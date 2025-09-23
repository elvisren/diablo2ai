#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified frame pipeline + PyQt GUI (dark theme).

- Source node: Camera / Video / Image
- Display (monitor) rectifier node
- Dynamic object detector (YOLOv11 via your ObjectDetector)
- Static object detector (dummy; implement later)
- Aggregation + drawing
- E2E timing + video "interval skip ratio" control

Requires:
  pip install PyQt5 opencv-python ultralytics

Place alongside:
  camera_capture.py        (user provided)
  screen_detector.py       (user provided)
  object_detector.py       (user provided)
"""

from __future__ import annotations

import sys
import time
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# --- user-provided helpers (cited) ---
# camera capture (camera device open/read)  [citation below]
from camera_capture import CameraCapture  # :contentReference[oaicite:3]{index=3}
# monitor/screen detector+rectifier         [citation below]
from screen_detector import ScreenRectifier  # :contentReference[oaicite:4]{index=4}
# YOLO-based object detector wrapper        [citation below]
from object_detector import ObjectDetector, DetectionResult  # :contentReference[oaicite:5]{index=5}

# --- Qt stuff ---
from PyQt5 import QtCore, QtGui, QtWidgets


# =========================
# Generic detection schema
# =========================

@dataclass
class ObjInstance:
    """Generic per-instance output, works for YOLO and traditional CV."""
    name: str
    bbox_xyxy: Tuple[int, int, int, int]         # (x1, y1, x2, y2) in the *current* frame coordinates
    conf: float = 0.0
    source: str = ""                              # e.g., "yolo", "static", "display"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameResult:
    """Result of a pipeline step."""
    frame: np.ndarray
    objects: List[ObjInstance]


def draw_instances(img: np.ndarray, instances: List[ObjInstance]) -> np.ndarray:
    """Draws bboxes + labels; non-destructive (returns a copy)."""
    out = img.copy()
    for det in instances:
        x1, y1, x2, y2 = map(int, det.bbox_xyxy)
        # box
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 255), 2, cv2.LINE_AA)
        # label
        label = det.name
        if det.conf is not None:
            label = f"{label} {det.conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(0, y1 - 8)
        cv2.rectangle(out, (x1, y - th - 6), (x1 + tw + 6, y), (60, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(out, label, (x1 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


# =========================
# Pipeline nodes
# =========================

class BaseNode:
    """All nodes consume np.ndarray (BGR) and return FrameResult."""
    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    def process(self, frame: np.ndarray) -> FrameResult:
        raise NotImplementedError


class NopNode(BaseNode):
    """Dummy node that passes through."""
    def __init__(self, name="nop"):
        super().__init__(name)

    def process(self, frame: np.ndarray) -> FrameResult:
        return FrameResult(frame=frame, objects=[])


class DisplayRectifierNode(BaseNode):
    """Find the monitor and warp/rectify to target 16:9."""
    def __init__(self, target_w: int = 1920, target_h: int = 1080):
        super().__init__("display_rectifier")
        self.rectifier = ScreenRectifier(target_w=target_w, target_h=target_h)

    def process(self, frame: np.ndarray) -> FrameResult:
        if not self.enabled:
            return FrameResult(frame=frame, objects=[])

        quad = self.rectifier.detect(frame)
        if quad is not None:
            # Option 1: fully rectify to target size (stable input for downstream)
            warped = self.rectifier.rectify(frame, quad)
            # Also record the screen quad as an "object" if you want to visualize it:
            x1, y1 = np.min(quad[:, 0]), np.min(quad[:, 1])
            x2, y2 = np.max(quad[:, 0]), np.max(quad[:, 1])
            inst = ObjInstance(name="display", bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                               conf=1.0, source="display")
            return FrameResult(frame=warped, objects=[inst])
        else:
            # If not found, pass-through (downstream can still try)
            return FrameResult(frame=frame, objects=[])


class DynamicObjectDetectorNode(BaseNode):
    """YOLOv11 (via your wrapper); detects 'moving' or general objects."""
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 conf: float = 0.25, iou: float = 0.45, imgsz: int = 1280):
        super().__init__("dynamic_object_detector")
        self.det = ObjectDetector(model_path=model_path, conf=conf, iou=iou, imgsz=imgsz, device=device)
        # Warmup lightly (safe even if Ultralytics missing)
        try:
            self.det.warmup()
        except Exception:
            pass

    def process(self, frame: np.ndarray) -> FrameResult:
        if not self.enabled:
            return FrameResult(frame=frame, objects=[])

        try:
            res: DetectionResult = self.det.detect(frame)
        except Exception as e:
            # If model fails, pass-through with nothing
            return FrameResult(frame=frame, objects=[])

        # Convert to our generic instances
        objs: List[ObjInstance] = []
        for d in res.detections:
            objs.append(ObjInstance(
                name=d.name,
                bbox_xyxy=tuple(map(int, d.xyxy)),
                conf=float(d.conf),
                source="yolo",
                meta={"cls_id": int(d.cls)}
            ))
        # You may choose to use res.annotated; we keep raw frame so drawing stays unified
        return FrameResult(frame=frame, objects=objs)


class StaticObjectDetectorNode(BaseNode):
    """Placeholder for your future CV2-based static detector."""
    def __init__(self):
        super().__init__("static_object_detector")
        # TODO: initialize your classical CV assets here

    def process(self, frame: np.ndarray) -> FrameResult:
        if not self.enabled:
            return FrameResult(frame=frame, objects=[])
        # Dummy: identify nothing (you can add template matching, feature matching, etc.)
        return FrameResult(frame=frame, objects=[])


# =========================
# Source node (Camera / Video / Image)
# =========================

class SourceMode:
    CAMERA = "Camera"
    VIDEO = "Video"
    IMAGE = "Image"


class SourceNode:
    """
    Provides frames on demand.

    For 'Video': supports "interval ratio" skipping:
      after each pipeline round-trip of duration T, we skip ahead by (ratio * T) seconds.
      ratio=1.0 => real-time; 0.5 => half; 2.0 => double, etc.
    For 'Camera': we ignore interval and just read next frame immediately.
    For 'Image': returns the same still each time.
    """

    def __init__(self):
        self.mode = SourceMode.CAMERA
        self.camera_id = 0
        self.video_path: Optional[str] = None
        self.image_path: Optional[str] = None
        self.interval_ratio: float = 1.0

        self._cam: Optional[CameraCapture] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._still: Optional[np.ndarray] = None
        self._video_fps: float = 0.0

    def set_mode(self, mode: str):
        self.mode = mode

    def set_camera(self, cam_id: int = 0):
        self.camera_id = cam_id

    def set_video(self, path: str):
        self.video_path = path

    def set_image(self, path: str):
        self.image_path = path

    def set_interval_ratio(self, r: float):
        self.interval_ratio = max(0.0, float(r))

    def open(self):
        self.close()
        if self.mode == SourceMode.CAMERA:
            self._cam = CameraCapture(device_index=self.camera_id, width=1920, height=1080, fps=30)
            self._cam.open()  # may raise if camera missing
        elif self.mode == SourceMode.VIDEO:
            if not self.video_path or not Path(self.video_path).exists():
                raise RuntimeError("Video path not set or not found.")
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise RuntimeError("Could not open video.")
            self._video_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        elif self.mode == SourceMode.IMAGE:
            if not self.image_path or not Path(self.image_path).exists():
                raise RuntimeError("Image path not set or not found.")
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to read image.")
            self._still = img
        else:
            raise RuntimeError("Unknown mode")

    def restart(self):
        """For 'Video', rewind to start. For others, re-open."""
        if self.mode == SourceMode.VIDEO and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.open()

    def close(self):
        if self._cam is not None:
            self._cam.release()
            self._cam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._still = None
        self._video_fps = 0.0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.mode == SourceMode.CAMERA:
            if self._cam is None:
                return False, None
            ok, frame = self._cam.read()
            return ok, frame
        elif self.mode == SourceMode.VIDEO:
            if self._cap is None:
                return False, None
            ok, frame = self._cap.read()
            return ok, frame
        elif self.mode == SourceMode.IMAGE:
            if self._still is None:
                return False, None
            return True, self._still.copy()
        else:
            return False, None

    def skip_ahead_by_seconds(self, dt: float):
        """Only relevant for VIDEO."""
        if self.mode != SourceMode.VIDEO or self._cap is None or self._video_fps <= 0:
            return
        frames_to_skip = int(round(self._video_fps * max(0.0, dt)))
        # Fast-skip via grab()
        for _ in range(frames_to_skip):
            self._cap.grab()


# =========================
# Pipeline orchestrator
# =========================

class Pipeline:
    def __init__(self, source: SourceNode, rectifier: DisplayRectifierNode,
                 dyn_det: DynamicObjectDetectorNode, stat_det: StaticObjectDetectorNode):
        self.source = source
        self.nodes: List[BaseNode] = [rectifier, dyn_det, stat_det]

    def run_once(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[ObjInstance], float]:
        """
        Returns:
          (input_frame_display, final_frame_display, aggregated_instances, latency_s)
        Where input_frame_display is the *input to pipeline* (after Source read),
        and final_frame_display is the *last stage output* with boxes drawn.
        """
        t0 = time.time()

        ok, frame_in = self.source.read()
        if not ok or frame_in is None:
            return None, None, [], 0.0

        input_for_view = frame_in.copy()  # show pipeline input (left)

        # Process through nodes
        agg: List[ObjInstance] = []
        cur = frame_in
        for node in self.nodes:
            res = node.process(cur)
            cur = res.frame
            if res.objects:
                agg.extend(res.objects)

        # Draw aggregated detections on final frame
        final = draw_instances(cur, agg)

        latency = time.time() - t0

        # Handle interval skipping policy:
        if self.source.mode == SourceMode.VIDEO:
            # Skip ahead by "ratio * latency" seconds
            skip_dt = self.source.interval_ratio * latency
            self.source.skip_ahead_by_seconds(skip_dt)
        # For camera: read immediately next time (no sleep).
        # For image: do nothing.

        return input_for_view, final, agg, latency


# =========================
# Qt worker thread
# =========================

class Worker(QtCore.QObject):
    frameReady = QtCore.pyqtSignal(np.ndarray, np.ndarray, list, float)  # (left_img, right_img, objs, latency)
    stopped = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._running = False
        self._lock = threading.Lock()

    @QtCore.pyqtSlot()
    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        self._loop()

    def _loop(self):
        while True:
            with self._lock:
                if not self._running:
                    self.stopped.emit()
                    return
            try:
                left, right, objs, latency = self.pipeline.run_once()
                if left is None or right is None:
                    # If source is video and ended, emit stopped
                    if self.pipeline.source.mode == SourceMode.VIDEO:
                        with self._lock:
                            self._running = False
                        self.stopped.emit()
                        return
                    else:
                        # For camera/image, continue trying
                        time.sleep(0.01)
                        continue
                self.frameReady.emit(left, right, objs, latency)
            except Exception as e:
                self.error.emit(str(e))
                with self._lock:
                    self._running = False
                self.stopped.emit()
                return

    @QtCore.pyqtSlot()
    def stop(self):
        with self._lock:
            self._running = False


# =========================
# Qt GUI
# =========================

def cvimg_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    """Convert BGR numpy image to QImage (RGB888)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


class ImagePane(QtWidgets.QLabel):
    """Aspect-correct 16:9 display pane."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 180)  # keep a reasonable minimum

    def set_frame(self, img_bgr: np.ndarray):
        qimg = cvimg_to_qimage(img_bgr)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # ----- Window sizing + layout -----
        self.setWindowTitle("AI Object Pipeline")
        # Initial width 2000; compute height to fit controls + two 16:9 panes
        init_width = 2000
        control_h = 120  # rough; we’ll pack tightly to avoid white gap
        # Two 16:9 panes side-by-side => each ~ (init_width- paddings)/2
        pane_w = (init_width - 40) // 2
        pane_h = int(pane_w * 9 / 16)
        init_height = control_h + pane_h + 40
        self.resize(init_width, init_height)

        # ----- Controls (top row) -----
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(10, 10, 10, 0)
        top.setSpacing(12)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems([SourceMode.CAMERA, SourceMode.VIDEO, SourceMode.IMAGE])

        self.camera_id_spin = QtWidgets.QSpinBox()
        self.camera_id_spin.setRange(0, 16)
        self.camera_id_spin.setValue(0)

        self.video_path_edit = QtWidgets.QLineEdit()
        self.video_browse_btn = QtWidgets.QPushButton("Browse…")

        self.image_path_edit = QtWidgets.QLineEdit()
        self.image_browse_btn = QtWidgets.QPushButton("Browse…")

        self.interval_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.interval_ratio_spin.setRange(0.0, 8.0)
        self.interval_ratio_spin.setDecimals(2)
        self.interval_ratio_spin.setSingleStep(0.25)
        self.interval_ratio_spin.setValue(1.0)

        self.use_dynamic_cb = QtWidgets.QCheckBox("Dynamic object detector (YOLO)")
        self.use_dynamic_cb.setChecked(True)
        self.use_static_cb = QtWidgets.QCheckBox("Static object detector")
        self.use_static_cb.setChecked(False)

        self.run_btn = QtWidgets.QPushButton("Run")

        # Grouping / labels
        top.addWidget(QtWidgets.QLabel("Source:"))
        top.addWidget(self.source_combo)

        self.cam_box = QtWidgets.QWidget()
        cam_l = QtWidgets.QHBoxLayout(self.cam_box)
        cam_l.setContentsMargins(0,0,0,0)
        cam_l.addWidget(QtWidgets.QLabel("Camera ID:"))
        cam_l.addWidget(self.camera_id_spin)
        top.addWidget(self.cam_box)

        self.vid_box = QtWidgets.QWidget()
        vid_l = QtWidgets.QHBoxLayout(self.vid_box)
        vid_l.setContentsMargins(0,0,0,0)
        vid_l.addWidget(QtWidgets.QLabel("Video:"))
        vid_l.addWidget(self.video_path_edit, 1)
        vid_l.addWidget(self.video_browse_btn)
        top.addWidget(self.vid_box)

        self.img_box = QtWidgets.QWidget()
        img_l = QtWidgets.QHBoxLayout(self.img_box)
        img_l.setContentsMargins(0,0,0,0)
        img_l.addWidget(QtWidgets.QLabel("Image:"))
        img_l.addWidget(self.image_path_edit, 1)
        img_l.addWidget(self.image_browse_btn)
        top.addWidget(self.img_box)

        top.addWidget(QtWidgets.QLabel("Interval ratio:"))
        top.addWidget(self.interval_ratio_spin)

        top.addStretch(1)

        top.addWidget(self.use_dynamic_cb)
        top.addWidget(self.use_static_cb)
        top.addWidget(self.run_btn)

        # ----- Two image panes (bottom row) -----
        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(10, 8, 10, 10)
        bottom.setSpacing(12)

        self.left_pane = ImagePane()
        self.left_pane.setFixedSize(pane_w, pane_h)
        self.right_pane = ImagePane()
        self.right_pane.setFixedSize(pane_w, pane_h)

        bottom.addWidget(self.left_pane, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        bottom.addWidget(self.right_pane, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # ----- Pack layouts -----
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        layout.addLayout(top, 0)
        layout.addLayout(bottom, 1)

        # Ensure no white gap by forcing sizes on resize
        self._pane_h_fixed = pane_h

        # Dark theme (Fusion + palette)
        self._apply_dark_theme()

        # ----- Pipeline bits (created on Run) -----
        self.source_node: Optional[SourceNode] = None
        self.rectifier_node: Optional[DisplayRectifierNode] = None
        self.dynamic_node: Optional[DynamicObjectDetectorNode] = None
        self.static_node: Optional[StaticObjectDetectorNode] = None
        self.pipeline: Optional[Pipeline] = None

        # Worker thread
        self.thread: Optional[QtCore.QThread] = None
        self.worker: Optional[Worker] = None

        # Signals / slots
        self.source_combo.currentTextChanged.connect(self._on_source_mode_changed)
        self.video_browse_btn.clicked.connect(self._browse_video)
        self.image_browse_btn.clicked.connect(self._browse_image)
        self.run_btn.clicked.connect(self._on_run_clicked)

        self._on_source_mode_changed(self.source_combo.currentText())

    # --- Dark theme ---
    def _apply_dark_theme(self):
        QtWidgets.QApplication.setStyle("Fusion")
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(37, 37, 38))
        pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(37, 37, 38))
        pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 48))
        pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(14, 99, 156))
        pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        self.setPalette(pal)

    # --- Layout control to avoid gap ---
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Make panes fill the available width, keep 16:9, keep same size
        avail_w = self.width() - 40
        pane_w = avail_w // 2
        pane_h = int(pane_w * 9 / 16)
        self.left_pane.setFixedSize(pane_w, pane_h)
        self.right_pane.setFixedSize(pane_w, pane_h)

    # --- Source UI switching ---
    def _on_source_mode_changed(self, mode: str):
        self.cam_box.setVisible(mode == SourceMode.CAMERA)
        self.vid_box.setVisible(mode == SourceMode.VIDEO)
        self.img_box.setVisible(mode == SourceMode.IMAGE)

    def _browse_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video", "", "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)")
        if path:
            self.video_path_edit.setText(path)

    def _browse_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All Files (*)")
        if path:
            self.image_path_edit.setText(path)

    # --- Run/Stop/Restart ---
    def _on_run_clicked(self):
        text = self.run_btn.text().strip().lower()
        if text in ("run", "restart"):
            self._start_pipeline(restart=(text == "restart"))
        elif text == "stop":
            self._stop_pipeline()

    def _start_pipeline(self, restart: bool = False):
        # If an old worker exists, stop it first
        self._stop_pipeline(finalize_only=True)

        # Build source
        self.source_node = SourceNode()
        mode = self.source_combo.currentText()
        self.source_node.set_mode(mode)
        self.source_node.set_camera(self.camera_id_spin.value())
        self.source_node.set_video(self.video_path_edit.text().strip())
        self.source_node.set_image(self.image_path_edit.text().strip())
        self.source_node.set_interval_ratio(self.interval_ratio_spin.value())

        # Build nodes
        self.rectifier_node = DisplayRectifierNode(target_w=1920, target_h=1080)
        self.rectifier_node.enabled = True  # always on

        self.dynamic_node = DynamicObjectDetectorNode(model_path=None, device=None, conf=0.25, iou=0.45, imgsz=1280)
        self.dynamic_node.enabled = self.use_dynamic_cb.isChecked()

        self.static_node = StaticObjectDetectorNode()
        self.static_node.enabled = self.use_static_cb.isChecked()

        self.pipeline = Pipeline(self.source_node, self.rectifier_node, self.dynamic_node, self.static_node)

        # Open source
        try:
            if restart and self.source_node.mode == SourceMode.VIDEO:
                self.source_node.open()
                self.source_node.restart()
            else:
                self.source_node.open()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Source Error", str(e))
            return

        # Start worker thread
        self.thread = QtCore.QThread()
        self.worker = Worker(self.pipeline)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.start)
        self.worker.frameReady.connect(self._on_frame_ready)
        self.worker.stopped.connect(self._on_worker_stopped)
        self.worker.error.connect(self._on_worker_error)

        self.thread.start()
        self.run_btn.setText("Stop")

    def _stop_pipeline(self, finalize_only: bool = False):
        if self.worker:
            try:
                self.worker.stop()
            except Exception:
                pass
        if self.thread:
            self.thread.quit()
            self.thread.wait(1000)
        if not finalize_only:
            # flip button state
            self.run_btn.setText("Restart")

    def _on_worker_stopped(self):
        # When video finishes, let user restart (rewind)
        self.run_btn.setText("Restart")
        # leave the last frames on screen

    def _on_worker_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Pipeline Error", msg)
        self.run_btn.setText("Restart")

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, list, float)
    def _on_frame_ready(self, left_bgr: np.ndarray, right_bgr: np.ndarray, objs: list, latency: float):
        # Show images
        self.left_pane.set_frame(left_bgr)
        self.right_pane.set_frame(right_bgr)
        # Optionally: show latency in window title
        self.setWindowTitle(f"AI Object Pipeline  —  latency: {latency*1000:.1f} ms  —  detections: {len(objs)}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
