#!/usr/bin/env python3
# app.py
import sys, time, logging, signal, zoneinfo, datetime, os
from pathlib import Path
from typing import Optional, Literal
from collections import deque

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from camera_capture import CameraCapture
from screen_detector import ScreenRectifier
from detectors.object_detector import ObjectDetector, DetectionResult

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("qt-app")

# ---------------- Utils ----------------
def bgr_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    """Return a QImage that OWNS its data (avoid dangling NumPy buffer)."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
    return img.copy()  # ensure independent buffer

def shanghai_timestamp() -> str:
    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    now = datetime.datetime.now(tz)
    return now.strftime("%Y%m%d_%H%M%S")

def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

# ---------------- Video ingest worker (unchanged behavior) ----------------
class VideoWorker(QtCore.QThread):
    """
    Reads frames from 'camera' or 'file', detects screen & rectifies, and emits:
      - frame_pair(QImage annotated_left, QImage rectified_right)
      - rectified_frame_ready(ndarray, float timestamp_seconds)
      - file_progress(int pct, int idx, int total, float t_cur, float t_total)
      - file_finished()
    """
    frame_pair = QtCore.pyqtSignal(QtGui.QImage, QtGui.QImage)
    rectified_frame_ready = QtCore.pyqtSignal(object, float)
    file_progress = QtCore.pyqtSignal(int, int, int, float, float)
    file_finished = QtCore.pyqtSignal()

    def __init__(self,
                 mode: Literal["camera", "file"],
                 device: int,
                 width: int,
                 height: int,
                 fps: int,
                 target_w: int,
                 target_h: int,
                 video_path: Optional[Path] = None):
        super().__init__()
        self.mode = mode
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.video_path = str(video_path) if video_path else None
        self.rectifier = ScreenRectifier(target_w, target_h)
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

    def _open_source(self) -> bool:
        if self.mode == "camera":
            cam = CameraCapture(self.device, self.width, self.height, self.fps)
            try:
                cam.open()
            except Exception as e:
                log.error("Camera open failed: %s", e)
                return False
            self._cap = cam.cap
            return True
        else:
            if not self.video_path or not os.path.isfile(self.video_path):
                log.error("Video file not found: %s", self.video_path)
                return False
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                log.error("Failed to open video file: %s", self.video_path)
                return False
            self._cap = cap
            fps_file = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            log.info("Opened video %s @ %.3f fps, %d frames", self.video_path, fps_file, total)
            return True

    def run(self):
        if not self._open_source():
            return

        self._running = True
        last_quad = None

        src_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if src_fps <= 0:
            src_fps = float(self.fps if self.mode == "camera" else 30.0)
        period = 1.0 / max(1e-6, src_fps)

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if self.mode == "file" else 0
        cur_idx = 0
        t0 = time.perf_counter()

        try:
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    if self.mode == "file":
                        self.file_progress.emit(100, total_frames, total_frames, float(cur_idx/src_fps), float(total_frames/src_fps))
                        self.file_finished.emit()
                        break
                    self.msleep(5)
                    continue

                if self.mode == "file":
                    cur_idx += 1

                # ---------- Detect display + rectify ----------
                quad = self.rectifier.detect(frame)
                if quad is None and last_quad is not None:
                    quad = last_quad

                if quad is not None:
                    last_quad = quad
                    left_img = self.rectifier.annotate(frame, quad)
                    rectified = self.rectifier.rectify(frame, quad)
                else:
                    left_img = frame
                    rectified = np.zeros((self.rectifier.target_h, self.rectifier.target_w, 3), np.uint8)

                # Emit previews and the raw rectified frame for downstream detection
                if not self._running: break
                self.frame_pair.emit(bgr_to_qimage(left_img), bgr_to_qimage(rectified))
                if not self._running: break
                self.rectified_frame_ready.emit(rectified, time.perf_counter())

                # ---------- Progress pacing for files ----------
                if self.mode == "file":
                    if total_frames > 0:
                        pct = int(round(100.0 * cur_idx / max(1, total_frames)))
                        t_cur = float(cur_idx / src_fps)
                        t_tot = float(total_frames / src_fps)
                    else:
                        ratio = float(self._cap.get(cv2.CAP_PROP_POS_AVI_RATIO) or 0.0)
                        pct = int(round(100.0 * ratio))
                        t_cur = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                        t_tot = 0.0
                    self.file_progress.emit(min(100, pct), cur_idx, total_frames, t_cur, t_tot)

                    expected_elapsed = cur_idx * period
                    now = time.perf_counter() - t0
                    delay = expected_elapsed - now
                    if delay > 0:
                        self.msleep(int(delay * 1000))
                else:
                    self.msleep(1)
        finally:
            if self._cap is not None:
                try: self._cap.release()
                except Exception: pass
                self._cap = None

    def stop(self):
        self._running = False
        self.wait(1000)

# ---------------- Detection worker (latest-frame only) ----------------
class DetectionWorker(QtCore.QThread):
    """
    Consumes rectified frames and runs heavy detection.
    Drop strategy: keep a maxlen=1 queue to always process the newest frame.
    Emits:
      - annotated_ready(QImage, ndarray raw_annotated, float tstamp)
      - results_ready(list[Detection], float tstamp)  # optional for your logic
    """
    annotated_ready = QtCore.pyqtSignal(QtGui.QImage, object, float)
    results_ready = QtCore.pyqtSignal(list, float)

    def __init__(self, detector: ObjectDetector, parent=None):
        super().__init__(parent)
        self.detector = detector
        self._running = False
        self._queue: deque[tuple[np.ndarray, float]] = deque(maxlen=1)
        self._lock = QtCore.QMutex()

    @QtCore.pyqtSlot(object, float)
    def push_frame(self, frame_bgr: np.ndarray, tstamp: float):
        # Replace any existing frame with the newest
        with QtCore.QMutexLocker(self._lock):
            if len(self._queue) == self._queue.maxlen:
                self._queue.pop()
            self._queue.append((frame_bgr, tstamp))

    def run(self):
        self._running = True
        self.detector.warmup()
        while self._running:
            item = None
            with QtCore.QMutexLocker(self._lock):
                if self._queue:
                    item = self._queue.pop()
            if item is None:
                self.msleep(2)
                continue

            frame_bgr, ts = item
            try:
                result: DetectionResult = self.detector.detect(frame_bgr)
            except Exception as e:
                log.error("Detection error: %s", e)
                # Draw a small warning overlay but keep UI alive
                annotated = frame_bgr.copy()
                cv2.putText(annotated, f"Detection error: {e}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                qimg = bgr_to_qimage(annotated)
                self.annotated_ready.emit(qimg, annotated, ts)
                continue

            qimg = bgr_to_qimage(result.annotated)
            self.annotated_ready.emit(qimg, result.annotated, ts)
            # If you want raw results downstream, emit here:
            # self.results_ready.emit(result.detections, ts)

    def stop(self):
        self._running = False
        self.wait(1000)

# ---------------- Recorder ----------------
class VideoRecorder(QtCore.QObject):
    """Constant-FPS writer; duplicates frames to keep timeline if processing lags."""
    def __init__(self, out_dir: Path, fps: int = 30):
        super().__init__()
        self.out_dir = out_dir
        self.fps = max(1, int(fps))
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_size: Optional[tuple] = None
        self.next_t: Optional[float] = None
        self.last_frame: Optional[np.ndarray] = None

    def start(self, frame_size: tuple[int, int]) -> Path:
        self.stop()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"diablo2_{shanghai_timestamp()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(path), fourcc, self.fps, frame_size, True)
        if not self.writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter")
        self.frame_size, self.next_t, self.last_frame = frame_size, None, None
        return path

    def stop(self):
        if self.writer is not None:
            try: self.writer.release()
            finally:
                self.writer = None
                self.frame_size = None
                self.next_t = None
                self.last_frame = None

    def on_frame(self, frame: np.ndarray, tstamp: float):
        if self.writer is None: return
        assert self.frame_size is not None
        fh, fw = frame.shape[:2]
        if (fw, fh) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)

        period = 1.0 / self.fps
        if self.next_t is None:
            self.next_t = tstamp

        while self.next_t + 1e-6 < tstamp:
            self.writer.write(self.last_frame if self.last_frame is not None else frame)
            self.next_t += period

        self.writer.write(frame)
        self.last_frame = frame
        self.next_t += period

# ---------------- Aspect container (unchanged) ----------------
class AspectContainer(QtWidgets.QWidget):
    """Keeps its single child (e.g., ImagePane) at a fixed aspect ratio."""
    def __init__(self, aspect_w: int, aspect_h: int, child: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self._aw, self._ah = aspect_w, aspect_h
        self._child = child
        self._child.setParent(self)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        r = self.rect()
        if self._ah == 0:
            self._child.setGeometry(r)
            return
        target = self._aw / self._ah
        cur = r.width() / max(1, r.height())
        if cur > target:
            h = r.height()
            w = int(round(h * target))
            x = r.x() + (r.width() - w) // 2
            y = r.y()
            self._child.setGeometry(QtCore.QRect(x, y, w, h))
        else:
            w = r.width()
            h = int(round(w / target))
            x = r.x()
            y = r.y() + (r.height() - h) // 2
            self._child.setGeometry(QtCore.QRect(x, y, w, h))

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtCore.Qt.black)

# ---------------- Image panes ----------------
class ImagePane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[QtGui.QImage] = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

    def setImage(self, img: QtGui.QImage):
        self._image = img
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        r = self.rect()
        painter.fillRect(r, QtCore.Qt.black)
        if self._image is None: return
        pm = QtGui.QPixmap.fromImage(self._image)
        src_aspect = self._image.width() / max(1, self._image.height())
        dst_aspect = r.width() / max(1, r.height())

        if abs(src_aspect - dst_aspect) < 1e-3:
            scaled = pm.scaled(r.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            painter.drawPixmap(r, scaled, scaled.rect())
        elif src_aspect > dst_aspect:
            h = r.height(); w = int(round(h * src_aspect))
            scaled = pm.scaled(w, h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            x = (w - r.width()) // 2
            painter.drawPixmap(r, scaled, QtCore.QRect(x, 0, r.width(), r.height()))
        else:
            w = r.width(); h = int(round(w / src_aspect))
            scaled = pm.scaled(w, h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            y = (h - r.height()) // 2
            painter.drawPixmap(r, scaled, QtCore.QRect(0, y, r.width(), r.height()))

# ---------------- Main Window ----------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("diablo II resurrected AI analyzer")

        # Core workers
        self.worker: Optional[VideoWorker] = None
        self.detector = ObjectDetector(model_path=None, conf=0.25, iou=0.45, max_det=200)
        self.detWorker: Optional[DetectionWorker] = None

        self.recorder = VideoRecorder(Path("./snapshots"), fps=30)
        self.recording_path: Optional[Path] = None

        # ========== LEFT COLUMN ==========
        self.deviceSpin = QtWidgets.QSpinBox(); self.deviceSpin.setRange(0, 32); self.deviceSpin.setValue(0)
        self.widthSpin  = QtWidgets.QSpinBox(); self.widthSpin.setRange(160, 7680); self.widthSpin.setValue(1920)
        self.heightSpin = QtWidgets.QSpinBox(); self.heightSpin.setRange(120, 4320); self.heightSpin.setValue(1080)
        self.fpsSpin    = QtWidgets.QSpinBox(); self.fpsSpin.setRange(1, 120);      self.fpsSpin.setValue(30)
        self.camBtn     = QtWidgets.QPushButton("Read from camera")

        camBox = QtWidgets.QGroupBox("Camera")
        camGrid = QtWidgets.QGridLayout(camBox); camGrid.setContentsMargins(8,8,8,8); camGrid.setHorizontalSpacing(8)
        row = 0
        camGrid.addWidget(QtWidgets.QLabel("Device"),  row, 0); camGrid.addWidget(self.deviceSpin,  row, 1)
        camGrid.addWidget(QtWidgets.QLabel("Width"),   row, 2); camGrid.addWidget(self.widthSpin,   row, 3)
        camGrid.addWidget(QtWidgets.QLabel("Height"),  row, 4); camGrid.addWidget(self.heightSpin,  row, 5)
        camGrid.addWidget(QtWidgets.QLabel("FPS"),     row, 6); camGrid.addWidget(self.fpsSpin,     row, 7)
        camGrid.addWidget(self.camBtn, row+1, 0, 1, 8)

        self.videoPathEdit = QtWidgets.QLineEdit("")
        self.browseBtn = QtWidgets.QPushButton("Open file…")
        self.fileBtn   = QtWidgets.QPushButton("Play from file")
        self.fileProgress = QtWidgets.QProgressBar(); self.fileProgress.setRange(0, 100); self.fileProgress.setValue(0)
        self.fileTime = QtWidgets.QLabel("--:-- / --:--")

        fileBox = QtWidgets.QGroupBox("Video file")
        fileGrid = QtWidgets.QGridLayout(fileBox); fileGrid.setContentsMargins(8,8,8,8); fileGrid.setHorizontalSpacing(8)
        fileGrid.addWidget(QtWidgets.QLabel("Path"), 0,0)
        fileGrid.addWidget(self.videoPathEdit, 0,1,1,2)
        fileGrid.addWidget(self.browseBtn, 0,3)
        fileGrid.addWidget(self.fileBtn, 1,0)
        fileGrid.addWidget(self.fileProgress, 1,1,1,2)
        fileGrid.addWidget(self.fileTime, 1,3)

        leftTop = QtWidgets.QVBoxLayout()
        leftTop.setContentsMargins(8,8,8,4); leftTop.setSpacing(8)
        leftTop.addWidget(camBox)
        leftTop.addWidget(fileBox)
        leftTopW = QtWidgets.QWidget(); leftTopW.setLayout(leftTop)

        self.leftPane = ImagePane()

        # ========== RIGHT COLUMN ==========
        self.targetWSpin= QtWidgets.QSpinBox(); self.targetWSpin.setRange(320, 7680); self.targetWSpin.setValue(3840)
        self.targetHSpin= QtWidgets.QSpinBox(); self.targetHSpin.setRange(180, 4320); self.targetHSpin.setValue(2160)
        self.savePathEdit = QtWidgets.QLineEdit("./snapshots/")
        self.saveBtn = QtWidgets.QPushButton("Start recording")

        grp_display = QtWidgets.QGroupBox("Display and Record")
        g1 = QtWidgets.QGridLayout(grp_display); g1.setContentsMargins(8,8,8,8); g1.setHorizontalSpacing(8); g1.setVerticalSpacing(6)
        g1.addWidget(QtWidgets.QLabel("Target Width"),  0, 0); g1.addWidget(self.targetWSpin, 0, 1)
        g1.addWidget(QtWidgets.QLabel("Target Height"), 0, 2); g1.addWidget(self.targetHSpin, 0, 3)

        grp_save = QtWidgets.QGroupBox("Save processed result")
        g2 = QtWidgets.QGridLayout(grp_save); g2.setContentsMargins(8,8,8,8); g2.setHorizontalSpacing(8); g2.setVerticalSpacing(6)
        g2.addWidget(QtWidgets.QLabel("Save dir"), 0, 0); g2.addWidget(self.savePathEdit, 0, 1, 1, 2); g2.addWidget(self.saveBtn, 0, 3)

        rightTop = QtWidgets.QVBoxLayout()
        rightTop.setContentsMargins(8,8,8,4); rightTop.setSpacing(8)
        rightTop.addWidget(grp_display)
        rightTop.addWidget(grp_save)
        rightTopW = QtWidgets.QWidget(); rightTopW.setLayout(rightTop)

        self.rightPane = ImagePane()

        self.leftAspect  = AspectContainer(3840, 2160, self.leftPane)
        self.rightAspect = AspectContainer(3840, 2160, self.rightPane)

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0,0,0,0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(0)

        grid.addWidget(leftTopW,          0, 0)
        grid.addWidget(rightTopW,         0, 1)
        grid.addWidget(self.leftAspect,   1, 0)
        grid.addWidget(self.rightAspect,  1, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 1)

        self.leftTopW = leftTopW
        self.rightTopW = rightTopW

        # Minimum heights to avoid clipped buttons on some OS themes
        self._base_controls_h = max(self.leftTopW.sizeHint().height(), self.rightTopW.sizeHint().height())
        pad = 12
        self._controls_h_min = self._base_controls_h + pad
        for w in (self.leftTopW, self.rightTopW):
            w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
            w.setMinimumHeight(self._controls_h_min)

        self._preview_aspect = 32.0 / 9.0
        self._resizing_guard = False

        min_w = 1200
        min_h = self._controls_h_min + int(round(min_w / self._preview_aspect))
        self.setMinimumSize(min_w, min_h)

        # ---- Signals
        self.camBtn.clicked.connect(self.toggle_camera)
        self.fileBtn.clicked.connect(self.toggle_file)
        self.browseBtn.clicked.connect(self.choose_video)
        self.saveBtn.clicked.connect(self.toggle_recording)
        self.targetWSpin.valueChanged.connect(self._on_target_changed)
        self.targetHSpin.valueChanged.connect(self._on_target_changed)

        # Detector worker
        self._start_detector_worker()

        app = QtWidgets.QApplication.instance()
        if app:
            app.aboutToQuit.connect(self.cleanup)

        # Initial state
        self._set_camera_controls_enabled(True)
        self._set_file_controls_enabled(True)

        self._apply_dark_theme()
        self.resize(2000, self._controls_h_min + int(round(2000 / self._preview_aspect)))

    # ---------- Theme ----------
    def _apply_dark_theme(self):
        palette = QtGui.QPalette()
        bg = QtGui.QColor(18, 18, 22)
        panel = QtGui.QColor(28, 28, 35)
        text = QtGui.QColor(230, 230, 235)
        acc = QtGui.QColor(120, 170, 255)

        palette.setColor(QtGui.QPalette.Window, bg)
        palette.setColor(QtGui.QPalette.Base, panel)
        palette.setColor(QtGui.QPalette.AlternateBase, bg)
        palette.setColor(QtGui.QPalette.WindowText, text)
        palette.setColor(QtGui.QPalette.Text, text)
        palette.setColor(QtGui.QPalette.Button, panel)
        palette.setColor(QtGui.QPalette.ButtonText, text)
        palette.setColor(QtGui.QPalette.Highlight, acc)
        palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.white)
        self.setPalette(palette)

        self.setStyleSheet("""
        QWidget { color: #E6E6EB; }
        QGroupBox {
            border: 1px solid #3a3f4b; border-radius: 10px; margin-top: 12px;
            background: #1c1c23; padding: 8px 10px 10px 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #9fb0ff; }
        QPushButton {
            background: #2a2f3b; border: 1px solid #3f4656; border-radius: 8px; padding: 8px 12px;
        }
        QPushButton:hover { border-color: #7aa2ff; background: #2f3544; }
        QPushButton:pressed { background: #242a36; }
        QLineEdit, QSpinBox {
            background: #20232c; border: 1px solid #3a3f4b; border-radius: 6px; padding: 4px 6px;
        }
        QLineEdit:hover, QSpinBox:hover { border-color: #7aa2ff; }
        QProgressBar {
            background: #20232c; border: 1px solid #3a3f4b; border-radius: 6px; text-align: center;
        }
        QProgressBar::chunk { background-color: #7aa2ff; }
        """)

    # ---------- Resize logic ----------
    def resizeEvent(self, e: QtGui.QResizeEvent):
        if self._resizing_guard:
            return super().resizeEvent(e)
        self._resizing_guard = True
        try:
            new_w, new_h = e.size().width(), e.size().height()
            old_w, old_h = e.oldSize().width(), e.oldSize().height()

            controls_h = max(self.leftTopW.height(), self.rightTopW.height(), self._controls_h_min)
            dw = abs(new_w - (old_w if old_w > 0 else new_w))
            dh = abs(new_h - (old_h if old_h > 0 else new_h))
            if dw >= dh:
                target_h = controls_h + int(round(new_w / self._preview_aspect))
                if target_h != new_h:
                    self.resize(new_w, max(target_h, self.minimumHeight()))
            else:
                preview_h = max(1, new_h - controls_h)
                target_w = int(round(preview_h * self._preview_aspect))
                if target_w != new_w:
                    self.resize(max(target_w, self.minimumWidth()), new_h)
        finally:
            self._resizing_guard = False
        super().resizeEvent(e)

    # ---------- Source toggles ----------
    def toggle_camera(self):
        if self.worker and self.worker.isRunning() and getattr(self.worker, "mode", None) == "camera":
            self._stop_worker()
            self.camBtn.setText("Read from camera")
            self._set_file_controls_enabled(True)
            self._set_camera_controls_enabled(True)
            return

        if self.worker and getattr(self.worker, "mode", None) == "file":
            self._stop_worker()
            self.fileBtn.setText("Play from file")
            self.fileProgress.setValue(0)
            self.fileTime.setText("--:-- / --:--")

        dev = self.deviceSpin.value()
        w   = self.widthSpin.value()
        h   = self.heightSpin.value()
        fps = self.fpsSpin.value()
        tw  = self.targetWSpin.value()
        th  = self.targetHSpin.value()

        self.worker = VideoWorker("camera", dev, w, h, fps, tw, th)
        self._connect_worker_common()
        self.worker.start()

        self.camBtn.setText("Stop camera")
        self._set_file_controls_enabled(False)
        self._set_camera_controls_enabled(True)

    def toggle_file(self):
        if self.worker and self.worker.isRunning() and getattr(self.worker, "mode", None) == "file":
            self._stop_worker()
            self.fileBtn.setText("Play from file")
            self._set_camera_controls_enabled(True)
            self._set_file_controls_enabled(True)
            return

        if self.worker and getattr(self.worker, "mode", None) == "camera":
            self._stop_worker()
            self.camBtn.setText("Read from camera")

        vpath = self.videoPathEdit.text().strip()
        if not vpath:
            self.choose_video()
            vpath = self.videoPathEdit.text().strip()
            if not vpath:
                return

        tw  = self.targetWSpin.value()
        th  = self.targetHSpin.value()

        self.worker = VideoWorker("file", 0, 0, 0, 0, tw, th, Path(vpath))
        self._connect_worker_common()
        self.worker.file_progress.connect(self.on_file_progress)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.start()

        self.fileBtn.setText("Stop from file")
        self._set_camera_controls_enabled(False)
        self._set_file_controls_enabled(True)
        self.fileProgress.setValue(0)
        self.fileTime.setText("00:00 / " + self._read_total_time(vpath))

    def choose_video(self):
        start_dir = Path(self.savePathEdit.text()).expanduser().resolve()
        if not start_dir.exists():
            start_dir = Path("./snapshots").resolve()
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", str(start_dir),
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        if fname:
            self.videoPathEdit.setText(fname)

    def _connect_worker_common(self):
        self.worker.frame_pair.connect(self.on_frame_pair)
        # Important: push rectified frames to the detector worker
        self.worker.rectified_frame_ready.connect(self._on_rectified_for_detection)
        self.worker.finished.connect(lambda: log.info("Ingest worker finished"))

    def _stop_worker(self):
        if self.worker:
            try:
                try: self.worker.frame_pair.disconnect(self.on_frame_pair)
                except TypeError: pass
                try: self.worker.rectified_frame_ready.disconnect(self._on_rectified_for_detection)
                except TypeError: pass
                try: self.worker.file_progress.disconnect(self.on_file_progress)
                except Exception: pass
                try: self.worker.file_finished.disconnect(self.on_file_finished)
                except Exception: pass
                self.worker.stop()
            finally:
                self.worker = None
        # Keep detector worker alive; it’s shared across runs and just idles without frames
        if self.recording_path is not None:
            self.stop_recording()

    def _set_file_controls_enabled(self, enabled: bool):
        self.videoPathEdit.setEnabled(enabled)
        self.browseBtn.setEnabled(enabled)
        self.fileBtn.setEnabled(enabled)

    def _set_camera_controls_enabled(self, enabled: bool):
        self.deviceSpin.setEnabled(enabled)
        self.widthSpin.setEnabled(enabled)
        self.heightSpin.setEnabled(enabled)
        self.fpsSpin.setEnabled(enabled)
        self.camBtn.setEnabled(enabled)

    # ---------- Display / record ----------
    def on_frame_pair(self, left: QtGui.QImage, _right_raw: QtGui.QImage):
        # Left shows the source with screen quad overlay.
        self.leftPane.setImage(left)
        # Right will be set by detector output (annotated). We ignore _right_raw here.
        self.rightPane.setImage(_right_raw)

    @QtCore.pyqtSlot(object, float)
    def _on_rectified_for_detection(self, rectified_bgr: np.ndarray, tstamp: float):
        if self.detWorker is not None:
            self.detWorker.push_frame(rectified_bgr, tstamp)

    @QtCore.pyqtSlot(QtGui.QImage, object, float)
    def _on_annotated_ready(self, qimg: QtGui.QImage, raw_annotated: np.ndarray, tstamp: float):
        self.rightPane.setImage(qimg)  # overwrite with boxes once ready
        if self.recording_path is not None:
            self.recorder.on_frame(raw_annotated, tstamp)

    def _on_target_changed(self):
        # If you want live resizing of rectified/detected output,
        # you can restart the ingest worker with new target dims.
        pass

    def toggle_recording(self):
        if self.recording_path is None:
            out_dir = Path(self.savePathEdit.text()).expanduser().resolve()
            self.recorder.out_dir = out_dir
            tw = self.targetWSpin.value()
            th = self.targetHSpin.value()
            try:
                self.recording_path = self.recorder.start((tw, th))
                self.saveBtn.setText("Stop recording")
                self.savePathEdit.setText(str(out_dir))
                log.info("Recording to %s", self.recording_path)
            except Exception as e:
                log.error("Failed to start recording: %s", e)
                self.recording_path = None
        else:
            self.stop_recording()

    def stop_recording(self):
        try:
            self.recorder.stop()
            log.info("Saved video: %s", self.recording_path)
        finally:
            self.recording_path = None
            self.saveBtn.setText("Start recording")

    # ---------- File progress ----------
    def on_file_progress(self, pct: int, idx: int, total: int, t_cur: float, t_tot: float):
        if total > 0:
            self.fileProgress.setRange(0, 100)
            self.fileProgress.setValue(max(0, min(100, pct)))
            self.fileTime.setText(f"{fmt_time(t_cur)} / {fmt_time(t_tot)}")
        else:
            self.fileProgress.setRange(0, 0)
            self.fileTime.setText(f"{fmt_time(t_cur)} / --:--")

    def on_file_finished(self):
        self.fileProgress.setRange(0, 100)
        self.fileProgress.setValue(100)
        self.fileBtn.setText("Play from file")
        self._set_camera_controls_enabled(True)

    def _read_total_time(self, vpath: str) -> str:
        try:
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened(): return "--:--"
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            n   = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            cap.release()
            if fps > 0 and n > 0:
                return fmt_time(n / fps)
        except Exception:
            pass
        return "--:--"

    # ---------- Detector worker lifecycle ----------
    def _start_detector_worker(self):
        if self.detWorker is not None:
            return
        self.detWorker = DetectionWorker(self.detector, self)
        self.detWorker.annotated_ready.connect(self._on_annotated_ready)
        self.detWorker.start()

    # ---------- Cleanup ----------
    def closeEvent(self, event: QtGui.QCloseEvent):
        self.cleanup()
        super().closeEvent(event)

    def cleanup(self):
        self._stop_worker()
        if self.detWorker:
            try:
                self.detWorker.stop()
            except Exception:
                pass
            self.detWorker = None

# ---------------- main ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    ping = QtCore.QTimer(); ping.start(100); ping.timeout.connect(lambda: None)

    w = MainWindow()
    w.show()
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        w.cleanup()

if __name__ == "__main__":
    main()
