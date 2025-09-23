from __future__ import annotations
import time
import threading
from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np

# NOTE: per your package layout, imports use src.app...
from src.app.gui.theme import apply_dark_theme
from src.app.gui.widgets import ImagePane
from src.app.pipeline.display import DisplayRectifierNode
from src.app.pipeline.dynamic import DynamicObjectDetectorNode
from src.app.pipeline.source import SourceMode, SourceNode
from src.app.pipeline.static import StaticObjectDetectorNode
from src.app.pipeline.utils import draw_instances


class Worker(QtCore.QObject):
    frameReady = QtCore.pyqtSignal(np.ndarray, np.ndarray, list, float)
    stopped = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, pipeline):
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

    @QtCore.pyqtSlot()
    def stop(self):
        with self._lock:
            self._running = False

    def _loop(self):
        while True:
            with self._lock:
                if not self._running:
                    self.stopped.emit()
                    return
            try:
                left, right, objs, latency = self.pipeline.run_once()
                if left is None or right is None:
                    if self.pipeline.source.mode == SourceMode.VIDEO:
                        with self._lock:
                            self._running = False
                        self.stopped.emit()
                        return
                    else:
                        time.sleep(0.01)
                        continue
                self.frameReady.emit(left, right, objs, latency)
            except Exception as e:
                self.error.emit(str(e))
                with self._lock:
                    self._running = False
                self.stopped.emit()
                return


class Pipeline:
    def __init__(self, source: SourceNode, rectifier: DisplayRectifierNode,
                 dyn_det: DynamicObjectDetectorNode, stat_det: StaticObjectDetectorNode):
        self.source = source
        self.nodes = [rectifier, dyn_det, stat_det]

    def run_once(self):
        t0 = time.time()
        ok, frame_in = self.source.read()
        if not ok or frame_in is None:
            return None, None, [], 0.0

        left = frame_in.copy()

        agg = []
        cur = frame_in
        for node in self.nodes:
            if not node.enabled:
                continue
            res = node.process(cur)
            cur = res.frame
            if res.objects:
                agg.extend(res.objects)

        right = draw_instances(cur, agg)
        latency = time.time() - t0

        if self.source.mode == SourceMode.VIDEO:
            # Forward video by latency * skip_ratio (seconds)
            self.source.skip_ahead_by_seconds(latency * self.source.skip_ratio)

        return left, right, agg, latency


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Object Pipeline")

        init_width = 2000
        control_h = 120
        pane_w = (init_width - 40) // 2
        pane_h = int(pane_w * 9 / 16)
        init_height = control_h + pane_h + 40
        self.resize(init_width, init_height)

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

        # Skip ratio (video only)
        self.skip_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.skip_ratio_spin.setRange(0.1, 8.0)
        self.skip_ratio_spin.setDecimals(2)
        self.skip_ratio_spin.setSingleStep(0.25)
        self.skip_ratio_spin.setValue(1.0)
        self.skip_ratio_label = QtWidgets.QLabel("Skip ratio:")

        self.use_dynamic_cb = QtWidgets.QCheckBox("Dynamic object detector (YOLO or motion)")
        self.use_dynamic_cb.setChecked(True)
        self.use_static_cb = QtWidgets.QCheckBox("Static object detector")
        self.use_static_cb.setChecked(False)

        self.run_btn = QtWidgets.QPushButton("Run")

        # Left control group
        top.addWidget(QtWidgets.QLabel("Source:"))
        top.addWidget(self.source_combo)

        self.cam_box = QtWidgets.QWidget()
        cam_l = QtWidgets.QHBoxLayout(self.cam_box)
        cam_l.setContentsMargins(0, 0, 0, 0)
        cam_l.addWidget(QtWidgets.QLabel("Camera ID:"))
        cam_l.addWidget(self.camera_id_spin)
        top.addWidget(self.cam_box)

        self.vid_box = QtWidgets.QWidget()
        vid_l = QtWidgets.QHBoxLayout(self.vid_box)
        vid_l.setContentsMargins(0, 0, 0, 0)
        vid_l.addWidget(QtWidgets.QLabel("Video:"))
        vid_l.addWidget(self.video_path_edit, 1)
        vid_l.addWidget(self.video_browse_btn)
        top.addWidget(self.vid_box)

        self.img_box = QtWidgets.QWidget()
        img_l = QtWidgets.QHBoxLayout(self.img_box)
        img_l.setContentsMargins(0, 0, 0, 0)
        img_l.addWidget(QtWidgets.QLabel("Image:"))
        img_l.addWidget(self.image_path_edit, 1)
        img_l.addWidget(self.image_browse_btn)
        top.addWidget(self.img_box)

        # Skip ratio widgets (hidden unless Video)
        top.addWidget(self.skip_ratio_label)
        top.addWidget(self.skip_ratio_spin)

        # Center stats with fixed-width font
        top.addStretch(1)
        mono = QtGui.QFont("Menlo")  # macOS
        if not QtGui.QFontInfo(mono).fixedPitch():
            mono = QtGui.QFont("Courier New")
        mono.setStyleHint(QtGui.QFont.TypeWriter)
        mono.setFixedPitch(True)

        self.latency_label = QtWidgets.QLabel("Latency:")
        self.latency_value = QtWidgets.QLabel("0.0 ms")
        self.latency_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.latency_value.setMinimumWidth(90)
        self.latency_value.setMaximumWidth(110)
        self.latency_value.setFont(mono)

        self.detect_label = QtWidgets.QLabel("Det:")
        self.detect_value = QtWidgets.QLabel("  0")
        self.detect_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.detect_value.setMinimumWidth(30)
        self.detect_value.setMaximumWidth(40)
        self.detect_value.setFont(mono)

        stat_box = QtWidgets.QWidget()
        stat_l = QtWidgets.QHBoxLayout(stat_box)
        stat_l.setContentsMargins(0, 0, 0, 0)
        stat_l.setSpacing(6)
        stat_l.addWidget(self.latency_label)
        stat_l.addWidget(self.latency_value)
        stat_l.addSpacing(12)
        stat_l.addWidget(self.detect_label)
        stat_l.addWidget(self.detect_value)
        top.addWidget(stat_box)
        top.addStretch(1)

        # Right control group
        top.addWidget(self.use_dynamic_cb)
        top.addWidget(self.use_static_cb)
        top.addWidget(self.run_btn)

        # Bottom images
        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(10, 8, 10, 10)
        bottom.setSpacing(12)
        self.left_pane = ImagePane()
        self.left_pane.setFixedSize(pane_w, pane_h)
        self.right_pane = ImagePane()
        self.right_pane.setFixedSize(pane_w, pane_h)
        bottom.addWidget(self.left_pane, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        bottom.addWidget(self.right_pane, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(top, 0)
        layout.addLayout(bottom, 1)

        apply_dark_theme(self)

        # Pipeline state
        self.source_node: SourceNode | None = None
        self.rectifier_node: DisplayRectifierNode | None = None
        self.dynamic_node: DynamicObjectDetectorNode | None = None
        self.static_node: StaticObjectDetectorNode | None = None
        self.pipeline: Pipeline | None = None

        self.thread: QtCore.QThread | None = None
        self.worker: Worker | None = None

        # Signals
        self.source_combo.currentTextChanged.connect(self._on_source_mode_changed)
        self.video_browse_btn.clicked.connect(self._browse_video)
        self.image_browse_btn.clicked.connect(self._browse_image)
        self.run_btn.clicked.connect(self._on_run_clicked)

        self._on_source_mode_changed(self.source_combo.currentText())

    # ---------- Lifecycle helpers ----------

    def _shutdown_thread(self):
        """
        Safely stop and dispose of the worker/thread if they exist, guarding against
        'wrapped C/C++ object of type QThread has been deleted'.
        """
        # Ask worker to stop first (if still around)
        if self.worker is not None:
            try:
                self.worker.stop()
            except Exception:
                pass

        # Quit and join the thread if it's still a valid object and running
        t = self.thread
        if t is not None:
            try:
                if t.isRunning():
                    t.quit()
                    t.wait(2000)
            except RuntimeError:
                # The underlying C++ object may already be deleted; ignore
                pass
            except Exception:
                pass

        # Drop strong refs so Python doesn't call into a deleted wrapper later
        self.worker = None
        self.thread = None

    # ---------- Qt Events ----------

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        avail_w = self.width() - 40
        pane_w = avail_w // 2
        pane_h = int(pane_w * 9 / 16)
        self.left_pane.setFixedSize(pane_w, pane_h)
        self.right_pane.setFixedSize(pane_w, pane_h)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._shutdown_thread()
        super().closeEvent(event)

    # ---------- UI handlers ----------

    def _on_source_mode_changed(self, mode: str):
        self.cam_box.setVisible(mode == SourceMode.CAMERA)
        self.vid_box.setVisible(mode == SourceMode.VIDEO)
        self.img_box.setVisible(mode == SourceMode.IMAGE)
        # Skip ratio visibility only for video
        self.skip_ratio_label.setVisible(mode == SourceMode.VIDEO)
        self.skip_ratio_spin.setVisible(mode == SourceMode.VIDEO)

    def _browse_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", "", "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        if path:
            self.video_path_edit.setText(path)

    def _browse_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All Files (*)"
        )
        if path:
            self.image_path_edit.setText(path)

    def _on_run_clicked(self):
        text = self.run_btn.text().strip().lower()
        if text in ("run", "restart"):
            self._start_pipeline(restart=(text == "restart"))
        elif text == "stop":
            self._stop_pipeline()

    def _start_pipeline(self, restart: bool = False):
        # Stop & dispose any previous pipeline *before* creating a new one
        self._shutdown_thread()

        self.source_node = SourceNode()
        mode = self.source_combo.currentText()
        self.source_node.set_mode(mode)
        self.source_node.set_camera(self.camera_id_spin.value())
        self.source_node.set_video(self.video_path_edit.text().strip())
        self.source_node.set_image(self.image_path_edit.text().strip())
        self.source_node.set_skip_ratio(self.skip_ratio_spin.value())

        self.rectifier_node = DisplayRectifierNode(target_w=1920, target_h=1080)
        self.rectifier_node.enabled = True
        self.dynamic_node = DynamicObjectDetectorNode(model_path="../../models/best_aliyun.pt", conf=0.25, iou=0.45, imgsz=1280, device=None)
        self.dynamic_node.enabled = self.use_dynamic_cb.isChecked()
        self.static_node = StaticObjectDetectorNode()
        self.static_node.enabled = self.use_static_cb.isChecked()

        self.pipeline = Pipeline(self.source_node, self.rectifier_node, self.dynamic_node, self.static_node)

        try:
            if restart and self.source_node.mode == SourceMode.VIDEO:
                self.source_node.open()
                self.source_node.restart()
            else:
                self.source_node.open()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Source Error", str(e))
            return

        # New thread/worker
        self.thread = QtCore.QThread(self)  # parented for safety
        self.worker = Worker(self.pipeline)
        self.worker.moveToThread(self.thread)

        # Lifecycle wiring
        self.thread.started.connect(self.worker.start)
        self.worker.frameReady.connect(self._on_frame_ready)
        self.worker.stopped.connect(self._on_worker_stopped)
        self.worker.error.connect(self._on_worker_error)
        # Ensure thread quits when worker stops; then delete objects
        self.worker.stopped.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.stopped.connect(self.worker.deleteLater)

        self.thread.start()
        self.run_btn.setText("Stop")

    def _stop_pipeline(self, finalize_only: bool = False):
        # Gracefully stop and dispose; safe if already stopped/deleted
        self._shutdown_thread()
        if not finalize_only:
            self.run_btn.setText("Restart")

    def _on_worker_stopped(self):
        self.run_btn.setText("Restart")

    def _on_worker_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Pipeline Error", msg)
        self.run_btn.setText("Restart")

    @QtCore.pyqtSlot(np.ndarray, np.ndarray, list, float)
    def _on_frame_ready(self, left_bgr: np.ndarray, right_bgr: np.ndarray, objs: list, latency: float):
        self.left_pane.set_frame(left_bgr)
        self.right_pane.set_frame(right_bgr)
        # Update center stats (latency 1 decimal; detections width 3; monospaced)
        self.latency_value.setText(f"{latency*1000:.1f} ms")
        self.detect_value.setText(f"{len(objs):3d}")
