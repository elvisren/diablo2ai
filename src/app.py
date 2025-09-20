#!/usr/bin/env python3
# qt_app.py
import sys, time, logging, signal, zoneinfo, datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from camera_capture import CameraCapture
from screen_detector import ScreenRectifier

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("qt-app")

# ---------------- Helpers ----------------
def bgr_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)

def shanghai_timestamp() -> str:
    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    now = datetime.datetime.now(tz)
    return now.strftime("%Y%m%d_%H%M%S")

# ---------------- Worker thread ----------------
class VideoWorker(QtCore.QThread):
    frame_pair = QtCore.pyqtSignal(QtGui.QImage, QtGui.QImage)
    rectified_frame_ready = QtCore.pyqtSignal(object, float)

    def __init__(self, device: int, width: int, height: int, fps: int,
                 target_w: int, target_h: int):
        super().__init__()
        self.cam = CameraCapture(device, width, height, fps)
        self.rectifier = ScreenRectifier(target_w, target_h)
        self._running = False

    def run(self):
        try:
            self.cam.open()
        except Exception as e:
            log.error("Camera open failed: %s", e)
            return

        self._running = True
        last_quad = None

        try:
            while self._running:
                ok, frame = self.cam.read()
                if not ok:
                    self.msleep(5)
                    continue

                # ---- FIX: no "or" with NumPy arrays; check None explicitly
                quad = self.rectifier.detect(frame)
                if quad is None and last_quad is not None:
                    quad = last_quad

                if quad is not None:
                    last_quad = quad
                    display_l = self.rectifier.annotate(frame, quad)
                    rectified = self.rectifier.rectify(frame, quad)
                else:
                    display_l = frame
                    rectified = np.zeros((self.rectifier.target_h, self.rectifier.target_w, 3), np.uint8)

                if not self._running:
                    break
                self.frame_pair.emit(bgr_to_qimage(display_l), bgr_to_qimage(rectified))

                if not self._running:
                    break
                self.rectified_frame_ready.emit(rectified, time.perf_counter())

                self.msleep(1)
        finally:
            self.cam.release()

    def stop(self):
        self._running = False
        self.wait(1000)  # join up to 1s

# ---------------- Recorder (constant-FPS) ----------------
class VideoRecorder(QtCore.QObject):
    """
    Writes constant-FPS video. If processing lags, duplicates frames so playback stays smooth.
    """
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
        name = f"diablo2_{shanghai_timestamp()}.mp4"
        path = self.out_dir / name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(path), fourcc, self.fps, frame_size, True)
        if not self.writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter")
        self.frame_size = frame_size
        self.next_t = None
        self.last_frame = None
        return path

    def stop(self):
        if self.writer is not None:
            try:
                self.writer.release()
            finally:
                self.writer = None
                self.frame_size = None
                self.next_t = None
                self.last_frame = None

    def on_frame(self, frame: np.ndarray, tstamp: float):
        if self.writer is None:
            return
        assert self.frame_size is not None
        fh, fw = frame.shape[:2]
        if (fw, fh) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)

        frame_period = 1.0 / self.fps
        if self.next_t is None:
            self.next_t = tstamp

        while self.next_t + 1e-6 < tstamp:
            self.writer.write(self.last_frame if self.last_frame is not None else frame)
            self.next_t += frame_period

        self.writer.write(frame)
        self.last_frame = frame
        self.next_t += frame_period

# ---------------- Aspect-locked bottom pair ----------------
class AspectPair(QtWidgets.QWidget):
    """
    Two image panes side-by-side that:
      - are equal size
      - fill all available bottom space with no gaps
      - keep overall aspect = 2 * (pane_aspect)
    """
    def __init__(self, pane_aspect: float = 16/9, parent=None):
        super().__init__(parent)
        self.pane_aspect = pane_aspect
        self.leftPane  = _ImagePane()
        self.rightPane = _ImagePane()
        self.setContentsMargins(0, 0, 0, 0)
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.leftPane,  0, 0)
        lay.addWidget(self.rightPane, 0, 1)
        lay.setColumnStretch(0, 1)
        lay.setColumnStretch(1, 1)

    def setAspect(self, aspect: float):
        self.pane_aspect = max(0.1, float(aspect))
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        h = 180
        w = int(2 * self.pane_aspect * h)
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        h = 120
        w = int(2 * self.pane_aspect * h)
        return QtCore.QSize(w, h)

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        W = self.width()
        H = self.height()
        desired_h = int(round(W / (2.0 * self.pane_aspect)))
        if desired_h > H:
            desired_h = H
            W_needed = int(round(2.0 * self.pane_aspect * desired_h))
            left_w  = W_needed // 2
            right_w = W_needed - left_w
            x0 = (W - W_needed) // 2
            self.leftPane.setGeometry(x0, 0, left_w, desired_h)
            self.rightPane.setGeometry(x0 + left_w, 0, right_w, desired_h)
        else:
            top = (H - desired_h) // 2
            left_w = W // 2
            right_w = W - left_w
            self.leftPane.setGeometry(0, top, left_w, desired_h)
            self.rightPane.setGeometry(left_w, top, right_w, desired_h)

class _ImagePane(QtWidgets.QWidget):
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
        painter.fillRect(self.rect(), QtCore.Qt.black)
        if self._image is None:
            return
        target = self.rect()
        img_pm = QtGui.QPixmap.fromImage(self._image)
        src_aspect = self._image.width() / max(1, self._image.height())
        dst_aspect = target.width() / max(1, target.height())

        # Fill pane completely (crop if needed), keep aspect
        if abs(src_aspect - dst_aspect) < 1e-3:
            pm = img_pm.scaled(target.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            painter.drawPixmap(target, pm, pm.rect())
        elif src_aspect > dst_aspect:
            h = target.height()
            w = int(round(h * src_aspect))
            pm = img_pm.scaled(w, h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            x = (w - target.width()) // 2
            painter.drawPixmap(target, pm, QtCore.QRect(x, 0, target.width(), target.height()))
        else:
            w = target.width()
            h = int(round(w / src_aspect))
            pm = img_pm.scaled(w, h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            y = (h - target.height()) // 2
            painter.drawPixmap(target, pm, QtCore.QRect(0, y, target.width(), target.height()))

# ---------------- Main window ----------------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Screen Detector (Qt)")
        self.worker: Optional[VideoWorker] = None
        self.recorder = VideoRecorder(Path("./snapshots"), fps=30)
        self.recording_path: Optional[Path] = None
        self._resizing = False

        # --- Controls (top) ---
        self.deviceSpin = QtWidgets.QSpinBox(); self.deviceSpin.setRange(0, 32); self.deviceSpin.setValue(0)
        self.widthSpin  = QtWidgets.QSpinBox(); self.widthSpin.setRange(160, 7680); self.widthSpin.setValue(1920)
        self.heightSpin = QtWidgets.QSpinBox(); self.heightSpin.setRange(120, 4320); self.heightSpin.setValue(1080)
        self.fpsSpin    = QtWidgets.QSpinBox(); self.fpsSpin.setRange(1, 120);      self.fpsSpin.setValue(30)
        self.targetWSpin= QtWidgets.QSpinBox(); self.targetWSpin.setRange(320, 7680); self.targetWSpin.setValue(3840)
        self.targetHSpin= QtWidgets.QSpinBox(); self.targetHSpin.setRange(180, 4320); self.targetHSpin.setValue(2160)

        self.runBtn = QtWidgets.QPushButton("Run")
        self.savePathEdit = QtWidgets.QLineEdit("./snapshots/")
        self.saveBtn = QtWidgets.QPushButton("Save to file")

        self.topBox = QtWidgets.QWidget()
        top = QtWidgets.QGridLayout(self.topBox)
        top.setContentsMargins(8, 8, 8, 4)
        top.setHorizontalSpacing(8)
        r = 0
        top.addWidget(QtWidgets.QLabel("Device"), r, 0); top.addWidget(self.deviceSpin, r, 1)
        top.addWidget(QtWidgets.QLabel("Width"),  r, 2); top.addWidget(self.widthSpin,  r, 3)
        top.addWidget(QtWidgets.QLabel("Height"), r, 4); top.addWidget(self.heightSpin, r, 5)
        top.addWidget(QtWidgets.QLabel("FPS"),    r, 6); top.addWidget(self.fpsSpin,    r, 7)
        r += 1
        top.addWidget(QtWidgets.QLabel("Target W"), r, 0); top.addWidget(self.targetWSpin, r, 1)
        top.addWidget(QtWidgets.QLabel("Target H"), r, 2); top.addWidget(self.targetHSpin, r, 3)
        top.addWidget(self.runBtn, r, 6)
        r += 1
        top.addWidget(QtWidgets.QLabel("Save dir"), r, 0); top.addWidget(self.savePathEdit, r, 1, 1, 5)
        top.addWidget(self.saveBtn, r, 6)

        # --- Bottom aspect-locked pair ---
        pane_aspect = self.targetWSpin.value() / self.targetHSpin.value()
        self.pair = AspectPair(pane_aspect)
        self.pair.setContentsMargins(0, 0, 0, 0)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.topBox, 0)
        self.layout.addWidget(self.pair, 1)

        self.runBtn.clicked.connect(self.toggle_run)
        self.saveBtn.clicked.connect(self.toggle_recording)
        self.targetWSpin.valueChanged.connect(self._on_target_aspect_changed)
        self.targetHSpin.valueChanged.connect(self._on_target_aspect_changed)

        app = QtWidgets.QApplication.instance()
        if app:
            app.aboutToQuit.connect(self.cleanup)

        self.resize(1200, 720)
        QtCore.QTimer.singleShot(0, self._snap_window_to_aspect)

    # Keep window height matched to aspect so no outer gaps appear
    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        if self._resizing:
            return
        self._snap_window_to_aspect()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.cleanup()
        super().closeEvent(event)

    def _snap_window_to_aspect(self):
        self._resizing = True
        try:
            aspect = self.targetWSpin.value() / max(1, self.targetHSpin.value())
            top_h = self.topBox.sizeHint().height()
            total_w = self.width() - (self.layout.contentsMargins().left() + self.layout.contentsMargins().right())
            pair_h = int(round(total_w / (2.0 * aspect)))
            wanted = (self.layout.contentsMargins().top() + top_h + pair_h +
                      self.layout.contentsMargins().bottom())
            if abs(self.height() - wanted) > 1:
                self.resize(self.width(), wanted)
        finally:
            self._resizing = False

    def _on_target_aspect_changed(self):
        a = self.targetWSpin.value() / max(1, self.targetHSpin.value())
        self.pair.setAspect(a)
        self._snap_window_to_aspect()

    # ---- Runtime control
    def toggle_run(self):
        if self.worker and self.worker.isRunning():
            self.stop_worker()
            self.runBtn.setText("Run")
            return

        dev = self.deviceSpin.value()
        w   = self.widthSpin.value()
        h   = self.heightSpin.value()
        fps = self.fpsSpin.value()
        tw  = self.targetWSpin.value()
        th  = self.targetHSpin.value()

        self.worker = VideoWorker(dev, w, h, fps, tw, th)
        self.worker.frame_pair.connect(self.on_frame_pair)
        self.worker.rectified_frame_ready.connect(self.on_rectified_frame)
        self.worker.finished.connect(lambda: log.info("Worker finished"))
        self.worker.start()
        self.runBtn.setText("Stop")

    def stop_worker(self):
        if self.worker:
            try:
                # disconnect first to avoid late emits into deleted widgets
                try: self.worker.frame_pair.disconnect(self.on_frame_pair)
                except TypeError: pass
                try: self.worker.rectified_frame_ready.disconnect(self.on_rectified_frame)
                except TypeError: pass
                self.worker.stop()
            finally:
                self.worker = None
        if self.recording_path is not None:
            self.stop_recording()

    def cleanup(self):
        self.stop_worker()

    # ---- Display
    def on_frame_pair(self, left: QtGui.QImage, right: QtGui.QImage):
        self.pair.leftPane.setImage(left)
        self.pair.rightPane.setImage(right)

    # ---- Recording
    def toggle_recording(self):
        if self.recording_path is None:
            out_dir = Path(self.savePathEdit.text()).expanduser().resolve()
            self.recorder.out_dir = out_dir
            tw = self.targetWSpin.value()
            th = self.targetHSpin.value()
            try:
                self.recording_path = self.recorder.start((tw, th))
                self.saveBtn.setText("Stop saving")
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
            self.saveBtn.setText("Save to file")

    def on_rectified_frame(self, frame, tstamp: float):
        if self.recording_path is None:
            return
        self.recorder.on_frame(frame, tstamp)

# ---------------- main ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)

    # Graceful Ctrl-C: quit event loop instead of raising KeyboardInterrupt
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
