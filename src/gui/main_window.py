from __future__ import annotations
import datetime, zoneinfo
from pathlib import Path
from typing import Optional, Literal
import logging, cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from ..pipeline.pipeline import Pipeline
from ..pipeline.sources import CameraSource, FileSource
from ..pipeline.stages import ScreenStage, ObjectStage
from ..pipeline.recorder import VideoRecorder
from .widgets import ImagePane, AspectContainer

log = logging.getLogger("gui")

def bgr_to_qimage(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QtGui.QImage(rgb.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
    return img.copy()

def sh_timestamp() -> str:
    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    now = datetime.datetime.now(tz)
    return now.strftime("%Y%m%d_%H%M%S")

class IngestThread(QtCore.QThread):
    frame_pair = QtCore.pyqtSignal(QtGui.QImage, QtGui.QImage)
    file_progress = QtCore.pyqtSignal(int, int, int, float, float)
    file_finished = QtCore.pyqtSignal()

    def __init__(self, pipeline: Pipeline, mode: Literal["camera","file"]):
        super().__init__()
        self.pipeline = pipeline
        self.mode = mode
        self._running = False

    def run(self):
        self._running = True
        try: self.pipeline.open()
        except Exception as e:
            log.error("Failed to open source: %s", e); return

        src = self.pipeline.source
        total = src.frame_count() if self.mode == "file" else 0
        try:
            while self._running:
                ok, bundle = self.pipeline.step()
                if not ok:
                    if self.mode == "file":
                        self.file_progress.emit(100, total, total, src.current_seconds(), src.duration_seconds())
                        self.file_finished.emit()
                    break
                left = bundle.left_preview_bgr if bundle.left_preview_bgr is not None else bundle.raw_bgr
                right_pref = bundle.right_preview_bgr if bundle.right_preview_bgr is not None else None
                right = right_pref if right_pref is not None else (bundle.rectified_bgr if bundle.rectified_bgr is not None else bundle.raw_bgr)
                if left is not None and right is not None:
                    self.frame_pair.emit(bgr_to_qimage(left), bgr_to_qimage(right))
                self.msleep(1)
        finally:
            try: self.pipeline.close()
            except Exception: pass

    def stop(self):
        self._running = False
        self.wait(1000)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diablo II Analyzer — OOP Pipeline")

        # Controls (left)
        self.deviceSpin = QtWidgets.QSpinBox(); self.deviceSpin.setRange(0, 32); self.deviceSpin.setValue(0)
        self.widthSpin  = QtWidgets.QSpinBox(); self.widthSpin.setRange(160, 7680); self.widthSpin.setValue(1920)
        self.heightSpin = QtWidgets.QSpinBox(); self.heightSpin.setRange(120, 4320); self.heightSpin.setValue(1080)
        self.fpsSpin    = QtWidgets.QSpinBox(); self.fpsSpin.setRange(1, 240); self.fpsSpin.setValue(30)
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
        leftTop.addWidget(camBox)
        leftTop.addWidget(fileBox)
        self.leftTopW = QtWidgets.QWidget(); self.leftTopW.setLayout(leftTop)

        self.leftPane = ImagePane()

        # Right controls
        self.targetWSpin= QtWidgets.QSpinBox(); self.targetWSpin.setRange(320, 7680); self.targetWSpin.setValue(3840)
        self.targetHSpin= QtWidgets.QSpinBox(); self.targetHSpin.setRange(180, 4320); self.targetHSpin.setValue(2160)
        self.saveDirEdit = QtWidgets.QLineEdit("./snapshots/")
        self.recBtn = QtWidgets.QPushButton("Start recording")

        dispBox = QtWidgets.QGroupBox("Display & Record")
        dGrid = QtWidgets.QGridLayout(dispBox); dGrid.setContentsMargins(8,8,8,8); dGrid.setHorizontalSpacing(8)
        dGrid.addWidget(QtWidgets.QLabel("Target W"), 0,0); dGrid.addWidget(self.targetWSpin, 0,1)
        dGrid.addWidget(QtWidgets.QLabel("Target H"), 0,2); dGrid.addWidget(self.targetHSpin, 0,3)
        dGrid.addWidget(QtWidgets.QLabel("Save dir"), 1,0); dGrid.addWidget(self.saveDirEdit, 1,1,1,2); dGrid.addWidget(self.recBtn, 1,3)

        rightTop = QtWidgets.QVBoxLayout()
        rightTop.addWidget(dispBox)
        self.rightTopW = QtWidgets.QWidget(); self.rightTopW.setLayout(rightTop)

        self.rightPane = ImagePane()

        self.leftAspect  = AspectContainer(3840, 2160, self.leftPane)
        self.rightAspect = AspectContainer(3840, 2160, self.rightPane)

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0,0,0,0)
        grid.setHorizontalSpacing(0); grid.setVerticalSpacing(0)
        grid.addWidget(self.leftTopW, 0, 0); grid.addWidget(self.rightTopW, 0, 1)
        grid.addWidget(self.leftAspect, 1, 0); grid.addWidget(self.rightAspect, 1, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 0);    grid.setRowStretch(1, 1)

        # Minimum heights for buttons
        self._base_controls_h = max(self.leftTopW.sizeHint().height(), self.rightTopW.sizeHint().height())
        pad = 12
        self._controls_h_min = self._base_controls_h + pad
        for w in (self.leftTopW, self.rightTopW):
            w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
            w.setMinimumHeight(self._controls_h_min)

        # Aspect & resizing guard
        self._preview_aspect = 32.0 / 9.0
        self._resizing_guard = False

        min_w = 1200
        min_h = self._controls_h_min + int(round(min_w / self._preview_aspect))
        self.setMinimumSize(min_w, min_h)

        # Signals
        self.camBtn.clicked.connect(self.toggle_camera)
        self.fileBtn.clicked.connect(self.toggle_file)
        self.browseBtn.clicked.connect(self.choose_video)
        self.recBtn.clicked.connect(self.toggle_recording)

        # Dark theme + initial size
        self._apply_dark_theme()
        self.resize(2000, self._controls_h_min + int(round(2000 / self._preview_aspect)))

        self.ingest: Optional['IngestThread'] = None
        self.pipeline: Optional[Pipeline] = None
        self.recorder = VideoRecorder(Path("./snapshots"), fps=30)
        self.recording = False

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

    def resizeEvent(self, e: QtGui.QResizeEvent):
        if self._resizing_guard:
            return super().resizeEvent(e)
        self._resizing_guard = True
        try:
            new_w, new_h = e.size().width(), e.size().height()
            old_w, old_h = e.oldSize().width(), e.oldSize().height()
            controls_h = max(self._controls_h_min, self.leftTopW.height(), self.rightTopW.height())
            if abs(new_w - old_w) > abs(new_h - old_h):
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

    def _make_pipeline(self, mode: str, **kwargs) -> Pipeline:
        if mode == "camera":
            src = CameraSource(kwargs.get("device",0), kwargs.get("width",1920), kwargs.get("height",1080), kwargs.get("fps",30))
        else:
            src = FileSource(Path(kwargs["path"]))
        stages = [ScreenStage(self.targetWSpin.value(), self.targetHSpin.value()), ObjectStage()]
        return Pipeline(src, stages)

    def _start_ingest(self, mode: str, **kwargs):
        if getattr(self, 'ingest', None):
            self.ingest.stop(); self.ingest = None
        self.pipeline = self._make_pipeline(mode, **kwargs)
        self.ingest = IngestThread(self.pipeline, mode)  # type: ignore
        self.ingest.frame_pair.connect(self.on_frame_pair)
        self.ingest.file_progress.connect(self.on_file_progress)
        self.ingest.file_finished.connect(self.on_file_finished)
        self.ingest.start()

    def on_frame_pair(self, left: QtGui.QImage, right: QtGui.QImage):
        self.leftPane.setImage(left)
        self.rightPane.setImage(right)

    def on_file_progress(self, pct: int, idx: int, total: int, t_cur: float, t_tot: float):
        if total > 0:
            self.fileProgress.setRange(0, 100)
            self.fileProgress.setValue(max(0, min(100, pct)))
            self.fileTime.setText(f"{self._fmt(t_cur)} / {self._fmt(t_tot)}")
        else:
            self.fileProgress.setRange(0, 0)
            self.fileTime.setText(f"{self._fmt(t_cur)} / --:--")

    def on_file_finished(self):
        self.fileProgress.setRange(0, 100); self.fileProgress.setValue(100)
        self.fileBtn.setText("Play from file")

    def _fmt(self, s: float) -> str:
        s = max(0, int(round(s))); m, s = divmod(s, 60); h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def toggle_camera(self):
        if getattr(self, 'ingest', None) and self.ingest.mode == "camera":
            self.ingest.stop(); self.ingest = None
            self.camBtn.setText("Read from camera")
            return
        self._start_ingest("camera",
            device=self.deviceSpin.value(),
            width=self.widthSpin.value(),
            height=self.heightSpin.value(),
            fps=self.fpsSpin.value()
        )
        self.camBtn.setText("Stop camera")
        self.fileBtn.setText("Play from file")

    def toggle_file(self):
        if getattr(self, 'ingest', None) and self.ingest.mode == "file":
            self.ingest.stop(); self.ingest = None
            self.fileBtn.setText("Play from file")
            return
        path = self.videoPathEdit.text().strip()
        if not path:
            self.choose_video(); path = self.videoPathEdit.text().strip()
            if not path: return
        self._start_ingest("file", path=path)
        self.fileBtn.setText("Stop from file")
        self.camBtn.setText("Read from camera")

    def choose_video(self):
        start_dir = Path(self.saveDirEdit.text()).expanduser().resolve()
        if not start_dir.exists():
            start_dir = Path("./snapshots").resolve()
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video", str(start_dir),
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)")
        if fname: self.videoPathEdit.setText(fname)

    def toggle_recording(self):
        if not hasattr(self, 'recorder'): return
        if not getattr(self, 'recording', False):
            out_dir = Path(self.saveDirEdit.text()).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            self.recorder.out_dir = out_dir
            self.recorder.start((self.targetWSpin.value(), self.targetHSpin.value()),
                                f"detected_{sh_timestamp()}.mp4")
            self.recording = True
            self.recBtn.setText("Stop recording")
        else:
            self.recorder.stop()
            self.recording = False
            self.recBtn.setText("Start recording")
