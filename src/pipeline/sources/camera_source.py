from __future__ import annotations
import logging, cv2
from ..interfaces import VideoSource
log = logging.getLogger("pipeline.sources.camera")

try:
    from src.pipeline.sources.camera_capture import CameraCapture as _UserCamera
except Exception as _e:
    _UserCamera = None
    log.debug("User CameraCapture not available: %s", _e)

class CameraSource(VideoSource):
    def __init__(self, device: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        self.device, self._w, self._h, self._fps = device, width, height, fps
        self._idx = 0
        self._cam = None

    def open(self) -> None:
        if _UserCamera is not None:
            cam = _UserCamera(self.device, self._w, self._h, self._fps)
            cam.open()
            self._cam = cam
            log.info("Camera opened via CameraCapture: dev=%d size=%dx%d fps~%s",
                     self.device, self._w, self._h, self._fps)
        else:
            cap = cv2.VideoCapture(self.device, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.device)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.device}")
            if self._w: cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
            if self._h: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
            if self._fps: cap.set(cv2.CAP_PROP_FPS, self._fps)
            self._cam = cap
            log.info("Camera opened via cv2: dev=%d size=%dx%d fps~%.1f",
                     self.device, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self):
        if self._cam is None:
            return False, None
        ok, frame = self._cam.read()
        if ok: self._idx += 1
        return ok, frame

    def fps(self) -> float:
        if _UserCamera is not None and isinstance(self._cam, _UserCamera):
            return float(self._fps or 30.0)
        if self._cam is None: return float(self._fps or 30.0)
        v = self._cam.get(cv2.CAP_PROP_FPS) or 0.0
        return float(v if v > 0 else (self._fps or 30.0))

    def frame_count(self) -> int: return 0
    def current_index(self) -> int: return self._idx
    def duration_seconds(self) -> float: return 0.0
    def current_seconds(self) -> float:
        f = self.fps() or 30.0
        return float(self._idx) / float(f)

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.release()
            finally:
                self._cam = None
