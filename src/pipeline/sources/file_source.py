from __future__ import annotations
from typing import Optional
from pathlib import Path
import logging, cv2
from ..interfaces import VideoSource
log = logging.getLogger("pipeline.sources.file")

class FileSource(VideoSource):
    def __init__(self, path: Path):
        self.path = str(path)
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.path}")
        self.cap = cap
        log.info("Video opened: %s @ %.3f fps, frames=%d",
                 self.path, cap.get(cv2.CAP_PROP_FPS) or 0.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

    def read(self):
        if self.cap is None: return False, None
        return self.cap.read()

    def fps(self) -> float:
        if self.cap is None: return 0.0
        v = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        return float(v if v > 0 else 30.0)

    def frame_count(self) -> int:
        if self.cap is None: return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def current_index(self) -> int:
        if self.cap is None: return 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

    def duration_seconds(self) -> float:
        f = self.frame_count(); fps = self.fps() or 30.0
        return float(f) / float(fps) if f > 0 else 0.0

    def current_seconds(self) -> float:
        if self.cap is None: return 0.0
        ms = self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
        return float(ms) / 1000.0

    def close(self) -> None:
        if self.cap is not None:
            try: self.cap.release()
            finally:
                self.cap = None
