from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import cv2, numpy as np, logging
log = logging.getLogger("pipeline.recorder")

class VideoRecorder:
    def __init__(self, out_dir: Path, fps: int = 30):
        self.out_dir = out_dir
        self.fps = max(1, int(fps))
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_size: Optional[Tuple[int, int]] = None

    def start(self, frame_size: tuple[int, int], filename: str) -> Path:
        self.stop()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / filename
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(path), fourcc, self.fps, frame_size, True)
        if not self.writer or not self.writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter")
        self.frame_size = frame_size
        return path

    def write(self, frame: np.ndarray):
        if self.writer is None: return
        fh, fw = frame.shape[:2]
        W, H = self.frame_size or (fw, fh)
        if (fw, fh) != (W, H):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        self.writer.write(frame)

    def stop(self):
        if self.writer is not None:
            try: self.writer.release()
            finally:
                self.writer = None
                self.frame_size = None
