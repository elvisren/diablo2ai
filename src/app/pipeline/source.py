from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

class SourceMode:
    CAMERA = "Camera"
    VIDEO = "Video"
    IMAGE = "Image"

class SourceNode:
    """Provides frames from camera/video/image.
    For Video: after each pipeline iteration of duration T, it can skip ahead by (skip_ratio * T) seconds.
    """
    def __init__(self):
        self.mode = SourceMode.CAMERA
        self.camera_id = 0
        self.video_path: str | None = None
        self.image_path: str | None = None
        self.skip_ratio: float = 1.0

        self._cap: cv2.VideoCapture | None = None
        self._still: np.ndarray | None = None
        self._video_fps: float = 0.0

    def set_mode(self, mode: str):
        self.mode = mode

    def set_camera(self, cam_id: int):
        self.camera_id = int(cam_id)

    def set_video(self, path: str):
        self.video_path = path or None

    def set_image(self, path: str):
        self.image_path = path or None

    def set_skip_ratio(self, r: float):
        self.skip_ratio = max(0.1, float(r))

    def open(self):
        self.close()
        if self.mode == SourceMode.CAMERA:
            self._cap = cv2.VideoCapture(self.camera_id)
            if not self._cap.isOpened():
                raise RuntimeError("Could not open camera.")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
        elif self.mode == SourceMode.VIDEO:
            if not self.video_path or not Path(self.video_path).exists():
                raise RuntimeError("Video path not set or not found.")
            self._cap = cv2.VideoCapture(self.video_path)
            if not self._cap.isOpened():
                raise RuntimeError("Could not open video.")
            fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
            self._video_fps = fps if fps > 1e-3 else 30.0
        elif self.mode == SourceMode.IMAGE:
            if not self.image_path or not Path(self.image_path).exists():
                raise RuntimeError("Image path not set or not found.")
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Failed to read image.")
            self._still = img
        else:
            raise RuntimeError("Unknown source mode.")

    def restart(self):
        if self.mode == SourceMode.VIDEO and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.open()

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._still = None
        self._video_fps = 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.mode in (SourceMode.CAMERA, SourceMode.VIDEO):
            if self._cap is None:
                return False, None
            ok, frame = self._cap.read()
            return ok, frame
        elif self.mode == SourceMode.IMAGE:
            if self._still is None:
                return False, None
            return True, self._still.copy()
        return False, None

    def skip_ahead_by_seconds(self, dt: float):
        if self.mode != SourceMode.VIDEO or self._cap is None or self._video_fps <= 0:
            return
        frames_to_skip = int(round(self._video_fps * max(0.0, dt)))
        for _ in range(frames_to_skip):
            self._cap.grab()
