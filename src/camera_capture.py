#!/usr/bin/env python3
# camera_capture.py
import cv2
import logging
from typing import List, Tuple, Optional

log = logging.getLogger("camera")

class CameraCapture:
    """
    Cross-version, mac-friendly camera capture using AVFoundation first.
    Thread-safe enough for a single reader thread.
    """
    def __init__(self, device_index: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    @staticmethod
    def list_devices(max_index: int = 8) -> List[int]:
        found = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            cap.release()
            if ok:
                found.append(i)
        if not found:
            log.warning("No cameras found in [0..%d]. Check permissions.", max_index-1)
        return found

    def open(self):
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera device index {self.device_index}")

        if self.width  > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        if self.height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps    > 0: cap.set(cv2.CAP_PROP_FPS,          self.fps)

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        log.info("Opened device %d @ %dx%d ~%.1f fps", self.device_index, aw, ah, afps)
        self.cap = cap

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            finally:
                self.cap = None
