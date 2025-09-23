from __future__ import annotations
from .base import BaseNode
from .types import FrameResult

class StaticObjectDetectorNode(BaseNode):
    """Dummy static detector (no-op)."""
    def __init__(self):
        super().__init__("static_object_detector")

    def process(self, frame):
        if not self.enabled:
            return FrameResult(frame=frame, objects=[])
        return FrameResult(frame=frame, objects=[])
