from __future__ import annotations
import numpy as np
from typing import Optional
from .base import BaseNode
from .screen_detector import ScreenRectifier
from .types import FrameResult

# Reuse your robust monitor finder/rectifier (user-provided module)


class DisplayRectifierNode(BaseNode):
    """
    Finds the monitor using ScreenRectifier and returns the rectified display
    (e.g., 1920x1080) as the frame for downstream nodes.

    If no display is found, passes the original frame through unchanged.

    NOTE: Per request, this node does NOT output any bounding boxes/objects.
    """
    def __init__(self, target_w: int = 1920, target_h: int = 1080, trim_bezel: bool = True):
        super().__init__("display_rectifier")
        self.rect = ScreenRectifier(
            target_w=target_w,
            target_h=target_h,
            trim_bezel=trim_bezel,
            max_trim_px=40,
        )

    def process(self, frame) -> FrameResult:
        if not self.enabled:
            return FrameResult(frame=frame, objects=[])

        quad = self.rect.detect(frame)
        if quad is None:
            # Not found: pass-through (downstream can still run)
            return FrameResult(frame=frame, objects=[])

        # Rectify to target size and forward only the warped frame
        warped = self.rect.rectify(frame, quad)
        return FrameResult(frame=warped, objects=[])
