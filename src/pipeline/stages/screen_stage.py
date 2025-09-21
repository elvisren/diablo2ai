from __future__ import annotations
import logging, numpy as np
from ..interfaces import FrameBundle, PipelineStage
from src.pipeline.stages.screen_detector import ScreenRectifier

log = logging.getLogger("pipeline.stages.screen")

class ScreenStage(PipelineStage):
    def __init__(self, target_w: int = 3840, target_h: int = 2160):
        self.rectifier = ScreenRectifier(target_w, target_h)

    def process(self, bundle: FrameBundle) -> FrameBundle:
        frame = bundle.raw_bgr
        if frame is None:
            return bundle
        quad = self.rectifier.detect(frame)
        if quad is None:
            left = frame
            rectified = np.zeros((self.rectifier.target_h, self.rectifier.target_w, 3), np.uint8)
        else:
            left = self.rectifier.annotate(frame, quad)
            rectified = self.rectifier.rectify(frame, quad)
        bundle.left_preview_bgr = left
        bundle.rectified_bgr = rectified
        return bundle
