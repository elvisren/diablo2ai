from __future__ import annotations
import logging, cv2
from typing import Optional
from ..interfaces import FrameBundle, PipelineStage
from ...detectors.object_detector import ObjectDetector, DetectionResult

log = logging.getLogger("pipeline.stages.object")

class ObjectStage(PipelineStage):
    def __init__(self, detector: Optional[ObjectDetector] = None):
        self.detector = detector or ObjectDetector(model_path=None, conf=0.25, iou=0.45, max_det=200)

    def process(self, bundle: FrameBundle) -> FrameBundle:
        rect = bundle.rectified_bgr
        if rect is None:
            return bundle
        try:
            result: DetectionResult = self.detector.detect(rect)
            annotated = result.annotated if hasattr(result, "annotated") else rect
            bundle.right_preview_bgr = annotated
        except Exception as e:
            log.error("Detection error: %s", e)
            ann = rect.copy()
            cv2.putText(ann, f"Detection error: {e}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
            bundle.right_preview_bgr = ann
        return bundle
