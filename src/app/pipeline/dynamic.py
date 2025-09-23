from __future__ import annotations
import cv2
import numpy as np
from typing import List, Optional
from .base import BaseNode
from .types import FrameResult, ObjInstance

class DynamicObjectDetectorNode(BaseNode):
    """YOLOv11 if available (ultralytics), otherwise motion-based fallback."""
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.25,
        imgsz: int = 640,
        device: Optional[str] = None,
        task: str = "detect",          # <-- make task explicit (detect/segment/classify/pose/obb)
        fuse: bool = True,             # optionally fuse for a tiny speedup
    ):
        super().__init__("dynamic_object_detector")
        self.model_path = model_path or "yolo11x.pt"
        self.conf = float(conf)
        self.iou = float(iou)
        # keep imgsz >= 32 and divisible by 32 for best results
        self.imgsz = max(32, int(imgsz // 32 * 32))
        self.device = device
        self.task = task
        self.fuse = fuse

        self._backend = None
        self._yolo_model = None
        self._prev_gray = None
        self._init_backend()

    def _init_backend(self):
        try:
            from ultralytics import YOLO  # type: ignore

            # Prefer passing task in the constructor; fall back to setting attribute for older versions
            try:
                self._yolo_model = YOLO(self.model_path, task=self.task)
            except TypeError:
                self._yolo_model = YOLO(self.model_path)
                # Older/alternate versions: set task directly if available
                try:
                    setattr(self._yolo_model, "task", self.task)
                except Exception:
                    pass

            # Optional: fuse model for slightly faster inference
            try:
                if self.fuse and hasattr(self._yolo_model, "fuse"):
                    self._yolo_model.fuse()
            except Exception:
                pass

            self._backend = "yolo"
        except Exception:
            # If ultralytics isn't available or model load fails, fall back to motion
            self._backend = "motion"

    def _detect_motion(self, frame: np.ndarray) -> List[ObjInstance]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_gray is None:
            self._prev_gray = gray
            return []
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        instances: List[ObjInstance] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 400:
                continue
            x, y, w, h = cv2.boundingRect(c)
            instances.append(
                ObjInstance(
                    name="moving_object",
                    bbox_xyxy=(x, y, x + w, y + h),
                    conf=0.50,
                    source="motion",
                )
            )
        return instances

    def _detect_yolo(self, frame: np.ndarray) -> List[ObjInstance]:
        instances: List[ObjInstance] = []
        try:
            # You can use either .predict(...) or the __call__ shortcut; keeping .predict for clarity
            res = self._yolo_model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                device=self.device or None,
            )
            if not res:
                return instances
            r0 = res[0]
            names = r0.names
            boxes = getattr(r0, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                return instances

            xyxy_np = boxes.xyxy.cpu().numpy()
            conf_np = boxes.conf.cpu().numpy()
            cls_np = boxes.cls.cpu().numpy()

            for xyxy, conf, cls in zip(xyxy_np, conf_np, cls_np):
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                cls_id = int(cls)
                name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                instances.append(
                    ObjInstance(
                        name=name,
                        bbox_xyxy=(x1, y1, x2, y2),
                        conf=float(conf),
                        source="yolo",
                        meta={"cls_id": cls_id},
                    )
                )
        except Exception:
            # Downgrade to motion if inference fails repeatedly
            self._backend = "motion"
        return instances

    def process(self, frame: np.ndarray) -> FrameResult:
        if not self.enabled:
            return FrameResult(frame=frame, objects=[])
        if self._backend == "yolo" and self._yolo_model is not None:
            objs = self._detect_yolo(frame)
        else:
            objs = self._detect_motion(frame)
        return FrameResult(frame=frame, objects=objs)