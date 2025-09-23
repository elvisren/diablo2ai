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
        imgsz: int = 1280,
        device: Optional[str] = None,
        task: str = "detect",          # detect/segment/classify/pose/obb
        fuse: bool = True,             # optionally fuse for a tiny speedup
    ):
        super().__init__("dynamic_object_detector")
        self.model_path = model_path
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
        from ultralytics import YOLO  # type: ignore

        self._yolo_model = YOLO(self.model_path, task=self.task)

        # Optional: fuse model for slightly faster inference
        try:
            if self.fuse and hasattr(self._yolo_model, "fuse"):
                self._yolo_model.fuse()
        except Exception:
            raise RuntimeError("Failed to fuse YOLO model")

        self._backend = "yolo"

    def _detect_yolo(self, frame: np.ndarray) -> List[ObjInstance]:
        instances: List[ObjInstance] = []
        # Call YOLO
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

        H, W = frame.shape[:2]
        for xyxy, conf, cls in zip(xyxy_np, conf_np, cls_np):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            cls_id = int(cls)
            name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

            # Create instance
            inst = ObjInstance(
                name=name,
                bbox_xyxy=(x1, y1, x2, y2),
                conf=float(conf),
                source="yolo",
                meta={"cls_id": cls_id},
            )
            instances.append(inst)

        # --- Console prints when detections exist ---
        if instances:
            print(f"[YOLO] Detected {len(instances)} object(s) on frame {W}x{H}:")
            for i, inst in enumerate(instances, 1):
                x1, y1, x2, y2 = inst.bbox_xyxy
                w, h = (x2 - x1), (y2 - y1)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                print(
                    f"  #{i}: cls='{inst.name}' (id={inst.meta.get('cls_id')}) "
                    f"conf={inst.conf:.3f} "
                    f"bbox(xyxy)=({x1},{y1},{x2},{y2}) "
                    f"size(w×h)=({w}×{h}) center=({cx:.1f},{cy:.1f})"
                )

        return instances

    def _detect_motion(self, frame: np.ndarray) -> List[ObjInstance]:
        """Minimal fallback: no motion detection implemented; returns no objects."""
        return []

    def process(self, frame: np.ndarray) -> FrameResult:
        if not self.enabled:
            return FrameResult(frame=frame, objects=[])
        if self._backend == "yolo" and self._yolo_model is not None:
            objs = self._detect_yolo(frame)
        else:
            objs = self._detect_motion(frame)
        return FrameResult(frame=frame, objects=objs)
