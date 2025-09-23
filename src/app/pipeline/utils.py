from __future__ import annotations
import cv2
import numpy as np
from typing import List
from .types import ObjInstance

def draw_instances(img: np.ndarray, instances: List[ObjInstance]) -> np.ndarray:
    out = img.copy()
    for det in instances:
        x1, y1, x2, y2 = map(int, det.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 255), 2, cv2.LINE_AA)
        label = det.name
        if det.conf is not None:
            label = f"{label} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(0, y1 - 8)
        cv2.rectangle(out, (x1, y - th - 6), (x1 + tw + 6, y), (60, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(out, label, (x1 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out
