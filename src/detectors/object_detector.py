#!/usr/bin/env python3
# detectors/object_detector.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("object-detector")

# ---------- Public data types ----------
@dataclass
class Detection:
    xyxy: Tuple[int, int, int, int]  # (x1,y1,x2,y2)
    conf: float
    cls: int
    name: str

@dataclass
class DetectionResult:
    annotated: np.ndarray                 # BGR image with boxes drawn
    detections: List[Detection]           # raw detections (for your later use)


class ObjectDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 100,
        device: Optional[str] = None,
    ):
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.device = device

        # --- NEW: Resolve default model candidates if model_path not given ---
        self.model_path = self._resolve_model_path(model_path)
        self._use_ultralytics = False
        self._names = None
        self._model = None
        self._init_backend()

    def _resolve_model_path(self, user_path: Optional[str]) -> str:
        if user_path:  # explicit wins
            return user_path

        here = Path(__file__).resolve()                 # .../src/detectors/object_detector.py
        src_root = here.parent.parent                   # .../src
        candidates = [
            src_root / "models" / "yolo11m.pt",         # …/src/models/yolo11m.pt   (your layout)
            Path.cwd() / "models" / "yolo11m.pt",       # …/<cwd>/models/yolo11m.pt (fallback)
            here.parent / "yolo11m.pt",                 # …/src/detectors/yolo11m.pt (last resort)
        ]
        for p in candidates:
            if p.exists():
                log.info("Using YOLO model: %s", p)
                return str(p)
        # If none exist, still return the first candidate so Ultralytics can error clearly
        log.warning("YOLO weights not found in default locations; expected one of: %s", ", ".join(map(str, candidates)))
        return str(candidates[0])

    # ---------- Backend selection ----------
    def _init_backend(self):
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self.model_path)
            # Try to get class names (dict mapping id->name)
            try:
                # Newer Ultralytics models expose .names at model or first task model
                self._names = getattr(self._model.model, "names", None) or getattr(self._model, "names", None)
            except Exception:
                self._names = None
            self._use_ultralytics = True
            log.info("Ultralytics loaded: %s", self.model_path)
        except Exception as e:
            self._model = None
            self._use_ultralytics = False
            log.warning("Ultralytics not available (%s). Falling back to simple POC detector.", e)

    def warmup(self):
        """Optional warmup; safe to call even without Ultralytics."""
        if not self._use_ultralytics:
            return
        try:
            # Small blank to compile kernels
            dummy = np.zeros((320, 320, 3), np.uint8)
            _ = self.detect(dummy)
        except Exception:
            pass

    # ---------- Main API ----------
    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        if self._use_ultralytics:
            return self._detect_yolo(frame_bgr)
        else:
            return self._detect_fallback(frame_bgr)

    # ---------- YOLO path ----------
    def _detect_yolo(self, img_bgr: np.ndarray) -> DetectionResult:
        # Ultralytics expects RGB by default
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self._model.predict(
            img_rgb,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
            device=self.device if self.device else None,
        )

        annotated = img_bgr.copy()
        outs: List[Detection] = []
        if not results:
            return DetectionResult(annotated=annotated, detections=outs)

        r = results[0]
        try:
            boxes = r.boxes
        except Exception:
            boxes = None

        if boxes is not None:
            xyxy = getattr(boxes, "xyxy", None)
            confs = getattr(boxes, "conf", None)
            clss = getattr(boxes, "cls", None)

            if xyxy is not None:
                xyxy = xyxy.cpu().numpy().astype(np.int32)
                confs = (confs.cpu().numpy() if confs is not None else np.zeros((len(xyxy),), dtype=float))
                clss = (clss.cpu().numpy().astype(int) if clss is not None else np.zeros((len(xyxy),), dtype=int))

                for (x1, y1, x2, y2), cf, c in zip(xyxy, confs, clss):
                    name = self._names.get(int(c), str(int(c))) if isinstance(self._names, dict) else str(int(c))
                    outs.append(Detection((int(x1), int(y1), int(x2), int(y2)), float(cf), int(c), name))
                    self._draw_box(annotated, (x1, y1, x2, y2), name, cf)

        return DetectionResult(annotated=annotated, detections=outs)

    # ---------- Fallback POC (very crude) ----------
    def _detect_fallback(self, img_bgr: np.ndarray) -> DetectionResult:
        """
        Very simple moving-edge boxes finder to prove the interface.
        Replace ASAP with a real model if Ultralytics isn't available.
        """
        annotated = img_bgr.copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 120, 240)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outs: List[Detection] = []
        H, W = gray.shape[:2]
        area_th = 0.002 * W * H
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < area_th:
                continue
            self._draw_box(annotated, (x, y, x + w, y + h), "obj", 0.30)
            outs.append(Detection((x, y, x + w, y + h), 0.30, 0, "obj"))
        return DetectionResult(annotated=annotated, detections=outs)

    # ---------- Drawing ----------
    @staticmethod
    def _draw_box(img: np.ndarray, xyxy: Tuple[int, int, int, int], label: str, conf: float):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 220, 255), 2, cv2.LINE_AA)
        txt = f"{label} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(0, y1 - 8)
        cv2.rectangle(img, (x1, y - th - 6), (x1 + tw + 6, y), (60, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(img, txt, (x1 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
