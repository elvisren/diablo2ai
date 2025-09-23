from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from .base import BaseNode
from .types import FrameResult, ObjInstance


class StaticObjectDetectorNode(BaseNode):
    """
    Detects Diablo II: Resurrected static HUD elements:
      - Left orb (HP / blood) and its text "生命 cur/max"
      - Right orb (Mana) and its text "法力 cur/max"

    Strategy
    --------
    1) Use your given normalized hints to define two ROIs where the orbs live.
       Then refine each orb center/radius with HoughCircles/contours (robust to
       color/fill changes since edges persist even when empty).
    2) From the refined circle, compute the text ROI using your offsets:
         - HP text center: directly above the orb: dy = 0.35/16.5 (of H)
         - text width     ≈ 2.4/29 (of W)
         - height: adaptive (based on ROI), with safe clamps.
    3) OCR the cropped text if a backend is available (EasyOCR > Tesseract).
       We parse numbers with regex; if OCR is unavailable/unsuccessful, we still
       emit boxes and leave values as None.
    4) Return an annotated frame and a FrameResult with four ObjInstances:
         - "blood_ball", "blood_ball_text", "mana_ball", "mana_ball_text"

    Internal state
    --------------
    self.stats = {
        "blood": {"current": int|None, "max": int|None},
        "mana":  {"current": int|None, "max": int|None},
    }
    """

    # ---- ratios from user (relative to full image W,H) ----
    # red orb hint center:
    RED_CX_N = 8.2 / 29.0
    RED_CY_N = 15.4 / 16.5
    # mana orb is horizontally mirrored:
    BLUE_CX_N = 1.0 - RED_CX_N
    BLUE_CY_N = RED_CY_N
    # text block geometry (relative to full frame, anchored by orb):
    TEXT_DY_UP_N = 0.35 / 16.5    # center ABOVE orb top
    TEXT_W_N = 2.4 / 29.0         # width of text band (of W)
    TEXT_H_N = 0.9 * (0.35 / 16.5)  # heuristic height; tuned from your hint

    def __init__(self):
        super().__init__("static_object_detector")
        self._ocr_backend = None  # "easyocr" | "tesseract" | None
        self._init_ocr_backend()
        self.stats: Dict[str, Dict[str, Optional[int]]] = {
            "blood": {"current": None, "max": None},
            "mana": {"current": None, "max": None},
        }

    # ------------------------- Public API -------------------------

    def process(self, frame: np.ndarray) -> FrameResult:
        if not self.enabled:
            return FrameResult(frame=frame, objects=[])

        out = frame.copy()
        H, W = out.shape[:2]
        objects: List[ObjInstance] = []

        # Detect HP orb
        hp_center, hp_radius, hp_bbox = self._locate_orb(
            out, W, H, (self.RED_CX_N, self.RED_CY_N)
        )
        if hp_center is not None and hp_radius is not None and hp_bbox is not None:
            x1, y1, x2, y2 = hp_bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            objects.append(
                ObjInstance(
                    name="blood_ball",
                    bbox_xyxy=(x1, y1, x2, y2),
                    conf=1.0,
                    source="static",
                    meta={"center": hp_center, "radius": hp_radius},
                )
            )

            # HP text box
            tb = self._text_bbox_from_orb(W, H, hp_center, hp_radius)
            tx1, ty1, tx2, ty2 = tb
            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)

            text_img = self._prep_text_roi(out, tb)
            hp_text, hp_nums = self._ocr_and_parse(text_img, is_hp=True)

            # update internal stats
            if hp_nums is not None:
                cur, mx = hp_nums
                self.stats["blood"]["current"] = cur
                self.stats["blood"]["max"] = mx

            objects.append(
                ObjInstance(
                    name="blood_ball_text",
                    bbox_xyxy=(tx1, ty1, tx2, ty2),
                    conf=1.0 if hp_text else 0.0,
                    source=self._ocr_backend or "static",
                    meta={"raw_text": hp_text, "parsed": self.stats["blood"]},
                )
            )

        # Detect Mana orb
        mp_center, mp_radius, mp_bbox = self._locate_orb(
            out, W, H, (self.BLUE_CX_N, self.BLUE_CY_N)
        )
        if mp_center is not None and mp_radius is not None and mp_bbox is not None:
            x1, y1, x2, y2 = mp_bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
            objects.append(
                ObjInstance(
                    name="mana_ball",
                    bbox_xyxy=(x1, y1, x2, y2),
                    conf=1.0,
                    source="static",
                    meta={"center": mp_center, "radius": mp_radius},
                )
            )

            tb = self._text_bbox_from_orb(W, H, mp_center, mp_radius)
            tx1, ty1, tx2, ty2 = tb
            cv2.rectangle(out, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)

            text_img = self._prep_text_roi(out, tb)
            mp_text, mp_nums = self._ocr_and_parse(text_img, is_hp=False)

            if mp_nums is not None:
                cur, mx = mp_nums
                self.stats["mana"]["current"] = cur
                self.stats["mana"]["max"] = mx

            objects.append(
                ObjInstance(
                    name="mana_ball_text",
                    bbox_xyxy=(tx1, ty1, tx2, ty2),
                    conf=1.0 if mp_text else 0.0,
                    source=self._ocr_backend or "static",
                    meta={"raw_text": mp_text, "parsed": self.stats["mana"]},
                )
            )

        return FrameResult(frame=out, objects=objects)

    # ---------------------- Orb localization ----------------------

    def _locate_orb(
        self,
        img: np.ndarray,
        W: int,
        H: int,
        hint_center_n: Tuple[float, float],
    ) -> Tuple[Optional[Tuple[int, int]], Optional[int], Optional[Tuple[int, int, int, int]]]:
        """
        Returns (center(x,y), radius_px, bbox_xyxy) or (None,...)
        """
        cx_hint = int(hint_center_n[0] * W)
        cy_hint = int(hint_center_n[1] * H)

        # A generous ROI around the hint (orb size ~ 0.20*H in many captures)
        roi_half_w = int(0.16 * W)
        roi_half_h = int(0.16 * H)
        x1 = np.clip(cx_hint - roi_half_w, 0, W - 1)
        y1 = np.clip(cy_hint - roi_half_h, 0, H - 1)
        x2 = np.clip(cx_hint + roi_half_w, 0, W - 1)
        y2 = np.clip(cy_hint + roi_half_h, 0, H - 1)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None, None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Edge-friendly preproc (robust to fill/empty)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.Canny(gray, 40, 120)

        # Hough search for circles inside ROI
        # radius range ~ [0.08H, 0.16H], scaled to ROI
        minR = int(0.07 * H)
        maxR = int(0.16 * H)
        # Convert to ROI scale by noting ROI is a crop of full image
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(0.25 * min(roi.shape[:2])),
            param1=120,
            param2=20,
            minRadius=int(minR * (roi.shape[0] / H)),
            maxRadius=int(maxR * (roi.shape[0] / H)),
        )

        best_center = None
        best_radius = None

        if circles is not None and len(circles) > 0:
            c = np.uint16(np.around(circles))[0]
            # pick circle closest to hint center within ROI coords
            hx = cx_hint - x1
            hy = cy_hint - y1
            dists = [(int(x), int(y), int(r), (x - hx) ** 2 + (y - hy) ** 2) for x, y, r in c]
            x, y, r, _ = min(dists, key=lambda t: t[3])
            best_center = (x1 + x, y1 + y)
            best_radius = int(r)
        else:
            # Fallback: largest circular-ish contour near hint
            cnt_center, cnt_radius = self._largest_roundish_contour(edges)
            if cnt_center is not None and cnt_radius is not None:
                best_center = (x1 + cnt_center[0], y1 + cnt_center[1])
                best_radius = int(cnt_radius)

        if best_center is None or best_radius is None:
            return None, None, None

        bx1 = int(best_center[0] - best_radius)
        by1 = int(best_center[1] - best_radius)
        bx2 = int(best_center[0] + best_radius)
        by2 = int(best_center[1] + best_radius)
        bx1, by1 = max(bx1, 0), max(by1, 0)
        bx2, by2 = min(bx2, W - 1), min(by2, H - 1)

        return best_center, best_radius, (bx1, by1, bx2, by2)

    @staticmethod
    def _largest_roundish_contour(edges: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1000:  # ignore tiny
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            r = int(r)
            circ_area = math.pi * (r ** 2)
            fill_ratio = area / (circ_area + 1e-6)
            # Prefer well-filled, reasonably circular shapes
            score = fill_ratio * area
            if score > best_score:
                best_score = score
                best = ((int(x), int(y)), r)
        return best if best is not None else (None, None)

    # ---------------------- Text ROI + OCR -----------------------

    def _text_bbox_from_orb(
        self, W: int, H: int, orb_center: Tuple[int, int], orb_radius: int
    ) -> Tuple[int, int, int, int]:
        cx, cy = orb_center
        orb_top_y = cy - orb_radius

        text_cy = int(orb_top_y - H * self.TEXT_DY_UP_N)
        text_w = int(W * self.TEXT_W_N)
        text_h = max(18, int(H * self.TEXT_H_N))

        x1 = int(cx - text_w // 2)
        x2 = int(cx + text_w // 2)
        y1 = int(text_cy - text_h // 2)
        y2 = int(text_cy + text_h // 2)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        return (x1, y1, x2, y2)

    @staticmethod
    def _prep_text_roi(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return roi

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Contrast & binarize (works with various capture qualities)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Normalize contrast and then adaptive threshold
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        thr = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 7)

        # Enlarge a bit to help OCR
        scale = 2
        enlarged = cv2.resize(thr, (thr.shape[1] * scale, thr.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
        return enlarged

    def _init_ocr_backend(self) -> None:
        try:
            import pytesseract  # CPU-only
            self._ocr_backend = "tesseract"
            return
        except Exception:
            pass

        # Fallback to EasyOCR only if you really want it:
        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            self._ocr_backend = "easyocr"
        except Exception:
            self._ocr_backend = None

    def _ocr_and_parse(
        self, text_img: np.ndarray, is_hp: bool
    ) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """
        Returns (raw_text, (current, max)) when successful, else (raw_text_or_None, None).
        """
        raw = None
        if text_img is None or text_img.size == 0:
            return None, None

        # Run OCR
        try:
            if self._ocr_backend == "easyocr":
                # result is list of [bbox, text, conf]
                out = self._easyocr_reader.readtext(text_img)
                # Concatenate best strips
                raw = "".join([t for _, t, c in out if c >= 0.2]).strip()
            elif self._ocr_backend == "tesseract":
                import pytesseract
                raw = pytesseract.image_to_string(text_img, lang="chi_sim+eng", config="--psm 7").strip()
        except Exception:
            raw = None

        # Parse numbers like: "生命 1234/5678" or "法力 375/375"
        # Be tolerant to OCR noise/spaces.
        import re

        if raw:
            # Normalize common OCR glitches
            s = raw.replace(" ", "").replace("：", ":").replace("|", "/").replace("\\", "/")
            s = s.replace("，", ",").replace("。", ".")
            # Remove stray non-digit between numbers
            m = re.search(r"(\d{1,6})\s*[/]\s*(\d{1,6})", s)
            if m:
                cur = int(m.group(1))
                mx = int(m.group(2))
                return raw, (cur, mx)

        # If label word was recognized but numbers failed, still return raw
        return raw, None
