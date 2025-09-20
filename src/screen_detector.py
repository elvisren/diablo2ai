#!/usr/bin/env python3
# screen_detector.py
import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple

log = logging.getLogger("detector")

# ---------- Utility geometry ----------
def _order_quad(pts4: np.ndarray) -> np.ndarray:
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _line_from_segment(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(float, [p1[0], p1[1], p2[0], p2[1]])
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    n = max(1e-9, np.hypot(a, b))
    a, b, c = a/n, b/n, c/n
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return np.array([a, b, c], dtype=np.float64)

def _intersect_lines(L1: np.ndarray, L2: np.ndarray) -> Optional[np.ndarray]:
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1*b2 - a2*b1
    if abs(det) < 1e-9:
        return None
    x = (b1*c2 - b2*c1) / det
    y = (c1*a2 - c2*a1) / det
    return np.array([x, y], dtype=np.float64)

def _warp_to_size(frame: np.ndarray, quad: np.ndarray, w: int, h: int) -> np.ndarray:
    src = _order_quad(quad.astype(np.float32))
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h))

# ---------- LSD factory (OpenCV-version safe) ----------
def _make_lsd():
    refine_val = None
    if hasattr(cv2, "LSD_REFINE_STD"): refine_val = cv2.LSD_REFINE_STD
    try:
        if refine_val is not None:
            return cv2.createLineSegmentDetector(refine=refine_val)
    except TypeError:
        pass
    try:
        return cv2.createLineSegmentDetector()
    except Exception:
        return None

# ---------- Manual corner picker ----------
class CornerPicker:
    """Hook up to a Qt/HighGUI window name and click TL->TR->BR->BL when enabled."""
    def __init__(self, win_name: str):
        self.win = win_name
        self.points: List[Tuple[int, int]] = []
        self.enabled = False
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if not self.enabled:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
            if len(self.points) == 4:
                self.enabled = False
                log.info("Manual corners selected.")

    def reset_and_enable(self):
        self.points.clear()
        self.enabled = True
        log.info("Click TL -> TR -> BR -> BL")

    def get_quad(self) -> Optional[np.ndarray]:
        if len(self.points) == 4:
            return np.array(self.points, dtype=np.float32)
        return None

# ---------- Screen detector ----------
class ScreenRectifier:
    """
    Finds the display quad using line detection (LSD with Hough fallback),
    draws it on the input, and rectifies to a target width/height.
    """
    def __init__(self, target_w: int = 3840, target_h: int = 2160):
        self.target_w = target_w
        self.target_h = target_h
        self.last_quad: Optional[np.ndarray] = None
        self.H_prev: Optional[np.ndarray] = None
        self.alpha = 0.85  # EMA smoothing

    def _detect_quad(self, frame: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame.shape[:2]
        down = 2
        h, w = H // down, W // down
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray  = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)

        lsd = _make_lsd()
        lines = None
        if lsd is not None:
            try:
                lines, _, _, _ = lsd.detect(gray)
            except Exception:
                lines = None

        segments = []
        min_len = 0.15 * max(w, h)
        if lines is not None and len(lines) > 0:
            for L in lines:
                x1, y1, x2, y2 = L[0]
                length = np.hypot(x2 - x1, y2 - y1)
                if length < min_len: continue
                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1))) % 180.0
                segments.append((np.array([x1, y1]), np.array([x2, y2]), length, angle))
        else:
            edges = cv2.Canny(gray, 60, 180)
            hp = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                 minLineLength=int(min_len), maxLineGap=20)
            if hp is None: return None
            for x1, y1, x2, y2 in hp[:, 0, :]:
                length = np.hypot(x2 - x1, y2 - y1)
                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1))) % 180.0
                segments.append((np.array([x1, y1]), np.array([x2, y2]), length, angle))
        if not segments: return None

        def is_vertical(theta):   return abs(theta - 90.0) <= 20.0
        def is_horizontal(theta): return min(abs(theta - 0.0), abs(theta - 180.0)) <= 20.0

        vertical = [s for s in segments if is_vertical(s[3])]
        horiz    = [s for s in segments if is_horizontal(s[3])]
        if len(vertical) < 2 or len(horiz) < 2: return None

        def line_and_offset(seg):
            p1, p2, *_ = seg
            L = _line_from_segment(p1, p2)
            cx, cy = w/2.0, h/2.0
            offset = L[0]*cx + L[1]*cy + L[2]
            return L, offset

        v_sorted = sorted([line_and_offset(s) for s in vertical], key=lambda x: x[1])
        h_sorted = sorted([line_and_offset(s) for s in horiz],    key=lambda x: x[1])
        L_left,  L_right  = v_sorted[0][0], v_sorted[-1][0]
        L_top,   L_bottom = h_sorted[0][0], h_sorted[-1][0]

        p_tl = _intersect_lines(L_left,  L_top)
        p_tr = _intersect_lines(L_right, L_top)
        p_br = _intersect_lines(L_right, L_bottom)
        p_bl = _intersect_lines(L_left,  L_bottom)
        if any(p is None for p in (p_tl, p_tr, p_br, p_bl)): return None

        quad_small = np.vstack([p_tl, p_tr, p_br, p_bl]).astype(np.float32)
        area = cv2.contourArea(quad_small.reshape(-1,1,2))
        if area < 0.25 * w * h: return None
        return quad_small * np.array([down, down], dtype=np.float32)

    # ---- public API
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        return self._detect_quad(frame)

    def rectify(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        src = _order_quad(quad.astype(np.float32))
        dst = np.array([[0, 0],
                        [self.target_w - 1, 0],
                        [self.target_w - 1, self.target_h - 1],
                        [0, self.target_h - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        if self.H_prev is None:
            H_smooth = H
        else:
            H_smooth = (self.alpha * self.H_prev + (1 - self.alpha) * H).astype(np.float32)
        self.H_prev = H_smooth
        return cv2.warpPerspective(frame, H_smooth, (self.target_w, self.target_h))

    def annotate(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        out = frame.copy()
        cv2.polylines(out, [quad.astype(int)], True, (0, 0, 255), 3, cv2.LINE_AA)
        return out

    def warp_to_size(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        return _warp_to_size(frame, quad, self.target_w, self.target_h)
