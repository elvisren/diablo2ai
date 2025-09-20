#!/usr/bin/env python3
# screen_detector.py
import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple

log = logging.getLogger("detector")

# ----------------- small helpers -----------------
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
    c = x1 * y2 - x2 * y1
    n = max(1e-9, np.hypot(a, b))
    a, b, c = a / n, b / n, c / n
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return np.array([a, b, c], dtype=np.float64)

def _intersect_lines(L1: np.ndarray, L2: np.ndarray) -> Optional[np.ndarray]:
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float64)

def _warp_to_size(frame: np.ndarray, quad: np.ndarray, w: int, h: int) -> np.ndarray:
    src = _order_quad(quad.astype(np.float32))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h))

def _make_lsd():
    refine_val = None
    if hasattr(cv2, "LSD_REFINE_STD"):
        refine_val = cv2.LSD_REFINE_STD
    try:
        if refine_val is not None:
            return cv2.createLineSegmentDetector(refine=refine_val)
    except TypeError:
        pass
    try:
        return cv2.createLineSegmentDetector()
    except Exception:
        return None

# ----------------- manual corner picker (unchanged API) -----------------
class CornerPicker:
    """Click TL -> TR -> BR -> BL when enabled."""
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

# ----------------- bezel trimming -----------------
def _estimate_inward_shift(gray: np.ndarray, p1: np.ndarray, p2: np.ndarray, max_px: int = 40) -> int:
    """
    For a side (p1->p2), sample perpendicular intensity gradient from the edge inward
    up to max_px. Return the offset (pixels) where the gradient magnitude peaks.
    This tends to land near the bezel->screen transition.
    """
    p1 = p1.astype(np.float32)
    p2 = p2.astype(np.float32)
    v = p2 - p1
    L = np.linalg.norm(v)
    if L < 1.0:
        return 0
    t = v / L
    n = np.array([-t[1], t[0]])  # inward normal will be decided by sampling two directions

    # choose "inward" by checking which side grows brighter/has more variance
    k = int(max(16, L // 50))  # sample points along the edge
    xs = np.linspace(0, 1, k)
    pts_on_edge = (p1[None, :] + xs[:, None] * v[None, :])

    # probe both directions a few pixels to decide inwards
    test = 6
    a = np.clip(pts_on_edge + n[None, :] * test, [0, 0], [gray.shape[1] - 1, gray.shape[0] - 1])
    b = np.clip(pts_on_edge - n[None, :] * test, [0, 0], [gray.shape[1] - 1, gray.shape[0] - 1])
    avals = gray[a[:, 1].astype(int), a[:, 0].astype(int)]
    bvals = gray[b[:, 1].astype(int), b[:, 0].astype(int)]
    inward = n if np.var(avals) > np.var(bvals) else -n

    # build a profile of gradient magnitude averaged along the edge
    shifts = np.arange(0, max_px)
    mags = []
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    for s in shifts:
        samp = pts_on_edge + inward[None, :] * s
        xs_ = np.clip(samp[:, 0].astype(int), 0, gray.shape[1] - 1)
        ys_ = np.clip(samp[:, 1].astype(int), 0, gray.shape[0] - 1)
        g = np.hypot(gx[ys_, xs_], gy[ys_, xs_]).mean()
        mags.append(g)
    idx = int(np.argmax(mags))
    return idx

def _shrink_quad_inwards(gray: np.ndarray, quad: np.ndarray, max_px: int = 40) -> np.ndarray:
    """Move each side inward by an estimated bezel width."""
    q = _order_quad(quad).astype(np.float32)
    tl, tr, br, bl = q
    shifts = [
        _estimate_inward_shift(gray, tl, tr, max_px),  # top
        _estimate_inward_shift(gray, tr, br, max_px),  # right
        _estimate_inward_shift(gray, br, bl, max_px),  # bottom
        _estimate_inward_shift(gray, bl, tl, max_px),  # left
    ]

    def move_segment(p1, p2, pixels):
        v = p2 - p1
        L = np.linalg.norm(v)
        if L < 1:
            return p1, p2
        n = np.array([-(v[1] / L), (v[0] / L)], dtype=np.float32)
        return p1 + n * pixels, p2 + n * pixels

    t1, t2 = move_segment(tl, tr, shifts[0])
    r1, r2 = move_segment(tr, br, shifts[1])
    b1, b2 = move_segment(br, bl, shifts[2])
    l1, l2 = move_segment(bl, tl, shifts[3])

    # recompute intersections
    L_top = _line_from_segment(t1, t2)
    L_right = _line_from_segment(r1, r2)
    L_bottom = _line_from_segment(b1, b2)
    L_left = _line_from_segment(l1, l2)

    p_tl = _intersect_lines(L_left, L_top)
    p_tr = _intersect_lines(L_right, L_top)
    p_br = _intersect_lines(L_right, L_bottom)
    p_bl = _intersect_lines(L_left, L_bottom)
    if any(p is None for p in (p_tl, p_tr, p_br, p_bl)):
        return q  # fallback: do not shrink if unstable
    out = np.vstack([p_tl, p_tr, p_br, p_bl]).astype(np.float32)
    return out

# ----------------- main detector -----------------
class ScreenRectifier:
    """
    Robust screen quad finder:
      1) Contour-based rectangle (largest, centered, near-orthogonal)
      2) Fallback to line grouping (LSD/Hough)
    Optional bezel trimming estimates the inner active area.
    Rectification uses EMA-smoothed homography (like your original).
    """
    def __init__(self, target_w: int = 3840, target_h: int = 2160, trim_bezel: bool = True, max_trim_px: int = 40):
        self.target_w = target_w
        self.target_h = target_h
        self.trim_bezel = trim_bezel
        self.max_trim_px = max_trim_px
        self.last_quad: Optional[np.ndarray] = None
        self.H_prev: Optional[np.ndarray] = None
        self.alpha = 0.85  # EMA smoothing

    # ---------- public API ----------
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        quad = self._detect_by_contours(frame)
        if quad is None:
            quad = self._detect_by_lines(frame)
        if quad is None:
            return None

        # bezel trim (on the original-resolution gray)
        if self.trim_bezel:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            quad = _shrink_quad_inwards(gray, quad, max_px=self.max_trim_px)
        self.last_quad = quad
        return quad

    def rectify(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        src = _order_quad(quad.astype(np.float32))
        dst = np.array([[0, 0],
                        [self.target_w - 1, 0],
                        [self.target_w - 1, self.target_h - 1],
                        [0, self.target_h - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        H_smooth = H if self.H_prev is None else (self.alpha * self.H_prev + (1 - self.alpha) * H).astype(np.float32)
        self.H_prev = H_smooth
        return cv2.warpPerspective(frame, H_smooth, (self.target_w, self.target_h))

    def annotate(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        out = frame.copy()
        cv2.polylines(out, [quad.astype(int)], True, (0, 0, 255), 3, cv2.LINE_AA)
        return out

    def warp_to_size(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        return _warp_to_size(frame, quad, self.target_w, self.target_h)

    # ---------- detectors ----------
    def _detect_by_contours(self, frame: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame.shape[:2]
        scale = 960.0 / max(H, W)
        if scale < 1.0:
            small = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = frame.copy()
            scale = 1.0
        sh, sw = small.shape[:2]

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # sharpen a bit to separate bezel vs screen
        gray_sharp = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)

        edges = cv2.Canny(gray_sharp, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        # Center preference & rectangle quality scoring
        cx, cy = sw / 2.0, sh / 2.0
        best = None
        best_score = -1e9

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.20 * sw * sh:
                continue  # too small

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            approx = approx.reshape(-1, 2).astype(np.float32)
            if not cv2.isContourConvex(approx.astype(np.int32)):
                continue

            # orthogonality score: angles near 90°
            def angle(a, b, c):
                v1 = a - b
                v2 = c - b
                cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                return np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            pts = _order_quad(approx)
            angs = [angle(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]
            ortho = -np.mean([abs(a - 90) for a in angs])  # higher is better

            # centered score: distance of centroid to image center
            M = cv2.moments(approx)
            if abs(M["m00"]) < 1e-6:
                continue
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            center_penalty = -np.hypot(cX - cx, cY - cy) / max(sw, sh)

            # aspect ratio sanity (allow portrait/landscape; very loose)
            w_est = np.linalg.norm(pts[1] - pts[0])
            h_est = np.linalg.norm(pts[3] - pts[0])
            ar = (w_est + 1e-6) / (h_est + 1e-6)
            ar_penalty = -0.0
            if ar < 0.4 or ar > 3.0:
                ar_penalty = -5.0  # strongly discourage absurd shapes

            score = (area / (sw * sh)) * 2.0 + ortho * 0.5 + center_penalty * 3.0 + ar_penalty
            if score > best_score:
                best_score = score
                best = pts

        if best is None:
            return None

        # upscale back to original resolution
        if scale != 1.0:
            best = best / np.float32(scale)
        return best

    def _detect_by_lines(self, frame: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame.shape[:2]
        down = 2
        h, w = H // down, W // down
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)

        lsd = _make_lsd()
        lines = None
        if lsd is not None:
            try:
                lines, _, _, _ = lsd.detect(gray)
            except Exception:
                lines = None

        segments = []
        min_len = 0.18 * max(w, h)
        if lines is not None and len(lines) > 0:
            for L in lines:
                x1, y1, x2, y2 = L[0]
                length = np.hypot(x2 - x1, y2 - y1)
                if length < min_len:
                    continue
                angle = (np.degrees(np.arctan2((y2 - y1), (x2 - x1))) + 180.0) % 180.0
                segments.append((np.array([x1, y1]), np.array([x2, y2]), length, angle))
        else:
            edges = cv2.Canny(gray, 60, 180)
            hp = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                 minLineLength=int(min_len), maxLineGap=20)
            if hp is None:
                return None
            for x1, y1, x2, y2 in hp[:, 0, :]:
                length = np.hypot(x2 - x1, y2 - y1)
                angle = (np.degrees(np.arctan2((y2 - y1), (x2 - x1))) + 180.0) % 180.0
                segments.append((np.array([x1, y1]), np.array([x2, y2]), length, angle))
        if not segments:
            return None

        def is_vertical(theta):   return abs(theta - 90.0) <= 20.0
        def is_horizontal(theta): return min(abs(theta - 0.0), abs(theta - 180.0)) <= 20.0

        vertical = [s for s in segments if is_vertical(s[3])]
        horiz = [s for s in segments if is_horizontal(s[3])]
        if len(vertical) < 2 or len(horiz) < 2:
            return None

        def line_and_offset(seg):
            p1, p2, *_ = seg
            L = _line_from_segment(p1, p2)
            cx, cy = w / 2.0, h / 2.0
            offset = L[0] * cx + L[1] * cy + L[2]
            return L, offset

        v_sorted = sorted([line_and_offset(s) for s in vertical], key=lambda x: x[1])
        h_sorted = sorted([line_and_offset(s) for s in horiz], key=lambda x: x[1])
        L_left, L_right = v_sorted[0][0], v_sorted[-1][0]
        L_top, L_bottom = h_sorted[0][0], h_sorted[-1][0]

        p_tl = _intersect_lines(L_left, L_top)
        p_tr = _intersect_lines(L_right, L_top)
        p_br = _intersect_lines(L_right, L_bottom)
        p_bl = _intersect_lines(L_left, L_bottom)
        if any(p is None for p in (p_tl, p_tr, p_br, p_bl)):
            return None

        quad_small = np.vstack([p_tl, p_tr, p_br, p_bl]).astype(np.float32)
        area = cv2.contourArea(quad_small.reshape(-1, 1, 2))
        if area < 0.25 * w * h:
            return None
        return quad_small * np.array([down, down], dtype=np.float32)

# ----------------- module-level passthroughs (kept) -----------------
def _warp_to_size(frame: np.ndarray, quad: np.ndarray, w: int, h: int) -> np.ndarray:
    src = _order_quad(quad.astype(np.float32))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h))
