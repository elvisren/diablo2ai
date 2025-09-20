#!/usr/bin/env python3
# screen_detector.py
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple, List

# ----------------- helpers (same names where used before) -----------------

def _order_quad(pts4: np.ndarray) -> np.ndarray:
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _warp_to_size(frame: np.ndarray, quad: np.ndarray, w: int, h: int) -> np.ndarray:
    src = _order_quad(quad.astype(np.float32))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h))

def _quad_area(quad: np.ndarray) -> float:
    return cv2.contourArea(quad.reshape(4, 1, 2).astype(np.float32))

def _poly_orthogonality_score(pts: np.ndarray) -> float:
    p = _order_quad(pts)
    diffs = []
    for i in range(4):
        a, b, c = p[(i - 1) % 4], p[i], p[(i + 1) % 4]
        v1 = a - b; v2 = c - b
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        diffs.append(abs(ang - 90.0))
    return float(np.mean(diffs))

def _mean_inside_outside(gray: np.ndarray, quad: np.ndarray, inward_px=8, outward_px=8) -> float:
    q = _order_quad(quad).astype(np.float32)
    h, w = gray.shape[:2]

    def collect(p1, p2, nvec, di, do):
        N = max(12, int(np.linalg.norm(p2 - p1) / 12))
        ts = np.linspace(0, 1, N)
        edge = (p1[None, :] + ts[:, None] * (p2 - p1)[None, :])
        a = np.clip((edge + nvec * di).round().astype(int), [0, 0], [w - 1, h - 1])
        b = np.clip((edge - nvec * do).round().astype(int), [0, 0], [w - 1, h - 1])
        return gray[a[:, 1], a[:, 0]], gray[b[:, 1], b[:, 0]]

    inside_vals, outside_vals = [], []
    for i in range(4):
        p1, p2 = q[i], q[(i + 1) % 4]
        v = p2 - p1
        L = max(1e-6, np.linalg.norm(v))
        n = np.array([-(v[1] / L), (v[0] / L)], dtype=np.float32)
        a, b = collect(p1, p2, n, inward_px, outward_px)
        a2, b2 = collect(p1, p2, -n, inward_px, outward_px)
        if np.var(a) > np.var(a2):
            inside_vals.append(a); outside_vals.append(b)
        else:
            inside_vals.append(a2); outside_vals.append(b2)
    return float(np.mean(np.concatenate(inside_vals)) - np.mean(np.concatenate(outside_vals)))

def _make_lsd():
    try:
        if hasattr(cv2, "LSD_REFINE_STD"):
            return cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_STD)
        return cv2.createLineSegmentDetector()
    except Exception:
        return None

def _line_from_segment(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(float, [p1[0], p1[1], p2[0], p2[1]])
    a = y1 - y2; b = x2 - x1; c = x1 * y2 - x2 * y1
    n = max(1e-9, np.hypot(a, b))
    a, b, c = a / n, b / n, c / n
    if a < 0 or (a == 0 and b < 0):  # normalize sign
        a, b, c = -a, -b, -c
    return np.array([a, b, c], dtype=np.float32)

def _intersect_lines(L1: np.ndarray, L2: np.ndarray) -> Optional[np.ndarray]:
    a1, b1, c1 = L1; a2, b2, c2 = L2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9: return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float32)

# ---------- snap all sides to a uniform border (inner by default) ----------

def _snap_to_uniform_border(gray: np.ndarray, quad: np.ndarray, search_px: int = 60, pick_inner: bool = True) -> np.ndarray:
    q = _order_quad(quad.astype(np.float32))
    h, w = gray.shape[:2]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    shifted = []
    for i in range(4):
        p1, p2 = q[i], q[(i + 1) % 4]
        v = p2 - p1
        L = max(1e-6, np.linalg.norm(v))
        t = v / L
        n = np.array([-t[1], t[0]], dtype=np.float32)

        # choose inside direction by variance
        N = max(16, int(L // 18))
        ts = np.linspace(0, 1, N)
        edge = (p1[None, :] + ts[:, None] * v[None, :]).astype(np.float32)
        probe = 6
        a = np.clip((edge + n[None, :] * probe).round().astype(int), [0,0], [w-1,h-1])
        b = np.clip((edge - n[None, :] * probe).round().astype(int), [0,0], [w-1,h-1])
        inside_dir = n if np.var(gray[a[:,1], a[:,0]]) > np.var(gray[b[:,1], b[:,0]]) else -n

        direction = inside_dir if pick_inner else -inside_dir

        # 1-D gradient ridge search
        best_s, best_mag = 0, -1.0
        for s in range(0, search_px + 1):
            samp = edge + direction[None, :] * s
            xs = np.clip(samp[:, 0].astype(int), 0, w - 1)
            ys = np.clip(samp[:, 1].astype(int), 0, h - 1)
            mag = float(np.mean(np.hypot(gx[ys, xs], gy[ys, xs])))
            if mag > best_mag:
                best_mag, best_s = mag, s

        p1s, p2s = p1 + direction * best_s, p2 + direction * best_s
        shifted.append(_line_from_segment(p1s, p2s))

    P = [
        _intersect_lines(shifted[3], shifted[0]),
        _intersect_lines(shifted[0], shifted[1]),
        _intersect_lines(shifted[1], shifted[2]),
        _intersect_lines(shifted[2], shifted[3]),
    ]
    if any(p is None for p in P):
        return q
    return _order_quad(np.stack(P, axis=0))

# ----------------- main detector (PUBLIC API KEPT) -----------------

class ScreenRectifier:
    """
    Public API (unchanged):
      - detect(frame) -> Optional[np.ndarray]
      - rectify(frame, quad) -> np.ndarray
      - annotate(frame, quad) -> np.ndarray
      - warp_to_size(frame, quad) -> np.ndarray
    """

    def __init__(self, target_w: int = 3840, target_h: int = 2160, trim_bezel: bool = True, max_trim_px: int = 40):
        self.target_w = target_w
        self.target_h = target_h
        self.trim_bezel = trim_bezel
        self.max_trim_px = max_trim_px

        # internal state
        self.last_quad: Optional[np.ndarray] = None
        self._H_ema: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._frame_i: int = 0

        # tuning
        self._prefer_inner_border = True
        self._force_redetect_every = 120
        self._ema_alpha = 0.85
        self._lk_win = (15, 15)
        self._lk_levels = 3
        self._lk_err_thresh = 20.0
        self._reproj_thresh_px = 8.0

    # ---------- PUBLIC ----------
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        self._frame_i += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tracked_ok = False
        if self.last_quad is not None and self._prev_gray is not None and self._prev_pts is not None:
            tracked_ok = self._track(gray)

        if (self.last_quad is None) or (not tracked_ok) or (self._frame_i % self._force_redetect_every == 0):
            quad = self._detect_monitor(frame)
            if quad is not None:
                # enforce uniform border (inner by default)
                quad = _snap_to_uniform_border(gray, quad, search_px=60, pick_inner=self._prefer_inner_border)
                self._set_tracking_seed(gray, quad)

        quad_out = None if self.last_quad is None else self.last_quad.copy()
        if quad_out is not None and self.trim_bezel and self.max_trim_px > 0:
            quad_out = _snap_to_uniform_border(
                gray, quad_out, search_px=min(20, self.max_trim_px), pick_inner=True
            )
        return quad_out

    def rectify(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        src = _order_quad(quad.astype(np.float32))
        dst = np.array([[0, 0],
                        [self.target_w - 1, 0],
                        [self.target_w - 1, self.target_h - 1],
                        [0, self.target_h - 1]], dtype=np.float32)
        H_now = cv2.getPerspectiveTransform(src, dst)
        if self._H_ema is None:
            self._H_ema = H_now
        else:
            a = self._ema_alpha
            self._H_ema = (a * self._H_ema + (1 - a) * H_now).astype(np.float32)
        return cv2.warpPerspective(frame, self._H_ema, (self.target_w, self.target_h))

    def annotate(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        out = frame.copy()
        cv2.polylines(out, [quad.astype(int)], True, (0, 0, 255), 3, cv2.LINE_AA)
        return out

    def warp_to_size(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        return _warp_to_size(frame, quad, self.target_w, self.target_h)

    # ---------- internal ----------
    def _set_tracking_seed(self, gray: np.ndarray, quad: np.ndarray):
        self.last_quad = _order_quad(quad.astype(np.float32))
        self._prev_pts = self.last_quad.reshape(4, 1, 2).astype(np.float32)
        self._prev_gray = gray
        self._H_ema = None

    def _track(self, gray: np.ndarray) -> bool:
        p0 = self._prev_pts
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, p0, None,
            winSize=self._lk_win, maxLevel=self._lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        st = st.reshape(-1); err = err.reshape(-1)
        ok = (st == 1) & (err < self._lk_err_thresh)
        if ok.sum() < 3:
            return False

        p0_ok = p0.reshape(-1, 2)[ok]
        p1_ok = p1.reshape(-1, 2)[ok]
        H, _ = cv2.findHomography(p0_ok, p1_ok, cv2.RANSAC, 3.0)
        if H is None:
            return False

        q_old = self.last_quad.reshape(-1, 1, 2)
        q_new = cv2.perspectiveTransform(q_old, H).reshape(4, 2)
        reproj = float(np.mean(np.linalg.norm(q_new - self.last_quad, axis=1)))

        self.last_quad = _order_quad(q_new.astype(np.float32))
        self._prev_pts = self.last_quad.reshape(4, 1, 2).astype(np.float32)
        self._prev_gray = gray

        return reproj <= self._reproj_thresh_px

    # ---- detection: contours first, then line-based fallback ----
    def _detect_monitor(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        quad = self._detect_by_contours(frame_bgr)
        if quad is not None:
            return quad
        return self._detect_by_lines(frame_bgr)

    def _detect_by_contours(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        scale = 960.0 / max(H, W)
        small = cv2.resize(frame_bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame_bgr.copy()
        sh, sw = small.shape[:2]

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        sharp = cv2.addWeighted(gray, 1.6, cv2.GaussianBlur(gray, (0, 0), 3), -0.6, 0)

        edges = cv2.Canny(sharp, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cx, cy = sw / 2.0, sh / 2.0
        img_area = sw * sh
        best, best_score = None, -1e9

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.12 * img_area:  # relaxed
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True).reshape(-1, 2)
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx.astype(np.int32)):
                continue

            pts = approx.astype(np.float32)
            A = _quad_area(pts)
            rect_err = _poly_orthogonality_score(pts)
            M = cv2.moments(pts)
            if abs(M["m00"]) < 1e-6:
                continue
            x0, y0 = M["m10"]/M["m00"], M["m01"]/M["m00"]
            center_penalty = np.hypot(x0 - cx, y0 - cy) / max(sw, sh)
            bezel_delta = _mean_inside_outside(gray, pts, 8, 8) / 255.0

            score = (
                2.2 * (A / img_area)
                - 0.7 * rect_err
                - 3.0 * center_penalty
                + 1.2 * bezel_delta
            )
            if score > best_score:
                best_score, best = score, pts

        if best is None:
            return None
        if scale < 1:
            best = best / np.float32(scale)
        return _order_quad(best)

    def _detect_by_lines(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        down = 2
        h, w = H // down, W // down
        small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)

        lsd = _make_lsd()
        segments: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        min_len = 0.18 * max(w, h)

        if lsd is not None:
            try:
                lines, _, _, _ = lsd.detect(gray)
            except Exception:
                lines = None
            if lines is not None:
                for L in lines:
                    x1, y1, x2, y2 = L[0]
                    length = float(np.hypot(x2 - x1, y2 - y1))
                    if length < min_len: continue
                    angle = (np.degrees(np.arctan2((y2 - y1), (x2 - x1))) + 180.0) % 180.0
                    segments.append((np.array([x1, y1]), np.array([x2, y2]), length, angle))

        if not segments:
            edges = cv2.Canny(gray, 60, 180)
            hp = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                 minLineLength=int(min_len), maxLineGap=20)
            if hp is None:
                return None
            for x1, y1, x2, y2 in hp[:, 0, :]:
                length = float(np.hypot(x2 - x1, y2 - y1))
                angle = (np.degrees(np.arctan2((y2 - y1), (x2 - x1))) + 180.0) % 180.0
                segments.append((np.array([x1, y1]), np.array([x2, y2]), length, angle))

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
        if area < 0.20 * w * h:
            return None

        return quad_small * np.array([down, down], dtype=np.float32)
