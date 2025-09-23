#!/usr/bin/env python3
# screen_detector.py
from __future__ import annotations

import argparse
import os
import time
import cv2
import numpy as np
from typing import Optional, Tuple, List

# ----------------- helpers (kept names) -----------------

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

def _quad_center(quad: np.ndarray) -> Tuple[float, float]:
    q = _order_quad(quad)
    return float(q[:, 0].mean()), float(q[:, 1].mean())

def _point_in_quad(pt: np.ndarray, quad: np.ndarray) -> bool:
    return cv2.pointPolygonTest(quad.reshape(-1,1,2), (float(pt[0]), float(pt[1])), False) >= 0

# ----------------- robust inward refinement (NEW) -----------------

def _consistent_inward_normals(quad: np.ndarray) -> List[np.ndarray]:
    """Normals that all point toward the quad centroid."""
    q = _order_quad(quad.astype(np.float32))
    cx, cy = q[:,0].mean(), q[:,1].mean()
    normals = []
    for i in range(4):
        p1, p2 = q[i], q[(i+1)%4]
        v = p2 - p1
        L = max(1e-6, np.linalg.norm(v))
        t = v / L
        n = np.array([-t[1], t[0]], dtype=np.float32)  # one normal
        mid = 0.5*(p1+p2)
        # choose direction that points toward centroid
        if np.dot(np.array([cx,cy],dtype=np.float32) - mid, n) < 0:
            n = -n
        normals.append(n)
    return normals

def _score_along_normal(gray: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                        edge_pts: np.ndarray, nvec: np.ndarray,
                        s: int, delta: int, strip: int) -> float:
    """
    Evaluate boundary-likeness at offset s:
      α*|∇| + β*(inside_edge - outside_edge) + γ*(outside_mean - inside_mean)
    where inside is at s+delta, outside at s-delta.
    """
    h, w = gray.shape[:2]
    # central line (at s), plus two strips (inside/outside)
    def sample(off):
        samp = edge_pts + nvec[None,:]*off
        xs = np.clip(samp[:,0].astype(int), 0, w-1)
        ys = np.clip(samp[:,1].astype(int), 0, h-1)
        return xs, ys

    xs_c, ys_c = sample(s)
    xs_i, ys_i = sample(s + delta)
    xs_o, ys_o = sample(max(0, s - delta))

    # widen to strip by averaging a few sub-offsets along normal
    def strip_mean(xs, ys, base_off, sign):
        acc_g, acc_i = 0.0, 0.0
        steps = max(1, strip)
        for k in range(steps):
            o = base_off + sign*k
            xs_k, ys_k = sample(o)
            acc_g += float(np.mean(np.hypot(gx[ys_k, xs_k], gy[ys_k, xs_k])))
            acc_i += float(np.mean(gray[ys_k, xs_k]))
        return acc_g/steps, acc_i/steps

    g_c = float(np.mean(np.hypot(gx[ys_c, xs_c], gy[ys_c, xs_c])))
    g_i, m_i = strip_mean(xs_i, ys_i, s + delta, +1)
    g_o, m_o = strip_mean(xs_o, ys_o, max(0, s - delta), -1)

    # weights (tuned conservatively; raise β if bezel reflectance is tricky)
    alpha, beta, gamma = 1.0, 1.8, 0.7
    return alpha*g_c + beta*(g_i - g_o) + gamma*(m_o - m_i)

def _refine_to_inner(gray: np.ndarray, outer_quad: np.ndarray,
                     search_in_px: int = 140,
                     bezel_min_px: int = 4,
                     bezel_max_px: int = 100,
                     delta: int = 6,
                     strip: int = 3) -> np.ndarray:
    """
    Refine each side of 'outer_quad' strictly INWARD to the inner panel edge
    using a robust objective. Normals are globally consistent (centroid-based).
    """
    q = _order_quad(outer_quad.astype(np.float32))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    normals = _consistent_inward_normals(q)

    inner_lines = []
    for i in range(4):
        p1, p2 = q[i], q[(i+1)%4]
        v = p2 - p1
        L = max(1e-6, np.linalg.norm(v))
        n = normals[i]
        # edge samples along the side
        N = max(24, int(L // 14))
        ts = np.linspace(0, 1, N).astype(np.float32)
        edge_pts = (p1[None,:] + ts[:,None]*(p2 - p1)[None,:]).astype(np.float32)

        # scan inward offsets
        s_min = max(0, bezel_min_px)
        s_max = max(s_min + 1, min(search_in_px, bezel_max_px + 40))
        best_s, best_val = s_min, -1e9
        for s in range(s_min, s_max+1):
            val = _score_along_normal(gray, gx, gy, edge_pts, n, s, delta, strip)
            if val > best_val:
                best_val, best_s = val, s

        # build line at best_s
        p1s, p2s = p1 + n*best_s, p2 + n*best_s
        inner_lines.append(_line_from_segment(p1s, p2s))

    # intersect to form inner quad
    P = [
        _intersect_lines(inner_lines[3], inner_lines[0]),
        _intersect_lines(inner_lines[0], inner_lines[1]),
        _intersect_lines(inner_lines[1], inner_lines[2]),
        _intersect_lines(inner_lines[2], inner_lines[3]),
    ]
    if any(p is None for p in P):
        return q  # fallback
    q_in = _order_quad(np.stack(P, axis=0))

    # keep inner strictly inside outer; if not, nudge inward uniformly
    if not all(_point_in_quad(q_in[k], q) for k in range(4)):
        # move a little further inward along each side's normal
        bump = 6.0
        q_fix = q_in.copy()
        for i in range(4):
            p = q_in[i]
            q_fix[i] = p + normals[i]*bump
        q_in = _order_quad(q_fix.astype(np.float32))

    return q_in

# ----------------- main detector (PUBLIC API KEPT) -----------------

class ScreenRectifier:
    """
    Public API (unchanged):
      - detect(frame) -> Optional[np.ndarray]          # returns INNER quad
      - rectify(frame, quad) -> np.ndarray
      - annotate(frame, quad) -> np.ndarray
      - warp_to_size(frame, quad) -> np.ndarray
    """

    def __init__(
        self,
        target_w: int = 3840,
        target_h: int = 2160,
        # backward-compat args
        trim_bezel: bool = True,
        max_trim_px: int = 60,
        # hierarchy thresholds
        min_pair_area_ratio: float = 0.80,
        max_pair_area_ratio: float = 0.995,
        max_center_delta_frac: float = 0.03,
        # inward refinement
        search_in_px: int = 140,
        bezel_min_px: int = 4,
        bezel_max_px: int = 100,
        **kwargs,
    ):
        self.target_w = target_w
        self.target_h = target_h
        self.trim_bezel = bool(trim_bezel)
        self.max_trim_px = int(max_trim_px)
        self.min_pair_area_ratio = float(min_pair_area_ratio)
        self.max_pair_area_ratio = float(max_pair_area_ratio)
        self.max_center_delta_frac = float(max_center_delta_frac)
        self.search_in_px = int(search_in_px)
        self.bezel_min_px = int(bezel_min_px)
        self.bezel_max_px = max(int(bezel_max_px), self.max_trim_px)
        _ = kwargs

    # ---------- PUBLIC ----------
    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        pair = self._detect_inner_outer(frame)
        return None if pair is None else pair[1]

    def rectify(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        src = _order_quad(quad.astype(np.float32))
        dst = np.array([[0, 0],
                        [self.target_w - 1, 0],
                        [self.target_w - 1, self.target_h - 1],
                        [0, self.target_h - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, H, (self.target_w, self.target_h))

    def annotate(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        out = frame.copy()
        cv2.polylines(out, [quad.astype(int)], True, (0, 255, 0), 3, cv2.LINE_AA)  # inner in green
        return out

    def warp_to_size(self, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        return _warp_to_size(frame, quad, self.target_w, self.target_h)

    # ---------- internal: return (outer_quad, inner_quad) ----------
    def _detect_inner_outer(self, frame_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        outer = self._detect_largest_quad(frame_bgr)
        if outer is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        inner = _refine_to_inner(
            gray, outer,
            search_in_px=max(self.search_in_px, self.max_trim_px),
            bezel_min_px=self.bezel_min_px,
            bezel_max_px=self.bezel_max_px,
            delta=6,
            strip=3,
        )

        # sanity checks
        if not all(_point_in_quad(inner[k], outer) for k in range(4)):
            # pull inner slightly toward centroid to ensure containment
            cx, cy = _quad_center(outer)
            v = inner - np.array([[cx,cy]], dtype=np.float32)
            inner = inner - 0.03 * v  # 3% toward center
            inner = _order_quad(inner.astype(np.float32))

        if _quad_area(inner) < 0.60 * _quad_area(outer):
            # if too small due to glare/over-shift, ease back outward a bit
            inner = _order_quad(0.15*outer + 0.85*inner)

        return outer, inner

    # Largest acceptable big quad (seed)
    def _detect_largest_quad(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        max_side = max(W, H)
        scale = 1280.0 / max_side if max_side > 1280 else 1.0
        small = frame_bgr if scale == 1.0 else cv2.resize(frame_bgr, (int(W*scale), int(H*scale)), cv2.INTER_AREA)
        sh, sw = small.shape[:2]

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        min_area_small = (2.0 / 3.0) * (sw * sh)
        best = None
        best_area = -1.0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            pts = approx.reshape(-1, 2).astype(np.float32)
            if not cv2.isContourConvex(pts.astype(np.int32)):
                continue
            area = cv2.contourArea(pts)
            if area < min_area_small:
                continue
            if _poly_orthogonality_score(pts) > 12.0:
                continue
            if area > best_area:
                best_area = area
                best = _order_quad(pts)

        if best is None:
            return None
        return best / np.float32(scale) if scale != 1.0 else best

# ----------------- GUI demo (mac-friendly, no prints required) -----------------

def _draw_two_quads(img: np.ndarray, outer: Optional[np.ndarray], inner: Optional[np.ndarray]) -> np.ndarray:
    vis = img.copy()
    if outer is not None:
        cv2.polylines(vis, [outer.astype(int)], True, (255, 0, 0), 4, cv2.LINE_AA)  # BLUE = outer bezel
    if inner is not None:
        cv2.polylines(vis, [inner.astype(int)], True, (0, 255, 0), 4, cv2.LINE_AA)  # GREEN = inner screen
    # legend
    cv2.rectangle(vis, (20, 20), (420, 110), (20, 20, 20), -1)
    cv2.circle(vis, (44, 56), 10, (255, 0, 0), -1); cv2.putText(vis, "Outer (bezel)", (64, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240,240,240), 2, cv2.LINE_AA)
    cv2.circle(vis, (44, 96), 10, (0, 255, 0), -1); cv2.putText(vis, "Inner (display)", (64, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240,240,240), 2, cv2.LINE_AA)
    return vis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect monitor inner/outer borders and visualize (GUI only).")
    parser.add_argument("--image", default="../../snapshots/display_day.png", help="Path to input image")
    parser.add_argument("--save", default="", help="Optional path to save visualization (PNG/JPG)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"[error] Cannot read image: {args.image}")

    rect = ScreenRectifier()
    pair = rect._detect_inner_outer(img)
    vis = img.copy()
    if pair is None:
        cv2.rectangle(vis, (20, 20), (20 + 540, 20 + 60), (20, 20, 20), -1)
        cv2.putText(vis, "NO SCREEN DETECTED", (36, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        outer_q, inner_q = pair
        vis = _draw_two_quads(vis, outer_q, inner_q)

    # Optional save
    if args.save:
        _ = cv2.imwrite(args.save, vis)

    win_name = "Screen detector (blue=outer, green=inner) — ESC/q to close, s to save"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    max_w, max_h = 1600, 1000
    h, w = vis.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    disp = vis if scale >= 0.999 else cv2.resize(vis, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
    cv2.imshow(win_name, disp)

    save_path = args.save
    while True:
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord('q')):  # ESC or 'q'
            break
        if key == ord('s'):
            if not save_path:
                base, ext = os.path.splitext(os.path.basename(args.image))
                stamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(os.path.dirname(args.image), f"{base}_vis_{stamp}.png")
            _ = cv2.imwrite(save_path, vis)

    cv2.destroyAllWindows()
