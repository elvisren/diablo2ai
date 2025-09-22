#!/usr/bin/env python3
# generate_d2r_yolo_data.py  —  with guards + optional IoU-based non-overlap
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml
import re

# =========================================================
#                        CONFIG
# =========================================================
@dataclass
class Config:
    # Inputs
    monsters_dir: Path = Path("./monstor_image")     # *_graphic.png (sprites with alpha)
    other_dir: Path    = Path("./other_image")       # expects cursor_graphic.png
    # Outputs
    out_root: Path     = Path("./out")
    # Image geometry
    image_size: Tuple[int, int] = (1920, 1080)       # (W, H)
    # Train/Val counts
    total_train_images: int = 900
    total_val_images:   int = 100
    # Per-image placement
    monsters_per_image: int = 20
    duplicates_min: int = 1
    duplicates_max: int = 4
    # Sprite scales (monster)
    min_sprite_h: int = 80
    max_sprite_h: int = 360
    # Cursor scales
    cursor_min_h: int = 32
    cursor_max_h: int = 120
    # “3D” angle grids
    yaws: List[int]   = None
    pitches: List[int] = None
    # Aug
    aug_brightness: float = 0.15
    aug_contrast: float   = 0.10
    jpeg_quality: int = 90
    # Cache location
    cache_dir: Path = Path("./cache")
    # Reproducibility
    seed: int = 1337

    # -------- NEW: safety & overlap controls --------
    # Minimum bbox size in pixels (after clipping) to keep an instance
    min_bbox_w: int = 2
    min_bbox_h: int = 2

    # Non-overlap controls for monsters
    avoid_overlap_monsters: bool = True
    max_iou_monsters: float = 0.50  # reject placement if IoU > this with any existing box

    # Non-overlap controls for cursor
    avoid_overlap_cursor: bool = True
    max_iou_cursor: float = 0.60

    # Try multiple random positions before giving up on an instance
    place_attempts: int = 30

    def __post_init__(self):
        if self.yaws is None:
            self.yaws = [-35, -25, -15, -7, 0, 7, 15, 25, 35]
        if self.pitches is None:
            self.pitches = [-20, -12, -6, 0, 6, 12, 20]


# =========================================================
#                      UTILITIES
# =========================================================
SAFE_NAME_RE = re.compile(r"(.*)_graphic\.png$", re.IGNORECASE)

def parse_class_name(p: Path) -> Optional[str]:
    m = SAFE_NAME_RE.match(p.name)
    if not m:
        return None
    base = m.group(1)
    # Strip trailing "01"
    if base.endswith("01"):
        base = base[:-2]
    return base

def list_monster_sprites(src: Path) -> Dict[str, Path]:
    sprites = {}
    for fp in src.rglob("*_graphic.png"):
        cname = parse_class_name(fp)
        if cname and cname.lower() != "cursor":
            sprites[cname] = fp
    return sprites

def find_cursor_sprite(other_dir: Path) -> Path:
    p = other_dir / "cursor_graphic.png"
    if not p.exists():
        raise SystemExit(f"Required cursor image not found: {p}")
    return p

def ensure_dirs_for_split(cfg: Config, split: str):
    (cfg.out_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (cfg.out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

def write_classes(classes: List[str], out_root: Path):
    (out_root / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

def write_dataset_yaml(cfg: Config, classes: List[str]):
    ds = {
        "path": str(cfg.out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)},
    }
    with open(cfg.out_root / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(ds, f, sort_keys=True)

def apply_bc_jitter(img_bgr: np.ndarray, max_brightness: float, max_contrast: float):
    b = 1.0 + (random.uniform(-max_brightness, max_brightness) if max_brightness > 0 else 0.0)
    c = 1.0 + (random.uniform(-max_contrast, max_contrast) if max_contrast > 0 else 0.0)
    out = cv2.convertScaleAbs(img_bgr, alpha=c, beta=(b - 1.0) * 128.0)
    return out


# =========================================================
#                 “FAKE 3D” (Yaw/Pitch Warp)
# =========================================================
def sprite_to_bgra(png_path: Path) -> np.ndarray:
    pil = Image.open(png_path).convert("RGBA")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGBA2BGRA)

def rotation_matrix_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    Ry = np.array([
        [ math.cos(yaw), 0, math.sin(yaw)],
        [ 0,             1, 0            ],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ], dtype=np.float32)
    Rx = np.array([
        [1, 0,               0              ],
        [0, math.cos(pitch),-math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ], dtype=np.float32)
    return Rx @ Ry

def warp_perspective_with_3d(bgra: np.ndarray, yaw: int, pitch: int) -> np.ndarray:
    h, w = bgra.shape[:2]
    corners = np.array([
        [-w/2, -h/2, 0],
        [ w/2, -h/2, 0],
        [ w/2,  h/2, 0],
        [-w/2,  h/2, 0],
    ], dtype=np.float32)
    R = rotation_matrix_yaw_pitch(yaw, pitch)
    f = 1.2 * max(w, h)
    K = np.array([[f, 0, 0],
                  [0, f, 0],
                  [0, 0, 1]], dtype=np.float32)
    z_off = 2.0 * max(w, h)
    pts3d = (R @ corners.T).T
    pts3d[:, 2] += z_off
    pts2d = (K @ pts3d.T).T
    pts2d = pts2d[:, :2] / pts2d[:, 2:3]
    min_xy = pts2d.min(axis=0)
    pts2d_shift = pts2d - min_xy
    out_w = int(max(1, math.ceil(pts2d_shift[:, 0].max())))
    out_h = int(max(1, math.ceil(pts2d_shift[:, 1].max())))
    src = np.array([[0, 0],[w-1, 0],[w-1, h-1],[0, h-1]], dtype=np.float32)
    dst = pts2d_shift.astype(np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    warped = cv2.warpPerspective(bgra, H, (out_w, out_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    return cv2.GaussianBlur(warped, (0, 0), sigmaX=0.5)

def cache_warp_grid(cache_dir: Path, cname: str, png_path: Path,
                    yaws: List[int], pitches: List[int]) -> Dict[Tuple[int,int], Path]:
    out_map: Dict[Tuple[int,int], Path] = {}
    mdir = cache_dir / cname
    mdir.mkdir(parents=True, exist_ok=True)
    base = sprite_to_bgra(png_path)
    for y in yaws:
        for p in pitches:
            outp = mdir / f"{y}_{p}.png"
            out_map[(y, p)] = outp
            if not outp.exists():
                warped = warp_perspective_with_3d(base, y, p)
                cv2.imwrite(str(outp), warped)
    return out_map


# =========================================================
#                    COMPOSITING / LABELS
# =========================================================
def alpha_composite_onto(canvas_bgr: np.ndarray, sprite_bgra: np.ndarray, x: int, y: int
                         ) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    H, W = canvas_bgr.shape[:2]
    h, w = sprite_bgra.shape[:2]
    x1, y1, x2, y2 = x, y, x + w, y + h

    dx1 = max(0, x1); dy1 = max(0, y1)
    dx2 = min(W, x2); dy2 = min(H, y2)
    if dx1 >= dx2 or dy1 >= dy2:
        return canvas_bgr, (0, 0, 0, 0)  # outside

    sx1 = dx1 - x1; sy1 = dy1 - y1
    sx2 = sx1 + (dx2 - dx1)
    sy2 = sy1 + (dy2 - dy1)

    roi = canvas_bgr[dy1:dy2, dx1:dx2]
    spr = sprite_bgra[sy1:sy2, sx1:sx2]

    if spr.shape[2] == 4:
        alpha = spr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = spr[:, :, :3].astype(np.float32)
        base = roi.astype(np.float32)
        out = alpha * rgb + (1.0 - alpha) * base
        canvas_bgr[dy1:dy2, dx1:dx2] = out.astype(np.uint8)

        nz = np.argwhere(alpha.squeeze() > 0)
        if nz.size == 0:
            return canvas_bgr, (0, 0, 0, 0)
        yy = nz[:, 0] + dy1
        xx = nz[:, 1] + dx1
        bx1, by1, bx2, by2 = int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())
        return canvas_bgr, (bx1, by1, bx2, by2)
    else:
        canvas_bgr[dy1:dy2, dx1:dx2] = spr[:, :, :3]
        return canvas_bgr, (dx1, dy1, dx2 - 1, dy2 - 1)

def resize_keep_aspect(bgra: np.ndarray, target_h: int) -> np.ndarray:
    h, w = bgra.shape[:2]
    scale = target_h / max(1, h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(bgra, (new_w, target_h), interpolation=cv2.INTER_AREA)

def bbox_xyxy_valid(b: Tuple[int,int,int,int], cfg: Config) -> bool:
    x1, y1, x2, y2 = b
    return (x2 - x1) >= cfg.min_bbox_w and (y2 - y1) >= cfg.min_bbox_h

def bbox_to_yolo_line(bbox_xyxy: Tuple[int,int,int,int], class_id: int, W: int, H: int
                      ) -> Optional[str]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    return f"{class_id} {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}"

def iou_xyxy(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    aw = max(0, ax2 - ax1); ah = max(0, ay2 - ay1)
    bw = max(0, bx2 - bx1); bh = max(0, by2 - by1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# =========================================================
#                    SAMPLING / ROSTER
# =========================================================
def build_balanced_roster(class_names: List[str], total_images: int,
                          per_image: int) -> List[str]:
    total_instances = total_images * per_image
    per_class = math.ceil(total_instances / len(class_names))
    roster = []
    for cname in class_names:
        roster += [cname] * per_class
    random.shuffle(roster)
    return roster[:total_instances]

def choose_duplicates(dup_min: int, dup_max: int) -> int:
    return random.randint(dup_min, dup_max)


# =========================================================
#                         MAIN
# =========================================================
def synthesize_split(split: str,
                     num_images: int,
                     cfg: Config,
                     class_names: List[str],
                     name_to_id: Dict[str, int],
                     cache_index: Dict[str, Dict[Tuple[int,int], Path]],
                     cursor_name: str):
    ensure_dirs_for_split(cfg, split)
    W, H = cfg.image_size
    img_dir = cfg.out_root / "images" / split
    lbl_dir = cfg.out_root / "labels" / split

    # Balanced roster for monsters (cursor excluded)
    monster_names = [c for c in class_names if c != cursor_name]
    roster = build_balanced_roster(monster_names, num_images, cfg.monsters_per_image)
    roster_ptr = 0

    def new_background():
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        bg[:] = (15, 15, 15)
        return bg

    pbar = tqdm(range(num_images), desc=f"Synthesize[{split}]")
    for idx in pbar:
        canvas = new_background()
        labels: List[str] = []
        placed_boxes: List[Tuple[int,int,int,int]] = []  # keep bboxes to check IoU
        placed = 0

        # -------- Place monsters --------
        while placed < cfg.monsters_per_image and roster_ptr < len(roster):
            cname = roster[roster_ptr]; roster_ptr += 1
            dups = choose_duplicates(cfg.duplicates_min, cfg.duplicates_max)

            for _ in range(dups):
                if placed >= cfg.monsters_per_image:
                    break

                yaw = random.choice(cfg.yaws)
                pitch = random.choice(cfg.pitches)
                spr = cv2.imread(str(cache_index[cname][(yaw, pitch)]), cv2.IMREAD_UNCHANGED)

                target_h = random.randint(cfg.min_sprite_h, cfg.max_sprite_h)
                spr_r = resize_keep_aspect(spr, target_h)

                # jitter
                rgb = spr_r[:, :, :3]
                a = spr_r[:, :, 3:4]
                rgb = apply_bc_jitter(rgb, cfg.aug_brightness, cfg.aug_contrast)
                spr_r = np.concatenate([rgb, a], axis=2)

                sh, sw = spr_r.shape[:2]

                # Try multiple placements to satisfy IoU constraint & bbox size
                success = False
                for _try in range(cfg.place_attempts):
                    x = random.randint(-sw // 4, W - sw + sw // 4)
                    y = random.randint(-sh // 4, H - sh + sh // 4)
                    # Dry-run compositing: get bbox without modifying canvas
                    tmp_canvas = canvas.copy()
                    tmp_canvas, bbox = alpha_composite_onto(tmp_canvas, spr_r, x, y)
                    if not bbox_xyxy_valid(bbox, cfg):
                        continue
                    if cfg.avoid_overlap_monsters:
                        too_much = any(iou_xyxy(bbox, pb) > cfg.max_iou_monsters for pb in placed_boxes)
                        if too_much:
                            continue
                    # Commit compositing
                    canvas, bbox = alpha_composite_onto(canvas, spr_r, x, y)
                    if not bbox_xyxy_valid(bbox, cfg):
                        continue
                    line = bbox_to_yolo_line(bbox, name_to_id[cname], W, H)
                    if line:
                        labels.append(line)
                        placed_boxes.append(bbox)
                        placed += 1
                        success = True
                        break
                if not success:
                    # Could not place this instance respecting IoU/min-size; skip it
                    continue

        # -------- Place exactly ONE cursor --------
        yaw = random.choice(cfg.yaws)
        pitch = random.choice(cfg.pitches)
        cur = cv2.imread(str(cache_index[cursor_name][(yaw, pitch)]), cv2.IMREAD_UNCHANGED)
        target_h = random.randint(cfg.cursor_min_h, cfg.cursor_max_h)
        cur_r = resize_keep_aspect(cur, target_h)
        rgb = cur_r[:, :, :3]
        a = cur_r[:, :, 3:4]
        rgb = apply_bc_jitter(rgb, cfg.aug_brightness * 0.5, cfg.aug_contrast * 0.5)
        cur_r = np.concatenate([rgb, a], axis=2)
        sh, sw = cur_r.shape[:2]

        # Try to place cursor respecting its IoU rule (if enabled)
        for _try in range(cfg.place_attempts):
            x = random.randint(0, max(0, W - sw))
            y = random.randint(0, max(0, H - sh))
            tmp_canvas = canvas.copy()
            tmp_canvas, bbox = alpha_composite_onto(tmp_canvas, cur_r, x, y)
            if not bbox_xyxy_valid(bbox, cfg):
                continue
            if cfg.avoid_overlap_cursor:
                too_much = any(iou_xyxy(bbox, pb) > cfg.max_iou_cursor for pb in placed_boxes)
                if too_much:
                    continue
            canvas, bbox = alpha_composite_onto(canvas, cur_r, x, y)
            if not bbox_xyxy_valid(bbox, cfg):
                continue
            line = bbox_to_yolo_line(bbox, name_to_id[cursor_name], W, H)
            if line:
                labels.append(line)
                placed_boxes.append(bbox)
                break
        else:
            # Could not place cursor under IoU/size constraints — place once without IoU restriction
            x = random.randint(0, max(0, W - sw))
            y = random.randint(0, max(0, H - sh))
            canvas, bbox = alpha_composite_onto(canvas, cur_r, x, y)
            if bbox_xyxy_valid(bbox, cfg):
                line = bbox_to_yolo_line(bbox, name_to_id[cursor_name], W, H)
                if line:
                    labels.append(line)
                    placed_boxes.append(bbox)

        # Save (trim blanks)
        img_name = f"img_{idx:06d}.jpg"
        lbl_name = f"img_{idx:06d}.txt"
        cv2.imwrite(str(img_dir / img_name), canvas,
                    [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality])
        cleaned = [s for s in labels if s and s.strip()]
        (lbl_dir / lbl_name).write_text("\n".join(cleaned), encoding="utf-8")


def main():
    cfg = Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Collect sprites
    monster_sprites = list_monster_sprites(cfg.monsters_dir)
    if not monster_sprites:
        raise SystemExit(f"No '*_graphic.png' sprites found under {cfg.monsters_dir}")

    cursor_png = find_cursor_sprite(cfg.other_dir)

    # Build classes (monsters + cursor)
    class_names = sorted(monster_sprites.keys())
    cursor_class_name = parse_class_name(cursor_png) or "cursor"
    if cursor_class_name not in class_names:
        class_names.append(cursor_class_name)

    name_to_id = {n: i for i, n in enumerate(class_names)}

    # Write metadata
    write_classes(class_names, cfg.out_root)
    write_dataset_yaml(cfg, class_names)

    # Precompute caches
    cache_index: Dict[str, Dict[Tuple[int,int], Path]] = {}
    print("Precomputing caches …")
    for cname in tqdm(class_names, desc="Cache precompute"):
        if cname == cursor_class_name:
            cache_index[cname] = cache_warp_grid(cfg.cache_dir, cname, cursor_png,
                                                 cfg.yaws, cfg.pitches)
        else:
            cache_index[cname] = cache_warp_grid(cfg.cache_dir, cname,
                                                 monster_sprites[cname],
                                                 cfg.yaws, cfg.pitches)

    # Synthesize splits
    random.seed(cfg.seed + 1); np.random.seed(cfg.seed + 1)
    synthesize_split("train", cfg.total_train_images, cfg,
                     class_names, name_to_id, cache_index, cursor_class_name)

    random.seed(cfg.seed + 2); np.random.seed(cfg.seed + 2)
    synthesize_split("val", cfg.total_val_images, cfg,
                     class_names, name_to_id, cache_index, cursor_class_name)

    print("\nDone.")
    print(f"Train images  => {(cfg.out_root / 'images' / 'train').resolve()}")
    print(f"Train labels  => {(cfg.out_root / 'labels' / 'train').resolve()}")
    print(f"Val images    => {(cfg.out_root / 'images' / 'val').resolve()}")
    print(f"Val labels    => {(cfg.out_root / 'labels' / 'val').resolve()}")
    print(f"Classes       => {(cfg.out_root / 'classes.txt').resolve()}")
    print(f"Dataset YAML  => {(cfg.out_root / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()
