#!/usr/bin/env python3
"""
Generate YOLOv11 training/validation data for Diablo II: Resurrected objects.

Key features
- Reads multi-format assets (png/jpg/jpeg/bmp/webp/tif/tiff) from raw_images/* folders.
- Builds balanced samplers so backgrounds, monsters, portals, gates, waypoints, cursors are used ~equally.
- Composes positives by pasting sprites onto full backgrounds (no pre-resize of bg), then letterboxes
  to 640x640 without cropping. Labels are adjusted to the letterboxed coordinates.
- Applies mild 3D-ish effects via perspective/affine warps (no relative scale change beyond the
  required shrink factors). Cursor is always on top **and is labeled** as class `cursor`.
- Prevents near-complete occlusion: no object may hide >90% of another; tries to re-place if needed.
- Generates negatives from backgrounds only (optionally layered/warped) — no labeled objects.
- Emits standard YOLO folder structure and a dataset YAML listing class names.
- Detailed progress bars for each generation phase.

Folder expectations under RAW_ROOT (default: ./raw_images):
  - negative_example/    (backgrounds; also used for negatives)
  - monster_image/       (monster sprites; filenames like Name_graphic.png or Name01_graphic.png)
  - blue_portal/
  - red_portal/
  - diablo_gate/         (filenames: {major|minor}_{open|close}_id.png)
  - waypoint/
  - cursor/

Output structure under OUT_ROOT (default: ./yolo_input):
  yolo_input/
    images/{train,val}/xxx.jpg
    labels/{train,val}/xxx.txt
    dataset.yaml

Usage examples
  python generate_d2r_yolo_data.py
    --raw-root raw_images --out-root yolo_input
    --train-pos 2000 --train-neg 1000 --val-pos 400 --val-neg 200
    --img-size 640 --seed 42

"""
from __future__ import annotations

import argparse
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# ---------------------------
# Config and constants
# ---------------------------

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

SPRITE_SCALE = {
    "monster": 2,      # shrink to 1/2
    "blue_portal": 0.5,
    "red_portal": 0.5,
    "diablo_gate": 0.5,
    "waypoint": 0.5,
    "cursor": 1,      # cursor shrinks to 1/4
}

# Mild 3D-ish transform limits (kept conservative so relative scale is basically preserved)
MAX_ROT_DEG = 6.0
MAX_SHEAR = 0.08
MAX_PERSP = 0.04  # as a fraction of width/height offset for corner jitter

# Overlap rule: disallow when overlap area exceeds 90% of the smaller box area
MAX_OCCLUDE_RATIO = 0.80

@dataclass
class CFG:
    raw_root: Path
    out_root: Path
    img_size: int = 640
    seed: int = 42
    # counts
    train_pos: int = 2000
    train_neg: int = 1000
    val_pos: int = 400
    val_neg: int = 200
    # composition counts
    monsters_per_img: int = 20
    portals_min: int = 0
    portals_max: int = 2
    gates_min: int = 0
    gates_max: int = 2
    wps_min: int = 0
    wps_max: int = 2
    # negatives: number of extra bg overlays per image
    neg_max_extra_layers: int = 3

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    files.sort()
    return files


def load_rgba(path: Path) -> Image.Image:
    # Robustly load any supported format into RGBA, handling palette/alpha correctly
    with Image.open(path) as im:
        im.load()
        return im.convert("RGBA")


def shrink_exact(img: Image.Image, factor: float) -> Image.Image:
    if factor == 1.0:
        return img
    w, h = img.size
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    return img.resize((nw, nh), Image.LANCZOS)


def rand_affine_params() -> Tuple[float, float]:
    rot = random.uniform(-MAX_ROT_DEG, MAX_ROT_DEG)
    shear = random.uniform(-MAX_SHEAR, MAX_SHEAR)
    return rot, shear


def perspective_coeffs(src_pts: List[Tuple[float, float]], dst_pts: List[Tuple[float, float]]):
    # Solve for 8 perspective coefficients mapping src -> dst for PIL
    matrix = []
    rhs = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    A = np.array(matrix, dtype=np.float64)
    B = np.array(rhs, dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(res.tolist())


def random_3d_warp(img: Image.Image) -> Image.Image:
    """Apply small rotation+shear+perspective without changing relative scale noticeably."""
    w, h = img.size
    # Affine (rotate+shear) around center
    rot, shear = rand_affine_params()
    aff = Image.Transform.AFFINE

    rotated = img.rotate(rot, resample=Image.BICUBIC, expand=False)

    # Then shear horizontally a little
    shear_x = shear
    a = 1
    b = shear_x
    c = 0
    d = 0
    e = 1
    f = 0
    sheared = rotated.transform(
        (w, h), aff, (a, b, c, d, e, f), resample=Image.BICUBIC
    )

    # Perspective jitter on corners
    jitter = MAX_PERSP
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dx = w * jitter
    dy = h * jitter
    dst = [
        (random.uniform(-dx, dx), random.uniform(-dy, dy)),
        (w + random.uniform(-dx, dx), random.uniform(-dy, dy)),
        (w + random.uniform(-dx, dx), h + random.uniform(-dy, dy)),
        (random.uniform(-dx, dx), h + random.uniform(-dy, dy)),
    ]
    coeffs = perspective_coeffs(src, dst)
    warped = sheared.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    return warped


def alpha_bbox(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    # Return bounding box of non-zero alpha; assumes RGBA
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.split()[3].getbbox()
    return bbox


def intersect_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def letterbox(img: Image.Image, new_size: int) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """Resize with unchanged aspect ratio and pad to (new_size,new_size)."""
    w, h = img.size
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    # pad
    pad_w = new_size - nw
    pad_h = new_size - nh
    left = pad_w // 2
    top = pad_h // 2
    canvas = Image.new("RGB", (new_size, new_size), (0, 0, 0))
    if resized.mode == "RGBA":
        bg = Image.new("RGB", resized.size, (0, 0, 0))
        bg.paste(resized, mask=resized.split()[3])
        resized = bg
    canvas.paste(resized, (left, top))
    return canvas, scale, (left, top)


def normalize_yolo(box: Tuple[int, int, int, int], scale: float, pad: Tuple[int, int], out_size: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    # After letterbox: x' = x*scale + pad_x; y' = y*scale + pad_y
    px, py = pad
    nx1 = x1 * scale + px
    ny1 = y1 * scale + py
    nx2 = x2 * scale + px
    ny2 = y2 * scale + py
    cx = ((nx1 + nx2) / 2.0) / out_size
    cy = ((ny1 + ny2) / 2.0) / out_size
    w = (nx2 - nx1) / out_size
    h = (ny2 - ny1) / out_size
    return cx, cy, w, h

# ---------------------------
# Asset parsing & samplers
# ---------------------------

@dataclass
class Asset:
    path: Path
    name: str   # class name
    kind: str   # monster | blue_portal | red_portal | diablo_gate | waypoint | background | cursor


def parse_monster_name(path: Path) -> str:
    name = path.stem  # e.g., "Fallen_graphic" or "Fallen01_graphic"
    # drop trailing "_graphic"
    name = re.sub(r"_graphic$", "", name, flags=re.IGNORECASE)
    # remove trailing 01
    name = re.sub(r"01$", "", name)
    return name


def parse_gate_name(path: Path) -> Optional[str]:
    m = re.match(r"^(major|minor)_(open|close)_", path.stem, flags=re.IGNORECASE)
    if not m:
        return None
    a, b = m.group(1).lower(), m.group(2).lower()
    return f"diablo_gate_{a}_{b}"


def build_assets(cfg: CFG) -> Dict[str, List[Asset]]:
    rr = cfg.raw_root
    folders = {
        "background": rr / "negative_example",
        "monster": rr / "monster_image",
        "blue_portal": rr / "blue_portal",
        "red_portal": rr / "red_portal",
        "diablo_gate": rr / "diablo_gate",
        "waypoint": rr / "waypoint",
        "cursor": rr / "cursor",
    }

    assets: Dict[str, List[Asset]] = {k: [] for k in folders.keys()}

    # backgrounds
    for p in list_images(folders["background"]):
        assets["background"].append(Asset(p, name="__bg__", kind="background"))

    # monsters
    for p in list_images(folders["monster"]):
        nm = parse_monster_name(p)
        assets["monster"].append(Asset(p, name=nm, kind="monster"))

    # portals
    for p in list_images(folders["blue_portal"]):
        assets["blue_portal"].append(Asset(p, name="blue_portal", kind="blue_portal"))
    for p in list_images(folders["red_portal"]):
        assets["red_portal"].append(Asset(p, name="red_portal", kind="red_portal"))

    # diablo gates
    for p in list_images(folders["diablo_gate"]):
        nm = parse_gate_name(p)
        if nm:
            assets["diablo_gate"].append(Asset(p, name=nm, kind="diablo_gate"))

    # waypoints
    for p in list_images(folders["waypoint"]):
        assets["waypoint"].append(Asset(p, name="waypoint", kind="waypoint"))

    # cursor (required)
    for p in list_images(folders["cursor"]):
        assets["cursor"].append(Asset(p, name="cursor", kind="cursor"))

    # sanity checks
    required = ["background", "monster", "blue_portal", "red_portal", "diablo_gate", "waypoint", "cursor"]
    missing = [k for k in required if len(assets[k]) == 0]
    if missing:
        raise FileNotFoundError(
            "Missing or empty required folders/files: " + ", ".join(missing)
        )

    return assets


class BalancedSampler:
    """Round-robin sampler with reshuffle per full pass; tracks usage counts."""

    def __init__(self, items: Sequence[Asset], seed: int = 0):
        if not items:
            raise ValueError("BalancedSampler requires non-empty items")
        self.items: List[Asset] = list(items)
        self.n = len(self.items)
        self.idx = 0
        self.rng = random.Random(seed)
        self.counts = [0] * self.n
        self._reshuffle()

    def _reshuffle(self):
        self.order = list(range(self.n))
        self.rng.shuffle(self.order)
        self.idx = 0

    def next_one(self) -> Asset:
        if self.idx >= self.n:
            self._reshuffle()
        i = self.order[self.idx]
        self.idx += 1
        self.counts[i] += 1
        return self.items[i]

    def next_many(self, k: int) -> List[Asset]:
        return [self.next_one() for _ in range(k)]

# ---------------------------
# Composition
# ---------------------------

def place_on_canvas(
    canvas: Image.Image,
    sprite: Image.Image,
    existing: List[Tuple[int, int, int, int]],
    max_attempts: int = 80,
) -> Optional[Tuple[int, int, Tuple[int, int, int, int]]]:
    W, H = canvas.size
    # compute sprite visible bbox (alpha)
    bbox_local = alpha_bbox(sprite)
    if bbox_local is None:
        return None
    sx1, sy1, sx2, sy2 = bbox_local
    vis_w = sx2 - sx1
    vis_h = sy2 - sy1
    if vis_w <= 1 or vis_h <= 1:
        return None
    # allowed positions so that visible bbox stays inside canvas
    min_x = 0
    min_y = 0
    max_x = W - vis_w
    max_y = H - vis_h
    if max_x <= min_x or max_y <= min_y:
        return None

    for _ in range(max_attempts):
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        # bbox in canvas coords
        bbox = (x, y, x + vis_w, y + vis_h)
        # check occlusion vs existing
        ok = True
        for b in existing:
            inter = intersect_area(b, bbox)
            if inter == 0:
                continue
            a_small = min(area(b), area(bbox))
            if a_small == 0:
                continue
            if inter / a_small > MAX_OCCLUDE_RATIO:
                ok = False
                break
        if ok:
            return x - sx1, y - sy1, bbox  # paste offset (top-left of full sprite), and visible bbox
    return None


def prepare_sprite(asset: Asset) -> Tuple[Image.Image, str]:
    img = load_rgba(asset.path)
    factor = SPRITE_SCALE.get(asset.kind, 1.0)
    img = shrink_exact(img, factor)
    # small 3D-ish warp for all except cursor (keep arrow crisp)
    if asset.kind != "cursor":
        img = random_3d_warp(img)
    return img, asset.name


def compose_positive(
    cfg: CFG,
    bg_asset: Asset,
    monster_s: BalancedSampler,
    portal_s: BalancedSampler,
    gate_s: BalancedSampler,
    wp_s: BalancedSampler,
    cursor_s: BalancedSampler,
) -> Tuple[Image.Image, List[Tuple[str, Tuple[int, int, int, int]]]]:
    # Background unmodified
    bg = load_rgba(bg_asset.path)
    canvas = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    canvas.paste(bg, (0, 0))

    labels: List[Tuple[str, Tuple[int, int, int, int]]] = []
    placed_bboxes: List[Tuple[int, int, int, int]] = []

    # Determine counts
    n_mon = cfg.monsters_per_img
    n_portals = random.randint(cfg.portals_min, cfg.portals_max)
    n_gates = random.randint(cfg.gates_min, cfg.gates_max)
    n_wps = random.randint(cfg.wps_min, cfg.wps_max)

    # Monsters
    for asset in monster_s.next_many(n_mon):
        spr, name = prepare_sprite(asset)
        pos = place_on_canvas(canvas, spr, placed_bboxes)
        if pos is None:
            continue
        ox, oy, bbox = pos
        canvas.alpha_composite(spr, (ox, oy))
        labels.append((name, bbox))
        placed_bboxes.append(bbox)

    # Portals (combined sampler includes both blue/red assets shuffled)
    for _ in range(n_portals):
        asset = portal_s.next_one()
        spr, name = prepare_sprite(asset)
        pos = place_on_canvas(canvas, spr, placed_bboxes)
        if pos is None:
            continue
        ox, oy, bbox = pos
        canvas.alpha_composite(spr, (ox, oy))
        labels.append((name, bbox))
        placed_bboxes.append(bbox)

    # Diablo gates
    for _ in range(n_gates):
        asset = gate_s.next_one()
        spr, name = prepare_sprite(asset)
        pos = place_on_canvas(canvas, spr, placed_bboxes)
        if pos is None:
            continue
        ox, oy, bbox = pos
        canvas.alpha_composite(spr, (ox, oy))
        labels.append((name, bbox))
        placed_bboxes.append(bbox)

    # Waypoints
    for _ in range(n_wps):
        asset = wp_s.next_one()
        spr, name = prepare_sprite(asset)
        pos = place_on_canvas(canvas, spr, placed_bboxes)
        if pos is None:
            continue
        ox, oy, bbox = pos
        canvas.alpha_composite(spr, (ox, oy))
        labels.append((name, bbox))
        placed_bboxes.append(bbox)

    # Cursor always on top (labeled)
    cur_asset = cursor_s.next_one()
    cur_img, _ = prepare_sprite(cur_asset)
    bbox_local = alpha_bbox(cur_img)
    if bbox_local is not None:
        W, H = canvas.size
        sx1, sy1, sx2, sy2 = bbox_local
        vis_w, vis_h = sx2 - sx1, sy2 - sy1
        if vis_w > 0 and vis_h > 0 and vis_w < W and vis_h < H:
            x = random.randint(0, W - vis_w)
            y = random.randint(0, H - vis_h)
            canvas.alpha_composite(cur_img, (x - sx1, y - sy1))
            cursor_bbox = (x, y, x + vis_w, y + vis_h)
            labels.append(("cursor", cursor_bbox))

    return canvas, labels


def compose_negative(cfg: CFG, bg_asset: Asset, bg_sampler: BalancedSampler) -> Image.Image:
    base = load_rgba(bg_asset.path)
    canvas = Image.new("RGBA", base.size, (0, 0, 0, 0))
    canvas.paste(base, (0, 0))
    # Optionally paste a few other backgrounds with mild warp to add diversity
    extra_layers = random.randint(0, cfg.neg_max_extra_layers)
    for _ in range(extra_layers):
        other = bg_sampler.next_one()
        lay = load_rgba(other.path)
        lay = random_3d_warp(lay)
        # random small alpha blend by reducing opacity
        alpha = random.randint(60, 140)
        lay = lay.copy()
        L = lay.split()[3].point(lambda a: min(a, alpha))
        lay.putalpha(L)
        # random offset but inside canvas
        W, H = canvas.size
        lw, lh = lay.size
        if lw <= 0 or lh <= 0:
            continue
        x = random.randint(-lw // 4, W - 3 * lw // 4)
        y = random.randint(-lh // 4, H - 3 * lh // 4)
        canvas.alpha_composite(lay, (x, y))
    return canvas

# ---------------------------
# Saving & labels
# ---------------------------

def write_yolo_txt(path: Path, items: List[Tuple[int, float, float, float, float]]):
    lines = []
    for cls_id, cx, cy, w, h in items:
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ---------------------------
# Main pipeline
# ---------------------------

def build_class_list(assets: Dict[str, List[Asset]]) -> List[str]:
    names = []
    # monsters (unique names)
    seen = set()
    for a in assets["monster"]:
        if a.name not in seen:
            seen.add(a.name)
            names.append(a.name)
    # portals
    names.extend(["blue_portal", "red_portal"])  # fixed order
    # gates (collect present combinations)
    gate_names = sorted({a.name for a in assets["diablo_gate"]})
    names.extend(gate_names)
    # waypoint and cursor
    names.append("waypoint")
    names.append("cursor")
    return names


def save_dataset_yaml(cfg: CFG, class_names: List[str]):
    yaml_path = cfg.out_root / "dataset.yaml"
    content = (
        f"path: {cfg.out_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names: {class_names}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def prepare_output_folders(out_root: Path):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        ensure_dir(out_root / sub)


def pipeline(cfg: CFG):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    assets = build_assets(cfg)

    # Class names and mapping
    class_names = build_class_list(assets)
    name_to_id = {n: i for i, n in enumerate(class_names)}

    # Output dirs
    if cfg.out_root.exists():
        shutil.rmtree(cfg.out_root)
    prepare_output_folders(cfg.out_root)
    save_dataset_yaml(cfg, class_names)

    # Samplers (balanced)
    bg_sampler = BalancedSampler(assets["background"], seed=cfg.seed + 1)
    monster_sampler = BalancedSampler(assets["monster"], seed=cfg.seed + 2)
    portal_items = assets["blue_portal"] + assets["red_portal"]
    portal_sampler = BalancedSampler(portal_items, seed=cfg.seed + 3)
    gate_sampler = BalancedSampler(assets["diablo_gate"], seed=cfg.seed + 4)
    wp_sampler = BalancedSampler(assets["waypoint"], seed=cfg.seed + 5)
    cursor_sampler = BalancedSampler(assets["cursor"], seed=cfg.seed + 6)

    # Helper to save one composed image+labels to split
    def save_example(split: str, idx: int, image: Image.Image, labels_raw: List[Tuple[str, Tuple[int, int, int, int]]]):
        # Letterbox to cfg.img_size and convert labels
        boxed, scale, pad = letterbox(image, cfg.img_size)
        # convert to RGB
        if boxed.mode != "RGB":
            bg = Image.new("RGB", boxed.size, (0, 0, 0))
            if "A" in boxed.getbands():
                bg.paste(boxed, mask=boxed.split()[3])
            else:
                bg.paste(boxed)
            boxed = bg

        img_name = f"{split}_{idx:06d}.jpg"
        label_name = f"{split}_{idx:06d}.txt"
        img_path = cfg.out_root / "images" / split / img_name
        label_path = cfg.out_root / "labels" / split / label_name

        boxed.save(img_path, quality=95)

        yolo_items: List[Tuple[int, float, float, float, float]] = []
        for cname, box in labels_raw:
            if cname not in name_to_id:
                continue
            cx, cy, w, h = normalize_yolo(box, scale, pad, cfg.img_size)
            if w <= 0 or h <= 0:
                continue
            yolo_items.append((name_to_id[cname], cx, cy, w, h))
        write_yolo_txt(label_path, yolo_items)

    # Generate TRAIN positives
    pbar = tqdm(range(cfg.train_pos), desc="Generating train positives", ncols=100)
    for i in pbar:
        bg = bg_sampler.next_one()
        img, labels = compose_positive(cfg, bg, monster_sampler, portal_sampler, gate_sampler, wp_sampler, cursor_sampler)
        save_example("train", i, img, labels)

    # Generate TRAIN negatives
    pbar = tqdm(range(cfg.train_neg), desc="Generating train negatives", ncols=100)
    for i in pbar:
        bg = bg_sampler.next_one()
        img = compose_negative(cfg, bg, bg_sampler)
        # no labels
        save_example("train", i + cfg.train_pos, img, [])

    # Generate VAL positives
    pbar = tqdm(range(cfg.val_pos), desc="Generating val positives", ncols=100)
    for i in pbar:
        bg = bg_sampler.next_one()
        img, labels = compose_positive(cfg, bg, monster_sampler, portal_sampler, gate_sampler, wp_sampler, cursor_sampler)
        save_example("val", i, img, labels)

    # Generate VAL negatives
    pbar = tqdm(range(cfg.val_neg), desc="Generating val negatives", ncols=100)
    for i in pbar:
        bg = bg_sampler.next_one()
        img = compose_negative(cfg, bg, bg_sampler)
        save_example("val", i + cfg.val_pos, img, [])

    print("\nDone.")
    print(f"Dataset root: {cfg.out_root}")
    print("Class map:")
    for i, n in enumerate(class_names):
        print(f"  {i:>3} -> {n}")


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> CFG:
    ap = argparse.ArgumentParser(description="Generate YOLOv11 data for D2R")
    ap.add_argument("--raw-root", type=Path, default=Path("raw_images"))
    ap.add_argument("--out-root", type=Path, default=Path("yolo_input"))
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-pos", type=int, default=2000)
    ap.add_argument("--train-neg", type=int, default=1000)
    ap.add_argument("--val-pos", type=int, default=400)
    ap.add_argument("--val-neg", type=int, default=200)
    ap.add_argument("--monsters-per-img", type=int, default=20)
    ap.add_argument("--portals-min", type=int, default=0)
    ap.add_argument("--portals-max", type=int, default=2)
    ap.add_argument("--gates-min", type=int, default=0)
    ap.add_argument("--gates-max", type=int, default=2)
    ap.add_argument("--wps-min", type=int, default=0)
    ap.add_argument("--wps-max", type=int, default=2)
    ap.add_argument("--neg-max-extra-layers", type=int, default=3)

    args = ap.parse_args()

    cfg = CFG(
        raw_root=args.raw_root,
        out_root=args.out_root,
        img_size=args.img_size,
        seed=args.seed,
        train_pos=args.train_pos,
        train_neg=args.train_neg,
        val_pos=args.val_pos,
        val_neg=args.val_neg,
        monsters_per_img=args.monsters_per_img,
        portals_min=args.portals_min,
        portals_max=args.portals_max,
        gates_min=args.gates_min,
        gates_max=args.gates_max,
        wps_min=args.wps_min,
        wps_max=args.wps_max,
        neg_max_extra_layers=args.neg_max_extra_layers,
    )
    return cfg


def main():
    cfg = parse_args()
    pipeline(cfg)


if __name__ == "__main__":
    main()
