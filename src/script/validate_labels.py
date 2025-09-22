#!/usr/bin/env python3
# validate_labels.py
from pathlib import Path

DATASET_YAML = Path("out/dataset.yaml")

import yaml, math

def check_line(parts, nc):
    if len(parts) != 5:
        return "expected 5 columns"
    try:
        c = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
    except Exception:
        return "non-numeric value(s)"
    if c < 0 or c >= nc:
        return f"class id {c} out of range 0..{nc-1}"
    if any(math.isnan(v) or math.isinf(v) for v in (x, y, w, h)):
        return "NaN/Inf present"
    # normalized in [0,1], allow tiny epsilon
    eps = 1e-6
    if not (-eps <= x <= 1+eps and -eps <= y <= 1+eps and w > 0 and h > 0 and w <= 1+eps and h <= 1+eps):
        return f"values out of [0,1] or non-positive: x={x:.3f},y={y:.3f},w={w:.6f},h={h:.6f}"
    return None

def main():
    if not DATASET_YAML.exists():
        raise SystemExit(f"Not found: {DATASET_YAML}")
    cfg = yaml.safe_load(DATASET_YAML.read_text())
    names = cfg["names"]
    if isinstance(names, dict):
        nc = 1 + max(int(k) for k in names.keys()) if names else 0
    else:
        nc = len(names)
    problems = 0
    for split in ("train", "val"):
        lbl_dir = Path("out/labels") / split
        if not lbl_dir.exists():
            print(f"[skip] {lbl_dir} not found")
            continue
        for p in sorted(lbl_dir.glob("*.txt")):
            text = p.read_text().strip()
            if not text:
                print(f"[WARN] empty label file: {p}")
                continue
            for i, line in enumerate(text.splitlines(), 1):
                parts = line.strip().split()
                err = check_line(parts, nc)
                if err:
                    print(f"[BAD ] {p}:{i}: {err} | line='{line}'")
                    problems += 1
    if problems == 0:
        print("✓ All labels look sane.")
    else:
        print(f"Found {problems} problematic label lines. Fix or delete those images/labels and retry.")
if __name__ == "__main__":
    main()
