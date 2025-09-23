#!/usr/bin/env python3
# Simple YOLO11 training script (object detection)

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics import settings
settings.update({'tensorboard': True})


def pick_device() -> str:
    # Prefer Apple MPS on macOS, else CUDA, else CPU
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_data_yaml(data_dir: Path) -> str:
    # Common names people use
    for name in ("dataset.yaml", "data.yaml"):
        p = data_dir / name
        if p.is_file():
            return str(p)
    print(f"[error] Couldn’t find dataset yaml in {data_dir} (expected dataset.yaml or data.yaml).", file=sys.stderr)
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 for object detection.")
    parser.add_argument("--weights", default="../../models/yolo11x.pt", help="Path to YOLO11 weights")
    parser.add_argument("--data_dir", default="yolo_input", help="Directory containing dataset.yaml / data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (short side)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 lets Ultralytics auto-tune)")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--name", default="yolo11x_train", help="Run name (saved under runs/)")
    parser.add_argument("--resume", action="store_true", help="Resume last training run")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_yaml = find_data_yaml(data_dir)

    device = pick_device()

    print(f"[info] device={device}")
    print(f"[info] data={data_yaml}")
    print(f"[info] weights={args.weights}")

    # Explicitly set task to avoid warnings
    model = YOLO(args.weights, task="detect")

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        batch=args.batch,
        workers=args.workers,
        project="runs",        # all runs saved here
        name=args.name,        # runs/<task>/<name> (e.g., runs/detect/yolo11x_train)
        cache="ram",           # speed up training if you have RAM
        resume=args.resume,    # resume if a previous run exists
        plots=True,            # save training plots
        save=True,             # save checkpoints
        seed=42,                # make results reproducible-ish
        verbose=True,
    )

    print("\n✅ Training started. To view live metrics in TensorBoard:")
    print("   tensorboard --logdir runs\n")


if __name__ == "__main__":
    main()
