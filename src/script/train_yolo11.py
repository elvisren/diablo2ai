#!/usr/bin/env python3
# train_yolo11.py
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11 on synthetic D2R data with early stopping")
    p.add_argument("--data", type=str, default="out/dataset.yaml",
                   help="Ultralytics dataset.yaml path")
    p.add_argument("--model", type=str, default="../models/yolo11x.pt",
                   help="Base weights path (must exist; no auto-download)")
    p.add_argument("--imgsz", type=int, default=1080, help="Training image size (square or int)")
    p.add_argument("--epochs", type=int, default=500, help="Max epochs")
    p.add_argument("--patience", type=int, default=50,
                   help="Early stopping patience (epochs with no val improvement)")
    p.add_argument("--batch", type=int, default=-1,
                   help="Batch size (-1 = auto). Use a smaller value if you OOM.")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: 'auto', 'cpu', 'mps', 'cuda', or '0,1' etc.")
    p.add_argument("--project", type=str, default="runs/train", help="Project directory")
    p.add_argument("--name", type=str, default="d2r-yolo11x", help="Run name")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    p.add_argument("--cos_lr", action="store_true",
                   help="Use cosine LR schedule (recommended for long runs)")
    p.add_argument("--lr0", type=float, default=None, help="Override initial LR (optional)")
    p.add_argument("--weight_decay", type=float, default=None, help="Override weight decay (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Base model not found: {model_path}\n"
            f"Please place the weights there (no auto-download by design)."
        )

    # Load model strictly from local weights (won't download).
    model = YOLO(str(model_path))

    # Assemble training kwargs
    train_kwargs = dict(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,     # early stopping on val metric
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        verbose=True,
        save=True,                  # save last, best, checkpoints
        save_period=0,              # set >0 to save every N epochs
        pretrained=False,           # we already loaded pretrained weights explicitly
        deterministic=True,
        plots=True,                 # save learning curve plots
    )

    # Optional sched/optim tweaks
    if args.cos_lr:
        train_kwargs["cos_lr"] = True
    if args.lr0 is not None:
        train_kwargs["lr0"] = float(args.lr0)
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = float(args.weight_decay)

    # Kick off training (Ultralytics shows a live progress bar + per-epoch metrics)
    results = model.train(**train_kwargs)

    # Validate on the val split and print summary
    metrics = model.val(split="val", imgsz=args.imgsz, device=args.device, plots=True)
    # metrics is a SimpleNamespace; common fields include:
    # metrics.box.map, map50, map75, precision, recall, fitness
    print("\n=== Final Validation Metrics ===")
    try:
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50:    {metrics.box.map50:.4f}")
        print(f"mAP75:    {metrics.box.map75:.4f}")
        print(f"Precision:{metrics.box.precision:.4f}")
        print(f"Recall:   {metrics.box.recall:.4f}")
        print(f"Fitness:  {metrics.fitness:.4f}")
    except Exception:
        print(metrics)

    # Helpful paths
    run_dir = Path(results.save_dir)  # runs/train/<name> (or .../nameX)
    print(f"\nArtifacts saved in: {run_dir.resolve()}")
    print(f" - Best weights: { (run_dir / 'weights' / 'best.pt').resolve() }")
    print(f" - Last weights: { (run_dir / 'weights' / 'last.pt').resolve() }")
    print(f" - Curves/plots: { (run_dir / 'results.png').resolve() }")
    print(f" - CSV metrics : { (run_dir / 'results.csv').resolve() }")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
