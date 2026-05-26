"""
Fine-tune HockeyAI YOLOv8 on the 8-class set (original 7 + new `shot`).

Assumes you've already:
  1. Run util/extract_label_frames.py to populate data/labels/images/
  2. Run util/prelabel_frames.py to seed data/labels/labels/
  3. Hand-corrected the labels in your preferred tool
     (see data/labels/README.md)

This script:
  1. Pairs each .jpg in --images-dir with its .txt in --labels-dir
  2. Splits 80/20 train/val deterministically by filename hash
  3. Creates the ultralytics directory structure under --work-dir
     (symlinks back to the originals — no copies)
  4. Writes data.yaml
  5. Calls ultralytics YOLO(...).train(...) starting from the cached
     HockeyAI weights (so the existing classes' learning is preserved)

Usage:
    python3 util/train_yolo_finetune.py \\
        --epochs 50 --batch 8 --device mps

Tip on --device:
  cpu  — slow but always works
  mps  — Apple Silicon GPU; ~5-10x faster than cpu on M2/M3
  cuda — NVIDIA; only if you have one
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_seg.net_detection import _load_model_lazy  # ensures model cached locally


CLASS_ORDER = [
    "centriod", "faceoff", "goal", "goalie",
    "player", "puck", "referee", "shot",
]


def _split_train_val(stems: list[str], val_frac: float, seed: str = "split-v1"):
    """Deterministic split by hash(seed + stem)."""
    train, val = [], []
    for stem in sorted(stems):
        h = hashlib.md5((seed + ":" + stem).encode()).hexdigest()
        # First 8 hex chars → int [0, 16^8); compare to val_frac * 16^8
        bucket = int(h[:8], 16) / float(0x10000_0000)
        (val if bucket < val_frac else train).append(stem)
    return train, val


def _symlink_files(stems, src_images, src_labels, dst_images, dst_labels):
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for ext, sdir, ddir in (
            (".jpg", src_images, dst_images),
            (".txt", src_labels, dst_labels),
        ):
            src = sdir / (stem + ext)
            dst = ddir / (stem + ext)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.resolve(), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/labels/images")
    ap.add_argument("--labels-dir", default="data/labels/labels")
    ap.add_argument("--work-dir",   default="data/labels/_yolo_workdir",
                    help="ultralytics dir layout — symlinks back to originals")
    ap.add_argument("--val-frac",   type=float, default=0.2)
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--batch",      type=int,   default=8)
    ap.add_argument("--imgsz",      type=int,   default=640)
    ap.add_argument("--device",     default="mps",
                    help="mps (Apple Silicon), cpu, or cuda")
    ap.add_argument("--run-name",   default="hockeyai_shot_finetune")
    ap.add_argument("--base-weights", default=None,
                    help="path to base YOLOv8 weights to fine-tune from. "
                         "If omitted, uses the cached HockeyAI download.")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    work_dir   = Path(args.work_dir)

    image_stems = {p.stem for p in images_dir.glob("*.jpg")}
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    paired = sorted(image_stems & label_stems)
    print(f"Found {len(image_stems)} images, {len(label_stems)} labels, "
          f"{len(paired)} paired", file=sys.stderr)
    if not paired:
        print("ERROR: no paired (image, label) found. Run extract + prelabel first.",
              file=sys.stderr); return 2

    train_stems, val_stems = _split_train_val(paired, args.val_frac)
    print(f"Split: {len(train_stems)} train, {len(val_stems)} val", file=sys.stderr)

    # Build ultralytics dir layout via symlinks
    if work_dir.exists():
        shutil.rmtree(work_dir)
    _symlink_files(train_stems, images_dir, labels_dir,
                   work_dir / "images" / "train", work_dir / "labels" / "train")
    _symlink_files(val_stems,   images_dir, labels_dir,
                   work_dir / "images" / "val",   work_dir / "labels" / "val")

    # data.yaml
    yaml_path = work_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {work_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write(f"nc: {len(CLASS_ORDER)}\n")
        f.write("names:\n")
        for i, name in enumerate(CLASS_ORDER):
            f.write(f"  {i}: {name}\n")
    print(f"Wrote {yaml_path}", file=sys.stderr)

    # Resolve base weights
    if args.base_weights:
        base_weights = args.base_weights
    else:
        model = _load_model_lazy()
        if model is None:
            print("ERROR: cannot resolve HockeyAI weights and no --base-weights",
                  file=sys.stderr); return 2
        # ultralytics YOLO objects expose .ckpt_path for the original .pt file
        base_weights = str(model.ckpt_path) if hasattr(model, "ckpt_path") else None
        if not base_weights:
            print("ERROR: could not extract weights path from cached model",
                  file=sys.stderr); return 2
    print(f"Base weights: {base_weights}", file=sys.stderr)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed", file=sys.stderr); return 2

    yolo = YOLO(base_weights)
    print(f"Training: epochs={args.epochs}  batch={args.batch}  "
          f"device={args.device}  imgsz={args.imgsz}", file=sys.stderr)
    # ultralytics prepends `runs/<task>/` to the project path unless it's
    # absolute — pass the resolved absolute path so the output lands
    # under work_dir as advertised, not under cwd/runs/detect/...
    project_path = (work_dir / "runs").resolve()
    yolo.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.run_name,
        project=str(project_path),
        exist_ok=True,
    )
    print("Training complete.", file=sys.stderr)
    print(f"Best model: {work_dir / 'runs' / args.run_name / 'weights' / 'best.pt'}",
          file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
