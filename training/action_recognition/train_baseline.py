#!/usr/bin/env python3
"""Phase 1 baseline: per-frame ResNet18 on clip-averaged images.

Trains a 3-class classifier (no_event / shot_save / goal) on the
mean-frame of each clip. This is a sanity-check baseline — if it
can't beat random (macro-F1 > 0.40), we have a labeling or data
problem before investing in Phase 2 (X3D-M).

Targets MPS on Apple Silicon, falls back to CPU.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training" / "action_recognition"))

from dataset import (
    load_manifest, split_by_game, ActionClipDataset, class_weights,
    LABEL_NAMES, LABEL_TO_IDX,
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(n_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """ResNet18 with replaced head. ~11M params, fast to train."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
    return per_class_metrics(np.array(all_labels), np.array(all_preds))


def per_class_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    n_classes = len(LABEL_NAMES)
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        tp = int(((preds == i) & (labels == i)).sum())
        fp = int(((preds == i) & (labels != i)).sum())
        fn = int(((preds != i) & (labels == i)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn,
                              "p": prec, "r": rec, "f1": f1}
    macro_f1 = sum(per_class[n]["f1"] for n in LABEL_NAMES) / n_classes
    return {"per_class": per_class, "macro_f1": macro_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                     default=REPO / "training" / "action_recognition" / "manifest.json")
    ap.add_argument("--out-dir", type=Path,
                     default=REPO / "training" / "action_recognition" / "runs" / "baseline_resnet18")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-frames", type=int, default=8,
                     help="Frames per clip — averaged in per_frame_mean mode.")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit-train", type=int, default=0,
                     help="If >0, cap training clips (for quick smoke runs).")
    ap.add_argument("--ffmpeg-only", action="store_true",
                     help="Use ffmpeg subprocess instead of PyAV. Faster "
                          "in practice on this dataset because clips are short.")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}", file=sys.stderr)

    records = load_manifest(args.manifest)
    print(f"loaded {len(records)} clips", file=sys.stderr)
    splits = split_by_game(records)
    for s, recs in splits.items():
        c = Counter(r.label for r in recs)
        print(f"  {s:5} {len(recs):>4} clips  {dict(c)}", file=sys.stderr)

    train_recs = splits["train"]
    if args.limit_train > 0:
        train_recs = train_recs[:args.limit_train]
        print(f"capped training set to {len(train_recs)}", file=sys.stderr)

    use_pyav = not args.ffmpeg_only
    train_ds = ActionClipDataset(train_recs, mode="middle_frame",
                                    n_frames=args.n_frames, size=args.size,
                                    use_pyav=use_pyav, augment=True)
    val_ds   = ActionClipDataset(splits["val"], mode="middle_frame",
                                    n_frames=args.n_frames, size=args.size,
                                    use_pyav=use_pyav, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.workers,
                                 persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              persistent_workers=args.workers > 0)

    model = build_model(n_classes=len(LABEL_NAMES), pretrained=True).to(device)
    w = class_weights(train_recs).to(device)
    print(f"class weights: {dict(zip(LABEL_NAMES, w.cpu().tolist()))}", file=sys.stderr)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_log = []
    best_val_macro_f1 = 0.0
    best_path = args.out_dir / "best.pt"

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_n    = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_n    += x.size(0)
            if (i + 1) % 10 == 0:
                avg = running_loss / running_n
                elapsed = time.time() - epoch_start
                eta = elapsed * (len(train_loader) - (i+1)) / (i+1)
                print(f"  ep{epoch+1} batch {i+1}/{len(train_loader)}  "
                      f"loss={avg:.3f}  ({elapsed:.0f}s, ETA {eta:.0f}s)",
                      file=sys.stderr)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / max(running_n, 1)
        elapsed = time.time() - epoch_start
        per_class_str = "  ".join(
            f"{n}: P={m['p']:.2f} R={m['r']:.2f} F1={m['f1']:.2f}"
            for n, m in val_metrics["per_class"].items()
        )
        print(f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.3f}  "
              f"val macro-F1={val_metrics['macro_f1']:.3f}  ({elapsed:.0f}s)",
              file=sys.stderr)
        print(f"  {per_class_str}", file=sys.stderr)

        train_log.append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class": val_metrics["per_class"],
            "elapsed_sec": elapsed,
        })

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "epoch": epoch + 1,
            }, best_path)
            print(f"  → saved best ({best_val_macro_f1:.3f}) to {best_path}",
                  file=sys.stderr)

    (args.out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))

    # Final test set eval with best checkpoint
    print(f"\n=== TEST EVAL (loading best checkpoint) ===", file=sys.stderr)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_ds = ActionClipDataset(splits["test"], mode="middle_frame",
                                   n_frames=args.n_frames, size=args.size,
                                   use_pyav=use_pyav, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)
    test_metrics = evaluate(model, test_loader, device)
    print(f"TEST macro-F1 = {test_metrics['macro_f1']:.3f}", file=sys.stderr)
    for n, m in test_metrics["per_class"].items():
        print(f"  {n:<10} P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})", file=sys.stderr)
    (args.out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    # ── Phase 1 go/no-go ──────────────────────────────────────────
    print(f"\n=== PHASE 1 DECISION ===", file=sys.stderr)
    macro = test_metrics["macro_f1"]
    goal_r = test_metrics["per_class"]["goal"]["r"]
    if macro > 0.40 and goal_r > 0.20:
        print(f"✅ Macro-F1 {macro:.3f} > 0.40 AND goal recall {goal_r:.3f} > 0.20 "
              f"→ PROCEED to Phase 2 (X3D-M)", file=sys.stderr)
    elif macro > 0.40:
        print(f"⚠ Macro-F1 {macro:.3f} > 0.40 but goal recall {goal_r:.3f} <= 0.20 "
              f"— overall ok but goals weak. Consider oversampling or X3D first.",
              file=sys.stderr)
    else:
        print(f"❌ Macro-F1 {macro:.3f} <= 0.40 — STOP, investigate labels/data",
              file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
