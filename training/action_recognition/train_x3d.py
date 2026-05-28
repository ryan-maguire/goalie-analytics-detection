#!/usr/bin/env python3
"""Phase 2: X3D-M fine-tune on hockey clips.

3D CNN trained on Kinetics-400, head replaced with 3-way classifier:
  no_event / shot_save / goal

Key design choices:
  - Frames per clip: 16 (X3D-M's native input length)
  - Spatial size: 224x224 (X3D-M's native)
  - Clip mode: extract 16 frames evenly from each window
  - Same game-disjoint train/val/test split as the baseline
  - sqrt_inv class weighting
  - Random horizontal flip augmentation (hockey is L/R symmetric)
  - Optional: freeze backbone for first N epochs (faster initial training,
    then unfreeze for fine-tuning)

Target: beat baseline test goal F1 (0.122) by a wide margin —
the temporal modeling is the entire point of moving to X3D.
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
from torch.utils.data import DataLoader

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


def build_x3d_m(n_classes: int = 3) -> nn.Module:
    """X3D-M with Kinetics-400 pretrained weights + replaced head.

    X3D-M's native input is 16 frames × 224×224 with a hard-coded
    AvgPool3d(kT=16, kH=7, kW=7) at the head. To support arbitrary
    input shapes (so we can train faster at 8 frames × 160 px), we
    swap the fixed pool for AdaptiveAvgPool3d((1, 1, 1)) which works
    at any post-backbone resolution.
    """
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m",
                            pretrained=True, source="github")
    # head structure: ResNetBasicHead
    #   .pool      = ProjectedPool (pre_conv → pre_norm → pre_act →
    #                pool (the fixed AvgPool3d(16,7,7) that breaks) →
    #                post_conv → post_act)
    #   .dropout, .proj (Linear 2048→400), .output_pool (adaptive)
    # Replace ONLY the inner pool with adaptive so the pipeline accepts
    # any input shape; the surrounding convs that lift channel count
    # 192 → 432 → 2048 stay intact.
    head = model.blocks[5]
    head.pool.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    in_feats = head.proj.in_features
    head.proj = nn.Linear(in_feats, n_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all params except the new head — for warmup epochs."""
    for name, p in model.named_parameters():
        p.requires_grad = "blocks.5.proj" in name


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(y.tolist())
    return per_class_metrics(np.array(all_labels), np.array(all_preds))


def per_class_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    n = len(LABEL_NAMES)
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
    macro_f1 = sum(per_class[n]["f1"] for n in LABEL_NAMES) / len(LABEL_NAMES)
    return {"per_class": per_class, "macro_f1": macro_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                     default=REPO / "training" / "action_recognition" / "manifest.json")
    ap.add_argument("--out-dir", type=Path,
                     default=REPO / "training" / "action_recognition" / "runs" / "x3d_m_v1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=2,
                     help="Epochs to train with frozen backbone (head only)")
    ap.add_argument("--batch-size", type=int, default=8,
                     help="X3D is 3D — smaller batch than 2D ResNet")
    ap.add_argument("--lr", type=float, default=1e-3,
                     help="LR for the head; backbone uses lr/10 after unfreeze")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-frames", type=int, default=16,
                     help="X3D-M's native input length")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit-train", type=int, default=0,
                     help="If >0, cap training clips (for smoke runs).")
    ap.add_argument("--ffmpeg-only", action="store_true",
                     help="Use ffmpeg subprocess instead of PyAV.")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}", file=sys.stderr)

    records = load_manifest(args.manifest)
    splits = split_by_game(records)
    for s, recs in splits.items():
        c = Counter(r.label for r in recs)
        print(f"  {s:5} {len(recs):>4} clips  {dict(c)}", file=sys.stderr)

    train_recs = splits["train"]
    if args.limit_train > 0:
        train_recs = train_recs[:args.limit_train]
        print(f"capped training set to {len(train_recs)}", file=sys.stderr)

    use_pyav = not args.ffmpeg_only
    train_ds = ActionClipDataset(train_recs, mode="clip",
                                    n_frames=args.n_frames, size=args.size,
                                    use_pyav=use_pyav, augment=True)
    val_ds = ActionClipDataset(splits["val"], mode="clip",
                                  n_frames=args.n_frames, size=args.size,
                                  use_pyav=use_pyav, augment=False)
    test_ds = ActionClipDataset(splits["test"], mode="clip",
                                   n_frames=args.n_frames, size=args.size,
                                   use_pyav=use_pyav, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.workers,
                                 persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.workers,
                               persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers)

    print(f"\nbuilding X3D-M with Kinetics-400 weights...", file=sys.stderr)
    model = build_x3d_m(n_classes=len(LABEL_NAMES)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.1f}M", file=sys.stderr)

    w = class_weights(train_recs).to(device)
    print(f"  class weights: {dict(zip(LABEL_NAMES, [round(v,2) for v in w.cpu().tolist()]))}",
          file=sys.stderr)
    criterion = nn.CrossEntropyLoss(weight=w)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_log = []
    best_val_macro_f1 = 0.0
    best_val_goal_f1 = 0.0    # also track goal F1 separately
    best_path = args.out_dir / "best.pt"

    # Phase A: warmup with frozen backbone, head-only training
    print(f"\n=== Warmup: head-only ({args.warmup_epochs} epochs) ===", file=sys.stderr)
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.warmup_epochs):
        model.train()
        epoch_start = time.time()
        rl, rn = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rl += loss.item() * x.size(0); rn += x.size(0)
            if (i + 1) % 10 == 0:
                avg = rl / rn
                elapsed = time.time() - epoch_start
                eta = elapsed * (len(train_loader) - (i+1)) / (i+1)
                print(f"  [warm ep{epoch+1}] batch {i+1}/{len(train_loader)}  "
                      f"loss={avg:.3f}  ({elapsed:.0f}s, ETA {eta:.0f}s)",
                      file=sys.stderr)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        train_loss = rl / max(rn, 1)
        pc = "  ".join(f"{n}: F1={m['f1']:.2f}"
                        for n, m in val_metrics['per_class'].items())
        print(f"warm ep{epoch+1}/{args.warmup_epochs}  loss={train_loss:.3f}  "
              f"val macro-F1={val_metrics['macro_f1']:.3f}  goal-F1={val_metrics['per_class']['goal']['f1']:.3f}  "
              f"({elapsed:.0f}s)", file=sys.stderr)
        print(f"  {pc}", file=sys.stderr)
        train_log.append({"phase": "warm", "epoch": epoch+1, "train_loss": train_loss,
                           "val_metrics": val_metrics, "elapsed_sec": elapsed})

    # Phase B: full fine-tune (unfreeze backbone, lower LR)
    print(f"\n=== Fine-tune: full unfrozen ({args.epochs - args.warmup_epochs} epochs) ===",
          file=sys.stderr)
    unfreeze_all(model)
    # Two-group LR: head at full LR, backbone at lr/10
    head_params = [p for n, p in model.named_parameters() if "blocks.5.proj" in n]
    backbone_params = [p for n, p in model.named_parameters() if "blocks.5.proj" not in n]
    optimizer = torch.optim.AdamW([
        {"params": head_params,      "lr": args.lr},
        {"params": backbone_params,  "lr": args.lr / 10},
    ], weight_decay=1e-4)
    remaining = args.epochs - args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

    for ep_idx in range(remaining):
        epoch = args.warmup_epochs + ep_idx + 1
        model.train()
        epoch_start = time.time()
        rl, rn = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rl += loss.item() * x.size(0); rn += x.size(0)
            if (i + 1) % 10 == 0:
                avg = rl / rn
                elapsed = time.time() - epoch_start
                eta = elapsed * (len(train_loader) - (i+1)) / (i+1)
                print(f"  [ep{epoch}] batch {i+1}/{len(train_loader)}  "
                      f"loss={avg:.3f}  ({elapsed:.0f}s, ETA {eta:.0f}s)",
                      file=sys.stderr)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        train_loss = rl / max(rn, 1)
        pc = "  ".join(f"{n}: F1={m['f1']:.2f}"
                        for n, m in val_metrics['per_class'].items())
        print(f"epoch {epoch}/{args.epochs}  loss={train_loss:.3f}  "
              f"val macro-F1={val_metrics['macro_f1']:.3f}  goal-F1={val_metrics['per_class']['goal']['f1']:.3f}  "
              f"({elapsed:.0f}s)", file=sys.stderr)
        print(f"  {pc}", file=sys.stderr)
        train_log.append({"phase": "ft", "epoch": epoch, "train_loss": train_loss,
                           "val_metrics": val_metrics, "elapsed_sec": elapsed})

        # Save on either macro-F1 or goal-F1 new best
        save = False
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]; save = True
        if val_metrics["per_class"]["goal"]["f1"] > best_val_goal_f1:
            best_val_goal_f1 = val_metrics["per_class"]["goal"]["f1"]; save = True
        if save:
            torch.save({
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "epoch": epoch,
            }, best_path)
            print(f"  → saved best (macro={best_val_macro_f1:.3f}, "
                  f"goal={best_val_goal_f1:.3f}) to {best_path}", file=sys.stderr)

    (args.out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))

    # Test eval
    print(f"\n=== TEST EVAL (best checkpoint) ===", file=sys.stderr)
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    print(f"TEST macro-F1 = {test_metrics['macro_f1']:.3f}", file=sys.stderr)
    for n, m in test_metrics['per_class'].items():
        print(f"  {n:<10} P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})", file=sys.stderr)
    (args.out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    # Phase 2 go/no-go
    print(f"\n=== PHASE 2 DECISION ===", file=sys.stderr)
    macro = test_metrics["macro_f1"]
    goal_f1 = test_metrics["per_class"]["goal"]["f1"]
    baseline_macro = 0.471   # ResNet18 single-frame baseline
    baseline_goal  = 0.122
    if goal_f1 > baseline_goal * 2:
        print(f"✅ Goal F1 {goal_f1:.3f} > 2× baseline ({baseline_goal:.3f}) "
              f"— X3D temporal model is working. Ship to Phase 3 (integration).",
              file=sys.stderr)
    elif macro > baseline_macro:
        print(f"⚠ Macro-F1 {macro:.3f} > baseline ({baseline_macro:.3f}) but "
              f"goal F1 {goal_f1:.3f} not 2× better than {baseline_goal:.3f}. "
              f"Iterate (oversampling, focal loss, more epochs).", file=sys.stderr)
    else:
        print(f"❌ Macro-F1 {macro:.3f} <= baseline ({baseline_macro:.3f}). "
              f"X3D didn't help. Investigate.", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
