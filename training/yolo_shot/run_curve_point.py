"""Orchestrator: run the full YOLO-shot pipeline for ONE curve point.

For a given train_size N, this:
  1. Picks the first N match_ids from splits.json["train_pool"]
  2. Ensures HockeyAI features are extracted for those + test matches
     (skips if cached on disk)
  3. Extracts positive frames (shot moments) for the train subset
  4. Mines hard-negative frames for the train subset
  5. Pre-labels every frame with HockeyAI's 7 classes
  6. Auto-labels the shot bbox on positives, using TIGHT factors
     (W=1.0, H=0.8 — fixes the prior 1.8/1.5 over-generous baseline)
  7. Builds a per-curve-point dataset (symlinks) covering only the
     train subset's frames
  8. Fine-tunes YOLO from HockeyAI weights for --epochs
  9. Runs inference on the TEST matches at 1fps
 10. Bridges per-second probs → predictions.csv
 11. Calls training/learning_curve/run_curve.py record

This runs as a SHELL of subprocess.run() calls — each step is an
existing util/ script, kept as the source of truth for its phase.

Idempotency: each phase checks for its output on disk and skips if
already present. Force a re-run with --force-phase=<name>.

Usage:
    # ONE-TIME setup
    python3 training/yolo_shot/synth_customer.py    # creates CUST_LEARNCURVE.json
    python3 training/learning_curve/splits.py       # creates splits.json

    # Per curve point
    python3 training/yolo_shot/run_curve_point.py --train-size 4 --epochs 30
    python3 training/yolo_shot/run_curve_point.py --train-size 8 --epochs 30
    python3 training/yolo_shot/run_curve_point.py --train-size 12 --epochs 30
    python3 training/yolo_shot/run_curve_point.py --train-size 16 --epochs 50

Each run records a (train_size, F1) point into
training/learning_curve/results.json. After the last one, plot:
    python3 training/learning_curve/run_curve.py plot \\
        --baseline-f1 0.422 --target-f1 0.90
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO       = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS     = REPO / "training" / "learning_curve" / "splits.json"
RECORD_CMD = REPO / "training" / "learning_curve" / "run_curve.py"
CUST_JSON  = REPO / "data" / "customers" / "CUST_LEARNCURVE.json"

# Default tight bbox factors — the prior failure was W=1.8, H=1.5
# (test F1=0.232). These constrain the shot bbox to roughly the
# goalmouth + immediate slot.
DEFAULT_W_FACTOR = 1.0
DEFAULT_H_FACTOR = 0.8


def sh(cmd: list, *, check: bool = True) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    rc = subprocess.call([str(c) for c in cmd])
    if rc != 0 and check:
        raise RuntimeError(f"command failed (rc={rc}): {' '.join(map(str, cmd))}")
    return rc


def phase_features(vids: list[str], force: bool) -> None:
    """Extract per-second HockeyAI features for each vID. Cached."""
    feat_dir = REPO / "data" / "output" / "yolo_features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    for vid in vids:
        tsv = feat_dir / f"{vid}.tsv"
        if tsv.exists() and not force:
            continue
        video = REPO / "data" / "videos" / f"{vid}.mp4"
        if not video.exists():
            print(f"  [skip] no video for {vid}", file=sys.stderr)
            continue
        sh([sys.executable, "util/extract_yolo_features.py",
            "--video", video, "--out", tsv, "--fps", 1.0])


def phase_positives(train_vids: list[str], force: bool) -> None:
    cmd = [sys.executable, "util/extract_label_frames_v2.py",
           "--customers", CUST_JSON,
           "--vIDs", *train_vids]
    if force:
        # The script is idempotent on filename (skips if exists), so
        # "force" means deleting the existing _pos_ images first.
        img_dir = REPO / "data" / "labels" / "images"
        for f in img_dir.glob("*_pos_*.jpg"):
            mid = f.stem.split("_")[0]
            if mid in train_vids:
                f.unlink()
    sh(cmd)


def phase_hard_negs(train_vids: list[str], target_count: int) -> None:
    sh([sys.executable, "util/sample_hard_negatives.py",
        "--customers", CUST_JSON,
        "--target-count", target_count])
    # The script picks top-N globally; we don't restrict to train_vids
    # here because the features-dir already determines which vIDs are
    # candidates (and we only ran feature extraction for train + test).
    # Test-vid hardnegs are filtered out at dataset-assembly time below.


def phase_prelabel(force: bool) -> None:
    cmd = [sys.executable, "util/prelabel_frames.py"]
    if force:
        cmd.append("--force")
    sh(cmd)


def phase_autolabel(train_vids: list[str], w_factor: float,
                     h_factor: float) -> None:
    # autolabel_shots.py supports --filter for one substring; loop per vID
    for vid in train_vids:
        sh([sys.executable, "util/autolabel_shots.py",
            "--commit",
            "--filter", vid,
            "--w-factor", w_factor,
            "--h-factor", h_factor])


def phase_build_dataset(train_vids: list[str], curve_dir: Path) -> Path:
    """Symlink the train subset's frames + labels into curve_dir/dataset."""
    src_imgs = REPO / "data" / "labels" / "images"
    src_lbls = REPO / "data" / "labels" / "labels"
    ds_imgs = curve_dir / "dataset" / "images"
    ds_lbls = curve_dir / "dataset" / "labels"
    for d in (ds_imgs, ds_lbls):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    n = 0
    train_set = set(train_vids)
    for img in src_imgs.glob("*.jpg"):
        mid = img.stem.split("_")[0]
        if mid not in train_set:
            continue
        lbl = src_lbls / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        os.symlink(img.resolve(), ds_imgs / img.name)
        os.symlink(lbl.resolve(), ds_lbls / lbl.name)
        n += 1
    print(f"  dataset: {n} frame/label pairs", file=sys.stderr)
    return curve_dir / "dataset"


def phase_train(dataset: Path, work_dir: Path, run_name: str,
                 epochs: int, batch: int, device: str) -> Path:
    sh([sys.executable, "util/train_yolo_finetune.py",
        "--images-dir", dataset / "images",
        "--labels-dir", dataset / "labels",
        "--work-dir",   work_dir,
        "--run-name",   run_name,
        "--epochs",     epochs,
        "--batch",      batch,
        "--device",     device])
    # ultralytics dumps to work_dir/runs/<run_name>/weights/best.pt
    weights = work_dir / "runs" / run_name / "weights" / "best.pt"
    if not weights.exists():
        # Fallback search — ultralytics' exact path varies by version
        candidates = list(work_dir.glob("**/weights/best.pt"))
        if candidates:
            weights = max(candidates, key=lambda p: p.stat().st_mtime)
    if not weights.exists():
        raise FileNotFoundError(f"no best.pt under {work_dir}")
    return weights


def phase_infer(weights: Path, test_vids: list[str],
                 probs_dir: Path) -> None:
    sh([sys.executable, "util/predict_shots_yolo.py",
        "--weights",   weights,
        "--vIDs",      *test_vids,
        "--out-dir",   probs_dir,
        "--fps",       1.0,
        "--force"])


def phase_bridge(probs_dir: Path, test_vids: list[str], preds_csv: Path,
                  threshold: float, min_dur: int, pre: int, post: int) -> None:
    sh([sys.executable, "training/yolo_shot/probs_to_predictions.py",
        "--probs-dir", probs_dir,
        "--out",       preds_csv,
        "--vIDs",      *test_vids,
        "--threshold", threshold,
        "--min-dur",   min_dur,
        "--pre",       pre,
        "--post",      post])


def phase_record(train_size: int, preds_csv: Path, notes: str) -> None:
    sh([sys.executable, RECORD_CMD,
        "record",
        "--train-size",  train_size,
        "--predictions", preds_csv,
        "--notes",       notes])


# ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size",   type=int, required=True)
    ap.add_argument("--epochs",       type=int, default=30)
    ap.add_argument("--batch",        type=int, default=8)
    ap.add_argument("--device",       default="mps",
                    help="ultralytics device (mps for Apple Silicon, "
                         "cuda for NVIDIA, cpu fallback)")
    ap.add_argument("--w-factor",     type=float, default=DEFAULT_W_FACTOR)
    ap.add_argument("--h-factor",     type=float, default=DEFAULT_H_FACTOR)
    ap.add_argument("--hardneg-target", type=int, default=1500,
                    help="approx # of hard-negative frames to mine globally")
    ap.add_argument("--infer-threshold", type=float, default=0.50)
    ap.add_argument("--infer-min-dur", type=int,   default=3)
    ap.add_argument("--infer-pre",     type=int,   default=5)
    ap.add_argument("--infer-post",    type=int,   default=5)
    ap.add_argument("--notes",         default="")
    ap.add_argument("--force-features",  action="store_true")
    ap.add_argument("--force-positives", action="store_true")
    ap.add_argument("--force-prelabel",  action="store_true")
    args = ap.parse_args()

    os.chdir(REPO)
    if not SPLITS.exists():
        print(f"ERROR: {SPLITS} missing. Run "
              f"training/learning_curve/splits.py first.", file=sys.stderr)
        return 2
    if not CUST_JSON.exists():
        print(f"ERROR: {CUST_JSON} missing. Run "
              f"training/yolo_shot/synth_customer.py first.", file=sys.stderr)
        return 2

    splits = json.loads(SPLITS.read_text())
    train_pool = splits["train_pool"]
    test_mids  = splits["test_match_ids"]
    if args.train_size > len(train_pool):
        print(f"ERROR: train_size={args.train_size} > train_pool "
              f"({len(train_pool)})", file=sys.stderr)
        return 3
    train_mids = train_pool[: args.train_size]
    train_vids = [str(m) for m in train_mids]
    test_vids  = [str(m) for m in test_mids]

    curve_dir = REPO / "runs" / f"yolo_curve_n{args.train_size}"
    curve_dir.mkdir(parents=True, exist_ok=True)
    run_name  = f"hockeyai_shot_n{args.train_size}"
    weights_dir = curve_dir / "work"
    weights_dir.mkdir(parents=True, exist_ok=True)
    probs_dir   = curve_dir / "probs"
    probs_dir.mkdir(parents=True, exist_ok=True)
    preds_csv   = curve_dir / "preds.csv"

    print(f"\n=== curve point: train_size={args.train_size} ===", file=sys.stderr)
    print(f"  train: {train_vids}", file=sys.stderr)
    print(f"  test:  {test_vids}", file=sys.stderr)
    print(f"  curve_dir: {curve_dir}", file=sys.stderr)

    t0 = time.time()

    # 1-2. Feature extraction (cached) for train + test
    phase_features(train_vids + test_vids, args.force_features)

    # 3. Positives for train
    phase_positives(train_vids, args.force_positives)

    # 4. Hard negs (global pool — assembly step below filters out test)
    phase_hard_negs(train_vids, args.hardneg_target)

    # 5. Prelabel everything
    phase_prelabel(args.force_prelabel)

    # 6. Auto-label shot bbox on positives (TIGHT factors)
    phase_autolabel(train_vids, args.w_factor, args.h_factor)

    # 7. Per-curve-point dataset (symlinks, train subset only)
    dataset = phase_build_dataset(train_vids, curve_dir)

    # 8. Train
    weights = phase_train(dataset, weights_dir, run_name,
                           args.epochs, args.batch, args.device)
    print(f"  weights: {weights}", file=sys.stderr)

    # 9. Inference on test
    phase_infer(weights, test_vids, probs_dir)

    # 10. Bridge to predictions.csv
    phase_bridge(probs_dir, test_vids, preds_csv,
                  args.infer_threshold, args.infer_min_dur,
                  args.infer_pre, args.infer_post)

    # 11. Record curve point
    notes = args.notes or (
        f"yolo+hardneg n={args.train_size} epochs={args.epochs} "
        f"bbox={args.w_factor}x{args.h_factor} "
        f"thr={args.infer_threshold} min_dur={args.infer_min_dur} "
        f"pre={args.infer_pre} post={args.infer_post}"
    )
    phase_record(args.train_size, preds_csv, notes)

    print(f"\n=== curve point n={args.train_size} done in "
          f"{(time.time() - t0)/60:.1f} min ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
