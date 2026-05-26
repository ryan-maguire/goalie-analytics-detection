"""One audio-only curve point: train + infer test + bridge + record."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO       = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS     = REPO / "training" / "learning_curve" / "splits.json"
RESULTS    = REPO / "training" / "learning_curve" / "results_audio.json"
RECORD_CMD = REPO / "training" / "learning_curve" / "run_curve.py"
BRIDGE     = REPO / "training" / "yolo_shot" / "probs_to_predictions.py"


def sh(cmd, *, check=True):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    rc = subprocess.call([str(c) for c in cmd])
    if rc != 0 and check:
        raise RuntimeError(f"command failed (rc={rc})")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, required=True)
    ap.add_argument("--epochs",     type=int, default=50)
    ap.add_argument("--batch",      type=int, default=64)
    ap.add_argument("--hidden",     type=int, default=64)
    ap.add_argument("--n-layers",   type=int, default=2)
    ap.add_argument("--seq-len",    type=int, default=32)
    ap.add_argument("--stride-train", type=int, default=4)
    ap.add_argument("--stride-infer", type=int, default=4)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--infer-threshold", type=float, default=0.50)
    ap.add_argument("--infer-min-dur",   type=int,   default=3)
    ap.add_argument("--infer-pre",       type=int,   default=5)
    ap.add_argument("--infer-post",      type=int,   default=5)
    ap.add_argument("--notes",      default="")
    args = ap.parse_args()

    os.chdir(REPO)
    splits = json.loads(SPLITS.read_text())
    train_pool = splits["train_pool"]; test_mids = splits["test_match_ids"]
    if args.train_size > len(train_pool):
        print(f"ERROR: train_size={args.train_size} > train_pool "
              f"({len(train_pool)})", file=sys.stderr); return 3
    train_vids = [str(m) for m in train_pool[:args.train_size]]
    test_vids  = [str(m) for m in test_mids]

    curve_dir = REPO / "runs" / f"audio_curve_n{args.train_size}"
    work_dir  = curve_dir / "work"
    probs_dir = curve_dir / "probs"
    preds_csv = curve_dir / "preds.csv"
    work_dir.mkdir(parents=True, exist_ok=True)
    probs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== AUDIO curve point: n={args.train_size} ===", file=sys.stderr)
    print(f"  train: {train_vids}", file=sys.stderr)
    print(f"  test:  {test_vids}",  file=sys.stderr)
    t0 = time.time()

    sh([sys.executable, SCRIPT_DIR / "train.py",
        "--train-vids", *train_vids,
        "--out-dir",    work_dir,
        "--epochs",     args.epochs, "--batch", args.batch,
        "--hidden",     args.hidden, "--n-layers", args.n_layers,
        "--seq-len",    args.seq_len, "--stride", args.stride_train,
        "--lr",         args.lr, "--device", args.device])
    weights = work_dir / "best.pt"
    if not weights.exists():
        print(f"ERROR: no best.pt", file=sys.stderr); return 4

    sh([sys.executable, SCRIPT_DIR / "infer.py",
        "--weights", weights, "--vIDs", *test_vids,
        "--out-dir", probs_dir, "--seq-len", args.seq_len,
        "--stride", args.stride_infer, "--device", args.device])

    sh([sys.executable, BRIDGE,
        "--probs-dir", probs_dir, "--out", preds_csv,
        "--vIDs",      *test_vids,
        "--threshold", args.infer_threshold,
        "--min-dur",   args.infer_min_dur,
        "--pre",       args.infer_pre, "--post", args.infer_post])

    notes = args.notes or (
        f"audio-only n={args.train_size} epochs={args.epochs} "
        f"thr={args.infer_threshold}")
    sh([sys.executable, RECORD_CMD,
        "--results-path", RESULTS,
        "record",
        "--train-size", args.train_size,
        "--predictions", preds_csv,
        "--notes",       notes])
    print(f"\n=== audio n={args.train_size} done in {(time.time()-t0)/60:.1f} min ===",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
