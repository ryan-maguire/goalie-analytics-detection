"""Generate splits.json for the learning-curve experiment.

Inputs (auto-detected from the repo):
  - paired matches    = match_id appearing in BOTH data/videos/full_<id>.mp4
                        AND data/ground_truth/gt_<id>.csv
  - GT shot count     = used to stratify so test set isn't skewed
                        toward low-shot games

Outputs:
  splits.json with keys:
    test_match_ids     fixed holdout, NEVER trained on
    train_pool         everything else (drawn from for curve points)
    curve_sizes        [n1, n2, n3, n4] training-set sizes to evaluate
    seed               random seed used (for reproducibility)

Usage:
    python3 training/learning_curve/splits.py --test-frac 0.25 \
        --curve-sizes 3 6 9 12 --seed 7

The script prints diagnostics + writes splits.json next to itself.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO     = Path(__file__).resolve().parents[2]
VIDEO_DIR = REPO / "data" / "videos"
GT_DIR    = REPO / "data" / "ground_truth"
OUT_PATH  = Path(__file__).resolve().parent / "splits.json"

THREAT_ACTIONS = {"Shots", "Goals", "OZ play"}


def paired_match_ids() -> list[int]:
    """Return match_ids with BOTH video and GT files present."""
    vids = {p.stem.replace("full_", "")
            for p in VIDEO_DIR.glob("full_*.mp4")
            if p.stem.replace("full_", "").isdigit()}
    gts  = {p.stem.replace("gt_", "")
            for p in GT_DIR.glob("gt_*.csv")}
    return sorted(int(m) for m in vids & gts)


def gt_shot_count(match_id: int) -> int:
    """Approximate shot count for stratified splitting. Counts threat-action
    rows (Shots, Goals, OZ play) in the GT CSV. Returns 0 if missing/empty."""
    path = GT_DIR / f"gt_{match_id}.csv"
    if not path.exists():
        return 0
    n = 0
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("action") or "").strip() in THREAT_ACTIONS:
                    n += 1
    except Exception:
        return 0
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac",   type=float, default=0.25,
                    help="Fraction of paired games held out as test (default 0.25)")
    ap.add_argument("--min-test",    type=int, default=3,
                    help="Minimum test-set size — overrides --test-frac")
    ap.add_argument("--curve-sizes", type=int, nargs="+", default=None,
                    help="Training-set sizes to evaluate. Default: auto from train_pool")
    ap.add_argument("--seed",        type=int, default=7)
    args = ap.parse_args()

    paired = paired_match_ids()
    if not paired:
        print("ERROR: no paired matches found "
              f"(checked {VIDEO_DIR} and {GT_DIR}).", file=sys.stderr)
        return 2

    counts = {m: gt_shot_count(m) for m in paired}
    print(f"Paired matches: {len(paired)}", file=sys.stderr)
    for m in paired:
        print(f"  {m}: {counts[m]} threat rows", file=sys.stderr)

    n_test = max(args.min_test, round(len(paired) * args.test_frac))
    n_test = min(n_test, len(paired) - 1)

    # Stratified split: sort by shot count, take every k-th into test so
    # both sets span the shot-count range. Beats random for small N.
    by_count = sorted(paired, key=lambda m: counts[m])
    step = max(1, len(by_count) // n_test)
    test_ids = sorted(by_count[i] for i in range(0, len(by_count), step))[:n_test]

    train_pool = sorted(m for m in paired if m not in test_ids)

    # Auto-generate curve sizes if not given. Aim for 4 well-spaced points.
    if args.curve_sizes is None:
        N = len(train_pool)
        if N <= 0:
            print("ERROR: empty train pool after holdout.", file=sys.stderr)
            return 3
        if N >= 12:
            curve_sizes = [N // 4, N // 2, 3 * N // 4, N]
        elif N >= 4:
            curve_sizes = sorted({max(1, N // 4), max(2, N // 2),
                                   max(3, 3 * N // 4), N})
        else:
            curve_sizes = list(range(1, N + 1))
    else:
        curve_sizes = sorted(set(args.curve_sizes))

    splits = {
        "seed":          args.seed,
        "video_dir":     str(VIDEO_DIR),
        "gt_dir":        str(GT_DIR),
        "test_match_ids": test_ids,
        "train_pool":    train_pool,
        "curve_sizes":   curve_sizes,
        "shot_counts":   {str(m): counts[m] for m in paired},
        "notes": (
            "test_match_ids are NEVER used for training. "
            "For each size in curve_sizes, draw a deterministic "
            "subset of train_pool[0:size] (sorted ascending) for "
            "reproducibility. Increase test-frac if N grows."
        ),
    }
    OUT_PATH.write_text(json.dumps(splits, indent=2))

    print(f"\nsplits written → {OUT_PATH}", file=sys.stderr)
    print(f"  test:       {test_ids}  (n={len(test_ids)})", file=sys.stderr)
    print(f"  train_pool: {train_pool}  (n={len(train_pool)})", file=sys.stderr)
    print(f"  curve:      {curve_sizes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
