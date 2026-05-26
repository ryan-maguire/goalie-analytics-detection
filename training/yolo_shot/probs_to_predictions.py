"""Bridge: per-second `shot` probability TSVs → predictions.csv that
training/learning_curve/eval.py consumes.

Sidesteps util/yolo_probs_to_windows.py because:
  - it requires a customer JSON (we don't need one for the curve eval)
  - it filters by VID_TO_HUDL (which doesn't list hudl-fetched match IDs)
  - its output is cv_seg-format JSON, but we want CSV anyway

Algorithm matches yolo_probs_to_windows.py so curve points are
comparable to cv_seg's documented output:
  1. For each second, hit = (shot_max_conf >= threshold)
  2. Merge consecutive hit-seconds (allow ≤2-sec gaps for single drops)
  3. Drop runs shorter than --min-dur
  4. Pad each run by --pre / --post (clamped to [0, max_t])
  5. Merge overlapping padded windows
  6. Emit predictions.csv: vID,start_s,end_s,confidence

Usage:
    python3 training/yolo_shot/probs_to_predictions.py \\
        --probs-dir runs/yolo_curve_n8/probs \\
        --out runs/yolo_curve_n8/preds.csv \\
        --vIDs 2069975 2072194 2073809 ... \\
        --threshold 0.50 --min-dur 3 --pre 5 --post 5
"""

import argparse
import csv
import sys
from pathlib import Path


def load_probs(tsv: Path) -> list[tuple[int, float]]:
    rows = []
    with open(tsv) as f:
        f.readline()                                # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                rows.append((int(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return rows


def runs_above(probs, threshold, max_gap=2):
    """Return [(start_s, end_s, mean_conf)] runs of consecutive seconds
    with conf >= threshold, allowing up to `max_gap` seconds of below-
    threshold values inside a single run (silences single-frame drops)."""
    out = []
    cur_s = cur_last = None
    cur_confs: list[float] = []
    for t, p in probs:
        if p >= threshold:
            if cur_s is None:
                cur_s = t; cur_last = t; cur_confs = [p]
            elif t - cur_last <= max_gap:
                cur_last = t; cur_confs.append(p)
            else:
                out.append((cur_s, cur_last,
                             sum(cur_confs) / len(cur_confs)))
                cur_s = t; cur_last = t; cur_confs = [p]
    if cur_s is not None:
        out.append((cur_s, cur_last, sum(cur_confs) / len(cur_confs)))
    return out


def merge_overlap(windows):
    """Merge overlapping / touching (start, end, conf) tuples."""
    if not windows:
        return []
    ws = sorted(windows)
    merged = [list(ws[0])]
    for s, e, c in ws[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = max(merged[-1][2], c)   # keep max conf
        else:
            merged.append([s, e, c])
    return [tuple(m) for m in merged]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dir", required=True, type=Path,
                    help="dir of {vID}.tsv files from predict_shots_yolo.py")
    ap.add_argument("--out",       required=True, type=Path,
                    help="output predictions CSV")
    ap.add_argument("--vIDs",      nargs="+", required=True,
                    help="vID(s) to include — typically the test set")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min-dur",   type=int,   default=3)
    ap.add_argument("--pre",       type=int,   default=5)
    ap.add_argument("--post",      type=int,   default=5)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_videos = n_windows = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vID", "start_s", "end_s", "confidence"])
        for vid in args.vIDs:
            tsv = args.probs_dir / f"{vid}.tsv"
            if not tsv.exists():
                print(f"  [skip] no probs for {vid} ({tsv})", file=sys.stderr)
                continue
            probs = load_probs(tsv)
            if not probs:
                continue
            max_t = probs[-1][0]
            runs = runs_above(probs, args.threshold)
            kept = [(s, e, c) for s, e, c in runs
                     if (e - s + 1) >= args.min_dur]
            padded = [(max(0, s - args.pre), min(max_t, e + args.post), c)
                       for s, e, c in kept]
            merged = merge_overlap(padded)
            for s, e, c in merged:
                w.writerow([vid, s, e, f"{c:.4f}"])
            n_videos += 1
            n_windows += len(merged)
            print(f"  {vid}: {len(probs)} secs → {len(runs)} raw runs → "
                  f"{len(kept)} after min-dur → {len(merged)} merged windows",
                  file=sys.stderr)

    print(f"\n{n_windows} windows across {n_videos} videos → {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
