"""
Convert per-second `shot` probabilities (from predict_shots_yolo.py) into
cv_seg-compatible threat window JSONs that the eval can score.

Algorithm:
  1. For each second, hit = (shot_max_conf >= threshold).
  2. Merge consecutive hit-seconds into runs.
  3. Drop runs shorter than --min-dur.
  4. Pad each run by --pre/--post to give some IoU room with GT.
  5. Emit as one JSON file per video matching cv_seg's gt_seg_{vID}.json
     schema, plus a stub gt_seg_{vID}_meta.json with target_filter=True
     so the eval restricts GT to opponent-team shots (apples-to-apples
     vs cv_seg).

The colour fields aren't used because target_filter mode doesn't score
attribution. We set them to a placeholder string so the JSON shape
matches cv_seg's writer.

Usage:
    python3 util/yolo_probs_to_windows.py \\
        --probs-dir data/output/yolo_shot_probs \\
        --customers data/customers/CUST000048.json data/customers/CUST000031.json \\
        --out-dir data/output/runs/yolo_shot \\
        --threshold 0.50 --min-dur 3 --pre 5 --post 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_cv_seg_output import VID_TO_HUDL


def _load_opp_color(customer_paths):
    """{vID: opponentGoalieColor}. Used only as a placeholder string."""
    out = {}
    for p in customer_paths:
        for rec in json.load(open(p)):
            vid = str(rec.get("vID", "")).strip()
            if vid:
                out[vid] = rec.get("opponentGoalieColor") or "Unknown"
    return out


def _load_probs(tsv_path):
    """Return [(t, shot_max_conf)]."""
    rows = []
    with open(tsv_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                t = int(parts[0]); p = float(parts[1])
            except ValueError:
                continue
            rows.append((t, p))
    return rows


def _runs_above(probs, threshold):
    """Return [(start, end_inclusive)] runs of consecutive seconds with
    p >= threshold."""
    runs = []
    cur_start = None
    last_t = None
    for t, p in probs:
        if p >= threshold:
            if cur_start is None:
                cur_start = t
                last_t = t
            else:
                # Allow up to 1 second gap (single missed frame)
                if t - last_t <= 2:
                    last_t = t
                else:
                    runs.append((cur_start, last_t))
                    cur_start = t
                    last_t = t
        # below threshold — close any open run
        else:
            if cur_start is not None:
                runs.append((cur_start, last_t))
                cur_start = None
                last_t = None
    if cur_start is not None:
        runs.append((cur_start, last_t))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dir",  default="data/output/yolo_shot_probs")
    ap.add_argument("--customers",  nargs="+", required=True)
    ap.add_argument("--out-dir",    default="data/output/runs/yolo_shot")
    ap.add_argument("--threshold",  type=float, default=0.50,
                    help="shot_max_conf threshold to count a second as 'shot'")
    ap.add_argument("--min-dur",    type=int, default=3,
                    help="min run duration (sec) before padding to keep")
    ap.add_argument("--pre",        type=int, default=5,
                    help="seconds of pre-roll to add to each kept run")
    ap.add_argument("--post",       type=int, default=5,
                    help="seconds of post-roll to add to each kept run")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    opp_color_of = _load_opp_color(args.customers)

    summary_rows = []
    for tsv in sorted(Path(args.probs_dir).glob("*.tsv")):
        vid = tsv.stem
        if vid not in VID_TO_HUDL:
            continue
        probs = _load_probs(str(tsv))
        if not probs:
            continue
        max_t = probs[-1][0]
        runs = _runs_above(probs, args.threshold)
        kept = [(s, e) for s, e in runs if (e - s + 1) >= args.min_dur]
        # Pad and clip to [0, max_t]
        windows = []
        for s, e in kept:
            ws = max(0, s - args.pre)
            we = min(max_t, e + args.post)
            windows.append((ws, we))
        # Merge any overlaps that resulted from padding
        windows.sort()
        merged = []
        for s, e in windows:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Write cv_seg-compatible gt_seg_{vID}.json
        opp_color = opp_color_of.get(vid, "Unknown")
        segments = [
            {
                "segment_start":       int(s),
                "segment_end":         int(e),
                "segmentHasThreat":    True,
                "threat_goalie_color": opp_color,
            }
            for s, e in merged
        ]
        seg_path  = os.path.join(args.out_dir, f"gt_seg_{vid}.json")
        meta_path = os.path.join(args.out_dir, f"gt_seg_{vid}_meta.json")
        with open(seg_path, "w") as f:
            json.dump(segments, f, indent=2)
        meta = {
            "vID":              vid,
            "method":           "yolo_shot_finetune",
            "processed_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "threat_segments":  len(segments),
            "total_segments":   len(segments),
            "target_filter": {
                "applied":         True,
                "target_color":    opp_color,
                "prefilter_total": len(segments),
                "postfilter_total": len(segments),
            },
            "thresholds": {
                "yolo_shot_threshold": args.threshold,
                "min_dur":             args.min_dur,
                "pre":                 args.pre,
                "post":                args.post,
            },
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        summary_rows.append((vid, len(probs), len(runs), len(kept), len(merged)))

    print(f"threshold={args.threshold}  min_dur={args.min_dur}  "
          f"pre={args.pre}  post={args.post}")
    print(f"{'vID':<16}  {'n_sec':>6}  {'runs':>5}  {'kept':>5}  {'merged':>6}")
    for vid, ns, nr, nk, nm in summary_rows:
        print(f"  {vid:<16}  {ns:>6}  {nr:>5}  {nk:>5}  {nm:>6}")


if __name__ == "__main__":
    sys.exit(main() or 0)
