"""Strict-window F1 evaluator for the learning-curve experiment.

Uses the SAME IoU / matching primitives as eval/eval_cv_seg_output.py
(imported, not duplicated) so the F1 number is directly comparable
to the documented 0.422 baseline.

Inputs:
  predictions_csv   path to a CSV with columns:
                      vID,start_s,end_s[,confidence,team]
  splits_path       path to splits.json (used to pull test_match_ids
                      + gt_dir)

Output:
  prints a JSON summary {aggregate: {p, r, f1}, per_video: {...}}
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eval.eval_cv_seg_output import (  # noqa: E402
    ThreatWindow,
    load_ground_truth_windows,
    greedy_match,
    pad_window,
)


# Match the strict-F1 setup the project documents in EVAL_NOTES.md
DEFAULT_IOU_THRESHOLD = 0.30
DEFAULT_GT_PAD_SEC    = 2.0    # widen Hudl's 12-s clips slightly
DEFAULT_WINDOW_DIFF   = 8      # gap-merge param for GT (matches v25)


def load_predictions(path: Path) -> dict[str, list[ThreatWindow]]:
    """vID → list of predicted ThreatWindows."""
    by_vid: dict[str, list[ThreatWindow]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vid = str(row["vID"]).strip()
                s   = float(row["start_s"])
                e   = float(row["end_s"])
            except (KeyError, ValueError) as ex:
                print(f"  [skip] bad row: {row} ({ex})", file=sys.stderr)
                continue
            by_vid[vid].append(ThreatWindow(
                start=s, end=e,
                team=(row.get("team") or None),
                source="prediction",
            ))
    return by_vid


def eval_predictions(
    predictions_csv: Path,
    splits_path:     Path,
    iou_threshold:   float = DEFAULT_IOU_THRESHOLD,
    gt_pad_sec:      float = DEFAULT_GT_PAD_SEC,
    window_diff:     int   = DEFAULT_WINDOW_DIFF,
) -> dict:
    splits = json.loads(splits_path.read_text())
    test_ids: list[int] = splits["test_match_ids"]
    gt_dir = Path(splits["gt_dir"])

    preds = load_predictions(predictions_csv)

    per_video: dict[str, dict] = {}
    agg_tp = agg_fp = agg_fn = 0

    for mid in test_ids:
        vid = str(mid)
        gt_path = gt_dir / f"gt_{mid}.csv"
        if not gt_path.exists():
            print(f"  [warn] missing GT for {mid} → skipping", file=sys.stderr)
            continue
        gt_raw = load_ground_truth_windows(str(gt_path), window_diff)
        gt = [pad_window(w, gt_pad_sec) for w in gt_raw]
        pred = preds.get(vid, [])

        _matches, unmatched_gt, unmatched_pred = greedy_match(
            gt, pred, iou_threshold,
        )
        tp = len(gt) - len(unmatched_gt)
        fn = len(unmatched_gt)
        fp = len(unmatched_pred)
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_video[vid] = {"tp": tp, "fp": fp, "fn": fn,
                          "p": round(p, 4), "r": round(r, 4),
                          "f1": round(f1, 4),
                          "n_gt": len(gt), "n_pred": len(pred)}
        agg_tp += tp; agg_fp += fp; agg_fn += fn

    P = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else 0.0
    R = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0

    return {
        "settings": {
            "iou_threshold": iou_threshold,
            "gt_pad_sec":    gt_pad_sec,
            "window_diff":   window_diff,
        },
        "aggregate": {
            "tp": agg_tp, "fp": agg_fp, "fn": agg_fn,
            "p": round(P, 4), "r": round(R, 4), "f1": round(F, 4),
        },
        "per_video": per_video,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",   required=True, type=Path)
    ap.add_argument("--splits",        type=Path,
                    default=Path(__file__).resolve().parent / "splits.json")
    ap.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    ap.add_argument("--gt-pad-sec",    type=float, default=DEFAULT_GT_PAD_SEC)
    ap.add_argument("--window-diff",   type=int,   default=DEFAULT_WINDOW_DIFF)
    args = ap.parse_args()

    result = eval_predictions(
        args.predictions, args.splits,
        iou_threshold=args.iou_threshold,
        gt_pad_sec=args.gt_pad_sec,
        window_diff=args.window_diff,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
