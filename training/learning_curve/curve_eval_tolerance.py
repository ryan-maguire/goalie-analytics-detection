"""Δ-second event-spotting eval (alternative to strict-IoU window F1).

For each test video:
  - GT events  = midpoint of each merged GT window
  - Pred events = midpoint of each merged prediction window
  - Greedy 1:1 match on min |gt_t - pred_t|; accept if distance ≤ Δ seconds
  - TP = matched pairs; FN = unmatched GT; FP = unmatched predictions

This is the metric used by SoccerNet action-spotting + most production
sports-vision systems. ±5s tolerance is "find the moment within 5
seconds" — useful when the downstream consumer (a coach reviewing
goalie reactions) doesn't need second-perfect localisation.

Outputs the SAME JSON shape as curve_eval.py so existing tooling can
read either.
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
)


DEFAULT_TOLERANCE_S = 5.0
DEFAULT_WINDOW_DIFF = 8


def load_predictions(path: Path) -> dict[str, list[ThreatWindow]]:
    by_vid: dict[str, list[ThreatWindow]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vid = str(row["vID"]).strip()
                s   = float(row["start_s"])
                e   = float(row["end_s"])
            except (KeyError, ValueError):
                continue
            by_vid[vid].append(ThreatWindow(start=s, end=e, source="prediction"))
    return by_vid


def midpoints(windows: list[ThreatWindow]) -> list[float]:
    return [0.5 * (w.start + w.end) for w in windows]


def greedy_tolerance_match(
    gt_t: list[float],
    pred_t: list[float],
    tolerance_s: float,
) -> tuple[int, int, int]:
    """Greedy 1:1 matching by min |gt - pred| ≤ tolerance.
    Returns (tp, fp, fn)."""
    if not gt_t and not pred_t:
        return 0, 0, 0
    candidates = []
    for i, g in enumerate(gt_t):
        for j, p in enumerate(pred_t):
            d = abs(g - p)
            if d <= tolerance_s:
                candidates.append((d, i, j))
    candidates.sort()
    matched_g, matched_p = set(), set()
    for d, i, j in candidates:
        if i in matched_g or j in matched_p:
            continue
        matched_g.add(i); matched_p.add(j)
    tp = len(matched_g)
    fn = len(gt_t) - tp
    fp = len(pred_t) - len(matched_p)
    return tp, fp, fn


def eval_predictions(
    predictions_csv: Path,
    splits_path: Path,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    window_diff: int = DEFAULT_WINDOW_DIFF,
) -> dict:
    splits = json.loads(splits_path.read_text())
    test_ids = splits["test_match_ids"]
    gt_dir = Path(splits["gt_dir"])

    preds = load_predictions(predictions_csv)

    per_video: dict[str, dict] = {}
    agg_tp = agg_fp = agg_fn = 0

    for mid in test_ids:
        vid = str(mid)
        gt_path = gt_dir / f"gt_{mid}.csv"
        if not gt_path.exists():
            continue
        gt_w = load_ground_truth_windows(str(gt_path), window_diff)
        gt_t = midpoints(gt_w)
        pred_t = midpoints(preds.get(vid, []))
        tp, fp, fn = greedy_tolerance_match(gt_t, pred_t, tolerance_s)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_video[vid] = {
            "tp": tp, "fp": fp, "fn": fn,
            "p": round(p, 4), "r": round(r, 4), "f1": round(f1, 4),
            "n_gt": len(gt_t), "n_pred": len(pred_t),
        }
        agg_tp += tp; agg_fp += fp; agg_fn += fn

    P = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else 0.0
    R = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0

    return {
        "settings": {
            "metric":      "event_spotting_midpoint",
            "tolerance_s": tolerance_s,
            "window_diff": window_diff,
        },
        "aggregate": {
            "tp": agg_tp, "fp": agg_fp, "fn": agg_fn,
            "p": round(P, 4), "r": round(R, 4), "f1": round(F, 4),
        },
        "per_video": per_video,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--splits",      type=Path,
                    default=Path(__file__).resolve().parent / "splits.json")
    ap.add_argument("--tolerance-s", type=float, default=DEFAULT_TOLERANCE_S)
    ap.add_argument("--window-diff", type=int,   default=DEFAULT_WINDOW_DIFF)
    args = ap.parse_args()
    result = eval_predictions(args.predictions, args.splits,
                                tolerance_s=args.tolerance_s,
                                window_diff=args.window_diff)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
