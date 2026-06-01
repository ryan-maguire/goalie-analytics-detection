"""Run candidate_list.py across the test set and produce a
coach's-eye-view validation report.

Per game:
  - n_candidates surfaced
  - n_gt_shots in the game
  - recall@5s, recall@10s (how many real shots got a candidate near them)
  - n_false_positives (candidates with no GT shot within ±5s)
  - sample of the candidate list

Aggregate:
  - overall recall@5s, recall@10s
  - "review budget" (avg candidates per minute)
  - top false-positive timestamps (for error analysis)

Output:
  <out_dir>/VALIDATION_REPORT.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))                                  # for `eval.*`
sys.path.insert(0, str(SCRIPT))
sys.path.insert(0, str(REPO / "training" / "learning_curve"))

from candidate_list import (  # noqa: E402
    generate_candidates, annotate_with_gt, fmt_mmss,
)
from eval.eval_cv_seg_output import load_ground_truth_windows  # noqa: E402


def recall_at_tolerance(rows: list[dict], gt_mids: list[float],
                          tolerance_s: float) -> float:
    """Fraction of GT events whose nearest candidate is within tolerance.

    Returns 0.0 when there is no ground truth — recall is undefined, but
    0.0 keeps the aggregate `sum(n_gt * recall)` correct (0 * 0 = 0) and,
    unlike 1.0, doesn't render a misleading "100%" in the per-game table.
    The per-game rows display "n/a" when n_gt == 0 (see report builder).
    """
    if not gt_mids:
        return 0.0
    n_hit = 0
    for gm in gt_mids:
        nearest = min((abs(r["t_seconds"] - gm) for r in rows), default=float("inf"))
        if nearest <= tolerance_s:
            n_hit += 1
    return n_hit / len(gt_mids)


def video_duration_s(probs_dir: Path, vid: str) -> int:
    f = probs_dir / f"{vid}.tsv"
    if not f.exists():
        return 0
    last_t = 0
    with open(f) as fh:
        fh.readline()
        for line in fh:
            try: last_t = max(last_t, int(float(line.split("\t")[0])))
            except (ValueError, IndexError): pass
    return last_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path,
                    default=REPO / "training" / "learning_curve" / "splits.json")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--yolo-probs-dir",  type=Path,
                    default=REPO / "runs" / "yolo_curve_n16" / "probs")
    ap.add_argument("--audio-probs-dir", type=Path,
                    default=REPO / "runs" / "audio_curve_n16" / "probs")
    ap.add_argument("--gt-dir", type=Path,
                    default=REPO / "data" / "ground_truth")
    ap.add_argument("--weight-yolo",  type=float, default=0.5)
    ap.add_argument("--weight-audio", type=float, default=0.5)
    ap.add_argument("--threshold",    type=float, default=0.40)
    ap.add_argument("--nms-distance", type=int,   default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads(args.splits.read_text())
    test_ids = splits["test_match_ids"]

    per_video_summary = []
    all_fps_top = []
    for mid in test_ids:
        vid = str(mid)
        try:
            rows = generate_candidates(
                vid, args.yolo_probs_dir, args.audio_probs_dir,
                args.weight_yolo, args.weight_audio,
                args.threshold, args.nms_distance,
            )
        except FileNotFoundError as e:
            print(f"  [skip] {vid}: {e}", file=sys.stderr); continue
        gt_path = args.gt_dir / f"gt_{mid}.csv"
        gt_windows = load_ground_truth_windows(str(gt_path), 8) if gt_path.exists() else []
        gt_mids = [0.5 * (w.start + w.end) for w in gt_windows]
        rows = annotate_with_gt(rows, args.gt_dir, vid)

        r5  = recall_at_tolerance(rows, gt_mids, 5.0)
        r10 = recall_at_tolerance(rows, gt_mids, 10.0)
        n_match5 = sum(1 for r in rows
                        if r.get("gt_match_t_seconds") is not None)
        n_fp = len(rows) - n_match5
        dur = video_duration_s(args.yolo_probs_dir, vid)
        cand_per_min = len(rows) / max(1, dur / 60)
        per_video_summary.append({
            "vid": vid, "n_candidates": len(rows), "n_gt": len(gt_mids),
            "recall_5s": r5, "recall_10s": r10,
            "n_fp_at_5s": n_fp, "dur_s": dur,
            "cand_per_min": cand_per_min,
            "sample_rows": rows[:10],
            "all_rows": rows,
        })
        # Collect top-confidence false positives for error analysis
        for r in rows:
            if r.get("gt_match_t_seconds") is None:
                all_fps_top.append((r["confidence"], vid, r["t_mmss"]))
    all_fps_top.sort(reverse=True)

    # Aggregate
    total_gt   = sum(s["n_gt"] for s in per_video_summary)
    total_hits5  = sum(s["n_gt"] * s["recall_5s"]  for s in per_video_summary)
    total_hits10 = sum(s["n_gt"] * s["recall_10s"] for s in per_video_summary)
    total_cand = sum(s["n_candidates"] for s in per_video_summary)
    total_dur  = sum(s["dur_s"] for s in per_video_summary)
    agg_r5  = total_hits5  / max(1, total_gt)
    agg_r10 = total_hits10 / max(1, total_gt)
    avg_cand_per_min = total_cand / max(1, total_dur / 60)

    # ──────────────── Write report ────────────────
    lines = ["# Candidate-list validation report",
              "",
              f"Generated against test set of {len(per_video_summary)} games.",
              "",
              "## How to read this",
              "",
              "For each game, the model surfaces a ranked list of candidate "
              "shot moments. A coach reviews the list, accepts true shots, "
              "rejects false positives. The two key numbers are:",
              "",
              "- **Recall@5s**: fraction of real shots within ±5s of a candidate",
              "- **Recall@10s**: same with ±10s tolerance",
              "- **Candidates/min**: review-load proxy (lower = less coach effort)",
              "",
              "## Aggregate over test set",
              "",
              f"- **Recall@5s:  {agg_r5:.1%}**",
              f"- **Recall@10s: {agg_r10:.1%}**",
              f"- Avg candidates per minute: {avg_cand_per_min:.1f}",
              f"- Total candidates: {total_cand} across {total_gt} real GT shots",
              "",
              "## Per-game breakdown",
              "",
              "| game | dur | GT shots | candidates | cand/min | R@5s | R@10s | FP@5s |",
              "|---|---|---|---|---|---|---|---|"]
    for s in per_video_summary:
        r5_disp  = "n/a" if s["n_gt"] == 0 else f"{s['recall_5s']:.1%}"
        r10_disp = "n/a" if s["n_gt"] == 0 else f"{s['recall_10s']:.1%}"
        lines.append(
            f"| {s['vid']} | {fmt_mmss(s['dur_s'])} | {s['n_gt']} | "
            f"{s['n_candidates']} | {s['cand_per_min']:.1f} | "
            f"{r5_disp} | {r10_disp} | {s['n_fp_at_5s']} |"
        )

    # Per-game samples
    lines += ["", "## Sample candidate lists (top 10 per game)", ""]
    for s in per_video_summary:
        r5_hdr = "n/a" if s["n_gt"] == 0 else f"{s['recall_5s']:.0%}"
        lines += [f"### Game {s['vid']}  ({s['n_candidates']} candidates, "
                   f"{s['n_gt']} GT shots, R@5s={r5_hdr})",
                   "",
                   "| rank | time | conf | GT match | Δs |",
                   "|---|---|---|---|---|"]
        for r in s["sample_rows"]:
            chk = "✓" if r["gt_match_t_seconds"] is not None else "✗"
            ds  = f"{r['gt_match_delta_s']:.1f}" if r["gt_match_delta_s"] is not None else "—"
            lines.append(
                f"| {r['rank']} | {r['t_mmss']} | {r['confidence']:.2f} | "
                f"{chk} {r['gt_match_t_mmss']} | {ds} |"
            )
        lines.append("")

    # Top false positives for error analysis
    lines += ["## Top-10 false positives across test set "
              "(highest-conf candidates with no GT shot within ±5s)",
              "",
              "| conf | game | time |",
              "|---|---|---|"]
    for conf, vid, mmss in all_fps_top[:10]:
        lines.append(f"| {conf:.2f} | {vid} | {mmss} |")

    lines += ["",
              "## Configuration used",
              "",
              "```",
              f"fusion          = YOLO + audio, weights {args.weight_yolo}/{args.weight_audio}",
              f"YOLO probs dir  = {args.yolo_probs_dir}",
              f"audio probs dir = {args.audio_probs_dir}",
              f"threshold       = {args.threshold}",
              f"NMS distance    = {args.nms_distance} seconds",
              "```",
              ""]

    report = args.out_dir / "VALIDATION_REPORT.md"
    report.write_text("\n".join(lines))
    print(f"\nwrote {report}", file=sys.stderr)
    print(f"AGG Recall@5s:  {agg_r5:.1%}", file=sys.stderr)
    print(f"AGG Recall@10s: {agg_r10:.1%}", file=sys.stderr)
    print(f"Avg cand/min:   {avg_cand_per_min:.1f}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
