#!/usr/bin/env python3
"""Validate scoreboard_tracker goal detections against ground truth.

For each GT goal in data/ground_truth/gt_{hudl}.csv:
  - Check if a detected goal_event's lookback window covers GT[start]
  - If yes: TP, with offset = detected_t_sec - GT[start]
  - If no:  FN

For each detected goal_event:
  - If no GT goal falls in the lookback window: FP

Reports: precision, recall, F1, offset distribution.

Usage:
    python3 tools/validate_scoreboard_goals.py \\
        --vID dwGsP6QKDs8 --hudl-id 2070269
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

REPO = Path(__file__).resolve().parents[1]


# vID → hudl_id (matches eval/eval_cv_seg_output.py VID_TO_HUDL)
VID_TO_HUDL = {
    "SX5xNJlh6eQ": 2073056, "bfEKgtOIkQU": 2072195,
    "mjEeE7p2Hz8": 2073809, "n2cy8b755Tg": 2127046,
    "v0lxSTbXfw8": 2073810, "dwGsP6QKDs8": 2070269,
    "Fjc9hmK8_3U": 2070260, "HNG0jKYY12g": 2095275,
    "J8WkcuTsD5I": 2072194, "kQVdtRa4o_A": 2127034,
    "krxhPVLGLz8": 2108724, "KYtM20r9BuM": 2072196,
    "q5yj6sAFQeY": 2127052, "zOQrPK7IJ24": 2127035,
}


def load_gt_goals(hudl_id: int) -> list[dict]:
    """Return list of {start, end, team, half} for action==Goals rows."""
    path = REPO / "data" / "ground_truth" / f"gt_{hudl_id}.csv"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if (row.get("action") or "").strip() == "Goals":
                out.append({
                    "start": int(row["start"]),
                    "end":   int(row["end"]),
                    "team":  row.get("team", "").strip(),
                    "half":  row.get("half", "").strip(),
                })
    return sorted(out, key=lambda g: g["start"])


def load_detected(vid: str) -> list[dict]:
    path = REPO / "data" / "output" / "scoreboard" / vid / "goal_events.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def match(gt_goals: list[dict], detected: list[dict]) -> dict:
    """Greedy match: for each detected, find the earliest unmatched GT
    goal whose start falls in [lookback_start, lookback_end]."""
    matched_gt: set[int] = set()
    tps = []
    fps = []
    for d in detected:
        lo = d["lookback_start"]
        hi = d["lookback_end"]
        candidate = None
        for i, g in enumerate(gt_goals):
            if i in matched_gt:
                continue
            if lo <= g["start"] <= hi:
                candidate = (i, g)
                break
        if candidate:
            i, g = candidate
            matched_gt.add(i)
            offset = d["detected_t_sec"] - g["start"]
            tps.append({"gt_idx": i, "gt": g, "det": d, "offset_sec": offset})
        else:
            fps.append(d)
    fns = [{"gt_idx": i, "gt": g}
           for i, g in enumerate(gt_goals) if i not in matched_gt]
    return {"tp": tps, "fp": fps, "fn": fns}


def report(vid: str, hudl_id: int, gt: list[dict], detected: list[dict], m: dict) -> str:
    tp, fp, fn = len(m["tp"]), len(m["fp"]), len(m["fn"])
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    offsets = [t["offset_sec"] for t in m["tp"]]
    lines = []
    lines.append(f"# Scoreboard goal-detection validation")
    lines.append(f"")
    lines.append(f"- vID:          `{vid}`  (hudl_id={hudl_id})")
    lines.append(f"- GT goals:     {len(gt)}")
    lines.append(f"- Detected:     {len(detected)}")
    lines.append(f"")
    lines.append(f"## Confusion")
    lines.append(f"")
    lines.append(f"| metric | count |")
    lines.append(f"|---|---|")
    lines.append(f"| TP (detected ∩ GT) | {tp} |")
    lines.append(f"| FP (detected, no GT) | {fp} |")
    lines.append(f"| FN (GT, not detected) | {fn} |")
    lines.append(f"| **Precision** | **{prec:.3f}** |")
    lines.append(f"| **Recall**    | **{rec:.3f}**  |")
    lines.append(f"| **F1**        | **{f1:.3f}**  |")
    if offsets:
        lines.append(f"")
        lines.append(f"## Detection offset (detected_t_sec - GT_goal_start)")
        lines.append(f"")
        lines.append(f"| stat | value |")
        lines.append(f"|---|---|")
        lines.append(f"| min  | {min(offsets):+d}s |")
        lines.append(f"| p25  | {sorted(offsets)[len(offsets)//4]:+d}s |")
        lines.append(f"| median | {median(offsets):+.0f}s |")
        lines.append(f"| p75  | {sorted(offsets)[3*len(offsets)//4]:+d}s |")
        lines.append(f"| max  | {max(offsets):+d}s |")
        lines.append(f"| mean | {mean(offsets):+.1f}s |")

    lines.append(f"")
    lines.append(f"## TP matches")
    lines.append(f"")
    lines.append(f"| GT t | GT team | detected t | offset | side | score |")
    lines.append(f"|---|---|---|---|---|---|")
    for t in m["tp"]:
        g, d = t["gt"], t["det"]
        lines.append(f"| {g['start']}s | {g['team']} | {d['detected_t_sec']}s "
                       f"| {t['offset_sec']:+d}s | {d['side']} | "
                       f"{d['score_before']}→{d['score_after']} |")

    if m["fp"]:
        lines.append(f"")
        lines.append(f"## FP (detected but no GT goal in lookback window)")
        lines.append(f"")
        lines.append(f"| detected t | side | score | lookback |")
        lines.append(f"|---|---|---|---|")
        for d in m["fp"]:
            lines.append(f"| {d['detected_t_sec']}s | {d['side']} | "
                           f"{d['score_before']}→{d['score_after']} | "
                           f"[{d['lookback_start']}-{d['lookback_end']}]s |")

    if m["fn"]:
        lines.append(f"")
        lines.append(f"## FN (GT goal not detected)")
        lines.append(f"")
        lines.append(f"| GT t | GT team |")
        lines.append(f"|---|---|")
        for f in m["fn"]:
            g = f["gt"]
            lines.append(f"| {g['start']}s | {g['team']} |")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vID", required=True)
    ap.add_argument("--hudl-id", type=int, default=None,
                     help="Override hudl-id; defaults to VID_TO_HUDL[vID]")
    ap.add_argument("--out", default=None,
                     help="Output markdown path. Default: data/output/evals/scoreboard_validation_{vID}.md")
    args = ap.parse_args()

    hudl_id = args.hudl_id or VID_TO_HUDL.get(args.vID)
    if not hudl_id:
        print(f"ERROR: no hudl_id for {args.vID}", file=sys.stderr); sys.exit(1)

    gt = load_gt_goals(hudl_id)
    det = load_detected(args.vID)
    if not gt and not det:
        print(f"ERROR: no GT and no detections for {args.vID}", file=sys.stderr); sys.exit(1)

    m = match(gt, det)
    out_text = report(args.vID, hudl_id, gt, det, m)
    out_path = Path(args.out) if args.out else \
               REPO / "data" / "output" / "evals" / f"scoreboard_validation_{args.vID}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text)
    print(out_text)
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
