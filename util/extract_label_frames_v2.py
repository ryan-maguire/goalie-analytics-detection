"""
v2 of extract_label_frames.py — dynamic shot-anchor selection.

For each opponent-team GT 'Shots' window [start, end], scan EVERY second
in that range, run HockeyAI to score goal-class confidence, and extract
ONE positive frame: the second with the highest goal confidence. If no
goal is detected anywhere in the window (camera elsewhere), skip the
shot entirely — we don't want training positives without a visible goal.

This fixes the v1 mistake of anchoring at GT_start+1s. Validation
(per util/preview_gt_windows.py) showed the actual shot moment is
typically at +5 to +8s, not +1s.

This script ONLY (re-)extracts positives. Negatives and hard-negatives
from v1 stay in place — they're unaffected by the anchor-timing issue.

Filename remains `{vID}_pos_{best_t:05d}.jpg` (absolute second), so
labeling tools / training pipelines work unchanged.

Usage:
    # 1. Delete existing positives (v1 anchored at wrong moment)
    rm data/labels/images/*_pos_*.jpg data/labels/labels/*_pos_*.txt

    # 2. Re-extract
    python3 util/extract_label_frames_v2.py \\
        --customers data/customers/CUST000048.json data/customers/CUST000031.json
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_cv_seg_output import VID_TO_HUDL, _team_names_match
from cv_seg.net_detection import _load_model_lazy, CLS_GOAL


def _load_vid_to_opponent_team(customer_paths):
    out = {}
    for p in customer_paths:
        for rec in json.load(open(p)):
            vid = str(rec.get("vID", "")).strip()
            if vid:
                out[vid] = rec.get("opponentGoalieTeamName") or ""
    return out


def _load_gt_shot_windows(gt_csv, opp_team):
    out = []
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            if row.get("action", "").strip().lower() != "shots":
                continue
            team = row.get("team", "").strip()
            if opp_team and not _team_names_match(team, opp_team):
                continue
            try:
                s = int(float(row["start"])); e = int(float(row["end"]))
            except (ValueError, KeyError):
                continue
            out.append((s, e))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="data/videos")
    ap.add_argument("--gt-dir",     default="data/ground_truth")
    ap.add_argument("--customers",  nargs="+", required=True)
    ap.add_argument("--out-dir",    default="data/labels/images")
    ap.add_argument("--vIDs",       nargs="*", default=None)
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="lowest goal-conf to consider when picking the "
                         "best second. Low so signal-starved camera angles "
                         "still get a chance.")
    ap.add_argument("--min-conf-to-keep", type=float, default=0.25,
                    help="if best second's goal-conf is below this, SKIP the "
                         "shot entirely (no useful frame in window).")
    args = ap.parse_args()

    import cv2

    model = _load_model_lazy()
    if model is None:
        print("ERROR: HockeyAI not available", file=sys.stderr); return 2
    cls_goal_idx = None
    for idx, name in model.names.items():
        if name == CLS_GOAL:
            cls_goal_idx = idx; break
    if cls_goal_idx is None:
        print("ERROR: no goal class in model", file=sys.stderr); return 2

    opp = _load_vid_to_opponent_team(args.customers)
    vids = args.vIDs or sorted(VID_TO_HUDL.keys())
    os.makedirs(args.out_dir, exist_ok=True)

    totals = {"shots_considered": 0, "extracted": 0, "skipped_no_goal": 0}
    for vid in vids:
        video_path = os.path.join(args.videos_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            continue
        hudl = VID_TO_HUDL.get(vid)
        gt_csv = os.path.join(args.gt_dir, f"gt_{hudl}.csv")
        if not os.path.exists(gt_csv):
            continue
        shots = _load_gt_shot_windows(gt_csv, opp.get(vid, ""))
        if not shots:
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  cannot open {video_path}", file=sys.stderr); continue
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur_sec = int(n_frames / fps_v)

        n_extracted = n_skipped = 0
        for s, e in shots:
            totals["shots_considered"] += 1
            # Scan every second in [s, e], track (t, best_conf, best_frame)
            best_t = None
            best_conf = -1.0
            best_frame = None
            for t in range(max(0, s), min(dur_sec, e + 1)):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                results = model.predict(source=frame, conf=args.conf_floor,
                                        verbose=False, classes=[cls_goal_idx])
                if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                    continue
                conf = float(results[0].boxes.conf.max().item())
                if conf > best_conf:
                    best_conf = conf
                    best_t = t
                    best_frame = frame.copy()

            if best_frame is None or best_conf < args.min_conf_to_keep:
                n_skipped += 1
                totals["skipped_no_goal"] += 1
                continue

            out_path = os.path.join(args.out_dir, f"{vid}_pos_{best_t:05d}.jpg")
            cv2.imwrite(out_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            n_extracted += 1
            totals["extracted"] += 1

        cap.release()
        print(f"  {vid}: extracted {n_extracted}/{len(shots)}  "
              f"skipped_no_goal={n_skipped}", file=sys.stderr)

    print(f"\nTotals: extracted={totals['extracted']}  "
          f"skipped_no_goal={totals['skipped_no_goal']}  "
          f"of {totals['shots_considered']} shots considered",
          file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
