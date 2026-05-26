"""
Mine hard-negative frames for the approach-2.5 retraining.

The approach-2 failure (F1=0.23) was caused by training-data
distribution mismatch: 352 negatives can't represent the ~30,000+
non-shot game seconds. This script samples frames from the actual
non-shot distribution, focusing on seconds that visually LOOK like
shots (goal visible + nearby players) — exactly the cases where the
model over-fired at game-time inference.

Sources:
  - util/extract_yolo_features.py output TSVs (already computed for the
    original 9 videos: bfEK, dwGs, Fjc, J8Wk, krxh, KYtM, mjEe, SX5x, v0lx)
  - data/ground_truth/gt_<hudl>.csv (to exclude shot seconds)

Score per candidate second:
    score = goal_conf_max * player_weight * activity_weight
  where:
    player_weight  = 1.0 if n_player >= 5 else 0.5
    activity_weight = 1.0 if n_goalie >= 1 else 0.7
This biases the pool toward "goal in frame with bodies near it" —
the visual signature that the over-firing approach-2 model learned
to call a shot.

Picks top N by score, extracts the frames, writes them as
`{vID}_hardneg2_{t:05d}.jpg`. Subsequent prelabel+autolabel cycles
will skip existing files and process only the new hardnegs.

Usage:
    python3 util/sample_hard_negatives.py \\
        --customers data/customers/CUST000048.json data/customers/CUST000031.json \\
        --target-count 2000
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_cv_seg_output import VID_TO_HUDL, _team_names_match


# Buffer around GT shot windows to avoid sampling near-shot frames
# as negatives (those might still be shot-influenced)
SHOT_BUFFER_SEC = 5


def _load_opp(customer_paths):
    out = {}
    for p in customer_paths:
        for rec in json.load(open(p)):
            vid = str(rec.get("vID", "")).strip()
            if vid:
                out[vid] = rec.get("opponentGoalieTeamName") or ""
    return out


def _load_shot_seconds(gt_csv, opp_team, buffer_sec):
    secs: set[int] = set()
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
            for t in range(max(0, s - buffer_sec), e + buffer_sec + 1):
                secs.add(t)
    return secs


def _load_features(tsv_path):
    """Yield (t, n_goal, goal_conf_max, n_goalie, n_player) per row."""
    with open(tsv_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            try:
                t = int(float(parts[idx["t"]]))
                n_goal   = int(parts[idx["n_goal"]])
                gc       = float(parts[idx["goal_conf_max"]])
                n_goalie = int(parts[idx["n_goalie"]])
                n_player = int(parts[idx["n_player"]])
            except (ValueError, KeyError):
                continue
            yield t, n_goal, gc, n_goalie, n_player


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default="data/output/yolo_features")
    ap.add_argument("--gt-dir",       default="data/ground_truth")
    ap.add_argument("--videos-dir",   default="data/videos")
    ap.add_argument("--customers",    nargs="+", required=True)
    ap.add_argument("--out-dir",      default="data/labels/images")
    ap.add_argument("--target-count", type=int, default=2000,
                    help="total hardneg2 frames to extract")
    ap.add_argument("--min-goal-conf", type=float, default=0.30,
                    help="candidates need at least this goal_conf_max to qualify")
    args = ap.parse_args()

    import cv2

    opp = _load_opp(args.customers)
    os.makedirs(args.out_dir, exist_ok=True)

    # Build global candidate pool (vID, t, score)
    candidates = []
    feature_files = sorted(Path(args.features_dir).glob("*.tsv"))
    print(f"Found {len(feature_files)} feature TSVs", file=sys.stderr)

    for tsv in feature_files:
        vid = tsv.stem
        # Fallback: hudl-fetched matches have vID == hudl_id (numeric).
        # Skip non-numeric vIDs that aren't in VID_TO_HUDL (those are
        # genuinely unknown).
        if vid not in VID_TO_HUDL and not vid.isdigit():
            continue
        hudl = VID_TO_HUDL[vid] if vid in VID_TO_HUDL else int(vid)
        gt_csv = os.path.join(args.gt_dir, f"gt_{hudl}.csv")
        if not os.path.exists(gt_csv):
            continue
        shot_secs = _load_shot_seconds(gt_csv, opp.get(vid, ""),
                                       SHOT_BUFFER_SEC)
        n_added = 0
        for t, n_goal, gc, n_goalie, n_player in _load_features(str(tsv)):
            if t in shot_secs:
                continue
            if n_goal < 1 or gc < args.min_goal_conf:
                continue
            player_w   = 1.0 if n_player >= 5 else 0.5
            activity_w = 1.0 if n_goalie >= 1 else 0.7
            score = gc * player_w * activity_w
            candidates.append((score, vid, t))
            n_added += 1
        print(f"  {vid}: {n_added} candidates (after shot/goal-conf filters)",
              file=sys.stderr)

    candidates.sort(reverse=True)  # highest score first
    picked = candidates[: args.target_count]
    print(f"\nTotal candidates: {len(candidates)}", file=sys.stderr)
    print(f"Picking top {len(picked)} (target={args.target_count})", file=sys.stderr)

    # Group picks by video so we open each capture only once
    per_vid: dict[str, list[int]] = {}
    for _, vid, t in picked:
        per_vid.setdefault(vid, []).append(t)

    n_written = n_skipped_exists = 0
    for vid, ts in sorted(per_vid.items()):
        video_path = os.path.join(args.videos_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"  [skip] {vid}: no video", file=sys.stderr); continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [skip] {vid}: cannot open", file=sys.stderr); continue
        ts.sort()  # seek monotonically increasing
        n_v = 0
        for t in ts:
            out_path = os.path.join(args.out_dir,
                                    f"{vid}_hardneg2_{t:05d}.jpg")
            if os.path.exists(out_path):
                n_skipped_exists += 1
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                n_v += 1; n_written += 1
        cap.release()
        print(f"  {vid}: wrote {n_v} hardneg2 frames", file=sys.stderr)

    print(f"\nTotals: wrote={n_written}  skipped_existing={n_skipped_exists}",
          file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
