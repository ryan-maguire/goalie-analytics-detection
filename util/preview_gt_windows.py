"""
Pick N GT shot windows, extract every-second frames from start to end,
overlay goal (green) and derived-shot (red) bboxes, and write into
per-window subdirectories so you can flip through and validate:
  - the GT window actually contains a shot
  - the camera frames the goal at the right moment
  - the derived shot bbox covers the right area
  - HockeyAI's goal detection holds across the play

Output layout:
    {out_dir}/{vID}_{start}s-{end}s/
        t{abs_t:05d}.jpg    # absolute video second
        t{abs_t:05d}.jpg
        ...
        meta.txt            # GT pos_x, pos_y, team, action notes

Usage:
    python3 util/preview_gt_windows.py \\
        --customers data/customers/CUST000048.json data/customers/CUST000031.json \\
        --out-dir data/labels/_preview_gt \\
        --windows-per-video 1   # 9 videos × 1 = 9 windows
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_cv_seg_output import VID_TO_HUDL, _team_names_match
from cv_seg.net_detection import _load_model_lazy, CLS_GOAL


W_FACTOR = 1.8
H_FACTOR = 1.5


def _load_vid_to_opponent_team(customer_paths):
    out = {}
    for p in customer_paths:
        for rec in json.load(open(p)):
            vid = str(rec.get("vID", "")).strip()
            if vid:
                out[vid] = rec.get("opponentGoalieTeamName") or ""
    return out


def _load_gt_shots(gt_csv, opp_team):
    """Return [(start, end, pos_x, pos_y, team, half)]."""
    out = []
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            if row.get("action", "").strip().lower() != "shots":
                continue
            team = row.get("team", "").strip()
            if opp_team and not _team_names_match(team, opp_team):
                continue
            try:
                s = int(float(row["start"]))
                e = int(float(row["end"]))
                px = float(row["pos_x"]) if row.get("pos_x") else None
                py = float(row["pos_y"]) if row.get("pos_y") else None
                half = row.get("half", "")
            except (ValueError, KeyError):
                continue
            out.append((s, e, px, py, team, half))
    return out


def _expand_bbox(cx, cy, w, h, w_factor, h_factor):
    nw = min(1.0, w * w_factor)
    nh = min(1.0, h * h_factor)
    ncx = max(nw / 2, min(1.0 - nw / 2, cx))
    ncy = max(nh / 2, min(1.0 - nh / 2, cy))
    return ncx, ncy, nw, nh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="data/videos")
    ap.add_argument("--gt-dir",     default="data/ground_truth")
    ap.add_argument("--customers",  nargs="+", required=True)
    ap.add_argument("--out-dir",    default="data/labels/_preview_gt")
    ap.add_argument("--vIDs",       nargs="*", default=None)
    ap.add_argument("--windows-per-video", type=int, default=1)
    ap.add_argument("--conf",       type=float, default=0.25)
    args = ap.parse_args()

    import cv2

    model = _load_model_lazy()
    if model is None:
        print("ERROR: HockeyAI not available", file=sys.stderr); return 2
    cls_goal_idx = None
    for idx, name in model.names.items():
        if name == CLS_GOAL:
            cls_goal_idx = idx; break

    opp = _load_vid_to_opponent_team(args.customers)
    vids = args.vIDs or sorted(VID_TO_HUDL.keys())
    os.makedirs(args.out_dir, exist_ok=True)

    total_windows = 0
    total_frames  = 0

    for vid in vids:
        video_path = os.path.join(args.videos_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            continue
        hudl = VID_TO_HUDL.get(vid)
        gt_csv = os.path.join(args.gt_dir, f"gt_{hudl}.csv")
        if not os.path.exists(gt_csv):
            continue
        shots = _load_gt_shots(gt_csv, opp.get(vid, ""))
        if not shots:
            continue

        # Pick the first N shots per video (deterministic, reproducible)
        picks = shots[:args.windows_per_video]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  cannot open {video_path}", file=sys.stderr); continue
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0

        for shot_start, shot_end, px, py, team, half in picks:
            subdir = Path(args.out_dir) / f"{vid}_{shot_start:04d}s-{shot_end:04d}s"
            subdir.mkdir(parents=True, exist_ok=True)
            # meta.txt
            with open(subdir / "meta.txt", "w") as f:
                f.write(f"vID:       {vid}\n")
                f.write(f"hudl_id:   {hudl}\n")
                f.write(f"window:    [{shot_start}, {shot_end}] seconds "
                        f"(duration {shot_end - shot_start}s)\n")
                f.write(f"team:      {team}\n")
                f.write(f"half:      {half}\n")
                f.write(f"GT pos:    pos_x={px}, pos_y={py} (rink coordinate "
                        f"system not yet decoded)\n")
                f.write(f"\nFrames: t<sec>.jpg, sec = absolute video second.\n")
                f.write(f"Green bbox = HockeyAI goal detection.\n")
                f.write(f"Red bbox   = derived shot area (goal x{W_FACTOR}w × {H_FACTOR}h).\n")

            for t in range(shot_start, shot_end + 1):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                H, W = frame.shape[:2]

                results = model.predict(source=frame, conf=args.conf, verbose=False,
                                        classes=[cls_goal_idx] if cls_goal_idx is not None else None)
                goal = None
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_i = int(boxes.conf.argmax().item())
                    x1, y1, x2, y2 = boxes.xyxy[best_i].cpu().numpy().tolist()
                    gconf = float(boxes.conf[best_i].item())
                    cx = ((x1 + x2) / 2) / W
                    cy = ((y1 + y2) / 2) / H
                    gw = (x2 - x1) / W
                    gh = (y2 - y1) / H
                    goal = (cx, cy, gw, gh, gconf)

                # Draw
                if goal:
                    cx, cy, gw, gh, gconf = goal
                    gx1 = int((cx - gw / 2) * W); gy1 = int((cy - gh / 2) * H)
                    gx2 = int((cx + gw / 2) * W); gy2 = int((cy + gh / 2) * H)
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    scx, scy, sw, sh = _expand_bbox(cx, cy, gw, gh, W_FACTOR, H_FACTOR)
                    sx1 = int((scx - sw / 2) * W); sy1 = int((scy - sh / 2) * H)
                    sx2 = int((scx + sw / 2) * W); sy2 = int((scy + sh / 2) * H)
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
                    label = f"t={t}s  goal_conf={gconf:.2f}"
                else:
                    label = f"t={t}s  NO GOAL DETECTED"

                # Status banner top-left
                cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 0), -1)
                cv2.putText(frame, label, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                # Bottom: GT context
                cv2.rectangle(frame, (0, H - 40), (W, H), (0, 0, 0), -1)
                offset = t - shot_start
                offset_label = f"+{offset}s of GT shot [{shot_start},{shot_end}]"
                cv2.putText(frame, offset_label, (10, H - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 2)

                out = subdir / f"t{t:05d}.jpg"
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                total_frames += 1

            total_windows += 1
            print(f"  {vid} [{shot_start}s-{shot_end}s] → {subdir}", file=sys.stderr)
        cap.release()

    print(f"\nDone: {total_windows} windows, {total_frames} frames written.",
          file=sys.stderr)
    print(f"Browse: {args.out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
