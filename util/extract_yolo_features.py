"""
Extract per-second HockeyAI YOLO features for a video. Output is a TSV
that downstream training (util/train_shot_classifier.py) can consume
many times without re-running expensive inference.

Per-second feature vector (columns):
    t              second
    n_puck         puck detections (any conf >= conf_floor)
    puck_conf_max  max puck confidence
    puck_conf_mean mean puck confidence
    n_goal         goal detections
    goal_conf_max  max goal confidence
    n_goalie       goalie detections
    goalie_conf_max max goalie confidence
    n_player       player detections
    n_centroid     centroid-class detections
    n_faceoff      faceoff-class detections
    n_referee      referee detections
    puck_goal_min_dist  min center-to-center distance, normalised
                        by frame diagonal (NaN if either missing)
    goalie_goal_min_dist same for goalie<->goal
    puck_in_goal_roi    1 if any puck bbox overlaps any goal bbox
    puck_in_goal_ext2   1 if any puck bbox overlaps 2x-expanded goal box

Usage:
    python3 util/extract_yolo_features.py \\
        --video data/videos/bfEKgtOIkQU.mp4 \\
        --out   data/output/yolo_features/bfEKgtOIkQU.tsv \\
        [--fps 1.0] [--conf 0.05]

Cost: ~150ms per frame × duration_sec × fps. At 1fps a 70-min game
is ~600s of inference. Idempotent — skips if output already exists
(use --force to overwrite).
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_seg.net_detection import _load_model_lazy, CLS_GOAL, CLS_GOALIE, CLS_PLAYER


CLASS_NAMES = {  # populated from model.names at runtime, but mapped here for clarity
    "centriod": "centriod",   # sic — that's how the model spells it
    "faceoff":  "faceoff",
    "goal":     CLS_GOAL,
    "goalie":   CLS_GOALIE,
    "player":   CLS_PLAYER,
    "puck":     "puck",
    "referee":  "referee",
}


def _bbox_center(xyxy):
    return ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)


def _bbox_overlap(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw > 0 and ih > 0


def _expand_bbox(bbox, factor):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w  = (x2 - x1) * factor
    h  = (y2 - y1) * factor
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--fps",   type=float, default=1.0)
    ap.add_argument("--conf",  type=float, default=0.05,
                    help="low conf floor — features include count/max even "
                         "for low-conf detections so downstream training can "
                         "learn its own thresholds")
    ap.add_argument("--force", action="store_true",
                    help="overwrite output if it exists")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"[skip] {args.out} already exists (use --force to overwrite)",
              file=sys.stderr)
        return 0

    import cv2

    model = _load_model_lazy()
    if model is None:
        print("ERROR: HockeyAI model not available", file=sys.stderr)
        return 2

    cls_idx_of: dict[str, int] = {}
    for idx, name in model.names.items():
        if name in CLASS_NAMES:
            cls_idx_of[name] = idx
    print(f"Class indices: {cls_idx_of}", file=sys.stderr)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        return 2
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps_video
    print(f"Video: {duration:.0f}s @ {fps_video:.1f}fps", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = [
        "t",
        "n_puck", "puck_conf_max", "puck_conf_mean",
        "n_goal", "goal_conf_max",
        "n_goalie", "goalie_conf_max",
        "n_player", "n_centroid", "n_faceoff", "n_referee",
        "puck_goal_min_dist",
        "goalie_goal_min_dist",
        "puck_in_goal_roi",
        "puck_in_goal_ext2",
    ]

    step = 1.0 / args.fps
    n_samples = int(duration / step)

    with open(args.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for i in range(n_samples):
            t = i * step
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            H, W = frame.shape[:2]
            diag = math.hypot(W, H)

            results = model.predict(source=frame, conf=args.conf, verbose=False)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                # write a zero-feature row so the time series stays dense
                row = [t] + [0] * (len(cols) - 1)
                f.write("\t".join(str(v) for v in row) + "\n")
                continue
            boxes = results[0].boxes
            pucks, goals, goalies, players = [], [], [], []
            n_centroid = n_faceoff = n_referee = 0
            for j in range(len(boxes)):
                cls_idx  = int(boxes.cls[j].item())
                conf     = float(boxes.conf[j].item())
                xyxy     = tuple(boxes.xyxy[j].cpu().numpy().tolist())
                cls_name = model.names.get(cls_idx, "")
                entry    = {"conf": conf, "xyxy": xyxy}
                if   cls_name == "puck":     pucks.append(entry)
                elif cls_name == CLS_GOAL:   goals.append(entry)
                elif cls_name == CLS_GOALIE: goalies.append(entry)
                elif cls_name == CLS_PLAYER: players.append(entry)
                elif cls_name == "centriod": n_centroid += 1
                elif cls_name == "faceoff":  n_faceoff  += 1
                elif cls_name == "referee":  n_referee  += 1

            n_puck   = len(pucks)
            n_goal   = len(goals)
            n_goalie = len(goalies)
            n_player = len(players)
            puck_conf_max  = max((p["conf"] for p in pucks),  default=0.0)
            puck_conf_mean = (sum(p["conf"] for p in pucks) / n_puck) if pucks else 0.0
            goal_conf_max  = max((g["conf"] for g in goals),  default=0.0)
            goalie_conf_max = max((g["conf"] for g in goalies), default=0.0)

            puck_goal_min = ""
            puck_in_goal  = 0
            puck_in_ext2  = 0
            if pucks and goals:
                puck_goal_min = min(
                    math.hypot(*[a - b for a, b in zip(_bbox_center(p["xyxy"]),
                                                         _bbox_center(g["xyxy"]))])
                    for p in pucks for g in goals
                ) / diag
                for p in pucks:
                    for g in goals:
                        if _bbox_overlap(p["xyxy"], g["xyxy"]):
                            puck_in_goal = 1
                        if _bbox_overlap(p["xyxy"], _expand_bbox(g["xyxy"], 2.0)):
                            puck_in_ext2 = 1
                puck_goal_min = f"{puck_goal_min:.4f}"

            goalie_goal_min = ""
            if goalies and goals:
                goalie_goal_min_val = min(
                    math.hypot(*[a - b for a, b in zip(_bbox_center(gk["xyxy"]),
                                                         _bbox_center(g["xyxy"]))])
                    for gk in goalies for g in goals
                ) / diag
                goalie_goal_min = f"{goalie_goal_min_val:.4f}"

            row = [
                t,
                n_puck, f"{puck_conf_max:.4f}", f"{puck_conf_mean:.4f}",
                n_goal, f"{goal_conf_max:.4f}",
                n_goalie, f"{goalie_conf_max:.4f}",
                n_player, n_centroid, n_faceoff, n_referee,
                puck_goal_min,
                goalie_goal_min,
                puck_in_goal,
                puck_in_ext2,
            ]
            f.write("\t".join(str(v) for v in row) + "\n")
            if (i + 1) % 300 == 0:
                print(f"  {i+1}/{n_samples} ({(i+1)*100//n_samples}%)", file=sys.stderr)

    cap.release()
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
