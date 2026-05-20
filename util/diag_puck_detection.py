"""
Diagnostic: characterise puck class detection quality on amateur hockey
footage. Run YOLO at 1 fps over a 5-min clip and report:

  - per-second detection rate (frames with ≥1 puck, ≥1 goal)
  - puck confidence distribution
  - puck<->goal pixel distance distribution when both present

This is a feasibility check before building YOLO-driven shot detection.
If puck recall is poor on amateur broadcasts, approach 1 (puck-near-goal)
is dead and we pivot to goalie-reaction or other signals.

Usage:
    python3 util/diag_puck_detection.py \
        --video data/videos/bfEKgtOIkQU.mp4 \
        --start 0 --duration 300 --fps 1
"""

import argparse
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

# Reuse cv_seg's lazy-loaded model and its constants
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_seg.net_detection import _load_model_lazy, CLS_GOAL, CLS_GOALIE


def _bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _bbox_size(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x2 - x1, y2 - y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    required=True)
    ap.add_argument("--start",    type=float, default=0.0,   help="start time (s)")
    ap.add_argument("--duration", type=float, default=300.0, help="seconds to scan")
    ap.add_argument("--fps",      type=float, default=1.0,   help="sample rate")
    ap.add_argument("--conf",     type=float, default=0.05,
                    help="minimum confidence to keep — set LOW to see the "
                         "raw distribution, not just supra-threshold dets")
    args = ap.parse_args()

    import cv2  # late import keeps --help fast

    model = _load_model_lazy()
    if model is None:
        print("ERROR: could not load HockeyAI model", file=sys.stderr)
        return 2

    cls_puck   = None
    cls_goal   = None
    cls_goalie = None
    for idx, name in model.names.items():
        if name == "puck":         cls_puck   = idx
        elif name == CLS_GOAL:     cls_goal   = idx
        elif name == CLS_GOALIE:   cls_goalie = idx
    print(f"Model classes: {model.names}")
    print(f"  puck idx={cls_puck}  goal idx={cls_goal}  goalie idx={cls_goalie}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        return 2

    step  = 1.0 / args.fps
    times = [args.start + i * step for i in range(int(args.duration / step))]

    n_frames_total          = 0
    n_with_puck             = 0
    n_with_goal             = 0
    n_with_goalie           = 0
    n_with_puck_and_goal    = 0
    n_with_all_three        = 0
    puck_confs:    list[float] = []
    goal_confs:    list[float] = []
    puck_widths:   list[float] = []
    puck_to_goal_dists: list[float] = []
    multi_puck_frames = 0
    multi_goal_frames = 0
    # v2 metrics: top-1 puck filtering + bbox-overlap geometry
    n_top1_near_goal        = 0   # top-conf puck center within 5% of frame diag
    n_top1_inside_goal_roi  = 0   # top-conf puck bbox overlaps goal bbox at all
    n_top1_inside_goal_ext  = 0   # top-conf puck bbox overlaps goal bbox expanded 2x
    sustained_near_goal_runs: list[int] = []
    near_goal_streak = 0

    print(f"Scanning {len(times)} frames "
          f"(start={args.start}s, dur={args.duration}s, fps={args.fps}, "
          f"conf={args.conf}) …")
    last_log = -1
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        n_frames_total += 1
        H, W = frame.shape[:2]
        diag = math.hypot(W, H)

        results = model.predict(source=frame, conf=args.conf, verbose=False)
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        pucks  = []
        goals  = []
        goalies = []
        for j in range(len(boxes)):
            cls_idx = int(boxes.cls[j].item())
            conf    = float(boxes.conf[j].item())
            xyxy    = boxes.xyxy[j].cpu().numpy().tolist()
            entry   = {"conf": conf, "xyxy": xyxy}
            if cls_idx == cls_puck:    pucks.append(entry)
            elif cls_idx == cls_goal:  goals.append(entry)
            elif cls_idx == cls_goalie: goalies.append(entry)

        if pucks:
            n_with_puck += 1
            puck_confs.append(max(p["conf"] for p in pucks))
            for p in pucks:
                w, h = _bbox_size(p["xyxy"])
                puck_widths.append(w)
            if len(pucks) > 1:
                multi_puck_frames += 1
        if goals:
            n_with_goal += 1
            goal_confs.append(max(g["conf"] for g in goals))
            if len(goals) > 1:
                multi_goal_frames += 1
        if goalies:
            n_with_goalie += 1
        if pucks and goals:
            n_with_puck_and_goal += 1
            # Min distance puck-center to goal-center, normalized by frame diagonal
            min_d = min(
                math.hypot(*[a - b for a, b in zip(_bbox_center(p["xyxy"]),
                                                     _bbox_center(g["xyxy"]))])
                for p in pucks for g in goals
            )
            puck_to_goal_dists.append(min_d / diag)

            # v2: top-1 puck only (highest conf), plus bbox-overlap geometry
            top_puck = max(pucks, key=lambda p: p["conf"])
            top_goal = max(goals, key=lambda g: g["conf"])
            pcx, pcy = _bbox_center(top_puck["xyxy"])
            gcx, gcy = _bbox_center(top_goal["xyxy"])
            top1_dist_frac = math.hypot(pcx - gcx, pcy - gcy) / diag
            if top1_dist_frac <= 0.05:
                n_top1_near_goal += 1

            # bbox overlap: does the top-1 puck bbox intersect the goal bbox?
            px1, py1, px2, py2 = top_puck["xyxy"]
            gx1, gy1, gx2, gy2 = top_goal["xyxy"]
            iw = max(0, min(px2, gx2) - max(px1, gx1))
            ih = max(0, min(py2, gy2) - max(py1, gy1))
            inside_goal      = (iw > 0 and ih > 0)
            # Expanded goal ROI: 2x the goal bbox dimensions, centered on goal
            gw, gh = gx2 - gx1, gy2 - gy1
            ex1, ey1 = gcx - gw, gcy - gh
            ex2, ey2 = gcx + gw, gcy + gh
            iew = max(0, min(px2, ex2) - max(px1, ex1))
            ieh = max(0, min(py2, ey2) - max(py1, ey1))
            inside_ext = (iew > 0 and ieh > 0)

            if inside_goal:
                n_top1_inside_goal_roi += 1
            if inside_ext:
                n_top1_inside_goal_ext += 1
            if inside_ext:
                near_goal_streak += 1
            else:
                if near_goal_streak > 0:
                    sustained_near_goal_runs.append(near_goal_streak)
                near_goal_streak = 0
        else:
            if near_goal_streak > 0:
                sustained_near_goal_runs.append(near_goal_streak)
            near_goal_streak = 0
        if pucks and goals and goalies:
            n_with_all_three += 1

        if i // 30 != last_log:
            last_log = i // 30
            print(f"  {i+1}/{len(times)} frames scanned …", file=sys.stderr)

    cap.release()

    def _summarise(name, xs, fmt="{:.3f}"):
        if not xs:
            print(f"  {name}: (no samples)")
            return
        xs = sorted(xs)
        n = len(xs)
        med = statistics.median(xs)
        p25 = xs[n // 4]
        p75 = xs[(3 * n) // 4]
        print(f"  {name}: n={n}  "
              f"min={fmt.format(xs[0])}  p25={fmt.format(p25)}  "
              f"med={fmt.format(med)}  p75={fmt.format(p75)}  max={fmt.format(xs[-1])}")

    print("\n=== Detection rates ===")
    print(f"  frames scanned:            {n_frames_total}")
    print(f"  with ≥1 puck:              {n_with_puck} ({n_with_puck/max(1,n_frames_total):.1%})")
    print(f"  with ≥1 goal:              {n_with_goal} ({n_with_goal/max(1,n_frames_total):.1%})")
    print(f"  with ≥1 goalie:            {n_with_goalie} ({n_with_goalie/max(1,n_frames_total):.1%})")
    print(f"  with puck AND goal:        {n_with_puck_and_goal} ({n_with_puck_and_goal/max(1,n_frames_total):.1%})")
    print(f"  with puck AND goal AND goalie: {n_with_all_three} ({n_with_all_three/max(1,n_frames_total):.1%})")
    print(f"  frames with multiple pucks: {multi_puck_frames}")
    print(f"  frames with multiple goals: {multi_goal_frames}")

    print("\n=== Distributions ===")
    _summarise("puck max-conf per frame", puck_confs)
    _summarise("goal max-conf per frame", goal_confs)
    _summarise("puck bbox width (px)",     puck_widths, fmt="{:.1f}")
    _summarise("puck<->goal min dist (fraction of frame diagonal)", puck_to_goal_dists)

    # close out any trailing streak
    if near_goal_streak > 0:
        sustained_near_goal_runs.append(near_goal_streak)

    print("\n=== v2: top-1 puck + geometry ===")
    print(f"  top-1 puck center within 5% diagonal of goal center: "
          f"{n_top1_near_goal} / {n_with_puck_and_goal} frames")
    print(f"  top-1 puck bbox intersects goal bbox:                 "
          f"{n_top1_inside_goal_roi} / {n_with_puck_and_goal}")
    print(f"  top-1 puck bbox intersects 2x-expanded goal ROI:      "
          f"{n_top1_inside_goal_ext} / {n_with_puck_and_goal}")
    if sustained_near_goal_runs:
        sustained_near_goal_runs.sort()
        n = len(sustained_near_goal_runs)
        print(f"  sustained near-goal runs (consecutive 1-fps frames "
              f"with top-1 puck in 2x goal ROI):")
        print(f"    count={n}  total_frames={sum(sustained_near_goal_runs)}  "
              f"min={sustained_near_goal_runs[0]}  "
              f"med={sustained_near_goal_runs[n//2]}  "
              f"max={sustained_near_goal_runs[-1]}")
        # Bucket by duration
        buckets = Counter()
        for r in sustained_near_goal_runs:
            if   r == 1: buckets["1s"]   += 1
            elif r == 2: buckets["2s"]   += 1
            elif r <= 5: buckets["3-5s"] += 1
            else:        buckets["6+s"]  += 1
        print(f"    duration buckets: {dict(buckets)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
