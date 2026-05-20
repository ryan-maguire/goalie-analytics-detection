"""
Diagnostic: compare goal-area motion vs whole-frame motion across a 5-min
clip, with GT shot timestamps overlaid. Purpose: validate the hypothesis
that motion localized to a 2x-padded YOLO goal ROI has better signal-to-
noise than the existing whole-frame motion signal.

If goal-area motion spikes at GT shots and stays low elsewhere — while
whole-frame motion is flat or noisy across the same span — then the
goal-anchored motion build is worth doing. If both signals look similar,
abort.

Usage:
    python3 util/diag_goal_area_motion.py \
        --video data/videos/bfEKgtOIkQU.mp4 \
        --gt-csv data/ground_truth/gt_2072195.csv \
        --start 0 --duration 300
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_seg.net_detection import _load_model_lazy, CLS_GOAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    required=True)
    ap.add_argument("--gt-csv",   required=True)
    ap.add_argument("--start",    type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--pad",      type=float, default=2.0,
                    help="goal-bbox expansion factor (e.g. 2.0 = 2x w & 2x h)")
    ap.add_argument("--conf",     type=float, default=0.5,
                    help="YOLO conf threshold for goal class")
    args = ap.parse_args()

    import cv2
    import numpy as np

    # Load GT shot windows that overlap [start, start+duration]
    gt_shots = []
    with open(args.gt_csv) as f:
        for row in csv.DictReader(f):
            if row.get("action", "").strip().lower() != "shots":
                continue
            try:
                s = float(row["start"]); e = float(row["end"])
            except (ValueError, KeyError):
                continue
            if e < args.start or s > args.start + args.duration:
                continue
            gt_shots.append((s, e))
    print(f"GT shots in clip: {gt_shots}", file=sys.stderr)

    model = _load_model_lazy()
    if model is None:
        print("ERROR: HockeyAI model not available", file=sys.stderr)
        return 2
    cls_goal_idx = None
    for idx, name in model.names.items():
        if name == CLS_GOAL:
            cls_goal_idx = idx; break
    if cls_goal_idx is None:
        print("ERROR: goal class not in model", file=sys.stderr); return 2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr); return 2

    times = [args.start + i for i in range(int(args.duration))]
    prev_small = None
    last_goal_bbox = None  # carry forward when frame has no detection

    print("t\twhole_motion\tgoal_motion\thas_goal_bbox\tis_gt_shot")
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (W // 2, H // 2))
        sh, sw = small.shape

        # YOLO goal detection (top-conf goal bbox if any)
        results = model.predict(source=frame, conf=args.conf, verbose=False,
                                classes=[cls_goal_idx])
        cur_goal_bbox = None
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_i = int(boxes.conf.argmax().item())
            x1, y1, x2, y2 = boxes.xyxy[best_i].cpu().numpy().tolist()
            cur_goal_bbox = (x1, y1, x2, y2)
            last_goal_bbox = cur_goal_bbox
        use_bbox = cur_goal_bbox or last_goal_bbox  # carry-forward

        if prev_small is None:
            prev_small = small
            print(f"{t}\t0.000\t0.000\t{int(cur_goal_bbox is not None)}\t0")
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_small, small, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Whole-frame motion (same central crop as compute_motion_thirds)
        y0, y1 = sh // 5, 4 * sh // 5
        x0, x1 = sw // 8, 7 * sw // 8
        whole = float(np.mean(mag[y0:y1, x0:x1]))

        # Goal-area motion: expand goal bbox by `pad` factor, then crop in
        # SMALL-FRAME coordinates (divide by 2 since small is W//2 x H//2).
        if use_bbox is not None:
            gx1, gy1, gx2, gy2 = use_bbox
            gcx = (gx1 + gx2) / 2; gcy = (gy1 + gy2) / 2
            gw  = (gx2 - gx1) * args.pad; gh = (gy2 - gy1) * args.pad
            ex1 = max(0, int((gcx - gw / 2) / 2))
            ey1 = max(0, int((gcy - gh / 2) / 2))
            ex2 = min(sw, int((gcx + gw / 2) / 2))
            ey2 = min(sh, int((gcy + gh / 2) / 2))
            roi = mag[ey1:ey2, ex1:ex2]
            goal_motion = float(np.mean(roi)) if roi.size else 0.0
        else:
            goal_motion = 0.0

        is_gt = int(any(s <= t < e for s, e in gt_shots))
        print(f"{t}\t{whole:.3f}\t{goal_motion:.3f}\t"
              f"{int(cur_goal_bbox is not None)}\t{is_gt}")

        prev_small = small

    cap.release()


if __name__ == "__main__":
    sys.exit(main() or 0)
