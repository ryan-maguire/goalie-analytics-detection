"""
Run the fine-tuned HockeyAI+shot YOLOv8 model on each video at 1fps,
extract per-second `shot` class probabilities, write a TSV per video.

The TSV is the slow, one-time output. Downstream conversion to windows
(util/yolo_probs_to_windows.py) is cheap and threshold-tunable.

Output columns per row:
    t                 second
    shot_max_conf     max conf of any class-7 (shot) detection on frame
    shot_n            number of class-7 detections on frame (any conf)
    goal_max_conf     for sanity / debugging — max goal-class conf

Usage:
    python3 util/predict_shots_yolo.py \\
        --weights runs/detect/data/labels/_yolo_workdir/runs/hockeyai_shot_finetune/weights/best.pt \\
        --videos-dir data/videos \\
        --out-dir data/output/yolo_shot_probs
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_cv_seg_output import VID_TO_HUDL


SHOT_CLASS_ID = 7
GOAL_CLASS_ID = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",    required=True)
    ap.add_argument("--videos-dir", default="data/videos")
    ap.add_argument("--out-dir",    default="data/output/yolo_shot_probs")
    ap.add_argument("--vIDs",       nargs="*", default=None)
    ap.add_argument("--fps",        type=float, default=1.0)
    ap.add_argument("--conf-floor", type=float, default=0.05,
                    help="lowest conf to track. Set low so threshold tuning "
                         "downstream isn't bounded by this.")
    ap.add_argument("--force",      action="store_true")
    args = ap.parse_args()

    import cv2
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed", file=sys.stderr); return 2

    if not os.path.exists(args.weights):
        print(f"ERROR: weights not found: {args.weights}", file=sys.stderr); return 2
    model = YOLO(args.weights)
    print(f"Loaded model: {args.weights}", file=sys.stderr)
    print(f"Classes: {model.names}", file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    vids = args.vIDs or sorted(VID_TO_HUDL.keys())

    for vid in vids:
        out_path = os.path.join(args.out_dir, f"{vid}.tsv")
        if os.path.exists(out_path) and not args.force:
            print(f"  [skip] {out_path} exists", file=sys.stderr); continue
        video_path = os.path.join(args.videos_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"  [skip] {vid}: no video", file=sys.stderr); continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [skip] {vid}: cannot open", file=sys.stderr); continue
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_sec = int(n_frames / fps_v)
        print(f"  {vid}: {dur_sec}s", file=sys.stderr)

        step = 1.0 / args.fps
        n_samples = int(dur_sec / step)
        with open(out_path, "w") as f:
            f.write("t\tshot_max_conf\tshot_n\tgoal_max_conf\n")
            for i in range(n_samples):
                t = i * step
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                results = model.predict(source=frame, conf=args.conf_floor,
                                        verbose=False)
                shot_max = 0.0; shot_n = 0; goal_max = 0.0
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    for j in range(len(boxes)):
                        cls_idx = int(boxes.cls[j].item())
                        conf = float(boxes.conf[j].item())
                        if cls_idx == SHOT_CLASS_ID:
                            shot_n += 1
                            if conf > shot_max:
                                shot_max = conf
                        elif cls_idx == GOAL_CLASS_ID:
                            if conf > goal_max:
                                goal_max = conf
                f.write(f"{int(t)}\t{shot_max:.4f}\t{shot_n}\t{goal_max:.4f}\n")
                if (i + 1) % 600 == 0:
                    print(f"    {i+1}/{n_samples}", file=sys.stderr)
        cap.release()
        print(f"    wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
