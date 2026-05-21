"""
Pre-label extracted frames with HockeyAI YOLO detections so the human
labeler only has to add a `shot` bbox (and fix the rare existing-class
mistake). Writes YOLO-format .txt files alongside each image.

YOLO label format (one line per detection):
    class_id cx cy w h   (all normalized 0-1)

Class ID assignments (matches HockeyAI's existing model.names plus the
new `shot` class appended at index 7):
    0  centriod
    1  faceoff
    2  goal
    3  goalie
    4  player
    5  puck
    6  referee
    7  shot   (NEW — labeler adds for positive frames)

Also writes data/labels/classes.txt with the class names in index
order — labelImg requires this in YOLO mode.

Usage:
    python3 util/prelabel_frames.py \\
        --images-dir data/labels/images \\
        --labels-dir data/labels/labels \\
        [--conf 0.25]
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_seg.net_detection import _load_model_lazy


CLASS_ORDER = [
    "centriod",
    "faceoff",
    "goal",
    "goalie",
    "player",
    "puck",
    "referee",
    "shot",   # new — appended at index 7
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/labels/images")
    ap.add_argument("--labels-dir", default="data/labels/labels")
    ap.add_argument("--conf",       type=float, default=0.25,
                    help="confidence threshold for pre-label detections. "
                         "Lower = more bboxes to delete, higher = more to add. "
                         "0.25 is a reasonable starting point.")
    ap.add_argument("--force",      action="store_true",
                    help="overwrite existing .txt files")
    args = ap.parse_args()

    import cv2

    os.makedirs(args.labels_dir, exist_ok=True)

    # Write classes.txt (labelImg / many tools require this)
    classes_path = os.path.join(os.path.dirname(args.labels_dir.rstrip("/")) or ".",
                                "classes.txt")
    with open(classes_path, "w") as f:
        for name in CLASS_ORDER:
            f.write(name + "\n")
    print(f"Wrote {classes_path}", file=sys.stderr)

    model = _load_model_lazy()
    if model is None:
        print("ERROR: HockeyAI model not available", file=sys.stderr); return 2

    # Build class-name → our-class-id map; HockeyAI's internal indices
    # may differ from our preferred order. We standardize on CLASS_ORDER.
    name_to_id = {name: i for i, name in enumerate(CLASS_ORDER)}

    images = sorted(Path(args.images_dir).glob("*.jpg"))
    print(f"Found {len(images)} images. conf threshold={args.conf}",
          file=sys.stderr)

    n_written = 0; n_skipped = 0; n_no_dets = 0
    for img_path in images:
        out_path = Path(args.labels_dir) / (img_path.stem + ".txt")
        if out_path.exists() and not args.force:
            n_skipped += 1; continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARN: cannot read {img_path}", file=sys.stderr); continue
        H, W = frame.shape[:2]

        results = model.predict(source=frame, conf=args.conf, verbose=False)
        lines = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_idx_yolo = int(boxes.cls[i].item())
                cls_name = model.names.get(cls_idx_yolo)
                if cls_name not in name_to_id:
                    continue
                our_id = name_to_id[cls_name]
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                # YOLO format: normalized center + size
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                # Clamp to [0, 1] defensively (some edge cases produce
                # bboxes slightly outside the frame)
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                lines.append(f"{our_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            n_no_dets += 1
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        n_written += 1
        if n_written % 100 == 0:
            print(f"  pre-labeled {n_written}/{len(images)}", file=sys.stderr)

    print(f"\nDone. wrote={n_written}  skipped_existing={n_skipped}  "
          f"frames_with_no_dets={n_no_dets}", file=sys.stderr)
    print(f"Class file: {classes_path}", file=sys.stderr)
    print(f"Labels dir: {args.labels_dir}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
