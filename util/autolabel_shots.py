"""
Auto-label `shot` bboxes on positive frames using the goal-class
detections that prelabel_frames.py already wrote.

For each *_pos_*.txt file that doesn't already have a class-7 entry:
  1. Parse existing YOLO detections
  2. Pick the highest-confidence goal bbox (class 2). Falls back to
     looking at the .jpg via fresh HockeyAI inference if no goal is in
     the pre-label (sometimes the pre-label conf=0.25 threshold
     filtered out a low-but-present goal).
  3. Expand it: width × W_FACTOR, height × H_FACTOR, centered on the
     goal center. This is the shot-area bbox.
  4. Clamp to [0, 1] and write a new line `7 cx cy w h` appended to
     the .txt file.

Two modes:
  --preview <dir>   Render annotated preview JPEGs into <dir> so you
                    can spot-check before committing. Does NOT modify
                    any .txt files. Recommended first pass.
  --commit          Actually append the class-7 line to each .txt.
                    Skips files that already have a class-7 entry.

Usage:
    # 1. Generate previews for ONE video to validate
    python3 util/autolabel_shots.py --preview data/labels/_preview \\
        --filter bfEKgtOIkQU

    # 2. If previews look right, commit for that video
    python3 util/autolabel_shots.py --commit --filter bfEKgtOIkQU

    # 3. If happy with the approach, run on all videos
    python3 util/autolabel_shots.py --commit
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Class ID conventions (must match prelabel_frames.py CLASS_ORDER)
CLS_GOAL_ID = 2
CLS_SHOT_ID = 7

# Default expansion factors. Overridable via --w-factor / --h-factor.
# History: 1.8 × 1.5 was the original tuning. Prior approach-2 retrain
# (test F1 0.232) showed it was too generous — the model learned to
# fire on any goal+player vicinity. Recommended tighter defaults for
# retraining: --w-factor 1.0 --h-factor 0.8 (covers goalmouth + slot
# but not the whole offensive zone).
W_FACTOR = 1.8
H_FACTOR = 1.5


def _parse_yolo_txt(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Return [(class_id, cx, cy, w, h)] from a YOLO label file."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            continue
        out.append((cid, cx, cy, w, h))
    return out


def _pick_goal(dets: list[tuple]) -> tuple | None:
    """Return the first goal bbox (or None). YOLO label format doesn't
    store conf, so 'first' = the order HockeyAI wrote them, which is
    confidence-descending."""
    for d in dets:
        if d[0] == CLS_GOAL_ID:
            return d
    return None


def _expand_bbox(cx, cy, w, h, w_factor, h_factor) -> tuple[float, float, float, float]:
    nw = min(1.0, w * w_factor)
    nh = min(1.0, h * h_factor)
    # Re-center, clamping so the bbox stays inside [0,1]
    ncx = max(nw / 2, min(1.0 - nw / 2, cx))
    ncy = max(nh / 2, min(1.0 - nh / 2, cy))
    return ncx, ncy, nw, nh


def _yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return int(x1), int(y1), int(x2), int(y2)


def main():
    global W_FACTOR, H_FACTOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/labels/images")
    ap.add_argument("--labels-dir", default="data/labels/labels")
    ap.add_argument("--preview",    default=None,
                    help="dir to write annotated preview JPEGs (validation mode)")
    ap.add_argument("--commit",     action="store_true",
                    help="actually append class-7 lines to .txt files")
    ap.add_argument("--filter",     default=None,
                    help="restrict to .jpg/.txt files whose name contains this "
                         "substring (e.g. a vID for single-video validation)")
    ap.add_argument("--fallback-yolo", action="store_true",
                    help="if pre-label has no goal, run HockeyAI on the .jpg "
                         "with conf=0.05 to try to find one (slower)")
    ap.add_argument("--w-factor", type=float, default=W_FACTOR,
                    help=f"width-expansion factor for shot bbox (default {W_FACTOR})")
    ap.add_argument("--h-factor", type=float, default=H_FACTOR,
                    help=f"height-expansion factor for shot bbox (default {H_FACTOR})")
    args = ap.parse_args()

    # Override the module-level defaults so the rest of main() picks them up
    W_FACTOR = args.w_factor
    H_FACTOR = args.h_factor

    if not args.preview and not args.commit:
        print("ERROR: pick at least one of --preview or --commit", file=sys.stderr)
        return 2

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    preview_dir = Path(args.preview) if args.preview else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # Find positive frames
    pos_imgs = sorted(images_dir.glob("*_pos_*.jpg"))
    if args.filter:
        pos_imgs = [p for p in pos_imgs if args.filter in p.name]
    print(f"Found {len(pos_imgs)} positive .jpg files"
          f"{' (filtered)' if args.filter else ''}", file=sys.stderr)

    # Lazy imports
    if preview_dir or args.fallback_yolo:
        import cv2
    if args.fallback_yolo:
        from cv_seg.net_detection import _load_model_lazy
        model = _load_model_lazy()
        if model is None:
            print("ERROR: --fallback-yolo set but HockeyAI not available",
                  file=sys.stderr); return 2

    n_autolabeled = 0
    n_already_labeled = 0
    n_no_goal = 0
    n_fallback_hit = 0
    for img_path in pos_imgs:
        txt_path = labels_dir / (img_path.stem + ".txt")
        dets = _parse_yolo_txt(txt_path)
        if any(d[0] == CLS_SHOT_ID for d in dets):
            n_already_labeled += 1
            continue
        goal = _pick_goal(dets)

        # Fallback: re-run YOLO with lower conf
        if goal is None and args.fallback_yolo:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            H, W = frame.shape[:2]
            results = model.predict(source=frame, conf=0.05, verbose=False)
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best = None
                best_conf = -1.0
                for i in range(len(boxes)):
                    cls_idx = int(boxes.cls[i].item())
                    name = model.names.get(cls_idx, "")
                    if name != "goal":
                        continue
                    conf = float(boxes.conf[i].item())
                    if conf > best_conf:
                        best_conf = conf
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                        cx = ((x1 + x2) / 2) / W
                        cy = ((y1 + y2) / 2) / H
                        bw = (x2 - x1) / W
                        bh = (y2 - y1) / H
                        best = (CLS_GOAL_ID, cx, cy, bw, bh)
                if best:
                    goal = best
                    n_fallback_hit += 1

        if goal is None:
            n_no_goal += 1
            continue

        _, gcx, gcy, gw, gh = goal
        scx, scy, sw, sh = _expand_bbox(gcx, gcy, gw, gh, W_FACTOR, H_FACTOR)
        shot_line = f"{CLS_SHOT_ID} {scx:.6f} {scy:.6f} {sw:.6f} {sh:.6f}"

        if args.commit:
            with open(txt_path, "a") as f:
                # Ensure we don't append onto the same line
                existing = txt_path.read_text()
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(shot_line + "\n")

        if preview_dir:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            H, W = frame.shape[:2]
            # Draw goal bbox in green, shot bbox in red
            gx1, gy1, gx2, gy2 = _yolo_to_xyxy(gcx, gcy, gw, gh, W, H)
            sx1, sy1, sx2, sy2 = _yolo_to_xyxy(scx, scy, sw, sh, W, H)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
            cv2.putText(frame, "goal (green) -> shot (red)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            out_path = preview_dir / img_path.name
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        n_autolabeled += 1

    print(f"\nResults:")
    print(f"  auto-labeled (new shot bbox written/previewed): {n_autolabeled}")
    print(f"  already had class-7 entry (skipped):            {n_already_labeled}")
    print(f"  no goal detection in pre-label (skipped):       {n_no_goal}")
    if args.fallback_yolo:
        print(f"  ↳ of those, rescued by fallback YOLO:           {n_fallback_hit}")
    if preview_dir:
        print(f"  preview JPEGs at: {preview_dir}")
    if args.commit:
        print(f"  COMMITTED to .txt files at: {labels_dir}")
    else:
        print(f"  DRY RUN — no .txt files modified. Re-run with --commit when ready.")


if __name__ == "__main__":
    sys.exit(main() or 0)
