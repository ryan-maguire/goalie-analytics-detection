"""
detect_hockeyai.py

Standalone diagnostic that runs the SimulaMet-HOST HockeyAI pre-trained
YOLOv8 model (https://huggingface.co/SimulaMet-HOST/HockeyAI) against
sampled frames from each video. Writes annotated output for visual
inspection.

The model was fine-tuned on 2,101 frames from broadcast Swedish Hockey
League (SHL) footage and detects 7 classes:
  Center Ice
  Faceoff Dots
  Goal Frame      ← our target
  Goaltender      ← our target
  Players
  Puck
  Referee

The relevant question for our use: do "Goal Frame" and "Goaltender"
detections transfer to amateur arena footage with different camera
angles, lighting, and image quality? If yes, we have a viable
attribution signal that classical CV failed to provide.

This is purely diagnostic — model is downloaded, run, results
visualized. No integration with cv_seg.

Usage:
    python3 detect_hockeyai.py \\
        --video-dir data/videos \\
        --output-dir hockeyai_validation \\
        [--quick]      # 10 frames/video instead of 50
        [--vIDs ...]   # restrict to specific videos

Dependencies (install once):
    pip install ultralytics huggingface_hub
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

HF_REPO_ID = "SimulaMet-HOST/HockeyAI"
HF_MODEL_FILENAME = "HockeyAI_model_weight.pt"

# Class names from the HockeyAI model card. The model returns class
# indices; we map them to readable names. Order matters here — must
# match the order the model was trained with.
# Note: the model card lists classes alphabetically but YOLO assigns
# indices by training-time order. We'll discover the actual mapping
# from the loaded model and just use what it reports.
HOCKEYAI_CLASSES_OF_INTEREST = {
    "goal":     (0, 0, 255),     # red boxes — goal frame, primary target
    "goalie":   (0, 255, 0),     # green boxes — goaltender, primary target
    "player":   (255, 200, 0),   # cyan boxes — players, context
    # Other classes (faceoff, puck, referee, others) render in default
    # gray as low-priority context.
}
DEFAULT_COLOR = (200, 200, 200)


def load_model():
    """Download (if needed) and load the HockeyAI YOLOv8 model.

    Uses huggingface_hub to fetch the .pt file once; subsequent runs
    use the cached copy. Then loads with ultralytics.YOLO.
    """
    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: missing dependency. Install with:", file=sys.stderr)
        print(f"    pip install ultralytics huggingface_hub", file=sys.stderr)
        print(f"  ({e})", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading model from huggingface ({HF_REPO_ID})...", file=sys.stderr)
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME,
    )
    print(f"  cached at: {model_path}", file=sys.stderr)
    print(f"Loading YOLOv8 model...", file=sys.stderr)
    model = YOLO(model_path)
    print(f"  classes: {model.names}", file=sys.stderr)
    return model


# ---------------------------------------------------------------------------
# Inference + annotation
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_name: str
    side:       str        # 'left' / 'right' based on bbox center
    bbox:       tuple      # (x, y, w, h) in pixel coords
    confidence: float

    def to_dict(self):
        return asdict(self)


@dataclass
class FrameResult:
    frame_idx:     int
    timestamp_sec: float
    detections:    list = field(default_factory=list)
    n_per_class:   dict = field(default_factory=dict)


def run_inference(model, frame_bgr: np.ndarray, conf_threshold: float = 0.25) -> list[Detection]:
    """Run the model on a single frame; return Detection objects."""
    h, w = frame_bgr.shape[:2]

    # YOLOv8 expects RGB or BGR? ultralytics handles both via cv2 input
    # (auto-converts internally based on input source). Pass BGR directly.
    results = model.predict(
        source=frame_bgr,
        conf=conf_threshold,
        verbose=False,
    )
    if not results:
        return []

    res = results[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return []

    detections: list[Detection] = []
    for i in range(len(boxes)):
        cls_idx = int(boxes.cls[i].item())
        cls_name = model.names.get(cls_idx, f"class_{cls_idx}")
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        bx, by = int(x1), int(y1)
        bw, bh = int(x2 - x1), int(y2 - y1)
        cx = bx + bw / 2
        side = "left" if cx < w / 2 else "right"
        detections.append(Detection(
            class_name=cls_name, side=side,
            bbox=(bx, by, bw, bh), confidence=conf,
        ))
    return detections


def draw_detections(frame: np.ndarray, dets: list[Detection],
                    classes_to_show: list[str] | None = None) -> np.ndarray:
    """Annotate a frame with detection boxes. If classes_to_show is
    given, only those classes are drawn."""
    out = frame.copy()
    for d in dets:
        if classes_to_show and d.class_name not in classes_to_show:
            continue
        color = HOCKEYAI_CLASSES_OF_INTEREST.get(d.class_name, DEFAULT_COLOR)
        x, y, w, h = d.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{d.class_name[:4]} {d.confidence:.2f}"
        cv2.putText(out, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def make_panel(orig: np.ndarray, fr: FrameResult) -> np.ndarray:
    """Four-panel layout to highlight the targets we care about most.
    top-left: original
    top-right: all detections (shows what the model finds in general)
    bottom-left: Goal Frame + Goaltender only (the targets)
    bottom-right: Players only (context for crowd/density)
    """
    h, w = orig.shape[:2]
    target_w = 640
    scale = target_w / w
    target_h = int(h * scale)

    def resize(img):
        return cv2.resize(img, (target_w, target_h))

    def label_panel(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (320, 30), (0, 0, 0), -1)
        cv2.putText(out, text, (5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return out

    panels = [
        resize(label_panel(orig.copy(), f"original t={fr.timestamp_sec:.0f}s")),
        resize(label_panel(draw_detections(orig, fr.detections),
                           f"all classes ({len(fr.detections)})")),
        resize(label_panel(draw_detections(orig, fr.detections,
                                           classes_to_show=["goal", "goalie"]),
                           f"goal+goalie")),
        resize(label_panel(draw_detections(orig, fr.detections,
                                           classes_to_show=["player"]),
                           f"players only")),
    ]
    top = np.concatenate([panels[0], panels[1]], axis=1)
    bot = np.concatenate([panels[2], panels[3]], axis=1)
    return np.concatenate([top, bot], axis=0)


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def pick_sample_frames(duration_sec: float, n_samples: int) -> list[float]:
    return list(np.linspace(30, duration_sec - 30, n_samples).astype(float))


def process_video(model, video_path: str, vID: str,
                  output_dir: str, n_samples: int) -> dict:
    print(f"[{vID}] processing {video_path}", file=sys.stderr)
    out_dir = Path(output_dir) / vID
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"vID": vID, "status": "could_not_open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    print(f"[{vID}] fps={fps:.1f}, duration={duration:.0f}s, sampling {n_samples} frames",
          file=sys.stderr)

    timestamps = pick_sample_frames(duration, n_samples)
    results: list[FrameResult] = []
    t_start = time.time()

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue

        dets = run_inference(model, frame)
        fr = FrameResult(
            frame_idx=int(t * fps),
            timestamp_sec=t,
            detections=dets,
            n_per_class=dict(Counter(d.class_name for d in dets)),
        )
        results.append(fr)

        # Write annotated panel
        panel = make_panel(frame, fr)
        ts_int = int(t)
        cv2.imwrite(str(out_dir / f"frame_{ts_int:05d}.png"), panel,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])
        # Per-frame JSON
        with open(out_dir / f"frame_{ts_int:05d}.json", "w") as f:
            json.dump({
                "vID": vID,
                "timestamp_sec": t,
                "detections":   [d.to_dict() for d in fr.detections],
                "n_per_class":  fr.n_per_class,
            }, f, indent=2)

    cap.release()
    elapsed = time.time() - t_start

    # Aggregate stats
    total_per_class: Counter = Counter()
    frames_with_class: Counter = Counter()
    for r in results:
        for cls, n in r.n_per_class.items():
            total_per_class[cls] += n
            frames_with_class[cls] += 1
    summary = {
        "vID": vID,
        "duration_sec": duration,
        "n_samples": len(results),
        "elapsed_sec": round(elapsed, 1),
        "frames_with_goal":     frames_with_class.get("goal", 0),
        "frames_with_goalie":   frames_with_class.get("goalie", 0),
        "frames_with_player":   frames_with_class.get("player", 0),
        "total_goal_dets":      total_per_class.get("goal", 0),
        "total_goalie_dets":    total_per_class.get("goalie", 0),
        "total_player_dets":    total_per_class.get("player", 0),
    }
    print(f"[{vID}] done in {elapsed:.1f}s — "
          f"goal: {summary['frames_with_goal']}/{len(results)} frames "
          f"({summary['total_goal_dets']} dets); "
          f"goalie: {summary['frames_with_goalie']}/{len(results)} frames "
          f"({summary['total_goalie_dets']} dets)",
          file=sys.stderr)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video-dir", default="data/videos")
    p.add_argument("--output-dir", default="hockeyai_validation")
    p.add_argument("--vIDs", nargs="*", default=None)
    p.add_argument("--quick", action="store_true",
                   help="Sample 10 frames per video instead of 50")
    p.add_argument("--conf-threshold", type=float, default=0.25,
                   help="Confidence threshold for detections (default 0.25)")
    return p.parse_args()


def main():
    args = parse_args()
    n_samples = 10 if args.quick else 50

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        print(f"ERROR: {video_dir} not found", file=sys.stderr)
        sys.exit(1)

    if args.vIDs:
        vIDs = args.vIDs
    else:
        vIDs = [f.stem.removeprefix("full_") for f in sorted(video_dir.glob("full_*.mp4"))]
    if not vIDs:
        print(f"ERROR: no videos in {video_dir}", file=sys.stderr)
        sys.exit(1)

    # Load model once
    model = load_model()
    print(f"\nProcessing {len(vIDs)} videos: {vIDs}", file=sys.stderr)

    summaries = []
    for vID in vIDs:
        video_path = str(video_dir / f"full_{vID}.mp4")
        if not os.path.exists(video_path):
            print(f"  SKIP: {video_path} not found", file=sys.stderr)
            continue
        try:
            s = process_video(model, video_path, vID, args.output_dir, n_samples)
            summaries.append(s)
        except Exception as e:
            print(f"  [{vID}] failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    # Summary
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.txt", "w") as f:
        f.write("HockeyAI YOLOv8 detection diagnostic — summary\n")
        f.write(f"  generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"  samples per video: {n_samples}\n")
        f.write(f"  conf threshold: {args.conf_threshold}\n")
        f.write(f"  model: {HF_REPO_ID} / {HF_MODEL_FILENAME}\n")
        f.write("\n")
        f.write(f"  vIDs that detect 'goal' and/or 'goalie' frequently are\n")
        f.write(f"  candidates for net-based attribution. <50% detection rate\n")
        f.write(f"  on either suggests domain transfer from SHL broadcast to\n")
        f.write(f"  amateur footage isn't reliable for that arena.\n\n")
        f.write(f"  {'vID':<14} {'samples':<8} {'goal':<14} {'goalie':<14} {'player':<14}\n")
        f.write(f"  {'-'*14} {'-'*8} {'-'*14} {'-'*14} {'-'*14}\n")
        for s in summaries:
            n = s["n_samples"]
            g = s["frames_with_goal"]
            gt = s["frames_with_goalie"]
            pl = s["frames_with_player"]
            f.write(f"  {s['vID']:<14} {n:<8} "
                    f"{g}/{n} ({100*g/n:>3.0f}%)   "
                    f"{gt}/{n} ({100*gt/n:>3.0f}%)   "
                    f"{pl}/{n} ({100*pl/n:>3.0f}%)\n")
    print(f"\nWrote summary to {out_dir / 'summary.txt'}", file=sys.stderr)


if __name__ == "__main__":
    main()
