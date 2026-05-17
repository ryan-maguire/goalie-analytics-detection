"""
Net detection — alternative attribution signal using a pre-trained
YOLOv8 hockey detection model (SimulaMet-HOST/HockeyAI on Hugging Face).

The model detects goal frames and goaltenders. When both classes co-occur
on the same side of the frame, that's a very high-confidence attribution
signal — independent of (and more direct than) motion asymmetry.

Lifecycle:
  - Model is downloaded lazily on first use (cached locally by
    huggingface_hub thereafter, ~52MB)
  - Model is loaded once per cv_seg run and reused across all windows
  - If `ultralytics` is not installed, the module returns None for
    every call and cv_seg falls back to motion attribution

Per-window cost: 5 frames × ~150ms inference = ~750ms per window.
For a typical 70-window video, that adds ~50s to a cv_seg run.

Validation history (5-video diagnostic):
  - On 4 of 5 amateur arena videos, when the model detects BOTH a goal
    AND a goalie in the same frame, they're on the same side 99% of
    the time (79/85 co-occurrences across the four videos).
  - The 5th video (n2cy8b755Tg) has wire mesh in front of the camera
    that produces spurious goal detections. We mitigate this by
    requiring confidence ≥ 0.50 AND multi-frame agreement.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from . import constants as C

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants — exposed for tuning
# ---------------------------------------------------------------------------

#: Hugging Face model identifier
NET_MODEL_REPO_ID = "SimulaMet-HOST/HockeyAI"
NET_MODEL_FILENAME = "HockeyAI_model_weight.pt"

#: Confidence threshold below which detections are discarded.
#: 0.50 was chosen from the 5-video diagnostic — at this threshold the
#: spurious detections on n2cy8b755Tg drop sharply (139 → ~30) while
#: legitimate detections on other videos retain median conf 0.70+.
NET_DET_CONF_THRESHOLD = 0.50

#: Number of frames sampled per threat window. Evenly spaced.
#: v23.10 (2026-05-15): bumped 5 → 10 to address low HockeyAI fire rate
#: on color-collision videos. Baseline measurement on HNG0jKYY12g:
#: HockeyAI fired on only 12 of 94 windows (13%) at frames-per-window=5,
#: leaving 87% to fall back to motion or last-assigned-color. Doubling
#: the frame sample widens the chance that ≥2 frames contain a co-occurring
#: goal+goalie detection that passes NET_DET_CONF_THRESHOLD.
#: Cost: ~50ms extra per window × 70 windows ≈ 3.5s extra per video.
#: Revert to 5 if this regresses non-collision videos.
NET_DET_FRAMES_PER_WINDOW = 10

#: Minimum number of frames in a window where goal+goalie co-occur on
#: the same side, before we'll override motion-asymmetry. Set to 2 to
#: avoid making decisions on a single frame's evidence (we saw 1/25
#: side disagreement on mjEeE7p2Hz8 in validation; multi-frame
#: agreement filters that out).
NET_DET_MIN_COOCCUR_FRAMES = 2

#: Minimum fraction of co-occurring frames that must agree on side.
#: Default 0.66 — supermajority. Below this we treat the window as
#: ambiguous and fall back to motion.
NET_DET_MAJORITY_FRACTION = 0.66

#: Class names exposed by the HockeyAI model. Lowercase singular —
#: discovered empirically (the model card lists them with different
#: casing, which caused a bug during the diagnostic phase).
CLS_GOAL   = "goal"
CLS_GOALIE = "goalie"
CLS_PLAYER = "player"


# ---------------------------------------------------------------------------
# Module-level cached model
# ---------------------------------------------------------------------------

_MODEL: Optional[object] = None
_MODEL_LOAD_FAILED = False  # sticky — don't retry every call


def _load_model_lazy():
    """Load the YOLOv8 model on first call. Cached for subsequent calls.

    Returns:
        The loaded YOLO model object, or None if the model can't be
        loaded (missing ultralytics package, network failure on first
        download, etc.).
    """
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_FAILED:
        return None

    try:
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError as e:
        log.warning(
            f"Net detection unavailable: missing dependency ({e}). "
            f"Install with: pip install ultralytics huggingface_hub. "
            f"Falling back to motion attribution."
        )
        _MODEL_LOAD_FAILED = True
        return None

    try:
        log.info(f"Loading HockeyAI model ({NET_MODEL_REPO_ID})...")
        model_path = hf_hub_download(
            repo_id=NET_MODEL_REPO_ID,
            filename=NET_MODEL_FILENAME,
        )
        _MODEL = YOLO(model_path)
        log.info(f"  model classes: {_MODEL.names}")
        return _MODEL
    except Exception as e:
        log.warning(
            f"Net detection unavailable: failed to load model ({e}). "
            f"Falling back to motion attribution."
        )
        _MODEL_LOAD_FAILED = True
        return None


def reset_model_cache():
    """Force re-load on next call. Used in tests."""
    global _MODEL, _MODEL_LOAD_FAILED
    _MODEL = None
    _MODEL_LOAD_FAILED = False


# ---------------------------------------------------------------------------
# Per-window detection
# ---------------------------------------------------------------------------

@dataclass
class FrameDetection:
    """Per-frame summary of what the model saw."""
    timestamp:   float
    goal_side:   Optional[str]   # 'left' / 'right' / None
    goalie_side: Optional[str]
    n_goals:   int = 0
    n_goalies: int = 0


@dataclass
class WindowAttribution:
    """Per-window aggregated attribution signal from net detection."""
    side:                Optional[str]   # 'left' / 'right' / None
    confidence_label:    str             # 'cooccur', 'goalie_only', 'goal_only', 'none'
    n_frames_sampled:    int
    n_cooccur_frames:    int             # frames with goal AND goalie
    n_goal_only_frames:  int
    n_goalie_only_frames: int

    def is_confident(self) -> bool:
        return self.side is not None


def _frame_top_side_for_class(detections, class_name: str, frame_w: int) -> Optional[str]:
    """Return the side of the highest-confidence detection of `class_name`
    in this frame, or None if no detection of that class."""
    matching = [d for d in detections if d["class_name"] == class_name]
    if not matching:
        return None
    best = max(matching, key=lambda d: d["confidence"])
    cx = best["bbox"][0] + best["bbox"][2] / 2
    return "left" if cx < frame_w / 2 else "right"


def _classify_frame(detections, frame_w: int) -> FrameDetection:
    """Reduce a list of YOLO detections to per-class side summaries."""
    n_goals   = sum(1 for d in detections if d["class_name"] == CLS_GOAL)
    n_goalies = sum(1 for d in detections if d["class_name"] == CLS_GOALIE)
    return FrameDetection(
        timestamp=0.0,  # filled in by caller
        goal_side=_frame_top_side_for_class(detections,   CLS_GOAL,   frame_w),
        goalie_side=_frame_top_side_for_class(detections, CLS_GOALIE, frame_w),
        n_goals=n_goals,
        n_goalies=n_goalies,
    )


def _aggregate_window(frames: list[FrameDetection]) -> WindowAttribution:
    """Combine per-frame detections into a single window-level decision.

    Decision priority:
      1) If there are >= MIN_COOCCUR co-occurrence frames AND a majority
         agree on side → confident attribution from co-occurrence.
      2) Else, if there are >= MIN_COOCCUR goalie-only frames AND a
         majority agree on side → moderate attribution from goalie alone.
      3) Else, if there are >= MIN_COOCCUR goal-only frames AND a
         majority agree on side → weaker attribution from goal alone.
      4) Else → no attribution signal; caller falls back to motion.
    """
    cooccur_sides:    list[str] = []
    goalie_only_sides: list[str] = []
    goal_only_sides:   list[str] = []

    for fr in frames:
        has_goal   = fr.goal_side   is not None
        has_goalie = fr.goalie_side is not None
        if has_goal and has_goalie and fr.goal_side == fr.goalie_side:
            cooccur_sides.append(fr.goal_side)
        elif has_goalie and not has_goal:
            goalie_only_sides.append(fr.goalie_side)
        elif has_goal and not has_goalie:
            goal_only_sides.append(fr.goal_side)
        # If goal and goalie disagree on side within the same frame, we
        # treat it as ambiguous and contribute neither.

    n_cooccur     = len(cooccur_sides)
    n_goalie_only = len(goalie_only_sides)
    n_goal_only   = len(goal_only_sides)

    def _majority_side(sides: list[str]) -> Optional[str]:
        if len(sides) < NET_DET_MIN_COOCCUR_FRAMES:
            return None
        counts = Counter(sides)
        winner, n_winner = counts.most_common(1)[0]
        if n_winner / len(sides) >= NET_DET_MAJORITY_FRACTION:
            return winner
        return None

    # Priority 1: co-occurrence
    side = _majority_side(cooccur_sides)
    if side:
        return WindowAttribution(
            side=side, confidence_label="cooccur",
            n_frames_sampled=len(frames),
            n_cooccur_frames=n_cooccur,
            n_goal_only_frames=n_goal_only,
            n_goalie_only_frames=n_goalie_only,
        )

    # Priority 2: goalie-only
    side = _majority_side(goalie_only_sides)
    if side:
        return WindowAttribution(
            side=side, confidence_label="goalie_only",
            n_frames_sampled=len(frames),
            n_cooccur_frames=n_cooccur,
            n_goal_only_frames=n_goal_only,
            n_goalie_only_frames=n_goalie_only,
        )

    # Priority 3: goal-only
    side = _majority_side(goal_only_sides)
    if side:
        return WindowAttribution(
            side=side, confidence_label="goal_only",
            n_frames_sampled=len(frames),
            n_cooccur_frames=n_cooccur,
            n_goal_only_frames=n_goal_only,
            n_goalie_only_frames=n_goalie_only,
        )

    return WindowAttribution(
        side=None, confidence_label="none",
        n_frames_sampled=len(frames),
        n_cooccur_frames=n_cooccur,
        n_goal_only_frames=n_goal_only,
        n_goalie_only_frames=n_goalie_only,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_attribution_signal(
    video_path:      str,
    t_start:         int,
    t_end:           int,
    n_frames:        int = NET_DET_FRAMES_PER_WINDOW,
    conf_threshold:  float = NET_DET_CONF_THRESHOLD,
) -> Optional[WindowAttribution]:
    """Sample frames from [t_start, t_end] and return the dominant
    side from net+goalie co-occurrence detections.

    Returns:
        WindowAttribution with side='left'/'right' if confident,
        WindowAttribution with side=None if no clear signal,
        None if the model couldn't be loaded (caller should fall back
        to motion attribution).

    The two None cases are distinct: model-not-loaded returns None
    (no signal possible at all), while ambiguous-detections returns
    a WindowAttribution with side=None (signal was tried but didn't
    converge).
    """
    if t_end <= t_start:
        return None

    model = _load_model_lazy()
    if model is None:
        return None

    if not os.path.exists(video_path):
        log.warning(f"net_detection: video not found: {video_path}")
        return None

    # Lazy import cv2 — keeps test imports fast
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        log.warning(f"net_detection: cv2/numpy not available ({e})")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"net_detection: could not open video: {video_path}")
        return None
    try:
        # Sample timestamps evenly within the window
        if n_frames < 1:
            n_frames = 1
        if n_frames == 1:
            timestamps = [(t_start + t_end) / 2]
        else:
            step = (t_end - t_start) / (n_frames + 1)
            timestamps = [t_start + step * (i + 1) for i in range(n_frames)]

        frames: list[FrameDetection] = []
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_h, frame_w = frame.shape[:2]
            results = model.predict(
                source=frame,
                conf=conf_threshold,
                verbose=False,
            )
            if not results:
                frames.append(FrameDetection(timestamp=t, goal_side=None,
                                              goalie_side=None))
                continue
            res = results[0]
            boxes = res.boxes
            dets: list[dict] = []
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_idx = int(boxes.cls[i].item())
                    cls_name = model.names.get(cls_idx, f"class_{cls_idx}")
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    dets.append({
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    })
            fd = _classify_frame(dets, frame_w)
            fd.timestamp = t
            frames.append(fd)
    finally:
        cap.release()

    if not frames:
        return None
    return _aggregate_window(frames)
