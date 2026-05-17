"""
Per-frame signal extraction.

Each helper takes a BGR frame (and optionally a precomputed HSV array)
and returns a single signal value. The orchestrator computes HSV ONCE
per sampled frame and passes it into every helper that needs it,
avoiding 3× redundant cvtColor calls per second of video.
"""

import subprocess
from typing import Optional

import cv2
import numpy as np

from . import constants as C
from .io_utils import probe_video_dims, ffmpeg_available
from .logger import log


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------

def detect_red_light(
    frame: np.ndarray,
    hsv: Optional[np.ndarray] = None,
) -> float:
    """
    Detect goal light flash. Returns fraction of corner ROI pixels that
    are saturated red. A value above C.RED_LIGHT_THRESH indicates a light
    flash. Checks both top corners and returns the maximum.

    Args:
        frame: BGR frame (used for shape).
        hsv: Optional precomputed full-frame HSV. If supplied, both
            corner ROIs are sliced from it instead of running two extra
            cvtColor calls per frame.
    """
    h, w = frame.shape[:2]
    roi_h = max(1, int(h * C.RED_LIGHT_ROI_FRAC))
    roi_w = max(1, int(w * C.RED_LIGHT_ROI_FRAC))

    if hsv is not None:
        hsv_corners = [
            hsv[0:roi_h, 0:roi_w],
            hsv[0:roi_h, w - roi_w:w],
        ]
    else:
        hsv_corners = [
            cv2.cvtColor(frame[0:roi_h, 0:roi_w],         cv2.COLOR_BGR2HSV),
            cv2.cvtColor(frame[0:roi_h, w - roi_w:w],     cv2.COLOR_BGR2HSV),
        ]

    max_frac = 0.0
    for hsv_roi in hsv_corners:
        if hsv_roi.size == 0:
            continue
        # Red spans 0-10 and 170-180 in hue
        mask1 = cv2.inRange(hsv_roi, np.array([0, 150, 100]),   np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv_roi, np.array([170, 150, 100]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)
        roi_h_actual, roi_w_actual = hsv_roi.shape[:2]
        frac = red_pixels / max(roi_h_actual * roi_w_actual, 1)
        max_frac = max(max_frac, frac)

    return float(max_frac)


def detect_centre_faceoff(
    frame: np.ndarray,
    hsv: Optional[np.ndarray] = None,
) -> float:
    """
    Detect centre-ice faceoff. Returns a confidence score 0–1.

    Two-part test: (a) red centre circle via HoughCircles in the centre
    band, and (b) roughly equal player density in left vs right halves.

    Args:
        frame: BGR frame.
        hsv: Optional precomputed HSV. If None, computed here.
    """
    h, w = frame.shape[:2]
    if hsv is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ── Part 1: Red circle detection ────────────────────────────────────
    mask1 = cv2.inRange(hsv, np.array([0, 100, 80]),   np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)

    centre_band = red_mask.copy()
    centre_band[:h // 3, :] = 0
    centre_band[2 * h // 3:, :] = 0

    blurred = cv2.GaussianBlur(centre_band, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=w // 4,
        param1=50,
        param2=C.FACEOFF_CIRCLE_THRESH,
        minRadius=w // 20,
        maxRadius=w // 6,
    )
    has_red_circle = circles is not None

    # ── Part 2: Player density symmetry ─────────────────────────────────
    v_channel = hsv[:, :, 2]
    ice_thresh = np.percentile(v_channel, 60)
    player_mask = (v_channel < ice_thresh).astype(np.uint8)
    player_mask[:h // 5, :]     = 0
    player_mask[4 * h // 5:, :] = 0

    left_density  = player_mask[:, :w // 2].sum()
    right_density = player_mask[:, w // 2:].sum()
    total = left_density + right_density

    if total == 0:
        density_score = 0.0
    else:
        ratio = min(left_density, right_density) / max(left_density, right_density, 1)
        density_score = float(ratio)

    symmetric = density_score >= C.FACEOFF_DENSITY_RATIO

    if has_red_circle and symmetric:
        return 1.0
    if has_red_circle:
        return 0.6
    if symmetric:
        return 0.3
    return 0.0


def compute_motion_thirds(
    small_prev: Optional[np.ndarray],
    curr_gray: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray]:
    """
    Compute optical flow magnitude split into left / middle / right
    horizontal thirds plus the overall mean.

    Returns: (motion_total, motion_left, motion_mid, motion_right,
              small_curr) — the small-resized version of curr_gray is
    returned so the caller can cache it as the next iteration's
    small_prev, halving per-frame resize work.

    All four motion values are mean flow magnitude on the same vertical
    band (central 60% of frame height). The horizontal split is applied
    to the central 75% of frame width, divided into equal thirds. This
    is the primary signal for v23 window attribution.
    """
    h, w = curr_gray.shape
    small_curr = cv2.resize(curr_gray, (w // 2, h // 2))

    if small_prev is None:
        return 0.0, 0.0, 0.0, 0.0, small_curr

    flow = cv2.calcOpticalFlowFarneback(
        small_prev, small_curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    ch, cw = magnitude.shape
    y0, y1 = ch // 5, 4 * ch // 5
    x0, x1 = cw // 8, 7 * cw // 8
    centre = magnitude[y0:y1, x0:x1]

    third_w = centre.shape[1] // 3
    if third_w == 0:
        mean_total = float(np.mean(centre)) if centre.size else 0.0
        return mean_total, 0.0, 0.0, 0.0, small_curr

    left  = centre[:, :third_w]
    mid   = centre[:, third_w:2 * third_w]
    right = centre[:, 2 * third_w:]

    return (
        float(np.mean(centre)),
        float(np.mean(left)),
        float(np.mean(mid)),
        float(np.mean(right)),
        small_curr,
    )


def detect_scene_type(
    frame: np.ndarray,
    hsv: Optional[np.ndarray] = None,
) -> str:
    """
    Classify the shot as 'wide' or 'close'. Wide broadcast shots show
    one or two horizontal blue lines; close-ups show none.

    Args:
        frame: BGR frame (used for shape).
        hsv: Optional precomputed HSV.
    """
    h, w = frame.shape[:2]
    if hsv is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([140, 255, 255]))
    centre_w = blue_mask[:, w // 4: 3 * w // 4]
    edges = cv2.Canny(centre_w, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(w * 0.15),
        minLineLength=int(w * 0.10),
        maxLineGap=20,
    )
    if lines is None:
        return "close"

    horizontal = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 10 or angle > 170:
            horizontal += 1

    return "wide" if horizontal >= 1 else "close"


def measure_bench_activity(frame: np.ndarray) -> float:
    """
    Measure activity in the bench/crowd ROI (top strip of frame).
    Returns mean V-channel value normalised to 0–1.
    """
    h, w = frame.shape[:2]
    roi_h = max(1, int(h * C.ACTIVITY_ROI_FRAC))
    roi   = frame[:roi_h, w // 4: 3 * w // 4]
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2])) / 255.0


def detect_celebration_clustering(
    frame: np.ndarray,
    motion_score: float,
    hsv: Optional[np.ndarray] = None,
) -> tuple[float, str]:
    """
    Detect goal celebration via asymmetric player clustering.

    Returns (score, side) where score is 0–1 confidence and side is
    'left', 'right', or 'none'. side indicates which net was just
    scored on.

    Args:
        frame: BGR frame (used for shape).
        motion_score: Motion score for the same frame; sub-threshold
            motion short-circuits before any HSV work.
        hsv: Optional precomputed HSV.
    """
    if motion_score < C.CELEBRATION_MOTION_MIN:
        return 0.0, "none"

    h, w = frame.shape[:2]
    if hsv is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ice_top    = int(h * 0.25)
    ice_bottom = int(h * 0.80)
    ice_region = hsv[ice_top:ice_bottom, :, 2]

    ice_thresh   = np.percentile(ice_region, 65)
    player_mask  = (ice_region < ice_thresh).astype(np.uint8)

    mid_x        = w // 2
    left_region  = player_mask[:, :mid_x]
    right_region = player_mask[:, mid_x:]

    left_density  = left_region.sum()  / max(left_region.size, 1)
    right_density = right_region.sum() / max(right_region.size, 1)

    if max(left_density, right_density) < C.CELEBRATION_MIN_DENSITY:
        return 0.0, "none"

    if left_density == 0 and right_density == 0:
        return 0.0, "none"
    dominant  = max(left_density, right_density)
    recessive = min(left_density, right_density)
    # Floor recessive at MIN_DENSITY/4 so a near-empty half doesn't make
    # the ratio explode just because the dominant side cleared the
    # threshold by a hair. The fixed 1e-6 floor previously here let
    # tiny densities (e.g. 0.0005 vs 0.04) score full confidence.
    recessive_floor = C.CELEBRATION_MIN_DENSITY / 4
    ratio = dominant / max(recessive, recessive_floor)

    if ratio < C.CELEBRATION_DENSITY_RATIO:
        return 0.0, "none"

    score = min(1.0, (ratio - C.CELEBRATION_DENSITY_RATIO) /
                     (C.CELEBRATION_DENSITY_RATIO * 2))
    side  = "left" if left_density >= right_density else "right"
    return float(score), side


# ---------------------------------------------------------------------------
# Frame iteration: ffmpeg-pipe (fast path) and OpenCV decode loop (fallback)
# ---------------------------------------------------------------------------

def iter_frames_ffmpeg(video_path: str, sample_fps: int, w: int, h: int):
    """
    Stream BGR frames from ffmpeg at exactly `sample_fps`, yielding
    numpy arrays of shape (h, w, 3). One ffmpeg invocation, one decode
    pass — significantly faster than OpenCV's "read every frame and
    skip most" loop on high-fps source videos.
    """
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", video_path,
        "-vf", f"fps={sample_fps}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    frame_size = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=frame_size * 4)
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _extract_signal_one_frame(
    frame: np.ndarray,
    small_prev: Optional[np.ndarray],
    t: int,
) -> tuple[dict, np.ndarray]:
    """
    Run all per-frame signal extractors on a single BGR frame and
    return (signal_dict, small_curr_for_next_iteration). HSV is
    computed ONCE here and passed into every helper that needs it.

    detect_red_light removed in v23.5 (no goal lights on amateur
    rinks); confirmed not coming back.
    measure_bench_activity removed in v23.5 then restored in v23.6 —
    activity-spike events still aren't standalone predictions, but
    they're useful as co-confirmers for motion_auto_close windows.
    See windows.py for the consumer logic.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    faceoff   = detect_centre_faceoff(frame, hsv=hsv)
    motion, motion_left, motion_mid, motion_right, small_curr = compute_motion_thirds(
        small_prev, gray
    )
    scene    = detect_scene_type(frame, hsv=hsv)
    activity = measure_bench_activity(frame)
    celeb_score, celeb_side = detect_celebration_clustering(frame, motion, hsv=hsv)

    signal = {
        "t":            t,
        # red_light removed in v23.5 — don't restore.
        "faceoff":      faceoff,
        "motion":       motion,
        "motion_left":  motion_left,
        "motion_mid":   motion_mid,
        "motion_right": motion_right,
        "scene":        scene,
        "activity":     activity,
        "celeb":        celeb_score,
        "celeb_side":   celeb_side,
    }
    return signal, small_curr


def extract_frame_signals(
    video_path: str,
    sample_fps: int = 1,
) -> tuple[list[dict], float]:
    """
    Sample video at sample_fps and extract per-second signal vectors.

    Prefers an ffmpeg-based frame stream (one decode pass at the target
    rate), and falls back to OpenCV's per-frame read+skip loop if
    ffmpeg is unavailable or its pipe fails.

    NOTE: The downstream pipeline (windows.py, attribution.py,
    postprocess.py) assumes signal["t"] is an integer wall-clock
    second. That holds only when sample_fps == 1; other values would
    break range-based motion sampling and threshold tuning. We enforce
    that here rather than silently mis-time the rest of the pipeline.
    """
    if sample_fps != 1:
        raise ValueError(
            f"extract_frame_signals currently requires sample_fps=1 "
            f"(downstream pipeline assumes integer-second timestamps); got {sample_fps}"
        )

    w, h, native_fps, duration = probe_video_dims(video_path)
    log.info(f"  Video: {duration:.0f}s, {native_fps:.1f}fps, "
             f"sampling at {sample_fps}fps ({w}x{h})")

    use_ffmpeg = ffmpeg_available()
    signals: list[dict] = []
    small_prev: Optional[np.ndarray] = None

    if use_ffmpeg:
        log.info("  Using ffmpeg pipe for frame extraction (fast path)")
        try:
            for t, frame in enumerate(iter_frames_ffmpeg(video_path, sample_fps, w, h)):
                signal, small_prev = _extract_signal_one_frame(frame, small_prev, t)
                signals.append(signal)
                if (t + 1) % 60 == 0:
                    log.info(f"    ... {t + 1}s / {duration:.0f}s processed")
        except Exception as e:
            log.warning(f"  ffmpeg frame stream failed: {e} — "
                        f"falling back to OpenCV decode loop")
            signals = []
            small_prev = None
            use_ffmpeg = False

    if not use_ffmpeg:
        log.info("  Using OpenCV decode loop for frame extraction (fallback)")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        try:
            frame_step = max(1, int(native_fps / sample_fps))
            frame_idx = 0
            t = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step == 0:
                    signal, small_prev = _extract_signal_one_frame(frame, small_prev, t)
                    signals.append(signal)
                    t += 1
                frame_idx += 1
                if frame_idx > 0 and frame_idx % (int(native_fps) * 60) == 0:
                    log.info(f"    ... {t}s / {duration:.0f}s processed")
        finally:
            cap.release()

    log.info(f"  Frame extraction complete: {len(signals)} samples")
    return signals, duration
