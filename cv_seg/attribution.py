"""
Side detection and goalie colour attribution.

v23 attribution is motion-based: the side under attack has higher
sustained optical-flow energy. Colour sampling is only used at
game-start side detection (when no config override is supplied).
"""

from math import ceil
from typing import Optional

import cv2
import numpy as np

from . import constants as C
from .colors import is_light_jersey, jersey_color_to_hsv_range
from .logger import log


# ---------------------------------------------------------------------------
# Game-start side detection (CV)
# ---------------------------------------------------------------------------

def detect_goalie_sides_cv(
    video_path: str,
    goalie_color_a: str,
    goalie_color_b: str,
    probe_sec: int = 120,
) -> dict:
    """
    Detect which net each goalie defends by analysing the first
    probe_sec seconds of the video. Used only when no config override
    is supplied.

    Returns: {goalie_color_a: "left"|"right", goalie_color_b: "left"|"right"}
    """
    log.info("  CV side detection: sampling opening period...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    crease_top = int(h * C.GOAL_ROI_TOP_FRAC)
    crease_bot = int(h * C.GOAL_ROI_BOTTOM_FRAC)
    left_roi   = (0,                                crease_top, int(w * C.GOAL_ROI_SIDE_FRAC), crease_bot)
    right_roi  = (int(w * (1 - C.GOAL_ROI_SIDE_FRAC)), crease_top, w,                       crease_bot)

    hsv_range_a = jersey_color_to_hsv_range(goalie_color_a)
    hsv_range_b = jersey_color_to_hsv_range(goalie_color_b)

    sample_times = list(range(10, min(probe_sec, 90), 16))   # ~5 samples
    a_left_total = a_right_total = b_left_total = b_right_total = 0.0
    n_samples = 0

    cap = cv2.VideoCapture(video_path)
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t * fps)
        ret, frame = cap.read()
        if not ret:
            continue

        left_crop  = frame[left_roi[1]:left_roi[3],   left_roi[0]:left_roi[2]]
        right_crop = frame[right_roi[1]:right_roi[3], right_roi[0]:right_roi[2]]

        def _frac(roi, rng):
            if roi.size == 0: return 0.0
            hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, rng[0], rng[1])
            return float(cv2.countNonZero(mask)) / max(roi.shape[0] * roi.shape[1], 1)

        a_left_total  += _frac(left_crop,  hsv_range_a)
        a_right_total += _frac(right_crop, hsv_range_a)
        b_left_total  += _frac(left_crop,  hsv_range_b)
        b_right_total += _frac(right_crop, hsv_range_b)
        n_samples += 1
    cap.release()

    if n_samples == 0:
        log.warning("  Side detection: no frames sampled — defaulting to A=left, B=right")
        return {goalie_color_a: "left", goalie_color_b: "right"}

    a_left  = a_left_total  / n_samples
    a_right = a_right_total / n_samples
    b_left  = b_left_total  / n_samples
    b_right = b_right_total / n_samples

    log.info(f"  Colour A ({goalie_color_a[:20]}): left={a_left:.3f} right={a_right:.3f}")
    log.info(f"  Colour B ({goalie_color_b[:20]}): left={b_left:.3f} right={b_right:.3f}")

    if (abs(a_left - a_right) < C.SIDE_DETECTION_TIE_EPS
            and abs(b_left - b_right) < C.SIDE_DETECTION_TIE_EPS):
        a_side = "left" if is_light_jersey(goalie_color_a) else "right"
        b_side = "right" if a_side == "left" else "left"
        log.warning(f"  Side detection ambiguous — using brightness heuristic: "
                    f"A={a_side}, B={b_side}")
    else:
        a_side = "left" if a_left >= a_right else "right"
        b_side = "right" if a_side == "left" else "left"
        log.info(f"  Side detection result: A={a_side}, B={b_side}")

    return {goalie_color_a: a_side, goalie_color_b: b_side}


# ---------------------------------------------------------------------------
# Period side maps (config-driven, deterministic)
# ---------------------------------------------------------------------------

def detect_period_side_maps(
    video_path: str,
    goalie_color_a: str,
    goalie_color_b: str,
    duration: float,
    periods_config: Optional[list] = None,
    target_start_side: Optional[str] = None,
    opponent_start_side: Optional[str] = None,
    target_color: Optional[str] = None,
    opponent_color: Optional[str] = None,
) -> list[tuple[int, dict]]:
    """
    Build a list of (start_second, side_map) tuples so the caller can
    look up which goalie is on which side at any time during the game.

    Config-only, deterministic. Target goalie defends `target_start_side`
    in period 1; the side alternates every period thereafter. Opponent
    is always on the opposite side.

    LEAGUE-RULE ASSUMPTION: This function bakes in strict every-period
    alternation. That matches NHL/college regulation play but NOT
    every league or tournament format (some alternate every other
    period, some flip only between regulation and OT, etc.). If you
    need a different rule, supply the alternation pattern through the
    config rather than editing this function — the caller already
    passes periods_config which can be extended.
    """
    if not (periods_config and isinstance(periods_config, list)):
        log.warning("  No periods in config — cannot build period side map")
        return []
    if not target_start_side or not opponent_start_side:
        log.warning("  targetStartSide/opponentStartSide missing — cannot build period side map")
        return []
    if not target_color:
        log.warning("  targetGoalieColor missing — cannot build period side map")
        return []

    tss = target_start_side.strip().lower()
    oss = opponent_start_side.strip().lower()
    if tss not in ("left", "right") or oss not in ("left", "right"):
        log.warning(f"  Config sides invalid ({target_start_side}/{opponent_start_side}) — "
                    f"cannot build period side map")
        return []
    if tss == oss:
        log.warning(f"  Config sides are identical ({tss}) — cannot build period side map")
        return []

    target_is_a = (target_color == goalie_color_a)
    target_is_b = (target_color == goalie_color_b)
    if not (target_is_a or target_is_b):
        log.warning(f"  Target colour '{target_color}' matches neither "
                    f"A='{goalie_color_a}' nor B='{goalie_color_b}' — "
                    f"cannot build period side map")
        return []

    target_col_name   = goalie_color_a if target_is_a else goalie_color_b
    opponent_col_name = goalie_color_b if target_is_a else goalie_color_a

    sorted_periods = sorted(
        (p for p in periods_config if isinstance(p, dict)
         and p.get("start") is not None),
        key=lambda p: int(p["start"]),
    )
    if not sorted_periods:
        log.warning("  No usable period entries — cannot build period side map")
        return []

    result: list[tuple[int, dict]] = []
    for idx, p in enumerate(sorted_periods):
        p_start = int(p["start"])
        period_num = p.get("num", idx + 1)
        if idx % 2 == 0:
            tgt_side, opp_side = tss, oss
        else:
            tgt_side, opp_side = oss, tss
        side_map = {
            target_col_name:   tgt_side,
            opponent_col_name: opp_side,
        }
        result.append((p_start, side_map))
        log.info(f"  Period {period_num} side map (config) at {p_start}s: "
                 f"{target_col_name[:15]}={tgt_side}, "
                 f"{opponent_col_name[:15]}={opp_side}")

    if result and result[0][0] > 0:
        result.insert(0, (0, result[0][1]))
    return result


# ---------------------------------------------------------------------------
# Per-window goalie attribution (motion asymmetry)
# ---------------------------------------------------------------------------

def assign_goalie_colors(
    windows: list[dict],
    goalie_color_a: str,
    goalie_color_b: str,
    initial_side_map: dict,
    signals: list[dict],
    duration: float,
    period_side_maps: Optional[list[tuple[int, dict]]] = None,
    target_color: Optional[str] = None,
    video_path: Optional[str] = None,
    use_net_detection: bool = False,
) -> list[dict]:
    """
    For each candidate window, determine which goalie is being threatened
    and assign threat_goalie_color and threat_goalie_side.

    Attribution priority:
      1) Net detection (if use_net_detection=True and video_path provided):
         sample frames in the window, run HockeyAI YOLOv8 inference,
         and attribute based on goal+goalie co-occurrence sides.
      2) Motion asymmetry: whichever side of the rink has more optical
         flow during the window (+ pre-roll).
      3) Fallback ladder for ambiguous cases.

    The side → goalie colour mapping comes from the period-aware side map.

    Args:
        target_color: RESERVED — currently unused. An earlier
            implementation used it as the fallback default when motion
            gave no usable signal, but that approach regressed v0
            (offensive-end clips of the target team got mis-attributed
            to target colour). The parameter is retained for forward
            compatibility.
        video_path: Local path to the source video, required for net
            detection. Ignored if use_net_detection is False.
        use_net_detection: If True, attempt YOLOv8-based attribution
            from goal/goalie detections per window. Defaults to False
            so the v23.6.1 baseline behaviour is preserved.
    """
    from .side_map import side_map_at

    # Net-detection module is imported lazily so cv_seg still works
    # when ultralytics/huggingface_hub aren't installed.
    if use_net_detection and video_path:
        from . import net_detection
    else:
        net_detection = None

    dur_int = ceil(duration)
    threat_duration = {goalie_color_a: 0, goalie_color_b: 0}

    # Build dense numpy arrays indexed by integer second so we can slice
    # ranges instead of summing dict values inside a Python loop. Missing
    # seconds are zero-filled.
    if signals:
        max_t = max(int(s["t"]) for s in signals)
    else:
        max_t = -1
    n = max_t + 1
    motion_left_arr  = np.zeros(n, dtype=np.float32)
    motion_right_arr = np.zeros(n, dtype=np.float32)
    has_signal       = np.zeros(n, dtype=bool)
    for s in signals:
        ti = int(s["t"])
        if 0 <= ti < n:
            motion_left_arr[ti]  = s.get("motion_left",  0.0)
            motion_right_arr[ti] = s.get("motion_right", 0.0)
            has_signal[ti]       = True

    def _motion_asymmetry(ws: int, we: int) -> tuple[str, float, float, float]:
        """Return (attacked_side, mean_left, mean_right, ratio)."""
        t_start = max(0, ws - C.MOTION_ATTR_PRE_ROLL_SEC)
        t_end   = max(t_start, min(we, n))
        if t_end <= t_start:
            return "ambiguous", 0.0, 0.0, 1.0

        mask = has_signal[t_start:t_end]
        if not mask.any():
            return "ambiguous", 0.0, 0.0, 1.0
        l_slice = motion_left_arr[t_start:t_end][mask]
        r_slice = motion_right_arr[t_start:t_end][mask]
        mean_l = float(l_slice.mean())
        mean_r = float(r_slice.mean())

        diff = abs(mean_r - mean_l)
        if diff < C.MOTION_ATTR_ABS_FLOOR:
            return "ambiguous", mean_l, mean_r, 1.0

        stronger = max(mean_l, mean_r)
        weaker   = max(min(mean_l, mean_r), 1e-6)
        ratio    = stronger / weaker

        if ratio < C.MOTION_ATTR_RATIO:
            return "ambiguous", mean_l, mean_r, ratio

        side = "right" if mean_r > mean_l else "left"
        return side, mean_l, mean_r, ratio

    def _color_defending(side: str, sm: dict) -> str:
        if sm.get(goalie_color_a) == side:
            return goalie_color_a
        if sm.get(goalie_color_b) == side:
            return goalie_color_b
        return goalie_color_a

    assigned: list[dict] = []
    motion_decisions    = 0
    fallback_decisions  = 0
    net_decisions       = 0
    last_assigned_color: Optional[str] = None
    last_assigned_side:  Optional[str] = None

    for w in windows:
        ws, we = w["start"], min(w["end"], dur_int)
        if we <= ws:
            continue

        sm = side_map_at(ws, initial_side_map, period_side_maps)

        # Always compute the motion signal — used either as the primary
        # decision (no net detection) or as a fallback when net
        # detection has no confident answer.
        attacked_side, m_left, m_right, ratio = _motion_asymmetry(ws, we)

        # ── Priority 1: net detection (if enabled) ──────────────────
        net_attribution = None
        if net_detection is not None:
            try:
                net_attribution = net_detection.detect_attribution_signal(
                    video_path=video_path, t_start=ws, t_end=we,
                )
            except Exception as e:
                log.warning(f"net_detection failed for window [{ws}-{we}]: {e}")
                net_attribution = None

        if net_attribution is not None and net_attribution.is_confident():
            color = _color_defending(net_attribution.side, sm)
            side  = net_attribution.side
            attribution_src = f"net_{net_attribution.confidence_label}"
            net_decisions += 1

        # ── Priority 2: motion asymmetry ────────────────────────────
        elif attacked_side != "ambiguous":
            color = _color_defending(attacked_side, sm)
            side  = attacked_side
            motion_decisions += 1
            attribution_src = "motion"

        else:
            # ── Priority 3: ambiguous-motion fallback ladder ────────
            # See historical comments below — unchanged from v23.6.
            #   1) Hard-trigger source (goal_light / faceoff windows
            #      were placed *because* something happened on a
            #      specific side — but the source dict here doesn't
            #      record the side, so this is a no-op for now and
            #      kept as a comment for the future).
            #   2) Whichever side has any edge in average motion.
            #   3) Inherit from the previous attributed window — most
            #      sequences are bursts of pressure on the same end.
            #   4) Last-resort: goalie_color_a (deterministic).
            if m_right > m_left:
                color = _color_defending("right", sm)
                side  = "right"
                attribution_src = "fallback_motion_edge"
            elif m_left > m_right:
                color = _color_defending("left", sm)
                side  = "left"
                attribution_src = "fallback_motion_edge"
            elif last_assigned_color is not None and last_assigned_side is not None:
                color = last_assigned_color
                side  = sm.get(color, last_assigned_side)
                attribution_src = "fallback_inherit_prev"
            else:
                color = goalie_color_a
                side  = sm.get(color, "left")
                attribution_src = "fallback_default_a"
            fallback_decisions += 1

        # Build the assigned record, including net-detection diagnostics
        # if we ran net detection on this window.
        assignment = {
            "segmentHasThreat":    True,
            "threat_goalie_color": color,
            "threat_goalie_side":  side,
            "segment_start":       ws,
            "segment_end":         we,
            "_motion_left":        round(m_left, 3),
            "_motion_right":       round(m_right, 3),
            "_motion_ratio":       round(ratio, 2),
            "_attribution_src":    attribution_src,
        }
        if net_attribution is not None:
            assignment["_net_n_cooccur"]     = net_attribution.n_cooccur_frames
            assignment["_net_n_goal_only"]   = net_attribution.n_goal_only_frames
            assignment["_net_n_goalie_only"] = net_attribution.n_goalie_only_frames
            assignment["_net_n_sampled"]     = net_attribution.n_frames_sampled
            assignment["_net_label"]         = net_attribution.confidence_label

        assigned.append(assignment)
        threat_duration[color] += (we - ws)
        last_assigned_color = color
        last_assigned_side  = side

    if net_detection is not None:
        log.info(f"  Attribution: {net_decisions} net, {motion_decisions} motion, "
                 f"{fallback_decisions} fallback (of {len(assigned)} windows)")
    else:
        log.info(f"  Motion-based attribution: {motion_decisions} motion, "
                 f"{fallback_decisions} fallback (of {len(assigned)} windows)")

    return assigned
