#!/usr/bin/env python3
"""Scoreboard OCR tracker — v1: broadcast overlay only.

Samples a video at 1 fps, OCRs the bottom strip looking for the
broadcast-style score overlay (e.g. "0 NW [icon] 5:51 2nd NAS 2"),
builds a time series of (t, home_score, away_score, period, clock),
detects score-change events, and emits goal-event windows with a
backward-search range of [T-60, T-20] seconds.

Physical LED scoreboards (the upper-corner kind found in most amateur
videos) are NOT handled in v1 — they're tiny in the frame and need
either a specialized 7-segment recognizer or super-resolution
preprocessing. Will be v2 work.

Outputs:
  data/output/scoreboard/{vID}/timeseries.json   — per-second OCR snapshots
  data/output/scoreboard/{vID}/goal_events.json  — detected score changes
  data/output/scoreboard/{vID}/recovery_seg.json — cv_seg-schema windows
                                                    spanning the lookback range
                                                    for each goal event

Usage:
  python3 tools/scoreboard_tracker.py \\
      --vID dwGsP6QKDs8 \\
      --customer-id CUST000031 \\
      --lookback-pre 60 --lookback-post 20
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VIDEOS_DIR = REPO / "data" / "videos"
OUT_BASE   = REPO / "data" / "output" / "scoreboard"

# Lazy imports — keeps --help fast
def _import_runtime():
    import cv2  # noqa
    import easyocr  # noqa
    return cv2, easyocr


# ─── Overlay parsing ─────────────────────────────────────────────────
# The broadcast overlay observed in dwGsP6QKDs8 / v0lxSTbXfw8 / hudl_2073810
# follows the layout: [score] [HOME_ABBR] [icon] [M:SS] [period] [AWAY_ABBR] [score]
#
# EasyOCR returns these as separate tokens. We don't care about exact
# layout fidelity — we just need to extract home_score, away_score,
# (optionally clock + period) and trust temporal smoothing to handle
# per-frame noise.

_CLOCK_RE  = re.compile(r"^\d{1,2}[:.]\d{2}$")     # 5:51 or 5.51 (OCR sometimes reads : as .)
_PERIOD_RE = re.compile(r"^(?:1st|2nd|3rd|OT|ot|SO|so)$", re.IGNORECASE)
_SCORE_RE  = re.compile(r"^\d{1,2}$")              # 0-99 from regex; capped at MAX_SCORE below
_TEAM_RE   = re.compile(r"^[A-Z]{2,5}[a-z]?$")     # NW, NAS, COLTIA, NAs (case noise)

# Max plausible hockey score per side per game. Any OCR result above
# this is almost certainly a misread (e.g. picking up shot-on-goal or
# penalty-minute totals which can hit 30+, OR misreading "2" as "29").
# Hockey games very rarely exceed 10 per side; 15 is a safe ceiling.
MAX_PLAUSIBLE_SCORE = 15


@dataclass
class OcrSnapshot:
    t_sec:        int
    home_score:   Optional[int] = None
    away_score:   Optional[int] = None
    clock:        Optional[str] = None
    period:       Optional[str] = None
    raw_tokens:   list[str] = None
    confidence:   float = 0.0     # mean of contributing token confidences

    def is_valid(self) -> bool:
        """At least one side parsed. Per-side independence matters because
        EasyOCR consistently misses the HOME score on dwGs/v0lxS overlays
        (leftmost digit on a blue background) while reliably catching AWAY
        (rightmost digit on a dark background). Tracking sides
        independently keeps every away-goal recovery even when home-side
        OCR is broken."""
        return self.home_score is not None or self.away_score is not None


def parse_overlay(detections: list[tuple],
                    crop_width: Optional[int] = None) -> tuple[Optional[int], Optional[int],
                                                                Optional[str], Optional[str],
                                                                list[str], float]:
    """Given EasyOCR detections [(bbox, text, conf), ...] from the bottom
    strip, extract home/away scores + clock + period.

    Position is computed as ABSOLUTE x-center of each bbox, normalized
    by `crop_width` (or by the rightmost detected x if crop_width is
    None). This is more robust than rank-position because OCR garbage
    tokens to the right of the real scoreboard shift index-based
    rankings.

    Returns (home, away, clock, period, raw_tokens, mean_conf).
    """
    if not detections:
        return None, None, None, None, [], 0.0

    def x_center(bbox):
        xs = [p[0] for p in bbox]
        return sum(xs) / 4
    def x_right(bbox):
        return max(p[0] for p in bbox)

    # If crop_width wasn't passed, infer from the rightmost detected x
    if crop_width is None:
        crop_width = max(x_right(b) for b, _, _ in detections) or 1

    ordered = sorted(detections, key=lambda d: x_center(d[0]))
    tokens   = [t for _, t, _ in ordered]
    raw_strs = [t.strip() for t in tokens]

    # Collect (relative_x in [0,1], value, conf) for each numeric token
    nums: list[tuple[float, int, float]] = []
    clock = period = None
    confs = []
    for bbox, text, conf in ordered:
        s = text.strip()
        confs.append(conf)
        if _CLOCK_RE.match(s):
            clock = s.replace(".", ":")
        elif _PERIOD_RE.match(s):
            period = s.lower()
        elif _SCORE_RE.match(s):
            rel_x = x_center(bbox) / crop_width
            nums.append((rel_x, int(s), conf))

    mean_conf = sum(confs) / len(confs) if confs else 0
    if not nums:
        return None, None, clock, period, raw_strs, mean_conf

    # Layout zones for a broadcast overlay:
    #   [home_score: 0.0-0.40] [HOME][icon][clock][period][AWAY][away_score: 0.60-1.0]
    # Filter to numerics in the left zone (home) and right zone (away).
    home_candidates = [(x, v, c) for (x, v, c) in nums if x < 0.40]
    away_candidates = [(x, v, c) for (x, v, c) in nums if x > 0.60]

    # Pick highest-confidence numeric per side, capped at MAX_PLAUSIBLE_SCORE.
    # Out-of-range values are almost certainly OCR misreads of stats
    # (shots, PIM) or single-digit misreads like 2→29.
    def _pick(cands):
        if not cands: return None
        best = max(cands, key=lambda t: t[2])
        return best[1] if 0 <= best[1] <= MAX_PLAUSIBLE_SCORE else None
    home_score = _pick(home_candidates)
    away_score = _pick(away_candidates)

    return home_score, away_score, clock, period, raw_strs, mean_conf


# ─── Video sampling ───────────────────────────────────────────────────

def sample_video_to_snapshots(video_path: Path, reader,
                                target_fps: float = 1.0,
                                crop_bottom_frac: float = 0.15,
                                start_sec: int = 60,
                                end_sec_offset: int = 60) -> list[OcrSnapshot]:
    """Walk the video at target_fps, OCR the bottom strip per frame,
    return per-second snapshots. Skips first start_sec (intro/warmup)
    and last end_sec_offset (post-game) to avoid noise."""
    cv2, _ = _import_runtime()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n_frames / fps
    print(f"  video: {dur:.0f}s @ {fps:.1f}fps  ({n_frames} frames)", file=sys.stderr)
    end_sec = int(dur - end_sec_offset)
    print(f"  sampling [{start_sec}, {end_sec}]s @ {target_fps:.1f} fps", file=sys.stderr)

    step_frames = int(round(fps / target_fps))
    snapshots: list[OcrSnapshot] = []

    t0 = time.time()
    last_log = t0
    frame_idx = 0
    sample_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step_frames != 0:
            frame_idx += 1; continue
        t_sec = int(frame_idx / fps)
        if t_sec < start_sec:
            frame_idx += 1; continue
        if t_sec > end_sec:
            break

        # Crop bottom strip where the overlay lives. Two passes:
        #   1. Full-width crop catches away score (rightmost, dark bg) +
        #      clock + period reliably.
        #   2. Left-half crop catches home score (leftmost, on a colored
        #      panel) which the full-width pass tends to drop because
        #      EasyOCR de-prioritizes edge-of-image characters.
        h, w = frame.shape[:2]
        full_crop = frame[int(h * (1 - crop_bottom_frac)):h, :]
        left_crop = frame[int(h * (1 - crop_bottom_frac)):h, :int(w * 0.30)]
        try:
            det_full = reader.readtext(full_crop)
            det_left = reader.readtext(left_crop)
        except Exception:
            det_full, det_left = [], []
        # Merge: use full-pass for away/clock/period; if home is None,
        # fall back to the leftmost numeric token in the left-pass.
        full_w  = full_crop.shape[1]
        left_w  = left_crop.shape[1]
        home, away, clock, period, tokens, conf = parse_overlay(det_full, crop_width=full_w)
        if home is None and det_left:
            # In the left-only crop, the home score now occupies the
            # ENTIRE width so it lands in the "left zone" trivially.
            # Pass left_w so the position calc is meaningful.
            home_l, _, _, _, _, _ = parse_overlay(det_left, crop_width=left_w)
            if home_l is not None:
                home = home_l
        snapshots.append(OcrSnapshot(
            t_sec=t_sec, home_score=home, away_score=away,
            clock=clock, period=period, raw_tokens=tokens, confidence=conf,
        ))
        sample_count += 1

        # Heartbeat every 30s of wall time
        if time.time() - last_log > 30:
            elapsed = time.time() - t0
            rate = sample_count / elapsed
            remaining = (end_sec - t_sec) / target_fps
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{t_sec:>5}s game / {sample_count:>4} samples]  "
                  f"valid={sum(1 for s in snapshots if s.is_valid()):>4}  "
                  f"rate={rate:.1f}/s  ETA={eta:.0f}s",
                  file=sys.stderr)
            last_log = time.time()
        frame_idx += 1

    cap.release()
    print(f"  done: {sample_count} samples in {time.time()-t0:.0f}s "
          f"({sum(1 for s in snapshots if s.is_valid())} valid)",
          file=sys.stderr)
    return snapshots


# ─── Score-change detection ───────────────────────────────────────────

def smoothed_scores(snapshots: list[OcrSnapshot],
                     window: int = 5,
                     min_consecutive: int = 3) -> list[tuple[int, Optional[int], Optional[int]]]:
    """Three-pass cleaning:

    Pass 1 — Rolling-window MEDIAN over `window` samples (per side).
             Absorbs single-frame OCR misreads like 0→7→0.

    Pass 2 — MONOTONIC NON-DECREASING constraint per side. Scores can
             only go up. If the median series tries to dip (e.g. from a
             stale OCR misread), keep the previous value. Crucial
             because the broadcast overlay can flicker between display
             modes and OCR can pick up stale or unrelated digits.

    Pass 3 — Forward-fill nulls so the score is defined at every
             timestamp once the first valid reading lands. Lets
             score-change detection see continuous values.

    A new score value is also required to appear in `min_consecutive`
    samples within a 10-sample neighborhood to be accepted — prevents
    a single-frame OCR spike (like "2" → "5") from triggering a fake
    +3-goal cascade.
    """
    out = []
    n = len(snapshots)
    half = window // 2

    # Pass 1: rolling median. Require >= MIN_QUORUM non-null values in
    # the window for the median to be considered meaningful. Otherwise
    # a single-frame OCR misread in a sparse neighborhood becomes the
    # "median" (median of [15] = 15 even though 15 is OCR noise).
    MIN_QUORUM = 3
    medians = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        homes = sorted([x.home_score for x in snapshots[lo:hi]
                         if x.home_score is not None])
        aways = sorted([x.away_score for x in snapshots[lo:hi]
                         if x.away_score is not None])
        h_med = homes[len(homes)//2] if len(homes) >= MIN_QUORUM else None
        a_med = aways[len(aways)//2] if len(aways) >= MIN_QUORUM else None
        medians.append((snapshots[i].t_sec, h_med, a_med))

    # Pass 2: monotonic non-decreasing, accept jumps up to MAX_STEP
    # provided they have sustained support. Larger jumps are rejected
    # as OCR noise.
    #
    # Why allow >+1: OCR has dropouts so we might miss a stretch where
    # the score went from 1 → 2 (never seeing "1"). When score then
    # appears as 2 with sustained support, we accept the jump and
    # emit goal events for each integer in between in detect_score_changes.
    #
    # Goals don't realistically come >3 apart in close succession in
    # the same OCR window, so MAX_STEP=4 catches almost all real cases
    # while rejecting 0→15 style misreads.
    LOOKAHEAD = 15
    MAX_STEP = 4
    def _monotonic_with_support(series: list[tuple[int, Optional[int], Optional[int]]],
                                  side: str) -> list[Optional[int]]:
        accepted: list[Optional[int]] = []
        cur: Optional[int] = None
        for i, row in enumerate(series):
            v = row[1] if side == "home" else row[2]
            if v is None:
                accepted.append(cur); continue
            if cur is None:
                # First valid reading. Be conservative: require sustained
                # support (multiple consistent readings) before locking
                # in the initial score — single-frame initial misreads
                # are the largest source of error.
                lookahead = [x[1] if side == "home" else x[2]
                              for x in series[i:i+LOOKAHEAD]]
                support = sum(1 for x in lookahead if x == v)
                if support >= min_consecutive:
                    cur = v
                accepted.append(cur); continue
            if v == cur:
                accepted.append(cur); continue
            if v < cur:
                # Score went DOWN — impossible, ignore
                accepted.append(cur); continue
            if v - cur > MAX_STEP:
                # Jump too large — almost certainly OCR misread (e.g.
                # 2 → 15 because Tesseract read shots-on-goal as score).
                accepted.append(cur); continue
            # v in (cur, cur+MAX_STEP]: candidate forward jump. Require
            # sustained support to commit.
            lookahead = [x[1] if side == "home" else x[2]
                          for x in series[i:i+LOOKAHEAD]]
            support = sum(1 for x in lookahead if x == v)
            if support >= min_consecutive:
                cur = v
            accepted.append(cur)
        return accepted

    home_filt = _monotonic_with_support(medians, "home")
    away_filt = _monotonic_with_support(medians, "away")
    return [(medians[i][0], home_filt[i], away_filt[i]) for i in range(n)]


@dataclass
class GoalEvent:
    detected_t_sec:    int      # time score change visible in OCR
    side:              str      # "home" or "away"
    score_before:      int
    score_after:       int
    lookback_start:    int      # search window for actual goal moment
    lookback_end:      int


def detect_score_changes(smoothed: list[tuple[int, Optional[int], Optional[int]]],
                           lookback_pre: int = 60,
                           lookback_post: int = 20) -> list[GoalEvent]:
    """Walk the smoothed series; emit a GoalEvent every time home or away
    score increments. Backward window: [T-lookback_pre, T-lookback_post]."""
    events: list[GoalEvent] = []
    prev_home = prev_away = None
    for (t, h, a) in smoothed:
        if h is not None:
            if prev_home is not None and h > prev_home:
                # Emit one event per unit increment (handles cases where
                # OCR misses several seconds and we see a +2 jump)
                for s in range(prev_home + 1, h + 1):
                    events.append(GoalEvent(
                        detected_t_sec=t, side="home",
                        score_before=s-1, score_after=s,
                        lookback_start=max(0, t - lookback_pre),
                        lookback_end=max(0, t - lookback_post),
                    ))
            prev_home = h
        if a is not None:
            if prev_away is not None and a > prev_away:
                for s in range(prev_away + 1, a + 1):
                    events.append(GoalEvent(
                        detected_t_sec=t, side="away",
                        score_before=s-1, score_after=s,
                        lookback_start=max(0, t - lookback_pre),
                        lookback_end=max(0, t - lookback_post),
                    ))
            prev_away = a
    return events


# ─── Threat-color lookup (matches run_fusion_pipeline.py) ─────────────

def load_threat_color(customer_id: str, vid: str) -> str:
    cust_json = REPO / "data" / "customers" / f"{customer_id}.json"
    if not cust_json.exists():
        return "Unknown"
    try:
        for rec in json.loads(cust_json.read_text()):
            if str(rec.get("vID")) == vid:
                return rec.get("targetGoalieColor") or "Unknown"
    except Exception:
        pass
    return "Unknown"


def goal_events_to_recovery_seg(vid: str, customer_id: str,
                                   events: list[GoalEvent]) -> list[dict]:
    """Convert detected goal events into cv_seg-format windows.
    Each window covers the [lookback_start, lookback_end] range so
    metrics_seg can analyze the moment the goal actually happened."""
    threat_color = load_threat_color(customer_id, vid)
    out = []
    for e in events:
        out.append({
            "segmentHasThreat":     True,
            "threat_goalie_color":  threat_color,
            "threat_goalie_side":   "unknown",
            "segment_start":        int(e.lookback_start),
            "segment_end":          int(e.lookback_end),
            "source_signals":       ["scoreboard_ocr"],
            "n_overlapping_raw":    1,
            "_scoreboard_event":    asdict(e),
        })
    return out


def video_path_for(vid: str) -> Optional[Path]:
    for cand in (VIDEOS_DIR / f"full_{vid}.mp4", VIDEOS_DIR / f"{vid}.mp4"):
        if cand.exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vID", required=True)
    ap.add_argument("--customer-id", required=True)
    ap.add_argument("--target-fps", type=float, default=1.0)
    ap.add_argument("--crop-bottom", type=float, default=0.15,
                     help="Fraction of frame height (from bottom) to OCR. 0.15 = bottom 15%%.")
    ap.add_argument("--min-consecutive", type=int, default=2,
                    help="Sustained-support count for accepting a score "
                         "value (out of LOOKAHEAD=15 samples). Default was "
                         "3 — empirically too strict on broadcast overlays "
                         "with brief flickers (validated F1=0.571 / R=0.40 "
                         "on dwGs+v0lxS). Default lowered to 2 on 2026-05-29.")
    ap.add_argument("--smooth-window", type=int, default=5,
                     help="Median filter window in seconds.")
    ap.add_argument("--lookback-pre",  type=int, default=180,
                    help="Backward search horizon: goal moment expected "
                         "between [t-pre, t-post] of the OCR score-change. "
                         "Default 180s matches the empirically validated "
                         "config on dwGs/v0lxS (F1=0.571 at mc=3, 0.667 at "
                         "mc=2 on dwGs). The previous 60s default was "
                         "aspirational, not measured.")
    ap.add_argument("--lookback-post", type=int, default=5,
                    help="See --lookback-pre. Default 5s matches validated config.")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="Default: data/output/scoreboard/<vID>/")
    args = ap.parse_args()

    out_dir = args.out_dir or (OUT_BASE / args.vID)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = video_path_for(args.vID)
    if not src:
        print(f"ERROR: no video file for {args.vID}", file=sys.stderr); sys.exit(1)
    print(f"[scoreboard_tracker] vID={args.vID}", file=sys.stderr)
    print(f"  video: {src}", file=sys.stderr)
    print(f"  output: {out_dir}", file=sys.stderr)

    _, easyocr = _import_runtime()
    print(f"  loading EasyOCR…", file=sys.stderr)
    t_load = time.time()
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print(f"  loaded in {time.time()-t_load:.0f}s", file=sys.stderr)

    snapshots = sample_video_to_snapshots(
        src, reader,
        target_fps=args.target_fps,
        crop_bottom_frac=args.crop_bottom,
    )

    # Persist raw time series
    ts_path = out_dir / "timeseries.json"
    ts_path.write_text(json.dumps([asdict(s) for s in snapshots], indent=2))
    print(f"  wrote {ts_path}  ({len(snapshots)} samples)", file=sys.stderr)

    # Smooth + detect events
    smoothed = smoothed_scores(snapshots, window=args.smooth_window,
                                  min_consecutive=args.min_consecutive)
    events   = detect_score_changes(smoothed,
                                      lookback_pre=args.lookback_pre,
                                      lookback_post=args.lookback_post)

    ev_path = out_dir / "goal_events.json"
    ev_path.write_text(json.dumps([asdict(e) for e in events], indent=2))
    print(f"  wrote {ev_path}  ({len(events)} goal events detected)", file=sys.stderr)

    # cv_seg-format recovery windows
    rec = goal_events_to_recovery_seg(args.vID, args.customer_id, events)
    rec_path = out_dir / "recovery_seg.json"
    rec_path.write_text(json.dumps(rec, indent=2))
    print(f"  wrote {rec_path}", file=sys.stderr)

    # Summary
    print(f"\n[summary]", file=sys.stderr)
    valid = [s for s in snapshots if s.is_valid()]
    print(f"  valid OCR snapshots: {len(valid)}/{len(snapshots)} "
          f"({len(valid)*100//max(len(snapshots),1)}%)", file=sys.stderr)
    if events:
        for e in events:
            print(f"  goal: side={e.side:5}  score {e.score_before}→{e.score_after}  "
                  f"detected at {e.detected_t_sec}s  "
                  f"lookback [{e.lookback_start}-{e.lookback_end}]s", file=sys.stderr)
    else:
        print(f"  no goal events detected", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
