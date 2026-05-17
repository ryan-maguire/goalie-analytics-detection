"""
Snapshot: would shot-centering tighten predicted windows without hurting eval IoU?

This script does NOT modify cv_seg. It reads the existing eval JSON and the
per-second signals trace for each video, and simulates shrinking each TP
predicted window to a peak-centered ±N-second clip. It reports the
before/after IoU distribution so we can decide whether to actually ship the
shot-centering refinement to cv_seg.

What it answers:
  - If we shrink every TP window to peak-centered ±8s (clamped), what happens
    to mean IoU?
  - How many windows have a clear motion peak vs. uniform motion (fallback)?
  - Per-video: which videos benefit most, which (if any) get worse?

What it does NOT answer:
  - Whether shrinking changes FN/FP counts (that requires re-running matching,
    which is more involved). The snapshot only looks at existing TP pairs.
  - What the end-user clip actually looks like (that's a separate UX check).

Usage:
  python eval_snapshot_shot_centering.py \\
      --eval-json data/output/evals/eval_20260513T031821.json \\
      --signals-dir data/output/runs/cv_seg \\
      [--pre 8] [--post 8] [--peak-floor 4.0]

The signals dir must contain gt_seg_{vID}_signals.json files. If missing,
the script attempts a best-effort GCS download from
gs://goalie_video_bucket/analyze_video/01-segment_detection/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from statistics import mean, median
from typing import Optional

log = logging.getLogger("snapshot")
logging.basicConfig(
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ── Defaults — chosen to produce 16s target windows by default ────────────
DEFAULT_PRE_PEAK_SEC  = 8
DEFAULT_POST_PEAK_SEC = 8
# Below this motion intensity, treat the window as "no clear peak" and
# fall back to the original width. Tuned to MOTION_THRESH in cv_seg/constants.
DEFAULT_PEAK_FLOOR    = 4.0
# Don't produce a window shorter than this — matches cv_seg's MIN_THREAT_DUR.
MIN_OUTPUT_WIDTH      = 15

# GCS source for signals.json files if not on disk
GCS_BUCKET            = "goalie_video_bucket"
GCS_SIGNALS_PREFIX    = "analyze_video/01-segment_detection"


@dataclass
class SnapshotResult:
    vID: str
    matches:           int = 0
    fallback_no_peak:  int = 0          # peak below floor — kept original
    fallback_too_narrow: int = 0        # would shrink below MIN — kept original
    shrunk:            int = 0          # actually shrunk
    iou_before:        list[float] = None
    iou_after:         list[float] = None
    width_before:      list[float] = None
    width_after:       list[float] = None

    def __post_init__(self):
        if self.iou_before  is None: self.iou_before  = []
        if self.iou_after   is None: self.iou_after   = []
        if self.width_before is None: self.width_before = []
        if self.width_after  is None: self.width_after  = []


# ── Helpers ───────────────────────────────────────────────────────────────

def load_signals(vID: str, signals_dir: str,
                 allow_gcs_download: bool = True) -> Optional[list[dict]]:
    """Load gt_seg_{vID}_signals.json — per-second {motion, faceoff, activity, ...}.
    Returns the list of per-second dicts, or None if unavailable."""
    path = os.path.join(signals_dir, f"gt_seg_{vID}_signals.json")

    if not os.path.exists(path) and allow_gcs_download:
        try:
            from google.cloud import storage  # type: ignore
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob_name = f"{GCS_SIGNALS_PREFIX}/gt_seg_{vID}_signals.json"
            blob = bucket.blob(blob_name)
            if blob.exists():
                os.makedirs(signals_dir, exist_ok=True)
                blob.download_to_filename(path)
                log.info(f"  GCS: downloaded {blob_name}")
        except Exception as e:
            log.warning(f"  GCS download failed for {vID}: {e}")

    if not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"  Failed to read {path}: {e}")
        return None

    # cv_seg writes signals.json with the per-second data under the
    # "per_second" key (added in v23.10). Older signals.json files
    # without per_second can't be used by this snapshot — there's no
    # other place that motion-intensity time-series lives.
    if isinstance(data, dict) and "per_second" in data:
        return data["per_second"]
    if isinstance(data, dict) and "signals" in data:
        # Defensive: older / alternative naming
        return data["signals"]
    if isinstance(data, list):
        return data
    log.warning(
        f"  {path}: no per_second field. This needs cv_seg v23.10+ which "
        f"writes raw per-second signals into signals.json. Re-run cv_seg "
        f"for this vID, or ignore if all eval videos pre-date v23.10."
    )
    return None


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Standard IoU for two intervals on the same axis."""
    inter_start = max(a_start, b_start)
    inter_end   = min(a_end,   b_end)
    inter = max(0.0, inter_end - inter_start)
    if inter == 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


# ── Shot-locator strategies ──────────────────────────────────────────────
#
# Each strategy takes (signals, start_t, end_t) and returns
# (snap_time, signal_strength) where snap_time is the time we'd center
# on, and signal_strength is a value the caller can compare against
# `peak_floor` to decide whether the signal is strong enough to act on.

def _signals_in_range(signals: list[dict],
                      start_t: int, end_t: int) -> list[dict]:
    """Slice the per-second signals to those within [start_t, end_t]."""
    return [s for s in signals
            if "t" in s and start_t <= s["t"] <= end_t]


def locate_peak(signals: list[dict],
                start_t: int, end_t: int) -> tuple[Optional[int], float]:
    """Find time of maximum motion intensity in [start_t, end_t].
    Returns (peak_time, peak_value) or (None, 0)."""
    in_range = _signals_in_range(signals, start_t, end_t)
    if not in_range:
        return None, 0.0
    best = max(in_range, key=lambda s: s.get("motion", 0) or 0)
    return best["t"], float(best.get("motion", 0) or 0)


def locate_gradient(signals: list[dict],
                    start_t: int, end_t: int) -> tuple[Optional[int], float]:
    """Find time of steepest motion derivative (positive direction only).

    A shot moment is hypothesised to correlate with a sharp transition
    in motion intensity — puck snap, save, follow-through — rather than
    sustained high motion. argmax of d(motion)/dt should catch this
    better than argmax of motion itself.

    Returns (gradient_peak_time, gradient_magnitude). Magnitude can be
    compared to peak_floor / 2 (gradients are inherently smaller than
    absolute values) to decide if the signal is meaningful.
    """
    in_range = _signals_in_range(signals, start_t, end_t)
    if len(in_range) < 2:
        return None, 0.0
    # Sort by t to ensure correct derivative order
    in_range.sort(key=lambda s: s["t"])

    best_t   = None
    best_g   = -float("inf")
    for prev, curr in zip(in_range, in_range[1:]):
        g = (curr.get("motion", 0) or 0) - (prev.get("motion", 0) or 0)
        if g > best_g:
            best_g = g
            best_t = curr["t"]
    return best_t, max(0.0, best_g)


def locate_gradient_then_peak(signals: list[dict],
                              start_t: int, end_t: int,
                              search_radius: int = 3,
                              ) -> tuple[Optional[int], float]:
    """Find the steepest gradient, then snap to the highest motion
    within ±search_radius seconds of that gradient.

    Intuition: gradient says "something happened here", peak nearby
    says "the action centered HERE". This corrects for the case where
    the gradient fires on a transition into a sustained-motion zone —
    centering on the transition would miss the action.
    """
    grad_t, grad_g = locate_gradient(signals, start_t, end_t)
    if grad_t is None:
        return None, 0.0

    snap_lo = max(start_t, grad_t - search_radius)
    snap_hi = min(end_t,   grad_t + search_radius)
    peak_t, peak_v = locate_peak(signals, snap_lo, snap_hi)
    if peak_t is None:
        return grad_t, grad_g
    # Signal strength: report gradient magnitude (the trigger), not the
    # peak (we're using peak only for the snap location)
    return peak_t, grad_g


def locate_whistle_anchored(signals: list[dict],
                            start_t: int, end_t: int,
                            ) -> tuple[Optional[int], float]:
    """Snap to the time of the highest faceoff/whistle activity within
    the window. Note: whistles in cv_seg appear as elevated 'faceoff'
    scores around faceoff moments. End of play usually correlates with
    a whistle. Used as a comparison baseline for the other strategies.
    Returns (faceoff_peak_time, faceoff_strength)."""
    in_range = _signals_in_range(signals, start_t, end_t)
    if not in_range:
        return None, 0.0
    best = max(in_range, key=lambda s: s.get("faceoff", 0) or 0)
    return best["t"], float(best.get("faceoff", 0) or 0)


# Strategy registry — name → (locator_fn, default_floor)
# default_floor is a per-strategy minimum signal strength. Below this
# we fall back to the original window rather than acting on noise.
# Gradient floors are deliberately permissive — a synthetic test showed
# 2.0 cut off the legitimate ramp case. 1.0 is closer to "any meaningful
# upward motion change". Tune from real-data comparison results.
LOCATOR_STRATEGIES = {
    "peak":                (locate_peak,                4.0),
    "gradient":            (locate_gradient,            1.0),
    "gradient_then_peak":  (locate_gradient_then_peak,  1.0),
    "whistle":             (locate_whistle_anchored,    0.3),
}


def find_peak_motion(signals: list[dict],
                     start_t: int, end_t: int) -> tuple[Optional[int], float]:
    """Backward-compatible wrapper for the original peak strategy."""
    return locate_peak(signals, start_t, end_t)


def refine_window(
    orig_start: float, orig_end: float,
    signals: list[dict],
    pre_sec: int, post_sec: int, peak_floor: float,
    min_width: int,
    locator,
) -> tuple[float, float, str]:
    """Return (new_start, new_end, status) where status is one of
    'shrunk', 'fallback_no_peak', 'fallback_too_narrow'.

    `locator` is one of the functions in LOCATOR_STRATEGIES — it picks
    the snap time and reports the signal strength.
    """
    snap_t, signal_strength = locator(signals, int(orig_start), int(orig_end))

    if snap_t is None or signal_strength < peak_floor:
        return orig_start, orig_end, "fallback_no_peak"

    # Centre on snap, clamp to original boundaries (never expand)
    new_start = max(orig_start, snap_t - pre_sec)
    new_end   = min(orig_end,   snap_t + post_sec)

    if (new_end - new_start) < min_width:
        return orig_start, orig_end, "fallback_too_narrow"

    return new_start, new_end, "shrunk"


# ── Main analysis ─────────────────────────────────────────────────────────

def analyse(eval_json_path: str, signals_dir: str,
            pre: int, post: int, peak_floor: float,
            min_width: int, allow_gcs: bool,
            strategy: str = "peak",
            verbose: bool = True) -> dict:
    """Run the full snapshot analysis and return a structured result."""
    if strategy not in LOCATOR_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            f"Choose from: {sorted(LOCATOR_STRATEGIES)}"
        )
    locator, default_floor = LOCATOR_STRATEGIES[strategy]
    # If the caller didn't override peak_floor (passed the function default),
    # use the strategy's tuned default. Different strategies have very
    # different absolute scales (gradient ≪ raw motion intensity).
    effective_floor = peak_floor if peak_floor != DEFAULT_PEAK_FLOOR else default_floor

    with open(eval_json_path, encoding="utf-8") as f:
        eval_data = json.load(f)

    if verbose:
        log.info(f"Loaded eval from {eval_json_path}")
        log.info(f"  Strategy:      {strategy}")
        log.info(f"  Pre/Post peak: {pre}/{post}s  (target window: {pre+post}s)")
        log.info(f"  Signal floor:  {effective_floor}  (strategy default: {default_floor})")
        log.info(f"  Min output:    {min_width}s")
        log.info("")

    per_video: list[SnapshotResult] = []

    for v in eval_data.get("videos", []):
        if v.get("status") != "ok":
            continue
        vID = v["vID"]
        matches = [m for m in v.get("matches", [])
                   if m.get("gt_start") is not None
                   and m.get("pred_start") is not None]
        if not matches:
            continue

        signals = load_signals(vID, signals_dir, allow_gcs_download=allow_gcs)
        if signals is None:
            log.warning(f"  [{vID}] no signals — skipping ({len(matches)} TPs ignored)")
            continue

        result = SnapshotResult(vID=vID, matches=len(matches))

        for m in matches:
            gt_s, gt_e = m["gt_start"],   m["gt_end"]
            pr_s, pr_e = m["pred_start"], m["pred_end"]
            iou_before = iou(gt_s, gt_e, pr_s, pr_e)

            new_s, new_e, status = refine_window(
                pr_s, pr_e, signals, pre, post, effective_floor, min_width,
                locator=locator,
            )
            iou_after = iou(gt_s, gt_e, new_s, new_e)

            result.iou_before.append(iou_before)
            result.iou_after.append(iou_after)
            result.width_before.append(pr_e - pr_s)
            result.width_after.append(new_e - new_s)

            if status == "fallback_no_peak":      result.fallback_no_peak += 1
            elif status == "fallback_too_narrow": result.fallback_too_narrow += 1
            else:                                 result.shrunk += 1

        per_video.append(result)
        if verbose:
            log.info(
                f"  [{vID}] {result.matches:3d} TPs  "
                f"shrunk={result.shrunk:3d}  "
                f"no-peak={result.fallback_no_peak:3d}  "
                f"narrow={result.fallback_too_narrow:3d}  "
                f"IoU {mean(result.iou_before):.3f} → {mean(result.iou_after):.3f}  "
                f"width {median(result.width_before):.0f}s → {median(result.width_after):.0f}s"
            )

    # Aggregate
    all_iou_before    = [x for r in per_video for x in r.iou_before]
    all_iou_after     = [x for r in per_video for x in r.iou_after]
    all_width_before  = [x for r in per_video for x in r.width_before]
    all_width_after   = [x for r in per_video for x in r.width_after]
    total_matches     = sum(r.matches for r in per_video)
    total_shrunk      = sum(r.shrunk for r in per_video)
    total_no_peak     = sum(r.fallback_no_peak for r in per_video)
    total_too_narrow  = sum(r.fallback_too_narrow for r in per_video)

    # Track IoU loss/gain per match
    iou_improved = sum(1 for b, a in zip(all_iou_before, all_iou_after) if a > b + 0.01)
    iou_unchanged = sum(1 for b, a in zip(all_iou_before, all_iou_after) if abs(a - b) <= 0.01)
    iou_worsened = sum(1 for b, a in zip(all_iou_before, all_iou_after) if a < b - 0.01)

    # Critical question: how many matches that PASS the 0.2 IoU threshold
    # before still pass after? (i.e. how many TPs would we LOSE?)
    threshold = 0.2
    pass_before = sum(1 for x in all_iou_before if x >= threshold)
    pass_after  = sum(1 for x in all_iou_after  if x >= threshold)
    would_lose  = sum(1 for b, a in zip(all_iou_before, all_iou_after)
                      if b >= threshold and a < threshold)
    would_gain  = sum(1 for b, a in zip(all_iou_before, all_iou_after)
                      if b < threshold and a >= threshold)

    if verbose:
        print()
        print("=" * 70)
        print(f"AGGREGATE — strategy: {strategy}")
        print("=" * 70)
        print(f"Total TPs analysed:    {total_matches}")
    if total_matches == 0:
        if verbose:
            print()
            print("No TP matches found in any video. Either:")
            print("  - The eval JSON has no matched pairs (all FN / FP).")
            print("  - No signals.json files are available (need cv_seg v23.10+).")
            print("  - Check the WARNING messages above for per-video reasons.")
        return {"params": {"pre": pre, "post": post, "peak_floor": peak_floor,
                           "min_width": min_width},
                "totals": {"matches": 0}}
    if verbose:
        print(f"  shrunk:              {total_shrunk}  ({100*total_shrunk/total_matches:.1f}%)")
        print(f"  fallback no-peak:    {total_no_peak}  ({100*total_no_peak/total_matches:.1f}%)")
        print(f"  fallback too-narrow: {total_too_narrow}  ({100*total_too_narrow/total_matches:.1f}%)")
        print()
        print(f"Mean IoU:     {mean(all_iou_before):.4f} → {mean(all_iou_after):.4f}  "
              f"(Δ {mean(all_iou_after) - mean(all_iou_before):+.4f})")
        print(f"Median IoU:   {median(all_iou_before):.4f} → {median(all_iou_after):.4f}")
        print()
        print(f"Per-match IoU change:")
        print(f"  improved (+>0.01):  {iou_improved}  ({100*iou_improved/total_matches:.1f}%)")
        print(f"  unchanged (±0.01):  {iou_unchanged}  ({100*iou_unchanged/total_matches:.1f}%)")
        print(f"  worsened (<-0.01):  {iou_worsened}  ({100*iou_worsened/total_matches:.1f}%)")
        print()
        print(f"At IoU >= {threshold} threshold:")
        print(f"  passing before:     {pass_before}  ({100*pass_before/total_matches:.1f}%)")
        print(f"  passing after:      {pass_after}   ({100*pass_after/total_matches:.1f}%)")
        print(f"  would lose:         {would_lose}  ({100*would_lose/total_matches:.1f}%)")
        print(f"  would gain:         {would_gain}  ({100*would_gain/total_matches:.1f}%)")
        print()
        print(f"Mean window width:    {mean(all_width_before):.1f}s → {mean(all_width_after):.1f}s")
        print(f"Median window width:  {median(all_width_before):.1f}s → {median(all_width_after):.1f}s")
        print()
        print("Recommendation:")
        if mean(all_iou_after) >= mean(all_iou_before):
            print(f"  GREEN: mean IoU did not decrease.")
            print(f"  Shrinking is safe to ship — proceed to cv_seg implementation.")
        elif mean(all_iou_after) >= mean(all_iou_before) - 0.02:
            print(f"  YELLOW: mean IoU dropped {mean(all_iou_before)-mean(all_iou_after):.3f}.")
            print(f"  Marginal trade-off. Decide based on UX importance vs. metric loss.")
        else:
            print(f"  RED: mean IoU dropped {mean(all_iou_before)-mean(all_iou_after):.3f}.")
            print(f"  Don't ship without a different shot-locating strategy.")

    return {
        "strategy": strategy,
        "params": {"pre": pre, "post": post, "peak_floor": effective_floor,
                   "min_width": min_width},
        "totals": {
            "matches": total_matches, "shrunk": total_shrunk,
            "fallback_no_peak": total_no_peak,
            "fallback_too_narrow": total_too_narrow,
        },
        "iou": {
            "mean_before":   mean(all_iou_before),
            "mean_after":    mean(all_iou_after),
            "median_before": median(all_iou_before),
            "median_after":  median(all_iou_after),
            "improved": iou_improved, "unchanged": iou_unchanged,
            "worsened": iou_worsened,
        },
        "threshold": {
            "value": threshold,
            "pass_before": pass_before, "pass_after": pass_after,
            "would_lose": would_lose,   "would_gain": would_gain,
        },
        "width": {
            "mean_before":   mean(all_width_before),
            "mean_after":    mean(all_width_after),
            "median_before": median(all_width_before),
            "median_after":  median(all_width_after),
        },
        "per_video": [
            {
                "vID": r.vID, "matches": r.matches,
                "shrunk": r.shrunk,
                "fallback_no_peak": r.fallback_no_peak,
                "fallback_too_narrow": r.fallback_too_narrow,
                "iou_before_mean": mean(r.iou_before) if r.iou_before else 0,
                "iou_after_mean":  mean(r.iou_after)  if r.iou_after  else 0,
                "width_before_median": median(r.width_before) if r.width_before else 0,
                "width_after_median":  median(r.width_after)  if r.width_after  else 0,
            }
            for r in per_video
        ],
    }


def main():
    p = argparse.ArgumentParser(
        description="Snapshot: would shot-centering improve cv_seg eval IoU?"
    )
    p.add_argument("--eval-json", required=True,
                   help="Path to the cv_seg eval JSON (eval_*.json)")
    p.add_argument("--signals-dir",
                   default="data/output/runs/cv_seg",
                   help="Directory containing gt_seg_{vID}_signals.json")
    p.add_argument("--pre",  type=int, default=DEFAULT_PRE_PEAK_SEC,
                   help=f"Seconds before peak (default {DEFAULT_PRE_PEAK_SEC})")
    p.add_argument("--post", type=int, default=DEFAULT_POST_PEAK_SEC,
                   help=f"Seconds after peak (default {DEFAULT_POST_PEAK_SEC})")
    p.add_argument("--peak-floor", type=float, default=DEFAULT_PEAK_FLOOR,
                   help=f"Min motion to count as peak (default {DEFAULT_PEAK_FLOOR}). "
                        f"Strategy-specific defaults apply when this flag is left unset.")
    p.add_argument("--min-width", type=int, default=MIN_OUTPUT_WIDTH,
                   help=f"Min output window width (default {MIN_OUTPUT_WIDTH})")
    p.add_argument("--strategy", default="peak",
                   choices=sorted(LOCATOR_STRATEGIES.keys()),
                   help="Shot-locator strategy (default: peak). "
                        "'peak' = original. 'gradient' = steepest motion change. "
                        "'gradient_then_peak' = find gradient, snap to nearby peak. "
                        "'whistle' = anchor on faceoff signal.")
    p.add_argument("--compare", action="store_true",
                   help="Run all strategies and print a side-by-side comparison "
                        "table. Useful as the cheap test before committing to "
                        "cv_seg refinement.")
    p.add_argument("--no-gcs", action="store_true",
                   help="Skip GCS download of missing signals.json files")
    p.add_argument("--output", default=None,
                   help="If set, write the result JSON to this path")
    args = p.parse_args()

    if args.compare:
        # Run all strategies, suppress per-strategy verbose output,
        # print a comparison table.
        results = {}
        for name in ["peak", "gradient", "gradient_then_peak", "whistle"]:
            log.info(f"Running strategy: {name}")
            results[name] = analyse(
                eval_json_path=args.eval_json,
                signals_dir=args.signals_dir,
                pre=args.pre, post=args.post, peak_floor=args.peak_floor,
                min_width=args.min_width,
                allow_gcs=not args.no_gcs,
                strategy=name,
                verbose=False,
            )

        # Print the comparison
        print()
        print("=" * 82)
        print("STRATEGY COMPARISON")
        print("=" * 82)
        print(f"{'strategy':<22} {'IoU before':>10} {'IoU after':>9} {'Δ':>7}  "
              f"{'lose@.2':>8} {'gain@.2':>8} {'med width':>9}")
        for name, r in results.items():
            iou_b = r['iou']['mean_before']
            iou_a = r['iou']['mean_after']
            delta = iou_a - iou_b
            lose  = r['threshold']['would_lose']
            gain  = r['threshold']['would_gain']
            mw    = r['width']['median_after']
            verdict = ("GREEN " if delta >= 0
                       else "YELLOW" if delta >= -0.02
                       else "RED   ")
            print(f"  {name:<20} {iou_b:>10.3f} {iou_a:>9.3f} {delta:>+7.3f}  "
                  f"{lose:>8d} {gain:>8d} {mw:>8.0f}s  {verdict}")
        print()
        print("Interpretation:")
        print("  Δ        = mean IoU after - before (negative = degradation)")
        print("  lose@.2  = TPs falling below 0.2 IoU (= becoming FN in eval)")
        print("  gain@.2  = TPs newly passing 0.2 IoU")
        print("  GREEN  >= 0.000  ship-safe")
        print("  YELLOW >= -0.020 marginal, consider tradeoff")
        print("  RED    <  -0.020 don't ship")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            log.info(f"Comparison results written to {args.output}")
        return

    result = analyse(
        eval_json_path=args.eval_json,
        signals_dir=args.signals_dir,
        pre=args.pre, post=args.post, peak_floor=args.peak_floor,
        min_width=args.min_width,
        allow_gcs=not args.no_gcs,
        strategy=args.strategy,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        log.info(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
