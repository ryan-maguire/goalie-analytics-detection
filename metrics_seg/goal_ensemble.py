"""Multi-call goal-confirmation ensemble.

v10's goal classifier has intrinsic precision 0.875 but production
STRICT precision sits at 0.625 because:
  - Single-call FPs sneak through
  - cv_seg attribution errors compound

Goals are rare (~2-6 per game), so each FP/FN drags F1 hard. Trading
3× Gemini cost on the ~5% of windows with predicted goals is a
favorable swap. This module:

  1. Takes the first-call result + the video bytes + a callable for
     additional Gemini calls
  2. Fires 2 extra calls with diversified temperatures
  3. Optionally validates against the fused per-second probs (does
     the window contain a sustained signal peak?)
  4. Returns the corroborated result (goals kept, downgraded, or
     unchanged depending on votes + signal)

It builds on the existing v8 multi-call vote infrastructure in
`analyze_clip_metrics` — this module is the "if first call says
goal, go deeper" branch with the extra prob-signal veto.

Default: disabled (feature flag --goal-ensemble).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("metrics_seg.goal_ensemble")


# Tunable per the spec
ENSEMBLE_EXTRA_CALLS_TEMPS = (0.0, 0.3)   # 2 extras at these temps
PROB_VETO_PEAK_MIN         = 0.50          # peak prob threshold for veto
PROB_VETO_SUSTAINED_SEC    = 3             # peak must be sustained N consecutive secs


@dataclass
class EnsembleTrace:
    triggered:        bool = False
    extras_attempted: int  = 0
    extras_succeeded: int  = 0
    per_call_goals:   list[int] = field(default_factory=list)
    n_yes_goal:       int  = 0
    n_total_calls:    int  = 0
    prob_peak:        float = 0.0
    prob_sustained_s: int   = 0
    prob_veto:        bool = False
    final_goals:      int  = 0
    decision:         str  = "untouched"   # untouched|confirmed|downgraded|extras_failed


def _sustained_seconds_above(probs_window: np.ndarray, thresh: float) -> int:
    """Longest run of consecutive seconds with prob >= thresh."""
    best = cur = 0
    for v in probs_window:
        if v >= thresh:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _peak_in_window(
    probs: Optional[np.ndarray],
    segment_start: int,
    segment_end: int,
) -> tuple[float, int]:
    """Returns (peak_value, longest_sustained_above_PROB_VETO_PEAK_MIN)."""
    if probs is None or len(probs) == 0:
        return 1.0, 999   # no probs available → can't veto, fail-safe to ALLOW
    lo = max(0, int(segment_start))
    hi = min(len(probs), int(segment_end) + 1)
    if hi <= lo:
        return 0.0, 0
    sub = probs[lo:hi]
    return float(sub.max()), _sustained_seconds_above(sub, PROB_VETO_PEAK_MIN)


def confirm_goal(
    *,
    first_result:     dict,
    video_bytes:      bytes,
    prompt_text:      str,
    segment_start:    int,
    segment_end:      int,
    call_gemini:      Callable[[bytes, str, int, float], tuple[Optional[dict], dict]],
    fused_probs:      Optional[np.ndarray] = None,
    extra_call_temps: tuple[float, ...] = ENSEMBLE_EXTRA_CALLS_TEMPS,
) -> tuple[dict, EnsembleTrace]:
    """If first_result['goals'] >= 1, run a verification ensemble.
    Returns (possibly-modified result dict, trace).

    `call_gemini(video_bytes, prompt_text, segment_start, temperature)
    -> (response_dict | None, per_call_trace)` is the caller-provided
    Gemini invocation — same signature as the existing
    `_call_gemini_for_metrics` in 01_detect_segment_metrics.py.
    """
    trace = EnsembleTrace()
    first_goals = int(first_result.get("goals", 0) or 0)
    if first_goals < 1:
        # No goal claimed — nothing to verify
        trace.final_goals = first_goals
        trace.per_call_goals = [first_goals]
        trace.n_total_calls = 1
        trace.decision = "untouched"
        return first_result, trace

    trace.triggered = True
    yes_votes = 1 if first_goals >= 1 else 0
    per_call = [first_goals]

    # Fire extras
    for t in extra_call_temps:
        trace.extras_attempted += 1
        try:
            resp, _ = call_gemini(video_bytes, prompt_text, segment_start, t)
        except Exception as e:
            log.warning(f"goal_ensemble: extra call failed at temp={t}: {e}")
            continue
        if resp is None:
            continue
        trace.extras_succeeded += 1
        g = int(resp.get("goals", 0) or 0)
        per_call.append(g)
        if g >= 1:
            yes_votes += 1

    trace.per_call_goals = per_call
    trace.n_yes_goal = yes_votes
    trace.n_total_calls = len(per_call)

    if trace.extras_succeeded == 0:
        trace.decision = "extras_failed"
        trace.final_goals = first_goals
        return first_result, trace

    # Majority vote: >=2 of (1 first + 2 extras) say goal
    vote_passes = yes_votes >= 2

    # Prob-signal veto
    peak, sustained = _peak_in_window(fused_probs, segment_start, segment_end)
    trace.prob_peak = peak
    trace.prob_sustained_s = sustained
    prob_supports = (peak >= PROB_VETO_PEAK_MIN
                      and sustained >= PROB_VETO_SUSTAINED_SEC)
    trace.prob_veto = (not prob_supports) and (fused_probs is not None
                                                  and len(fused_probs) > 0)

    if vote_passes and prob_supports:
        trace.decision = "confirmed"
        trace.final_goals = first_goals
        return first_result, trace

    # Downgrade: vote failed OR prob vetoed
    downgraded = dict(first_result)
    downgraded["goals"] = 0
    # Maintain identity: saves = shotsOnNet - goals
    son = int(downgraded.get("shotsOnNet", 0) or 0)
    downgraded["saves"] = max(0, son)
    downgraded["_goal_ensemble_overrode"] = True
    downgraded["_goal_ensemble_reason"] = (
        "vote_failed" if not vote_passes
        else "prob_signal_veto")
    trace.decision = "downgraded"
    trace.final_goals = 0
    return downgraded, trace
