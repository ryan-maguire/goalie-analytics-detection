"""Reconcile manually-logged goals against AI-detected segments.

A coach logs the exact time range of every goal scored AGAINST this goalie in
the Add/Edit form (`manual_goals_logged` on the video record) — that's the only
kind of goal the pipeline can validate, since the AI only analyzes this goalie's
net. Between stage 2 (metrics) and stage 3 (feedback), we compare each logged
goal to the AI-detected segments and:

  * stamp every AI segment with ``was_ai_detected=True`` /
    ``manually_verified=False`` (so the UI can badge/filter all clips);
  * for each logged goal:
      - OVERLAP  → mark the overlapping segment(s) ``manually_verified=True``
                   and record the goal as ``was_ai_detected=True``;
      - MISS     → inject a synthetic segment at the goal's boundaries
                   (``was_ai_detected=False``, ``manually_verified=True``) with
                   a goal-bearing ``metrics`` block so it flows through the rest
                   of the pipeline and reaches Coach Review; record the goal as
                   ``was_ai_detected=False``.

(The old per-goal ``scored_on`` field was removed — all logged goals are
goals-against; any legacy value is ignored.)

This module is pure (no I/O) so the overlap/injection logic is unit-testable.
"""

import math
from typing import Any, Optional


def _num(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        n = float(v)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Closed-interval overlap test: [a_start,a_end] vs [b_start,b_end]."""
    return a_start <= b_end and a_end >= b_start


def _make_injected_segment(start: int, end: int, goalie_side: Optional[str]) -> dict:
    """Synthetic segment for an AI-missed goal. Carries a goal-bearing metrics
    block so it survives the ``segmentHasThreat && metrics is not None`` filter
    and gets the same stage-3 enrichment as AI clips."""
    side = goalie_side or "unknown"
    return {
        "segment_start": start,
        "segment_end": end,
        "segmentHasThreat": True,
        "threat_goalie_side": side,
        "source_signals": ["manual_goal"],
        "was_ai_detected": False,
        "manually_verified": True,
        "metrics": {
            "shots": 1,
            "shotsOnNet": 1,
            "saves": 0,
            "goals": 1,
            "rebounds": 0,
            "observed_goalie_side": side,
            "goal_criteria": {},
            "shot_timestamps": [],
        },
    }


def reconcile_manual_goals(
    segments: list,
    manual_goals: Optional[list],
    *,
    goalie_side: Optional[str] = None,
) -> tuple[list, list]:
    """Stamp AI segments, overlap-match / inject manual goals-against.

    Pure function. Mutates the ``was_ai_detected`` / ``manually_verified`` flags
    on the input ``segments`` (and may append injected segments), and returns
    ``(segments_out, manual_goals_out)`` where each goal dict gains a computed
    ``was_ai_detected`` (or ``None`` for own-team goals).
    """
    # Default flags on every AI-produced segment.
    for s in segments:
        if isinstance(s, dict):
            s.setdefault("was_ai_detected", True)
            s.setdefault("manually_verified", False)

    goals_out = [dict(g) for g in (manual_goals or []) if isinstance(g, dict)]
    injected: list = []

    for g in goals_out:
        # Every logged goal is a goal AGAINST the analyzed goalie — the only kind
        # the pipeline can validate (the AI only sees this goalie's net). The old
        # per-goal `scored_on` field was removed; any legacy value is ignored.
        t0 = _num(g.get("start_time"))
        t1 = _num(g.get("end_time"))
        if t0 is None or t1 is None:
            g["was_ai_detected"] = None
            continue
        if t1 < t0:
            t0, t1 = t1, t0

        overlapping = [
            s for s in segments
            if isinstance(s, dict)
            and _overlaps(
                t0, t1,
                _num(s.get("segment_start"), 0.0),
                _num(s.get("segment_end"), 0.0),
            )
        ]

        if overlapping:
            for s in overlapping:
                s["manually_verified"] = True
            g["was_ai_detected"] = True
        else:
            seg = _make_injected_segment(
                int(math.floor(t0)), int(math.ceil(t1)), goalie_side,
            )
            injected.append(seg)
            g["was_ai_detected"] = False

    return segments + injected, goals_out
