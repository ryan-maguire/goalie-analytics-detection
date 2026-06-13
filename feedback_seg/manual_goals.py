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


def _is_threat(s: Any) -> bool:
    """A rendered clip: only these become visible/output clips, so only these
    are eligible to match a coach goal (fixes matching invisible segments)."""
    return isinstance(s, dict) and bool(s.get("segmentHasThreat")) and s.get("metrics") is not None


def _goals_of(s: dict) -> float:
    return _num((s.get("metrics") or {}).get("goals"), 0.0) or 0.0


def reconcile_manual_goals(
    segments: list,
    manual_goals: Optional[list],
    *,
    goalie_side: Optional[str] = None,
) -> tuple[list, list]:
    """Reconcile coach-logged goals (ground truth) against AI segments.

    Pure function (no I/O). For each logged goal:
      * OVERLAP a rendered threat clip → PROMOTE it to a confirmed goal (coach
        overrides even an AI "save" call): metrics.goals≥1, manually_verified,
        was_ai_detected=True, goal_unconfirmed=False;
      * no overlap → INJECT a goal clip at the logged time (was_ai_detected=False).
    Then any AI goal clip NOT corroborated by a coach goal is flagged
    ``goal_unconfirmed=True`` (only when goals were logged) so the UI can keep it
    out of Save% until a user confirms it.

    Returns ``(segments_out, manual_goals_out)``; each goal gets a computed
    ``was_ai_detected`` (True=matched, False=injected, None=malformed time).
    """
    for s in segments:
        if isinstance(s, dict):
            s.setdefault("was_ai_detected", True)
            s.setdefault("manually_verified", False)
            s.setdefault("goal_unconfirmed", False)

    goals_out = [dict(g) for g in (manual_goals or []) if isinstance(g, dict)]
    injected: list = []
    claimed: set = set()  # id() of threat segments already matched to a goal

    for g in goals_out:
        # Every logged goal is a goal AGAINST the analyzed goalie. (The old
        # per-goal `scored_on` field was removed; any legacy value is ignored.)
        t0 = _num(g.get("start_time"))
        t1 = _num(g.get("end_time"))
        if t0 is None or t1 is None:
            g["was_ai_detected"] = None
            continue
        if t1 < t0:
            t0, t1 = t1, t0

        # Best-overlapping unclaimed rendered clip.
        best, best_ov = None, 0.0
        for s in segments:
            if not _is_threat(s) or id(s) in claimed:
                continue
            s0 = _num(s.get("segment_start"), 0.0)
            s1 = _num(s.get("segment_end"), 0.0)
            if not _overlaps(t0, t1, s0, s1):
                continue
            ov = min(t1, s1) - max(t0, s0)  # may be 0 for a boundary touch
            if best is None or ov > best_ov:
                best, best_ov = s, ov

        if best is not None:
            claimed.add(id(best))
            m = best.get("metrics")
            if not isinstance(m, dict):
                m = {}
                best["metrics"] = m
            if _num(m.get("goals"), 0.0) < 1:
                m["goals"] = 1  # promote: coach says goal, even if AI called it a save
            best["manually_verified"] = True
            best["was_ai_detected"] = True
            best["goal_unconfirmed"] = False
            g["was_ai_detected"] = True
        else:
            injected.append(_make_injected_segment(
                int(math.floor(t0)), int(math.ceil(t1)), goalie_side,
            ))
            g["was_ai_detected"] = False

    # AI goal clips the coach didn't corroborate → "likely not a goal" (only when
    # the coach provided ground truth). Confirmed/injected goals are skipped.
    if goals_out:
        for s in segments:
            if _is_threat(s) and _goals_of(s) > 0 and s.get("was_ai_detected") is True and not s.get("manually_verified"):
                s["goal_unconfirmed"] = True

    return segments + injected, goals_out
