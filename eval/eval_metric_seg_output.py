"""
eval_metric_seg_output.py

Evaluate the metrics-detection step (gt_metrics_*.json from
01_detect_segment_metrics.py) against Hudl ground-truth event CSVs.

Where the cv_seg eval asked "did we find the right WINDOWS?", this
eval asks "given the windows cv_seg produced, did the metrics model
count the right things INSIDE them?"

Pipeline:
  1. Load gt_metrics_{vID}.json — cv_seg threat segments enriched
     with per-segment metrics (shots, goals, etc.) from Gemini.
  2. Load gt_{hudl_id}.csv — Hudl-tagged events with timestamps,
     teams, and action types ('Shots' / 'Goals' / 'OZ play').
  3. Resolve team→color mapping from the customer file (same logic
     as eval_cv_seg_output) so we know which team's events count
     toward each window's expected metrics.
  4. Read cv_seg's gt_seg_{vID}_meta.json sidecar to detect whether
     cv_seg ran with --target-filter (the default, v23.7+). When set,
     game-level totals are restricted to the OPPONENT team only,
     matching the segments that survived cv_seg's filter. Without
     this restriction, the predicted totals (only target-defended
     threats) get compared against the full Hudl game and look
     ~50% under-predicted.
  5. For each threat segment with metrics:
     - Find Hudl events overlapping the segment's [start, end].
     - Filter to events from the OPPOSING team (since their shots
       are the threat to this window's goalie).
     - Count Shots and Goals → expected metrics.
     - Compare to predicted metrics from the model.
  6. Aggregate per-video and across videos. Report:
     - Game totals: predicted vs actual shots/goals (target-restricted
       in target_filter mode)
     - Per-window MAE on shots and goals
     - Confusion-matrix-style stats on goals (the rare-event metric)
     - Diagnostic TSV with every window's predicted vs actual

Usage:
  python eval_metric_seg_output.py --vIDs mjEeE7p2Hz8 \\
      --customer-id CUST000048

  # Multi-customer
  python eval_metric_seg_output.py --customer-id CUST000048 CUST000031

Outputs (to data/output/evals/):
  eval_metrics_{ts}.txt       — text report
  eval_metrics_{ts}.json      — full structured results
  eval_metrics_{ts}_per_window.tsv  — one row per window for inspection
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional


# Reuse cv_seg eval's helpers — VID_TO_HUDL mapping, customer-file
# parsing, color-matching logic. Kept in sync by import rather than
# copy-paste.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cv_seg_output import (  # noqa: E402
    VID_TO_HUDL,
    DEFAULT_DATA_ROOT,
    DEFAULT_GT_DIR,
    fetch_customer_file,
    build_team_color_map_from_customer,
    load_team_color_map,
    _colors_match,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_METRICS_DIR = os.path.join(DEFAULT_DATA_ROOT, "output", "runs", "metrics_seg")
DEFAULT_OUTPUT_DIR  = os.path.join(DEFAULT_DATA_ROOT, "output", "evals")
# cv_seg's prediction directory — needed to read the _meta.json sidecar
# so we can detect target_filter mode and adjust the eval accordingly.
# Default matches run_pipeline.py's --local-output-dir layout. Override
# with --cv-seg-dir to evaluate ad-hoc runs.
DEFAULT_CV_SEG_DIR  = os.path.join(DEFAULT_DATA_ROOT, "output", "runs", "cv_seg")

# GCS — mirrors the cv_seg eval's pattern. When a metrics or cv_seg
# meta file isn't on disk, the eval falls back to downloading from
# these prefixes. Prefixes match the v23.7 path migration.
GCS_BUCKET           = "goalie_video_bucket"
GCS_METRICS_PREFIX   = "analyze_video/02-segment_metrics"
GCS_CV_SEG_PREFIX    = "analyze_video/01-segment_detection"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HudlEvent:
    """One row from gt_{hudl_id}.csv."""
    start:  float
    end:    float
    team:   str
    action: str        # 'Shots' | 'Goals' | 'OZ play'
    half:   int

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.start + self.end)


@dataclass
class WindowEval:
    """Per-window comparison: model vs ground truth."""
    vID:           str
    segment_start: float
    segment_end:   float
    threat_color:  Optional[str]

    # Model-predicted (from gt_metrics_*.json)
    pred_shots:  int
    pred_goals:  int

    # Hudl-derived ground truth — STRICT (team-filtered)
    gt_shots:    int
    gt_goals:    int

    # Hudl-derived ground truth — UNFILTERED (any team's events
    # overlapping this window). Independent of cv_seg's attribution
    # quality; tells us how many Hudl events occurred in this time
    # window regardless of which goalie was supposedly under threat.
    gt_shots_unfiltered: int = 0
    gt_goals_unfiltered: int = 0

    # Diagnostic — which Hudl events were attributed here
    matched_shots_team:  Optional[str] = None  # whose shots were counted
    n_overlapping_total: int = 0               # all overlapping events,
                                               # before team filtering
    notes: list[str] = field(default_factory=list)

    @property
    def shot_delta(self) -> int:
        return self.pred_shots - self.gt_shots

    @property
    def goal_delta(self) -> int:
        return self.pred_goals - self.gt_goals

    @property
    def shot_delta_unfiltered(self) -> int:
        return self.pred_shots - self.gt_shots_unfiltered

    @property
    def goal_delta_unfiltered(self) -> int:
        return self.pred_goals - self.gt_goals_unfiltered


@dataclass
class VideoEvalResult:
    vID:     str
    hudl_id: Optional[int]
    status:  str = "ok"
    notes:   list[str] = field(default_factory=list)

    windows: list[WindowEval] = field(default_factory=list)

    # Game-level totals — STRICT (team-filtered)
    total_pred_shots:  int = 0
    total_gt_shots:    int = 0
    total_pred_goals:  int = 0
    total_gt_goals:    int = 0

    # Game-level totals — UNFILTERED (any team)
    total_gt_shots_unfiltered: int = 0
    total_gt_goals_unfiltered: int = 0

    # Total Hudl events (any team) and target-team-only events for
    # the whole game — used to sanity-check the team-attribution flip.
    hudl_total_shots:  int = 0   # all Shots events in the CSV
    hudl_total_goals:  int = 0

    # v11: shot-level evaluation results.  Optional — None when the
    # metrics output is v10 (no shot_timestamps field present).
    shot_eval: Optional["ShotEvalResult"] = None

    # v11+: window-refinement diagnostics (only populated when segments
    # have segment_start_refined / segment_end_refined fields, added by
    # metrics_seg's _refine_all_segments).  Tracks how aggressively the
    # refinement tightened windows on this video.
    refinement: Optional[dict] = None


# ── v11 shot-level eval: per-shot match against Hudl GT ───────────────────
#
# v11 added shot_timestamps to metrics_seg output: one structured entry
# per shot with timestamp (MM:SS within the clip), location, release type,
# and outcome. This eval track validates each predicted timestamp against
# the Hudl GT Shots events.
#
# Matching is one-to-one (greedy by closest midpoint): if Gemini split a
# single GT shot into two predictions, we count one TP and one FP, never
# two TPs. This is the right counting discipline because coaches care
# about per-shot accuracy, not just totals.
#
# Three failure modes for a GT shot to surface clearly:
#   - fn (covered, missed): GT shot in time cv_seg flagged as threat,
#     no prediction landed inside it — Gemini's failure
#   - fn_uncovered: GT shot in time cv_seg never flagged as a threat —
#     cv_seg's failure, Gemini never got to see the clip
#   - tp_strict: prediction inside GT window
# We report both recall (TP / TP+FN+FN_uncovered) and recall_within_coverage
# (TP / TP+FN) so cv_seg vs metric_seg errors can be attributed cleanly.


@dataclass
class ShotMatch:
    """One predicted shot from gt_metrics[*].shot_timestamps and its
    match to a Hudl GT Shots event (if any)."""
    vID:            str
    segment_start:  float
    abs_timestamp:  float          # segment_start + parsed MM:SS
    timestamp_str:  str
    location:       str
    release:        str
    outcome:        str

    # Match result (None when no match)
    gt_shot_start:  Optional[float] = None
    gt_shot_end:    Optional[float] = None
    gt_team:        Optional[str]   = None
    matched:        bool            = False
    match_class:    str             = "no_match"  # 'inside' | 'tolerance' | 'no_match'


@dataclass
class ShotEvalResult:
    """Per-video shot-level evaluation.  All counts are AFTER one-to-one
    matching; a single Hudl shot is consumed by at most one prediction."""
    vID:           str
    matches:       list[ShotMatch] = field(default_factory=list)

    tp_strict:     int = 0    # pred falls inside a GT shot window
    tp_lenient:    int = 0    # pred falls within ±tolerance of a GT window
    fp:            int = 0    # pred matched no GT shot
    fn:            int = 0    # GT shot was in cv_seg-covered time, missed
    fn_uncovered:  int = 0    # GT shot was in time cv_seg never flagged

    n_total_pred:    int = 0  # total predicted shots from shot_timestamps
    n_total_gt:      int = 0  # total GT Shots events (target-team if known)
    n_gt_covered:    int = 0  # GT shots that fell inside a cv_seg segment
    n_gt_uncovered:  int = 0  # GT shots NOT inside any cv_seg segment

    @property
    def precision(self) -> float:
        denom = self.tp_strict + self.fp
        return self.tp_strict / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """End-to-end recall: TP / all GT shots (including cv_seg misses).

        Denominator is the total GT count, NOT tp_strict+fn+fn_uncovered.
        A tolerance-matched GT is removed from `fn` (it was consumed by a
        prediction) but is scored fp, not tp_strict — so the old formula
        dropped it from both numerator and denominator and inflated recall.
        n_total_gt is the stable, partition-independent denominator."""
        denom = self.n_total_gt
        return self.tp_strict / denom if denom else 0.0

    @property
    def recall_within_coverage(self) -> float:
        """Recall restricted to time cv_seg actually flagged.  Isolates
        metrics_seg's contribution from cv_seg's coverage. Denominator is
        the covered-GT count (same tolerance-match caveat as recall)."""
        denom = self.n_gt_covered
        return self.tp_strict / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2*p*r/(p+r) if (p+r) else 0.0

    @property
    def f1_within_coverage(self) -> float:
        p, r = self.precision, self.recall_within_coverage
        return 2*p*r/(p+r) if (p+r) else 0.0


# ---------------------------------------------------------------------------
# Hudl CSV loading
# ---------------------------------------------------------------------------

def load_hudl_events(gt_path: str) -> list[HudlEvent]:
    """Parse a Hudl gt_{hudl_id}.csv into HudlEvent objects."""
    events: list[HudlEvent] = []
    with open(gt_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                events.append(HudlEvent(
                    start=float(row["start"]),
                    end=float(row["end"]),
                    team=(row.get("team") or "").strip(),
                    action=(row.get("action") or "").strip(),
                    half=int(row.get("half") or 1),
                ))
            except (ValueError, KeyError) as e:
                log.warning(f"Skipping malformed Hudl row in {gt_path}: {e}")
    return events


# ---------------------------------------------------------------------------
# Metrics output loading
# ---------------------------------------------------------------------------

def _try_gcs_download(blob_name: str, local_path: str) -> bool:
    """Best-effort download of a single GCS object to local_path.
    Returns True iff the file landed on disk. Failures logged, never
    raised — the caller decides how to handle absence.
    """
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        log.warning("  google-cloud-storage not installed — skipping GCS download")
        return False
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            return False
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        blob.download_to_filename(local_path)
        log.info(f"  GCS: downloaded {blob_name} -> {local_path}")
        return True
    except Exception as e:
        log.warning(f"  GCS download failed for {blob_name}: {e}")
        return False


def load_metrics_output(
    vID: str,
    metrics_dir: str,
    allow_gcs_download: bool = True,
) -> Optional[list[dict]]:
    """Load gt_metrics_{vID}.json. Returns None if missing/unreadable.

    With allow_gcs_download=True (default), falls back to downloading
    from gs://{GCS_BUCKET}/{GCS_METRICS_PREFIX}/ when the file isn't
    already on disk. This mirrors what eval_cv_seg_output.py does for
    its predictions and is what makes the eval usable from a fresh
    machine without manual prefetching.
    """
    path = os.path.join(metrics_dir, f"gt_metrics_{vID}.json")

    if not os.path.exists(path) and allow_gcs_download:
        _try_gcs_download(f"{GCS_METRICS_PREFIX}/gt_metrics_{vID}.json", path)

    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"Failed to read {path}: {e}")
        return None


def load_cv_seg_meta(
    vID: str,
    cv_seg_dir: str,
    allow_gcs_download: bool = True,
) -> Optional[dict]:
    """Load gt_seg_{vID}_meta.json. Returns None if missing/unreadable.

    The eval reads this sidecar to detect cv_seg's target_filter mode.
    Missing meta isn't fatal — older cv_seg outputs don't have it, and
    the eval falls back to its previous behaviour (count all overlapping
    events without target-team restriction at the game-totals level).

    With allow_gcs_download=True (default), falls back to downloading
    from gs://{GCS_BUCKET}/{GCS_CV_SEG_PREFIX}/ when not on disk.
    """
    path = os.path.join(cv_seg_dir, f"gt_seg_{vID}_meta.json")

    if not os.path.exists(path) and allow_gcs_download:
        _try_gcs_download(f"{GCS_CV_SEG_PREFIX}/gt_seg_{vID}_meta.json", path)

    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"[{vID}] failed to read cv_seg meta {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Window evaluation
# ---------------------------------------------------------------------------

def _team_name_tokens(name: Optional[str]) -> set:
    """Tokenise a team-name string for fuzzy matching. Lowercase,
    whitespace-split, drop empties. Trailing punctuation is stripped
    from each token so 'jr.' and 'jr' compare equal — Hudl CSVs
    sometimes write "Philadelphia Jr. Flyers 19U AA" where the
    customer file has "Jr Flyers 19U". Empty input → empty set."""
    if not name:
        return set()
    raw_tokens = name.lower().strip().split()
    # Strip common trailing punctuation. We don't touch internal chars
    # (so apostrophes inside "coeur d'alene" survive) — only the very
    # end of each token, which is where the punctuation-mismatch bugs
    # we've seen actually originate.
    cleaned = (t.rstrip(".,;:!?") for t in raw_tokens)
    return {t for t in cleaned if t}


def _team_names_match(a: Optional[str], b: Optional[str]) -> bool:
    """Tolerant team-name comparison.

    Returns True if `a` and `b` refer to the same team across the
    customer-file/Hudl boundary. Mirrors the implementation in
    eval_cv_seg_output.py — see that file for the full design notes.

    Three rules in order:
      1. Exact match (after stripping whitespace)
      2. Case-insensitive exact match
      3. Token-set subset, both sides ≥2 tokens (handles 'Team South
         Dakota' ≈ 'Team South Dakota 19U' but rejects 'Team' alone)
    """
    if a is None or b is None:
        return False
    a_clean = a.strip()
    b_clean = b.strip()
    if not a_clean or not b_clean:
        return False
    if a_clean == b_clean:
        return True
    if a_clean.lower() == b_clean.lower():
        return True
    a_tok = _team_name_tokens(a_clean)
    b_tok = _team_name_tokens(b_clean)
    if len(a_tok) < 2 or len(b_tok) < 2:
        return False
    return a_tok.issubset(b_tok) or b_tok.issubset(a_tok)


def _events_overlap_window(
    event:    HudlEvent,
    win_start: float,
    win_end:   float,
) -> bool:
    """Any-overlap: the event's interval intersects the window's interval.

    Half-open boundaries — touching ends don't count. A shot at exactly
    the window boundary is unusual enough that the user probably wants
    to inspect it manually rather than have the eval silently include
    or exclude it.
    """
    return event.end > win_start and event.start < win_end


def _resolve_target_team(
    threat_color:    Optional[str],
    team_color_map:  Optional[dict[str, str]],
) -> Optional[str]:
    """Given a window's threat_goalie_color (the DEFENDING goalie's
    jersey color), find which TEAM is shooting at that goalie. Their
    'Shots' events in the Hudl CSV are the ones that count for this
    window.

    Important: the team_color_map produced by
    build_team_color_map_from_customer is keyed
        {team_name: color_of_OPPOSING_goalie_they_threaten}
    NOT {team_name: their_own_jersey_color}.
    This is because cv_seg eval used it to answer "predicted threat
    color X — which team should this be attributed to?" → "the team
    whose mapped color is X, since that team threatens that color."

    So to find the SHOOTING team for a window with threat_color=X,
    we look up which team's mapped color EQUALS X. (Earlier draft of
    this function got this backwards by inverting the match.)

    Returns the shooting-team name, or None if we can't resolve it
    (no mapping, no color match, or color-collision case).
    """
    if not team_color_map or not threat_color:
        return None

    matching_teams: list[str] = []
    for team, color in team_color_map.items():
        if team.startswith("_"):
            continue  # sentinel keys (_collision, _target_tokens, etc.)
        if _colors_match(color, threat_color):
            matching_teams.append(team)

    # In a 2-team game we expect exactly one match. Zero or many means
    # the mapping is ambiguous for this window — return None so we fall
    # back to the all-overlapping-events path with a note.
    if len(matching_teams) == 1:
        return matching_teams[0]
    return None


def evaluate_window(
    vID:            str,
    segment:        dict,
    hudl_events:    list[HudlEvent],
    team_color_map: Optional[dict[str, str]],
) -> Optional[WindowEval]:
    """Build a per-window comparison record. Returns None for non-threat
    segments and for threats with no metrics (failed Gemini calls).

    Computes BOTH:
      - team-filtered counts (gt_shots / gt_goals): only events from
        the team SHOOTING at this window's goalie
      - unfiltered counts (gt_shots_unfiltered / gt_goals_unfiltered):
        all events overlapping the window regardless of team

    The team-filtered numbers are stricter and the right answer when
    cv_seg's attribution is correct. The unfiltered numbers tell us
    "in the time window cv_seg flagged as a threat, how many real
    Hudl events occurred at all?" — which is the question that matters
    for the METRIC model's quality, independent of upstream attribution
    errors. Both are reported; the user can read whichever fits their
    question.
    """
    if not segment.get("segmentHasThreat"):
        return None
    metrics = segment.get("metrics")
    if metrics is None:
        return None

    win_start = segment["segment_start"]
    win_end   = segment["segment_end"]
    threat_color = segment.get("threat_goalie_color")

    # All overlapping events (any team)
    overlapping = [e for e in hudl_events if _events_overlap_window(e, win_start, win_end)]

    # Unfiltered counts — any team's events in this window
    gt_shots_unfiltered = sum(1 for e in overlapping if e.action == "Shots")
    gt_goals_unfiltered = sum(1 for e in overlapping if e.action == "Goals")

    # Team-filtered counts — only events from the team SHOOTING at
    # this window's goalie (correct answer when cv_seg attribution is right)
    target_team = _resolve_target_team(threat_color, team_color_map)
    notes: list[str] = []
    if target_team:
        # Fuzzy match — handles 'Team South Dakota' (customer file) vs
        # 'Team South Dakota 19U' (Hudl CSV).
        relevant = [e for e in overlapping
                    if _team_names_match(e.team, target_team)]
    else:
        # Fall back to unfiltered if we have no mapping. Caller is
        # already noting why (no map, collision, etc.).
        relevant = overlapping
        notes = ["no team mapping — strict counts == unfiltered counts"]

    gt_shots = sum(1 for e in relevant if e.action == "Shots")
    gt_goals = sum(1 for e in relevant if e.action == "Goals")

    return WindowEval(
        vID=vID,
        segment_start=win_start,
        segment_end=win_end,
        threat_color=threat_color,
        pred_shots=int(metrics.get("shots", 0)),
        pred_goals=int(metrics.get("goals", 0)),
        gt_shots=gt_shots,
        gt_goals=gt_goals,
        gt_shots_unfiltered=gt_shots_unfiltered,
        gt_goals_unfiltered=gt_goals_unfiltered,
        matched_shots_team=target_team,
        n_overlapping_total=len(overlapping),
        notes=notes,
    )


def _parse_mm_ss(s: str) -> Optional[float]:
    """Parse 'MM:SS' (or 'M:SS') to seconds.  Returns None if malformed.
    Accepts 0..99 minutes — long segments are theoretically possible,
    and double-digit MM is what the v11 prompt produces."""
    if not isinstance(s, str):
        return None
    parts = s.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        mm = int(parts[0])
        ss = int(parts[1])
    except ValueError:
        return None
    if mm < 0 or ss < 0 or ss >= 60:
        return None
    return float(mm * 60 + ss)


# Tolerance for "shot just outside the GT window" matching.  Hudl windows
# are nominally 12s wide centered on the moment; the human tagger's hand
# is not pixel-perfect so a few seconds of slop is reasonable.  Set to 0
# to require strict containment.
SHOT_MATCH_TOLERANCE_SEC = 3


def evaluate_shot_timestamps(
    vID:            str,
    metrics_data:   list[dict],
    hudl_events:    list[HudlEvent],
    team_color_map: Optional[dict[str, str]],
    *,
    tolerance_sec:  float = SHOT_MATCH_TOLERANCE_SEC,
) -> Optional[ShotEvalResult]:
    """Validate each predicted shot_timestamps entry against Hudl GT.

    Strategy:
      1. Collect all predictions across all threat segments, converting
         clip-relative MM:SS to absolute video seconds.
      2. Collect all GT Shots events (team-filtered when possible).
      3. Greedy one-to-one match: each prediction takes the closest
         unmatched GT shot that contains it (or is within tolerance).
      4. Remaining GT shots are FN, sub-classified by whether they fall
         in time cv_seg covered with a threat segment (fn — metric_seg's
         miss) or outside any segment (fn_uncovered — cv_seg's miss).

    Returns None if the metrics output has no shot_timestamps field at
    all (i.e. it's v10 output).  The caller skips the section gracefully.
    """
    # 1. Predictions
    threat_segments = [s for s in metrics_data if s.get("segmentHasThreat")]

    # Detect whether any segment has shot_timestamps (v11 marker).
    any_has_field = any("shot_timestamps" in (s.get("metrics") or {})
                       for s in threat_segments)
    if not any_has_field:
        return None

    pred_shots: list[ShotMatch] = []
    for seg in threat_segments:
        m = seg.get("metrics") or {}
        ts_list = m.get("shot_timestamps")
        if not isinstance(ts_list, list):
            continue
        seg_start = float(seg.get("segment_start") or 0)
        seg_end   = float(seg.get("segment_end")   or 0)
        for ts in ts_list:
            if not isinstance(ts, dict):
                continue
            ts_str = ts.get("timestamp", "") or ""
            offset = _parse_mm_ss(ts_str)
            if offset is None:
                log.warning(
                    f"  [{vID}] segment {seg_start}-{seg_end}: "
                    f"malformed shot_timestamps timestamp {ts_str!r} — skipping"
                )
                continue
            abs_t = seg_start + offset
            # Clamp to segment bounds — Gemini occasionally returns a
            # timestamp slightly past the clip end due to off-by-one
            # in its MM:SS rounding. Treating those as the segment end
            # keeps them matchable.
            if abs_t > seg_end:
                abs_t = seg_end
            pred_shots.append(ShotMatch(
                vID=vID,
                segment_start=seg_start,
                abs_timestamp=abs_t,
                timestamp_str=ts_str,
                location=ts.get("location", "") or "",
                release=ts.get("release", "") or "",
                outcome=ts.get("outcome", "") or "",
            ))

    # 2. GT Shots events
    target_team = None
    threat_color_seen = None
    for seg in threat_segments:
        if seg.get("threat_goalie_color"):
            threat_color_seen = seg["threat_goalie_color"]
            break
    if threat_color_seen and team_color_map:
        target_team = _resolve_target_team(threat_color_seen, team_color_map)

    all_gt_shots = [e for e in hudl_events if e.action == "Shots"]
    if target_team:
        gt_shots_relevant = [e for e in all_gt_shots
                             if _team_names_match(e.team, target_team)]
    else:
        gt_shots_relevant = all_gt_shots[:]   # no mapping; use unfiltered

    # 3. Classify each GT shot by coverage (is it inside any cv_seg threat segment?)
    def _gt_is_covered(gt: HudlEvent) -> bool:
        # A GT shot is "covered" if its midpoint falls inside any
        # threat segment.  Using midpoint avoids edge-of-window
        # ambiguity (some GT clips share boundaries with segments).
        mid = gt.midpoint
        for seg in threat_segments:
            ss = float(seg.get("segment_start") or 0)
            se = float(seg.get("segment_end")   or 0)
            if ss <= mid <= se:
                return True
        return False

    gt_covered   = [g for g in gt_shots_relevant if _gt_is_covered(g)]
    gt_uncovered = [g for g in gt_shots_relevant if not _gt_is_covered(g)]

    # 4. Greedy one-to-one matching.  Process predictions in time order;
    # for each, find the closest UNMATCHED GT shot whose window (with
    # tolerance) contains it.  Closest = smallest distance from
    # prediction to GT midpoint.
    pred_shots_sorted = sorted(pred_shots, key=lambda p: p.abs_timestamp)
    available_gt = list(gt_covered)   # uncovered GTs are not eligible
                                       # to match (their cv_seg segment
                                       # doesn't exist, so predictions
                                       # there are impossible).
    available_gt.sort(key=lambda g: g.start)

    for p in pred_shots_sorted:
        best_gt = None
        best_class = None
        best_dist = float("inf")
        for g in available_gt:
            # Strict: prediction inside [start, end]
            if g.start <= p.abs_timestamp <= g.end:
                dist = abs(p.abs_timestamp - g.midpoint)
                if dist < best_dist:
                    best_dist = dist
                    best_gt = g
                    best_class = "inside"
                continue
            # Lenient: within ±tolerance
            if (g.start - tolerance_sec) <= p.abs_timestamp <= (g.end + tolerance_sec):
                dist = abs(p.abs_timestamp - g.midpoint)
                # Strict matches always preferred over lenient.
                if best_class != "inside" and dist < best_dist:
                    best_dist = dist
                    best_gt = g
                    best_class = "tolerance"
        if best_gt is not None:
            p.gt_shot_start = best_gt.start
            p.gt_shot_end   = best_gt.end
            p.gt_team       = best_gt.team
            p.matched       = True
            p.match_class   = best_class
            available_gt.remove(best_gt)

    # Aggregate
    result = ShotEvalResult(vID=vID, matches=pred_shots_sorted)
    result.n_total_pred   = len(pred_shots_sorted)
    result.n_total_gt     = len(gt_shots_relevant)
    result.n_gt_covered   = len(gt_covered)
    result.n_gt_uncovered = len(gt_uncovered)
    result.fn_uncovered   = len(gt_uncovered)
    result.fn             = len(available_gt)   # remaining unmatched covered GTs

    for p in pred_shots_sorted:
        if p.match_class == "inside":
            result.tp_strict  += 1
            result.tp_lenient += 1
        elif p.match_class == "tolerance":
            result.tp_lenient += 1
            # Note: tolerance matches are NOT counted as TP_strict.
            # We score this as a FP under the strict rule but record
            # the lenient match in tp_lenient for diagnostic visibility.
            result.fp += 1
        else:
            result.fp += 1

    return result



def evaluate_video(
    vID:            str,
    gt_dir:         str,
    metrics_dir:    str,
    cv_seg_dir:     str,
    team_color_map_global: dict[str, dict[str, str]],
    allow_gcs_download: bool = True,
) -> VideoEvalResult:
    hudl_id = VID_TO_HUDL.get(vID)
    result = VideoEvalResult(vID=vID, hudl_id=hudl_id)

    if hudl_id is None:
        result.status = "skipped"
        result.notes.append("no hudl_id mapping")
        return result

    gt_path = os.path.join(gt_dir, f"gt_{hudl_id}.csv")
    if not os.path.exists(gt_path):
        result.status = "skipped"
        result.notes.append(f"ground truth not found: {gt_path}")
        return result

    try:
        hudl_events = load_hudl_events(gt_path)
    except Exception as e:
        result.status = "error"
        result.notes.append(f"failed to load Hudl CSV: {e}")
        return result

    metrics_output = load_metrics_output(
        vID, metrics_dir, allow_gcs_download=allow_gcs_download
    )
    if metrics_output is None:
        result.status = "missing_prediction"
        result.notes.append(f"gt_metrics_{vID}.json not found in {metrics_dir}")
        return result

    # Read cv_seg's meta sidecar to detect target_filter mode. Missing
    # meta is fine — older cv_seg outputs don't have it.
    cv_seg_meta = load_cv_seg_meta(
        vID, cv_seg_dir, allow_gcs_download=allow_gcs_download
    )
    target_filter_applied = bool(
        cv_seg_meta
        and cv_seg_meta.get("target_filter", {}).get("applied")
    )

    # Per-video team mapping (using string key first, int fallback —
    # mirrors what cv_seg eval does)
    team_map = (team_color_map_global.get(str(hudl_id))
                or team_color_map_global.get(hudl_id))

    if team_map is not None and isinstance(team_map.get("_collision"), list):
        result.notes.append(
            f"target/opponent share color token(s) {team_map['_collision']} "
            f"— per-window team filtering disabled, counting all overlapping events"
        )
        team_map = None
    elif team_map is None:
        result.notes.append("no team→color mapping — counting all overlapping events")

    # Game totals — when cv_seg's target_filter is on, restrict the
    # totals to the opponent team (the team whose shots threaten our
    # target goalie). Without this, the predicted game totals (which
    # only cover target-defended threats) get compared against the
    # FULL Hudl game (including target-team shots at the opponent
    # goalie), producing a ~50% under-prediction artefact.
    #
    # Per-window stats are unaffected — `evaluate_window` already does
    # the right thing per-segment via `_resolve_target_team`.
    if target_filter_applied and team_map is not None:
        opponent_team = team_map.get("_opponent_team_name")
        if opponent_team:
            pre_shots = sum(1 for e in hudl_events if e.action == "Shots")
            pre_goals = sum(1 for e in hudl_events if e.action == "Goals")
            # Fuzzy match — handles 'Team South Dakota' (customer file)
            # vs 'Team South Dakota 19U' (Hudl CSV) and similar suffix
            # mismatches across customer-file/Hudl boundary.
            result.hudl_total_shots = sum(
                1 for e in hudl_events
                if e.action == "Shots" and _team_names_match(e.team, opponent_team)
            )
            result.hudl_total_goals = sum(
                1 for e in hudl_events
                if e.action == "Goals" and _team_names_match(e.team, opponent_team)
            )
            result.notes.append(
                f"target_filter mode: game totals restricted to opponent "
                f"team {opponent_team!r} only "
                f"(shots {pre_shots} → {result.hudl_total_shots}, "
                f"goals {pre_goals} → {result.hudl_total_goals})"
            )
        else:
            # Old customer-file format without the new sentinel.
            result.hudl_total_shots = sum(1 for e in hudl_events if e.action == "Shots")
            result.hudl_total_goals = sum(1 for e in hudl_events if e.action == "Goals")
            result.notes.append(
                "target_filter is ON in cv_seg meta but team_map lacks "
                "_opponent_team_name sentinel (old customer-file format) — "
                "game totals will look ~50% under-predicted"
            )
    else:
        # Original behaviour: any-team game totals.
        result.hudl_total_shots = sum(1 for e in hudl_events if e.action == "Shots")
        result.hudl_total_goals = sum(1 for e in hudl_events if e.action == "Goals")
        if target_filter_applied and team_map is None:
            result.notes.append(
                "target_filter is ON in cv_seg meta but no team→color map "
                "available — game totals will look ~50% under-predicted"
            )

    # Build per-window evaluations
    for segment in metrics_output:
        we = evaluate_window(vID, segment, hudl_events, team_map)
        if we is not None:
            result.windows.append(we)
            result.total_pred_shots += we.pred_shots
            result.total_gt_shots   += we.gt_shots
            result.total_pred_goals += we.pred_goals
            result.total_gt_goals   += we.gt_goals
            result.total_gt_shots_unfiltered += we.gt_shots_unfiltered
            result.total_gt_goals_unfiltered += we.gt_goals_unfiltered

    # v11: shot-level evaluation against Hudl Shots events.
    # Returns None when metrics_output is v10 (no shot_timestamps).
    result.shot_eval = evaluate_shot_timestamps(
        vID, metrics_output, hudl_events, team_map,
    )

    # v11+: window-refinement diagnostics — describe how much the
    # refinement tightened windows on this video.  None if no segment
    # has the refined fields (i.e. metrics_seg pre-refinement).
    result.refinement = _summarize_refinement(metrics_output)

    return result


def _summarize_refinement(metrics_output: list[dict]) -> Optional[dict]:
    """Summarise segment_start_refined / segment_end_refined fields.
    Returns None if the metrics_output has no segments with refined
    fields (i.e. it predates the refinement feature)."""
    threats = [s for s in metrics_output if s.get("segmentHasThreat")]
    if not threats:
        return None
    has_refined = any("segment_start_refined" in s for s in threats)
    if not has_refined:
        return None

    sources: dict[str, int] = {}
    orig_widths: list[int] = []
    refined_widths: list[int] = []
    for s in threats:
        sources[s.get("refinement_source", "unknown")] = \
            sources.get(s.get("refinement_source", "unknown"), 0) + 1
        if "segment_start_refined" in s and "segment_end_refined" in s:
            orig_w    = (s.get("segment_end") or 0) - (s.get("segment_start") or 0)
            refined_w = s["segment_end_refined"] - s["segment_start_refined"]
            orig_widths.append(int(orig_w))
            refined_widths.append(int(refined_w))

    if not refined_widths:
        return None

    def _median(xs: list[int]) -> float:
        s = sorted(xs)
        n = len(s)
        if n == 0: return 0
        if n % 2 == 1: return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    return {
        "n_threat_segments":     len(threats),
        "refinement_sources":    sources,
        "orig_width_median":     _median(orig_widths),
        "orig_width_mean":       round(sum(orig_widths) / len(orig_widths), 1),
        "refined_width_median":  _median(refined_widths),
        "refined_width_mean":    round(sum(refined_widths) / len(refined_widths), 1),
        "median_shrinkage":      _median(orig_widths) - _median(refined_widths),
    }


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def _shot_metrics(
    windows: list[WindowEval],
    *,
    use_unfiltered: bool = False,
) -> dict:
    """MAE, signed bias, exact-match rate over per-window shot counts.

    use_unfiltered: if True, compares pred_shots against
        gt_shots_unfiltered (any team's events overlapping the window)
        rather than the team-filtered gt_shots.
    """
    if not windows:
        return {"n": 0, "mae": None, "bias": None, "exact_pct": None,
                "rmse": None}
    n = len(windows)
    if use_unfiltered:
        signed = [w.shot_delta_unfiltered for w in windows]
    else:
        signed = [w.shot_delta for w in windows]
    abs_errors = [abs(d) for d in signed]
    mae  = sum(abs_errors) / n
    bias = sum(signed) / n
    exact = sum(1 for d in signed if d == 0) / n
    rmse = (sum(d * d for d in signed) / n) ** 0.5
    return {
        "n":         n,
        "mae":       round(mae, 3),
        "rmse":      round(rmse, 3),
        "bias":      round(bias, 3),
        "exact_pct": round(100 * exact, 1),
    }


def _goal_classifier_metrics(
    windows: list[WindowEval],
    *,
    use_unfiltered: bool = False,
) -> dict:
    """Treat 'goals >= 1 in this window' as a binary classifier."""
    tp = fp = fn = tn = 0
    for w in windows:
        actual_pos = (w.gt_goals_unfiltered if use_unfiltered else w.gt_goals) >= 1
        pred_pos   = w.pred_goals >= 1
        if actual_pos and pred_pos:
            tp += 1
        elif actual_pos and not pred_pos:
            fn += 1
        elif not actual_pos and pred_pos:
            fp += 1
        else:
            tn += 1
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1: Optional[float] = None
    if p is not None and r is not None and (p + r) > 0:
        f1 = 2 * p * r / (p + r)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(p, 4) if p is not None else None,
        "recall":    round(r, 4) if r is not None else None,
        "f1":        round(f1, 4) if f1 is not None else None,
    }


def aggregate(results: list[VideoEvalResult]) -> dict:
    """Combine per-video results into overall stats. Reports BOTH
    team-filtered (strict) and unfiltered metrics.

    Why both:
      - team-filtered shows model performance assuming cv_seg's
        attribution is correct (the strict end-to-end evaluation)
      - unfiltered shows model performance independent of cv_seg
        attribution (does the model count what it sees in the clip?)
      - the gap between the two quantifies how much of the metric
        eval's error is actually upstream cv_seg attribution error
    """
    all_windows: list[WindowEval] = []
    for r in results:
        all_windows.extend(r.windows)

    total_pred_shots = sum(r.total_pred_shots for r in results)
    total_gt_shots   = sum(r.total_gt_shots   for r in results)
    total_pred_goals = sum(r.total_pred_goals for r in results)
    total_gt_goals   = sum(r.total_gt_goals   for r in results)
    total_gt_shots_unfiltered = sum(r.total_gt_shots_unfiltered for r in results)
    total_gt_goals_unfiltered = sum(r.total_gt_goals_unfiltered for r in results)

    # v11 shot-level aggregate.  Only populated if at least one video
    # has shot_eval set (i.e. v11 metrics output present).
    shot_evals = [r.shot_eval for r in results if r.shot_eval is not None]
    shot_aggregate: Optional[dict] = None
    if shot_evals:
        tp_strict  = sum(s.tp_strict     for s in shot_evals)
        tp_lenient = sum(s.tp_lenient    for s in shot_evals)
        fp         = sum(s.fp            for s in shot_evals)
        fn         = sum(s.fn            for s in shot_evals)
        fn_uncov   = sum(s.fn_uncovered  for s in shot_evals)
        n_pred     = sum(s.n_total_pred  for s in shot_evals)
        n_gt       = sum(s.n_total_gt    for s in shot_evals)
        n_covered  = sum(s.n_gt_covered  for s in shot_evals)
        n_uncov    = sum(s.n_gt_uncovered for s in shot_evals)
        precision = tp_strict / (tp_strict + fp) if (tp_strict + fp) else 0
        # Pooled recall uses the total GT count as denominator (not
        # tp_strict+fn+fn_uncov): tolerance matches are removed from `fn`
        # but scored fp, so the old denominator dropped them and inflated
        # recall. n_gt = n_covered + n_uncov is the stable denominator.
        recall    = tp_strict / n_gt if n_gt else 0
        recall_w  = tp_strict / n_covered if n_covered else 0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0
        f1_w      = 2*precision*recall_w/(precision+recall_w) if (precision+recall_w) else 0
        shot_aggregate = {
            "n_videos_with_shot_timestamps": len(shot_evals),
            "n_total_pred_shots": n_pred,
            "n_total_gt_shots":   n_gt,
            "n_gt_covered":       n_covered,
            "n_gt_uncovered":     n_uncov,
            "tp_strict":          tp_strict,
            "tp_lenient":         tp_lenient,
            "fp":                 fp,
            "fn":                 fn,
            "fn_uncovered":       fn_uncov,
            "precision":          round(precision, 4),
            "recall":             round(recall, 4),
            "recall_within_coverage": round(recall_w, 4),
            "f1":                 round(f1, 4),
            "f1_within_coverage": round(f1_w, 4),
        }

    return {
        "n_videos":          len(results),
        "n_windows":         len(all_windows),

        # STRICT (team-filtered)
        "total_pred_shots":  total_pred_shots,
        "total_gt_shots":    total_gt_shots,
        "shots_diff":        total_pred_shots - total_gt_shots,
        "shots_ratio":       round(_safe_div(total_pred_shots, total_gt_shots), 3)
                              if total_gt_shots else None,
        "total_pred_goals":  total_pred_goals,
        "total_gt_goals":    total_gt_goals,
        "goals_diff":        total_pred_goals - total_gt_goals,
        "shot_window_metrics":  _shot_metrics(all_windows),
        "goal_classifier":      _goal_classifier_metrics(all_windows),

        # UNFILTERED (any team's events)
        "total_gt_shots_unfiltered": total_gt_shots_unfiltered,
        "shots_diff_unfiltered":     total_pred_shots - total_gt_shots_unfiltered,
        "shots_ratio_unfiltered":    round(_safe_div(total_pred_shots, total_gt_shots_unfiltered), 3)
                                      if total_gt_shots_unfiltered else None,
        "total_gt_goals_unfiltered": total_gt_goals_unfiltered,
        "goals_diff_unfiltered":     total_pred_goals - total_gt_goals_unfiltered,
        "shot_window_metrics_unfiltered": _shot_metrics(all_windows, use_unfiltered=True),
        "goal_classifier_unfiltered":     _goal_classifier_metrics(all_windows, use_unfiltered=True),

        # v11 shot-level aggregate (None if no v11 output present)
        "shot_timestamps_eval": shot_aggregate,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_text_report(
    results: list[VideoEvalResult],
    summary: dict,
    args:    argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("metrics_seg evaluation report")
    lines.append(f"  generated:    {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"  gt_dir:       {args.gt_dir}")
    lines.append(f"  metrics_dir:  {args.metrics_dir}")
    lines.append(f"  cv_seg_dir:   {args.cv_seg_dir}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Two views are reported throughout:")
    lines.append("  STRICT     — only counts Hudl events from the team SHOOTING")
    lines.append("               at the window's threat_goalie_color. End-to-end")
    lines.append("               eval (depends on cv_seg attribution being correct).")
    lines.append("  UNFILTERED — counts ALL Hudl events overlapping the window")
    lines.append("               regardless of team. Tests the metric model in")
    lines.append("               isolation, independent of cv_seg attribution.")
    lines.append("")
    lines.append("If STRICT looks much worse than UNFILTERED, the gap is upstream")
    lines.append("cv_seg attribution error, not metric-model error.")
    lines.append("")

    # ── OVERALL ─────────────────────────────────────────────────────────
    lines.append("OVERALL — game-level totals")
    lines.append("-" * 78)
    lines.append(f"  videos:                 {summary['n_videos']}")
    lines.append(f"  windows compared:       {summary['n_windows']}")
    lines.append("")
    lines.append(f"  {'':<18} {'STRICT':>14} {'UNFILTERED':>14}")
    lines.append(f"  {'predicted shots':<18} {summary['total_pred_shots']:>14} "
                 f"{summary['total_pred_shots']:>14}")
    lines.append(f"  {'actual shots':<18} {summary['total_gt_shots']:>14} "
                 f"{summary['total_gt_shots_unfiltered']:>14}")
    lines.append(f"  {'shots diff':<18} {summary['shots_diff']:>+14d} "
                 f"{summary['shots_diff_unfiltered']:>+14d}")
    if summary["shots_ratio"] is not None and summary["shots_ratio_unfiltered"] is not None:
        lines.append(f"  {'pred/actual ratio':<18} {summary['shots_ratio']:>14.2f} "
                     f"{summary['shots_ratio_unfiltered']:>14.2f}")
    lines.append("")
    lines.append(f"  {'predicted goals':<18} {summary['total_pred_goals']:>14} "
                 f"{summary['total_pred_goals']:>14}")
    lines.append(f"  {'actual goals':<18} {summary['total_gt_goals']:>14} "
                 f"{summary['total_gt_goals_unfiltered']:>14}")
    lines.append(f"  {'goals diff':<18} {summary['goals_diff']:>+14d} "
                 f"{summary['goals_diff_unfiltered']:>+14d}")
    lines.append("")

    # ── SHOT-COUNT PER-WINDOW ───────────────────────────────────────────
    lines.append("PER-WINDOW shot-count accuracy")
    lines.append("-" * 78)
    sm  = summary["shot_window_metrics"]
    smu = summary["shot_window_metrics_unfiltered"]
    if sm["n"] == 0:
        lines.append("  (no windows to score)")
    else:
        lines.append(f"  windows:        {sm['n']}")
        lines.append(f"  {'':<22} {'STRICT':>10} {'UNFILTERED':>12}")
        lines.append(f"  {'mean abs error (MAE)':<22} {sm['mae']:>10} {smu['mae']:>12}")
        lines.append(f"  {'root mean sq error':<22} {sm['rmse']:>10} {smu['rmse']:>12}")
        lines.append(f"  {'signed bias':<22} {sm['bias']:>+10.3f} {smu['bias']:>+12.3f}")
        lines.append(f"  {'exact-match rate':<22} {sm['exact_pct']:>9}% {smu['exact_pct']:>11}%")
    lines.append("")

    # ── GOAL CLASSIFIER ─────────────────────────────────────────────────
    gm  = summary["goal_classifier"]
    gmu = summary["goal_classifier_unfiltered"]
    lines.append("GOAL DETECTION — binary classifier (goals>=1 in window)")
    lines.append("-" * 78)
    lines.append(f"  {'':<18} {'STRICT':>10} {'UNFILTERED':>12}")
    lines.append(f"  {'TP':<18} {gm['tp']:>10} {gmu['tp']:>12}")
    lines.append(f"  {'FP':<18} {gm['fp']:>10} {gmu['fp']:>12}")
    lines.append(f"  {'FN':<18} {gm['fn']:>10} {gmu['fn']:>12}")
    lines.append(f"  {'TN':<18} {gm['tn']:>10} {gmu['tn']:>12}")

    def _fmt_pct(v):
        return f"{v:.3f}" if v is not None else "    —"
    lines.append(f"  {'precision':<18} {_fmt_pct(gm['precision']):>10} "
                 f"{_fmt_pct(gmu['precision']):>12}")
    lines.append(f"  {'recall':<18} {_fmt_pct(gm['recall']):>10} "
                 f"{_fmt_pct(gmu['recall']):>12}")
    lines.append(f"  {'F1':<18} {_fmt_pct(gm['f1']):>10} {_fmt_pct(gmu['f1']):>12}")
    lines.append("")

    # ── PER VIDEO ───────────────────────────────────────────────────────
    lines.append("PER VIDEO (STRICT counts)")
    lines.append("-" * 78)
    header = (f"  {'vID':<13} {'hudl':>8}  {'status':<18}"
              f"{'win':>4}{'PrSh':>6}{'GtSh':>6}{'ΔSh':>6}"
              f"{'PrGl':>5}{'GtGl':>5}{'ΔGl':>5}")
    lines.append(header)
    for r in results:
        d_shots = r.total_pred_shots - r.total_gt_shots
        d_goals = r.total_pred_goals - r.total_gt_goals
        lines.append(
            f"  {r.vID:<13} {(r.hudl_id or 0):>8}  {r.status:<18}"
            f"{len(r.windows):>4}"
            f"{r.total_pred_shots:>6}{r.total_gt_shots:>6}{d_shots:>+6}"
            f"{r.total_pred_goals:>5}{r.total_gt_goals:>5}{d_goals:>+5}"
        )
        for note in r.notes:
            lines.append(f"      note: {note}")
    lines.append("")

    # ── SANITY ──────────────────────────────────────────────────────────
    lines.append("SANITY — Hudl game totals (all teams) vs window-level coverage")
    lines.append("-" * 78)
    lines.append("  Helps catch attribution mistakes — if Hudl recorded N "
                 "shots in a game but only M show up in our window-level "
                 "totals, the missing N-M either (a) fell outside any "
                 "cv_seg window, or (b) belonged to a team we filtered out.")
    for r in results:
        if r.status != "ok":
            continue
        if r.hudl_total_shots:
            lines.append(f"  {r.vID:<13}  hudl-game shots: {r.hudl_total_shots:>3}, "
                         f"window-attributed STRICT: {r.total_gt_shots:>3} "
                         f"({100*r.total_gt_shots/r.hudl_total_shots:>5.1f}%), "
                         f"UNFILTERED: {r.total_gt_shots_unfiltered:>3} "
                         f"({100*r.total_gt_shots_unfiltered/r.hudl_total_shots:>5.1f}%)")
        else:
            lines.append(f"  {r.vID:<13}  hudl-game shots: 0")
        lines.append(f"  {r.vID:<13}  hudl-game goals: {r.hudl_total_goals:>3}, "
                     f"window-attributed STRICT: {r.total_gt_goals:>3}, "
                     f"UNFILTERED: {r.total_gt_goals_unfiltered:>3}")

    # v11 shot-timestamps eval — only printed if at least one video has it.
    se = summary.get("shot_timestamps_eval")
    if se:
        lines.append("")
        lines.append("=" * 78)
        lines.append("SHOT-TIMESTAMPS EVAL (v11)")
        lines.append("-" * 78)
        lines.append(f"  Videos with shot_timestamps:   {se['n_videos_with_shot_timestamps']}")
        lines.append(f"  Total predicted shots:         {se['n_total_pred_shots']}")
        lines.append(f"  Total Hudl GT shots:           {se['n_total_gt_shots']}")
        lines.append(f"    in cv_seg-covered time:      {se['n_gt_covered']}")
        lines.append(f"    in uncovered time (cv miss): {se['n_gt_uncovered']}")
        lines.append("")
        lines.append(f"  Counts (one-to-one matching, ±3s tolerance for lenient):")
        lines.append(f"    TP (pred inside GT window):  {se['tp_strict']}")
        lines.append(f"    TP (lenient — w/ tolerance): {se['tp_lenient']}")
        lines.append(f"    FP (pred matched no GT):     {se['fp']}")
        lines.append(f"    FN (GT missed in coverage):  {se['fn']}")
        lines.append(f"    FN (GT in cv_seg miss):      {se['fn_uncovered']}")
        lines.append("")
        lines.append(f"  End-to-end (P/R/F1):           "
                     f"P={se['precision']}  R={se['recall']}  F1={se['f1']}")
        lines.append(f"  Within-coverage  (P/R/F1):     "
                     f"P={se['precision']}  R={se['recall_within_coverage']}  "
                     f"F1={se['f1_within_coverage']}")
        lines.append("")
        lines.append("  Per-video shot-eval (sorted by F1):")
        lines.append(f"    {'vID':<14}  {'TP':>4}{'FP':>4}{'FN':>4}{'FN_u':>5}  "
                     f"{'P':>5}{'R':>5}{'F1':>5}  {'R_in':>5}{'F1_in':>6}")
        sorted_rs = sorted(
            (r for r in results if r.shot_eval is not None),
            key=lambda r: r.shot_eval.f1, reverse=True,
        )
        for r in sorted_rs:
            s = r.shot_eval
            lines.append(
                f"    {r.vID:<14}  {s.tp_strict:>4}{s.fp:>4}{s.fn:>4}{s.fn_uncovered:>5}  "
                f"{s.precision:>5.2f}{s.recall:>5.2f}{s.f1:>5.2f}  "
                f"{s.recall_within_coverage:>5.2f}{s.f1_within_coverage:>6.2f}"
            )

    # v11+ window-refinement diagnostics — only printed if at least one
    # video has the refined fields present.
    refined_results = [r for r in results if r.refinement]
    if refined_results:
        lines.append("")
        lines.append("=" * 78)
        lines.append("WINDOW REFINEMENT (v11+ shot-centered tightening)")
        lines.append("-" * 78)
        lines.append("  Per video — median segment width before/after shot-centering refinement")
        lines.append(f"  {'vID':<14}  {'N':>3} {'orig med':>9} {'ref med':>9} "
                     f"{'orig mean':>10} {'ref mean':>9}  sources")
        for r in refined_results:
            ref = r.refinement
            srcs = ref.get("refinement_sources", {})
            src_str = ", ".join(
                f"{k}={v}" for k, v in sorted(srcs.items(), key=lambda kv: -kv[1])
            )
            lines.append(
                f"  {r.vID:<14}  {ref['n_threat_segments']:>3} "
                f"{ref['orig_width_median']:>8.0f}s {ref['refined_width_median']:>8.0f}s "
                f"{ref['orig_width_mean']:>9.0f}s {ref['refined_width_mean']:>8.0f}s  {src_str}"
            )

    return "\n".join(lines) + "\n"


def write_per_shot_tsv(
    results: list[VideoEvalResult],
    path:    str,
) -> int:
    """Write per-shot match diagnostics.  Returns row count.
    No-op (creates empty file) if no shot_eval data present."""
    columns = [
        "vID", "segment_start", "abs_timestamp", "timestamp_str",
        "location", "release", "outcome",
        "matched", "match_class",
        "gt_shot_start", "gt_shot_end", "gt_team",
    ]
    n_rows = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for r in results:
            if r.shot_eval is None:
                continue
            for m in r.shot_eval.matches:
                row = [
                    m.vID, m.segment_start, m.abs_timestamp, m.timestamp_str,
                    m.location, m.release, m.outcome,
                    m.matched, m.match_class,
                    m.gt_shot_start, m.gt_shot_end, m.gt_team or "",
                ]
                f.write("\t".join("" if v is None else str(v) for v in row) + "\n")
                n_rows += 1
    return n_rows


def write_per_window_tsv(
    results: list[VideoEvalResult],
    path:    str,
) -> None:
    columns = [
        "vID", "segment_start", "segment_end", "threat_color",
        "pred_shots",
        "gt_shots",            "shot_delta",
        "gt_shots_unfilt",     "shot_delta_unfilt",
        "pred_goals",
        "gt_goals",            "goal_delta",
        "gt_goals_unfilt",     "goal_delta_unfilt",
        "matched_team", "n_overlapping_total", "notes",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for r in results:
            for w in r.windows:
                row = [
                    w.vID, w.segment_start, w.segment_end,
                    w.threat_color or "",
                    w.pred_shots,
                    w.gt_shots, w.shot_delta,
                    w.gt_shots_unfiltered, w.shot_delta_unfiltered,
                    w.pred_goals,
                    w.gt_goals, w.goal_delta,
                    w.gt_goals_unfiltered, w.goal_delta_unfiltered,
                    w.matched_shots_team or "",
                    w.n_overlapping_total,
                    "; ".join(w.notes),
                ]
                f.write("\t".join("" if v is None else str(v) for v in row) + "\n")


def write_reports(
    results: list[VideoEvalResult],
    summary: dict,
    args:    argparse.Namespace,
) -> tuple[str, str, str]:
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    json_path = os.path.join(args.output_dir, f"eval_metrics_{timestamp}.json")
    txt_path  = os.path.join(args.output_dir, f"eval_metrics_{timestamp}.txt")
    tsv_path  = os.path.join(args.output_dir, f"eval_metrics_{timestamp}_per_window.tsv")
    # v11: per-shot diagnostics from shot_timestamps validation.
    per_shot_tsv = os.path.join(args.output_dir, f"eval_metrics_{timestamp}_per_shot.tsv")

    def _shot_eval_payload(r: VideoEvalResult) -> Optional[dict]:
        if r.shot_eval is None:
            return None
        s = r.shot_eval
        return {
            "n_total_pred":   s.n_total_pred,
            "n_total_gt":     s.n_total_gt,
            "n_gt_covered":   s.n_gt_covered,
            "n_gt_uncovered": s.n_gt_uncovered,
            "tp_strict":      s.tp_strict,
            "tp_lenient":     s.tp_lenient,
            "fp":             s.fp,
            "fn":             s.fn,
            "fn_uncovered":   s.fn_uncovered,
            "precision":      round(s.precision, 4),
            "recall":         round(s.recall, 4),
            "recall_within_coverage": round(s.recall_within_coverage, 4),
            "f1":             round(s.f1, 4),
            "f1_within_coverage": round(s.f1_within_coverage, 4),
        }

    json_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {
            "gt_dir":      args.gt_dir,
            "metrics_dir": args.metrics_dir,
            "cv_seg_dir":  args.cv_seg_dir,
        },
        "summary": summary,
        "videos": [
            {
                "vID":     r.vID,
                "hudl_id": r.hudl_id,
                "status":  r.status,
                "n_windows":        len(r.windows),
                "total_pred_shots": r.total_pred_shots,
                "total_gt_shots":   r.total_gt_shots,
                "total_gt_shots_unfiltered": r.total_gt_shots_unfiltered,
                "total_pred_goals": r.total_pred_goals,
                "total_gt_goals":   r.total_gt_goals,
                "total_gt_goals_unfiltered": r.total_gt_goals_unfiltered,
                "hudl_total_shots": r.hudl_total_shots,
                "hudl_total_goals": r.hudl_total_goals,
                "shot_eval":        _shot_eval_payload(r),
                "refinement":       r.refinement,
                "notes":   r.notes,
            }
            for r in results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_text_report(results, summary, args))
    write_per_window_tsv(results, tsv_path)
    n_shot_rows = write_per_shot_tsv(results, per_shot_tsv)

    return json_path, txt_path, tsv_path, per_shot_tsv, n_shot_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate metric-seg output against Hudl GT")
    p.add_argument("--vIDs", nargs="*", default=None,
                   help="Specific vIDs to evaluate. If omitted, all known vIDs are tried.")
    p.add_argument("--gt-dir", default=DEFAULT_GT_DIR,
                   help=f"Directory containing gt_{{hudl_id}}.csv files (default: {DEFAULT_GT_DIR})")
    p.add_argument("--metrics-dir", default=DEFAULT_METRICS_DIR,
                   help=f"Directory containing gt_metrics_{{vID}}.json (default: {DEFAULT_METRICS_DIR})")
    p.add_argument("--cv-seg-dir", default=DEFAULT_CV_SEG_DIR,
                   help=f"Directory containing cv_seg outputs including the "
                        f"gt_seg_{{vID}}_meta.json sidecar — read to detect "
                        f"target_filter mode and restrict GT events to the "
                        f"opponent team only when applicable. "
                        f"(default: {DEFAULT_CV_SEG_DIR})")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Where to write eval reports (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--customer-id", dest="customer_ids", default=None,
                   nargs="+", metavar="CUSTOMER_ID",
                   help="One or more customer IDs (e.g. CUST000048 "
                        "CUST000031) — for each, fetches its customer file "
                        "from GCS and merges per-vID records into a single "
                        "team→color map. Use multiple values when the eval "
                        "set spans more than one customer.")
    p.add_argument("--customer-cache-dir", default=None,
                   help="Local cache dir for customer file. Defaults to "
                        "<data_root>/customers/.")
    p.add_argument("--team-color-map", default=None,
                   help="Hand-written team→color map JSON (legacy). Used if "
                        "--customer-id is not provided.")
    p.add_argument("--no-gcs", action="store_true",
                   help="Don't download anything from GCS — fail clearly if "
                        "a needed file isn't already local.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    target_vids = args.vIDs if args.vIDs else list(VID_TO_HUDL.keys())

    # Build team→color map (same priority as cv_seg eval: customer-id
    # first, then legacy --team-color-map flag, then nothing).
    team_color_map: dict[str, dict[str, str]] = {}
    if args.customer_ids:
        cache_dir = args.customer_cache_dir or os.path.join(DEFAULT_DATA_ROOT, "customers")
        log.info(f"Loading customer file(s) for {args.customer_ids} "
                 f"(cache_dir={cache_dir})")
        for customer_id in args.customer_ids:
            records = fetch_customer_file(
                customer_id, cache_dir,
                allow_gcs_download=(not args.no_gcs),
            )
            if records:
                per_customer_map, warnings = build_team_color_map_from_customer(records)
                for w in warnings:
                    log.warning(f"  customer file ({customer_id}): {w}")
                # Detect cross-customer key collisions
                overlapping = set(per_customer_map) & set(team_color_map)
                if overlapping:
                    log.warning(
                        f"  customer file ({customer_id}): {len(overlapping)} "
                        f"vID(s) already mapped — later values win: "
                        f"{sorted(overlapping)}"
                    )
                team_color_map.update(per_customer_map)
                log.info(f"  customer file ({customer_id}): added mapping for "
                         f"{len(per_customer_map)} video(s)")
            else:
                log.warning(f"  customer file ({customer_id}) unavailable — "
                            f"videos for this customer will be evaluated "
                            f"without team filtering")
        log.info(f"  combined: team→color map covers "
                 f"{len(team_color_map)} video(s) across "
                 f"{len(args.customer_ids)} customer(s)")
    elif args.team_color_map:
        team_color_map = load_team_color_map(args.team_color_map)

    log.info(f"Evaluating {len(target_vids)} video(s)")

    results: list[VideoEvalResult] = []
    for vID in target_vids:
        log.info(f"[{vID}] evaluating...")
        try:
            res = evaluate_video(
                vID=vID,
                gt_dir=args.gt_dir,
                metrics_dir=args.metrics_dir,
                cv_seg_dir=args.cv_seg_dir,
                team_color_map_global=team_color_map,
                allow_gcs_download=(not args.no_gcs),
            )
        except Exception as e:
            log.exception(f"[{vID}] unhandled error: {e}")
            res = VideoEvalResult(vID=vID, hudl_id=VID_TO_HUDL.get(vID),
                                  status="error", notes=[f"unhandled: {e}"])
        log.info(f"[{vID}] {res.status} — windows={len(res.windows)} "
                 f"shots: pred={res.total_pred_shots} "
                 f"gt(strict)={res.total_gt_shots} "
                 f"gt(unfilt)={res.total_gt_shots_unfiltered} "
                 f"goals: pred={res.total_pred_goals} "
                 f"gt(strict)={res.total_gt_goals} "
                 f"gt(unfilt)={res.total_gt_goals_unfiltered}")
        results.append(res)

    summary = aggregate(results)
    json_path, txt_path, tsv_path, per_shot_tsv, n_shot_rows = write_reports(
        results, summary, args
    )

    log.info("=" * 60)
    log.info(f"Shots — pred={summary['total_pred_shots']}  "
             f"strict={summary['total_gt_shots']} "
             f"(diff {summary['shots_diff']:+d})  "
             f"unfilt={summary['total_gt_shots_unfiltered']} "
             f"(diff {summary['shots_diff_unfiltered']:+d})")
    log.info(f"Goals — pred={summary['total_pred_goals']}  "
             f"strict={summary['total_gt_goals']} "
             f"(diff {summary['goals_diff']:+d})  "
             f"unfilt={summary['total_gt_goals_unfiltered']} "
             f"(diff {summary['goals_diff_unfiltered']:+d})")
    sm  = summary["shot_window_metrics"]
    smu = summary["shot_window_metrics_unfiltered"]
    if sm["n"]:
        log.info(f"Per-window shot MAE — strict={sm['mae']}  unfilt={smu['mae']}")
        log.info(f"Per-window exact match — strict={sm['exact_pct']}%  "
                 f"unfilt={smu['exact_pct']}%")
    gm  = summary["goal_classifier"]
    gmu = summary["goal_classifier_unfiltered"]
    def _fmt(v): return f"{v:.3f}" if v is not None else "—"
    log.info(f"Goal classifier (strict)   : "
             f"P={_fmt(gm['precision'])} R={_fmt(gm['recall'])} F1={_fmt(gm['f1'])} "
             f"(TP={gm['tp']} FP={gm['fp']} FN={gm['fn']})")
    log.info(f"Goal classifier (unfilt)   : "
             f"P={_fmt(gmu['precision'])} R={_fmt(gmu['recall'])} F1={_fmt(gmu['f1'])} "
             f"(TP={gmu['tp']} FP={gmu['fp']} FN={gmu['fn']})")

    # v11 shot-timestamps eval summary — emitted only if present.
    se = summary.get("shot_timestamps_eval")
    if se:
        log.info(f"Shot-timestamps eval ({se['n_videos_with_shot_timestamps']} videos):"
                 f"  P={_fmt(se['precision'])} R={_fmt(se['recall'])} "
                 f"F1={_fmt(se['f1'])} "
                 f"(TP={se['tp_strict']} FP={se['fp']} "
                 f"FN={se['fn']} FN_uncov={se['fn_uncovered']})")
        log.info(f"  recall within coverage:  R={_fmt(se['recall_within_coverage'])}  "
                 f"F1={_fmt(se['f1_within_coverage'])}")
        log.info(f"  pred={se['n_total_pred_shots']} "
                 f"GT={se['n_total_gt_shots']} "
                 f"(covered={se['n_gt_covered']} uncovered={se['n_gt_uncovered']})")

    log.info(f"Reports written:")
    log.info(f"  {json_path}")
    log.info(f"  {txt_path}")
    log.info(f"  {tsv_path}")
    log.info(f"  {per_shot_tsv}  ({n_shot_rows} shot rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
