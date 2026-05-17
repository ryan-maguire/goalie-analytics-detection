#!/usr/bin/env python3
"""
eval_cv_seg_output.py — evaluate cv_seg threat-segment outputs against
Hudl-style ground-truth CSVs.

Pipeline per video
==================
1. Resolve vID → hudl_id via VID_TO_HUDL.
2. Load ground truth from <data_root>/ground_truth/gt_{hudl_id}.csv.
   Filter rows to action ∈ {Shots, Goals}, group by team, then merge
   rows inside each team whose gap to the previous merged window is
   less than --window-diff seconds. The result is the per-team list
   of ground-truth threat windows.
3. Load cv_seg prediction from <data_root>/output/cv_seg/gt_seg_{vID}.json
   (and the matching _meta.json). If the prediction file isn't on
   disk, try downloading from GCS (goalie_video_bucket /
   analyze_video/01-segment_detection/). If neither works, the video
   is reported as missing.
4. Match predicted threat windows to ground-truth windows using IoU ≥ 0.3.
   Matching is done per-team (so we can score attribution).
5. Compute precision, recall, F1, plus a confusion matrix for goalie
   attribution among matched windows.

Reports are written to <data_root>/output/evals/eval_{timestamp}.json
and .txt. Per-video and overall aggregates are included.

Where <data_root> comes from
============================
By default <data_root> is <repo_root>/data, where <repo_root> is the
directory holding this script's parent (so the script and the data
sit side by side in the repo). To use the original Linux layout
(/data/ground_truth, /data/output/...) set the environment variable
CV_SEG_DATA_ROOT=/data, or pass --gt-dir / --pred-dir / --output-dir
explicitly.

Team→colour mapping
===================
The ground-truth CSV identifies threats by team name; the cv_seg
prediction labels them by goalie colour. To score attribution we need
to know which team maps to goalie_color_a vs goalie_color_b.

Recommended: pass --customer-id CUST000048 and let the eval pull
gs://goalie_video_bucket/customerID/CUST000048.json, which contains
target/opponent team names and colors per video. The eval translates
that into the team→colour map automatically and handles multi-token
colours like "White and Green" via primary-colour fuzzy matching.

Legacy: --team-color-map points at a hand-written JSON of the form

    {
        "2069765": {                              # hudl_id (str)
            "St. Jude Knights 19U": "white",      # this team's SHOTS
            "North Shore Warhawks 19U AA": "blue" # threaten the OTHER goalie
        },
        ...
    }

A team's value is the goalie colour they SHOOT AT (i.e. the opposing
goalie's jersey colour). If a video is missing from the map the
attribution metrics for that video are skipped and the eval falls
back to windows-only matching, which is logged in the report.

If target and opponent share a primary colour (e.g. both jerseys
contain "blue") attribution is unverifiable for that video and the
eval falls back to windows-only with a warning in the report.

target_filter mode (cv_seg v23.7+)
==================================
When cv_seg is run with --target-filter (the default), its output
contains ONLY threat segments where the target goalie is being
threatened — opponent-threat and no-threat segments are dropped.
The eval detects this by reading the prediction's _meta.json sidecar
and looking for `target_filter.applied == True`. When set, the eval
restricts GT to rows from the opponent team only, so recall and
precision compare like-with-like. Without this filtering, every
target-team GT row would be a guaranteed False Negative.

The attribution confusion matrix is also degenerate in this mode
(cv_seg only emits one colour, GT is restricted to one team — every
match is "correct" by construction). This is noted in the per-video
report.

Usage
=====
    python eval_cv_seg_output.py
    python eval_cv_seg_output.py --customer-id CUST000048
    python eval_cv_seg_output.py --customer-id CUST000048 CUST000031
    python eval_cv_seg_output.py --window-diff 30 --iou-threshold 0.5
    python eval_cv_seg_output.py --vIDs SX5xNJlh6eQ KYtM20r9BuM
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# vID → Hudl ID mapping. Covers only vIDs that have customer-file
# records (i.e. videos cv_seg can actually be run against). Six entries
# were pruned 2026-05-13 (QqC7B7gKTiw, RqlWRSDFGzo, xHBKJWbsTsg,
# YnIF8qUm5RM, U7NUbWad0A8, Z8BjH4-XDdg) — those Hudl IDs were never
# paired with customer records, so they showed up as missing_prediction
# noise in every eval run. Re-add only when their customer records exist.
VID_TO_HUDL: dict[str, int] = {
    # CUST000048 (5 vIDs)
    "SX5xNJlh6eQ": 2073056, "bfEKgtOIkQU": 2072195,
    "mjEeE7p2Hz8": 2073809, "n2cy8b755Tg": 2127046,
    "v0lxSTbXfw8": 2073810,
    # CUST000031 (9 vIDs)
    "dwGsP6QKDs8": 2070269, "Fjc9hmK8_3U": 2070260,
    "HNG0jKYY12g": 2095275, "J8WkcuTsD5I": 2072194,
    "kQVdtRa4o_A": 2127034, "krxhPVLGLz8": 2108724,
    "KYtM20r9BuM": 2072196, "q5yj6sAFQeY": 2127052,
    "zOQrPK7IJ24": 2127035,
}

DEFAULT_WINDOW_DIFF   = 20
DEFAULT_IOU_THRESHOLD = 0.3

# Default data root: $CV_SEG_DATA_ROOT if set, else <repo_root>/data,
# where repo_root is the parent of the directory holding this script.
# This keeps the eval working out of the box on a developer machine
# (where '/data' is typically not writable) while still letting a
# Linux deployment point at '/data' via the env var.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
DEFAULT_DATA_ROOT  = os.environ.get("CV_SEG_DATA_ROOT") or os.path.join(_REPO_ROOT, "data")
DEFAULT_GT_DIR     = os.path.join(DEFAULT_DATA_ROOT, "ground_truth")
# Default pred-dir matches what run_pipeline.py writes when invoked
# with --local-output-dir data/output/runs (the canonical production
# layout). Override with --pred-dir to evaluate ad-hoc cv_seg runs
# written to a different location.
DEFAULT_PRED_DIR   = os.path.join(DEFAULT_DATA_ROOT, "output", "runs", "cv_seg")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATA_ROOT, "output", "evals")

GCS_BUCKET           = "goalie_video_bucket"
# Path migration: cv_seg writes to analyze_video/01-segment_detection
# now (was analysis_video/segment_detection prior to v23.7's path
# renumbering). The eval downloads from this prefix when a prediction
# isn't already on disk.
GCS_OUTPUT_PREFIX    = "analyze_video/01-segment_detection"
GCS_CUSTOMER_PREFIX  = "customerID"

THREAT_ACTIONS = {"Shots", "Goals"}

# Module logger configured in main() so import-time effects stay clean.
log = logging.getLogger("eval_cv_seg")


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class ThreatWindow:
    """A merged ground-truth or predicted threat window."""
    start: float
    end:   float
    team:  Optional[str]  = None       # ground-truth side
    color: Optional[str]  = None       # prediction side
    period: Optional[int] = None       # GT 'half' column when known
    source: Optional[str] = None       # for debugging: 'shots+goals' or pred 'source'

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class VideoEvalResult:
    vID:          str
    hudl_id:      Optional[int]
    duration:     Optional[float]   = None
    gt_windows:   list[ThreatWindow] = field(default_factory=list)
    pred_windows: list[ThreatWindow] = field(default_factory=list)

    # Window-level (merged across team/color)
    tp_window:    int = 0
    fp_window:    int = 0
    fn_window:    int = 0

    # Lenient metric: fraction of GT windows whose midpoint falls inside
    # at least one predicted window. Computed alongside the strict IoU
    # metrics so the report shows both. Bypasses the prediction-width
    # vs Hudl's-12-second-clip mismatch entirely — answers "is cv_seg
    # looking in the right place?" independent of "how big are its
    # windows?".
    gt_midpoint_hits: int = 0   # GT windows where midpoint is covered
    gt_total_for_mid: int = 0   # = len(gt_windows); duplicated for arithmetic clarity

    # Pred-side reciprocal: fraction of predictions whose midpoint
    # falls inside at least one GT window. Tells us the precision side
    # of the lenient metric.
    pred_midpoint_hits: int = 0
    pred_total_for_mid: int = 0

    # Attribution-level (only on matched windows where mapping is known)
    attr_correct: int = 0
    attr_wrong:   int = 0
    attr_skipped: int = 0  # matched but no team→color mapping available

    # Per-match diagnostics
    matches: list[dict] = field(default_factory=list)

    # Signal-trace sidecar (gt_seg_{vID}_signals.json) keyed by
    # (segment_start, segment_end). Populated when cv_seg produced
    # the file; used to attribute FPs to their originating signal.
    signal_trace: Optional[dict] = None

    # Status / errors
    status:      str = "ok"
    notes:       list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.tp_window + self.fp_window
        return self.tp_window / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp_window + self.fn_window
        return self.tp_window / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def midpoint_recall(self) -> Optional[float]:
        """Fraction of GT windows whose midpoint is covered by some prediction."""
        return (self.gt_midpoint_hits / self.gt_total_for_mid
                if self.gt_total_for_mid else None)

    @property
    def midpoint_precision(self) -> Optional[float]:
        """Fraction of predictions whose midpoint lands inside some GT window."""
        return (self.pred_midpoint_hits / self.pred_total_for_mid
                if self.pred_total_for_mid else None)

    @property
    def attribution_accuracy(self) -> Optional[float]:
        d = self.attr_correct + self.attr_wrong
        return self.attr_correct / d if d else None


# --------------------------------------------------------------------------
# Ground-truth loading
# --------------------------------------------------------------------------

def load_ground_truth_windows(
    csv_path: str,
    window_diff: int,
) -> list[ThreatWindow]:
    """
    Read a Hudl-style CSV and return merged threat windows split by team.

    Merging rule (from spec): within each team, scan rows sorted by
    start time. Begin a new window with the first Shots/Goals row.
    Extend the open window's end to the row's end whenever the gap from
    the previous row's *end* to this row's *start* is < window_diff.
    Any larger gap closes the current window and opens a new one.

    Note: rows in the CSV often overlap (each is a fixed 12-second clip
    around the event), so gap can be negative. Negative gap is always
    < window_diff, which is the intended behaviour — overlapping events
    merge.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    threat_rows = [r for r in rows if r.get("action") in THREAT_ACTIONS]
    if not threat_rows:
        return []

    # Group by team, then merge inside each team.
    by_team: dict[str, list[dict]] = defaultdict(list)
    for r in threat_rows:
        by_team[r["team"]].append(r)

    merged: list[ThreatWindow] = []
    for team, team_rows in by_team.items():
        # Sort by start to keep merging deterministic.
        team_rows.sort(key=lambda x: int(x["start"]))
        cur_start = int(team_rows[0]["start"])
        cur_end   = int(team_rows[0]["end"])
        cur_period = _safe_int(team_rows[0].get("half"))
        for r in team_rows[1:]:
            r_start = int(r["start"])
            r_end   = int(r["end"])
            gap = r_start - cur_end
            if gap < window_diff:
                # Extend current window. Period flips are unusual but
                # possible if a clip straddles the period clock — keep
                # the earliest period (matches segment behaviour).
                cur_end = max(cur_end, r_end)
            else:
                merged.append(ThreatWindow(
                    start=cur_start, end=cur_end, team=team,
                    period=cur_period, source="shots+goals",
                ))
                cur_start = r_start
                cur_end   = r_end
                cur_period = _safe_int(r.get("half"))
        merged.append(ThreatWindow(
            start=cur_start, end=cur_end, team=team,
            period=cur_period, source="shots+goals",
        ))

    merged.sort(key=lambda w: w.start)
    return merged


def _safe_int(v) -> Optional[int]:
    try:
        return int(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Customer file loading (GCS-backed team→color mapping)
# --------------------------------------------------------------------------
#
# Customer files at gs://goalie_video_bucket/customerID/{customerID}.json
# are arrays of per-video records with fields:
#
#   vID, targetGoalieTeamName, targetGoalieColor,
#        opponentGoalieTeamName, opponentGoalieColor
#
# We translate this into the per-video team→color mapping that the
# eval already understands. Important subtlety: the customer file says
# "Amherst's goalie wears black/red". The eval needs the OPPOSITE —
# "when Amherst takes a shot, they're threatening the WHITE goalie".
# So a team's value in the eval map is the OPPOSING goalie's color.

# Single-token primary colors we look for inside customer color strings
# like "White and Green". Order matters only when a string contains two
# of these (e.g. "Black and Red" → we'll capture both as a set).
_PRIMARY_COLORS = (
    "white", "black", "red", "blue", "green", "yellow",
    "orange", "purple", "pink", "gold", "silver", "grey", "gray",
    "navy", "maroon", "teal", "tan",
)


def _color_tokens(color_str: Optional[str]) -> set[str]:
    """
    Extract primary-color tokens from a customer-file color string.
    'White and Green' → {'white', 'green'}; 'Black' → {'black'}.
    Unknown words are ignored. Returns empty set on None / empty input.
    """
    if not color_str:
        return set()
    lower = color_str.lower()
    return {c for c in _PRIMARY_COLORS if c in lower}


def _customer_file_local_path(customer_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"customerID_{customer_id}.json")


def fetch_customer_file(
    customer_id: str,
    cache_dir: str,
    allow_gcs_download: bool = True,
) -> Optional[list[dict]]:
    """
    Return the parsed list of per-video records for `customer_id`,
    downloading from gs://goalie_video_bucket/customerID/{id}.json
    into cache_dir if not already on disk. Returns None on any
    failure (logged, never raised).
    """
    local_path = _customer_file_local_path(customer_id, cache_dir)

    if not os.path.exists(local_path) and allow_gcs_download:
        if not _download_customer_file(customer_id, local_path):
            return None

    if not os.path.exists(local_path):
        log.warning(f"  customer file not found locally and GCS disabled: {local_path}")
        return None

    try:
        with open(local_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"  failed to read {local_path}: {e}")
        return None

    if not isinstance(data, list):
        log.warning(f"  customer file {local_path} is not a JSON array — ignoring")
        return None
    return data


def _download_customer_file(customer_id: str, local_path: str) -> bool:
    """Best-effort GCS download. Mirrors the prediction-download pattern."""
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        log.warning("  google-cloud-storage not installed — cannot fetch customer file")
        return False

    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
    except Exception as e:
        log.warning(f"  GCS client init failed: {e}")
        return False

    blob_name = f"{GCS_CUSTOMER_PREFIX}/{customer_id}.json"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        blob = bucket.blob(blob_name)
        if not blob.exists():
            log.warning(f"  GCS: {blob_name} not present")
            return False
        blob.download_to_filename(local_path)
        log.info(f"  GCS: downloaded {blob_name} -> {local_path}")
        return True
    except Exception as e:
        log.warning(f"  GCS download error for {blob_name}: {e}")
        return False


def build_team_color_map_from_customer(
    customer_records: list[dict],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """
    Convert a customer file's array of per-video records into the
    {hudl_id_str: {team_name: opposing_goalie_color}} map the eval
    already consumes. Records are keyed by the customer file's vID,
    which we look up in VID_TO_HUDL.

    Returns (map, warnings). Warnings are emitted when:
      - a record has a vID not in VID_TO_HUDL (probably extra customer
        data we don't have ground truth for — non-fatal, just skipped)
      - target and opponent share a primary color (caller should fall
        back to windows-only for that video; we still emit the entry
        so downstream code can detect the collision and skip it)
      - either color string has no recognisable primary-color token
    """
    warnings: list[str] = []
    out: dict[str, dict[str, str]] = {}

    for rec in customer_records:
        vID    = rec.get("vID")
        target_team = rec.get("targetGoalieTeamName")
        target_col  = rec.get("targetGoalieColor")
        opp_team    = rec.get("opponentGoalieTeamName")
        opp_col     = rec.get("opponentGoalieColor")

        if not all([vID, target_team, target_col, opp_team, opp_col]):
            warnings.append(
                f"customer record incomplete (vID={vID!r}) — missing one of "
                f"target/opponent team/color"
            )
            continue

        hudl_id = VID_TO_HUDL.get(vID)
        if hudl_id is None:
            # Not in our eval set — this is fine, just skip silently
            # at INFO level rather than warning.
            log.info(f"  customer record for vID={vID} has no hudl_id mapping — skipping")
            continue

        target_tokens = _color_tokens(target_col)
        opp_tokens    = _color_tokens(opp_col)

        if not target_tokens:
            warnings.append(f"vID={vID}: targetGoalieColor {target_col!r} "
                            f"has no recognisable primary color token")
        if not opp_tokens:
            warnings.append(f"vID={vID}: opponentGoalieColor {opp_col!r} "
                            f"has no recognisable primary color token")

        # Collision detection: if the two color strings share any
        # primary-color token, attribution is ambiguous. We still
        # populate the entry so the eval can detect and report it,
        # but we mark it with a sentinel so the comparator can skip.
        shared = target_tokens & opp_tokens
        if shared:
            warnings.append(
                f"vID={vID}: target and opponent share color token(s) "
                f"{sorted(shared)} — attribution will be skipped for this video "
                f"(target={target_col!r}, opponent={opp_col!r})"
            )

        # When TARGET shoots, they threaten the OPPONENT goalie (and vice
        # versa). The map's value is the color of the threatened goalie.
        out[str(hudl_id)] = {
            target_team: opp_col,    # original strings preserved
            opp_team:    target_col,
            # Sidecar metadata used by the comparator for fuzzy matching
            # and collision skipping. Underscore prefix marks these as
            # non-team keys so any future iteration over team names can
            # filter them out trivially.
            "_target_tokens":   sorted(target_tokens),
            "_opponent_tokens": sorted(opp_tokens),
            "_collision":       sorted(shared) if shared else None,
            # Team-name sentinels — needed by callers that have to
            # restrict GT events to one team's perspective (e.g. when
            # cv_seg's target_filter is on, only the opponent's shots
            # threaten the target goalie, so only rows with
            # team == _opponent_team_name belong in the comparison).
            "_target_team_name":   target_team,
            "_opponent_team_name": opp_team,
            "_target_color":       target_col,
            "_opponent_color":     opp_col,
        }

    return out, warnings


# --------------------------------------------------------------------------
# Prediction loading (local + GCS fallback)
# --------------------------------------------------------------------------

def load_prediction(
    vID: str,
    pred_dir: str,
    allow_gcs_download: bool = True,
) -> tuple[Optional[list[dict]], Optional[dict], Optional[dict], list[str]]:
    """
    Return (segments, meta, signal_trace, notes).

    segments is the parsed JSON list from gt_seg_{vID}.json.
    meta is the parsed _meta.json sidecar, or None if missing.
    signal_trace is the parsed _signals.json sidecar (which lists the
        raw-window source signals overlapping each final threat segment),
        or None if missing. The signals sidecar is optional — older
        cv_seg outputs won't have it, and the eval should still work.
    notes captures download/skip events.
    """
    notes: list[str] = []
    seg_path    = os.path.join(pred_dir, f"gt_seg_{vID}.json")
    meta_path   = os.path.join(pred_dir, f"gt_seg_{vID}_meta.json")
    trace_path  = os.path.join(pred_dir, f"gt_seg_{vID}_signals.json")

    if not os.path.exists(seg_path) and allow_gcs_download:
        ok = _try_download_from_gcs(vID, pred_dir)
        notes.append("downloaded from GCS" if ok else "GCS download failed")

    if not os.path.exists(seg_path):
        notes.append(f"prediction file not found at {seg_path}")
        return None, None, None, notes

    try:
        with open(seg_path, encoding="utf-8") as f:
            segments = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        notes.append(f"failed to read {seg_path}: {e}")
        return None, None, None, notes

    meta: Optional[dict] = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            notes.append(f"meta sidecar unreadable ({e}) — attribution disabled")
    else:
        notes.append("meta sidecar missing — attribution disabled")

    signal_trace: Optional[dict] = None
    if os.path.exists(trace_path):
        try:
            with open(trace_path, encoding="utf-8") as f:
                signal_trace = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            notes.append(f"signals sidecar unreadable ({e}) — FP trace disabled")
    # No "missing" note here on purpose — older cv_seg outputs simply
    # don't have this file and that's fine.

    return segments, meta, signal_trace, notes


def _try_download_from_gcs(vID: str, pred_dir: str) -> bool:
    """
    Best-effort download of gt_seg_{vID}.json and _meta.json from GCS.
    Returns True iff the segments file landed on disk. Failures are
    logged but never raise.
    """
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        log.warning("  google-cloud-storage not installed — skipping GCS download")
        return False

    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
    except Exception as e:
        log.warning(f"  GCS client init failed: {e}")
        return False

    os.makedirs(pred_dir, exist_ok=True)
    targets = [
        (f"{GCS_OUTPUT_PREFIX}/gt_seg_{vID}.json",
         os.path.join(pred_dir, f"gt_seg_{vID}.json")),
        (f"{GCS_OUTPUT_PREFIX}/gt_seg_{vID}_meta.json",
         os.path.join(pred_dir, f"gt_seg_{vID}_meta.json")),
        # Signals sidecar (added in cv_seg v23.2). Older cv_seg outputs
        # in GCS won't have this — the loop tolerates absence.
        (f"{GCS_OUTPUT_PREFIX}/gt_seg_{vID}_signals.json",
         os.path.join(pred_dir, f"gt_seg_{vID}_signals.json")),
    ]

    seg_ok = False
    for blob_name, local_path in targets:
        try:
            blob = bucket.blob(blob_name)
            if not blob.exists():
                log.info(f"  GCS: {blob_name} not present")
                continue
            blob.download_to_filename(local_path)
            log.info(f"  GCS: downloaded {blob_name} -> {local_path}")
            if blob_name.endswith(f"gt_seg_{vID}.json"):
                seg_ok = True
        except Exception as e:
            log.warning(f"  GCS download error for {blob_name}: {e}")

    return seg_ok


def predicted_threat_windows(
    segments: list[dict],
) -> list[ThreatWindow]:
    """Convert cv_seg's full timeline to threat-only ThreatWindow objects."""
    out: list[ThreatWindow] = []
    for seg in segments:
        if not seg.get("segmentHasThreat"):
            continue
        out.append(ThreatWindow(
            start=float(seg["segment_start"]),
            end=float(seg["segment_end"]),
            color=seg.get("threat_goalie_color"),
            source=seg.get("_attribution_src", "pred"),
        ))
    return out


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def iou(a: ThreatWindow, b: ThreatWindow) -> float:
    inter_start = max(a.start, b.start)
    inter_end   = min(a.end,   b.end)
    inter = max(0.0, inter_end - inter_start)
    union = a.duration + b.duration - inter
    return inter / union if union > 0 else 0.0


def pad_window(w: ThreatWindow, pad: float) -> ThreatWindow:
    """
    Return a new ThreatWindow with start/end expanded by `pad` seconds
    on each side, clamped at zero. Used to widen 12-second Hudl GT
    clips before IoU so they're comparable in width to cv_seg's
    naturally wider predicted windows. Metadata (team, color, period,
    source) is preserved.
    """
    if pad <= 0:
        return w
    return ThreatWindow(
        start=max(0.0, w.start - pad),
        end=w.end + pad,
        team=w.team, color=w.color,
        period=w.period, source=w.source,
    )


def compute_midpoint_hits(
    gt: list[ThreatWindow],
    pred: list[ThreatWindow],
) -> tuple[int, int]:
    """
    Return (gt_hits, pred_hits) where:
      gt_hits   = # GT windows whose midpoint lies inside SOME prediction
      pred_hits = # predictions whose midpoint lies inside SOME GT window

    These are deliberately many-to-many — a single wide prediction
    can cover multiple GT midpoints (and would, on the v23 cv_seg
    output where MAX_OPEN_WINDOW_SEC=120 produces blobs that swallow
    several Hudl shots). That's the right accounting for "did cv_seg
    look in the right place?" — a 120-second prediction that covers
    three GT midpoints really did flag all three threats, even if
    its IoU with each is below 0.3.
    """
    gt_hits = 0
    for g in gt:
        gm = 0.5 * (g.start + g.end)
        for p in pred:
            if p.start <= gm <= p.end:
                gt_hits += 1
                break

    pred_hits = 0
    for p in pred:
        pm = 0.5 * (p.start + p.end)
        for g in gt:
            if g.start <= pm <= g.end:
                pred_hits += 1
                break

    return gt_hits, pred_hits


def greedy_match(
    gt: list[ThreatWindow],
    pred: list[ThreatWindow],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """
    Greedy 1-to-1 matching: pair (gt_i, pred_j) with the highest IoU first,
    drop both, repeat. Returns (matched_pairs, unmatched_gt, unmatched_pred).

    Greedy on the highest-IoU pair is optimal-ish for sparse, mostly
    non-overlapping windows and keeps the eval report trivially
    explainable. Hungarian assignment would change ≤1% of typical
    cases here while doubling the code.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            score = iou(g, p)
            if score >= iou_threshold:
                candidates.append((score, i, j))
    candidates.sort(reverse=True)  # highest IoU first

    matched_gt:   set[int] = set()
    matched_pred: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for score, i, j in candidates:
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        pairs.append((i, j, score))

    unmatched_gt   = set(range(len(gt)))   - matched_gt
    unmatched_pred = set(range(len(pred))) - matched_pred
    return pairs, unmatched_gt, unmatched_pred


# --------------------------------------------------------------------------
# Per-video evaluation
# --------------------------------------------------------------------------

def evaluate_video(
    vID: str,
    gt_dir: str,
    pred_dir: str,
    window_diff: int,
    iou_threshold: float,
    team_to_color_global: dict[str, dict[str, str]],
    allow_gcs_download: bool = True,
    gt_pad: float = 0.0,
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
        result.notes.append(f"ground-truth not found: {gt_path}")
        return result

    try:
        gt_windows = load_ground_truth_windows(gt_path, window_diff)
    except Exception as e:
        result.status = "error"
        result.notes.append(f"failed to load ground truth: {e}")
        return result
    # Store the RAW (un-padded) GT windows for the report and diagnostics
    # so users see the source-of-truth boundaries, not the padded ones.
    result.gt_windows = gt_windows

    segments, meta, signal_trace, pred_notes = load_prediction(
        vID, pred_dir, allow_gcs_download=allow_gcs_download
    )
    result.notes.extend(pred_notes)
    if segments is None:
        result.status = "missing_prediction"
        return result

    pred_windows = predicted_threat_windows(segments)
    result.pred_windows = pred_windows
    result.signal_trace = signal_trace
    if meta:
        result.duration = meta.get("video_duration_sec")

    # ── target_filter detection ───────────────────────────────────────
    # cv_seg v23.7+ supports --no-target-filter; the default is ON, which
    # means the prediction file contains ONLY segments where the target
    # goalie is being threatened. Opponent-threat segments are dropped
    # before write. To make recall/precision meaningful in that mode,
    # we have to drop the corresponding opponent-shoots GT rows too —
    # otherwise every opponent-team GT row is a guaranteed False
    # Negative and recall craters by ~50% for no real reason.
    #
    # The cv_seg meta sidecar records the filter outcome under
    # "target_filter": {"applied": bool, "target_color": str, ...}.
    # When applied, the only GT events that should match are those from
    # the OPPONENT team (since their shots threaten our target goalie).
    target_filter_applied = bool(
        meta and meta.get("target_filter", {}).get("applied")
    )

    # Resolve team→color mapping early so we can use it for filtering
    # AND for later attribution scoring. Same code as before, just
    # moved up.
    team_map = team_to_color_global.get(str(hudl_id)) or team_to_color_global.get(hudl_id)
    have_attr_mapping = team_map is not None

    collision: Optional[list] = None
    if have_attr_mapping and isinstance(team_map.get("_collision"), list):
        collision = team_map["_collision"]
        have_attr_mapping = False
        result.notes.append(
            f"target/opponent share color token(s) {collision} — "
            f"attribution cannot be verified, skipped for this video"
        )
    elif not have_attr_mapping:
        result.notes.append("no team→color mapping — attribution skipped")

    # Apply target-filter GT restriction if the prediction was generated
    # in target-filter mode. We need a team→color mapping to know which
    # GT rows belong to the opponent team.
    if target_filter_applied:
        if team_map is None:
            # Without the customer file we can't restrict GT by team.
            # The eval will run, but recall will be artificially low
            # because opponent-team GT rows still count as FNs.
            result.notes.append(
                "target_filter is ON in prediction but no team→color map "
                "available — recall will be artificially low (opponent-team "
                "GT rows have no matching predictions)"
            )
        else:
            opponent_team_name = team_map.get("_opponent_team_name")
            if opponent_team_name:
                pre_count = len(gt_windows)
                # Fuzzy match — handles suffix differences like
                # 'Team South Dakota' (customer file) vs
                # 'Team South Dakota 19U' (Hudl CSV). See
                # _team_names_match for the matching rules.
                gt_windows = [w for w in gt_windows
                              if _team_names_match(w.team, opponent_team_name)]
                result.gt_windows = gt_windows  # update stored reference
                result.notes.append(
                    f"target_filter mode: GT restricted to opponent team "
                    f"{opponent_team_name!r} only "
                    f"({pre_count} → {len(gt_windows)} windows); "
                    f"target-team GT rows have no matching predictions "
                    f"by design"
                )
            else:
                # Old customer-file map without the new sentinel keys.
                # Don't filter — log so it's obvious.
                result.notes.append(
                    "target_filter is ON in prediction but team_map lacks "
                    "_opponent_team_name sentinel (old customer-file format) — "
                    "recall will be artificially low"
                )

    # ── Lenient metric: midpoint coverage ─────────────────────────────
    # Computed against the RAW (un-padded) GT and predictions. The
    # question this answers is "did cv_seg fire near each GT shot,
    # and are most predictions near a real shot?" — which is the
    # cleanest signal of model behaviour independent of window-width
    # mismatches.
    gt_hits, pred_hits = compute_midpoint_hits(gt_windows, pred_windows)
    result.gt_midpoint_hits   = gt_hits
    result.gt_total_for_mid   = len(gt_windows)
    result.pred_midpoint_hits = pred_hits
    result.pred_total_for_mid = len(pred_windows)

    # ── Strict metric: IoU-based matching ─────────────────────────────
    # Optionally pad GT windows symmetrically. Hudl publishes 12-second
    # clips around each shot, but the actual threat extends earlier
    # (zone entry, build-up) and later (rebound coverage). gt_pad lets
    # the eval account for that without changing the underlying GT.
    gt_for_match = [pad_window(g, gt_pad) for g in gt_windows]
    if gt_pad > 0:
        result.notes.append(f"matching GT padded by ±{gt_pad:g}s")

    # team_map and have_attr_mapping were resolved earlier (above), so
    # we can use them here for window matching and attribution scoring.

    # Window matching: do it once across all teams/colors so a missed
    # GT window from team A can't be 'matched' by a pred window assigned
    # to team B. We require both window overlap AND (when mapping is
    # available) team/color agreement.
    pairs, unmatched_gt, unmatched_pred = greedy_match(
        gt_for_match, pred_windows, iou_threshold
    )

    # Convert pairs into TPs vs attribution-mismatch downgrades.
    for gi, pj, score in pairs:
        gw = gt_windows[gi]   # raw GT shown to user
        pw = pred_windows[pj]
        # When gt_pad>0, `score` reflects IoU of (padded GT, pred). Keep
        # both numbers in the record so the diagnostic TSV makes the
        # padding effect transparent.
        raw_iou = iou(gw, pw)
        match_record = {
            "gt_start":   gw.start,
            "gt_end":     gw.end,
            "gt_team":    gw.team,
            "pred_start": pw.start,
            "pred_end":   pw.end,
            "pred_color": pw.color,
            "iou":        round(score, 3),
            "iou_raw":    round(raw_iou, 3) if gt_pad > 0 else None,
            "attribution": "skipped",
        }

        # Fuzzy team-name lookup: customer files use the canonical team
        # name; the GT CSV may have a slight variant (e.g. spaces,
        # punctuation). Try exact first, then case-insensitive, then
        # token-overlap heuristic.
        expected_color = _lookup_team_color(team_map, gw.team) if have_attr_mapping else None

        if have_attr_mapping and expected_color is not None:
            actual_color   = pw.color
            if actual_color is None:
                match_record["attribution"] = "missing_color"
                result.attr_skipped += 1
            elif _colors_match(actual_color, expected_color):
                match_record["attribution"] = "correct"
                result.attr_correct += 1
            else:
                match_record["attribution"] = f"wrong (expected {expected_color})"
                result.attr_wrong += 1
        else:
            result.attr_skipped += 1

        result.tp_window += 1
        result.matches.append(match_record)

    result.fn_window = len(unmatched_gt)
    result.fp_window = len(unmatched_pred)

    # In target-filter mode the attribution confusion matrix is
    # logically degenerate: cv_seg only emitted target-color segments,
    # and we've already restricted GT to the opponent-team rows that
    # threaten that target color. So every matched pair is correct by
    # construction — high attr_correct counts are expected and don't
    # reflect anything cv_seg "got right". Note this explicitly so the
    # report doesn't mislead.
    if target_filter_applied and have_attr_mapping:
        result.notes.append(
            "attribution counts are vacuous in target_filter mode "
            "(cv_seg emits only one color, GT is pre-restricted to one "
            "team — every match is 'correct' by construction)"
        )

    # Capture the unmatched windows for the report so failure modes are
    # explorable without re-running.
    for gi in sorted(unmatched_gt):
        gw = gt_windows[gi]
        result.matches.append({
            "gt_start": gw.start, "gt_end": gw.end, "gt_team": gw.team,
            "pred_start": None, "pred_end": None, "pred_color": None,
            "iou": 0.0, "attribution": "n/a (FN)",
        })
    for pj in sorted(unmatched_pred):
        pw = pred_windows[pj]
        result.matches.append({
            "gt_start": None, "gt_end": None, "gt_team": None,
            "pred_start": pw.start, "pred_end": pw.end, "pred_color": pw.color,
            "iou": 0.0, "attribution": "n/a (FP)",
        })

    return result


def _colors_match(predicted: Optional[str], expected: Optional[str]) -> bool:
    """
    Compare cv_seg's predicted color against a customer-file color string.

    cv_seg outputs single tokens like 'white' or 'blue'; the customer
    file gives multi-token strings like 'White and Green' or 'Blue and
    Red'. A match means the predicted token appears as a primary-color
    token inside the expected string. Falls back to exact-string-equal
    when neither side parses to a known primary color (so a custom
    color like 'crimson' still works if cv_seg and the customer file
    agree on the literal spelling).
    """
    if predicted is None or expected is None:
        return False
    pred_tokens = _color_tokens(predicted)
    exp_tokens  = _color_tokens(expected)
    if pred_tokens and exp_tokens:
        return bool(pred_tokens & exp_tokens)
    # Fallback: exact compare for colors we don't recognise.
    return predicted.strip().lower() == expected.strip().lower()


def _team_name_tokens(name: Optional[str]) -> set:
    """Tokenise a team-name string for fuzzy matching. Lowercase,
    whitespace-split, drop empties. Trailing punctuation is stripped
    from each token so 'jr.' and 'jr' compare equal — Hudl CSVs
    sometimes write "Philadelphia Jr. Flyers 19U AA" where the
    customer file has "Jr Flyers 19U". Empty input → empty set."""
    if not name:
        return set()
    raw_tokens = name.lower().strip().split()
    cleaned = (t.rstrip(".,;:!?") for t in raw_tokens)
    return {t for t in cleaned if t}


def _team_names_match(a: Optional[str], b: Optional[str]) -> bool:
    """Tolerant team-name comparison.

    Returns True if `a` and `b` refer to the same team across the
    customer-file/Hudl boundary. Returns False otherwise.

    The comparison is symmetric and uses three rules in order:
      1. Exact match (after stripping whitespace) — the common case
      2. Case-insensitive exact match
      3. Token-set subset: one name's tokens are entirely contained in
         the other's, AND both names have at least 2 tokens. This
         handles the suffix mismatches we've actually observed:
           - 'Team South Dakota'  ≈  'Team South Dakota 19U'
           - 'Northshore Warhawks 19U'  ≈  'North Shore Warhawks 19U AA'
         The ≥2 tokens guard prevents 'Team' alone from matching
         'Team South Dakota 19U'.

    Note: this is intentionally narrower than _lookup_team_color's
    50%-overlap rule. The subset rule rejects partial overlaps that
    could be false positives (e.g. 'Boston Hockey 19U' vs 'Boston
    Pride 19U' share 'boston' and '19u' but are different teams).
    """
    if a is None or b is None:
        return False

    a_clean = a.strip()
    b_clean = b.strip()
    if not a_clean or not b_clean:
        return False

    # Rule 1: exact (preserves prior behaviour for callers that used ==)
    if a_clean == b_clean:
        return True

    # Rule 2: case-insensitive
    if a_clean.lower() == b_clean.lower():
        return True

    # Rule 3: token-subset, both sides ≥2 tokens
    a_tok = _team_name_tokens(a_clean)
    b_tok = _team_name_tokens(b_clean)
    if len(a_tok) < 2 or len(b_tok) < 2:
        return False
    return a_tok.issubset(b_tok) or b_tok.issubset(a_tok)


def _lookup_team_color(team_map: dict, team_name: Optional[str]) -> Optional[str]:
    """
    Resolve a GT team name to an expected goalie color from the team_map.
    Tries: exact match, case-insensitive match, then a token-overlap
    heuristic (≥ 50% of the longer name's tokens shared) so 'Northshore
    Warhawks 19U' matches 'North Shore Warhawks 19U AA'. Returns None
    when no candidate clears the bar.

    Sentinel keys (those starting with '_') are skipped.
    """
    if team_name is None:
        return None
    real_keys = {k: v for k, v in team_map.items() if not k.startswith("_")}

    if team_name in real_keys:
        return real_keys[team_name]

    target = team_name.lower().strip()
    for k, v in real_keys.items():
        if k.lower().strip() == target:
            return v

    # Token overlap: tokenize on whitespace, compare set sizes. Pick the
    # candidate with the highest overlap that's also at least 50% of the
    # longer name's token count. This is intentionally generous because
    # team names in different sources can differ by spacing (Northshore
    # vs North Shore), suffix (19U vs 19U AA), or punctuation.
    target_tokens = set(target.split())
    if not target_tokens:
        return None
    best_v: Optional[str] = None
    best_overlap = 0.0
    for k, v in real_keys.items():
        k_tokens = set(k.lower().split())
        if not k_tokens:
            continue
        overlap = len(target_tokens & k_tokens)
        threshold = 0.5 * max(len(target_tokens), len(k_tokens))
        if overlap >= threshold and overlap > best_overlap:
            best_overlap = overlap
            best_v = v
    return best_v


# --------------------------------------------------------------------------
# Diagnostic dump
# --------------------------------------------------------------------------

# Classification thresholds for unmatched items. These are not eval
# metrics — they exist purely to label diagnostic rows so a human can
# scan the TSV and see "scale issue" vs "missed entirely" at a glance.
DIAG_NEAR_IOU_LOW   = 0.05  # below this counts as "no spatial overlap"
DIAG_NEAR_IOU_HIGH  = 0.30  # the eval's match threshold; we mirror it
DIAG_NEAR_TIME_SEC  = 30    # within this many seconds of midpoint = "nearby"


def _midpoint(w: ThreatWindow) -> float:
    return 0.5 * (w.start + w.end)


def _nearest_by_midpoint(
    target: ThreatWindow,
    pool: list[ThreatWindow],
    exclude_indices: set[int],
) -> tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Return (index, time_delta_sec, iou) of the nearest pool item by
    midpoint distance. Returns (None, None, None) if pool is empty
    (after exclusions).
    """
    target_mid = _midpoint(target)
    best_i: Optional[int]      = None
    best_dist: Optional[float] = None
    best_iou:  Optional[float] = None
    for i, candidate in enumerate(pool):
        if i in exclude_indices:
            continue
        dist = abs(_midpoint(candidate) - target_mid)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_i    = i
            best_iou  = iou(target, candidate)
    return best_i, best_dist, best_iou


def _classify_fn(nearest_iou: Optional[float],
                 nearest_dist: Optional[float]) -> str:
    """
    Classify an unmatched GT window by what's wrong with the closest pred:
      no_pred                 — pool was empty
      missed_entirely         — nearest pred is far away in time
      scale_mismatch          — nearest pred overlaps but IoU < threshold
      no_spatial_overlap      — nearest pred is close in time but doesn't overlap
    """
    if nearest_iou is None:
        return "no_pred"
    if nearest_dist is not None and nearest_dist > DIAG_NEAR_TIME_SEC:
        return "missed_entirely"
    if nearest_iou >= DIAG_NEAR_IOU_LOW:
        return "scale_mismatch"
    return "no_spatial_overlap"


def _classify_fp(nearest_iou: Optional[float],
                 nearest_dist: Optional[float]) -> str:
    """
    Classify an unmatched prediction by what's likely wrong:
      no_gt                   — pool was empty
      spurious                — nearest GT is far away (cv_seg fired with no nearby shot)
      scale_mismatch          — overlaps a GT window but IoU < threshold
      adjacent_to_gt          — close in time but no overlap (e.g. cv_seg slightly missed)
    """
    if nearest_iou is None:
        return "no_gt"
    if nearest_dist is not None and nearest_dist > DIAG_NEAR_TIME_SEC:
        return "spurious"
    if nearest_iou >= DIAG_NEAR_IOU_LOW:
        return "scale_mismatch"
    return "adjacent_to_gt"


def build_diagnostics(results: list[VideoEvalResult]) -> tuple[list[dict], dict]:
    """
    Produce one diagnostic record per unmatched item (FN or FP) across
    all videos, plus a summary tally of how those unmatched items
    classify. The summary is what you actually look at first; the
    per-row table is for drilling in.

    For FNs, the nearest-prediction lookup excludes predictions that
    were already matched to some other GT window — otherwise a single
    wide prediction could appear as "the nearest prediction" for two
    different GT windows and confuse the picture.
    """
    rows: list[dict] = []

    fn_class_count: Counter = Counter()
    fp_class_count: Counter = Counter()
    fp_source_when_spurious: Counter = Counter()  # which signal source over-fires?
    scale_iou_samples: list[float] = []           # for distribution awareness

    for res in results:
        if res.status not in ("ok",):
            continue

        # Re-derive matched indices from the matches list so we know
        # which preds were "claimed" by some GT.
        matched_pred_starts = {
            (m["pred_start"], m["pred_end"])
            for m in res.matches
            if m.get("pred_start") is not None and m.get("attribution") not in ("n/a (FP)",)
            and m.get("gt_start") is not None  # actual matches only
        }

        matched_pred_idx: set[int] = set()
        for j, pw in enumerate(res.pred_windows):
            if (pw.start, pw.end) in matched_pred_starts:
                matched_pred_idx.add(j)

        matched_gt_starts = {
            (m["gt_start"], m["gt_end"])
            for m in res.matches
            if m.get("gt_start") is not None and m.get("pred_start") is not None
        }
        matched_gt_idx: set[int] = set()
        for i, gw in enumerate(res.gt_windows):
            if (gw.start, gw.end) in matched_gt_starts:
                matched_gt_idx.add(i)

        # ── Unmatched GT windows (FNs) ──────────────────────────────
        for i, gw in enumerate(res.gt_windows):
            if i in matched_gt_idx:
                continue
            ni, ndist, niou = _nearest_by_midpoint(
                gw, res.pred_windows, exclude_indices=matched_pred_idx
            )
            cls = _classify_fn(niou, ndist)
            fn_class_count[cls] += 1
            if cls == "scale_mismatch" and niou is not None:
                scale_iou_samples.append(niou)

            row = {
                "vID":        res.vID,
                "kind":       "FN",
                "class":      cls,
                "gt_start":   gw.start,
                "gt_end":     gw.end,
                "gt_team":    gw.team,
                "gt_period":  gw.period,
                "pred_start": res.pred_windows[ni].start if ni is not None else None,
                "pred_end":   res.pred_windows[ni].end   if ni is not None else None,
                "pred_color": res.pred_windows[ni].color if ni is not None else None,
                "pred_src":   res.pred_windows[ni].source if ni is not None else None,
                "midpoint_delta_sec": round(ndist, 1) if ndist is not None else None,
                "iou":        round(niou, 3) if niou is not None else None,
            }
            rows.append(row)

        # ── Unmatched predictions (FPs) ─────────────────────────────
        for j, pw in enumerate(res.pred_windows):
            if j in matched_pred_idx:
                continue
            ni, ndist, niou = _nearest_by_midpoint(
                pw, res.gt_windows, exclude_indices=matched_gt_idx
            )
            cls = _classify_fp(niou, ndist)
            fp_class_count[cls] += 1
            if cls == "spurious" and pw.source:
                fp_source_when_spurious[pw.source] += 1
            if cls == "scale_mismatch" and niou is not None:
                scale_iou_samples.append(niou)

            row = {
                "vID":        res.vID,
                "kind":       "FP",
                "class":      cls,
                "gt_start":   res.gt_windows[ni].start if ni is not None else None,
                "gt_end":     res.gt_windows[ni].end   if ni is not None else None,
                "gt_team":    res.gt_windows[ni].team  if ni is not None else None,
                "gt_period":  res.gt_windows[ni].period if ni is not None else None,
                "pred_start": pw.start,
                "pred_end":   pw.end,
                "pred_color": pw.color,
                "pred_src":   pw.source,
                "midpoint_delta_sec": round(ndist, 1) if ndist is not None else None,
                "iou":        round(niou, 3) if niou is not None else None,
            }
            rows.append(row)

    # Distribution of near-miss IoUs is the single most useful signal
    # for "would lowering the threshold help?" questions.
    iou_dist: Optional[dict] = None
    if scale_iou_samples:
        scale_iou_samples.sort()
        n = len(scale_iou_samples)
        iou_dist = {
            "n":        n,
            "min":      round(scale_iou_samples[0], 3),
            "p25":      round(scale_iou_samples[n // 4], 3),
            "median":   round(scale_iou_samples[n // 2], 3),
            "p75":      round(scale_iou_samples[(3 * n) // 4], 3),
            "max":      round(scale_iou_samples[-1], 3),
        }

    summary = {
        "fn_classification":        dict(fn_class_count),
        "fp_classification":        dict(fp_class_count),
        "fp_spurious_by_source":    dict(fp_source_when_spurious),
        "near_miss_iou_distribution": iou_dist,
    }
    return rows, summary


def write_diagnostics_tsv(rows: list[dict], path: str) -> None:
    """Write diagnostic records as a TSV that opens cleanly in Excel/Numbers."""
    columns = [
        "vID", "kind", "class",
        "gt_start", "gt_end", "gt_team", "gt_period",
        "pred_start", "pred_end", "pred_color", "pred_src",
        "midpoint_delta_sec", "iou",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(
                "" if row.get(c) is None else str(row.get(c))
                for c in columns
            ) + "\n")


# --------------------------------------------------------------------------
# FP signal trace
# --------------------------------------------------------------------------
#
# Each cv_seg prediction carries (in the _signals.json sidecar) the
# set of raw-window source signals — motion, goal_light, faceoff,
# crowd_roar, celebration, activity_spike, motion_auto_close, motion_eof
# — that contributed to the final segment. For false positives, the
# source-signal combination is the most actionable diagnostic: it
# tells us which upstream signal is over-firing.

def _trace_lookup(
    trace: Optional[dict],
    pw: ThreatWindow,
) -> tuple[list[str], int]:
    """
    Find the trace entry whose (segment_start, segment_end) matches pw.
    Returns (source_signals, n_overlapping_raw). Returns ([], 0) when
    no trace is available or no exact match is found.

    Match is by half-open interval equality with a 0.5s tolerance to
    accommodate the float vs int representation differences between
    cv_seg's output and the eval's window dataclass.
    """
    if not trace:
        return [], 0
    segments = trace.get("segments") or []
    for entry in segments:
        if (abs(entry["segment_start"] - pw.start) < 0.5
                and abs(entry["segment_end"] - pw.end) < 0.5):
            return list(entry.get("source_signals") or []), int(entry.get("n_overlapping_raw") or 0)
    return [], 0


def build_fp_trace(
    results: list[VideoEvalResult],
) -> tuple[list[dict], dict]:
    """
    For every FP prediction across all videos, look up its source
    signals from the cv_seg signal trace and build:
      rows    — one record per FP for the TSV dump
      summary — aggregate counts by source-signal combination, plus
                per-video top-source breakdown

    A "source combination" is the sorted tuple of source tags joined
    with '+'. So a window hit by motion alone is "motion"; one hit by
    motion AND celebration is "celebration+motion". Counting by
    combination (rather than just by individual signal) is the more
    useful aggregation because it tells you which signals are
    co-firing on bogus windows vs which are firing alone.
    """
    rows: list[dict] = []
    by_combo: Counter = Counter()
    by_combo_video: dict[str, Counter] = {}
    individual_signals: Counter = Counter()  # raw count per signal regardless of co-firing
    no_trace_count = 0

    for res in results:
        if res.status != "ok":
            continue

        # Re-derive matched-pred indices from the matches list.
        matched_pred_starts = {
            (m["pred_start"], m["pred_end"])
            for m in res.matches
            if m.get("pred_start") is not None
            and m.get("gt_start") is not None
        }

        for pw in res.pred_windows:
            if (pw.start, pw.end) in matched_pred_starts:
                continue  # this pred was matched to a GT — not an FP

            sources, n_raw = _trace_lookup(res.signal_trace, pw)
            if not sources and not res.signal_trace:
                # No trace at all for this video — skip the source
                # accounting but still record the FP. Tracked
                # separately so the summary surfaces the gap.
                no_trace_count += 1

            combo = "+".join(sources) if sources else "(no_trace)"
            by_combo[combo] += 1
            by_combo_video.setdefault(res.vID, Counter())[combo] += 1
            for s in sources:
                individual_signals[s] += 1

            rows.append({
                "vID":         res.vID,
                "pred_start":  pw.start,
                "pred_end":    pw.end,
                "pred_dur":    round(pw.end - pw.start, 1),
                "pred_color":  pw.color,
                "attr_src":    pw.source,         # cv_seg's _attribution_src
                "source_combo": combo,
                "n_raw":       n_raw,
            })

    summary = {
        "fp_total":                 len(rows),
        "fp_by_source_combo":       dict(by_combo.most_common()),
        "fp_by_individual_signal":  dict(individual_signals.most_common()),
        "fp_top_combo_per_video": {
            vID: counter.most_common(3)
            for vID, counter in by_combo_video.items()
        },
        "fps_without_trace":        no_trace_count,
    }
    return rows, summary


def write_fp_trace_tsv(rows: list[dict], path: str) -> None:
    """One row per FP. Same shape rules as the existing diagnostics TSV."""
    columns = [
        "vID", "pred_start", "pred_end", "pred_dur", "pred_color",
        "attr_src", "source_combo", "n_raw",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(
                "" if row.get(c) is None else str(row.get(c))
                for c in columns
            ) + "\n")


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

def aggregate(results: list[VideoEvalResult]) -> dict:
    """Sum TP/FP/FN across videos and compute global P/R/F1 + attribution."""
    tp = sum(r.tp_window for r in results)
    fp = sum(r.fp_window for r in results)
    fn = sum(r.fn_window for r in results)
    attr_c = sum(r.attr_correct for r in results)
    attr_w = sum(r.attr_wrong   for r in results)
    attr_s = sum(r.attr_skipped for r in results)

    # Midpoint-coverage accumulators
    mh_gt   = sum(r.gt_midpoint_hits   for r in results)
    mh_gt_n = sum(r.gt_total_for_mid   for r in results)
    mh_pr   = sum(r.pred_midpoint_hits for r in results)
    mh_pr_n = sum(r.pred_total_for_mid for r in results)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    attr_acc = attr_c / (attr_c + attr_w) if (attr_c + attr_w) else None

    mid_recall    = mh_gt / mh_gt_n if mh_gt_n else None
    mid_precision = mh_pr / mh_pr_n if mh_pr_n else None
    mid_f1: Optional[float] = None
    if mid_recall is not None and mid_precision is not None and (mid_recall + mid_precision):
        mid_f1 = 2 * mid_recall * mid_precision / (mid_recall + mid_precision)

    statuses = Counter(res.status for res in results)

    return {
        "videos_total":              len(results),
        "videos_by_status":          dict(statuses),
        "tp_window":                 tp,
        "fp_window":                 fp,
        "fn_window":                 fn,
        "precision":                 round(p, 4),
        "recall":                    round(r, 4),
        "f1":                        round(f1, 4),
        "midpoint_recall":           round(mid_recall, 4)    if mid_recall    is not None else None,
        "midpoint_precision":        round(mid_precision, 4) if mid_precision is not None else None,
        "midpoint_f1":               round(mid_f1, 4)        if mid_f1        is not None else None,
        "midpoint_gt_hits":          mh_gt,
        "midpoint_gt_total":         mh_gt_n,
        "midpoint_pred_hits":        mh_pr,
        "midpoint_pred_total":       mh_pr_n,
        "attribution_correct":       attr_c,
        "attribution_wrong":         attr_w,
        "attribution_skipped":       attr_s,
        "attribution_accuracy":      round(attr_acc, 4) if attr_acc is not None else None,
    }


def render_text_report(
    results: list[VideoEvalResult],
    summary: dict,
    args: argparse.Namespace,
    diagnostics_summary: Optional[dict] = None,
    fp_trace_summary: Optional[dict] = None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("cv_seg evaluation report")
    lines.append(f"  generated:       {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"  window_diff:     {args.window_diff}s")
    lines.append(f"  iou_threshold:   {args.iou_threshold}")
    lines.append(f"  gt_pad:          ±{args.gt_pad:g}s")
    lines.append(f"  gt_dir:          {args.gt_dir}")
    lines.append(f"  pred_dir:        {args.pred_dir}")
    lines.append("=" * 78)

    lines.append("")
    lines.append("OVERALL")
    lines.append("-" * 78)
    lines.append(f"  videos:               {summary['videos_total']} "
                 f"(by status: {summary['videos_by_status']})")
    lines.append("")
    lines.append(f"  ── STRICT (IoU ≥ {args.iou_threshold}"
                 f"{', GT padded' if args.gt_pad > 0 else ''}) ──")
    lines.append(f"  TP / FP / FN:         {summary['tp_window']} / "
                 f"{summary['fp_window']} / {summary['fn_window']}")
    lines.append(f"  precision:            {summary['precision']:.4f}")
    lines.append(f"  recall:               {summary['recall']:.4f}")
    lines.append(f"  F1:                   {summary['f1']:.4f}")

    # Lenient block: midpoint coverage. Independent of IoU/gt_pad — it
    # always answers "did cv_seg fire near each shot" using raw GT.
    lines.append("")
    lines.append("  ── LENIENT (midpoint coverage, raw GT) ──")
    if summary["midpoint_recall"] is not None:
        lines.append(f"  midpoint recall:      {summary['midpoint_recall']:.4f} "
                     f"({summary['midpoint_gt_hits']} / "
                     f"{summary['midpoint_gt_total']} GT windows had a pred over their midpoint)")
    if summary["midpoint_precision"] is not None:
        lines.append(f"  midpoint precision:   {summary['midpoint_precision']:.4f} "
                     f"({summary['midpoint_pred_hits']} / "
                     f"{summary['midpoint_pred_total']} preds landed midpoint inside a GT window)")
    if summary["midpoint_f1"] is not None:
        lines.append(f"  midpoint F1:          {summary['midpoint_f1']:.4f}")

    lines.append("")
    lines.append("  ── ATTRIBUTION ──")
    if summary["attribution_accuracy"] is not None:
        lines.append(f"  attribution accuracy: {summary['attribution_accuracy']:.4f} "
                     f"({summary['attribution_correct']} / "
                     f"{summary['attribution_correct'] + summary['attribution_wrong']})")
    else:
        lines.append("  attribution accuracy: n/a (no team→color mapping for any video)")
    if summary["attribution_skipped"]:
        lines.append(f"  attribution skipped:  {summary['attribution_skipped']} "
                     f"(no mapping or pred color unavailable)")

    lines.append("")
    lines.append("PER VIDEO")
    lines.append("-" * 78)
    header = (f"  {'vID':<13}{'hudl':>9}  {'status':<18}"
              f"{'GT':>4}{'Pred':>5}{'TP':>4}{'FP':>4}{'FN':>4}"
              f"{'P':>7}{'R':>7}{'F1':>7}{'mRec':>7}{'mPre':>7}{'Attr':>7}")
    lines.append(header)
    for res in results:
        attr = res.attribution_accuracy
        attr_str = f"{attr:.2f}" if attr is not None else "  -"
        mrec = res.midpoint_recall
        mrec_str = f"{mrec:.2f}" if mrec is not None else "  -"
        mpre = res.midpoint_precision
        mpre_str = f"{mpre:.2f}" if mpre is not None else "  -"
        lines.append(
            f"  {res.vID:<13}{(res.hudl_id or 0):>9}  {res.status:<18}"
            f"{len(res.gt_windows):>4}{len(res.pred_windows):>5}"
            f"{res.tp_window:>4}{res.fp_window:>4}{res.fn_window:>4}"
            f"{res.precision:>7.2f}{res.recall:>7.2f}{res.f1:>7.2f}"
            f"{mrec_str:>7}{mpre_str:>7}{attr_str:>7}"
        )
        for note in res.notes:
            lines.append(f"      note: {note}")

    if diagnostics_summary:
        lines.append("")
        lines.append("DIAGNOSTICS — what's behind the unmatched items")
        lines.append("-" * 78)

        fn_cls = diagnostics_summary.get("fn_classification") or {}
        fp_cls = diagnostics_summary.get("fp_classification") or {}
        fp_src = diagnostics_summary.get("fp_spurious_by_source") or {}
        iou_d  = diagnostics_summary.get("near_miss_iou_distribution")

        # Order classes from "measurement issue" to "model issue" so the
        # eye scans top-down through likely root causes.
        fn_order = ["scale_mismatch", "no_spatial_overlap",
                    "missed_entirely", "no_pred"]
        fp_order = ["scale_mismatch", "adjacent_to_gt",
                    "spurious", "no_gt"]

        if fn_cls:
            lines.append("")
            lines.append("  Missed GT windows (FNs) by likely cause:")
            for k in fn_order:
                v = fn_cls.get(k, 0)
                if v:
                    lines.append(f"    {k:<24} {v:>4}")
            for k, v in fn_cls.items():
                if k not in fn_order and v:
                    lines.append(f"    {k:<24} {v:>4}")

        if fp_cls:
            lines.append("")
            lines.append("  Extra predictions (FPs) by likely cause:")
            for k in fp_order:
                v = fp_cls.get(k, 0)
                if v:
                    lines.append(f"    {k:<24} {v:>4}")
            for k, v in fp_cls.items():
                if k not in fp_order and v:
                    lines.append(f"    {k:<24} {v:>4}")

        if fp_src:
            lines.append("")
            lines.append("  Spurious FPs by cv_seg source signal "
                         "(where do bogus windows come from?):")
            for src, n in sorted(fp_src.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {src:<24} {n:>4}")

        if iou_d:
            lines.append("")
            lines.append("  Near-miss IoU distribution "
                         "(unmatched items that DID overlap a window):")
            lines.append(f"    n={iou_d['n']}  min={iou_d['min']}  "
                         f"p25={iou_d['p25']}  median={iou_d['median']}  "
                         f"p75={iou_d['p75']}  max={iou_d['max']}")
            lines.append("    (if median > 0.20, lowering --iou-threshold "
                         "to 0.2 would convert many of these to TPs)")

        lines.append("")
        lines.append("  How to read this:")
        lines.append("    scale_mismatch     = right place, wrong size — eval issue, "
                     "not a model issue")
        lines.append("    no_spatial_overlap = nearby in time but no overlap — "
                     "small timing offset")
        lines.append("    missed_entirely    = GT window has no nearby prediction — "
                     "real recall miss")
        lines.append("    spurious           = prediction has no nearby GT shot — "
                     "real precision miss")

    if fp_trace_summary and fp_trace_summary.get("fp_total"):
        lines.append("")
        lines.append("FP SIGNAL TRACE — which signals are firing on bogus predictions")
        lines.append("-" * 78)
        total = fp_trace_summary["fp_total"]
        no_trace = fp_trace_summary.get("fps_without_trace", 0)
        lines.append(f"  total FPs:            {total}")
        if no_trace:
            lines.append(f"  FPs without trace:    {no_trace} "
                         f"(cv_seg run pre-dates the _signals.json sidecar — "
                         f"re-run cv_seg to populate)")

        combos = fp_trace_summary.get("fp_by_source_combo") or {}
        if combos:
            lines.append("")
            lines.append("  FP count by source combination "
                         "(which signals fired together to produce the FP):")
            # Show top 10 to keep the report scan-friendly. The TSV
            # has the long tail.
            for combo, count in list(combos.items())[:10]:
                pct = 100 * count / total
                lines.append(f"    {combo:<40} {count:>4}  ({pct:>4.1f}%)")
            if len(combos) > 10:
                lines.append(f"    ... and {len(combos) - 10} other combinations "
                             f"(see _fp_trace.tsv)")

        individual = fp_trace_summary.get("fp_by_individual_signal") or {}
        if individual:
            lines.append("")
            lines.append("  FP count by individual signal "
                         "(double-counts FPs hit by multiple signals — "
                         "use this to rank levers):")
            for sig, count in individual.items():
                lines.append(f"    {sig:<28} {count:>4}")

        per_video = fp_trace_summary.get("fp_top_combo_per_video") or {}
        if per_video:
            lines.append("")
            lines.append("  Top FP source per video (where to look first):")
            for vID, top in per_video.items():
                if top:
                    summary_str = ", ".join(f"{c}({n})" for c, n in top)
                    lines.append(f"    {vID:<14} {summary_str}")

        lines.append("")
        lines.append("  How to read this:")
        lines.append("    Each FP is one cv_seg threat segment that didn't match a GT shot.")
        lines.append("    'source_combo' = the SET of raw-window source signals that")
        lines.append("    overlapped that segment. If 'motion' alone dominates, the motion")
        lines.append("    threshold is the lever. If 'celebration+motion' dominates, those")
        lines.append("    signals are co-firing on non-shot game flow (scrums, line changes).")

    return "\n".join(lines) + "\n"


def write_reports(
    results: list[VideoEvalResult],
    summary: dict,
    args: argparse.Namespace,
) -> tuple[str, str, Optional[str], Optional[str]]:
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        raise SystemExit(
            f"Cannot create output directory {args.output_dir!r}: {e}.\n"
            f"  Pass --output-dir <path> to write somewhere writable, "
            f"or set CV_SEG_DATA_ROOT to override the default data root."
        )
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    json_path     = os.path.join(args.output_dir, f"eval_{timestamp}.json")
    txt_path      = os.path.join(args.output_dir, f"eval_{timestamp}.txt")
    tsv_path      = os.path.join(args.output_dir, f"eval_{timestamp}_diagnostics.tsv")
    fp_trace_path = os.path.join(args.output_dir, f"eval_{timestamp}_fp_trace.tsv")

    diag_rows, diag_summary = build_diagnostics(results)
    fp_trace_rows, fp_trace_summary = build_fp_trace(results)

    json_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {
            "window_diff":   args.window_diff,
            "iou_threshold": args.iou_threshold,
            "gt_pad":        args.gt_pad,
            "gt_dir":        args.gt_dir,
            "pred_dir":      args.pred_dir,
        },
        "summary": summary,
        "diagnostics_summary": diag_summary,
        "fp_trace_summary": fp_trace_summary,
        "videos": [
            {
                "vID":           r.vID,
                "hudl_id":       r.hudl_id,
                "status":        r.status,
                "duration":      r.duration,
                "gt_windows":    [_window_to_dict(w) for w in r.gt_windows],
                "pred_windows":  [_window_to_dict(w) for w in r.pred_windows],
                "metrics": {
                    "tp": r.tp_window,
                    "fp": r.fp_window,
                    "fn": r.fn_window,
                    "precision": round(r.precision, 4),
                    "recall":    round(r.recall, 4),
                    "f1":        round(r.f1, 4),
                    "midpoint_recall":
                        round(r.midpoint_recall, 4)
                        if r.midpoint_recall is not None else None,
                    "midpoint_precision":
                        round(r.midpoint_precision, 4)
                        if r.midpoint_precision is not None else None,
                    "midpoint_gt_hits":  r.gt_midpoint_hits,
                    "midpoint_gt_total": r.gt_total_for_mid,
                    "midpoint_pred_hits":  r.pred_midpoint_hits,
                    "midpoint_pred_total": r.pred_total_for_mid,
                    "attribution_accuracy":
                        round(r.attribution_accuracy, 4)
                        if r.attribution_accuracy is not None else None,
                    "attribution_correct": r.attr_correct,
                    "attribution_wrong":   r.attr_wrong,
                    "attribution_skipped": r.attr_skipped,
                },
                "matches":       r.matches,
                "notes":         r.notes,
            }
            for r in results
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_text_report(results, summary, args,
                                   diag_summary, fp_trace_summary))

    tsv_written: Optional[str] = None
    if diag_rows:
        write_diagnostics_tsv(diag_rows, tsv_path)
        tsv_written = tsv_path

    fp_trace_written: Optional[str] = None
    if fp_trace_rows:
        write_fp_trace_tsv(fp_trace_rows, fp_trace_path)
        fp_trace_written = fp_trace_path

    return json_path, txt_path, tsv_written, fp_trace_written


def _window_to_dict(w: ThreatWindow) -> dict:
    return {
        "start": w.start, "end": w.end,
        "team": w.team, "color": w.color,
        "period": w.period, "source": w.source,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate cv_seg threat-segment outputs against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vIDs", nargs="*", default=None,
        help="Restrict evaluation to these video IDs (default: all in VID_TO_HUDL)",
    )
    p.add_argument(
        "--window-diff", type=int, default=DEFAULT_WINDOW_DIFF,
        help="Max gap (seconds) between adjacent shots/goals to merge into one GT window",
    )
    p.add_argument(
        "--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
        help="Minimum IoU for a predicted window to count as matching a GT window",
    )
    p.add_argument(
        "--gt-pad", type=float, default=0.0,
        help="Pad each GT window by this many seconds on each side before IoU "
             "matching. Hudl publishes 12-second clips around shots; the actual "
             "threat extends earlier (zone entry) and later (rebound). A pad of "
             "10-20 brings GT widths into the same range as cv_seg's natural "
             "windows. Doesn't affect midpoint-recall.",
    )
    p.add_argument(
        "--gt-dir", default=DEFAULT_GT_DIR,
        help="Directory containing gt_{hudl_id}.csv files",
    )
    p.add_argument(
        "--pred-dir", default=DEFAULT_PRED_DIR,
        help="Directory containing gt_seg_{vID}.json prediction files",
    )
    p.add_argument(
        "--pred-dirs", nargs="+", default=None, metavar="LABEL=DIR",
        help="Sweep mode: evaluate multiple cv_seg configurations and emit a "
             "side-by-side comparison report. Each entry is LABEL=DIR, e.g. "
             "--pred-dirs baseline=data/output/cv_seg "
             "motion0.8=data/output/cv_seg_motion08 "
             "motion1.0=data/output/cv_seg_motion10. "
             "Overrides --pred-dir.",
    )
    p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory for the eval reports",
    )
    p.add_argument(
        "--team-color-map", default=None,
        help="Path to JSON mapping {hudl_id: {team_name: goalie_color}} "
             "for attribution scoring (manual fallback when --customer-id "
             "is not used)",
    )
    p.add_argument(
        "--customer-id", dest="customer_ids", default=None, nargs="+",
        metavar="CUSTOMER_ID",
        help="One or more customer IDs (e.g. CUST000048 CUST000031) — for "
             f"each, fetches gs://{GCS_BUCKET}/{GCS_CUSTOMER_PREFIX}/"
             "{customer-id}.json and merges all per-vID records into a "
             "single team→color map. Use multiple values when the eval "
             "set spans more than one customer — each customer's records "
             "cover only their own vIDs, so missing one means those "
             "videos get evaluated without team filtering. "
             "Takes precedence over --team-color-map.",
    )
    p.add_argument(
        "--customer-cache-dir", default=None,
        help="Directory to cache downloaded customer files "
             "(default: <data_root>/customers)",
    )
    p.add_argument(
        "--no-gcs", action="store_true",
        help="Disable GCS download fallback for both prediction files "
             "and customer files",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Reduce log verbosity to WARNING",
    )
    return p.parse_args(argv)


def load_team_color_map(path: Optional[str]) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    if not os.path.exists(path):
        log.warning(f"team-color-map {path} not found — attribution scoring disabled")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"failed to load team-color-map {path}: {e}")
        return {}
    # Normalise keys to strings so callers can use either int or str hudl_ids.
    return {str(k): v for k, v in data.items()}


def _parse_pred_dirs(specs: list[str]) -> list[tuple[str, str]]:
    """
    Parse --pred-dirs values like 'baseline=path/to/dir' into
    [(label, dir), ...]. A bare path with no '=' gets the basename
    as its label so users can do --pred-dirs cv_seg_motion08
    cv_seg_motion10 and still get sensible labels.
    """
    out: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for spec in specs:
        if "=" in spec:
            label, _, path = spec.partition("=")
            label = label.strip()
            path = path.strip()
        else:
            path = spec.strip()
            label = os.path.basename(path.rstrip(os.sep)) or path
        if not label or not path:
            raise SystemExit(f"--pred-dirs entry malformed: {spec!r}")
        if label in seen_labels:
            raise SystemExit(f"--pred-dirs duplicate label: {label!r}")
        seen_labels.add(label)
        out.append((label, path))
    return out


def _evaluate_one_config(
    label: str,
    pred_dir: str,
    target_vids: list[str],
    args: argparse.Namespace,
    team_color_map: dict[str, dict[str, str]],
) -> tuple[list[VideoEvalResult], dict]:
    """Run the per-video evaluation loop for a single pred_dir; return
    the per-video results and the aggregate summary. Used by both the
    single-config and sweep paths."""
    log.info(f"=== Config: {label}  ({pred_dir}) ===")
    results: list[VideoEvalResult] = []
    for vID in target_vids:
        log.info(f"[{label}/{vID}] evaluating...")
        try:
            res = evaluate_video(
                vID=vID,
                gt_dir=args.gt_dir,
                pred_dir=pred_dir,
                window_diff=args.window_diff,
                iou_threshold=args.iou_threshold,
                team_to_color_global=team_color_map,
                allow_gcs_download=(not args.no_gcs),
                gt_pad=args.gt_pad,
            )
        except Exception as e:
            log.exception(f"[{label}/{vID}] unhandled error during evaluation: {e}")
            res = VideoEvalResult(
                vID=vID, hudl_id=VID_TO_HUDL.get(vID),
                status="error", notes=[f"unhandled: {e}"],
            )
        log.info(f"[{label}/{vID}] {res.status} — TP={res.tp_window} "
                 f"FP={res.fp_window} FN={res.fn_window} F1={res.f1:.3f}")
        results.append(res)
    summary = aggregate(results)
    return results, summary


def render_sweep_report(
    configs: list[tuple[str, str]],
    summaries: dict[str, dict],
    args: argparse.Namespace,
) -> str:
    """
    Render a single-page text comparison across configs. Layout: one
    row per metric, one column per config, plus a delta-vs-first column
    so it's obvious which way each knob moves the needle.
    """
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("cv_seg sweep comparison")
    lines.append(f"  generated:       {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append(f"  window_diff:     {args.window_diff}s")
    lines.append(f"  iou_threshold:   {args.iou_threshold}")
    lines.append(f"  gt_pad:          ±{args.gt_pad:g}s")
    lines.append("  configs:")
    for label, pred_dir in configs:
        lines.append(f"    {label:<20} {pred_dir}")
    lines.append("=" * 90)

    labels = [lab for lab, _ in configs]
    baseline = labels[0]

    # Each metric row: ("display name", key in summary, format string,
    #   higher_is_better). higher_is_better drives the delta sign hint
    #   (▲/▼) so the eye doesn't need to translate "F1 went up" vs
    #   "FP went down".
    metrics = [
        ("STRICT precision",     "precision",          "{:.4f}", True),
        ("STRICT recall",        "recall",             "{:.4f}", True),
        ("STRICT F1",            "f1",                 "{:.4f}", True),
        ("STRICT TP",            "tp_window",          "{:d}",   True),
        ("STRICT FP",            "fp_window",          "{:d}",   False),
        ("STRICT FN",            "fn_window",          "{:d}",   False),
        ("LENIENT mid recall",   "midpoint_recall",    "{:.4f}", True),
        ("LENIENT mid precision","midpoint_precision", "{:.4f}", True),
        ("LENIENT mid F1",       "midpoint_f1",        "{:.4f}", True),
        ("Attribution accuracy", "attribution_accuracy","{:.4f}", True),
    ]

    name_w = 24
    col_w  = 14

    # Header row
    header = " " * name_w
    for lab in labels:
        header += f"{lab:>{col_w}}"
    if len(labels) > 1:
        header += f"  Δ vs {baseline}"
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))

    for display, key, fmt, higher_is_better in metrics:
        row = f"{display:<{name_w}}"
        baseline_val = summaries[baseline].get(key)
        for lab in labels:
            val = summaries[lab].get(key)
            cell = "—" if val is None else fmt.format(val)
            row += f"{cell:>{col_w}}"
        if len(labels) > 1 and baseline_val is not None:
            # Delta cell shows the LAST config vs baseline (the most
            # likely thing the user wants when sweeping incrementally).
            last_val = summaries[labels[-1]].get(key)
            if last_val is None:
                row += "          —"
            else:
                delta = last_val - baseline_val
                arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "·")
                # Color the arrow's "good"-ness for the eye:
                better = ((delta > 0) == higher_is_better) and delta != 0
                tag = "(better)" if better else ("(worse)" if delta != 0 else "")
                if isinstance(baseline_val, int) and isinstance(last_val, int):
                    row += f"   {arrow} {delta:+d} {tag}"
                else:
                    row += f"   {arrow} {delta:+.4f} {tag}"
        lines.append(row)

    # Per-video sub-table for STRICT F1 and LENIENT mid_recall — these
    # are usually the two we care about most. If anything, this is
    # what makes the report worth scanning.
    lines.append("")
    lines.append("Per-video STRICT F1 / LENIENT midpoint recall")
    lines.append("-" * 90)
    sub_header = f"  {'vID':<13}{'hudl':>9}"
    for lab in labels:
        sub_header += f"{(lab + ' F1'):>{col_w}}{(lab + ' mRec'):>{col_w + 2}}"
    lines.append(sub_header)

    # We need the per-video results too, not just summaries. The
    # caller passes them into a parallel structure stored on each
    # summary dict under "_results" so we don't change the public
    # aggregate() shape.
    sample_results = summaries[labels[0]].get("_results") or []
    for i, sample_res in enumerate(sample_results):
        vID = sample_res.vID
        hudl = sample_res.hudl_id or 0
        row = f"  {vID:<13}{hudl:>9}"
        for lab in labels:
            res_list = summaries[lab].get("_results") or []
            res = res_list[i] if i < len(res_list) else None
            f1 = res.f1 if res else 0.0
            mrec = res.midpoint_recall if res and res.midpoint_recall is not None else None
            mrec_str = f"{mrec:.2f}" if mrec is not None else "  -"
            row += f"{f1:>{col_w}.2f}{mrec_str:>{col_w + 2}}"
        lines.append(row)

    return "\n".join(lines) + "\n"


def write_sweep_report(
    configs: list[tuple[str, str]],
    summaries: dict[str, dict],
    per_config_results: dict[str, list[VideoEvalResult]],
    args: argparse.Namespace,
) -> tuple[str, str]:
    """Persist sweep comparison as both .txt and .json. Per-config
    individual reports are NOT written here — the user asked for
    a single comparison report. Run individual evals separately if
    you want full per-config reports."""
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        raise SystemExit(
            f"Cannot create output directory {args.output_dir!r}: {e}.\n"
            f"  Pass --output-dir <path> to write somewhere writable."
        )
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    json_path = os.path.join(args.output_dir, f"eval_sweep_{timestamp}.json")
    txt_path  = os.path.join(args.output_dir, f"eval_sweep_{timestamp}.txt")

    # Attach per-video results to each summary so the renderer can
    # build the per-video sub-table without us threading another arg
    # through the call.
    enriched: dict[str, dict] = {}
    for label, summary in summaries.items():
        enriched[label] = {**summary, "_results": per_config_results[label]}

    # JSON payload — strip the _results sentinel since VideoEvalResult
    # isn't natively JSON-serialisable. We expose the per-video metric
    # subset that's most useful for downstream charting.
    json_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {
            "window_diff":   args.window_diff,
            "iou_threshold": args.iou_threshold,
            "gt_pad":        args.gt_pad,
            "gt_dir":        args.gt_dir,
        },
        "configs": [{"label": lab, "pred_dir": d} for lab, d in configs],
        "summaries": {
            lab: {k: v for k, v in s.items() if not k.startswith("_")}
            for lab, s in enriched.items()
        },
        "per_video": {
            lab: [
                {
                    "vID": res.vID,
                    "hudl_id": res.hudl_id,
                    "status": res.status,
                    "f1": round(res.f1, 4),
                    "precision": round(res.precision, 4),
                    "recall": round(res.recall, 4),
                    "midpoint_recall":
                        round(res.midpoint_recall, 4)
                        if res.midpoint_recall is not None else None,
                    "midpoint_precision":
                        round(res.midpoint_precision, 4)
                        if res.midpoint_precision is not None else None,
                    "attribution_accuracy":
                        round(res.attribution_accuracy, 4)
                        if res.attribution_accuracy is not None else None,
                }
                for res in per_config_results[lab]
            ]
            for lab in [c[0] for c in configs]
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_sweep_report(configs, enriched, args))

    return json_path, txt_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    target_vids = args.vIDs if args.vIDs else list(VID_TO_HUDL.keys())

    # Build the team→color attribution map. Priority order:
    #   1) --customer-id (fetches from GCS, parses, builds the map)
    #   2) --team-color-map (legacy hand-written JSON)
    #   3) {} (eval runs in windows-only mode)
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
                # Detect cross-customer key collisions (shouldn't happen
                # in practice — vIDs are per-customer — but a misfiled
                # record could cause silent overwrites).
                overlapping = set(per_customer_map) & set(team_color_map)
                if overlapping:
                    log.warning(
                        f"  customer file ({customer_id}): {len(overlapping)} "
                        f"vID(s) already mapped by an earlier customer file — "
                        f"later values win: {sorted(overlapping)}"
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

    # ── Sweep mode ─────────────────────────────────────────────────────
    if args.pred_dirs:
        configs = _parse_pred_dirs(args.pred_dirs)
        log.info(f"Sweep mode: {len(configs)} config(s) over "
                 f"{len(target_vids)} video(s)")

        summaries: dict[str, dict] = {}
        per_config_results: dict[str, list[VideoEvalResult]] = {}
        for label, pred_dir in configs:
            results, summary = _evaluate_one_config(
                label=label, pred_dir=pred_dir,
                target_vids=target_vids, args=args,
                team_color_map=team_color_map,
            )
            summaries[label] = summary
            per_config_results[label] = results

        json_path, txt_path = write_sweep_report(
            configs, summaries, per_config_results, args,
        )
        log.info("=" * 60)
        log.info("Sweep comparison summary:")
        for label, _ in configs:
            s = summaries[label]
            mid_f1_str = (f"{s['midpoint_f1']:.3f}"
                          if s.get('midpoint_f1') is not None else " —")
            log.info(f"  {label:<20} STRICT F1={s['f1']:.3f}  "
                     f"LENIENT mid F1={mid_f1_str}")
        log.info(f"Reports written:")
        log.info(f"  {json_path}")
        log.info(f"  {txt_path}")
        return 0

    # ── Single-config mode (original behaviour) ───────────────────────
    log.info(f"Evaluating {len(target_vids)} video(s) "
             f"(window_diff={args.window_diff}s, iou>={args.iou_threshold})")

    results, summary = _evaluate_one_config(
        label="default", pred_dir=args.pred_dir,
        target_vids=target_vids, args=args,
        team_color_map=team_color_map,
    )

    json_path, txt_path, tsv_path, fp_trace_path = write_reports(
        results, summary, args
    )

    log.info("=" * 60)
    log.info(f"STRICT  P={summary['precision']:.3f}  "
             f"R={summary['recall']:.3f}  F1={summary['f1']:.3f}")
    if summary["midpoint_recall"] is not None:
        log.info(f"LENIENT midpoint  "
                 f"R={summary['midpoint_recall']:.3f}  "
                 f"P={summary['midpoint_precision']:.3f}  "
                 f"F1={summary['midpoint_f1']:.3f}")
    if summary["attribution_accuracy"] is not None:
        log.info(f"Attribution accuracy: {summary['attribution_accuracy']:.3f}")
    log.info(f"Reports written:")
    log.info(f"  {json_path}")
    log.info(f"  {txt_path}")
    if tsv_path:
        log.info(f"  {tsv_path}")
    if fp_trace_path:
        log.info(f"  {fp_trace_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
