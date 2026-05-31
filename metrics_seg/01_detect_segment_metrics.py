"""
01_detect_segment_metrics.py

v8 changes (over v7 — code-only, no prompt change):
  - Video clips are now sent INLINE to Gemini via Part.from_bytes
    instead of being uploaded to GCS first and referenced by URI.
    This eliminates the GCS upload (the bottleneck behind workers=4
    upload contention causing TimeoutErrors) and removes the temp-blob
    lifecycle entirely. No token-cost change — Gemini receives the
    same video either way; transport doesn't affect tokenization.
    Tradeoff: in multi-call vote paths the same bytes are
    re-transmitted on each call (no server-side cache), but the
    cumulative wire cost is small for 5-12 MB clips and we save the
    GCS round-trip.
  - Added MAX_INLINE_VIDEO_BYTES guard (18 MB) — if an unusually
    encoded clip exceeds the limit, fail clearly rather than letting
    the API return an opaque size error.
  - gcs_upload_file and gcs_delete_blob are no longer called by the
    pipeline. They're left in the module as no-op-from-our-side
    library functions in case external callers reference them, but
    nothing in this file depends on them anymore.
  - The end-of-run GCS "safety sweep" of TEMP_PREFIX was removed
    since no temp blobs are ever created.

v6 changes (over v4 — note: v5 was reverted):
  - PROMPT (goal-rubric): Added an explicit anti-rationalization rule
    forbidding goals=1 inferences based on the shotsOnNet=saves+goals
    identity. The v5 5-video run surfaced a new failure mode where the
    model wrote "a goal is counted to maintain the identity" — turning
    a clear shot-on-net into a goals=1 claim that bypasses the truth
    table. v6 makes the resolution rule explicit: when a SOG cannot
    satisfy any truth-table path, set goals=0 and treat the shot as
    a save (saves = shotsOnNet). This addresses 2 of the 9 v5 FPs
    (SX5 116-137, v0 785-905) without loosening any path.

  - LOGIC: Added conditional 3-call majority vote for `goals`. When the
    first call returns goals >= 1, the wrapper fires 2 more calls at
    temperature=0.3 and takes a majority vote. goals=1 is kept only if
    2+ of 3 calls agree. After voting, `saves` is re-derived as
    `shotsOnNet - goals` to preserve the identity. goal_criteria is
    preserved from the first call (no merging of boolean flags across
    runs). Targets the run-to-run Gemini variance that produced v4's
    mj 6/0/0 → v5 2/0/4 swing without any rubric change. The vote only
    fires on goal-claim windows (~5-10 per video), so total realised
    cost is ~1.0-1.3x.

  - Path 5 from v5 is REVERTED. The truth table returns to v4's 4 paths.

v4 changes (over v3.1):
  - PROMPT: Added anti-narration rule and per-shot enumeration discipline
    to the `shots` definition (analog of the SOG anti-narration rule from
    v3). Strengthened final calibration check question 2 to require
    timestamp + position + release type per shot. Added Example 10
    illustrating the long-busy-window over-counting failure mode.
  - LOGIC:  Added conditional 3-call median for `shots` and `shotsOnNet`.
    When a single call returns shots > MULTICALL_SHOTS_THRESHOLD (=4),
    the wrapper fires 2 more calls and takes the median of `shots` and
    `shotsOnNet` across all 3 results. The first call stays at
    temperature=0 so cheap-clip behaviour is unchanged and reproducible.
    The 2 extra calls use temperature=0.3 to introduce variance — at
    temperature=0 the extras would be deterministic copies of the first
    and the median would be a no-op. Other fields (goals, rebounds,
    goal_criteria) come from the first call. After median, `saves` is
    recomputed as `shotsOnNet - goals` to preserve the identity. Median
    only fires on long busy clips, not on every call, so the typical
    cost is ~1.0–1.3x not 3x.

For each threat window detected by 01_detect_goalie_segments.py, extracts the
corresponding video clip and sends it to Gemini to count hockey metrics:
  - shots        : a deliberate shot attempt by the opponent — wind-up,
                   snap, or release with the puck headed toward the net.
                   Excludes dump-ins, passes, rims, deflections without a
                   shooting motion, and any puck movement during stoppages.
                   Includes goalie saves, misses wide/high, and shots
                   blocked by a defender. Typical: 1–4 per 120-second clip
                   of sustained pressure.
  - shotsOnNet   : subset of `shots` where the goalie actually had to
                   handle the puck — every shotOnNet has a goalie save,
                   a goal, or a post hit that would have entered.
                   Misses-wide and skater-blocked shots are NOT shotsOnNet.
                   Typically 40–70% of `shots`.
  - saves        : shotsOnNet stopped by the goalie (not goals)
  - rebounds     : loose pucks after a save attempt
  - goals        : pucks that crossed the goal line

Reads:
  gs://goalie_video_bucket/analyze_video/01-segment_detection/gt_seg_{vID}.json
  gs://goalie_video_bucket/ground_truth_video/full_video/full_{vID}.mp4

Writes:
  gs://goalie_video_bucket/analyze_video/02-segment_metrics/gt_metrics_{vID}.json

Output schema — each segment from step 01 is preserved exactly, with a
"metrics" key added for threat segments (null for no-threat segments):

  {
    "segmentHasThreat": true,
    "threat_goalie_color": "White and Green",
    "threat_goalie_side": "left",
    "segment_start": 342,
    "segment_end": 378,
    "metrics": {
      "shots": 2,
      "shotsOnNet": 1,
      "saves": 1,
      "rebounds": 1,
      "goals": 0,
      "observed_goalie_side": "left",   -- side the model saw in the clip;
                                           recorded for telemetry only and
                                           never overrides the step-1 side
                                           assignment (which comes from config)
      "goal_criteria": { ... }           -- raw goal signal flags for debugging
    }
  }

Usage:
    python3 01_detect_segment_metrics.py --vID U7NUbWad0A8 --customID CUST000048

    # Multiple videos
    python3 01_detect_segment_metrics.py \\
        --vID U7NUbWad0A8 SX5xNJlh6eQ KYtM20r9BuM \\
        --customID CUST000048

    # Limit parallel workers (default: 2)
    python3 01_detect_segment_metrics.py --vID U7NUbWad0A8 --customID CUST000048 --workers 3

Model: gemini-2.5-pro
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import pathlib
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# Make `from metrics_seg import X` resolvable when this file is run
# directly as `python3 metrics_seg/01_detect_segment_metrics.py` (the
# common invocation in tools/* harnesses). Without this, the v14
# subpackage imports below silently fall through to
# _V14_IMPROVEMENTS_AVAILABLE=False and every --flash-screen /
# --goal-ensemble / --prefilter-threshold flag becomes a no-op.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from google.api_core import exceptions as gcore_exceptions
from google.cloud import storage
from google import genai
from google.genai import types

# Pipeline progress reporting → customer JSON's analyticsStatus.
# No-op when --progress-stage-idx is not passed (standalone usage).
try:
    from util import progress as _pipeline_progress
except ImportError:
    _pipeline_progress = None

# Set by main() from --progress-stage-idx + --customID. None → progress
# reporting off (standalone usage).
_PROGRESS_STAGE_IDX: int | None = None
_PROGRESS_CUSTOMER_ID: str | None = None

# v14 improvements (see IMPROVEMENTS_SPEC.md). All additive; off by
# default. Imported lazily-safe: failures to import do NOT break the
# existing pipeline.
try:
    from metrics_seg import cache as _v14_cache
    from metrics_seg import prefilter as _v14_prefilter
    from metrics_seg import audio_context as _v14_audio_ctx
    from metrics_seg import goal_ensemble as _v14_goal_ensemble
    from metrics_seg import calibration as _v14_calibration
    from metrics_seg import flash_screen as _v14_flash
    _V14_IMPROVEMENTS_AVAILABLE = True
except ImportError:
    _V14_IMPROVEMENTS_AVAILABLE = False

# Module-level config dict populated by main() from CLI args. Used by
# the deep inner functions (analyze_clip_metrics, _run_one) without
# requiring signature plumbing. All keys default such that the existing
# v13 behavior is preserved when no v14 flags are set.
_V14_CONFIG: dict = {
    "enabled":             False,
    "prefilter_threshold": 0.0,
    "use_context":         False,
    "goal_ensemble":       False,
    "flash_screen":        False,
    "no_cache":            False,
    "cache_dir":           None,
    "probs_dir_yolo":      None,
    "probs_dir_audio":     None,
    "audio_features_dir":  None,
    "calibration_dir":     None,
    "gt_dir":              None,
    # Per-vID memoization populated by _v14_load_probs_for_vid
    "_probs_cache":        {},
}


def _v14_load_probs_for_vid(vid: str):
    """Lazy-load fused probs for a vID. Returns FusedProbs or None."""
    if not _V14_IMPROVEMENTS_AVAILABLE:
        return None
    yolo_dir  = _V14_CONFIG.get("probs_dir_yolo")
    audio_dir = _V14_CONFIG.get("probs_dir_audio")
    if not (yolo_dir or audio_dir):
        return None
    pc = _V14_CONFIG["_probs_cache"]
    if vid in pc:
        return pc[vid]
    fp = _v14_prefilter.load_fused_probs(
        vid,
        pathlib.Path(yolo_dir) if yolo_dir else None,
        pathlib.Path(audio_dir) if audio_dir else None,
    )
    pc[vid] = fp
    return fp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL   = "gemini-2.5-pro"
# NOTE: Must match GEMINI_MODEL in 01_detect_goalie_segments.py — update both together.
# Overridable at runtime via the --model CLI flag (see parse_args()).
# The cache layer (v14) keys on model_name, so switching models naturally
# invalidates cached responses for the new model.
# See SegmentDict in 01_detect_goalie_segments.py for the shared segment schema.
PROJECT_ID = "goalie-analytics-pro-dev"
REGION     = "us-central1"
GCS_BUCKET     = "goalie_video_bucket"

VIDEO_PREFIX   = "ground_truth_video/full_video"
INPUT_PREFIX   = "analyze_video/01-segment_detection"
OUTPUT_PREFIX  = "analyze_video/02-segment_metrics"
TEMP_PREFIX    = "analyze_video/00-temp_parts"  # UNUSED since v8 — kept only because
                                                # other tooling may reference it.
                                                # Safe to remove once you're sure
                                                # nothing else reads it.

MAX_RETRIES       = 6
RETRY_BACKOFF_BASE = 60     # initial delay in seconds; doubles each retry
RETRY_BACKOFF_MAX  = 60     # CAP per-attempt delay. Original code had no cap, so
                            # 6 retries totalled ~30 minutes per segment. Capped at
                            # 60s the worst case is 6×60 = 6min, which matches
                            # typical Gemini transient-error recovery times.
EXTRACT_CLIP_TIMEOUT_SEC = 300  # ffmpeg subprocess timeout — 5 minutes is
                                # generous for clips < 2 minutes long.

# v8: Inline video bytes (no GCS round-trip).
#
# Vertex AI Gemini accepts video as either a GCS URI or inline bytes,
# capped at ~20 MB total request size. Our clips are 30-60s 720p H.264
# at ~1.5 Mbps = 5-12 MB, so they fit comfortably inline. Sending bytes
# directly eliminates the upload-to-GCS / delete-from-GCS round-trip
# (the bottleneck behind workers=4 timing out at 4-thread upload contention),
# and removes the temp-blob lifecycle management entirely.
#
# The Files API (genai.upload_file) is NOT available with vertexai=True
# clients — it's exclusive to the consumer Gemini API key path. Inline
# bytes are the only no-GCS option for Vertex.
#
# Vertex's documented hard limit is 20 MB request size. We set the
# threshold below that to leave room for the prompt and JSON envelope.
MAX_INLINE_VIDEO_BYTES = 18 * 1024 * 1024  # 18 MB — leaves ~2 MB headroom

SIDE_UNKNOWN = "unknown"  # sentinel for unresolved goalie side (matches goalie_segments.py)

# v4: Conditional multi-call median.
# When a single Gemini call returns shots > MULTICALL_SHOTS_THRESHOLD,
# we fire 2 additional calls and take the median of `shots` and
# `shotsOnNet` across all 3 results. Other fields are kept from the
# first call. This targets the long-busy-window over-counting failure
# mode where Gemini hallucinates speculative shots.
#
# IMPORTANT: at temperature=0 Gemini is deterministic, so the extra
# calls would return identical results and the median would be a
# no-op. We bump temperature on the extras to introduce variance,
# while keeping the first call at temp=0 so the cheap-clip path is
# unchanged and reproducible.
MULTICALL_SHOTS_THRESHOLD   = 4    # trigger median when shots > 4
MULTICALL_EXTRA_CALLS       = 2    # fire 2 more calls (total 3)
MULTICALL_VARIANCE_WARN     = 4    # warn if max-min spread on shots exceeds this
MULTICALL_EXTRA_TEMPERATURE = 0.3  # temperature for the 2 extra calls

# v6: Conditional multi-call majority vote on `goals`.
# When a single Gemini call returns goals >= 1, we fire 2 additional calls
# at temperature=0.3 and take a majority vote. goals=1 is kept only if 2+
# of 3 calls agree. This targets the run-to-run Gemini variance that
# produced v4's mj 6/0/0 → v5 2/0/4 swing without any rubric change.
#
# Trigger: goals >= 1 on the first call. We do not vote on goals=0 windows
# (those would target FN recovery, which is a separate iteration).
#
# Field scope: `goals` only. goal_criteria is preserved from the first
# call; merging boolean flags across runs would create incoherent records.
# After the vote, `saves` is re-derived as `shotsOnNet - goals` to keep
# the identity true.
MULTICALL_GOAL_VOTE_TRIGGER  = 1    # fire vote when first-call goals >= 1
MULTICALL_GOAL_VOTE_QUORUM   = 2    # require 2-of-3 majority for goals=1

# Structured output schema — enforces JSON output and eliminates markdown wrapper handling
METRICS_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "shots":               {"type": "INTEGER"},
        "shotsOnNet":          {"type": "INTEGER"},
        "saves":               {"type": "INTEGER"},
        "rebounds":            {"type": "INTEGER"},
        "goals":               {"type": "INTEGER"},
        "observed_goalie_side": {"type": "STRING"},
        # ── v11: structured per-shot enumeration ──────────────────────
        # The v10 prompt already required Gemini to enumerate each shot
        # internally with timestamp/location/release as an anti-narration
        # discipline, but the enumeration lived in prose. v11 asks Gemini
        # to return that enumeration as a structured list, which becomes
        # the basis for shot-centered window refinement downstream.
        # len(shot_timestamps) == shots is enforced by the prompt as an
        # identity. outcome ∈ {goal, save, miss, blocked} maps to the
        # other count fields (saves are saves+goals SOG; miss/blocked
        # are non-SOG).
        "shot_timestamps": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "timestamp": {"type": "STRING"},  # MM:SS
                    "location":  {"type": "STRING"},
                    "release":   {"type": "STRING"},
                    "outcome":   {"type": "STRING"},  # goal|save|miss|blocked
                    # ── v13: evidence-anchored truth-table features ──
                    # Required for every entry. The truth table is:
                    #   feature_puck_traveling_toward_net    must be True (REQUIRED)
                    #   feature_puck_release_frame_visible   ≥1 of these must be True (EVIDENCE)
                    #   feature_puck_impact_sound_audible    ↑
                    #   feature_puck_carrier_holds_or_passes must be False (DISQUALIFIER)
                    # Discrimination scores from the 14-video probe (Jan 2026):
                    #   traveling_toward_net  disc +0.82 (cleanest)
                    #   release_frame_visible disc +0.61
                    #   impact_sound_audible  disc +0.51
                    #   carrier_holds_or_passes disc −0.58 (anti-anchor)
                    "feature_puck_traveling_toward_net":    {"type": "BOOLEAN"},
                    "feature_puck_release_frame_visible":   {"type": "BOOLEAN"},
                    "feature_puck_impact_sound_audible":    {"type": "BOOLEAN"},
                    "feature_puck_carrier_holds_or_passes": {"type": "BOOLEAN"},
                },
                "required": [
                    "timestamp", "location", "release", "outcome",
                    "feature_puck_traveling_toward_net",
                    "feature_puck_release_frame_visible",
                    "feature_puck_impact_sound_audible",
                    "feature_puck_carrier_holds_or_passes",
                ],
            },
        },
        "goal_criteria": {
            "type": "OBJECT",
            "properties": {
                # ── Legacy v6/v7/v8/v9 features (still in schema for
                #    backward compatibility; see metrics_v10.txt for
                #    which are actually used) ────────────────────────
                "anchor_puck_crosses_line":       {"type": "BOOLEAN"},
                "anchor_ref_points_at_net":       {"type": "BOOLEAN"},
                "anchor_puck_retrieved_from_net": {"type": "BOOLEAN"},
                "support_whistle":                {"type": "BOOLEAN"},
                "support_crowd_spike":            {"type": "BOOLEAN"},
                "support_celebration":            {"type": "BOOLEAN"},
                "support_centre_ice_faceoff":     {"type": "BOOLEAN"},
                "disqualifier_active":            {"type": "BOOLEAN"},
                "anchor_puck_crosses_line_timestamp": {"type": "STRING"},
                # ── v10 features (new, drive Path B / Path C) ────
                # scoreboard_change: visible scoreboard digit change
                #   during the clip. Anchor for Path B.
                # attacking_team_skates_to_bench: post-goal fist-bump
                #   ritual. Anchor for Path C.
                # crowd_cheer_sustained: sustained (>3s) crowd cheer
                #   rising from baseline. Anchor for Path C.
                "scoreboard_change":              {"type": "BOOLEAN"},
                "attacking_team_skates_to_bench": {"type": "BOOLEAN"},
                "crowd_cheer_sustained":          {"type": "BOOLEAN"},
                # ── Confirming detail (required when goals >= 1) ──
                "confirming_detail":              {"type": "STRING"},
                "decision_notes":                 {"type": "STRING"},
            },
            "required": [
                "anchor_puck_crosses_line", "anchor_ref_points_at_net",
                "anchor_puck_retrieved_from_net", "support_whistle",
                "support_crowd_spike", "support_celebration",
                "support_centre_ice_faceoff", "disqualifier_active",
                "anchor_puck_crosses_line_timestamp",
                "scoreboard_change", "attacking_team_skates_to_bench",
                "crowd_cheer_sustained",
                "confirming_detail", "decision_notes",
            ],
        },
    },
    "required": ["shots", "shotsOnNet", "saves", "rebounds", "goals",
                 "observed_goalie_side", "shot_timestamps", "goal_criteria"],
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# Loaded from prompts/metrics_v{N}.txt to keep prompt iterations
# diff-clean and version-trackable independent of code changes.
# To switch versions, change PROMPT_VERSION below.
PROMPT_VERSION = "v14.1"
_PROMPT_PATH = pathlib.Path(__file__).parent / "prompts" / f"metrics_{PROMPT_VERSION}.txt"
try:
    METRICS_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()
except FileNotFoundError as _e:
    raise SystemExit(
        f"Prompt file not found: {_PROMPT_PATH}.\n"
        f"  Either restore the file, or update PROMPT_VERSION in "
        f"01_detect_segment_metrics.py to a version that exists."
    ) from _e



# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":    self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "msg":   record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in ("args", "created", "exc_info", "exc_text", "filename",
                           "funcName", "levelname", "levelno", "lineno", "message",
                           "module", "msecs", "msg", "name", "pathname", "process",
                           "processName", "relativeCreated", "stack_info", "thread",
                           "threadName") and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _setup_logging() -> None:
    """Configure root logger to emit JSON to stdout. Called from
    main() rather than at import time so importing this module as a
    library does not mutate the importer's logging configuration.

    Idempotent: subsequent calls are no-ops (won't add duplicate
    handlers)."""
    root = logging.getLogger()
    if any(getattr(h, "_segmetrics_handler", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    handler._segmetrics_handler = True  # type: ignore[attr-defined]
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    for noisy in ("urllib3", "google.auth", "google.api_core"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# Module-level logger handle; handlers attached lazily by main().
# Importing this module as a library does NOT configure logging.
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

_gcs_client = None
_gemini_client = None

def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def _get_gemini_client():
    """Cache the Gemini client at module scope. The client carries
    auth state and a connection pool; recreating it per-video wastes
    both. Thread-safe to share since the underlying transport handles
    its own locking."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
    return _gemini_client


def gcs_download_to_temp(bucket_name: str, blob_name: str, suffix: str) -> str:
    blob   = _get_gcs_client().bucket(bucket_name).blob(blob_name)
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    log.info("Downloading from GCS", extra={"src": f"gs://{bucket_name}/{blob_name}"})
    blob.download_to_filename(tmp.name)
    return tmp.name


def gcs_read_json(bucket_name: str, blob_name: str):
    blob = _get_gcs_client().bucket(bucket_name).blob(blob_name)
    return json.loads(blob.download_as_text())


def gcs_write_json(bucket_name: str, blob_name: str, data) -> None:
    blob = _get_gcs_client().bucket(bucket_name).blob(blob_name)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
    log.info("Written to GCS", extra={"dst": f"gs://{bucket_name}/{blob_name}"})


def _write_metrics_output(
    output: list,
    vID: str,
    *,
    no_gcs: bool = False,
    output_dir: str | None = None,
) -> None:
    """
    Write gt_metrics_{vID}.json to local, GCS, or both.

    Rules:
      - If output_dir is set: write locally there.
      - If no_gcs is False: also write to GCS.
      - If no_gcs is True and output_dir is not set: raise (no destination).
    """
    filename = f"gt_metrics_{vID}.json"
    wrote_somewhere = False

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        local_path = os.path.join(output_dir, filename)
        with open(local_path, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"[{vID}] Written locally to {local_path}")
        wrote_somewhere = True

    if not no_gcs:
        gcs_write_json(GCS_BUCKET, f"{OUTPUT_PREFIX}/{filename}", output)
        wrote_somewhere = True

    if not wrote_somewhere:
        log.warning(
            f"[{vID}] No output destination — pass --output-dir or drop --no-gcs"
        )


def _write_trace_sidecar(
    trace_payload: dict,
    vID: str,
    *,
    no_gcs: bool = False,
    output_dir: str | None = None,
) -> None:
    """Write the gt_metrics_{vID}_trace.json sidecar.

    Same destination policy as _write_metrics_output. The trace is
    OPTIONAL data: if it can't be written we log and continue rather
    than fail the whole run."""
    filename = f"gt_metrics_{vID}_trace.json"
    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            local_path = os.path.join(output_dir, filename)
            with open(local_path, "w") as f:
                json.dump(trace_payload, f, indent=2)
            log.info(f"[{vID}] Trace sidecar written locally to {local_path}")
        if not no_gcs:
            gcs_write_json(GCS_BUCKET, f"{OUTPUT_PREFIX}/{filename}", trace_payload)
    except Exception as e:
        log.warning(f"[{vID}] Failed to write trace sidecar (non-fatal): {e}")


def _output_already_exists(
    vID: str,
    *,
    no_gcs: bool,
    output_dir: str | None,
) -> bool:
    """Return True if gt_metrics_{vID}.json already exists at the
    configured destination(s).

    Logic mirrors _write_metrics_output's destination rules: if
    output_dir is set we check local; if --no-gcs is not set we
    check GCS too. Skip-existing requires ALL configured destinations
    to have the file (so re-running with a new --output-dir doesn't
    silently skip a missing local copy)."""
    filename = f"gt_metrics_{vID}.json"
    checks: list[bool] = []
    if output_dir:
        checks.append(os.path.exists(os.path.join(output_dir, filename)))
    if not no_gcs:
        try:
            blob = _get_gcs_client().bucket(GCS_BUCKET).blob(f"{OUTPUT_PREFIX}/{filename}")
            checks.append(blob.exists())
        except Exception as e:
            log.warning(f"[{vID}] Could not check GCS existence: {e}")
            checks.append(False)
    if not checks:
        # No destinations configured — can't determine; safer to re-run.
        return False
    return all(checks)


def _summarize_traces(trace_map: dict[int, dict]) -> dict:
    """Aggregate per-segment traces into a per-video summary dict.

    Captures vote-fire rates and total Gemini call count, which is
    what we need to validate the cost claim against actual data
    (rather than the docstring's hand-waved 1.3-1.5x estimate)."""
    n = len(trace_map)
    if n == 0:
        return {
            "n_segments": 0,
            "total_gemini_calls": 0,
            "shot_vote_fired": 0,
            "goal_vote_fired": 0,
            "goal_vote_kept": 0,
            "goal_vote_rejected": 0,
        }

    traces = list(trace_map.values())
    total_calls       = sum(t.get("n_calls", 0) for t in traces)
    shot_fired        = sum(1 for t in traces if t.get("shot_vote_triggered"))
    goal_fired        = sum(1 for t in traces if t.get("goal_vote_triggered"))
    goal_kept         = sum(1 for t in traces if t.get("goal_vote_outcome") == "kept")
    goal_rejected     = sum(1 for t in traces if t.get("goal_vote_outcome") == "rejected")

    # The "cost ratio" the docstring claimed: average calls per
    # segment, vs the cheap-path baseline of 1.0 calls per segment.
    cost_ratio = total_calls / n if n > 0 else 0.0

    return {
        "n_segments":          n,
        "total_gemini_calls":  total_calls,
        "cost_ratio":          round(cost_ratio, 3),
        "shot_vote_fired":     shot_fired,
        "shot_vote_pct":       round(100 * shot_fired / n, 1),
        "goal_vote_fired":     goal_fired,
        "goal_vote_pct":       round(100 * goal_fired / n, 1),
        "goal_vote_kept":      goal_kept,
        "goal_vote_rejected":  goal_rejected,
    }


# ---------------------------------------------------------------------------
# Shot-centered window refinement (v11+)
# ---------------------------------------------------------------------------

# Target output window width: pre_peak + post_peak = total seconds.
# 16 seconds (8/8) was picked over Hudl's 12s because hockey threat
# sequences need a few seconds of buildup before the release and a few
# seconds of follow-through after. The 4 extra seconds vs Hudl mostly
# absorbs goalie reactions and rebounds.
REFINE_PRE_SHOT_SEC  = 8
REFINE_POST_SHOT_SEC = 8

# Floor matching cv_seg's MIN_THREAT_DUR. If a refinement would shrink
# below this we expand symmetrically until the floor is met or we hit
# the original segment's boundaries.
REFINE_MIN_WIDTH_SEC = 15


def _parse_clip_mm_ss(s: str) -> int | None:
    """Parse 'MM:SS' to integer seconds.  Returns None on malformed
    input.  Mirrors the parser in eval_metric_seg_output.py."""
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
    return mm * 60 + ss


def _refine_segment_window(segment: dict) -> dict:
    """Compute shot-centered refined boundaries for a single segment.

    Returns the segment dict with two new keys added:
      - segment_start_refined: int seconds (= segment_start if no refinement)
      - segment_end_refined:   int seconds (= segment_end   if no refinement)
      - refinement_source:     "shot_timestamps" | "fallback_no_shots"
                             | "fallback_unparseable" | "fallback_no_metrics"
                             | "fallback_not_threat"

    Originals (segment_start, segment_end) are NEVER mutated.
    Downstream consumers can choose which to use.

    Why this lives in metrics_seg rather than cv_seg:
      The signal driving the refinement (shot_timestamps) only exists
      after Gemini has run.  cv_seg has no access to shot moments —
      its motion-peak proxy was tested in eval_snapshot_shot_centering
      and failed (mean IoU dropped 0.05).  Gemini's shot timestamps
      were the alternative that worked (79-91% within-coverage recall
      on the 5-video v11 eval).
    """
    refined = dict(segment)  # shallow copy — we mutate the copy

    orig_start = segment.get("segment_start")
    orig_end   = segment.get("segment_end")

    # Default: refined == original.  This handles both the no-threat
    # case and any other case where we can't compute refinement.
    refined["segment_start_refined"] = orig_start
    refined["segment_end_refined"]   = orig_end

    if not segment.get("segmentHasThreat"):
        refined["refinement_source"] = "fallback_not_threat"
        return refined

    metrics = segment.get("metrics")
    if not isinstance(metrics, dict):
        refined["refinement_source"] = "fallback_no_metrics"
        return refined

    ts_list = metrics.get("shot_timestamps")
    if not isinstance(ts_list, list) or len(ts_list) == 0:
        refined["refinement_source"] = "fallback_no_shots"
        return refined

    # Parse all shot timestamps to absolute seconds
    abs_shot_times: list[int] = []
    for ts in ts_list:
        if not isinstance(ts, dict):
            continue
        offset = _parse_clip_mm_ss(ts.get("timestamp", "") or "")
        if offset is None:
            continue
        abs_t = orig_start + offset
        # Clamp to segment bounds — Gemini occasionally returns a
        # timestamp slightly past the clip end due to off-by-one in
        # its MM:SS rounding.  Defensive clamp.
        abs_t = max(orig_start, min(orig_end, abs_t))
        abs_shot_times.append(abs_t)

    if not abs_shot_times:
        refined["refinement_source"] = "fallback_unparseable"
        return refined

    # Compute refined boundaries based on shot positions.
    # Single shot:   center ±REFINE_*_SHOT_SEC on the shot
    # Multiple shots: span from first - PRE to last + POST
    first_shot = min(abs_shot_times)
    last_shot  = max(abs_shot_times)

    refined_start = max(orig_start, first_shot - REFINE_PRE_SHOT_SEC)
    refined_end   = min(orig_end,   last_shot  + REFINE_POST_SHOT_SEC)

    # Enforce minimum width.  If refinement would produce <MIN_WIDTH,
    # expand symmetrically until we hit MIN_WIDTH or the original
    # segment boundary, whichever comes first.
    width = refined_end - refined_start
    if width < REFINE_MIN_WIDTH_SEC:
        deficit = REFINE_MIN_WIDTH_SEC - width
        # Distribute the deficit half before, half after the shot(s)
        pad_before = deficit // 2
        pad_after  = deficit - pad_before
        new_start  = max(orig_start, refined_start - pad_before)
        new_end    = min(orig_end,   refined_end   + pad_after)
        # If one side hit the boundary, push the other side harder
        consumed_before = refined_start - new_start
        consumed_after  = new_end - refined_end
        if consumed_before < pad_before:
            # Push end further
            extra = pad_before - consumed_before
            new_end = min(orig_end, new_end + extra)
        if consumed_after < pad_after:
            extra = pad_after - consumed_after
            new_start = max(orig_start, new_start - extra)
        refined_start, refined_end = new_start, new_end

    refined["segment_start_refined"] = int(refined_start)
    refined["segment_end_refined"]   = int(refined_end)
    refined["refinement_source"]     = "shot_timestamps"
    return refined


def _refine_all_segments(segments: list[dict]) -> tuple[list[dict], dict]:
    """Apply _refine_segment_window to every segment.  Returns the
    refined segment list AND a summary dict for logging."""
    out = [_refine_segment_window(s) for s in segments]
    sources: dict[str, int] = {}
    width_deltas: list[int] = []
    for s in out:
        src = s.get("refinement_source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        # Track width change for shot_timestamps-refined segments only
        if src == "shot_timestamps":
            orig_w     = s["segment_end"] - s["segment_start"]
            refined_w  = s["segment_end_refined"] - s["segment_start_refined"]
            width_deltas.append(refined_w - orig_w)
    summary = {
        "refinement_sources": sources,
        "n_refined":          len(width_deltas),
        "avg_width_delta":    (sum(width_deltas) / len(width_deltas)) if width_deltas else 0,
        "min_width_delta":    min(width_deltas) if width_deltas else 0,
        "max_width_delta":    max(width_deltas) if width_deltas else 0,
    }
    return out, summary


def gcs_upload_file(local_path: str, bucket_name: str, blob_name: str) -> str:
    """UNUSED since v8 — kept for backwards compat with any external
    callers. v8 sends clip bytes inline via Part.from_bytes; no GCS
    upload is needed."""
    blob = _get_gcs_client().bucket(bucket_name).blob(blob_name)
    blob.chunk_size = 8 * 1024 * 1024
    blob.upload_from_filename(local_path)
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    log.info("Uploaded clip to GCS", extra={"uri": gcs_uri})
    return gcs_uri


def gcs_delete_blob(bucket_name: str, blob_name: str) -> None:
    """UNUSED since v8 — kept for backwards compat. See gcs_upload_file."""
    try:
        blob = _get_gcs_client().bucket(bucket_name).blob(blob_name)
        if blob.exists():
            blob.delete()
            log.info("Deleted GCS temp clip", extra={"blob": blob_name})
    except Exception as e:
        log.warning(f"Could not delete GCS blob {blob_name}: {e}")


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def extract_clip(input_path: str, start_sec: int, end_sec: int, output_path: str) -> None:
    """Extract a clip using stream copy.

    -ss is placed AFTER -i (slow seek mode). With -i first then -ss, ffmpeg
    decodes from the start of the file up to the seek point — slower than
    fast/keyframe seek but gives sub-keyframe-accurate start times when
    combined with -c copy. The original code placed -ss BEFORE -i which
    is fast but lands the clip on the previous keyframe, potentially
    several seconds before the requested start. For threat windows of
    15-60s we care about the seconds.

    Even with this layout, -c copy can only cut at keyframes, so the
    actual clip start may still snap to a nearby keyframe. To get
    truly frame-accurate clips we'd need to re-encode (-c:v libx264),
    which is much slower. The current setup is a reasonable middle
    ground: typical broadcast keyframes are 2s apart, so worst-case
    drift is ~2s rather than ~10s.

    Subprocess has a hard timeout to prevent hangs from corrupted
    input or stalled filesystem reads from blocking the pipeline.
    """
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i",  input_path,
        "-ss", str(start_sec),
        "-t",  str(duration),
        "-c",  "copy",
        # avoid_negative_ts is needed because slow-seek + stream copy
        # can produce negative timestamps in the output that break
        # downstream players.
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=EXTRACT_CLIP_TIMEOUT_SEC,
                       capture_output=True, text=True)
    except subprocess.TimeoutExpired as e:
        log.error(
            f"ffmpeg timed out after {EXTRACT_CLIP_TIMEOUT_SEC}s",
            extra={"start_sec": start_sec, "end_sec": end_sec,
                   "input": input_path}
        )
        raise
    except subprocess.CalledProcessError as e:
        log.error(
            f"ffmpeg failed (exit code {e.returncode})",
            extra={"start_sec": start_sec, "end_sec": end_sec,
                   "stderr": (e.stderr or "")[:500]}
        )
        raise


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def _call_gemini_for_metrics(
    video_bytes: bytes,
    prompt: str,
    segment_start: int,
    gemini_client,
    *,
    temperature: float = 0.0,
) -> dict | None:
    """
    Send a single prompt+video request to Gemini and parse one metrics dict.

    The video is sent INLINE as bytes (Part.from_bytes) — no GCS upload
    or referenced URI. Vertex's request size cap is ~20 MB; the caller
    is responsible for ensuring video_bytes is well under that limit.
    See analyze_clip_metrics for the size guard.

    Args:
        video_bytes: Raw mp4 bytes of the clip. The caller reads the
            file once and passes the same bytes to all calls in a
            multi-call vote, so we don't re-read the file from disk
            on each retry/extra.
        temperature: Sampling temperature. Default 0 for deterministic
            results on the first call. The multi-call paths use a
            non-zero value on extras to introduce variance — at
            temperature=0 every extra call would return the same answer
            as the first and the median/vote would be a no-op.

    Returns the parsed metrics dict (with the shotsOnNet=saves+goals
    identity already enforced) or None after exhausting MAX_RETRIES.

    Failure modes handled:
      - JSON parse errors (with truncation recovery)
      - Empty / safety-blocked / recitation-blocked responses
        (early-abort: no point retrying, the model will refuse again)
      - Transient API errors (DEADLINE_EXCEEDED, ServiceUnavailable,
        ResourceExhausted, etc) — retried with capped exponential
        backoff
      - Non-transient API errors — single attempt, no retry
      - Empty / generic confirming_detail when goals >= 1 — flagged
        in logs and the goals claim is downgraded to 0 (treats it as
        a hallucination since the prompt requires concrete detail)
    """
    client = gemini_client
    last_error: BaseException | None = None

    # Per-attempt delay is capped to keep the worst-case wait per
    # segment manageable. The original code went 60 → 120 → ... → 960s
    # for total ~30min/segment; capped at 60s the worst case is ~5min,
    # which matches typical Gemini transient-error recovery times.
    def _backoff(attempt: int) -> float:
        delay = min(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
        # Add up to 30% jitter to avoid thundering herd if many segments
        # hit the same rate limit at once.
        return delay + random.uniform(0, 0.3) * delay

    for attempt in range(1, MAX_RETRIES + 1):
        response = None
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=8192,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=METRICS_RESPONSE_SCHEMA,
                ),
            )

            # ── Detect terminal failures BEFORE trying to parse ──────────
            # When Gemini blocks a response on safety / recitation / other
            # policy reasons, response.text is empty/None. Retrying won't
            # help — the model will refuse again. We surface this clearly
            # rather than letting it loop through MAX_RETRIES of
            # JSONDecodeErrors (which costs ~5 minutes of pointless
            # backoff before giving up).
            finish_reason = _extract_finish_reason(response)
            if finish_reason in TERMINAL_FINISH_REASONS:
                log.error(
                    f"Gemini terminal block on attempt {attempt}: {finish_reason}",
                    extra={
                        "segment_start": segment_start,
                        "finish_reason": finish_reason,
                    }
                )
                return None

            raw = (response.text or "").strip()
            if not raw:
                # Empty response with no terminal finish_reason — odd.
                # Could be an internal Gemini hiccup; allow ONE retry.
                log.warning(
                    f"Empty response on attempt {attempt} "
                    f"(finish_reason={finish_reason!r})",
                    extra={"segment_start": segment_start}
                )
                last_error = RuntimeError(f"empty response, finish_reason={finish_reason}")
                if attempt < MAX_RETRIES:
                    time.sleep(_backoff(attempt))
                continue

            # ── Parse JSON, with truncation recovery ────────────────────
            try:
                metrics = json.loads(raw)
            except json.JSONDecodeError as parse_err:
                metrics = _try_recover_truncated_json(raw, segment_start)
                if metrics is None:
                    # Real parse failure with a non-empty body.
                    raise parse_err

            # ── Enforce shotsOnNet = saves + goals ──────────────────────
            sog   = metrics.get("shotsOnNet", 0)
            saves = metrics.get("saves", 0)
            goals = metrics.get("goals", 0)
            if sog != saves + goals:
                log.warning(
                    "Metrics inconsistency: shotsOnNet != saves + goals — correcting",
                    extra={
                        "segment_start": segment_start,
                        "shotsOnNet": sog, "saves": saves, "goals": goals,
                    }
                )
                metrics["shotsOnNet"] = saves + goals

            # ── v11: shot_timestamps post-processing ─────────────────────
            # The v11 prompt asks Gemini to enumerate each shot as a
            # structured record with timestamp/location/release/outcome.
            # The smoke test on Fjc9hmK8_3U revealed two small data-
            # cleanliness drifts vs. the unstructured-prose v10 path:
            #
            #   1. ~7% of segments had len(shot_timestamps) > shots
            #      (e.g. shots=4, ts=5). The structured list has fewer
            #      guardrails than the count discipline, so Gemini's
            #      under-counting language ("rather UNDER-count than
            #      inflate") protects `shots` but lets the enumeration
            #      bloat by one or two entries.
            #   2. ~3 segments had outcome="goal" in shot_timestamps
            #      while goals=0 — Gemini saw something goal-like but
            #      failed the v10 anchor checks. The structured field
            #      has no goal_criteria gate.
            #
            # We enforce both invariants here so downstream code
            # (window-refinement, eval validation) can trust the field.
            # Rule 1 truncates the structured list to the count.
            # Rule 2 demotes goal→save (a "near goal" without anchor
            # confirmation is the same shape as a save in the count
            # identity above).
            ts_list = metrics.get("shot_timestamps")
            if isinstance(ts_list, list):
                if len(ts_list) > metrics.get("shots", 0):
                    log.warning(
                        "shot_timestamps length exceeds shots — truncating",
                        extra={
                            "segment_start": segment_start,
                            "shots": metrics.get("shots", 0),
                            "ts_len": len(ts_list),
                        }
                    )
                    metrics["shot_timestamps"] = ts_list[: metrics.get("shots", 0)]
                    ts_list = metrics["shot_timestamps"]

                if metrics.get("goals", 0) == 0:
                    n_demoted = 0
                    for ts in ts_list:
                        if isinstance(ts, dict) and ts.get("outcome") == "goal":
                            ts["outcome"] = "save"
                            n_demoted += 1
                    if n_demoted:
                        log.warning(
                            "shot_timestamps had outcome='goal' in a "
                            "goals=0 segment — demoting to 'save'",
                            extra={
                                "segment_start": segment_start,
                                "n_demoted": n_demoted,
                            }
                        )

                # ── v13: server-side truth-table enforcement ──────────────
                # Belt + suspenders. The prompt requires the truth table,
                # but Gemini may still emit entries that violate it.
                # We drop entries where:
                #   - REQUIRED (puck_traveling_toward_net) is False
                #   - DISQUALIFIER (puck_carrier_holds_or_passes) is True
                #   - Neither EVIDENCE feature (release_frame_visible OR
                #     impact_sound_audible) is True
                #
                # An entry that's missing any of the four feature fields
                # is preserved (backward compatibility for any v11-style
                # responses that slip through schema validation). This
                # check only PRUNES; it doesn't add or rewrite entries.
                n_pruned_required    = 0
                n_pruned_disqualifier = 0
                n_pruned_no_evidence = 0
                kept: list = []
                for ts in ts_list:
                    if not isinstance(ts, dict):
                        kept.append(ts)
                        continue
                    # Use .get() with default None so missing fields
                    # don't prune (backward-compat).
                    feat_travel    = ts.get("feature_puck_traveling_toward_net")
                    feat_release   = ts.get("feature_puck_release_frame_visible")
                    feat_impact    = ts.get("feature_puck_impact_sound_audible")
                    feat_disqualif = ts.get("feature_puck_carrier_holds_or_passes")

                    # If ALL four fields are missing (v11-style entry),
                    # keep it for back-compat — don't apply truth table.
                    if (feat_travel is None and feat_release is None and
                        feat_impact is None and feat_disqualif is None):
                        kept.append(ts)
                        continue

                    # REQUIRED rule — explicit False prunes.
                    if feat_travel is False:
                        n_pruned_required += 1
                        continue
                    # DISQUALIFIER rule — explicit True prunes.
                    if feat_disqualif is True:
                        n_pruned_disqualifier += 1
                        continue
                    # EVIDENCE rule — both explicit False prunes.
                    if feat_release is False and feat_impact is False:
                        n_pruned_no_evidence += 1
                        continue
                    kept.append(ts)

                n_pruned = (n_pruned_required + n_pruned_disqualifier
                            + n_pruned_no_evidence)
                if n_pruned > 0:
                    metrics["shot_timestamps"] = kept
                    ts_list = kept
                    # Re-derive shots count from the pruned list.
                    new_shots = len(kept)
                    old_shots = metrics.get("shots", 0)
                    metrics["shots"] = new_shots
                    # shotsOnNet bounded by shots.
                    if metrics.get("shotsOnNet", 0) > new_shots:
                        metrics["shotsOnNet"] = new_shots
                    # saves = shotsOnNet - goals (preserve the identity).
                    metrics["saves"] = max(
                        0, metrics.get("shotsOnNet", 0) - metrics.get("goals", 0)
                    )
                    log.info(
                        "v13 truth-table prune — adjusted counts",
                        extra={
                            "segment_start": segment_start,
                            "pruned_required": n_pruned_required,
                            "pruned_disqualifier": n_pruned_disqualifier,
                            "pruned_no_evidence": n_pruned_no_evidence,
                            "shots_before": old_shots,
                            "shots_after": new_shots,
                        }
                    )

            # ── Validate confirming_detail when goals >= 1 ──────────────
            # The prompt requires concrete visual specifics (player number,
            # exact location, etc.) when claiming a goal. An empty or
            # generic detail string is evidence the model is hallucinating
            # — downgrade goals to 0 and re-derive saves.
            criteria = metrics.get("goal_criteria") or {}
            if goals >= 1:
                detail = (criteria.get("confirming_detail") or "").strip()
                if not detail or _is_generic_detail(detail):
                    log.warning(
                        "Goals claimed but confirming_detail is empty/generic "
                        "— downgrading goals to 0",
                        extra={
                            "segment_start": segment_start,
                            "first_goals": goals,
                            "confirming_detail": detail,
                        }
                    )
                    metrics["goals"] = 0
                    metrics["saves"] = max(0, metrics.get("shotsOnNet", 0))
                    # Also re-cascade the shot_timestamps outcome demotion
                    # since the segment is no longer a goal segment.
                    ts_list2 = metrics.get("shot_timestamps")
                    if isinstance(ts_list2, list):
                        n_demoted2 = 0
                        for ts in ts_list2:
                            if isinstance(ts, dict) and ts.get("outcome") == "goal":
                                ts["outcome"] = "save"
                                n_demoted2 += 1
                        if n_demoted2:
                            log.info(
                                "shot_timestamps outcome='goal' demoted "
                                "after confirming_detail downgrade",
                                extra={
                                    "segment_start": segment_start,
                                    "n_demoted": n_demoted2,
                                }
                            )

            # ── Debug log goal_criteria when interesting ────────────────
            if criteria:
                anchors_hit = [k for k in (
                    "anchor_puck_crosses_line",
                    "anchor_ref_points_at_net",
                    "anchor_puck_retrieved_from_net",
                ) if criteria.get(k)]
                supports_hit = [k for k in (
                    "support_whistle", "support_crowd_spike",
                    "support_celebration", "support_centre_ice_faceoff",
                ) if criteria.get(k)]
                if metrics.get("goals", 0) > 0 or anchors_hit:
                    log.info(
                        "Goal criteria",
                        extra={
                            "segment_start": segment_start,
                            "goals": metrics.get("goals", 0),
                            "anchors": anchors_hit,
                            "supports": supports_hit,
                            "disqualifier": criteria.get("disqualifier_active", False),
                            "notes": criteria.get("decision_notes", ""),
                        }
                    )

            log.info(
                "Metrics extracted",
                extra={"segment_start": segment_start, "metrics": {
                    k: v for k, v in metrics.items() if k != "goal_criteria"
                }, "attempt": attempt}
            )
            return metrics

        except json.JSONDecodeError as e:
            raw_text = (response.text if response is not None else "") or ""
            pos = getattr(e, "pos", 0)
            ctx_start = max(0, pos - 100)
            ctx_end = min(len(raw_text), pos + 100)
            log.warning(
                f"JSON parse error on attempt {attempt}: {e}",
                extra={
                    "segment_start": segment_start,
                    "raw_len": len(raw_text),
                    "error_pos": pos,
                    "context": raw_text[ctx_start:ctx_end],
                }
            )
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(_backoff(attempt))

        except _TRANSIENT_API_EXCEPTIONS as e:
            last_error = e
            if attempt < MAX_RETRIES:
                delay = _backoff(attempt)
                log.warning(
                    f"Transient error on attempt {attempt} — retrying in {delay:.1f}s",
                    extra={"segment_start": segment_start, "error": str(e)}
                )
                time.sleep(delay)
            else:
                log.error(
                    f"Transient error on final attempt {attempt}: {e}",
                    extra={"segment_start": segment_start}
                )

        except Exception as e:
            # Fallback string-match for transient errors that aren't
            # in the typed exception list (network reset, SSL, etc.)
            # Also catches genai.errors.APIError (the new google.genai
            # SDK raises these instead of api_core exceptions, so the
            # typed _TRANSIENT_API_EXCEPTIONS tuple — which uses
            # api_core types — never matched 429s. Added 429/
            # RESOURCE_EXHAUSTED to the string heuristic to recover
            # rate-limited calls properly.)
            last_error = e
            transient_by_message = any(k in str(e) for k in [
                "SSL", "EOF", "timed out", "timeout", "Connection", "reset",
                "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE",
                "504", "DEADLINE_EXCEEDED", "500", "INTERNAL",
            ])
            if transient_by_message and attempt < MAX_RETRIES:
                delay = _backoff(attempt)
                log.warning(
                    f"Transient (by message) error on attempt {attempt} — "
                    f"retrying in {delay:.1f}s",
                    extra={"segment_start": segment_start, "error": str(e)}
                )
                time.sleep(delay)
            else:
                log.error(
                    f"Non-transient error on attempt {attempt}: {e}",
                    extra={"segment_start": segment_start}
                )
                return None

    log.error(
        f"All {MAX_RETRIES} attempts failed",
        extra={"segment_start": segment_start, "last_error": str(last_error)}
    )
    return None


# ---------------------------------------------------------------------------
# Gemini response inspection helpers
# ---------------------------------------------------------------------------

# Terminal finish_reasons: retrying won't help, the model will refuse again.
# These names match the Gemini API's FinishReason enum values when stringified.
TERMINAL_FINISH_REASONS = frozenset({
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
})

# Non-terminal: STOP (success), MAX_TOKENS (truncation — caught by JSON
# recovery path), OTHER (worth one retry).

_TRANSIENT_API_EXCEPTIONS = (
    gcore_exceptions.ServiceUnavailable,    # 503
    gcore_exceptions.DeadlineExceeded,      # 504
    gcore_exceptions.ResourceExhausted,     # 429
    gcore_exceptions.InternalServerError,   # 500
    gcore_exceptions.Aborted,               # 409
    gcore_exceptions.RetryError,            # generic retry wrapper
)


def _extract_finish_reason(response) -> str | None:
    """Pull the finish_reason from the first candidate, as a string.

    The Gemini SDK returns finish_reason as either an enum or a string
    depending on version. Coerce to string for consistent comparison.
    """
    try:
        cand = response.candidates[0]
        fr = getattr(cand, "finish_reason", None)
        if fr is None:
            return None
        # enum.name preferred (e.g. FinishReason.SAFETY -> "SAFETY")
        return getattr(fr, "name", str(fr)).upper()
    except (AttributeError, IndexError, TypeError):
        return None


def _try_recover_truncated_json(raw: str, segment_start: int) -> dict | None:
    """Attempt to recover from JSON truncated mid-string by trimming
    back to the last balanced closing brace. Returns the parsed dict
    or None if recovery fails."""
    last_brace = raw.rfind("}")
    if last_brace <= 0:
        return None
    candidate = raw[: last_brace + 1]
    depth = 0
    end = -1
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end <= 0:
        return None
    try:
        recovered = json.loads(candidate[:end])
        log.warning(
            "Recovered metrics from truncated JSON",
            extra={
                "segment_start": segment_start,
                "raw_len": len(raw),
                "recovered_len": end,
            }
        )
        return recovered
    except json.JSONDecodeError:
        return None


# Generic phrases the model uses when it can't actually pin down a goal
# but the rubric tries to force it. Anything matching is treated as
# evidence the goal claim is fabricated. Lowercase for comparison.
_GENERIC_DETAIL_PATTERNS = (
    "a goal is counted",
    "to maintain the identity",
    "to satisfy the rubric",
    "the truth table",
    "appears to be a goal",
    "likely a goal",
    "i believe",
)


def _is_generic_detail(detail: str) -> bool:
    """Return True if confirming_detail looks like generic hockey
    narration rather than a specific visual observation. The prompt
    requires per-claim concrete detail (player number, exact location,
    etc.) — generic phrases are a hallucination tell."""
    s = detail.lower()
    if any(pat in s for pat in _GENERIC_DETAIL_PATTERNS):
        return True
    # Very short detail strings are almost certainly not concrete.
    # The schema allows empty when goals=0; for goals>=1 we expect
    # a sentence describing what was seen.
    if len(detail.split()) < 5:
        return True
    return False




def _median_int(values: list[int]) -> int:
    """Median of a non-empty list of ints. Even-length lists round down."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    if n % 2 == 1:
        return s[n // 2]
    # Even — return the lower of the two middle values (deterministic, conservative).
    return s[n // 2 - 1]


def analyze_clip_metrics(
    clip_path: str,
    goalie_color: str,
    goalie_side: str,
    duration: int,
    segment_start: int,
    gemini_client,
    *,
    opponent_color: str = "the attacking team",
) -> tuple[dict | None, dict]:
    """
    Read clip bytes from disk and send to Gemini for metric counting.
    Returns (parsed dict, trace). The trace is always populated even
    when the metrics dict is None — it documents what happened (call
    counts, per-call results, vote decisions) for the
    gt_metrics_*_trace.json sidecar. Returns (None, trace) on failure
    after all retries.

    v4: Conditional 3-call median for `shots` and `shotsOnNet`.
    v6: Conditional 3-call majority vote for `goals`.
    v8: Video sent INLINE (no GCS upload). Bytes are read once and
        passed to all calls in the multi-call vote, so re-reading the
        file from disk on each retry is avoided. Tradeoff: the same
        bytes are re-transmitted on each call (no server-side cache),
        but cumulative wire cost is small for our 5-12 MB clips and
        we save the GCS upload + delete round-trip entirely.

    When both vote conditions trigger on the same clip, the extras are
    SHARED — we fire 2 extras and use them for both votes. After
    voting, `saves` is re-derived from `shotsOnNet - goals` to preserve
    the identity. goal_criteria is preserved from the first call
    (no merging boolean flags across runs).
    """
    # Trace accumulator — populated as we go, returned even on failure.
    trace: dict = {
        "segment_start": segment_start,
        "duration": duration,
        "n_calls": 0,
        "shot_vote_triggered": False,
        "goal_vote_triggered": False,
        "shot_vote_outcome":   None,    # "applied" | "extras_failed"
        "goal_vote_outcome":   None,    # "kept" | "rejected" | "extras_failed"
        "per_call_shots":      [],
        "per_call_shotsOnNet": [],
        "per_call_goals":      [],
        "first_call_shots":    None,
        "first_call_goals":    None,
        "median_shots":        None,
        "median_shotsOnNet":   None,
        "yes_votes":           None,
        "final_shots":         None,
        "final_shotsOnNet":    None,
        "final_goals":         None,
        "failure_reason":      None,
    }

    # ── Read clip bytes inline ─────────────────────────────────────────
    # Vertex's request size limit is ~20 MB. Our clips at 30-60s 720p
    # H.264 are 5-12 MB, but a guard is still prudent: an unusual
    # encoding or longer-than-expected clip could exceed it. If a clip
    # is too large we surface a clear failure rather than letting the
    # underlying API return an opaque size error several seconds in.
    try:
        clip_size = os.path.getsize(clip_path)
    except OSError as e:
        trace["failure_reason"] = f"clip_read_failed: {e}"
        log.error(
            f"Failed to stat clip file: {e}",
            extra={"segment_start": segment_start, "clip_path": clip_path}
        )
        return None, trace

    if clip_size > MAX_INLINE_VIDEO_BYTES:
        trace["failure_reason"] = (
            f"clip_too_large: {clip_size} bytes > {MAX_INLINE_VIDEO_BYTES} limit"
        )
        log.error(
            f"Clip exceeds inline-byte limit ({clip_size} > "
            f"{MAX_INLINE_VIDEO_BYTES}); cannot send to Gemini without GCS path",
            extra={
                "segment_start": segment_start,
                "clip_size_bytes": clip_size,
                "limit": MAX_INLINE_VIDEO_BYTES,
            }
        )
        return None, trace

    with open(clip_path, "rb") as f:
        video_bytes = f.read()

    prompt = METRICS_PROMPT.format(
        duration=duration,
        goalie_color=goalie_color,
        opponent_color=opponent_color,
        side=goalie_side or SIDE_UNKNOWN,
    )

    # ── v14 Phase 2: Flash pre-filter screening ────────────────────────
    # When --flash-screen is on, ask Flash (cheap) whether this clip has
    # any shot activity before paying for a Pro call. Fail-safe: any
    # Flash error escalates to Pro anyway.
    if _V14_IMPROVEMENTS_AVAILABLE and _V14_CONFIG.get("flash_screen"):
        screen = _v14_flash.screen_clip(
            video_bytes, duration, gemini_client,
            enabled=True, fail_safe=True,
        )
        trace["flash_screen_fired"]      = True
        trace["flash_screen_shots_any"]  = screen.shots_any
        trace["flash_screen_confidence"] = screen.confidence
        trace["flash_screen_failed"]     = screen.failed
        trace["flash_screen_fail_reason"] = screen.fail_reason
        if _v14_flash.should_skip_pro(screen):
            log.info(
                "Flash screen → skip Pro",
                extra={
                    "segment_start": segment_start,
                    "shots_any":  screen.shots_any,
                    "confidence": screen.confidence,
                }
            )
            trace["flash_screen_skipped_pro"] = True
            trace["final_shots"]      = 0
            trace["final_shotsOnNet"] = 0
            trace["final_goals"]      = 0
            return _v14_flash.null_metrics_for_skip(
                segment_start, segment_start + duration), trace
        trace["flash_screen_skipped_pro"] = False

    # ── First call (temperature=0, deterministic anchor) ──────────────
    first = _call_gemini_for_metrics(video_bytes, prompt, segment_start, gemini_client)
    trace["n_calls"] = 1
    if first is None:
        trace["failure_reason"] = "first_call_failed"
        return None, trace

    first_shots = first.get("shots", 0)
    first_goals = first.get("goals", 0)
    trace["first_call_shots"] = first_shots
    trace["first_call_goals"] = first_goals
    trace["per_call_shots"].append(first_shots)
    trace["per_call_shotsOnNet"].append(first.get("shotsOnNet", 0))
    trace["per_call_goals"].append(first_goals)

    shot_vote_triggered = first_shots > MULTICALL_SHOTS_THRESHOLD
    goal_vote_triggered = first_goals >= MULTICALL_GOAL_VOTE_TRIGGER
    trace["shot_vote_triggered"] = shot_vote_triggered
    trace["goal_vote_triggered"] = goal_vote_triggered

    if not shot_vote_triggered and not goal_vote_triggered:
        # Cheap path — no vote conditions met, single call is enough.
        trace["final_shots"]      = first_shots
        trace["final_shotsOnNet"] = first.get("shotsOnNet", 0)
        trace["final_goals"]      = first_goals
        return first, trace

    # ── Multi-call extras (shared across both votes) ──────────────
    log.info(
        "Multi-call extras triggered",
        extra={
            "segment_start": segment_start,
            "first_shots": first_shots,
            "first_goals": first_goals,
            "shot_vote": shot_vote_triggered,
            "goal_vote": goal_vote_triggered,
            "extra_calls": MULTICALL_EXTRA_CALLS,
        }
    )

    extras: list[dict] = []
    for i in range(MULTICALL_EXTRA_CALLS):
        extra = _call_gemini_for_metrics(
            video_bytes, prompt, segment_start, gemini_client,
            temperature=MULTICALL_EXTRA_TEMPERATURE,
        )
        trace["n_calls"] += 1
        if extra is not None:
            extras.append(extra)
            trace["per_call_shots"].append(extra.get("shots", 0))
            trace["per_call_shotsOnNet"].append(extra.get("shotsOnNet", 0))
            trace["per_call_goals"].append(extra.get("goals", 0))
        else:
            log.warning(
                f"Multi-call extra {i+1}/{MULTICALL_EXTRA_CALLS} failed — continuing",
                extra={"segment_start": segment_start}
            )

    if not extras:
        # All extras failed — fall back to the first call rather than
        # losing the window entirely.
        log.warning(
            "Multi-call extras all failed — using first-call result",
            extra={
                "segment_start": segment_start,
                "first_shots": first_shots,
                "first_goals": first_goals,
            }
        )
        if shot_vote_triggered:
            trace["shot_vote_outcome"] = "extras_failed"
        if goal_vote_triggered:
            trace["goal_vote_outcome"] = "extras_failed"
        trace["final_shots"]      = first_shots
        trace["final_shotsOnNet"] = first.get("shotsOnNet", 0)
        trace["final_goals"]      = first_goals
        return first, trace

    pool = [first] + extras

    # Start the merged result from the first call. We'll overwrite
    # only the fields the votes touch.
    merged = dict(first)

    # ── Shot vote: median of shots and shotsOnNet ─────────────────
    if shot_vote_triggered:
        all_shots = [r.get("shots", 0) for r in pool]
        all_son   = [r.get("shotsOnNet", 0) for r in pool]
        med_shots = _median_int(all_shots)
        med_son   = _median_int(all_son)
        trace["median_shots"]      = med_shots
        trace["median_shotsOnNet"] = med_son
        trace["shot_vote_outcome"] = "applied"

        spread_shots = max(all_shots) - min(all_shots)
        if spread_shots > MULTICALL_VARIANCE_WARN:
            log.warning(
                "High variance across multi-call shot results",
                extra={
                    "segment_start": segment_start,
                    "shots_per_call": all_shots,
                    "shotsOnNet_per_call": all_son,
                    "spread_shots": spread_shots,
                }
            )

        merged["shots"]      = med_shots
        merged["shotsOnNet"] = med_son
        if merged["shotsOnNet"] > merged["shots"]:
            merged["shotsOnNet"] = merged["shots"]

        log.info(
            "Multi-call shot median applied",
            extra={
                "segment_start": segment_start,
                "n_calls": len(pool),
                "shots_per_call": all_shots,
                "shotsOnNet_per_call": all_son,
                "median_shots": med_shots,
                "median_shotsOnNet": med_son,
                "first_shots": first_shots,
            }
        )

    # ── Goal vote: majority-of-3 (binary) ─────────────────────────
    if goal_vote_triggered:
        yes_votes = sum(1 for r in pool if r.get("goals", 0) >= MULTICALL_GOAL_VOTE_TRIGGER)
        per_call_goals = [r.get("goals", 0) for r in pool]
        quorum_met = yes_votes >= MULTICALL_GOAL_VOTE_QUORUM
        trace["yes_votes"] = yes_votes

        if quorum_met:
            vote_outcome = "kept"
        else:
            merged["goals"] = 0
            vote_outcome = "rejected"
        trace["goal_vote_outcome"] = vote_outcome

        log.info(
            "Multi-call goal vote applied",
            extra={
                "segment_start": segment_start,
                "n_calls": len(pool),
                "first_call_goals": first_goals,
                "per_call_goals": per_call_goals,
                "yes_votes": yes_votes,
                "quorum_required": MULTICALL_GOAL_VOTE_QUORUM,
                "outcome": vote_outcome,
                "final_goals": merged["goals"],
            }
        )

    # ── Re-derive saves to preserve identity ──────────────────────
    goals_final = merged.get("goals", 0)
    if merged.get("shotsOnNet", 0) < goals_final:
        merged["shotsOnNet"] = goals_final
    merged["saves"] = max(0, merged["shotsOnNet"] - goals_final)

    trace["final_shots"]      = merged.get("shots", 0)
    trace["final_shotsOnNet"] = merged.get("shotsOnNet", 0)
    trace["final_goals"]      = merged.get("goals", 0)

    return merged, trace


# ---------------------------------------------------------------------------
# Parallel segment dispatch
# ---------------------------------------------------------------------------

async def _dispatch_segments_async(
    segments:       list[dict],
    threat_indices: list[int],
    workers:        int,
    local_video:    str,
    goalie_color:   str,
    opponent_color: str,
    gemini_client,
    vID:            str,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """
    Dispatch all threat segments to Gemini in parallel, bounded by a semaphore.

    Returns (results_map, trace_map):
      results_map: {segment_index_in_full_list: enriched_segment_dict}
        Segments that fail are stored with metrics=None.
      trace_map:   {segment_index_in_full_list: trace_dict}
        Per-segment trace data (call counts, vote outcomes, per-call
        results) used to populate the trace sidecar.

    Concurrency: previous implementation was async-shaped but every
    operation inside _run_one was synchronous and blocking, so the
    semaphore serialised work rather than paralleising it. This
    version offloads the blocking ffmpeg + Gemini calls to a
    ThreadPoolExecutor via run_in_executor, which makes the worker
    count meaningful again.
    """
    results_map: dict[int, dict] = {}
    trace_map:   dict[int, dict] = {}
    semaphore = asyncio.Semaphore(workers)
    loop      = asyncio.get_running_loop()
    # Match thread pool size to semaphore — there's no point in having
    # more threads than concurrent slots, and threads cost stack memory.
    executor  = ThreadPoolExecutor(max_workers=workers,
                                   thread_name_prefix=f"seg-{vID}")

    def _blocking_segment_work(
        clip_path: str, start: int, end: int, duration: int,
        color: str, side: str,
    ) -> tuple[dict | None, dict]:
        """All the synchronous heavy work for one segment, run in
        a worker thread. Returns (metrics, trace)."""
        extract_clip(local_video, start, end, clip_path)
        return analyze_clip_metrics(
            clip_path, color, side, duration, start, gemini_client,
            opponent_color=opponent_color,
        )

    async def _run_one(seg_idx_in_threats: int, seg_idx_in_all: int) -> None:
        async with semaphore:
            segment  = segments[seg_idx_in_all]
            start    = segment["segment_start"]
            end      = segment["segment_end"]
            duration = end - start
            side     = segment.get("threat_goalie_side") or SIDE_UNKNOWN
            color = segment.get("threat_goalie_color")
            if not color:
                log.warning(
                    f"[{vID}] Threat segment {start}–{end}s has no goalie color "
                    f"— using config default: {goalie_color}"
                )
                color = goalie_color

            # ─── v14 PRE-FILTER: skip Gemini if peak prob below threshold ───
            if (_V14_CONFIG["enabled"]
                    and _V14_CONFIG["prefilter_threshold"] > 0):
                fp = _v14_load_probs_for_vid(vID)
                if fp is not None:
                    skip, peak = _v14_prefilter.should_skip(
                        fp, start, end, _V14_CONFIG["prefilter_threshold"])
                    if skip:
                        log.info(f"[{vID}] PREFILTER SKIP segment {start}-{end}s "
                                  f"(peak={peak:.3f} < {_V14_CONFIG['prefilter_threshold']})")
                        results_map[seg_idx_in_all] = dict(segment) | {
                            "metrics": _v14_prefilter.null_metrics_dict(peak)}
                        trace_map[seg_idx_in_all] = {
                            "segment_start": start, "duration": duration,
                            "failure_reason": None,
                            "_prefilter_skip": True,
                            "_prefilter_peak_conf": round(peak, 4),
                        }
                        return

            log.info(
                f"Processing threat {seg_idx_in_threats + 1}/{len(threat_indices)}",
                extra={"segment_start": start, "segment_end": end, "duration": duration}
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                clip_path = tmp.name

            metrics: dict | None = None
            trace: dict = {
                "segment_start": start,
                "duration": duration,
                "failure_reason": "not_run",
            }
            try:
                try:
                    # CRITICAL: run blocking work in the executor so the
                    # event loop can dispatch other segments concurrently.
                    metrics, trace = await loop.run_in_executor(
                        executor, _blocking_segment_work,
                        clip_path, start, end, duration, color, side,
                    )
                finally:
                    if os.path.exists(clip_path):
                        os.unlink(clip_path)
            except Exception as e:
                log.error(
                    f"[{vID}] Segment at {start}-{end}s failed: {e}",
                    extra={"segment_start": start, "exc_type": type(e).__name__}
                )
                trace["failure_reason"] = f"{type(e).__name__}: {e}"

            results_map[seg_idx_in_all] = dict(segment) | {"metrics": metrics}
            trace_map[seg_idx_in_all]   = trace

            # Pipeline progress: report after each threat completes
            # (works correctly under asyncio concurrency — len(results_map)
            # is the actual count of completed threats).
            if _pipeline_progress is not None and _PROGRESS_STAGE_IDX is not None:
                _pipeline_progress.report(
                    customer_id=_PROGRESS_CUSTOMER_ID, vid=vID,
                    stage_idx=_PROGRESS_STAGE_IDX,
                    current=len(results_map),
                    total=len(threat_indices),
                )

    try:
        await asyncio.gather(*[
            _run_one(i, seg_idx)
            for i, seg_idx in enumerate(threat_indices)
        ])
    finally:
        executor.shutdown(wait=False)

    return results_map, trace_map


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------

def process_video(
    vID: str,
    customID: str,
    config: list | dict,
    workers: int,
    no_gcs: bool = False,
    segments_dir: str | None = None,
    local_video_dir: str | None = None,
    output_dir: str | None = None,
    skip_existing: bool = False,
) -> bool:
    """
    Run the metrics detection pipeline for a single video.

    Downloads the full video, loads the step-01 segment predictions,
    extracts each threat window as a short clip, sends it to Gemini for
    metric counting (shots, saves, goals, rebounds), and writes the
    enriched output (and a per-segment trace sidecar) to GCS and/or local.

    Args:
        no_gcs:          If True, skip GCS reads when a local file exists.
        segments_dir:    Local directory for gt_seg_{vID}.json. If set, read
                         from here instead of GCS.
        local_video_dir: Local directory for full_{vID}.mp4. If set, skip
                         GCS video download. Falls back to glob-matching
                         *_{vID}.mp4 if exact name is not present.
        output_dir:      Local directory to write gt_metrics_{vID}.json.
                         If set, output is written here in addition to GCS
                         (or instead of GCS, when no_gcs=True).
        skip_existing:   If True, skip processing when output already
                         exists at the configured destination.

    Returns True on success, False if a fatal error occurred.
    """
    log.info(f"{'='*60}")
    log.info(f"Starting metrics detection for vID: {vID}")
    log.info(f"{'='*60}")

    # 0. Skip-existing check (before any I/O / config parsing)
    if skip_existing and _output_already_exists(vID, no_gcs=no_gcs, output_dir=output_dir):
        log.info(f"[{vID}] Output already exists — skipping (--skip-existing)")
        return True

    # 1. Resolve jersey colors
    if isinstance(config, list):
        record = next((r for r in config if str(r.get("vID")) == str(vID)), None)
        if record is None:
            log.error(f"[{vID}] No record found in config — skipping.")
            return False
    else:
        record = config

    goalie_color = record.get("targetGoalieColor")
    if not goalie_color:
        log.error(f"[{vID}] Config missing targetGoalieColor — skipping.")
        return False

    opponent_color = record.get("opponentGoalieColor", "the attacking team")

    log.info(f"[{vID}] Goalie: {goalie_color} | Opponent: {opponent_color}")

    # 2. Load step 01 predictions — local first if available
    segments = None
    if segments_dir:
        local_seg_path = os.path.join(segments_dir, f"gt_seg_{vID}.json")
        if os.path.exists(local_seg_path):
            try:
                with open(local_seg_path, "r") as f:
                    segments = json.load(f)
                log.info(f"[{vID}] Loaded segments from local: {local_seg_path}")
            except Exception as e:
                log.warning(f"[{vID}] Failed to load local segments ({e}) — falling back")

    if segments is None:
        if no_gcs:
            log.error(f"[{vID}] --no-gcs set but local segments not found "
                      f"(segments_dir={segments_dir}). Aborting.")
            return False
        input_blob = f"{INPUT_PREFIX}/gt_seg_{vID}.json"
        try:
            segments = gcs_read_json(GCS_BUCKET, input_blob)
            log.info(f"[{vID}] Loaded segments from GCS: {input_blob}")
        except Exception as e:
            log.error(f"[{vID}] Failed to load segment predictions: {e}")
            return False

    threat_segments = [s for s in segments if s.get("segmentHasThreat")]
    log.info(
        f"[{vID}] Loaded {len(segments)} segments ({len(threat_segments)} threats)"
    )

    if not threat_segments:
        log.warning(f"[{vID}] No threat segments found — writing pass-through output.")
        output = [dict(s) | {"metrics": None} for s in segments]
        _write_metrics_output(
            output, vID, no_gcs=no_gcs, output_dir=output_dir,
        )
        return True

    # 3. Get the cached Gemini client (one per process, shared across videos)
    gemini_client = _get_gemini_client()

    # 4. Obtain full video — prefer local if available
    local_video = None
    cleanup_local_video = False
    if local_video_dir:
        # Exact match first
        candidate = os.path.join(local_video_dir, f"full_{vID}.mp4")
        if os.path.exists(candidate):
            local_video = candidate
            log.info(f"[{vID}] Using local video: {local_video}")
        else:
            # Fallback: any *.mp4 / *.mov containing the vID. Useful for
            # devs who store videos with descriptive prefixes but don't
            # want to symlink. We pick the first match alphabetically
            # for determinism.
            for ext in ("mp4", "mov", "mkv"):
                matches = sorted(glob.glob(os.path.join(local_video_dir, f"*{vID}*.{ext}")))
                if matches:
                    local_video = matches[0]
                    log.info(
                        f"[{vID}] Using local video by glob match: {local_video} "
                        f"(exact name {os.path.basename(candidate)} not found)"
                    )
                    break

    if local_video is None:
        if no_gcs:
            log.error(f"[{vID}] --no-gcs set but local video not found "
                      f"(local_video_dir={local_video_dir}). Aborting.")
            return False
        video_blob = f"{VIDEO_PREFIX}/full_{vID}.mp4"
        try:
            local_video = gcs_download_to_temp(GCS_BUCKET, video_blob, suffix=".mp4")
            cleanup_local_video = True
        except Exception as e:
            log.error(f"[{vID}] Failed to download video: {e}")
            return False

    try:
        # ── 5. Process all threat segments in parallel ─────────────────────
        threat_indices = [i for i, s in enumerate(segments) if s.get("segmentHasThreat")]

        results_map, trace_map = asyncio.run(_dispatch_segments_async(
            segments=segments,
            threat_indices=threat_indices,
            workers=workers,
            local_video=local_video,
            goalie_color=goalie_color,
            opponent_color=opponent_color,
            gemini_client=gemini_client,
            vID=vID,
        ))

        # 6. Rebuild full segment list, inserting metrics for threats
        output = []
        for i, seg in enumerate(segments):
            if i in results_map:
                output.append(results_map[i])
            else:
                # No-threat segment — pass through with metrics: null
                output.append(dict(seg) | {"metrics": None})

        # 6b. Apply shot-centered window refinement.  Adds
        # segment_start_refined / segment_end_refined / refinement_source
        # to every segment without mutating the originals.  Non-threat
        # and metrics-less segments get refined == original via fallback.
        output, refine_summary = _refine_all_segments(output)
        log.info(
            f"[{vID}] Window refinement applied",
            extra={
                "vID": vID,
                **refine_summary,
            }
        )

        # Side corrections removed.
        # Step 1 derives sides deterministically from the customer config,
        # so we trust that assignment end-to-end. observed_goalie_side is
        # still populated on each segment for telemetry but is not used
        # to mutate the output.

        # 7. Summary stats
        succeeded = sum(1 for s in output if s.get("segmentHasThreat") and s.get("metrics") is not None)
        failed    = sum(1 for s in output if s.get("segmentHasThreat") and s.get("metrics") is None)
        # Use `is not None` rather than truthiness — an empty-dict metrics
        # value would be silently excluded otherwise.
        total_goals = sum(
            s["metrics"].get("goals", 0)
            for s in output
            if s.get("metrics") is not None
        )
        total_sog = sum(
            s["metrics"].get("shotsOnNet", 0)
            for s in output
            if s.get("metrics") is not None
        )

        # Vote-fire-rate + per-call summary from the trace map. This
        # lets us validate the cost claim ("realised cost ~1.3-1.5x")
        # against actual data per video.
        vote_summary = _summarize_traces(trace_map)
        log.info(
            f"[{vID}] Metrics complete",
            extra={
                "vID": vID,
                "threat_segments": len(threat_indices),
                "succeeded": succeeded,
                "failed": failed,
                "total_shotsOnNet": total_sog,
                "total_goals": total_goals,
                **vote_summary,
            }
        )

        # 8. Write output — local and/or GCS depending on flags
        _write_metrics_output(output, vID, no_gcs=no_gcs, output_dir=output_dir)

        # 9. Write trace sidecar — gt_metrics_{vID}_trace.json. Same
        # destination policy as the main output, but as a SEPARATE file
        # so the main output schema stays untouched and downstream
        # consumers don't have to know about traces.
        trace_payload = {
            "vID":           vID,
            "summary":       vote_summary,
            "per_segment":   [trace_map[i] for i in sorted(trace_map.keys())],
        }
        _write_trace_sidecar(trace_payload, vID, no_gcs=no_gcs, output_dir=output_dir)

        return True

    finally:
        if cleanup_local_video and local_video and os.path.exists(local_video):
            os.unlink(local_video)
            log.info(f"[{vID}] Deleted local video.")

        # Note: previous versions did a GCS "safety sweep" here for any
        # leftover temp clips at TEMP_PREFIX. Since v8, clips are sent
        # inline via Part.from_bytes — no GCS upload happens at all,
        # so there are no temp blobs to clean up. The sweep was removed
        # for that reason.


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Detect per-window metrics for threat segments")
    parser.add_argument("--vID", required=True, nargs="+", metavar="vID",
                        help="One or more video IDs to process")
    parser.add_argument("--customID", required=True,
                        help="Customer config JSON blob name (e.g. CUST000048)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel Gemini requests within a single "
                             "video (default: 2). Concurrency uses real threads now, "
                             "so this number is meaningful.")
    parser.add_argument("--video-workers", type=int, default=1,
                        help="Number of videos to process concurrently (default: 1). "
                             "Total in-flight Gemini requests is workers * video-workers, "
                             "so be mindful of API quota.")
    # Local-first overrides
    parser.add_argument("--no-gcs", action="store_true",
                        help="Prefer local files for segments and video input. "
                             "Output is still written to GCS unless --output-dir is also set.")
    parser.add_argument("--segments-dir", default=None,
                        help="Local directory containing gt_seg_{vID}.json. "
                             "If set, read from here instead of GCS.")
    parser.add_argument("--local-video-dir", default=None,
                        help="Local directory containing full_{vID}.mp4. "
                             "Falls back to glob *{vID}*.mp4/.mov/.mkv if exact name "
                             "not found. If set, skip GCS video download.")
    parser.add_argument("--output-dir", default=None,
                        help="Local directory to write gt_metrics_{vID}.json. "
                             "If set, output is written here instead of (or in addition to) GCS.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip processing a vID when the output file already "
                             "exists at the configured destination. Useful for "
                             "batch reprocessing or recovery from partial failures.")

    # ─── v14 improvements (see IMPROVEMENTS_SPEC.md) ─── all opt-in
    g = parser.add_argument_group("v14 improvements (opt-in)")
    g.add_argument("--cache-dir", default=None,
                    help="Directory for the Gemini-response cache "
                         "(default: ~/.cache/metrics_seg). Set to empty "
                         "string or use --no-cache to disable.")
    g.add_argument("--no-cache", action="store_true",
                    help="Disable the Gemini-response cache.")
    g.add_argument("--prefilter-threshold", type=float, default=0.0,
                    help="Skip Gemini calls for windows whose peak "
                         "YOLO+audio fused prob is below this threshold "
                         "(default 0.0, i.e. disabled). Try 0.30 to "
                         "save ~30-50%% of calls.")
    g.add_argument("--use-context", action="store_true",
                    help="Prepend audio + visual prior context to the "
                         "Gemini prompt (off by default).")
    g.add_argument("--goal-ensemble", action="store_true",
                    help="When a clip's first call returns goals>=1, "
                         "fire 2 more calls + prob-signal veto to "
                         "confirm. Defaults off.")
    g.add_argument("--flash-screen", action="store_true",
                    help="Phase-2 stub: use Gemini Flash to screen out "
                         "obviously-empty clips before Pro call. "
                         "Currently fail-safe positive (always escalates).")
    g.add_argument("--probs-dir-yolo", default=None,
                    help="Directory with per-second YOLO probs TSVs (one per vID). "
                         "Used by --prefilter-threshold, --use-context, --goal-ensemble.")
    g.add_argument("--probs-dir-audio", default=None,
                    help="Directory with per-second audio probs TSVs.")
    g.add_argument("--audio-features-dir", default=None,
                    help="Directory with per-second audio feature TSVs "
                         "(used by --use-context for audio markers).")
    g.add_argument("--calibration-dir", default=None,
                    help="Where to log calibration data "
                         "(default: data/output/calibration). Always on; "
                         "GT comparison only logged if --gt-dir given.")
    g.add_argument("--gt-dir", default=None,
                    help="Optional GT CSV dir for calibration logging.")

    # ─── alt-orchestrator hooks ─── used by tools/run_fusion_pipeline.py
    g2 = parser.add_argument_group("alt-orchestrator hooks")
    g2.add_argument("--local-seg-json", default=None,
                     help="Path to a local gt_seg_{vID}.json (overrides "
                          "GCS read). Used by tools/run_fusion_pipeline.py.")
    g2.add_argument("--no-gcs-upload", action="store_true",
                     help="Skip uploading gt_metrics_*.json to GCS. "
                          "Output still written to --output-dir if set.")
    parser.add_argument("--model", default=None,
                        help=f"Override the Gemini model used for metrics "
                             f"extraction. Default: {GEMINI_MODEL}. Useful "
                             f"for A/B testing cheaper Flash variants "
                             f"(e.g. gemini-3.5-flash, gemini-2.5-flash). "
                             f"The v14 cache (when enabled) keys on the "
                             f"model name, so switching models cleanly "
                             f"invalidates cached responses.")
    parser.add_argument("--vertex-location", default=None,
                        help=f"Override the Vertex AI location used by the "
                             f"genai client. Default: {REGION}. Some preview "
                             f"models (e.g. gemini-3.x family) are only "
                             f"served through the 'global' routing pool — "
                             f"pass --vertex-location global to reach them.")
    parser.add_argument("--progress-stage-idx", type=int, default=None,
                        choices=[1, 2, 3],
                        help="When set (1=cv_seg, 2=metrics_seg, 3=feedback_seg), "
                             "writes 'Processing (X%%)' to the vID's "
                             "analyticsStatus in the customer JSON (local "
                             "+ GCS) as each threat completes. Standalone "
                             "use (no flag) leaves the customer config "
                             "untouched. Set by run_pipeline.py when "
                             "orchestrating multi-stage runs.")
    parser.add_argument("--prompt-version", default=None,
                        help=f"Override the prompt file used. Default: "
                             f"{PROMPT_VERSION} (loaded from "
                             f"metrics_seg/prompts/metrics_<version>.txt). "
                             f"Useful for A/B testing prompt revisions "
                             f"without changing the constant. The cache "
                             f"key does NOT include the prompt version, "
                             f"so callers comparing prompt variants should "
                             f"either disable the cache (--no-cache) or "
                             f"point at separate output dirs.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _process_video_safely(vID: str, args, config) -> tuple[str, bool]:
    """Wrap process_video for batch use — never raises; always returns
    (vID, ok). Used as the unit of work when --video-workers > 1."""
    # v14 alt-orchestrator hook: --local-seg-json takes priority over
    # --segments-dir if set. Convert to segments_dir form for re-use.
    segments_dir = args.segments_dir
    if getattr(args, "local_seg_json", None):
        # Use the parent dir; process_video will read gt_seg_{vID}.json from it
        segments_dir = str(pathlib.Path(args.local_seg_json).parent)
    try:
        ok = process_video(
            vID=vID,
            customID=args.customID,
            config=config,
            workers=args.workers,
            no_gcs=args.no_gcs or getattr(args, "no_gcs_upload", False),
            segments_dir=segments_dir,
            local_video_dir=args.local_video_dir,
            output_dir=args.output_dir,
            skip_existing=args.skip_existing,
        )
    except Exception as e:
        log.error(f"[{vID}] Unhandled exception in process_video: {e}",
                  extra={"exc_type": type(e).__name__})
        ok = False

    # ─── v14 calibration logging ───
    if _V14_IMPROVEMENTS_AVAILABLE and ok:
        try:
            _v14_log_calibration(vID, args)
        except Exception as e:
            log.warning(f"[{vID}] calibration logging failed: {e}")

    return vID, ok


def _v14_log_calibration(vid: str, args) -> None:
    """Read the just-written metrics output, aggregate per-game totals,
    optionally compare against GT, log to data/output/calibration."""
    if not args.output_dir:
        return    # need a local copy to read totals from
    out_path = pathlib.Path(args.output_dir) / f"gt_metrics_{vid}.json"
    if not out_path.exists():
        return
    try:
        data = json.loads(out_path.read_text())
    except Exception:
        return
    totals = _v14_calibration.GameTotals()
    for seg in (data if isinstance(data, list) else []):
        m = seg.get("metrics") or {}
        if not isinstance(m, dict):
            continue
        totals.shots        += int(m.get("shots", 0) or 0)
        totals.shots_on_net += int(m.get("shotsOnNet", 0) or 0)
        totals.saves        += int(m.get("saves", 0) or 0)
        totals.goals        += int(m.get("goals", 0) or 0)

    gt_totals = None
    gt_dir = _V14_CONFIG.get("gt_dir") or args.gt_dir
    if gt_dir:
        gt_csv = pathlib.Path(gt_dir) / f"gt_{vid}.csv"
        if gt_csv.exists():
            try:
                import csv as _csv
                gt = _v14_calibration.GameTotals()
                with open(gt_csv, newline="") as f:
                    for row in _csv.DictReader(f):
                        action = (row.get("action") or "").strip()
                        if action == "Shots":   gt.shots_on_net += 1
                        if action == "Goals":   gt.goals += 1
                gt.shots = gt.shots_on_net   # rough: count Shots rows as SOG
                gt.saves = max(0, gt.shots_on_net - gt.goals)
                gt_totals = gt
            except Exception:
                pass

    log_dir = _V14_CONFIG.get("calibration_dir")
    log_dir = pathlib.Path(log_dir) if log_dir else None
    _v14_calibration.log_run(vid, totals, gt_totals,
                                extra={"output_dir": args.output_dir,
                                       "prefilter_threshold": _V14_CONFIG.get("prefilter_threshold")},
                                log_dir=log_dir)


def main():
    _setup_logging()  # configure handlers; no-op if already configured
    args = parse_args()

    # Honor --model + --vertex-location overrides before any Gemini
    # client constructs. The client is module-cached, so these mutations
    # need to happen before the first _get_gemini_client() call.
    if args.model:
        global GEMINI_MODEL
        log.info(f"Overriding Gemini model: {GEMINI_MODEL} → {args.model}")
        GEMINI_MODEL = args.model
    if args.vertex_location:
        global REGION
        log.info(f"Overriding Vertex location: {REGION} → {args.vertex_location}")
        REGION = args.vertex_location
    if args.progress_stage_idx is not None:
        global _PROGRESS_STAGE_IDX, _PROGRESS_CUSTOMER_ID
        _PROGRESS_STAGE_IDX = args.progress_stage_idx
        _PROGRESS_CUSTOMER_ID = args.customID
        log.info(f"Progress reporting enabled: stage_idx={_PROGRESS_STAGE_IDX} "
                  f"customer_id={_PROGRESS_CUSTOMER_ID}")
    if args.prompt_version:
        global METRICS_PROMPT, PROMPT_VERSION
        new_path = pathlib.Path(__file__).parent / "prompts" / f"metrics_{args.prompt_version}.txt"
        if not new_path.exists():
            raise SystemExit(
                f"--prompt-version {args.prompt_version}: prompt file "
                f"not found at {new_path}")
        log.info(f"Overriding prompt: {PROMPT_VERSION} → {args.prompt_version}")
        PROMPT_VERSION = args.prompt_version
        METRICS_PROMPT = new_path.read_text(encoding="utf-8").strip()

    # Populate v14 improvements config from CLI flags. Flags default such
    # that this is a no-op unless the user opts in.
    if _V14_IMPROVEMENTS_AVAILABLE:
        any_v14 = (args.prefilter_threshold > 0 or args.use_context
                    or args.goal_ensemble or args.flash_screen
                    or args.no_cache or args.cache_dir is not None)
        _V14_CONFIG.update({
            "enabled":             any_v14,
            "prefilter_threshold": args.prefilter_threshold,
            "use_context":         args.use_context,
            "goal_ensemble":       args.goal_ensemble,
            "flash_screen":        args.flash_screen,
            "no_cache":            args.no_cache,
            "cache_dir":           args.cache_dir,
            "probs_dir_yolo":      args.probs_dir_yolo,
            "probs_dir_audio":     args.probs_dir_audio,
            "audio_features_dir":  args.audio_features_dir,
            "calibration_dir":     args.calibration_dir,
            "gt_dir":              args.gt_dir,
        })
        if any_v14:
            log.info(f"v14 improvements active: prefilter≥{args.prefilter_threshold} "
                      f"context={args.use_context} ensemble={args.goal_ensemble} "
                      f"flash={args.flash_screen} cache={not args.no_cache}")
            # Initialize cache singleton with configured dir
            if not args.no_cache:
                _v14_cache.set_default_cache(
                    _v14_cache.GeminiResponseCache(cache_dir=args.cache_dir))

    # Config loading: skip GCS if --local-seg-json is set AND customID
    # config can come from existing local dirs.
    config_filename = args.customID if args.customID.endswith(".json") else f"{args.customID}.json"
    config_blob = f"customerID/{config_filename}"

    # Local-config short-circuit for alt orchestrator
    local_config_path = pathlib.Path("data/customers") / config_filename
    if args.local_seg_json and local_config_path.exists():
        log.info(f"Loading config from local: {local_config_path}")
        try:
            config = json.loads(local_config_path.read_text())
        except Exception as e:
            log.error(f"Failed to load local config: {e}")
            sys.exit(1)
    else:
        log.info(f"Loading config from gs://{GCS_BUCKET}/{config_blob}")
        try:
            config = gcs_read_json(GCS_BUCKET, config_blob)
        except Exception as e:
            log.error(f"Failed to load config from gs://{GCS_BUCKET}/{config_blob}: {e}")
            sys.exit(1)

    succeeded, failed = [], []

    if args.video_workers <= 1 or len(args.vID) == 1:
        # Sequential path — preserves original ordering and log
        # interleaving for the common single-video / small-batch case.
        for vID in args.vID:
            _, ok = _process_video_safely(vID, args, config)
            (succeeded if ok else failed).append(vID)
    else:
        # Concurrent path — process multiple videos in parallel using
        # a thread pool. Threads not processes because the bottleneck
        # is I/O (GCS, ffmpeg subprocess, Gemini network calls), and
        # we want to share the loaded config + auth state.
        log.info(
            f"Processing {len(args.vID)} videos with {args.video_workers} "
            f"video-level workers × {args.workers} per-video workers "
            f"= up to {args.video_workers * args.workers} concurrent Gemini calls"
        )
        with ThreadPoolExecutor(max_workers=args.video_workers,
                                thread_name_prefix="vid") as ex:
            futures = {ex.submit(_process_video_safely, vID, args, config): vID
                       for vID in args.vID}
            from concurrent.futures import as_completed
            for fut in as_completed(futures):
                vID, ok = fut.result()
                (succeeded if ok else failed).append(vID)

    log.info("=" * 60)
    log.info(f"Completed: {len(succeeded)}/{len(args.vID)} videos succeeded.")
    if succeeded:
        log.info(f"  Succeeded: {succeeded}")
    if failed:
        log.warning(f"  Failed:    {failed}")
    log.info("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()