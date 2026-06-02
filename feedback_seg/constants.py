"""Configuration constants for feedback_seg.

These are intentionally module-level (not CLI args) because changing
them requires deliberate review and re-testing.
"""

# ── GCP / GCS configuration ────────────────────────────────────────────
PROJECT_ID  = "goalie-analytics-pro-dev"
REGION      = "us-central1"
BUCKET_NAME = "goalie_video_bucket"

# Bucket prefixes (relative paths inside BUCKET_NAME)
VIDEO_PREFIX  = "ground_truth_video/full_video"
INPUT_PREFIX  = "analyze_video/02-segment_metrics"
OUTPUT_PREFIX = "analyze_video/03-segment_goalie_feedback"
TEMP_PREFIX   = "analyze_video/00-temp_parts"

# ── Gemini configuration ───────────────────────────────────────────────
# gemini-2.5-flash: the 2.5→3.x bump (commit 948c438) was rejected, and
# gemini-3.5-flash 404s on Vertex for this project. Stay on 2.5-flash.
GEMINI_MODEL = "gemini-2.5-flash"

# Default parallelism. The CLI exposes --workers to override.
COACH_PARALLEL_WORKERS = 3

# Max output tokens for both per-window and summary calls. The 2048
# default was producing truncated JSON when Gemini wrote a verbose
# four-pillar summary.
MAX_OUTPUT_TOKENS = 8192

# ── Inline bytes vs GCS-URI threshold ──────────────────────────────────
# When a clip file is below this size, we send it inline as bytes (no
# GCS round-trip). When it exceeds this size we fall back to the
# upload-URI-delete pattern. Vertex AI's inline part size limit is
# documented as ~20MB; we keep a 2MB margin.
INLINE_BYTES_MAX_SIZE = 18 * 1024 * 1024  # 18 MiB

# ── Retry configuration ────────────────────────────────────────────────
MAX_RETRIES        = 4         # was 6 — bounded total wait
RETRY_BACKOFF_BASE = 30.0      # initial wait, seconds
RETRY_BACKOFF_CAP  = 240.0     # was unbounded — cap upper jitter range

# ── ffmpeg clip-extraction tolerance ───────────────────────────────────
# After stream-copy extraction, if the resulting clip's duration
# differs from the requested duration by more than this many seconds,
# fall back to a precise re-encode. Source video keyframe intervals
# are typically 1-10s, so 2.0 is a reasonable threshold.
CLIP_DURATION_TOLERANCE_SEC = 2.0
