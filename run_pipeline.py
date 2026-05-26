"""
run_pipeline.py — Goalie analytics end-to-end runner.

Runs stage-1 → metrics_seg → feedback_seg sequentially for one or more
video IDs. Stage 1 is **hybrid** by default: fusion-wide first (YOLO+
audio candidate peaks → ±5/15s merged windows), with cv_seg as a
safety net when fusion under-produces.

Stage-1 modes (mutually exclusive flags):
    (default)          hybrid  — fusion → cv_seg fallback below threshold
    --pure-fusion-stage1       — fusion only, no fallback
    --legacy-cv-seg            — cv_seg only (pre-fusion default)

The hybrid fallback fires per-vID when fusion produces fewer than
--hybrid-min-windows windows (default 30). Calibrated from the
14-game validation: 30 catches the pathological case
(kQVdtRa4o_A @ 24) without clobbering the surprise fusion win at
zOQrPK7IJ24 @ 34.

Validation history:
  3-game sample  (Goal F1 +0.106, Shot e2e F1 +0.059)
                 — data/output/evals/fusion_wide_validation.md
  11-game expand (Goal F1 -0.072, Shot e2e F1 +0.026)
                 — data/output/evals/fusion_wide_validation_11vid.md
                 — motivated the hybrid mode as the new default.

Fusion stage 1 REQUIRES YOLO+audio per-second probs to already exist at
runs/yolo_curve_n16/probs/{vID}.tsv and runs/audio_curve_n16/probs/{vID}.tsv.
If they don't exist for a vID, either extract them first or use
--legacy-cv-seg.

Usage:
    # Single video, hybrid stage 1 (default)
    python run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8

    # Multiple videos
    python run_pipeline.py --customer_id CUST000048 \\
        --vID mjEeE7p2Hz8 Fjc9hmK8_3U HNG0jKYY12g

    # Also keep local copies of all stage outputs
    python run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8 \\
        --local-output-dir data/output/runs

    # Tune per-stage Gemini parallelism
    python run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8 \\
        --metrics-workers 4 --feedback-workers 6

    # Pure fusion (no cv_seg fallback)
    python run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8 \\
        --pure-fusion-stage1

    # Roll back to legacy cv_seg motion windows for stage 1
    python run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8 \\
        --legacy-cv-seg

    # Custom hybrid fallback threshold
    python run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8 \\
        --hybrid-min-windows 40

    # Run only specific stages (1=stage1, 2=metrics_seg, 3=feedback_seg).
    # Default is "all". Publish to 04-final_video runs only when 3 is
    # requested AND succeeds.
    python run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8 \\
        --steps 3                              # feedback_seg + publish only
    python run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8 \\
        --steps 2 3                            # skip stage 1
    python run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8 \\
        --steps 1                              # stage 1 only, no publish

Behavior:
- Stages run per-vID: cv_seg → metrics_seg → feedback_seg, all for one
  vID, then move to the next. If any stage fails for a vID, the runner
  skips that vID's remaining stages and moves to the next vID.
- Stages already write their JSON output to the canonical GCS path:
    cv_seg       → gs://goalie_video_bucket/analyze_video/01-segment_detection/gt_seg_{vID}.json
    metrics_seg  → gs://goalie_video_bucket/analyze_video/02-segment_metrics/gt_metrics_{vID}.json
    feedback_seg → gs://goalie_video_bucket/analyze_video/03-segment_goalie_feedback/gt_feedback_{vID}.json
- Final publish step (only when ALL three stages succeeded for a vID):
  copies gt_feedback_{vID}.json to
    gs://goalie_video_bucket/analyze_video/04-final_video/{vID}.json
  Note the rename: the prefix is dropped so the published artifact is
  identified by vID alone.
- All stage stdout/stderr is captured into a single master log file at
  log/{timestamp}.log along with timing stats. Stage logs go to stdout
  too so progress is visible while the run is in flight.

Requirements:
- This file lives in the project root, with cv_seg/, metrics_seg/, and
  feedback_seg/ as sibling directories.
- Run from the project root so `python -m cv_seg` and `python -m
  feedback_seg` resolve correctly.
- Customer config and ground-truth video must already exist in GCS at
  the paths each stage expects.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── Stage configuration ──────────────────────────────────────────────
#
# Each stage is described by a function that builds the argv list given
# a customer_id, vID, and runtime options. Subprocess invocation keeps
# stages isolated — no shared global state, no import-time side effects
# leaking between cv_seg's PyTorch/OpenCV init and feedback_seg's Vertex
# client. Exit code is the source of truth for success/failure.

PROJECT_ROOT = Path(__file__).resolve().parent

# metrics_seg is a flat script, not a package, so it's invoked by path
# rather than `python -m metrics_seg`.
METRICS_SEG_SCRIPT = PROJECT_ROOT / "metrics_seg" / "01_detect_segment_metrics.py"

# ── Stage 04 publish (final_video) ────────────────────────────────────
#
# After all three pipeline stages succeed, copy the feedback_seg output
# to a "final" location with a stripped filename. The publish destination
# is what downstream consumers (UI, API, dashboards) read from.
PUBLISH_BUCKET     = "goalie_video_bucket"
PUBLISH_SRC_PREFIX = "analyze_video/03-segment_goalie_feedback"
PUBLISH_DST_PREFIX = "analyze_video/04-final_video"


def cv_seg_argv(customer_id: str,
                vID: str,
                local_output_dir: str | None,
                workers: int | None,
                target_filter: bool = True) -> list[str]:
    """Build argv for cv_seg (LEGACY stage 1 — kept for rollback via
    --legacy-cv-seg flag).

    cv_seg writes to GCS by default. --output-dir is its local mirror
    path; --no-local suppresses local writes when we don't want them.

    `workers` is accepted for signature parity with the other stage
    builders but is unused — cv_seg is single-threaded by design.

    `target_filter` defaults to True (match cv_seg's own default). When
    False, --no-target-filter is appended so cv_seg writes ALL segments
    regardless of which goalie they target. Useful for debugging
    attribution; the default is what production runs want.
    """
    customID = customer_id if customer_id.endswith(".json") else f"{customer_id}.json"
    argv = [
        sys.executable, "-m", "cv_seg",
        "--vID", vID,
        "--customID", customID,
    ]
    if not target_filter:
        argv += ["--no-target-filter"]
    if local_output_dir:
        argv += ["--output-dir", str(Path(local_output_dir) / "cv_seg")]
    else:
        argv += ["--no-local"]
    return argv


# Default stage-1 builder: fusion-wide pipeline (YOLO+audio candidate
# peaks → seg JSON with ±5/15s windows). Replaces cv_seg as the
# production default per the 3-game validation that lifted:
#   Goal F1       0.645 → 0.750 (+0.106)
#   Shot e2e F1   0.371 → 0.430 (+0.059)
#   Goal precision 0.867 → 1.000 (+0.133, perfect)
# See data/output/evals/fusion_wide_validation.md for the validation.
FUSION_PIPELINE_SCRIPT = PROJECT_ROOT / "tools" / "run_fusion_pipeline.py"

# Pad seconds for fusion windows. Wider --post gives Gemini enough
# aftermath context for goal-confirmation paths (Path B / Path C in
# metrics_v13.txt). 5/15 was the validated sweet spot.
FUSION_PRE_SEC  = 5
FUSION_POST_SEC = 15

# Hybrid stage-1 fallback threshold: if fusion produces fewer than this
# many windows, run cv_seg as a safety-net fallback for that vID.
# Calibration from the 14-game (3 + 11) validation:
#   - kQVdtRa4o_A had 24 fusion windows and lost -0.160 Shot F1 vs cv_seg
#   - zOQrPK7IJ24 had 34 fusion windows and WON +0.300 Goal F1 vs cv_seg
#   - every other vID had >=53 fusion windows
# Threshold of 30 catches the pathological case (kQVdt) without
# clobbering the win on zOQrP. See data/output/evals/fusion_wide_validation_11vid.md.
HYBRID_MIN_WINDOWS_DEFAULT = 30


def fusion_stage1_argv(customer_id: str,
                       vID: str,
                       local_output_dir: str | None,
                       workers: int | None,
                       target_filter: bool = True) -> list[str]:
    """Build argv for the fusion stage-1 (the new production default).

    Calls tools/run_fusion_pipeline.py to:
      1. Generate candidate shot peaks from YOLO+audio fused probs
      2. Expand peaks by ±FUSION_PRE_SEC/FUSION_POST_SEC → merged windows
      3. Write cv_seg-format gt_seg_{vID}.json that metrics_seg consumes
    The --skip-metrics flag stops the fusion orchestrator from also
    invoking metrics_seg — run_pipeline.py drives that separately
    via the metrics_seg stage in STAGES.

    `target_filter` is accepted for signature parity but unused (fusion
    candidate peaks are pre-filtered by the upstream model anyway).
    `workers` is also unused — candidate generation is single-threaded.
    """
    # The fusion orchestrator writes to <out-dir>/gt_seg_{vID}.json.
    # When --local-output-dir is set, use a subdir for parity with the
    # cv_seg path; otherwise default to data/output/runs/cv_seg_fusion_wide
    # so downstream stages find a consistent location.
    if local_output_dir:
        out_dir = Path(local_output_dir) / "cv_seg_fusion_wide"
    else:
        out_dir = PROJECT_ROOT / "data" / "output" / "runs" / "cv_seg_fusion_wide"
    return [
        sys.executable, str(FUSION_PIPELINE_SCRIPT),
        "--customer_id", customer_id,
        "--vID", vID,
        "--skip-metrics",                             # this stage only generates the seg JSON
        "--pre", str(FUSION_PRE_SEC),
        "--post", str(FUSION_POST_SEC),
        "--out-dir", str(out_dir),
    ]


def metrics_seg_argv(customer_id: str,
                     vID: str,
                     local_output_dir: str | None,
                     workers: int | None,
                     stage1_seg_dir: Path | None = None,
                     local_video_dir: Path | None = None) -> list[str]:
    """Build argv for metrics_seg.

    metrics_seg writes to GCS by default. --output-dir adds a local
    mirror; absence of --no-gcs keeps GCS upload on.

    `workers` controls --workers (parallel Gemini requests within one
    video). When None, metrics_seg's built-in default is used (2).

    `stage1_seg_dir` points metrics_seg at the local seg JSON dir
    produced by stage 1 (fusion or cv_seg). When set, metrics_seg
    reads gt_seg_{vID}.json from there instead of GCS. Required for
    the fusion path (where seg JSON is local-only by default).

    `local_video_dir` short-circuits the GCS video download.
    """
    # metrics_seg's --customID accepts the bare ID without .json
    customID = customer_id.removesuffix(".json")
    argv = [
        sys.executable, str(METRICS_SEG_SCRIPT),
        "--vID", vID,
        "--customID", customID,
    ]
    if workers is not None:
        argv += ["--workers", str(workers)]
    if local_output_dir:
        argv += ["--output-dir", str(Path(local_output_dir) / "metrics_seg")]
    if stage1_seg_dir is not None:
        argv += ["--segments-dir", str(stage1_seg_dir)]
    if local_video_dir is not None:
        argv += ["--local-video-dir", str(local_video_dir)]
    return argv


def feedback_seg_argv(customer_id: str,
                      vID: str,
                      local_output_dir: str | None,
                      workers: int | None) -> list[str]:
    """Build argv for feedback_seg.

    feedback_seg writes to GCS by default. --output-dir adds a local
    mirror; absence of --no-gcs keeps GCS upload on.

    `workers` controls --workers (parallel Gemini requests within one
    video). When None, feedback_seg's built-in default is used (3,
    from COACH_PARALLEL_WORKERS).
    """
    argv = [
        sys.executable, "-m", "feedback_seg",
        "--customer_id", customer_id,
        "--vID", vID,
    ]
    if workers is not None:
        argv += ["--workers", str(workers)]
    if local_output_dir:
        argv += ["--output-dir", str(Path(local_output_dir) / "feedback_seg")]
    return argv


# Stage 1 builder is selected at runtime based on --legacy-cv-seg flag.
# Default is fusion (validated +0.106 goal F1 lift across 3 games).
# build_stages() returns the right STAGES tuple list to use.
def build_stages(use_legacy_cv_seg: bool = False) -> list[tuple[str, callable]]:
    stage1_name    = "cv_seg"
    stage1_builder = fusion_stage1_argv if not use_legacy_cv_seg else cv_seg_argv
    return [
        (stage1_name,    stage1_builder),
        ("metrics_seg",  metrics_seg_argv),
        ("feedback_seg", feedback_seg_argv),
    ]

# Default STAGES (fusion). Overridden in main() if --legacy-cv-seg is set.
STAGES = build_stages(use_legacy_cv_seg=False)

# Map step number (1-based, matching the new GCS path numbering) to
# stage tuple. Used to filter STAGES by --steps.
#   1 → stage1 (fusion or cv_seg) → seg JSON for metrics_seg
#   2 → metrics_seg               → analyze_video/02-segment_metrics
#   3 → feedback_seg              → analyze_video/03-segment_goalie_feedback
#                                   (publish to 04-final_video runs after 3 succeeds)
STEP_TO_STAGE_NAME = {1: "cv_seg", 2: "metrics_seg", 3: "feedback_seg"}
STAGE_NAME_TO_STEP = {v: k for k, v in STEP_TO_STAGE_NAME.items()}
ALL_STEPS = [1, 2, 3]


# ── Logging ──────────────────────────────────────────────────────────

class TeeLogger:
    """Writes to a file and stdout simultaneously, line-buffered."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(log_path, "w", buffering=1, encoding="utf-8")

    def write(self, line: str) -> None:
        if not line.endswith("\n"):
            line = line + "\n"
        self.fh.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    def header(self, title: str) -> None:
        bar = "=" * 70
        self.write(bar)
        self.write(title)
        self.write(bar)

    def close(self) -> None:
        self.fh.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {sec:.0f}s"


# ── Stage execution ──────────────────────────────────────────────────

def run_stage(stage_name: str,
              argv: list[str],
              tee: TeeLogger) -> tuple[bool, float]:
    """Run one stage as a subprocess. Capture combined stdout+stderr
    to the master log line-by-line as it streams.

    Returns (success, elapsed_seconds).
    """
    tee.write(f"[{now_iso()}] >>> START stage={stage_name}")
    tee.write(f"[{now_iso()}]     command: {' '.join(argv)}")
    start = time.monotonic()

    # Merge stderr into stdout so the log preserves causal ordering of
    # progress messages (which go to stderr via logging) and final-line
    # summaries (which sometimes go to stdout). bufsize=1 + text=True
    # gives line-buffered streaming.
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
    except FileNotFoundError as e:
        elapsed = time.monotonic() - start
        tee.write(f"[{now_iso()}] !!! FAILED to launch stage={stage_name}: {e}")
        return False, elapsed

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            # Strip trailing newline so TeeLogger.write doesn't double it
            tee.write(f"  [{stage_name}] {line.rstrip()}")
    finally:
        proc.wait()

    elapsed = time.monotonic() - start
    ok = (proc.returncode == 0)
    status = "OK" if ok else f"FAIL (exit={proc.returncode})"
    tee.write(f"[{now_iso()}] <<< END   stage={stage_name} status={status} "
              f"elapsed={fmt_duration(elapsed)}")
    return ok, elapsed


def _count_seg_json_windows(seg_path: Path) -> int:
    """Count segments in a stage-1 seg JSON. Returns 0 on any failure
    (missing file, malformed JSON, unexpected schema)."""
    try:
        data = json.loads(seg_path.read_text())
        if isinstance(data, list):
            return len(data)
        # cv_seg historically wrote a list; defensively support a wrapped
        # object in case the schema ever evolves.
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            return len(data["segments"])
        return 0
    except Exception:
        return 0


def run_video(customer_id: str,
              vID: str,
              local_output_dir: str | None,
              workers_by_stage: dict[str, int | None],
              steps_to_run: set[int],
              target_filter: bool,
              tee: TeeLogger,
              stage1_mode: str = "hybrid",
              hybrid_min_windows: int = HYBRID_MIN_WINDOWS_DEFAULT,
              local_video_dir: Path | None = None) -> tuple[bool, dict[str, float], str | None]:
    """Run the requested stages for a single vID, stopping at first failure.

    Stages in the chain that aren't in `steps_to_run` are silently
    skipped (they won't appear in per_stage). Stages that are requested
    but skipped due to an upstream requested-stage failure also won't
    appear in per_stage — the failure_stage tells the caller why.

    `target_filter` is forwarded only to cv_seg (the only stage that
    knows about it). The other stage builders ignore it.

    stage1_mode controls which backend drives stage 1:
      "hybrid"         — fusion first; if it produces fewer than
                          `hybrid_min_windows`, re-run cv_seg and use
                          its output for metrics_seg. Default (the
                          safety net for low-fusion-density games).
      "pure_fusion"    — fusion only, no fallback.
      "legacy_cv_seg"  — cv_seg motion-window detector (pre-fusion
                          default).

    Returns (overall_success, per_stage_elapsed_seconds, failure_stage):
      overall_success: True iff every requested stage that ran returned
        OK. A vID where no requested stage failed is "success".
      per_stage: {stage_name: elapsed_seconds} for stages that actually
        ran. Stages skipped for any reason are absent. The hybrid
        fallback (when triggered) appears as "cv_seg_fallback".
      failure_stage: name of the failing stage, or None on full success.
    """
    tee.header(f"VIDEO {vID}  customer={customer_id}  stage1_mode={stage1_mode}")
    per_stage: dict[str, float] = {}

    # Stage list is picked at runtime. The "hybrid" mode runs fusion
    # first and may follow up with cv_seg, so it shares its builder
    # setup with "pure_fusion".
    use_legacy_cv_seg = (stage1_mode == "legacy_cv_seg")
    stages = build_stages(use_legacy_cv_seg=use_legacy_cv_seg)

    # Where stage-1 writes its seg JSON. metrics_seg needs to read from
    # here when stage-1 is fusion (which doesn't upload to GCS).
    if use_legacy_cv_seg:
        stage1_seg_dir = (Path(local_output_dir) / "cv_seg"
                           if local_output_dir else None)
    else:
        # fusion default — see fusion_stage1_argv()
        stage1_seg_dir = (Path(local_output_dir) / "cv_seg_fusion_wide"
                           if local_output_dir
                           else PROJECT_ROOT / "data" / "output" / "runs" / "cv_seg_fusion_wide")

    # Active stage-1 backend. Flips to "cv_seg" mid-flight if the hybrid
    # fallback fires; metrics_seg wiring keys off this.
    active_backend = "cv_seg" if use_legacy_cv_seg else "fusion"

    for stage_name, argv_builder in stages:
        step_num = STAGE_NAME_TO_STEP[stage_name]
        if step_num not in steps_to_run:
            tee.write(f"[{now_iso()}] --- Skipping stage={stage_name} "
                      f"(step {step_num} not requested)")
            continue

        builder_kwargs = {}
        # cv_seg legacy takes target_filter; fusion ignores it but
        # accepts the kwarg for signature parity.
        if stage_name == "cv_seg":
            builder_kwargs["target_filter"] = target_filter
        # metrics_seg in fusion-backend mode reads seg JSON from local
        # dir + local video dir. cv_seg-backend mode (legacy or hybrid
        # fallback) lets metrics_seg fall back to its default GCS read.
        if stage_name == "metrics_seg" and active_backend == "fusion":
            builder_kwargs["stage1_seg_dir"] = stage1_seg_dir
            if local_video_dir is not None:
                builder_kwargs["local_video_dir"] = local_video_dir

        argv = argv_builder(
            customer_id, vID, local_output_dir,
            workers_by_stage.get(stage_name),
            **builder_kwargs,
        )
        ok, elapsed = run_stage(stage_name, argv, tee)
        per_stage[stage_name] = elapsed
        if not ok:
            tee.write(f"[{now_iso()}] !!! Skipping remaining stages for vID={vID}")
            return False, per_stage, stage_name

        # ── Hybrid fallback ───────────────────────────────────────────
        # After the stage-1 step (its name is "cv_seg" regardless of
        # which backend ran), inspect the produced seg JSON. If fusion
        # under-produced, re-run cv_seg as a safety net.
        if (stage_name == "cv_seg"
                and stage1_mode == "hybrid"
                and active_backend == "fusion"
                and stage1_seg_dir is not None):
            seg_path = stage1_seg_dir / f"gt_seg_{vID}.json"
            n_windows = _count_seg_json_windows(seg_path)
            tee.write(f"[{now_iso()}] --- hybrid: fusion produced "
                      f"{n_windows} windows (threshold={hybrid_min_windows})")
            if n_windows < hybrid_min_windows:
                tee.write(f"[{now_iso()}] --- hybrid: below threshold "
                          f"→ falling back to cv_seg")
                cv_argv = cv_seg_argv(
                    customer_id, vID, local_output_dir, None,
                    target_filter=target_filter,
                )
                ok2, elapsed2 = run_stage("cv_seg_fallback", cv_argv, tee)
                per_stage["cv_seg_fallback"] = elapsed2
                if not ok2:
                    tee.write(f"[{now_iso()}] !!! cv_seg fallback failed; "
                              f"skipping remaining stages for vID={vID}")
                    return False, per_stage, "cv_seg_fallback"
                # Switch metrics_seg to read from cv_seg (GCS or local).
                active_backend = "cv_seg"
                stage1_seg_dir = (Path(local_output_dir) / "cv_seg"
                                   if local_output_dir else None)
            else:
                tee.write(f"[{now_iso()}] --- hybrid: above threshold "
                          f"→ keeping fusion output")

    return True, per_stage, None


# ── Stage 04 publish ─────────────────────────────────────────────────

def publish_final(vID: str, tee: TeeLogger) -> tuple[bool, float]:
    """Copy gt_feedback_{vID}.json from 03-segment_goalie_feedback to
    04-final_video/{vID}.json (note the rename).

    Uses google.cloud.storage's server-side copy via Bucket.copy_blob —
    no download/upload cycle, no local temp file.

    Returns (success, elapsed_seconds). On failure, logs the reason and
    continues; the caller decides how to surface this in the summary.

    Failure modes handled:
      - source blob does not exist (shouldn't happen given the gating
        in main(), but defended against)
      - GCS auth/permission errors
      - any other transient GCS error (single attempt; if you need
        retry here, lift the call_with_retry helper from feedback_seg)
    """
    src_blob_name = f"{PUBLISH_SRC_PREFIX}/gt_feedback_{vID}.json"
    dst_blob_name = f"{PUBLISH_DST_PREFIX}/{vID}.json"

    tee.write(f"[{now_iso()}] >>> START stage=publish_final")
    tee.write(f"[{now_iso()}]     src: gs://{PUBLISH_BUCKET}/{src_blob_name}")
    tee.write(f"[{now_iso()}]     dst: gs://{PUBLISH_BUCKET}/{dst_blob_name}")
    start = time.monotonic()

    try:
        # Import locally so the runner can still parse --help on a
        # machine that doesn't have google-cloud-storage installed.
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(PUBLISH_BUCKET)
        src_blob = bucket.blob(src_blob_name)

        if not src_blob.exists():
            elapsed = time.monotonic() - start
            tee.write(f"[{now_iso()}] !!! Source blob does not exist; "
                      f"feedback_seg may have failed silently")
            tee.write(f"[{now_iso()}] <<< END   stage=publish_final "
                      f"status=FAIL (no source) elapsed={fmt_duration(elapsed)}")
            return False, elapsed

        # Server-side copy. This is atomic from the consumer's
        # perspective — readers either see the old content (if any) or
        # the new content, never a partial write.
        bucket.copy_blob(
            blob=src_blob,
            destination_bucket=bucket,
            new_name=dst_blob_name,
        )
        elapsed = time.monotonic() - start
        tee.write(f"[{now_iso()}] <<< END   stage=publish_final status=OK "
                  f"elapsed={fmt_duration(elapsed)}")
        return True, elapsed

    except Exception as e:
        elapsed = time.monotonic() - start
        tee.write(f"[{now_iso()}] !!! publish_final raised: "
                  f"{type(e).__name__}: {e}")
        tee.write(f"[{now_iso()}] <<< END   stage=publish_final "
                  f"status=FAIL (exception) elapsed={fmt_duration(elapsed)}")
        return False, elapsed


# ── Main ─────────────────────────────────────────────────────────────

def _positive_int(value: str) -> int:
    """argparse type for a strictly positive integer."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer")
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"must be >= 1, got {n}"
        )
    return n


def _step_value(value: str) -> str:
    """argparse type for --steps. Accepts 'all' or '1'/'2'/'3'.

    Returns the value as a normalized lowercase string. Resolution to
    the actual step set happens in main() so we have one place that
    knows about ALL_STEPS.
    """
    v = value.strip().lower()
    if v == "all":
        return "all"
    if v in {"1", "2", "3"}:
        return v
    raise argparse.ArgumentTypeError(
        f"{value!r} is not a valid step. Use 'all' or one of: 1, 2, 3"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end goalie analytics pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--customer_id", "--customID", "--customer-id",
        dest="customer_id", required=True,
        help="Customer config key, e.g. CUST000048",
    )
    p.add_argument(
        "--vID", required=True, nargs="+", metavar="vID",
        help="One or more video IDs to process",
    )
    p.add_argument(
        "--local-output-dir", dest="local_output_dir", default=None,
        help="If set, each stage also writes its output JSON to this "
             "directory (in addition to GCS). Subdirectories cv_seg/, "
             "metrics_seg/, feedback_seg/ are created under it. "
             "Default: GCS-only.",
    )
    p.add_argument(
        "--log-dir", dest="log_dir", default="log",
        help="Directory for the master log file (default: log/).",
    )
    p.add_argument(
        "--metrics-workers", dest="metrics_workers",
        type=_positive_int, default=None,
        help="Parallel Gemini requests within metrics_seg per video. "
             "When omitted, metrics_seg's built-in default (2) is used. "
             "Higher = faster per video, more Vertex quota consumed. "
             "Watch for 429 ResourceExhausted in the log if you push it.",
    )
    p.add_argument(
        "--feedback-workers", dest="feedback_workers",
        type=_positive_int, default=None,
        help="Parallel Gemini requests within feedback_seg per video. "
             "When omitted, feedback_seg's built-in default (3) is used. "
             "Tune independently from --metrics-workers — the two stages "
             "have different per-call costs (feedback_seg sends longer "
             "video parts and longer prompts).",
    )
    p.add_argument(
        "--steps", dest="steps",
        nargs="+", type=_step_value, default=["all"],
        metavar="STEP",
        help="Which pipeline stages to run, by number. "
             "1 = cv_seg, 2 = metrics_seg, 3 = feedback_seg. "
             "Default 'all' is equivalent to '1 2 3'. "
             "Examples: --steps 3 (feedback_seg only), "
             "--steps 2 3 (skip cv_seg). "
             "When 3 is requested AND succeeds, the 04 publish step "
             "(copy gt_feedback_{vID}.json → 04-final_video/{vID}.json) "
             "runs automatically. "
             "When 3 is not requested, no publish runs. "
             "Skipping a middle stage (e.g. --steps 1 3) is allowed but "
             "the downstream stage will fail unless its input already "
             "exists in GCS from a prior run.",
    )
    p.add_argument(
        "--no-target-filter", dest="target_filter",
        action="store_false", default=True,
        help="Disable cv_seg's target-color segment filter. By default, "
             "cv_seg drops every segment that isn't a threat against the "
             "customer's targetGoalieColor — this is the production "
             "behaviour and roughly halves downstream Gemini cost. Pass "
             "this flag when debugging attribution issues to keep all "
             "segments (target-threat, opponent-threat, no-threat) in "
             "cv_seg's output. Has no effect when --steps doesn't "
             "include 1. Has no effect under fusion stage 1 (the default).",
    )
    stage1_grp = p.add_mutually_exclusive_group()
    stage1_grp.add_argument(
        "--legacy-cv-seg", dest="stage1_mode_legacy",
        action="store_true", default=False,
        help="Stage 1 backend = cv_seg only (the pre-fusion default). "
             "Useful for rollback and for vIDs that don't yet have "
             "YOLO+audio probs cached. Mutually exclusive with "
             "--pure-fusion-stage1.",
    )
    stage1_grp.add_argument(
        "--pure-fusion-stage1", dest="stage1_mode_pure_fusion",
        action="store_true", default=False,
        help="Stage 1 backend = fusion only, with no cv_seg safety net. "
             "Use when you want to study fusion's behavior in isolation. "
             "Mutually exclusive with --legacy-cv-seg.",
    )
    p.add_argument(
        "--hybrid-min-windows", dest="hybrid_min_windows",
        type=_positive_int, default=HYBRID_MIN_WINDOWS_DEFAULT,
        help="Hybrid-mode threshold: if fusion produces fewer than this "
             f"many windows for a vID, re-run cv_seg as a fallback and "
             f"use its output for metrics_seg. Default {HYBRID_MIN_WINDOWS_DEFAULT}. "
             "Calibrated from the 14-game validation: 30 catches the "
             "fusion-under-produced case (kQVdtRa4o_A @ 24 windows) "
             "without clobbering low-window-count fusion wins "
             "(zOQrPK7IJ24 @ 34 windows won +0.300 Goal F1). Ignored "
             "under --legacy-cv-seg or --pure-fusion-stage1.",
    )
    p.add_argument(
        "--local-video-dir", dest="local_video_dir", default=None, type=Path,
        help="Local directory containing full_{vID}.mp4. Skips the GCS "
             "video download in metrics_seg. Only used when the active "
             "stage-1 backend is fusion (default and hybrid-without-fallback).",
    )
    args = p.parse_args()
    if args.stage1_mode_legacy:
        args.stage1_mode = "legacy_cv_seg"
    elif args.stage1_mode_pure_fusion:
        args.stage1_mode = "pure_fusion"
    else:
        args.stage1_mode = "hybrid"
    return args


def main() -> int:
    args = parse_args()

    # Resolve --steps. "all" is equivalent to all step numbers; mixing
    # "all" with explicit numbers is treated the same as "all".
    if "all" in args.steps:
        steps_to_run: set[int] = set(ALL_STEPS)
    else:
        steps_to_run = {int(s) for s in args.steps}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"{timestamp}.log"
    tee = TeeLogger(log_path)

    # Per-stage worker counts. None means "use the stage's own default".
    workers_by_stage: dict[str, int | None] = {
        "cv_seg":       None,  # cv_seg is single-threaded; flag is unused
        "metrics_seg":  args.metrics_workers,
        "feedback_seg": args.feedback_workers,
    }

    def _w(stage: str) -> str:
        v = workers_by_stage.get(stage)
        return f"{v}" if v is not None else "(stage default)"

    overall_start = time.monotonic()
    tee.header(f"PIPELINE RUN  started={now_iso()}")
    tee.write(f"customer_id:      {args.customer_id}")
    tee.write(f"vIDs:             {args.vID}")
    tee.write(f"steps:            "
              f"{sorted(steps_to_run)}  "
              f"({', '.join(STEP_TO_STAGE_NAME[s] for s in sorted(steps_to_run))})")
    tee.write(f"local-output-dir: {args.local_output_dir or '(none — GCS-only)'}")
    mode_desc = {
        "hybrid":        f"hybrid (fusion → cv_seg fallback if <{args.hybrid_min_windows} windows; default)",
        "pure_fusion":   "pure_fusion (no cv_seg fallback)",
        "legacy_cv_seg": "cv_seg (LEGACY — motion windows)",
    }[args.stage1_mode]
    tee.write(f"stage-1 backend:  {mode_desc}")
    if args.stage1_mode in ("legacy_cv_seg", "hybrid"):
        tee.write(f"target-filter:    "
                  f"{'enabled (cv_seg drops non-target segments)' if args.target_filter else 'DISABLED (cv_seg keeps all segments)'}")
    tee.write(f"metrics workers:  {_w('metrics_seg')}")
    tee.write(f"feedback workers: {_w('feedback_seg')}")
    tee.write(f"log file:         {log_path}")
    tee.write(f"project root:     {PROJECT_ROOT}")
    tee.write("")

    # Per-vID results:
    #   vID → {
    #       stages: {name: elapsed_seconds},
    #       success: bool,                 # all three pipeline stages succeeded
    #       failure_stage: str | None,     # which pipeline stage failed (if any)
    #       publish_success: bool | None,  # 04 publish result; None if not attempted
    #       publish_elapsed: float | None,
    #   }
    results: dict[str, dict] = {}

    for vID in args.vID:
        ok, per_stage, failure_stage = run_video(
            customer_id=args.customer_id,
            vID=vID,
            local_output_dir=args.local_output_dir,
            workers_by_stage=workers_by_stage,
            steps_to_run=steps_to_run,
            target_filter=args.target_filter,
            tee=tee,
            stage1_mode=args.stage1_mode,
            hybrid_min_windows=args.hybrid_min_windows,
            local_video_dir=args.local_video_dir,
        )

        # Stage 04 publish — runs only when:
        #   1. step 3 (feedback_seg) was in the requested step set, AND
        #   2. all stages that did run succeeded (step 3 included).
        # If step 3 wasn't requested, no publish; per_stage won't contain
        # feedback_seg either, so the publish slot stays None.
        publish_success: bool | None = None
        publish_elapsed: float | None = None
        if 3 not in steps_to_run:
            tee.write(f"[{now_iso()}] --- Skipping publish_final for vID={vID} "
                      f"(step 3 not requested)")
        elif not ok:
            tee.write(f"[{now_iso()}] !!! Skipping publish_final for vID={vID} "
                      f"(upstream failure at {failure_stage})")
        else:
            publish_success, publish_elapsed = publish_final(vID, tee)

        results[vID] = {
            "success": ok,
            "stages": per_stage,
            "failure_stage": failure_stage,
            "publish_success": publish_success,
            "publish_elapsed": publish_elapsed,
        }

    overall_elapsed = time.monotonic() - overall_start

    # ── Summary ─────────────────────────────────────────────────────
    tee.header("RUN SUMMARY")
    succeeded = [v for v, r in results.items() if r["success"]]
    failed    = [v for v, r in results.items() if not r["success"]]
    # Publish failures are tracked separately — a vID where all three
    # stages succeeded but the 04 copy failed counts as "succeeded
    # pipeline / failed publish".
    publish_failed = [v for v, r in results.items()
                      if r["success"] and r["publish_success"] is False]
    tee.write(f"Total elapsed:    {fmt_duration(overall_elapsed)}")
    tee.write(f"Videos processed: {len(args.vID)}")
    tee.write(f"  Succeeded:      {len(succeeded)}  {succeeded if succeeded else ''}")
    tee.write(f"  Failed:         {len(failed)}  {failed if failed else ''}")
    if publish_failed:
        tee.write(f"  Publish failed: {len(publish_failed)}  {publish_failed}")
    tee.write("")

    # Per-vID timing table
    tee.write("Per-video timing:")
    header = (f"  {'vID':<16}  {'stage1':>10}  {'fallback':>10}  "
              f"{'metrics_seg':>12}  {'feedback_seg':>13}  "
              f"{'publish':>9}  {'total':>10}  status")
    tee.write(header)
    tee.write("  " + "-" * (len(header) - 2))
    for vID in args.vID:
        r = results[vID]
        s = r["stages"]
        cv   = fmt_duration(s.get("cv_seg",          0.0)) if "cv_seg"          in s else "—"
        fbk  = fmt_duration(s.get("cv_seg_fallback", 0.0)) if "cv_seg_fallback" in s else "—"
        me   = fmt_duration(s.get("metrics_seg",     0.0)) if "metrics_seg"     in s else "—"
        fb   = fmt_duration(s.get("feedback_seg",    0.0)) if "feedback_seg"    in s else "—"
        if r["publish_elapsed"] is not None:
            pub = fmt_duration(r["publish_elapsed"])
        else:
            pub = "—"
        total_seconds = sum(s.values()) + (r["publish_elapsed"] or 0.0)
        total = fmt_duration(total_seconds)
        if r["success"] and r["publish_success"]:
            status = "OK"
        elif r["success"] and r["publish_success"] is False:
            status = "FAIL @ publish_final"
        else:
            status = f"FAIL @ {r['failure_stage']}"
        tee.write(f"  {vID:<16}  {cv:>10}  {fbk:>10}  {me:>12}  {fb:>13}  "
                  f"{pub:>9}  {total:>10}  {status}")

    tee.write("")
    tee.write(f"[{now_iso()}] DONE")
    tee.close()

    # Exit non-zero on any failure: pipeline stage failure OR publish failure
    return 0 if (not failed and not publish_failed) else 1


if __name__ == "__main__":
    sys.exit(main())
