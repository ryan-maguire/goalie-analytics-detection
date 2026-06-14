"""Cloud Run Job entrypoint — runs the pipeline for one vID.

The Service (deploy/api/main.py) dispatches Job executions with env
overrides. This script translates those env vars into the run_pipeline.py
CLI invocation and delegates. Keeping run_pipeline.py as the single
source of truth for stage orchestration means local + production use
the exact same code path.

Env vars (all optional except CUSTOMER_ID + VID):
  CUSTOMER_ID         required
  VID                 required
  STAGE1_MODE         "hybrid" | "pure_fusion" | "legacy_cv_seg"
                      (default: "hybrid")
  STEPS               comma-separated subset of "1,2,3" (default: "1,2,3")
  HYBRID_MIN_WINDOWS  int (default: 30)
  METRICS_WORKERS     int (optional — stage default if unset)
  FEEDBACK_WORKERS    int (optional — stage default if unset)
  LOCAL_VIDEO_DIR     path (optional — useful for warm-cache testing
                      but production deployments leave this unset so
                      run_pipeline.py downloads from GCS)

On unrecoverable validation failure, writes 'Failed: <reason>' to the
vID's analyticsStatus and exits non-zero so Cloud Run reports the Job
execution as failed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from util import progress as _pp


def _require(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        print(f"FATAL: env var {var} is required", file=sys.stderr)
        sys.exit(2)
    return val


def main() -> int:
    customer_id = _require("CUSTOMER_ID")
    vID         = _require("VID")
    stage1_mode = os.environ.get("STAGE1_MODE", "hybrid")
    steps       = os.environ.get("STEPS", "1,2,3").split(",")
    hybrid_min  = os.environ.get("HYBRID_MIN_WINDOWS", "30")
    mw          = os.environ.get("METRICS_WORKERS")
    fw          = os.environ.get("FEEDBACK_WORKERS")
    local_video_dir = os.environ.get("LOCAL_VIDEO_DIR")

    # Production analysis reads the app-uploaded video from the upload bucket.
    # Point every pipeline stage at that prefix (the constants modules read
    # GCS_VIDEO_PREFIX; env propagates through the subprocess below). The
    # default (ground_truth_video/full_video) is reserved for eval runs.
    os.environ.setdefault(
        "GCS_VIDEO_PREFIX", "analyze_video/00-segement-video-upload")

    # Verify the source video the app uploaded is actually present in GCS
    # before kicking off the pipeline, so a missing upload fails fast with a
    # clear reason instead of crashing a stage mid-run. Skipped when a local
    # video dir is provided (the pipeline uses the local file).
    if not local_video_dir:
        try:
            # Heartbeat so the status moves off the dispatch's "Processing (0%)"
            # while the (potentially multi-minute) source-video fetch runs —
            # otherwise it looks frozen at 0% until stage 1 starts reporting.
            try:
                _pp._update_status(customer_id, vID, "Processing (1%) — fetching video")
            except Exception:
                pass
            from util.ensure_video import ensure_video
            result = ensure_video(vID, customer_id)
            print(f"[worker] ensure_video: {result}", flush=True)
        except Exception as e:
            print(f"[worker] ensure_video failed: {e}", file=sys.stderr, flush=True)
            try:
                _pp.mark_failed(customer_id, vID, reason=f"video fetch: {e}")
            except Exception as e2:
                print(f"[worker] mark_failed swallowed error: {e2}", file=sys.stderr)
            return 1

    argv = [
        sys.executable, str(REPO / "run_pipeline.py"),
        "--customer_id", customer_id,
        "--vID", vID,
        "--hybrid-min-windows", hybrid_min,
        "--steps", *[s.strip() for s in steps if s.strip()],
    ]
    if stage1_mode == "pure_fusion":
        argv.append("--pure-fusion-stage1")
    elif stage1_mode == "legacy_cv_seg":
        argv.append("--legacy-cv-seg")
    # else: hybrid (default)

    if mw:
        argv += ["--metrics-workers", mw]
    if fw:
        argv += ["--feedback-workers", fw]
    if local_video_dir:
        argv += ["--local-video-dir", local_video_dir]

    print(f"[worker] running: {' '.join(argv)}", flush=True)
    rc = subprocess.call(argv, cwd=str(REPO))
    if rc != 0:
        # run_pipeline.py mid-stage failures already leave analyticsStatus
        # at the last "Processing (X%)" they wrote — but if the orchestrator
        # itself crashes (e.g. before any stage logged), the field would be
        # stuck at "Ready for Analysis". Mark Failed for visibility.
        try:
            _pp.mark_failed(customer_id, vID, reason=f"worker exit {rc}")
        except Exception as e:
            print(f"[worker] mark_failed swallowed error: {e}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
