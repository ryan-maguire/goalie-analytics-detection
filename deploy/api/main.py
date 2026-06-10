"""FastAPI Cloud Run Service for the goalie-analytics pipeline.

Endpoints:
  POST /run                                — dispatch pipeline run(s) as
                                             Cloud Run Job executions
  GET  /status/{customer_id}/{vID}         — read analyticsStatus from
                                             the customer config in GCS
  GET  /health                             — liveness probe

Auth: IAM-based. The Service is deployed with --no-allow-unauthenticated,
so the caller MUST present a Google-issued ID token via the
`Authorization: Bearer <token>` header. The Service's own service account
needs `roles/run.developer` (or narrower equivalents) on the Job project
to dispatch executions.

Architecture: API is a thin dispatcher. POST /run validates inputs,
creates a Cloud Run Job execution per vID with env-var overrides
(CUSTOMER_ID, VID, STAGE1_MODE, STEPS, HYBRID_MIN_WINDOWS), and returns
the execution name(s) immediately. The actual pipeline work happens in
the Job worker (see deploy/worker/run.py). Status is observable via
GET /status, which polls the customer JSON's analyticsStatus field
(updated by util/progress.py during the run — see commit d8a0584).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

# A run that hasn't updated its heartbeat in this long is treated as dead. The
# worker writes updatedDate on every progress tick (util/progress.py); if the
# Cloud Run Job is OOM/timeout/SIGKILL-ed it stops ticking and the record would
# otherwise sit at 'Processing (X%)' forever.
STALE_RUN_SECS = 45 * 60


def _parse_iso(value) -> Optional[datetime]:
    """Parse an ISO-8601 'updatedDate' to an aware UTC datetime, else None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# customer_id flows into a GCS object path; restrict to a safe charset so
# a caller can't traverse to or probe arbitrary keys (e.g. "../foo").
# Intentionally permissive on format (CUST000031, CUST_LEARNCURVE, …) —
# it only forbids path-control characters.
_CUST_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

# Google Cloud clients — lazy-init at first use so module import doesn't
# fail in local dev environments without credentials.
_GCS_CLIENT = None
_RUN_CLIENT = None

log = logging.getLogger("api.main")
logging.basicConfig(level=logging.INFO)

# ── Config (override via env in the deployed service) ────────────────
PROJECT_ID  = os.environ.get("PROJECT_ID",  "goalie-analytics-pro-dev")
REGION      = os.environ.get("REGION",      "us-central1")
JOB_NAME    = os.environ.get("JOB_NAME",    "goalie-pipeline-worker")
GCS_BUCKET  = os.environ.get("GCS_BUCKET",  "goalie_video_bucket")
GCS_PREFIX  = os.environ.get("GCS_PREFIX",  "customerID")

app = FastAPI(
    title="goalie-analytics pipeline API",
    description="Dispatches pipeline runs as Cloud Run Job executions "
                "and exposes status polling.",
    version="1.0.0",
)


# ── Request / response schemas ───────────────────────────────────────

class RunRequest(BaseModel):
    customer_id: str = Field(..., description="e.g. 'CUST000031'")
    vID:         list[str] = Field(..., min_length=1, description=
        "One or more video IDs. Each vID dispatches its own Job execution.")
    stage1_mode: str = Field("hybrid", pattern="^(hybrid|pure_fusion|legacy_cv_seg)$",
        description="hybrid (default) | pure_fusion | legacy_cv_seg")
    steps: list[int] = Field([1, 2, 3], description=
        "Pipeline steps to run (1=stage1, 2=metrics_seg, 3=feedback_seg)")
    hybrid_min_windows: int = Field(30, ge=1, description=
        "Fusion-output threshold below which the hybrid mode falls back "
        "to cv_seg. Default 30 (calibrated from 14-game validation).")
    metrics_workers: Optional[int] = Field(None, ge=1, le=8, description=
        "Per-Job parallelism for metrics_seg (Gemini concurrency). "
        "Default = stage's built-in (2).")
    feedback_workers: Optional[int] = Field(None, ge=1, le=8, description=
        "Per-Job parallelism for feedback_seg. Default = built-in (3).")

    @field_validator("customer_id")
    @classmethod
    def _valid_customer_id(cls, v: str) -> str:
        if not _CUST_ID_RE.match(v):
            raise ValueError(
                "customer_id may contain only letters, digits, '_', '.', '-'")
        return v

    @field_validator("vID")
    @classmethod
    def _no_duplicate_vids(cls, v: list[str]) -> list[str]:
        # Duplicate vIDs would dispatch the same video twice, racing on the
        # same customer-JSON status field and GCS output paths.
        if len(v) != len(set(v)):
            raise ValueError("vID list must not contain duplicate IDs")
        return v


class RunResponse(BaseModel):
    customer_id: str
    executions: list[dict]   # [{vID, execution_name, status_url, error}]


# ── Helpers ──────────────────────────────────────────────────────────

def _gcs_client():
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        from google.cloud import storage
        _GCS_CLIENT = storage.Client(project=PROJECT_ID)
    return _GCS_CLIENT


def _run_client():
    global _RUN_CLIENT
    if _RUN_CLIENT is None:
        from google.cloud import run_v2
        _RUN_CLIENT = run_v2.JobsClient()
    return _RUN_CLIENT


def _dispatch_job(req: RunRequest, vID: str) -> str:
    """Create one Job execution with env overrides. Returns execution name."""
    from google.cloud import run_v2

    env = [
        run_v2.EnvVar(name="CUSTOMER_ID", value=req.customer_id),
        run_v2.EnvVar(name="VID",         value=vID),
        run_v2.EnvVar(name="STAGE1_MODE", value=req.stage1_mode),
        run_v2.EnvVar(name="STEPS",       value=",".join(str(s) for s in req.steps)),
        run_v2.EnvVar(name="HYBRID_MIN_WINDOWS", value=str(req.hybrid_min_windows)),
    ]
    if req.metrics_workers is not None:
        env.append(run_v2.EnvVar(name="METRICS_WORKERS", value=str(req.metrics_workers)))
    if req.feedback_workers is not None:
        env.append(run_v2.EnvVar(name="FEEDBACK_WORKERS", value=str(req.feedback_workers)))

    job_full = f"projects/{PROJECT_ID}/locations/{REGION}/jobs/{JOB_NAME}"
    request = run_v2.RunJobRequest(
        name=job_full,
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(env=env),
            ],
            task_count=1,
        ),
    )
    op = _run_client().run_job(request=request)
    # Operation name encodes the execution id; the response after
    # completion would contain Execution, but we don't wait for it.
    return op.operation.name


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "project": PROJECT_ID, "region": REGION,
            "job": JOB_NAME}


@app.post("/run", response_model=RunResponse)
def run_pipeline(req: RunRequest):
    """Dispatch one Cloud Run Job execution per vID. Returns immediately.

    Dispatch is per-vID and independent: a failure on one vID neither
    aborts the others nor discards the execution names of vIDs already
    dispatched. Each execution entry carries an ``error`` field (None on
    success). A blanket 502 is raised ONLY if every vID failed — so a
    client that retries on 502 cannot double-dispatch the vIDs that had
    already succeeded.
    """
    executions = []
    for vID in req.vID:
        try:
            op_name = _dispatch_job(req, vID)
            executions.append({
                "vID": vID,
                "execution_name": op_name,
                "status_url": f"/status/{req.customer_id}/{vID}",
                "error": None,
            })
            log.info(f"dispatched: customer={req.customer_id} vID={vID} op={op_name}")
        except Exception as e:
            log.error(f"dispatch failed for {req.customer_id}/{vID}: {e}")
            executions.append({
                "vID": vID,
                "execution_name": None,
                "status_url": None,
                "error": str(e),
            })
    if executions and all(e["error"] for e in executions):
        raise HTTPException(
            status_code=502,
            detail=f"All {len(executions)} dispatch(es) failed; "
                   f"first error: {executions[0]['error']}",
        )
    return RunResponse(customer_id=req.customer_id, executions=executions)


@app.get("/status/{customer_id}/{vID}")
def get_status(customer_id: str, vID: str):
    """Return the vID's analyticsStatus from the GCS customer config.

    Status values written by the pipeline:
      - 'Ready for Analysis'   — initial, untouched
      - 'Processing (X%)'      — mid-run; X reflects 33% per stage +
                                  per-step fraction within active stage
      - 'Complete'             — full 3-stage success
      - 'Failed: <reason>'     — pipeline aborted
    """
    if not _CUST_ID_RE.match(customer_id.removesuffix(".json")):
        raise HTTPException(status_code=400, detail="Invalid customer_id")
    cust_blob = customer_id if customer_id.endswith(".json") else f"{customer_id}.json"
    blob_path = f"{GCS_PREFIX}/{cust_blob}"
    try:
        bucket = _gcs_client().bucket(GCS_BUCKET)
        blob = bucket.blob(blob_path)
        data = json.loads(blob.download_as_text())
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Could not read gs://{GCS_BUCKET}/{blob_path}: {e}",
        )
    if not isinstance(data, list):
        raise HTTPException(status_code=500,
            detail="Customer config is not a list of vID records")
    rec = next((r for r in data if r.get("vID") == vID), None)
    if rec is None:
        raise HTTPException(status_code=404,
            detail=f"No record for vID={vID} in customer {customer_id}")

    status = rec.get("analyticsStatus")
    updated = rec.get("updatedDate")

    # Stale-run watchdog: a 'Processing' record whose heartbeat is older than
    # STALE_RUN_SECS means the worker died mid-run. Surface a derived Failed
    # status so readers stop polling — derived only, we do NOT mutate the file
    # from a GET (that would be a racy side effect on a read path).
    stalled = False
    if isinstance(status, str) and status.startswith("Processing"):
        ts = _parse_iso(updated)
        if ts is not None:
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > STALE_RUN_SECS:
                stalled = True
                status = f"Failed: stalled (no update for {int(age // 60)} min)"

    return {
        "customer_id":     customer_id,
        "vID":             vID,
        "analyticsStatus": status,
        "updatedDate":     updated,
        "stalled":         stalled,
    }
