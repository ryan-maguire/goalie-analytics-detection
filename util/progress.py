"""Pipeline progress reporting → customer JSON's analyticsStatus field.

Used by cv_seg / metrics_seg / feedback_seg / fusion stage 1 to update the
per-vID `analyticsStatus` field as work progresses. Format: "Processing (X%)"
during work, "Complete" when all three stages succeed.

Math: total progress = ((stage_idx - 1) + current/total) / 3 × 100, where
stage_idx ∈ {1, 2, 3} and current/total is the within-stage step. So:
  - stage 1 mid: 0%–33%
  - stage 2 mid: 33%–67%
  - stage 3 mid: 67%–100%

Writes are best-effort to both LOCAL (data/customers/<customer_id>.json)
and GCS (customerID/<customer_id>.json in the bucket). Either failing
does NOT raise — progress reporting must not break the pipeline.

The functions are no-ops if `stage_idx` is None — stages call these
helpers unconditionally and pass through their own CLI flag, so running
a stage standalone (no `--progress-stage-idx`) leaves the customer
JSON untouched.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("util.progress")

_GCS_RMW_RETRIES = 8


def _now_iso() -> str:
    """UTC ISO-8601 'last touched' heartbeat. Written on every progress tick so
    a watchdog can tell a live run from a worker that died mid-run (the customer
    record otherwise keeps a stale 'Processing (X%)' forever)."""
    return datetime.now(timezone.utc).isoformat()

REPO = Path(__file__).resolve().parents[1]
LOCAL_DIR = REPO / "data" / "customers"
GCS_BUCKET = "goalie_video_bucket"
GCS_PREFIX = "customerID"


def report(
    customer_id: Optional[str],
    vid: str,
    *,
    stage_idx: Optional[int],
    current: int,
    total: int,
) -> None:
    """Write 'Processing (X%)' to the vid's analyticsStatus.

    No-op if customer_id or stage_idx is None, or total <= 0.
    Errors during write are logged but never raised.
    """
    if not customer_id or stage_idx is None or total <= 0:
        return
    if stage_idx not in (1, 2, 3):
        log.warning(f"progress.report: invalid stage_idx={stage_idx}; expected 1, 2, or 3")
        return
    # Clamp current to [0, total] so reordering or off-by-one doesn't
    # produce 105% during a long run
    current = max(0, min(current, total))
    frac = ((stage_idx - 1) + current / total) / 3.0
    pct = int(round(frac * 100))
    # Stamp updatedDate on every tick so the status is a heartbeat, not just a
    # percentage — lets a watchdog distinguish a slow-but-alive run from a dead
    # worker frozen at "Processing (X%)".
    _apply_fields(customer_id, vid, {
        "analyticsStatus": f"Processing ({pct}%)",
        "updatedDate": _now_iso(),
    })


def mark_complete(
    customer_id: Optional[str],
    vid: str,
    *,
    analytics_duration_secs: Optional[float] = None,
    segment_duration_secs: Optional[float] = None,
) -> None:
    """Mark the vid Complete and record run-summary fields.

    Always sets analyticsStatus='Complete' and analyticsUpdateDate (the
    success date, "MM/DD/YYYY"). When provided, also sets:
      - analyticsDuration: total pipeline wall time, "MM:SS"
      - segmentDuration:   total analyzed-clip duration, "HH:MM"
    """
    if not customer_id:
        return
    fields = {
        "analyticsStatus": "Complete",
        "analyticsUpdateDate": _today_mmddyyyy(),
        "updatedDate": _now_iso(),
    }
    if analytics_duration_secs is not None:
        fields["analyticsDuration"] = _fmt_mmss(analytics_duration_secs)
    if segment_duration_secs is not None:
        fields["segmentDuration"] = _fmt_hhmm(segment_duration_secs)
    _apply_fields(customer_id, vid, fields)


def mark_failed(customer_id: Optional[str], vid: str, *, reason: str = "") -> None:
    """Write 'Failed: <reason>' to the vid's analyticsStatus."""
    if not customer_id:
        return
    status = f"Failed: {reason}" if reason else "Failed"
    _apply_fields(customer_id, vid, {"analyticsStatus": status, "updatedDate": _now_iso()})


def _fmt_mmss(secs: float) -> str:
    """Total duration as MM:SS (minutes may exceed 59 for long runs)."""
    secs = max(0, int(round(secs)))
    return f"{secs // 60:02d}:{secs % 60:02d}"


def _fmt_hhmm(secs: float) -> str:
    """Total duration as HH:MM (seconds truncated)."""
    secs = max(0, int(round(secs)))
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"


def _today_mmddyyyy() -> str:
    """Today's date as MM/DD/YYYY in PIPELINE_TZ (default America/Chicago),
    falling back to UTC if the zone database is unavailable."""
    tz_name = os.environ.get("PIPELINE_TZ", "America/Chicago")
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo(tz_name))
    except Exception:
        now = datetime.now(timezone.utc)
    return now.strftime("%m/%d/%Y")


def _update_status(customer_id: str, vid: str, status: str) -> None:
    """Status-only convenience wrapper around _apply_fields."""
    _apply_fields(customer_id, vid, {"analyticsStatus": status})


def _apply_fields(customer_id: str, vid: str, fields: dict) -> None:
    """Read-modify-write the customer JSON, local AND GCS. Best effort.
    Merges `fields` into the matching vID record(s)."""
    cust_file = customer_id if customer_id.endswith(".json") else f"{customer_id}.json"
    local_path = LOCAL_DIR / cust_file
    gcs_blob = f"{GCS_PREFIX}/{cust_file}"

    # Update local (authoritative for in-process consistency). Use flock
    # to be safe across concurrent stage subprocesses, even though the
    # current orchestrator runs stages serially.
    data = None
    if local_path.exists():
        try:
            with open(local_path, "r+") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list):
                        updated = False
                        for rec in data:
                            if rec.get("vID") == vid:
                                rec.update(fields)
                                updated = True
                        if updated:
                            f.seek(0)
                            f.truncate()
                            json.dump(data, f, indent=2)
                finally:
                    try: fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception: pass
        except Exception as e:
            log.warning(f"progress: local write failed for {local_path}: {e}")

    # Push to GCS with OPTIMISTIC CONCURRENCY. The customer JSON holds every
    # vID for the customer and is written by multiple racing writers: this
    # progress tick, concurrent Cloud Run Job executions for *other* vIDs of the
    # same customer, the gateway's dispatch reset, and add/edit-video saves. A
    # naive download→upload here is last-write-wins and silently reverts another
    # writer's record (e.g. a sibling vID's Complete back to Processing). Read at
    # a captured generation and write with if_generation_match, retrying on
    # conflict. Still best-effort: never raise.
    try:
        from google.cloud import storage as gcs_storage
        from google.api_core.exceptions import PreconditionFailed, NotFound
        client = gcs_storage.Client()
        bucket = client.bucket(GCS_BUCKET)

        for attempt in range(_GCS_RMW_RETRIES):
            if attempt:
                # Jittered backoff so racing writers don't collide every retry.
                time.sleep(random.uniform(0, min(0.05 * (2 ** attempt), 0.5)))

            meta = bucket.get_blob(gcs_blob)
            if meta is None:
                # No GCS copy yet. Only create from the local view we just wrote
                # (avoid materializing an empty file); skip otherwise.
                if not isinstance(data, list):
                    break
                generation, gcs_data = 0, data
            else:
                generation = meta.generation
                try:
                    pinned = bucket.blob(gcs_blob, generation=generation)
                    gcs_data = json.loads(pinned.download_as_text())
                except (PreconditionFailed, NotFound):
                    continue  # changed/deleted between get_blob and download

            if not isinstance(gcs_data, list):
                break
            updated = False
            for rec in gcs_data:
                if isinstance(rec, dict) and rec.get("vID") == vid:
                    rec.update(fields)
                    updated = True
            if not updated:
                break  # this vID isn't in the file — nothing to write

            try:
                bucket.blob(gcs_blob).upload_from_string(
                    json.dumps(gcs_data, indent=2),
                    content_type="application/json",
                    if_generation_match=generation,
                )
                break
            except PreconditionFailed:
                continue  # another writer won the race — re-read and retry
        else:
            log.warning(
                f"progress: GCS write exhausted {_GCS_RMW_RETRIES} retries for "
                f"gs://{GCS_BUCKET}/{gcs_blob} (write contention)")
    except ImportError:
        pass  # google-cloud-storage not installed; local-only is fine
    except Exception as e:
        log.warning(f"progress: GCS write failed for gs://{GCS_BUCKET}/{gcs_blob}: {e}")
