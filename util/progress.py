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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("util.progress")

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
    _update_status(customer_id, vid, f"Processing ({pct}%)")


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
    _update_status(customer_id, vid, status)


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

    # Push to GCS — read-modify-write because the GCS copy is the source
    # of truth for downstream readers and another process could be
    # mutating it. Best-effort; never raise.
    try:
        from google.cloud import storage as gcs_storage
        client = gcs_storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_blob)
        try:
            gcs_data = json.loads(blob.download_as_text())
        except Exception:
            # If we wrote locally just now, fall back to the local view.
            # If GCS is empty AND local missing, nothing to update.
            gcs_data = data
        if isinstance(gcs_data, list):
            updated = False
            for rec in gcs_data:
                if rec.get("vID") == vid:
                    rec.update(fields)
                    updated = True
            if updated:
                blob.upload_from_string(
                    json.dumps(gcs_data, indent=2),
                    content_type="application/json",
                )
    except ImportError:
        pass  # google-cloud-storage not installed; local-only is fine
    except Exception as e:
        log.warning(f"progress: GCS write failed for gs://{GCS_BUCKET}/{gcs_blob}: {e}")
