"""Ensure a game's full video is present (and current) in GCS before the
detection pipeline runs.

The CV pipeline (cv_seg / YOLO / audio) processes actual pixels + audio, so
it needs the real file at:

    gs://goalie_video_bucket/ground_truth_video/full_video/full_<vID>.mp4

App-submitted games are NOT pre-fetched there (the web app streams the
YouTube URL straight to Gemini and never uploads the file). This module
fetches the video from the customer config's `eventVideoURL` with yt-dlp and
uploads it — but only when needed:

  - object missing in GCS                         → download + upload
  - YouTube upload timestamp > GCS object's       → download + upload
    last-updated time (video was re-published)
  - otherwise                                     → up to date, no-op

Freshness caveat: YouTube only exposes an *upload/publish* timestamp, not an
"edited in place" time. This catches a (re)published / replaced video, but
NOT a silent in-place edit (YouTube doesn't surface those). On any error
reading the YouTube timestamp we KEEP the existing GCS copy (fail-safe — we
don't re-download on uncertainty).

Worker memory note: Cloud Run's /tmp is memory-backed, and full games can be
multiple GB. If downloads OOM the worker, either raise the Job's --memory or
set ENSURE_VIDEO_MAX_HEIGHT (e.g. 720) to fetch a smaller rendition.

CLI:
    python3 util/ensure_video.py --vID <vID> --customer-id <CUST...> [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Mirror the pipeline's GCS layout (cv_seg/constants.py + metrics_seg).
GCS_BUCKET   = os.environ.get("GCS_BUCKET", "goalie_video_bucket")
VIDEO_PREFIX = os.environ.get("VIDEO_PREFIX", "ground_truth_video/full_video")
CONFIG_PREFIX = os.environ.get("GCS_PREFIX", "customerID")
# Optional max rendition height to bound download size on small workers.
_MAX_HEIGHT = os.environ.get("ENSURE_VIDEO_MAX_HEIGHT")


def _storage_client():
    from google.cloud import storage
    return storage.Client()


def _youtube_url_for(vID: str, customer_id: str, bucket_name: str) -> Optional[str]:
    """Read the customer config from GCS and return the vID's eventVideoURL."""
    cust_file = customer_id if customer_id.endswith(".json") else f"{customer_id}.json"
    blob = _storage_client().bucket(bucket_name).blob(f"{CONFIG_PREFIX}/{cust_file}")
    try:
        data = json.loads(blob.download_as_text())
    except Exception as e:
        raise RuntimeError(f"could not read customer config "
                           f"{CONFIG_PREFIX}/{cust_file}: {e}")
    if not isinstance(data, list):
        raise RuntimeError(f"customer config {cust_file} is not a list")
    rec = next((r for r in data if str(r.get("vID")) == str(vID)), None)
    if rec is None:
        raise RuntimeError(f"no record for vID={vID} in {cust_file}")
    url = (rec.get("eventVideoURL") or "").strip()
    return url or None


def _youtube_upload_epoch(url: str) -> Optional[float]:
    """Best-effort UTC epoch of the YouTube upload/publish time, via yt-dlp
    metadata only (no download). None if it can't be determined."""
    try:
        import yt_dlp
        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"  ensure_video: could not read YouTube metadata ({e}); "
              f"keeping existing GCS copy", file=sys.stderr)
        return None
    ts = info.get("timestamp")
    if isinstance(ts, (int, float)):
        return float(ts)
    upload_date = info.get("upload_date")  # 'YYYYMMDD'
    if isinstance(upload_date, str) and len(upload_date) == 8 and upload_date.isdigit():
        dt = datetime(int(upload_date[:4]), int(upload_date[4:6]),
                      int(upload_date[6:8]), tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _download_to(url: str, dest_dir: str) -> Path:
    """Download the best mp4 (optionally height-capped) into dest_dir; return
    the produced file path."""
    import yt_dlp
    height_cap = f"[height<={_MAX_HEIGHT}]" if _MAX_HEIGHT else ""
    fmt = (f"bestvideo{height_cap}[ext=mp4]+bestaudio[ext=m4a]/"
           f"best{height_cap}[ext=mp4]/best{height_cap}/best")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(dest_dir, "video.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = [p for p in Path(dest_dir).iterdir() if p.is_file()]
    if not files:
        raise RuntimeError("yt-dlp produced no output file")
    # Prefer the merged .mp4; fall back to whatever single file exists.
    return next((p for p in files if p.suffix.lower() == ".mp4"), files[0])


def ensure_video(
    vID: str,
    customer_id: str,
    *,
    bucket_name: str = GCS_BUCKET,
    video_prefix: str = VIDEO_PREFIX,
    force: bool = False,
) -> dict:
    """Make sure full_<vID>.mp4 exists and is current in GCS.

    Returns a dict {"action": ..., "blob": gs_uri}. `action` is one of:
      present    — already in GCS and up to date (no work)
      downloaded — was missing; fetched + uploaded
      refreshed  — existed but stale; re-fetched + re-uploaded
    Raises on hard failure (no URL + missing object, or download error).
    """
    blob_path = f"{video_prefix}/full_{vID}.mp4"
    gs_uri = f"gs://{bucket_name}/{blob_path}"
    blob = _storage_client().bucket(bucket_name).blob(blob_path)
    exists = blob.exists()

    if exists and not force:
        url = _youtube_url_for(vID, customer_id, bucket_name)
        if not url:
            # Can't check freshness without the URL — keep what we have.
            print(f"  ensure_video: {gs_uri} present; no eventVideoURL to "
                  f"check freshness — keeping existing", file=sys.stderr)
            return {"action": "present", "blob": gs_uri}
        yt_epoch = _youtube_upload_epoch(url)
        if yt_epoch is None:
            return {"action": "present", "blob": gs_uri}
        blob.reload()  # populate .updated
        gcs_epoch = blob.updated.timestamp() if blob.updated else 0.0
        if yt_epoch <= gcs_epoch:
            print(f"  ensure_video: {gs_uri} is current "
                  f"(YouTube {int(yt_epoch)} <= GCS {int(gcs_epoch)})",
                  file=sys.stderr)
            return {"action": "present", "blob": gs_uri}
        print(f"  ensure_video: {gs_uri} STALE — YouTube upload "
              f"({int(yt_epoch)}) newer than GCS copy ({int(gcs_epoch)}); "
              f"re-fetching", file=sys.stderr)
        action = "refreshed"
    else:
        url = _youtube_url_for(vID, customer_id, bucket_name)
        if not url:
            raise RuntimeError(
                f"no video in GCS for vID={vID} and no eventVideoURL in "
                f"{customer_id} config to fetch it from")
        print(f"  ensure_video: {gs_uri} {'forced' if force else 'missing'} — "
              f"fetching from {url}", file=sys.stderr)
        action = "downloaded"

    tmpdir = tempfile.mkdtemp(prefix="ensure_video_")
    try:
        src = _download_to(url, tmpdir)
        size_mb = src.stat().st_size / 1_048_576
        print(f"  ensure_video: downloaded {size_mb:.0f} MB → uploading to {gs_uri}",
              file=sys.stderr)
        blob.upload_from_filename(str(src), content_type="video/mp4")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return {"action": action, "blob": gs_uri}


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensure full_<vID>.mp4 is in GCS.")
    ap.add_argument("--vID", required=True)
    ap.add_argument("--customer-id", required=True)
    ap.add_argument("--force", action="store_true",
                    help="re-download even if present and current")
    args = ap.parse_args()
    try:
        result = ensure_video(args.vID, args.customer_id, force=args.force)
    except Exception as e:
        print(f"ensure_video FAILED: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
