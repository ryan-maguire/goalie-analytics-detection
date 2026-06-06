"""Verify a game's source video is present in GCS before the pipeline runs.

The app uploads the video for analysis directly to GCS (the front end writes
the file and records its gs:// URI in the customer config's `eventVideoURL`).
The detection pipeline (cv_seg / metrics_seg / feedback_seg) reads that file
straight from GCS, so this module no longer fetches anything — it just
pre-flights the upload:

  - object present in GCS   → OK, pipeline proceeds
  - object missing          → hard fail (the worker marks the vID Failed
                              with a clear reason instead of letting a stage
                              crash mid-run)

Source of truth is the record's `eventVideoURL`, which is now a full gs:// URI
under:

    gs://goalie_video_bucket/analyze_video/00-segement-video-upload/full_<vID>.mp4

If `eventVideoURL` is absent (legacy/eval records), we fall back to
constructing the path from GCS_VIDEO_PREFIX (the same prefix the pipeline
stages read from) so this check stays aligned with what actually gets read.

CLI:
    python3 util/ensure_video.py --vID <vID> --customer-id <CUST...>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Tuple

# Mirror the pipeline's GCS layout (cv_seg/constants.py + metrics_seg).
GCS_BUCKET    = os.environ.get("GCS_BUCKET", "goalie_video_bucket")
# Where the pipeline reads the source video from. Production overrides this via
# GCS_VIDEO_PREFIX; default matches the eval/ground-truth corpus.
VIDEO_PREFIX  = os.environ.get("GCS_VIDEO_PREFIX", "ground_truth_video/full_video")
CONFIG_PREFIX = os.environ.get("GCS_PREFIX", "customerID")


def _storage_client():
    from google.cloud import storage
    return storage.Client()


def _parse_gs_uri(uri: str) -> Optional[Tuple[str, str]]:
    """Split a gs://bucket/path/to/object URI into (bucket, blob). None if not
    a gs:// URI."""
    if not uri.startswith("gs://"):
        return None
    rest = uri[len("gs://"):]
    bucket, _, blob = rest.partition("/")
    if not bucket or not blob:
        return None
    return bucket, blob


def _event_video_url(vID: str, customer_id: str, bucket_name: str) -> Optional[str]:
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


def ensure_video(
    vID: str,
    customer_id: str,
    *,
    bucket_name: str = GCS_BUCKET,
    video_prefix: str = VIDEO_PREFIX,
) -> dict:
    """Confirm the source video for (vID, customer) exists in GCS.

    Resolution order:
      1. The record's `eventVideoURL` if it's a gs:// URI (authoritative).
      2. Otherwise gs://{bucket}/{video_prefix}/full_<vID>.mp4.

    Returns {"action": "present", "blob": gs_uri}. Raises RuntimeError if the
    object is missing or the config can't be resolved.
    """
    url = _event_video_url(vID, customer_id, bucket_name)
    parsed = _parse_gs_uri(url) if url else None
    if parsed:
        src_bucket, blob_path = parsed
    else:
        if url:
            # A non-gs:// eventVideoURL is no longer supported (videos are
            # uploaded to GCS, not linked). Surface it rather than silently
            # falling back, but still try the conventional path below.
            print(f"  ensure_video: eventVideoURL is not a gs:// URI "
                  f"({url!r}); falling back to {video_prefix}/full_{vID}.mp4",
                  file=sys.stderr)
        src_bucket = bucket_name
        blob_path = f"{video_prefix}/full_{vID}.mp4"

    gs_uri = f"gs://{src_bucket}/{blob_path}"
    blob = _storage_client().bucket(src_bucket).blob(blob_path)
    if not blob.exists():
        raise RuntimeError(
            f"source video missing in GCS: {gs_uri} "
            f"(vID={vID}, customer={customer_id}). The app must upload the "
            f"video before analysis.")
    print(f"  ensure_video: {gs_uri} present", file=sys.stderr)
    return {"action": "present", "blob": gs_uri}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify the source video for a vID exists in GCS.")
    ap.add_argument("--vID", required=True)
    ap.add_argument("--customer-id", required=True)
    args = ap.parse_args()
    try:
        result = ensure_video(args.vID, args.customer_id)
    except Exception as e:
        print(f"ensure_video FAILED: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
