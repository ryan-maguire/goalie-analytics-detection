"""Generate V4 signed URLs for private GCS objects (e.g. browser video
playback) — without a downloaded service-account key.

Videos live in a PRIVATE bucket, so a browser can't fetch a `gs://` URI or a
plain `https://storage.googleapis.com/...` link. A **signed URL** embeds a
time-limited signature in the query string, letting an unauthenticated client
GET the object until the URL expires. The browser never authenticates to GCS.

Keyless signing on Cloud Run
----------------------------
There's no private key in the runtime, so signing is delegated to the IAM
Credentials API (`signBlob`). For that to work the runtime service account
needs `roles/iam.serviceAccountTokenCreator` ON ITSELF, and the
`iamcredentials.googleapis.com` API must be enabled. We pass the SA email +
a fresh access token to generate_signed_url(), which routes signing through
IAM automatically.

Whichever service signs must hold that grant. Today it's the web app's
runtime SA (301726916294-compute@developer.gserviceaccount.com). If you ever
sign from the pipeline API instead, grant its SA (goalie-pipeline-sa@…) the
same role.

Usage (library)
---------------
    from util.signed_url import signed_video_url
    url = signed_video_url("mjEeE7p2Hz8")          # default upload prefix
    # → return {"url": url} from your endpoint; frontend sets <video src>.

CLI (smoke test)
----------------
    python3 util/signed_url.py --vID mjEeE7p2Hz8 --minutes 360
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from typing import Optional

GCS_BUCKET = os.environ.get("GCS_BUCKET", "goalie_video_bucket")
# Where analysis videos are uploaded. Matches GCS_VIDEO_PREFIX used by the
# pipeline stages (see cv_seg/constants.py).
VIDEO_PREFIX = os.environ.get(
    "GCS_VIDEO_PREFIX", "analyze_video/00-segement-video-upload")
# Generous default so a long game doesn't expire mid-viewing; the frontend
# can re-request on a playback error if it ever does.
DEFAULT_EXPIRY_MIN = int(os.environ.get("SIGNED_URL_EXPIRY_MIN", "360"))


def signed_url_for_blob(
    blob_name: str,
    *,
    bucket_name: str = GCS_BUCKET,
    minutes: int = DEFAULT_EXPIRY_MIN,
    method: str = "GET",
) -> str:
    """Return a V4 signed URL for `blob_name` in `bucket_name`.

    Works keyless on Cloud Run (IAM signBlob) and with a local key/ADC that
    carries a private key. Does not check that the object exists — signing is
    purely cryptographic; a URL for a missing object 404s on access.
    """
    import google.auth
    from google.auth.transport import requests as ga_requests
    from google.cloud import storage

    creds, _ = google.auth.default()
    # Populate creds.token (+ .service_account_email on Cloud Run) so the
    # keyless IAM signBlob path is available.
    creds.refresh(ga_requests.Request())

    blob = storage.Client(credentials=creds).bucket(bucket_name).blob(blob_name)

    kwargs = dict(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method=method,
    )
    # On Cloud Run (no private key) route signing through IAM. Local creds
    # that already have a signer (e.g. a key file) don't need these and would
    # reject them, so only pass when there's no built-in signer.
    sa_email = getattr(creds, "service_account_email", None)
    if sa_email and sa_email != "default" and not hasattr(creds, "signer"):
        kwargs["service_account_email"] = sa_email
        kwargs["access_token"] = creds.token

    return blob.generate_signed_url(**kwargs)


def signed_video_url(
    vID: str,
    *,
    bucket_name: str = GCS_BUCKET,
    video_prefix: str = VIDEO_PREFIX,
    minutes: int = DEFAULT_EXPIRY_MIN,
) -> str:
    """Signed GET URL for a game's uploaded video (`full_<vID>.mp4`)."""
    return signed_url_for_blob(
        f"{video_prefix}/full_{vID}.mp4",
        bucket_name=bucket_name,
        minutes=minutes,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a V4 signed playback URL for a game video.")
    ap.add_argument("--vID", required=True)
    ap.add_argument("--minutes", type=int, default=DEFAULT_EXPIRY_MIN,
                    help=f"URL lifetime in minutes (default {DEFAULT_EXPIRY_MIN})")
    args = ap.parse_args()
    try:
        print(signed_video_url(args.vID, minutes=args.minutes))
    except Exception as e:
        print(f"signed_url FAILED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
