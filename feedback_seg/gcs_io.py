"""GCS read/write/upload/delete helpers.

Single module-level GCS client (one per process), reused across all
threads (the GCS SDK client is thread-safe).
"""

import json
import tempfile
from typing import Any, Optional

from google.cloud import storage

from .constants import BUCKET_NAME
from .logger import log


_gcs_client: Optional[storage.Client] = None


def get_gcs_client() -> storage.Client:
    """Return the module-level singleton GCS client."""
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def get_bucket() -> storage.Bucket:
    return get_gcs_client().bucket(BUCKET_NAME)


def gcs_read_json(bucket: storage.Bucket, blob_path: str) -> Any:
    return json.loads(bucket.blob(blob_path).download_as_text())


def gcs_blob_exists(bucket: storage.Bucket, blob_path: str) -> bool:
    return bucket.blob(blob_path).exists()


def gcs_download_to_temp(bucket: storage.Bucket, blob_path: str,
                         suffix: str) -> str:
    """Download a blob to a NamedTemporaryFile; return the local path.

    Caller is responsible for deleting the temp file.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    log.info("Downloading from GCS",
             extra={"src": f"gs://{BUCKET_NAME}/{blob_path}"})
    bucket.blob(blob_path).download_to_filename(tmp.name)
    return tmp.name


def gcs_upload_file(bucket: storage.Bucket, local_path: str,
                    blob_path: str) -> str:
    """Upload local file to GCS; return the gs:// URI."""
    blob = bucket.blob(blob_path)
    blob.chunk_size = 8 * 1024 * 1024
    blob.upload_from_filename(local_path)
    uri = f"gs://{BUCKET_NAME}/{blob_path}"
    log.info("Uploaded clip to GCS", extra={"uri": uri})
    return uri


def gcs_delete_blob(bucket: storage.Bucket, blob_path: str) -> None:
    """Delete a GCS blob; swallow errors (idempotent cleanup)."""
    try:
        b = bucket.blob(blob_path)
        if b.exists():
            b.delete()
    except Exception as e:
        log.warning(f"Could not delete GCS blob {blob_path}: {e}")


def gcs_write_json(bucket: storage.Bucket, blob_path: str, data: Any) -> None:
    bucket.blob(blob_path).upload_from_string(
        json.dumps(data, indent=2),
        content_type="application/json",
    )
    log.info("Written to GCS",
             extra={"dst": f"gs://{BUCKET_NAME}/{blob_path}"})
