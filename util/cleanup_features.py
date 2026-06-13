"""Post-reprocess cleanup of derived, user-owned features keyed to a video.

When a video is rerun through the pipeline its clips are regenerated. A clipID
is ``{vID}_{start}_{end}`` (see feedback_seg/video.py::make_clip_id), so any
shift in segment boundaries — common when the model/prompt changes, which is
usually *why* a video is rerun — produces new clipIDs and orphans every stored
reference to the old ones. This module purges EVERY derived reference tied to
the reprocessed vID so nothing stale lingers:

  - Favorites          customerID/<cust>_favorites.json   (clipFavorites[])
  - Playlists          customerID/<cust>_playlists.json   (playlists[].clipIds[])
  - Public shares      public_shares/<slug>.json          (clips[] where vID matches)
  - Coach feedback     <vID>_feedback.json                (entire file — it is vID-scoped)

All writes use optimistic concurrency (generation-match RMW) so a racing UI
write or a sibling pipeline run can't be silently clobbered, mirroring
util/progress.py. The whole thing is BEST-EFFORT: a failure in any store is
logged and swallowed so it can never fail the pipeline run.
"""

import copy
import json
import random
import time

BUCKET = "goalie_video_bucket"
CUST_PREFIX = "customerID"
SHARE_PREFIX = "public_shares/"

_GCS_RMW_RETRIES = 10


def _bare_customer(customer_id: str) -> str:
    """Normalize an arg like 'CUST000048', 'CUST000048.json', or
    'customerID/CUST000048' down to the bare 'CUST000048' the side-files use."""
    c = str(customer_id or "")
    if c.startswith(f"{CUST_PREFIX}/"):
        c = c[len(CUST_PREFIX) + 1:]
    if c.endswith(".json"):
        c = c[: -len(".json")]
    return c


def _belongs_to_vid(clip_id, vid: str) -> bool:
    """True when a stored clipID belongs to this video. clipID is
    ``{vID}_{start}_{end}``, so an exact-vID or '<vID>_' prefix match is it."""
    s = str(clip_id)
    return s == vid or s.startswith(f"{vid}_")


def _rmw(bucket, blob_name, mutate_fn, *, default, log):
    """Generation-match read-modify-write of one JSON blob.

    ``mutate_fn(data) -> (new_data, changed)``. When the blob is absent we start
    from a deep copy of ``default``; if nothing changed we never write (so a
    missing file is left missing). Returns True iff a write happened. Raises only
    when retries are exhausted under contention — the caller swallows it.
    """
    from google.api_core.exceptions import PreconditionFailed, NotFound

    for attempt in range(_GCS_RMW_RETRIES):
        if attempt:
            # Jittered backoff so racing writers don't collide every retry.
            time.sleep(random.uniform(0, min(0.05 * (2 ** attempt), 0.5)))

        meta = bucket.get_blob(blob_name)
        if meta is None:
            generation, data = 0, copy.deepcopy(default)
        else:
            generation = meta.generation
            try:
                pinned = bucket.blob(blob_name, generation=generation)
                data = json.loads(pinned.download_as_text())
            except (PreconditionFailed, NotFound):
                continue  # changed/deleted between get_blob and download — retry

        new_data, changed = mutate_fn(data)
        if not changed:
            return False

        try:
            bucket.blob(blob_name).upload_from_string(
                json.dumps(new_data, indent=2),
                content_type="application/json",
                if_generation_match=generation,
            )
            return True
        except PreconditionFailed:
            continue  # another writer won the race — re-read and retry

    raise RuntimeError(f"RMW exhausted {_GCS_RMW_RETRIES} retries for {blob_name}")


def cleanup_features_for_vid(customer_id: str, vid: str, *, log=print) -> dict:
    """Purge every derived reference to ``vid`` for ``customer_id``.

    Best-effort and side-effect-only: returns a summary of how many references
    were removed per store (or -1 on a store-level error). Never raises.
    """
    cust = _bare_customer(customer_id)
    summary = {"favorites": 0, "playlists": 0, "shares": 0, "feedback": 0, "validations": 0}

    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        log(f"cleanup: google-cloud-storage unavailable — skipped vID={vid}")
        return summary

    try:
        client = gcs_storage.Client()
        bucket = client.bucket(BUCKET)
    except Exception as e:  # noqa: BLE001
        log(f"cleanup: GCS client init failed ({type(e).__name__}: {e}) — skipped")
        return summary

    # ── 1. Favorites ────────────────────────────────────────────────
    def _fav_mutate(doc):
        if isinstance(doc, list):  # tolerate an older bare-list shape
            doc = {"clientID": cust, "clipFavorites": doc}
        if not isinstance(doc, dict):
            return doc, False
        favs = doc.get("clipFavorites")
        if not isinstance(favs, list):
            return doc, False
        kept = [c for c in favs if not _belongs_to_vid(c, vid)]
        removed = len(favs) - len(kept)
        if not removed:
            return doc, False
        doc["clipFavorites"] = kept
        summary["favorites"] = removed
        return doc, True

    try:
        _rmw(bucket, f"{CUST_PREFIX}/{cust}_favorites.json", _fav_mutate,
             default={"clientID": cust, "clipFavorites": []}, log=log)
    except Exception as e:  # noqa: BLE001
        summary["favorites"] = -1
        log(f"cleanup: favorites failed for {cust} ({type(e).__name__}: {e})")

    # ── 2. Playlists ────────────────────────────────────────────────
    def _pl_mutate(doc):
        pls = doc.get("playlists") if isinstance(doc, dict) else None
        if not isinstance(pls, list):
            return doc, False
        removed = 0
        for pl in pls:
            ids = pl.get("clipIds") if isinstance(pl, dict) else None
            if not isinstance(ids, list):
                continue
            kept = [c for c in ids if not _belongs_to_vid(c, vid)]
            if len(kept) != len(ids):
                removed += len(ids) - len(kept)
                pl["clipIds"] = kept
        if not removed:
            return doc, False
        summary["playlists"] = removed
        return doc, True

    try:
        _rmw(bucket, f"{CUST_PREFIX}/{cust}_playlists.json", _pl_mutate,
             default={"clientID": cust, "playlists": []}, log=log)
    except Exception as e:  # noqa: BLE001
        summary["playlists"] = -1
        log(f"cleanup: playlists failed for {cust} ({type(e).__name__}: {e})")

    # ── 3. Public recruiting shares (no vID index — scan the prefix) ──
    def _share_mutate(doc):
        if not isinstance(doc, dict):
            return doc, False
        if str(doc.get("clientID")) != cust:
            return doc, False  # someone else's share — leave it
        clips = doc.get("clips")
        if not isinstance(clips, list):
            return doc, False
        kept = [c for c in clips if str((c or {}).get("vID")) != vid]
        removed = len(clips) - len(kept)
        if not removed:
            return doc, False
        doc["clips"] = kept
        if not kept:
            doc["revoked"] = True  # nothing left to show — turn the link off
        summary["shares"] += removed
        return doc, True

    try:
        for blob in client.list_blobs(BUCKET, prefix=SHARE_PREFIX):
            if not blob.name.endswith(".json"):
                continue
            try:
                _rmw(bucket, blob.name, _share_mutate, default={}, log=log)
            except Exception as e:  # noqa: BLE001
                log(f"cleanup: share {blob.name} failed ({type(e).__name__}: {e})")
    except Exception as e:  # noqa: BLE001
        summary["shares"] = -1
        log(f"cleanup: share scan failed ({type(e).__name__}: {e})")

    # ── 4. Coach feedback (the whole file is scoped to this vID) ──────
    try:
        fb = bucket.blob(f"{vid}_feedback.json")
        if fb.exists():
            fb.delete()
            summary["feedback"] = 1
    except Exception as e:  # noqa: BLE001
        summary["feedback"] = -1
        log(f"cleanup: feedback delete failed for {vid} ({type(e).__name__}: {e})")

    # ── 5. Clip validations (Confirm/Decline side-file, keyed by clipID) ──
    def _val_mutate(doc):
        vals = doc.get("clipValidations") if isinstance(doc, dict) else None
        if not isinstance(vals, dict):
            return doc, False
        stale = [cid for cid in vals if _belongs_to_vid(cid, vid)]
        if not stale:
            return doc, False
        for cid in stale:
            vals.pop(cid, None)
        summary["validations"] = len(stale)
        return doc, True

    try:
        _rmw(bucket, f"{CUST_PREFIX}/{cust}_validations.json", _val_mutate,
             default={"clientID": cust, "clipValidations": {}}, log=log)
    except Exception as e:  # noqa: BLE001
        summary["validations"] = -1
        log(f"cleanup: validations failed for {cust} ({type(e).__name__}: {e})")

    return summary
