"""Pipeline orchestrator for feedback_seg.

Loads metrics from GCS (or local), parallelizes per-window analysis,
generates the game summary, and writes the assembled output.
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from google import genai
from google.cloud import storage

from .constants import (
    BUCKET_NAME, COACH_PARALLEL_WORKERS, INPUT_PREFIX, OUTPUT_PREFIX,
    PROJECT_ID, REGION, TEMP_PREFIX, VIDEO_PREFIX,
)
from .gcs_io import (
    gcs_blob_exists, gcs_delete_blob, gcs_download_to_temp,
    gcs_read_json, gcs_upload_file, gcs_write_json, get_bucket,
)
from .gemini import (
    analyze_window, generate_summary, should_use_inline_bytes,
)
from .logger import log
from .video import extract_clip, make_clip_id, make_temp_clip_path


# ── Per-window record assembly ───────────────────────────────────────

def _build_record(segment: dict, clip_id: str, analysis: dict) -> dict:
    """Build the output record for a successfully-analyzed window."""
    metrics = segment.get("metrics") or {}
    goals = metrics.get("goals", 0) or 0

    record = {
        **segment,
        "clipID":          clip_id,
        "clip_start_time": segment["segment_start"],
        "clip_end_time":   segment["segment_end"],
        "clip_duration":   segment["segment_end"] - segment["segment_start"],
        "clipShot":        (metrics.get("shots", 0) or 0) > 0,
        "clipShotCount":   metrics.get("shots", 0) or 0,
        "clipSave":        (metrics.get("saves", 0) or 0) > 0,
        "clipSaveCount":   metrics.get("saves", 0) or 0,
        "clipHasGoal":     goals > 0,
    }
    if "error" in analysis:
        record["error"] = analysis["error"]
    else:
        record["technical_reasoning"] = analysis.get("technical_reasoning", "")
        record["goalie_positioning"]  = analysis.get("goalie_positioning", {})
        record["coaching_feedback"]   = analysis.get("coaching_feedback", {})
        # Optional caveats list — empty if Gemini omitted it
        record["analysis_confidence_caveats"] = analysis.get(
            "analysis_confidence_caveats", []
        )
    return record


def _error_record(segment: dict, clip_id: str, error: str) -> dict:
    """Build an output record for a window that failed to be analyzed."""
    return _build_record(segment, clip_id, {"error": error})


# ── Per-window worker ────────────────────────────────────────────────

def _process_window_task(args: tuple, ctx: dict) -> tuple[int, dict]:
    """Process one threat window: extract clip, send to Gemini, return record.

    Module-level so the ThreadPoolExecutor doesn't need a closure.

    Inline-bytes path: if the extracted clip fits the inline limit
    (`INLINE_BYTES_MAX_SIZE`), send it directly as bytes — no GCS
    round-trip. Falls back to upload-URI-delete only for oversized clips.
    """
    idx, segment = args
    vID            = ctx["vID"]
    total_windows  = ctx["total_windows"]
    local_video    = ctx["local_video"]
    bucket         = ctx["bucket"]
    gemini_client  = ctx["gemini_client"]
    goalie_color   = ctx["goalie_color"]
    opponent_color = ctx["opponent_color"]
    temp_gcs_lock  = ctx["temp_gcs_lock"]
    temp_gcs_paths = ctx["temp_gcs_paths"]

    start   = segment["segment_start"]
    end     = segment["segment_end"]
    clip_id = make_clip_id(vID, start, end)

    log.info(
        f"Processing window {idx + 1}/{total_windows}",
        extra={"segment_start": start, "segment_end": end, "clip_id": clip_id},
    )

    clip_path = make_temp_clip_path(suffix=".mp4")
    try:
        try:
            extract_clip(local_video, start, end, clip_path)
        except Exception as e:
            log.error(
                f"Failed to extract clip: {e}",
                extra={"segment_start": start},
            )
            return idx, _error_record(segment, clip_id, f"extract_clip: {e}")

        # Decide inline-bytes vs GCS-URI based on clip size
        if should_use_inline_bytes(clip_path):
            gcs_uri: Optional[str] = None
            gcs_clip_blob: Optional[str] = None
        else:
            # Oversized — must upload, and bucket must be available
            if bucket is None:
                msg = (
                    f"Clip exceeds inline-bytes size and no GCS bucket is "
                    f"configured (running in --no-gcs mode). Cannot proceed."
                )
                log.error(msg, extra={"segment_start": start})
                return idx, _error_record(segment, clip_id, msg)

            gcs_clip_blob = f"{TEMP_PREFIX}/{clip_id}.mp4"
            try:
                gcs_uri = gcs_upload_file(bucket, clip_path, gcs_clip_blob)
                with temp_gcs_lock:
                    temp_gcs_paths.add(gcs_clip_blob)
            except Exception as e:
                log.error(
                    f"Failed to upload clip: {e}",
                    extra={"segment_start": start},
                )
                return idx, _error_record(segment, clip_id, f"gcs_upload: {e}")

        # Send to Gemini
        analysis = analyze_window(
            gemini_client, clip_path, gcs_uri, segment,
            goalie_color, opponent_color,
        )

        # If we uploaded, clean up the temp blob
        if gcs_clip_blob is not None and bucket is not None:
            gcs_delete_blob(bucket, gcs_clip_blob)
            with temp_gcs_lock:
                temp_gcs_paths.discard(gcs_clip_blob)

        return idx, _build_record(segment, clip_id, analysis)

    finally:
        # Always clean up the local clip file
        try:
            if os.path.exists(clip_path):
                os.remove(clip_path)
        except OSError:
            pass


# ── Main pipeline ────────────────────────────────────────────────────

def process_video(
    customer_id: str,
    vID: str,
    workers: int = COACH_PARALLEL_WORKERS,
    local_video: Optional[str] = None,
    local_metrics: Optional[str] = None,
    local_config: Optional[str] = None,
    no_gcs: bool = False,
    output_dir: Optional[str] = None,
    progress_stage_idx: Optional[int] = None,
) -> bool:
    """Run feedback_seg for one video.

    Returns True on success, False on configuration or input failure.

    Args:
        customer_id: Customer config key, e.g. 'CUST000048'.
        vID: Video ID.
        workers: Parallel Gemini workers (per-window). Default 3.
        local_video: If provided, use this local file instead of
            downloading the video from GCS.
        local_metrics: If provided, read metrics from this local JSON
            file instead of GCS.
        local_config: If provided, read customer config from this local
            JSON file instead of GCS.
        no_gcs: If True, do not write output to GCS. Requires
            output_dir to be set.
        output_dir: Local directory to write output JSON to (in
            addition to or instead of GCS).
    """
    bucket: Optional[storage.Bucket] = None if no_gcs else get_bucket()
    gemini_client = genai.Client(
        vertexai=True, project=PROJECT_ID, location=REGION,
    )

    # ── 1. Load customer config ─────────────────────────────────────
    config = _load_config(bucket, customer_id, local_config)
    records = config if isinstance(config, list) else [config]
    match = next((r for r in records if str(r.get("vID", "")) == str(vID)),
                 None)
    if not match:
        log.error(f"No record for vID={vID} in customer config")
        return False

    goalie_color   = match.get("targetGoalieColor")
    opponent_color = match.get("opponentGoalieColor")
    if not goalie_color or not opponent_color:
        log.error("Config missing targetGoalieColor or opponentGoalieColor")
        return False

    log.info(f"[{vID}] Goalie: {goalie_color} / Opponent: {opponent_color}")

    # ── 2. Load metrics input ───────────────────────────────────────
    all_segments = _load_metrics(bucket, vID, local_metrics)
    if all_segments is None:
        return False

    threat_windows = [
        s for s in all_segments
        if s.get("segmentHasThreat") and s.get("metrics") is not None
    ]
    log.info(
        f"[{vID}] {len(all_segments)} total segments, "
        f"{len(threat_windows)} threat windows to analyse"
    )
    if not threat_windows:
        log.warning(f"[{vID}] No threat windows with metrics — nothing to do.")
        return True

    # ── 3. Get the full video ───────────────────────────────────────
    if local_video:
        if not os.path.exists(local_video):
            log.error(f"--local-video path does not exist: {local_video}")
            return False
        video_path = local_video
        cleanup_video_after = False
    else:
        if bucket is None:
            log.error(
                "Cannot download video without a bucket. "
                "Pass --local-video to use a local file."
            )
            return False
        video_blob = f"{VIDEO_PREFIX}/full_{vID}.mp4"
        video_path = gcs_download_to_temp(bucket, video_blob, suffix=".mp4")
        cleanup_video_after = True

    # ── 4. Per-window analysis ──────────────────────────────────────
    results: dict[int, dict] = {}
    results_lock  = threading.Lock()
    temp_gcs_lock = threading.Lock()
    temp_gcs_paths: set[str] = set()

    window_ctx = {
        "vID": vID,
        "total_windows": len(threat_windows),
        "local_video": video_path,
        "bucket": bucket,
        "gemini_client": gemini_client,
        "goalie_color": goalie_color,
        "opponent_color": opponent_color,
        "temp_gcs_lock": temp_gcs_lock,
        "temp_gcs_paths": temp_gcs_paths,
    }

    try:
        log.info(
            f"[{vID}] Analysing {len(threat_windows)} windows "
            f"with {workers} workers"
        )
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="feedback",
        ) as pool:
            futures = {
                pool.submit(_process_window_task, (idx, seg), window_ctx): idx
                for idx, seg in enumerate(threat_windows)
            }
            for future in as_completed(futures):
                idx = futures[future]
                exc = future.exception()
                if exc:
                    # Worker-exception fix (Defect 1): write an error
                    # record so the output preserves one-record-per-window.
                    log.error(
                        f"Worker raised on idx={idx}: {exc}",
                        extra={"error_type": type(exc).__name__},
                    )
                    seg = threat_windows[idx]
                    clip_id = make_clip_id(
                        vID, seg["segment_start"], seg["segment_end"],
                    )
                    with results_lock:
                        results[idx] = _error_record(
                            seg, clip_id,
                            f"unhandled worker exception: {exc}",
                        )
                else:
                    idx, record = future.result()
                    with results_lock:
                        results[idx] = record

                # Pipeline progress: report after each window completes
                # (success or error — both count toward the % done).
                if progress_stage_idx is not None:
                    try:
                        from util import progress as _pp
                        with results_lock:
                            n_done = len(results)
                        _pp.report(
                            customer_id=customer_id, vid=vID,
                            stage_idx=progress_stage_idx,
                            current=n_done,
                            total=len(threat_windows),
                        )
                    except ImportError:
                        pass

        window_records = [results[i] for i in range(len(threat_windows))]

    finally:
        # Clean up the downloaded video (if we downloaded it)
        if cleanup_video_after:
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except OSError:
                pass
        # Safety sweep for any orphaned GCS clips
        if bucket is not None:
            with temp_gcs_lock:
                orphans = list(temp_gcs_paths)
            for gcs_path in orphans:
                gcs_delete_blob(bucket, gcs_path)

    # ── 5. Generate summary ─────────────────────────────────────────
    log.info(f"[{vID}] Generating coaches summary")
    summary_data = generate_summary(
        gemini_client, window_records, goalie_color, opponent_color,
    )

    # ── 6. Assemble and write output ────────────────────────────────
    succeeded = sum(1 for r in window_records if "error" not in r)
    failed    = len(window_records) - succeeded

    final_output = [
        {
            "type": "summary",
            "response": {
                "videoID":                 vID,
                "goalie_jersey_color":     goalie_color,
                "opposition_jersey_color": opponent_color,
                "coaches_summary":         summary_data.get("coaches_summary", ""),
                "coaches_overall_rating":  summary_data.get("coaches_overall_rating", "N/A"),
                "windows_analysed":        len(window_records),
                "windows_succeeded":       succeeded,
                "windows_failed":          failed,
            },
        },
        {
            "type": "windows",
            "response": window_records,
        },
    ]

    _write_output(final_output, vID, bucket, no_gcs, output_dir)
    log.info(
        f"[{vID}] Complete",
        extra={"succeeded": succeeded, "failed": failed},
    )
    return True


# ── Helper: load customer config ─────────────────────────────────────

def _load_config(
    bucket: Optional[storage.Bucket],
    customer_id: str,
    local_config: Optional[str],
) -> Any:
    if local_config:
        log.info(f"Loading config from local file: {local_config}")
        if not os.path.exists(local_config):
            log.error(
                f"--local-config path does not exist: {local_config!r}. "
                f"Resolved relative to cwd={os.getcwd()!r}. "
                f"Pass an absolute path or cd to the right directory."
            )
            sys.exit(1)
        import json as _json
        with open(local_config) as f:
            return _json.load(f)

    if bucket is None:
        log.error(
            "No --local-config and no GCS bucket. "
            "Pass --local-config for --no-gcs mode."
        )
        sys.exit(1)

    config_filename = (
        customer_id if customer_id.endswith(".json") else f"{customer_id}.json"
    )
    config_blob = f"customerID/{config_filename}"
    log.info(f"Loading config from gs://{BUCKET_NAME}/{config_blob}")
    return gcs_read_json(bucket, config_blob)


# ── Helper: load metrics input ───────────────────────────────────────

def _load_metrics(
    bucket: Optional[storage.Bucket],
    vID: str,
    local_metrics: Optional[str],
) -> Optional[list]:
    if local_metrics:
        log.info(f"Loading metrics from local file: {local_metrics}")
        if not os.path.exists(local_metrics):
            log.error(f"--local-metrics path does not exist: {local_metrics}")
            return None
        import json as _json
        with open(local_metrics) as f:
            return _json.load(f)

    if bucket is None:
        log.error(
            "No --local-metrics and no GCS bucket. "
            "Pass --local-metrics for --no-gcs mode."
        )
        return None

    input_blob = f"{INPUT_PREFIX}/gt_metrics_{vID}.json"
    log.info(f"Loading metrics from gs://{BUCKET_NAME}/{input_blob}")
    if not gcs_blob_exists(bucket, input_blob):
        log.error(
            f"Metrics file not found at gs://{BUCKET_NAME}/{input_blob}. "
            f"Run metrics_seg for {vID} first."
        )
        return None
    return gcs_read_json(bucket, input_blob)


# ── Helper: write final output ───────────────────────────────────────

def _write_output(
    final_output: list,
    vID: str,
    bucket: Optional[storage.Bucket],
    no_gcs: bool,
    output_dir: Optional[str],
) -> None:
    """Write the assembled output to GCS and/or a local directory."""
    import json as _json

    output_blob_path = f"{OUTPUT_PREFIX}/gt_feedback_{vID}.json"
    output_filename  = f"gt_feedback_{vID}.json"

    wrote_anywhere = False

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        local_out = os.path.join(output_dir, output_filename)
        with open(local_out, "w") as f:
            _json.dump(final_output, f, indent=2)
        log.info(f"Written to local: {local_out}")
        wrote_anywhere = True

    if not no_gcs:
        if bucket is None:
            bucket = get_bucket()
        gcs_write_json(bucket, output_blob_path, final_output)
        wrote_anywhere = True

    if not wrote_anywhere:
        log.warning(
            f"[{vID}] No output written: --no-gcs but --output-dir not set."
        )
