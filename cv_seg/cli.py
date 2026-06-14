"""
Command-line interface for the CV goalie segment detector.

Usage examples:
    # Run from the repo root (so relative paths resolve)
    python -m cv_seg --vID mjEeE7p2Hz8 --customID CUST000048.json

    # Multiple videos
    python -m cv_seg --vID mjEeE7p2Hz8 Fjc9hmK8_3U --customID CUST000048.json

    # Use local video instead of downloading from GCS
    python -m cv_seg --vID mjEeE7p2Hz8 --customID CUST000048.json \
        --local-video data/full_video/full_mjEeE7p2Hz8.mp4 --no-gcs

    # Tune whistle threshold
    python -m cv_seg --vID Fjc9hmK8_3U --customID CUST000048.json \
        --whistle-thresh 3.5
"""

import argparse
import json
import os
import sys

from . import constants as C
from .io_utils import HAS_GCS, gcs_read_json
from .logger import log
from .pipeline import process_video


def parse_args():
    p = argparse.ArgumentParser(
        description="Pure-CV goalie threat segment detector (no Gemini)"
    )
    p.add_argument("--vID", required=True, nargs="+", metavar="vID",
                   help="One or more video IDs to process")
    p.add_argument("--customID", required=True,
                   help="Customer config JSON blob name in GCS (e.g. CUST000048.json)")
    p.add_argument("--local-video", dest="local_video", default=None,
                   help="Path to a local video file (skips GCS download). "
                        "Only valid when processing a single vID.")
    p.add_argument("--output-dir", dest="output_dir", default="data/outputs",
                   help="Local directory to write output JSONs (default: data/outputs)")
    p.add_argument("--no-gcs", dest="write_gcs", action="store_false",
                   help="Skip writing to GCS (write to --output-dir only)")
    p.add_argument("--progress-stage-idx", type=int, default=None, choices=[1, 2, 3],
                   help="When set (typically 1 for cv_seg as stage 1), writes "
                        "'Processing (33%%)' to the vID's analyticsStatus on "
                        "successful per-vid completion. cv_seg has no clean "
                        "per-step counter so progress is coarse (0%% → 33%% "
                        "per vid). Set by run_pipeline.py.")
    p.add_argument("--no-local", dest="write_local", action="store_false",
                   help="Skip writing to local --output-dir (write to GCS only)")
    p.add_argument("--whistle-thresh", dest="whistle_thresh", type=float, default=None,
                   help=f"Whistle z-score threshold (default: {C.WHISTLE_ENERGY_THRESH}). "
                        "Raise if too many false whistles; lower if real whistles are missed.")
    p.add_argument("--motion-thresh", dest="motion_thresh", type=float, default=None,
                   help=f"Motion magnitude threshold (default: {C.MOTION_THRESH}). "
                        "Raise if too many short motion windows; lower if shots are missed.")
    # --red-light-thresh removed in v23.5 (red_light detector dropped).
    p.add_argument("--max-threat-dur", dest="max_threat_dur", type=int, default=None,
                   help=f"Hard cap on final threat segment length in seconds "
                        f"(default: {C.MAX_THREAT_DUR}). Segments longer than this are "
                        f"split. Lower this if multi-shot blobs are swallowing "
                        f"distinct events; raise it if real sustained-pressure "
                        f"sequences are being chopped up.")
    p.add_argument("--max-open-window", dest="max_open_window", type=int, default=None,
                   help=f"Auto-close cap for an open motion window in seconds "
                        f"(default: {C.MAX_OPEN_WINDOW_SEC}). Distinct from "
                        f"--max-threat-dur — this fires earlier in the pipeline, "
                        f"only on motion-driven windows. Lower this to make the "
                        f"motion source produce shorter raw windows.")
    p.add_argument("--no-net-detection", dest="use_net_detection",
                   action="store_false", default=True,
                   help="Disable HockeyAI YOLOv8 attribution and use motion "
                        "asymmetry only (v23.6.1 baseline behaviour). Default "
                        "is to use net detection. Requires `ultralytics` and "
                        "`huggingface_hub` to be installed; if they're not, "
                        "cv_seg falls back to motion automatically.")
    p.add_argument("--no-target-filter", dest="target_filter",
                   action="store_false", default=True,
                   help="Keep ALL segments (target-threat, opponent-threat, "
                        "and no-threat) in the output. Default is to drop "
                        "everything except segments where threat_goalie_color "
                        "matches the customer's targetGoalieColor. Use this "
                        "flag when debugging attribution issues — you usually "
                        "want the default for production runs since it cuts "
                        "downstream Gemini costs roughly in half.")
    return p.parse_args()


def _load_config(customID: str) -> dict:
    """Load customer config from local file or GCS."""
    config_filename = customID if customID.endswith(".json") else f"{customID}.json"
    gcs_config_path = f"customerID/{config_filename}"
    if os.path.exists(config_filename):
        with open(config_filename) as f:
            config = json.load(f)
        log.info(f"Loaded config from local file: {config_filename}")
        return config
    if HAS_GCS:
        try:
            config = gcs_read_json(C.GCS_BUCKET, gcs_config_path)
            log.info(f"Loaded config from GCS: gs://{C.GCS_BUCKET}/{gcs_config_path}")
            return config
        except Exception as e:
            log.error(f"Failed to load config from GCS gs://{C.GCS_BUCKET}/{gcs_config_path}: {e}")
            sys.exit(1)
    log.error(f"Config file not found locally and GCS unavailable: {config_filename}")
    sys.exit(1)


def _apply_threshold_overrides(args) -> None:
    """
    Apply per-run threshold overrides by mutating the constants module
    so all functions pick up the new values without signature changes.

    CAVEAT: This makes the package non-reentrant. Two pipelines running
    concurrently in the same process (e.g. via a thread pool) will see
    each other's threshold overrides. The CLI runs videos serially so
    this is fine in practice, but if you ever embed process_video() in
    a server, refactor to thread thresholds through the call as a
    Config object instead.
    """
    if args.whistle_thresh is not None:
        C.WHISTLE_ENERGY_THRESH = args.whistle_thresh
        log.info(f"Overriding WHISTLE_ENERGY_THRESH → {C.WHISTLE_ENERGY_THRESH}")
    if args.motion_thresh is not None:
        C.MOTION_THRESH = args.motion_thresh
        log.info(f"Overriding MOTION_THRESH → {C.MOTION_THRESH}")
    # --red-light-thresh removed in v23.5
    if args.max_threat_dur is not None:
        C.MAX_THREAT_DUR = args.max_threat_dur
        log.info(f"Overriding MAX_THREAT_DUR → {C.MAX_THREAT_DUR}")
    if args.max_open_window is not None:
        C.MAX_OPEN_WINDOW_SEC = args.max_open_window
        log.info(f"Overriding MAX_OPEN_WINDOW_SEC → {C.MAX_OPEN_WINDOW_SEC}")


def main():
    args = parse_args()

    if args.local_video and len(args.vID) > 1:
        print("[error] --local-video can only be used with a single --vID", file=sys.stderr)
        sys.exit(1)

    config = _load_config(args.customID)
    output_dir = args.output_dir if args.write_local else None
    _apply_threshold_overrides(args)

    # Pipeline progress: stage 1 is coarse-grained — cv_seg doesn't
    # surface a per-step counter, so each vid jumps 0% → 33% on success.
    # Import lazily so standalone cv_seg (no progress flag) has no extra
    # import cost.
    _pp = None
    if args.progress_stage_idx is not None:
        try:
            from util import progress as _pp
        except ImportError:
            pass

    succeeded, failed = [], []
    for vID in args.vID:
        # Intra-stage-1 progress: map frame-extraction fraction (0..1) into this
        # stage's band so the status climbs instead of sitting at 0% until the
        # stage finishes. Best-effort; no-op when progress isn't wired.
        on_progress = None
        if _pp is not None:
            def on_progress(frac: float, _vID=vID):
                _pp.report(
                    customer_id=args.customID, vid=_vID,
                    stage_idx=args.progress_stage_idx,
                    current=int(frac * 1000), total=1000,
                )
        ok = process_video(
            vID=vID,
            config=config,
            local_video=args.local_video,
            write_gcs=args.write_gcs,
            output_dir=output_dir,
            use_net_detection=args.use_net_detection,
            target_filter=args.target_filter,
            on_progress=on_progress,
        )
        (succeeded if ok else failed).append(vID)
        if _pp is not None and ok:
            _pp.report(
                customer_id=args.customID, vid=vID,
                stage_idx=args.progress_stage_idx,
                current=1, total=1,
            )

    log.info("=" * 60)
    log.info(f"Completed: {len(succeeded)}/{len(args.vID)} videos succeeded.")
    if succeeded:
        log.info(f"  Succeeded: {succeeded}")
    if failed:
        log.warning(f"  Failed: {failed}")
    log.info("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
