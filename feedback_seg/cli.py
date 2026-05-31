"""Command-line entry point for feedback_seg.

Usage:
    python -m feedback_seg --customer_id CUST000048 --vID U7NUbWad0A8

For local development (no GCS round-trips):
    python -m feedback_seg \\
        --customer_id CUST000048 --vID mjEeE7p2Hz8 \\
        --local-config /path/to/CUST000048.json \\
        --local-video data/videos/full_mjEeE7p2Hz8.mp4 \\
        --local-metrics data/output/metrics_v10/gt_metrics_mjEeE7p2Hz8.json \\
        --output-dir data/output/feedback \\
        --no-gcs
"""

import argparse
import sys

from .constants import COACH_PARALLEL_WORKERS
from .pipeline import process_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m feedback_seg",
        description="Enrich threat-window metrics with Gemini goalie "
                    "coaching feedback. Stage 4 of the goalie analytics "
                    "pipeline (cv_seg → metrics_seg → feedback_seg).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--customer_id", "--customID", "--customer-id",
        dest="customer_id", required=True,
        help="Customer config key, e.g. CUST000048",
    )
    p.add_argument(
        "--vID", required=True,
        help="Video ID to process, e.g. U7NUbWad0A8",
    )
    p.add_argument(
        "--workers", type=int, default=COACH_PARALLEL_WORKERS,
        help=f"Parallel Gemini workers per video (default: {COACH_PARALLEL_WORKERS}).",
    )
    p.add_argument(
        "--local-video", dest="local_video", default=None,
        help="Use a local video file instead of downloading from GCS.",
    )
    p.add_argument(
        "--local-metrics", dest="local_metrics", default=None,
        help="Read metrics from a local JSON file instead of GCS. "
             "Typically the output of metrics_seg.",
    )
    p.add_argument(
        "--local-config", dest="local_config", default=None,
        help="Read customer config from a local JSON file instead of GCS.",
    )
    p.add_argument(
        "--no-gcs", dest="no_gcs", action="store_true",
        help="Do not write output to GCS. Requires --output-dir.",
    )
    p.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Local directory to write output JSON to "
             "(in addition to or instead of GCS).",
    )
    p.add_argument(
        "--progress-stage-idx", type=int, default=None, choices=[1, 2, 3],
        help="When set (typically 3 for feedback_seg), writes "
             "'Processing (X%%)' to the vID's analyticsStatus in the "
             "customer JSON (local + GCS) as each window completes. "
             "Standalone use (no flag) leaves the customer config "
             "untouched. Set by run_pipeline.py.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_gcs and not args.output_dir:
        print(
            "ERROR: --no-gcs requires --output-dir to be set so output "
            "can be written somewhere.",
            file=sys.stderr,
        )
        sys.exit(2)

    ok = process_video(
        customer_id=args.customer_id,
        vID=args.vID,
        workers=args.workers,
        local_video=args.local_video,
        local_metrics=args.local_metrics,
        local_config=args.local_config,
        no_gcs=args.no_gcs,
        output_dir=args.output_dir,
        progress_stage_idx=args.progress_stage_idx,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
