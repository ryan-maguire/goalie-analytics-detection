"""Fusion-as-Stage-1 alternative orchestrator.

Replaces cv_seg's motion-based windows with our YOLO+audio
candidate peaks for downstream stages. Output schema matches
cv_seg's `gt_seg_<vID>.json` so metrics_seg + feedback_seg consume
it without changes.

ADDITIVE — does not modify the existing `run_pipeline.py`. Use this
when you want to evaluate metrics_seg on fusion-based windows
instead of cv_seg windows.

Pipeline:
    candidate_list peaks  →  expand ±PRE/POST  →  merge overlaps
       →  write gt_seg_<vID>.json (cv_seg schema)
       →  invoke metrics_seg as a subprocess

Usage:
    python3 tools/run_fusion_pipeline.py \\
        --customer_id CUST_LEARNCURVE \\
        --vID 2073809 \\
        --pre 5 --post 5 \\
        --max-candidates 80

Output:
    Local: <out>/gt_seg_<vID>.json + <out>/gt_metrics_<vID>.json
    (GCS upload is skipped — this is a side-channel pipeline)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO       = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training" / "learning_curve"))

from candidate_list import generate_candidates  # noqa: E402


METRICS_SEG_SCRIPT = REPO / "metrics_seg" / "01_detect_segment_metrics.py"


def expand_and_merge(
    peak_seconds: list[int],
    pre: int,
    post: int,
    max_t: int,
) -> list[tuple[int, int]]:
    """Each peak t → window [t-pre, t+post]; merge overlapping wins.
    Capped at max_t."""
    if not peak_seconds:
        return []
    windows = sorted([(max(0, t - pre), min(max_t, t + post))
                       for t in peak_seconds])
    merged: list[list[int]] = [list(windows[0])]
    for s, e in windows[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def write_seg_json(
    vid: str,
    windows: list[tuple[int, int]],
    out_path: Path,
    threat_color: str = "Unknown",
    threat_side:  str = "unknown",
) -> None:
    """Write gt_seg_<vID>.json in the cv_seg-compatible schema.

    cv_seg writes a FLAT LIST of segment dicts at the top level (not a
    wrapped object). metrics_seg iterates it as `for s in segments`.

    Each segment needs at minimum:
      - segment_start, segment_end (ints)
      - segmentHasThreat (bool)
      - threat_goalie_color (str — drives color attribution in metrics_seg)
      - threat_goalie_side (str — defaults to "unknown" sentinel)
    """
    segments = [
        {
            "segmentHasThreat":     True,
            "threat_goalie_color":  threat_color,
            "threat_goalie_side":   threat_side,
            "segment_start":        int(s),
            "segment_end":          int(e),
            # provenance metadata — ignored by metrics_seg, useful for diagnosis
            "source_signals":       ["fusion_peak"],
            "n_overlapping_raw":    1,
        }
        for s, e in windows
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(segments, indent=2))
    # Sidecar for our own bookkeeping (doesn't break metrics_seg)
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps({
        "version":         "fusion-1",
        "vid":             vid,
        "n_segments":      len(segments),
        "method":          "fusion_yolo_audio_candidate_peaks",
        "processed_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer_id",     required=True)
    ap.add_argument("--vID",             required=True)
    ap.add_argument("--out-dir",         default=str(REPO / "data" / "output" / "runs" / "fusion_pipeline"),
                    type=Path)
    ap.add_argument("--yolo-probs-dir",  type=Path,
                    default=REPO / "runs" / "yolo_curve_n16" / "probs")
    ap.add_argument("--audio-probs-dir", type=Path,
                    default=REPO / "runs" / "audio_curve_n16" / "probs")
    ap.add_argument("--weight-yolo",     type=float, default=0.5)
    ap.add_argument("--weight-audio",    type=float, default=0.5)
    ap.add_argument("--threshold",       type=float, default=0.40)
    ap.add_argument("--nms-distance",    type=int,   default=8)
    ap.add_argument("--max-candidates",  type=int,   default=80)
    ap.add_argument("--pre",             type=int,   default=5,
                    help="seconds before peak in the expanded window")
    ap.add_argument("--post",            type=int,   default=5,
                    help="seconds after peak")
    ap.add_argument("--threat-color",    default=None,
                    help="threat_goalie_color for the seg JSON. If omitted, "
                         "looked up from data/customers/<CUST>.json by vID.")
    ap.add_argument("--skip-metrics",    action="store_true",
                    help="only write the seg JSON, don't run metrics_seg")
    ap.add_argument("--workers",         type=int, default=2,
                    help="metrics_seg parallelism")
    ap.add_argument("--dry-run",         action="store_true",
                    help="print metrics_seg argv instead of executing")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate candidates
    print(f"[1/3] generating candidates for {args.vID}…", file=sys.stderr)
    rows = generate_candidates(
        vid             = args.vID,
        yolo_probs_dir  = args.yolo_probs_dir,
        audio_probs_dir = args.audio_probs_dir,
        weight_yolo     = args.weight_yolo,
        weight_audio    = args.weight_audio,
        threshold       = args.threshold,
        nms_distance    = args.nms_distance,
        max_candidates  = args.max_candidates,
    )
    if not rows:
        print(f"  no candidates above threshold — exiting", file=sys.stderr)
        return 0

    # 2. Expand peaks into windows + merge overlaps. t_seconds may be a
    # float (e.g. 12.7s) — round to the nearest int rather than letting
    # downstream int() truncate so a 12.7s peak gives a window centred
    # on 13s, not 12s.
    peak_secs = [round(float(r["t_seconds"])) for r in rows]
    # Determine video duration from the probs TSV (max t)
    probs_tsv = args.yolo_probs_dir / f"{args.vID}.tsv"
    max_t = 7200       # 2-hour default cap
    if probs_tsv.exists():
        with open(probs_tsv) as f:
            f.readline()
            for line in f:
                try:
                    max_t = max(max_t, int(float(line.split("\t")[0])))
                except (ValueError, IndexError):
                    pass
    windows = expand_and_merge(peak_secs, args.pre, args.post, max_t)
    print(f"  {len(rows)} peaks → {len(windows)} merged windows after "
          f"±{args.pre}/{args.post}s expansion", file=sys.stderr)

    # Look up threat color from customer config if not given
    threat_color = args.threat_color
    if threat_color is None:
        cust_json = REPO / "data" / "customers" / f"{args.customer_id}.json"
        if cust_json.exists():
            for rec in json.loads(cust_json.read_text()):
                if str(rec.get("vID")) == args.vID:
                    threat_color = rec.get("targetGoalieColor") or "Unknown"
                    break
        if threat_color is None:
            threat_color = "Unknown"
            print(f"  WARN: no targetGoalieColor found for {args.vID} in "
                  f"{cust_json} — using 'Unknown'", file=sys.stderr)
        else:
            print(f"  resolved threat_goalie_color = '{threat_color}'", file=sys.stderr)

    # 3. Write the seg JSON
    seg_json_path = args.out_dir / f"gt_seg_{args.vID}.json"
    write_seg_json(args.vID, windows, seg_json_path,
                    threat_color=threat_color)
    print(f"[2/3] wrote {seg_json_path}", file=sys.stderr)

    if args.skip_metrics:
        print(f"[3/3] --skip-metrics → done", file=sys.stderr)
        return 0

    # 4. Invoke metrics_seg pointing at our local JSON
    #
    # metrics_seg's --customID is the customer (CUST...) name; --vID is
    # the video. It expects the seg JSON at the GCS path by default; for
    # local-only pipeline we'd need either:
    #   - upload seg JSON to GCS, or
    #   - patch metrics_seg to accept --local-seg-json
    # The patch is part of the wiring work in 01_detect_segment_metrics.py.
    metrics_argv = [
        sys.executable, str(METRICS_SEG_SCRIPT),
        "--customID", args.customer_id,
        "--vID", args.vID,
        "--workers", str(args.workers),
        "--local-seg-json", str(seg_json_path),       # added by wiring
        "--output-dir", str(args.out_dir),
        "--no-gcs-upload",                            # added by wiring
    ]
    print(f"[3/3] invoking metrics_seg…", file=sys.stderr)
    print(f"  $ {' '.join(metrics_argv)}", file=sys.stderr)
    if args.dry_run:
        print(f"  (--dry-run; not executing)", file=sys.stderr)
        return 0
    rc = subprocess.call(metrics_argv)
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
