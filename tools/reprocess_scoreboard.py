#!/usr/bin/env python3
"""Reprocess a saved scoreboard timeseries.json with updated smoothing
+ event-detection logic. Lets us iterate on the post-OCR pipeline
without spending 20+ minutes re-OCRing the video.

Usage:
    python3 tools/reprocess_scoreboard.py --vID dwGsP6QKDs8 --customer-id CUST000031
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Reuse the production tracker's helpers
import importlib.util
spec = importlib.util.spec_from_file_location("sb", str(REPO / "tools" / "scoreboard_tracker.py"))
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vID", required=True)
    ap.add_argument("--customer-id", required=True)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--min-consecutive", type=int, default=3)
    ap.add_argument("--lookback-pre",  type=int, default=60)
    ap.add_argument("--lookback-post", type=int, default=20)
    args = ap.parse_args()

    base = REPO / "data" / "output" / "scoreboard" / args.vID
    ts_path = base / "timeseries.json"
    if not ts_path.exists():
        print(f"ERROR: no timeseries at {ts_path}", file=sys.stderr); sys.exit(1)

    raw = json.loads(ts_path.read_text())
    # Re-cast OCR output as snapshots; apply MAX_PLAUSIBLE_SCORE filter
    # (necessary because the original OCR run wrote raw values; the
    # post-pass cap from the new parser only applies inside the live
    # capture path).
    snapshots = []
    for r in raw:
        h = r.get("home_score")
        a = r.get("away_score")
        if h is not None and (h < 0 or h > sb.MAX_PLAUSIBLE_SCORE):
            h = None
        if a is not None and (a < 0 or a > sb.MAX_PLAUSIBLE_SCORE):
            a = None
        snapshots.append(sb.OcrSnapshot(
            t_sec=r["t_sec"], home_score=h, away_score=a,
            clock=r.get("clock"), period=r.get("period"),
            raw_tokens=r.get("raw_tokens", []), confidence=r.get("confidence", 0),
        ))

    n_valid = sum(1 for s in snapshots if s.is_valid())
    print(f"loaded {len(snapshots)} snapshots, {n_valid} valid after MAX_SCORE filter",
          file=sys.stderr)

    smoothed = sb.smoothed_scores(snapshots,
                                     window=args.smooth_window,
                                     min_consecutive=args.min_consecutive)
    events = sb.detect_score_changes(smoothed,
                                       lookback_pre=args.lookback_pre,
                                       lookback_post=args.lookback_post)

    # Dump the smoothed series so we can debug
    sm_path = base / "smoothed.json"
    sm_path.write_text(json.dumps([
        {"t": t, "home": h, "away": a} for (t, h, a) in smoothed
    ], indent=2))

    ev_path = base / "goal_events.json"
    ev_path.write_text(json.dumps([asdict(e) for e in events], indent=2))

    rec = sb.goal_events_to_recovery_seg(args.vID, args.customer_id, events)
    rec_path = base / "recovery_seg.json"
    rec_path.write_text(json.dumps(rec, indent=2))

    # Print event summary
    print(f"\n[reprocess summary]", file=sys.stderr)
    print(f"  wrote {sm_path}", file=sys.stderr)
    print(f"  wrote {ev_path}  ({len(events)} events)", file=sys.stderr)
    print(f"  wrote {rec_path}", file=sys.stderr)
    if events:
        for e in events:
            print(f"  goal: side={e.side:5}  {e.score_before}→{e.score_after}  "
                  f"detected t={e.detected_t_sec}s  "
                  f"lookback [{e.lookback_start}-{e.lookback_end}]s",
                  file=sys.stderr)
    else:
        print(f"  no goal events", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
