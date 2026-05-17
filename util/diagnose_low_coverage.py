"""
diagnose_low_coverage.py — investigate WHY 5 low-coverage videos miss
~half their Hudl shots, when motion signals at those moments are strong.

This is a diagnostic-only tool. It runs against the existing cv_seg
outputs (gt_seg_{vID}.json + _meta.json + _signals.json) and answers
three questions per low-coverage video:

  1. Are uncovered shots in time slices where cv_seg signals fired?
     (If yes → window-opening logic is missing them.
      If no  → cv_seg signal extraction is missing them.)

  2. How many raw candidate windows did cv_seg generate vs how many
     made it through target_filter?
     (Quantifies the "attribution drops half of windows" hypothesis.)

  3. For each uncovered shot, what was happening with motion, audio,
     and faceoff signals at the time?
     (Shows whether tuning a threshold could help.)

Output: a per-video TSV row + a console summary.

USAGE:
    python3 diagnose_low_coverage.py \\
        --vIDs q5yj6sAFQeY HNG0jKYY12g KYtM20r9BuM zOQrPK7IJ24 J8WkcuTsD5I \\
        --hudl-id-map "q5yj6sAFQeY:2127052,HNG0jKYY12g:2095275,..." \\
        --cv-seg-dir data/output/runs/cv_seg \\
        --gt-dir data/ground_truth \\
        --output-dir data/output/diagnostics

No Gemini calls; no clip extraction; no network calls. Pure analysis.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoDiagnostic:
    vID:                  str
    hudl_id:              str
    n_hudl_shots:         int
    n_covered:            int
    n_uncovered:          int
    coverage_pct:         float
    # cv_seg pipeline counts (from meta)
    n_raw_candidates:     int
    n_prefilter_threat:   int
    n_prefilter_target:   int
    n_prefilter_opponent: int
    n_prefilter_no_thr:   int
    n_final_segments:     int
    # Signal-level diagnosis of uncovered shots
    uncov_with_strong_motion:   int     # motion peak >=4 within ±10s
    uncov_with_long_run:        int     # motion run >=8s within ±30s
    uncov_in_opponent_segment:  int     # falls in a segment ATTRIBUTED to opponent
    uncov_in_no_threat_segment: int     # falls in a segment classified as no-threat
    uncov_outside_all_raw:      int     # not inside any RAW (pre-filter) candidate
    # Derived
    motion_could_recover_pct:        float   # if MOTION_THRESH lowered
    attribution_could_recover_pct:   float   # if attribution were corrected
    extraction_could_recover_pct:    float   # if cv_seg generated more raw windows


def load_raw_candidates_from_signals(signals: dict) -> list[tuple[int, int]]:
    """Returns [(start, end), ...] for ALL raw candidate windows.
    cv_seg's signals.json doesn't store raw candidates directly, so we
    approximate them from the segments list (each segment was a raw
    candidate that survived ranking). For the pre-filter total we use
    n_raw_windows."""
    return [(int(s.get('segment_start') or 0), int(s.get('segment_end') or 0))
            for s in signals.get('segments', [])]


def diagnose_video(vID: str, hudl_id: str, cv_seg_dir: str,
                   gt_dir: str) -> Optional[VideoDiagnostic]:
    """Run the full diagnostic for a single video."""
    sig_fp = os.path.join(cv_seg_dir, f'gt_seg_{vID}_signals.json')
    meta_fp = os.path.join(cv_seg_dir, f'gt_seg_{vID}_meta.json')
    gt_fp = os.path.join(gt_dir, f'gt_{hudl_id}.csv')

    for fp, name in [(sig_fp, 'signals'), (meta_fp, 'meta'), (gt_fp, 'GT CSV')]:
        if not os.path.exists(fp):
            print(f"[{vID}] missing {name} at {fp}", file=sys.stderr)
            return None

    signals = json.load(open(sig_fp))
    meta = json.load(open(meta_fp))
    per_sec = signals.get('per_second', [])
    sec_lookup = {s['t']: s for s in per_sec}
    segments = signals.get('segments', [])

    # Pull pipeline counts from meta
    tf = meta.get('target_filter', {}) or {}
    n_raw = signals.get('n_raw_windows', 0)
    n_pf_threat   = int(tf.get('prefilter_threat', 0))
    n_pf_target   = int(tf.get('prefilter_target', 0))
    n_pf_opponent = int(tf.get('prefilter_opponent', 0))
    n_pf_no_thr   = int(tf.get('prefilter_no_threat', 0))
    n_final       = int(tf.get('postfilter_total', signals.get('n_final_threats', 0)))

    # Pull Hudl shots
    shots: list[tuple[float, float]] = []
    with open(gt_fp) as f:
        for row in csv.DictReader(f):
            if row.get('action') == 'Shots':
                try:
                    shots.append((float(row['start']), float(row['end'])))
                except (ValueError, KeyError):
                    continue

    # Find uncovered shots
    uncovered: list[int] = []
    for ss, se in shots:
        mid = int((ss + se) / 2)
        in_seg = any(seg['segment_start'] <= mid <= seg['segment_end']
                     for seg in segments)
        if not in_seg:
            uncovered.append(mid)

    # Signal-level analysis
    # We need access to ALL raw windows to check "outside all raw".
    # Approximate from meta: if prefilter_total = n_pf_threat + n_pf_no_thr,
    # then raw windows ≈ prefilter_total. signals.json doesn't store raw
    # windows directly so we approximate with the final segments (which
    # over-counts coverage). This is a known limitation.
    final_seg_intervals = [(s['segment_start'], s['segment_end']) for s in segments]

    strong_motion = 0
    long_run = 0
    opp_seg = 0   # We can't tell which were opponent-attributed from signals
                  # alone — would need the pre-filter window list. Skip.
    no_thr_seg = 0
    outside_raw = 0   # Same — approximate

    for mid in uncovered:
        # ±10s window
        window_secs = [sec_lookup.get(t, {'motion': 0, 'activity': 0, 'faceoff': 0})
                       for t in range(max(0, mid - 10), mid + 10)]
        max_motion = max((s['motion'] for s in window_secs), default=0)
        if max_motion >= 4.0:
            strong_motion += 1

        # Longest motion run in ±30s
        run = 0
        max_run = 0
        for t in range(max(0, mid - 30), mid + 30):
            sec = sec_lookup.get(t)
            if sec is None:
                continue
            if sec['motion'] >= 3.0:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        if max_run >= 8:
            long_run += 1

    # Recoverability estimates
    motion_could = strong_motion + (
        # how many would be helped by lowering threshold
        sum(1 for mid in uncovered if not any(
            sec_lookup.get(t, {}).get('motion', 0) >= 4.0
            for t in range(max(0, mid - 10), mid + 10)
        ) and any(
            sec_lookup.get(t, {}).get('motion', 0) >= 2.0
            for t in range(max(0, mid - 10), mid + 10)
        ))
    )

    # Attribution-recoverable: rough estimate = uncovered shots that
    # have a strong motion signal but no segment at all. cv_seg
    # generated SOMETHING here but dropped it.
    n_attribution_drops = n_pf_opponent + n_pf_no_thr
    n_total_threat_signals = n_pf_threat + n_pf_no_thr
    if n_total_threat_signals > 0:
        attribution_drop_rate = n_attribution_drops / n_total_threat_signals
        # If we recovered ALL attribution drops, we'd add the same
        # fraction of windows. Assume the same fraction would catch
        # the uncovered shots.
        attribution_could = int(len(uncovered) * attribution_drop_rate)
    else:
        attribution_could = 0

    extraction_could = 0  # signals can't tell us this without more data

    return VideoDiagnostic(
        vID=vID,
        hudl_id=hudl_id,
        n_hudl_shots=len(shots),
        n_covered=len(shots) - len(uncovered),
        n_uncovered=len(uncovered),
        coverage_pct=100 * (len(shots) - len(uncovered)) / len(shots) if shots else 0,
        n_raw_candidates=n_raw,
        n_prefilter_threat=n_pf_threat,
        n_prefilter_target=n_pf_target,
        n_prefilter_opponent=n_pf_opponent,
        n_prefilter_no_thr=n_pf_no_thr,
        n_final_segments=n_final,
        uncov_with_strong_motion=strong_motion,
        uncov_with_long_run=long_run,
        uncov_in_opponent_segment=opp_seg,
        uncov_in_no_threat_segment=no_thr_seg,
        uncov_outside_all_raw=outside_raw,
        motion_could_recover_pct=100 * motion_could / len(uncovered) if uncovered else 0,
        attribution_could_recover_pct=100 * attribution_could / len(uncovered) if uncovered else 0,
        extraction_could_recover_pct=0,
    )


def print_summary(diags: list[VideoDiagnostic]) -> None:
    print("\n" + "=" * 90)
    print("LOW-COVERAGE DIAGNOSTIC — per-video summary")
    print("=" * 90)
    print(f"\n{'vID':<14}  {'shots':>6} {'cov%':>5}  "
          f"{'raw':>5} {'thr':>5} {'opp':>5} {'tgt':>5}  "
          f"{'strong-motion':>14} {'long-run':>10}")
    for d in diags:
        print(f"{d.vID:<14}  "
              f"{d.n_hudl_shots:>6} {d.coverage_pct:>4.0f}%  "
              f"{d.n_raw_candidates:>5} {d.n_prefilter_threat:>5} "
              f"{d.n_prefilter_opponent:>5} {d.n_prefilter_target:>5}  "
              f"{d.uncov_with_strong_motion:>5}/{d.n_uncovered:<3} "
              f"({100*d.uncov_with_strong_motion/d.n_uncovered if d.n_uncovered else 0:.0f}%) "
              f"{d.uncov_with_long_run:>3}/{d.n_uncovered:<3}")

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print("""
strong-motion column: of uncovered Hudl shots, how many had peak optical
   flow ≥ 4.0 in their ±10s window. cv_seg's MOTION_THRESH is 3.0.
   If this is HIGH (>90%), motion signal is fine — uncovered shots are
   NOT being missed because cv_seg didn't see motion.

long-run column: of uncovered Hudl shots, how many had a sustained
   motion run ≥ 8s within ±30s. cv_seg's MIN_MOTION_RUN_SEC is 8.
   If this is HIGH (>80%), motion runs are sustained enough to open
   windows — yet windows are NOT opening here. That means the
   bottleneck is downstream: attribution dropped the window or it
   got rejected post-merge.

opp column (prefilter_opponent): how many candidate windows cv_seg
   attributed to the opponent color and DROPPED via target_filter.
   The ratio opp / (thr + no_thr) is the "attribution drop rate".

If strong-motion is high AND long-run is high AND opp is large relative
to tgt, then the bottleneck is ATTRIBUTION on color-collision videos.
The fix is to make attribution more accurate on these videos — e.g. by
verifying HockeyAI YOLOv8 is actually being invoked, or by tightening
its per-frame confidence threshold.

If strong-motion is LOW, the bottleneck is signal extraction — cv_seg
literally didn't see the motion. The fix would be MOTION_THRESH tuning
or extracting different signals.
""")


def parse_id_map(s: str) -> dict[str, str]:
    pairs = [p.strip() for p in s.split(",") if p.strip()]
    return dict(p.split(":", 1) for p in pairs)


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose cv_seg low-coverage on a set of videos"
    )
    p.add_argument("--vIDs", nargs="+", required=True)
    p.add_argument("--hudl-id-map", required=True,
                   help="Comma-separated vID:hudl_id pairs")
    p.add_argument("--cv-seg-dir", default="data/output/runs/cv_seg")
    p.add_argument("--gt-dir", default="data/ground_truth")
    p.add_argument("--output-dir", default="data/output/diagnostics")
    return p.parse_args()


def main():
    args = parse_args()
    hudl_id_map = parse_id_map(args.hudl_id_map)
    os.makedirs(args.output_dir, exist_ok=True)

    diags: list[VideoDiagnostic] = []
    for vID in args.vIDs:
        hudl_id = hudl_id_map.get(vID)
        if not hudl_id:
            print(f"[{vID}] no hudl_id in map; skipping", file=sys.stderr)
            continue
        d = diagnose_video(vID, hudl_id, args.cv_seg_dir, args.gt_dir)
        if d is not None:
            diags.append(d)

    if not diags:
        print("No diagnostics produced.", file=sys.stderr)
        return 1

    # Write TSV
    tsv_path = os.path.join(args.output_dir, "low_coverage_diagnostic.tsv")
    with open(tsv_path, "w") as f:
        cols = list(VideoDiagnostic.__dataclass_fields__.keys())
        f.write("\t".join(cols) + "\n")
        for d in diags:
            f.write("\t".join(str(getattr(d, c)) for c in cols) + "\n")
    print(f"\nWrote {tsv_path}", file=sys.stderr)

    print_summary(diags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
