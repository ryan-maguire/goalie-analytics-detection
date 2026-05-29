"""Produce a ranked list of candidate shot moments for one game.

This is the SHIP-IT entrypoint for the goalie-analytics tool: given a
game (vID), surface ~30-60 candidate timestamps for a coach to review.
Each candidate is one timestamp (mm:ss) plus a confidence score.

Default model = YOLO+audio late fusion (50/50 weighted average of
per-second probabilities). Best Δ=10s recall in our sweep was 85%
with this fusion at threshold=0.40, nms=8.

Outputs:
  - <out>/<vID>_candidates.csv   machine-readable ranked list
  - <out>/<vID>_candidates.md    human-readable for coach review

Usage:
    python3 tools/candidate_list.py --vID 2073809 \\
        --out-dir candidate_output/

    # Custom fusion weights or thresholds:
    python3 tools/candidate_list.py --vID 2073809 \\
        --weight-yolo 0.6 --weight-audio 0.4 \\
        --threshold 0.35 --nms-distance 8 \\
        --max-candidates 80
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))                                  # for `eval.*`
sys.path.insert(0, str(REPO / "training" / "learning_curve"))   # for probs_to_points

from probs_to_points import load_probs, smooth, find_peaks  # noqa: E402


def fmt_mmss(t_seconds: float) -> str:
    m = int(t_seconds // 60)
    s = int(t_seconds % 60)
    return f"{m:02d}:{s:02d}"


def fuse_probs(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    T = min(len(a) for a in arrays)
    out = np.zeros(T, dtype=np.float32)
    for a, w in zip(arrays, weights):
        out += w * a[:T]
    return out


def generate_candidates(
    vid: str,
    yolo_probs_dir: Path,
    audio_probs_dir: Path,
    weight_yolo: float = 0.5,
    weight_audio: float = 0.5,
    threshold: float = 0.40,
    nms_distance: int = 8,
    smooth_k: int = 3,
    max_candidates: Optional[int] = None,
) -> list[dict]:
    """Returns a ranked list of {rank, t_seconds, t_mmss, confidence}."""
    yolo_tsv  = yolo_probs_dir  / f"{vid}.tsv"
    audio_tsv = audio_probs_dir / f"{vid}.tsv"
    arrays = []
    weights = []
    if yolo_tsv.exists():
        arrays.append(load_probs(yolo_tsv)); weights.append(weight_yolo)
    if audio_tsv.exists():
        arrays.append(load_probs(audio_tsv)); weights.append(weight_audio)
    if not arrays:
        raise FileNotFoundError(
            f"No probs found for vID={vid} in {yolo_probs_dir} or {audio_probs_dir}")
    # Renormalise weights to sum to 1
    s = sum(weights)
    weights = [w / s for w in weights]
    fused = fuse_probs(arrays, weights)
    smoothed = smooth(fused, smooth_k)
    peaks = find_peaks(smoothed, threshold, nms_distance)
    # Rank by descending confidence
    peaks.sort(key=lambda x: -x[1])
    if max_candidates is not None:
        peaks = peaks[:max_candidates]
    rows = []
    for rank, (t, conf) in enumerate(peaks, start=1):
        rows.append({
            "rank":       rank,
            "t_seconds":  round(float(t), 1),
            "t_mmss":     fmt_mmss(t),
            "confidence": round(float(conf), 4),
        })
    return rows


def annotate_with_gt(rows: list[dict], gt_dir: Path, vid: str,
                      tolerance_s: float = 5.0,
                      window_diff: int = 8) -> list[dict]:
    """Add 'gt_match' field showing the closest GT shot moment within
    tolerance, or None if no match. Optional — only used when we have GT."""
    from eval.eval_cv_seg_output import load_ground_truth_windows
    gt_path = gt_dir / f"gt_{vid}.csv"
    if not gt_path.exists():
        return rows
    windows = load_ground_truth_windows(str(gt_path), window_diff)
    gt_mids = [(0.5 * (w.start + w.end), w) for w in windows]
    annotated = []
    for r in rows:
        t = r["t_seconds"]
        best_d = float("inf"); best_gt = None
        for gm, gw in gt_mids:
            d = abs(t - gm)
            if d < best_d:
                best_d = d; best_gt = (gm, gw)
        if best_gt is not None and best_d <= tolerance_s:
            r = {**r, "gt_match_t_seconds": int(best_gt[0]),
                 "gt_match_t_mmss":   fmt_mmss(best_gt[0]),
                 "gt_match_delta_s":  round(best_d, 1),
                 "gt_team":           best_gt[1].team or "—"}
        else:
            r = {**r, "gt_match_t_seconds": None,
                 "gt_match_t_mmss":   "—",
                 "gt_match_delta_s":  None,
                 "gt_team":           "—"}
        annotated.append(r)
    return annotated


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("rank,t_seconds,t_mmss,confidence\n")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md(rows: list[dict], path: Path, vid: str, has_gt: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Shot-candidate review — game {vid}",
              "",
              f"**{len(rows)} candidates** ranked by confidence.",
              ""]
    if has_gt:
        n_match  = sum(1 for r in rows if r.get("gt_match_t_seconds") is not None)
        lines += [f"Of these, **{n_match}/{len(rows)} ({n_match/max(1,len(rows))*100:.0f}%)** "
                  f"match a GT shot within ±5 seconds.", ""]
        lines += ["| rank | time | conf | GT match | Δs | team |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            chk = "✓" if r["gt_match_t_seconds"] is not None else "—"
            ds  = f"{r['gt_match_delta_s']:.1f}" if r["gt_match_delta_s"] is not None else "—"
            lines.append(
                f"| {r['rank']} | {r['t_mmss']} | {r['confidence']:.2f} | "
                f"{chk} {r['gt_match_t_mmss']} | {ds} | {r['gt_team']} |")
    else:
        lines += ["| rank | time | conf |",
                  "|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['rank']} | {r['t_mmss']} | {r['confidence']:.2f} |")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vID", required=True,
                    help="match_id of the game to process")
    ap.add_argument("--yolo-probs-dir",  type=Path,
                    default=REPO / "runs" / "yolo_curve_n16" / "probs")
    ap.add_argument("--audio-probs-dir", type=Path,
                    default=REPO / "runs" / "audio_curve_n16" / "probs")
    ap.add_argument("--gt-dir", type=Path,
                    default=REPO / "data" / "ground_truth",
                    help="optional — if provided, marks GT-matched candidates")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--weight-yolo",  type=float, default=0.5)
    ap.add_argument("--weight-audio", type=float, default=0.5)
    ap.add_argument("--threshold",    type=float, default=0.40)
    ap.add_argument("--nms-distance", type=int,   default=8)
    ap.add_argument("--smooth-k",     type=int,   default=3)
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="cap output (default = all peaks above threshold)")
    ap.add_argument("--tolerance-s",  type=float, default=5.0,
                    help="±s for GT annotation")
    args = ap.parse_args()

    rows = generate_candidates(
        vid=args.vID,
        yolo_probs_dir=args.yolo_probs_dir,
        audio_probs_dir=args.audio_probs_dir,
        weight_yolo=args.weight_yolo,
        weight_audio=args.weight_audio,
        threshold=args.threshold,
        nms_distance=args.nms_distance,
        smooth_k=args.smooth_k,
        max_candidates=args.max_candidates,
    )

    has_gt = args.gt_dir is not None and (args.gt_dir / f"gt_{args.vID}.csv").exists()
    if has_gt:
        rows = annotate_with_gt(rows, args.gt_dir, args.vID,
                                  tolerance_s=args.tolerance_s)

    csv_path = args.out_dir / f"{args.vID}_candidates.csv"
    md_path  = args.out_dir / f"{args.vID}_candidates.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path, args.vID, has_gt)
    print(f"  CSV → {csv_path}", file=sys.stderr)
    print(f"  MD  → {md_path}",  file=sys.stderr)
    print(f"  {len(rows)} candidates surfaced", file=sys.stderr)
    if has_gt:
        n_match = sum(1 for r in rows if r.get("gt_match_t_seconds") is not None)
        print(f"  {n_match}/{len(rows)} match GT within ±{args.tolerance_s}s",
              file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
