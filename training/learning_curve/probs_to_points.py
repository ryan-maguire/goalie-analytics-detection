"""Convert per-second probs TSV → point-event predictions for the
tolerance metric.

Pipeline:
  1. Load per-second shot probabilities
  2. Smooth with a small kernel (3-sec moving average) to reduce noise
  3. Find local maxima above --threshold
  4. Apply NMS: if two peaks within --nms-distance sec, drop the lower-conf one
  5. Emit predictions.csv with one point per shot (start_s = end_s = peak_t)

Output schema matches the existing predictions.csv consumed by both
eval scripts. Each point becomes a degenerate window of duration 0.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_probs(tsv: Path) -> np.ndarray:
    rows = []
    with open(tsv) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            try:
                t = int(float(parts[0])); p = float(parts[1])
            except ValueError:
                continue
            rows.append((t, p))
    if not rows:
        return np.zeros(0, dtype=np.float32)
    T = max(t for t, _ in rows) + 1
    a = np.zeros(T, dtype=np.float32)
    for t, p in rows:
        a[t] = p
    return a


def smooth(probs: np.ndarray, k: int = 3) -> np.ndarray:
    """Centered moving average of length k."""
    if k <= 1 or len(probs) == 0:
        return probs
    pad = k // 2
    padded = np.pad(probs, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(padded, kernel, mode="valid")


def find_peaks(probs: np.ndarray, threshold: float,
                nms_distance: int = 3) -> list[tuple[int, float]]:
    """Return [(t, conf)] of local maxima above threshold, after NMS."""
    if len(probs) == 0:
        return []
    # Local maximum: prob[i] >= prob[i-1] and > prob[i+1]
    cand: list[tuple[float, int]] = []
    for i in range(len(probs)):
        if probs[i] < threshold:
            continue
        left  = probs[i - 1] if i > 0 else -1
        right = probs[i + 1] if i + 1 < len(probs) else -1
        if probs[i] >= left and probs[i] > right:
            cand.append((float(probs[i]), i))
    # NMS: highest conf first, suppress neighbours within nms_distance
    cand.sort(reverse=True)
    kept: list[tuple[int, float]] = []
    suppressed = set()
    for conf, t in cand:
        if t in suppressed:
            continue
        kept.append((t, conf))
        for j in range(max(0, t - nms_distance), t + nms_distance + 1):
            suppressed.add(j)
    kept.sort(key=lambda x: x[0])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs-dir", required=True, type=Path,
                    help="dir of <vID>.tsv per-second probs files")
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--vIDs",      nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--smooth-k",  type=int,   default=3)
    ap.add_argument("--nms-distance", type=int, default=3,
                    help="suppress peaks within this many seconds of a higher peak")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_vids = n_pts = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vID", "start_s", "end_s", "confidence"])
        for vid in args.vIDs:
            tsv = args.probs_dir / f"{vid}.tsv"
            if not tsv.exists():
                print(f"  [skip] no probs for {vid}", file=sys.stderr); continue
            probs = load_probs(tsv)
            smoothed = smooth(probs, args.smooth_k)
            peaks = find_peaks(smoothed, args.threshold, args.nms_distance)
            for t, c in peaks:
                w.writerow([vid, t, t, f"{c:.4f}"])
            n_vids += 1; n_pts += len(peaks)
            print(f"  {vid}: {len(probs)} secs → {len(peaks)} point events",
                  file=sys.stderr)
    print(f"\n{n_pts} points across {n_vids} videos → {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
