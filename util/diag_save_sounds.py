"""
Diagnostic: characterise audio signature of save events. For each GT
shot window, extract spectral features and compare to random non-shot
windows of equal duration. The goal is to find a feature (or set of
features) that discriminates "shot was made" from generic game audio.

If a feature shows clear separation, we can build a save-sound detector
on top of cv_seg/audio.py's existing whistle/crowd_roar pattern. If
nothing discriminates, audio is dead and we pivot to YOLO training.

Bands tested (Hz):
    0-200, 200-500, 500-1000, 1000-2000, 2000-4500, 4500-8000

Other features:
    peak/mean energy ratio (impacts spike)
    transient count via librosa onset_detect

Usage:
    python3 util/diag_save_sounds.py \\
        --video data/videos/bfEKgtOIkQU.mp4 \\
        --gt-csv data/ground_truth/gt_2072195.csv
"""

import argparse
import csv
import random
import sys
import statistics
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_seg.io_utils import load_audio_via_ffmpeg_pipe


BANDS = [
    ("0-200",       0,    200),
    ("200-500",     200,  500),
    ("500-1000",    500,  1000),
    ("1000-2000",   1000, 2000),
    ("2000-4500",   2000, 4500),
    ("4500-8000",   4500, 8000),
]


def _band_energy_ratio(stft, freqs, lo, hi):
    """Mean band-energy ratio over the STFT window."""
    mask = (freqs >= lo) & (freqs <= hi)
    band  = stft[mask, :].sum(axis=0)
    total = stft.sum(axis=0) + 1e-8
    return float((band / total).mean())


def _window_features(y, sr, t_start, t_end):
    """Compute spectral + transient features for a single window."""
    import librosa
    i0 = int(t_start * sr)
    i1 = int(t_end * sr)
    if i1 <= i0 or i1 > len(y):
        return None
    seg = y[i0:i1]
    if not np.isfinite(seg).all():
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
    if len(seg) < sr:  # need at least 1s
        return None

    hop = sr // 4
    n_fft = sr
    stft = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    feats = {}
    for name, lo, hi in BANDS:
        feats[f"band_{name}"] = _band_energy_ratio(stft, freqs, lo, hi)

    # Overall RMS and peak/mean
    rms = float(np.sqrt(np.mean(seg ** 2)))
    feats["rms"] = rms
    feats["peak_over_mean"] = float(np.max(np.abs(seg)) / (np.mean(np.abs(seg)) + 1e-8))

    # Transient count via onset detection
    try:
        onset_env = librosa.onset.onset_strength(y=seg, sr=sr, hop_length=hop)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env,
                                            sr=sr, hop_length=hop)
        feats["n_onsets"] = int(len(onsets))
    except Exception:
        feats["n_onsets"] = 0

    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",   required=True)
    ap.add_argument("--gt-csv",  required=True)
    ap.add_argument("--sr",      type=int, default=16000)
    ap.add_argument("--n-nonshot-samples", type=int, default=50,
                    help="random non-shot windows to compare against")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    # Load GT shot windows
    gt_shots = []
    with open(args.gt_csv) as f:
        for row in csv.DictReader(f):
            if row.get("action", "").strip().lower() != "shots":
                continue
            try:
                s = float(row["start"]); e = float(row["end"])
            except (ValueError, KeyError):
                continue
            gt_shots.append((s, e))
    print(f"GT shot windows: {len(gt_shots)}", file=sys.stderr)

    # Load audio
    print(f"Loading audio @ {args.sr} Hz …", file=sys.stderr)
    y, sr = load_audio_via_ffmpeg_pipe(args.video, sr=args.sr)
    if y is None:
        print("ERROR: audio load failed", file=sys.stderr)
        return 2
    duration = len(y) / sr
    print(f"Duration: {duration:.1f}s, samples: {len(y)}", file=sys.stderr)

    # Random non-shot windows (same length distribution as GT shots)
    rng = random.Random(args.seed)
    shot_durs = [e - s for s, e in gt_shots]
    nonshot_windows = []
    attempts = 0
    while len(nonshot_windows) < args.n_nonshot_samples and attempts < 1000:
        attempts += 1
        d = rng.choice(shot_durs) if shot_durs else 12.0
        t0 = rng.uniform(0, max(0, duration - d))
        t1 = t0 + d
        # reject if overlaps any GT shot
        if any(not (t1 <= gs or t0 >= ge) for gs, ge in gt_shots):
            continue
        nonshot_windows.append((t0, t1))
    print(f"Non-shot windows sampled: {len(nonshot_windows)}", file=sys.stderr)

    # Compute features for both groups
    shot_feats    = [f for f in (_window_features(y, sr, s, e) for s, e in gt_shots)    if f]
    nonshot_feats = [f for f in (_window_features(y, sr, s, e) for s, e in nonshot_windows) if f]
    print(f"Shot features computed: {len(shot_feats)}", file=sys.stderr)
    print(f"Non-shot features computed: {len(nonshot_feats)}", file=sys.stderr)

    if not shot_feats or not nonshot_feats:
        print("ERROR: no features", file=sys.stderr)
        return 2

    # Compare distributions per feature
    feat_keys = list(shot_feats[0].keys())
    print("\n=== Feature discrimination (shot vs non-shot) ===")
    print(f"  {'feature':<18}  {'shot_med':>10}  {'nonshot_med':>11}  {'ratio':>6}  "
          f"{'shot_p25':>10}  {'shot_p75':>10}  {'nons_p25':>10}  {'nons_p75':>10}")
    rows = []
    for k in feat_keys:
        sv = sorted(f[k] for f in shot_feats)
        nv = sorted(f[k] for f in nonshot_feats)
        sm = sv[len(sv)//2]
        nm = nv[len(nv)//2]
        ratio = sm / nm if nm > 0 else float('inf')
        sp25 = sv[len(sv)//4]
        sp75 = sv[(3*len(sv))//4]
        np25 = nv[len(nv)//4]
        np75 = nv[(3*len(nv))//4]
        rows.append((abs(ratio - 1.0), k, sm, nm, ratio, sp25, sp75, np25, np75))
    # Sort by absolute deviation from 1.0 (best discriminators first)
    rows.sort(reverse=True)
    for _, k, sm, nm, ratio, sp25, sp75, np25, np75 in rows:
        print(f"  {k:<18}  {sm:>10.4f}  {nm:>11.4f}  {ratio:>6.2f}  "
              f"{sp25:>10.4f}  {sp75:>10.4f}  {np25:>10.4f}  {np75:>10.4f}")

    # Simple discriminability score: for each feature, fraction of shots
    # whose value exceeds the non-shot median (or vice versa for ratio<1)
    print("\n=== Discriminability (above-nonshot-median fraction) ===")
    for k in feat_keys:
        nm = statistics.median(f[k] for f in nonshot_feats)
        above = sum(1 for f in shot_feats if f[k] > nm) / len(shot_feats)
        print(f"  {k:<18}  shot fraction above non-shot median = {above:.2f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
