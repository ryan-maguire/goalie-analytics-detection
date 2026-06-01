"""Per-second audio feature TSV per video.

Pipeline:
  1. ffmpeg extracts mono 16 kHz WAV to a temp path
  2. librosa computes a small per-second feature vector
  3. Write a TSV next to data/output/audio_features/<vID>.tsv

Per-second feature columns (29 total, all scalars):
    t                      second
    rms_mean rms_max rms_dyn        loudness
    sc_mean sc_std                  spectral centroid (brightness)
    sbw_mean sbw_std                spectral bandwidth
    sro_mean                        spectral rolloff
    zcr_mean zcr_std                zero-crossing rate (whistles/impacts)
    onset_mean onset_max            onset strength (puck strikes, whistles)
    mfcc0..mfcc12                   13 MFCCs (averaged across the second)

Designed to mirror extract_yolo_features.py's TSV layout (column 0 = t)
so downstream training/audio_shot reuses the temporal-A dataset
pattern without changes apart from feature names + N_FEATURES.

Usage:
    python3 util/extract_audio_features.py \\
        --video data/videos/2069765.mp4 \\
        --out   data/output/audio_features/2069765.tsv
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


SR     = 16000      # audio sample rate after ffmpeg
HOP    = 512        # librosa default
N_MELS = 32         # informs mel spectrum length (not directly written)
N_MFCC = 13


def extract_wav(video: Path, dst: Path) -> bool:
    """Use ffmpeg to extract mono SR-Hz WAV. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-ac", "1",                # mono
        "-ar", str(SR),
        "-vn", "-acodec", "pcm_s16le",
        str(dst),
    ]
    return subprocess.call(cmd) == 0


def per_second_features(wav_path: Path) -> np.ndarray:
    import librosa  # heavy import — keep local
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    if sr != SR:
        raise RuntimeError(f"unexpected sample rate {sr}")
    duration_s = int(len(y) // sr)
    if duration_s <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Compute frame-level features once at HOP, then aggregate per second
    rms       = librosa.feature.rms(y=y, hop_length=HOP)[0]
    sc        = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
    sbw       = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP)[0]
    sro       = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP)[0]
    zcr       = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)  # (13, F)

    fps_frames = sr / HOP                                # frames per second
    rows = []
    for t in range(duration_s):
        a = int(round(t * fps_frames))
        b = int(round((t + 1) * fps_frames))
        if b <= a:
            continue
        sl_r   = rms[a:b]; sl_sc = sc[a:b]; sl_sb = sbw[a:b]
        sl_ro  = sro[a:b]; sl_z  = zcr[a:b]; sl_o  = onset_env[a:b]
        row = [
            float(t),
            float(sl_r.mean()),  float(sl_r.max()),  float(sl_r.max() - sl_r.min()),
            float(sl_sc.mean()), float(sl_sc.std()),
            float(sl_sb.mean()), float(sl_sb.std()),
            float(sl_ro.mean()),
            float(sl_z.mean()),  float(sl_z.std()),
            float(sl_o.mean()),  float(sl_o.max()),
        ]
        for k in range(N_MFCC):
            row.append(float(mfcc[k, a:b].mean()))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


HEADER = ["t",
          "rms_mean", "rms_max", "rms_dyn",
          "sc_mean", "sc_std",
          "sbw_mean", "sbw_std",
          "sro_mean",
          "zcr_mean", "zcr_std",
          "onset_mean", "onset_max",
          *[f"mfcc{k}" for k in range(N_MFCC)]]
N_FEATURES_AUDIO = len(HEADER) - 1   # excludes 't'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out",   required=True, type=Path)
    ap.add_argument("--keep-wav", action="store_true",
                    help="keep the intermediate WAV in --wav-dir")
    ap.add_argument("--wav-dir", type=Path, default=None)
    ap.add_argument("--force",   action="store_true",
                    help="overwrite existing --out")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f"[skip] {args.out} exists", file=sys.stderr)
        return 0
    if not args.video.exists():
        print(f"ERROR: video missing {args.video}", file=sys.stderr)
        return 2

    if args.keep_wav and args.wav_dir is None:
        print("ERROR: --keep-wav requires --wav-dir", file=sys.stderr)
        return 2

    if args.keep_wav:
        wav_path = args.wav_dir / f"{args.video.stem}.wav"
        cleanup = False
    else:
        tmp_dir = Path(tempfile.mkdtemp(prefix="audio_feat_"))
        wav_path = tmp_dir / f"{args.video.stem}.wav"
        cleanup = True

    t0 = time.time()
    print(f"  ffmpeg → {wav_path}", file=sys.stderr)
    # try/finally so the temp WAV dir is removed on EVERY exit path,
    # including an unexpected raise from per_second_features (librosa OOM,
    # bad audio) which previously leaked a full game-audio WAV in /tmp.
    try:
        if not extract_wav(args.video, wav_path):
            print("  ERROR: ffmpeg failed", file=sys.stderr)
            return 3

        print(f"  librosa per-second features…", file=sys.stderr)
        feats = per_second_features(wav_path)
        if feats.size == 0:
            print("  WARN: empty feature matrix", file=sys.stderr)
            return 4

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\t".join(HEADER) + "\n")
            for row in feats:
                f.write("\t".join(f"{v:.4f}" for v in row) + "\n")
        print(f"  wrote {args.out}  ({feats.shape[0]} seconds, "
              f"{feats.shape[1]} cols, {(time.time()-t0):.1f}s)",
              file=sys.stderr)
        return 0
    finally:
        if cleanup:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
