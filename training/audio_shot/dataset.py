"""Audio-feature sequences for the audio-only shot model.

Parallels training/temporal_shot/dataset.py but reads the 25-D
per-second feature vectors produced by util/extract_audio_features.py
(rms / spectral / zcr / onset / 13 MFCCs).

Reuses the same TemporalShotDataset shape & semantics so the
training and inference loops are otherwise identical.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eval.eval_cv_seg_output import load_ground_truth_windows  # noqa: E402


# Must match util/extract_audio_features.py HEADER (minus 't')
FEATURE_COLS = [
    "rms_mean", "rms_max", "rms_dyn",
    "sc_mean", "sc_std",
    "sbw_mean", "sbw_std",
    "sro_mean",
    "zcr_mean", "zcr_std",
    "onset_mean", "onset_max",
    *[f"mfcc{k}" for k in range(13)],
]
N_FEATURES = len(FEATURE_COLS)
NAN_SENTINEL = 0.0


def load_features(tsv_path: Path) -> np.ndarray:
    if not tsv_path.exists():
        raise FileNotFoundError(tsv_path)
    rows: list[tuple[int, list[float]]] = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                t = int(float(row["t"]))
            except (KeyError, ValueError):
                continue
            vec = []
            for c in FEATURE_COLS:
                v = row.get(c, "")
                if v in ("", "nan", "None", None):
                    vec.append(NAN_SENTINEL)
                else:
                    try:
                        vec.append(float(v))
                    except ValueError:
                        vec.append(NAN_SENTINEL)
            rows.append((t, vec))
    if not rows:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    T = max(t for t, _ in rows) + 1
    feats = np.zeros((T, N_FEATURES), dtype=np.float32)
    for t, vec in rows:
        feats[t] = vec
    return feats


def derive_labels(
    gt_csv: Path,
    n_seconds: int,
    window_diff: int = 8,
    label_pad: int = 1,
) -> np.ndarray:
    if n_seconds <= 0:
        return np.zeros((0,), dtype=np.float32)
    labels = np.zeros(n_seconds, dtype=np.float32)
    try:
        windows = load_ground_truth_windows(str(gt_csv), window_diff)
    except FileNotFoundError:
        return labels
    for w in windows:
        s = max(0, int(w.start) - label_pad)
        e = min(n_seconds, int(w.end) + label_pad + 1)
        labels[s:e] = 1.0
    return labels


class AudioShotDataset(Dataset):
    def __init__(
        self,
        vids:          list[str],
        features_dir:  Path,
        gt_dir:        Path,
        seq_len:       int = 32,
        stride:        int = 4,
        window_diff:   int = 8,
        label_pad:     int = 1,
        time_window:   tuple[float, float] | None = None,
    ):
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.vid_stats: dict[str, dict] = {}
        for vid in vids:
            feats = load_features(features_dir / f"{vid}.tsv")
            labels = derive_labels(gt_dir / f"gt_{vid}.csv",
                                    len(feats), window_diff, label_pad)
            T = len(feats)
            if time_window is not None:
                lo = int(time_window[0] * T)
                hi = int(time_window[1] * T)
            else:
                lo, hi = 0, T
            n_pos = int(labels[lo:hi].sum())
            n_seq = 0
            for start in range(lo, max(lo + 1, hi - seq_len + 1), stride):
                end = start + seq_len
                if end > hi:
                    break
                self.samples.append((feats[start:end], labels[start:end]))
                n_seq += 1
            self.vid_stats[vid] = {
                "T":      T,
                "window": (lo, hi),
                "pos_s":  n_pos,
                "n_seq":  n_seq,
            }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f, y = self.samples[i]
        return (torch.from_numpy(f).float(),
                torch.from_numpy(y).float())
