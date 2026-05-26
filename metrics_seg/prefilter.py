"""Per-second prob pre-filter for cv_seg windows.

Many cv_seg threat windows contain no shots (camera pans, scrums,
dump-ins). Each currently costs a ~$0.05 Gemini call to confirm
"nothing happened." This module loads our pre-computed YOLO+audio
fused per-second shot probabilities for the vID and lets the
pipeline skip Gemini for windows where the peak prob is below a
threshold.

Recall@10s of the underlying fused probs is 86% (validated on the
6-game held-out test set), so threshold=0.30 retains the vast
majority of true positives while killing the easy false positives.

Default: disabled (threshold=0). Enable with --prefilter-threshold.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("metrics_seg.prefilter")


def _load_probs_tsv(tsv_path: Path) -> np.ndarray:
    """Per-second TSV → np.ndarray indexed by second."""
    if not tsv_path.exists():
        return np.zeros(0, dtype=np.float32)
    rows: list[tuple[int, float]] = []
    with open(tsv_path) as f:
        f.readline()                                       # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                rows.append((int(float(parts[0])), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        return np.zeros(0, dtype=np.float32)
    T = max(t for t, _ in rows) + 1
    arr = np.zeros(T, dtype=np.float32)
    for t, p in rows:
        arr[t] = p
    return arr


@dataclass
class FusedProbs:
    """Per-second fused YOLO+audio shot probability for one vID."""
    vid: str
    probs: np.ndarray       # length T (seconds), values in [0,1]
    yolo_dir: Optional[Path]
    audio_dir: Optional[Path]
    weight_yolo: float
    weight_audio: float


def load_fused_probs(
    vid: str,
    yolo_probs_dir: Optional[Path],
    audio_probs_dir: Optional[Path],
    weight_yolo: float = 0.5,
    weight_audio: float = 0.5,
) -> FusedProbs:
    """50/50 (configurable) weighted average of YOLO + audio per-second
    probs. Either side missing → uses only what's available. Both
    missing → returns empty array (which causes every window to be
    kept by `should_skip`)."""
    arrays = []
    weights = []
    if yolo_probs_dir is not None:
        a = _load_probs_tsv(yolo_probs_dir / f"{vid}.tsv")
        if a.size > 0:
            arrays.append(a); weights.append(weight_yolo)
    if audio_probs_dir is not None:
        a = _load_probs_tsv(audio_probs_dir / f"{vid}.tsv")
        if a.size > 0:
            arrays.append(a); weights.append(weight_audio)
    if not arrays:
        log.warning(f"prefilter: no probs found for vID={vid} in "
                     f"{yolo_probs_dir} or {audio_probs_dir} — "
                     f"prefilter will keep every window")
        return FusedProbs(vid, np.zeros(0, dtype=np.float32),
                            yolo_probs_dir, audio_probs_dir,
                            weight_yolo, weight_audio)
    s = sum(weights)
    weights = [w / s for w in weights]
    T = min(len(a) for a in arrays)
    fused = np.zeros(T, dtype=np.float32)
    for a, w in zip(arrays, weights):
        fused += w * a[:T]
    return FusedProbs(vid, fused, yolo_probs_dir, audio_probs_dir,
                        weight_yolo, weight_audio)


def peak_in_window(
    probs: FusedProbs,
    window_start: int,
    window_end: int,
) -> float:
    """Max prob within [window_start, window_end) in seconds."""
    if probs.probs.size == 0:
        return 1.0   # no probs available → treat as "definitely keep"
    lo = max(0, int(window_start))
    hi = min(len(probs.probs), int(window_end) + 1)
    if hi <= lo:
        return 0.0
    return float(probs.probs[lo:hi].max())


def should_skip(
    probs: FusedProbs,
    window_start: int,
    window_end: int,
    threshold: float,
) -> tuple[bool, float]:
    """Return (skip, peak_conf). Skip when peak_conf < threshold AND
    threshold > 0 (a threshold of 0 disables the filter)."""
    if threshold <= 0:
        return False, peak_in_window(probs, window_start, window_end)
    peak = peak_in_window(probs, window_start, window_end)
    return (peak < threshold), peak


def null_metrics_dict(peak_conf: float) -> dict:
    """Return the dict that would normally come from Gemini, with all
    counts zeroed. Schema must match what downstream consumers expect."""
    return {
        "shots":      0,
        "shotsOnNet": 0,
        "saves":      0,
        "goals":      0,
        "shot_timestamps": [],
        # All goal-criteria booleans default false:
        "anchor_puck_crosses_line":               False,
        "anchor_ref_points_at_net":               False,
        "anchor_puck_retrieved_from_net":         False,
        "anchor_attacking_team_skates_to_bench":  False,
        "anchor_scoreboard_change":               False,
        "support_celebration":                    False,
        "support_centre_ice_faceoff":             False,
        "_prefilter_skip":      True,
        "_prefilter_peak_conf": round(peak_conf, 4),
    }
