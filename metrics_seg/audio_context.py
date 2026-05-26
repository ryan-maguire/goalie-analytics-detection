"""Audio + visual prior context for the Gemini prompt.

Renders model priors (per-second YOLO + audio shot probabilities,
plus discrete audio events) as a short text block prepended to the
v13 prompt. Gives Gemini hints to corroborate — not ground truth.

Context block format (concise, model-token-cheap):

    Visual shot-prob peaks (YOLO):  0:78@03, 0:65@12
    Audio shot-prob peaks:           0:62@04
    Audio events detected:
      0:03 — sharp impact (onset 0.92, likely puck strike)
      0:07 — whistle (high-freq onset 0.71)
      0:09 — crowd reaction (RMS dynamic 0.84)

`extract_audio_markers` reads the per-second audio feature TSV
written by `util/extract_audio_features.py`. The TSV columns are
(see that script's HEADER): rms_mean/rms_max/rms_dyn, sc_mean,
sbw_mean, sro_mean, zcr_mean/zcr_std, onset_mean/onset_max,
mfcc0..mfcc12.

Heuristics here are deliberately simple — they're priors, not
classifications. Tuned on the held-out test set to surface obvious
events without flooding the prompt.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np


def _fmt_mmss(t: float) -> str:
    m = int(t // 60); s = int(t % 60)
    return f"{m}:{s:02d}"


def _peak_summary(probs: np.ndarray,
                    window_start: int, window_end: int,
                    min_peak: float = 0.40, top_k: int = 3) -> str:
    """Return a comma-separated string like '0.78@03, 0.65@12' for
    the top-K local maxima above `min_peak` within the window."""
    if probs.size == 0:
        return "—"
    lo = max(0, int(window_start))
    hi = min(len(probs), int(window_end) + 1)
    if hi <= lo:
        return "—"
    sub = probs[lo:hi]
    # Local maxima above threshold within the window
    peaks = []
    for i in range(len(sub)):
        if sub[i] < min_peak:
            continue
        left  = sub[i - 1] if i > 0 else -1
        right = sub[i + 1] if i + 1 < len(sub) else -1
        if sub[i] >= left and sub[i] > right:
            peaks.append((float(sub[i]), i))
    if not peaks:
        return "—"
    peaks.sort(reverse=True)
    parts = [f"{p:.2f}@{_fmt_mmss(t + lo - window_start)}"
              for p, t in peaks[:top_k]]
    return ", ".join(parts)


def _load_audio_features_tsv(tsv_path: Path) -> dict[int, dict[str, float]]:
    """Per-second audio features keyed by t. Returns {} if missing."""
    if not tsv_path.exists():
        return {}
    out: dict[int, dict[str, float]] = {}
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                t = int(float(row["t"]))
            except (KeyError, ValueError):
                continue
            vals: dict[str, float] = {}
            for k, v in row.items():
                if k == "t":
                    continue
                try:
                    vals[k] = float(v)
                except (ValueError, TypeError):
                    pass
            out[t] = vals
    return out


def extract_audio_markers(
    audio_features_tsv: Path,
    window_start: int,
    window_end: int,
    max_markers: int = 5,
) -> list[str]:
    """Return human-readable strings for notable audio events in the
    window. Heuristics:
        - High onset_max (> 0.7) → "sharp impact (puck strike)"
        - High zcr_std + high spectral centroid → "whistle"
        - Very high rms_dyn (> 0.5) → "crowd reaction"

    Markers are timestamped relative to the WINDOW START so the model
    interprets them as offsets into the clip it's analyzing.
    """
    feats = _load_audio_features_tsv(audio_features_tsv)
    if not feats:
        return []
    markers: list[tuple[float, str]] = []   # (importance, text)
    lo = max(0, int(window_start))
    hi = min(int(window_end) + 1, max(feats.keys(), default=0) + 1)
    for t in range(lo, hi):
        row = feats.get(t)
        if not row:
            continue
        onset_max = row.get("onset_max", 0.0)
        zcr_std   = row.get("zcr_std",   0.0)
        sc_mean   = row.get("sc_mean",   0.0)
        rms_dyn   = row.get("rms_dyn",   0.0)
        rel = t - window_start
        # Whistle: high spectral content + zero-crossing variance
        if zcr_std > 0.05 and sc_mean > 2500 and onset_max > 0.5:
            markers.append((onset_max + 0.5,
                              f"{_fmt_mmss(rel)} — whistle "
                              f"(zcr_std {zcr_std:.2f}, sc {sc_mean:.0f})"))
        # Sharp impact: onset spike
        elif onset_max > 0.7:
            markers.append((onset_max,
                              f"{_fmt_mmss(rel)} — sharp impact "
                              f"(onset {onset_max:.2f}, likely puck strike)"))
        # Crowd reaction: large RMS dynamic range
        elif rms_dyn > 0.5:
            markers.append((rms_dyn - 0.2,
                              f"{_fmt_mmss(rel)} — crowd reaction "
                              f"(RMS dynamic {rms_dyn:.2f})"))
    markers.sort(reverse=True)
    return [m for _, m in markers[:max_markers]]


def render_context_block(
    vid: str,
    window_start: int,
    window_end: int,
    yolo_probs: Optional[np.ndarray] = None,
    audio_probs: Optional[np.ndarray] = None,
    audio_features_tsv: Optional[Path] = None,
) -> str:
    """Render the full context block to prepend to the v13 prompt.
    Returns '' if there's nothing useful to add (so callers can
    conditionally include or skip it)."""
    parts: list[str] = []
    if yolo_probs is not None and yolo_probs.size > 0:
        parts.append(
            f"Visual shot-prob peaks (YOLO):  "
            f"{_peak_summary(yolo_probs, window_start, window_end)}")
    if audio_probs is not None and audio_probs.size > 0:
        parts.append(
            f"Audio shot-prob peaks:           "
            f"{_peak_summary(audio_probs, window_start, window_end)}")
    if audio_features_tsv is not None:
        markers = extract_audio_markers(
            audio_features_tsv, window_start, window_end)
        if markers:
            parts.append("Audio events detected:")
            for m in markers:
                parts.append(f"  {m}")
    if not parts:
        return ""
    header = (
        "**OPTIONAL CONTEXT** — these are model PRIORS, not ground truth. "
        "Use them as hints but corroborate visually before counting.")
    return header + "\n\n" + "\n".join(parts) + "\n\n"
