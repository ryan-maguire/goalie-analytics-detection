"""
Audio-based signal detection: whistles and crowd roars.

Both detectors take a pre-loaded waveform and sample rate so the parent
pipeline can read the WAV ONCE and pass it to both detectors instead of
each calling librosa.load() independently.
"""

from typing import Optional

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from . import constants as C
from .logger import log


def detect_whistles(
    y: Optional[np.ndarray],
    sr: int,
    duration_sec: float,
) -> list[float]:
    """
    Detect whistle events in a pre-loaded audio waveform.

    Method: STFT, extract energy in the 2–4.5 kHz band, normalise by
    total energy, z-score normalise (so the threshold is independent of
    recording level), apply minimum duration and refractory filters.
    """
    log.info("  Analysing audio for whistle events...")
    if y is None or len(y) == 0:
        log.warning("  No audio waveform available — skipping whistle detection")
        return []
    if not HAS_LIBROSA:
        log.warning("  librosa not installed — skipping whistle detection")
        return []

    # Defensively scrub NaN/Inf before STFT; a single bad sample propagates
    # through the FFT and silently zeros every detection downstream.
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    hop_length = sr // 4     # 0.25s hop
    n_fft      = sr          # 1s window
    stft       = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs      = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_mask    = (freqs >= C.WHISTLE_FREQ_LOW) & (freqs <= C.WHISTLE_FREQ_HIGH)
    band_energy  = stft[band_mask, :].sum(axis=0)
    total_energy = stft.sum(axis=0) + 1e-8
    ratio = band_energy / total_energy

    mu, sigma = ratio.mean(), ratio.std() + 1e-8
    z = (ratio - mu) / sigma

    above   = z > C.WHISTLE_ENERGY_THRESH
    hop_sec = hop_length / sr

    raw_events: list[tuple[float, float]] = []
    in_whistle = False
    start_t = 0.0
    for i, flag in enumerate(above):
        t = i * hop_sec
        if flag and not in_whistle:
            in_whistle = True
            start_t = t
        elif not flag and in_whistle:
            in_whistle = False
            dur = t - start_t
            if dur >= C.WHISTLE_MIN_DUR_SEC:
                raw_events.append((start_t, t))
    if in_whistle:
        end_t = len(above) * hop_sec
        if (end_t - start_t) >= C.WHISTLE_MIN_DUR_SEC:
            raw_events.append((start_t, end_t))

    # Apply refractory period
    whistles: list[float] = []
    last_accepted = -999.0
    for start, end in raw_events:
        midpoint = start + (end - start) / 2
        if midpoint - last_accepted >= C.WHISTLE_REFRACTORY_SEC:
            whistles.append(midpoint)
            last_accepted = midpoint

    log.info(f"  Detected {len(whistles)} whistle events "
             f"(from {len(raw_events)} raw detections above z>{C.WHISTLE_ENERGY_THRESH})")
    return whistles


def detect_crowd_roar_spikes(
    y: Optional[np.ndarray],
    sr: int,
    duration_sec: float,
) -> list[float]:
    """
    Detect crowd roar spikes — low-frequency energy bursts (20–500 Hz)
    that follow goals. Uses z-score normalisation so the threshold is
    independent of arena volume.
    """
    log.info("  Analysing audio for crowd roar spikes...")
    if y is None or len(y) == 0:
        log.warning("  No audio waveform available — skipping crowd-roar detection")
        return []
    if not HAS_LIBROSA:
        log.warning("  librosa not installed — skipping crowd-roar detection")
        return []

    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    hop_length = sr // 2
    n_fft      = sr
    stft       = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs      = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_mask    = (freqs >= C.CROWD_FREQ_LOW) & (freqs <= C.CROWD_FREQ_HIGH)
    band_energy  = stft[band_mask, :].sum(axis=0)
    total_energy = stft.sum(axis=0) + 1e-8
    ratio = band_energy / total_energy

    mu, sigma = ratio.mean(), ratio.std() + 1e-8
    z = (ratio - mu) / sigma

    above   = z > C.CROWD_ENERGY_THRESH
    hop_sec = hop_length / sr

    raw_events: list[tuple[float, float]] = []
    in_burst, start_t = False, 0.0
    for i, flag in enumerate(above):
        t = i * hop_sec
        if flag and not in_burst:
            in_burst = True
            start_t  = t
        elif not flag and in_burst:
            in_burst = False
            if (t - start_t) >= C.CROWD_MIN_DUR_SEC:
                raw_events.append((start_t, t))
    if in_burst:
        end_t = len(above) * hop_sec
        if (end_t - start_t) >= C.CROWD_MIN_DUR_SEC:
            raw_events.append((start_t, end_t))

    crowd_spikes: list[float] = []
    last_accepted = -999.0
    for start, end in raw_events:
        midpoint = start + (end - start) / 2
        if midpoint - last_accepted >= C.CROWD_REFRACTORY_SEC:
            crowd_spikes.append(midpoint)
            last_accepted = midpoint

    log.info(f"  Detected {len(crowd_spikes)} crowd roar spikes "
             f"(from {len(raw_events)} raw detections above z>{C.CROWD_ENERGY_THRESH})")
    return crowd_spikes
