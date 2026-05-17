"""
IO helpers: GCS, video metadata probing, and audio extraction.

The audio-extraction function is careful to clean up its tempfile on
every error path so a failed extraction never leaves an orphan WAV on
disk.
"""

import json
import os
import subprocess
import tempfile
from typing import Optional

import cv2
import numpy as np

try:
    from google.cloud import storage as gcs_storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

from .logger import log


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _gcs_client():
    if not HAS_GCS:
        raise RuntimeError("google-cloud-storage not installed — cannot use GCS.")
    return gcs_storage.Client()


def gcs_download_to_temp(bucket_name: str, blob_name: str, suffix: str = ".mp4") -> str:
    """Download a GCS object to a temp file and return the path."""
    log.info(f"Downloading gs://{bucket_name}/{blob_name} ...")
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = tmp.name
    blob.download_to_filename(path)
    size_mb = os.path.getsize(path) / 1_048_576
    log.info(f"  Downloaded {size_mb:.1f} MB → {path}")
    return path


def gcs_write_json(bucket_name: str, blob_name: str, data: object) -> None:
    """Write a JSON-serialisable object to GCS."""
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")
    log.info(f"  Wrote gs://{bucket_name}/{blob_name}")


def gcs_read_json(bucket_name: str, blob_name: str) -> object:
    """Read a JSON object from GCS."""
    client = _gcs_client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    return json.loads(blob.download_as_text())


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def get_video_duration(video_path: str) -> float:
    """Return the duration of a video in seconds."""
    _, _, _, duration = probe_video_dims(video_path)
    return duration


def probe_video_dims(video_path: str) -> tuple[int, int, float, float]:
    """
    Return (width, height, native_fps, duration) for the video.
    Used to size raw-pixel chunks read from the ffmpeg pipe.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()
    return w, h, fps, duration


def ffmpeg_available() -> bool:
    """Quick check that ffmpeg is on PATH and runnable."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio_wav(video_path: str) -> str:
    """
    Extract mono 16kHz WAV from the video using ffmpeg. Returns the temp
    file path on success.

    Cleans up its own tempfile on EVERY failure path before re-raising.
    A previous version leaked the tempfile when ffmpeg succeeded but the
    diagnostic later raised RuntimeError.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16 kHz
        "-vn",                # no video
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _safe_unlink(wav_path)
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[:200]}")

    # Audio quality diagnostic. The outer try/finally guarantees we never
    # leak the tempfile, regardless of which step raises (size check,
    # numpy ops on a malformed WAV, etc.). RuntimeError is the signal to
    # re-raise; everything else is downgraded to a warning.
    keep_file = False
    try:
        file_size = os.path.getsize(wav_path)
        if file_size < 1024:
            raise RuntimeError(
                f"Audio extraction produced nearly empty file ({file_size} bytes) — "
                f"video likely has no audio track. ffmpeg stderr: {result.stderr[:200]}"
            )

        # Sample the first 10 seconds for silence (numpy.frombuffer is ~50×
        # faster than struct.unpack on this size). A malformed WAV that
        # makes numpy raise is treated as a non-fatal diagnostic failure.
        try:
            with open(wav_path, "rb") as f:
                f.seek(44)  # skip WAV header
                raw = f.read(16000 * 2 * 10)
            if len(raw) >= 2:
                samples = np.frombuffer(raw[:len(raw) // 2 * 2], dtype=np.int16)
                if samples.size:
                    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    if rms < 10:
                        raise RuntimeError(
                            f"Audio track is silent (RMS={rms:.1f}) — "
                            f"video may have a muted or empty audio stream. "
                            f"Audio signals (whistle, crowd roar) will be unavailable."
                        )
                    log.info(f"  Audio OK: {file_size//1024}KB extracted, "
                             f"opening RMS={rms:.0f} (non-silent ✓)")
        except RuntimeError:
            raise  # silence check — propagate
        except Exception as e:  # diagnostic failure (OSError, numpy errors, ...)
            log.warning(f"  Audio diagnostic skipped (non-fatal): {e}")

        keep_file = True  # only reached if no RuntimeError was raised
    finally:
        if not keep_file:
            _safe_unlink(wav_path)

    return wav_path


def _safe_unlink(path: str) -> None:
    """Unlink a file, swallowing any OSError. Used on cleanup paths."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(wav_path: str, sr: int = 16000) -> tuple[Optional[np.ndarray], int]:
    """
    Load a WAV file into a numpy array. Centralises the librosa.load call
    so detect_whistles / detect_crowd_roar_spikes can share one read of
    the file. On a 60-minute game this saves ~115 MB of redundant I/O.

    Returns (y, sr). On failure returns (None, sr) and logs a warning.
    The audio detectors handle a None waveform by returning [].
    """
    import librosa
    log.info(f"  Loading audio waveform from {wav_path}...")
    try:
        y, sr_out = librosa.load(wav_path, sr=sr, mono=True)
        log.info(f"  Audio loaded: {len(y)/sr_out:.0f}s at {sr_out}Hz")
        return y, sr_out
    except Exception as e:
        log.warning(f"  Audio load failed: {e} — audio detectors will be skipped")
        return None, sr


def load_audio_via_ffmpeg_pipe(
    video_path: str,
    sr: int = 16000,
) -> tuple[Optional[np.ndarray], int]:
    """
    Decode a video's audio track straight into a numpy float32 array,
    skipping the WAV-on-disk roundtrip. Roughly equivalent to calling
    extract_audio_wav() + load_audio(), but with no temp file and no
    second decode pass.

    On any ffmpeg or decode failure, returns (None, sr). Callers should
    fall back to extract_audio_wav() + load_audio().
    """
    if not ffmpeg_available():
        return None, sr

    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", video_path,
        "-ac", "1",                # mono
        "-ar", str(sr),            # target sample rate
        "-vn",                     # no video
        "-f", "s16le",             # signed 16-bit little-endian PCM
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning(f"  ffmpeg audio pipe failed to launch: {e}")
        return None, sr

    if proc.returncode != 0:
        stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-200:]
        log.warning(f"  ffmpeg audio pipe failed (rc={proc.returncode}): {stderr_tail}")
        return None, sr

    raw = proc.stdout
    if len(raw) < 2:
        log.warning("  ffmpeg audio pipe returned no samples")
        return None, sr

    # Trim to an even byte count, decode, normalise to float32 in [-1, 1].
    samples = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16)
    if samples.size == 0:
        return None, sr
    y = samples.astype(np.float32) / 32768.0
    if np.isnan(y).any():
        y = np.nan_to_num(y, copy=False)
    log.info(f"  Audio loaded via ffmpeg pipe: {len(y)/sr:.0f}s at {sr}Hz")
    return y, sr
