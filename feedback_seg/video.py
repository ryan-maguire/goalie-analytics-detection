"""ffmpeg clip extraction with keyframe-aware fallback.

`-c copy` is fast (no re-encode) but requires the seek timestamp to
align with a source keyframe. When it doesn't, ffmpeg silently extends
the clip to the previous keyframe, sometimes by 5-10 seconds. We
detect that with a duration probe and re-encode if needed.
"""

import json
import os
import subprocess
import tempfile
from typing import Optional

from .constants import CLIP_DURATION_TOLERANCE_SEC
from .logger import log


def _ffprobe_duration(path: str) -> Optional[float]:
    """Return the duration in seconds, or None if probing fails."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "json", path],
            check=True, capture_output=True, timeout=30,
        )
        meta = json.loads(result.stdout)
        return float(meta["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _stream_copy(input_path: str, start_sec: float, duration: float,
                 output_path: str) -> None:
    """Fast path: stream-copy. Subject to keyframe alignment."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec),
        "-t",  str(duration),
        "-i",  input_path,
        "-c",  "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, timeout=300)


def _reencode(input_path: str, start_sec: float, duration: float,
              output_path: str) -> None:
    """Slow path: precise seek + re-encode. Frame-accurate."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec),
        "-i",  input_path,
        "-t",  str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-c:a", "aac",
        output_path,
    ]
    subprocess.run(cmd, check=True, timeout=900)


def make_temp_clip_path(suffix: str = ".mp4") -> str:
    """Allocate a unique temp file path. Caller must delete the file."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def extract_clip(input_path: str, start_sec: int, end_sec: int,
                 output_path: str,
                 tolerance_sec: float = CLIP_DURATION_TOLERANCE_SEC) -> None:
    """Extract [start_sec, end_sec) from input_path → output_path.

    Strategy:
      1. Try fast stream-copy.
      2. ffprobe the result. If the duration is more than `tolerance_sec`
         off from requested duration (because the seek hit a non-keyframe
         and ffmpeg silently widened the clip), discard and re-encode.

    Raises:
        ValueError: if duration is non-positive (defends against
            malformed segments — shouldn't happen given upstream
            validation but cheap to check).
        subprocess.CalledProcessError: if ffmpeg itself fails.
        RuntimeError: if both stream-copy and re-encode produce
            unusably-misaligned clips (very unlikely but reported
            cleanly).
    """
    duration = float(end_sec - start_sec)
    if duration <= 0:
        raise ValueError(
            f"Invalid clip duration: {duration}s ({start_sec}s - {end_sec}s). "
            f"Upstream segment is malformed."
        )

    # Fast path
    _stream_copy(input_path, float(start_sec), duration, output_path)
    actual = _ffprobe_duration(output_path)
    if actual is None:
        # ffprobe couldn't read the file — definitely re-encode
        log.warning(
            f"Could not probe stream-copy clip; falling back to re-encode",
            extra={"start": start_sec, "end": end_sec},
        )
    elif abs(actual - duration) <= tolerance_sec:
        return  # success
    else:
        log.info(
            f"Stream-copy clip duration off by {actual - duration:+.1f}s; "
            f"re-encoding for accuracy",
            extra={"start": start_sec, "end": end_sec,
                   "requested": duration, "actual": actual},
        )

    # Slow path
    _reencode(input_path, float(start_sec), duration, output_path)
    actual = _ffprobe_duration(output_path)
    if actual is None:
        raise RuntimeError(
            f"Re-encoded clip is unreadable (start={start_sec}, end={end_sec})"
        )
    if abs(actual - duration) > tolerance_sec:
        raise RuntimeError(
            f"Re-encoded clip duration {actual:.1f}s differs from "
            f"requested {duration:.1f}s by more than {tolerance_sec}s "
            f"(start={start_sec}, end={end_sec})"
        )


def make_clip_id(vID: str, start: int, end: int) -> str:
    """Stable ID for a clip — ties output records to source windows."""
    return f"{vID}_{start}_{end}"
