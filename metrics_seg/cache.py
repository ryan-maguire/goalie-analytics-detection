"""Content-addressed cache for Gemini responses.

Avoids re-paying for identical requests. Key is sha256 over the
exact bytes/strings that would alter the model output:
    sha256(video_bytes || prompt_text || model_name || temperature)

The cache is purely opt-in and additive — the main pipeline calls
`get(key)` first, falls through to live Gemini on miss, then writes
back with `put(key, response)`.

Defaults to ~/.cache/metrics_seg. Overrideable via env var
METRICS_SEG_CACHE_DIR or constructor arg. Set the dir to "" or pass
`disabled=True` to no-op the cache (live path always used).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "metrics_seg"


def _shard_path(root: Path, key: str) -> Path:
    """2-char shard prefix to keep dirs small (e.g. ab/abcdef...json)."""
    return root / key[:2] / f"{key}.json"


def key_for(
    video_bytes: bytes,
    prompt_text: str,
    model_name: str,
    temperature: float,
) -> str:
    """Stable cache key. Anything that could change the model output
    must be in this hash."""
    h = hashlib.sha256()
    h.update(video_bytes)
    h.update(b"||")
    h.update(prompt_text.encode("utf-8"))
    h.update(b"||")
    h.update(model_name.encode("utf-8"))
    h.update(b"||")
    h.update(f"{temperature:.4f}".encode("utf-8"))
    return h.hexdigest()


class GeminiResponseCache:
    """Disk-backed JSON cache. Thread-safe for concurrent puts.

    No expiration — Gemini responses for identical inputs are
    expected to be (modulo temperature) deterministic, so caching
    indefinitely is correct. Manual cleanup is left to ops.
    """

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        disabled: bool = False,
    ):
        env_dir = os.environ.get("METRICS_SEG_CACHE_DIR")
        if env_dir == "":
            disabled = True
        if disabled:
            self.cache_dir: Optional[Path] = None
            self._disabled = True
            self._lock = threading.Lock()
            return
        if cache_dir is None:
            cache_dir = env_dir or DEFAULT_CACHE_DIR
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._disabled = False
        self._lock = threading.Lock()

    def disabled(self) -> bool:
        return self._disabled

    def get(self, key: str) -> Optional[dict]:
        if self._disabled:
            return None
        path = _shard_path(self.cache_dir, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def put(self, key: str, response: dict) -> None:
        if self._disabled:
            return
        path = _shard_path(self.cache_dir, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp file then rename
        tmp = path.with_suffix(".tmp")
        try:
            with self._lock:
                tmp.write_text(json.dumps(response, separators=(",", ":")))
                os.replace(tmp, path)
        except Exception:
            # Cache failures must never break the pipeline
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    def size(self) -> int:
        """Count cached entries — handy for stats / debugging."""
        if self._disabled or self.cache_dir is None:
            return 0
        return sum(1 for _ in self.cache_dir.glob("**/*.json"))

    def clear(self) -> int:
        """Wipe the cache. Returns count removed."""
        if self._disabled or self.cache_dir is None:
            return 0
        n = 0
        for p in list(self.cache_dir.glob("**/*.json")):
            try:
                p.unlink(); n += 1
            except Exception:
                pass
        return n


# Module-level convenience singleton — most callers want one cache
# per process tied to env config.
_default_cache: Optional[GeminiResponseCache] = None
_default_lock = threading.Lock()


def get_default_cache() -> GeminiResponseCache:
    global _default_cache
    if _default_cache is None:
        with _default_lock:
            if _default_cache is None:
                _default_cache = GeminiResponseCache()
    return _default_cache


def set_default_cache(cache: GeminiResponseCache) -> None:
    global _default_cache
    with _default_lock:
        _default_cache = cache
