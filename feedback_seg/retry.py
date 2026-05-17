"""Retry helper with type-based transient classification.

Replaces the v1 string-match approach (`"Connection" in str(e)`) which
both over- and under-retried. Uses google.api_core typed exceptions
plus a fallback string-match for SDK-internal errors that don't surface
as typed exceptions.
"""

import random
import time
from typing import Callable, TypeVar

from .constants import MAX_RETRIES, RETRY_BACKOFF_BASE, RETRY_BACKOFF_CAP
from .logger import log

# google.api_core exposes typed exceptions for the gRPC status codes
# Vertex returns. Falling back to bare strings if api_core is missing
# (it should always be present in a Vertex-AI-using environment, but
# we defend against it anyway because the retry helper is also
# imported during tests where api_core may not be).
try:
    from google.api_core import exceptions as gax
    _TRANSIENT_TYPES: tuple = (
        gax.TooManyRequests,        # 429 / RESOURCE_EXHAUSTED
        gax.ResourceExhausted,
        gax.ServiceUnavailable,     # 503 / UNAVAILABLE
        gax.DeadlineExceeded,       # 504 / DEADLINE_EXCEEDED
        gax.InternalServerError,    # 500 / INTERNAL
        gax.BadGateway,             # 502
        gax.GatewayTimeout,         # 504 (alternate)
        gax.Aborted,                # 409 / ABORTED — usually retryable
        ConnectionError,
        TimeoutError,
        # IO-layer pipe failures inside the Vertex SDK's HTTP/grpc
        # transport. BrokenPipeError is OSError, NOT ConnectionError,
        # so the entry above doesn't catch it. Observed on the n2cy
        # validation run (3/66 windows failed mid-Gemini-call).
        BrokenPipeError,            # errno 32 — pipe closed mid-stream
        ConnectionResetError,       # peer closed TCP — usually transient
        ConnectionAbortedError,     # local stack aborted the connection
    )
except ImportError:
    _TRANSIENT_TYPES = (
        ConnectionError, TimeoutError,
        BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
    )

# Fallback string-match list for SDK-internal errors that don't surface
# as typed exceptions. Kept narrow; only patterns we've observed.
_TRANSIENT_STRING_MATCHES = (
    "SSL: ", "EOF occurred", "Connection reset", "Connection aborted",
    "Read timed out", "Operation timed out", "stream removed",
    "Broken pipe",  # belt-and-braces: some SDKs wrap BrokenPipeError
                    # inside a generic exception with this string
)

T = TypeVar("T")


def _is_transient(e: BaseException) -> bool:
    if isinstance(e, _TRANSIENT_TYPES):
        return True
    msg = str(e)
    return any(p in msg for p in _TRANSIENT_STRING_MATCHES)


def call_with_retry(fn: Callable[..., T], *args, **kwargs) -> T:
    """Call `fn(*args, **kwargs)` with exponential backoff on transient errors.

    Bounded total wait: with defaults (MAX_RETRIES=4, base=30, cap=240)
    the worst-case is ~7-12 minutes per failing call, vs >30 minutes
    in the v1 code.
    """
    last_error: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — broad on purpose, we re-raise
            last_error = e
            if not _is_transient(e) or attempt >= MAX_RETRIES:
                raise
            # exponential ceiling, capped, with full jitter
            cap_for_attempt = min(RETRY_BACKOFF_BASE * (2 ** attempt),
                                  RETRY_BACKOFF_CAP)
            wait = random.uniform(RETRY_BACKOFF_BASE, cap_for_attempt)
            log.warning(
                f"Transient error attempt {attempt}/{MAX_RETRIES} — "
                f"retrying in {wait:.0f}s",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            time.sleep(wait)
    # Unreachable in practice — we either return or re-raise above
    assert last_error is not None
    raise last_error
