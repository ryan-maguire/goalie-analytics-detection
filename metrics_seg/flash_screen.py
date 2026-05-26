"""Gemini Flash screening stub (Phase 2).

Pro is ~10× more expensive than Flash. For "is there ANY shot-like
event in this clip?", Flash is plenty. This module sketches the
two-stage screening path:

    video clip
       │
       ▼
    Flash screener  ──→ shots_any=False, goal_likely=False
       │                    │
       │                    └─→ skip Pro call, return zeros
       │
       └─→ shots_any=True or goal_likely=True
            │
            ▼
        Pro v13 detailed analysis

Full integration deferred to a follow-on PR — Flash availability
in Vertex region needs verification, and a screening prompt needs
probe-driven design. This stub provides the interface so the wiring
code can be drafted now and switched on later.

Feature-flag: --flash-screen (default off).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("metrics_seg.flash_screen")


FLASH_MODEL = "gemini-2.5-flash"

# Stub prompt — real version lives in prompts/screen_flash_v1.txt (TBD).
# Designed to be ~50-100 tokens vs v13's ~3000.
STUB_SCREEN_PROMPT = """You are a hockey video screener. Watch this {duration}s
clip and decide whether ANY shot on goal might have occurred. Be
inclusive — if uncertain, return shots_any=true.

Return JSON only:
{{"shots_any": bool, "goal_likely": bool, "confidence": 0.0-1.0}}
"""


@dataclass
class ScreenResult:
    shots_any:   bool
    goal_likely: bool
    confidence:  float
    raw_response: dict | None = None


def screen_clip(
    video_bytes: bytes,
    duration: int,
    gemini_client,
    *,
    enabled: bool = False,
    fail_safe: bool = True,
) -> ScreenResult:
    """Run the Flash screener.

    If `enabled=False`, returns a fail-safe positive result so the
    caller always escalates to Pro (preserves baseline behavior).

    If `enabled=True`, calls Flash with the stub prompt. On any error
    and `fail_safe=True`, also returns a positive — never skip Pro
    silently on a Flash failure.

    NOTE: this is a stub. The actual Flash invocation is left as a
    TODO; the integration is documented in IMPROVEMENTS_SPEC.md §7.
    """
    if not enabled:
        # Fail-safe positive — caller proceeds to Pro v13 unconditionally
        return ScreenResult(shots_any=True, goal_likely=True,
                              confidence=0.0, raw_response=None)

    # TODO(Phase 2): real Flash call
    #   - Build prompt from STUB_SCREEN_PROMPT.format(duration=duration)
    #   - Wrap video_bytes in types.Part.from_bytes(...)
    #   - client.models.generate_content(model=FLASH_MODEL, ...)
    #   - Parse JSON response
    #   - Handle MAX_TOKENS / safety blocks / truncations
    log.warning("flash_screen.screen_clip called with enabled=True but "
                 "Phase 2 integration not yet implemented; returning "
                 "fail-safe positive")
    if fail_safe:
        return ScreenResult(True, True, 0.0, None)
    return ScreenResult(False, False, 0.0, None)


def should_skip_pro(result: ScreenResult,
                     min_confidence: float = 0.70) -> bool:
    """Decide whether to skip Pro v13 based on the Flash result.
    Skip only when Flash is confidently negative on both axes."""
    return (not result.shots_any
             and not result.goal_likely
             and result.confidence >= min_confidence)
