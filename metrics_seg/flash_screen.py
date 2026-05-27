"""Gemini Flash pre-filter screening (v14 Phase 2 — implemented).

Pro is ~10× more expensive than Flash. For "is there ANY shot-like
event in this clip?", Flash is plenty. Two-stage screening path:

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

Fail-safe: any error in Flash (timeout, malformed JSON, safety block,
MAX_TOKENS truncation) falls through to a positive result so the
caller escalates to Pro. We never drop a window silently because
Flash hiccupped.

Calibration target: ≥95% recall on shot-positive clips. Validated
on data/output/evals/flash_screen_validation.md.

Feature-flag: --flash-screen in metrics_seg (default off until
recall is locked in).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

# The Gemini SDK is imported lazily so this module can be unit-tested
# without google-cloud-* in the env (the tests stub gemini_client).
log = logging.getLogger("metrics_seg.flash_screen")


# Flash model. Both 2.5-flash and 3.5-flash are viable; 3.5-flash needs
# location="global" on Vertex (the parent client should already be
# configured for that location).
FLASH_MODEL = "gemini-2.5-flash"

# Token budget for the screening call. Two-field JSON {shots_any:bool,
# confidence:float} is small, BUT Gemini 2.5 Flash uses "thinking"
# tokens before structured output is emitted — 128 gets consumed by
# the reasoning prefix and finish_reason becomes MAX_TOKENS with no
# JSON. Two compensations: (1) raise the cap to 256 for safety margin,
# (2) disable thinking via ThinkingConfig in screen_clip().
SCREEN_MAX_TOKENS = 256

# Permissive screening prompt — bias toward escalation. False negatives
# cost a window's coaching signal; false positives only cost a Pro call.
SCREEN_PROMPT = """You are screening a {duration}-second hockey video clip.

Decide whether the clip contains ANY shot-like event:
- A puck released toward the goal (saved, scored, missed, or blocked)
- A goalie reacting to a puck (save, deflection, scramble)
- A goal celebration / faceoff at center ice (aftermath signals)

Return TRUE if any of those is present.
Return FALSE only if the clip is clearly:
- Play exclusively in the neutral or defensive zone, no shots attempted
- A pure faceoff with no follow-up shot
- A replay slow-motion of a previous play
- A commercial / station ID / static screen / freeze

WHEN UNCERTAIN, RETURN TRUE. Missing a real shot costs more than an
extra Pro call.

Respond with JSON only:
{{"shots_any": <true|false>, "confidence": <0.0-1.0>}}
"""

SCREEN_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "shots_any":  {"type": "BOOLEAN"},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["shots_any"],
}


@dataclass
class ScreenResult:
    shots_any:    bool
    goal_likely:  bool       # historically separate; we now derive from shots_any
    confidence:   float
    raw_response: dict | None = None
    failed:       bool = False
    fail_reason:  str  = ""


def screen_clip(
    video_bytes: bytes,
    duration: int,
    gemini_client,
    *,
    enabled: bool = False,
    fail_safe: bool = True,
    model: str = FLASH_MODEL,
) -> ScreenResult:
    """Run the Flash screener.

    If `enabled=False`, returns a fail-safe positive so the caller
    always escalates to Pro (preserves baseline behavior — used when
    the --flash-screen feature flag is off).

    If `enabled=True`, calls Flash with the short screening prompt.
    On any error (network, parse, MAX_TOKENS, safety block) and
    `fail_safe=True`, also returns a positive — never skip Pro
    silently on a Flash failure.
    """
    if not enabled:
        return ScreenResult(
            shots_any=True, goal_likely=True, confidence=0.0,
            raw_response=None, failed=False, fail_reason="disabled",
        )

    # Lazy import — keeps this module testable without the SDK.
    try:
        from google.genai import types
    except ImportError:
        log.warning("google-genai SDK not importable; falling back to "
                     "fail-safe positive")
        return ScreenResult(True, True, 0.0, None, True, "sdk_missing")

    try:
        prompt = SCREEN_PROMPT.format(duration=duration)
        response = gemini_client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=SCREEN_MAX_TOKENS,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=SCREEN_RESPONSE_SCHEMA,
                # Disable thinking so the JSON lands within budget
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        raw = (response.text or "").strip()
        if not raw:
            # Empty payload typically means safety block or MAX_TOKENS
            # truncation. Either way, can't trust the result.
            if fail_safe:
                return ScreenResult(True, True, 0.0, None, True, "empty_response")
            return ScreenResult(False, False, 0.0, None, True, "empty_response")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"flash_screen JSON parse failed: {e}; raw={raw[:200]!r}")
            if fail_safe:
                return ScreenResult(True, True, 0.0, None, True,
                                     f"parse_error: {e}")
            return ScreenResult(False, False, 0.0, None, True,
                                 f"parse_error: {e}")

        shots_any  = bool(parsed.get("shots_any", True))    # default True (safe)
        confidence = float(parsed.get("confidence", 0.5))   # default mid-confidence

        # goal_likely is conservative: only flip false when shots_any is
        # also false AND confidence is high. Practically goal_likely
        # rides on shots_any since the prompt's positive cases include
        # goal-aftermath signals.
        goal_likely = shots_any

        return ScreenResult(
            shots_any=shots_any, goal_likely=goal_likely,
            confidence=confidence, raw_response=parsed, failed=False,
        )

    except Exception as e:
        log.warning(f"flash_screen call failed ({type(e).__name__}: {e}); "
                     f"fail_safe={fail_safe}")
        if fail_safe:
            return ScreenResult(True, True, 0.0, None, True,
                                 f"{type(e).__name__}: {e}")
        return ScreenResult(False, False, 0.0, None, True,
                             f"{type(e).__name__}: {e}")


def should_skip_pro(result: ScreenResult,
                     min_confidence: float = 0.70) -> bool:
    """Decide whether to skip Pro v13 based on the Flash result.

    Skip only when:
      - Flash succeeded (didn't fall through fail-safe)
      - Flash said NO shot activity
      - Flash is confident (>= min_confidence)

    Default 0.70 threshold is intentionally cautious — we'd rather pay
    for a Pro call than drop a real shot window. Tune downward only
    after validation shows recall holds.
    """
    if result.failed:
        return False
    return (not result.shots_any
             and not result.goal_likely
             and result.confidence >= min_confidence)


def null_metrics_for_skip(segment_start: int,
                           segment_end: int) -> dict:
    """Build the placeholder metrics dict for a Flash-screened-out
    window. Same shape that Pro would produce on an empty clip, plus
    a marker so the caller / eval can attribute the zero to screening
    rather than to a no-shot finding."""
    return {
        "shots":              0,
        "shotsOnNet":         0,
        "saves":              0,
        "rebounds":           0,
        "goals":              0,
        "observed_goalie_side": "unknown",
        "shot_timestamps":    [],
        "_flash_screen_skip": True,
    }
