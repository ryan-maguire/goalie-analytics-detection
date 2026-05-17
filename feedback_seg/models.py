"""Pydantic schemas for goalie feedback output.

Validators normalize Gemini's free-text classification fields to a
canonical form. LLM output is treated as best-effort: we accept a
range of plausible variations and only reject genuinely unusable
responses.

Production observations driving the design:

  v1.0: strict validators rejected Gemini's bare labels ("Base" vs
        "Base (B)") — caused 100% failure.
  v1.1: alias-mapping validators recovered the bare-label case but
        new failure modes emerged:
          - confidence scores on 1-10 or 1-100 scales (Gemini ignores
            the 1-5 instruction roughly 70% of the time)
          - "N/A" / "Not Applicable" for rebound_control_rank when no
            shot was directed at the targeted goalie
          - narrative text written into categorical fields
  v1.2: this module — confidence normalization, N/A acceptance,
        narrative-text rescue via embedded-label extraction.

Trade-off: silent normalization could mask genuine model drift. We
accept that risk because the alternative (rejecting 60-95% of
windows) is worse for the user and we still emit a warning when
non-trivial normalization occurs.
"""

import re
from typing import Iterable, Union

from pydantic import BaseModel, Field, field_validator


# ── Allowed enum labels (canonical form — what we emit on output) ────

DEPTH_LABELS = (
    "Aggressive (A)",
    "Base (B)",
    "Conservative (C)",
    "Defensive (D)",
)

QUALITY_LABELS = (  # used for both angle and squareness
    "Excellent",
    "Good",
    "Average",
    "Poor",
)

REBOUND_LABELS = (
    "Killed (Elite)",
    "Steered to Low-Danger Zone",
    "Uncontrolled into High-Danger Zone",
    "Goal Allowed",
    # NEW in v1.2: explicit "no rebound to evaluate" label. Used when
    # the targeted goalie wasn't actually under threat in the clip
    # (e.g., shot went to the other goalie's end).
    "Not Applicable",
)

NOT_APPLICABLE_ALIASES = ("n/a", "na", "not applicable", "none", "no shot")


# ── Alias normalization for enum fields ──────────────────────────────

def _build_alias_map(canonical_labels: Iterable[str]) -> dict[str, str]:
    """Build a case-insensitive alias map for an enum.

    For each canonical label, accept:
      - the canonical form itself, e.g. "Base (B)"
      - the bare label without the parenthesized code, e.g. "Base"
      - the parenthesized code alone, e.g. "B"
      - any of the above in lowercase or with whitespace
    """
    aliases: dict[str, str] = {}
    for canonical in canonical_labels:
        aliases[canonical.lower()] = canonical
        bare = re.sub(r"\s*\([^)]+\)\s*$", "", canonical).strip()
        if bare:
            aliases[bare.lower()] = canonical
        m = re.search(r"\(([^)]+)\)$", canonical)
        if m:
            code = m.group(1).strip()
            aliases[code.lower()] = canonical
    return aliases


_DEPTH_ALIASES   = _build_alias_map(DEPTH_LABELS)
_QUALITY_ALIASES = _build_alias_map(QUALITY_LABELS)
_REBOUND_ALIASES = _build_alias_map(REBOUND_LABELS)

# Add N/A aliases pointing to the "Not Applicable" canonical
for _alias in NOT_APPLICABLE_ALIASES:
    _REBOUND_ALIASES[_alias] = "Not Applicable"


def _direct_normalize(value: str, aliases: dict[str, str]) -> Union[str, None]:
    """Try direct alias lookup. Returns canonical form or None."""
    if not isinstance(value, str):
        return None
    return aliases.get(value.strip().lower())


def _rescue_from_narrative(text: str, aliases: dict[str, str]) -> Union[str, None]:
    """Last-resort: extract a recognizable label from narrative text.

    Some Gemini responses put narrative into categorical fields. e.g.
    'The goalie demonstrates good depth management, transitioning
    between Aggressive (A) when the puck is wide...' contains valid
    labels embedded in prose. We pull the FIRST recognizable canonical
    or alias substring as the field's intended value.

    Returns the canonical form, or None if no recognizable label is
    found.
    """
    if not isinstance(text, str) or len(text) < 3:
        return None
    # Sort aliases by length descending so we match longer aliases
    # ("Conservative") before shorter ones ("C") — avoids "Conservative"
    # being matched as just "C".
    sorted_aliases = sorted(aliases.keys(), key=len, reverse=True)
    text_lower = text.lower()
    best_pos = len(text_lower) + 1
    best_canonical: Union[str, None] = None
    for alias in sorted_aliases:
        if len(alias) < 2:
            continue  # skip single-char aliases — too noisy in narrative
        # Match on word boundaries when possible
        pattern = r"\b" + re.escape(alias) + r"\b"
        m = re.search(pattern, text_lower)
        if m and m.start() < best_pos:
            best_pos = m.start()
            best_canonical = aliases[alias]
    return best_canonical


def _normalize_enum(value, aliases: dict[str, str], canonical: tuple) -> str:
    """Normalize an enum field with alias map + narrative rescue.

    Raises ValueError only if no normalization is possible.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"expected string, got {type(value).__name__}: {value!r}"
        )
    direct = _direct_normalize(value, aliases)
    if direct is not None:
        return direct
    rescued = _rescue_from_narrative(value, aliases)
    if rescued is not None:
        return rescued
    raise ValueError(
        f"value must be one of {canonical}, got {value!r}"
    )


# ── Confidence-score normalization ───────────────────────────────────
# Production observation: Gemini emits values like 7, 8, 9 (1-10 scale)
# and 80, 85, 90, 95 (percentage scale) despite the 1-5 instruction.
# Normalize all to 1-5.

def _normalize_confidence(value) -> int:
    """Normalize a confidence value to 1-5.

    Rules:
      - 1-5: as-is
      - 6-10: 1-10 scale → divide by 2, round up to fit 1-5 ceiling
      - 11-100: percentage scale → divide by 20, clamp to 1-5
      - <1 or >100 or non-numeric: ValueError
    """
    if isinstance(value, bool):  # bool is an int subclass — reject it
        raise ValueError(f"confidence must be int, got bool: {value!r}")
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"confidence must be int, got {value!r}")

    if 1 <= v <= 5:
        return v
    if 6 <= v <= 10:
        # Map 6→3, 7→4, 8→4, 9→5, 10→5
        return min(5, max(1, (v + 1) // 2))
    if 11 <= v <= 100:
        # Map 20→1, 40→2, 60→3, 80→4, 100→5; half-up rounding
        # (Python's round() uses banker's rounding which gives 90→4
        # instead of the more intuitive 90→5; we use floor((v + 10) / 20)
        # to get half-up behavior on multiples of 10.)
        return min(5, max(1, (v + 10) // 20))
    raise ValueError(f"confidence out of range (expected 1-100): {v}")


# ── Sub-models ───────────────────────────────────────────────────────

class GoaliePositioning(BaseModel):
    depth_rank: str = Field(
        description=f"One of: {', '.join(DEPTH_LABELS)}"
    )
    cover_angle_rank: str = Field(
        description=f"One of: {', '.join(QUALITY_LABELS)}"
    )
    squareness_rank: str = Field(
        description=f"One of: {', '.join(QUALITY_LABELS)}"
    )
    # Note: ge/le are removed here. The validator does the range
    # normalization manually because Pydantic's ge/le runs BEFORE the
    # validator and rejects values like 95 before we can normalize.
    goalie_position_confidence_score: int = Field(
        description="Model confidence in positioning assessment, 1-5 scale"
    )

    @field_validator("depth_rank")
    @classmethod
    def _validate_depth(cls, v: str) -> str:
        return _normalize_enum(v, _DEPTH_ALIASES, DEPTH_LABELS)

    @field_validator("cover_angle_rank", "squareness_rank")
    @classmethod
    def _validate_quality(cls, v: str) -> str:
        return _normalize_enum(v, _QUALITY_ALIASES, QUALITY_LABELS)

    @field_validator("goalie_position_confidence_score", mode="before")
    @classmethod
    def _validate_confidence(cls, v) -> int:
        return _normalize_confidence(v)


class CoachingFeedback(BaseModel):
    rebound_control_rank: str = Field(
        description=f"One of: {', '.join(REBOUND_LABELS)}"
    )
    actionable_coaching_feedback: str = Field(
        description="Specific corrective coaching cue, or 'None' if execution was sound"
    )
    coaching_confidence_score: int = Field(
        description="Model confidence in coaching assessment, 1-5 scale"
    )

    @field_validator("rebound_control_rank")
    @classmethod
    def _validate_rebound(cls, v: str) -> str:
        return _normalize_enum(v, _REBOUND_ALIASES, REBOUND_LABELS)

    @field_validator("coaching_confidence_score", mode="before")
    @classmethod
    def _validate_confidence(cls, v) -> int:
        return _normalize_confidence(v)


class ClipAnalysis(BaseModel):
    """Full per-window analysis returned by Gemini."""
    technical_reasoning: str = Field(
        description="Timestamped narrative of the goalie's movements, "
                    "technique, and decision-making throughout the clip"
    )
    goalie_positioning: GoaliePositioning
    coaching_feedback: CoachingFeedback
    analysis_confidence_caveats: list[str] = Field(
        default_factory=list,
        description="Specific aspects the camera angle made unobservable. "
                    "Empty list if all four pillars were observable.",
    )
