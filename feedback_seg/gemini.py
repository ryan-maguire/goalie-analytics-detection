"""Gemini analysis: per-window goalie feedback + game-level summary.

Two public functions:
  - analyze_window(): one threat-window clip → coaching analysis
  - generate_summary(): all per-window results → game-level patterns

Both use Vertex AI auth (matching metrics_seg) and structured response
schemas. Per-window analysis uses temperature=0 (deterministic
classification); summary also uses temperature=0 for run-to-run
reproducibility.
"""

import json
import os
import re
from typing import Optional

from google import genai
from google.genai import types

from .constants import (
    GEMINI_MODEL, INLINE_BYTES_MAX_SIZE, MAX_OUTPUT_TOKENS,
)
from .logger import log
from .models import ClipAnalysis
from .retry import call_with_retry


# ── Prompt content ───────────────────────────────────────────────────

BASE_GOALIE_INSTRUCTIONS = (
    "Act as a Professional Goaltending Scout and Technical Coach. "
    "Analyze the video provided using the following four-pillar framework. "
    "IMPORTANT: Provide specific timestamps (e.g., [00:02]) for every movement "
    "and save identified.\n\n"

    "### TIMESTAMP-GROUNDING REQUIREMENT\n"
    "Each technique claim must include both (a) an [MM:SS] timestamp and "
    "(b) a one-sentence visual description of what made you call it that "
    "technique. If you cannot supply both, omit the claim.\n\n"

    "### 1. CREASE MOVEMENT & TRANSITIONAL PLAY\n"
    "- Identify footwork: Shuffles, T-Pushes, or C-Cuts.\n"
    "- Assess Edge Control: Look for precise stops vs. over-sliding.\n"
    "- Post Integration: Identify RVH, VH, or Standing Post Lean when play is "
    "near the goal line.\n\n"

    "### 2. POSITIONAL GEOMETRY\n"
    "- Depth Management: Classify as Aggressive (outside crease), Base (on line), "
    "Conservative (in paint), or Defensive (near goal line).\n"
    "- Angle: [Excellent, Good, Average, Poor] — Alignment on the 'Centre Line' "
    "from net to puck.\n"
    "- Squareness: [Excellent, Good, Average, Poor] — Orientation of "
    "shoulders/hips to the puck.\n\n"

    "### 3. SAVE SELECTION & TECHNICAL EXECUTION\n"
    "- Technique: Categorize as Butterfly, Active Blocker/Glove, Pad Extension, "
    "or Lateral Release.\n"
    "- Stability: Look for a 'quiet' upper body vs. collapsing or 'busy' "
    "movements.\n\n"

    "### 4. REBOUND CONTROL & RECOVERY\n"
    "- Puck Direction: Evaluate if the goalie 'killed' the play or steered it to "
    "low-danger zones.\n"
    "- Feedback: Provide corrective feedback ONLY if the rebound was uncontrolled "
    "into the 'slot/house'.\n"
    "- Recovery: Check for eye-lead and rotation toward the new puck location.\n\n"

    "### HONEST OBSERVATIONAL CAVEATS\n"
    "If the camera angle, distance, or framing makes any aspect of the goalie's "
    "play unobservable (e.g., the blocker side is hidden by the net frame, the "
    "puck destination is off-camera), list those specific aspects in the "
    "`analysis_confidence_caveats` field. Do not fabricate observations to fill "
    "the four-pillar framework.\n\n"
)

COACHING_LOGIC_BLOB = (
    "### MOVEMENT DEFINITIONS TO USE:\n"
    "- SHUFFLE: Small, lateral movements for short distances. Goalie stays square "
    "with both skates on the ice, push-and-pull motion with edges.\n"
    "- T-PUSH: Powerful lateral move for longer distances. Lead skate turns toward "
    "target, push off trailing skate, finish by squaring up.\n"
    "- C-CUT: Forward or backward depth adjustment. Goalie carves a 'C' shape with "
    "one skate while maintaining centred balance.\n\n"

    "### EDGE CONTROL & STOPPING CRITERIA:\n"
    "1. PRECISE STOP: Momentum killed instantly; goalie is 'set' and stationary.\n"
    "2. CONTROLLED GLIDE: Smooth finish with minimal drift; stays on angle.\n"
    "3. OVER-SLIDING: Drifts past the puck's angle, losing short-side or causing "
    "a recovery scramble.\n"
    "4. UNDER-SLIDE: Stopping short of the post or angle, leaving net exposed.\n\n"

    "### POST INTEGRATION DEFINITIONS:\n"
    "- RVH: Post-side leg horizontal/flat on ice; lead pad seals post; goalie "
    "leans into the pipe.\n"
    "- VH: Post-side leg vertical (standing); back leg horizontal (butterfly).\n"
    "- STANDING POST LEAN: Goalie on feet; arm and shoulder flush against post; "
    "no ice seal.\n\n"

    "### POSITIONAL GEOMETRY DEFINITIONS:\n"
    "1. DEPTH MANAGEMENT (must use one of the four exact labels below):\n"
    "   'Aggressive (A)' — Heels entirely outside the blue paint. Used on rushes.\n"
    "   'Base (B)'       — Skates on the crease line. Standard for 80% of in-zone play.\n"
    "   'Conservative (C)' — Centre of the blue paint. Used below the dots or 2-on-1.\n"
    "   'Defensive (D)'  — Toes near the goal line. Used for post-integration or "
    "scrambles.\n\n"
    "2. ANGLE: must use one of 'Excellent', 'Good', 'Average', 'Poor'.\n"
    "   Excellent = perfectly centred / Good = minor shading / Average = one side "
    "exposed / Poor = out of shooting lane.\n"
    "3. SQUARENESS: must use one of 'Excellent', 'Good', 'Average', 'Poor'.\n"
    "   Excellent = shoulders/hips perpendicular to puck / Good = small leaks / "
    "Average = angled away / Poor = facing wrong direction.\n\n"

    "### SAVE SELECTION & TECHNICAL EXECUTION DEFINITIONS:\n"
    "- BUTTERFLY: Both knees drop; pads flare; stick covers five-hole.\n"
    "- ACTIVE BLOCKER/GLOVE: Hands move independently toward puck trajectory.\n"
    "- PAD EXTENSION: Stay in butterfly while extending one leg to reach the corner.\n"
    "- LATERAL RELEASE: Hold standing edge as long as possible before butterfly "
    "slide; prevents over-sliding.\n\n"
    "STABILITY — QUIET UPPER BODY:\n"
    "- QUIET (Elite): Torso upright/still; head tracks puck; shoulders level.\n"
    "- COLLAPSING (Poor): Chest falls forward or goalie sits back on heels.\n"
    "- BUSY/FLAILING (Poor): Excessive arm/shoulder movement; lack of balance.\n\n"

    "### REBOUND CONTROL — must use one of these exact labels:\n"
    "- 'Killed (Elite)':                          Puck absorbed into equipment; play stopped.\n"
    "- 'Steered to Low-Danger Zone':              Rebound directed to corners/boards; "
    "immediate threat neutralised.\n"
    "- 'Uncontrolled into High-Danger Zone':      Puck bounces into the Slot or House; "
    "second-chance threat created.\n"
    "- 'Goal Allowed':                            Rebound or initial shot resulted in a goal.\n"
    "- 'Not Applicable':                          No shot reached the targeted goalie in this clip "
    "(e.g., the threat resolved before reaching the net or the play was at the other end).\n\n"
    "RECOVERY — VISUAL LEAD SEQUENCE:\n"
    "- EYE-LEAD (Elite): Eyes snap to new puck location; shoulders rotate; body "
    "follows.\n"
    "- SCRAMBLING (Poor): Body moves before eyes find the puck; flat-footed; "
    "out of position.\n\n"

    "### CONFIDENCE SCORES — STRICT 1-5 SCALE\n"
    "Both `goalie_position_confidence_score` and `coaching_confidence_score` "
    "MUST be an integer in the set {1, 2, 3, 4, 5}.\n"
    "- 1 = very low confidence (camera distance, occlusion, ambiguous footage)\n"
    "- 2 = low confidence (key signals partially observable)\n"
    "- 3 = moderate confidence (most signals clear)\n"
    "- 4 = high confidence (clear footage, signals unambiguous)\n"
    "- 5 = very high confidence (close camera, all signals clearly visible)\n"
    "Do NOT use values from 0, 6, 7, 8, 9, 10, or any percentage scale (80, 95, etc.). "
    "If you find yourself wanting to express more granularity, default to 4 (high) "
    "rather than going outside 1-5.\n\n"

    "### FIELD-CONFUSION GUARD\n"
    "`depth_rank` MUST be exactly one of the four depth labels — never a quality label "
    "('Good', 'Excellent'), never narrative prose. If the goalie's depth varies across "
    "the clip, pick the MOST REPRESENTATIVE depth (the one held for the longest fraction "
    "of the clip) and put narrative explanations in `technical_reasoning` instead.\n\n"

    "Provide the analysis strictly in JSON format according to the response schema."
)

GOALIE_ANALYSIS_INSTRUCTIONS = BASE_GOALIE_INSTRUCTIONS + "\n\n" + COACHING_LOGIC_BLOB


# ── Vertex response schemas (raw dicts) ──────────────────────────────
# Vertex AI doesn't accept Pydantic models with $defs; raw dicts are
# the workaround. The Pydantic models in models.py are used to validate
# the *response* after Gemini returns it.

VERTEX_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "technical_reasoning": {"type": "string"},
        "goalie_positioning": {
            "type": "object",
            "properties": {
                "depth_rank": {
                    "type": "string",
                    "enum": ["Aggressive (A)", "Base (B)",
                             "Conservative (C)", "Defensive (D)"],
                },
                "cover_angle_rank": {
                    "type": "string",
                    "enum": ["Excellent", "Good", "Average", "Poor"],
                },
                "squareness_rank": {
                    "type": "string",
                    "enum": ["Excellent", "Good", "Average", "Poor"],
                },
                "goalie_position_confidence_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": [
                "depth_rank", "cover_angle_rank",
                "squareness_rank", "goalie_position_confidence_score",
            ],
        },
        "coaching_feedback": {
            "type": "object",
            "properties": {
                "rebound_control_rank": {
                    "type": "string",
                    "enum": ["Killed (Elite)",
                             "Steered to Low-Danger Zone",
                             "Uncontrolled into High-Danger Zone",
                             "Goal Allowed",
                             "Not Applicable"],
                },
                "actionable_coaching_feedback": {"type": "string"},
                "coaching_confidence_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": [
                "rebound_control_rank",
                "actionable_coaching_feedback",
                "coaching_confidence_score",
            ],
        },
        "analysis_confidence_caveats": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["technical_reasoning", "goalie_positioning", "coaching_feedback"],
}

SUMMARY_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "coaches_overall_rating": {"type": "string"},
        "coaches_summary":        {"type": "string"},
    },
    "required": ["coaches_overall_rating", "coaches_summary"],
}


# ── Per-window analysis ──────────────────────────────────────────────

def _build_video_part(clip_path: str, gcs_uri: Optional[str]) -> types.Part:
    """Build the video Part for the Gemini call.

    If the clip is small enough, send it as inline bytes (faster — no
    GCS round-trip). Otherwise use the gs:// URI (caller must have
    already uploaded).
    """
    if gcs_uri is not None:
        return types.Part.from_uri(file_uri=gcs_uri, mime_type="video/mp4")
    with open(clip_path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type="video/mp4")


def should_use_inline_bytes(clip_path: str) -> bool:
    """True if the clip fits within the inline-bytes size limit."""
    try:
        return os.path.getsize(clip_path) <= INLINE_BYTES_MAX_SIZE
    except OSError:
        return False


def analyze_window(
    client: genai.Client,
    clip_path: str,
    gcs_uri: Optional[str],
    segment: dict,
    goalie_color: str,
    opponent_color: str,
) -> dict:
    """Send one threat-window clip to Gemini and return validated coaching JSON.

    `gcs_uri` is required when the clip exceeds INLINE_BYTES_MAX_SIZE;
    if None, the clip is sent inline as bytes from `clip_path`.

    Returns the validated dict (suitable for inclusion in the output
    record) or `{"error": "..."}` on any failure.
    """
    start    = segment["segment_start"]
    end      = segment["segment_end"]
    duration = end - start
    metrics  = segment.get("metrics") or {}

    prompt = (
        f"<clip_context>\n"
        f"  Goalie:    {goalie_color} defending against {opponent_color} attackers\n"
        f"  Duration:  {duration}s\n"
        f"  Game time: {start}s–{end}s in the full video\n"
        f"</clip_context>\n\n"
        f"<tracking_data>\n"
        f"  Shots:        {metrics.get('shots', '?')}\n"
        f"  Shots on net: {metrics.get('shotsOnNet', '?')}\n"
        f"  Saves:        {metrics.get('saves', '?')}\n"
        f"  Rebounds:     {metrics.get('rebounds', '?')}\n"
        f"  Goals:        {metrics.get('goals', '?')}\n"
        f"</tracking_data>\n\n"
        f"Form your own visual assessment of the goalie's technique first by "
        f"watching the clip. Then cross-reference with the tracking_data above. "
        f"If your visual assessment disagrees with any value, note the discrepancy "
        f"in technical_reasoning.\n\n"
        f"Using the four-pillar framework, provide a detailed technical analysis "
        f"with timestamps. Return ONLY the JSON object."
    )

    try:
        video_part = _build_video_part(clip_path, gcs_uri)
        response = call_with_retry(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=[video_part, prompt],
            config=types.GenerateContentConfig(
                system_instruction=GOALIE_ANALYSIS_INSTRUCTIONS,
                response_mime_type="application/json",
                response_schema=VERTEX_RESPONSE_SCHEMA,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=0,
            ),
        )
        raw       = (response.text or "").strip()
        parsed    = json.loads(raw)
        validated = ClipAnalysis(**parsed)
        return validated.model_dump()
    except Exception as e:
        log.warning(
            f"Coaching analysis failed: {e}",
            extra={"segment_start": start, "error_type": type(e).__name__},
        )
        return {"error": str(e)}


# ── Game-level summary ───────────────────────────────────────────────

# Rubric for coaches_overall_rating. Documented in the prompt so Gemini
# computes deterministically rather than guessing a percentage.
SUMMARY_RATING_RUBRIC = (
    "### RATING RUBRIC — compute coaches_overall_rating deterministically\n"
    "Start at 60. Then apply ALL of the following adjustments:\n"
    "  +10 if depth_rank is 'Aggressive (A)' or 'Base (B)' on >= 60% of windows\n"
    "  +10 if cover_angle_rank is 'Excellent' or 'Good' on >= 60% of windows\n"
    "  +10 if squareness_rank is 'Excellent' or 'Good' on >= 60% of windows\n"
    "  +5  if rebound_control_rank is 'Killed (Elite)' or 'Steered to Low-Danger Zone' "
    "on >= 60% of windows\n"
    "  -5  per window where rebound_control_rank == 'Goal Allowed'\n"
    "  -5  if cover_angle_rank or squareness_rank is 'Poor' on >= 30% of windows\n"
    "Clamp to [0, 100]. Output as 'NN%' (e.g. '78%')."
)


def generate_summary(
    client: genai.Client,
    window_records: list[dict],
    goalie_color: str,
    opponent_color: str,
) -> dict:
    """Aggregate per-window analyses into a game-level coaches' summary.

    Returns dict with keys 'coaches_summary' and 'coaches_overall_rating'.
    Failure modes:
      - Gemini returns truncated JSON: regex-fallback extracts what it can
      - Gemini returns un-parseable text: returns a clear error message
      - Vertex call fails entirely: returns the error in the summary text
    """
    feedback_lines = []
    for i, rec in enumerate(window_records):
        cf = rec.get("coaching_feedback", {}) or {}
        gp = rec.get("goalie_positioning", {}) or {}
        m  = rec.get("metrics") or {}
        feedback_lines.append(
            f"  <window id=\"{i+1}\" time=\"{rec.get('segment_start')}s–{rec.get('segment_end')}s\">"
            f"depth={gp.get('depth_rank','?')} angle={gp.get('cover_angle_rank','?')} "
            f"square={gp.get('squareness_rank','?')} rebound={cf.get('rebound_control_rank','?')} "
            f"goals={m.get('goals','?')} sog={m.get('shotsOnNet','?')} "
            f"coaching=\"{cf.get('actionable_coaching_feedback','None')}\""
            f"</window>"
        )

    prompt = (
        f"You are reviewing {len(window_records)} threat window clips for the "
        f"{goalie_color} goalie vs {opponent_color} attackers.\n\n"
        f"<windows>\n" + "\n".join(feedback_lines) + "\n</windows>\n\n"
        f"Using the window data above, identify PATTERNS across ALL windows — "
        f"not a per-window summary. Look for:\n"
        f"  - Technical elements that appear consistently across windows (strengths)\n"
        f"  - The single most recurring weakness or error pattern (work item)\n"
        f"  - Whether performance improved, declined, or stayed flat across the game\n\n"
        f"Write a concise coaches_summary paragraph using four-pillar framework language.\n\n"
        f"{SUMMARY_RATING_RUBRIC}\n\n"
        f"Return ONLY the JSON object."
    )

    try:
        response = call_with_retry(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SUMMARY_RESPONSE_SCHEMA,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=0,  # was 0.3 — pinned for run-to-run reproducibility
            ),
        )
        return parse_summary_response(response.text or "")

    except Exception as e:
        log.warning(f"Summary generation failed: {e}")
        return {
            "coaches_summary":        f"Summary generation failed: {e}",
            "coaches_overall_rating": "N/A",
        }


def parse_summary_response(raw: str) -> dict:
    """Parse the summary response with three fallback levels.

    Extracted as a public function so it can be unit-tested independently
    of Gemini.
    """
    raw = (raw or "").strip()

    # Strip any markdown code fences Gemini wraps around the JSON
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()

    # Happy path — clean JSON parse
    try:
        parsed = json.loads(raw)
        return {
            "coaches_summary":        parsed.get("coaches_summary", ""),
            "coaches_overall_rating": parsed.get("coaches_overall_rating", "N/A"),
        }
    except json.JSONDecodeError:
        pass

    # Truncated-JSON fallback. Tighter regex than v1 — terminates at
    # the closing quote followed by either `,` or `}` so we don't
    # accidentally swallow trailing JSON wrapper text.
    summary_match = re.search(
        r'"coaches_summary"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]',
        raw,
    )
    rating_match = re.search(
        r'"coaches_overall_rating"\s*:\s*"([^"]+)"',
        raw,
    )
    # If summary's closing-quote tighter regex misses (likely because the
    # JSON really is truncated mid-string), fall back to the open-ended
    # form so we still extract what we can.
    if not summary_match:
        summary_match = re.search(
            r'"coaches_summary"\s*:\s*"((?:[^"\\]|\\.)*)',
            raw,
        )

    if summary_match:
        try:
            extracted = (summary_match.group(1)
                                       .encode()
                                       .decode("unicode_escape"))
        except UnicodeDecodeError:
            # If unicode_escape can't decode, use the raw captured text
            extracted = summary_match.group(1)
        log.warning(
            "Summary JSON malformed — extracted fields via regex",
            extra={"raw_len": len(raw),
                   "rating": rating_match.group(1) if rating_match else "N/A"},
        )
        return {
            "coaches_summary":        extracted,
            "coaches_overall_rating": rating_match.group(1) if rating_match else "N/A",
        }

    # Regex also missed — log preview but don't leak raw into the field
    log.error(
        "Summary parsing failed — neither JSON parse nor regex matched",
        extra={"raw_len": len(raw), "raw_preview": raw[:200]},
    )
    return {
        "coaches_summary":        "Summary generation failed: response could not be parsed.",
        "coaches_overall_rating": "N/A",
    }
