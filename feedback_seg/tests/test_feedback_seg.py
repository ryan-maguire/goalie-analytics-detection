"""Tests for feedback_seg.

These tests deliberately avoid real Vertex/GCS calls. The Gemini and
GCS modules are imported but not exercised end-to-end; only the pure
helpers (validators, regex parsing, record assembly, retry
classification) are tested here.
"""

import json
import os
import tempfile
from unittest.mock import patch

import pytest


# ────────────────────────────────────────────────────────────────────
# Pydantic validators (Optimization h)
# ────────────────────────────────────────────────────────────────────

class TestModelValidators:
    def test_valid_clip_analysis(self):
        from feedback_seg.models import ClipAnalysis
        data = {
            "technical_reasoning": "[00:01] Goalie skates to centre.",
            "goalie_positioning": {
                "depth_rank": "Base (B)",
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Killed (Elite)",
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }
        ca = ClipAnalysis(**data)
        assert ca.goalie_positioning.depth_rank == "Base (B)"

    def test_invalid_depth_rank_rejected(self):
        """A genuinely unknown form is rejected. (Note: 'Base' alone IS
        accepted — see TestEnumAliases — so we use a string that
        matches no known canonical, alias, or short-code form.)"""
        from feedback_seg.models import ClipAnalysis
        data = {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": "Mediocre",   # not in any alias map
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Killed (Elite)",
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }
        with pytest.raises(Exception):
            ClipAnalysis(**data)

    def test_invalid_quality_rank_rejected(self):
        from feedback_seg.models import ClipAnalysis
        data = {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": "Base (B)",
                "cover_angle_rank": "Decent",   # not in enum
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Killed (Elite)",
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }
        with pytest.raises(Exception):
            ClipAnalysis(**data)

    def test_invalid_rebound_rank_rejected(self):
        from feedback_seg.models import ClipAnalysis
        data = {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": "Base (B)",
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Mediocre",  # not in enum
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }
        with pytest.raises(Exception):
            ClipAnalysis(**data)

    def test_confidence_score_normalizes_high_value(self):
        """Production observation: Gemini emits values like 8, 95.
        Validator normalizes to 1-5."""
        from feedback_seg.models import ClipAnalysis
        data = {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": "Base (B)",
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 95,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Killed (Elite)",
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 8,
            },
        }
        ca = ClipAnalysis(**data)
        # 95% → 5 (95/20 = 4.75 → round to 5)
        assert ca.goalie_positioning.goalie_position_confidence_score == 5
        # 8/10 → 4
        assert ca.coaching_feedback.coaching_confidence_score == 4

    def test_confidence_score_too_high_rejected(self):
        """Values > 100 are still rejected (genuinely outside any
        plausible confidence scale)."""
        from feedback_seg.models import ClipAnalysis
        data = {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": "Base (B)",
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 999,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Killed (Elite)",
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }
        with pytest.raises(Exception):
            ClipAnalysis(**data)

    def test_caveats_default_empty_list(self):
        """analysis_confidence_caveats is optional, defaults to []."""
        from feedback_seg.models import ClipAnalysis
        data = {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": "Base (B)",
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Killed (Elite)",
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }
        ca = ClipAnalysis(**data)
        assert ca.analysis_confidence_caveats == []

    def test_caveats_populated_when_present(self):
        from feedback_seg.models import ClipAnalysis
        data = {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": "Base (B)",
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Killed (Elite)",
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
            "analysis_confidence_caveats": [
                "Blocker side obscured by net frame",
                "Puck destination off-camera",
            ],
        }
        ca = ClipAnalysis(**data)
        assert len(ca.analysis_confidence_caveats) == 2


# ────────────────────────────────────────────────────────────────────
# Enum alias mapping (production observed: Gemini emits bare labels)
# ────────────────────────────────────────────────────────────────────

class TestEnumAliases:
    """Production v1 run had 66 of 68 windows fail because Gemini emitted
    'Base' / 'Defensive' instead of 'Base (B)' / 'Defensive (D)'. These
    tests verify the alias map normalizes those (and other plausible
    forms) to canonical output."""

    def _wrap_for_depth(self, depth_value: str) -> dict:
        return {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": depth_value,
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Killed (Elite)",
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }

    def _wrap_for_rebound(self, rebound_value: str) -> dict:
        return {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": "Base (B)",
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": rebound_value,
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }

    # — depth aliases —
    def test_depth_bare_label_normalizes(self):
        from feedback_seg.models import ClipAnalysis
        ca = ClipAnalysis(**self._wrap_for_depth("Base"))
        assert ca.goalie_positioning.depth_rank == "Base (B)"

    def test_depth_canonical_passes_through(self):
        from feedback_seg.models import ClipAnalysis
        ca = ClipAnalysis(**self._wrap_for_depth("Aggressive (A)"))
        assert ca.goalie_positioning.depth_rank == "Aggressive (A)"

    def test_depth_short_code_normalizes(self):
        from feedback_seg.models import ClipAnalysis
        ca = ClipAnalysis(**self._wrap_for_depth("D"))
        assert ca.goalie_positioning.depth_rank == "Defensive (D)"

    def test_depth_lowercase_normalizes(self):
        from feedback_seg.models import ClipAnalysis
        ca = ClipAnalysis(**self._wrap_for_depth("base"))
        assert ca.goalie_positioning.depth_rank == "Base (B)"

    def test_depth_with_whitespace_normalizes(self):
        from feedback_seg.models import ClipAnalysis
        ca = ClipAnalysis(**self._wrap_for_depth("  Conservative  "))
        assert ca.goalie_positioning.depth_rank == "Conservative (C)"

    # — quality (angle/squareness) aliases —
    def test_quality_canonical_passes_through(self):
        from feedback_seg.models import ClipAnalysis
        data = self._wrap_for_depth("Base")
        data["goalie_positioning"]["cover_angle_rank"] = "Excellent"
        data["goalie_positioning"]["squareness_rank"] = "Poor"
        ca = ClipAnalysis(**data)
        assert ca.goalie_positioning.cover_angle_rank == "Excellent"
        assert ca.goalie_positioning.squareness_rank == "Poor"

    def test_quality_lowercase_normalizes(self):
        from feedback_seg.models import ClipAnalysis
        data = self._wrap_for_depth("Base")
        data["goalie_positioning"]["cover_angle_rank"] = "good"
        ca = ClipAnalysis(**data)
        assert ca.goalie_positioning.cover_angle_rank == "Good"

    # — rebound aliases —
    def test_rebound_bare_form_normalizes(self):
        from feedback_seg.models import ClipAnalysis
        # Gemini might say "Killed" instead of "Killed (Elite)"
        ca = ClipAnalysis(**self._wrap_for_rebound("Killed"))
        assert ca.coaching_feedback.rebound_control_rank == "Killed (Elite)"

    def test_rebound_canonical_passes_through(self):
        from feedback_seg.models import ClipAnalysis
        ca = ClipAnalysis(**self._wrap_for_rebound("Goal Allowed"))
        assert ca.coaching_feedback.rebound_control_rank == "Goal Allowed"

    def test_rebound_short_code_normalizes(self):
        from feedback_seg.models import ClipAnalysis
        ca = ClipAnalysis(**self._wrap_for_rebound("Elite"))
        assert ca.coaching_feedback.rebound_control_rank == "Killed (Elite)"

    def test_rebound_unknown_label_rejected(self):
        from feedback_seg.models import ClipAnalysis
        with pytest.raises(Exception):
            ClipAnalysis(**self._wrap_for_rebound("Mediocre Rebound"))


# ────────────────────────────────────────────────────────────────────
# Confidence-score normalization (production drift mitigation, v1.2)
# ────────────────────────────────────────────────────────────────────

class TestConfidenceNormalization:
    """Production v1.1 run had 45+ windows fail because Gemini emitted
    confidence values outside the 1-5 range. The validator now
    normalizes 1-100 input to 1-5."""

    @pytest.mark.parametrize("input_value, expected", [
        # Native 1-5 scale: as-is
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        # 1-10 scale: divide by 2 with ceiling
        (6, 3), (7, 4), (8, 4), (9, 5), (10, 5),
        # Percentage scale: divide by 20, round
        (20, 1), (40, 2), (60, 3), (80, 4), (90, 5), (95, 5), (100, 5),
        # Exact production-observed values
        (85, 4),  # 85/20 = 4.25 → 4
        (90, 5),  # 90/20 = 4.5 → 5 (banker's rounding rounds half to even, but Python's round uses HALF_EVEN — verify behavior)
    ])
    def test_normalize(self, input_value, expected):
        from feedback_seg.models import _normalize_confidence
        assert _normalize_confidence(input_value) == expected

    def test_zero_rejected(self):
        from feedback_seg.models import _normalize_confidence
        with pytest.raises(ValueError):
            _normalize_confidence(0)

    def test_negative_rejected(self):
        from feedback_seg.models import _normalize_confidence
        with pytest.raises(ValueError):
            _normalize_confidence(-1)

    def test_over_100_rejected(self):
        from feedback_seg.models import _normalize_confidence
        with pytest.raises(ValueError):
            _normalize_confidence(150)

    def test_non_numeric_rejected(self):
        from feedback_seg.models import _normalize_confidence
        with pytest.raises(ValueError):
            _normalize_confidence("five")

    def test_bool_rejected(self):
        """bool is an int subclass in Python — guard against it."""
        from feedback_seg.models import _normalize_confidence
        with pytest.raises(ValueError):
            _normalize_confidence(True)


# ────────────────────────────────────────────────────────────────────
# N/A rebound rank & narrative rescue (production drift mitigation, v1.2)
# ────────────────────────────────────────────────────────────────────

class TestRebound_NotApplicable:
    """When the targeted goalie isn't actually under threat in the clip,
    Gemini correctly responds with 'N/A' or 'Not Applicable'. We accept
    these as the canonical 'Not Applicable' label."""

    def _wrap(self, rebound_value: str) -> dict:
        return {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": "Base (B)",
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": rebound_value,
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }

    @pytest.mark.parametrize("input_value", [
        "N/A", "n/a", "NA", "Not Applicable", "not applicable", "None",
        "no shot",
    ])
    def test_na_aliases_normalize(self, input_value):
        from feedback_seg.models import ClipAnalysis
        ca = ClipAnalysis(**self._wrap(input_value))
        assert ca.coaching_feedback.rebound_control_rank == "Not Applicable"


class TestNarrativeRescue:
    """When Gemini puts narrative text into a categorical field, try to
    rescue the intended label by extracting the FIRST recognizable
    canonical or alias substring."""

    def _wrap(self, depth_value: str) -> dict:
        return {
            "technical_reasoning": "x",
            "goalie_positioning": {
                "depth_rank": depth_value,
                "cover_angle_rank": "Good",
                "squareness_rank": "Excellent",
                "goalie_position_confidence_score": 4,
            },
            "coaching_feedback": {
                "rebound_control_rank": "Killed (Elite)",
                "actionable_coaching_feedback": "None",
                "coaching_confidence_score": 5,
            },
        }

    def test_narrative_with_aggressive_first(self):
        """Production observation: 'The goalie demonstrates good depth
        management, transitioning between Aggressive (A) when the puck
        is wide, Base (B)...' — should rescue Aggressive (the first
        valid label)."""
        from feedback_seg.models import ClipAnalysis
        narrative = (
            "The goalie demonstrates good depth management, transitioning "
            "between Aggressive (A) when the puck is wide, Base (B) for "
            "high slot play, and Defensive (D) when anticipating shots."
        )
        ca = ClipAnalysis(**self._wrap(narrative))
        assert ca.goalie_positioning.depth_rank == "Aggressive (A)"

    def test_narrative_with_no_recognizable_label_rejected(self):
        from feedback_seg.models import ClipAnalysis
        narrative = "The goalie was kind of all over the place, hard to say."
        with pytest.raises(Exception):
            ClipAnalysis(**self._wrap(narrative))

    def test_quality_label_in_depth_field_does_not_falsely_rescue(self):
        """'Good' is a valid alias for the QUALITY enum but not for the
        DEPTH enum. Rescue should fail clean rather than picking up a
        cross-field match."""
        from feedback_seg.models import ClipAnalysis
        with pytest.raises(Exception):
            ClipAnalysis(**self._wrap("Good"))


# ────────────────────────────────────────────────────────────────────
# Retry classifier (Defect 2)
# ────────────────────────────────────────────────────────────────────

class TestRetryClassifier:
    def test_typed_resource_exhausted_is_transient(self):
        from feedback_seg.retry import _is_transient
        try:
            from google.api_core import exceptions as gax
        except ImportError:
            pytest.skip("google.api_core not available")
        assert _is_transient(gax.ResourceExhausted("quota")) is True

    def test_typed_service_unavailable_is_transient(self):
        from feedback_seg.retry import _is_transient
        try:
            from google.api_core import exceptions as gax
        except ImportError:
            pytest.skip("google.api_core not available")
        assert _is_transient(gax.ServiceUnavailable("unavailable")) is True

    def test_value_error_is_not_transient(self):
        from feedback_seg.retry import _is_transient
        assert _is_transient(ValueError("bad")) is False

    def test_generic_runtime_error_with_innocent_text_is_not_transient(self):
        """The v1 string-match approach would have retried 'Connection
        between video and prompt failed' because 'Connection' is a
        substring. Type-based check rejects it."""
        from feedback_seg.retry import _is_transient
        assert _is_transient(
            RuntimeError("Connection between video and prompt was malformed")
        ) is False

    def test_string_fallback_for_ssl_errors(self):
        from feedback_seg.retry import _is_transient
        assert _is_transient(RuntimeError("SSL: handshake failed")) is True

    # ── v1.2.1: IO-layer pipe failures ─────────────────────────────
    # Observed on n2cy8b755Tg validation run: 3/66 windows failed
    # with [Errno 32] Broken pipe inside client.models.generate_content.
    # BrokenPipeError is OSError, not ConnectionError, so the prior
    # ConnectionError entry didn't catch it.

    def test_broken_pipe_is_transient(self):
        from feedback_seg.retry import _is_transient
        assert _is_transient(BrokenPipeError("[Errno 32] Broken pipe")) is True

    def test_connection_reset_is_transient(self):
        from feedback_seg.retry import _is_transient
        assert _is_transient(ConnectionResetError("peer reset")) is True

    def test_connection_aborted_is_transient(self):
        from feedback_seg.retry import _is_transient
        assert _is_transient(ConnectionAbortedError("local abort")) is True

    def test_string_fallback_for_wrapped_broken_pipe(self):
        """Some SDKs wrap BrokenPipeError inside a generic exception
        without the original __cause__. The string-fallback catches it."""
        from feedback_seg.retry import _is_transient
        assert _is_transient(RuntimeError("write failed: Broken pipe")) is True

    def test_broken_pipe_retries_then_succeeds(self):
        """Real-world behaviour: a transient pipe failure followed by
        a successful response. The retry helper should retry once and
        return the success."""
        from feedback_seg.retry import call_with_retry

        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 2:
                raise BrokenPipeError("[Errno 32] Broken pipe")
            return "ok"

        # Patch backoff to zero so the test doesn't actually sleep
        with patch("feedback_seg.retry.RETRY_BACKOFF_BASE", 0.0), \
             patch("feedback_seg.retry.RETRY_BACKOFF_CAP", 0.0):
            result = call_with_retry(flaky)

        assert result == "ok"
        assert len(attempts) == 2

    def test_max_retries_then_raises(self):
        from feedback_seg.retry import call_with_retry
        from feedback_seg import constants

        attempts = []

        def always_fail():
            attempts.append(1)
            raise ValueError("never retryable")

        with pytest.raises(ValueError):
            call_with_retry(always_fail)
        # ValueError isn't transient → 1 attempt only
        assert len(attempts) == 1


# ────────────────────────────────────────────────────────────────────
# Summary regex fallback (Defect 6)
# ────────────────────────────────────────────────────────────────────

class TestSummaryParsing:
    def test_clean_json(self):
        from feedback_seg.gemini import parse_summary_response
        raw = json.dumps({
            "coaches_summary": "Good performance.",
            "coaches_overall_rating": "75%",
        })
        result = parse_summary_response(raw)
        assert result["coaches_summary"] == "Good performance."
        assert result["coaches_overall_rating"] == "75%"

    def test_strips_markdown_fences(self):
        from feedback_seg.gemini import parse_summary_response
        raw = (
            "```json\n"
            + json.dumps({
                "coaches_summary": "Good.",
                "coaches_overall_rating": "75%",
            })
            + "\n```"
        )
        result = parse_summary_response(raw)
        assert result["coaches_summary"] == "Good."

    def test_truncated_json_extracts_via_regex(self):
        """JSON is cut off mid-string. Regex fallback should recover
        the partial summary."""
        from feedback_seg.gemini import parse_summary_response
        raw = (
            '{\n'
            '  "coaches_overall_rating": "78%",\n'
            '  "coaches_summary": "Good baseline depth, recurring '
            'issue with squareness when shooting from the right'
        )  # cut off mid-string, no closing quote/brace
        result = parse_summary_response(raw)
        assert "Good baseline depth" in result["coaches_summary"]
        assert result["coaches_overall_rating"] == "78%"

    def test_truncated_json_with_closing_quote_uses_tight_regex(self):
        """When the summary closes properly but the rest of the JSON
        is malformed, the tight regex catches it cleanly without
        swallowing trailing garbage."""
        from feedback_seg.gemini import parse_summary_response
        raw = (
            '{\n'
            '  "coaches_overall_rating": "78%",\n'
            '  "coaches_summary": "Good baseline depth.",\n'
            'GARBAGE_FROM_TRUNCATED_RESPONSE'
        )
        result = parse_summary_response(raw)
        assert result["coaches_summary"] == "Good baseline depth."

    def test_completely_unparseable_returns_clean_failure(self):
        from feedback_seg.gemini import parse_summary_response
        raw = "this is not JSON at all"
        result = parse_summary_response(raw)
        assert "could not be parsed" in result["coaches_summary"]
        assert result["coaches_overall_rating"] == "N/A"

    def test_empty_input_returns_clean_failure(self):
        from feedback_seg.gemini import parse_summary_response
        result = parse_summary_response("")
        assert result["coaches_overall_rating"] == "N/A"

    def test_unicode_escapes_decoded(self):
        from feedback_seg.gemini import parse_summary_response
        # JSON-escaped newline and quote in the summary
        raw = (
            '{"coaches_summary": "Line one.\\nLine two with \\"quotes\\".",'
            ' "coaches_overall_rating": "70%"}'
        )
        result = parse_summary_response(raw)
        assert "Line two" in result["coaches_summary"]


# ────────────────────────────────────────────────────────────────────
# Clip ID and record assembly
# ────────────────────────────────────────────────────────────────────

class TestClipID:
    def test_clip_id_format(self):
        from feedback_seg.video import make_clip_id
        assert make_clip_id("ABC123", 30, 75) == "ABC123_30_75"


class TestRecordAssembly:
    def _seg(self, **overrides):
        base = {
            "segmentHasThreat":    True,
            "segment_start":       30,
            "segment_end":         75,
            "threat_goalie_color": "White and Blue",
            "metrics": {
                "shots":      3,
                "shotsOnNet": 2,
                "saves":      2,
                "rebounds":   1,
                "goals":      0,
            },
        }
        base.update(overrides)
        return base

    def test_success_record_includes_all_fields(self):
        from feedback_seg.pipeline import _build_record
        analysis = {
            "technical_reasoning": "[00:01] Good.",
            "goalie_positioning": {"depth_rank": "Base (B)"},
            "coaching_feedback": {"rebound_control_rank": "Killed (Elite)"},
            "analysis_confidence_caveats": [],
        }
        rec = _build_record(self._seg(), "vid_30_75", analysis)
        assert rec["clipID"] == "vid_30_75"
        assert rec["clip_start_time"] == 30
        assert rec["clip_end_time"] == 75
        assert rec["clip_duration"] == 45
        assert rec["clipShot"] is True
        assert rec["clipShotCount"] == 3
        assert rec["clipSave"] is True
        assert rec["clipSaveCount"] == 2
        assert rec["clipHasGoal"] is False
        assert rec["technical_reasoning"] == "[00:01] Good."
        assert "error" not in rec

    def test_error_record(self):
        from feedback_seg.pipeline import _error_record
        rec = _error_record(self._seg(), "vid_30_75", "ffmpeg failed")
        assert rec["error"] == "ffmpeg failed"
        assert rec["clipID"] == "vid_30_75"
        assert rec["clip_duration"] == 45
        # No analysis fields when there's an error
        assert "technical_reasoning" not in rec

    def test_zero_metric_values_dont_break_truthiness(self):
        from feedback_seg.pipeline import _build_record
        seg = self._seg(metrics={"shots": 0, "saves": 0, "goals": 0})
        analysis = {
            "technical_reasoning": "x",
            "goalie_positioning": {"depth_rank": "Base (B)"},
            "coaching_feedback": {"rebound_control_rank": "Killed (Elite)"},
        }
        rec = _build_record(seg, "vid_0_45", analysis)
        assert rec["clipShot"] is False
        assert rec["clipShotCount"] == 0
        assert rec["clipSave"] is False
        assert rec["clipHasGoal"] is False

    def test_none_metrics_handled(self):
        from feedback_seg.pipeline import _build_record
        seg = self._seg()
        seg["metrics"] = None
        analysis = {
            "technical_reasoning": "x",
            "goalie_positioning": {},
            "coaching_feedback": {},
        }
        rec = _build_record(seg, "vid_0_45", analysis)
        # Should not raise; clipShot etc default to False/0
        assert rec["clipShotCount"] == 0
        assert rec["clipHasGoal"] is False


# ────────────────────────────────────────────────────────────────────
# Inline-bytes size gate
# ────────────────────────────────────────────────────────────────────

class TestInlineBytesGate:
    def test_small_file_returns_true(self, tmp_path):
        from feedback_seg.gemini import should_use_inline_bytes
        f = tmp_path / "small.mp4"
        f.write_bytes(b"x" * 1024)
        assert should_use_inline_bytes(str(f)) is True

    def test_large_file_returns_false(self, tmp_path):
        from feedback_seg.gemini import should_use_inline_bytes
        from feedback_seg.constants import INLINE_BYTES_MAX_SIZE
        f = tmp_path / "big.mp4"
        # write just over the limit
        f.write_bytes(b"x" * (INLINE_BYTES_MAX_SIZE + 1))
        assert should_use_inline_bytes(str(f)) is False

    def test_missing_file_returns_false(self):
        from feedback_seg.gemini import should_use_inline_bytes
        assert should_use_inline_bytes("/nonexistent/path.mp4") is False


# ────────────────────────────────────────────────────────────────────
# Video extraction: invalid duration
# ────────────────────────────────────────────────────────────────────

class TestVideoExtraction:
    def test_zero_duration_raises(self):
        from feedback_seg.video import extract_clip
        with pytest.raises(ValueError):
            extract_clip("doesn't_matter.mp4", 30, 30, "out.mp4")

    def test_negative_duration_raises(self):
        from feedback_seg.video import extract_clip
        with pytest.raises(ValueError):
            extract_clip("doesn't_matter.mp4", 30, 20, "out.mp4")
