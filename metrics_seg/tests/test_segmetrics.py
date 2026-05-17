"""Unit tests for 01_detect_segment_metrics.py.

These cover the helpers and post-parse logic that don't require
hitting Gemini or GCS. Run with: python -m pytest tests/ -q
"""
import importlib.util
import json
import sys
from pathlib import Path

# Import the module by path since the filename starts with a digit
HERE = Path(__file__).parent
SRC = HERE.parent / "01_detect_segment_metrics.py"
spec = importlib.util.spec_from_file_location("segmetrics", str(SRC))
seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seg)


# -------------------------------------------------------------------
# _is_generic_detail
# -------------------------------------------------------------------

class TestIsGenericDetail:
    def test_empty_is_generic(self):
        assert seg._is_generic_detail("")

    def test_short_is_generic(self):
        # Less than 5 words = generic
        assert seg._is_generic_detail("goal scored")
        assert seg._is_generic_detail("a b c d")

    def test_5_words_is_concrete_unless_pattern(self):
        # 5+ words and no generic patterns
        assert not seg._is_generic_detail("player number fourteen scored a goal")

    def test_known_pattern_caught(self):
        assert seg._is_generic_detail(
            "a goal is counted to maintain the identity from the rubric"
        )
        assert seg._is_generic_detail(
            "appears to be a goal but I cannot tell exactly"
        )

    def test_concrete_observation_passes(self):
        detail = ("Player #14 in the black jersey celebrates after the puck "
                  "enters the left side of the net at 0:42")
        assert not seg._is_generic_detail(detail)


# -------------------------------------------------------------------
# _try_recover_truncated_json
# -------------------------------------------------------------------

class TestRecoverTruncatedJson:
    def test_recovers_from_truncation_inside_string(self):
        # Mimic Gemini truncation: cut off mid-string in decision_notes
        truncated = (
            '{"shots": 2, "shotsOnNet": 1, "saves": 1, "rebounds": 0, "goals": 0, '
            '"observed_goalie_side": "left", '
            '"goal_criteria": {"anchor_puck_crosses_line": false}, '
            '"decision_notes": "the puck was'
        )
        # _try_recover walks back to last } — but here there's only one nested
        # } from goal_criteria. Should recover that partial dict.
        recovered = seg._try_recover_truncated_json(truncated, segment_start=100)
        # In this case there's no balanced TOP-level closing brace, so recovery
        # should fail.
        assert recovered is None

    def test_recovers_when_balanced_close_exists(self):
        # Trailing junk after the last complete brace
        valid_then_junk = (
            '{"shots": 2, "shotsOnNet": 1, "saves": 1, "rebounds": 0, '
            '"goals": 0, "observed_goalie_side": "left", '
            '"goal_criteria": {"anchor_puck_crosses_line": false, '
            '"anchor_ref_points_at_net": false, "anchor_puck_retrieved_from_net": false, '
            '"support_whistle": false, "support_crowd_spike": false, '
            '"support_celebration": false, "support_centre_ice_faceoff": false, '
            '"disqualifier_active": false, "anchor_puck_crosses_line_timestamp": "", '
            '"confirming_detail": "", "decision_notes": "no goal"}}'
            '\nSome trailing text that broke the parse'
        )
        recovered = seg._try_recover_truncated_json(valid_then_junk, segment_start=100)
        assert recovered is not None
        assert recovered["shots"] == 2
        assert recovered["goal_criteria"]["disqualifier_active"] is False

    def test_returns_none_on_no_close_brace(self):
        assert seg._try_recover_truncated_json('{"shots": 2, "shotsOnNet"', 100) is None
        assert seg._try_recover_truncated_json('', 100) is None


# -------------------------------------------------------------------
# _median_int
# -------------------------------------------------------------------

class TestMedianInt:
    def test_odd_length(self):
        assert seg._median_int([1, 5, 3]) == 3
        assert seg._median_int([1]) == 1

    def test_even_length_returns_lower(self):
        # Documented behaviour: deterministic, conservative — lower of the two
        assert seg._median_int([1, 2, 3, 4]) == 2
        assert seg._median_int([1, 1]) == 1

    def test_empty_returns_zero(self):
        assert seg._median_int([]) == 0


# -------------------------------------------------------------------
# _summarize_traces
# -------------------------------------------------------------------

class TestSummarizeTraces:
    def test_empty_input(self):
        out = seg._summarize_traces({})
        assert out["n_segments"] == 0
        assert out["total_gemini_calls"] == 0

    def test_single_cheap_segment(self):
        traces = {0: {"n_calls": 1, "shot_vote_triggered": False,
                      "goal_vote_triggered": False}}
        out = seg._summarize_traces(traces)
        assert out["n_segments"] == 1
        assert out["total_gemini_calls"] == 1
        assert out["cost_ratio"] == 1.0
        assert out["shot_vote_pct"] == 0.0

    def test_mixed_votes(self):
        traces = {
            0: {"n_calls": 1, "shot_vote_triggered": False,
                "goal_vote_triggered": False},
            1: {"n_calls": 3, "shot_vote_triggered": True,
                "goal_vote_triggered": True, "goal_vote_outcome": "kept"},
            2: {"n_calls": 3, "shot_vote_triggered": False,
                "goal_vote_triggered": True, "goal_vote_outcome": "rejected"},
            3: {"n_calls": 1, "shot_vote_triggered": False,
                "goal_vote_triggered": False},
        }
        out = seg._summarize_traces(traces)
        assert out["n_segments"] == 4
        assert out["total_gemini_calls"] == 8  # 1+3+3+1
        assert out["cost_ratio"] == 2.0
        assert out["shot_vote_fired"] == 1
        assert out["shot_vote_pct"] == 25.0
        assert out["goal_vote_fired"] == 2
        assert out["goal_vote_pct"] == 50.0
        assert out["goal_vote_kept"] == 1
        assert out["goal_vote_rejected"] == 1


# -------------------------------------------------------------------
# _extract_finish_reason
# -------------------------------------------------------------------

class TestExtractFinishReason:
    def test_string_finish_reason(self):
        # Some SDK versions return strings
        class FakeCand:
            finish_reason = "SAFETY"
        class FakeResp:
            candidates = [FakeCand()]
        assert seg._extract_finish_reason(FakeResp()) == "SAFETY"

    def test_enum_finish_reason(self):
        # Other SDK versions return enums with .name
        class FakeEnum:
            name = "RECITATION"
            def __str__(self):
                return "FinishReason.RECITATION"
        class FakeCand:
            finish_reason = FakeEnum()
        class FakeResp:
            candidates = [FakeCand()]
        assert seg._extract_finish_reason(FakeResp()) == "RECITATION"

    def test_missing_candidates_returns_none(self):
        class FakeResp:
            candidates = []
        assert seg._extract_finish_reason(FakeResp()) is None

    def test_no_candidates_attr(self):
        class FakeResp:
            pass
        assert seg._extract_finish_reason(FakeResp()) is None


# -------------------------------------------------------------------
# Prompt loader
# -------------------------------------------------------------------

class TestPromptLoader:
    def test_prompt_loaded(self):
        # Should be a substantial prompt, not a stub
        assert len(seg.METRICS_PROMPT) > 10000
        assert "ice hockey video analyst" in seg.METRICS_PROMPT

    def test_prompt_format_works(self):
        # All four expected placeholders must format without error
        out = seg.METRICS_PROMPT.format(
            duration=120, goalie_color="white",
            opponent_color="black", side="left",
        )
        assert "white" in out
        assert "black" in out
        assert "120" in out


# -------------------------------------------------------------------
# Terminal finish reasons set
# -------------------------------------------------------------------

class TestTerminalFinishReasons:
    def test_safety_is_terminal(self):
        assert "SAFETY" in seg.TERMINAL_FINISH_REASONS

    def test_stop_is_not_terminal(self):
        # STOP = success; should not be in the terminal set
        assert "STOP" not in seg.TERMINAL_FINISH_REASONS

    def test_max_tokens_is_not_terminal(self):
        # MAX_TOKENS = retried via JSON recovery; not terminal
        assert "MAX_TOKENS" not in seg.TERMINAL_FINISH_REASONS


# -------------------------------------------------------------------
# Inline bytes size guard (v8)
# -------------------------------------------------------------------

class TestInlineBytesSizeGuard:
    """Verify the v8 inline-byte path's clip-too-large guard.

    Uses analyze_clip_metrics with a fake oversize file; we don't need
    a real Gemini client because the size check happens BEFORE any
    Gemini call."""

    def test_oversize_clip_returns_failure_trace(self, tmp_path):
        # Write a fake clip that exceeds the limit
        big = tmp_path / "fake_oversize.mp4"
        big.write_bytes(b"\x00" * (seg.MAX_INLINE_VIDEO_BYTES + 1024))

        metrics, trace = seg.analyze_clip_metrics(
            clip_path=str(big),
            goalie_color="white",
            goalie_side="left",
            duration=30,
            segment_start=100,
            gemini_client=None,  # never reached
        )
        assert metrics is None
        assert trace["failure_reason"].startswith("clip_too_large:")
        assert trace["n_calls"] == 0

    def test_missing_clip_returns_failure_trace(self, tmp_path):
        # Path doesn't exist
        metrics, trace = seg.analyze_clip_metrics(
            clip_path=str(tmp_path / "nonexistent.mp4"),
            goalie_color="white",
            goalie_side="left",
            duration=30,
            segment_start=100,
            gemini_client=None,
        )
        assert metrics is None
        assert trace["failure_reason"].startswith("clip_read_failed:")
        assert trace["n_calls"] == 0

    def test_max_inline_video_bytes_constant_is_reasonable(self):
        # Should be at most 20 MB (Vertex hard limit) and at least 5 MB
        # (otherwise we'd reject clips that work today)
        assert 5 * 1024 * 1024 <= seg.MAX_INLINE_VIDEO_BYTES <= 20 * 1024 * 1024
