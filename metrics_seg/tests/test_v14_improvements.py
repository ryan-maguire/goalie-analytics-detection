"""Unit tests for the v14 improvements modules.

No Gemini calls. No network. Designed to pass without google.cloud
SDK installed (each module is import-safe).

Run:
    python3 -m pytest metrics_seg/tests/test_v14_improvements.py -v
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metrics_seg import cache as cache_mod
from metrics_seg import prefilter as prefilter_mod
from metrics_seg import audio_context as audio_ctx_mod
from metrics_seg import goal_ensemble as goal_ens_mod
from metrics_seg import calibration as calib_mod
from metrics_seg import flash_screen as flash_mod


# ─── cache ────────────────────────────────────────────────────────────
class TestCache(unittest.TestCase):
    def test_key_stable(self):
        k1 = cache_mod.key_for(b"abc", "prompt", "gemini-2.5-pro", 0.0)
        k2 = cache_mod.key_for(b"abc", "prompt", "gemini-2.5-pro", 0.0)
        self.assertEqual(k1, k2)

    def test_key_changes_on_input_change(self):
        base = cache_mod.key_for(b"abc", "prompt", "gemini-2.5-pro", 0.0)
        cases = [
            cache_mod.key_for(b"abd", "prompt", "gemini-2.5-pro", 0.0),    # bytes
            cache_mod.key_for(b"abc", "prompt2", "gemini-2.5-pro", 0.0),   # prompt
            cache_mod.key_for(b"abc", "prompt", "gemini-2.5-flash", 0.0),  # model
            cache_mod.key_for(b"abc", "prompt", "gemini-2.5-pro", 0.3),    # temp
        ]
        for c in cases:
            self.assertNotEqual(base, c)

    def test_put_get_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            c = cache_mod.GeminiResponseCache(cache_dir=td)
            key = cache_mod.key_for(b"v", "p", "m", 0.0)
            payload = {"shots": 3, "saves": 2, "goals": 1}
            self.assertIsNone(c.get(key))
            c.put(key, payload)
            self.assertEqual(c.get(key), payload)
            self.assertEqual(c.size(), 1)

    def test_disabled_is_noop(self):
        c = cache_mod.GeminiResponseCache(disabled=True)
        key = cache_mod.key_for(b"v", "p", "m", 0.0)
        c.put(key, {"x": 1})
        self.assertIsNone(c.get(key))
        self.assertEqual(c.size(), 0)


# ─── prefilter ────────────────────────────────────────────────────────
def _write_probs_tsv(path: Path, probs: list[float]) -> None:
    with open(path, "w") as f:
        f.write("t\tshot_max_conf\tshot_n\tgoal_max_conf\n")
        for t, p in enumerate(probs):
            f.write(f"{t}\t{p:.4f}\t0\t0.0\n")


class TestPrefilter(unittest.TestCase):
    def _make_dirs(self):
        d = tempfile.mkdtemp()
        yolo = Path(d) / "yolo"; yolo.mkdir()
        audio = Path(d) / "audio"; audio.mkdir()
        return yolo, audio

    def test_load_returns_fused(self):
        yolo, audio = self._make_dirs()
        _write_probs_tsv(yolo / "abc.tsv",  [0.1, 0.2, 0.8, 0.3])
        _write_probs_tsv(audio / "abc.tsv", [0.0, 0.5, 0.4, 0.7])
        fp = prefilter_mod.load_fused_probs("abc", yolo, audio)
        # 50/50 weighted average
        self.assertAlmostEqual(fp.probs[0], 0.05, places=4)
        self.assertAlmostEqual(fp.probs[2], 0.6,  places=4)

    def test_peak_in_window(self):
        yolo, audio = self._make_dirs()
        _write_probs_tsv(yolo / "abc.tsv", [0.1, 0.9, 0.2, 0.3, 0.1])
        _write_probs_tsv(audio / "abc.tsv", [0.0, 0.0, 0.0, 0.0, 0.0])
        fp = prefilter_mod.load_fused_probs("abc", yolo, audio)
        self.assertAlmostEqual(prefilter_mod.peak_in_window(fp, 0, 4), 0.45, places=4)

    def test_should_skip_threshold_zero_disables(self):
        yolo, audio = self._make_dirs()
        _write_probs_tsv(yolo / "abc.tsv", [0.01] * 10)
        _write_probs_tsv(audio / "abc.tsv", [0.01] * 10)
        fp = prefilter_mod.load_fused_probs("abc", yolo, audio)
        skip, _ = prefilter_mod.should_skip(fp, 0, 9, threshold=0.0)
        self.assertFalse(skip)

    def test_should_skip_low_prob_window(self):
        yolo, audio = self._make_dirs()
        _write_probs_tsv(yolo / "abc.tsv", [0.05] * 10)
        _write_probs_tsv(audio / "abc.tsv", [0.10] * 10)
        fp = prefilter_mod.load_fused_probs("abc", yolo, audio)
        skip, peak = prefilter_mod.should_skip(fp, 0, 9, threshold=0.30)
        self.assertTrue(skip)
        self.assertLess(peak, 0.30)

    def test_null_metrics_dict_schema(self):
        d = prefilter_mod.null_metrics_dict(0.12)
        for k in ("shots", "shotsOnNet", "saves", "goals"):
            self.assertEqual(d[k], 0)
        self.assertTrue(d["_prefilter_skip"])


# ─── audio context ────────────────────────────────────────────────────
class TestAudioContext(unittest.TestCase):
    def test_peak_summary_empty(self):
        s = audio_ctx_mod._peak_summary(np.zeros(0), 0, 10)
        self.assertEqual(s, "—")

    def test_peak_summary_finds_local_maxima(self):
        probs = np.array([0.1, 0.5, 0.8, 0.3, 0.7, 0.2, 0.1])
        s = audio_ctx_mod._peak_summary(probs, 0, 6, min_peak=0.4, top_k=3)
        # Should find both peaks 0.8@2 and 0.7@4
        self.assertIn("0.80", s); self.assertIn("0.70", s)

    def test_render_context_block_empty(self):
        out = audio_ctx_mod.render_context_block("vid", 0, 10)
        self.assertEqual(out, "")

    def test_render_context_block_with_probs(self):
        probs = np.array([0.1, 0.8, 0.2, 0.7, 0.1])
        out = audio_ctx_mod.render_context_block(
            "vid", 0, 4, yolo_probs=probs, audio_probs=probs)
        self.assertIn("OPTIONAL CONTEXT", out)
        self.assertIn("Visual shot-prob peaks", out)
        self.assertIn("Audio shot-prob peaks", out)


# ─── goal ensemble ────────────────────────────────────────────────────
class TestGoalEnsemble(unittest.TestCase):
    def test_no_goal_in_first_no_ensemble(self):
        first = {"shots": 5, "saves": 5, "goals": 0, "shotsOnNet": 5}
        called = []
        def cg(b, p, s, t): called.append(t); return ({"goals": 1}, {})
        result, trace = goal_ens_mod.confirm_goal(
            first_result=first, video_bytes=b"v", prompt_text="p",
            segment_start=0, segment_end=10, call_gemini=cg, fused_probs=None)
        self.assertEqual(result["goals"], 0)
        self.assertEqual(called, [])
        self.assertEqual(trace.decision, "untouched")

    def test_goal_confirmed_by_vote_and_prob(self):
        first = {"shots": 5, "saves": 4, "goals": 1, "shotsOnNet": 5}
        def cg(b, p, s, t): return ({"goals": 1}, {})
        probs = np.array([0.6] * 30)  # sustained well above threshold
        result, trace = goal_ens_mod.confirm_goal(
            first_result=first, video_bytes=b"v", prompt_text="p",
            segment_start=0, segment_end=20, call_gemini=cg, fused_probs=probs)
        self.assertEqual(result["goals"], 1)
        self.assertEqual(trace.decision, "confirmed")
        self.assertEqual(trace.n_yes_goal, 3)

    def test_goal_downgraded_by_vote_failure(self):
        first = {"shots": 5, "saves": 4, "goals": 1, "shotsOnNet": 5}
        def cg(b, p, s, t): return ({"goals": 0}, {})
        probs = np.array([0.6] * 30)
        result, trace = goal_ens_mod.confirm_goal(
            first_result=first, video_bytes=b"v", prompt_text="p",
            segment_start=0, segment_end=20, call_gemini=cg, fused_probs=probs)
        self.assertEqual(result["goals"], 0)
        self.assertEqual(trace.decision, "downgraded")
        self.assertTrue(result["_goal_ensemble_overrode"])
        self.assertEqual(result["_goal_ensemble_reason"], "vote_failed")

    def test_goal_downgraded_by_prob_veto(self):
        first = {"shots": 5, "saves": 4, "goals": 1, "shotsOnNet": 5}
        # 2 of 3 votes say goal — vote passes
        responses = iter([{"goals": 1}, {"goals": 0}])
        def cg(b, p, s, t): return (next(responses), {})
        probs = np.zeros(30)   # no signal → veto
        result, trace = goal_ens_mod.confirm_goal(
            first_result=first, video_bytes=b"v", prompt_text="p",
            segment_start=0, segment_end=20, call_gemini=cg, fused_probs=probs)
        self.assertEqual(result["goals"], 0)
        self.assertEqual(trace.decision, "downgraded")
        self.assertEqual(result["_goal_ensemble_reason"], "prob_signal_veto")


# ─── calibration ──────────────────────────────────────────────────────
class TestCalibration(unittest.TestCase):
    def test_log_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            ld = Path(td)
            pred = calib_mod.GameTotals(shots=50, shots_on_net=40,
                                          saves=37, goals=3)
            gt   = calib_mod.GameTotals(shots=52, shots_on_net=42,
                                          saves=40, goals=2)
            calib_mod.log_run("vid1", pred, gt, log_dir=ld)
            calib_mod.log_run("vid1", pred, gt, log_dir=ld)
            history = calib_mod.load_history("vid1", log_dir=ld)
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["delta"]["goals"], 1)

    def test_rolling_median_delta(self):
        with tempfile.TemporaryDirectory() as td:
            ld = Path(td)
            for shots_pred in (50, 52, 48, 51, 49):
                p = calib_mod.GameTotals(shots=shots_pred)
                gt = calib_mod.GameTotals(shots=50)
                calib_mod.log_run("vid1", p, gt, log_dir=ld)
            m = calib_mod.rolling_median_delta("vid1", "shots", log_dir=ld)
            self.assertEqual(m, 0)   # deltas [0, 2, -2, 1, -1] → median 0


# ─── flash screen ─────────────────────────────────────────────────────
class TestFlashScreen(unittest.TestCase):
    def test_disabled_fail_safe_positive(self):
        r = flash_mod.screen_clip(b"v", 30, gemini_client=None, enabled=False)
        self.assertTrue(r.shots_any)
        self.assertTrue(r.goal_likely)

    def test_should_skip_pro_only_when_confidently_negative(self):
        r1 = flash_mod.ScreenResult(False, False, 0.9)
        self.assertTrue(flash_mod.should_skip_pro(r1, min_confidence=0.7))
        r2 = flash_mod.ScreenResult(False, False, 0.5)  # low conf → don't skip
        self.assertFalse(flash_mod.should_skip_pro(r2, min_confidence=0.7))
        r3 = flash_mod.ScreenResult(True, False, 0.9)   # shots_any True → don't skip
        self.assertFalse(flash_mod.should_skip_pro(r3, min_confidence=0.7))


if __name__ == "__main__":
    unittest.main()
