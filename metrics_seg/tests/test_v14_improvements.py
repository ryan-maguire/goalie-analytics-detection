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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metrics_seg import prefilter as prefilter_mod
from metrics_seg import calibration as calib_mod
from metrics_seg import flash_screen as flash_mod


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
