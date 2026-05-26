"""Per-game calibration tracking.

Append-only log of (vID, predicted_totals, gt_totals, delta) per run.
Lives at data/output/calibration/<vID>.jsonl (one JSON-line per run).
Provides:

  - Historical accuracy per game/team
  - Optional `apply_correction(vID, predicted)` that scales predictions
    by the rolling-median delta. Read-only by default — analysts can
    inspect; production does not auto-apply.

This is the lightweight version. A heavier tracking system would
write to BigQuery / a structured DB; for now, jsonl + per-vID file
is enough.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "data" / "output" / "calibration"
_lock = threading.Lock()


@dataclass
class GameTotals:
    shots:      int = 0
    shots_on_net: int = 0
    saves:      int = 0
    goals:      int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "GameTotals":
        return cls(
            shots        = int(d.get("shots", 0) or 0),
            shots_on_net = int(d.get("shotsOnNet", d.get("shots_on_net", 0)) or 0),
            saves        = int(d.get("saves", 0) or 0),
            goals        = int(d.get("goals", 0) or 0),
        )

    def to_dict(self) -> dict:
        return {"shots": self.shots, "shotsOnNet": self.shots_on_net,
                 "saves": self.saves, "goals": self.goals}


def log_run(
    vid: str,
    predicted: GameTotals,
    ground_truth: Optional[GameTotals] = None,
    extra: Optional[dict] = None,
    log_dir: Optional[Path] = None,
) -> Path:
    """Append a record to <log_dir>/<vID>.jsonl. Returns the file path.

    ground_truth may be None — many runs don't have GT available.
    `extra` lets callers add fields like prompt_version, model_name,
    cache_hit_rate, prefilter_skipped, etc.
    """
    log_dir = log_dir or DEFAULT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    rec: dict = {
        "ts":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vID":       vid,
        "predicted": predicted.to_dict(),
    }
    if ground_truth is not None:
        rec["ground_truth"] = ground_truth.to_dict()
        rec["delta"] = {
            "shots":      predicted.shots - ground_truth.shots,
            "shotsOnNet": predicted.shots_on_net - ground_truth.shots_on_net,
            "saves":      predicted.saves - ground_truth.saves,
            "goals":      predicted.goals - ground_truth.goals,
        }
    if extra:
        rec["extra"] = extra

    path = log_dir / f"{vid}.jsonl"
    line = json.dumps(rec, separators=(",", ":")) + "\n"
    with _lock:
        with open(path, "a") as f:
            f.write(line)
    return path


def load_history(vid: str, log_dir: Optional[Path] = None) -> list[dict]:
    log_dir = log_dir or DEFAULT_DIR
    path = log_dir / f"{vid}.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def rolling_median_delta(
    vid: str,
    field: str = "shots",
    last_n: int = 5,
    log_dir: Optional[Path] = None,
) -> Optional[float]:
    """Median of the last `last_n` runs' delta for `field`.
    Returns None if no GT-bearing rows exist."""
    rows = load_history(vid, log_dir)
    deltas = [r["delta"][field] for r in rows
               if r.get("delta") and field in r.get("delta", {})]
    if not deltas:
        return None
    deltas = deltas[-last_n:]
    return statistics.median(deltas)


def apply_correction(
    vid: str,
    predicted: GameTotals,
    log_dir: Optional[Path] = None,
) -> GameTotals:
    """Apply rolling-median-delta correction. Subtracts the historical
    over-count (positive delta means model over-predicts → subtract
    from prediction). Floor at 0.

    USE WITH CARE — not auto-applied in production. Provided for
    analysts to inspect what calibrated predictions would look like.
    """
    out = GameTotals(predicted.shots, predicted.shots_on_net,
                       predicted.saves, predicted.goals)
    for field in ("shots", "shots_on_net", "saves", "goals"):
        # Field name in log uses camelCase shotsOnNet
        log_field = {"shots_on_net": "shotsOnNet"}.get(field, field)
        d = rolling_median_delta(vid, log_field, log_dir=log_dir)
        if d is None:
            continue
        cur = getattr(out, field)
        corrected = max(0, int(round(cur - d)))
        setattr(out, field, corrected)
    return out


def summary(vid: str, log_dir: Optional[Path] = None) -> dict:
    rows = load_history(vid, log_dir)
    return {
        "vID":      vid,
        "n_runs":   len(rows),
        "n_with_gt": sum(1 for r in rows if r.get("ground_truth")),
        "rolling_median_shots_delta":  rolling_median_delta(vid, "shots", log_dir=log_dir),
        "rolling_median_goals_delta":  rolling_median_delta(vid, "goals", log_dir=log_dir),
    }
