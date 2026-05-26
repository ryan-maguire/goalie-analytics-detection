#!/usr/bin/env python3
"""Compare metrics_seg outputs from gemini-3.5-flash vs gemini-2.5-pro.

Walks per-window pairs (matched by segment_start) and reports:
  - count of agreeing windows
  - count of windows that disagree, broken down by metric
  - aggregate per-game totals (shots, shotsOnNet, saves, goals)
  - per-window deltas for shots / saves / goals

Run AFTER tools/ab_test_flash35.sh has populated
data/output/runs/metrics_flash35/. Reads the Pro baselines from
data/output/runs/metrics_v13/.
"""
import json
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]
PRO_DIR   = REPO / "data" / "output" / "runs" / "metrics_v13"
FLASH_DIR = REPO / "data" / "output" / "runs" / "metrics_flash35"

VIDS = ["Fjc9hmK8_3U", "q5yj6sAFQeY", "KYtM20r9BuM"]
NUMERIC_FIELDS = ["shots", "shotsOnNet", "saves", "rebounds", "goals"]


def load_windows(path: Path) -> dict[int, dict]:
    """Return {segment_start: metrics_dict}."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("windows") or data.get("segments") or []
    out = {}
    for w in data:
        if not isinstance(w, dict):
            continue
        m = w.get("metrics")
        if not isinstance(m, dict):
            continue
        out[int(w["segment_start"])] = m
    return out


def diff_one(pro: dict, fla: dict) -> dict:
    """Return per-field deltas (fla - pro) for one window pair."""
    return {f: int(fla.get(f, 0) or 0) - int(pro.get(f, 0) or 0)
             for f in NUMERIC_FIELDS}


def main():
    print(f"{'vID':<14}  {'overlap':>7}  {'pro_only':>8}  {'fla_only':>8}  "
          f"{'shots_Δ':>8}  {'sON_Δ':>6}  {'saves_Δ':>8}  {'goals_Δ':>8}  agree%")
    print("-" * 100)

    all_per_window_deltas = []
    all_totals = {"pro": defaultdict(int), "fla": defaultdict(int)}

    for vid in VIDS:
        pro = load_windows(PRO_DIR   / f"gt_metrics_{vid}.json")
        fla = load_windows(FLASH_DIR / f"gt_metrics_{vid}.json")
        if not pro:
            print(f"{vid:<14}  Pro JSON missing — skip")
            continue
        if not fla:
            print(f"{vid:<14}  Flash JSON missing — skip (run ab_test_flash35.sh first)")
            continue

        overlap = sorted(set(pro) & set(fla))
        pro_only = sorted(set(pro) - set(fla))
        fla_only = sorted(set(fla) - set(pro))

        agree_count = 0
        sums = {f: 0 for f in NUMERIC_FIELDS}
        for s in overlap:
            d = diff_one(pro[s], fla[s])
            all_per_window_deltas.append((vid, s, d))
            if all(v == 0 for v in d.values()):
                agree_count += 1
            for f in NUMERIC_FIELDS:
                sums[f] += d[f]
        agree_pct = (agree_count / len(overlap) * 100) if overlap else 0.0

        # Game-level totals
        for s, m in pro.items():
            for f in NUMERIC_FIELDS:
                all_totals["pro"][f] += int(m.get(f, 0) or 0)
        for s, m in fla.items():
            for f in NUMERIC_FIELDS:
                all_totals["fla"][f] += int(m.get(f, 0) or 0)

        print(f"{vid:<14}  {len(overlap):>7}  {len(pro_only):>8}  {len(fla_only):>8}  "
              f"{sums['shots']:>+8}  {sums['shotsOnNet']:>+6}  "
              f"{sums['saves']:>+8}  {sums['goals']:>+8}  "
              f"{agree_pct:>5.1f}%")

    print()
    print("AGGREGATE PER-METRIC TOTALS (sums across all windows of all 3 games):")
    print(f"  {'field':<12}  {'Pro':>6}  {'Flash':>6}  {'Δ':>6}")
    for f in NUMERIC_FIELDS:
        p = all_totals["pro"][f]
        x = all_totals["fla"][f]
        print(f"  {f:<12}  {p:>6}  {x:>6}  {x-p:>+6}")

    # Goal-level analysis — the metric we care most about
    print()
    print("PER-WINDOW GOAL DISAGREEMENTS:")
    goal_disagree = [(v, s, d["goals"]) for v, s, d in all_per_window_deltas if d["goals"] != 0]
    if not goal_disagree:
        print("  (none — Pro and Flash agree on every window's goal count)")
    else:
        print(f"  {'vID':<14}  {'seg_start':>9}  {'Δ goals (fla-pro)':>18}")
        for v, s, d in goal_disagree:
            sign = "fla over-calls" if d > 0 else "fla under-calls"
            print(f"  {v:<14}  {s:>9}  {d:>+18}  ({sign})")


if __name__ == "__main__":
    main()
