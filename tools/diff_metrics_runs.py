"""Per-segment v13-vs-v14 diff analyzer.

Reads two metrics_seg outputs for the same vID + their trace sidecars +
the GT CSV. Produces a markdown report listing only the segments where
the two runs disagreed, with:

  - Side-by-side counts (shots, saves, goals)
  - Trace-level WHY (per-call goal vote, ensemble override, prefilter
    skip flags)
  - GT context (real shots/goals that occurred in the window)
  - Verdict (which version's count is closer to GT, if either)

This is a $0 qualitative tool — no Gemini calls. Use it to understand
why v14_ensemble downgraded a true goal, why v14_prefilter skipped a
window that contained a real shot, etc.

Usage:
    python3 tools/diff_metrics_runs.py \\
        --vID mjEeE7p2Hz8 \\
        --v13-dir data/output/runs/metrics_v13 \\
        --v14-dir data/output/runs/metrics_v14_ensemble \\
        --hudl-id 2073809 \\
        --out diff_mjEe_v13_vs_v14ensemble.md

If --hudl-id is omitted, looks up via eval/eval_cv_seg_output.py's
VID_TO_HUDL mapping for known YouTube-stem vIDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]


def fmt_mmss(t: float) -> str:
    m = int(t // 60); s = int(t % 60)
    return f"{m:02d}:{s:02d}"


@dataclass
class SegmentMetrics:
    """One run's metrics for one segment."""
    shots:        int = 0
    shots_on_net: int = 0
    saves:        int = 0
    goals:        int = 0
    n_shot_ts:    int = 0
    raw:          dict = field(default_factory=dict)

    @classmethod
    def from_metrics_dict(cls, m: Optional[dict]) -> "SegmentMetrics":
        if not isinstance(m, dict):
            return cls()
        return cls(
            shots        = int(m.get("shots", 0) or 0),
            shots_on_net = int(m.get("shotsOnNet", 0) or 0),
            saves        = int(m.get("saves", 0) or 0),
            goals        = int(m.get("goals", 0) or 0),
            n_shot_ts    = len(m.get("shot_timestamps") or []),
            raw          = m,
        )


def load_metrics_by_start(path: Path) -> dict[int, SegmentMetrics]:
    if not path.exists():
        return {}
    out: dict[int, SegmentMetrics] = {}
    for seg in json.loads(path.read_text()):
        if not isinstance(seg, dict):
            continue
        s = seg.get("segment_start")
        if s is None:
            continue
        out[int(s)] = SegmentMetrics.from_metrics_dict(seg.get("metrics"))
    return out


def load_trace_by_start(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and isinstance(raw.get("per_segment"), list):
        return {int(t["segment_start"]): t for t in raw["per_segment"]
                 if "segment_start" in t}
    if isinstance(raw, list):
        return {int(t["segment_start"]): t for t in raw
                 if isinstance(t, dict) and "segment_start" in t}
    return {}


def load_gt_events(gt_csv: Path,
                    only_actions: tuple[str, ...] = ("Shots", "Goals")) -> list[dict]:
    """Return [{start, end, action, team}] from a Hudl gt CSV.

    Hudl publishes 12-second clips around each event. start/end are
    seconds (int)."""
    if not gt_csv.exists():
        return []
    out = []
    with open(gt_csv, newline="") as f:
        for row in csv.DictReader(f):
            action = (row.get("action") or "").strip()
            if action not in only_actions:
                continue
            try:
                s = int(row["start"]); e = int(row["end"])
            except (KeyError, ValueError):
                continue
            out.append({"start": s, "end": e, "action": action,
                         "team": (row.get("team") or "").strip()})
    return out


def gt_in_window(gt_events: list[dict],
                  win_start: int, win_end: int) -> list[dict]:
    """GT events whose midpoint falls inside the window."""
    out = []
    for ev in gt_events:
        mid = 0.5 * (ev["start"] + ev["end"])
        if win_start <= mid <= win_end:
            out.append(ev)
    return out


def diff_segments(
    v13_metrics: dict[int, SegmentMetrics],
    v14_metrics: dict[int, SegmentMetrics],
    v13_trace:   dict[int, dict],
    v14_trace:   dict[int, dict],
    segments:    list[dict],
    gt_events:   list[dict],
) -> list[dict]:
    """Return one record per segment where the runs differ. Each record
    has all the context needed to render a row in the report."""
    diffs = []
    for seg in segments:
        s = int(seg.get("segment_start", -1))
        e = int(seg.get("segment_end", -1))
        if s < 0 or e < 0:
            continue
        v13 = v13_metrics.get(s, SegmentMetrics())
        v14 = v14_metrics.get(s, SegmentMetrics())
        # Did anything count differ?
        same = (v13.shots == v14.shots
                 and v13.shots_on_net == v14.shots_on_net
                 and v13.saves == v14.saves
                 and v13.goals == v14.goals)
        if same:
            continue
        gt_inside = gt_in_window(gt_events, s, e)
        n_gt_shots = sum(1 for g in gt_inside if g["action"] == "Shots")
        n_gt_goals = sum(1 for g in gt_inside if g["action"] == "Goals")
        diffs.append({
            "segment_start": s,
            "segment_end":   e,
            "v13":           v13,
            "v14":           v14,
            "v13_trace":     v13_trace.get(s, {}),
            "v14_trace":     v14_trace.get(s, {}),
            "gt_inside":     gt_inside,
            "n_gt_shots":    n_gt_shots,
            "n_gt_goals":    n_gt_goals,
        })
    return diffs


def verdict_for_field(v13: int, v14: int, gt: int) -> str:
    """Return one of: v13_closer / v14_closer / tied / both_off"""
    d13 = abs(v13 - gt)
    d14 = abs(v14 - gt)
    if d13 < d14: return "v13_closer"
    if d14 < d13: return "v14_closer"
    if d13 == d14 == 0: return "tied_correct"
    return "tied_off"


def trace_summary(trace: dict, v14: bool = False) -> str:
    """Human-readable trace info for a segment."""
    if not trace:
        return "—"
    bits = []
    n_calls = trace.get("n_calls", 0)
    if n_calls:
        bits.append(f"n_calls={n_calls}")
    pcg = trace.get("per_call_goals") or []
    if pcg and len(pcg) > 1:
        bits.append(f"per_call_goals={pcg}")
    sv = trace.get("shot_vote_outcome")
    if sv:
        bits.append(f"shot_vote={sv}")
    gv = trace.get("goal_vote_outcome")
    if gv:
        bits.append(f"goal_vote={gv}")
    if v14:
        # v14-specific trace fields
        if trace.get("_prefilter_skip"):
            bits.append(f"PREFILTER_SKIP@{trace.get('_prefilter_peak_conf',0):.2f}")
        if trace.get("_goal_ensemble_overrode"):
            bits.append(f"ENSEMBLE_OVERRODE({trace.get('_goal_ensemble_reason')})")
    fr = trace.get("failure_reason")
    if fr and fr != "not_run":
        bits.append(f"FAIL={fr}")
    return "; ".join(bits) if bits else "—"


def render_report(
    vid:        str,
    v13_label:  str,
    v14_label:  str,
    diffs:      list[dict],
    seg_total:  int,
    gt_events:  list[dict],
) -> str:
    n_diffs = len(diffs)
    n_gt_total = len(gt_events)

    # Aggregate verdict counters across all diffs
    goal_v13_closer = goal_v14_closer = goal_tied = 0
    shot_v13_closer = shot_v14_closer = shot_tied = 0
    for d in diffs:
        gv = verdict_for_field(d["v13"].goals, d["v14"].goals, d["n_gt_goals"])
        sv = verdict_for_field(d["v13"].shots_on_net, d["v14"].shots_on_net,
                                 d["n_gt_shots"])
        if gv == "v13_closer": goal_v13_closer += 1
        elif gv == "v14_closer": goal_v14_closer += 1
        else: goal_tied += 1
        if sv == "v13_closer": shot_v13_closer += 1
        elif sv == "v14_closer": shot_v14_closer += 1
        else: shot_tied += 1

    lines = [
        f"# metrics_seg diff: {v13_label} vs {v14_label}",
        "",
        f"**Video:** `{vid}`",
        f"**Segments compared:** {seg_total}",
        f"**Segments where output differed:** {n_diffs}",
        f"**GT events total in this video:** {n_gt_total} ({sum(1 for g in gt_events if g['action']=='Goals')} goals)",
        "",
        "## Verdict tally (only counting segments that differed)",
        "",
        "| field | v13 closer to GT | v14 closer to GT | tied |",
        "|---|---|---|---|",
        f"| goals | {goal_v13_closer} | {goal_v14_closer} | {goal_tied} |",
        f"| shotsOnNet | {shot_v13_closer} | {shot_v14_closer} | {shot_tied} |",
        "",
        "## Per-segment differences",
        ""]

    for d in diffs:
        s = d["segment_start"]; e = d["segment_end"]
        win = f"{fmt_mmss(s)}–{fmt_mmss(e)} ({s}-{e}s)"
        lines.append(f"### Segment {win}")
        lines.append("")
        lines.append("| field | v13 | v14 | Δ | GT (in window) | verdict |")
        lines.append("|---|---|---|---|---|---|")

        rows = [
            ("shots",      d["v13"].shots,        d["v14"].shots,        d["n_gt_shots"]),
            ("shotsOnNet", d["v13"].shots_on_net, d["v14"].shots_on_net, d["n_gt_shots"]),
            ("saves",      d["v13"].saves,        d["v14"].saves,        max(0, d["n_gt_shots"] - d["n_gt_goals"])),
            ("goals",      d["v13"].goals,        d["v14"].goals,        d["n_gt_goals"]),
        ]
        for name, a, b, gt in rows:
            delta = b - a
            delta_s = ("—" if delta == 0
                        else f"+{delta}" if delta > 0
                        else str(delta))
            ver = verdict_for_field(a, b, gt) if a != b else "—"
            lines.append(f"| {name} | {a} | {b} | {delta_s} | {gt} | {ver} |")

        # GT detail
        if d["gt_inside"]:
            lines.append("")
            lines.append("**GT events in window:**")
            for g in d["gt_inside"]:
                lines.append(f"- {fmt_mmss(g['start'])}-{fmt_mmss(g['end'])} "
                             f"`{g['action']}` `{g['team']}`")

        # Trace
        v13t = trace_summary(d["v13_trace"], v14=False)
        v14t = trace_summary(d["v14_trace"], v14=True)
        lines.append("")
        lines.append(f"- v13 trace: {v13t}")
        lines.append(f"- v14 trace: {v14t}")

        # If goals diverged, show per-call breakdown explicitly
        v13_pcg = d["v13_trace"].get("per_call_goals") or []
        v14_pcg = d["v14_trace"].get("per_call_goals") or []
        if (v13.goals if (v13 := d["v13"]) else 0) != (v14_g := d["v14"].goals):
            lines.append("")
            lines.append(f"- v13 per-call goal counts: {v13_pcg or '[single call]'}")
            lines.append(f"- v14 per-call goal counts: {v14_pcg or '[single call]'}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# Best-effort VID_TO_HUDL mapping for convenience
DEFAULT_VID_TO_HUDL = {
    "SX5xNJlh6eQ": 2073056, "bfEKgtOIkQU": 2072195,
    "mjEeE7p2Hz8": 2073809, "n2cy8b755Tg": 2127046,
    "v0lxSTbXfw8": 2073810, "dwGsP6QKDs8": 2070269,
    "Fjc9hmK8_3U": 2070260, "HNG0jKYY12g": 2095275,
    "J8WkcuTsD5I": 2072194, "kQVdtRa4o_A": 2127034,
    "krxhPVLGLz8": 2108724, "KYtM20r9BuM": 2072196,
    "q5yj6sAFQeY": 2127052, "zOQrPK7IJ24": 2127035,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vID", required=True)
    ap.add_argument("--v13-dir", required=True, type=Path)
    ap.add_argument("--v14-dir", required=True, type=Path)
    ap.add_argument("--gt-csv",  type=Path, default=None,
                    help="path to gt_<hudl_id>.csv (auto-resolved if --hudl-id given)")
    ap.add_argument("--hudl-id", type=int, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--v13-label", default="v13")
    ap.add_argument("--v14-label", default="v14")
    args = ap.parse_args()

    # Resolve GT CSV
    if args.gt_csv is None:
        hudl = args.hudl_id or DEFAULT_VID_TO_HUDL.get(args.vID)
        if hudl is None:
            print(f"ERROR: provide --gt-csv or --hudl-id "
                   f"(unknown vID {args.vID})", file=sys.stderr)
            return 2
        args.gt_csv = REPO / "data" / "ground_truth" / f"gt_{hudl}.csv"

    v13_path = args.v13_dir / f"gt_metrics_{args.vID}.json"
    v14_path = args.v14_dir / f"gt_metrics_{args.vID}.json"
    v13_trace_path = args.v13_dir / f"gt_metrics_{args.vID}_trace.json"
    v14_trace_path = args.v14_dir / f"gt_metrics_{args.vID}_trace.json"

    for p in (v13_path, v14_path):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr); return 3
    if not args.gt_csv.exists():
        print(f"ERROR: missing GT CSV {args.gt_csv}", file=sys.stderr); return 3

    v13_metrics = load_metrics_by_start(v13_path)
    v14_metrics = load_metrics_by_start(v14_path)
    v13_trace   = load_trace_by_start(v13_trace_path)
    v14_trace   = load_trace_by_start(v14_trace_path)

    # Build segment list (must come from one of the metrics JSONs to know the windows)
    segments = json.loads(v13_path.read_text())
    gt_events = load_gt_events(args.gt_csv)

    diffs = diff_segments(v13_metrics, v14_metrics,
                            v13_trace,   v14_trace,
                            segments, gt_events)

    report = render_report(
        vid=args.vID,
        v13_label=args.v13_label,
        v14_label=args.v14_label,
        diffs=diffs,
        seg_total=len(segments),
        gt_events=gt_events,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"  {len(diffs)} of {len(segments)} segments differed",
          file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
