"""Generate a synthetic customer JSON covering all paired hudl matches.

extract_label_frames_v2.py and sample_hard_negatives.py require a
--customers JSON to look up `targetGoalieTeamName` per vID (used to
filter GT to opponent-team shots). For hudl-fetched matches we don't
ship a curated customer file; this script emits a minimal one by
reading the team-perspective from each gt_<hudl>.csv directly.

For each gt CSV:
  - opp team = the team_name appearing in opponent rows; pick the
    most-common non-target team
  - target team = the OTHER team (the goalie team)
Heuristic: split rows by `team` field, take the two team names that
appear most often, and label the SMALLER-shot-count team as target
(goalies see more shots-against than their team takes — true on
average for the games in this corpus).

For the learning-curve scaffold a slightly noisy target/opp split is
fine: we want training positives that LOOK like shots, regardless of
which side the camera was on. If you need a clean split for eval-
attribution, build the customer JSON by hand.

Output: data/customers/CUST_LEARNCURVE.json with one record per
paired match, in the same shape the existing CUST000048.json uses.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO   = Path(__file__).resolve().parents[2]
GT_DIR = REPO / "data" / "ground_truth"
VID_DIR = REPO / "data" / "videos"
OUT    = REPO / "data" / "customers" / "CUST_LEARNCURVE.json"


def teams_in_gt(csv_path: Path) -> tuple[str, str]:
    """Return (target_team_name, opp_team_name) by row-count heuristic.
    Target = the team that takes FEWER threat-action shots (the goalie
    side, on average). Returns ('', '') if GT is empty/malformed."""
    if not csv_path.exists():
        return "", ""
    counts: Counter[str] = Counter()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("action") or "").strip() in {"Shots", "Goals"}:
                t = (row.get("team") or "").strip()
                if t:
                    counts[t] += 1
    if len(counts) < 2:
        # Fallback: pick whatever we have
        teams = list(counts.keys())
        return (teams[0] if teams else ""), (teams[0] if teams else "")
    top2 = counts.most_common(2)
    opp_team, _ = top2[0]      # more shots = opp (attacking)
    target_team, _ = top2[1]   # fewer shots = target (defending)
    return target_team, opp_team


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT,
                    help=f"customer JSON path (default {OUT})")
    args = ap.parse_args()

    records = []
    skipped = []
    paired = sorted({p.stem.replace("full_", "")
                     for p in VID_DIR.glob("full_*.mp4")
                     if p.stem.replace("full_", "").isdigit()}
                    & {p.stem.replace("gt_", "")
                       for p in GT_DIR.glob("gt_*.csv")})
    for mid_s in paired:
        gt = GT_DIR / f"gt_{mid_s}.csv"
        target, opp = teams_in_gt(gt)
        if not target:
            skipped.append(mid_s)
            continue
        records.append({
            "vID":                       mid_s,           # numeric str = hudl_id
            "hudlId":                    int(mid_s),
            "targetGoalieTeamName":      target,
            "opponentGoalieTeamName":    opp,
            "targetGoalieColor":         "Unknown",       # placeholder
            "opponentGoalieColor":       "Unknown",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2))
    print(f"wrote {args.out}  ({len(records)} records, {len(skipped)} skipped)",
          file=sys.stderr)
    if skipped:
        print(f"  skipped (no team rows in GT): {skipped}", file=sys.stderr)
    for r in records[:5]:
        print(f"  {r['vID']}  target={r['targetGoalieTeamName']!r}  "
              f"opp={r['opponentGoalieTeamName']!r}", file=sys.stderr)
    if len(records) > 5:
        print(f"  …({len(records) - 5} more)", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
