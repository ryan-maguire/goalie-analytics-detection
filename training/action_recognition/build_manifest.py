#!/usr/bin/env python3
"""Build training manifest for the action-recognition classifier.

For every game with a GT CSV and a stage-1 seg JSON:
  - Walk each stage-1 window
  - Classify it based on overlap with GT events:
        goal       — window overlaps a GT Goals row
        shot_save  — window overlaps a GT Shots row but no Goals row
        no_event   — window overlaps neither
  - Emit (vID, video_path, start_sec, end_sec, label, gt_team, source) records

Also samples ADDITIONAL no_event clips from random video time NOT
covered by stage-1 windows. These give the model exposure to true
non-event footage (line changes, neutral-zone play, etc.) which
stage-1 windows under-represent.

Output: training/action_recognition/manifest.json
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

VIDEOS_DIR  = REPO / "data" / "videos"
GT_DIR      = REPO / "data" / "ground_truth"
SEG_DIRS = [
    REPO / "data" / "output" / "runs" / "cv_seg",
    REPO / "data" / "output" / "runs" / "cv_seg_fusion_wide",
    REPO / "data" / "output" / "runs" / "cv_seg_fusion_wide_post25",
]
OUT_DIR     = REPO / "training" / "action_recognition"

# vID → hudl_id map (matches eval/eval_cv_seg_output.py VID_TO_HUDL)
VID_TO_HUDL = {
    "SX5xNJlh6eQ": 2073056, "bfEKgtOIkQU": 2072195,
    "mjEeE7p2Hz8": 2073809, "n2cy8b755Tg": 2127046,
    "v0lxSTbXfw8": 2073810, "dwGsP6QKDs8": 2070269,
    "Fjc9hmK8_3U": 2070260, "HNG0jKYY12g": 2095275,
    "J8WkcuTsD5I": 2072194, "kQVdtRa4o_A": 2127034,
    "krxhPVLGLz8": 2108724, "KYtM20r9BuM": 2072196,
    "q5yj6sAFQeY": 2127052, "zOQrPK7IJ24": 2127035,
}


@dataclass
class LabeledClip:
    vID:          str         # for youtube-IDed games; otherwise the hudl ID as string
    hudl_id:      int
    video_path:   str
    start_sec:    int
    end_sec:      int
    label:        str         # goal | shot_save | no_event
    overlapping_gt_count: int  # how many GT events overlap (sanity check)
    gt_team:      str = ""    # team that did the shot/goal, if any
    source:       str = ""    # 'stage1' or 'random_neg'


def load_gt(hudl_id: int) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
    """Return (shots, goals) lists of (start, end, team)."""
    path = GT_DIR / f"gt_{hudl_id}.csv"
    if not path.exists():
        return [], []
    shots, goals = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            action = (row.get("action") or "").strip()
            try:
                start = int(row["start"]); end = int(row["end"])
            except (ValueError, KeyError):
                continue
            team = (row.get("team") or "").strip()
            if action == "Shots":
                shots.append((start, end, team))
            elif action == "Goals":
                goals.append((start, end, team))
    return shots, goals


def load_seg_windows(seg_path: Path) -> list[tuple[int, int]]:
    """Return [(start, end), ...] for each window in the seg JSON."""
    try:
        data = json.loads(seg_path.read_text())
        if isinstance(data, dict):
            data = data.get("segments", [])
        return [(int(s["segment_start"]), int(s["segment_end"]))
                 for s in data if "segment_start" in s and "segment_end" in s]
    except Exception:
        return []


def find_seg_path(vid_key: str) -> Optional[Path]:
    """Find a stage-1 seg JSON for vid_key. Tries hudl-id first then vID."""
    for seg_dir in SEG_DIRS:
        candidate = seg_dir / f"gt_seg_{vid_key}.json"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def find_video_path(vid_key: str) -> Optional[Path]:
    for cand in (VIDEOS_DIR / f"full_{vid_key}.mp4", VIDEOS_DIR / f"{vid_key}.mp4"):
        if cand.exists() and cand.stat().st_size > 1024:  # exclude tiny placeholders
            return cand
    return None


def label_window(start: int, end: int,
                   shots: list, goals: list) -> tuple[str, int, str]:
    """Classify (start, end). Returns (label, n_overlap, team)."""
    overlapping_goals = [g for g in goals if g[0] < end and g[1] > start]
    overlapping_shots = [s for s in shots if s[0] < end and s[1] > start]
    if overlapping_goals:
        g = overlapping_goals[0]
        return "goal", len(overlapping_goals) + len(overlapping_shots), g[2]
    if overlapping_shots:
        s = overlapping_shots[0]
        return "shot_save", len(overlapping_shots), s[2]
    return "no_event", 0, ""


def video_duration_sec(path: Path) -> int:
    """Use ffprobe."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True, timeout=30,
        ).strip()
        return int(float(out))
    except Exception:
        return 0


def sample_random_negatives(vid_key: str, video_path: Path,
                              shots: list, goals: list,
                              taken_windows: list[tuple[int, int]],
                              n: int = 30,
                              clip_dur: int = 20,
                              seed: int = 42) -> list[LabeledClip]:
    """Randomly sample clips that DON'T overlap any GT event AND don't
    overlap any stage-1 window we've already labeled. These are pure
    no-event examples from neutral hockey time."""
    rng = random.Random(hash(vid_key) ^ seed)
    dur = video_duration_sec(video_path)
    if dur <= 60 + clip_dur:
        return []

    # Build forbidden ranges = GT events ± 10s + stage-1 windows
    forbidden = []
    for s, e, _ in shots + goals:
        forbidden.append((max(0, s - 10), e + 10))
    for s, e in taken_windows:
        forbidden.append((max(0, s - 5), e + 5))
    forbidden.sort()
    # Merge overlapping forbidden intervals
    merged: list[list[int]] = []
    for lo, hi in forbidden:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    out = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        start = rng.randint(60, dur - clip_dur - 60)
        end = start + clip_dur
        # Check forbidden overlap
        conflict = any(start < f_hi and end > f_lo for f_lo, f_hi in merged)
        if conflict: continue
        out.append(LabeledClip(
            vID=vid_key, hudl_id=VID_TO_HUDL.get(vid_key) or _hudl_from_filename(vid_key),
            video_path=str(video_path), start_sec=start, end_sec=end,
            label="no_event", overlapping_gt_count=0, gt_team="",
            source="random_neg",
        ))
    return out


def _hudl_from_filename(stem: str) -> int:
    """If stem is a hudl numeric ID (e.g. '2073809'), return it as int.
    Else return -1."""
    if stem.isdigit():
        return int(stem)
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR / "manifest.json")
    ap.add_argument("--negatives-per-game", type=int, default=30,
                     help="Random no-event clips to sample per game.")
    ap.add_argument("--include-fp-windows", action="store_true", default=True,
                     help="Include stage-1 false-positive windows (Pro said no event "
                          "but stage 1 fired) as no_event labels. Default on.")
    args = ap.parse_args()

    # Discover all videos (both youtube-IDed and hudl-IDed)
    all_vids: list[str] = []
    for p in sorted(VIDEOS_DIR.glob("full_*.mp4")):
        stem = p.stem.replace("full_", "")
        if stem not in all_vids:
            all_vids.append(stem)
    print(f"Found {len(all_vids)} videos in {VIDEOS_DIR}", file=sys.stderr)

    clips: list[LabeledClip] = []
    label_counts = Counter()
    per_game_stats = []
    no_video_count = 0
    no_seg_count = 0
    no_gt_count = 0

    # GT-centered clip duration (matches typical stage-1 window).
    # The model needs to see the event somewhere in the clip — not always
    # centered — so we add a small random offset jitter at sample time.
    GT_CLIP_DUR = 30

    def _exists(start, end, vid_key, tolerance=10):
        """Check if a window roughly overlapping (start, end) already exists
        for this vID — avoids near-duplicate clips."""
        for c in clips:
            if c.vID != vid_key: continue
            if abs(c.start_sec - start) < tolerance and abs(c.end_sec - end) < tolerance:
                return True
        return False

    for vid_key in all_vids:
        video_path = find_video_path(vid_key)
        if video_path is None:
            no_video_count += 1
            continue
        # Resolve hudl_id
        hudl_id = VID_TO_HUDL.get(vid_key) or _hudl_from_filename(vid_key)
        if hudl_id <= 0:
            print(f"  {vid_key}: no hudl mapping — skip", file=sys.stderr)
            continue
        shots, goals = load_gt(hudl_id)
        if not shots and not goals:
            no_gt_count += 1
            continue
        seg_path = find_seg_path(vid_key)
        if seg_path is None:
            seg_path = find_seg_path(str(hudl_id))
        windows = load_seg_windows(seg_path) if seg_path else []
        if not windows:
            no_seg_count += 1
        game_counts = Counter()

        # SOURCE 1 — stage-1 windows (matches production distribution).
        # Each window gets its GT-overlap label.
        for (s, e) in windows:
            label, n_overlap, team = label_window(s, e, shots, goals)
            clips.append(LabeledClip(
                vID=vid_key, hudl_id=hudl_id, video_path=str(video_path),
                start_sec=s, end_sec=e, label=label,
                overlapping_gt_count=n_overlap, gt_team=team,
                source="stage1",
            ))
            game_counts[label] += 1
            label_counts[label] += 1

        # SOURCE 2 — GT-centered clips for every GT event.
        # Ensures the goal class has enough positive examples (only ~33
        # goal events fall in stage-1 windows in our current 14-vID set;
        # this lifts it to all 145 across 33 games).
        for ev_start, ev_end, team in goals:
            mid = (ev_start + ev_end) // 2
            cs = max(0, mid - GT_CLIP_DUR // 2)
            ce = cs + GT_CLIP_DUR
            if _exists(cs, ce, vid_key, tolerance=15):
                continue
            clips.append(LabeledClip(
                vID=vid_key, hudl_id=hudl_id, video_path=str(video_path),
                start_sec=cs, end_sec=ce, label="goal",
                overlapping_gt_count=1, gt_team=team, source="gt_centered",
            ))
            game_counts["goal"] += 1
            label_counts["goal"] += 1
        for ev_start, ev_end, team in shots:
            # Skip if this shot's start is within any GT goal (already added)
            if any(g[0] <= ev_start <= g[1] for g in goals):
                continue
            mid = (ev_start + ev_end) // 2
            cs = max(0, mid - GT_CLIP_DUR // 2)
            ce = cs + GT_CLIP_DUR
            if _exists(cs, ce, vid_key, tolerance=15):
                continue
            clips.append(LabeledClip(
                vID=vid_key, hudl_id=hudl_id, video_path=str(video_path),
                start_sec=cs, end_sec=ce, label="shot_save",
                overlapping_gt_count=1, gt_team=team, source="gt_centered",
            ))
            game_counts["shot_save"] += 1
            label_counts["shot_save"] += 1

        # SOURCE 3 — random negatives from non-event time
        neg_clips = sample_random_negatives(
            vid_key, video_path, shots, goals, windows,
            n=args.negatives_per_game,
        )
        for c in neg_clips:
            clips.append(c)
            game_counts["no_event"] += 1
            label_counts["no_event"] += 1
        per_game_stats.append((vid_key, hudl_id, len(windows),
                                len(neg_clips), dict(game_counts),
                                len(shots), len(goals)))

    print(f"\nLoaded {len(clips)} labeled clips from {len(per_game_stats)} games", file=sys.stderr)
    print(f"  videos missing:    {no_video_count}", file=sys.stderr)
    print(f"  GT missing:        {no_gt_count}", file=sys.stderr)
    print(f"  stage-1 missing:   {no_seg_count}", file=sys.stderr)
    print(f"\nLabel distribution:", file=sys.stderr)
    for label, n in label_counts.most_common():
        print(f"  {label:12} {n:>5}", file=sys.stderr)

    print(f"\nPer-game summary:", file=sys.stderr)
    for vid_key, hudl_id, n_windows, n_neg, counts, n_shots_gt, n_goals_gt in per_game_stats:
        g = counts.get("goal", 0)
        ss = counts.get("shot_save", 0)
        ne = counts.get("no_event", 0)
        print(f"  {vid_key[:14]:<14}  hudl={hudl_id}  "
              f"win={n_windows:>3}  neg={n_neg:>2}  "
              f"goal={g:>2} shot={ss:>3} ne={ne:>3}  "
              f"(GT: {n_shots_gt} shots / {n_goals_gt} goals)",
              file=sys.stderr)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_games":        len(per_game_stats),
        "label_counts":   dict(label_counts),
        "clips":          [asdict(c) for c in clips],
    }, indent=2))
    print(f"\nwrote {out_path}  ({len(clips)} clips, {out_path.stat().st_size//1024} KB)",
          file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
