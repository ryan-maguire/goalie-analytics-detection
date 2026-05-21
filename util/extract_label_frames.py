"""
Extract candidate frames for HockeyAI fine-tuning on a new `shot` class.

Three frame types extracted into data/labels/images/:
  - pos:     2 frames per opponent-team GT 'Shots' window, at start+1
             and start+2 seconds (Hudl convention puts the shot ~1s in)
  - neg:     1 random non-shot frame per shot (balanced-ish)
  - hardneg: 1 frame per FP from the most recent *_fp_trace.tsv,
             at the FP window's midpoint

Filename convention: `{vID}_{pos|neg|hardneg}_{t:05d}.jpg`. The
companion util/prelabel_frames.py reads this naming to know which
frames are positives (only positives need a manual `shot` bbox).

GT filtering uses eval's fuzzy team-name matcher so customer-file
short names like 'Jr Flyers 19U' match Hudl long names like
'Philadelphia Jr. Flyers 19U AA'.

Usage:
    python3 util/extract_label_frames.py \\
        --videos-dir data/videos \\
        --gt-dir data/ground_truth \\
        --customers data/customers/CUST000048.json data/customers/CUST000031.json \\
        --fp-trace data/output/evals/eval_<latest>_fp_trace.tsv \\
        --out-dir data/labels/images \\
        [--vIDs SX5xNJlh6eQ bfEKgtOIkQU ...]
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_cv_seg_output import VID_TO_HUDL, _team_names_match


def _load_vid_to_opponent_team(customer_paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in customer_paths:
        with open(p) as f:
            cfg = json.load(f)
        for rec in cfg:
            vid = str(rec.get("vID", "")).strip()
            opp = rec.get("opponentGoalieTeamName") or ""
            if vid:
                out[vid] = opp
    return out


def _load_gt_shot_windows(gt_csv: str, target_team: str) -> list[tuple[int, int]]:
    """Return [(start_sec, end_sec)] for opponent-team Shots rows."""
    windows = []
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            if row.get("action", "").strip().lower() != "shots":
                continue
            team = row.get("team", "").strip()
            if target_team and not _team_names_match(team, target_team):
                continue
            try:
                s = int(float(row["start"])); e = int(float(row["end"]))
            except (ValueError, KeyError):
                continue
            windows.append((s, e))
    return windows


def _load_fp_midpoints(fp_trace_tsv: str) -> dict[str, list[int]]:
    """Return {vID: [t1, t2, ...]} — midpoints of FP windows."""
    fps: dict[str, list[int]] = {}
    if not os.path.exists(fp_trace_tsv):
        return fps
    with open(fp_trace_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            vid = row.get("vID", "").strip()
            try:
                ps = float(row["pred_start"]); pe = float(row["pred_end"])
            except (ValueError, KeyError):
                continue
            if not vid:
                continue
            fps.setdefault(vid, []).append(int((ps + pe) / 2))
    return fps


def _extract_frame(cap, t_sec: float):
    cap.set(_CV2_PROP_POS_MSEC, t_sec * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="data/videos")
    ap.add_argument("--gt-dir",     default="data/ground_truth")
    ap.add_argument("--customers",  nargs="+", required=True)
    ap.add_argument("--fp-trace",   default=None,
                    help="path to eval_<ts>_fp_trace.tsv for hard negatives. "
                         "If omitted, uses the most recent in data/output/evals/")
    ap.add_argument("--out-dir",    default="data/labels/images")
    ap.add_argument("--vIDs",       nargs="*", default=None,
                    help="restrict to these vIDs (default: all in VID_TO_HUDL)")
    ap.add_argument("--pos-offsets-sec", type=int, nargs="+", default=[1, 2],
                    help="seconds to offset from GT shot start for positive frames")
    ap.add_argument("--neg-per-shot", type=float, default=0.5,
                    help="random negatives per positive shot (0.5 → ~1 neg per 2 pos)")
    ap.add_argument("--max-hardnegs-per-video", type=int, default=12,
                    help="cap hard-negative frames per video (avoids FP-heavy "
                         "videos dominating the dataset)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import cv2
    global _CV2_PROP_POS_MSEC
    _CV2_PROP_POS_MSEC = cv2.CAP_PROP_POS_MSEC

    # Resolve FP trace
    if args.fp_trace is None:
        evals = sorted(Path("data/output/evals").glob("*_fp_trace.tsv"))
        if evals:
            args.fp_trace = str(evals[-1])
            print(f"Using FP trace: {args.fp_trace}", file=sys.stderr)
    fp_midpoints = _load_fp_midpoints(args.fp_trace) if args.fp_trace else {}
    print(f"Loaded FPs for {len(fp_midpoints)} videos", file=sys.stderr)

    opp_team_of = _load_vid_to_opponent_team(args.customers)
    print(f"Loaded opponent-team for {len(opp_team_of)} vIDs", file=sys.stderr)

    vids = args.vIDs or sorted(VID_TO_HUDL.keys())
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    totals = {"pos": 0, "neg": 0, "hardneg": 0}
    for vid in vids:
        video_path = os.path.join(args.videos_dir, f"{vid}.mp4")
        if not os.path.exists(video_path):
            print(f"  skip {vid}: no video at {video_path}", file=sys.stderr); continue
        hudl = VID_TO_HUDL.get(vid)
        if not hudl:
            print(f"  skip {vid}: no hudl mapping", file=sys.stderr); continue
        opp_team = opp_team_of.get(vid, "")
        gt_csv = os.path.join(args.gt_dir, f"gt_{hudl}.csv")
        if not os.path.exists(gt_csv):
            print(f"  skip {vid}: no GT at {gt_csv}", file=sys.stderr); continue
        shots = _load_gt_shot_windows(gt_csv, opp_team)
        if not shots:
            print(f"  skip {vid}: 0 shots in GT", file=sys.stderr); continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  skip {vid}: cannot open video", file=sys.stderr); continue
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = int(n_frames / fps_v)
        print(f"  {vid}: {duration_sec}s, {len(shots)} shots", file=sys.stderr)

        # Positives — 2 frames per shot at configured offsets
        n_pos = 0
        shot_seconds: set[int] = set()
        for s, e in shots:
            for off in args.pos_offsets_sec:
                t = s + off
                if t >= duration_sec:
                    continue
                shot_seconds.add(t)
                fname = f"{vid}_pos_{t:05d}.jpg"
                out_path = os.path.join(args.out_dir, fname)
                if os.path.exists(out_path):
                    n_pos += 1; continue
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if ok:
                    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    n_pos += 1
        # Track all GT shot seconds (including the full window) for neg sampling
        gt_full_seconds: set[int] = set()
        for s, e in shots:
            for t in range(max(0, s - 3), min(duration_sec, e + 3)):
                gt_full_seconds.add(t)

        # Random negatives
        n_neg_target = int(len(shots) * args.neg_per_shot)
        n_neg = 0
        attempts = 0
        while n_neg < n_neg_target and attempts < n_neg_target * 20:
            attempts += 1
            t = rng.randint(0, max(0, duration_sec - 1))
            if t in gt_full_seconds:
                continue
            fname = f"{vid}_neg_{t:05d}.jpg"
            out_path = os.path.join(args.out_dir, fname)
            if os.path.exists(out_path):
                n_neg += 1; continue
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                n_neg += 1

        # Hard negatives from FP trace, capped per video
        n_hardneg = 0
        fps_for_vid = fp_midpoints.get(vid, [])
        rng.shuffle(fps_for_vid)
        for t in fps_for_vid[:args.max_hardnegs_per_video]:
            if t < 0 or t >= duration_sec:
                continue
            # Skip if this midpoint happens to coincide with a real shot second
            if t in gt_full_seconds:
                continue
            fname = f"{vid}_hardneg_{t:05d}.jpg"
            out_path = os.path.join(args.out_dir, fname)
            if os.path.exists(out_path):
                n_hardneg += 1; continue
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                n_hardneg += 1

        cap.release()
        print(f"    pos={n_pos}  neg={n_neg}  hardneg={n_hardneg}",
              file=sys.stderr)
        totals["pos"]     += n_pos
        totals["neg"]     += n_neg
        totals["hardneg"] += n_hardneg

    print(f"\nTotals: pos={totals['pos']}  neg={totals['neg']}  "
          f"hardneg={totals['hardneg']}  "
          f"(grand total {sum(totals.values())} frames)", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
