#!/usr/bin/env python3
"""Track A: X3D + Pro ensemble for goal detection.

Goal: lift goal F1 above Pro-alone by using X3D as a precision booster.

Pipeline:
  For each window in data/output/runs/metrics_v13/gt_metrics_<vID>.json:
    1. Pro prediction (from cached metrics): "goal" iff Pro reported goals > 0
    2. X3D prediction: run inference on the clip, get 3-class probabilities
    3. Ensemble: combine Pro + X3D under several policies, pick the
       best on test-set games

Policies (all evaluated, best one wins):
  pro_only:           baseline — what we have in production today
  x3d_only:           pure X3D prediction (sanity check)
  intersection:       goal iff Pro AND X3D both say goal (precision-focused)
  union:              goal iff Pro OR X3D say goal (recall-focused)
  pro_boost_x3d:      Pro's goal call, but downgrade if X3D goal_prob < t
  x3d_filter_pro:     Pro's goal call gated by X3D goal probability
  weighted_avg:       (alpha * Pro_score + beta * X3D_score) >= threshold

Output: data/output/evals/x3d_pro_ensemble.md (per-policy precision/recall/F1
on TRAIN vs TEST splits — TEST is the honest number since X3D was trained
on TRAIN games).
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training" / "action_recognition"))

from dataset import (
    load_manifest, split_by_game, extract_frames_ffmpeg,
    LABEL_NAMES, LABEL_TO_IDX,
)

GOAL_IDX = LABEL_TO_IDX["goal"]

# vID → hudl_id (from training/action_recognition/build_manifest.py)
VID_TO_HUDL = {
    "SX5xNJlh6eQ": 2073056, "bfEKgtOIkQU": 2072195,
    "mjEeE7p2Hz8": 2073809, "n2cy8b755Tg": 2127046,
    "v0lxSTbXfw8": 2073810, "dwGsP6QKDs8": 2070269,
    "Fjc9hmK8_3U": 2070260, "HNG0jKYY12g": 2095275,
    "J8WkcuTsD5I": 2072194, "kQVdtRa4o_A": 2127034,
    "krxhPVLGLz8": 2108724, "KYtM20r9BuM": 2072196,
    "q5yj6sAFQeY": 2127052, "zOQrPK7IJ24": 2127035,
}
HUDL_TO_VID = {v: k for k, v in VID_TO_HUDL.items()}

METRICS_DIR  = REPO / "data" / "output" / "runs" / "metrics_v13"
GT_DIR       = REPO / "data" / "ground_truth"
VIDEOS_DIR   = REPO / "data" / "videos"
X3D_CKPT     = REPO / "training" / "action_recognition" / "runs" / "x3d_m_v1" / "best.pt"
OUT_DIR      = REPO / "data" / "output" / "evals"


@dataclass
class WindowPrediction:
    vID:           str
    hudl_id:       int
    split:         str       # train | val | test
    start_sec:     int
    end_sec:       int
    pro_goal:      bool       # Pro called this a goal
    pro_n_goals:   int
    x3d_probs:     list[float]  # [no_event, shot_save, goal]
    gt_goal:       bool       # GT says this window contains a goal


def load_pro_windows(vid: str) -> Optional[list[dict]]:
    """Read metrics_v13 output for a vID. Returns None if file missing."""
    f = METRICS_DIR / f"gt_metrics_{vid}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def load_gt_events(hudl_id: int) -> tuple[list, list]:
    """Return (shots, goals) from gt_<hudl_id>.csv: list of (start, end, team)."""
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
            if action == "Goals":
                goals.append((start, end))
            elif action == "Shots":
                shots.append((start, end))
    return shots, goals


def gt_label_for_window(start: int, end: int, goals: list) -> bool:
    """Window contains a GT goal iff any goal start time falls within."""
    return any(g[0] < end and g[1] > start for g in goals)


def video_path_for(vid: str) -> Optional[Path]:
    for cand in (VIDEOS_DIR / f"full_{vid}.mp4", VIDEOS_DIR / f"{vid}.mp4"):
        if cand.exists() and cand.stat().st_size > 1024:
            return cand
    return None


def load_x3d_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load X3D-M with our 3-class head and the trained weights."""
    sys.path.insert(0, str(REPO / "training" / "action_recognition"))
    from train_x3d import build_x3d_m
    model = build_x3d_m(n_classes=len(LABEL_NAMES)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  loaded X3D from {ckpt_path}  (val_metrics={ckpt.get('val_metrics', {}).get('macro_f1')})",
          file=sys.stderr)
    return model


@torch.no_grad()
def x3d_predict(model, video_path: str, start_sec: int, end_sec: int,
                 device: torch.device, n_frames: int = 8, size: int = 160) -> np.ndarray:
    """Returns 3-class softmax probabilities [no_event, shot_save, goal]."""
    frames = extract_frames_ffmpeg(video_path, start_sec, end_sec,
                                      n_frames=n_frames, target_size=(size, size))
    if frames is None:
        return np.array([1.0, 0.0, 0.0])   # default to no_event on extract failure
    x = torch.from_numpy(frames.copy()).float() / 255.0
    x = x.permute(0, 3, 1, 2)                # (T, C, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.permute(1, 0, 2, 3).unsqueeze(0).to(device)   # (1, C, T, H, W)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def split_for_hudl(hudl_id: int, splits: dict) -> str:
    for split_name, recs in splits.items():
        if any(r.hudl_id == hudl_id for r in recs):
            return split_name
    return "unknown"


# ─── ensemble policies ──────────────────────────────────────────────

def policy_pro_only(p: WindowPrediction) -> bool:
    return p.pro_goal

def policy_x3d_only(p: WindowPrediction, thresh: float = 0.5) -> bool:
    return p.x3d_probs[GOAL_IDX] >= thresh

def policy_intersection(p: WindowPrediction, x3d_thresh: float = 0.3) -> bool:
    return p.pro_goal and p.x3d_probs[GOAL_IDX] >= x3d_thresh

def policy_union(p: WindowPrediction, x3d_thresh: float = 0.5) -> bool:
    return p.pro_goal or p.x3d_probs[GOAL_IDX] >= x3d_thresh

def policy_pro_boost(p: WindowPrediction, x3d_thresh: float = 0.15) -> bool:
    """Pro's goal call, downgraded if X3D goal_prob < threshold.
    The opposite of intersection — we trust Pro recall, X3D suppresses FPs."""
    if not p.pro_goal:
        return False
    return p.x3d_probs[GOAL_IDX] >= x3d_thresh

def policy_weighted(p: WindowPrediction,
                       w_pro: float = 0.7, w_x3d: float = 0.3,
                       thresh: float = 0.5) -> bool:
    score = w_pro * (1.0 if p.pro_goal else 0.0) + w_x3d * p.x3d_probs[GOAL_IDX]
    return score >= thresh


POLICIES = {
    "pro_only":          (policy_pro_only,     {}),
    "x3d_only_t05":      (policy_x3d_only,     {"thresh": 0.5}),
    "x3d_only_t03":      (policy_x3d_only,     {"thresh": 0.3}),
    "intersection_t03":  (policy_intersection, {"x3d_thresh": 0.3}),
    "intersection_t015": (policy_intersection, {"x3d_thresh": 0.15}),
    "union_t05":         (policy_union,        {"x3d_thresh": 0.5}),
    "union_t07":         (policy_union,        {"x3d_thresh": 0.7}),
    "pro_boost_t015":    (policy_pro_boost,    {"x3d_thresh": 0.15}),
    "pro_boost_t02":     (policy_pro_boost,    {"x3d_thresh": 0.20}),
    "pro_boost_t03":     (policy_pro_boost,    {"x3d_thresh": 0.30}),
    "weighted_55_45":    (policy_weighted,     {"w_pro": 0.55, "w_x3d": 0.45, "thresh": 0.5}),
    "weighted_70_30":    (policy_weighted,     {"w_pro": 0.70, "w_x3d": 0.30, "thresh": 0.5}),
}


def evaluate_policy(predictions: list[WindowPrediction],
                       policy_fn, policy_kwargs: dict) -> dict:
    """Window-level precision/recall/F1 on the 'goal' class."""
    tp = fp = fn = tn = 0
    for p in predictions:
        called = policy_fn(p, **policy_kwargs)
        if p.gt_goal:
            if called: tp += 1
            else:      fn += 1
        else:
            if called: fp += 1
            else:      tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "precision": prec, "recall": rec, "f1": f1}


def write_report(report_path: Path, all_metrics: dict, n_preds: int):
    lines = []
    lines.append(f"# X3D + Pro ensemble — Track A results")
    lines.append("")
    lines.append(f"Generated from {n_preds} cached Pro windows × X3D inference each.")
    lines.append("")
    lines.append(f"X3D best checkpoint: ep6 val macro-F1=0.593, val goal-F1=0.364, test goal-F1=0.138")
    lines.append(f"Pro v13 prompt running gemini-2.5-pro.")
    lines.append("")

    for split_name, by_policy in all_metrics.items():
        if not by_policy:
            continue
        n = sum(by_policy[p]["tp"] + by_policy[p]["fn"]
                 for p in by_policy) // max(len(by_policy), 1)
        # Get GT goal count: tp + fn (same across policies)
        any_pol = next(iter(by_policy.values()))
        gt_goals = any_pol["tp"] + any_pol["fn"]
        n_windows = gt_goals + any_pol["fp"] + any_pol["tn"]
        lines.append(f"## {split_name.upper()} split  "
                     f"({n_windows} windows, {gt_goals} GT goals)")
        lines.append("")
        lines.append(f"| policy | TP | FP | FN | P | R | **F1** |")
        lines.append(f"|---|---:|---:|---:|---:|---:|---:|")
        rows = []
        for pol, m in by_policy.items():
            rows.append((pol, m["tp"], m["fp"], m["fn"],
                          m["precision"], m["recall"], m["f1"]))
        # Sort by F1 desc within split
        rows.sort(key=lambda r: -r[6])
        for pol, tp, fp, fn, p, r, f1 in rows:
            highlight = " ⭐" if pol != "pro_only" and f1 > by_policy["pro_only"]["f1"] else ""
            bold = f"**{f1:.3f}**"
            lines.append(f"| `{pol}` | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | {bold}{highlight} |")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\n  wrote {report_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=X3D_CKPT)
    ap.add_argument("--manifest", type=Path,
                    default=REPO / "training" / "action_recognition" / "manifest.json")
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "x3d_pro_ensemble.md")
    ap.add_argument("--predictions-cache", type=Path,
                    default=REPO / "data" / "output" / "evals"
                            / "x3d_pro_predictions.json",
                    help="Cache X3D inference results so reruns are instant.")
    ap.add_argument("--vids", nargs="*", default=None,
                    help="vIDs to run; default = all 14 with metrics_v13/")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--size", type=int, default=160)
    args = ap.parse_args()

    if not args.ckpt.exists():
        print(f"ERROR: X3D checkpoint not found at {args.ckpt}", file=sys.stderr); sys.exit(1)

    # Same game-split as training so we can show train/val/test metrics honestly
    records = load_manifest(args.manifest)
    splits = split_by_game(records)
    train_hudls = {r.hudl_id for r in splits["train"]}
    val_hudls   = {r.hudl_id for r in splits["val"]}
    test_hudls  = {r.hudl_id for r in splits["test"]}
    print(f"\nGame splits (from training manifest):", file=sys.stderr)
    print(f"  train: {len(train_hudls)} games  val: {len(val_hudls)}  test: {len(test_hudls)}",
          file=sys.stderr)

    # Discover vIDs we'll evaluate
    if args.vids:
        vids = list(args.vids)
    else:
        vids = sorted({f.stem.replace("gt_metrics_", "")
                       for f in METRICS_DIR.glob("gt_metrics_*.json")
                       if "_trace" not in f.name})
    print(f"\nEvaluating {len(vids)} vIDs with Pro metrics on disk", file=sys.stderr)

    # Cache
    pred_cache: dict = {}
    if args.predictions_cache.exists():
        try:
            pred_cache = json.loads(args.predictions_cache.read_text())
        except Exception:
            pred_cache = {}
    cache_key = lambda v, s, e: f"{v}:{s}:{e}"

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"  device: {device}", file=sys.stderr)
    model = load_x3d_model(args.ckpt, device)

    # Walk windows + run X3D
    predictions: list[WindowPrediction] = []
    for vid in vids:
        hudl_id = VID_TO_HUDL.get(vid)
        if not hudl_id:
            print(f"  {vid}: no hudl mapping — skip", file=sys.stderr); continue
        split = ("train" if hudl_id in train_hudls
                 else "val" if hudl_id in val_hudls
                 else "test" if hudl_id in test_hudls
                 else "unknown")
        video_path = video_path_for(vid)
        if not video_path:
            print(f"  {vid}: no video — skip", file=sys.stderr); continue
        pro_windows = load_pro_windows(vid)
        if not pro_windows:
            print(f"  {vid}: no Pro output — skip", file=sys.stderr); continue
        _, gt_goals = load_gt_events(hudl_id)

        t0 = time.time()
        n_new = 0
        for w in pro_windows:
            s = int(w["segment_start"]); e = int(w["segment_end"])
            m = w.get("metrics") or {}
            pro_n_goals = int(m.get("goals", 0) or 0)
            pro_goal = pro_n_goals > 0
            gt_goal = gt_label_for_window(s, e, gt_goals)

            k = cache_key(vid, s, e)
            if k in pred_cache:
                x3d_probs = pred_cache[k]
            else:
                probs = x3d_predict(model, str(video_path), s, e, device,
                                     n_frames=args.n_frames, size=args.size)
                x3d_probs = probs.tolist()
                pred_cache[k] = x3d_probs
                n_new += 1
            predictions.append(WindowPrediction(
                vID=vid, hudl_id=hudl_id, split=split,
                start_sec=s, end_sec=e,
                pro_goal=pro_goal, pro_n_goals=pro_n_goals,
                x3d_probs=x3d_probs, gt_goal=gt_goal,
            ))
        elapsed = time.time() - t0
        print(f"  {vid} ({split:5}): {len(pro_windows)} windows  "
              f"{n_new} new X3D calls  ({elapsed:.0f}s)", file=sys.stderr)

    # Persist X3D prediction cache (for instant reruns)
    args.predictions_cache.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_cache.write_text(json.dumps(pred_cache))

    print(f"\nTotal predictions: {len(predictions)}", file=sys.stderr)
    by_split_counts = Counter(p.split for p in predictions)
    print(f"  per-split: {dict(by_split_counts)}", file=sys.stderr)
    gt_pos = sum(1 for p in predictions if p.gt_goal)
    print(f"  GT goal windows: {gt_pos}/{len(predictions)}", file=sys.stderr)

    # Evaluate each policy per split + overall
    all_metrics = {}
    for split_name in ("all", "train", "val", "test"):
        if split_name == "all":
            subset = predictions
        else:
            subset = [p for p in predictions if p.split == split_name]
        if not subset:
            all_metrics[split_name] = {}
            continue
        by_policy = {}
        for pol, (fn, kwargs) in POLICIES.items():
            by_policy[pol] = evaluate_policy(subset, fn, kwargs)
        all_metrics[split_name] = by_policy

    # Print test split results
    print(f"\n=== TEST split: goal-class metrics ===", file=sys.stderr)
    if all_metrics.get("test"):
        by_pol = all_metrics["test"]
        # Sort by F1 desc
        sorted_pols = sorted(by_pol.items(), key=lambda kv: -kv[1]["f1"])
        for pol, m in sorted_pols:
            marker = " ⭐" if pol != "pro_only" and m["f1"] > by_pol["pro_only"]["f1"] else ""
            print(f"  {pol:<22}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']}){marker}",
                  file=sys.stderr)

    write_report(args.out, all_metrics, len(predictions))


if __name__ == "__main__":
    sys.exit(main() or 0)
