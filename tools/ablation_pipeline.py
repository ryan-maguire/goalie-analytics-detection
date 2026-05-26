"""End-to-end metrics_seg ablation runner.

ONE SCRIPT that runs everything needed to compare v13 baseline against
each v14 improvement (in isolation and combined), across multiple
videos, and produces a single comparison report.

  For each vID in --vIDs:
    1. Run cv_seg (Step 1) if its output doesn't exist
    2. Run eval/eval_cv_seg_output.py (Step 1 eval)
    3. For each variant in VARIANTS:
         a. Run metrics_seg with that variant's flags (skip if output exists)
         b. Run eval/eval_metric_seg_output.py against that output
    4. (resumable — skipping anything already on disk)

  After all videos × all variants finish:
    - Parse every eval text/JSON output
    - Build a comparison matrix (variant × video × metric)
    - Print to stdout AND write ablation_report.md

Cost warning: each metrics_seg run = ~$1-3 in Gemini Pro calls.
Default 3 videos × 5 variants × ~$2 = ~$30. Override --variants or
--vIDs to scope down. Use --dry-run to plan without running anything.

Usage:
    # Dry-run plan (no Gemini, no money)
    python3 tools/ablation_pipeline.py --dry-run

    # Run defaults (3 vIDs from CUST000048, 5 variants)
    python3 tools/ablation_pipeline.py

    # Custom set
    python3 tools/ablation_pipeline.py \\
        --vIDs mjEeE7p2Hz8 SX5xNJlh6eQ bfEKgtOIkQU \\
        --customID CUST000048 \\
        --variants v13 v14_prefilter v14_all
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]


# ─── Defaults ─────────────────────────────────────────────────────────
# Per-vID customer mapping (extracted from eval/eval_cv_seg_output.py
# VID_TO_HUDL grouping comments).
VID_TO_CUST: dict[str, str] = {
    # CUST000048
    "SX5xNJlh6eQ": "CUST000048",
    "bfEKgtOIkQU": "CUST000048",
    "mjEeE7p2Hz8": "CUST000048",
    "n2cy8b755Tg": "CUST000048",
    "v0lxSTbXfw8": "CUST000048",
    # CUST000031
    "dwGsP6QKDs8": "CUST000031",
    "Fjc9hmK8_3U": "CUST000031",
    "HNG0jKYY12g": "CUST000031",
    "J8WkcuTsD5I": "CUST000031",
    "kQVdtRa4o_A": "CUST000031",
    "krxhPVLGLz8": "CUST000031",
    "KYtM20r9BuM": "CUST000031",
    "q5yj6sAFQeY": "CUST000031",
    "zOQrPK7IJ24": "CUST000031",
}

# All 14 vIDs across both customers — used for scaled experiments
ALL_PAIRED_VIDS = list(VID_TO_CUST.keys())

# Smaller default for quick smoke tests
DEFAULT_VIDS = ALL_PAIRED_VIDS[:3]

DEFAULT_CUST = "CUST000048"   # only used as fallback for unknown vIDs

# Each variant is (name, extra_flags_for_metrics_seg)
VARIANTS: dict[str, list[str]] = {
    "v13": [],   # baseline — no v14 flags at all
    "v14_prefilter": [
        "--prefilter-threshold", "0.30",
        "--probs-dir-yolo",  str(REPO / "runs" / "yolo_curve_n16"  / "probs"),
        "--probs-dir-audio", str(REPO / "runs" / "audio_curve_n16" / "probs"),
    ],
    "v14_context": [
        "--use-context",
        "--probs-dir-yolo",  str(REPO / "runs" / "yolo_curve_n16"  / "probs"),
        "--probs-dir-audio", str(REPO / "runs" / "audio_curve_n16" / "probs"),
        "--audio-features-dir", str(REPO / "data" / "output" / "audio_features"),
    ],
    "v14_ensemble": [
        "--goal-ensemble",
        "--probs-dir-yolo",  str(REPO / "runs" / "yolo_curve_n16"  / "probs"),
        "--probs-dir-audio", str(REPO / "runs" / "audio_curve_n16" / "probs"),
    ],
    "v14_all": [
        "--prefilter-threshold", "0.30",
        "--use-context",
        "--goal-ensemble",
        "--probs-dir-yolo",  str(REPO / "runs" / "yolo_curve_n16"  / "probs"),
        "--probs-dir-audio", str(REPO / "runs" / "audio_curve_n16" / "probs"),
        "--audio-features-dir", str(REPO / "data" / "output" / "audio_features"),
    ],
}


@dataclass
class RunResult:
    """One (vID × variant) result."""
    vid: str
    variant: str
    # Step 2 metric eval
    goal_strict_p:  Optional[float] = None
    goal_strict_r:  Optional[float] = None
    goal_strict_f1: Optional[float] = None
    goal_unfilt_f1: Optional[float] = None
    predicted_goals:  Optional[int] = None
    actual_goals:     Optional[int] = None
    predicted_shots:  Optional[int] = None
    actual_shots:     Optional[int] = None
    shot_mae:    Optional[float] = None
    shot_e2e_f1: Optional[float] = None
    shot_inwin_f1: Optional[float] = None
    n_windows: Optional[int] = None
    # Diagnostics
    metrics_path:    Optional[Path] = None
    eval_path:       Optional[Path] = None
    notes:           list[str] = field(default_factory=list)


# ─── Subprocess helpers ───────────────────────────────────────────────
def sh(cmd: list, *, check: bool = True, dry_run: bool = False) -> int:
    pretty = " ".join(str(c) for c in cmd)
    print(f"  $ {pretty}", flush=True)
    if dry_run:
        return 0
    rc = subprocess.call([str(c) for c in cmd])
    if rc != 0 and check:
        raise RuntimeError(f"command failed (rc={rc}): {pretty}")
    return rc


def cust_for(vid: str, fallback: str = DEFAULT_CUST) -> str:
    """Look up the customer ID for a vID; fall back if not in map."""
    return VID_TO_CUST.get(vid, fallback)


def run_cv_seg(vid: str, cust: str, *, dry_run: bool) -> Path:
    out_dir = REPO / "data" / "output" / "runs" / "cv_seg"
    out_path = out_dir / f"gt_seg_{vid}.json"
    if out_path.exists() and not dry_run:
        print(f"  [cv_seg] {vid}: cached → skip")
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = REPO / "data" / "videos" / f"{vid}.mp4"
    if not video_path.exists():
        # Try full_<vid>.mp4 fallback
        video_path = REPO / "data" / "videos" / f"full_{vid}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"no video for {vid} (tried .mp4 and full_*.mp4)")
    sh([sys.executable, "-m", "cv_seg",
        "--vID", vid, "--customID", cust,
        "--local-video", str(video_path),
        "--output-dir", str(out_dir),
        "--no-gcs"], dry_run=dry_run)
    return out_path


def run_cv_seg_eval(vids: list[str], *, dry_run: bool) -> Path:
    eval_dir = REPO / "data" / "output" / "evals" / "cv_seg_ablation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, "eval/eval_cv_seg_output.py",
        "--vIDs", *vids,
        "--pred-dir", str(REPO / "data" / "output" / "runs" / "cv_seg"),
        "--gt-dir",   str(REPO / "data" / "ground_truth"),
        "--output-dir", str(eval_dir)], dry_run=dry_run, check=False)
    return eval_dir


def run_metrics_seg(vid: str, cust: str, variant: str, flags: list[str],
                      *, workers: int, dry_run: bool) -> Path:
    out_dir = REPO / "data" / "output" / "runs" / f"metrics_{variant}"
    out_path = out_dir / f"gt_metrics_{vid}.json"
    if out_path.exists() and not dry_run:
        print(f"  [metrics_seg/{variant}] {vid}: cached → skip")
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "metrics_seg/01_detect_segment_metrics.py",
            "--vID", vid, "--customID", cust,
            "--segments-dir", str(REPO / "data" / "output" / "runs" / "cv_seg"),
            "--local-video-dir", str(REPO / "data" / "videos"),
            "--output-dir", str(out_dir),
            "--no-gcs",
            "--workers", str(workers),
            *flags]
    # Common to all variants (for calibration logging)
    if "--gt-dir" not in flags:
        cmd += ["--gt-dir", str(REPO / "data" / "ground_truth")]
    sh(cmd, dry_run=dry_run)
    return out_path


def run_metric_eval(vid: str, variant: str, *, dry_run: bool) -> Path:
    out_dir = REPO / "data" / "output" / "evals" / f"metrics_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, "eval/eval_metric_seg_output.py",
        "--vIDs", vid,
        "--metrics-dir", str(REPO / "data" / "output" / "runs" / f"metrics_{variant}"),
        "--cv-seg-dir",  str(REPO / "data" / "output" / "runs" / "cv_seg"),
        "--gt-dir",      str(REPO / "data" / "ground_truth"),
        "--output-dir",  str(out_dir),
        "--no-gcs"], dry_run=dry_run, check=False)
    return out_dir


# ─── Eval parsing ─────────────────────────────────────────────────────
_RE_FLOAT = r"([0-9]+\.[0-9]+|nan)"
_RE_INT   = r"([0-9]+)"
_PATTERNS = {
    # GOAL DETECTION block
    "goal_strict_p":   re.compile(r"^\s*precision\s+" + _RE_FLOAT + r"\s+" + _RE_FLOAT, re.M),
    "goal_strict_r":   re.compile(r"^\s*recall\s+" + _RE_FLOAT + r"\s+" + _RE_FLOAT, re.M),
    "goal_strict_f1":  re.compile(r"^\s*F1\s+" + _RE_FLOAT + r"\s+" + _RE_FLOAT, re.M),
    "predicted_goals": re.compile(r"^\s*predicted goals\s+" + _RE_INT + r"\s+" + _RE_INT, re.M),
    "actual_goals":    re.compile(r"^\s*actual goals\s+" + _RE_INT + r"\s+" + _RE_INT, re.M),
    "predicted_shots": re.compile(r"^\s*predicted shots\s+" + _RE_INT + r"\s+" + _RE_INT, re.M),
    "actual_shots":    re.compile(r"^\s*actual shots\s+" + _RE_INT + r"\s+" + _RE_INT, re.M),
    "shot_mae":        re.compile(r"^\s*mean abs error \(MAE\)\s+" + _RE_FLOAT + r"\s+" + _RE_FLOAT, re.M),
    "n_windows":       re.compile(r"^\s*windows compared:\s+" + _RE_INT, re.M),
    "shot_e2e_f1":     re.compile(r"End-to-end \(P/R/F1\):\s+P=" + _RE_FLOAT
                                    + r"\s+R=" + _RE_FLOAT + r"\s+F1=" + _RE_FLOAT),
    "shot_inwin_f1":   re.compile(r"Within-coverage\s+\(P/R/F1\):\s+P=" + _RE_FLOAT
                                    + r"\s+R=" + _RE_FLOAT + r"\s+F1=" + _RE_FLOAT),
}


def parse_eval_text(txt: str) -> dict:
    """Extract key metrics from the eval_metric_seg_output.py text report."""
    out: dict = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(txt)
        if not m:
            continue
        if key in ("predicted_goals", "actual_goals",
                    "predicted_shots", "actual_shots", "n_windows"):
            out[key] = int(m.group(1))
        elif key in ("shot_e2e_f1", "shot_inwin_f1"):
            # P, R, F1 — keep F1
            out[key] = float(m.group(3))
        else:
            out[key] = float(m.group(1))
    return out


def find_latest_eval_txt(eval_dir: Path, vid_substr: str = "") -> Optional[Path]:
    """Return the most recent .txt eval report in eval_dir."""
    if not eval_dir.exists():
        return None
    candidates = list(eval_dir.glob("eval_*.txt"))
    if not candidates:
        return None
    if vid_substr:
        v = [c for c in candidates if vid_substr in c.read_text()]
        if v:
            return max(v, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ─── Report builder ───────────────────────────────────────────────────
def write_report(results: list[RunResult], out_path: Path) -> None:
    lines = ["# metrics_seg ablation report",
              "",
              f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
              ""]

    # Cross-tab: variant × vid → goal_strict_f1
    vids = sorted({r.vid for r in results})
    variants = sorted({r.variant for r in results})

    def cell(vid, variant, attr) -> str:
        r = next((r for r in results if r.vid == vid and r.variant == variant), None)
        if r is None:
            return "—"
        v = getattr(r, attr)
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    def add_table(title: str, attr: str) -> None:
        lines.append(f"## {title} — `{attr}`")
        lines.append("")
        hdr = "| variant | " + " | ".join(vids) + " | mean |"
        sep = "|" + "---|" * (len(vids) + 2)
        # Use extend() not += — `lines += [...]` would create a local
        # `lines` and trigger UnboundLocalError on the prior append().
        lines.extend([hdr, sep])
        for v in variants:
            cells = [cell(vid, v, attr) for vid in vids]
            try:
                vals = [float(c) for c in cells if c != "—"]
                mean = f"{sum(vals)/len(vals):.3f}" if vals else "—"
            except ValueError:
                mean = "—"
            lines.append(f"| {v} | " + " | ".join(cells) + f" | **{mean}** |")
        lines.append("")

    add_table("Goal F1 (STRICT)",  "goal_strict_f1")
    add_table("Goal precision (STRICT)", "goal_strict_p")
    add_table("Goal recall (STRICT)",    "goal_strict_r")
    add_table("Goal F1 (UNFILTERED)", "goal_unfilt_f1")
    add_table("Shot end-to-end F1", "shot_e2e_f1")
    add_table("Shot within-coverage F1", "shot_inwin_f1")
    add_table("Shot MAE (lower = better)", "shot_mae")
    add_table("Windows compared", "n_windows")

    # Per-video per-variant raw counts
    lines += ["## Raw per-(vID, variant) counts",
               "",
               "| vID | variant | n_windows | pred_goals | actual_goals | "
               "pred_shots | actual_shots | goal_F1 | shot_e2e_F1 |",
               "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.vid} | {r.variant} | "
            f"{r.n_windows or '—'} | {r.predicted_goals or '—'} | "
            f"{r.actual_goals or '—'} | {r.predicted_shots or '—'} | "
            f"{r.actual_shots or '—'} | "
            f"{(f'{r.goal_strict_f1:.3f}' if r.goal_strict_f1 is not None else '—')} | "
            f"{(f'{r.shot_e2e_f1:.3f}' if r.shot_e2e_f1 is not None else '—')} |"
        )

    # Winner per metric (by mean across vids)
    lines += ["", "## Winners (by mean across videos)",
               "", "| metric | best variant | mean |", "|---|---|---|"]
    for attr in ("goal_strict_f1", "shot_e2e_f1", "shot_inwin_f1"):
        per_variant: dict[str, list[float]] = {}
        for r in results:
            v = getattr(r, attr)
            if v is not None:
                per_variant.setdefault(r.variant, []).append(v)
        if not per_variant:
            lines.append(f"| {attr} | — | — |"); continue
        means = {k: sum(vs) / len(vs) for k, vs in per_variant.items()}
        best = max(means.items(), key=lambda kv: kv[1])
        lines.append(f"| {attr} | **{best[0]}** | {best[1]:.4f} |")
    # MAE is lower-better
    per_variant_mae: dict[str, list[float]] = {}
    for r in results:
        if r.shot_mae is not None:
            per_variant_mae.setdefault(r.variant, []).append(r.shot_mae)
    if per_variant_mae:
        means_mae = {k: sum(vs) / len(vs) for k, vs in per_variant_mae.items()}
        best_mae = min(means_mae.items(), key=lambda kv: kv[1])
        lines.append(f"| shot_mae (lower=better) | **{best_mae[0]}** | {best_mae[1]:.4f} |")

    lines += ["", "## Honest read",
               "",
               "- One game = ~7-10 goal events → 95% CI on goal F1 is ±0.30.",
               "- Aggregate across all videos before concluding.",
               "- Look at per-variant trends; ignore single-cell wobble.",
               "- Shot-count improvements (MAE, e2e F1) are more statistically reliable than goal F1."]

    out_path.write_text("\n".join(lines))


# ─── Main orchestration ───────────────────────────────────────────────
def check_availability(vid: str) -> dict:
    """Return what's missing/present for this vID. Used to warn the user
    when v14 variants will silently fall back to baseline behavior."""
    hudl_map = {  # local copy; the diff tool has the canonical one
        "SX5xNJlh6eQ": 2073056, "bfEKgtOIkQU": 2072195,
        "mjEeE7p2Hz8": 2073809, "n2cy8b755Tg": 2127046,
        "v0lxSTbXfw8": 2073810, "dwGsP6QKDs8": 2070269,
        "Fjc9hmK8_3U": 2070260, "HNG0jKYY12g": 2095275,
        "J8WkcuTsD5I": 2072194, "kQVdtRa4o_A": 2127034,
        "krxhPVLGLz8": 2108724, "KYtM20r9BuM": 2072196,
        "q5yj6sAFQeY": 2127052, "zOQrPK7IJ24": 2127035,
    }
    hudl = hudl_map.get(vid)
    res = {
        "has_video":  (REPO / "data" / "videos" / f"{vid}.mp4").exists()
                       or (REPO / "data" / "videos" / f"full_{vid}.mp4").exists(),
        "has_gt":     hudl is not None and
                       (REPO / "data" / "ground_truth" / f"gt_{hudl}.csv").exists(),
        "has_yolo_probs": (
            (REPO / "runs" / "yolo_curve_n16" / "probs" / f"{vid}.tsv").exists()
            or (hudl is not None and
                 (REPO / "runs" / "yolo_curve_n16" / "probs" / f"{hudl}.tsv").exists())
        ),
        "has_audio_probs": (
            (REPO / "runs" / "audio_curve_n16" / "probs" / f"{vid}.tsv").exists()
            or (hudl is not None and
                 (REPO / "runs" / "audio_curve_n16" / "probs" / f"{hudl}.tsv").exists())
        ),
        "has_audio_features": (
            (REPO / "data" / "output" / "audio_features" / f"{vid}.tsv").exists()
            or (hudl is not None and
                 (REPO / "data" / "output" / "audio_features" / f"{hudl}.tsv").exists())
        ),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vIDs", nargs="+", default=DEFAULT_VIDS,
                    help=f"video IDs to run (default: {DEFAULT_VIDS})")
    ap.add_argument("--all-paired", action="store_true",
                    help=f"shortcut: use all {len(ALL_PAIRED_VIDS)} known paired vIDs across both customers")
    ap.add_argument("--customID", default=None,
                    help="DEPRECATED — kept for backwards compat. Customer is now "
                         "looked up per-vID via VID_TO_CUST. Pass to override the "
                         "fallback for unknown vIDs.")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                    help=f"variant names to test (default: all of {list(VARIANTS)})")
    ap.add_argument("--workers", type=int, default=2,
                    help="metrics_seg per-video Gemini parallelism")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running anything (no $$)")
    ap.add_argument("--report-path", type=Path,
                    default=REPO / "data" / "output" / "evals" / "ablation_report.md")
    ap.add_argument("--yes", action="store_true",
                    help="skip the cost-confirmation prompt")
    args = ap.parse_args()

    os.chdir(REPO)

    if args.all_paired:
        args.vIDs = ALL_PAIRED_VIDS

    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        print(f"ERROR: unknown variants: {unknown}. "
              f"Available: {list(VARIANTS)}", file=sys.stderr)
        return 2

    # ── Availability check ───────────────────────────────────────
    print("=" * 78)
    print("AVAILABILITY CHECK (what each vID has on disk)")
    print("=" * 78)
    print(f"{'vID':<14} {'cust':<11} {'video':>6} {'gt':>4} {'yolo_probs':>12} "
           f"{'audio_probs':>12} {'audio_feats':>12}")
    print("-" * 78)
    skip_vids = []
    no_probs_vids = []
    for vid in args.vIDs:
        cust = cust_for(vid, fallback=args.customID or DEFAULT_CUST)
        a = check_availability(vid)
        if not a["has_video"] or not a["has_gt"]:
            skip_vids.append(vid)
        if not (a["has_yolo_probs"] and a["has_audio_probs"]):
            no_probs_vids.append(vid)
        ok = lambda b: "  ✓" if b else "  ✗"
        print(f"{vid:<14} {cust:<11} {ok(a['has_video']):>6} {ok(a['has_gt']):>4} "
              f"{ok(a['has_yolo_probs']):>12} {ok(a['has_audio_probs']):>12} "
              f"{ok(a['has_audio_features']):>12}")
    print()
    if skip_vids:
        print(f"⚠️  {len(skip_vids)} vIDs missing video or GT — will fail. "
              f"Skipping: {skip_vids}", file=sys.stderr)
        args.vIDs = [v for v in args.vIDs if v not in skip_vids]
    if no_probs_vids:
        print(f"⚠️  {len(no_probs_vids)} vIDs missing YOLO/audio probs — "
              f"prefilter/ensemble variants will NO-OP for these "
              f"(silently fall back to baseline). "
              f"v14_context still works (uses audio_features instead).")
        print(f"    Affected: {no_probs_vids}\n")

    # Variant-availability advice
    requested_needs_probs = any(v in args.variants
                                  for v in ("v14_prefilter", "v14_ensemble", "v14_all"))
    if requested_needs_probs and no_probs_vids:
        usable = len(args.vIDs) - len(no_probs_vids)
        print(f"NOTE: prefilter/ensemble/all variants will only actually fire on "
              f"{usable} of {len(args.vIDs)} vIDs. For the other {len(no_probs_vids)}, "
              f"they'll just rerun v13 baseline at full cost.\n")

    # ── Cost estimate ─────────────────────────────────────────────
    n_videos = len(args.vIDs)
    n_variants = len(args.variants)
    avg_cost_per_run = 2.00  # USD, rough Gemini Pro estimate per game
    est_total = n_videos * n_variants * avg_cost_per_run
    print("=" * 70)
    print("ABLATION PLAN")
    print(f"  videos:   {args.vIDs}  ({n_videos})")
    print(f"  variants: {args.variants}  ({n_variants})")
    print(f"  total metrics_seg runs:  {n_videos * n_variants}")
    print(f"  estimated cost:          ~${est_total:.0f}  "
           f"(${avg_cost_per_run:.2f} avg/run × {n_videos*n_variants})")
    print(f"  estimated wall time:     ~{n_videos*n_variants*12} min")
    print(f"  dry_run:                 {args.dry_run}")
    print("=" * 70)

    if not args.dry_run and not args.yes:
        ans = input("Proceed? (y/N) ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 0

    # ── Execute ───────────────────────────────────────────────────
    results: list[RunResult] = []

    for vid in args.vIDs:
        cust = cust_for(vid, fallback=args.customID or DEFAULT_CUST)
        print(f"\n══ {vid} (customer={cust}) ══")
        # Step 1: cv_seg (cached)
        try:
            run_cv_seg(vid, cust, dry_run=args.dry_run)
        except Exception as e:
            print(f"  [cv_seg] {vid} FAILED: {e}")
            continue

    # Run cv_seg eval once across all vids
    print(f"\n══ Step 1 eval (all vIDs) ══")
    run_cv_seg_eval(args.vIDs, dry_run=args.dry_run)

    for vid in args.vIDs:
        cust = cust_for(vid, fallback=args.customID or DEFAULT_CUST)
        for variant in args.variants:
            print(f"\n── {vid} / {variant} (cust={cust}) ──")
            r = RunResult(vid=vid, variant=variant)
            try:
                metrics_path = run_metrics_seg(
                    vid, cust, variant, VARIANTS[variant],
                    workers=args.workers, dry_run=args.dry_run)
                r.metrics_path = metrics_path
            except Exception as e:
                print(f"  metrics_seg FAILED: {e}")
                r.notes.append(f"metrics_seg failed: {e}")
                results.append(r)
                continue
            try:
                eval_dir = run_metric_eval(vid, variant, dry_run=args.dry_run)
                txt = find_latest_eval_txt(eval_dir, vid_substr=vid)
                if txt:
                    parsed = parse_eval_text(txt.read_text())
                    r.eval_path = txt
                    for k, v in parsed.items():
                        setattr(r, k, v)
            except Exception as e:
                print(f"  eval FAILED: {e}")
                r.notes.append(f"eval failed: {e}")
            results.append(r)

    # ── Report ────────────────────────────────────────────────────
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        write_report(results, args.report_path)
        print(f"\n══ Report written → {args.report_path} ══")
        print("\nTop-line goal F1 by variant (mean across videos):")
        per_v: dict[str, list[float]] = {}
        for r in results:
            if r.goal_strict_f1 is not None:
                per_v.setdefault(r.variant, []).append(r.goal_strict_f1)
        for v in args.variants:
            vs = per_v.get(v, [])
            mean = sum(vs) / len(vs) if vs else float("nan")
            print(f"  {v:<20s} mean F1 = {mean:.4f}  (n={len(vs)})")
    else:
        print(f"\n══ dry_run — nothing executed ══")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
