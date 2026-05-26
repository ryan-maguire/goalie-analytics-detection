"""Multi-video validation of fusion-wide architecture vs v13 baseline.

For each vID in --vIDs:
  1. cv_seg (CPU, free) — cached if output exists
  2. Generate fusion-wide seg JSON (--pre 5 --post 15) — free, fast
  3. metrics_seg on cv_seg windows → "v13" baseline (~$3)
  4. metrics_seg on fusion-wide windows → "fusion_wide" (~$3)
  5. Eval both
Then aggregate side-by-side comparison report with per-vID and
mean-across-vIDs F1 metrics.

Resumable: skips any (vID, variant) where the metrics JSON already
exists on disk. Pre-existing v13 / fusion-wide outputs from prior
runs are reused.

Default vIDs: 3 known-working (have YOLO+audio probs cached for the
fusion candidate list to fire).

Usage:
    python3 tools/validate_fusion_wide.py --yes   # default 3 vids
    python3 tools/validate_fusion_wide.py --dry-run

Cost estimate: ~$3-4 per new metrics_seg run × (2 variants × n_vids
not already cached). Default plan: $12 for 2 new vIDs × 2 variants.
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
sys.path.insert(0, str(REPO / "tools"))

# Reuse the per-vID customer mapping from the ablation pipeline
VID_TO_CUST: dict[str, str] = {
    "SX5xNJlh6eQ": "CUST000048", "bfEKgtOIkQU": "CUST000048",
    "mjEeE7p2Hz8": "CUST000048", "n2cy8b755Tg": "CUST000048",
    "v0lxSTbXfw8": "CUST000048", "dwGsP6QKDs8": "CUST000031",
    "Fjc9hmK8_3U": "CUST000031", "HNG0jKYY12g": "CUST000031",
    "J8WkcuTsD5I": "CUST000031", "kQVdtRa4o_A": "CUST000031",
    "krxhPVLGLz8": "CUST000031", "KYtM20r9BuM": "CUST000031",
    "q5yj6sAFQeY": "CUST000031", "zOQrPK7IJ24": "CUST000031",
}

# vIDs that have YOLO + audio probs cached → can run fusion pipeline
HAS_PROBS = {"mjEeE7p2Hz8", "dwGsP6QKDs8", "J8WkcuTsD5I"}
DEFAULT_VIDS = ["mjEeE7p2Hz8", "dwGsP6QKDs8", "J8WkcuTsD5I"]


# ─── Subprocess helpers ───────────────────────────────────────────────
def sh(cmd: list, *, check: bool = True, dry_run: bool = False) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd[:6])}{'...' if len(cmd) > 6 else ''}",
          flush=True)
    if dry_run:
        return 0
    rc = subprocess.call([str(c) for c in cmd])
    if rc != 0 and check:
        raise RuntimeError(f"command failed (rc={rc})")
    return rc


def run_cv_seg(vid: str, cust: str, *, dry_run: bool) -> Path:
    out_dir = REPO / "data" / "output" / "runs" / "cv_seg"
    out_path = out_dir / f"gt_seg_{vid}.json"
    if out_path.exists() and not dry_run:
        print(f"  [cv_seg] {vid}: cached → skip")
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = REPO / "data" / "videos" / f"{vid}.mp4"
    if not video_path.exists():
        video_path = REPO / "data" / "videos" / f"full_{vid}.mp4"
    sh([sys.executable, "-m", "cv_seg",
        "--vID", vid, "--customID", cust,
        "--local-video", str(video_path),
        "--output-dir", str(out_dir),
        "--no-gcs"], dry_run=dry_run)
    return out_path


def run_fusion_seg(vid: str, cust: str, *, pre: int, post: int,
                     dry_run: bool) -> Path:
    out_dir = REPO / "data" / "output" / "runs" / "cv_seg_fusion_wide"
    out_path = out_dir / f"gt_seg_{vid}.json"
    if out_path.exists() and not dry_run:
        print(f"  [fusion_seg] {vid}: cached → skip")
        return out_path
    sh([sys.executable, "tools/run_fusion_pipeline.py",
        "--customer_id", cust,
        "--vID", vid,
        "--skip-metrics",
        "--pre", str(pre), "--post", str(post),
        "--out-dir", str(out_dir)], dry_run=dry_run)
    return out_path


def run_metrics_seg(vid: str, cust: str, variant: str,
                      segments_dir: Path, *, workers: int,
                      dry_run: bool) -> Path:
    out_dir = REPO / "data" / "output" / "runs" / f"metrics_{variant}"
    out_path = out_dir / f"gt_metrics_{vid}.json"
    if out_path.exists() and not dry_run:
        print(f"  [metrics_seg/{variant}] {vid}: cached → skip ${0}")
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, "metrics_seg/01_detect_segment_metrics.py",
        "--vID", vid, "--customID", cust,
        "--segments-dir", str(segments_dir),
        "--local-video-dir", str(REPO / "data" / "videos"),
        "--output-dir", str(out_dir),
        "--no-gcs",
        "--workers", str(workers)], dry_run=dry_run)
    return out_path


def run_eval(vid: str, variant: str, segments_dir: Path,
              *, dry_run: bool) -> Path:
    out_dir = REPO / "data" / "output" / "evals" / f"validate_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, "eval/eval_metric_seg_output.py",
        "--vIDs", vid,
        "--metrics-dir", str(REPO / "data" / "output" / "runs" / f"metrics_{variant}"),
        "--cv-seg-dir",  str(segments_dir),
        "--gt-dir",      str(REPO / "data" / "ground_truth"),
        "--output-dir",  str(out_dir),
        "--no-gcs"], dry_run=dry_run, check=False)
    return out_dir


# ─── Eval-text parser (reuse from ablation_pipeline) ──────────────────
_RE = lambda p: re.compile(p, re.M)
_PATTERNS = {
    "predicted_goals": _RE(r"^\s*predicted goals\s+(\d+)"),
    "actual_goals":    _RE(r"^\s*actual goals\s+(\d+)"),
    "predicted_shots": _RE(r"^\s*predicted shots\s+(\d+)"),
    "actual_shots":    _RE(r"^\s*actual shots\s+(\d+)"),
    "shot_mae":        _RE(r"^\s*mean abs error \(MAE\)\s+([0-9.]+)"),
    "n_windows":       _RE(r"^\s*windows compared:\s+(\d+)"),
    "goal_p":          _RE(r"^\s*precision\s+([0-9.]+|—|nan)\s+"),
    "goal_r":          _RE(r"^\s*recall\s+([0-9.]+|—|nan)\s+"),
    "goal_f1":         _RE(r"^\s*F1\s+([0-9.]+|—|nan)\s+"),
    "goal_tp":         _RE(r"^\s*TP\s+(\d+)\s+"),
    "goal_fp":         _RE(r"^\s*FP\s+(\d+)\s+"),
    "goal_fn":         _RE(r"^\s*FN\s+(\d+)\s+"),
    "shot_e2e_p":      _RE(r"End-to-end \(P/R/F1\):\s+P=([0-9.]+)"),
    "shot_e2e_r":      _RE(r"End-to-end \(P/R/F1\):\s+P=[0-9.]+\s+R=([0-9.]+)"),
    "shot_e2e_f1":     _RE(r"End-to-end \(P/R/F1\):\s+P=[0-9.]+\s+R=[0-9.]+\s+F1=([0-9.]+)"),
    "shot_inwin_p":    _RE(r"Within-coverage\s+\(P/R/F1\):\s+P=([0-9.]+)"),
    "shot_inwin_r":    _RE(r"Within-coverage\s+\(P/R/F1\):\s+P=[0-9.]+\s+R=([0-9.]+)"),
    "shot_inwin_f1":   _RE(r"Within-coverage\s+\(P/R/F1\):\s+P=[0-9.]+\s+R=[0-9.]+\s+F1=([0-9.]+)"),
}


def parse_eval(txt: str) -> dict:
    out = {}
    for k, pat in _PATTERNS.items():
        m = pat.search(txt)
        if not m: continue
        v = m.group(1)
        if v in ("—", "nan"):
            out[k] = None
        elif k.startswith(("predicted_", "actual_", "n_", "goal_tp", "goal_fp", "goal_fn")):
            try: out[k] = int(v)
            except ValueError: out[k] = None
        else:
            try: out[k] = float(v)
            except ValueError: out[k] = None
    return out


def find_eval_txt(eval_dir: Path, vid: str) -> Optional[Path]:
    candidates = list(eval_dir.glob("eval_metrics_*.txt"))
    if not candidates: return None
    matching = [c for c in candidates if vid in c.read_text()]
    if not matching: return None
    return max(matching, key=lambda p: p.stat().st_mtime)


def report(results: dict) -> str:
    """results = {vid: {variant: parsed_eval_dict}}"""
    lines = ["# fusion-wide validation report",
              "",
              f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
              "",
              "## Per-video metrics"]
    for vid in results:
        lines += ["", f"### {vid}", "",
                   "| metric | v13 (cv_seg) | fusion_wide | Δ |",
                   "|---|---|---|---|"]
        v13 = results[vid].get("v13", {})
        fw  = results[vid].get("fusion_wide", {})
        for label, key, fmt, better in [
            ("Goal F1 (STRICT)",    "goal_f1",      "{:.3f}", "higher"),
            ("Goal precision",      "goal_p",       "{:.3f}", "higher"),
            ("Goal recall",         "goal_r",       "{:.3f}", "higher"),
            ("Goal TP",             "goal_tp",      "{:d}",   "higher"),
            ("Goal FP",             "goal_fp",      "{:d}",   "lower"),
            ("Goal FN",             "goal_fn",      "{:d}",   "lower"),
            ("Shot end-to-end F1",  "shot_e2e_f1",  "{:.3f}", "higher"),
            ("Shot end-to-end recall", "shot_e2e_r","{:.3f}", "higher"),
            ("Within-cov F1",       "shot_inwin_f1","{:.3f}", "higher"),
            ("Within-cov recall",   "shot_inwin_r", "{:.3f}", "higher"),
            ("Shot MAE",            "shot_mae",     "{:.3f}", "lower"),
            ("Predicted shots",     "predicted_shots", "{:d}", ""),
            ("Predicted goals",     "predicted_goals", "{:d}", ""),
            ("n_windows",           "n_windows",    "{:d}",   ""),
        ]:
            a = v13.get(key)
            b = fw.get(key)
            sa = fmt.format(a) if a is not None else "—"
            sb = fmt.format(b) if b is not None else "—"
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                d = b - a
                arrow = ""
                if better == "higher":
                    arrow = " ✅" if d > 0 else " ❌" if d < 0 else ""
                elif better == "lower":
                    arrow = " ✅" if d < 0 else " ❌" if d > 0 else ""
                sd = (f"+{d:.3f}" if isinstance(d, float) and d > 0
                       else (f"{d:.3f}" if isinstance(d, float)
                              else f"{'+' if d>0 else ''}{d}")) + arrow
            else:
                sd = "—"
            lines.append(f"| {label} | {sa} | {sb} | {sd} |")

    # Aggregate
    lines += ["", "## Aggregate (mean across vIDs that have both runs)", ""]
    lines += ["| metric | v13 (mean) | fusion_wide (mean) | Δ |",
               "|---|---|---|---|"]
    for label, key in [
        ("Goal F1",           "goal_f1"),
        ("Goal precision",    "goal_p"),
        ("Goal recall",       "goal_r"),
        ("Shot end-to-end F1","shot_e2e_f1"),
        ("Shot end-to-end R", "shot_e2e_r"),
        ("Within-cov F1",     "shot_inwin_f1"),
        ("Within-cov R",      "shot_inwin_r"),
        ("Shot MAE",          "shot_mae"),
    ]:
        v13s, fws = [], []
        for vid in results:
            v13 = results[vid].get("v13", {}).get(key)
            fw  = results[vid].get("fusion_wide", {}).get(key)
            if isinstance(v13, (int, float)) and isinstance(fw, (int, float)):
                v13s.append(v13); fws.append(fw)
        if not v13s:
            lines.append(f"| {label} | — | — | — |"); continue
        m13 = sum(v13s) / len(v13s); mfw = sum(fws) / len(fws)
        d = mfw - m13
        sign = "+" if d >= 0 else ""
        better = "lower" if key == "shot_mae" else "higher"
        arrow = ""
        if better == "higher":
            arrow = " ✅" if d > 0.01 else " ❌" if d < -0.01 else ""
        else:
            arrow = " ✅" if d < -0.01 else " ❌" if d > 0.01 else ""
        lines.append(f"| {label} | {m13:.3f} | {mfw:.3f} | {sign}{d:.3f}{arrow} |")

    lines += ["", "## Headline",
               "",
               "If most ✅ marks appear in the Δ column AND aggregate "
               "metrics improved, fusion-wide is the new production "
               "default. If mixed or regressive, stick with v13."]
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vIDs", nargs="+", default=DEFAULT_VIDS,
                    help=f"vIDs to validate (default {DEFAULT_VIDS})")
    ap.add_argument("--pre", type=int, default=5)
    ap.add_argument("--post", type=int, default=15,
                    help="window padding (5/15 gave +0.42 goal F1 on mjEe)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="skip the cost-confirmation prompt")
    ap.add_argument("--report-path", type=Path,
                    default=REPO / "data" / "output" / "evals" / "fusion_wide_validation.md")
    args = ap.parse_args()

    os.chdir(REPO)

    # Sanity check + cost estimate
    print("=" * 70)
    print("FUSION-WIDE VALIDATION PLAN")
    print(f"  vIDs:    {args.vIDs}")
    print(f"  windows: ±{args.pre}/{args.post}s")
    print(f"  variants: v13 + fusion_wide")
    print()
    runs_needed = []
    for vid in args.vIDs:
        cust = VID_TO_CUST.get(vid)
        if cust is None:
            print(f"  ⚠️  {vid}: no customer mapping — SKIPPING")
            continue
        if vid not in HAS_PROBS:
            print(f"  ⚠️  {vid}: no YOLO/audio probs cached — fusion won't fire, SKIPPING")
            continue
        for variant in ("v13", "fusion_wide"):
            out_path = REPO / "data" / "output" / "runs" / f"metrics_{variant}" / f"gt_metrics_{vid}.json"
            if not out_path.exists():
                runs_needed.append((vid, variant, cust))
                print(f"    needed: {vid} / {variant}  (cust={cust})")
            else:
                print(f"    cached: {vid} / {variant}")
    print()
    est_cost = len(runs_needed) * 3
    est_time = len(runs_needed) * 15
    print(f"  new Gemini runs needed: {len(runs_needed)}")
    print(f"  estimated cost:         ~${est_cost}")
    print(f"  estimated wall time:    ~{est_time} min")
    print("=" * 70)

    if not args.dry_run and not args.yes and runs_needed:
        ans = input("Proceed? (y/N) ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted."); return 0

    # Resolve which vIDs to process
    vids_to_process = [v for v in args.vIDs
                        if VID_TO_CUST.get(v) and v in HAS_PROBS]
    if not vids_to_process:
        print("No vIDs to process. Aborting."); return 2

    # ── Execute ───────────────────────────────────────────────
    cv_seg_dir = REPO / "data" / "output" / "runs" / "cv_seg"
    fusion_seg_dir = REPO / "data" / "output" / "runs" / "cv_seg_fusion_wide"

    for vid in vids_to_process:
        cust = VID_TO_CUST[vid]
        print(f"\n══ {vid} (customer={cust}) ══")

        # 1. cv_seg (cached)
        try:
            run_cv_seg(vid, cust, dry_run=args.dry_run)
        except Exception as e:
            print(f"  cv_seg FAILED: {e} — skipping vid"); continue

        # 2. fusion-wide seg JSON (cached)
        try:
            run_fusion_seg(vid, cust, pre=args.pre, post=args.post,
                            dry_run=args.dry_run)
        except Exception as e:
            print(f"  fusion_seg FAILED: {e} — skipping vid"); continue

        # 3. v13 baseline metrics (cached)
        try:
            run_metrics_seg(vid, cust, "v13", cv_seg_dir,
                              workers=args.workers, dry_run=args.dry_run)
        except Exception as e:
            print(f"  v13 metrics_seg FAILED: {e}")

        # 4. fusion-wide metrics (cached)
        try:
            run_metrics_seg(vid, cust, "fusion_wide", fusion_seg_dir,
                              workers=args.workers, dry_run=args.dry_run)
        except Exception as e:
            print(f"  fusion_wide metrics_seg FAILED: {e}")

        # 5. evals
        try:
            run_eval(vid, "v13", cv_seg_dir, dry_run=args.dry_run)
        except Exception: pass
        try:
            run_eval(vid, "fusion_wide", fusion_seg_dir, dry_run=args.dry_run)
        except Exception: pass

    # ── Parse + report ────────────────────────────────────────
    if args.dry_run:
        print("\n══ dry-run — no report ══")
        return 0

    results: dict[str, dict[str, dict]] = {}
    for vid in vids_to_process:
        results[vid] = {}
        for variant in ("v13", "fusion_wide"):
            eval_dir = REPO / "data" / "output" / "evals" / f"validate_{variant}"
            txt = find_eval_txt(eval_dir, vid)
            if txt is None:
                continue
            results[vid][variant] = parse_eval(txt.read_text())

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report(results))
    print(f"\n══ Report written → {args.report_path} ══")

    # Headline numbers
    print("\nHeadline (mean across vIDs with both runs):")
    for key, label in [("goal_f1", "Goal F1"),
                        ("shot_e2e_f1", "Shot e2e F1"),
                        ("shot_inwin_f1", "Within-cov F1"),
                        ("shot_inwin_r", "Within-cov recall")]:
        v13s, fws = [], []
        for vid in results:
            v13 = results[vid].get("v13", {}).get(key)
            fw  = results[vid].get("fusion_wide", {}).get(key)
            if isinstance(v13, (int, float)) and isinstance(fw, (int, float)):
                v13s.append(v13); fws.append(fw)
        if v13s:
            m13 = sum(v13s) / len(v13s); mfw = sum(fws) / len(fws)
            print(f"  {label:<22s}  v13 {m13:.3f}  →  fusion_wide {mfw:.3f}  "
                  f"(Δ {mfw - m13:+.3f}, n={len(v13s)})")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
