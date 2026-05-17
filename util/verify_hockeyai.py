"""
verify_hockeyai.py — confirm HockeyAI YOLOv8 attribution is available
and usable in this environment.

Answers four questions in order:
  1. Is ultralytics installed?  (If no → pip install required.)
  2. Is huggingface_hub installed?  (Required for model download.)
  3. Can we load the HockeyAI model right now?  (Download + load test.)
  4. Does the model produce sensible detections on a test frame?  (Smoke test.)

If all four pass, the recipe for re-running cv_seg with HockeyAI is
printed at the end. If any fail, the script explains how to fix it.

USAGE:
    python util/verify_hockeyai.py

    # Optional: provide a path to a test video frame to run inference on
    python util/verify_hockeyai.py --test-frame /path/to/frame.jpg

No video needed for the basic check — the model load step alone confirms
whether HockeyAI is ready to fire.
"""

from __future__ import annotations

import argparse
import os
import sys


def check_ultralytics() -> tuple[bool, str]:
    """Return (ok, message)."""
    try:
        import ultralytics
        return True, f"ultralytics {ultralytics.__version__}"
    except ImportError as e:
        return False, f"NOT INSTALLED ({e})"


def check_huggingface_hub() -> tuple[bool, str]:
    try:
        import huggingface_hub
        return True, f"huggingface_hub {huggingface_hub.__version__}"
    except ImportError as e:
        return False, f"NOT INSTALLED ({e})"


def check_model_load(cv_seg_module_dir: str = ".") -> tuple[bool, str]:
    """Try to actually load HockeyAI. Downloads on first call (~52MB)."""
    # Put the repo root on sys.path so `from cv_seg.net_detection ...`
    # resolves correctly regardless of where this script is invoked from.
    if cv_seg_module_dir not in sys.path:
        sys.path.insert(0, cv_seg_module_dir)
    try:
        # Use cv_seg's own loader so we test the production code path
        from cv_seg.net_detection import _load_model_lazy
        model = _load_model_lazy()
        if model is None:
            return False, "Loader returned None — check logs for details"
        # Inspect basics
        names = getattr(model, "names", None) or {}
        return True, f"loaded; classes = {list(names.values()) if names else '?'}"
    except ModuleNotFoundError as e:
        return False, (f"could not import cv_seg ({e}). "
                       f"Run this from the repo root, or pass "
                       f"--repo-root /path/to/repo.")
    except Exception as e:
        return False, f"FAILED: {type(e).__name__}: {e}"


def check_inference(test_frame: str, cv_seg_module_dir: str = ".") -> tuple[bool, str]:
    """Run HockeyAI on a test frame and report detections."""
    if cv_seg_module_dir not in sys.path:
        sys.path.insert(0, cv_seg_module_dir)
    try:
        from cv_seg.net_detection import _load_model_lazy, NET_DET_CONF_THRESHOLD
        model = _load_model_lazy()
        if model is None:
            return False, "Model not loaded (skipping inference)"
        results = model(test_frame, conf=NET_DET_CONF_THRESHOLD, verbose=False)
        if not results:
            return False, "Inference returned empty result"
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return True, "No detections at threshold (this is OK for non-hockey frames)"
        # Tally per class
        cls_ids = boxes.cls.tolist()
        cls_names = [model.names.get(int(c), str(int(c))) for c in cls_ids]
        from collections import Counter
        tally = Counter(cls_names)
        return True, f"{len(boxes)} detections: {dict(tally)}"
    except Exception as e:
        return False, f"FAILED: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="Verify HockeyAI is ready to fire in cv_seg")
    ap.add_argument("--test-frame", default=None,
                    help="Optional: path to a JPG/PNG frame to run inference on")
    ap.add_argument("--repo-root", default=None,
                    help="Path to the repo root containing cv_seg/. "
                         "Default: current working directory.")
    args = ap.parse_args()

    # Resolve repo root. Accept either an explicit --repo-root, or default
    # to CWD if cv_seg/ is there.
    repo_root = args.repo_root or os.getcwd()
    cv_seg_path = os.path.join(repo_root, "cv_seg")
    if not os.path.isdir(cv_seg_path):
        print(f"ERROR: cv_seg/ not found at {cv_seg_path}.", file=sys.stderr)
        print("       Run this script from the repo root, or pass "
              "--repo-root /path/to/repo.", file=sys.stderr)
        return 2

    print("=" * 70)
    print("HockeyAI verification")
    print("=" * 70)
    print()

    # Step 1
    ok1, msg1 = check_ultralytics()
    print(f"[{'✓' if ok1 else '✗'}] ultralytics:        {msg1}")

    # Step 2
    ok2, msg2 = check_huggingface_hub()
    print(f"[{'✓' if ok2 else '✗'}] huggingface_hub:    {msg2}")

    if not (ok1 and ok2):
        print()
        print("FIX:")
        print("    pip install ultralytics huggingface_hub")
        print()
        print("Then re-run this script.")
        return 1

    # Step 3 — actually load the model (download happens here on first run)
    print()
    print("Loading HockeyAI model (may download ~52MB on first run)...")
    ok3, msg3 = check_model_load(cv_seg_module_dir=repo_root)
    print(f"[{'✓' if ok3 else '✗'}] model load:         {msg3}")

    if not ok3:
        print()
        print("Possible causes:")
        print("  - First-time download blocked by firewall (model is on huggingface.co)")
        print("  - Disk full / cache directory unwritable")
        print("  - HF Hub auth issue (HockeyAI model is public but check ~/.cache/huggingface)")
        return 1

    # Step 4 — optional inference smoke test
    if args.test_frame:
        if not os.path.exists(args.test_frame):
            print(f"\n[✗] test-frame not found: {args.test_frame}")
            return 1
        print()
        ok4, msg4 = check_inference(args.test_frame, cv_seg_module_dir=repo_root)
        print(f"[{'✓' if ok4 else '✗'}] inference smoke:    {msg4}")
        if not ok4:
            return 1

    # All good — print the recipe
    print()
    print("=" * 70)
    print("HockeyAI is ready. Recipe for re-running the 5 collision videos:")
    print("=" * 70)
    print("""
1. Clear stale cv_seg outputs for the 5 collision videos:

       rm -f data/output/runs/cv_seg/gt_seg_q5yj6sAFQeY*
       rm -f data/output/runs/cv_seg/gt_seg_HNG0jKYY12g*
       rm -f data/output/runs/cv_seg/gt_seg_KYtM20r9BuM*
       rm -f data/output/runs/cv_seg/gt_seg_zOQrPK7IJ24*
       rm -f data/output/runs/cv_seg/gt_seg_J8WkcuTsD5I*
       rm -f data/output/runs/metrics_seg/gt_metrics_q5yj6sAFQeY*
       rm -f data/output/runs/metrics_seg/gt_metrics_HNG0jKYY12g*
       rm -f data/output/runs/metrics_seg/gt_metrics_KYtM20r9BuM*
       rm -f data/output/runs/metrics_seg/gt_metrics_zOQrPK7IJ24*
       rm -f data/output/runs/metrics_seg/gt_metrics_J8WkcuTsD5I*

2. Re-run Steps 1+2 for the 5 collision videos. HockeyAI is on by
   default in cv_seg's CLI (`use_net_detection=True`), so the same
   `run_pipeline.py` invocation should now route through it:

       python run_pipeline.py --customer_id CUST000048 \\
           --vID n2cy8b755Tg \\
           --steps 1 2 --local-output-dir data/output/runs

       python run_pipeline.py --customer_id CUST000031 \\
           --vID HNG0jKYY12g kQVdtRa4o_A q5yj6sAFQeY zOQrPK7IJ24 \\
           --steps 1 2 --local-output-dir data/output/runs

       # KYtM20r9BuM and J8WkcuTsD5I — check which customer they belong to.
       # (Per CONTEXT.md they're CUST000048 and CUST000031 respectively.)

   NOTE: per-video cv_seg runtime grows by ~50s when HockeyAI fires
   (5 frames × ~150ms × ~70 windows per video). Budget ~10 min extra
   total for the 5 collision videos.

3. Verify HockeyAI actually fired this time. After cv_seg completes,
   spot-check the meta:

       python -c "
       import json
       d = json.load(open('data/output/runs/cv_seg/gt_seg_HNG0jKYY12g_meta.json'))
       print('attribution:', d.get('attribution'))
       "

   If you see attribution_method='hockeyai' (or similar non-'?' value),
   the model fired. If it's still '?', cv_seg silently fell back to
   motion attribution — check cv_seg logs for the reason.

4. Re-run the eval and zip results:

       python eval/eval_metric_seg_output.py --customer-id CUST000048 CUST000031

       zip -j v11_2_hockeyai_$(date +%Y%m%d_%H%M%S).zip \\
           $(ls -t data/output/evals/eval_metrics_*.txt | head -1) \\
           $(ls -t data/output/evals/eval_metrics_*.json | head -1) \\
           $(ls -t data/output/evals/eval_metrics_*_per_window.tsv | head -1) \\
           $(ls -t data/output/evals/eval_metrics_*_per_shot.tsv | head -1) \\
           $(ls -t data/output/runs/cv_seg/gt_seg_*_meta.json | head -5)
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
