"""
Audit GT/video coverage and (optionally) download missing videos.

For each GT csv in data/ground_truth/:
  - Resolve its hudl_id from filename
  - Look up the matching vID via VID_TO_HUDL + customer JSON records
  - Categorize: have_video / missing_video / no_vid_mapping
  - Optionally download the missing videos via gcloud storage cp

Usage:
    # Audit only
    python3 util/audit_gt_coverage.py

    # Audit + download anything mapped but missing locally
    python3 util/audit_gt_coverage.py --download
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_cv_seg_output import VID_TO_HUDL
from cv_seg import constants as C


def _collect_vid_to_hudl(customer_dir: Path) -> dict[str, int]:
    """Union of VID_TO_HUDL + any extra records in customer JSONs that
    have both vID and hudlId fields. Customer JSONs are the only source
    of new mappings if someone adds games beyond the eval's hardcoded
    table."""
    out: dict[str, int] = dict(VID_TO_HUDL)
    for p in sorted(customer_dir.glob("*.json")):
        try:
            cfg = json.load(open(p))
        except Exception:
            continue
        for rec in cfg:
            vid = str(rec.get("vID", "")).strip()
            hudl = rec.get("hudlId") or rec.get("hudl_id")
            if vid and hudl:
                try:
                    out[vid] = int(hudl)
                except (ValueError, TypeError):
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir",        default="data/ground_truth")
    ap.add_argument("--videos-dir",    default="data/videos")
    ap.add_argument("--customer-dir",  default="data/customers")
    ap.add_argument("--bucket",        default=C.GCS_BUCKET)
    ap.add_argument("--prefix",        default=C.VIDEO_PREFIX)
    ap.add_argument("--download",      action="store_true",
                    help="actually run `gcloud storage cp` for missing videos")
    args = ap.parse_args()

    vid_to_hudl = _collect_vid_to_hudl(Path(args.customer_dir))
    hudl_to_vid = {h: v for v, h in vid_to_hudl.items()}

    have_video, missing_video, no_mapping = [], [], []
    for gt in sorted(Path(args.gt_dir).glob("gt_*.csv")):
        # Filename pattern: gt_<hudl_id>.csv
        stem = gt.stem
        try:
            hudl_id = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        vid = hudl_to_vid.get(hudl_id)
        if vid is None:
            no_mapping.append(hudl_id)
            continue
        # Pipeline.py accepts either convention (it just uses the path
        # given to --local-video). Check both.
        full_form = Path(args.videos_dir) / f"full_{vid}.mp4"
        bare_form = Path(args.videos_dir) / f"{vid}.mp4"
        if full_form.exists() or bare_form.exists():
            have_video.append((hudl_id, vid))
        else:
            missing_video.append((hudl_id, vid))

    print(f"=== GT coverage audit ===")
    print(f"GT files total:       {len(have_video) + len(missing_video) + len(no_mapping)}")
    print(f"  ✅ mapped + video local:  {len(have_video)}")
    print(f"  📥 mapped + video missing: {len(missing_video)}")
    print(f"  ❓ unmapped (no vID):      {len(no_mapping)}")

    if have_video:
        print(f"\n--- ready to use ---")
        for h, v in have_video:
            print(f"  gt_{h}.csv  ↔  {v}.mp4")

    if missing_video:
        print(f"\n--- mapped but video missing locally ---")
        for h, v in missing_video:
            print(f"  gt_{h}.csv  →  needs {v}.mp4 from "
                  f"gs://{args.bucket}/{args.prefix}/full_{v}.mp4")

    if no_mapping:
        print(f"\n--- ORPHAN GTs (no vID known — can't download without one) ---")
        for h in no_mapping:
            print(f"  gt_{h}.csv  ❓ no vID in VID_TO_HUDL or any customer JSON")
        print(f"\n  To use these GTs, add their vID/hudl pair to either:")
        print(f"    1) eval/eval_cv_seg_output.py:VID_TO_HUDL  (canonical)")
        print(f"    2) a customer JSON in {args.customer_dir}/  "
              f"with vID + hudlId fields")

    if args.download and missing_video:
        print(f"\n=== downloading {len(missing_video)} missing videos ===")
        os.makedirs(args.videos_dir, exist_ok=True)
        ok, fail = [], []
        for h, v in missing_video:
            src = f"gs://{args.bucket}/{args.prefix}/full_{v}.mp4"
            dst = os.path.join(args.videos_dir, f"full_{v}.mp4")
            print(f"  {v} …", flush=True)
            rc = subprocess.call(["gcloud", "storage", "cp", src, dst])
            if rc == 0 and os.path.exists(dst):
                ok.append(v)
                # Create the {vID}.mp4 → full_{vID}.mp4 symlink (matches
                # convention used by run_fast_set.sh / pipeline.py)
                link = os.path.join(args.videos_dir, f"{v}.mp4")
                if not os.path.lexists(link):
                    os.symlink(f"full_{v}.mp4", link)
            else:
                fail.append(v)
        print(f"\ndownloaded ok: {len(ok)}  {ok}")
        print(f"failed:        {len(fail)}  {fail}")
        return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
