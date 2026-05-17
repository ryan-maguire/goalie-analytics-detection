"""
discrim_probe_shots_phase2.py — α framing probe.

Phase-1 (atomic observation) showed four features with discrimination
≥+0.40. v13 deployed those features as a truth-table inside the
production "list every shot" prompt. Result: 100% of returned entries
had REQUIRED=true, DISQUALIFIER=false, EVIDENCE=true. The features
got gamed — Gemini answered them in whatever direction listed the
shot, not based on what it observed.

This script runs Phase 2: the same features under three different
framings, on the same clip set as Phase 1. Compares feature fire
rates across framings to see which (if any) survive classification
framing pressure.

The three framings:
  F1 (control / atomic):
      "At second X, is feature Y true?" — no shots, no classification.
      This is Phase 1 from discrim_probe_shots.py. We re-use those
      results (loaded from disk) rather than re-running.

  F2 (truth-table positive):
      "List shots, with each entry showing all four feature booleans.
       Truth table: REQUIRED traveling→net, EVIDENCE release|impact,
       DISQUALIFIER carrier_holds." This is the v13 production
       framing. We re-use v13 production output (loaded from disk)
       rather than re-running.

  F3 (truth-table INVERTED) — the gaming detector:
      "List shots, with each entry showing all four feature booleans.
       Truth table: REQUIRED carrier_holds, EVIDENCE NOT release |
       NOT impact, DISQUALIFIER traveling→net." Opposite of the
       v13 framing. Real discrimination would mean shots stop being
       listed (or come back with feature answers consistent with the
       observation). Gaming would mean Gemini lists "shots" anyway,
       with features now flipped to whatever direction the new
       framing requires.

INTERPRETATION
==============
For each feature, compute the fire rate under each framing on
real-shot-clips and on zone-noise-clips. A feature is REAL only if:
  - F1 disc:        rate(real) − rate(noise) >= +0.40
  - F1 vs F2:       fire rate similar in F1 and F2 (didn't inflate)
  - F1 vs F3:       fire rate ≈ rate * (F3-listing-rate / F1-listing-rate)
                    i.e. the feature tracks observation, not framing

A feature is GAMED if:
  - F2 fire rate is near 100% on every listed entry (v13 did this)
  - F3 fire rate stays near where F3's framing demands

USAGE
=====
    python3 discrim_probe_shots_phase2.py \\
        --hudl-id-map "mjEeE7p2Hz8:2073809,dwGsP6QKDs8:2070269,..." \\
        --gt-dir data/ground_truth \\
        --video-dir data/videos \\
        --metrics-dir data/output/runs/metrics_seg \\
        --phase1-dir data/output/discrim_probe_shots \\
        --output-dir data/output/discrim_probe_shots_phase2

COST
====
~30 clips × 1 call (F3) × ~$0.005 = ~$0.15
Runtime: ~10 minutes wall-clock.

We do NOT re-run F1 (Phase 1 already done) or F2 (v13 production
output is on disk). We only need to run F3.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed.", file=sys.stderr)
    sys.exit(1)


# Features we're probing
FEATURES = [
    "puck_traveling_toward_net",
    "puck_release_frame_visible",
    "puck_impact_sound_audible",
    "puck_carrier_holds_or_passes",
]

# Phase-1 clip dimensions (must match Phase 1 to allow comparison)
CLIP_HALF_WIDTH_SEC = 15
MIN_INTER_SHOT_GAP  = 60


@dataclass
class ClipRecord:
    clip_id:    str
    label:      str
    vID:        str
    video_path: str
    t_start:    float
    t_end:      float
    clip_path:  Optional[str] = None


# ---------------------------------------------------------------------------
# F3 — inverted truth-table prompt
# ---------------------------------------------------------------------------

def build_f3_inverted_prompt() -> str:
    """The inverted truth-table prompt.

    The framing tells Gemini that a shot is defined by exactly the
    OPPOSITE feature values from the v13 production framing. If features
    are observational, Gemini will list far fewer "shots" (since most
    real-shot moments don't have the inverted features true). If
    features are just rationalisations, Gemini will list shots anyway
    and flip its feature answers to match the new framing.
    """
    return """You are an expert hockey video analyst.

I will show you a 30-second clip. Identify every distinct shot attempt
in it. Use this truth table to decide whether a candidate moment is
a shot:

A candidate is a real shot ONLY IF:
  REQUIRED (must be TRUE):
    `feature_puck_carrier_holds_or_passes`
      — The dominant action is the puck-carrier maintaining possession
        or passing to a teammate.
  EVIDENCE (at least ONE must be TRUE):
    `feature_NOT_puck_release_frame_visible`
      — The puck does NOT leave the stick — no visible release frame.
    `feature_NOT_puck_impact_sound_audible`
      — There is NO crisp puck-impact sound.
  DISQUALIFIER (must be FALSE):
    `feature_puck_traveling_toward_net`
      — The puck does NOT visibly travel toward the goal net.

For each shot that satisfies the table, return:
  - timestamp (MM:SS within this clip)
  - location, release, outcome (as usual)
  - the four feature booleans

Return JSON:
{
  "shots": <int>,
  "shot_timestamps": [
    {
      "timestamp": "MM:SS",
      "location": "...",
      "release": "...",
      "outcome": "...",
      "feature_puck_traveling_toward_net":    <bool>,
      "feature_puck_release_frame_visible":   <bool>,
      "feature_puck_impact_sound_audible":    <bool>,
      "feature_puck_carrier_holds_or_passes": <bool>
    }
  ]
}

It is much better to UNDER-count than to inflate. Return ONLY the
JSON object."""


# ---------------------------------------------------------------------------
# Clip extraction (re-use Phase 1's clip set if present, else rebuild)
# ---------------------------------------------------------------------------

def load_phase1_clips(phase1_dir: str) -> list[ClipRecord]:
    """Load the clip set from Phase 1's raw output for stable comparison."""
    fp = os.path.join(phase1_dir, "phase1_raw.json")
    if not os.path.exists(fp):
        return []
    data = json.load(open(fp))
    clips: list[ClipRecord] = []
    seen = set()
    for entry in data:
        cid = entry["clip_id"]
        if cid in seen:
            continue
        seen.add(cid)
        # Phase-1 raw doesn't record video paths — we'll rebuild
        # from the clip directory.
        clip_path = os.path.join(phase1_dir, "clips", f"{cid}.mp4")
        if not os.path.exists(clip_path):
            continue
        clips.append(ClipRecord(
            clip_id=cid,
            label=entry["clip_label"],
            vID=entry["vID"],
            video_path="",  # unused
            t_start=0,
            t_end=0,
            clip_path=clip_path,
        ))
    return clips


# ---------------------------------------------------------------------------
# Gemini probe
# ---------------------------------------------------------------------------

def _get_client(project_id: str, region: str):
    return genai.Client(vertexai=True, project=project_id, location=region)


def query_gemini(client, clip_path: str, prompt: str, model_name: str,
                 max_retries: int = 3) -> Optional[dict]:
    with open(clip_path, "rb") as f:
        clip_bytes = f.read()
    contents = [
        genai_types.Part.from_bytes(data=clip_bytes, mime_type="video/mp4"),
        genai_types.Part.from_text(text=prompt),
    ]
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0,
    )
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model_name, contents=contents, config=config,
            )
            text = resp.text
            if not text:
                continue
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"    JSON decode error attempt {attempt+1}: {e}",
                  file=sys.stderr)
            time.sleep(2)
        except Exception as e:
            print(f"    API error attempt {attempt+1}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(5)
    return None


# ---------------------------------------------------------------------------
# Analysis: compare F1, F2, F3 fire rates per feature per label
# ---------------------------------------------------------------------------

def load_f1_results(phase1_dir: str) -> list[dict]:
    """Phase 1 atomic-observation results.

    Returns one row per (clip, second, feature, label, fired)."""
    fp = os.path.join(phase1_dir, "phase1_probe_results.tsv")
    if not os.path.exists(fp):
        return []
    rows = []
    with open(fp) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows


def load_f2_results(phase1_dir: str) -> list[dict]:
    """Phase 1's Phase-2 output (the v11-style framing). One row per
    listed shot_timestamps entry."""
    fp = os.path.join(phase1_dir, "phase2_raw.json")
    if not os.path.exists(fp):
        return []
    return json.load(open(fp))


def run_f3(clips: list[ClipRecord], output_dir: str, model_name: str,
           project_id: str, region: str) -> list[dict]:
    """Run F3 (inverted truth-table framing) on every clip."""
    client = _get_client(project_id, region)
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict] = []
    prompt = build_f3_inverted_prompt()
    for i, clip in enumerate(clips):
        print(f"  [{i+1}/{len(clips)}] F3 on {clip.clip_id} ({clip.label})",
              file=sys.stderr)
        resp = query_gemini(client, clip.clip_path, prompt, model_name)
        if resp is not None:
            results.append({
                "clip_id":    clip.clip_id,
                "clip_label": clip.label,
                "vID":        clip.vID,
                "response":   resp,
            })
    with open(os.path.join(output_dir, "f3_raw.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  F3 raw written to {output_dir}/f3_raw.json", file=sys.stderr)
    return results


def aggregate_results(f1_rows: list[dict], f2_results: list[dict],
                      f3_results: list[dict], output_dir: str) -> dict:
    """For each feature × clip-label, report fire rate under each framing."""

    # F1: per-(label, feature) — average fire rate across all (clip, second)
    # observations.
    f1_stats: dict[tuple[str, str], dict] = {}
    for r in f1_rows:
        key = (r["label"], r["feature"])
        s = f1_stats.setdefault(key, {"n": 0, "fired": 0})
        s["n"] += 1
        if r["fired"] == "1":
            s["fired"] += 1
    for key, s in f1_stats.items():
        s["rate"] = s["fired"] / s["n"] if s["n"] else 0

    # F2 and F3: per-(label, feature) — average fire rate across all
    # LISTED shot_timestamps entries.
    def aggregate_listed(results: list[dict]):
        out: dict[tuple[str, str], dict] = {}
        n_clips_by_label: dict[str, int] = {}
        n_listed_by_label: dict[str, int] = {}
        for entry in results:
            label = entry["clip_label"]
            n_clips_by_label[label] = n_clips_by_label.get(label, 0) + 1
            ts_list = entry["response"].get("shot_timestamps") or []
            n_listed_by_label[label] = n_listed_by_label.get(label, 0) + len(ts_list)
            for ts in ts_list:
                if not isinstance(ts, dict):
                    continue
                for feat in FEATURES:
                    key = (label, feat)
                    s = out.setdefault(key, {"n": 0, "fired": 0})
                    s["n"] += 1
                    if ts.get(f"feature_{feat}") is True:
                        s["fired"] += 1
        for key, s in out.items():
            s["rate"] = s["fired"] / s["n"] if s["n"] else 0
        return out, n_clips_by_label, n_listed_by_label

    f2_stats, f2_clips, f2_listed = aggregate_listed(f2_results)
    f3_stats, f3_clips, f3_listed = aggregate_listed(f3_results)

    # Build the readable comparison
    print("\n" + "=" * 90, file=sys.stderr)
    print("PHASE-2 FRAMING PROBE — feature fire rates across three framings",
          file=sys.stderr)
    print("=" * 90, file=sys.stderr)
    print(f"\nClip listing rates:", file=sys.stderr)
    print(f"  F2 (v13 framing):       real_shot clips: "
          f"{f2_listed.get('real_shot', 0)} entries from "
          f"{f2_clips.get('real_shot', 0)} clips  "
          f"(avg {f2_listed.get('real_shot', 0)/max(f2_clips.get('real_shot', 1),1):.1f}/clip)",
          file=sys.stderr)
    print(f"                          zone_noise clips: "
          f"{f2_listed.get('zone_noise', 0)} entries from "
          f"{f2_clips.get('zone_noise', 0)} clips  "
          f"(avg {f2_listed.get('zone_noise', 0)/max(f2_clips.get('zone_noise', 1),1):.1f}/clip)",
          file=sys.stderr)
    print(f"  F3 (inverted framing):  real_shot clips: "
          f"{f3_listed.get('real_shot', 0)} entries from "
          f"{f3_clips.get('real_shot', 0)} clips  "
          f"(avg {f3_listed.get('real_shot', 0)/max(f3_clips.get('real_shot', 1),1):.1f}/clip)",
          file=sys.stderr)
    print(f"                          zone_noise clips: "
          f"{f3_listed.get('zone_noise', 0)} entries from "
          f"{f3_clips.get('zone_noise', 0)} clips  "
          f"(avg {f3_listed.get('zone_noise', 0)/max(f3_clips.get('zone_noise', 1),1):.1f}/clip)",
          file=sys.stderr)

    # Per-feature comparison
    print(f"\nPer-feature fire rates (label: framing → rate):", file=sys.stderr)
    print(f"\n  {'feature':<40} {'label':<12} "
          f"{'F1_rate':>8} {'F2_rate':>8} {'F3_rate':>8} {'verdict':<25}",
          file=sys.stderr)
    print("  " + "-" * 100, file=sys.stderr)

    summary: dict[str, dict] = {}
    for feat in FEATURES:
        for label in ("real_shot", "zone_noise"):
            f1r = f1_stats.get((label, feat), {}).get("rate", 0)
            f2r = f2_stats.get((label, feat), {}).get("rate", 0)
            f3r = f3_stats.get((label, feat), {}).get("rate", 0)
            # Verdict heuristics
            verdict = ""
            # If F2 is near 100% and F1 is moderate, that's the v13 gaming pattern
            if f2r >= 0.95 and f1r < 0.85:
                verdict = "GAMED in F2"
            # If F3 also tracks F2 in the inverted direction, full gaming
            if f3r <= 0.05 and f2r >= 0.95:
                verdict = "FULL GAMING (F2↑F3↓)"
            elif f3r >= 0.95 and f2r <= 0.05:
                verdict = "FULL GAMING (F2↓F3↑)"
            # If F1≈F2 (within 0.15), the framing didn't change the observation
            elif abs(f2r - f1r) <= 0.15:
                verdict = "SURVIVES framing"
            print(f"  {feat:<40} {label:<12} "
                  f"{f1r:>7.2f}  {f2r:>7.2f}  {f3r:>7.2f}  {verdict:<25}",
                  file=sys.stderr)
            summary[f"{label}__{feat}"] = {
                "f1": f1r, "f2": f2r, "f3": f3r, "verdict": verdict,
            }
        print("  " + "-" * 100, file=sys.stderr)

    # Aggregate verdict
    print(f"\nAGGREGATE", file=sys.stderr)
    print(f"  Total entries listed under v13 framing (F2): "
          f"{sum(f2_listed.values())} (from "
          f"{sum(f2_clips.values())} clips)", file=sys.stderr)
    print(f"  Total entries listed under INVERTED framing (F3): "
          f"{sum(f3_listed.values())} (from "
          f"{sum(f3_clips.values())} clips)", file=sys.stderr)
    if sum(f3_listed.values()) >= 0.5 * sum(f2_listed.values()):
        print("\n  CONCLUSION: Gemini lists similar numbers of shots under both",
              file=sys.stderr)
        print("              framings — features are FRAMING-RESPONSIVE, not",
              file=sys.stderr)
        print("              observation-anchored. The v13 truth table cannot",
              file=sys.stderr)
        print("              constrain over-counting via prompt design alone.",
              file=sys.stderr)
    else:
        print("\n  CONCLUSION: Gemini lists meaningfully fewer shots under",
              file=sys.stderr)
        print(f"              inverted framing ({sum(f3_listed.values())} vs "
              f"{sum(f2_listed.values())}). Some features may be",
              file=sys.stderr)
        print("              observation-anchored — look at SURVIVES verdicts",
              file=sys.stderr)
        print("              above for v14 candidates.", file=sys.stderr)

    # Persist
    out = {
        "f1_stats":         {f"{k[0]}__{k[1]}": v for k, v in f1_stats.items()},
        "f2_stats":         {f"{k[0]}__{k[1]}": v for k, v in f2_stats.items()},
        "f3_stats":         {f"{k[0]}__{k[1]}": v for k, v in f3_stats.items()},
        "f2_n_clips":       f2_clips,
        "f2_n_listed":      f2_listed,
        "f3_n_clips":       f3_clips,
        "f3_n_listed":      f3_listed,
        "summary":          summary,
    }
    with open(os.path.join(output_dir, "phase2_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase-2 framing probe (α)")
    p.add_argument("--phase1-dir", required=True,
                   help="Phase-1 output directory (we need its clips/ and "
                        "phase1_probe_results.tsv and phase2_raw.json)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--project-id", default="goalie-analytics-pro-dev")
    p.add_argument("--region", default="us-central1")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Phase-1 dir: {args.phase1_dir}", file=sys.stderr)
    print(f"Output dir:  {args.output_dir}", file=sys.stderr)
    print(f"Model:       {args.model}", file=sys.stderr)
    print()

    # Load Phase 1 results (F1)
    f1_rows = load_f1_results(args.phase1_dir)
    if not f1_rows:
        print("ERROR: phase1_probe_results.tsv not found in "
              f"{args.phase1_dir}", file=sys.stderr)
        return 1
    print(f"Loaded {len(f1_rows)} F1 observations from Phase 1", file=sys.stderr)

    # Load Phase 1's Phase 2 (F2 — v11-style framing)
    f2_results = load_f2_results(args.phase1_dir)
    if not f2_results:
        print("ERROR: phase2_raw.json not found in "
              f"{args.phase1_dir}", file=sys.stderr)
        return 1
    print(f"Loaded {len(f2_results)} F2 results from Phase 1", file=sys.stderr)

    # Load clips for F3
    clips = load_phase1_clips(args.phase1_dir)
    if not clips:
        print("ERROR: no clips found at "
              f"{args.phase1_dir}/clips/", file=sys.stderr)
        return 1
    print(f"Loaded {len(clips)} clips from Phase 1", file=sys.stderr)
    print()

    # Run F3
    print(f"Running F3 (inverted truth-table) on {len(clips)} clips...",
          file=sys.stderr)
    f3_results = run_f3(clips, args.output_dir, args.model,
                        args.project_id, args.region)

    # Aggregate and report
    aggregate_results(f1_rows, f2_results, f3_results, args.output_dir)

    print(f"\nDone. Outputs in {args.output_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
