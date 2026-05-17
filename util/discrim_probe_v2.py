"""
discrim_probe_v2.py — goal-context discrimination probe

Companion to discrim_probe.py. Same 44 clips, same feature list, but
the prompt is now framed as a GOAL-DETECTION task with a candidate
v10 truth table embedded. The point is to test whether Gemini changes
its per-feature observation rates when the framing pushes it toward a
goal-confirmation answer.

If per-feature YES rates stay close to probe 1 (the atomic-observation
probe), the truth-table design from probe 1 is trustworthy.

If per-feature YES rates go UP on FP/neg clips (especially the
features that anchor the v10 paths), the model is gaming the
truth table to satisfy the framing — and we need to handle that
in v10 design (or accept lower real-world precision).

Pipeline:
  1. Load existing clips/cache from discrim_probe.py output dir
  2. Re-query each clip with the goal-context prompt
  3. Save responses
  4. Compute deltas: per_clip_v2 features vs per_clip features
  5. Output:
       - per_clip_v2.json
       - probe_results_v2.tsv
       - feature_analysis_v2.tsv (same format as probe 1)
       - probe_delta.tsv (rate-change comparison vs probe 1)

USAGE:
    python3 discrim_probe_v2.py \\
        --hudl-id-map "mjEeE7p2Hz8:2073809,..." \\
        --gt-dir data/ground_truth \\
        --video-dir data/videos \\
        --metrics-dir data/output/metrics_v8 \\
        --output-dir data/output/discrim_probe \\
        --probe1-results data/output/discrim_probe/per_clip.json
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Re-use the clip-building logic from probe 1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discrim_probe import (
    FEATURES,
    FEATURE_NAMES,
    ClipRecord,
    build_clip_set,
    _get_genai_client,
    parse_id_map,
)


# ---------------------------------------------------------------------------
# Goal-context prompt (different from probe 1)
# ---------------------------------------------------------------------------

def build_goal_context_prompt() -> str:
    """Goal-detection-style prompt that asks the model to evaluate a
    candidate v10 truth table.

    Critically: this prompt is ABOUT goals. It tells the model the
    purpose. Then we observe whether features fire at the same rate
    as in probe 1's atomic-observation framing.
    """
    feature_block = "\n".join(
        f"  {i+1}. **{name}**: {desc}"
        for i, (name, desc) in enumerate(FEATURES)
    )
    return f"""You are evaluating a clip from an amateur ice-hockey game to determine if a goal was scored against the defending goalie. The camera is a handheld view from the stands; there is no broadcast graphics.

A goal is confirmed if ANY of the following three paths is satisfied:

**Path A**: `puck_retrieved_from_net` AND `centre_ice_faceoff_visible`
**Path B**: `scoreboard_change` AND `ref_point_at_net`
**Path C**: `attacking_team_skates_to_bench` AND `crowd_cheer_sustained`

NEGATIVE ANCHOR — if you observe any of `goalie_glove_save`,
`goalie_freezes_puck`, `defender_clears_puck`, or
`rebound_chase_with_no_goal`, set goals = 0 regardless of the paths.

For each of the 27 observation questions below, answer YES or NO based on what is visibly or audibly present. If YES, give a MM:SS timestamp within the clip. Be conservative — only answer YES if you can clearly observe the thing in question.

Then determine if a goal was scored using the truth table above.

OBSERVATION QUESTIONS:
{feature_block}

OUTPUT FORMAT — strict JSON, no other text:

{{
  "observations": {{
    "celebration_visible":         {{"answer": "YES" | "NO", "timestamp": "MM:SS" | "", "note": "<one short phrase if YES>"}},
    "celebration_at_end_of_clip":  {{"answer": "YES" | "NO", "timestamp": "MM:SS" | "", "note": ""}},
    ... (one entry per feature, in the same order as the questions above)
  }},
  "path_evaluation": {{
    "path_A_satisfied": "YES" | "NO",
    "path_B_satisfied": "YES" | "NO",
    "path_C_satisfied": "YES" | "NO",
    "negative_anchor_fired": "YES" | "NO"
  }},
  "goals": 0 | 1,
  "goals_reasoning": "<brief explanation tying the path evaluation to the goals decision>"
}}

Output ONLY the JSON object. No prose, no markdown, no explanation outside the structure."""


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------

def query_gemini_v2(client, clip_path: str, prompt: str, model_name: str) -> Optional[dict]:
    """Single Gemini call with the goal-context prompt."""
    try:
        from google.genai import types
    except ImportError:
        print("ERROR: pip install google-genai", file=sys.stderr)
        sys.exit(1)
    try:
        with open(clip_path, "rb") as f:
            video_bytes = f.read()
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part(inline_data=types.Blob(
                    mime_type="video/mp4", data=video_bytes,
                )),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        text = response.text or ""
        return json.loads(text)
    except Exception as e:
        print(f"  Gemini error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def run_probe2(clips: list[ClipRecord], output_dir: str, model_name: str,
               use_vertex: bool, project_id: str, region: str,
               probe1_results_path: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    prompt = build_goal_context_prompt()

    per_clip_v2_path = os.path.join(output_dir, "per_clip_v2.json")
    probe_results_v2_path = os.path.join(output_dir, "probe_results_v2.tsv")
    feature_analysis_v2_path = os.path.join(output_dir, "feature_analysis_v2.tsv")
    probe_delta_path = os.path.join(output_dir, "probe_delta.tsv")

    # Load probe 1 results for comparison
    probe1_data: dict[str, dict] = {}
    if os.path.exists(probe1_results_path):
        probe1_data = json.load(open(probe1_results_path))
        print(f"Loaded {len(probe1_data)} probe 1 cached responses", file=sys.stderr)
    else:
        print(f"WARNING: probe 1 results not found at {probe1_results_path}; "
              f"delta analysis will be empty.", file=sys.stderr)

    # Resume support for probe 2
    cached: dict[str, dict] = {}
    if os.path.exists(per_clip_v2_path):
        cached = json.load(open(per_clip_v2_path))
        print(f"Loaded {len(cached)} probe 2 cached responses", file=sys.stderr)

    client = _get_genai_client(use_vertex, project_id, region)
    print(f"\nProbe 2 (goal-context): {len(clips)} clips with {model_name}...", file=sys.stderr)

    for i, clip in enumerate(clips, 1):
        if clip.clip_id in cached and not cached[clip.clip_id].get("_error"):
            continue
        print(f"  [{i}/{len(clips)}] {clip.clip_id} ({clip.label})...", file=sys.stderr)
        result = query_gemini_v2(client, clip.clip_path, prompt, model_name)
        if result is None:
            cached[clip.clip_id] = {"_error": True}
        else:
            cached[clip.clip_id] = result
        with open(per_clip_v2_path, "w") as f:
            json.dump(cached, f, indent=2)

    # Per-clip × feature TSV (same shape as probe 1)
    with open(probe_results_v2_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["clip_id", "label", "vID", "source"]
                   + FEATURE_NAMES
                   + ["pred_goals", "path_A", "path_B", "path_C", "neg_anchor"])
        for clip in clips:
            row = [clip.clip_id, clip.label, clip.vID, clip.source]
            response = cached.get(clip.clip_id, {})
            obs = response.get("observations", {}) if isinstance(response, dict) else {}
            for feat in FEATURE_NAMES:
                entry = obs.get(feat, {})
                if isinstance(entry, dict):
                    row.append(entry.get("answer", "?"))
                else:
                    row.append("?")
            pe = response.get("path_evaluation", {}) if isinstance(response, dict) else {}
            row.append(response.get("goals", "?") if isinstance(response, dict) else "?")
            row.append(pe.get("path_A_satisfied", "?"))
            row.append(pe.get("path_B_satisfied", "?"))
            row.append(pe.get("path_C_satisfied", "?"))
            row.append(pe.get("negative_anchor_fired", "?"))
            w.writerow(row)

    # Feature discrimination analysis (same format as probe 1)
    by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    label_counts: dict[str, int] = defaultdict(int)
    for clip in clips:
        label_counts[clip.label] += 1
        response = cached.get(clip.clip_id, {})
        obs = response.get("observations", {}) if isinstance(response, dict) else {}
        for feat in FEATURE_NAMES:
            entry = obs.get(feat, {})
            ans = entry.get("answer", "") if isinstance(entry, dict) else ""
            if ans == "YES":
                by_label[clip.label][feat] += 1

    with open(feature_analysis_v2_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "feature", "n_goal", "n_fp", "n_neg",
            "rate_goal", "rate_fp", "rate_neg",
            "discrim_vs_fp", "discrim_vs_neg",
            "verdict",
        ])
        for feat in FEATURE_NAMES:
            n_g = by_label["goal"].get(feat, 0)
            n_f = by_label["fp"].get(feat, 0)
            n_n = by_label["neg"].get(feat, 0)
            r_g = n_g / max(label_counts["goal"], 1)
            r_f = n_f / max(label_counts["fp"], 1)
            r_n = n_n / max(label_counts["neg"], 1)
            d_fp = r_g - r_f
            d_neg = r_g - r_n
            if d_fp >= 0.3 and d_neg >= 0.3:
                verdict = "STRONG_GOAL_SIGNAL"
            elif d_fp >= 0.15 and d_neg >= 0.15:
                verdict = "moderate_goal_signal"
            elif d_fp <= -0.15:
                verdict = "ANTI_SIGNAL_(fires_more_on_FP)"
            elif r_g + r_f + r_n >= 2.5:
                verdict = "noise/decoy"
            else:
                verdict = "weak/uninformative"
            w.writerow([
                feat, n_g, n_f, n_n,
                f"{r_g:.2f}", f"{r_f:.2f}", f"{r_n:.2f}",
                f"{d_fp:+.2f}", f"{d_neg:+.2f}", verdict,
            ])

    # Delta analysis vs probe 1
    if probe1_data:
        # Aggregate probe 1 rates
        p1_by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for clip in clips:
            response = probe1_data.get(clip.clip_id, {})
            obs = response.get("observations", {}) if isinstance(response, dict) else {}
            for feat in FEATURE_NAMES:
                entry = obs.get(feat, {})
                ans = entry.get("answer", "") if isinstance(entry, dict) else ""
                if ans == "YES":
                    p1_by_label[clip.label][feat] += 1

        with open(probe_delta_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow([
                "feature",
                "p1_goal", "p2_goal", "Δ_goal",
                "p1_fp",   "p2_fp",   "Δ_fp",
                "p1_neg",  "p2_neg",  "Δ_neg",
                "interpretation",
            ])
            for feat in FEATURE_NAMES:
                rates = {}
                for lbl in ["goal", "fp", "neg"]:
                    p1_n = p1_by_label[lbl].get(feat, 0)
                    p2_n = by_label[lbl].get(feat, 0)
                    n_total = max(label_counts[lbl], 1)
                    rates[lbl] = (p1_n / n_total, p2_n / n_total)

                # Interpretation: did framing inflate the FP rate?
                d_fp = rates["fp"][1] - rates["fp"][0]
                d_neg = rates["neg"][1] - rates["neg"][0]
                d_goal = rates["goal"][1] - rates["goal"][0]
                if abs(d_fp) >= 0.15 or abs(d_neg) >= 0.15:
                    if d_fp > 0 or d_neg > 0:
                        interp = "FRAMING_INFLATES_FP"
                    else:
                        interp = "framing_suppresses_fp"
                elif abs(d_goal) >= 0.15:
                    interp = "framing_shifts_goal_rate" if d_goal > 0 else "framing_suppresses_goal"
                else:
                    interp = "stable"
                w.writerow([
                    feat,
                    f"{rates['goal'][0]:.2f}", f"{rates['goal'][1]:.2f}", f"{d_goal:+.2f}",
                    f"{rates['fp'][0]:.2f}",   f"{rates['fp'][1]:.2f}",   f"{d_fp:+.2f}",
                    f"{rates['neg'][0]:.2f}",  f"{rates['neg'][1]:.2f}",  f"{d_neg:+.2f}",
                    interp,
                ])

    print(f"\nWrote:")
    print(f"  {per_clip_v2_path}")
    print(f"  {probe_results_v2_path}")
    print(f"  {feature_analysis_v2_path}")
    if probe1_data:
        print(f"  {probe_delta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hudl-id-map", required=True,
                   help="Comma-separated vID:hudl_id pairs (same as probe 1)")
    p.add_argument("--video-dir", default="data/videos")
    p.add_argument("--gt-dir", default="data/ground_truth")
    p.add_argument("--metrics-dir", required=True,
                   help="Source dir with v8 metrics outputs")
    p.add_argument("--output-dir", default="data/output/discrim_probe")
    p.add_argument("--workspace", default=None,
                   help="Where to read clips from (defaults to <output-dir>/clips, "
                        "matching probe 1)")
    p.add_argument("--probe1-results", default=None,
                   help="Path to probe 1's per_clip.json (defaults to "
                        "<output-dir>/per_clip.json)")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--use-api-key", action="store_true")
    p.add_argument("--project-id", default="goalie-analytics-pro-dev")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--n-negatives", type=int, default=10)
    p.add_argument("--pad-seconds", type=float, default=15.0)
    return p.parse_args()


def main():
    args = parse_args()
    hudl_map = parse_id_map(args.hudl_id_map)
    workspace = args.workspace or os.path.join(args.output_dir, "clips")
    probe1_results = args.probe1_results or os.path.join(args.output_dir, "per_clip.json")

    clips = build_clip_set(
        video_dir=args.video_dir,
        gt_dir=args.gt_dir,
        metrics_dir=args.metrics_dir,
        hudl_id_map=hudl_map,
        workspace=workspace,
        pad_seconds=args.pad_seconds,
        n_random_negatives=args.n_negatives,
        fp_v8_metrics_dir=args.metrics_dir,
    )
    if not clips:
        print("ERROR: no clips built", file=sys.stderr)
        sys.exit(1)

    by_label = defaultdict(int)
    for c in clips:
        by_label[c.label] += 1
    print(f"\nClip set summary:", file=sys.stderr)
    for label, n in sorted(by_label.items()):
        print(f"  {label}: {n}", file=sys.stderr)

    run_probe2(
        clips, args.output_dir, args.model,
        use_vertex=(not args.use_api_key),
        project_id=args.project_id,
        region=args.region,
        probe1_results_path=probe1_results,
    )


if __name__ == "__main__":
    main()
