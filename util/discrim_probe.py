"""
discrim_probe.py — standalone goal-discrimination diagnostic

Goal: figure out which visual/audio cues Gemini RELIABLY DIFFERENTIATES
between known goals, known false positives (from v8), and random
non-goal play. The output drives v10 truth table design — bottom-up
from observed model behavior, not top-down from hockey theory.

This script is INDEPENDENT of metrics_seg/. It does not modify the
prompt. It runs its own structured probe against a curated clip set.

Pipeline:
  1. Build clip set from Hudl GT + v8 FPs + random negatives.
     ~40 clips total, each labeled (in our records, NOT to the model)
     as 'goal', 'fp', or 'neg'.
  2. For each clip, send a single Gemini call with a structured,
     label-free observation prompt asking for ~25 atomic features.
  3. Aggregate responses into a feature-by-clip table.
  4. Compute per-feature fire rates per class and discrimination scores.
  5. Output:
       - probe_results.tsv  (one row per clip × feature)
       - feature_analysis.tsv  (one row per feature with rates+scores)
       - per_clip.json  (full Gemini responses for inspection)

USAGE:
    python3 discrim_probe.py \\
        --hudl-id-map "mjEeE7p2Hz8:2073809,SX5xNJlh6eQ:2073056,..." \\
        --gt-dir data/ground_truth \\
        --video-dir data/videos \\
        --metrics-dir data/output/metrics_v8 \\
        --output-dir data/output/discrim_probe

Requirements:
    pip install google-genai

Cost: ~40 clips × 1 Gemini call each. At ~$0.005 / call this is ~$0.20
of API cost.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Feature set — atomic observations the model is asked to make
# ---------------------------------------------------------------------------
#
# Design principles:
#  - Each feature is a single observable thing, not a judgment
#  - Each feature is YES/NO (with optional MM:SS timestamp for true features)
#  - Mix of real candidate features + deliberate decoys
#  - The order is randomized in the prompt to avoid framing effects

FEATURES = [
    # ── Visual goal-anchor features ──
    ("celebration_visible",
     "Visible celebration by 2+ attacking-team players (raised sticks, "
     "embracing, jumping, glove-pumps). Single-player reaction does NOT count."),
    ("celebration_at_end_of_clip",
     "If celebration_visible is true: did the celebration occur in the "
     "LAST 15 seconds of the clip (vs. the beginning)?"),
    ("ref_point_at_net",
     "Did a referee point a horizontal arm directly toward a goal net? "
     "Distinguish from ref signaling icing (parallel arm) or penalty "
     "(arm raised UP)."),
    ("ref_point_at_north_end",
     "If ref_point_at_net is true: was the ref pointing at the net at "
     "the FAR/UPPER end of the camera frame (north end of the rink)?"),
    ("ref_point_at_south_end",
     "If ref_point_at_net is true: was the ref pointing at the net at "
     "the NEAR/LOWER end of the camera frame (south end of the rink)?"),
    ("puck_visible_in_net",
     "Was the puck visibly seen INSIDE the net (between the posts and "
     "behind the goal line, with the goalie not in possession)?"),
    ("puck_retrieved_from_net",
     "Did a player or referee visibly reach INTO the net through the "
     "front opening to remove the puck? Distinguish from defender "
     "clearing the puck or goalie freezing it."),

    # ── Save-anchor features (anti-signals for goals) ──
    ("goalie_glove_save",
     "Did the defending goalie cleanly catch the puck in their glove?"),
    ("goalie_pad_save",
     "Did the defending goalie deflect a shot with their pads?"),
    ("goalie_freezes_puck",
     "Did the defending goalie smother/cover/freeze the puck under "
     "their body or glove?"),
    ("defender_clears_puck",
     "Did a defender visibly clear the puck OUT of the crease area "
     "after a save?"),
    ("rebound_chase_with_no_goal",
     "Was there a chaotic rebound scramble in front of the net that "
     "ended with the puck being cleared OUT (away from the net), not "
     "with the puck going IN?"),

    # ── Faceoff features ──
    ("centre_ice_faceoff_visible",
     "Did both teams visibly skate to and line up at CENTRE ice for "
     "a faceoff (puck dropped at the centre faceoff dot)?"),
    ("zone_faceoff_visible",
     "Did a faceoff occur at one of the IN-ZONE faceoff dots (not "
     "centre ice, but at one of the corner dots inside a defending "
     "blue line)?"),

    # ── Audio features ──
    ("whistle_audible",
     "Was a referee whistle clearly audible at any point in the clip?"),
    ("multiple_whistles",
     "Were 2+ distinct whistles audible (typical of a confirmed-goal "
     "stoppage where refs blow multiple times)?"),
    ("crowd_cheer_sustained",
     "Was a sustained crowd cheer audible (more than 3 seconds, rising "
     "from baseline noise)?"),
    ("crowd_groan",
     "Was a crowd groan or 'awww' reaction audible (typical of a save "
     "or near-miss)?"),
    ("scoreboard_change",
     "Did a visible scoreboard digit (numeric goal count) change "
     "during the clip?"),
    ("horn_audible",
     "Was a goal horn or buzzer audible (NOT just a whistle)?"),

    # ── Behavioral / contextual ──
    ("attacking_team_skates_to_bench",
     "Did the attacking team visibly skate toward their own bench "
     "for a fist-bump line (typical post-goal celebration ritual)?"),
    ("goalie_retrieves_puck_from_own_net",
     "Did the defending goalie themselves reach into the net to "
     "retrieve the puck?"),
    ("camera_zooms_in_on_celebration",
     "Did the camera visibly zoom in or hold on a player or group "
     "celebrating, as opposed to following continuous play?"),
    ("play_continues_no_stoppage",
     "Did play visibly CONTINUE without a whistle stoppage during "
     "or after the candidate goal moment?"),

    # ── Decoys (we expect these to fire equally regardless of class) ──
    ("players_on_ice_visible",
     "Were there players visibly skating on the ice (decoy — should "
     "fire on every clip)?"),
    ("puck_visible_at_some_point",
     "Was the puck visible at any point in the clip (decoy)?"),
]

FEATURE_NAMES = [f[0] for f in FEATURES]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt() -> str:
    """Build the label-free observation prompt.

    Critical: the prompt does NOT say 'is this a goal' or 'classify this'.
    It asks for atomic observations. The model has no way to know what
    answer we want — which is the point.
    """
    feature_block = "\n".join(
        f"  {i+1}. **{name}**: {desc}"
        for i, (name, desc) in enumerate(FEATURES)
    )
    return f"""You are watching a short clip of an ice-hockey game from an amateur arena (handheld camera up in the stands; no broadcast graphics). For each of the observation questions below, answer YES or NO based on what is visibly or audibly present in the clip. If YES, give a MM:SS timestamp within the clip showing where you observed it.

Be conservative. Only answer YES if you can clearly observe the thing in question — do not infer or assume. If the camera angle is unclear, answer NO.

OBSERVATION QUESTIONS:
{feature_block}

OUTPUT FORMAT — strict JSON, no other text:

{{
  "observations": {{
    "celebration_visible":         {{"answer": "YES" | "NO", "timestamp": "MM:SS" | "", "note": "<one short phrase if YES>"}},
    "celebration_at_end_of_clip":  {{"answer": "YES" | "NO", "timestamp": "MM:SS" | "", "note": ""}},
    ... (one entry per feature, in the same order as the questions above)
  }}
}}

Output ONLY the JSON object. No prose, no markdown, no explanation outside the structure."""


# ---------------------------------------------------------------------------
# Clip extraction via ffmpeg
# ---------------------------------------------------------------------------

def extract_clip(video_path: str, t_start: float, t_end: float,
                 out_path: str) -> bool:
    """Extract [t_start, t_end] from video to out_path. Returns True on success."""
    duration = max(0.5, t_end - t_start)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{max(0, t_start):.2f}",
        "-i", video_path,
        "-t", f"{duration:.2f}",
        "-c", "copy",  # stream copy, no re-encoding
        out_path,
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, timeout=60)
        if result.returncode != 0:
            # Re-encode fallback (some videos can't be stream-copied at arbitrary timestamps)
            cmd_reenc = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{max(0, t_start):.2f}",
                "-i", video_path,
                "-t", f"{duration:.2f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                "-c:a", "aac",
                out_path,
            ]
            result = subprocess.run(cmd_reenc, check=False, capture_output=True, timeout=180)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except Exception as e:
        print(f"  ffmpeg failed for {out_path}: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Build clip set
# ---------------------------------------------------------------------------

@dataclass
class ClipRecord:
    clip_id:   str          # "goal_mjEeE_t261", "fp_v0lx_t499", "neg_bfEK_t1230"
    label:    str           # "goal" / "fp" / "neg"
    vID:      str
    video_path: str
    t_start:  float
    t_end:    float
    source:   str           # "hudl_goal" / "v8_fp" / "random_neg"
    extra:    dict = field(default_factory=dict)
    clip_path: str = ""     # filled after extraction


def build_clip_set(
    video_dir: str,
    gt_dir: str,
    metrics_dir: str,
    hudl_id_map: dict[str, str],
    workspace: str,
    pad_seconds: float = 15.0,
    n_random_negatives: int = 10,
    fp_v8_metrics_dir: Optional[str] = None,
) -> list[ClipRecord]:
    """Build the test clip set. Returns ClipRecords with clip_path filled in."""
    rng = random.Random(42)
    clips: list[ClipRecord] = []

    # 1. Hudl Goals → 'goal' clips
    print("Building goal clips...", file=sys.stderr)
    for vID, hudl_id in hudl_id_map.items():
        gt_path = os.path.join(gt_dir, f"gt_{hudl_id}.csv")
        video_path = os.path.join(video_dir, f"full_{vID}.mp4")
        if not os.path.exists(gt_path) or not os.path.exists(video_path):
            print(f"  [{vID}] missing GT or video; skipping", file=sys.stderr)
            continue
        with open(gt_path) as f:
            for row in csv.DictReader(f):
                if row.get("action") != "Goals":
                    continue
                t_start = max(0, float(row["start"]) - pad_seconds)
                t_end = float(row["end"]) + pad_seconds
                clip_id = f"goal_{vID[:5]}_t{int(float(row['start']))}"
                clips.append(ClipRecord(
                    clip_id=clip_id, label="goal", vID=vID,
                    video_path=video_path, t_start=t_start, t_end=t_end,
                    source="hudl_goal",
                    extra={"team": row.get("team", ""), "hudl_id": hudl_id},
                ))

    # 2. v8 false positives → 'fp' clips
    print("Building fp clips from v8 metrics...", file=sys.stderr)
    if fp_v8_metrics_dir and os.path.isdir(fp_v8_metrics_dir):
        for vID, hudl_id in hudl_id_map.items():
            metrics_file = os.path.join(fp_v8_metrics_dir, f"gt_metrics_{vID}.json")
            gt_path = os.path.join(gt_dir, f"gt_{hudl_id}.csv")
            video_path = os.path.join(video_dir, f"full_{vID}.mp4")
            if not (os.path.exists(metrics_file) and os.path.exists(gt_path)
                    and os.path.exists(video_path)):
                continue
            metrics = json.load(open(metrics_file))
            gt_goals = []
            with open(gt_path) as f:
                for row in csv.DictReader(f):
                    if row.get("action") == "Goals":
                        gt_goals.append((float(row["start"]), float(row["end"])))
            for s in metrics:
                m = s.get("metrics") or {}
                if (m.get("goals") or 0) < 1:
                    continue
                ws = s.get("segment_start", 0)
                we = s.get("segment_end", 0)
                # Is this an FP? (no Hudl Goal overlaps)
                if any(ge > ws and gs < we for gs, ge in gt_goals):
                    continue
                # FP — use the cv_seg window as the clip span
                clip_id = f"fp_{vID[:5]}_t{int(ws)}"
                clips.append(ClipRecord(
                    clip_id=clip_id, label="fp", vID=vID,
                    video_path=video_path, t_start=float(ws),
                    t_end=float(we), source="v8_fp",
                    extra={"threat_color": s.get("threat_goalie_color", "")},
                ))

    # 3. Random negatives → 'neg' clips (random Shots events that are NOT
    #    near any Goal — gives us a baseline of "looks like normal play")
    print("Building random negative clips...", file=sys.stderr)
    neg_candidates: list[tuple[str, str, float, float]] = []
    for vID, hudl_id in hudl_id_map.items():
        gt_path = os.path.join(gt_dir, f"gt_{hudl_id}.csv")
        video_path = os.path.join(video_dir, f"full_{vID}.mp4")
        if not (os.path.exists(gt_path) and os.path.exists(video_path)):
            continue
        shots = []
        goals = []
        with open(gt_path) as f:
            for row in csv.DictReader(f):
                if row.get("action") == "Goals":
                    goals.append((float(row["start"]), float(row["end"])))
                elif row.get("action") == "Shots":
                    shots.append((float(row["start"]), float(row["end"])))
        for ss, se in shots:
            # Skip if any Goal overlaps within ±30s of this shot
            if any(abs(gs - ss) < 30 or abs(ge - se) < 30 for gs, ge in goals):
                continue
            t_start = max(0, ss - pad_seconds)
            t_end = se + pad_seconds
            neg_candidates.append((vID, video_path, t_start, t_end))

    rng.shuffle(neg_candidates)
    for vID, video_path, t_start, t_end in neg_candidates[:n_random_negatives]:
        clip_id = f"neg_{vID[:5]}_t{int(t_start + pad_seconds)}"
        clips.append(ClipRecord(
            clip_id=clip_id, label="neg", vID=vID,
            video_path=video_path, t_start=t_start, t_end=t_end,
            source="random_neg",
        ))

    # 4. Extract all clips
    print(f"\nExtracting {len(clips)} clips...", file=sys.stderr)
    os.makedirs(workspace, exist_ok=True)
    extracted: list[ClipRecord] = []
    for clip in clips:
        out_path = os.path.join(workspace, f"{clip.clip_id}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            clip.clip_path = out_path
            extracted.append(clip)
            continue
        ok = extract_clip(clip.video_path, clip.t_start, clip.t_end, out_path)
        if ok:
            clip.clip_path = out_path
            extracted.append(clip)
        else:
            print(f"  failed to extract {clip.clip_id}", file=sys.stderr)
    print(f"  {len(extracted)} / {len(clips)} clips extracted", file=sys.stderr)

    return extracted


# ---------------------------------------------------------------------------
# Gemini probe
# ---------------------------------------------------------------------------

def _get_genai_client(use_vertex: bool, project_id: str, region: str):
    """Create a genai.Client. Defaults to Vertex AI auth (like metrics_seg);
    falls back to API-key auth if --use-api-key is set."""
    try:
        from google import genai
    except ImportError:
        print("ERROR: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    if use_vertex:
        # Same path metrics_seg uses — relies on
        # `gcloud auth application-default login` credentials.
        return genai.Client(vertexai=True, project=project_id, location=region)
    else:
        # API-key path; reads GOOGLE_API_KEY or GEMINI_API_KEY env var.
        return genai.Client()


def query_gemini(client, clip_path: str, prompt: str, model_name: str) -> Optional[dict]:
    """Single Gemini call. Returns parsed JSON or None on error."""
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


# ---------------------------------------------------------------------------
# Run probe + analysis
# ---------------------------------------------------------------------------

def run_probe(clips: list[ClipRecord], output_dir: str, model_name: str,
              use_vertex: bool, project_id: str, region: str) -> None:
    """Run the probe across all clips and write outputs."""
    os.makedirs(output_dir, exist_ok=True)
    prompt = build_prompt()

    per_clip_path = os.path.join(output_dir, "per_clip.json")
    probe_results_path = os.path.join(output_dir, "probe_results.tsv")
    feature_analysis_path = os.path.join(output_dir, "feature_analysis.tsv")

    # Resume support
    cached: dict[str, dict] = {}
    if os.path.exists(per_clip_path):
        cached = json.load(open(per_clip_path))
        print(f"Loaded {len(cached)} cached responses", file=sys.stderr)

    # Create the client once and reuse across all calls
    client = _get_genai_client(use_vertex, project_id, region)
    print(f"\nProbing {len(clips)} clips with {model_name} "
          f"(auth: {'vertex' if use_vertex else 'api-key'})...", file=sys.stderr)

    for i, clip in enumerate(clips, 1):
        if clip.clip_id in cached and not cached[clip.clip_id].get("_error"):
            continue
        print(f"  [{i}/{len(clips)}] {clip.clip_id} ({clip.label})...", file=sys.stderr)
        result = query_gemini(client, clip.clip_path, prompt, model_name)
        if result is None:
            cached[clip.clip_id] = {"_error": True}
        else:
            cached[clip.clip_id] = result
        # Save after each call to support resume
        with open(per_clip_path, "w") as f:
            json.dump(cached, f, indent=2)

    # Per-clip × feature TSV
    with open(probe_results_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["clip_id", "label", "vID", "source"] + FEATURE_NAMES)
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
            w.writerow(row)

    # Feature discrimination analysis
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

    with open(feature_analysis_path, "w", newline="") as f:
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
            d_fp  = r_g - r_f
            d_neg = r_g - r_n

            # Verdict heuristic
            if d_fp >= 0.3 and d_neg >= 0.3:
                verdict = "STRONG_GOAL_SIGNAL"
            elif d_fp >= 0.15 and d_neg >= 0.15:
                verdict = "moderate_goal_signal"
            elif d_fp <= -0.15:
                verdict = "ANTI_SIGNAL_(fires_more_on_FP)"
            elif r_g + r_f + r_n >= 2.5:  # fires on most clips
                verdict = "noise/decoy"
            else:
                verdict = "weak/uninformative"

            w.writerow([
                feat, n_g, n_f, n_n,
                f"{r_g:.2f}", f"{r_f:.2f}", f"{r_n:.2f}",
                f"{d_fp:+.2f}", f"{d_neg:+.2f}", verdict,
            ])

    print(f"\nWrote:")
    print(f"  {per_clip_path}")
    print(f"  {probe_results_path}")
    print(f"  {feature_analysis_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_id_map(s: str) -> dict[str, str]:
    out = {}
    for entry in s.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"bad --hudl-id-map entry: {entry}")
        k, v = entry.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hudl-id-map", required=True,
                   help="Comma-separated vID:hudl_id pairs (e.g. "
                        "'mjEeE7p2Hz8:2073809,SX5xNJlh6eQ:2073056')")
    p.add_argument("--video-dir", default="data/videos")
    p.add_argument("--gt-dir", default="data/ground_truth")
    p.add_argument("--metrics-dir", required=True,
                   help="Source dir with v8 metrics outputs (used to "
                        "find FP windows for the probe).")
    p.add_argument("--output-dir", default="data/output/discrim_probe")
    p.add_argument("--workspace", default=None,
                   help="Where to extract clips (defaults to a temp dir under output-dir)")
    p.add_argument("--model", default="gemini-2.5-flash",
                   help="Gemini model name (default: gemini-2.5-flash)")
    p.add_argument("--use-api-key", action="store_true",
                   help="Use API-key auth (GOOGLE_API_KEY/GEMINI_API_KEY env "
                        "var) instead of Vertex AI. Default is Vertex AI auth, "
                        "matching how metrics_seg authenticates.")
    p.add_argument("--project-id", default="goalie-analytics-pro-dev",
                   help="GCP project for Vertex AI (default matches metrics_seg)")
    p.add_argument("--region", default="us-central1",
                   help="GCP region for Vertex AI (default matches metrics_seg)")
    p.add_argument("--n-negatives", type=int, default=10,
                   help="Number of random non-goal clips to include")
    p.add_argument("--pad-seconds", type=float, default=15.0,
                   help="Padding around Hudl Goal events when building clips")
    return p.parse_args()


def main():
    args = parse_args()
    hudl_map = parse_id_map(args.hudl_id_map)
    workspace = args.workspace or os.path.join(args.output_dir, "clips")

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

    print(f"\nClip set summary:", file=sys.stderr)
    by_label = defaultdict(int)
    for c in clips:
        by_label[c.label] += 1
    for label, n in sorted(by_label.items()):
        print(f"  {label}: {n}", file=sys.stderr)

    run_probe(
        clips, args.output_dir, args.model,
        use_vertex=(not args.use_api_key),
        project_id=args.project_id,
        region=args.region,
    )


if __name__ == "__main__":
    main()
