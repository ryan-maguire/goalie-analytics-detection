"""
discrim_probe_shots.py — shot-moment discrimination diagnostic (v3)

Goal: figure out which visual/audio features Gemini RELIABLY DIFFERENTIATES
between real shot moments and zone-pressure-but-no-shot moments. The
output drives v13 prompt redesign — bottom-up from observed model
behavior, not top-down from "what should logically be a shot."

This is the same pattern that drove the v10 goal-detection breakthrough
(F1 0.42 → 0.625). It is INDEPENDENT of metrics_seg/. Does not modify
the production prompt. Runs its own structured probe against a curated
clip set.

WHY THIS IS NEEDED
==================
v11.2 produces shot_timestamps with within-coverage recall 0.71 but
precision 0.28. The 2.2-shots-per-window ratio is consistent across
clean, collision, weak, and strong videos — so the over-counting is
structural to Gemini's behavior under the v11 prompt, not a video-
quality issue.

The investigation summary's lesson #2 applies here: "LLMs game whatever
path you give them." When v11 asks Gemini to enumerate each shot with
timestamp + location + release + outcome, Gemini pattern-matches on
"must enumerate something" and produces 2-3 entries even when only 1
real shot occurred. The fix is to anchor shot timestamps on hard
visual features (like v10 used scoreboard_change + bench fist-bump),
not to add more "be more conservative" instructions.

PROBE DESIGN
============
DIFFERENT from the goal probe (which is per-clip binary). Shot moments
require timestamp-level discrimination, not whole-clip yes/no.

Two phases:

Phase 1 — Per-second atomic observations:
    For each clip, ask Gemini to scan ~5-10 candidate seconds (chosen
    to include both real shot seconds and zone-pressure-but-no-shot
    seconds). For each candidate second, ask atomic feature questions:
      - "Is there a clear puck-leaves-stick frame here?"
      - "Is the goalie making a save attempt?"
      - "Is the puck visible airborne or sliding toward the net?"
      - "Is play continuous (no whistle in the next 2s)?"
    The label (real-shot vs zone-pressure-noise) is NOT shown.

Phase 2 — Under v11 framing:
    Run the same clips through the production v11 shot_timestamps
    prompt. Compare what Gemini reports vs Hudl. Compute per-feature
    "framing inflation" — which atomic features inflate under
    classification pressure (the failure mode v10 found for goals).

OUTPUT
======
For each feature:
  - fire_rate_on_real_shots (atomic phase)
  - fire_rate_on_zone_pressure_noise (atomic phase)
  - discrimination_score = real_rate - noise_rate
  - framing_delta = (phase 2 inflation - phase 1 baseline)

A feature is a STRONG candidate for v13 if:
  - discrimination_score >= 0.40
  - framing_delta <= 0.20 (not heavily gamed under classification)

CLIP SET
========
~30 clips drawn from videos with clean attribution + clean GT:
  - 20 "real shot" clips: ±15s padding around a Hudl Shots event,
    AT LEAST 60s away from any other Hudl shot to avoid contamination.
  - 10 "zone pressure noise" clips: cv_seg threat segments where
    Gemini predicted 2+ shots but Hudl GT had 0 shots in the window
    AND no shots in the surrounding ±60s (high-confidence ghost-shot
    windows).

USAGE
=====
    python3 discrim_probe_shots.py \\
        --hudl-id-map "mjEeE7p2Hz8:2073809,dwGsP6QKDs8:2070269,..." \\
        --gt-dir data/ground_truth \\
        --video-dir data/videos \\
        --metrics-dir data/output/runs/metrics_seg \\
        --output-dir data/output/discrim_probe_shots

COST
====
~30 clips × 2 calls each (Phase 1 + Phase 2) × ~$0.005 = ~$0.30
Runtime: ~10-15 minutes wall-clock with sequential calls.
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

# Gemini client. We use the same import pattern as discrim_probe.py.
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Feature definitions — what we ask about each candidate second
# ---------------------------------------------------------------------------
#
# Designed to be ATOMIC and VERIFIABLE (Gemini can answer from one frame
# or one second of video, not from theorising about "what type of play
# this is"). Features fall into three groups:
#
#   1. Direct shot evidence (highest discrimination expected)
#   2. Goalie reaction (correlates with real shots)
#   3. Continuity / non-shot evidence (anti-anchors — should fire on noise)
#
# Like the goal probe, we also include intentional decoys to check that
# Gemini isn't just saying yes to everything.

FEATURES = [
    # ── Direct shot evidence ──────────────────────────────────────────
    ("puck_release_frame_visible",
     "At this exact moment in the clip, is there a clear PUCK-LEAVES-STICK "
     "frame — a player's stick visibly contacts and releases the puck "
     "toward the net? Distinguish from a pass (puck travels along the ice "
     "between teammates) or a deke/carry (puck stays on the stick)."),

    ("puck_traveling_toward_net",
     "At this moment, is the puck visibly traveling in a STRAIGHT or curved "
     "line toward the goal net (vs. along the ice between players)?"),

    ("shot_lane_clear_to_net",
     "At this moment, does the puck-carrier have a clear shooting LANE to "
     "the net (no defender directly in front of them blocking the shot)?"),

    # ── Goalie reaction (correlates with real shots) ──────────────────
    ("goalie_drops_to_butterfly",
     "At this moment, is the defending goalie dropping to the butterfly "
     "(knees down, pads forming a V across the crease)? Distinguish from "
     "tracking play passively in their stance."),

    ("goalie_extends_glove_or_blocker",
     "At this moment, is the goalie actively extending their glove or "
     "blocker hand to intercept a puck?"),

    ("goalie_makes_visible_save",
     "At this moment, does the goalie visibly stop a puck (catch in glove, "
     "deflect with pad, smother with body)?"),

    # ── Continuity / non-shot signals ─────────────────────────────────
    ("play_continues_no_stoppage",
     "In the 2 seconds AFTER this moment, does play visibly continue with "
     "no whistle, no goalie freezing the puck, and no goal celebration?"),

    ("scramble_in_front_no_clear_shot",
     "At this moment, is there a chaotic scramble or scrum in the goal "
     "crease area where multiple players are battling for the puck but "
     "NO clear shot release is visible? (This is the 'zone pressure noise' "
     "feature — should fire on ghost-shot windows.)"),

    ("puck_carrier_holds_or_passes",
     "At this moment, does the puck-carrier visibly choose to HOLD the "
     "puck (maintaining possession) or PASS to a teammate, rather than "
     "shoot toward the net?"),

    # ── Audio (often the most reliable in real shots) ─────────────────
    ("puck_impact_sound_audible",
     "In the 1 second around this moment, is there an AUDIBLE crisp puck "
     "impact sound — puck-on-pad, puck-on-post, puck-on-glass? "
     "Distinguish from skate scraping, body contact, or stick-on-stick."),

    ("whistle_in_next_2s",
     "Is a referee whistle audible in the 2 seconds AFTER this moment?"),

    # ── Decoys (should fire on every clip) ────────────────────────────
    ("any_player_visible_with_puck",
     "Is there any player visibly carrying or near the puck at this moment "
     "(decoy — should fire on virtually every clip)?"),

    ("ice_surface_visible",
     "Is the ice surface visible in the frame at this moment (decoy)?"),
]

FEATURE_NAMES = [f[0] for f in FEATURES]


# ---------------------------------------------------------------------------
# Candidate second selection
# ---------------------------------------------------------------------------
#
# Each clip is 30 seconds wide. From it we select 3-5 candidate seconds
# to ask atomic questions about. For real-shot clips, the candidates
# include:
#   - The Hudl-real-shot moment (definitive +1 label)
#   - 1-2 seconds 5-10s before the shot (definitive -1 label, "zone pressure")
#   - 1-2 seconds 5-10s after the shot (definitive -1 label, "post-shot")
#
# For ghost-shot clips, candidates are 3-5 evenly-spaced seconds
# throughout the window (all labeled -1, "zone pressure but no shot").
#
# This produces ~100-150 labeled (clip, second, label) tuples from
# ~30 clips for analysis.


@dataclass
class CandidateSecond:
    """One labeled (clip, second_in_clip) pair to be probed."""
    clip_id:        str          # parent clip ID
    sec_in_clip:    int          # seconds offset from clip start
    label:          str          # 'real_shot' | 'pre_shot' | 'post_shot' | 'zone_noise'
    notes:          str = ""     # diagnostic (e.g. shot release time)


@dataclass
class ClipRecord:
    """One clip in the probe set."""
    clip_id:        str
    label:          str          # 'real_shot' or 'zone_noise'
    vID:            str
    video_path:     str
    t_start:        float        # absolute video seconds
    t_end:          float
    clip_path:      Optional[str] = None
    candidates:     list[CandidateSecond] = field(default_factory=list)
    extra:          dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_phase1_prompt(candidate_seconds: list[int]) -> str:
    """Phase 1: atomic, label-free observation prompt.

    Asks Gemini to answer feature questions for each candidate second
    in the clip. The prompt does NOT mention "shot detection" — we
    want to measure baseline feature fire rates without classification
    pressure.
    """
    candidate_str = ", ".join(f"second {s}" for s in candidate_seconds)

    feature_block = "\n".join(
        f"  {i+1}. `{name}`: {desc}"
        for i, (name, desc) in enumerate(FEATURES)
    )

    return f"""You are an expert hockey video analyst.

I will show you a short video clip. I need you to OBSERVE specific moments
in the clip and answer atomic yes/no questions about what's visible at
those moments. Do NOT interpret the play or guess what happened — just
describe what you literally see and hear at each candidate moment.

CANDIDATE MOMENTS to observe (seconds from clip start):
{candidate_str}

For EACH candidate moment, answer YES or NO to ALL of these features:
{feature_block}

Return a JSON object with the following shape — exactly one entry per
candidate moment, each with one boolean per feature:

{{
  "observations": [
    {{
      "second": <int>,
      "puck_release_frame_visible": true|false,
      "puck_traveling_toward_net": true|false,
      ... (all features)
    }},
    ... (one entry per candidate second)
  ]
}}

Critical rules:
  - Answer based ONLY on what you can SEE or HEAR at the exact moment.
  - Do NOT infer based on what happened before or after the moment
    (except for features that explicitly ask about "the next 2 seconds").
  - If you cannot tell, answer false. Do NOT speculate.
  - Return ONLY the JSON object. No prose explanation.
"""


def build_phase2_prompt() -> str:
    """Phase 2: run the v11 shot_timestamps prompt verbatim (or close to it).

    This measures what Gemini reports under the production classification
    framing — comparing to Phase 1 reveals which features get inflated
    when Gemini is asked to enumerate shots vs asked to describe what's
    visible.
    """
    # Note: we don't load the actual v11 prompt here because we want
    # a minimal version that captures the framing effect, not the
    # full goal-detection machinery. The key behavior we want to
    # measure is: under "list all shots" framing, what does Gemini
    # report at the same candidate seconds?
    return """You are an expert hockey video analyst. Watch this clip and
identify every distinct shot attempt — a player winding up and releasing
the puck toward the net.

For each shot, return:
  - timestamp (MM:SS within this clip — the moment the puck leaves the stick)
  - location (e.g. "high slot", "right point", "in close")
  - release (e.g. "wrist shot", "snap shot", "slap shot")
  - outcome (one of: goal, save, miss, blocked)

It is much better to UNDER-count than to inflate. If you cannot point
to a specific puck-leaves-stick frame, do NOT include the entry.

Return ONLY a JSON object:
{
  "shots": <int>,
  "shot_timestamps": [
    {"timestamp": "MM:SS", "location": "...", "release": "...", "outcome": "..."}
  ]
}
"""


# ---------------------------------------------------------------------------
# Clip extraction
# ---------------------------------------------------------------------------

def extract_clip(video_path: str, t_start: float, t_end: float,
                 out_path: str) -> bool:
    """Extract a clip with ffmpeg. Stream-copy first; re-encode if duration
    drifts > 2s (same heuristic as feedback_seg/video.py)."""
    duration = t_end - t_start
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(t_start), "-i", video_path,
        "-t", str(duration), "-c", "copy",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg failed: {e.stderr.decode()[:200]}", file=sys.stderr)
        return False
    # Probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", out_path],
        capture_output=True, text=True,
    )
    try:
        actual = float(probe.stdout.strip())
    except ValueError:
        return False
    if abs(actual - duration) > 2.0:
        # Re-encode for accuracy
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(t_start), "-i", video_path,
            "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-b:a", "96k",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return False
    return True


# ---------------------------------------------------------------------------
# Clip set construction
# ---------------------------------------------------------------------------

CLIP_HALF_WIDTH_SEC = 15   # ±15s around the candidate moment → 30s clips
MIN_INTER_SHOT_GAP  = 60   # exclude clips where another shot is within ±60s


def build_clip_set(
    video_dir: str,
    gt_dir: str,
    metrics_dir: str,
    hudl_id_map: dict[str, str],
    workspace: str,
    n_real_shots: int = 20,
    n_zone_noise: int = 10,
) -> list[ClipRecord]:
    """Build the test clip set."""
    rng = random.Random(42)
    clips: list[ClipRecord] = []

    # 1. Real-shot clips: pick Hudl Shots events that are isolated
    #    (no other shot within ±MIN_INTER_SHOT_GAP) so the clip's
    #    real-shot moment is unambiguous.
    print("Building real-shot clips...", file=sys.stderr)
    real_candidates: list[tuple[str, str, float, float, str]] = []
    for vID, hudl_id in hudl_id_map.items():
        gt_path = os.path.join(gt_dir, f"gt_{hudl_id}.csv")
        video_path = os.path.join(video_dir, f"full_{vID}.mp4")
        if not (os.path.exists(gt_path) and os.path.exists(video_path)):
            print(f"  [{vID}] missing GT or video; skipping", file=sys.stderr)
            continue
        shots = []
        with open(gt_path) as f:
            for row in csv.DictReader(f):
                if row.get("action") in ("Shots", "Goals"):
                    try:
                        shots.append((float(row["start"]), float(row["end"]),
                                      row.get("team", "")))
                    except (ValueError, KeyError):
                        continue
        # Isolated shots
        for ss, se, team in shots:
            mid = (ss + se) / 2
            if any(abs(mid - (os_ + oe) / 2) < MIN_INTER_SHOT_GAP
                   for os_, oe, _ in shots
                   if (os_, oe) != (ss, se)):
                continue
            t_start = max(0, mid - CLIP_HALF_WIDTH_SEC)
            t_end   = mid + CLIP_HALF_WIDTH_SEC
            real_candidates.append((vID, video_path, t_start, t_end, team))

    rng.shuffle(real_candidates)
    for i, (vID, video_path, t_start, t_end, team) in enumerate(real_candidates[:n_real_shots]):
        shot_sec_in_clip = CLIP_HALF_WIDTH_SEC  # the shot is at the middle
        clip_id = f"shot_{vID[:5]}_t{int(t_start + CLIP_HALF_WIDTH_SEC)}"
        clip = ClipRecord(
            clip_id=clip_id, label="real_shot", vID=vID,
            video_path=video_path, t_start=t_start, t_end=t_end,
            extra={"team": team},
        )
        # Candidate seconds: the real shot moment + 1 before + 1 after
        clip.candidates = [
            CandidateSecond(clip_id=clip_id, sec_in_clip=shot_sec_in_clip,
                            label="real_shot",
                            notes=f"hudl shot at absolute t={t_start + shot_sec_in_clip:.0f}"),
            CandidateSecond(clip_id=clip_id, sec_in_clip=max(0, shot_sec_in_clip - 8),
                            label="pre_shot"),
            CandidateSecond(clip_id=clip_id, sec_in_clip=min(29, shot_sec_in_clip + 8),
                            label="post_shot"),
        ]
        clips.append(clip)

    # 2. Zone-noise clips: cv_seg threat windows where Gemini predicted
    #    2+ shots but Hudl had 0 shots in the window AND ±MIN_INTER_SHOT_GAP.
    print("Building zone-noise (ghost-shot) clips...", file=sys.stderr)
    noise_candidates: list[tuple[str, str, float, float, list[int]]] = []
    for vID, hudl_id in hudl_id_map.items():
        metrics_file = os.path.join(metrics_dir, f"gt_metrics_{vID}.json")
        gt_path = os.path.join(gt_dir, f"gt_{hudl_id}.csv")
        video_path = os.path.join(video_dir, f"full_{vID}.mp4")
        if not (os.path.exists(metrics_file) and os.path.exists(gt_path)
                and os.path.exists(video_path)):
            continue
        metrics = json.load(open(metrics_file))
        hudl_shots = []
        with open(gt_path) as f:
            for row in csv.DictReader(f):
                if row.get("action") in ("Shots", "Goals"):
                    try:
                        hudl_shots.append((float(row["start"]), float(row["end"])))
                    except (ValueError, KeyError):
                        continue
        for seg in metrics:
            if not seg.get("segmentHasThreat"):
                continue
            m = seg.get("metrics") or {}
            n_shots = m.get("shots", 0) or 0
            if n_shots < 2:
                continue
            ws = float(seg.get("segment_start", 0))
            we = float(seg.get("segment_end", 0))
            # Window must have NO Hudl shots in or near it
            if any(ge > ws - MIN_INTER_SHOT_GAP and gs < we + MIN_INTER_SHOT_GAP
                   for gs, ge in hudl_shots):
                continue
            # Take 30s centered on the window midpoint
            mid = (ws + we) / 2
            t_start = max(0, mid - CLIP_HALF_WIDTH_SEC)
            t_end   = mid + CLIP_HALF_WIDTH_SEC
            # Candidate seconds: 5 evenly-spaced moments
            cand_secs = [3, 9, 15, 21, 27]
            noise_candidates.append((vID, video_path, t_start, t_end, cand_secs))

    rng.shuffle(noise_candidates)
    for vID, video_path, t_start, t_end, cand_secs in noise_candidates[:n_zone_noise]:
        clip_id = f"noise_{vID[:5]}_t{int(t_start + CLIP_HALF_WIDTH_SEC)}"
        clip = ClipRecord(
            clip_id=clip_id, label="zone_noise", vID=vID,
            video_path=video_path, t_start=t_start, t_end=t_end,
        )
        for s in cand_secs:
            clip.candidates.append(
                CandidateSecond(clip_id=clip_id, sec_in_clip=s, label="zone_noise")
            )
        clips.append(clip)

    # 3. Extract all clips
    print(f"\nExtracting {len(clips)} clips...", file=sys.stderr)
    os.makedirs(workspace, exist_ok=True)
    extracted: list[ClipRecord] = []
    for clip in clips:
        out_path = os.path.join(workspace, f"{clip.clip_id}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
            clip.clip_path = out_path
            extracted.append(clip)
            continue
        if extract_clip(clip.video_path, clip.t_start, clip.t_end, out_path):
            clip.clip_path = out_path
            extracted.append(clip)
        else:
            print(f"  failed to extract {clip.clip_id}", file=sys.stderr)
    print(f"  {len(extracted)} / {len(clips)} clips extracted", file=sys.stderr)
    return extracted


# ---------------------------------------------------------------------------
# Gemini probe execution
# ---------------------------------------------------------------------------

def _get_genai_client(project_id: str, region: str):
    """Vertex AI client, matching discrim_probe.py."""
    return genai.Client(vertexai=True, project=project_id, location=region)


def query_gemini(client, clip_path: str, prompt: str, model_name: str,
                 max_retries: int = 3) -> Optional[dict]:
    """Send a clip + prompt to Gemini, expect JSON back."""
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
            print(f"    JSON decode error attempt {attempt+1}: {e}", file=sys.stderr)
            time.sleep(2)
        except Exception as e:
            print(f"    API error attempt {attempt+1}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            time.sleep(5)
    return None


def run_probe(
    clips: list[ClipRecord],
    output_dir: str,
    model_name: str,
    project_id: str,
    region: str,
) -> dict:
    """Run both phases of the probe, save raw responses, return summary."""
    client = _get_genai_client(project_id, region)
    os.makedirs(output_dir, exist_ok=True)

    phase1_results: list[dict] = []
    phase2_results: list[dict] = []

    for i, clip in enumerate(clips):
        print(f"\n[{i+1}/{len(clips)}] {clip.clip_id} ({clip.label})", file=sys.stderr)

        # PHASE 1: per-second atomic observations
        cand_secs = [c.sec_in_clip for c in clip.candidates]
        p1_prompt = build_phase1_prompt(cand_secs)
        p1_resp = query_gemini(client, clip.clip_path, p1_prompt, model_name)
        if p1_resp is not None:
            phase1_results.append({
                "clip_id":    clip.clip_id,
                "clip_label": clip.label,
                "vID":        clip.vID,
                "candidates": [asdict(c) for c in clip.candidates],
                "response":   p1_resp,
            })

        # PHASE 2: v11-style framing
        p2_prompt = build_phase2_prompt()
        p2_resp = query_gemini(client, clip.clip_path, p2_prompt, model_name)
        if p2_resp is not None:
            phase2_results.append({
                "clip_id":    clip.clip_id,
                "clip_label": clip.label,
                "vID":        clip.vID,
                "response":   p2_resp,
            })

    # Persist raw responses
    with open(os.path.join(output_dir, "phase1_raw.json"), "w") as f:
        json.dump(phase1_results, f, indent=2)
    with open(os.path.join(output_dir, "phase2_raw.json"), "w") as f:
        json.dump(phase2_results, f, indent=2)
    print(f"\nRaw responses written to {output_dir}/phase1_raw.json and phase2_raw.json",
          file=sys.stderr)

    # Analysis
    return analyze_results(phase1_results, phase2_results, output_dir)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_results(
    phase1_results: list[dict],
    phase2_results: list[dict],
    output_dir: str,
) -> dict:
    """Aggregate per-feature discrimination scores."""
    # Build a per-(clip, second, label, feature) table from Phase 1
    rows: list[dict] = []
    for clip_entry in phase1_results:
        candidates = clip_entry["candidates"]
        observations = clip_entry["response"].get("observations", [])
        # Match observations to candidates by second
        for cand in candidates:
            obs = next((o for o in observations
                        if o.get("second") == cand["sec_in_clip"]), None)
            if obs is None:
                continue
            for feat in FEATURE_NAMES:
                rows.append({
                    "clip_id":  clip_entry["clip_id"],
                    "second":   cand["sec_in_clip"],
                    "label":    cand["label"],
                    "feature":  feat,
                    "fired":    bool(obs.get(feat, False)),
                })

    # Write the long-format probe results TSV
    p1_tsv_path = os.path.join(output_dir, "phase1_probe_results.tsv")
    with open(p1_tsv_path, "w") as f:
        f.write("clip_id\tsecond\tlabel\tfeature\tfired\n")
        for r in rows:
            f.write(f"{r['clip_id']}\t{r['second']}\t{r['label']}\t"
                    f"{r['feature']}\t{int(r['fired'])}\n")

    # Per-feature aggregation
    feature_stats: dict[str, dict] = {}
    for feat in FEATURE_NAMES:
        feat_rows = [r for r in rows if r["feature"] == feat]
        n_real   = sum(1 for r in feat_rows if r["label"] == "real_shot")
        n_noise  = sum(1 for r in feat_rows if r["label"] == "zone_noise")
        f_real   = sum(1 for r in feat_rows if r["label"] == "real_shot"  and r["fired"])
        f_noise  = sum(1 for r in feat_rows if r["label"] == "zone_noise" and r["fired"])
        # We're most interested in real_shot vs zone_noise discrimination.
        rate_real  = f_real / n_real if n_real else 0.0
        rate_noise = f_noise / n_noise if n_noise else 0.0
        feature_stats[feat] = {
            "n_real":     n_real,
            "n_noise":    n_noise,
            "fire_real":  f_real,
            "fire_noise": f_noise,
            "rate_real":  rate_real,
            "rate_noise": rate_noise,
            "discrimination": rate_real - rate_noise,
        }

    # Phase 2 inflation: per-clip, how many shots did Gemini report vs
    # how many real shots were in the clip?
    inflation_stats: dict[str, dict] = {}
    for entry in phase2_results:
        clip_label = entry["clip_label"]
        n_pred = len(entry["response"].get("shot_timestamps", []))
        # Real-shot clips have 1 real shot; zone-noise clips have 0.
        n_true = 1 if clip_label == "real_shot" else 0
        inflation_stats.setdefault(clip_label, {"n_clips": 0, "n_pred": 0, "n_true": 0})
        inflation_stats[clip_label]["n_clips"] += 1
        inflation_stats[clip_label]["n_pred"]  += n_pred
        inflation_stats[clip_label]["n_true"]  += n_true

    # Write the feature-analysis TSV
    fa_tsv_path = os.path.join(output_dir, "phase1_feature_analysis.tsv")
    with open(fa_tsv_path, "w") as f:
        f.write("feature\tn_real\tn_noise\tfire_real\tfire_noise\t"
                "rate_real\trate_noise\tdiscrimination\n")
        for feat in FEATURE_NAMES:
            s = feature_stats[feat]
            f.write(f"{feat}\t{s['n_real']}\t{s['n_noise']}\t"
                    f"{s['fire_real']}\t{s['fire_noise']}\t"
                    f"{s['rate_real']:.3f}\t{s['rate_noise']:.3f}\t"
                    f"{s['discrimination']:+.3f}\n")

    # Summary report
    summary = {
        "n_phase1_clips":     len(phase1_results),
        "n_phase2_clips":     len(phase2_results),
        "n_phase1_rows":      len(rows),
        "feature_stats":      feature_stats,
        "phase2_inflation":   inflation_stats,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print readable summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("DISCRIMINATION RESULTS — feature fire rates by label", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"  {'feature':<40} {'real':>6} {'noise':>6} {'disc':>6}", file=sys.stderr)
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6}", file=sys.stderr)
    for feat in sorted(FEATURE_NAMES,
                       key=lambda f: -feature_stats[f]["discrimination"]):
        s = feature_stats[feat]
        marker = ""
        if s["discrimination"] >= 0.40:
            marker = " <-- strong candidate"
        elif s["discrimination"] >= 0.25:
            marker = " <-- moderate"
        elif s["discrimination"] <= -0.20:
            marker = " <-- anti-anchor (fires on noise)"
        print(f"  {feat:<40} {s['rate_real']:>6.2f} "
              f"{s['rate_noise']:>6.2f} {s['discrimination']:>+6.2f}{marker}",
              file=sys.stderr)

    print("\n" + "=" * 70, file=sys.stderr)
    print("PHASE 2 INFLATION — Gemini under v11-style framing", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    for label, stats in inflation_stats.items():
        rate = stats["n_pred"] / stats["n_clips"] if stats["n_clips"] else 0
        print(f"  {label:<20} clips={stats['n_clips']:>3}  "
              f"pred shots={stats['n_pred']:>3}  "
              f"true shots={stats['n_true']:>3}  "
              f"avg pred/clip={rate:.2f}",
              file=sys.stderr)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_id_map(s: str) -> dict[str, str]:
    """Parse 'vID1:hudl1,vID2:hudl2,...' to a dict."""
    pairs = [p.strip() for p in s.split(",") if p.strip()]
    return dict(p.split(":", 1) for p in pairs)


def parse_args():
    p = argparse.ArgumentParser(
        description="Shot-moment discrimination probe (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--hudl-id-map", required=True,
                   help="Comma-separated vID:hudl_id pairs")
    p.add_argument("--gt-dir", required=True,
                   help="Directory of gt_{hudl_id}.csv files")
    p.add_argument("--video-dir", required=True,
                   help="Directory of full_{vID}.mp4 files")
    p.add_argument("--metrics-dir", required=True,
                   help="Directory of gt_metrics_{vID}.json files (for ghost-shot detection)")
    p.add_argument("--output-dir", required=True,
                   help="Where to write probe outputs")
    p.add_argument("--workspace", default=None,
                   help="Where to extract clips to (default: <output-dir>/clips)")
    p.add_argument("--n-real-shots", type=int, default=20)
    p.add_argument("--n-zone-noise", type=int, default=10)
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--project-id", default="goalie-analytics-pro-dev")
    p.add_argument("--region", default="us-central1")
    return p.parse_args()


def main():
    args = parse_args()
    hudl_id_map = parse_id_map(args.hudl_id_map)
    workspace = args.workspace or os.path.join(args.output_dir, "clips")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Hudl ID map: {hudl_id_map}", file=sys.stderr)
    print(f"Output dir:  {args.output_dir}", file=sys.stderr)
    print(f"Workspace:   {workspace}", file=sys.stderr)
    print(f"Model:       {args.model}", file=sys.stderr)

    clips = build_clip_set(
        video_dir=args.video_dir,
        gt_dir=args.gt_dir,
        metrics_dir=args.metrics_dir,
        hudl_id_map=hudl_id_map,
        workspace=workspace,
        n_real_shots=args.n_real_shots,
        n_zone_noise=args.n_zone_noise,
    )

    if not clips:
        print("ERROR: no clips extracted", file=sys.stderr)
        return 1

    print(f"\nClip set: {sum(1 for c in clips if c.label == 'real_shot')} real, "
          f"{sum(1 for c in clips if c.label == 'zone_noise')} noise",
          file=sys.stderr)

    summary = run_probe(
        clips=clips,
        output_dir=args.output_dir,
        model_name=args.model,
        project_id=args.project_id,
        region=args.region,
    )

    print(f"\nProbe complete. Outputs in {args.output_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
