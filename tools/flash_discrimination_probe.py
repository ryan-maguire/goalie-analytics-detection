#!/usr/bin/env python3
"""Phase C: probe Flash's ability to discriminate shot vs no-shot chunks.

Before building a recovery pass, validate that Flash 2.5/3.5-flash can
actually tell the difference between a clip containing a shot and one
that doesn't. The Phase 2 prefilter validation showed Flash over-calls
when the prompt is permissive ("any shot activity?"); this probe uses
a tighter prompt and measures real TPR / TNR on labeled samples.

Sampling:
  Positives: 20 windows from data/output/runs/metrics_v13/*.json
             where shots > 0. Chunk is centered on the first
             shot_timestamp (absolute = segment_start + offset).
  Negatives: 20 windows from the same files where
             shots = shotsOnNet = saves = goals = 0.
             Chunk is the window's middle 25 s.

Method:
  1. ffmpeg-extract a 25 s clip per sample.
  2. Send to Flash with the strict shot-attempt prompt.
  3. Record {flash_says_yes, confidence}.
  4. Compute TPR (sensitivity), TNR (specificity), and the threshold
     sweep so we know the right confidence cutoff for production.

Cost: ~40 Flash calls × $0.005 ≈ $0.20.

Decision rule for proceeding to Phase A:
  - TPR ≥ 0.85 at SOME confidence threshold (don't miss real shots)
  - TNR ≥ 0.70 at that same threshold  (don't add too many FPs)
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

METRICS_V13_DIR = REPO / "data" / "output" / "runs" / "metrics_v13"
VIDEOS_DIR      = REPO / "data" / "videos"

# Tighter than the Phase 2 prefilter prompt. Asks for explicit evidence
# rather than just "any shot-like signal".
SHOT_PROBE_PROMPT = """Watch this {duration}-second hockey clip.

Is there a CLEAR shot attempt on the goalie's net?

A "clear shot attempt" requires AT LEAST ONE of these signals:
  - Puck visibly released toward the net (wrist/snap/slap shot, redirect)
  - Goalie actively reacting to a puck — sliding, going down,
    making a save motion, freezing the puck
  - Puck crossing the goal line / goal celebration
  - Rebound chase in the crease

Return false if the clip shows only:
  - Play in the neutral / defensive zone with no shot toward net
  - A pure faceoff with no follow-up shot
  - Puck possession at center ice / line change
  - Replay slow-motion of a previous play
  - Commercial / static screen / freeze

Return JSON only: {{"shot_attempt": <true|false>, "confidence": <0.0-1.0>}}
"""

SHOT_PROBE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "shot_attempt": {"type": "BOOLEAN"},
        "confidence":   {"type": "NUMBER"},
    },
    "required": ["shot_attempt"],
}

DEFAULT_MODEL    = "gemini-3.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_PROJECT  = "goalie-analytics-pro-dev"
DEFAULT_CHUNK_S  = 25


@dataclass
class Sample:
    vid:          str
    label:        str           # "positive" | "negative"
    start_sec:    int           # chunk absolute start
    duration:    int           # chunk duration
    src_window_start: int       # the v13 window this came from
    src_window_end:   int
    notes:        str           # provenance


def parse_mmss(s: str) -> int:
    m = re.match(r"^(\d+):(\d{2})$", s.strip())
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def collect_samples(n_positive: int, n_negative: int,
                     chunk_s: int, seed: int = 42) -> list[Sample]:
    """Scan metrics_v13/ and build a balanced sample list."""
    positives: list[Sample] = []
    negatives: list[Sample] = []

    metrics_files = sorted(METRICS_V13_DIR.glob("gt_metrics_*.json"))
    for f in metrics_files:
        if "_trace" in f.name:
            continue
        vid = f.stem.replace("gt_metrics_", "")
        # Skip if no local video file
        if not any((VIDEOS_DIR / n).exists()
                    for n in (f"full_{vid}.mp4", f"{vid}.mp4")):
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        for w in data:
            m = (w.get("metrics") or {})
            if not m:
                continue
            seg_start = int(w.get("segment_start", 0))
            seg_end   = int(w.get("segment_end", seg_start + 30))
            shots     = int(m.get("shots", 0) or 0)
            shotsOnNet = int(m.get("shotsOnNet", 0) or 0)
            saves     = int(m.get("saves", 0) or 0)
            goals     = int(m.get("goals", 0) or 0)
            timestamps = m.get("shot_timestamps") or []

            if shots > 0 and timestamps:
                # Positive: center 25s on first shot_timestamp
                ts_rel = parse_mmss(timestamps[0].get("timestamp", "00:00"))
                center = seg_start + ts_rel
                start = max(0, center - chunk_s // 2)
                positives.append(Sample(
                    vid=vid, label="positive",
                    start_sec=start, duration=chunk_s,
                    src_window_start=seg_start, src_window_end=seg_end,
                    notes=f"first_shot_ts={ts_rel}s outcome={timestamps[0].get('outcome')}",
                ))
            elif shots == 0 and shotsOnNet == 0 and saves == 0 and goals == 0:
                # Negative: middle 25s of window
                center = (seg_start + seg_end) // 2
                start = max(0, center - chunk_s // 2)
                negatives.append(Sample(
                    vid=vid, label="negative",
                    start_sec=start, duration=chunk_s,
                    src_window_start=seg_start, src_window_end=seg_end,
                    notes="pro_all_zero",
                ))

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:n_positive] + negatives[:n_negative]


def extract_clip(vid: str, start_sec: int, duration: int, out_path: Path) -> bool:
    """ffmpeg-cut a clip. Returns True on success."""
    src = VIDEOS_DIR / f"full_{vid}.mp4"
    if not src.exists():
        src = VIDEOS_DIR / f"{vid}.mp4"
    if not src.exists():
        return False
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec), "-i", str(src),
        "-t", str(duration), "-c", "copy",
        "-y", str(out_path),
    ]
    try:
        rc = subprocess.run(cmd, check=False, timeout=60).returncode
        return rc == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def call_flash(video_bytes: bytes, duration: int, client) -> dict:
    """Single Flash call with the strict prompt. Returns parsed dict
    or {} on any failure.

    IMPORTANT: Gemini 2.5 Flash uses 'thinking' tokens before emitting
    structured output. With thinking enabled, a 128-token budget gets
    consumed by an "Here is the JSON requested:" prefix before the
    JSON itself can land — finish_reason becomes MAX_TOKENS and
    response.text is the partial prefix. Two fixes:
      - thinking_config(thinking_budget=0) disables thinking → cheap +
        deterministic structured output (preferred for binary classification)
      - max_output_tokens=256 safety margin even if thinking re-enables
    """
    from google.genai import types
    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[
                types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                SHOT_PROBE_PROMPT.format(duration=duration),
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=256,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=SHOT_PROBE_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = (response.text or "").strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}
    except Exception as e:
        return {"_error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-positive", type=int, default=20)
    ap.add_argument("--n-negative", type=int, default=20)
    ap.add_argument("--chunk-sec",  type=int, default=DEFAULT_CHUNK_S)
    ap.add_argument("--out",        default="data/output/evals/flash_discrim_probe.json")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    samples = collect_samples(args.n_positive, args.n_negative,
                                args.chunk_sec, args.seed)
    n_pos = sum(1 for s in samples if s.label == "positive")
    n_neg = sum(1 for s in samples if s.label == "negative")
    print(f"Sampled {n_pos} positive + {n_neg} negative chunks "
          f"from {len(set(s.vid for s in samples))} games "
          f"({args.chunk_sec}s each)", file=sys.stderr)
    if n_pos == 0 or n_neg == 0:
        print("Empty sample set — check METRICS_V13_DIR exists "
              "and contains gt_metrics_*.json", file=sys.stderr)
        sys.exit(1)

    # Late import so the help/dry-path stays import-light
    from google import genai
    client = genai.Client(vertexai=True,
                            project=DEFAULT_PROJECT,
                            location=DEFAULT_LOCATION)

    results = []
    with tempfile.TemporaryDirectory(prefix="flash_probe_") as tmpdir:
        for i, s in enumerate(samples, 1):
            clip = Path(tmpdir) / f"{i:03d}.mp4"
            ok = extract_clip(s.vid, s.start_sec, s.duration, clip)
            if not ok:
                print(f"  [{i}/{len(samples)}] EXTRACT FAILED  vid={s.vid} "
                      f"start={s.start_sec}", file=sys.stderr)
                results.append({"sample": s.__dict__, "extract_failed": True})
                continue
            video_bytes = clip.read_bytes()
            resp = call_flash(video_bytes, s.duration, client)
            shot_attempt = bool(resp.get("shot_attempt", False))
            confidence   = float(resp.get("confidence", 0.0))
            err = resp.get("_error", "")
            print(f"  [{i:>2}/{len(samples)}] {s.label:<8} vid={s.vid} "
                  f"start={s.start_sec:>5}s  shot_attempt={shot_attempt} "
                  f"conf={confidence:.2f}  {err}", file=sys.stderr)
            results.append({
                "sample":       s.__dict__,
                "flash":        resp,
                "shot_attempt": shot_attempt,
                "confidence":   confidence,
            })

    # ── Confusion matrix at multiple thresholds ──────────────────────
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)

    print("\n=== Threshold sweep ===")
    print(f"{'threshold':>10}  {'TPR':>6}  {'TNR':>6}  "
          f"{'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4}")
    print("-" * 60)
    for thr in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9):
        tp = fn = fp = tn = 0
        for r in results:
            if r.get("extract_failed"):
                continue
            label = r["sample"]["label"]
            says_yes = r["shot_attempt"] and r["confidence"] >= thr
            if label == "positive":
                if says_yes: tp += 1
                else:        fn += 1
            else:
                if says_yes: fp += 1
                else:        tn += 1
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        print(f"{thr:>10.2f}  {tpr:>6.2f}  {tnr:>6.2f}  "
              f"{tp:>4} {fn:>4} {fp:>4} {tn:>4}")

    print("\n=== Decision rule ===")
    print("Proceed to Phase A iff at SOME threshold:")
    print("  TPR >= 0.85   AND   TNR >= 0.70")


if __name__ == "__main__":
    main()
