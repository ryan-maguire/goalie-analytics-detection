#!/usr/bin/env python3
"""Flash recovery pass — find shots stage 1 missed.

Given an existing stage-1 seg JSON (cv_seg or fusion), compute the
complement (time NOT covered by any window), slice it into chunks at
the median window duration, ask Flash "is there a clear shot attempt?",
and append confidence-passing chunks as NEW windows to the seg JSON.

The augmented seg JSON is cv_seg-schema-compatible — metrics_seg
consumes it unchanged.

Inversion of the Phase 2 prefilter: the prefilter tried to drop
windows stage 1 over-produced (which doesn't exist — stage 1's
target-color filter already trims well). The recovery pass targets
the opposite failure: windows stage 1 missed entirely.

Cost model per ~60-min game:
  ~94 Flash calls × $0.005      = $0.47
  ~28 Pro calls (Flash positives) = paid downstream by metrics_seg
  marginal cost vs no-recovery   = ~$1.90/game

Usage:
    python3 tools/flash_recovery_pass.py \\
        --vID krxhPVLGLz8 \\
        --in-seg  data/output/runs/cv_seg/gt_seg_krxhPVLGLz8.json \\
        --customer-id CUST000031 \\
        --out-seg data/output/runs/cv_seg_with_recovery/gt_seg_krxhPVLGLz8.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VIDEOS_DIR = REPO / "data" / "videos"
DEFAULT_MODEL    = "gemini-3.5-flash"
DEFAULT_LOCATION = "us-central1"
DEFAULT_PROJECT  = "goalie-analytics-pro-dev"

# Strict shot-attempt prompt — same as Phase C probe. Asks for explicit
# evidence rather than "any shot-like signal" (which over-calls).
SHOT_PROMPT = """Watch this {duration}-second hockey clip.

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

SHOT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "shot_attempt": {"type": "BOOLEAN"},
        "confidence":   {"type": "NUMBER"},
    },
    "required": ["shot_attempt"],
}


@dataclass
class Chunk:
    start_sec: int
    end_sec:   int
    @property
    def duration(self) -> int:
        return self.end_sec - self.start_sec


def load_seg(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("segments") or []
    return data if isinstance(data, list) else []


def median_window_duration(segments: list[dict], default: int = 25) -> int:
    if not segments:
        return default
    durs = []
    for s in segments:
        st = int(s.get("segment_start", 0))
        en = int(s.get("segment_end",   0))
        d = en - st
        if d > 0:
            durs.append(d)
    if not durs:
        return default
    durs.sort()
    return durs[len(durs) // 2]


def compute_complement(segments: list[dict], video_dur_sec: int) -> list[tuple[int, int]]:
    """Returns sorted, non-overlapping time intervals NOT covered by any
    window. Skips the first/last 60s of the game to avoid intro/postgame."""
    if not segments:
        return [(60, max(60, video_dur_sec - 60))]
    spans = sorted([(int(s["segment_start"]), int(s["segment_end"]))
                     for s in segments
                     if int(s.get("segment_end", 0)) > int(s.get("segment_start", 0))])
    merged: list[list[int]] = [list(spans[0])]
    for st, en in spans[1:]:
        if st <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    gaps: list[tuple[int, int]] = []
    cur = 60
    for st, en in merged:
        if st > cur:
            gaps.append((cur, st))
        cur = max(cur, en)
    if cur < video_dur_sec - 60:
        gaps.append((cur, video_dur_sec - 60))
    return gaps


def slice_into_chunks(gaps: list[tuple[int, int]], chunk_sec: int) -> list[Chunk]:
    """Cut each gap into back-to-back chunk_sec segments. Trims any
    trailing piece shorter than half a chunk (too narrow to be useful)."""
    out: list[Chunk] = []
    half = chunk_sec // 2
    for st, en in gaps:
        cur = st
        while cur + half <= en:
            chunk_end = min(en, cur + chunk_sec)
            if chunk_end - cur >= half:
                out.append(Chunk(cur, chunk_end))
            cur = chunk_end
    return out


def video_path_for(vid: str) -> Optional[Path]:
    for cand in (VIDEOS_DIR / f"full_{vid}.mp4", VIDEOS_DIR / f"{vid}.mp4"):
        if cand.exists():
            return cand
    return None


def video_duration_sec(path: Path) -> int:
    """ffprobe-based duration. Returns 0 on failure."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=30).strip()
        return int(float(out))
    except Exception:
        return 0


def extract_clip(src: Path, start: int, dur: int, out_path: Path) -> bool:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", str(start), "-i", str(src),
           "-t", str(dur), "-c", "copy",
           "-y", str(out_path)]
    try:
        rc = subprocess.run(cmd, check=False, timeout=60).returncode
        return rc == 0 and out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def call_flash(video_bytes: bytes, duration: int, client, model: str) -> dict:
    """Single Flash call with thinking disabled. Returns parsed dict
    or {} on any failure."""
    from google.genai import types
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                SHOT_PROMPT.format(duration=duration),
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=256,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=SHOT_SCHEMA,
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
        return {"_error": f"{type(e).__name__}: {e}"}


def load_threat_color(customer_id: str, vid: str) -> str:
    """Pull threat_goalie_color from the customer JSON the same way
    tools/run_fusion_pipeline.py does."""
    cust_json = REPO / "data" / "customers" / f"{customer_id}.json"
    if not cust_json.exists():
        return "Unknown"
    try:
        for rec in json.loads(cust_json.read_text()):
            if str(rec.get("vID")) == vid:
                return rec.get("targetGoalieColor") or "Unknown"
    except Exception:
        pass
    return "Unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vID", required=True)
    ap.add_argument("--in-seg",   required=True, type=Path,
                     help="Existing stage-1 seg JSON to augment.")
    ap.add_argument("--out-seg",  required=True, type=Path,
                     help="Output path for the augmented seg JSON.")
    ap.add_argument("--customer-id", required=True,
                     help="For threat_goalie_color attribution.")
    ap.add_argument("--chunk-sec", type=int, default=None,
                     help="Chunk duration. Default: median of existing windows.")
    ap.add_argument("--min-confidence", type=float, default=0.70,
                     help="Flash confidence threshold for recovery. Default 0.70.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--vertex-location", default=DEFAULT_LOCATION)
    ap.add_argument("--vertex-project",  default=DEFAULT_PROJECT)
    ap.add_argument("--dry-run", action="store_true",
                     help="Compute chunks + report cost, skip Flash calls.")
    ap.add_argument("--trace-out", type=Path, default=None,
                     help="Optional sidecar with per-chunk Flash decisions.")
    args = ap.parse_args()

    segments = load_seg(args.in_seg)
    print(f"loaded {len(segments)} existing windows from {args.in_seg}", file=sys.stderr)

    src = video_path_for(args.vID)
    if not src:
        print(f"ERROR: no video file for {args.vID}", file=sys.stderr)
        sys.exit(1)
    print(f"video: {src}", file=sys.stderr)
    dur = video_duration_sec(src)
    if dur <= 0:
        print(f"ERROR: ffprobe failed for {src}", file=sys.stderr)
        sys.exit(1)
    print(f"video duration: {dur}s", file=sys.stderr)

    chunk_sec = args.chunk_sec or median_window_duration(segments, default=25)
    print(f"chunk_sec: {chunk_sec}", file=sys.stderr)

    gaps = compute_complement(segments, dur)
    total_gap = sum(en - st for st, en in gaps)
    print(f"complement: {len(gaps)} gaps, total {total_gap}s ({total_gap*100//dur}%)",
          file=sys.stderr)

    chunks = slice_into_chunks(gaps, chunk_sec)
    print(f"chunks: {len(chunks)}  (est cost: ${len(chunks)*0.005:.2f} for Flash)",
          file=sys.stderr)

    if args.dry_run:
        print(f"--dry-run: not invoking Flash", file=sys.stderr)
        return 0

    threat_color = load_threat_color(args.customer_id, args.vID)
    print(f"threat_goalie_color: {threat_color}", file=sys.stderr)

    from google import genai
    client = genai.Client(vertexai=True,
                           project=args.vertex_project,
                           location=args.vertex_location)

    recovered: list[dict] = []
    trace: list[dict] = []
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="flash_recovery_") as tmpdir:
        for i, c in enumerate(chunks, 1):
            clip = Path(tmpdir) / f"{i:04d}.mp4"
            if not extract_clip(src, c.start_sec, c.duration, clip):
                trace.append({"start": c.start_sec, "end": c.end_sec,
                               "extract_failed": True})
                continue
            resp = call_flash(clip.read_bytes(), c.duration, client, args.model)
            shot = bool(resp.get("shot_attempt", False))
            conf = float(resp.get("confidence", 0.0))
            err  = resp.get("_error", "")
            kept = shot and conf >= args.min_confidence
            if (i % 10) == 0 or kept:
                print(f"  [{i:>3}/{len(chunks)}] {c.start_sec:>5}-{c.end_sec:>5}s  "
                      f"shot={shot} conf={conf:.2f}  "
                      f"{'KEPT' if kept else 'drop'}  {err}",
                      file=sys.stderr)
            trace.append({"start": c.start_sec, "end": c.end_sec,
                           "shot_attempt": shot, "confidence": conf,
                           "kept": kept, "error": err})
            if kept:
                recovered.append({
                    "segmentHasThreat":     True,
                    "threat_goalie_color":  threat_color,
                    "threat_goalie_side":   "unknown",
                    "segment_start":        int(c.start_sec),
                    "segment_end":          int(c.end_sec),
                    "source_signals":       ["flash_recovery"],
                    "n_overlapping_raw":    1,
                    "_recovery_confidence": conf,
                })
    elapsed = time.time() - t0
    print(f"\nFlash recovery: {len(recovered)} new windows from {len(chunks)} "
          f"chunks ({len(recovered)*100//max(len(chunks),1)}% kept), "
          f"{elapsed:.0f}s wall", file=sys.stderr)

    augmented = segments + recovered
    augmented.sort(key=lambda s: int(s.get("segment_start", 0)))

    args.out_seg.parent.mkdir(parents=True, exist_ok=True)
    args.out_seg.write_text(json.dumps(augmented, indent=2))
    print(f"wrote {len(augmented)} total windows → {args.out_seg}", file=sys.stderr)

    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_out.write_text(json.dumps({
            "vID":            args.vID,
            "chunk_sec":      chunk_sec,
            "min_confidence": args.min_confidence,
            "n_original_windows": len(segments),
            "n_chunks":       len(chunks),
            "n_recovered":    len(recovered),
            "elapsed_sec":    elapsed,
            "trace":          trace,
        }, indent=2))
        print(f"wrote trace → {args.trace_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
