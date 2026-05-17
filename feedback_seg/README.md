# feedback_seg

Stage 4 of the goalie analytics pipeline: takes the per-window metrics
output from `metrics_seg` and produces qualitative coaching feedback by
sending each threat-window clip to Gemini with a "professional
goaltending scout/coach" prompt.

```
cv_seg → metrics_seg → feedback_seg
```

## Output

Per video:
`gs://goalie_video_bucket/analyze_video/03-segment_goalie_feedback/gt_feedback_{vID}.json`

Two-record array:
1. `{"type": "summary", "response": {coaches_summary, coaches_overall_rating, ...}}`
2. `{"type": "windows", "response": [<one record per threat window>]}`

Each window record preserves all original metrics fields and adds:
- `clipID`, `clip_start_time`, `clip_end_time`, `clip_duration`
- `clipShot`, `clipShotCount`, `clipSave`, `clipSaveCount`, `clipHasGoal`
- `goalie_positioning` — depth/angle/squareness ranks + 1-5 confidence
- `coaching_feedback` — rebound rank + actionable cue + 1-5 confidence
- `technical_reasoning` — timestamped narrative
- `analysis_confidence_caveats` — what the camera angle hid (may be empty)

## Run

### Production (GCS)

```bash
python -m feedback_seg --customer_id CUST000048 --vID U7NUbWad0A8
```

### Local development (no GCS round-trips)

```bash
python -m feedback_seg \
    --customer_id CUST000048 \
    --vID mjEeE7p2Hz8 \
    --local-config /Users/me/CUST000048.json \
    --local-video data/videos/full_mjEeE7p2Hz8.mp4 \
    --local-metrics data/output/metrics_v10/gt_metrics_mjEeE7p2Hz8.json \
    --output-dir data/output/feedback_v1 \
    --no-gcs
```

Vertex AI auth is still required (uses `gcloud auth application-default
login` credentials) — `--no-gcs` only skips GCS file I/O, not Vertex.

## Architecture notes

- **Inline bytes vs GCS upload**: for clips < 18 MB (most threat windows
  at 30-45s) we send video bytes inline. Larger clips fall back to the
  upload-URI-delete pattern. This removes hundreds of GCS operations per
  video and saves several minutes of runtime.
- **Stream-copy with re-encode fallback**: ffmpeg `-c copy` is fast but
  silently extends clips when the seek timestamp falls between
  keyframes. We probe the result with `ffprobe`; if it's off by more
  than 2 seconds, we re-encode for accuracy.
- **Type-based retry**: transient errors are detected via
  `google.api_core.exceptions` types, not string matching. Bounded total
  wait: ~7-12 minutes worst case per failing window vs >30 minutes in
  v1.
- **Worker-exception → error record**: if a worker raises an unhandled
  exception, we write an error record for that window's index instead of
  silently dropping it. The output preserves a one-record-per-window
  invariant.
- **Pydantic enum validators**: every classification field
  (`depth_rank`, `cover_angle_rank`, `squareness_rank`,
  `rebound_control_rank`) is validated against its enum at parse time.
  Drift in Gemini's output is caught immediately, not propagated.
- **Deterministic summary rating**: `coaches_overall_rating` is computed
  by an explicit rubric in the summary prompt rather than Gemini
  guessing a percentage. Run-to-run reproducible.

## Tests

```bash
cd feedback_seg/
python -m pytest tests/ -q
```

Tests cover the pure helpers without exercising real Vertex/GCS calls.
