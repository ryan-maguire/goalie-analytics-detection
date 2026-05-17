# feedback_seg changelog

## v1.2.2 — 2026-05-09

### Changed
- **GCS path scheme renamed** from `analysis_video/...` to
  `analyze_video/NN-...` with stage-numbered prefixes:
  - `INPUT_PREFIX`:  `analysis_video/segment_metrics` →
    `analyze_video/02-segment_metrics`
  - `OUTPUT_PREFIX`: `analysis_video/segment_goalie_feedback` →
    `analyze_video/03-segment_goalie_feedback`
  - `TEMP_PREFIX`:   `analysis_video/temp_parts` →
    `analyze_video/00-temp_parts`
  README updated to match. No code or test changes; constants-only.
- This is a coordinated change across cv_seg, metrics_seg, and
  feedback_seg. Existing data at the old paths is not migrated; if
  you need historical runs to be readable, copy or symlink the old
  GCS prefixes manually.

## v1.2.1 — 2026-05-08

### Fixed
- **Broken-pipe failures during Gemini calls now retry instead of failing
  the window.** Observed on the n2cy8b755Tg validation run where 3/66
  windows failed with `[Errno 32] Broken pipe` raised inside
  `client.models.generate_content`. The prior `ConnectionError` entry in
  `_TRANSIENT_TYPES` did not catch these because `BrokenPipeError` is an
  `OSError` subclass, not a `ConnectionError`.

  Added to `feedback_seg/retry.py`:
  - `BrokenPipeError` (errno 32 — pipe closed mid-stream)
  - `ConnectionResetError` (peer closed TCP)
  - `ConnectionAbortedError` (local stack aborted)
  - `"Broken pipe"` added to `_TRANSIENT_STRING_MATCHES` as a backstop
    for SDKs that wrap the typed exception inside a generic one

  Expected impact: lifts n2cy success rate from 95.5% → ~100%. No effect
  on validation-failure or schema-mismatch errors (still non-retryable
  by design — see `_is_transient` semantics).

### Tests added
- `test_broken_pipe_is_transient`
- `test_connection_reset_is_transient`
- `test_connection_aborted_is_transient`
- `test_string_fallback_for_wrapped_broken_pipe`
- `test_broken_pipe_retries_then_succeeds` (end-to-end retry behaviour
  with mocked time)

All 81 tests pass (was 76 + 5 new).

### Not changed
- `video.py` — ffmpeg uses `subprocess.run(check=True, timeout=...)`
  with no piping into stdin, so there is no broken-pipe surface there.
  The existing `_ffprobe_duration` check already addresses the
  stream-copy-extends-clips concern.
- `gcs_upload_file` — the n2cy failures were inside the Gemini call,
  not GCS upload (the failed clips were small enough for the
  inline-bytes path). If upload-path failures appear later, that's a
  separate patch.

## v1.2 — 2026-05-07
- Confidence-score normalization (1-5, 1-10, 1-100 → 1-5)
- "Not Applicable" added as canonical rebound label
- Narrative-text rescue (extract first valid label from prose)
- Vertex `response_schema` with `enum` + `minimum`/`maximum` constraints
- Tightened prompt with explicit 1-5 instruction
