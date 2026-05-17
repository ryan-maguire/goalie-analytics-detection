# 01_detect_segment_metrics.py changelog

## v10 — LOCKED PRODUCTION (2026-05-07)

`prompts/metrics_v10.txt` is the locked production prompt as of
2026-05-07. Decision rationale below.

### 5-video eval results (cv_seg v23.7-net + metrics_v10)

                        STRICT   UNFILTERED
  Goal classifier:
    TP                      10           14
    FP                       6            2
    FN                       6            8
    Precision           0.625        0.875
    Recall              0.625        0.636
    F1                  0.625        0.737

The 0.737 vs 0.625 gap is cv_seg attribution error: 3 of v10's STRICT
"FPs" are real Hudl goals attributed to the wrong color by cv_seg.
The metric model's intrinsic precision is **0.875** independent of
cv_seg attribution.

### Trajectory across all metric prompt versions

  v6  → F1 0.000, P n/a,  R 0.000  (zero goals predicted, by design)
  v7  → F1 0.353, P 0.353, R 0.353  (added paths 5/6/7; FPs spiked)
  v8  → F1 0.421, P 0.333, R 0.571  (negative-anchor guard; failed)
  v9  → F1 0.125, P 0.500, R 0.071  (Path 4 only; too conservative)
  v10 → F1 0.625, P 0.625, R 0.625  ← shipped

### What worked

The breakthrough was switching from theory-based path design (v6-v9
all anchored on celebration + ref point + N + FO, none of which
discriminate goals from FPs) to evidence-based path design driven by
the discrimination probe (probe1 + probe2 in `discrim_probe.py` /
`discrim_probe_v2.py`).

Specifically:
  - `scoreboard_change` and `attacking_team_skates_to_bench` held
    stable across atomic-observation and goal-detection framings,
    indicating Gemini doesn't game them under framing pressure.
  - `puck_retrieved_from_net` and `centre_ice_faceoff_visible` (the
    v9 production anchors) inflated from ~30% atomic FP rate to ~75%
    under goal-detection framing — they were unreliable as path
    anchors specifically because the framing made them gameable.

### Known limitations carried forward

  - **6 FNs** (real goals the metric model missed). Most of these have
    neither Path B (no scoreboard visible in frame) nor Path C (no
    clear bench fist-bump line and/or no sustained crowd). Improving
    recall further would require either additional path features or
    multi-call voting. Future work, not a blocker.
  - **3 TP-unfilt** (real goals in wrong-color windows). These are
    cv_seg attribution errors the metric model would catch if cv_seg
    attribution improved. Possible v24 cv_seg work.
  - **SX5xNJlh6eQ** (the high-camera arena) had 0% STRICT shot
    coverage already in cv_seg — the metric model can't help here.

### Future work bracketed but NOT pursued

  - Probe round 3 on the FN cases to surface Path D candidates
  - cv_seg attribution improvements to close the STRICT/UNFILTERED gap
  - Multi-call voting tuned to v10's path structure
  - Per-video calibration where camera framing differs

---

## v8.4 — 2026-05-06

Bumps to `prompts/metrics_v10.txt`. Also adds three new BOOLEAN fields
to the `goal_criteria` schema: `scoreboard_change`,
`attacking_team_skates_to_bench`, `crowd_cheer_sustained`.
`PROMPT_VERSION = "v10"`.

### What changed (relative to v9)

The v6→v9 prompt evolution kept anchoring goal detection on
celebration + ref point + N + FO. 5-video evals showed a
precision/recall tradeoff but no version reached F1 > 0.42.

A two-phase discrimination probe (44 clips: 22 Hudl Goals + 12 v8 FPs
+ 10 random negatives) tested two questions:

  1. Which atomic features actually discriminate goals from FPs in
     this footage? (probe 1, label-free)
  2. Do those discrimination rates hold up when asked in a goal-
     detection framing? (probe 2, with truth table embedded)

Key probe findings:

  - **Celebration + ref point fire on 75-95% of FPs.** These signals,
    which v6-v9 used as required anchors, do not actually discriminate.
    Removing them from path requirements is the largest single design
    change in v10.
  - **`puck_retrieved_from_net` and `centre_ice_faceoff_visible`
    inflate from 25-33% atomic FP rate to 75% under goal-detection
    framing.** The model games these features when asked to confirm
    a goal. Demoted from path anchors in v10 (still observed for
    diagnostics).
  - **`scoreboard_change` held stable across both framings** (50%
    goals, 17-25% FPs). It is a verifiable visual state change,
    hard to fabricate.
  - **`attacking_team_skates_to_bench` was the only feature flagged
    STRONG_GOAL_SIGNAL** in probe 1 (+0.30 vs FPs, +0.34 vs negs).
    It held stable in probe 2.
  - **`crowd_cheer_sustained` held stable across framings** (55%
    goals, 33% FPs).

### Truth table after v10

| Path | scoreboard_change | ref_points_at_net | attacking_team_skates_to_bench | crowd_cheer_sustained |
| B    | T                 | T                 | any                            | any                   |
| C    | any               | any               | T                              | T                     |

A goal is confirmed if Path B OR Path C fires AND no disqualifier fires.

### Expected outcome

In-sample v10 dry-run on the 44-clip probe set:
  - Path B OR Path C: TP 18/22, FP 8/22, F1 0.75 (this is the in-
    sample upper bound; real-world F1 will be lower)
  - Model's own pred_goals (with full v10 paths and negative anchor):
    TP 19/22, FP 14, F1 0.69

Realistic real-world expectation on a fresh 5-video run:
  - F1 ≈ 0.55-0.65
  - Recall ≈ 0.55-0.65
  - Precision ≈ 0.55-0.65

If real F1 falls below 0.40, the in-sample number was overfit and we
should reconsider whether v10 paths are robust outside the test set.

### Risk

`scoreboard_change` and `attacking_team_skates_to_bench` carry residual
risk because they only met the discrimination bar on a 44-clip sample.
The in-sample numbers may not generalize. The empirical evidence is
stronger than for prior versions, but the test set is still small.

If v10 shows a precision regression vs v8 with little recall improvement,
the right answer is v9 (Path 4 only) — accept low recall in exchange
for high precision.

## v8.3 — 2026-05-06

Prompt-only update: bumps to `prompts/metrics_v9.txt`. No code changes
beyond `PROMPT_VERSION = "v9"`.

### What changed (relative to v8)

5-video v8 eval revealed the Path 7 negative anchor failed to filter
rebound-FP cases. The model self-reports "no save was observed" while
describing what is clearly a save followed by a rebound that did NOT
go in. We can't prompt our way around Gemini's tendency to rationalize
narrative goal-claims when given any pathway to do so.

v9 makes one surgical change: drop everything except Path 4.

### Truth table after v9

| Path | B  | C   | D | N | FO | crowd |
| 4    | T  | any | T | T | T  | any   | All four strong signals required

A goal is confirmed only when celebration AND ref point AND puck
retrieved from the net AND post-goal centre-ice faceoff are ALL
visibly present in the clip. This is the strictest possible standard
under our existing schema.

### Why this is the right move

Per-path STRICT precision in v8:
  Path 4 (B+D+N+FO):    1 TP /  0 FP   precision 1.00
  Path 3 (B+D+FO):      1 TP /  1 FP   precision 0.50
  Path 7 (B+C+D+crowd): 6 TP / 11 FP   precision 0.35

Only Path 4 is reliable. Paths 3 and 7 are coin-flips at best.

### Expected outcome (5-video set)

  v6 baseline: 0 TP / 0 FP / 17 FN, F1 0.000
  v7:          6 TP / 11 FP / 11 FN, F1 0.353
  v8:          8 TP / 16 FP /  6 FN, F1 0.421
  v9 target:   ~2 TP / 0-1 FP / ~15 FN, F1 ~0.20

Recall drops substantially. F1 drops. But precision goes from 33% to
~95%+. For a tool that reports goal counts to users, conservative-but-
correct beats permissive-but-noisy. Users would rather miss 5 of 7
goals than have to fact-check every claim.

### Risk

If even Path 4's strict signals don't work as anchors (e.g., the model
claims N=true and FO=true on a rebound-during-scramble clip), v9 will
still produce FPs. The 5-video v8 data showed Path 4 at 1/1 precision
but the sample size is tiny (1 fire). v9 may reveal Path 4's true
precision is lower at scale.

If v9 still produces >2-3 FPs across the 5 videos, the right answer is
v6 — accept that the metric model genuinely cannot reliably detect
goals on this footage and report goal counts as 0 always.

## v8.2 — 2026-05-06

Prompt-only update: bumps to `prompts/metrics_v8.txt`. No code changes
beyond `PROMPT_VERSION = "v8"`.

### What changed (relative to v7)

5-video v7 eval showed precision regression: 6 TP / 11 FP / F1 0.353 on
405 windows. Most FPs came from two patterns: (1) the `A` (red light)
anchor being claimed without visual evidence, (2) Path 6 (B+D+N) firing
on rebound-during-scramble situations that the model interpreted as
puck-retrieved-from-net.

v8 makes three surgical changes:

1. **`A` (anchor_puck_crosses_line) REMOVED entirely.** Amateur arena
   footage rarely has a working red goal light, and the model was
   observed claiming red-light activations that weren't visible. The
   puck itself is too small at typical camera distances to track
   reliably crossing the line. The schema fields `anchor_puck_crosses_line`
   and `anchor_puck_crosses_line_timestamp` are kept for backward
   compatibility but are now defined as always-false / always-empty.

2. **Path 6 (B+D+N without FO) REMOVED.** Diagnostic showed Gemini
   interpreting "puck near the net during a scramble" as N=true.
   Without FO to confirm a stoppage actually occurred, the evidence
   is unreliable. Path 4 (B+D+N+FO) remains for cases where both
   anchors are visible.

3. **Path 7 negative-anchor guard added.** Path 7 (B+C+D+crowd) now
   fails if a save by the {goalie_color} goalie is observed in the
   seconds immediately before the celebration — gloving, freezing,
   smothering, kicking out, or visible defender clearance. This
   directly targets the dominant FP pattern (rebound-during-scramble
   where the model fabricates "the rebound went in").

### Truth table after v8

| Path | B  | C   | D | N   | FO | crowd |
| 3    | T  | any | T | any | T  | any   | B + D + FO  
| 4    | T  | any | T | T   | T  | any   | B + D + N + FO (strongest)
| 7    | T  | T   | T | any | any | T    | B + C + D + crowd (with negative-save anchor)

Three paths total, all anchored on B (celebration) AND D (ref point).
Two paths require FO; one uses C+crowd as the FO substitute.

### Expected outcome

On the 5-video set:
  - v6 baseline: 0 TP / 0 FP / 17 FN, F1 0.000
  - v7:          6 TP / 11 FP / 11 FN, F1 0.353
  - v8 target:   ~4 TP / ~4 FP / ~13 FN, F1 ~0.40

Path 7 negative-anchor is the most uncertain change — it might
over-suppress real TPs if the model is confused about whether a save
occurred. If v8 recall drops below v7 by more than 1-2 TPs without
proportional FP reduction, the negative anchor is too aggressive.

### Predicted impact on each known goal

| Goal | v7 result | v8 prediction |
|---|---|---|
| t=261 NSW (mjEeE) | TP via Path 7 | likely TP (no save observed) |
| t=629 Amh (mjEeE) | FN | FN (clip-content issue, not prompt) |
| t=812 Amh (mjEeE) | FN | FN (no C, no FO, no N) |
| t=978 Amh (mjEeE) | TP via Path 6 | **likely FN** — Path 6 removed |
| t=1166 Amh (mjEeE) | TP via Path 7 | likely TP if no save observed |
| t=1240 Amh (mjEeE) | TP via Path 4 | TP (Path 4 unchanged) |
| t=1659 Amh (mjEeE) | FN | FN (model reads as save+unrelated FO) |

So mjEeE specifically: 4/7 TP → 3/7 TP. Net loss of 1 TP (t=978).
But the 4 Path 6 FPs across other videos should disappear.

## v8.1 — 2026-05-06

Prompt-only update: bumps to `prompts/metrics_v7.txt`. No code changes
beyond `PROMPT_VERSION = "v7"`.

### What changed

- **Truth table grew from 4 paths to 7.** The original 4 paths all
  required either FO (centre-ice faceoff) or N (puck retrieved from
  net) — the two least-ambiguous post-goal signals. Diagnostic on
  mjEeE7p2Hz8 showed 5 of 7 real Hudl goals had FO=False because the
  cv_seg window (30-45s) ended before the post-goal faceoff occurred.
  Three new paths were added that don't require FO:
    - **Path 5: A + B + D** — puck visibly crosses line + celebration +
      ref points at net. Catches goals where the model can timestamp
      the goal moment but the clip ends before the faceoff.
    - **Path 6: B + D + N** — celebration + ref point + puck retrieved
      from net. Was previously listed as a removed path (cited risk of
      N hallucination); reinstated based on observation that Gemini
      correctly distinguishes "puck retrieval" from "scrum/clearout"
      when the strict N definition is in the prompt.
    - **Path 7: B + C + D + crowd_spike** — celebration + whistle + ref
      point + sustained crowd noise. Highest FP risk path; only
      enabled with the crowd_spike guard.

- **Per-path hallucination guards added** in the prompt explaining
  exactly what each new path requires of the `confirming_detail`
  field. Path 7 has the most explicit anti-FP language because of its
  similarity to big-save patterns.

### Expected outcome

On mjEeE7p2Hz8 (7 Hudl goals):
  - v6 baseline: 0/7 goal recall
  - v7 target: 3-4/7 goal recall, FPs ≤ 2
  - Goals expected to flip TP: t=261 (Path 7), t=812 (Path 5), t=978 (Path 6)
  - Goals expected to stay missed: t=629 (only B+crowd), t=1240 (no D),
    t=1659 (model interprets as save-then-unrelated-FO)

If FP count rises above 2-3 across all 5 videos, revert to v6 or
tighten the new path guards.

## v8 — 2026-05-03

Inline-bytes refactor. Code-only; prompt unchanged.

### What changed

- **Video sent INLINE via `Part.from_bytes`** instead of uploaded to
  GCS first. Eliminates the upload-then-reference round-trip that was
  the bottleneck behind `--workers 4` upload contention timing out 27
  of 72 segments on a real run.
- **No token cost change.** Gemini receives the same video bytes either
  way; transport doesn't affect tokenization. Total tokens per call
  are determined by the clip's content (duration × resolution), not
  how the bytes got there.
- **`MAX_INLINE_VIDEO_BYTES = 18 MB` guard.** Vertex's hard request
  size cap is 20 MB; we leave 2 MB headroom for the prompt and JSON
  envelope. Our typical clips at 30-60s 720p H.264 are 5-12 MB so the
  guard rarely fires, but if an unusually encoded clip exceeds the
  limit we surface a clear failure (`failure_reason=clip_too_large`)
  rather than let the API return an opaque size error.
- **`gcs_upload_file` and `gcs_delete_blob` are no longer called.**
  Left in the module marked UNUSED for backwards-compat with any
  external callers; nothing in the pipeline references them.
- **End-of-run GCS safety sweep removed.** No temp blobs are ever
  created, so there's nothing to sweep.

### What did NOT change

- `Part.from_bytes` was used instead of the consumer Files API
  (`genai.upload_file`). The Files API is **not available** with
  Vertex AI clients (`genai.Client(vertexai=True, ...)`) — it's
  exclusive to the consumer Gemini API key path. Inline bytes are the
  only no-GCS option for Vertex. Confirmed via Google AI docs and the
  google-gemini/cookbook GitHub issue #394.
- The retry logic, backoff caps, transient-error detection, safety/
  recitation-block detection, JSON recovery, vote machinery, trace
  sidecar, and all v7 helpers are unchanged.

### Tradeoff to be aware of

In multi-call vote paths (shot vote and goal vote), the same clip
bytes are re-transmitted on each call — there's no server-side cache.
For a 10 MB clip that triggers the goal vote, that's 30 MB total wire
egress per segment vs the old "upload once, reference 3x" pattern's
~10 MB. Vote-fire rate on real data is ~5-10% of segments, so the
amortized wire-cost difference is small (~1.0-1.2x), and the saved
GCS upload time more than makes up for it.

If clip sizes ever climb (e.g. 1080p source or longer threat windows),
revisit: at 30+ MB per clip the inline path stops being a clear win
and a Files API migration starts to make sense — but Vertex would need
to add Files API support first, or you'd need to switch to the
consumer Gemini API.

---

## v7 — 2026-05-03

Comprehensive review pass. No prompt changes; the v6 prompt is unchanged
in `prompts/metrics_v6.txt` — only the surrounding code, error handling,
and infrastructure were touched.

### Critical bug fixes

- **Async dispatch now actually runs concurrently** (`_dispatch_segments_async`).
  Previous version was async-shaped but every operation inside `_run_one`
  was synchronous and blocking. With `--workers 2` you got serial work.
  Fix: blocking work now dispatched through `loop.run_in_executor` to a
  ThreadPoolExecutor so the worker count is meaningful again.

- **ffmpeg `-ss` placement corrected** (`extract_clip`). Was using fast
  keyframe seek before `-i`, which can land the clip up to one keyframe
  interval (~2-10s) before the requested start. Moved `-ss` after `-i`
  for slow-but-accurate seek. With `-c copy` the clip can still snap to
  the nearest keyframe (~2s drift on broadcasts), but no longer drifts
  by 5-10s as before.

- **ffmpeg subprocess now has timeout** (`extract_clip`). Was no timeout —
  a hung ffmpeg would block the segment forever. Now bounded by
  `EXTRACT_CLIP_TIMEOUT_SEC=300`.

### Error handling

- **Safety / recitation block detection** (`_call_gemini_for_metrics`).
  When Gemini returns `finish_reason=SAFETY/RECITATION/etc`, the response
  has no text; the old code fell through to JSONDecodeError and retried
  6 times with exponential backoff (~30 minutes) before giving up.
  Now we inspect `finish_reason` first and abort immediately on terminal
  failures.

- **Backoff capped at `RETRY_BACKOFF_MAX=60s`**. Previously 6 retries
  totalled ~30 minutes per failed segment. Now ~5 minutes worst case,
  matching typical Gemini transient-error recovery times.

- **Typed transient-error detection**. Was string-matching error messages
  (`"timeout" in str(e)`); now uses `google.api_core.exceptions` classes
  (ServiceUnavailable, DeadlineExceeded, ResourceExhausted, etc.) with
  string-match retained as fallback.

- **`confirming_detail` validation enforced**. The schema requires
  concrete visual specifics when `goals >= 1`; the model sometimes
  returns generic phrasing or empty strings. Now validated post-parse —
  empty/generic detail downgrades the goal claim to 0 (treats it as a
  hallucination since the prompt requires per-claim concrete specifics).

### Observability

- **Per-segment trace sidecar** (`gt_metrics_{vID}_trace.json`). For
  each threat segment, captures the call count, vote-fire status,
  per-call results, and final merged values. Lets you analyse model
  behaviour post-hoc the same way cv_seg's `_signals.json` sidecar
  works for FP analysis.

- **Vote-fire-rate summary** in the per-video log line. Replaces the
  docstring's hand-waved "1.3-1.5x cost" claim with a real `cost_ratio`
  measurement. You'll know per-video what fraction of segments fired
  each vote and how many calls in total.

### Performance

- **Multi-video concurrency**. New `--video-workers` flag. Previous
  code ran videos sequentially in `main()`; now you can process N
  videos in parallel. Total in-flight Gemini calls is `workers ×
  video-workers`.

- **Gemini client cached at module scope**. Was created per-video;
  now a single client is shared across all videos in the process,
  same as `_gcs_client`.

### Operability

- **`--skip-existing` flag**. Skips processing a vID when the output
  already exists at the configured destination. Useful for batch
  reprocessing or recovery from partial failures.

- **Local video glob fallback**. `--local-video-dir` was strict-match
  on `full_{vID}.mp4`; now falls back to `*{vID}*.mp4`/`.mov`/`.mkv`
  if exact name not found. Removes friction for devs storing videos
  with descriptive prefixes.

- **`_setup_logging()` deferred to `main()`**. Was called at import
  time, mutating root logger as a side effect of `import`. Now safe
  to import this module as a library without touching global logging.

### Code quality

- **Prompt extracted to `prompts/metrics_v6.txt`**. The 644-line prompt
  is no longer inline in the Python source. To switch versions, change
  `PROMPT_VERSION` near the top of the file.

- **Dead `_propagate_side_corrections` removed**. Was a deprecated
  no-op kept for "external imports" with no actual external imports.

- **Empty-dict metrics no longer silently excluded from totals**.
  `if s.get("metrics")` was treating `{}` as falsy; changed to
  `is not None`.

- **23 unit tests added** in `tests/test_segmetrics.py` covering
  helpers, post-parse validation, JSON recovery, vote summary, and
  finish-reason extraction.

### Things explicitly NOT changed

- The v6 prompt itself. Verified byte-identical after extraction to
  `prompts/metrics_v6.txt`.
- The vote-trigger thresholds (`MULTICALL_SHOTS_THRESHOLD=4`,
  `MULTICALL_GOAL_VOTE_TRIGGER=1`).
- The `shotsOnNet = saves + goals` identity enforcement.
- The Gemini File API migration was considered but deferred — the
  current GCS-roundtrip approach works, and switching is a larger
  refactor than this pass should bundle.
