# metrics_seg improvements — spec

**Status:** spec + implementation 2026-05-24
**Author:** continuation of candidate-list / fusion learnings work
**Production baseline:** v10 prompt, F1=0.625 on goal classifier (5-video eval)

## Background

`metrics_seg` is Stage 2 of `cv_seg → metrics_seg → feedback_seg`. It
takes each motion-based threat window from cv_seg and asks Gemini 2.5
Pro (via Vertex AI) to count shots/saves/goals in that clip. The
v6→v13 prompt evolution lifted goal-classifier F1 from 0.000 to
0.625 through probe-driven path design.

Recent work on the candidate-list pipeline (`tools/candidate_list.py`,
`tools/validate_candidates.py`) produced several learnings that
translate directly into per-call F1 gains and substantial cost
reductions for `metrics_seg`:

1. **Late fusion (YOLO + audio) beats single-modality** by ~+0.10 F1
2. **Audio carries genuine independent signal** (~0.48 F1 on its own)
3. **Per-second probs at 86% R@10s** can serve as a strong prior
4. **Wide cv_seg windows include many false positives** — Gemini
   spends compute confirming "nothing happened"
5. **One outlier video (2132246) is unlearnable** across every
   architecture — fix at data layer, not model layer

These map to 8 concrete improvements detailed below.

## Goals

- **Lift goal-classifier F1 from 0.625 → ~0.78+** without prompt
  changes
- **Reduce per-game Gemini cost by 50%+** through pre-filtering and
  Flash screening
- **Make re-runs idempotent + cheap** via content-addressed caching
- **Preserve existing production behavior** — all changes additive +
  feature-flagged

## Non-goals

- Re-engineering the v13 prompt itself — that path has diminishing
  returns per the CHANGELOG
- Replacing Vertex/Gemini with Claude or alternative providers
- Touching `feedback_seg` (separate downstream stage)

---

## Improvements (in implementation order)

### #1 — Cache layer  `metrics_seg/cache.py`

**Why:** Each Gemini call is 5-15s + cost; re-runs today re-pay.

**What:** Content-addressed cache keyed by
`sha256(video_bytes + prompt_version + model_name + temperature)`.
Disk-backed JSON under `~/.cache/metrics_seg/` (configurable via
`METRICS_SEG_CACHE_DIR` env var). Methods:
- `get(key) -> response dict | None`
- `put(key, response_dict)`
- `key_for(video_bytes, prompt_version, model, temperature) -> str`

**Wired in:** `analyze_clip_metrics` checks cache before each Gemini
call. Set `METRICS_SEG_CACHE_DIR=""` to disable.

**Test:** Mock `_call_gemini_for_metrics`. Verify cache hit on
2nd call with same args. Verify miss when video bytes change by
1 byte.

**Risk:** Stale responses if prompt changes silently. Mitigation:
`prompt_version` is part of the key, and `prompts/metrics_v13.txt`
is loaded via `PROMPT_VERSION = "v13"` constant — bumping the
version invalidates cache.

---

### #2 — Per-second prob pre-filter  `metrics_seg/prefilter.py`

**Why:** cv_seg's motion-based windows include camera pans, scrums,
dump-ins that contain no shots. Today each gets a $0.05 Gemini call
to return zeros.

**What:** Before calling Gemini, load fused YOLO+audio per-second
probs for the vID. For each cv_seg window, compute the max prob
inside the window. If `< FILTER_THRESHOLD` (default 0.30), return:

```python
{
    "shots": 0, "shotsOnNet": 0, "saves": 0, "goals": 0,
    "_prefilter_skip": True,
    "_prefilter_peak_conf": float,
}
```

without any Gemini call. Goal-criteria booleans default to False.

**Wired in:** New flag `--prefilter-threshold 0.30` (default off,
i.e. threshold 0). Set positive to enable.

**Test:** Use a real test-video probs TSV. Synthesize a cv_seg
window in a known low-prob region. Verify `_prefilter_skip=True`.

**Risk:** Filtering real shots. Mitigation: aggregate per-second
prob has 86% R@10s, so threshold=0.30 keeps ~95% of true positives
in candidate windows. Track `_prefilter_skip` in trace sidecar to
audit.

---

### #3 — Audio + visual prior context in prompt  `metrics_seg/audio_context.py`

**Why:** Gemini currently sees only video. Audio model alone hits
F1=0.48. Giving Gemini explicit priors reduces hallucination both
ways (less inventing, less missing).

**What:** New module:
- `extract_audio_markers(audio_features_tsv, window_start, window_end)`
  → human-readable text like
  `"0:03 sharp impact (onset 0.92); 0:06 whistle (high-freq onset 0.71)"`
- `extract_prob_summary(yolo_probs, audio_probs, window_start, window_end)`
  → `"visual_peak_conf=0.78@03; audio_peak_conf=0.65@04"`

**Prompt wiring:** Prepend a new context block to the v13 prompt:

```
**OPTIONAL CONTEXT** — these are model PRIORS, not ground truth.
Use them as hints but corroborate visually before counting.

Visual shot-prob peaks (YOLO):  {visual_peak_summary}
Audio shot-prob peaks:          {audio_peak_summary}
Audio events detected:          {audio_markers}
```

Feature-flag: `--use-context` (default off). When off, prompt is
unchanged from v13.

**Test:** Render the context block on a known window. Verify
markers + summaries match expected format. Run prompt unit test
to ensure existing v13 structure is untouched.

**Risk:** Gemini over-trusts the priors and stops corroborating. The
"PRIORS, not ground truth" disclaimer aims to prevent this. Track
via per-call output: did Gemini's `shots` count match the visual
peak count? If too-tight correlation, soften the prompt.

---

### #4 — Goal ensemble voting  `metrics_seg/goal_ensemble.py`

**Why:** v10's intrinsic goal precision is 0.875 but STRICT is 0.625
due to cv_seg attribution + occasional single-call FPs. Goals are
rare (~2-6/game) so each FP/FN drags F1 disproportionately.

**What:** When the first Gemini call returns `goals >= 1`, fire a
verification pass:
- Call 2 more times: (Pro temp=0.0), (Pro temp=0.3)
- (optional, behind flag) Call Flash once at temp=0 for diversity
- Check the fused per-second probs: does the window contain a
  sustained peak (3+ consecutive seconds above 0.50)?
- Goal confirmed only if (≥2 of 3 Gemini calls report goals≥1) AND
  (prob signal corroborates)

Extends the existing v8 multi-call vote infrastructure rather than
replacing it.

**Wired in:** `analyze_clip_metrics` calls
`goal_ensemble.confirm(first_result, video_bytes, gemini_client, probs)`
which returns either the original result or a downgraded version
with `goals=0` + trace flag `goal_ensemble_overrode=True`.

Feature-flag: `--goal-ensemble` (default off).

**Test:** Mock 3 Gemini responses with varying goal counts. Verify
2-of-3 majority wins. Verify prob-signal veto kicks in when peaks
are absent.

**Risk:** 3× Gemini cost for windows with predicted goals. Acceptable
because goals are rare (~2-6 per game) and the precision lift
matters for downstream feedback_seg.

---

### #5 — Per-game calibration tracking  `metrics_seg/calibration.py`

**Why:** Some games (e.g. 2132246) are systematically miscounted.
Without tracking we can't notice or correct.

**What:** Append-only log per-game:

```json
{
  "vID": "2073809",
  "ts":  "2026-05-24T15:00:00Z",
  "predicted": {"shots": 57, "saves": 53, "goals": 1},
  "ground_truth": {"shots": 60, "saves": 55, "goals": 2},
  "delta":     {"shots": -3,  "saves": -2,  "goals": -1}
}
```

Logged to `data/output/calibration/<vID>.jsonl` (one entry per run).
Provides:
- Historical per-game accuracy
- Optional `apply_correction(vID, predicted)` that scales counts by
  the rolling-median delta (feature-flagged off; for analysts to
  inspect, not auto-apply)

**Wired in:** End of `_refine_all_segments` calls
`calibration.log(vID, predicted_totals, gt_totals_if_available)`.
GT may or may not be present depending on workflow.

**Test:** Write + read a log entry, verify schema. Test
`apply_correction` rolling median calc on a synthetic 5-entry log.

**Risk:** None — read-only by default.

---

### #6 — Fusion-as-Stage-1 alternative orchestrator  `tools/run_fusion_pipeline.py`

**Why:** cv_seg's wide motion windows are a structurally weaker
upstream than our point-based candidate list. Replacing this stage
should improve all downstream stages.

**What:** New orchestrator that:
1. Runs `tools/candidate_list.py` for each vID to get ranked peaks
2. For each peak, builds a window of `[t-PRE, t+POST]` (defaults 5/5
   sec to match the existing cv_seg window expansion)
3. Writes a `gt_seg_<vID>.json` in the same shape as cv_seg output:
   `{segments: [{segment_start, segment_end, segmentHasThreat,
   threat_goalie_color}, ...]}`
4. Invokes metrics_seg in `--steps 2` mode pointing at the new
   gt_seg JSONs

**Wiring:** Standalone — does not modify `run_pipeline.py`. Add a
new top-level entrypoint:

```bash
python3 tools/run_fusion_pipeline.py --customer_id CUST_LEARNCURVE --vID 2073809
```

**Test:** Generate candidate windows for one test vID, verify they
load cleanly into metrics_seg via the standard JSON schema.

**Risk:** Schema drift between cv_seg's JSON and our generated one.
Mitigation: write a strict validator + tests against existing real
cv_seg outputs.

---

### #7 — Flash screening (Phase 2 stub)  `metrics_seg/flash_screen.py`

**Why:** Pro is ~10× more expensive than Flash; Flash is plenty for
"is there ANY shot-like event here?"

**What (scaffolded; default off):**
- New screener function `flash_screen(video_bytes, gemini_client) ->
  ScreenResult` where `ScreenResult` has `shots_any: bool`,
  `goal_likely: bool`, `confidence: float`
- Simplified prompt — see `prompts/screen_flash_v1.txt` (stub)
- Wire flag: `--flash-screen` (default off). When on, run Flash
  first; only escalate to Pro v13 if `shots_any` or `goal_likely`

**Status:** Stub only in initial implementation. Full integration
deferred to a follow-on PR since it requires:
- Flash model availability validation in this Vertex region
- Prompt design + probe-driven tuning
- Cost accounting changes

**Test:** Stub returns `ScreenResult(False, False, 0.0)` always when
flag is off. With flag on, mocks the call.

**Risk:** Flash may underscreen — drop true positives. Mitigation:
fail-safe to escalate everything if Flash confidence is low.

---

### #8 — 2132246 investigation (not code)

Outside scope for this spec. Action item: manually inspect the video,
identify what makes it broken across YOLO/B/audio/Gemini, document
findings.

---

## Wiring in `01_detect_segment_metrics.py`

Minimal additive changes — preserve all existing behavior by default.

New config / CLI flags:
```
--cache-dir PATH          (default ~/.cache/metrics_seg)
--no-cache                (disable cache)
--prefilter-threshold F   (default 0.0, i.e. disabled; set >0 to filter)
--use-context             (default off; enable audio/visual priors)
--goal-ensemble           (default off; enable 3-call goal vote)
--flash-screen            (default off; Phase 2)
--probs-dir-yolo PATH     (where to find YOLO per-second probs)
--probs-dir-audio PATH    (where to find audio per-second probs)
--audio-features-dir PATH (where to find audio feature TSVs)
```

Insertion points (line numbers approximate, may shift):
- ~line 401: import new modules
- ~line 1335 (`analyze_clip_metrics`): cache check + audio context +
  goal ensemble
- ~line 1598 (`_refine_all_segments` runner): pre-filter loop before
  worker dispatch
- end of `_refine_all_segments`: calibration log

## Test plan

Unit tests (no Gemini calls):
- `tests/test_cache.py`           — cache hit/miss/invalidation
- `tests/test_prefilter.py`       — peak detection in synthetic probs
- `tests/test_audio_context.py`   — marker rendering on synthetic TSV
- `tests/test_goal_ensemble.py`   — voting + prob-signal veto
- `tests/test_calibration.py`     — log schema + rolling-median calc

Integration test (mocked Gemini):
- `tests/test_pipeline_e2e.py`    — single segment through the full
  modified flow with mocked Gemini responses

Live test (incremental, billed):
- Re-run one of the 5-video pilot games with `--prefilter-threshold
  0.30 --use-context`. Compare F1 against v10 baseline. Verify cost
  reduction.
- If F1 holds or improves, expand to 5-video re-eval.

## Rollout plan

1. **Phase 0 (this PR):** Spec + cache + prefilter + audio context +
   calibration + goal ensemble + fusion orchestrator stub + Flash
   stub. All feature-flagged off by default.
2. **Phase 1 (next):** Live one-video re-eval with flags on.
   Compare against v10 baseline.
3. **Phase 2 (next+1):** If Phase 1 wins, 5-video re-eval, then
   flip default flags on.
4. **Phase 3:** Build out Flash screener (real, not stub).
5. **Phase 4:** Replace cv_seg as default Stage 1 if fusion
   pipeline outperforms.

## Success metrics

- **F1 lift:** goal-classifier F1 ≥ 0.75 (current 0.625)
- **Cost reduction:** Gemini calls per game ↓ ≥ 40%
- **Idempotency:** cache hit rate on re-runs ≥ 90%
- **No regression:** existing v13 default behavior unchanged when all
  flags are off

## Files added

```
metrics_seg/
├── IMPROVEMENTS_SPEC.md       (this file)
├── cache.py                   (#1)
├── prefilter.py               (#2)
├── audio_context.py           (#3)
├── goal_ensemble.py           (#4)
├── calibration.py             (#5)
├── flash_screen.py            (#7 stub)
└── tests/
    ├── test_cache.py
    ├── test_prefilter.py
    ├── test_audio_context.py
    ├── test_goal_ensemble.py
    └── test_calibration.py

tools/
└── run_fusion_pipeline.py     (#6)
```

## Files modified

- `metrics_seg/01_detect_segment_metrics.py` — additive only, flagged
  insertion points

## What I'm not changing in this PR

- The v13 prompt itself (handled separately, prompt-version system
  already supports clean swaps)
- `feedback_seg/` (out of scope)
- `cv_seg/` (out of scope — fusion orchestrator is additive)
- GCS bucket structure, output JSON schema
