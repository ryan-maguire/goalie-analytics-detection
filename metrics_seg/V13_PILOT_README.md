# v13 — evidence-based shot detection (pilot)

## What changed vs v11.2 (production lock)

v13 replaces the v11 "enumerate every shot" instruction with an
evidence-based truth table built from the 14-video discrimination probe.

The probe found four features with very strong discrimination between
real shots and zone-pressure-but-no-shot moments:

| feature                                 | real-shot rate | noise rate | disc   |
|-----------------------------------------|----------------|------------|--------|
| `feature_puck_traveling_toward_net`     | 0.82           | 0.00       | +0.82  |
| `feature_puck_release_frame_visible`    | 0.71           | 0.10       | +0.61  |
| `feature_puck_impact_sound_audible`     | 0.71           | 0.20       | +0.51  |
| `feature_puck_carrier_holds_or_passes`  | 0.12           | 0.70       | −0.58 (anti-anchor) |

For context: v10's strongest goal discriminator was
`attacking_team_skates_to_bench` at +0.34. The shot discriminators are
substantially stronger.

## v13 truth table

For each candidate shot moment, Gemini must evaluate four booleans.
A candidate becomes a real shot only if:

  - **REQUIRED:** `feature_puck_traveling_toward_net` is TRUE
  - **EVIDENCE:** at least one of
      `feature_puck_release_frame_visible` OR
      `feature_puck_impact_sound_audible` is TRUE
  - **DISQUALIFIER:** `feature_puck_carrier_holds_or_passes` is FALSE

The four boolean values are stored on each `shot_timestamps` entry —
Gemini commits to feature observations *before* the classification
decision, which is the v10 anti-gaming pattern. The script also
enforces the truth table server-side: any entry that violates it is
pruned, and counts are re-derived from the pruned list.

## Pilot test plan

### Step 1 — Single-video pilot (cheap test)

Run v13 on **Fjc9hmK8_3U** only. This is the same video v12 was
piloted against, so the deltas are directly comparable.

```bash
# Clear v11.2 metrics_seg output for the pilot video
rm -f data/output/runs/metrics_seg/gt_metrics_Fjc9hmK8_3U*

# Run v13 (PROMPT_VERSION is already bumped in the script)
python run_pipeline.py --customer_id CUST000048 \
    --vID Fjc9hmK8_3U \
    --steps 2 --local-output-dir data/output/runs

# Eval on this single video
python eval/eval_metric_seg_output.py \
    --vIDs Fjc9hmK8_3U \
    --customer-id CUST000048
```

### Expected pilot outcomes vs v11.2

**v11.2 baseline on Fjc9hmK8_3U:**
- Shot-ts predictions: 156
- Shot-ts TP: ~45-50 (within ±3s tolerance)
- Within-coverage recall: ~0.79
- Precision: ~0.30

**v13 prediction (honest estimates):**

If the truth table works as designed:
- Shot-ts predictions: ~60-80 (down from 156 — the over-counting collapses)
- Shot-ts TP: ~45-50 (recall holds because real shots satisfy the truth table)
- Within-coverage recall: 0.74-0.84 (small possible drop from strict gating)
- Precision: 0.45-0.65 (the upside)

If the truth table over-prunes (features get gamed under framing
pressure, an outcome the probe didn't measure):
- Shot-ts predictions: ~30-50
- Within-coverage recall: 0.55-0.65 (meaningful regression)
- Precision: 0.50+

**Decision rule:** If within-coverage recall on Fjc9hmK8_3U falls
below 0.65, the truth table is over-pruning. Stop and run a
phase-2 follow-up probe before shipping further. If recall holds
above 0.70 with precision > 0.45, proceed to the full 14-video eval.

### Step 2 — Full 14-video eval (only after pilot passes)

```bash
# Clear all v11.2 outputs
rm -f data/output/runs/metrics_seg/gt_metrics_*

# Run all 14 videos
python run_pipeline.py --customer_id CUST000048 CUST000031 \
    --vID <all 14 vIDs> \
    --steps 2 --local-output-dir data/output/runs

# Full eval
python eval/eval_metric_seg_output.py --customer-id CUST000048 CUST000031
```

## Files in this bundle

```
metrics_seg/
├── prompts/
│   └── metrics_v13.txt              # new — evidence-based truth table
├── 01_detect_segment_metrics.py     # updated — PROMPT_VERSION=v13, schema
│                                    #   adds 4 boolean fields, server-side
│                                    #   truth-table enforcement
└── CHANGELOG.md                     # unchanged (v13 entry goes in after
                                    #   pilot results)
```

## Rollback path

To restore v11.2:

```bash
# In metrics_seg/01_detect_segment_metrics.py:
#   change PROMPT_VERSION = "v13" back to PROMPT_VERSION = "v11"
# The v13 schema additions are backward-compatible (the four feature
# fields just won't be populated by v11 responses, and the truth-table
# enforcement skips entries missing all four fields).
```

The v11.2 metrics_seg output is preserved in
`data/output/evals/eval_metrics_20260514T132537.*` for regression
comparison.

## What this DOESN'T fix

v13 targets shot precision. It does NOT address:
- **Goal F1** — unchanged from v11.2 baseline (0.50 STRICT).
- **Coverage recall on the 5 collision videos** — that's the B-fix
  (HockeyAI attribution), shipped separately as `verify_hockeyai.py`.
- **Shot-timestamps recall in uncovered time** — cv_seg-side; no
  metrics_seg change can recover shots from time cv_seg didn't flag.

If you run B (HockeyAI verification) and v13 in parallel, their
effects compose: B raises end-to-end recall by recovering uncovered
shots, v13 raises precision by gating over-counted shots. The combined
goal F1 lift could be 0.50 → 0.60+ if both land within their
expected ranges.

## Honest caveats (from the v10 playbook)

1. **The probe measured discrimination atomically, not under framing
   pressure.** v10's investigation found two features that looked
   strong in phase 1 but got gamed when embedded in classification
   prompts. We did NOT run a phase-2-style framing probe before
   shipping v13. The single-video pilot is the safety net.

2. **Sample size is small** — 17 real-shot moments + 10 noise moments
   for the discrimination measurements. The patterns are consistent
   but the variance bars are real.

3. **The audio feature (`puck_impact_sound_audible`) may not work
   cross-video.** Some feeds have muffled audio. Phase 1 had a 71%
   rate on real shots; on a video with weak audio this could drop
   substantially and hurt recall.

4. **My recent track record on this project:** I was wrong about the
   cv_seg window dedup theory; I had to course-correct. I think the
   probe data is solid, but trust the pilot result over the
   prediction.
