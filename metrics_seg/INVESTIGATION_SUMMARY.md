# Goalie analytics pipeline — investigation summary (May 2026)

This memo summarizes the multi-session investigation that took the
end-to-end goalie analytics pipeline from "reports zero goals" to
"reports ~64% of goals at 62-87% precision." It is intended as a
record for future-me and as context for anyone picking up this work.

## TL;DR

| Component | Before | After | Locked version |
|---|---|---|---|
| cv_seg attribution accuracy | 0.74 | **0.87** | v23.7 |
| metric goal classifier F1 (STRICT) | 0.00 | **0.625** | v10 |
| metric goal classifier F1 (UNFILTERED) | 0.00 | **0.737** | v10 |
| End-to-end goal recall | 0/14 | **10/16** STRICT, 14/22 UNFILTERED | — |

## The journey, in three acts

### Act 1: cv_seg attribution (v23.7-net)

We knew cv_seg's attribution accuracy was 0.74 and limiting downstream
work. Several theories about why were tested and rejected:
- Camera-pan inversion (wrong)
- Broadcast-quality assumptions (wrong)
- Classical CV approaches via hand-tuned net detectors (didn't work)
- Threshold sweep on motion params (no improvement)

What worked: the SimulaMet-HOST/HockeyAI YOLOv8 model (52MB, free, on
Hugging Face) transfers well enough to amateur arena footage. After a
diagnostic that revealed:
- 60% of frames have `goal` detection (vs predicted ~30%)
- 99% side-agreement when goal+goalie co-occur in a frame (4 of 5 videos)

We integrated it into cv_seg as the primary attribution signal with
motion as fallback. Per-video runtime grew by ~50s. Attribution
accuracy moved from 0.74 → 0.87.

**Shipped:** `cv_seg_pkg_v23_7.zip`, 64/64 tests pass, default-on.

### Act 2: GT data integrity discovery

Mid-investigation we discovered the Hudl ground truth file we'd been
debugging against had 2 goals, but the actual game had 7 goals (Amherst
6, NSW 1). The original GT was incomplete. Two specific goal events
we'd built elaborate camera-pan-inversion theories about (t=389,
t=2923) didn't even exist in the actual game.

**Lesson:** GT integrity is the foundation everything else rests on.
If the eval system can spot data drift between GT versions
automatically, that'd save future-you from this kind of footgun. A
simple check: log `hudl_total_goals` per video and warn if it differs
from a previous run's count by >1. Worth adding as future work.

### Act 3: metric prompt iteration (v6 → v10)

After cv_seg v23.7 landed, the metric model still reported 0/7 goals
on mjEeE7p2Hz8. The bottleneck moved from cv_seg attribution to the
metric model itself. Five prompt iterations followed:

| | Recall | Precision | F1 | Notes |
|---|---|---|---|---|
| v6 | 0.000 | n/a | 0.000 | Original baseline |
| v7 | 0.353 | 0.353 | 0.353 | Added Paths 5/6/7 — FPs spiked |
| v8 | 0.571 | 0.333 | 0.421 | Negative-anchor guard — failed |
| v9 | 0.071 | 0.500 | 0.125 | Path 4 only — too conservative |
| v10 | 0.625 | 0.625 | 0.625 | Evidence-based redesign |

The breakthrough between v9 and v10 was a **two-phase discrimination
probe** that asked Gemini atomic observation questions about 22 known
goals + 12 known FPs + 10 random negatives. The probe revealed:

1. **Celebration + ref point fire on 75-95% of FPs.** The signals every
   prior version was anchored on don't actually discriminate goals
   from FPs in this footage.

2. **`puck_retrieved_from_net` and `centre_ice_faceoff_visible`
   inflate dramatically under goal-detection framing.** Atomic FP
   rate ~30% balloons to ~75% when the prompt asks the model to
   confirm a goal. The model games these features. v8's negative-
   anchor guards were defeated by the same pattern: model self-reports
   "no save observed" while describing what is clearly a save plus
   rebound.

3. **Three features held stable across framings:**
   - `scoreboard_change` (visible scoreboard digit change)
   - `attacking_team_skates_to_bench` (post-goal fist-bump line)
   - `crowd_cheer_sustained` (sustained >3s crowd cheer)

v10's truth table is just two paths from these stable features:
- Path B: `scoreboard_change` AND `ref_points_at_net`
- Path C: `attacking_team_skates_to_bench` AND `crowd_cheer_sustained`

In-sample probe predicted F1 0.75. Real-world F1 came in at 0.625
STRICT / 0.737 UNFILTERED. The probe's prediction was directionally
correct but slightly optimistic — expected.

**Shipped:** `metric_prompt_v10.zip`, 26/26 tests pass.

## Lessons that generalize

**1. Theory-based prompt design loses to evidence-based prompt design
   on this kind of task.** v6-v9 were all built on "what should
   logically signal a goal" (celebration, ref point, faceoff, puck-
   in-net). Each iteration's adjustments to those features failed
   similarly because the underlying anchors don't discriminate. v10's
   anchors came from measuring what Gemini actually reports
   reliably across framings.

**2. LLMs game whatever path you give them.** When v6-v9 added
   negative anchors or strict definitions to combat FPs, Gemini just
   self-reported satisfaction with the new requirements. The model
   isn't evading deliberately — it's pattern-matching on the
   path-confirmation framing and producing language that satisfies
   it. The fix is to anchor on features that are hard to fabricate
   (visual state changes like `scoreboard_change`), not to add more
   verification rules to features the model can rationalize.

**3. The discrimination probe pattern is reusable.** For any prompt
   where the model is asked to classify clips into categories, building
   a small clip set with known labels (kept hidden from the model) and
   asking atomic observation questions across the set surfaces which
   features actually discriminate. The two-phase variant (atomic vs
   in-context) catches features that look discriminative in isolation
   but get gamed in classification context.

**4. Don't trust precision/recall numbers from a 5-video sample.**
   The probe's in-sample F1 0.75 vs real F1 0.625 is one data point;
   variance is high at this sample size. Future per-video
   calibration may show v10's numbers shifting either direction
   meaningfully.

## Things deliberately left undone

- **Probe round 3 on the 6 v10 FNs to find a Path D candidate.** We
  could surface another high-discriminator feature, but marginal gains
  are uncertain.
- **cv_seg attribution improvements to close the STRICT/UNFILTERED
  gap.** v10's UNFILTERED F1 of 0.737 represents what the metric
  model could achieve if cv_seg attribution were perfect. ~12 percentage
  points of recall on the table.
- **Multi-call voting for v10 paths.** The script supports it; we
  haven't tuned it for v10's path structure.
- **Per-video / per-arena calibration.** Different arenas have
  different camera angles. v10 may underperform on cameras where the
  scoreboard isn't in frame.
- **GT version-drift detection.** A simple check that would have caught
  the 2-goal-vs-7-goal GT error earlier in the investigation.

## File inventory

cv_seg side:
- `cv_seg_pkg_v23_7.zip` — production cv_seg with HockeyAI net detection
- `detect_hockeyai.py` — diagnostic for net detection on new arenas

metrics_seg side:
- `metric_prompt_v10.zip` — production prompt + script + changelog
- `discrim_probe.py` — atomic-observation probe (label-free)
- `discrim_probe_v2.py` — goal-context probe (with truth table)

Investigation artifacts:
- `data/output/discrim_probe/per_clip.json` — raw probe 1 responses
- `data/output/discrim_probe/per_clip_v2.json` — raw probe 2 responses
- `data/output/discrim_probe/feature_analysis.tsv` — per-feature
  discrimination scores
- `data/output/discrim_probe/probe_delta.tsv` — framing-effect analysis

Eval results from each version are in `data/output/evals/eval_metrics_*`.
