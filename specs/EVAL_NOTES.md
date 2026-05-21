# cv_seg evaluation notes

Breadcrumbs for tuning decisions that are non-obvious from the code.
Whenever you flip a constant or enable/disable a pipeline step, add a
line here so the next maintainer doesn't accidentally re-tread the same
ground.

## v25 (current) — fast-set F1 0.422

### Constant tuning session (2026-05-19/20) — 6 experiments, 1 keeper

Fast set: bfEKgtOIkQU, dwGsP6QKDs8, krxhPVLGLz8.

| # | Change | F1 | Action |
|---|---|---|---|
| 1 | `MIN_CONFIRMATION_OVERLAP_SEC` + `CONFIRMATION_EVENT_WIDTH_SEC` 6→8 lockstep | 0.372 → **0.422** | **kept** |
| 2 | `MIN_THREAT_DUR` 15→20 | 0.428 | revert (+0.006 < 0.01 noise) |
| 3 | `MAX_OPEN_WINDOW_SEC` 30→25 | 0.388 | revert (cap fragmented MAC, preds 197→238) |
| 4 | Require 2 distinct confirmer types (structural, windows.py) | 0.103 | revert — recall collapse, equivalent to v23.5 |
| 5 | activity_spike confirms only when motion ≥ MOTION_THRESH | 0.422 | revert — no-op (gate filtered nothing) |
| 6 | `MIN_MAC_PEAK_MOTION` 5.0 then 7.0 (gate MAC by peak motion) | 0.422 | revert — gate trivially satisfied at every value |

**Settled lessons:**

- The confirmer-rule design space is empirically very thin. v23.6's
  rule sits at a tight optimum: any tightening either does nothing
  (#5) or collapses recall (#4). The lockstep widening (#1) is the
  only working lever in this space; 4→6 won, 6→8 won, **8→10 not
  tried this session** — worth one more attempt.
- The 30s MAC cap is at a structural sweet spot. Cutting it
  fragments long motion runs into multiple windows (#3 went from
  197 to 238 predictions); raising it brings back the v23.3
  problem. The previous 45→30 win wasn't about "shorter is
  better"; it landed at a sweet spot.
- `MIN_THREAT_DUR` 15→20 is functionally too weak a lever: the
  underlying motion-only FP cluster is only ~8 FPs in the fast
  set after the v25 lockstep win. Future tuning of this constant
  has very limited ceiling.
- **Motion baseline is far above MOTION_THRESH**. Per-video signal
  medians on the fast set are 6.43 / 10.66 / 12.42 against
  `MOTION_THRESH=3.0`. Minimum observed peak motion in any
  sustained 30s active streak is 7.86 — so any absolute peak
  threshold below ~13 is a no-op. **A fixed absolute motion
  threshold cannot generalize across videos**. The motion signal
  needs per-video z-score normalization for any further
  intensity-based gating to work.

### YOLO scope exploration (2026-05-20) — both paths killed

User asked: can we use the HockeyAI YOLOv8 model
(SimulaMet-HOST/HockeyAI) for shot detection rather than just
attribution, to push F1 > 0.90?

**Approach 1 (puck-near-goal): killed.** `util/diag_puck_detection.py`
on bfEKgtOIkQU 0-300s:

- Puck recall is high — 99% of frames have ≥1 puck at conf≥0.05.
- BUT 98% of frames have **multiple** pucks (model hallucinates
  small objects). At production conf=0.5, still 56% multi-puck.
- The highest-confidence puck per frame is **never** near the goal
  (0 / 176 frames had top-1 puck bbox overlapping goal bbox; 2 /
  176 in a 2x-expanded goal ROI). The genuine puck — when near the
  goal during a shot — gets a LOWER confidence than the spurious
  detections of equipment, logos, scoreboard elements.
- Conclusion: cannot use conf ranking to find "the real puck".

**Approach 2 (goal-anchored motion): killed.**
`util/diag_goal_area_motion.py` on bfEKgtOIkQU 0-300s with 4 GT
shots:

- Goal class is reliable (73% detection at conf≥0.5, median conf
  0.83). Spatial anchor works.
- BUT goal-area motion is only marginally better than whole-frame
  motion at separating shots from non-shots:

```
threshold sweep (fresh-bbox frames, 44 shot / 176 non-shot)
              whole-frame:   goal-area:
threshold     recall/prec    recall/prec
5             0.82 / 0.27    0.84 / 0.24
7             0.61 / 0.29    0.70 / 0.25
10            0.18 / 0.22    0.55 / 0.28
12            0.09 / 0.19    0.43 / 0.29
```

- Best achievable per-second precision is ~0.29 on either signal.
- Root cause: the camera frames the goal during plays. So
  "active near the goal" is essentially the same signal as "the
  camera is watching a play" — which includes scrums, forechecks,
  dump-ins, and zone entries, not just shots.

**Realistic F1 ceilings (estimated):**

| Approach | Realistic F1 | Effort |
|---|---|---|
| Current pipeline (here) | 0.42 | done |
| + per-video motion z-score normalization | ~0.50 | ~1 day |
| + audio improvements (save-sound detection) | ~0.55 | ~2-3 days |
| Custom-trained shot classifier | 0.70-0.85 | 1-3 weeks |
| F1 > 0.90 | not achievable with this YOLO model + this signal set | requires custom training |

### Exp 7 (2026-05-20): Lockstep 8→10 — no-op, lever exhausted

Aggregate F1 byte-identical to 8/8 baseline (0.422). No pass/fail
outcomes changed. The 7% reduction in confirmer-fitting window
(from 77% to 67% of a 30s MAC) doesn't catch any new MAC blobs —
every confirmer that worked at 8s overlap is positioned far enough
from window edges that 10s also works.

History curve: 4→6 huge win, 6→8 substantial win, 8→10 zero.
**This lever is done.** Higher values would just risk breaking
single-event confirmation.

### Exp 8 (2026-05-20): Per-video quantile motion thresholds — wrong architecture

Tried two settings:
- p75/p40: F1=0.058 (recall 0.031, only 4-5 segments per video)
- p60/p40: F1=0.074 (recall 0.041, slightly better but still dead)

Per-video thresholds were computed correctly (e.g., bfEK p60=7.46
vs fixed 3.0). **The failure is architectural, not parametric.**

Root cause: `MOTION_THRESH=3.0` is a **noise floor** (gates "any
motion vs none"), not an intensity gate. `MIN_MOTION_RUN_SEC=8`
was calibrated assuming the threshold is low — when basically
every active second clears it, 8 consecutive seconds is easy.
When the threshold rises to per-video p60, the 8-consecutive-
seconds requirement becomes the bottleneck and almost no real
plays clear it.

**Per-video adaptation needs different framing to work:**
1. As a *secondary* gate after window assembly (a la exp 6's peak-
   motion gate, but per-video adaptive — e.g., MAC requires at
   least one second above per-video p90). Doesn't touch the noise
   floor or the sustained-time requirement.
2. As a coupled change: per-video threshold + per-video
   MIN_MOTION_RUN_SEC. Each video's MIN_MOTION_RUN_SEC needs to
   scale inversely with how often motion clears the threshold.
   Bigger code change.

### Exp 9 (2026-05-20): Per-video MAC peak-motion gate at p90 — revert

Combined exp 6's shape (secondary gate, doesn't replace noise
floor) with exp 8's per-video adaptation (quantile threshold).
Per-video thresholds correctly computed: bfEK p90=11.45,
dw p90=16.38, krxh p90=17.71.

Result: F1 0.422 → 0.376 (-0.046). Recall dropped 0.639 → 0.505
(-0.134); precision essentially flat. Per-video segments cut by
13-17% but ~13 of those losses were TPs, not FPs.

Root cause: percentile over the full motion distribution
(including stoppages, intermission with ~0 motion) pulls p90 too
high. Real plays with MAC peaks between p75 and p90 of overall
motion get killed alongside the camera pans.

Possible follow-ups (all marginal expected lift):
- p80 or p75 — looser, more recall, less precision; likely
  converges to baseline ±noise.
- Percentile over ACTIVE seconds only (motion >= MOTION_THRESH)
  — pulls threshold down but the continuous-distribution problem
  applies regardless.

**Conclusion: this hypothesis class is structurally weak.** Motion
intensity alone — fixed or per-video adaptive — doesn't
discriminate real plays from camera-following-the-play. The
underlying problem is that the camera frames the goal during ALL
sustained-motion events, real shots and forechecks alike.

### Critical environment finding (2026-05-20): librosa was missing

The entire v25 tuning session ran with `librosa` not installed, so
both `detect_whistles` and `detect_crowd_roar_spikes` silently
returned empty lists. Every fast-set + outer-check run had:

```
WARNING   librosa not installed — skipping whistle detection
WARNING   librosa not installed — skipping crowd-roar detection
INFO      Hard triggers: N centre faceoffs, 0 whistles
```

**This re-frames exp 1.** The lockstep 6→8 win wasn't about
widening whistles (there were 0 whistles). It was about widening
the `activity_spike` confirmer alone. The in-code comment that
says "single whistle/activity_spike events still confirm because
event-width widens" was effectively "single activity_spike events".

After installing `librosa>=0.10,<1.0`:
- Fast set F1: 0.422 → **0.428** (Δ +0.006, within noise)
- bfEK 0.37→0.38, dw 0.47→0.48, krxh unchanged (0 whistles detected)
- Outer-9 F1: 0.409 → **0.411** (Δ +0.002, also within noise)
- Per-video deltas on outer-9 sum to +0.01 across 9 videos — net wash
- Confirmed: librosa install is essentially a no-op for F1. The
  signal that whistle / crowd_roar were nominally adding wasn't
  actually missing — activity_spike was carrying the load.

### Save-sound diagnostic — no useful signal beyond existing detectors

`util/diag_save_sounds.py` on bfEK with 66 GT shot windows vs 50
random non-shot windows:

| Feature | Shot med | Non-shot med | Ratio | % shots > non-shot median |
|---|---|---|---|---|
| rms (overall loudness) | 0.102 | 0.068 | 1.49 | 94% |
| band_2000-4500 (whistle) | 0.280 | 0.257 | 1.09 | 89% |
| band_0-200 (sub-bass) | 0.064 | 0.088 | 0.72 | 5% |
| other bands | ~1.0 | | | 50-70% |

**RMS is the only strong discriminator (94%), but it just measures
"active play is louder than idle moments" — not a save-specific
signal.** Using RMS as a confirmer would catch the same noisy
forecheck / scrum / crowd-reaction periods that already produce
FPs. The 2-4.5kHz whistle band already discriminates and is
already detected. No distinctive save-sound spectral signature in
the other bands. **Audio path is structurally exhausted.**

### YOLO shot-classifier approach 1 (2026-05-21, autonomous) — INSUFFICIENT

Per-second binary shot classifier using HockeyAI YOLO features
(no fine-tuning, just logistic regression on detection counts +
confidences + spatial geometry). Leave-one-video-out CV across
all 9 outer-set videos.

Pipeline:
1. `util/extract_yolo_features.py` — per-second HockeyAI inference
   → 15-column feature TSV per video (~90 min wall time for 9 videos)
2. `util/train_shot_classifier.py` — LOO CV logistic regression,
   positives = seconds inside opponent-team GT 'Shots' windows
   (matches cv_seg target_filter semantics). Uses eval's
   `_team_names_match()` for fuzzy team-name matching across the
   customer-file / Hudl-CSV boundary.

Result:

| Video | n   | pos% | AUC   | F1@0.5 |
|-------|-----|------|-------|--------|
| Fjc9hmK8_3U  | 4916 | 17.8% | 0.513 | 0.241 |
| J8WkcuTsD5I  | 4398 | 15.5% | 0.619 | 0.304 |
| KYtM20r9BuM  | 4334 | 13.5% | 0.633 | 0.291 |
| SX5xNJlh6eQ  | 3665 | 14.4% | 0.546 | 0.251 |
| bfEKgtOIkQU  | 4161 | 14.4% | 0.580 | 0.262 |
| dwGsP6QKDs8  | 4933 | 18.0% | 0.600 | 0.305 |
| krxhPVLGLz8  | 3550 |  9.5% | 0.563 | 0.184 |
| mjEeE7p2Hz8  | 3343 | 16.4% | 0.635 | 0.319 |
| v0lxSTbXfw8  | 3920 | 14.7% | 0.562 | 0.260 |
| **mean**     |      |       | **0.583** | **0.268** |

**Top feature weights from the final model:**
```
  goal_conf_max     +0.324
  puck_conf_max     +0.246
  n_player          +0.222
  puck_conf_mean    -0.171
  goalie_conf_max   -0.169
  n_referee         -0.101
  n_puck            -0.101   ← consistent with the multi-puck
                              noise finding in diag_puck_detection
                              (more pucks = less likely real)
  n_goal            -0.088
```

**Conclusion:** YOLO features DO carry shot signal (AUC > 0.5 on
every video), but it's weak. Mean AUC 0.583 implies a real but
modest per-second discriminator. Per-window F1 derived from this
would cap well below the current 0.428 motion-based baseline,
since per-second precision of ~0.20 cannot aggregate to per-window
precision much above ~0.40 even with smoothing.

**Approach 1 is insufficient for meaningful F1 improvement over
the current motion-based pipeline.** Stopped per the predefined
autonomous rule (AUC < 0.65 → document and stop, don't iterate).

The path to F1 > 0.55 requires approach 2: fine-tune HockeyAI
YOLOv8 on shot-bbox annotations. Estimated 10-15 hours manual
labeling for 200-500 shot frames, plus 1-2 days of training infra.
Realistic ceiling 0.70-0.85 if puck-detection quality improves
under fine-tuning.

### Untried (worth attempting next session)

- **`bfEKgtOIkQU` per-video FP investigation** — only video where
  exp 1 hurt (-0.05). Worth a dedicated FP-trace deep-dive before
  more general tuning.
- **Approach 2: fine-tune HockeyAI on shot-bbox labels.** See
  conclusion above. Multi-week effort with realistic ceiling
  0.70-0.85.
- **Temporal context features for approach 1.** Per-second feature
  vectors with ±5s rolling means / lagged values. Might push mean
  AUC from 0.58 to 0.65-0.70 — still not enough to integrate, but
  worth knowing before committing to approach 2.

### Outer check — VALIDATED (2026-05-20)

Real 9-video outer check with all videos run under post-exp1
constants (8/8 lockstep):

```
STRICT  P=0.307  R=0.610  F1=0.409
TP=177  FP=399  FN=113
```

Per-video pre→post exp1:

| Video         | Pre-exp1 | Post-exp1 |    Δ    |
|---------------|----------|-----------|---------|
| SX5xNJlh6eQ   |   0.33   |   0.39    |  +0.06  |
| bfEKgtOIkQU   |   0.42   |   0.37    |  -0.05  |
| mjEeE7p2Hz8   |   0.18   |   0.42    |  +0.24  |
| v0lxSTbXfw8   |   0.33   |   0.33    |   0.00  |
| dwGsP6QKDs8   |   0.44   |   0.47    |  +0.03  |
| Fjc9hmK8_3U   |   0.30   |   0.43    |  +0.13  |
| J8WkcuTsD5I   |   0.37   |   0.43    |  +0.06  |
| krxhPVLGLz8   |   0.15   |   0.43    |  +0.28  |
| KYtM20r9BuM   |   0.29   |   0.40    |  +0.11  |

**Verdict on spec rule 7:** PASS. Fast-set gain was +0.050;
outer-only videos gained avg +0.10 each — well over the 50%
representativeness floor. The fast set was actually conservative;
the outer videos benefited more from the lockstep change.

**mjEeE7p2Hz8 mystery solved.** Pre-exp1 it was at F1=0.18
(flagged as outlier). Post-exp1 it's at 0.42. The v23.6→8/8
lockstep especially helps signal-starved videos that previously
survived on weak single-whistle confirmations. This was the
outlier flagged for investigation — exp1 was the answer.

**Only regression:** bfEKgtOIkQU -0.05 (matches fast-set inner-loop
measurement). Single-video edge case — worth a per-video FP-trace
investigation next session, but doesn't change the overall verdict.

Restore script: `util/restore_outer_videos.sh` (uses gcloud
storage cp). The 6 outer-only .mp4 files (~4GB total) live in
data/videos/full_*.mp4 with {vID}.mp4 symlinks.

## v24 — 45→30 MAC cap

### MAX_OPEN_WINDOW_SEC 45 → 30
- 64.8% of fast-set FPs were exactly 45s MAC windows. Shorter cap
  reduces MAC window duration; trimmed-blob IoU with GT improves
  for borderline matches.
- F1 (3-video fast set): 0.281 → 0.366. Net win.
- See in-code comment in cv_seg/constants.py for full rationale.

## v23 — see below

### Cap-split (postprocess.cap_segment_length) — DISABLED
- Tried as "Option 2" alongside Option 1 (split_long_threats only).
- Eval result: F1 = 0.69 (Option 1 only) vs F1 = 0.56 (Option 1 + cap-split).
- Decision: keep cap_segment_length() defined but do NOT call it from
  process_video. Retained for future re-investigation.

### Whistle threshold history — settled at 2.5 z-scores
- 0.55 (original): nearly half the game triggered as "whistle" because
  skate scrapes and puck impacts also have 2-4 kHz energy.
- 1.5 → 2.0: still too noisy.
- 2.5 (current): selects roughly the top ~1% of band-energy spikes,
  which on broadcast hockey corresponds to actual referee whistles.

### Whistle grace window — settled at 12s
- 5s → 8s → 12s. 5s cut off goal-mouth scrambles after a stopped-play
  whistle that turns out to be a save-and-immediate-restart. 12s gives
  enough room for rebounds without re-creating false windows.

### Motion attribution ratio — settled at 1.15
- 1.25 was too strict (only 17% of windows were "motion"-decided; the
  rest fell through to fallback).
- 1.15 admits real OZ-pressure asymmetry while still rejecting noise.

### Side detection tie threshold — promoted to constant
- 0.005 was hard-coded inline. Now lives in constants as
  SIDE_DETECTION_TIE_EPS.

### Attribution fallback — now inherits from previous window
- Old behaviour: ambiguous windows defaulted to goalie_color_a, which
  biased one team in low-motion sequences (notably activity-spike
  windows during stoppages).
- New ladder: motion-edge → inherit-previous → default-A.
  See _attribution_src field on each segment for which branch fired.

## v22

- Latent bug: every segment's side was rewritten using the period-1
  side map regardless of which period the segment belonged to. Fixed
  in v23 by threading period_side_maps through apply_side_assignments
  and selecting per-segment via side_map.side_map_at(seg_start).

## Performance baseline

On a 60-minute 720p broadcast with ffmpeg available:
- Frame extraction: ~3.5 minutes (signals.py)
- Audio load via ffmpeg pipe: ~5 seconds (vs ~12s for WAV roundtrip)
- Audio detectors (whistle + crowd): ~8 seconds combined
- Window assembly + attribution + postprocess: <1 second
- Total: ~4 minutes wall time
