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

### Untried (worth attempting next session)

- `MIN_CONFIRMATION_OVERLAP_SEC` + `CONFIRMATION_EVENT_WIDTH_SEC`
  8→10 lockstep. The proven winning lever, not yet exhausted.
- Per-video motion z-score normalization (replace fixed
  `MOTION_THRESH=3.0` with `mean + N*std` per video). The session's
  most important finding suggests this is the highest-value
  unexplored direction within the existing motion architecture.

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
