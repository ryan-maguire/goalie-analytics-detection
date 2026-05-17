# Investigation tools — A & B

Two diagnostics for the v11.2 production lock's open follow-ups:

- **A:** `discrim_probe_shots.py` — discrimination probe for shot moments
- **B:** `diagnose_low_coverage.py` — diagnose why 5 specific videos miss ~half their shots

Both ship together because they target different bottlenecks:
- A targets **precision** (Gemini over-counts shots by ~2× per window)
- B targets **recall** (cv_seg covers only 33-68% of shots on 5 low-coverage videos)

---

## A — Shot discrimination probe

### What it does

Mirrors the discrimination-probe pattern that drove v10's goal-detection breakthrough (F1 0.42 → 0.625). For shots specifically:

1. Builds a clip set: ~20 real-shot clips (isolated Hudl Shots events) + ~10 ghost-shot clips (windows where v11 predicted 2+ shots but Hudl had 0).
2. Runs each clip through Gemini twice:
   - **Phase 1 (atomic):** asks per-second observation questions without classification framing
   - **Phase 2 (v11-style framing):** asks Gemini to enumerate shots like the production prompt does
3. For each feature, computes the discrimination score: `fire_rate_on_real_shots − fire_rate_on_zone_pressure_noise`
4. Outputs feature-by-feature discrimination scores, ranked.

### What success looks like

We're looking for 2-3 features with discrimination scores ≥ 0.40 that ALSO don't get inflated under framing pressure. Those become the hard anchors for a v13 shot-detection prompt — replacing v11's "enumerate shots" instruction with "anchor each shot on these specific verifiable features."

Reference benchmark: v10's goal probe found `scoreboard_change` (disc +0.33), `attacking_team_skates_to_bench` (disc +0.34), `crowd_cheer_sustained` (disc +0.22). Those three combined into v10's truth table.

### How to run

```bash
python util/discrim_probe_shots.py \
    --hudl-id-map "mjEeE7p2Hz8:2073809,dwGsP6QKDs8:2070269,bfEKgtOIkQU:2072195,Fjc9hmK8_3U:2070260,J8WkcuTsD5I:2072194" \
    --gt-dir data/ground_truth \
    --video-dir data/videos \
    --metrics-dir data/output/runs/metrics_seg \
    --output-dir data/output/discrim_probe_shots
```

Use 5+ videos in the id-map; the script will pick isolated real-shot moments from across them.

Cost: ~$0.30 in Gemini calls. Runtime: ~15 minutes wall-clock.

### Output

```
data/output/discrim_probe_shots/
├── clips/                              # extracted 30s clips
├── phase1_raw.json                     # per-clip atomic-feature responses
├── phase2_raw.json                     # per-clip v11-framing responses
├── phase1_probe_results.tsv            # long-format: clip × second × feature × fired
├── phase1_feature_analysis.tsv         # per-feature discrimination summary
└── summary.json                        # everything aggregated
```

The console summary at the end ranks features by discrimination score.

### Interpreting the output

**Strong candidate feature** (disc ≥ 0.40): include in v13 truth table.

**Moderate** (0.25 ≤ disc < 0.40): include conditionally — only if phase 2 framing inflation is small (≤ 0.20).

**Anti-anchor** (disc ≤ −0.20): fires on noise more than real shots. Document as a negative signal in v13 ("if this fires, downgrade shot count").

**Phase 2 inflation > Phase 1 baseline:** Gemini is gaming this feature under framing pressure. v10 found `puck_retrieved_from_net` and `centre_ice_faceoff_visible` did this for goals. Such features are unreliable as path anchors regardless of their raw discrimination score.

---

## B — Low-coverage diagnostic

### What it does

Analyzes cv_seg's existing output (no re-running cv_seg, no Gemini calls) to answer: **why does cv_seg miss ~half the Hudl shots on the 5 low-coverage videos?**

For each video it shows:
- Hudl shots total vs covered vs uncovered
- cv_seg pipeline counts: raw → threat → target → final
- Of the uncovered Hudl shots, how many had strong motion signal anyway

### Key finding (run against the 2026-05-14 baseline)

```
vID              shots  cov%    raw   thr   opp   tgt   strong-motion   long-run
q5yj6sAFQeY         78   35%    107    91    55    36     51/51  (100%)  51/51
HNG0jKYY12g         91   44%     94    92    50    42     51/51  (100%)  51/51
KYtM20r9BuM         67   40%     70    62    27    35     39/40  (98%)   24/40
zOQrPK7IJ24         68   50%     97    96    41    55     34/34  (100%)  34/34
J8WkcuTsD5I         90   47%     82    67    27    40     48/48  (100%)  38/48
```

**100% of uncovered shots had strong motion (peak ≥4.0) — well above MOTION_THRESH=3.0.** cv_seg saw the action. But cv_seg attributed roughly half of the threat windows (`opp` column) to the OPPONENT color, and `target_filter` dropped them.

### What to do about it

The bottleneck is **attribution accuracy on color-collision videos**, not threshold tuning. Concrete next steps in order of likely impact:

1. **Verify HockeyAI YOLOv8 is actually firing on these videos.** The investigation summary notes attribution went from 0.74 → 0.87 using HockeyAI, but the cv_seg meta files for these 5 videos don't record which attribution method was used. Run `find ~/.cache/huggingface -name '*.pt' -path '*HockeyAI*'` to see if the model is downloaded; check that `ultralytics` is installed.

2. **If HockeyAI is NOT firing**, install it and re-run cv_seg for these 5 videos. Expected: attribution drop rate (`opp` / `thr`) goes from ~55% to ~15%, recovering ~70-80% of the uncovered shots.

3. **If HockeyAI IS firing**, the per-frame confidence threshold may need tightening for collision videos specifically. The validation note in `net_detection.py` mentions video `n2cy8b755Tg` having "wire mesh that produces spurious goal detections" — similar artifacts may be hurting attribution on the 5 collision videos.

### How to run

```bash
python util/diagnose_low_coverage.py \
    --vIDs q5yj6sAFQeY HNG0jKYY12g KYtM20r9BuM zOQrPK7IJ24 J8WkcuTsD5I \
    --hudl-id-map "q5yj6sAFQeY:2127052,HNG0jKYY12g:2095275,KYtM20r9BuM:2072196,zOQrPK7IJ24:2127035,J8WkcuTsD5I:2072194" \
    --cv-seg-dir data/output/runs/cv_seg \
    --gt-dir data/ground_truth \
    --output-dir data/output/diagnostics
```

Cost: $0. Runtime: <30 seconds.

### Output

```
data/output/diagnostics/
└── low_coverage_diagnostic.tsv         # one row per video, all columns above
```

Plus a console summary with interpretation rules.

---

## Recommended sequencing

The two are independent — run them in parallel if convenient.

**B first** is the cheaper experiment. If the diagnostic confirms attribution is the bottleneck (it does, on the 2026-05-14 baseline), and HockeyAI isn't firing, a single re-run could potentially recover 100+ shots across the 5 videos. That alone would meaningfully raise end-to-end goal recall.

**A in parallel** is the harder experiment. The probe takes ~15 min + $0.30 in API cost. Best case (a feature with disc ≥ 0.40 emerges), it sets up v13 prompt redesign. Worst case (no strong discriminators), we've proven the precision ceiling is fundamental and can stop trying to fix it via prompt iteration.

Whichever fires first, send the output back and I'll do the readout.
