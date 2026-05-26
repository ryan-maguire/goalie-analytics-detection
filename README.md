# goalie-analytics-detection

Shot-moment detection from hockey broadcast video.

The production tool **surfaces a ranked list of candidate shot
timestamps** per game so a coach can review them quickly. Validated
recall on a held-out 6-game test set:

| metric | result |
|---|---|
| **Recall @ ±10 seconds** | **86.2%** |
| Recall @ ±5 seconds | 68.6% |
| Avg candidates per minute | 2.3 |

This is the reframing from "per-second binary shot/no-shot F1" to
"surface candidates within tolerance" — the metric coaching tools
actually use. The earlier F1=0.57 IoU number understated how useful
the model is for review-based workflows.

---

## Quick start — review one game

Already-trained models + cached predictions ship in this repo. For any
game in the test set:

```bash
python3 tools/candidate_list.py --vID 2073809 --out-dir candidate_output/
```

Output:
- `candidate_output/2073809_candidates.csv` — machine-readable ranked list
- `candidate_output/2073809_candidates.md`  — coach-facing markdown

Each row has `rank`, `t_mmss` (timestamp), `confidence`, and
(if GT is present) a `✓ / ✗` marker showing whether it matches a
real shot within ±5 seconds.

## Full validation report

```bash
python3 tools/validate_candidates.py --out-dir candidate_output/
```

Produces `candidate_output/VALIDATION_REPORT.md` with per-game stats,
sample top-10 candidates per game, and a global error analysis.

## Architecture

```
video.mp4 ──┬─→ extract_yolo_features.py  ─→ yolo_features/<vID>.tsv ─┐
            │                                                          ├─→ YOLO trained model ─→ probs/<vID>.tsv ┐
            │                                                          │                                          │   late fusion
            └─→ extract_audio_features.py ─→ audio_features/<vID>.tsv ─→ audio trained model ─→ probs/<vID>.tsv ┤   50/50
                                                                                                                  │   weighted avg
                                                                                                       candidate_list.py
                                                                                                                  │
                                                                                                                  ↓
                                                                                                  ranked CSV + coach MD
```

Late fusion of two cheap-to-train models beats either alone:
- **YOLO** (per-frame HockeyAI fine-tuned): visual signal — puck/goal/goalie/player locations
- **Audio** (mel-spec + GRU): puck strikes, whistles, crowd reactions
- **Fuse** at the per-second probability level (50/50)
- **Pick peaks** with smoothing + NMS, rank by confidence

Why this works: the two signals fail differently. YOLO misses when
the camera cuts away from the play; audio misses when the broadcast
mutes or the crowd is dead. Together they cover more.

## Layout

```
.
├── tools/                    # SHIP-IT entrypoints
│   ├── candidate_list.py     # produce ranked candidates for one game
│   └── validate_candidates.py# run across test set, write report
│
├── runs/
│   ├── yolo_curve_n16/
│   │   ├── work/.../best.pt  # trained YOLO weights
│   │   └── probs/<vID>.tsv   # per-second YOLO predictions (test set)
│   └── audio_curve_n16/
│       ├── work/best.pt      # trained audio weights
│       └── probs/<vID>.tsv   # per-second audio predictions (test set)
│
├── training/                 # KEPT for retraining on more data
│   ├── yolo_shot/            # YOLO training pipeline + orchestrator
│   ├── audio_shot/           # audio training pipeline + orchestrator
│   └── learning_curve/       # splits.json + eval primitives + peak-finding
│
├── util/                     # data-pipeline helpers
│   ├── extract_yolo_features.py    # per-second YOLO TSV from video
│   ├── extract_audio_features.py   # per-second audio TSV from video
│   ├── extract_label_frames_v2.py  # extract positive training frames
│   ├── sample_hard_negatives.py    # mine hard negative training frames
│   ├── prelabel_frames.py          # HockeyAI auto-prelabel
│   ├── autolabel_shots.py          # add shot-bbox to positives
│   ├── train_yolo_finetune.py      # YOLO fine-tune wrapper
│   ├── predict_shots_yolo.py       # YOLO inference → per-second probs
│   ├── detect_hockeyai.py          # HockeyAI detector wrapper
│   ├── yolo_probs_to_windows.py    # cv_seg-format JSON writer (legacy)
│   └── pyproject.toml
│
├── data/
│   ├── videos/               # 46 .mp4 files (~66 GB)
│   ├── ground_truth/         # 33 gt_<id>.csv files (Hudl InStat exports)
│   ├── customers/            # vID→team mapping JSONs (incl. CUST_LEARNCURVE.json)
│   ├── labels/               # YOLO training data
│   │   ├── images/           # extracted positive + hardneg frames
│   │   ├── labels/           # YOLO-format bbox annotations
│   │   ├── classes.txt
│   │   └── README.md
│   └── output/
│       ├── yolo_features/    # per-second HockeyAI features per video (cached)
│       └── audio_features/   # per-second librosa features per video (cached)
│
├── cv_seg/                   # LEGACY stage 1 — kept for --legacy-cv-seg rollback
├── eval/                     # Eval primitives — DON'T edit
├── metrics_seg/              # Stage 2 — Gemini-enhanced JSON
├── feedback_seg/             # Stage 3 — Gemini-enhanced JSON
├── specs/                    # design docs + EVAL_NOTES history
├── tools/
│   ├── candidate_list.py     # ranked candidates (also reusable as a lib)
│   ├── validate_candidates.py
│   ├── run_fusion_pipeline.py # stage-1 implementation for run_pipeline.py
│   └── validate_fusion_wide.py
├── run_pipeline.py           # 3-stage orchestrator (fusion stage 1 by default)
└── run_fast_set.sh
```

## Adding a new game

Once you've fetched `data/videos/full_<vID>.mp4` and `data/ground_truth/gt_<vID>.csv`:

```bash
# 1. Per-second features (cached; one-time per video)
python3 util/extract_yolo_features.py  --video data/videos/<vID>.mp4 --out data/output/yolo_features/<vID>.tsv  --fps 1.0
python3 util/extract_audio_features.py --video data/videos/<vID>.mp4 --out data/output/audio_features/<vID>.tsv

# 2. Run inference with the existing trained models
python3 util/predict_shots_yolo.py \
    --weights runs/yolo_curve_n16/work/runs/hockeyai_shot_n16/weights/best.pt \
    --vIDs <vID> --out-dir runs/yolo_curve_n16/probs --fps 1.0
python3 training/audio_shot/infer.py \
    --weights runs/audio_curve_n16/work/best.pt \
    --vIDs <vID> --out-dir runs/audio_curve_n16/probs

# 3. Produce candidates
python3 tools/candidate_list.py --vID <vID> --out-dir candidate_output/
```

## Retraining (if you add ≥10 more paired games)

```bash
# Regenerate customer JSON for the new pairs
python3 training/yolo_shot/synth_customer.py

# Regenerate splits.json (test_match_ids stay fixed by default; only train_pool grows)
python3 training/learning_curve/splits.py

# Retrain at a new train_size (uses existing util/ scripts under the hood)
python3 training/yolo_shot/run_curve_point.py   --train-size 22 --epochs 30
python3 training/audio_shot/run_curve_point.py  --train-size 22 --epochs 50

# Then re-run inference + candidate_list as above with the new best.pt files
```

## Full 3-stage pipeline (fusion-wide stage 1 → metrics_seg → feedback_seg)

`run_pipeline.py` chains stage-1 detection → Gemini metrics_seg →
Gemini feedback_seg. **As of the fusion-wide cutover** (validated
2026-05-22), stage 1 is the candidate-list pipeline above wrapped as
a cv_seg-compatible seg JSON writer, replacing the legacy motion-based
`cv_seg`. cv_seg remains in-tree behind a rollback flag.

```bash
# Default: fusion stage 1
python3 run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8

# Rollback to the original cv_seg motion-window stage 1
python3 run_pipeline.py --customer_id CUST000048 --vID mjEeE7p2Hz8 --legacy-cv-seg
```

Validated lift of fusion-wide over cv_seg on 3 paired games
(mjEeE7p2Hz8, dwGsP6QKDs8, J8WkcuTsD5I), aggregate means:

| metric                 | v13 (cv_seg) | fusion-wide | Δ        |
|------------------------|--------------|-------------|----------|
| Goal F1 (strict)       | 0.645        | **0.750**   | +0.106 ✅ |
| Goal precision         | 0.867        | **1.000**   | +0.133 ✅ |
| Goal recall            | 0.560        | **0.607**   | +0.047 ✅ |
| Shot end-to-end F1     | 0.371        | **0.430**   | +0.059 ✅ |
| Shot MAE (sec)         | 1.146        | **1.070**   | −0.075 ✅ |
| Within-cov recall      | 0.883        | 0.800       | −0.083 ❌ |

Full per-video tables in `data/output/evals/fusion_wide_validation.md`.
The one regression — within-cov recall — is a precision/recall trade
that lands on the right side for production (fewer false-positive
shots, no false-positive goals across 3 games).

**Prerequisite for fusion stage 1:** the per-second YOLO + audio probs
TSVs must already exist for the vID at:

- `runs/yolo_curve_n16/probs/{vID}.tsv`
- `runs/audio_curve_n16/probs/{vID}.tsv`

Use the inference steps in "Adding a new game" above to generate them
if missing, or pass `--legacy-cv-seg` to skip the requirement.

## What's NOT in this repo (yet)

- **The video → Hudl fetcher** lives in the sibling repo
  `../hudl-fetch/` (separate Playwright-based tool for pulling new
  games + GT CSVs).

## Honest limits

- Validated on 6 held-out games. Recall@10s = 86%, R@5s = 69%.
- One game (2132246) is consistently weak across every architecture
  we tried — possibly a camera-angle / lighting / labeling artifact.
  Worth investigating manually before adding more of that team's games.
- 2.3 candidates/min means a 60-min game has ~140 candidates — about
  10-20 min of coach review per game.
- This is NOT the path to F1 > 0.90 on per-second binary classification.
  See `specs/EVAL_NOTES.md` for the full architecture exploration and
  why that target is structurally hard on broadcast hockey video.
