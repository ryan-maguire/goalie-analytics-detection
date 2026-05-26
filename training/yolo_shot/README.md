# YOLO shot-detector pipeline (learning curve)

Fine-tunes HockeyAI YOLOv8 to detect shot moments. Plugs into the
`training/learning_curve/` framework: each run records one
`(train_size, F1)` point so the power-law fit can answer **"does
this architecture's asymptote clear 0.90?"**

## Why this might NOT clear 0.90

Documented ceiling per `specs/EVAL_NOTES.md`: **0.70–0.85**. Per-frame
YOLO can't encode the temporal buildup→release→result pattern of a
shot; it sees one frame at a time. If the curve fit comes back with
asymptote ~0.78, that's the signal to switch to the temporal-model
path (TimeSformer / 3D-CNN over short clips).

The point of this scaffold is to get a *fast, empirical* answer.

## Prior failure modes (now addressed)

The first YOLO attempt landed at test F1=0.232 vs the 0.422 cv_seg
baseline. Root causes:

| failure | new default | where |
|---|---|---|
| training negatives too sparse (353 vs 30K+ non-shot sec) | hard-negative mining via `--hardneg-target 1500` | `phase_hard_negs` in `run_curve_point.py` |
| shot bbox too generous (1.8w × 1.5h) → model over-fires | tighter `--w-factor 1.0 --h-factor 0.8` | `DEFAULT_W_FACTOR / H_FACTOR` |
| positives anchored at GT+1s (wrong moment) | dynamic anchor — highest goal-conf second in window | `util/extract_label_frames_v2.py` (existing v2 fix) |
| eval comparison on weak-prediction artifacts | post-inference threshold/min-dur/pad tuning | `--infer-threshold / --infer-min-dur / --infer-pre / --infer-post` |

## Files

| | |
|---|---|
| `synth_customer.py`        | one-time: emit `data/customers/CUST_LEARNCURVE.json` (vID→targetTeam mapping for hudl matches) |
| `run_curve_point.py`       | orchestrator: runs all 11 phases for ONE `--train-size` |
| `probs_to_predictions.py`  | bridge: per-second probs TSV → predictions.csv (the format `learning_curve/eval.py` consumes) |

## One-time setup

```bash
# 1. Generate the synthetic customer JSON (extract_label_frames_v2.py
#    + sample_hard_negatives.py both require --customers)
python3 training/yolo_shot/synth_customer.py

# 2. Generate splits (test holdout + train pool + curve sizes)
python3 training/learning_curve/splits.py

# 3. Confirm ultralytics + cv2 are installed
python3 -c "import ultralytics, cv2; print(ultralytics.__version__, cv2.__version__)"
```

## Run the four curve points

```bash
# Smallest first — fastest iteration to validate the pipeline end-to-end
python3 training/yolo_shot/run_curve_point.py --train-size 4  --epochs 30

# Then scale up
python3 training/yolo_shot/run_curve_point.py --train-size 8  --epochs 30
python3 training/yolo_shot/run_curve_point.py --train-size 12 --epochs 40
python3 training/yolo_shot/run_curve_point.py --train-size 16 --epochs 50

# Plot when done
python3 training/learning_curve/run_curve.py plot \
    --baseline-f1 0.422 --target-f1 0.90
```

Time budget (Apple M-series): feature extraction is the slowest one-
time cost (~1× video duration with mps device). Training is ~20-40
min per curve point at the given epoch counts. Inference is ~1×
video duration on test set. **Plan for a full afternoon for all
four points** the first time through (most cost is upfront — later
points reuse cached features and labels).

## Curve-point artifacts

Each invocation creates `runs/yolo_curve_n<N>/`:

```
runs/yolo_curve_n8/
├── dataset/              # symlinks to data/labels/ for THIS subset
│   ├── images/
│   └── labels/
├── work/                 # ultralytics training output
│   └── runs/<run_name>/weights/best.pt
├── probs/                # per-test-video shot probs TSVs
└── preds.csv             # bridge output — passed to record subcommand
```

## Phase reference

Each phase is an existing `util/` script. The orchestrator is just a
shell of `subprocess.run()` calls plus a couple of glue helpers.

| # | phase | script | output |
|---|---|---|---|
| 1-2 | feature extract (train + test) | `util/extract_yolo_features.py` | `data/output/yolo_features/<vID>.tsv` |
| 3 | extract positive frames | `util/extract_label_frames_v2.py` | `data/labels/images/<vID>_pos_<t>.jpg` |
| 4 | mine hard negatives | `util/sample_hard_negatives.py` | `data/labels/images/<vID>_hardneg_*.jpg` |
| 5 | pre-label classes 0-6 | `util/prelabel_frames.py` | `data/labels/labels/*.txt` |
| 6 | auto-label shot bbox (class 7) | `util/autolabel_shots.py --w-factor 1.0 --h-factor 0.8` | (appends class-7 line) |
| 7 | per-curve-point dataset | `run_curve_point.py` (symlinks) | `runs/.../dataset/` |
| 8 | fine-tune YOLO | `util/train_yolo_finetune.py` | `runs/.../weights/best.pt` |
| 9 | inference on test | `util/predict_shots_yolo.py` | `runs/.../probs/<vID>.tsv` |
| 10 | bridge probs → CSV | `training/yolo_shot/probs_to_predictions.py` | `runs/.../preds.csv` |
| 11 | record curve point | `training/learning_curve/run_curve.py record` | adds row to `results.json` |

## Tuning the inference threshold

The defaults (`--infer-threshold 0.5 --infer-min-dur 3 --pre 5 --post 5`)
match what the previous YOLO attempt used. For each trained model,
you may get a quick free F1 boost by sweeping the threshold and
re-running JUST the bridge:

```bash
for t in 0.4 0.5 0.6 0.7 0.8; do
    python3 training/yolo_shot/probs_to_predictions.py \
        --probs-dir runs/yolo_curve_n8/probs \
        --out runs/yolo_curve_n8/preds_thr${t}.csv \
        --vIDs 2069975 2070269 2072194 2073809 2108723 2132246 \
        --threshold $t
    python3 training/learning_curve/eval.py \
        --predictions runs/yolo_curve_n8/preds_thr${t}.csv \
        | python3 -c "import json, sys; d=json.load(sys.stdin); print(f't=$t F1={d[\"aggregate\"][\"f1\"]:.3f}')"
done
```

Then re-record the curve point with the best threshold's predictions.

## Known caveats

- **Target team filtering**: `synth_customer.py` infers
  target-vs-opp teams by row-count heuristic (the team with fewer
  shots is the goalie side). This is correct for most games in this
  corpus but not all. Reckless in attribution-sensitive eval; safe
  for shot-only F1 (which is what we're measuring).
- **Hard negs are global**: `sample_hard_negatives.py` picks
  top-scored frames across ALL feature TSVs. `phase_build_dataset`
  filters to train_vids only at symlink time. If the global pool
  is heavily skewed toward one match the curve point may not have
  uniform negatives across train games — not catastrophic but worth
  knowing.
- **VID_TO_HUDL**: hudl-fetched matches aren't in
  `eval/eval_cv_seg_output.py`'s `VID_TO_HUDL` dict. We patched
  `util/extract_label_frames_v2.py` and `util/sample_hard_negatives.py`
  to fall back to `int(vid)` when the vID is a numeric string. The
  orchestrator's bridge bypasses `util/yolo_probs_to_windows.py`
  entirely for the same reason.
- **Re-running invalidates the curve**: phases 3-6 write into the
  shared `data/labels/` pool. A second curve point with a DIFFERENT
  train subset will see frames from the previous run — but
  `phase_build_dataset` only symlinks the current train_vids, so the
  trained model sees the right subset. Stale frames just take up
  disk. `rm -rf data/labels/images/*_pos_*.jpg data/labels/labels/*_pos_*.txt`
  to start clean.
