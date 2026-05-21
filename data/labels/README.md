# HockeyAI fine-tuning labels

Hand-labeled bbox annotations for fine-tuning the HockeyAI YOLOv8 model
on a new `shot` class. Goal: push F1 above the current ~0.42 motion-
based ceiling toward 0.70+ on the 9-video outer set.

## File layout

```
data/labels/
  README.md           ← you are here
  classes.txt         ← class names in YOLO id order (written by prelabel_frames.py)
  images/             ← extracted JPGs
    {vID}_pos_{t}.jpg     ← positive frames (around GT shot moments)
    {vID}_neg_{t}.jpg     ← random non-shot frames
    {vID}_hardneg_{t}.jpg ← frames from current cv_seg FPs
  labels/             ← YOLO-format .txt, one per image
    {vID}_pos_{t}.txt     ← pre-labeled by prelabel_frames.py
    ...
  _yolo_workdir/      ← created by train_yolo_finetune.py (train/val split)
```

## Class scheme (8 total)

| id | name      | source     | who labels it |
|----|-----------|------------|---------------|
| 0  | centriod  | HockeyAI   | pre-labeled, you fix if wrong |
| 1  | faceoff   | HockeyAI   | pre-labeled, you fix if wrong |
| 2  | goal      | HockeyAI   | pre-labeled, you fix if wrong |
| 3  | goalie    | HockeyAI   | pre-labeled, you fix if wrong |
| 4  | player    | HockeyAI   | pre-labeled, you fix if wrong |
| 5  | puck      | HockeyAI   | pre-labeled, you fix if wrong |
| 6  | referee   | HockeyAI   | pre-labeled, you fix if wrong |
| 7  | **shot**  | **new**    | **you add manually on positive frames** |

**`shot` class definition:** draw a bbox enclosing the area where the
shot is being taken — typically the goal mouth + immediate slot. Include
the puck if visible, the shooter, and the defending goalie. For a
1-on-1 breakaway, the bbox covers the slot area. For a point shot, the
bbox covers the goal mouth + crease. **Only label `shot` on frames
where a shot attempt is actually happening at that moment** (puck on
stick about to be released, or puck mid-flight toward goal).

Negative and hard-negative frames: do NOT add a `shot` bbox — these
are training the model to *reject* shot-like situations that aren't shots.

## Workflow

### 1. Extract frames (already done if you see images/)

```bash
python3 util/extract_label_frames.py \
    --customers data/customers/CUST000048.json data/customers/CUST000031.json
```

### 2. Pre-label with existing HockeyAI

```bash
python3 util/prelabel_frames.py
```

This populates `labels/` with .txt files containing detections for the
existing 7 classes. Your job is to add `shot` bboxes on positives and
fix any wrong existing labels.

### 3. Hand-label using a YOLO-aware tool

**Recommended: [labelImg](https://github.com/heartexlabs/labelImg)** — pip
installable, runs locally, supports YOLO format natively.

```bash
pip install labelImg
labelImg data/labels/images data/labels/classes.txt data/labels/labels
```

Then in labelImg:
- View → Auto Save Mode → on
- View → Display Labels → on (so you see pre-labeled boxes)
- Toggle "YOLO" output format (bottom-left of toolbar) if not already
- Use `w` to draw a new bbox; pick class `shot` (index 7) for shot moments

Suggested pace: spend ≤30 seconds per frame. If a shot moment isn't
clear, skip the frame (delete the .jpg and the .txt — they won't
contribute to training).

**Alternative: [AnyLabeling](https://github.com/vietanhdev/anylabeling)**
— newer, model-assisted (can use SAM for box suggestions). Heavier
install but faster labeling.

### 4. Quick label sanity check

```bash
# Count how many positives have a shot bbox (class 7)
grep -l "^7 " data/labels/labels/*_pos_*.txt | wc -l
# Should be roughly equal to (positives_total).
# If much less, labeling is incomplete.

# Count negatives that accidentally have a shot bbox (should be 0)
grep -l "^7 " data/labels/labels/*_neg_*.txt | wc -l
grep -l "^7 " data/labels/labels/*_hardneg_*.txt | wc -l
```

### 5. Train

```bash
python3 util/train_yolo_finetune.py --epochs 50 --batch 8 --device mps
```

On Apple Silicon `--device mps` uses the GPU; ~30-60 min for 50 epochs
on ~1000 frames. The script splits 80/20 train/val deterministically by
filename hash so you can re-run with different epoch counts without
shuffling your validation set.

Best weights land at `data/labels/_yolo_workdir/runs/hockeyai_shot_finetune/weights/best.pt`.

### 6. Evaluate

After training, integrate the fine-tuned model by either:
1. Pointing `cv_seg/net_detection.py` at the new weights (replace
   `NET_MODEL_REPO_ID` / `NET_MODEL_FILENAME` with the local path), and
   adding a new candidate-window source for the `shot` class
2. Building a parallel `cv_seg/shot_detection.py` that runs the new
   model independently of net_detection.py

Re-run `zsh ./run_fast_set.sh` and the 9-video eval to measure F1.

## Realistic expectations

- AUC target ≥ 0.80 on per-frame `shot` class (vs 0.58 for approach 1
  without bbox labels). With ~580 positives and ~390 negatives, a well-
  tuned YOLO fine-tune should reach this.
- Translating to window-level F1: aim for 0.55-0.70. Anything above 0.70
  on the outer-9 outer-check would be excellent.
- If first training run yields AUC < 0.70 on val:
  - Check label quality (random sample ~20 frames, see if `shot`
    bboxes are sensible)
  - Try more epochs (100, 200)
  - Try `--imgsz 1024` for higher input resolution (slower)
  - Consider adding goalie-reaction class (see EVAL_NOTES.md future-work)
