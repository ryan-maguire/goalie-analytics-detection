# Scoreboard goal-detection validation

- vID:          `dwGsP6QKDs8`  (hudl_id=2070269)
- GT goals:     5
- Detected:     4

## Confusion

| metric | count |
|---|---|
| TP (detected ∩ GT) | 3 |
| FP (detected, no GT) | 1 |
| FN (GT, not detected) | 2 |
| **Precision** | **0.750** |
| **Recall**    | **0.600**  |
| **F1**        | **0.667**  |

## Detection offset (detected_t_sec - GT_goal_start)

| stat | value |
|---|---|
| min  | +10s |
| p25  | +10s |
| median | +132s |
| p75  | +164s |
| max  | +164s |
| mean | +102.0s |

## TP matches

| GT t | GT team | detected t | offset | side | score |
|---|---|---|---|---|---|
| 1713s | Nashville Preditors 19U | 1877s | +164s | away | 0→1 |
| 2617s | Nashville Preditors 19U | 2627s | +10s | away | 2→3 |
| 3553s | Nashville Preditors 19U | 3685s | +132s | away | 3→4 |

## FP (detected but no GT goal in lookback window)

| detected t | side | score | lookback |
|---|---|---|---|
| 1877s | away | 1→2 | [1697-1872]s |

## FN (GT goal not detected)

| GT t | GT team |
|---|---|
| 688s | Nashville Preditors 19U |
| 4109s | Northshore Warhawks 19U |