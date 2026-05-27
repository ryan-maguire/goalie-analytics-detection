# Scoreboard goal-detection validation

- vID:          `dwGsP6QKDs8`  (hudl_id=2070269)
- GT goals:     5
- Detected:     2

## Confusion

| metric | count |
|---|---|
| TP (detected ∩ GT) | 2 |
| FP (detected, no GT) | 0 |
| FN (GT, not detected) | 3 |
| **Precision** | **1.000** |
| **Recall**    | **0.400**  |
| **F1**        | **0.571**  |

## Detection offset (detected_t_sec - GT_goal_start)

| stat | value |
|---|---|
| min  | +10s |
| p25  | +10s |
| median | +71s |
| p75  | +132s |
| max  | +132s |
| mean | +71.0s |

## TP matches

| GT t | GT team | detected t | offset | side | score |
|---|---|---|---|---|---|
| 2617s | Nashville Preditors 19U | 2627s | +10s | away | 2→3 |
| 3553s | Nashville Preditors 19U | 3685s | +132s | away | 3→4 |

## FN (GT goal not detected)

| GT t | GT team |
|---|---|
| 688s | Nashville Preditors 19U |
| 1713s | Nashville Preditors 19U |
| 4109s | Northshore Warhawks 19U |