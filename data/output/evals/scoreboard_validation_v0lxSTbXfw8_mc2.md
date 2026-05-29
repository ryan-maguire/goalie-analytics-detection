# Scoreboard goal-detection validation

- vID:          `v0lxSTbXfw8`  (hudl_id=2073810)
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
| min  | +13s |
| p25  | +13s |
| median | +26s |
| p75  | +39s |
| max  | +39s |
| mean | +26.0s |

## TP matches

| GT t | GT team | detected t | offset | side | score |
|---|---|---|---|---|---|
| 3222s | North Shore Warhawks 19U AA | 3235s | +13s | home | 2→3 |
| 3550s | North Shore Warhawks 19U AA | 3589s | +39s | home | 3→4 |

## FN (GT goal not detected)

| GT t | GT team |
|---|---|
| 367s | Coeur d'Alene Lady Thunder 19U AA |
| 876s | North Shore Warhawks 19U AA |
| 2364s | North Shore Warhawks 19U AA |