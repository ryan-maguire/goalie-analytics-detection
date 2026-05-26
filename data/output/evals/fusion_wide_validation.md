# fusion-wide validation report

Generated: 2026-05-26 01:37:53 UTC

## Per-video metrics

### mjEeE7p2Hz8

| metric | v13 (cv_seg) | fusion_wide | Δ |
|---|---|---|---|
| Goal F1 (STRICT) | 0.600 | 0.727 | +0.127 ✅ |
| Goal precision | 1.000 | 1.000 | 0.000 |
| Goal recall | 0.429 | 0.571 | +0.142 ✅ |
| Goal TP | 3 | 4 | +1 ✅ |
| Goal FP | 0 | 0 | 0 |
| Goal FN | 4 | 3 | -1 ✅ |
| Shot end-to-end F1 | 0.353 | 0.422 | +0.068 ✅ |
| Shot end-to-end recall | 0.530 | 0.574 | +0.043 ✅ |
| Within-cov F1 | 0.407 | 0.473 | +0.066 ✅ |
| Within-cov recall | 0.875 | 0.812 | -0.062 ❌ |
| Shot MAE | 1.200 | 1.175 | -0.025 ✅ |
| Predicted shots | 129 | 114 | -15 |
| Predicted goals | 3 | 4 | +1 |
| n_windows | 70 | 57 | -13 |

### dwGsP6QKDs8

| metric | v13 (cv_seg) | fusion_wide | Δ |
|---|---|---|---|
| Goal F1 (STRICT) | 0.667 | 0.857 | +0.190 ✅ |
| Goal precision | 0.600 | 1.000 | +0.400 ✅ |
| Goal recall | 0.750 | 0.750 | 0.000 |
| Goal TP | 3 | 3 | 0 |
| Goal FP | 2 | 0 | -2 ✅ |
| Goal FN | 1 | 1 | 0 |
| Shot end-to-end F1 | 0.453 | 0.423 | -0.030 ❌ |
| Shot end-to-end recall | 0.477 | 0.425 | -0.053 ❌ |
| Within-cov F1 | 0.592 | 0.561 | -0.031 ❌ |
| Within-cov recall | 0.946 | 0.842 | -0.104 ❌ |
| Shot MAE | 0.897 | 0.923 | +0.026 ❌ |
| Predicted shots | 121 | 109 | -12 |
| Predicted goals | 5 | 3 | -2 |
| n_windows | 68 | 65 | -3 |

### J8WkcuTsD5I

| metric | v13 (cv_seg) | fusion_wide | Δ |
|---|---|---|---|
| Goal F1 (STRICT) | 0.667 | 0.667 | 0.000 |
| Goal precision | 1.000 | 1.000 | 0.000 |
| Goal recall | 0.500 | 0.500 | 0.000 |
| Goal TP | 1 | 1 | 0 |
| Goal FP | 0 | 0 | 0 |
| Goal FN | 1 | 1 | 0 |
| Shot end-to-end F1 | 0.305 | 0.446 | +0.140 ✅ |
| Shot end-to-end recall | 0.333 | 0.466 | +0.133 ✅ |
| Within-cov F1 | 0.420 | 0.543 | +0.123 ✅ |
| Within-cov recall | 0.829 | 0.746 | -0.083 ❌ |
| Shot MAE | 1.340 | 1.113 | -0.227 ✅ |
| Predicted shots | 101 | 96 | -5 |
| Predicted goals | 1 | 1 | 0 |
| n_windows | 50 | 53 | +3 |

## Aggregate (mean across vIDs that have both runs)

| metric | v13 (mean) | fusion_wide (mean) | Δ |
|---|---|---|---|
| Goal F1 | 0.645 | 0.750 | +0.106 ✅ |
| Goal precision | 0.867 | 1.000 | +0.133 ✅ |
| Goal recall | 0.560 | 0.607 | +0.047 ✅ |
| Shot end-to-end F1 | 0.371 | 0.430 | +0.059 ✅ |
| Shot end-to-end R | 0.447 | 0.488 | +0.041 ✅ |
| Within-cov F1 | 0.473 | 0.526 | +0.053 ✅ |
| Within-cov R | 0.883 | 0.800 | -0.083 ❌ |
| Shot MAE | 1.146 | 1.070 | -0.075 ✅ |

## Headline

If most ✅ marks appear in the Δ column AND aggregate metrics improved, fusion-wide is the new production default. If mixed or regressive, stick with v13.