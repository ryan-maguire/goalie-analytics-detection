# fusion-wide validation report

Generated: 2026-05-30 00:53:13 UTC

## Per-video metrics

### SX5xNJlh6eQ

| metric | v13 (cv_seg) | fusion_wide | Δ |
|---|---|---|---|
| Goal F1 (STRICT) | — | — | — |
| Goal precision | — | — | — |
| Goal recall | 0.000 | 0.000 | 0.000 |
| Goal TP | 0 | 0 | 0 |
| Goal FP | 0 | 0 | 0 |
| Goal FN | 3 | 3 | 0 |
| Shot end-to-end F1 | 0.265 | 0.258 | -0.007 ❌ |
| Shot end-to-end recall | 0.397 | 0.339 | -0.058 ❌ |
| Within-cov F1 | 0.311 | 0.321 | +0.010 ✅ |
| Within-cov recall | 0.714 | 0.700 | -0.014 ❌ |
| Shot MAE | 1.441 | 1.179 | -0.262 ✅ |
| Predicted shots | 126 | 101 | -25 |
| Predicted goals | 0 | 0 | 0 |
| n_windows | 59 | 56 | -3 |

### bfEKgtOIkQU

| metric | v13 (cv_seg) | fusion_wide | Δ |
|---|---|---|---|
| Goal F1 (STRICT) | — | — | — |
| Goal precision | 0.000 | 0.000 | 0.000 |
| Goal recall | 0.000 | 0.000 | 0.000 |
| Goal TP | 0 | 0 | 0 |
| Goal FP | 1 | 1 | 0 |
| Goal FN | 2 | 2 | 0 |
| Shot end-to-end F1 | 0.273 | 0.354 | +0.081 ✅ |
| Shot end-to-end recall | 0.491 | 0.468 | -0.024 ❌ |
| Within-cov F1 | 0.306 | 0.408 | +0.102 ✅ |
| Within-cov recall | 0.800 | 0.725 | -0.075 ❌ |
| Shot MAE | 1.173 | 1.193 | +0.020 ❌ |
| Predicted shots | 148 | 101 | -47 |
| Predicted goals | 1 | 1 | 0 |
| n_windows | 81 | 57 | -24 |

## Aggregate (mean across vIDs that have both runs)

Means are reported with 95% percentile bootstrap CIs (2000 resamples). Δ is the **paired** bootstrap on per-vID differences — so it preserves within-vID correlation that an unpaired bootstrap would dilute. A Δ CI that crosses zero means the result is consistent with noise.

| metric | v13 (mean [95% CI]) | fusion_wide (mean [95% CI]) | Δ (paired [95% CI]) |
|---|---|---|---|
| Goal F1 | — | — | — |
| Goal precision | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] ~ |
| Goal recall | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] ~ |
| Shot end-to-end F1 | 0.269 [0.265, 0.273] | 0.306 [0.258, 0.354] | +0.037 [-0.007, +0.081] ~ |
| Shot end-to-end R | 0.444 [0.397, 0.491] | 0.403 [0.339, 0.468] | -0.041 [-0.058, -0.024] ❌ |
| Within-cov F1 | 0.308 [0.306, 0.311] | 0.365 [0.321, 0.408] | +0.056 [+0.010, +0.102] ✅ |
| Within-cov R | 0.757 [0.714, 0.800] | 0.712 [0.700, 0.725] | -0.045 [-0.075, -0.014] ❌ |
| Shot MAE | 1.307 [1.173, 1.441] | 1.186 [1.179, 1.193] | -0.121 [-0.262, +0.020] ~ |

Legend: ✅ = CI excludes zero in the favorable direction. ❌ = CI excludes zero in the unfavorable direction. `~` = CI crosses zero (no signal above noise on this sample size).

## Headline

If most ✅ marks appear in the Δ column AND aggregate metrics improved, fusion-wide is the new production default. If mixed or regressive, stick with v13.