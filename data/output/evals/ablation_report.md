# metrics_seg ablation report

Generated: 2026-05-25 21:25:39 UTC

## Goal F1 (STRICT) — `goal_strict_f1`

| variant | SX5xNJlh6eQ | bfEKgtOIkQU | mjEeE7p2Hz8 | mean |
|---|---|---|---|---|
| v14_prefilter | — | — | 0.400 | **0.400** |

## Goal precision (STRICT) — `goal_strict_p`

| variant | SX5xNJlh6eQ | bfEKgtOIkQU | mjEeE7p2Hz8 | mean |
|---|---|---|---|---|
| v14_prefilter | — | 0.000 | 1.000 | **0.500** |

## Goal recall (STRICT) — `goal_strict_r`

| variant | SX5xNJlh6eQ | bfEKgtOIkQU | mjEeE7p2Hz8 | mean |
|---|---|---|---|---|
| v14_prefilter | 0.000 | 0.000 | 0.250 | **0.083** |

## Goal F1 (UNFILTERED) — `goal_unfilt_f1`

| variant | SX5xNJlh6eQ | bfEKgtOIkQU | mjEeE7p2Hz8 | mean |
|---|---|---|---|---|
| v14_prefilter | — | — | — | **—** |

## Shot end-to-end F1 — `shot_e2e_f1`

| variant | SX5xNJlh6eQ | bfEKgtOIkQU | mjEeE7p2Hz8 | mean |
|---|---|---|---|---|
| v14_prefilter | 0.243 | 0.317 | 0.356 | **0.305** |

## Shot within-coverage F1 — `shot_inwin_f1`

| variant | SX5xNJlh6eQ | bfEKgtOIkQU | mjEeE7p2Hz8 | mean |
|---|---|---|---|---|
| v14_prefilter | 0.290 | 0.355 | 0.412 | **0.352** |

## Shot MAE (lower = better) — `shot_mae`

| variant | SX5xNJlh6eQ | bfEKgtOIkQU | mjEeE7p2Hz8 | mean |
|---|---|---|---|---|
| v14_prefilter | 1.404 | 1.238 | 1.113 | **1.252** |

## Windows compared — `n_windows`

| variant | SX5xNJlh6eQ | bfEKgtOIkQU | mjEeE7p2Hz8 | mean |
|---|---|---|---|---|
| v14_prefilter | 57 | 80 | 71 | **69.333** |

## Raw per-(vID, variant) counts

| vID | variant | n_windows | pred_goals | actual_goals | pred_shots | actual_shots | goal_F1 | shot_e2e_F1 |
|---|---|---|---|---|---|---|---|---|
| mjEeE7p2Hz8 | v14_prefilter | 71 | 2 | 8 | 123 | 58 | 0.400 | 0.356 |
| SX5xNJlh6eQ | v14_prefilter | 57 | — | 3 | 110 | 44 | — | 0.243 |
| bfEKgtOIkQU | v14_prefilter | 80 | 1 | 2 | 144 | 59 | — | 0.317 |

## Winners (by mean across videos)

| metric | best variant | mean |
|---|---|---|
| goal_strict_f1 | **v14_prefilter** | 0.4000 |
| shot_e2e_f1 | **v14_prefilter** | 0.3054 |
| shot_inwin_f1 | **v14_prefilter** | 0.3522 |
| shot_mae (lower=better) | **v14_prefilter** | 1.2517 |

## Honest read

- One game = ~7-10 goal events → 95% CI on goal F1 is ±0.30.
- Aggregate across all videos before concluding.
- Look at per-variant trends; ignore single-cell wobble.
- Shot-count improvements (MAE, e2e F1) are more statistically reliable than goal F1.