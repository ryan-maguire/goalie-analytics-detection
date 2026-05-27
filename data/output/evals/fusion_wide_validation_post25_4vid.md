# fusion-wide validation report

Generated: 2026-05-27 06:04:25 UTC

## Per-video metrics

### krxhPVLGLz8

| metric | v13 (cv_seg) | fusion_wide | Δ |
|---|---|---|---|
| Goal F1 (STRICT) | 1.000 | 0.667 | -0.333 ❌ |
| Goal precision | 1.000 | 1.000 | 0.000 |
| Goal recall | 1.000 | 0.500 | -0.500 ❌ |
| Goal TP | 1 | 1 | 0 |
| Goal FP | 0 | 0 | 0 |
| Goal FN | 0 | 1 | +1 ❌ |
| Shot end-to-end F1 | 0.331 | 0.268 | -0.063 ❌ |
| Shot end-to-end recall | 0.373 | 0.453 | +0.080 ✅ |
| Within-cov F1 | 0.427 | 0.291 | -0.136 ❌ |
| Within-cov recall | 0.759 | 0.615 | -0.143 ❌ |
| Shot MAE | 1.143 | 1.659 | +0.516 ❌ |
| Predicted shots | 69 | 114 | +45 |
| Predicted goals | 1 | 1 | 0 |
| n_windows | 42 | 44 | +2 |

### v0lxSTbXfw8

| metric | v13 (cv_seg) | fusion_wide | Δ |
|---|---|---|---|
| Goal F1 (STRICT) | 0.800 | 0.889 | +0.089 ✅ |
| Goal precision | 0.667 | 0.800 | +0.133 ✅ |
| Goal recall | 1.000 | 1.000 | 0.000 |
| Goal TP | 2 | 4 | +2 ✅ |
| Goal FP | 1 | 1 | 0 |
| Goal FN | 0 | 0 | 0 |
| Shot end-to-end F1 | 0.313 | 0.276 | -0.037 ❌ |
| Shot end-to-end recall | 0.361 | 0.263 | -0.098 ❌ |
| Within-cov F1 | 0.380 | 0.360 | -0.019 ❌ |
| Within-cov recall | 0.605 | 0.476 | -0.129 ❌ |
| Shot MAE | 0.951 | 1.216 | +0.265 ❌ |
| Predicted shots | 94 | 63 | -31 |
| Predicted goals | 3 | 5 | +2 |
| n_windows | 61 | 37 | -24 |

### Fjc9hmK8_3U

| metric | v13 (cv_seg) | fusion_wide | Δ |
|---|---|---|---|
| Goal F1 (STRICT) | 1.000 | — | — |
| Goal precision | 1.000 | — | — |
| Goal recall | 1.000 | 0.000 | -1.000 ❌ |
| Goal TP | 1 | 0 | -1 ❌ |
| Goal FP | 0 | 0 | 0 |
| Goal FN | 0 | 2 | +2 ❌ |
| Shot end-to-end F1 | 0.375 | 0.408 | +0.033 ✅ |
| Shot end-to-end recall | 0.515 | 0.400 | -0.115 ❌ |
| Within-cov F1 | 0.432 | 0.522 | +0.090 ✅ |
| Within-cov recall | 0.809 | 0.700 | -0.110 ❌ |
| Shot MAE | 1.271 | 1.020 | -0.251 ✅ |
| Predicted shots | 170 | 101 | -69 |
| Predicted goals | 1 | 0 | -1 |
| n_windows | 85 | 51 | -34 |

### HNG0jKYY12g

| metric | v13 (cv_seg) | fusion_wide | Δ |
|---|---|---|---|
| Goal F1 (STRICT) | — | 0.333 | — |
| Goal precision | — | 0.500 | — |
| Goal recall | 0.000 | 0.250 | +0.250 ✅ |
| Goal TP | 0 | 1 | +1 ✅ |
| Goal FP | 0 | 1 | +1 ❌ |
| Goal FN | 2 | 3 | +1 ❌ |
| Shot end-to-end F1 | 0.165 | 0.301 | +0.136 ✅ |
| Shot end-to-end recall | 0.165 | 0.302 | +0.138 ✅ |
| Within-cov F1 | 0.233 | 0.391 | +0.158 ✅ |
| Within-cov recall | 0.400 | 0.565 | +0.165 ✅ |
| Shot MAE | 1.065 | 1.047 | -0.018 ✅ |
| Predicted shots | 85 | 73 | -12 |
| Predicted goals | 0 | 2 | +2 |
| n_windows | 62 | 43 | -19 |

## Aggregate (mean across vIDs that have both runs)

| metric | v13 (mean) | fusion_wide (mean) | Δ |
|---|---|---|---|
| Goal F1 | 0.900 | 0.778 | -0.122 ❌ |
| Goal precision | 0.834 | 0.900 | +0.067 ✅ |
| Goal recall | 0.750 | 0.438 | -0.312 ❌ |
| Shot end-to-end F1 | 0.296 | 0.313 | +0.017 ✅ |
| Shot end-to-end R | 0.353 | 0.355 | +0.001 |
| Within-cov F1 | 0.368 | 0.391 | +0.023 ✅ |
| Within-cov R | 0.643 | 0.589 | -0.054 ❌ |
| Shot MAE | 1.107 | 1.236 | +0.128 ❌ |

## Headline

If most ✅ marks appear in the Δ column AND aggregate metrics improved, fusion-wide is the new production default. If mixed or regressive, stick with v13.