# High-Danger Save % & Danger-Zone Splits
*The save percentage goalies, coaches, and recruiters actually speak in — broken out by shot danger, side, and period.*

## Summary & problem
Goalie performance is judged in the real world by **danger-weighted save percentage** — the youth/junior analog of the NHL's HDSV% (high-danger save %). A goalie who stops 9/10 from the slot is far more impressive than one who stops 9/10 of point shots, but our product today only exposes **raw counts** (`clipSaveCount`, `metrics.saves`, `metrics.shotsOnNet`) in `StatsDashboard`. Families and recruiters cannot answer "how does this kid do on high-danger chances?" — the single most-asked goalie question.

This feature computes **save % per danger tier** (high / medium / low) plus **situational splits** (by danger, by goalie side, by approximate period, and optionally by outcome save/rebound/goal), with **sample-size–aware error bars** so a 2-shot sample is never shown as a confident "100%".

Critically, **there is no reliable danger classification in production today.** `threat_type` exists in the old/mock schema but is **not populated** in production windows — do not use it. We ship in two stages:
- **v1 PROXY** (ships now): derive a `shot_danger` tier from fields we already have — chiefly the per-shot `metrics.shot_timestamps[].location` / `.release` taxonomy, plus `shotsOnNet` vs `shots`, `rebounds`, `clipHasGoal`, and cue-parsing of `technical_reasoning`.
- **v2 PIPELINE FIELD** (accuracy upgrade): add a first-class `shot_danger` field to each `shot_timestamps` entry via a `metrics_seg` Gemini prompt addition.

## Target users
- **Goalies** — see "I'm .850 on high-danger, .960 on low-danger" and where to focus.
- **Goalie families** — a single recruiter-credible number ("HDSV%") instead of raw saves.
- **Goalie coaches** — situational splits (danger × side × period) to target practice.
- **Recruiters (secondary)** — comparable danger-weighted stat across games/seasons.

## Data inputs
All from `gs://goalie_video_bucket/analyze_video/04-final_video/<vID>.json` (array of items; `item.type ∈ {summary, clips, windows}`). Per-clip fields consumed:

| Field | Type | Use |
|---|---|---|
| `metrics.shots` | int | denominator candidate (all shot attempts) |
| `metrics.shotsOnNet` | int | SOG = save% denominator |
| `metrics.saves` | int | save% numerator (`shotsOnNet - goals`) |
| `metrics.goals` | int | goals against |
| `metrics.rebounds` | int | danger proxy + rebound split |
| `metrics.observed_goalie_side` | str `left`/`right` | **side split** (preferred over `threat_goalie_side`) |
| `metrics.shot_timestamps[]` | array | **per-shot** `location`, `release`, `outcome` — primary proxy basis. **NEW to UI** (exists in pipeline schema + JSON, not yet in `types.ts`). |
| `clipShotCount` / `clipSaveCount` / `clipHasGoal` | int/bool | fallback when `metrics` missing |
| `technical_reasoning` | str | cue-parsing fallback (rush/slot/cross-ice/breakaway/rebound) |
| `coaching_feedback.rebound_control_rank` | str (qualitative) | rebound split context (needs rank→ordinal map) |
| `clip_start_time` / `clip_end_time` | float (sec) | **approximate period** derivation |
| `analysis_confidence_caveats`, `*_confidence_score` | str/num | down-weight / flag low-confidence clips |

**NEW field (v2, stage = `metrics_seg`):** `metrics.shot_timestamps[].shot_danger` ∈ `{high, medium, low}` (+ optional `shot_danger_confidence`). See Pipeline.

`summary.response`: `windows_analysed/succeeded/failed` → coverage caveat in the panel header.
Customer config (`customerID/<custID>.json`): `event_date`, `eventSeason`, `segmentDuration` for cross-game roll-ups (out of MVP scope; informs PerformanceHistory linkage later).

> Reality check on existing schema (grounded): `shot_timestamps[].location` already uses a fixed taxonomy in `metrics_seg/prompts/metrics_v14.1.txt`: *"right point," "high slot," "left circle," "low slot," "right circle," "left point," "right wing," "left wing," "in close," "behind net."* `release` ∈ *slap/wrist/snap/one-timer/redirect-tip/backhand*. `outcome` ∈ *goal/save/miss/blocked*. This is the backbone of an accurate v1 proxy — far better than text-parsing alone.

## Pipeline / backend changes

### Option A — v1 proxy (NO pipeline change)
Compute `shot_danger` **client-side** (UI) or in the API gateway from existing fields. No `metrics_seg` change. Ships immediately. Accuracy bounded by location-taxonomy fidelity (see Limitations).

### Option B — v2 first-class `shot_danger` (the accuracy upgrade)
Add danger classification **at the source**, where Gemini already sees the clip and the play. This is strictly better than the proxy because Gemini can read play type (rush vs cycle), traffic/screen, lateral movement, and pre-shot passing that location-string alone loses.

**Schema change** in `metrics_seg/01_detect_segment_metrics.py` → `METRICS_RESPONSE_SCHEMA.shot_timestamps.items`:
```python
"shot_danger":            {"type": "STRING"},   # high | medium | low
"shot_danger_confidence": {"type": "STRING"},   # high | medium | low
```
Add both to that item's `required` list. (Per-shot, so it composes with the existing v13 truth-table features.)

**Prompt addition** (append to the structured-shot-enumeration block, ~line 234 of `metrics_v14.1.txt`, as a new prompt version `v15`):
```
   9. `shot_danger`: classify THIS shot's scoring danger from where it
      was taken and how the play developed, NOT from whether it scored:
        - "high"   — shot from the low/high slot, "in close," a clear
                     rush/breakaway, a one-timer or cross-ice/royal-road
                     pass immediately before release, or a rebound/second
                     chance with the goalie out of position.
        - "medium" — shot from the circles or a sharp-angle wing chance
                     with some traffic/screen but no slot entry.
        - "low"    — shot from the point or perimeter, well-angled,
                     goalie set and square, no screen and no pre-shot
                     lateral movement.
      Judge danger by chance quality at release. Do NOT upgrade a shot
      to "high" merely because it became a goal, and do NOT downgrade a
      slot one-timer to "low" because it was saved.
  10. `shot_danger_confidence`: high | medium | low — your confidence in
      the danger call given camera angle and clarity.
```
After shipping v2, the UI prefers `shot_danger` and **falls back to the proxy** per-shot whenever the field is absent (older JSONs / re-analysis lag). **Deploy note:** detection repo has no auto-deploy — a `metrics_seg` change requires `bash deploy/deploy.sh` and re-analysis of games to populate the new field.

## Computation / logic

### Core formulas
For a tier T (or any split bucket):
```
SV%(T)  = saves(T) / shotsOnNet(T)          # saves(T) = shotsOnNet(T) - goals(T)
HDSV%   = SV%(high)                          # the headline number
```
- **Denominator = `shotsOnNet`** (shots that actually reached the goalie), NOT `shots`. Misses/blocked (`outcome ∈ {miss, blocked}`) are excluded from SV% but kept for a separate "chances faced" count.
- Per-shot bucketing uses `shot_timestamps[].outcome` to assign save vs goal to a tier. When a clip has `metrics.shotsOnNet` but a shorter/empty `shot_timestamps` array, fall back to clip-level proxy (below) and attribute the whole clip to one tier.

### v1 proxy heuristic (pseudocode)
```
function shotDanger(shot, clip):
    loc = lower(shot.location)         # from shot_timestamps taxonomy
    rel = lower(shot.release)
    txt = lower(clip.technical_reasoning)

    # 1) Location is the strongest signal (taxonomy is fixed/known)
    if loc contains any("in close","low slot","high slot","slot"):    base = HIGH
    elif loc contains any("circle","wing"):                            base = MEDIUM
    elif loc contains any("point","behind net"):                       base = LOW
    else:                                                              base = MEDIUM   # unknown → neutral

    # 2) Upgrades (chance-quality amplifiers)
    if rel contains any("one-timer","redirect","tip"):                base = max(base, HIGH if base>=MEDIUM else MEDIUM)
    if shot.outcome == "save" and clip.metrics.rebounds > 0:          bump(base, +1)   # 2nd chances
    if regex(txt, "breakaway|rush|odd-man|2-on-1|3-on-2|cross-ice|royal road|backdoor|wide open|in alone"):
                                                                      bump(base, +1)
    return clamp(base, LOW..HIGH)

# Clip-level fallback when shot_timestamps is missing/empty:
function clipDangerFallback(clip):
    if clip.clipHasGoal:                          return HIGH    # weak; flag low-confidence
    if clip.metrics.rebounds > 0:                 return HIGH
    if clip.metrics.shotsOnNet < clip.metrics.shots: return MEDIUM  # contested/blocked traffic
    text-cue scan as above; else                  return MEDIUM
```
`bump()` raises a tier by one level (LOW→MEDIUM→HIGH, capped). Each proxy-derived shot is tagged `danger_source="proxy"` so the UI can render a dotted/“est.” style and exclude from recruiter-export by default.

### Splits
- **By danger tier:** group shots by tier → `SV%(high|medium|low)`, with n per tier.
- **By side:** group by `metrics.observed_goalie_side` (`left`/`right`); fall back to `threat_goalie_side` if `observed_goalie_side` absent. Yields `SV%(left)` vs `SV%(right)` — exposes glove/blocker-side weakness.
- **By approximate period:** **period is NOT a field.** Approximate from `clip_start_time` relative to game span:
  ```
  game_start = min(clip_start_time over clips);  game_end = max(clip_end_time)
  frac = (clip_start_time - game_start) / (game_end - game_start)
  period = ceil(frac * NUM_PERIODS)              # NUM_PERIODS default 3
  ```
  **Assumptions (state in UI tooltip):** uniform real-time mapping, no intermission/overtime correction, clip times are continuous game time. Label as "approx. period (P1/P2/P3)". If game span < a threshold or coverage gaps are large, hide the period split.
- **By outcome (optional):** save vs rebound vs goal, using `outcome` / `metrics.rebounds` / `metrics.goals`, to show "of high-danger SOG: X saved clean, Y rebound-then-controlled, Z goals."

### Rank & confidence handling
Ranks (`depth_rank`, `rebound_control_rank`, etc.) are **qualitative strings** — never averaged raw. Define a single `rankToOrdinal` map (`elite/excellent=4, good/strong=3, average/fair=2, poor/weak=1, unknown=null`) reused across the app; only ordinals (null-excluded) are aggregated. Clips/shots below a confidence threshold (`goalie_position_confidence_score`, `coaching_confidence_score`, or presence of `analysis_confidence_caveats`) are counted but flagged, and excluded from the headline HDSV% by an optional "high-confidence only" toggle.

### Sample-size / error bars
Per bucket, show a Wilson 95% CI for the save proportion:
```
n = shotsOnNet(bucket);  p = saves/n
ci = wilson(p, n, z=1.96)            # returns [lo, hi]
```
Buckets with `n < MIN_N` (default 5) render the % greyed with an explicit "small sample (n=__)" label and a wide CI; never shown as a clean headline. HDSV% headline requires `n_high ≥ MIN_N` else shows "—" + "not enough high-danger shots yet."

## Frontend
**New component:** `DangerSavePanel` (`components/DangerSavePanel.tsx`), with a small reusable `SaveRateBar` (value + Wilson CI whisker) and `useDangerSplits(clips)` hook (`hooks/useDangerSplits.ts`) holding all proxy/aggregation logic so it's unit-testable and reusable by `PerformanceHistory` later.

**Placement:** new card in `StatsDashboard`, directly **below** the existing raw-count tiles (which it complements, not replaces). Respects the global `FilterBar` / `GameSelector` selection. **Add `shot_timestamps` to `ClipDetail.metrics` in `types.ts`** (currently absent) and a `ShotDetail` type.

**UX:** headline HDSV% big-number with CI + n; a three-row danger breakdown; a toggle row for split dimension (Danger / Side / Period / Outcome); "est." badge when any bucket is proxy-derived; "high-confidence only" toggle. Clicking a bucket deep-links to the existing `Timeline`/`ClipModal` filtered to those shots (reuse, no new viewer).

```
┌──────────────────────────────────────────────────────────────┐
│  HIGH-DANGER SAVE %                         [Danger▼] est. ⓘ  │
│                                                                │
│     .857   HDSV%   (n=14, 95% CI .60–.96)                      │
│     ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇░░░  |————●————|                       │
│                                                                │
│  Tier        SV%     n     ▕ save-rate w/ 95% CI               │
│  High      .857     14     ▕ ▇▇▇▇▇▇▇▇▇▇▇▇░ |——●——|             │
│  Medium    .923     26     ▕ ▇▇▇▇▇▇▇▇▇▇▇▇▇ |—●—|               │
│  Low       .975     40     ▕ ▇▇▇▇▇▇▇▇▇▇▇▇▇ |●|                 │
│  ----------------------------------------------------------    │
│  Splits:  [Danger] [ Side ] [Period] [Outcome]                 │
│  Side   L: .910 (n=33)   R: .944 (n=47)                        │
│  ⚠ small sample where n<5 shown greyed · ☐ high-confidence only │
│  Source: 12 windows analysed · danger est. from shot location  │
└──────────────────────────────────────────────────────────────┘
```

## Edge cases & limitations
- **No production danger field today** — v1 is a proxy; never present it as exact. Label "est." and exclude proxy data from recruiter export by default.
- **Proxy inaccuracy:** location taxonomy is Gemini-derived prose and not always precise; a "screened point shot" reads LOW but is genuinely dangerous, and slot shots can be mis-located. Text-cue scan is brittle (keyword matching). Expect tier mislabeling on a minority of shots → v2 field is the fix.
- **`shot_timestamps` shorter than `shotsOnNet`:** array may omit some SOG; reconcile by attributing leftover SOG to the clip-level fallback tier.
- **Small samples:** youth games yield few high-danger SOG; Wilson CI + `MIN_N` gating prevents misleading headline numbers.
- **Missing fields:** `metrics` can be null on no-threat clips; `observed_goalie_side` may be empty (fall back to `threat_goalie_side`, else exclude from side split). Mixed goalies in one game → side/period splits can blend two goalies; out of MVP scope, note in tooltip.
- **Period approximation:** no intermission/OT awareness; clip-time gaps distort `frac`. Hide split when coverage is too sparse.
- **Double-counting:** a goal is in `metrics.goals` and may also be a `shot_timestamps.outcome="goal"` — bucket once via shots array, reconcile against clip totals.

## Phasing & effort
- **MVP — v1 proxy (S–M):** `useDangerSplits` hook + `DangerSavePanel` + `SaveRateBar`, `types.ts` additions, Wilson CI, danger/side/period splits. Pure frontend; no pipeline/deploy. **~S–M.**
- **v2 — pipeline `shot_danger` (M):** add schema fields + `metrics_v15` prompt, validate on the eval set (pooled F1 / agreement vs a labeled danger set), `bash deploy/deploy.sh`, re-analyze games, switch UI to field-with-proxy-fallback. **~M** (mostly prompt iteration + re-analysis cost).
- **v3 (later):** cross-game HDSV% trend in `PerformanceHistory`; recruiter PDF export of field-backed (non-proxy) numbers only.

## Success metrics
- ≥80% of viewed games render a non-"—" HDSV% (sufficient high-danger n).
- v2 `shot_danger` ≥ 80% agreement with human labels on the eval set before it replaces the proxy headline.
- Engagement: % of dashboard sessions that toggle a split dimension; deep-link clicks from a danger bucket into `ClipModal`.
- Qualitative: coaches confirm the high-danger bucket matches their eye on a sample of clips.

## Open questions
1. Tier cut-points: 3 tiers (high/med/low) or NHL-style 2 (high vs rest)? Default 3.
2. Denominator for HDSV% — `shotsOnNet` only (proposed) or include misses as "chances"? 
3. `MIN_N` and `NUM_PERIODS` defaults (5 and 3) — confirm for youth/junior game lengths.
4. Should proxy-derived numbers ever surface in recruiter export, clearly flagged, or be field-only?
5. Compute proxy client-side (fast, MVP) vs in the API gateway (cacheable, shareable)? 
6. Cross-game roll-up keying — by `eventSeason`, by goalie, or both (multi-goalie games)?
