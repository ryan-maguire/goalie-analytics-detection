# Season Report Card
*The screenshot-worthy A–F grade card that turns a season of clips into a story families and coaches actually share.*

## Summary & problem
Goalie Analytics Pro already produces rich per-clip analysis, career trend lines (PerformanceHistory),
per-snapshot shape (MechanicsRadarChart), and per-game tiles (StatsDashboard). What it does **not** have
is a single, graded, narrative artifact that answers the question a parent or coach asks first:
*"How is the season going, and is it getting better?"*

The **Season Report Card** is a pure-frontend view that grades each goalie pillar **A–F** over a chosen
season, attaches a **trend arrow** (improving / steady / declining vs. the prior stretch of games), and
surfaces **auto-detected milestones** ("Squareness improved a full grade," "Best save% month: November").
It is deliberately a *report card*, not a chart: large letter grades, plain-language callouts, one screen,
built to be screenshotted and texted.

This is distinct from existing features:
- **PerformanceHistory** = line charts of metrics over time (analytical, exploratory).
- **MechanicsRadarChart** = one moment's shape across pillars (snapshot).
- **Season Report Card** = the *graded narrative layer* — a verdict, a direction, and milestones.

## Target users
- **Goalies** — a motivating, fair grade per skill area and proof of progress.
- **Families** — a digestible, shareable season recap; no jargon required to read a letter grade.
- **Coaches** — a fast roster-level read on where a goalie stands and which pillar is trending the wrong way.

## Data inputs (exact fields)
All data is already available via `services/api.ts` and `types.ts`. No new backend.

**Game list (season scope)** — `fetchUserGames(customerId)` → `UserGame[]`, plus raw customer config
records (`getVideoMetadata` / `fetchFormOptions` expose `eventSeason`). Fields used:
- `vID`, `eventName`, `eventDate` (→ `event_date`), `clipCount`
- `eventSeason` (raw customer record; used to scope the report)

**Per-game data** — `fetchGameData(vID)` → `DataItem[]` (normalized). Per `ClipDetail`:
- `goalie_positioning.depth_rank` (Aggressive / Base / Conservative / Defensive)
- `goalie_positioning.cover_angle_rank` (Excellent / Good / Average / Poor)
- `goalie_positioning.squareness_rank` (Excellent / Good / Average / Poor)
- `goalie_positioning.goalie_position_confidence_score` (1–5)
- `coaching_feedback.rebound_control_rank` (Killed (Elite) / Steered to Low-Danger Zone /
  Uncontrolled into High-Danger Zone / Goal Allowed / Not Applicable)
- `coaching_feedback.coaching_confidence_score` (1–5)
- `clipHasGoal` (boolean, normalized)
- `metrics.shots`, `metrics.shotsOnNet`, `metrics.saves`, `metrics.goals`

**Summary** — `SummaryResponse.coaches_overall_rating` (per-game), `event_date`, `goalie_season`.
Used only as a cross-check / display, **not** as a grade source (we re-derive grades from clips so the
pooling is correct and auditable).

## Pipeline / backend
**Pure frontend over existing JSON.** The report is computed in the browser from `fetchUserGames` +
N× `fetchGameData`. No detection-repo (`04-final_video/<vID>.json`) changes are required.

*Optional enhancement (Phase 2):* persist a computed snapshot via the existing `save-json` endpoint
(`SAVE_JSON_ENDPOINT`, already wired as `saveSegmentGT` uses it) under an id like
`season_report/<customerId>/<season>` so shared links render instantly without re-fetching every game.
This is a cache/share convenience only and never the source of truth.

## Computation / logic
All aggregation is **POOLED across clips** for the season. Never average per-game percentages and never
average per-game letter grades — pool the raw clip scores (and pool raw save/shot counts), then grade once.

### Step 1 — rank → ordinal → 0–100
Reuse and **extend** `utils/helpers.ts` `rankToScore`. The current implementation predates the
production taxonomy and has gaps (it keys on "save"/"corner"/"slot"/"shooter" rebound strings and lumps
"conservative" with "neutral"). Extend it to cover the exact production strings and to distinguish
Conservative from Defensive:

| Pillar | Rank string | Score |
|---|---|---|
| **Depth** | Aggressive | 100 |
| | Base | 85 |
| | Conservative | 65 |
| | Defensive | 45 |
| **Angle / Squareness** | Excellent | 100 |
| | Good | 80 |
| | Average | 60 |
| | Poor | 40 |
| **Rebound control** | Killed (Elite) | 100 |
| | Steered to Low-Danger Zone | 75 |
| | Uncontrolled into High-Danger Zone | 40 |
| | Goal Allowed | 10 |
| | Not Applicable | *excluded* |

Implementation note: add an optional `pillar` arg to `rankToScore` so the same string ("Average") is
unambiguous, and keep the existing zero-arg behavior for backward compatibility with current callers.
`Not Applicable` and missing/empty ranks return `null` (not 0) so they are **excluded** from the pool,
not counted as failures.

### Step 2 — pooled pillar score
For each pillar P over the season's clips:
```
poolScore(P) = sum(confWeight_c * score_c) / sum(confWeight_c)   over clips c where score_c != null
```
**Confidence weighting:** `confWeight_c = clamp(conf_c, 1, 5) / 5`, where `conf_c` is
`goalie_position_confidence_score` for positioning pillars (Depth/Angle/Squareness) and
`coaching_confidence_score` for Rebound. Missing confidence defaults to weight `0.6` (i.e. score 3/5)
so low-info clips still count but lightly. Weighting can be toggled off in UI (show raw mean) for transparency.

### Step 3 — Save% grade (counting stat, pooled)
```
saves   = sum(metrics.saves)  ;  goals = sum(metrics.goals)  over all season clips
shotsOnNet = saves + goals    (fallback to sum(metrics.shotsOnNet) if larger)
savePct = saves / shotsOnNet   (undefined if shotsOnNet == 0)
```
Map `savePct` to 0–100 on a goalie-realistic curve before grading (raw .900 should not read as a B):
```
savePctScore = clamp( (savePct - 0.820) / (0.960 - 0.820) * 100, 0, 100 )
```
So .820→F-floor, .890≈50 (D/C border), .920≈71 (C+), .940≈86 (B+/A-), .960+→A. Curve constants live in
one config block for easy tuning by the coaching team.

### Step 4 — 0–100 → letter grade
Single shared cutoff table for **every** pillar and Overall:
```
A 90–100   B 80–89   C 70–79   D 60–69   F <60
+/- bands (optional display): X7–X9 → "+", X0–X3 → "−" within the letter band, A capped at A (no A+).
```

### Step 5 — Overall grade
Weighted blend of pillar scores (weights in config, default):
```
Overall = 0.20*Depth + 0.20*Angle + 0.20*Squareness + 0.20*Rebound + 0.20*SavePctScore
```
Pillars with no gradable clips are dropped and weights renormalized over the rest. If
`coaches_overall_rating` exists it is shown as a secondary "Coach's read" chip, not blended.

### Step 6 — Trend arrow (recent vs prior)
Order season games by `event_date`. Define **recent window N = last 3 games** (config; if season < 6
games use `floor(games/2)`, min 1). Compute each pillar score **pooled over the recent N games' clips**
vs **pooled over all prior games' clips**.
```
delta = recentScore - priorScore
↑ improving   if delta >= +4
→ steady      if -4 < delta < +4
↓ declining   if delta <= -4
```
Thresholds in score points (≈ half a letter band). Show the delta and the windows on hover/caption
("Last 3 games vs. first 9"). If no prior window (≤ N games total) → trend = "Not enough games yet".

### Step 7 — Milestone detection
Run a small rule set over the ordered season; emit any that fire (cap display at ~5, newest first):
- **Full-grade jump:** a pillar's recent-window grade is ≥1 letter above its prior-window grade →
  "Squareness improved a full grade (C → B)."
- **Best save% month:** group games by calendar month; flag the month with the highest pooled save%
  (require ≥2 games in the month) → "Best save% month: November (.928)."
- **Pillar of the season:** highest-scoring pillar with ≥ X clips → "Strongest pillar: Angle (A−)."
- **Most improved:** pillar with the largest positive `delta` (≥ +6) → "Most improved: Rebound control."
- **Shutout-ish game:** any game with `goals == 0` and `shotsOnNet >= 15` → "Wall night vs. <opponent>."
- **Streak:** ≥3 consecutive games where Overall grade ≥ B → "3-game B+ streak."
Each milestone carries a type, label, value, and (where applicable) the contributing game(s) for deep-link.

## Frontend
New view **`SeasonReportCard`** (`components/SeasonReportCard.tsx`) plus a thin
`utils/seasonReport.ts` for the pure computation (unit-testable, no React). Extend `rankToScore` in
`utils/helpers.ts`. Reuse the existing **GameSelector** but in *season scope* — a season dropdown
(populated from `eventSeason`) drives which games are pooled.

UX:
- A **report-card header**: goalie name, season, big **Overall** grade, Coach's-read chip, game count.
- A grid of **per-pillar grade cards**: large letter, the 0–100 sub-score, a trend arrow with delta,
  the clip-count it was graded on, and a one-line plain-language verdict. Color via existing
  `getScoreColor` / `getBadgeColor` conventions.
- **Milestones list**: icon + sentence, each linking back to its game(s) where applicable.
- **Share / export**: "Export as image" (html-to-image/canvas of the card) and "Copy share link"
  (encodes customerId + season; resolves the optional saved snapshot). Built so the card alone is the
  shareable unit (no surrounding chrome in the export).
- Loading state while N games fetch (parallel `Promise.all`); a "graded on M of K clips" footnote for trust.

```
+----------------------------------------------------------------------+
|  SEASON REPORT CARD            Season: 2025-26 ▾     [Export] [Share] |
|  Avery Maguire · 12 games · graded on 318 / 340 clips                 |
|                                                                      |
|        OVERALL   [ B+ ]   ↑ improving (+5 vs first 9)                 |
|        Coach's read: "Strong, competing hard"                        |
|----------------------------------------------------------------------|
|  DEPTH        ANGLE          SQUARENESS     REBOUND        SAVE %     |
|  [ B  ]       [ A- ]         [ C+ ]         [ B- ]         [ B  ]     |
|   84 →steady   91 ↑+6         77 ↑+11        81 →steady     .921      |
|  104 clips    104 clips      104 clips      88 clips       12 games   |
|  "Patient,    "Tracks pucks  "Up a full     "Killing more  "Top-third|
|   set early"   well"          grade!"        rebounds"      of league"|
|----------------------------------------------------------------------|
|  MILESTONES                                                          |
|  ★ Squareness improved a full grade (C → B)                          |
|  ★ Best save% month: November (.928)                                 |
|  ★ Most improved: Squareness (+11)                                   |
|  ★ Wall night vs. Riverside (0 GA on 19 shots) →                     |
+----------------------------------------------------------------------+
```

## Edge cases
- **Single game in season:** grade it, but trend = "Not enough games yet" and milestones requiring a
  prior window are suppressed. Footnote: "Grades from a single game; treat as a snapshot."
- **Small clip samples per pillar:** if a pillar has `< 8` gradable clips, still grade but mark with a
  "low sample" badge and dim it; exclude it from "Pillar of the season" / "Most improved".
- **All `Not Applicable` rebound (e.g., no rebound situations):** rebound pillar shows "—" / "No data",
  excluded from Overall (weights renormalize).
- **Missing ranks / empty strings:** excluded from the pool (return `null`), never scored as 0.
- **`shotsOnNet == 0`:** Save% card shows "—"; excluded from Overall.
- **Mixed goalies under one customer:** scope by goalie (use `goalie_name`) if multiple appear; default
  to the most-clipped goalie and offer a goalie picker.
- **No `eventSeason` on records:** fall back to bucketing by calendar year of `event_date`.
- **Confidence all missing:** weighting silently degrades to the 0.6 default → effectively unweighted.

## Phasing & effort
- **Phase 1 (S):** Extend `rankToScore`; build `utils/seasonReport.ts` (pooling, grading, trend, save%);
  `SeasonReportCard` view with grade cards + trend arrows; season scope via GameSelector. Unit tests on
  the pure logic. *Deps: none beyond existing api.ts/types.ts.*
- **Phase 2 (S):** Milestone detection rule set + milestones list with deep-links.
- **Phase 3 (M):** Export-as-image + share link; optional `save-json` snapshot cache for shared links.
  *Deps: html-to-image (or canvas) lib; reuse SAVE_JSON_ENDPOINT.*

## Success metrics
- **Adoption:** % of active goalies who open the report card per season; export/share click rate.
- **Shareability:** number of exports/share-link generations (the core intent).
- **Trust:** low rate of "grade looks wrong" feedback; pooled grades reconcile with `coaches_overall_rating`
  within one letter band for ≥80% of games.
- **Engagement lift:** sessions that view the report card → higher clip-detail drill-through (milestone deep-links).

## Open questions
- Are the save% curve anchors (.820–.960) right across age levels, or should they scale by `goalie_season` / division?
- Trend window N = 3 — fixed, or a user toggle (last 3 / last 5 / last month)?
- Show +/- letter bands, or keep it to clean single letters for families?
- Should Overall blend in `coaches_overall_rating`, or keep the LLM read strictly as a secondary chip?
- Confidence weighting on by default? Coaches may prefer raw, unweighted grades for explainability.
- Milestone cap and ranking when many fire — fixed priority order, or relevance-scored?
