# Beaten Map
### See exactly *where* goals beat you — a net-face 6-zone heatmap, trended across the season and tied back to your positioning.

---

## Summary & problem

Goalies and their coaches already know *how many* goals went in and *where on the ice* the shots came from (the existing **ShotHeatmap** is an ice-location plot). What they cannot see today is the single most actionable thing for a goalie: **which part of the *net* the puck went through.** "You let in 9 goals this month" is not coachable. "**62% of the goals that beat you went glove-high, and they cluster on nights your depth is too deep**" is a training plan.

The **Beaten Map** is a NET-FACE heatmap: a 6-zone grid drawn over the goal mouth (glove/blocker × high/low, plus five-hole and body/other) showing where goals beat the goalie. It aggregates across all games in a season, surfaces the goalie's dominant leak, links every zone to the actual goal clips in `ClipModal`, and (Phase 3) correlates each zone with the goalie's positioning ranks to explain *why* the leak happens.

This is **not** the ShotHeatmap. ShotHeatmap = where on the *ice* shots originate (top-down rink). Beaten Map = where in the *net face* goals land (front-on goal grid). They answer orthogonal questions and live side by side.

**The catch:** the data to power this **does not exist yet.** No production field encodes a net zone, glove/blocker side, or puck-in-net coordinate. This feature requires a **new per-clip field, `beaten_location`, produced by a `metrics_seg` Gemini prompt addition**, scoped to goal clips only. Section 5 specifies it exactly.

---

## Target users

- **Goalies** — "Where do I keep getting beat?" The aha moment: one dominant red zone on the grid.
- **Goalie families** — a plain-language, visual answer to "how's their season going" beyond raw goal counts.
- **Goalie coaches** — turns a season of clips into a leak map; the zone↔positioning correlation (Phase 3) becomes a practice plan.

---

## Data inputs

### Existing per-clip fields (item.type == "clips") — used as-is, no change

Used directly by the Beaten Map:

| Field | Use |
|---|---|
| `clipID` | Link a zone's goals back to clips in `ClipModal`. |
| `clip_start_time`, `clip_end_time` | Jump-to-clip in `VideoPlayer`. |
| `clipHasGoal` | **Gate.** Beaten Map only ever plots clips where this is true. |
| `metrics.goals` | Goal count for the clip (a clip may contain >1 goal). |
| `goalie_positioning{depth_rank, cover_angle_rank, squareness_rank, goalie_position_confidence_score}` | Phase 3 zone↔positioning correlation. |
| `coaching_feedback{rebound_control_rank, actionable_coaching_feedback}` | Surfaced in the zone drill-down alongside the clip. |
| `metrics.observed_goalie_side` | Sanity context for left/right (screen-side) interpretation; **not** the same as glove/blocker side (see caveat in §5). |
| `analysis_confidence_caveats` | Shown so users understand reliability. |

### NEW field — `beaten_location` (and `beaten_location_confidence`)

> **Flagged: this is a `metrics_seg` ADD, not in production today.** It is added to the per-clip `metrics{}` object via a Gemini prompt change. It is **computed and meaningful ONLY for clips where `clipHasGoal == true`** (equivalently `metrics.goals >= 1`). For all non-goal clips it is omitted or set to `"not_applicable"`. See §5 for schema and prompt.

---

## Pipeline / backend changes

All changes are in **`goalie-analytics-detection` → `metrics_seg`** (the per-clip Gemini stage, prompt file `metrics_seg/prompts/metrics_v14.1.txt`). `cv_seg` and `feedback_seg` are untouched. The new field flows through unchanged into the published artifact at `gs://goalie_video_bucket/analyze_video/04-final_video/<vID>.json` and is served verbatim by the API gateway `/api/get-json`. **No API or schema change is needed in the gateway** — it passes the JSON through.

### 6-zone taxonomy (fixed, closed set)

Front-on view of the net from the shooter's perspective. **Glove/blocker are the GOALIE's hands**, not screen-left/right.

| Zone value | Plain language |
|---|---|
| `glove_high` | Over the glove, top of net |
| `glove_low` | Glove side, along the ice |
| `blocker_high` | Over the blocker, top of net |
| `blocker_low` | Blocker side, along the ice |
| `five_hole` | Between the pads / through the legs |
| `body_other` | Off the goalie / through the body / squeaker — beat them but no clean corner |
| `unknown` | **First-class value.** Net zone not determinable from the available footage. |

`unknown` is **not** an error or a null — it is an honest, expected outcome of broadcast/single-cam footage and must be counted, displayed, and reported as its own category.

### Exact JSON schema addition

Add two keys to the per-clip `metrics` object (the object documented around lines 887–936 of `metrics_v14.1.txt`, alongside `goals`, `observed_goalie_side`, `goal_criteria`):

```jsonc
"beaten_location": <
    "glove_high" | "glove_low" | "blocker_high" | "blocker_low" |
    "five_hole" | "body_other" | "unknown" | "not_applicable"
    // "not_applicable" when goals == 0 (no goal in this clip).
    // "unknown" when goals >= 1 but the net entry point is not visible.
    // For clips with multiple goals, report the zone of the MOST RECENT /
    // clearest goal and note the rest in beaten_location_notes.
>,
"beaten_location_confidence": <
    float 0.0–1.0 — confidence in the beaten_location call.
    0.0 when beaten_location is "not_applicable" or "unknown".
>,
"beaten_location_notes": <
    short string — concrete visual justification when a zone is named
    (e.g. "puck visibly enters top-left of net over the glove at 00:14"),
    or why it is "unknown" (e.g. "camera behind net, entry occluded").
    Empty string "" when "not_applicable". Generic phrasing is invalid.
>
```

This mirrors the existing pattern in the prompt: a closed enum + a confidence score + a `*_notes` string demanding a concrete visual fact (same discipline as `goal_criteria.confirming_detail` / `decision_notes`).

### Prompt-instruction sketch (add to `metrics_v14.1.txt`, in the goal-handling section)

> **Net-zone classification (goal clips only).**
> Run this step ONLY for clips where you have determined `goals >= 1`. If `goals == 0`, set `beaten_location` to `"not_applicable"`, `beaten_location_confidence` to `0.0`, and `beaten_location_notes` to `""`, and skip the rest of this step.
>
> For each goal, determine where the puck entered the net **from the shooter's front-on view of the {goalie_color} goalie**. The goalie's **glove** and **blocker** define left/right — DO NOT use screen-left/right. If you cannot reliably tell which hand is which, prefer `"unknown"` over guessing.
>
> Classify the entry into exactly one of: `glove_high`, `glove_low`, `blocker_high`, `blocker_low`, `five_hole`, `body_other`, or `"unknown"`.
> - `five_hole`: puck visibly passes between the pads/legs.
> - `body_other`: the goalie is beaten but the puck deflects off them or trickles in with no clean corner.
> - `"unknown"`: the net-entry moment is occluded, off-frame, too far, or the camera angle (e.g. behind the net, end-zone wide) does not let you see the puck cross the line into a specific zone. **Choosing `"unknown"` is correct and expected; a wrong specific zone is worse than `"unknown"`.**
>
> Set `beaten_location_confidence` to reflect how clearly you saw the entry (1.0 = puck unmistakably in a named corner; ≤0.5 if inferred). Put the concrete visual fact (or the occlusion reason) in `beaten_location_notes`. If the clip has multiple goals, report the clearest one and mention the others in the notes.

Honesty note baked into the prompt: broadcast/single-camera angles frequently occlude the net entry, so a substantial share of goals will and should be `"unknown"`. The model is explicitly told `"unknown"` beats a guess.

### Backfill plan for existing games

The new field will not appear retroactively. To populate it for already-processed games:

1. **Re-run `metrics_seg` only**, not the whole pipeline. `cv_seg` segment outputs are unchanged, so re-feed existing clip segments through the updated metrics prompt and republish `<vID>.json`. (Detection repo has **no auto-deploy**.)
2. **Scope the backfill to goal clips** — only clips with `clipHasGoal == true` need the new call, keeping cost/latency down.
3. Run a **batch reprocess** over the catalog of published `<vID>.json` artifacts, oldest-season-first, validating that every goal clip now has a `beaten_location` (even if `"unknown"`).
4. **Graceful frontend fallback:** the UI must treat a missing `beaten_location` on a goal clip as `"unknown"` so un-backfilled games never crash the grid (see §8).

### Redeploy note

After merging the prompt change, **redeploy the worker**: `bash deploy/deploy.sh`. (No gateway redeploy required; it passes JSON through.)

---

## Computation / logic

All computation is client-side over the clips array for the selected scope (single game or full season), gated on `clipHasGoal == true`.

1. **Per-zone goal counts.** For each goal clip, read `metrics.beaten_location` (fallback `"unknown"`). Accumulate `metrics.goals` into that zone's bucket. Result: a 7-bucket tally `{glove_high, glove_low, blocker_high, blocker_low, five_hole, body_other, unknown}`.
2. **Percentages.** Each zone's % = zone count ÷ total goals. **Two denominators, both shown:** `% of all goals` (includes unknown) and `% of *located* goals` (excludes unknown). The headline ("62% glove-high") should use **% of located goals** but always display the unknown rate next to it so the number is honest.
3. **Dominant-leak callout.** The largest *located* zone, with its count and %, becomes the page's aha banner. Suppress the banner when located-goal sample is below a threshold (see §8).
4. **Zone → clips index.** Keep `clipID` lists per zone so clicking a zone opens those goals in `ClipModal`.
5. **Zone ↔ positioning correlation (Phase 3).** For each zone, summarize the distribution of `goalie_positioning` ranks among the goals in that zone (e.g. share of `glove_high` goals where `depth_rank == "Deep"`). Surface only when a zone has enough located goals and the pattern is lopsided, phrased qualitatively (ranks are qualitative): *"glove_high goals cluster when your depth is rated 'Deep'."* Weight or annotate by `goalie_position_confidence_score`; never present as a hard statistic.
6. **Handling `unknown`.** Never folded into a real zone, never silently dropped. It is a labeled cell/legend entry, always reported, and excluded from the dominant-leak math.

---

## Frontend

### New component: `BeatenMap.tsx` (`goalie-analytics-pro-ui/components/`)

A NET-FACE 6-zone grid heatmap. Do **not** reuse or extend `ShotHeatmap.tsx` (top-down ice). New `types.ts` additions: `BeatenLocation` union + the per-clip metrics fields.

- **Props:** `clips: Clip[]` (already filtered to the selected game/season scope).
- **Render:** a goal-frame rectangle (posts + crossbar) divided into the 6 zones. Each zone is color-graded by goal count (light → deep red) and labeled with **count and % of located goals**. A small `body_other` band and a separate **`unknown` chip** sit outside the corner grid (unknown is not a net position).
- **Headline banner:** "X% of goals beat you **glove-high**" + the unknown-rate caveat.
- **Interaction:** click a zone → open `ClipModal` filtered to that zone's `clipID`s; each clip shows its `actionable_coaching_feedback` and jumps via `VideoPlayer`.
- **Phase 3 insight strip:** under the grid, the zone↔positioning correlation sentence(s).

### Placement

In the game/season analytics view, **next to `ShotHeatmap`** (paired "where shots come from" / "where goals go in"). Also a compact season-trend variant viable inside `PerformanceHistory` and as a tile in `StatsDashboard`. `MechanicsRadarChart` and `StatsDashboard` are untouched.

### ASCII mockup

```
┌─────────────────────────────────────────────────────────┐
│  BEATEN MAP — Where goals beat you      [ Season 25/26 ▾ ]│
│  ───────────────────────────────────────────────────────│
│  62% of located goals beat you GLOVE-HIGH                 │
│  (located goals: 13 of 18 · 5 unknown / 28%)             │
│                                                           │
│        G L O V E              B L O C K E R               │
│     ┌──────────────┬──────────────┐                      │
│ HIGH│  GLOVE_HIGH  │ BLOCKER_HIGH │   ← darker = more     │
│     │   ███████    │     ░░       │                       │
│     │   8 · 62%    │   1 · 8%     │                       │
│     ├──────┬───────┴───────┬──────┤                       │
│ LOW │GLOVE │  FIVE_HOLE    │BLOCK │                       │
│     │_LOW  │    ▒▒▒        │_LOW  │                       │
│     │ ░    │   2 · 15%     │ ▒    │                       │
│     │1 · 8%│               │1 · 8%│                       │
│     └──────┴───────────────┴──────┘                      │
│        [ body/other: 0 ]     [ unknown: 5 ]              │
│                                                           │
│  ▶ Insight: glove_high goals cluster when your depth     │
│    is rated "Deep" (8 of 8). Click a zone to watch.      │
└─────────────────────────────────────────────────────────┘
```

---

## Edge cases & limitations

- **Camera angle is the hard limit.** Broadcast/end-zone-wide/behind-net footage frequently occludes net entry. Expect a meaningful `unknown` rate; honesty about this is a feature, not a bug.
- **`unknown` is first-class.** Always shown and reported; never imputed into a real zone, never dropped.
- **Glove/blocker ≠ screen side.** The model classifies relative to the goalie's hands; if it cannot tell handedness, it must return `"unknown"`. Don't conflate with `observed_goalie_side` (screen-side).
- **Small samples.** A goalie may have very few located goals. Suppress the dominant-leak banner and Phase-3 correlations below a minimum located-goal count (proposed: **≥6 located goals**); show raw counts instead of bold percentages.
- **Multi-goal clips.** Only the clearest goal gets a precise zone; others noted in `beaten_location_notes`. Slight undercount in busy clips is acceptable and disclosed.
- **Un-backfilled games.** Missing `beaten_location` on a goal clip → render as `"unknown"`, never crash.
- **Confidence-aware shading (optional):** optionally de-emphasize low-`beaten_location_confidence` calls in the heat shading.

---

## Phasing & effort

- **Phase 1 — `metrics_seg` field + backfill (M).** Add `beaten_location` / `beaten_location_confidence` / `beaten_location_notes` to the prompt + output schema; validate enum incl. `unknown`/`not_applicable`; goal-clip-only scope; `bash deploy/deploy.sh`; batch backfill existing games' goal clips.
- **Phase 2 — UI grid (S).** `BeatenMap.tsx` net-face 6-zone heatmap with counts/%, unknown chip, dominant-leak banner, click-to-`ClipModal`; place beside `ShotHeatmap`. Add `types.ts` + `services/api.ts` typing.
- **Phase 3 — zone ↔ positioning correlation (M).** Aggregate `goalie_positioning` ranks per zone; qualitative insight strip; confidence weighting; small-sample guards.

---

## Success metrics

- **Coverage:** ≥95% of goal clips carry a non-missing `beaten_location` (any value, including `unknown`) after backfill.
- **Acceptable `unknown` rate:** treat **≤40% unknown across located footage** as the launch bar; **≤25%** as the target. Above 40%, surface a "limited angle" disclaimer and de-emphasize the headline. (We expect unknown to be non-trivial — this is honest, not a failure.)
- **Plausibility / accuracy:** on a hand-labeled sample of clearly-visible goals, ≥80% zone agreement with human labels.
- **Engagement:** % of users who click a zone to watch its clips; coach-reported usefulness of the Phase-3 insight.

---

## Open questions

1. **Minimum located-goal threshold** for showing the dominant-leak banner and correlations — 6? configurable?
2. **Headline denominator** — confirm "% of located goals" (excluding unknown) as the headline with unknown shown alongside.
3. **Multi-goal clips** — accept clearest-goal-only, or extend schema to a per-goal list of zones later?
4. **Confidence threshold** — should low-confidence calls be visually de-weighted or counted equally?
5. **Backfill cost/latency** — is re-running `metrics_seg` on goal clips only acceptable across the full catalog, and in what batch order?
6. **Phase-3 framing** — how strongly to phrase correlations given ranks are qualitative and `goalie_position_confidence_score` varies (advisory copy vs. a metric)?
7. **Handedness** — should the goalie's catching hand (L/R) be a profile setting to disambiguate glove/blocker when the model is unsure?
