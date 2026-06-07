# Leak Finder
*Find the systematic weakness that is actually costing you goals — across your whole season, in plain language, with the clips and a drill to fix it.*

## Summary & problem
Every existing surface in Goalie Analytics Pro is **descriptive** and mostly **per-snapshot**: StatsDashboard totals, MechanicsRadarChart shows an average shape, ShotHeatmap plots location, MyTraining shows current focus, PerformanceHistory trends a single number over time. None of them answer the question a goalie, parent, or coach actually asks: *"What is the one thing that keeps beating me, and what do I do about it?"*

The dataset's real, hard-to-replicate advantage is that **each clip ties mechanics ranks (depth / angle / squareness / rebound control) to a binary outcome (`clipHasGoal` vs `clipSave`) on a specific side**, and we have *many* clips across *many* games for the same goalie. That makes it possible to do **cross-game correlational leak detection**: hold the dataset up and ask "when squareness is Poor on plays to the glove side, how much more often is it a goal vs. your baseline?"

**Leak Finder** aggregates every clip in a customer's season, correlates each weakness dimension with the goal/save outcome, ranks the resulting **leaks** by how much they elevate goals-against above the goalie's own base rate (relative risk), filters them through statistical guardrails (minimum sample size + confidence weighting), and presents the top leaks as ranked, plain-language cards. Each card links to the supporting clips (deep-link into the existing `ClipModal`) and maps to a **recommended drill** from a small drill library defined here.

This is **prescriptive**, not descriptive — and it is **cross-game correlational**, which distinguishes it from MyTraining/MechanicsRadarChart (per-snapshot, descriptive averages).

## Target users
- **Goalies (13+)** — "Stop guessing what to work on. Here is the pattern that beats you and the drill for it."
- **Goalie families** — non-technical, plain-language "leaks" with evidence they can trust and a clear next action.
- **Goalie coaches** — fast triage across an athlete's season; jump straight to the clips that prove the pattern; assign the mapped drill.

## Data inputs (exact production fields)
All from existing analysis JSON at `gs://goalie_video_bucket/analyze_video/04-final_video/<vID>.json` (an array of items; `item.type ∈ {summary, clips, windows}`, UI treats `windows == clips`). Game list from `gs://…/customerID/<custID>.json`.

Per clip (`clips/windows.response[]`):
- **Outcome:** `clipHasGoal` (bool), `clipSave` (bool), `clipSaveCount`, `clipShot`, `clipShotCount`
- **Side:** `threat_goalie_side` and `metrics.observed_goalie_side` — lowercase strings `"left" | "right" | "unknown"` (sentinel `"unknown"` when unresolved)
- **Positioning** `goalie_positioning`:
  - `depth_rank` ∈ `["Aggressive (A)", "Base (B)", "Conservative (C)", "Defensive (D)"]`
  - `cover_angle_rank` ∈ `["Excellent", "Good", "Average", "Poor"]`
  - `squareness_rank` ∈ `["Excellent", "Good", "Average", "Poor"]`
  - `goalie_position_confidence_score` ∈ 1..5 (int)
- **Coaching** `coaching_feedback`:
  - `rebound_control_rank` ∈ `["Killed (Elite)", "Steered to Low-Danger Zone", "Uncontrolled into High-Danger Zone", "Goal Allowed", "Not Applicable"]`
  - `coaching_confidence_score` ∈ 1..5 (int)
  - `actionable_coaching_feedback` (string, shown as supporting context)
- **Identity / linking:** `clipID`, `clip_start_time`, `clip_end_time`, `segment_start`, `segment_end`, `segmentHasThreat`
- `analysis_confidence_caveats` (string[]) — surfaced as a "why we're unsure" note.

From `customerID/<custID>.json` per game: `vID`, `eventName`, `event_date`, `goalie_name`, `eventSeason`, `analyticsStatus` (only aggregate games whose analysis is published).

From `summary.response`: `goalie_jersey_color` (display), `coaches_overall_rating` (context).

**NEW fields:** none required for v1. Leak Finder is **pure frontend over existing 04-final_video JSON**. (See Pipeline / backend for an optional later enrichment.)

**Data caveats honored here:** `threat_type` is NOT reliably populated — **do not use it** (a later "danger tier" enrichment from spec 02 will replace this need). Ranks are **qualitative strings** and require a rank→ordinal map (below). There is **no precise shot xy** — Leak Finder is dimension-based, not location-based. Confidence scores exist and are used for triage/weighting, not as ground truth.

## Pipeline / backend changes
**v1: pure frontend over existing data. No pipeline, prompt, or gateway changes.** The UI already fetches `customerID/<custID>` (game list) and `analyze_video/04-final_video/<vID>` (per game) via the gateway `get-json?id=<path>` endpoint (`services/api.ts`). Leak Finder fans out over the game list and reuses those fetches (ideally a cached batch).

**Optional later enrichment (after spec 02 "danger tier" lands):**
- `metrics_seg` (`01_detect_segment_metrics.py`) or `feedback_seg` adds a per-clip `danger_tier` ∈ `{"low","medium","high"}` so leaks can be weighted by chance quality (a Poor-squareness goal on a high-danger chance is less of a "leak" than one on a low-danger chance).
  - Prompt-instruction sketch (feedback_seg `gemini.py`): *"Add field `danger_tier`: classify the scoring chance the goalie faced as 'low', 'medium', or 'high' based on shot proximity, lateral movement before the shot, and traffic. This rates the difficulty of the chance, independent of the outcome."*
- No new gateway endpoint is needed even then (still read via `get-json`).

## Computation / logic
### Rank → ordinal map (0 = best, higher = worse leak risk)
```
DEPTH      = { "Base (B)":0, "Aggressive (A)":1, "Conservative (C)":2, "Defensive (D)":3 }   // out-of-net (D) is highest risk
ANGLE      = { "Excellent":0, "Good":1, "Average":2, "Poor":3 }
SQUARE     = { "Excellent":0, "Good":1, "Average":2, "Poor":3 }
REBOUND    = { "Killed (Elite)":0, "Steered to Low-Danger Zone":1,
               "Uncontrolled into High-Danger Zone":2, "Goal Allowed":3, "Not Applicable":null }
```
A clip is in the **"weak" bucket** for a dimension when its ordinal ≥ the dimension's weak threshold:
- angle / squareness: weak when rank ∈ {`Average`, `Poor`} (ordinal ≥ 2)
- depth: weak when rank ∈ {`Conservative (C)`, `Defensive (D)`} (ordinal ≥ 2)
- rebound: weak when rank ∈ {`Uncontrolled into High-Danger Zone`, `Goal Allowed`} (ordinal ≥ 2); `Not Applicable` excluded from rebound leaks entirely.

### Side normalization
`side = clip.threat_goalie_side || clip.metrics.observed_goalie_side`. Map `"left"→"Glove side (left)"`, `"right"→"Blocker side (right)"` for display **only if** the goalie's catch hand is known; otherwise show neutral "left/right". `"unknown"` clips are excluded from *side-specific* leaks but still count in dimension-only leaks. (Catch hand is not in the data model — see Open questions; default to neutral left/right labels.)

### Candidate leak = (dimension, weak-bucket) optionally × side
Enumerate candidates: each of {depth, angle, squareness, rebound} weak bucket, evaluated (a) overall and (b) split by side. For each candidate compute, over all clips in scope (`clipHasGoal === true || clipSave === true`; ignore clips with neither outcome):

```
N_total      = clips in scope (with valid outcome)
goals_total  = count(clipHasGoal)
base_rate    = goals_total / N_total                      // this goalie's own goals-against rate

n_leak       = clips matching candidate (weak bucket [, side])
goals_leak   = count(clipHasGoal within those)
leak_rate    = goals_leak / n_leak

relative_risk (RR) = leak_rate / base_rate                // how much MORE often you get scored on
excess_goals       = goals_leak - n_leak * base_rate      // ≈ "extra" goals attributable to this leak
```

### Confidence weighting
Weight each clip's contribution by its relevant `*_confidence_score / 5` (use `goalie_position_confidence_score` for depth/angle/squareness candidates; `coaching_confidence_score` for rebound). Compute a **confidence-weighted leak_rate** in parallel and a candidate `avg_conf = mean(score)/5`. Low avg_conf demotes a leak (see scoring) and triggers a "lower confidence" badge.

### Statistical guardrails (what counts as a real leak)
A candidate is promoted to a **leak** only if ALL hold:
1. **Min sample:** `n_leak ≥ 5` AND `goals_leak ≥ 3` (avoids "3 of 3" noise). Side-split candidates require `n_leak ≥ 6` (sides halve the sample).
2. **Meaningful elevation:** `RR ≥ 1.5` (≥50% more goals than the goalie's own base rate). Hard-floor `base_rate ≥ 0.05` so a near-perfect season doesn't produce huge RR from one goal.
3. **Confidence floor:** `avg_conf ≥ 0.5` (mean rank confidence ≥ 2.5/5). Below this it may still show but flagged "low confidence, treat as a hint."

### Leak score (ranking) and de-duplication
```
leak_score = excess_goals * log2(RR) * avg_conf
```
Rank descending; show **top 3–5**. De-dup overlapping candidates: if a side-specific leak and its overall counterpart describe largely the same clips (Jaccard of clipID sets > 0.6), keep the higher `leak_score` only. Prefer the more specific (side) one when scores are close (within 10%), since it is more actionable.

### Plain-language rendering (template)
> "You were scored on **{RR}× more often** when **{dimension phrase}**{ side phrase}. **{goals_leak} of {n_leak}** such clips this season ended in a goal (your overall rate is {base_rate%})."

Example: *"You were scored on **3.4× more often** when your **squareness was Average or Poor on plays to your glove side**. **6 of 9** such clips this season ended in a goal (your overall rate is 18%)."*

Dimension phrases: depth→"you were too deep in your net (Conservative/Defensive)"; angle→"your angle was off the shooting lane (Average/Poor)"; squareness→"you weren't square to the puck (Average/Poor)"; rebound→"you gave up a dangerous rebound".

### Cross-game aggregation & scope
Default scope = current `eventSeason` (from game list). Aggregate the pooled clip set across all games in scope (pooled TP-style counting, consistent with the project's pooled-F1 convention — never average per-game rates). A per-game mini-breakdown is shown inside the leak card ("appears in 4 of 6 games") to prove it is systematic, not one bad night.

### Pseudocode
```ts
const clips = games.flatMap(g => clipsOf(g).map(c => ({...c, vID:g.vID, eventName:g.eventName, date:g.event_date})))
                   .filter(c => c.clipHasGoal || c.clipSave);
const base = mean(clips, c => c.clipHasGoal ? 1 : 0);          // base_rate, floored at 0.05
const candidates = buildCandidates(clips);                     // {dim, weakBucket, side?} × matching clipIDs
const leaks = candidates
  .map(cand => score(cand, clips, base))                       // RR, excess_goals, avg_conf, leak_score
  .filter(passesGuardrails)                                    // §guardrails
  .sort((a,b) => b.leak_score - a.leak_score);
const top = dedupe(leaks).slice(0, 5).map(l => ({ ...l, drill: DRILL_LIBRARY[l.dim][l.severityBucket] }));
```

### Leaks → drills mapping (drill library, keyed by weakness dimension)
| dimension | bucket condition | drill name | what it fixes | cue |
|---|---|---|---|---|
| squareness | Average/Poor | **Post-to-Post Squares** | shoulders/hips facing the puck on lateral plays | "lead with the chest, not the pads" |
| squareness | Poor + side-specific | **Glove/Blocker-side Walk-out Tracking** | re-squaring after lateral movement to one side | "turn the cage to the puck" |
| angle | Average/Poor | **Centre-Line Angle Ladder** | alignment on the puck-to-net line | "find the line, then set" |
| depth | Conservative (C)/Defensive (D) | **Depth Trigger / Edge Pushes** | challenging shooters instead of sitting deep | "meet the shot at the top of the crease" |
| depth | Aggressive (A) over-challenge | **Retreat-and-Set** | recovering depth on backdoor/lateral plays | "challenge, then recover" |
| rebound | Uncontrolled into High-Danger | **Rebound Steering (chest & pad seal)** | directing rebounds to the corners | "absorb or steer to the boards" |
| rebound | Goal Allowed (rebound-driven) | **Second-Save Recovery / Desperation** | reset speed after a rebound | "first save, reset, second save" |

`DRILL_LIBRARY` is a static frontend constant (`utils/drillLibrary.ts`); editable without a deploy of the pipeline. Each drill: `{ name, summary, cue, optionalVideoUrl? }`.

## Frontend
**New components (no duplication of existing surfaces):**
- `LeakFinder.tsx` — top-level view: scope selector (season / all-time, reuses GameSelector's season list), summary line, ranked list of `LeakCard`s.
- `LeakCard.tsx` — one leak: plain-language headline, RR badge, evidence bar, mapped drill, "View {n} clips" button.
- `utils/leakAnalysis.ts` (logic) + `utils/drillLibrary.ts` (drill table).

**Route / placement:** new tab/route `#/leaks` ("Leak Finder", icon: magnifier-on-net) in the main nav next to StatsDashboard. Also surface the **#1 leak as a teaser card** on the dashboard ("Your biggest leak this season →") linking into the full view. Reuses existing `ClipModal`/`VideoPlayer` for clip playback (pass `clipID` + `vID` to deep-link), and `ClipCard` for the clip strip inside a card.

**UX:** progressive disclosure — headline first, evidence and drill on the card, clips on demand. Badges: RR multiplier, sample (`6/9 clips`), "appears in N/M games", optional "lower confidence."

```
┌──────────────────────────────────────────────────────────────┐
│  Leak Finder            Season: 2025-26 ▼     [All-time]       │
│  We analysed 142 clips across 6 games.                         │
├──────────────────────────────────────────────────────────────┤
│  #1  ●●●  SQUARENESS — GLOVE SIDE              [ 3.4× goals ]   │
│  You were scored on 3.4× more often when you weren't square    │
│  to the puck on plays to your glove side.                      │
│   Evidence: 6 of 9 clips → goal   (your overall rate 18%)      │
│   Seen in 4 of 6 games            [ lower confidence? no ]     │
│   ► DRILL: Glove-side Walk-out Tracking  — "turn the cage      │
│            to the puck"                          [ Start ▸ ]    │
│   [ View 9 clips ▸ ]                                            │
├──────────────────────────────────────────────────────────────┤
│  #2  ●●   DEPTH — TOO DEEP                     [ 2.1× goals ]   │
│  ...                                                            │
└──────────────────────────────────────────────────────────────┘
```

## Edge cases & limitations
- **Small samples:** if no candidate clears the guardrails, show an empty state: "Not enough clips yet to find reliable patterns — keep adding games (we need ~5+ goals against in a dimension)." Never show a leak from `n < 5`.
- **Near-perfect season:** `base_rate` floor (0.05) prevents one goal from generating absurd RR; if `goals_total < 3` overall, hide the feature with a "great season — too few goals to analyse" message.
- **Missing fields:** clips missing `goalie_positioning`/`coaching_feedback` or with `rebound_control_rank == "Not Applicable"` are excluded from the relevant candidate, not the whole analysis. Clips with neither `clipHasGoal` nor `clipSave` are dropped (no outcome).
- **`unknown` side:** excluded from side-specific leaks; still counted in overall-dimension leaks. If >40% of clips are `unknown` side, suppress side-split candidates and note "side detection was unreliable this season."
- **Low confidence:** candidates with `avg_conf < 0.5` are demoted and badged; never the headline #1 unless nothing else qualifies.
- **No `threat_type` / no xy:** acknowledged in a footnote ("Leaks are based on mechanics + outcome, not shot location"); danger weighting deferred to spec 02.
- **Correlation ≠ causation:** card footnote — "These are patterns, not proof. Use the clips to confirm with your coach."

## Phasing & effort
- **MVP (S–M):** pure-frontend; dimensions = squareness + angle + depth + rebound, overall + side split; guardrails; top-3 leaks; static drill library; deep-link to ClipModal; dashboard teaser. ~M (one analytics util + two components + route).
- **Phase 2 (S):** per-game breakdown sparkline, confidence badges, "improving/worsening over season" trend per leak (compare first-half vs second-half clip sets).
- **Phase 3 (M, depends on spec 02):** danger-tier weighting so leaks reflect chance quality; richer drill content (linked drill videos); coach "assign drill" action wired to MyTraining.

## Success metrics
- **Engagement:** % of sessions that open Leak Finder; clip deep-links opened per session; drill "Start" clicks.
- **Trust:** thumbs-up/down on each leak card (reuses FeedbackModal pattern) → target ≥70% agreement.
- **Outcome (long-horizon):** for goalies who engaged with a leak's drill, does that dimension's goal-against rate fall in subsequent games vs. their own prior baseline? (the only rigorous proof the feature works).
- **Coverage:** % of active goalies with ≥1 qualifying leak (validates guardrail thresholds aren't too strict).

## Open questions
1. **Catch hand** is not in the data model — without it we can't reliably label "glove" vs "blocker," only left/right. Add `catch_hand` to customer config, or infer? (Default v1: neutral left/right.)
2. Guardrail constants (`n≥5`, `RR≥1.5`, `avg_conf≥0.5`) are first-principles guesses — tune against real customer datasets (CUST000031 / CUST000048) before launch.
3. Should scope default to current season or rolling last-N-games? (Season chosen for "systematic" framing.)
4. Is `excess_goals * log2(RR) * avg_conf` the right ranking, or should families see the **simplest** (highest RR) leak first regardless of volume? A/B candidate.
5. Should leaks combine dimensions (e.g., Poor squareness AND deep depth together)? Higher signal but smaller samples — defer until base feature validates.
6. When spec 02 danger tiers land, do we reweight historical leaks retroactively or only new clips?
