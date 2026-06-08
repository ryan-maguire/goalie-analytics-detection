# Coach Review Queue + Drill Assignment
*Triage the AI's per-clip calls in one fast lane — Confirm, Correct, and assign a fixing drill to the goalie's training — across a game or a whole season.*

## Summary & problem
Goalie Analytics Pro already collects human-in-the-loop feedback (`FeedbackModal` → `POST /api/submit-feedback`), but it does so **one clip at a time, reactively**, when a user happens to open a `ClipModal`. A coach reviewing an athlete has no surface that says *"here are the 14 calls the AI was least sure about — review these first."* Low-confidence calls are buried among dozens of high-confidence ones, so they rarely get verified, and the feedback we collect never closes the loop back into the athlete's training.

**Coach Review Queue** is a coach-primary view that turns the existing feedback pipe into a real workflow. It pulls every clip in a game (or season), **orders them by "needs review"** — lowest `goalie_position_confidence_score` / `coaching_confidence_score` and clips carrying `analysis_confidence_caveats` float to the top — and lets the coach blast through them:

- **Confirm** → emits an `"Accurate Analytic"` feedback record (reuses `submitPositiveFeedback`), which spec 11 reads to render a **coach-verified badge**.
- **Correct** → opens the existing `FeedbackModal` (`"Correction Analytic"` with field-level `suggested_value`), no new payload shape.
- **Assign a drill** → maps the clip's weak dimension to a drill via the existing `utils/drillLibrary.ts getDrill`, and persists the assignment so it flows into the goalie's `MyTraining`.
- **Bulk** Confirm / Assign across a filtered set (e.g. "confirm all clips with confidence ≥ 4", "assign Squareness drill to all Poor-squareness goal clips").

This is the connective tissue between three existing-but-disconnected features: the feedback loop, `drillLibrary`, and `MyTraining`. It consumes **spec 12 (confidence triage)** for the ordering primitive and produces **spec 11 (coach-verified state)**.

## Target users
- **Goalie coaches (primary)** — review an athlete's game in minutes; verify or fix the AI; prescribe the corrective drill. The whole view is built around their throughput.
- **Goalies (13+)** — receive coach-assigned drills in `MyTraining` with the originating clip and the coach's note attached ("Coach flagged this, work on it").
- **Goalie families** — see that a credentialed human reviewed the analysis (verified badges via spec 11), increasing trust in the numbers.

## Data inputs (exact production fields)
Clips from existing analysis JSON at `analyze_video/04-final_video/<vID>.json` (array of items; `item.type ∈ {summary, clips, windows}`; UI treats `windows == clips`), fetched via `fetchGameData(vID)`. Game list from `customerID/<custID>.json` via `fetchUserGames(custID)`.

Per clip (`clips/windows.response[]`) — all consumed read-only for triage:
- **Identity / linking:** `clipID`, `clip_start_time`, `clip_end_time` (numbers, seconds)
- **Calls:** `clipShot`, `clipSave`, `clipHasGoal` (bool), `clipShotCount`, `clipSaveCount`
- **Triage signals (the ordering key):**
  - `goalie_positioning.goalie_position_confidence_score` ∈ 1..5
  - `coaching_feedback.coaching_confidence_score` ∈ 1..5
  - `analysis_confidence_caveats` (string[]) — presence + contents
- **Weak-dimension inputs (drive `getDrill` mapping):**
  - `goalie_positioning.depth_rank`, `cover_angle_rank`, `squareness_rank`
  - `coaching_feedback.rebound_control_rank`, `actionable_coaching_feedback`
- **Context (display only):** `technical_reasoning`, `metrics.beaten_location` (+ `beaten_location_confidence`)

From `customerID/<custID>.json` per game: `vID`, `eventName`, `event_date`, `eventSeason`, `goalie_name`.

**Caveat handling:** `threat_type` is unreliable and ranks are qualitative — the queue never auto-corrects, it only *orders* and *surfaces*. A human always makes the call.

**NEW fields / documents (source):**
- `assignments/<custID>.json` (GCS, written via `save-json`) — drill-assignment records the coach creates. Schema in *Pipeline / backend*. Source: this feature.
- Coach-verified state is **not** a new field — it is derived by spec 11 from the existing `<vID>_feedback` doc (`feedback_type === "Accurate Analytic"`). No duplication here.

## Pipeline / backend
**No detection-pipeline change.** Entirely frontend + existing gateway, plus one new `save-json` document.

1. **Confirm / Correct — reuse existing feedback pipe (no new endpoint, no new payload):**
   - Confirm calls `submitPositiveFeedback(clip, vID)` → `submitFeedbackPayload(..., 'Accurate Analytic', '')`.
   - Correct opens `FeedbackModal` → `submitFeedbackPayload(..., 'Correction Analytic', note, corrections[])`.
   - Both append to `<vID>_feedback` (read back via `fetchClipFeedback(vID)`). Spec 11 renders verified state off this; this view also reads it to dim already-reviewed clips.

2. **Drill assignments — one new `save-json` document** (gateway `POST /api/save-json` already exists and is used by `saveSegmentGT`):
   - **Doc id:** `assignments/<custID>` (resolves to `gs://…/assignments/<custID>.json`).
   - **Shape:** an array of assignment records (read-modify-write the whole array, same pattern as `saveSegmentGT`):
     ```jsonc
     {
       "assignmentID": "asg_<uuid>",     // client-generated
       "custID": "CUST000048",
       "goalie_name": "…",               // from game metadata, for MyTraining filtering
       "vID": "…",
       "clipID": "…",
       "dimension": "squareness",         // LeakDimension: depth|angle|squareness|rebound
       "drill_name": "Post-to-Post Squares",   // from getDrill(...).name (snapshot)
       "drill_cue": "Lead with the chest, not the pads.",
       "coach_note": "Re-square earlier on glove-side walkouts.",
       "clip_start_time": 123.4,          // deep-link back into the clip
       "clip_end_time": 131.0,
       "status": "assigned",              // assigned|completed (goalie can mark done later)
       "assigned_by": "coach",
       "assigned_at": "2026-06-07T…Z"
     }
     ```
   - **New `services/api.ts` functions** (mirror `fetchSegmentGT`/`saveSegmentGT`): `fetchAssignments(custID)` and `saveAssignments(custID, records[])`. Bulk assign = build N records, single `saveAssignments` write.

3. **`MyTraining` read path:** add a "Coach-Assigned" section above the existing AI-curated `DRILLS_DB` block. On mount it calls `fetchAssignments(selectedClient)`, filters to the relevant goalie, and renders each assignment as a card (drill name, cue, coach note, "Watch the clip" deep-link, "Mark done" → flips `status` and re-saves). AI auto-recommendations remain as the fallback when there are no coach assignments.

## Computation / logic
**1. Needs-review score (consumes spec 12).** Spec 12 owns the canonical confidence-triage primitive; this view imports it. If spec 12 ships a helper, use it; otherwise the local fallback ordering is:

```
posConf   = clip.goalie_positioning.goalie_position_confidence_score ?? 3
coachConf = clip.coaching_feedback.coaching_confidence_score ?? 3
caveatN   = (clip.analysis_confidence_caveats ?? []).length

reviewPriority =
    (caveatN > 0 ? 100 : 0)          // any caveat → top tier
  + (5 - min(posConf, coachConf)) * 10   // lower of the two confidences dominates
  + (5 - ((posConf + coachConf) / 2))    // tiebreak on average
```
Sort **descending** by `reviewPriority`. Result: clips with caveats first, then lowest-confidence calls, then the rest. Goal clips (`clipHasGoal`) get a secondary highlight (coaches care most about goals against) but do **not** override the confidence ordering.

**2. Already-reviewed dimming.** Join clips against `fetchClipFeedback(vID)` by `clipID`. A clip with an existing `Accurate`/`Correction` record renders dimmed with a small check and drops below unreviewed clips (stable sort: unreviewed first, then by `reviewPriority`).

**3. Weak-dimension → drill mapping (reuse, do not reinvent).** Pick the clip's weakest dimension to seed the suggested drill, then call `getDrill(dimension, opts)`:
```
ranks = {
  depth:      depth_rank,        angle:    cover_angle_rank,
  squareness: squareness_rank,   rebound:  rebound_control_rank
}
// "weak" = rank contains 'poor' (or rebound contains 'slot'/'goal') — same
// heuristic MyTraining already uses for weakness counting.
weakDim = first dimension whose rank reads weak, preferring the one named in
          actionable_coaching_feedback; fall back to lowest-confidence area.
drill   = getDrill(weakDim, { rank: ranks[weakDim],
                              sideSpecific: /glove|blocker|left|right/.test(note) })
```
The coach can override the suggested dimension in the assign popover (4-way toggle). `getDrill` already special-cases `rebound` when the rank mentions a goal, and `squareness` when side-specific — reuse that behavior verbatim.

**4. Bulk operations.** Apply the current `FilterBar`/queue filter to get the visible set, then:
- *Bulk Confirm* → `Promise.all` of `submitPositiveFeedback` over the set (skip already-corrected clips).
- *Bulk Assign* → build one assignment record per clip using each clip's own `weakDim`, then a **single** `saveAssignments` write.
Both show a per-row progress state and a final toast ("12 confirmed, 0 failed").

## Frontend
**New top-level view `coach_review_queue`** added to the `currentView` union and `handleViewChange` signature in `App.tsx` (line 45 / 75), wired into `renderView()` (line 313) and `Header`. New component `components/CoachReviewQueue.tsx`. New `services/api.ts`: `fetchAssignments` / `saveAssignments`. New `MyTraining.tsx` "Coach-Assigned" section. No changes to `FeedbackModal` (reused as-is), `ClipModal`, or `VideoPlayer`.

**Composition / reuse:**
- Game + season selector (reuse `fetchUserGames`; "Whole season" = concat all games' clips).
- `FilterBar` for outcome/confidence filtering of the queue.
- Per-row inline `VideoPlayer`/thumbnail deep-link to `clip_start_time` (reuse existing player open path).
- **Correct** button → opens existing `FeedbackModal` with `existingFeedback` prefilled if any.
- **Assign** button → small popover with the 4-way dimension toggle (pre-set to `weakDim`), the resolved `getDrill` card (name + cue), a coach-note textarea, and Save → `saveAssignments`.

**UX notes:** queue is keyboard-driven for throughput — `C` confirm, `X` correct, `A` assign, `J/K` next/prev row. Confirmed rows animate out (or dim) so the unreviewed count is always the headline. A sticky header shows `N needs-review · M reviewed · K assigned`.

```
+---------------------------------------------------------------------------+
| Coach Review Queue            Game: [U16 vs Storm ▼]  ◻ Whole season       |
| 14 needs review · 6 reviewed · 3 drills assigned     [Bulk Confirm] [⚙]    |
+---------------------------------------------------------------------------+
| Filter: [Goals] [Low conf ≤2] [Has caveats]            sort: needs-review |
+---------------------------------------------------------------------------+
| ⚠  03:412  GOAL  squareness: Poor   conf P1/C2   "angle unclear, occluded"|
|    [▶ clip] technical: beaten glove-high on cross-crease …                 |
|    suggested drill ▸ Post-to-Post Squares  ("Lead with the chest…")       |
|        [ Confirm ✓ ]   [ Correct ✎ ]   [ Assign drill → MyTraining ＋ ]   |
+---------------------------------------------------------------------------+
| ⚠  08:117  SAVE  depth: Poor        conf P2/C3   "depth est. low-confidence"|
|    suggested drill ▸ Depth Trigger / Edge Pushes                          |
|        [ Confirm ✓ ]   [ Correct ✎ ]   [ Assign drill → MyTraining ＋ ]   |
+---------------------------------------------------------------------------+
| ✓  11:204  SAVE  (reviewed · Accurate)                       dimmed       |
+---------------------------------------------------------------------------+
```

## Edge cases
- **Missing confidence scores** — default to 3 (neutral) so the clip neither floats to top nor sinks; do not treat absent as 0.
- **Empty caveats array vs missing** — both = "no caveat"; only a non-empty array raises priority.
- **No weak dimension** (all ranks Good/Excellent, high confidence) — Assign defaults the toggle to the lowest-confidence dimension; coach must pick before Save.
- **Already-corrected clip** — Confirm is disabled (would conflict with the correction); show "already corrected" and offer Re-correct.
- **Double assignment** — if an assignment for the same `clipID`+`dimension` exists, prompt "replace existing?" rather than duplicating (`saveAssignments` is read-modify-write, so we can de-dupe).
- **`save-json` failure mid-bulk** — assignments are a single array write (atomic-ish); confirms are N independent posts, so report partial success per row.
- **Whole-season mode** — can be hundreds of clips; cap initial render, paginate the queue, and warn before a season-wide bulk action.
- **`threat_type` / qualitative ranks** — never used to auto-decide; display-only. The coach's Confirm/Correct is the source of truth.
- **Goalie identity** — assignments are stamped with `goalie_name` from game metadata so `MyTraining` shows the right athlete's drills in multi-goalie customers.

## Phasing & effort
- **P0 — Queue + Confirm/Correct (M).** New view, needs-review ordering (local fallback if spec 12 not yet merged), reuse `submitPositiveFeedback` + `FeedbackModal`, reviewed-dimming. Ships value with zero new persistence. *Depends on: nothing hard; soft dep on spec 12.*
- **P1 — Drill assignment (M).** `assignments/<custID>.json` doc, `fetch/saveAssignments`, assign popover over `getDrill`, `MyTraining` "Coach-Assigned" section + "Mark done". *Depends on: P0.*
- **P2 — Bulk operations (S).** Bulk Confirm and Bulk Assign over the filtered set with progress + toast. *Depends on: P0, P1.*
- **P3 — Verified integration (S).** Render spec 11's coach-verified badge inline and on `ClipCard`; share the `Accurate Analytic` read. *Depends on: spec 11.*

Hard cross-deps: **spec 12** (confidence-triage ordering primitive — P0 reuses it, falls back if absent) and **spec 11** (coach-verified badge — P3 consumes; P0's Confirm produces the underlying state).

## Success metrics
- **Review throughput** — clips reviewed per coach-minute (target: a full game's needs-review tier cleared in < 5 min).
- **Low-confidence coverage** — % of clips with conf ≤ 2 or non-empty caveats that get a human Confirm/Correct (target ≥ 80% within a week of a game publishing).
- **Correction yield** — share of reviews that are Corrections vs Confirms (signals real model-error rate feeding the training loop).
- **Assignment → training conversion** — # drills assigned that appear in `MyTraining`, and % marked done by the goalie.
- **Trust** — % of games carrying ≥1 coach-verified badge (spec 11), as a proxy for family/coach confidence.

## Open questions
- Does spec 12 expose a reusable `reviewPriority`/sort helper, or should this view own the fallback long-term? (Prefer importing spec 12's.)
- Assignment identity: per-clip de-dupe key should be `clipID` alone or `clipID + dimension`? (Spec assumes the latter — a clip can have two weak dimensions.)
- Should the goalie be able to dismiss/decline a coach-assigned drill, or only mark it done?
- Multi-goalie customers: do we need a goalie picker at the top of the queue, or is per-game `goalie_name` sufficient for filtering?
- Should Bulk Confirm be gated behind a confidence floor (e.g. only allow bulk-confirming conf ≥ 4) to prevent rubber-stamping low-confidence calls?
- Auth/attribution: `assigned_by` is hardcoded `"coach"` for now — do we have a coach identity to stamp once accounts exist?
