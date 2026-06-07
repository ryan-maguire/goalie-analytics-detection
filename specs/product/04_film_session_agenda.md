# Auto Film Session Agenda
*Walk into film day with zero prep: the ~5 most instructive clips, an overall session theme, and AI-drafted per-clip talking points — ready to play, telestrate, and voice over.*

## Summary & problem
Coaches today either review a full game tape (too long) or hand-curate clips in **FavoriteClips** (manual, time-consuming, and only as good as their memory). Neither produces a *teaching plan*. A good film session is not "here are the goals" — it is a deliberate mix of **teachable goals-against** (what to fix) and **elite saves** (what to reinforce), each with a concrete talking point and a session-level through-line.

**Auto Film Session** auto-curates that plan. For one game (or a week of games) it scores every analyzed clip by *instructional value*, selects a balanced ~5-clip agenda, and asks Gemini to draft one talking point per clip plus a single session theme. The coach opens the agenda, each item launches in the existing **ClipModal → VideoPlayer** (telestration + voiceover already built), and the coach can reorder / remove / add before sharing.

This is **AUTO-curation + talking points**, deliberately distinct from the manual **FavoriteClips** feature. It complements ClipModal, VideoPlayer, Timeline, and MyTraining rather than duplicating them.

## Target users
- **Coaches (primary).** Need a ready-to-run lesson plan with no prep; want to reorder and inject their own clips; want to export/share to the goalie or staff.
- **Goalies / families (secondary, read-only).** Get the same curated session as a self-review or pre-practice digest. Families see "the 5 clips that matter this week" without needing to interpret raw analytics.

## Data inputs
Source: `gs://goalie_video_bucket/analyze_video/04-final_video/<vID>.json` (the UI-facing final JSON; produced from feedback_seg output where `item.type` is `windows`, remapped to `clips` in `services/api.ts`). Per game we read the `summary` item and every clip in the `clips`/`windows` item.

**Fields used for SCORING / SELECTION** (per clip):
- `clipHasGoal` (bool), `metrics.goals` — a goal-against is high instructional value.
- `clipSaveCount` / `metrics.saves`, `metrics.shotsOnNet` — save volume / difficulty signal.
- `coaching_feedback.rebound_control_rank` — ordinal: `Killed (Elite)` (good) … `Goal Allowed` (bad). Extremes are teachable.
- `goalie_positioning.depth_rank` (`Aggressive (A)`/`Base (B)`/`Conservative (C)`/`Defensive (D)`), `cover_angle_rank` & `squareness_rank` (`Excellent`/`Good`/`Average`/`Poor`) — ordinal extremity = teachable moment.
- `coaching_feedback.coaching_confidence_score` and `goalie_positioning.goalie_position_confidence_score` (integers 1–5) — weight; never feature a low-confidence clip as a "fact."
- `clip_start_time` / `clip_end_time` — recency proxy within game; with cross-game `event_date`, a weekly recency weight.
- `analysis_confidence_caveats` (array) — non-empty → down-weight (footage limits).

**Fields used to DRAFT TALKING POINTS** (the rich free text):
- `technical_reasoning` (full four-pillar narrative with `[MM:SS]` timestamps).
- `coaching_feedback.actionable_coaching_feedback` (the corrective).
- `goalie_positioning.{depth_rank,cover_angle_rank,squareness_rank}` and `coaching_feedback.rebound_control_rank` (labels to anchor language).
- `summary.response.coaches_summary` + `coaches_overall_rating` (game context for the theme).

**NOT used:** `threat_type` / `threat_goalie_side` — `threat_type` is unreliable in production (per data caveats); we do not condition selection on it.

## Pipeline / backend changes

**Decision: do BOTH, in two phases.**

### (a) On-demand endpoint — MVP (chosen first)
A new gateway route **`POST /api/film-session`** (in `goalie-analytics-api-gateway`, or the UI's own `server.ts`, which already has Vertex access). Rationale: agendas are a *coach action*, not every video needs one; on-demand avoids reprocessing the whole pipeline and lets the coach trigger weekly rollups across arbitrary game sets. Heuristic clip selection (below) runs server-side in TS/Python; only the **top-N selected clips** are sent to Gemini for talking points — cheap, fast, no video re-analysis (text-only call).

Request:
```json
{ "customerId": "CUST000048", "vIDs": ["abc123"], "topN": 5, "persist": true }
```
Flow: fetch each `04-final_video/<vID>.json` via existing get-json → run selection → call Gemini with the selected clips' text → assemble `agenda` object → if `persist`, write via **`/api/save-json`** to `analyze_video/05-film_session/<vID>.json` (single game) or `customerID/<custID>/film_session_week_<isoWeek>.json` (rollup). Return the `agenda` object to the UI.

### (b) feedback_seg post-step — Phase 2 (scheduled / always-fresh)
Add a `generate_film_session(client, window_records, summary_data, goalie_color, opponent_color)` function in `feedback_seg/gemini.py`, called from `pipeline.py` right after `generate_summary()` (step 5). It reuses the same `genai.Client` and the already-assembled `window_records`, so it is a near-free add-on at pipeline time. It writes the `agenda` object as a **third element** of the final output array:
```json
{ "type": "film_session", "response": { ...agenda... } }
```
This rides along into `04-final_video/<vID>.json`, so the UI gets an agenda with no extra call. The on-demand endpoint (a) stays for rollups and for re-running after a coach edits.

### Gemini prompt sketch (talking points + theme)
Text-only call, `temperature=0`, `response_mime_type="application/json"`, structured `response_schema` (mirrors the existing `SUMMARY_RESPONSE_SCHEMA` pattern). Input = the N pre-selected clips only.

```
SYSTEM: You are a goaltending coach building a film-session lesson plan.
You are given N pre-selected clips, each already chosen for teaching value.
For EACH clip produce ONE talking point a coach can say out loud while the
clip plays. Ground every point in the clip's technical_reasoning and
actionable_coaching_feedback. Reuse the four-pillar vocabulary (depth,
angle, squareness, rebound control, recovery). If a clip has caveats,
hedge ("from this angle it looks like…"). Do NOT invent timestamps not
present in the source text.

Then write ONE session_theme: the single through-line connecting these
clips (e.g. "Depth management on lateral plays"), and order_rationale:
why this sequence teaches best (typically: anchor with a strength, build
through correctable reps, close on the highest-leverage fix).

USER:
<game_context>rating={coaches_overall_rating} summary="{coaches_summary}"</game_context>
<clips>
  <clip id="{clipID}" kind="{goal|elite_save|teachable}" time="{start}-{end}s">
    depth={depth_rank} angle={cover_angle_rank} square={squareness_rank}
    rebound={rebound_control_rank} goals={metrics.goals} sog={metrics.shotsOnNet}
    coach_conf={coaching_confidence_score}
    technical_reasoning="{technical_reasoning}"
    actionable="{actionable_coaching_feedback}"
    caveats={analysis_confidence_caveats}
  </clip>
  ... (N clips)
</clips>
Return ONLY the JSON object.
```

### Persisted JSON shape (`type: "film_session"` / saved via save-json)
```json
{
  "type": "film_session",
  "response": {
    "generated_at": "2026-06-07T14:00:00Z",
    "scope": "game",                         // "game" | "week"
    "vIDs": ["abc123"],
    "session_theme": "Holding depth on east-west plays",
    "order_rationale": "Open on an elite save to set the standard, ...",
    "coach_edited": false,                    // flips true once a coach reorders/edits
    "agenda": [
      {
        "clipID": "abc123_0123_0131",
        "vID": "abc123",
        "clip_start_time": 123, "clip_end_time": 131,
        "kind": "elite_save",                 // goal | elite_save | teachable
        "instructional_score": 8.7,           // for transparency / re-sort
        "talking_point": "Notice how he holds the standing edge ...",
        "source_confidence": 4,               // min of the two clip conf scores
        "order": 1
      }
    ]
  }
}
```
The UI may PATCH this object (reorder, remove, add a manual clip) and re-`save-json` with `coach_edited: true`.

## Computation / logic — clip selection

Goal: pick ~5 clips that maximize *instructional value* while **guaranteeing a mix** of teachable goals-against and elite saves (never all of one kind).

### Ordinal maps (qualitative → number)
```
DEPTH      = { "Aggressive (A)":3, "Base (B)":2, "Conservative (C)":1, "Defensive (D)":0 }   // extremity, not quality
QUALITY    = { "Excellent":3, "Good":2, "Average":1, "Poor":0 }      // angle, squareness
REBOUND    = { "Killed (Elite)":3, "Steered to Low-Danger Zone":2,
               "Uncontrolled into High-Danger Zone":1, "Goal Allowed":0, "Not Applicable":null }
```
"Teachable extremity" for a quality field = distance from the middle: `abs(QUALITY[x] - 1.5)`. Both **Excellent** (reinforce) and **Poor** (fix) score high; **Average** scores ~0.

### Per-clip instructional score
```
def score_clip(clip, game_date, week_latest_date):
    conf = min(num(clip.coaching_feedback.coaching_confidence_score, 3),
               num(clip.goalie_positioning.goalie_position_confidence_score, 3)) / 5.0   # 0.2..1.0

    # event value
    goal_val   = 1.0 if clip.clipHasGoal or clip.metrics.goals > 0 else 0.0
    save_val   = min(num(clip.clipSaveCount,0), 3) / 3.0
    sog_val    = min(num(clip.metrics.shotsOnNet,0), 3) / 3.0     # difficulty proxy

    # rebound extremity (both elite and goal-allowed are teachable)
    rb = REBOUND[clip.coaching_feedback.rebound_control_rank]
    rebound_val = 0.0 if rb is None else abs(rb - 1.5) / 1.5      # 0..1

    # positioning extremity
    pos_val = (abs(QUALITY[clip....cover_angle_rank] - 1.5)
             + abs(QUALITY[clip....squareness_rank] - 1.5)) / 3.0  # 0..1 (avg of two, /1.5)

    caveat_penalty = 0.85 if clip.analysis_confidence_caveats else 1.0

    raw = ( W_GOAL    * goal_val
          + W_SAVE    * save_val
          + W_SOG     * sog_val
          + W_REBOUND * rebound_val
          + W_POS     * pos_val )

    recency = recency_weight(clip, game_date, week_latest_date)   # 1.0 single-game; decays in rollup
    return raw * conf * caveat_penalty * recency
```

### Weights (tunable constants)
```
W_GOAL    = 4.0   # a goal-against is the most instructive single event
W_SAVE    = 2.0   # elite/high save activity
W_SOG     = 1.0   # shot difficulty / volume
W_REBOUND = 2.5   # rebound-control extremes are very teachable
W_POS     = 1.5   # angle/squareness extremes
recency_weight (rollup) = 0.5 ** (days_since / 7)   # half-life one week; =1.0 for single game
```

### Balanced selection (guarantee the mix)
```
def select_agenda(clips, N=5):
    scored = sorted([(score_clip(c), c) for c in valid(clips)], reverse=True)
    goals       = [c for s,c in scored if is_goal(c)]
    elite_saves = [c for s,c in scored if is_elite_save(c)]   # rebound=Killed OR (save & angle/square Excellent)
    others      = [c for s,c in scored if c not in goals and c not in elite_saves]

    picks = []
    picks += take(goals, min(2, len(goals)))          # up to 2 teachable goals
    picks += take(elite_saves, min(2, len(elite_saves)))  # up to 2 elite saves
    picks += take(by_score(others + leftovers), N - len(picks))  # fill remainder by score
    # Guarantee mix: if picks are all one kind and the other kind exists, swap lowest-scored
    enforce_mix(picks, goals, elite_saves)
    dedupe_overlapping(picks, min_gap=2s)             # avoid two clips of the same play
    return sort_by_teaching_order(picks)              # strength-first, then correctives, see prompt
```
- `valid()` drops clips with `error`, and (in MVP) drops `conf < 2/5` unless nothing else qualifies.
- `is_elite_save`: `rebound_control_rank == "Killed (Elite)"` OR (`clipSaveCount>0` AND angle/squareness `Excellent`).

### Cross-game "weekly" rollup
Pool clips from all `vIDs` whose game `event_date` falls in the ISO week (from `customerID/<custID>.json` games[].event_date). Apply `recency_weight` so the most recent game's clips surface. Cap **per-game** picks (e.g. max 2 per game) so one blowout doesn't dominate the week. Same balanced selection, then talking points + a *weekly* theme.

## Frontend

New component **`components/FilmSession.tsx`** + a tab in the existing nav (alongside FavoriteClips / MyTraining). New `services/api.ts` helpers: `fetchFilmSession(vID)` (get-json on `analyze_video/05-film_session/<vID>` with fallback to the inline `film_session` item in `04-final_video`), `generateFilmSession(customerId, vIDs)` (POST `/api/film-session`), `saveFilmSession(id, agenda)` (save-json).

Behavior:
- Renders the `session_theme` header + an **ordered list** of agenda items. Each row shows kind badge (Goal / Elite Save / Teachable), thumbnail/time, `instructional_score`, and the `talking_point`.
- Clicking a row opens the **existing `ClipModal`** with that `ClipDetail` and `videoId` — reusing VideoPlayer telestration + voiceover with **no changes**. ClipModal gets `onNext`/`onPrev` wired to walk the agenda in order (presentation mode).
- **Coach controls:** drag-to-reorder, remove, and "Add clip" (opens the existing clip browser / Timeline to inject a `ClipDetail`; injected clips carry `kind: "teachable"` and no AI talking point until regenerated). Edits set `coach_edited: true` and call `saveFilmSession`.
- **Regenerate** button → `generateFilmSession` (re-runs selection + Gemini).
- **Export / Share:** "Copy agenda" (markdown of theme + per-clip talking points + timestamps) and "Share link" (read-only view for goalie/family).

```
+--------------------------------------------------------------+
|  FILM SESSION  ·  vs Sharks · 2026-06-05      [Regenerate]   |
|  Theme: "Holding depth on east-west plays"   [Copy] [Share]  |
+--------------------------------------------------------------+
|  Why this order: open on a strength, build to the key fix.   |
+--------------------------------------------------------------+
| #1  [ELITE SAVE]  02:03–02:11   score 8.7        [▶] [✎] [✕] |
|     "Holds the standing edge, lateral release — this is the  |
|      standard. Note the quiet upper body."          conf ●●●●|
+--------------------------------------------------------------+
| #2  [TEACHABLE]   05:40–05:48   score 7.9        [▶] [✎] [✕] |
|     "Angle is Average here — square earlier to the puck."    |
+--------------------------------------------------------------+
| #3  [GOAL]        11:22–11:31   score 9.4        [▶] [✎] [✕] |
|     "Goal against: depth too Conservative, rebound into the  |
|      house. This is the week's #1 fix."             conf ●●● |
+--------------------------------------------------------------+
| #4 [ELITE SAVE] ...   #5 [GOAL] ...                          |
+--------------------------------------------------------------+
|  [ + Add clip ]                          [ ▶ Present in order ]|
+--------------------------------------------------------------+
```

## Edge cases & limitations
- **Few clips (< N analyzed):** show all available; skip the balance guarantee; theme still generated. If 0 clips, show empty state pointing to FavoriteClips.
- **All-good game (no goals, no Poor ranks):** selection falls back to top elite saves + highest-quality reps; theme reframes as "what to reinforce." Never fabricate a weakness.
- **All-bad game (multiple goals):** cap goals at 2 in the agenda, fill with the *least-bad* / best-effort saves so the session isn't purely demoralizing; theme names the single highest-leverage fix.
- **Low confidence everywhere:** if all clips `conf < 3`, still produce an agenda but surface a banner ("footage quality limited") and hedge talking points; never present low-conf clips as definitive.
- **Unreliable fields:** `threat_type`/`threat_goalie_side` excluded from logic per caveats.
- **Overlapping windows:** `dedupe_overlapping` prevents two near-identical clips of the same shot.
- **Gemini failure / truncation:** reuse the existing summary parse-fallback pattern; on total failure, ship the selected clips with **template** talking points ("Goal against — review depth & rebound") so the agenda is never empty.
- **Stale agenda:** if a video is re-analyzed, the inline `film_session` item refreshes (Phase 2); on-demand agendas show `generated_at` and a re-generate prompt if older than the source JSON.

## Phasing & effort
- **MVP — effort M.** On-demand `/api/film-session` endpoint + heuristic selection (this spec's algorithm) + single Gemini talking-point call + `FilmSession.tsx` reading/persisting via get-json/save-json; reuses ClipModal/VideoPlayer untouched. Single-game scope.
- **Phase 2 — effort M.** feedback_seg post-step (`generate_film_session`) writing the inline `film_session` item for always-fresh agendas; **weekly rollup** across games with recency weighting and per-game caps; scheduled "weekly digest" generation + persistence; share-link read-only view.
- **Phase 3 — effort S.** Weight auto-tuning from coach edits (which clips coaches keep vs. remove), MyTraining hand-off (turn a recurring agenda fix into a drill).

## Success metrics
- **Coach prep time saved:** median time from "open game" → "ready to present" (target: < 1 min vs. current manual curation). Instrument time-to-first-Present.
- **Agenda usage:** % of analyzed games where a coach opens FilmSession; % that reach "Present in order"; clips played per session.
- **Edit rate:** fraction of agenda items kept vs. removed (a proxy for selection quality; high keep-rate = good auto-curation; feeds Phase 3 tuning).
- **Talking-point usefulness:** thumbs up/down per talking point (reuses FeedbackModal pattern).
- **Share/export rate:** agendas shared to goalies/families per coach per week.
- **Weekly digest engagement:** open rate of the scheduled weekly rollup.

## Open questions
1. Default N — fixed at 5, or scale with game length / clip count (e.g. 4–7)?
2. Persist location: dedicated `analyze_video/05-film_session/<vID>.json` vs. only the inline `film_session` item — or both with one as cache?
3. Should goalies/families be able to *generate* their own agenda, or only view a coach's (and is the read-only share link authenticated)?
4. Weekly rollup trigger: coach-initiated vs. scheduled cron (and which day)? Where does the schedule live (gateway cron vs. pipeline)?
5. Telestration persistence — should a coach's drawings on agenda clips save with the agenda for re-presentation?
6. Talking-point length/tone — one sentence vs. a short bullet list; configurable per coach?
7. Weight calibration: ship the constants above as defaults and tune from real coach edit data, or pilot-tune first with a few coaches?
