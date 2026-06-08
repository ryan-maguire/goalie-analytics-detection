# Ask the Film
*Talk to your game tape. Ask a question in plain English — get a grounded answer and the exact clips behind it.*

## Summary & problem
Every other view in Goalie Analytics Pro answers a **pre-decided** question: Leak Finder finds recurring weaknesses, HDSV% breaks save % by danger, BeatenMap shows where goals went in. But the questions goalies, parents, and coaches actually ask are open-ended and personal — *"show me every time I was beaten on the rush in the 3rd period," "how was my rebound control this month?," "which games did I struggle glove-high?," "find the clips where I was too deep on a one-timer."* Today the only way to answer those is to manually scrub Timeline + toggle FilterBar across game after game.

**Ask the Film** is a conversational, agentic query surface over the goalie's entire published clip corpus. A natural-language question is translated by Gemini into a **structured filter** over known clip fields, matching clips are retrieved **in code** from the published analysis JSON (no video re-analysis), counts/aggregates are computed **in code**, and Gemini then **narrates a grounded answer that cites only the retrieved clips**. The user gets both a synthesized answer *and* a clickable strip of result clips that deep-link to `ClipModal` / `VideoPlayer`. This is the flagship differentiator: it turns a static analytics dashboard into something you can interrogate.

The hard design constraints are **anti-hallucination** (the LLM never invents a stat or a clip — numbers come from code, narration cites real `clipID`s) and **cost/latency** (two bounded Gemini calls over text only, aggressively cached).

## Target users
- **Goalies** — self-coach by asking the tape directly ("where do I keep getting beat?") without learning the filter UI.
- **Goalie families** — low-friction natural-language access; no analytics literacy required.
- **Goalie coaches** — fast cross-game pulls for film sessions ("every high-danger rush goal this season"), composes with Film Session agenda.
- **Recruiters (secondary)** — quick qualitative pulls, always clip-backed and honest about sample size.

## Data inputs
All evidence comes from the published analysis JSON at `gs://goalie_video_bucket/analyze_video/04-final_video/<vID>.json` (array of items; `item.type ∈ {summary, clips, windows}`; UI treats `windows == clips`). The customer's game list comes from `customerID/<custID>.json`. **No new pipeline fields are required for MVP** — Ask the Film reads what is already published. Fields consumed per clip:

| Field | Type | Use in filter / answer |
|---|---|---|
| `clipID` | str | citation key; deep-link to `ClipModal` |
| `clip_start_time` / `clip_end_time` / `clip_duration` | float (sec) | **approx. period** derivation; ordering; clip strip |
| `clipShot` / `clipShotCount` | bool / int | shot-faced filter |
| `clipSave` / `clipSaveCount` | bool / int | save filter / counts |
| `clipHasGoal` | bool | "beaten / goal against" filter |
| `clipShot`/`clipSave` exposed as `metrics.shots/saves` after normalize | — | prefer `metrics.*` when present |
| `technical_reasoning` | str | semantic-cue matching (rush, screen, cross-ice, breakaway) + answer grounding |
| `threat_goalie_side` | str (`left`/`right`) | side filter (fallback to `metrics.observed_goalie_side`) |
| `analysis_confidence_caveats[]` | str[] | low-confidence flag; surfaced in answer honesty |
| `goalie_positioning.depth_rank` / `cover_angle_rank` / `squareness_rank` | str (qualitative) | rank-threshold filters via `rankToOrdinal` map |
| `goalie_positioning.goalie_position_confidence_score` | 1–5 | confidence gating |
| `coaching_feedback.rebound_control_rank` | str (qualitative) | rebound-control filter |
| `coaching_feedback.actionable_coaching_feedback` | str | answer grounding |
| `coaching_feedback.coaching_confidence_score` | 1–5 | confidence gating |
| `metrics.shots / shotsOnNet / saves / rebounds / goals` | int | code-computed counts in the answer |
| `metrics.observed_goalie_side` | str | preferred side signal |
| `metrics.shot_timestamps[]{timestamp,location,release,outcome}` | array | location/release/outcome filtering ("glove-high"→`beaten_location`; "slot"→`location`) |
| `metrics.beaten_location` | str (`glove_high\|glove_low\|blocker_high\|blocker_low\|five_hole\|body_other\|unknown\|not_applicable`) | **goal-location filter** (goal clips only, v15) |
| `metrics.beaten_location_confidence` | 0.0–1.0 | gate beaten-location answers |
| `metrics.beaten_location_notes` | str | answer grounding |
| `summary.response.coaches_summary` / `coaches_overall_rating` | str | per-game context in cross-game answers |

Game-level (from `customerID/<custID>.json`, via `fetchUserGames`): `vID`, `eventName`, `event_date`, `goalie_name`, `eventSeason` → **date/season scoping** and clip→game attribution.

**NEW field (optional, v2 — stage `metrics_seg` prompt v16):** `metrics.shot_timestamps[].shot_danger ∈ {high,medium,low}` (same field proposed by spec 02). If present, enables a true "high-danger" filter dimension instead of the location proxy. Not required for MVP.

**CAVEATS honored throughout:** `threat_type` is **not reliable** — never expose it as a filter dimension. Ranks are **qualitative strings** — mapped to ordinals only for threshold comparisons, never averaged in an answer. Aggregates are **pooled** (sum TP/FP across clips), never per-game means.

## Pipeline / backend changes
**No detection-pipeline change for MVP.** Ask the Film is read-only over already-published JSON.

**New endpoint — recommended home: UI `server.ts` (Express).** It already has Vertex/GCS access (ADC + runtime SA), already brokers signed playback, and auto-deploys on push to main — so no separate deploy story and no exposing a Gemini key to the browser. Add:

```
POST /api/ask-film
  body: { customerId, question, scope?: {vIDs?: string[], season?: string, gameId?: string},
          history?: {role, content}[], maxClips?: number }
  resp: { answer: string, filter: ClipFilter, results: AskResult[],
          counts: AskCounts, routedTo?: 'leak_finder'|'hdsv'|'beaten_map'|null,
          truncated: boolean, fromCache: boolean }
```

`server.ts` calls Vertex Gemini via `@google-cloud/vertexai` (add dep) using the same project/ADC already configured. The browser never sees model credentials. (Alternative home: the Flask gateway — but it would need Vertex creds + its own deploy + the CORS `*` surface; `server.ts` is the lower-friction, already-credentialed choice. Decision noted in Open Questions.)

**Retrieval source:** reuse the existing `fetchGameData(vID)` path / `get-json` for each in-scope game, with the same `normalizeGameData` boolean/count coercion the UI already applies, so the filter logic sees identical types.

## Computation / logic

### Two-call flow (text/structured only — never sends video)
```
1. PLAN  (Gemini structured output → ClipFilter)
   in:  question + history + the ClipFilter JSON schema + the field dictionary
   out: a ClipFilter object (function-calling / responseSchema). No prose.
        Includes filter.route ∈ {none|leak_finder|hdsv|beaten_map} for hand-off.

2. RETRIEVE  (pure code, no LLM)
   load in-scope games → flatten clips → apply ClipFilter deterministically
   → rank → take top maxClips (default 12) → compute AskCounts in code.

3. NARRATE  (Gemini, grounded)
   in:  question + AskCounts (code-computed) + a COMPACT evidence list of the
        matched clips (clipID, game, time, side, ranks, beaten_location,
        technical_reasoning excerpt, caveats). NOTHING ELSE.
   out: a short answer that (a) uses ONLY numbers from AskCounts,
        (b) cites clips by clipID, (c) states sample size / low-confidence
        honestly, (d) never claims a stat not in AskCounts.
```

If `filter.route != none`, the endpoint may **delegate** to an existing computation instead of (or alongside) generic retrieval — e.g. route "where do I keep getting beat?" to Leak Finder's `analyzeLeaks(clips)`, "high-danger save %?" to `computeDangerSplits(clips)` (`useDangerSplits.ts`), "where do goals go in?" to BeatenMap's `beaten_location` tally. The NARRATE step then narrates that computation's output (still code-computed numbers), and the UI can surface a "Open in Leak Finder →" affordance. This keeps Ask the Film composed with, not duplicating, the dedicated views.

### ClipFilter schema (the structured translation target)
```ts
interface ClipFilter {
  scope: { season?: string; dateFrom?: string; dateTo?: string;
           vIDs?: string[]; lastNGames?: number };      // default: all games
  outcome?: ('goal'|'save'|'shot'|'rebound')[];          // maps clipHasGoal/clipSave/clipShot/metrics.rebounds
  beatenLocation?: BeatenLocation[];                      // 'glove_high'… ; goal clips only
  side?: ('left'|'right')[];                              // observed_goalie_side ?? threat_goalie_side
  shotLocation?: string[];                                // taxonomy: 'slot','in close','point',…
  release?: string[];                                     // 'one-timer','wrist','redirect-tip',…
  period?: (1|2|3)[];                                     // APPROX, derived from clip_start_time
  ranks?: { dimension: 'depth'|'cover_angle'|'squareness'|'rebound_control';
            op: '<='|'>=='|'=='; value: 'poor'|'average'|'good'|'elite' }[];
  textCues?: string[];                                    // ['rush','breakaway','screen','cross-ice']
  minConfidence?: number;                                 // gate by *_confidence_score / beaten_location_confidence
  route: 'none'|'leak_finder'|'hdsv'|'beaten_map';
  needsClarification?: { question: string };             // when scope/intent ambiguous
}
```
Gemini PLAN is given the exact enum values (the real `BeatenLocation` union, the `shot_timestamps[].location` taxonomy, the rank vocabulary) so it can only emit known values — invalid values are dropped server-side before retrieval.

### Deterministic retrieval (pseudocode)
```
clips = []
for game in inScopeGames(filter.scope, customerGames):
    data = normalize(getJson(game.vID))
    for clip in clipsOf(data):
        clip._game = game            // attach eventName/event_date/season/vID
        clips.push(clip)

matched = clips.filter(c => matchesFilter(c, filter))

function matchesFilter(c, f):
    if f.outcome && !outcomeMatches(c, f.outcome): return false        // clipHasGoal / clipSave / clipShot / rebounds>0
    if f.beatenLocation && !f.beatenLocation.includes(c.metrics?.beaten_location): return false
    if f.side && !f.side.includes(c.metrics?.observed_goalie_side ?? c.threat_goalie_side): return false
    if f.shotLocation && !anyShotMatches(c.metrics?.shot_timestamps, f.shotLocation): return false
    if f.release   && !anyShotMatches(c.metrics?.shot_timestamps, f.release, 'release'): return false
    if f.period && !f.period.includes(approxPeriod(c, gameSpan(c._game))): return false
    if f.ranks && !ranksSatisfied(c, f.ranks): return false            // rankToOrdinal compare
    if f.textCues && !cueHit(c.technical_reasoning + coaching_feedback, f.textCues): return false
    if f.minConfidence && clipConfidence(c) < f.minConfidence: return false
    return true

// approx period — same method as HDSV% spec (no period field exists):
approxPeriod(c, span) = clamp(ceil((c.clip_start_time - span.start)/(span.end-span.start) * 3), 1, 3)

// rank threshold via shared rankToOrdinal {elite/excellent:4, good/strong:3, average/fair:2, poor/weak:1, unknown:null}
```
Ranking for the top-N strip: prioritize (1) higher `*_confidence_score` / `beaten_location_confidence`, (2) recency (`event_date` desc), (3) goals before saves when intent is "beaten." `truncated=true` when matched > maxClips.

### AskCounts (computed in CODE — the only numbers the LLM may use)
```
{ totalMatched, byOutcome:{goal,save,shot,rebound}, byGame:[{vID,eventName,n}],
  bySide:{left,right}, byBeatenLocation:{glove_high:…}, byPeriod:{1,2,3},
  saveRate?: saves/shotsOnNet (pooled, with n),     // only if denominator >= MIN_N
  lowConfidenceCount }                               // clips with caveats or score<threshold
```
`saveRate` is pooled (sum saves / sum shotsOnNet across matched clips), suppressed to `null` with an explicit "not enough shots" note when `n < MIN_N` (=5, reuse from `useDangerSplits`). The LLM is forbidden from emitting any percentage not present here.

### Grounding rules (sent verbatim in the NARRATE system instruction)
1. Use ONLY the counts in `AskCounts`; never compute or estimate a new number.
2. Cite clips by `clipID` (rendered as chips in the strip); never invent a clip.
3. If `totalMatched == 0`, say so plainly and suggest a broader filter — do not fabricate.
4. State sample size and that ranks/danger/period are **qualitative/approximate** when relevant.
5. If `lowConfidenceCount` is high, caveat the confidence.
6. Keep it to ~4–6 sentences; defer detail to the clip strip.
7. Never reference `threat_type`.

### Latency & cost controls
- **Two Gemini calls only**, both **text-only** (no video, no images): PLAN (small structured) + NARRATE (compact evidence). Target p50 < 4 s.
- **Model:** `gemini-2.5-flash` for both calls (cheap, fast; PLAN is schema-constrained, NARRATE is short). `gemini-2.5-pro` is reserved for the detection pipeline; not needed here.
- **Caching:** (a) per-game normalized clip JSON cached in `server.ts` memory with an ETag/`_t` check (already the fetch pattern) so repeat questions don't re-pull GCS; (b) answer cache keyed by `hash(customerId + canonicalized ClipFilter)` — identical filters reuse the prior answer; (c) optional Vertex context caching of the static field-dictionary prompt prefix.
- **Bounds:** evidence list capped at `maxClips` (12) and each `technical_reasoning` excerpt truncated (~240 chars) to cap NARRATE tokens; scope defaults to all games but `lastNGames`/`season` shrink the corpus.
- **Rate-limit** per `customerId` (e.g. simple token bucket in `server.ts`) to cap spend.

## Frontend
**New top-level view `ask_film`.** Wire-up (per existing conventions):
- Add `'ask_film'` to the view unions in `App.tsx` (`handleViewChange` + `currentView`) and `Header.tsx` (`onNavigate`/`handleNavigation` + menu item, icon e.g. `MessageCircle`/`Sparkles`).
- Add a `case 'ask_film':` in `App.tsx` `renderView()` rendering `<AskFilm customerId={…} games={…} onOpenClip={…} onNavigate={…} />`.

**New component `components/AskFilm.tsx`** — a chat/search surface:
- Question input (with example-chip prompts), optional scope selector (All games / This season / This game) reusing `GameSelector`/season values from `fetchFormOptions`.
- Answer panel (the NARRATE text) with inline `clipID` chips.
- A horizontal **clip-result strip** reusing `ClipCard` (compact mode) — each card deep-links via existing `onExpand` → `ClipModal` + `VideoPlayer` (signed playback already handled).
- A "routed" banner when `routedTo` is set: "This looks like a Leak Finder question — Open in Leak Finder →" calling `onNavigate('leak_finder')`.
- Conversation history kept client-side and passed back as `history` for follow-ups ("…and just the 3rd period").

**New service fn `askFilm()` in `services/api.ts`** → `POST /api/ask-film` (same-origin Express; no gateway round-trip, no Gemini key in browser).

**No new types beyond `ClipFilter`/`AskResult`/`AskCounts`** added to `types.ts`; clip rendering reuses `ClipDetail`. (Add `metrics.shot_timestamps`/`beaten_location` to `ClipDetail.metrics` if not already present from spec 02/03.)

```
┌──────────────────────────────────────────────────────────────────┐
│  ASK THE FILM                         Scope: [ All games ▾ ]       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ "every time I was beaten on the rush in the 3rd period"  ▶ │  │
│  └────────────────────────────────────────────────────────────┘  │
│  Try: [beaten glove-high?] [rebound control this month] [too deep]│
│                                                                    │
│  ── Answer ──────────────────────────────────────────────────     │
│  Across 4 games you were beaten on the rush 5 times; 3 came in    │
│  the approx. 3rd period (#c1842, #c2017, #c2099). Two were        │
│  glove-high. Sample is small — judge the clips, not the rate. ⓘ    │
│  [ Looks like a Leak Finder question → Open in Leak Finder ]       │
│                                                                    │
│  ── 5 clips ─────────────────────────────────  showing 5 of 5     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                     │
│  │clip ▸│ │clip ▸│ │clip ▸│ │clip ▸│ │clip ▸│   (→ ClipModal)     │
│  │P3 GA │ │P3 GA │ │P3 SV │ │P2 GA │ │P3 GA │                     │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                     │
│  ⓘ "period" is approximate (from clip time); ranks are qualitative │
└──────────────────────────────────────────────────────────────────┘
```

## Edge cases & limitations
- **Hallucination risk** is the core threat; mitigated by code-computed `AskCounts` + clipID-only citation + the explicit grounding rules. Numbers never originate in the LLM.
- **`beaten_location` is goal-clips-only and v15+** — older JSONs lack it; "glove-high beaten" questions only match games re-analyzed under v15; the answer must say so when coverage is partial (use `beaten_location_confidence` to gate).
- **`shot_danger` not in production** — "high-danger" questions fall back to a location/cue proxy (slot/in-close/rush) until the v2 field ships; flag as approximate. Do not present proxy danger as exact.
- **Period is not a field** — derived from `clip_start_time`; no intermission/OT awareness; mixed continuity distorts it. Always labeled "approx."
- **Ranks are qualitative** — rank thresholds use the shared `rankToOrdinal` map; never narrated as numeric averages.
- **`threat_type` excluded** — unreliable; not a filter dimension even though it exists in schema.
- **Mixed-goalie games / multi-goalie configs** — clips are attributed by game, not by goalie identity within a game; side/period blends are possible (note in tooltip; out of MVP scope to disambiguate).
- **Empty / ambiguous question** → PLAN returns `needsClarification`; UI asks a one-line follow-up rather than guessing scope.
- **No-match** → honest "0 clips" answer + suggested broadening; never fabricated.
- **Stale corpus** — answers reflect last-published JSON; a game mid-reanalysis may under-count (surface `windows_failed` coverage when low).
- **Cost runaways** — bounded by two flash calls, capped evidence, answer cache, per-customer rate limit.

## Phasing & effort
- **MVP (M):** `POST /api/ask-film` in `server.ts` (Vertex flash, PLAN→retrieve→NARRATE), `ClipFilter` schema + deterministic `matchesFilter`/`AskCounts`, `AskFilm.tsx` + view wiring + `askFilm()` service, in-memory game cache + answer cache. Outcome/side/beaten-location/period/text-cue/rank filters over existing fields. **~M** (most effort is the filter executor + grounding prompt tuning). Deps: existing `fetchGameData`/`normalizeGameData`, `ClipCard`/`ClipModal`, `rankToOrdinal`.
- **v1.1 (S):** routing/delegation to Leak Finder (`analyzeLeaks`), HDSV% (`computeDangerSplits`), BeatenMap; follow-up conversation history; example-chip prompts.
- **v2 (S, depends on spec 02):** consume `shot_timestamps[].shot_danger` for a true high-danger filter once `metrics_seg` v16 ships and games are re-analyzed (`bash deploy/deploy.sh` + re-run).
- **v3 (M, later):** shareable answer permalinks; export an Ask result straight into a Film Session agenda; voice input.

## Success metrics
- **Groundedness:** 0 hallucinated stats/clips in a manual audit of ≥50 answers (every number traceable to `AskCounts`, every cited clipID present in results).
- **Resolution rate:** ≥70% of questions return ≥1 relevant clip without needing clarification.
- **Latency:** p50 end-to-end < 4 s; cache hit rate ≥ 40% on repeat/refined questions.
- **Engagement:** Ask the Film sessions per active user; deep-link clicks from the clip strip into `ClipModal`; route-through clicks into Leak Finder/HDSV/BeatenMap.
- **Cost:** average Gemini spend per question within target (two flash calls); rate-limit trips near zero.

## Open questions
1. **Endpoint home:** `server.ts` (already Vertex/GCS-credentialed, auto-deploys, no browser key) — confirmed default — vs the Flask gateway? Proposing `server.ts`.
2. **Model:** `gemini-2.5-flash` for both PLAN and NARRATE — adequate for structured + short narration, or does NARRATE need `pro` for tone? Start flash.
3. **Default scope:** all games vs `lastNGames`/current season, to bound corpus size and cost?
4. **Routing aggressiveness:** when a question matches a dedicated view, do we narrate inline *and* offer the route, or always redirect? Proposing narrate-inline + offer.
5. **Answer caching invalidation** — purge on new game publish / favorites change, or short TTL? Proposing TTL + ETag on per-game JSON.
6. **Multi-goalie disambiguation** within a game — defer (note in tooltip) or add a goalie filter in v2?
7. **`maxClips`/excerpt length** defaults (12 / ~240 chars) — confirm against token budget and UX.
