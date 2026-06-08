# Parent Digest

> The warm, plain-language progress note a family forwards to grandparents: "Claire's rebound control improved from a C to a B+ over her last 4 games; her biggest growth area is depth on the rush — here's one drill to work on." Cross-game, family-facing, positive-but-honest.

---

## 1. Summary & problem

Goalie families are the audience that actually opens the app week to week, but everything we surface is built for the goalie or the coach. The per-game `coaches_summary` (already written by `feedback_seg`) is technical, single-game, and uses four-pillar coaching language. The Season Report Card (spec 08) is a *graded* artifact — letter grades and a rubric — which is great for self-assessment but reads like a transcript, not a story. A grandparent, an aunt, or a non-hockey parent opening either of those does not get the one thing they want: **is my kid getting better, what are they great at, and what should they work on — said kindly.**

The **Parent Digest** translates the technical analytics into one short, encouraging, *cross-game* narrative paragraph plus a couple of celebratory highlight numbers. It is the thing a parent screenshots and texts to family. It never demoralizes: weaknesses are framed as "growth areas," and a bad stretch is handled honestly but warmly. It is explicitly **not** another grade and **not** another coach note.

Two ways to build it, and we ship both in sequence:
- **v1 (MVP):** a deterministic, **template-based** digest assembled purely from aggregated season trends on the frontend. No Gemini, no new backend, zero per-call cost.
- **v2 (upgrade):** a single **on-demand Gemini summarization pass** that takes the same aggregated stats + trends and emits a more fluent, varied, genuinely warm paragraph. One text-only call; optionally persisted.

---

## 2. Target users

| Role | Relationship to feature |
|------|------------------------|
| **Goalie family** (primary) | Non-technical parent / grandparent. Opens the digest, reads the narrative, forwards it. Does not want jargon, grades, or charts — wants reassurance + one concrete thing to work on. **This is the family-primary feature.** |
| **Goalie** (secondary) | Reads it as a morale/motivation surface. The encouraging tone matters most here — a teen goalie should never feel judged. |
| **Coach** (tertiary) | May glance at it to align on messaging, but the coach's real tool is the per-game `coaches_summary` and the Film Session (spec 04). Not the target. |

Design north star: "Grandma opens this on a phone, reads it in 30 seconds, understands how her grandkid is doing, and smiles."

### Differentiation (do not duplicate)
- **vs. per-game `coaches_summary`** (`feedback_seg`, technical, one game, four-pillar language): the Digest is **multi-game** and **non-technical**. It may *quote* feedback but never reproduces the pillar framework.
- **vs. Season Report Card (spec 08, graded):** the Report Card *grades* (A/B/C, rubric). The Digest *narrates* (warm prose, no rubric on display). They share aggregated inputs; the Digest can reuse the Report Card's pillar grades as an internal input for the "C → B+" phrasing, but presents them as a friendly story, not a transcript.
- **vs. RecruitingProfile (spec 05):** that is an external, weakness-hiding marketing artifact for recruiters. The Digest is internal, honest about growth areas, and shared *within the family*.

---

## 3. Data inputs (exact fields)

All inputs already exist. Per game: `gs://goalie_video_bucket/analyze_video/04-final_video/<vID>.json` (array of `{type, response}` items; `windows` remapped to `clips` in `services/api.ts`). Game list + metadata: `customerID/<custID>.json` via gateway `get-json`.

### Customer config (game records, for ordering & labels)
- `vID`, `eventName`, `event_date`, `goalie_name`, `eventSeason`

### Per-game summary (`summary.response`)
- `coaches_summary` (string — *source for tone/quote mining only, not shown verbatim*)
- `coaches_overall_rating` (string)

### Per-clip (each clip in `clips`)
- `clipHasGoal` (bool)
- `metrics.shotsOnNet`, `metrics.saves`, `metrics.goals` → per-game and season save%
- `goalie_positioning`: `depth_rank`, `cover_angle_rank`, `squareness_rank` (qualitative ranks)
- `coaching_feedback`: `rebound_control_rank`, `actionable_coaching_feedback` (the latter is the source of the single suggested drill / "work on" line)

### Optional cross-feature inputs
- Season Report Card (spec 08) pillar **grades per game** if available — used to phrase "C → B+" deltas. If spec 08 is not yet built, the Digest derives its own coarse grade from rank distributions (see §6) and degrades gracefully.

> **CAVEAT:** positioning and coaching ranks are *qualitative* strings (e.g. `"GOOD"`, `"FAIR"`, `"POOR"` / `"HIGH"`...). The Digest must map them to an ordinal scale once (`§6`) and never invent precision the data lacks. Pool aggregates across clips; small samples are flagged (`§8`).

---

## 4. Pipeline / backend

**No `metrics_seg` / `feedback_seg` pipeline change. No worker redeploy.** The Digest is a *consumer* of already-published JSON, like Leak Finder (spec 01).

### (a) v1 — frontend template, no backend, no Gemini (CHOSEN for MVP)
The entire digest is computed and rendered client-side in `goalie-analytics-pro-ui` from the already-fetched season clips. Justification:
- Zero per-call cost, zero new infra, zero latency, no Vertex quota exposure.
- Deterministic and testable — the same season always produces the same digest (no model drift; matters for a feature whose whole job is *trust* and *tone safety*).
- Families value reliability over prose flair; a sturdy template clears the bar.

The only risk is robotic phrasing — mitigated by a small bank of tone-varied template fragments selected by trend shape (`§6`).

### (b) v2 — on-demand Gemini summarization (UPGRADE, effort M)
Add one route — **`POST /api/parent-digest`** — preferably in `server.ts` (the UI Express server already has Vertex access; see existing `/api/video-url`, `/api/upload-url`) or in `goalie-analytics-api-gateway`. The frontend computes the **aggregated trend object** (`§6`) and POSTs it; the server makes **one text-only Gemini-2.5-flash call** (same client/config style as `feedback_seg/gemini.py`: `response_mime_type="application/json"`, `temperature` ~0.4 here — slightly above the pipeline's `0` because we *want* warm, varied prose, not reproducibility). Only the small aggregate JSON is sent — no clip text, no video, cheap and fast.

Rationale for on-demand vs. baking into the pipeline: a digest is a *family read action*, computed across an arbitrary set of games (a season, last-4, a tournament weekend), so it cannot be pinned to a single video's pipeline run. On-demand avoids reprocessing and supports "last 4 games" rollups.

#### Gemini prompt sketch (v2)
```
SYSTEM / INSTRUCTION:
You write a short, warm progress note for the FAMILY of a youth hockey
goalie — including non-hockey relatives. Plain language, no jargon, no
letter grades, 90–140 words, second-or-third person using the goalie's
first name. Celebrate improvement first. Frame every weakness as a
"growth area" with one encouraging next step. Never demoralize, never
imply the goalie is bad, never use the words "failure", "weak", "poor
performance". If the recent stretch declined, acknowledge it kindly and
pivot to effort + the plan. Output STRICT JSON only.

INPUT (the aggregated trend object, see §6):
{ goalie_name, games_played, span_label, save_pct_trend,
  most_improved_pillar, biggest_growth_area, suggested_drill,
  pillar_deltas, sample_size_flag, bad_stretch_flag }

OUTPUT JSON:
{
  "headline": "string  (<= 70 chars, e.g. 'Claire is trending up')",
  "narrative": "string  (90–140 words, warm paragraph)",
  "celebrate": "string  (one sentence on the single best thing)",
  "growth_area": "string  (one sentence, framed positively + the drill)",
  "tone_check": "string  (must be 'encouraging'; model self-asserts)"
}
```
**Persistence (optional):** the v2 response MAY be cached via gateway `save-json` at `customerID/<custID>/digests/<span_label>.json` so re-opens are instant and the family always sees the same wording they forwarded. Cache-bust when a new game is added to the span.

---

## 5. Computation / logic

All of this runs the same in v1 (renders the template) and v2 (feeds the prompt). Build it once as a pure function `buildDigestInputs(games): DigestInputs`.

### Rank → ordinal mapping (do once)
Map qualitative ranks to a 0–3 ordinal: `POOR/LOW→0, FAIR→1, GOOD→2, EXCELLENT/HIGH→3` (case-insensitive; unknown → omit, do not coerce to 0). Per game, average the ordinal per pillar across that game's clips.

### Aggregated trend inputs
- **games_played** = count of games in span; **span_label** = e.g. `"last 4 games"` or `"2025–26 season"`.
- **save_pct_trend**: season save% = `Σ saves / Σ shotsOnNet`; trend = compare first-half-of-span mean vs. second-half-of-span mean → `{direction: up|flat|down, from_pct, to_pct, delta}`.
- **pillar_deltas**: for each pillar (depth, cover_angle, squareness, rebound_control), early-window mean ordinal vs. late-window mean ordinal. Translate to friendly grades for phrasing (`§ grade map` below).
- **most_improved_pillar** = pillar with the largest positive delta (the "C → B+" story).
- **biggest_growth_area** = pillar with the lowest *current* mean ordinal (not the most-declined — we lead families toward the highest-leverage area, kindly).
- **suggested_drill** = `actionable_coaching_feedback` from the most recent game's clips that touches `biggest_growth_area` (pick the most recent non-empty; trim to one sentence). Fallback to a static drill map keyed by pillar.
- **sample_size_flag** = true if `games_played < 3` or `Σ shotsOnNet < 15`.
- **bad_stretch_flag** = true if `save_pct_trend.direction == down` AND delta beyond a small threshold.

### Grade map (for "C → B+" phrasing only; never shown as a transcript)
Bucket the 0–3 ordinal into friendly grades (e.g. `0→C, 1→C+/B-, 2→B/B+, 3→A`). If spec 08 grades are present, prefer those for consistency.

### Encouraging-tone rules (enforced in BOTH versions)
1. **Celebrate first.** The opening sentence is always about improvement or a strength — never the deficit.
2. **Weaknesses are "growth areas," singular and actionable** — exactly one, always paired with a concrete next step.
3. **Never demoralize.** Banned framings: ranking the kid against others, "bad/weak/poor," failure language, percentile shaming.
4. **Honest, not dishonest.** A real decline is acknowledged ("the last couple of games were tougher") then pivoted to effort + plan — we do not pretend it didn't happen (that breaks trust and is the difference vs. the recruiting profile).
5. **Concrete, small numbers only** — one or two highlight numbers (games played, save% move). No charts, no rubric tables.

---

## 6. Frontend

A new **`ParentDigest`** component in `goalie-analytics-pro-ui/components/`, surfaced two ways:
- A **card** at the top of the home/season view (the first warm thing a family sees), and
- A **full view** reachable from nav, with a prominent **Share** affordance.

UX:
- One readable paragraph (the narrative), a friendly **headline**, and **two highlight stat chips** (e.g. "8 games" · "Save% 78% → 84% ↑").
- **Share** button → native share sheet / copy-to-clipboard of the narrative + headline as plain text (so it pastes cleanly into a text thread to grandparents). v2 may also offer "copy as image".
- **Span selector**: "Last 4 games" / "This season" (drives `buildDigestInputs`).
- Tone is visual too: soft, warm styling — not the dashboard's data-grid look.
- v1 renders the template instantly; v2 shows a one-line "writing your digest…" state during the single Gemini call, then the same layout.

### ASCII mockup
```
+------------------------------------------------------------+
|  Family Progress Note                       [ Share ]      |
|  ----------------------------------------------------------|
|  Claire is trending up.                                    |
|                                                            |
|  Over her last 4 games Claire has been steady and          |
|  competitive in net. Her rebound control has come a long   |
|  way — moving from around a C to a B+ — and you can see     |
|  her staying square to more shots. Her biggest growth      |
|  area right now is her depth on the rush; the next step    |
|  is simple: a short "challenge-and-recover" drill a few    |
|  times a week. She's working hard and it's showing.        |
|                                                            |
|   [ 4 games ]   [ Save%  76% -> 84%  ^ ]                    |
|                                                            |
|  Span:  ( Last 4 games )  ( This season )                  |
+------------------------------------------------------------+
```

---

## 7. Edge cases

- **Bad stretch (declining save%).** Honor honesty rule #4: open on a real strength, acknowledge the tough stretch kindly, pivot to effort + the one drill. Never lead with the decline; never hide it. (`bad_stretch_flag` selects a dedicated, gentle template / prompt branch.)
- **Small sample (1–2 games, or < 15 shots).** Set `sample_size_flag`; soften all trend claims ("it's still early, but…"), suppress "C → B+"-style deltas (not enough data to claim a trend), keep it purely celebratory + one forward-looking note.
- **Cold start (zero analyzed games).** Show an empty-state card: "Your family progress note will appear after the first game is analyzed." No fabricated narrative.
- **Missing fields.** Ranks/feedback often partly absent — the ordinal map omits unknowns (never coerces to 0); if a pillar has no data it is excluded from delta/growth-area selection. If `actionable_coaching_feedback` is empty, fall back to the static drill map.
- **Flat trend.** Use a "consistent and reliable" framing rather than forcing an improvement story.
- **v2 Gemini failure / quota.** Fall back to the v1 template silently (same `buildDigestInputs`), so the family always gets a digest. Mirror `feedback_seg`'s graceful-failure pattern.
- **Tone safety regression (v2).** If `tone_check != "encouraging"` or any banned word appears in the output, discard the Gemini result and render the v1 template.

---

## 8. Phasing & effort

| Phase | Scope | Effort |
|------|-------|--------|
| **MVP — v1 template** | `buildDigestInputs` pure function (rank ordinal map, save% trend, pillar deltas, growth-area + drill selection, flags), tone-varied template bank, `ParentDigest` card + view + Share, span selector, all edge-case branches. No backend. | **S** |
| **Upgrade — v2 Gemini** | `POST /api/parent-digest` on `server.ts` (Vertex, single flash call, temp ~0.4), prompt + strict-JSON parse, tone-safety guard with v1 fallback, optional `save-json` cache + cache-bust on new game, "writing…" state. | **M** |

Ship v1 first; it stands alone and de-risks tone. v2 is purely a prose-quality upgrade over the same inputs.

---

## 9. Success metrics

- **Share rate** — % of digest views that trigger Share/copy (the core "forward to grandma" behavior).
- **Family engagement** — digest card open rate among family accounts; repeat opens per season.
- **Tone safety** — zero banned-word / demoralizing-output incidents (v2 guard rejections logged; target rejection rate trending down as prompt matures).
- **v1 → v2 lift** — A/B on share rate and a quick thumbs-up "Did this feel encouraging?" micro-survey.
- **Coverage** — % of active goalies with a renderable digest (non-cold-start) within a week of their latest game.

---

## 10. Open questions

1. **Span default** — last-4 vs. full season as the default view? (Lean last-4: recency is what families ask about.)
2. **Spec 08 dependency** — adopt the Report Card's pillar grades for "C → B+" phrasing, or keep the Digest's own coarse grade map to ship independently? (Lean independent for MVP, adopt later.)
3. **Drill source** — is mining `actionable_coaching_feedback` reliable/family-readable enough, or do we need a curated per-pillar drill library with video links?
4. **Goalie's own view** — should the goalie see the *exact* family wording, or a slightly different motivational variant?
5. **Cadence / notifications** — push a fresh digest automatically after each new analyzed game (email/text), or keep it pull-only in-app?
6. **v2 persistence** — cache wording so the family always re-reads what they forwarded, vs. always regenerate fresh? (Lean cache.)
7. **Multi-goalie families** — one digest per goalie (assumed) vs. a combined household note?
