# Product feature specs

Implementation-ready specs for net-new Goalie Analytics Pro features, derived
from the data the pipeline already produces (`analyze_video/04-final_video/<vID>.json`)
and the existing web app. Each spec is self-contained (data inputs, pipeline/
backend changes, computation, frontend + mockup, edge cases, phasing, success
metrics, open questions).

## Status

| # | Spec | Primary user | Pipeline change? | Status |
|---|------|--------------|------------------|--------|
| 01 | [Leak Finder](01_leak_finder.md) — cross-game weakness patterns → drills | Goalie / Coach | No | ✅ **shipped** |
| 02 | [High-Danger Save %](02_high_danger_save_pct.md) — danger-weighted save% + splits | Goalie / Coach / Recruiter | v1 none; v2 `metrics.shot_danger` | ✅ **shipped** (v1 proxy) |
| 03 | [Beaten Map](03_beaten_map.md) — net-face 6-zone goals-against grid | Goalie / Coach | Yes — `metrics.beaten_location` (v15) | ✅ **shipped** (catalog backfill in progress) |
| 04 | [Film Session Agenda](04_film_session_agenda.md) — auto-curated teaching clips + talking points | Coach | v1 none; v2 Gemini points | ✅ **shipped** (v1 frontend) |
| 05 | [Recruiting Profile](05_recruiting_profile.md) — shareable verified stats + highlight reel | Family / Goalie | public share endpoint (deferred) | ✅ **shipped** (in-app; public share deferred) |
| 06 | [Ask the Film](06_ask_the_film.md) — conversational query over clips | All | new Gemini query endpoint | ⬜ specced |
| 07 | [Coach Review Queue](07_coach_review_queue.md) — triage/confirm/correct + drill assignment | Coach | reuse submit-feedback + save-json | ⬜ specced |
| 08 | [Season Report Card](08_season_report_card.md) — A–F pillar grades + trend | Goalie / Family | No (frontend) | ⬜ specced |
| 09 | [Parent Digest](09_parent_digest.md) — plain-language progress summary | Family | v1 none; v2 Gemini | ⬜ specced |
| 10 | [Highlight & Goals-Against Reels](10_highlight_reels.md) — one-click shareable reels | All | v1 none; v2 stitched MP4 | ⬜ specced |
| 11 | [Verified-by-Coach Badge](11_verified_by_coach_badge.md) — trust mark from the feedback loop | All | reuse feedback / save-json | ⬜ specced (cross-cutting) |
| 12 | [Confidence Triage](12_confidence_triage.md) — model confidence as a first-class signal | All | No (frontend util) | ⬜ specced (cross-cutting) |

## Dependency map
- **12 (Confidence Triage)** is the shared model-confidence layer; **07 (Review Queue)** orders by it and **many analytics** (01/02/03/05/08) should adopt its `utils/confidence.ts` + "high-confidence only" toggle.
- **11 (Verified Badge)** is the human-verification complement to 12; it's produced by **07** and surfaced on ClipCard/StatsDashboard/RecruitingProfile.
- **09 (Parent Digest)** builds on **08 (Report Card)** aggregates. **10 (Reels)** reuses **05/04** scoring + the slug-scoped signed-URL security from **05**.
- **06 (Ask the Film)** is standalone (flagship AI), and can route certain questions to 01/02/03 computations.

## Recommended build order for 06–12
1. **12 Confidence Triage** — small, frontend, and unlocks 07 + improves every analytics surface.
2. **11 Verified-by-Coach Badge** — small; turns existing feedback into a trust signal.
3. **07 Coach Review Queue** — depends on 12+11; the coach workflow that generates verification + drill assignments.
4. **08 Season Report Card** → **09 Parent Digest** — quick frontend wins for goalie/family.
5. **10 Highlight Reels** — quick MVP (playlist), heavier Phase-2 export.
6. **06 Ask the Film** — the flagship; largest (Gemini query layer) but the biggest differentiator.

## Built-feature code (web app)
Leak Finder `utils/leakAnalysis.ts`+`utils/drillLibrary.ts`; HDSV% `hooks/useDangerSplits.ts`+`DangerSavePanel.tsx`; Beaten Map `BeatenMap.tsx` (+ `metrics_seg` v15 `beaten_location`); Film Session `utils/filmSession.ts`+`FilmSession.tsx`; Recruiting Profile `utils/recruitingProfile.ts`+`RecruitingProfile.tsx`.
