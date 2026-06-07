# Product feature specs

Implementation-ready specs for net-new Goalie Analytics Pro features, derived
from the data the pipeline already produces (`analyze_video/04-final_video/<vID>.json`)
and the existing web app. Each spec is self-contained (data inputs, pipeline/
backend changes, computation, frontend + mockup, edge cases, phasing, success
metrics, open questions).

| # | Spec | Primary user | Pipeline change? | Effort | Build order |
|---|------|--------------|------------------|--------|-------------|
| 01 | [Leak Finder](01_leak_finder.md) — cross-game weakness patterns → drills | Goalie / Coach | No (pure frontend over existing JSON) | M | **1st** |
| 02 | [High-Danger Save %](02_high_danger_save_pct.md) — danger-weighted save% + splits | Goalie / Coach / Recruiter | v1 proxy = none; v2 adds `metrics.shot_danger` | S → M | **2nd** |
| 03 | [Beaten Map](03_beaten_map.md) — net-face 6-zone goals-against grid | Goalie / Coach | Yes — `metrics.beaten_location` (metrics_seg, goal clips only) | M | 3rd |
| 04 | [Film Session Agenda](04_film_session_agenda.md) — auto-curated teaching clips + talking points | Coach | New Gemini generation step (feedback_seg post-step or on-demand endpoint) | M | 4th |
| 05 | [Recruiting Profile](05_recruiting_profile.md) — shareable verified stats + highlight reel | Family / Goalie (recruiters consume) | New public signed-URL endpoint + profile doc | M/L | 5th |

## Cross-spec dependencies
- **02 → 01, 05:** the `shot_danger` field (spec 02 v2) lets Leak Finder segment leaks by danger and lets the Recruiting Profile show danger-weighted save%. Both work without it; both get better with it.
- **03** is an independent `metrics_seg` prompt addition (`beaten_location`, goal clips only) — needs a worker redeploy (`bash deploy/deploy.sh`) + a backfill of existing games.
- **04, 05** reuse the existing `VideoPlayer` (native `<video>` + telestration + signed-URL playback) and `ClipModal` for delivery.

## Recommended sequencing rationale
1. **Leak Finder** first — zero pipeline work, highest "prescriptive" value, exercises the cross-game analytics layer the others build on.
2. **High-Danger Save %** — ship the proxy fast for an immediate stat upgrade, then add the `shot_danger` field.
3. **Beaten Map** — first pipeline field; high "aha" value once `shot_danger` work has warmed up the metrics_seg prompt-editing path.
4. **Film Session** — AI generation step; coach-facing, reuses player.
5. **Recruiting Profile** — most surface area (public route + security for serving private video to unauthenticated recruiters); do last, leverage stats from 01/02.

Security note (spec 05): serving private-bucket highlight video on a public page is the hard part — read that spec's security section before building.
