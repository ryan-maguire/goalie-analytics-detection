# Recruiting / Showcase Profile

> One shareable link that turns a goalie's verified season into a recruiter-ready profile: verified stats + an auto-cut highlight reel of best saves + bio. Built by the family, consumed by recruiters and scouts.

---

## 1. Summary & problem

Recruiting is where goalie families spend real money — camps, showcases, advisors, edited highlight tapes (often $300–$1,500 per cut, redone every season). Today a family using Goalie Analytics Pro has *better* data than a paid highlight editor (per-clip save detection, positioning ranks, coaching confidence) but no way to package it for a college coach or scout. They export to YouTube and lose all the verified context.

The **Recruiting / Showcase Profile** packages what we already compute into a single, clean, shareable artifact:

- **Verified season stats** (games, shots faced, save%, positioning quality) — provenance-stamped "verified by Goalie Analytics Pro" so a recruiter trusts the number instead of a parent's spreadsheet.
- **Auto-cut highlight reel** of the goalie's best saves, selected from `clipSave` + confidence + positioning rank quality — no manual editing.
- **Bio / metadata** (name, team, season, position).

This is the clearest path to **direct revenue from families** (the audience that already pays for recruiting). It is also the feature with the sharpest **security constraint**: the highlight video lives in a PRIVATE bucket, but the consumer (a recruiter) is **unauthenticated**. Serving private-bucket clips to an anonymous public page safely is the core engineering problem this spec solves.

### Differentiation from PathToPro

PathToPro is an **internal development pathway** — it shows the *goalie* where to improve, surfaces weaknesses, and is private. The Recruiting Profile is the opposite: an **external, polished, recruiter-facing marketing artifact** that hides weaknesses, shows only best saves, and is shared via a public link. They share data sources but have opposite audiences and opposite editorial intent. Do not merge them; cross-link them ("Ready to show coaches? Publish a Recruiting Profile" from PathToPro).

---

## 2. Target users

| Role | Relationship to feature |
|------|------------------------|
| **Goalie / family** (primary, builder) | Authenticated Pro user. Curates and publishes the profile, picks bio fields, optionally trims the auto-selected reel, hits "Publish", gets a link/PDF to send. **Pays.** |
| **College recruiter / scout** (consumer) | **Unauthenticated.** Receives a link (`/p/<slug>`) or PDF. Views stats + reel. Never logs in, never sees raw game footage, never sees coaching weaknesses. |
| **Goalie coach / advisor** (secondary consumer) | Same as recruiter — receives the public link. |

The builder is the customer; the recruiter is the audience the customer is trying to impress. Every design choice optimizes for "a busy D1 coach opens this on a phone for 20 seconds and trusts it."

---

## 3. Data inputs (exact fields)

All inputs already exist in the published analysis JSON (`gs://goalie_video_bucket/analyze_video/04-final_video/<vID>.json`, an array of `{type, response}` items) and the customer config (`customerID/<custID>.json`, a list of game records keyed by `vID`).

### Profile metadata / bio (from customer config game records + summary)
- `goalie_name`, `goalie_team`, `eventSeason` / `goalie_season`, `event_date`, `eventName`, `opposition_team`
- `coaches_overall_rating` (summary.response) — optional headline rating

### Season stat profile (per game, then aggregated)
From each game's `clips` items, `clip.metrics`:
- `metrics.shots`, `metrics.shotsOnNet`, `metrics.saves`, `metrics.rebounds`, `metrics.goals`
- `metrics.observed_goalie_side`, `metrics.goal_criteria` (for QA / criteria display only)

From `clip.goalie_positioning`:
- `depth_rank`, `cover_angle_rank`, `squareness_rank`, `goalie_position_confidence_score`

From `clip.coaching_feedback`:
- `rebound_control_rank`, `coaching_confidence_score`

### Highlight-reel selection (per clip)
- `clipID`, `clip_start_time`, `clip_end_time`, `clip_duration`
- `clipSave` (bool), `clipSaveCount`, `clipHasGoal` (bool — **exclude any true**)
- `goalie_position_confidence_score`, `coaching_confidence_score` (tie-break / quality gate)
- `depth_rank`, `cover_angle_rank`, `squareness_rank` (rank quality → reel ordering)
- `vID` of the source game (to resolve the private video object)

### Caveats (carry into computation)
- **`threat_type` is NOT reliable** — do not use it for selection or display.
- **Ranks are qualitative ordinals** — map to a numeric scale before any aggregate (§6).
- **Aggregate save%** = `sum(saves) / sum(shotsOnNet)` across games — define precisely, never average per-game percentages.
- **Danger-weighting is an OPTIONAL dependency** on spec 02's `shot_danger` field. If absent, ship plain save%; never block on it.

---

## 4. Pipeline / backend changes

No detection-pipeline changes. Three additions across the UI Express server and the Flask gateway.

### 4.1 Profile document (persisted via gateway `save-json`)
The builder publishes a self-contained profile doc so the public page **never recomputes from raw analysis** and never needs config access:

```
gs://goalie_video_bucket/profiles/<slug>.json
```

`<slug>` = unguessable token (e.g. `r-3f9a2c7e8b1d` — 128-bit random, base32, prefixed). The slug IS the access credential; treat it like a bearer token.

> **Reuse note:** the existing `save-json` handler upserts by `vID` into `<clientID>.json`. Either (a) extend it to accept a `docPath`/`kind: "profile"` so it writes to `profiles/<slug>.json` instead of `<clientID>.json`, or (b) add a sibling `save-profile` / `get-profile` route. Prefer a thin sibling route so the public read path stays minimal and cannot be tricked into returning a customer config.

Profile doc shape (frozen snapshot — see §6 provenance):
```json
{
  "slug": "r-3f9a2c7e8b1d",
  "clientID": "<custID>",
  "status": "published",            // draft | published | unpublished
  "createdAt": "2026-06-07T...Z",
  "publishedAt": "...",
  "bio": { "goalieName": "...", "team": "...", "season": "...",
           "gradYear": 2027, "position": "G", "heightWeight": "...",
           "headline": "..." },
  "stats": { "games": 8, "totalShots": 412, "totalShotsOnNet": 388,
             "totalSaves": 351, "savePct": 0.9046,
             "dangerWeightedSavePct": null,
             "positioning": { "depth": 2.4, "coverAngle": 2.1, "squareness": 2.6 },
             "computedAt": "...", "engineVersion": "metrics_seg v14.1" },
  "reel": [
     { "vID": "<vID>", "clipID": "...", "start": 123.4, "end": 130.9,
       "saveCount": 1, "score": 0.83 }
  ],
  "consent": { "videoConsent": true, "minorConsentBy": "parent@...",
               "acceptedTermsAt": "..." }
}
```

### 4.2 PUBLIC, no-auth, rate-limited signed-URL endpoint (the hard part)
The existing `/api/video-url` mints a 6h V4 signed URL but is **prefix-locked only to the full-game upload prefix** (`analyze_video/00-segement-video-upload/full_<vID>.mp4`) and currently has no auth and no rate limit. A public recruiting page must NOT reuse it as-is: it would hand any anonymous visitor a 6-hour URL to the **entire raw game video**, exposing footage outside the highlight clips (other players, weaknesses, goals against).

Add a dedicated, tightly-scoped endpoint:

```
GET /api/p/:slug/clip-url?clipID=<id>
```

Hard requirements:
1. **Slug-gated, no broad input.** Caller supplies only `slug` + `clipID`. The server loads `profiles/<slug>.json` (must be `status: "published"`), then verifies `clipID` is in that profile's `reel[]`. **No `uri`/`vID` query param is accepted** — the client can never name an object. This is the key difference from `/api/video-url`.
2. **Resolve object server-side.** From the matched reel entry the server knows `vID` → object `full_<vID>.mp4`. The signed URL is for THAT object only.
3. **Range-locked, short TTL.** TTL = **5 minutes** (not 6h). The player re-requests per clip; a leaked URL dies fast. Note: V4 read signed URLs cannot restrict byte ranges, so the URL still grants the whole `full_<vID>.mp4` for 5 min — see §8 mitigation (highlight-only re-encoded objects) and §11 open question.
4. **Rate limiting.** Per-IP token bucket (e.g. 30 req/min) AND per-slug ceiling (e.g. 300 url-mints/hour) to blunt scraping/enumeration. Slugs are unguessable so enumeration is already hard; the limit caps damage if a slug leaks.
5. **No CORS wildcard escalation.** Lock `Access-Control-Allow-Origin` to the public Pro UI origin for this route.
6. **Audit.** Log `{slug, clipID, ip, ts}` for takedown forensics and recruiter-view analytics (§10).

### 4.3 Optional: PDF export
Server-side render of the profile (stats + bio + QR to the live link + thumbnail frames) to a static PDF stored at `gs://…/profiles/<slug>.pdf`. Phase 2. The PDF carries no playable video — only the link/QR — which sidesteps the video-security problem for the offline artifact.

---

## 5. Computation / logic

### 5.1 Rank → ordinal map
Ranks are qualitative strings. Map to a 1–3 numeric scale (lower = better) for aggregation, e.g.:
```
elite/excellent → 1 ; good/adequate → 2 ; needs_improvement/poor → 3
```
Maintain ONE shared map (reuse the existing FilterBar/StatsDashboard rank vocabulary). Unknown/empty rank → excluded from the average (do not coerce to a default).

### 5.2 Season stat aggregation (pooled, never per-game mean)
Across all selected games for the season:
```
games            = count of games included
totalShots       = Σ metrics.shots
totalShotsOnNet  = Σ metrics.shotsOnNet
totalSaves       = Σ metrics.saves
savePct          = totalSaves / totalShotsOnNet        # guard /0 → null
totalGoals       = Σ metrics.goals
positioning.depth     = mean(ordinal(depth_rank))      over clips with a rank
positioning.coverAngle= mean(ordinal(cover_angle_rank))
positioning.squareness= mean(ordinal(squareness_rank))
```
> Display positioning as the qualitative band the mean falls in (e.g. mean 1.4 → "Elite"), not a bare decimal — recruiters read words, not 1.4.

### 5.3 Optional danger-weighted save%
Only if spec 02's `shot_danger` exists on clips:
```
dangerWeightedSavePct = Σ(weight_i · saved_i) / Σ(weight_i · onNet_i)
```
where `weight_i` from `shot_danger`. If the field is missing on ANY game in the set, omit the metric entirely (do not partially weight). Label clearly as "high-danger save%".

### 5.4 Highlight-reel auto-selection
Candidate set = clips where `clipSave === true` AND `clipHasGoal !== true` (**never include a goal against**). Score each:
```
score = w_save  · min(clipSaveCount, 3)/3
      + w_pos   · (1 - (ordinal_avg(positioning ranks) - 1)/2)   # better rank → higher
      + w_conf  · normalize(goalie_position_confidence_score)
      + w_ccnf  · normalize(coaching_confidence_score)
```
Suggested weights: save 0.40, pos 0.25, posConf 0.20, coachConf 0.15 (tunable; store with the doc).

Selection rules:
- **Quality gate:** drop clips with `goalie_position_confidence_score` below a floor (avoid showing a "save" the model wasn't sure about).
- **Rank, then cap:** take top N (default **8**, max 12) by score.
- **Length cap:** total reel ≤ **90s**; if exceeded, drop lowest-scoring clips until under cap (per-clip already bounded by `clip_duration`).
- **De-dupe / spread:** prefer spreading across games so the reel isn't one hot night.
- Persist the final ordered list as `reel[]` with `{vID, clipID, start=clip_start_time, end=clip_end_time, saveCount, score}`.
- Builder may manually remove/reorder before publish (cannot add a goal-against clip — UI hard-blocks `clipHasGoal`).

### 5.5 Verified-stats provenance
Snapshot at publish time: stamp `engineVersion` (e.g. `metrics_seg v14.1`), `computedAt`, and the list of source `vID`s into the doc. The public page renders from the frozen snapshot, so a later re-analysis can't silently change a published stat. Re-publishing recomputes and re-stamps. This snapshot is what backs the "verified by Goalie Analytics Pro" trust mark.

---

## 6. Frontend

Two surfaces: a **public consumer page** and a **private builder**.

### 6.1 Public route `/p/<slug>` (no auth)
Server-rendered or SPA route that reads `profiles/<slug>.json` via the public `get-profile` read. Mobile-first, fast, no app chrome. Highlight reel uses the existing **`VideoPlayer`** component driven by clip ranges (`start`/`end` from `reel[]`), but pointed at the new `/api/p/:slug/clip-url` endpoint instead of `/api/video-url` (add a prop/streamMode so VideoPlayer fetches the slug-scoped URL; reuse all its seek/range/clip-loop logic). If a slug is `unpublished`/missing → clean 404 ("This profile is no longer available").

```
┌──────────────────────────────────────────────────────────┐
│  GOALIE ANALYTICS PRO                  ✓ Verified Profile  │
├──────────────────────────────────────────────────────────┤
│   ●  Alex Rivera                                           │
│      Northstars 16U AAA · Goaltender · Class of 2027      │
│      2025–26 Season                                        │
│                                                            │
│   ┌───────── VERIFIED SEASON STATS ──────────┐            │
│   │  Games  Shots  Save %   High-Danger Sv%   │            │
│   │    8     388    90.5%        (n/a)         │            │
│   │  Positioning:  Depth Elite · Angle Good   │            │
│   │                Squareness Elite           │            │
│   └────────────────────────────────────────────┘          │
│                                                            │
│   HIGHLIGHTS  ▸ Best Saves (8 clips · 1:24)               │
│   ┌──────────────────────────────────────────┐           │
│   │              [ ▶  VideoPlayer ]            │           │
│   │   clip 1/8  ◀  ───●────────────  ▶        │           │
│   └──────────────────────────────────────────┘           │
│   thumbs: [■][■][■][■][■][■][■][■]                         │
│                                                            │
│   🔒 Stats & saves verified by Goalie Analytics Pro       │
│      from full-game video.   Engine v14.1                  │
│   [ Download PDF ]              powered by GAP             │
└──────────────────────────────────────────────────────────┘
```

Constraints on the public page:
- Renders ONLY highlight clips — never the full-game player, never coaching weaknesses, never goals-against, never other games' raw clips.
- "Verified" trust mark links to a short explainer of how stats are computed.
- No PII beyond what the builder consented to expose.

### 6.2 Private builder (authenticated, inside Pro UI)
New component `RecruitingProfileBuilder` reachable from nav and from a CTA in PathToPro.

```
┌─ Build Recruiting Profile ───────────────────────────────┐
│ Bio:  [Name][Team][Season][Grad Yr][Position][Headline]  │
│ Season stats (auto, verified):  8 GP · 90.5% · Elite pos │
│   [ ] include high-danger save% (unavailable this season)│
│ Auto highlight reel (8 of 23 eligible saves):            │
│   ☰ 1. vID a1 · 0:07 save (score .83)         [remove]   │
│   ☰ 2. vID b2 · 0:05 save (score .79)         [remove]   │
│   ☰ ...                              total 1:24 / 1:30    │
│ Consent: [✓] I consent to share this video publicly      │
│          [✓] Parent/guardian (athlete is a minor)        │
│ ────────────────────────────────────────────────────────│
│ [ Preview public page ]   [ Save draft ]   [ Publish ▸ ] │
│ Published link: https://…/p/r-3f9a2c7e8b1d  [copy][unpub]│
└──────────────────────────────────────────────────────────┘
```
- Pulls eligible games from the customer config; runs §5 computation client-side or via a compute call; previews exactly what recruiters see.
- "Publish" writes the doc (4.1) and flips `status: published`; "Unpublish" flips to `unpublished` (instant takedown). Drag-reorder reel; remove clips; cannot add a goal-against.

---

## 7. Edge cases & limitations

- **Minor consent / privacy.** Most goalies are minors. Publishing requires explicit parent/guardian consent checkbox, stored in `consent` block with timestamp. No publish without it. Surface this prominently.
- **Public video access control.** The slug is a bearer token — anyone with the link sees the reel. There is no per-recruiter auth. A leaked link = leaked highlights (acceptable: highlights are meant to be shared) but the 5-min TTL + slug-scoped endpoint means a leak does NOT expose the full game or other games. Document this clearly to families.
- **Whole-file signed URL exposure.** A signed clip URL still technically grants the full `full_<vID>.mp4` for 5 minutes (V4 can't byte-range-restrict). Mitigation: Phase 2 produces **separate highlight-only re-encoded clip objects** (`profiles/<slug>/clip_<n>.mp4`) so the signed URL only ever points at a trimmed file. Until then, short TTL + rate limit + unguessable slug are the controls. (See §11.)
- **Stale stats.** Published doc is a frozen snapshot. If a game is re-analyzed, the public number won't change until re-publish. Show "as of `computedAt`" and prompt the builder to re-publish when underlying data changes.
- **Small sample / no data.** If `games < 3` or `totalShotsOnNet` is tiny, save% is noisy. Show a "limited sample" note and require a games floor (e.g. ≥2) to publish; never display a 100%/0% from one shot.
- **No eligible saves.** If the candidate reel is empty (all saves low-confidence or only goals-against), block publish with guidance — don't ship an empty reel.
- **Takedown / unpublish.** Instant: flip `status`, public page 404s, clip-url endpoint refuses (only serves `published`). Also invalidate cached PDFs.
- **Divergent metrics.** `metrics.shots` ≥ `shotsOnNet`; only `shotsOnNet` drives save%. Make the denominator explicit on hover so a recruiter can't misread it.
- **`threat_type` unused** — never display it.

---

## 8. Phasing & effort

| Phase | Scope | Effort |
|-------|-------|--------|
| **MVP** | Private-link profile: builder UI + §5 aggregation + auto highlight reel; profile doc via save-profile; reel playback via slug-scoped clip-url endpoint (5-min TTL, rate-limited) over existing full-game objects; consent gate; "verified" mark. Link is shareable but page is minimal. | **M / L** |
| **Phase 2** | Public polished `/p/<slug>` page (mobile, trust mark, PDF export); **highlight-only re-encoded clip objects** (removes whole-file exposure); recruiter-view analytics; reorder/curate reel. | **L** |
| **Phase 3** | Recruiter-side features (save/compare profiles, contact goalie), advisor co-branding, monetized tiers. | **L+** |

---

## 9. Success metrics

- **Profiles created** and **published** (activation; published/created ratio).
- **Shares** — unique slug link opens, unique recruiter IPs per profile (from clip-url audit log).
- **Reel engagement** — % of viewers who play ≥1 clip, avg clips watched.
- **Conversion / upsell** — % of profile-creating accounts on a paid recruiting tier; revenue per published profile; renewal rate season-over-season (recruiting is annual).
- **Trust** — clicks on the "verified" explainer; recruiter-initiated contacts (Phase 3).

---

## 10. Open questions

- **Whole-file signed URL (security, top priority).** Accept 5-min full-file exposure for MVP, or block MVP on re-encoded highlight-only objects? Recommendation: ship MVP with short TTL + rate limit, fast-follow re-encoded clips in Phase 2. Confirm risk appetite with stakeholders.
- **Consent for minors** — is a checkbox sufficient, or do we need a verified parent email / e-sign? Legal review needed before public launch.
- **Slug lifetime & rotation** — can a family rotate a slug (invalidate an old shared link) without losing view analytics? Likely yes; design slug→profile as 1:many.
- **Rate-limit infra** — Cloud Run is stateless/multi-instance; per-IP token bucket needs shared state (Memorystore/Redis) or edge rate limiting (Cloud Armor). Which?
- **Public read path** — extend `get-json` or add a dedicated `get-profile` that can ONLY read `profiles/*` (recommended, to guarantee it can't leak a customer config)?
- **Danger-weighting** — is spec 02's `shot_danger` landing this season? Gates the headline "high-danger save%" stat recruiters care most about.
- **PDF fidelity** — static thumbnails + QR only, or animated/GIF preview? Affects render cost.
- **Branding** — is "verified by Goalie Analytics Pro" co-brandable with advisors/clubs (a revenue lever) or strictly ours (a trust lever)?
