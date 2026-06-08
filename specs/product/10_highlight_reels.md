# One-Click Highlight & Goals-Against Reels

> Two auto-compiled reels per game (and per season) — a **Best-Saves** reel for morale and sharing, and a **Goals-Against** reel for teaching — played back instantly as an auto-advancing playlist through the existing `VideoPlayer`. One click, no editing, no new video objects in the MVP.

---

## 1. Summary & problem

A goalie family using Goalie Analytics Pro can already drill into a single game, build a recruiter profile, or assemble a film session — but every one of those is a multi-step, deliberate workflow. After a Saturday game, the things a goalie, parent, or coach actually want are simpler and more immediate:

- "Let me watch all my good saves from today" (morale, sharing to the group chat).
- "Show me only the goals I let in so we can talk through them" (a 3-minute teaching loop).

Today that requires manually hunting clips, opening each in `ClipModal`, and remembering which were saves vs goals. There is no one-click "play my saves" or "play the goals against."

**One-Click Reels** turns each game (and the season) into two pre-built, auto-advancing playlists. It is the lightest-weight feature in the suite: no curation, no agenda, no publish step — press **Best Saves** or **Goals Against** and a playlist plays end to end, advancing clip ranges through the player we already ship. The reel is also a **shareable artifact** (a copyable link that re-opens the same playlist; Phase 2 exports a stitched MP4).

### Differentiation (do NOT duplicate these)

| Feature | Intent | Editorial | Output |
|---|---|---|---|
| **RecruitingProfile** (`utils/recruitingProfile.ts`, spec 05) | External marketing to recruiters | Curated, hides weaknesses, consent-gated, published public page | Polished profile + best-saves reel |
| **FilmSession** (`utils/filmSession.ts`, spec 04) | Coaching agenda | Themed, ordered, per-clip talking points | Teaching agenda |
| **FavoriteClips** | Personal keepsakes | Fully manual | Hand-picked list |
| **One-Click Reels** (THIS) | Instant watch + light share | **Zero-config, automatic, both saves AND goals-against** | Auto-advancing in-app playlist; shareable link |

This feature is deliberately *dumber and faster*: no bio, no agenda, no consent flow, no recruiter audience. It is the "tap and watch" companion to those heavier features. It reuses RecruitingProfile's best-saves scoring math and FilmSession's per-game spread/de-dupe logic rather than reinventing them.

---

## 2. Target users

- **Goalies** (primary): watch their own saves for morale; review goals-against for self-coaching; share the saves reel.
- **Families** (primary sharers): one tap to send "today's saves" to relatives / the team chat.
- **Coaches** (teaching): open the goals-against reel and walk through it on a tablet; spread-across-games season reel for trend talks.

All three are already-authenticated Pro users. There is no unauthenticated consumer in the MVP (that is RecruitingProfile's job); the Phase-2 share link reuses spec 05's slug-scoped security model so a shared reel still never exposes full-game video.

---

## 3. Data inputs (exact fields)

All inputs already exist in the published analysis JSON (`gs://goalie_video_bucket/analyze_video/04-final_video/<vID>.json`, an array of `{type, response}` items, normalized client-side in `services/api.ts`) and the customer config (`customerID/<custID>.json`, game records keyed by `vID`).

### Per clip (from each game's `clips` items)
- `clipID`, `clip_start_time`, `clip_end_time`, `clip_duration` — playback range + length capping.
- `clipSave` (bool), `clipSaveCount` — best-saves candidacy + score weight.
- `clipHasGoal` (bool) — goals-against candidacy; **hard-excluded** from the saves reel.
- `clipSave` interplay: a clip can have both a save and a goal (mixed sequence) — see §8.
- `goalie_positioning.goalie_position_confidence_score` — quality gate + score.
- `goalie_positioning.depth_rank`, `cover_angle_rank`, `squareness_rank` — rank-quality scoring.
- `coaching_feedback.rebound_control_rank` — teaching value for goals-against ordering.
- `coaching_feedback.actionable_coaching_feedback` — optional caption shown beside a goal clip.
- `coaching_feedback.coaching_confidence_score` — score tie-break / gate.
- `metrics.saves`, `metrics.goals` — fallbacks when the booleans are absent/ambiguous.
- `metrics.beaten_location` (string, goal clips only, v15) — secondary ordering / grouping of the goals-against reel. Optional dependency: degrade gracefully if absent.

### Per game / season (from customer config `games[]`)
- `vID`, `eventName`, `event_date` (and `eventSeason` if present) — reel scoping, labels, season grouping.

### Caveats (carry into computation)
- **`threat_type` is UNRELIABLE** — never use it for selection, ordering, or display.
- **Ranks are qualitative ordinals** — map to a numeric scale before any comparison; reuse the existing `qual()` mapping from `recruitingProfile.ts`/`filmSession.ts` (`excellent→3, good→2, average/fair→1, poor/weak→0`, unknown→null).
- **`beaten_location` is descriptive, not a severity score** — use it only to *group/spread* the goals-against reel, never to rank "worse" goals.

---

## 4. Pipeline / backend

### MVP — no new objects, no pipeline change
The MVP is **pure frontend**. A reel is just an ordered array of `{vID, clipID, start, end}` computed client-side from already-fetched analysis JSON. Playback reuses the existing private-video path: `VideoPlayer` requests a short-lived V4 signed URL from the Express server (`GET /api/video-url?uri=...`, prefix-locked to `analyze_video/00-segement-video-upload/full_<vID>.mp4` in `server.ts`) and seeks to `[start, end]`. No new GCS objects, no detection-pipeline work, no gateway change.

- A reel **spanning multiple games** (season reel) simply changes `videoId` between clips; `VideoPlayer` already remounts the `<video>` on `videoId` change and re-fetches a signed URL per game.
- Optional **persistence of a curated reel** (if the user reorders/removes) uses the gateway `save-json` handler that already exists (the same one `services/api.ts` calls for ground truth), writing a small reel doc keyed by `{custID, vID|season, kind}`. This is optional even within MVP — the default reels are deterministic and need no storage.

### Phase 2 — server-stitched exported MP4 + shareable link
A "Download MP4 / Share" action produces a single stitched file:
- Server-side `ffmpeg` concatenates the clip ranges (`-ss start -to end` per source `full_<vID>.mp4`, re-encoded to a uniform codec/resolution, then `concat`) into one MP4 written to `gs://goalie_video_bucket/reels/<slug>/<kind>.mp4`.
- **Security — reuse spec 05's slug-scoped, short-TTL design.** A shared reel link is `/r/<slug>` where `<slug>` is an unguessable 128-bit token that *is* the bearer credential. The public read path:
  - Serves ONLY the stitched reel object for that slug (server resolves the object from the slug doc — the client **never names an object**, never passes `uri`/`vID`).
  - Mints a **5-minute** V4 signed URL (not the 6h used for in-app playback), rate-limited per-IP and per-slug, CORS-locked to the Pro UI origin — exactly the controls in spec 05 §4.2.
  - **Never exposes `full_<vID>.mp4`.** Because the shared object is a *trimmed, stitched* file, a leaked URL cannot reveal full-game footage, other players, or (for a saves reel) any goals-against. This is the key reason Phase 2 stitches rather than sharing raw-clip signed URLs.
- The slug→reel mapping doc is persisted via gateway `save-json` (e.g. `reels/<slug>.json`), holding `{slug, custID, kind, scope, clips[], status, createdAt}`. `status: unpublished` → instant takedown (link 404s), mirroring spec 05.

---

## 5. Computation / logic

Reuse existing helpers (`qual`, `toNum`, per-game spread) from `recruitingProfile.ts` and `filmSession.ts`; do not fork the rank vocabulary.

### 5.1 Best-Saves reel
Candidate set: `clipSave === true` AND `clipHasGoal !== true` (a goal-against can never appear in a saves reel). Score each candidate (this mirrors `recruitingProfile.ts` so the two stay consistent):

```
score = 0.40 · min(clipSaveCount, 3)/3                     # save volume / difficulty proxy
      + 0.25 · posQuality                                   # mean ordinal of cover_angle + squareness, /3
      + 0.20 · norm(goalie_position_confidence_score)
      + 0.15 · norm(coaching_confidence_score)
```
- **Quality gate:** drop clips with `goalie_position_confidence_score` below a floor (reuse `CONF_FLOOR = 2`) — don't headline an unsure "save".
- **Order:** descending `score` (the morale reel leads with the best save).
- **Caps:** top **N = 8** clips (max 12); total length **≤ 90s** — drop lowest-scoring clips until under cap. Per-game cap of 3 for season reels.

### 5.2 Goals-Against reel
Candidate set: `clipHasGoal === true` (or `metrics.goals > 0` fallback). This reel is for teaching, so it does **not** apply the saves quality gate — every goal is worth reviewing. Order by **teaching value**, not "worst goal":

```
teachValue = 0.5 · reboundProblem      # reboundOrd low (uncontrolled/high-danger) → higher
           + 0.3 · positioningProblem  # mean ordinal of depth/angle/squareness, inverted
           + 0.2 · confidence          # prefer goals the model is sure about for clear teaching
```
Reuse `reboundOrd()` from `filmSession.ts`. Tie-break / secondary spread by `beaten_location` so the reel covers *different* beaten spots (glove-high, five-hole, etc.) rather than five of the same — grouping only, never severity ranking. If `beaten_location` is absent, fall back to chronological order within the game.
- **Order:** chronological within a single game (so a coach reviews the game as it unfolded) by default, with an optional "by teaching value" toggle; for season reels, order by `teachValue` and spread across games.
- **Caps:** all goals in a single-game reel (typically few); season reel capped at **N = 10**, **≤ 180s** (teaching tolerates longer than the morale reel). Optional per-goal caption from `actionable_coaching_feedback`.

### 5.3 Shared rules (both reels)
- **De-dupe:** drop clips of the same game within ±2s of an already-picked clip's `clip_start_time` (reuse FilmSession's overlap guard).
- **Per-game spread (season reels):** cap clips per `vID` so one big night doesn't dominate (saves cap 3/game; goals leave as-is per game but balance across games when over the season cap).
- **Length capping:** sum `clip_duration` (fallback `clip_end_time - clip_start_time`, then a default of 8s); drop lowest-priority clips until under the cap, but always keep at least one clip if any candidate exists.
- **Reel = ordered `{vID, clipID, eventName, eventDate, start, end, kind, score|teachValue}[]`** — the exact playlist `VideoPlayer` advances through.

---

## 6. Frontend

### Surfaces
- **Per-game:** two buttons in `StatsDashboard` (and/or the game header): **▶ Best Saves (n)** and **▶ Goals Against (n)**. Counts come straight from the candidate sets; a button is disabled with a tooltip when its count is 0 (§8).
- **Per-season:** the same two buttons on the season/all-games view, computing across every game's clips.
- **Reels view:** clicking a button opens a focused player overlay (a thin wrapper, e.g. `ReelPlayer`) rendering the existing `VideoPlayer` with the current clip's `videoId/start/end` and wiring `onNext`/`onPrev` to advance the playlist. `VideoPlayer` already loops a clip and exposes `SkipForward`/`SkipBack`; the wrapper just supplies the handlers and auto-advances on clip end.

### Auto-advance
`VideoPlayer`'s `onNext`/`onPrev` props already render the skip buttons. The wrapper:
- Passes `onNext`/`onPrev` to step the playlist index (remount happens automatically via the `videoId`/`start` change).
- Auto-advances when a clip reaches its end. `VideoPlayer` currently *loops* a single clip; add a minimal `autoAdvance`/`onClipEnd` prop so that at end-of-clip in reel mode it fires `onNext` instead of looping (last clip stops or loops the whole reel — chosen by a Loop toggle). This is the only `VideoPlayer` change required for the MVP.

### Share / copy
- **MVP:** "Copy link" copies a deep link that re-opens the same reel in the app for another logged-in user (e.g. `/app?reel=best_saves&vID=<vID>` or `&season=<s>`); also a "Copy clip timestamps" fallback.
- **Phase 2:** "Download MP4" and "Share public link" (`/r/<slug>`) appear here, backed by §4 stitching.

### ASCII mockup

```
┌─ StatsDashboard ── 2026-02-14 vs Bandits ───────────────────────────┐
│  Shots 31 · Saves 28 · GA 3 · Sv% 90.3                               │
│                                                                      │
│   [ ▶  Best Saves (8) ]      [ ▶  Goals Against (3) ]   [ Season ▾ ] │
└─────────────────────────────────────────────────────────────────────┘
        │ click
        ▼
┌─ REEL ── Best Saves · vs Bandits ─────────────────────────────  ✕ ─┐
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   [ ▶  VideoPlayer ]                          │  │
│  │                                                  ⏱ 0:06        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│   ⏮  clip 3 / 8  ⏭        ●━━━━━━━━━━━━━━━━━━━     1.0x  🔁  🔇      │
│   thumbs:  [■][■][▣][■][■][■][■][■]                                  │
│   "Glove save, top-right — squareness Good"  (caption, goals only)   │
│   ───────────────────────────────────────────────────────────────  │
│   [ Copy link ]   [ Download MP4 (Phase 2) ]   auto-advancing ✓     │
└─────────────────────────────────────────────────────────────────────┘
```

### UX notes
- Best-Saves opens unmuted-by-default? No — `VideoPlayer` defaults muted; keep that (autoplay policy) and surface the unmute control.
- Goals-Against shows the optional coaching caption under the scrubber; Best-Saves shows none (morale, not critique).
- A small "n of m clips · total m:ss" readout sets expectations.

---

## 7. Edge cases & limitations

- **No saves in the game.** Best-Saves button disabled with tooltip "No qualifying saves" (all saves below the confidence floor, or none detected). Never open an empty player.
- **No goals against (shutout).** Goals-Against button disabled with a celebratory tooltip "Shutout — no goals against!" Surface it as a positive, not an error.
- **All-goals / blowout game.** Goals-Against reel could be long; the 180s season cap and a single-game "show all goals" note keep it bounded. Best-Saves still applies its 90s cap.
- **Mixed clip (save AND goal in one sequence).** `clipHasGoal === true` wins: the clip is excluded from Best-Saves and included in Goals-Against (we never put a goal-against in a morale reel). Note this rule in code comments.
- **Low-confidence saves.** Gated out of Best-Saves (don't headline an unsure save); goals are NOT gated (every goal is teachable).
- **Length caps vs. few clips.** Always keep ≥1 clip if any candidate exists; never produce a 0-length reel from a non-empty candidate set.
- **Cross-game signed URLs (season reels).** Each game mints its own signed URL on demand; a slow first-clip-per-game load is expected. Pre-fetch the next clip's signed URL on `onNext` hover to hide latency.
- **Missing `beaten_location`.** Goals-Against falls back to chronological ordering; never block on it.
- **Stale analysis.** Reels recompute from current JSON each open (MVP has no snapshot), so they always reflect the latest analysis — acceptable for an ephemeral watch tool.
- **`threat_type`** — never displayed or used.

---

## 8. Phasing & effort

| Phase | Scope | Effort |
|---|---|---|
| **MVP** | Per-game + per-season Best-Saves & Goals-Against buttons in `StatsDashboard`; §5 selection (reuse `recruitingProfile`/`filmSession` helpers); `ReelPlayer` wrapper + minimal `VideoPlayer` `autoAdvance` prop; copy-link share; empty/shutout states. No new objects. | **S / M** |
| **Phase 2** | Server-stitched exported MP4 (`ffmpeg` concat) + public `/r/<slug>` share link reusing spec 05's slug-scoped, 5-min-TTL, rate-limited signed-URL security; reel persistence + light reorder/remove; thumbnails. | **L** |
| **Phase 3** | Captions/branding overlay on exported MP4, auto-share to socials, season "best of" auto-compile. | **M+** |

---

## 9. Success metrics

- **Reel opens per game** (Best-Saves vs Goals-Against split) — the core "tap and watch" activation.
- **Completion rate** — % of opened reels watched to the last clip (does the playlist hold attention?).
- **Saves-reel shares** — copy-link clicks (MVP) / `/r/<slug>` opens (Phase 2); unique recipients per reel.
- **Goals-reel coach usage** — opens on the season/teaching view; clips re-watched (rewind events).
- **Time-to-watch** — seconds from game open to first reel clip playing (target: one tap, <3s incl. signed-URL fetch).
- **Phase 2 export adoption** — MP4 downloads / share links created per active family.

---

## 10. Open questions

- **`VideoPlayer` change scope.** Add a thin `autoAdvance` + `onClipEnd` prop (recommended, minimal), or build playlist logic entirely in the wrapper by detecting end via `onReadyStateChange`/timeupdate? Confirm the cleanest seam given `handleNativeEnded`'s current loop behavior.
- **Default goals-against ordering** — chronological (replay the game) or by teaching value? Proposal: chronological default with a toggle. Validate with a coach.
- **Should mixed save+goal clips appear in BOTH reels** (save shown in Best-Saves, goal in Goals-Against) or strictly goals-only? Current proposal: goal wins, excluded from saves. Confirm.
- **MVP persistence** — ship default deterministic reels only, or allow reorder/remove + `save-json` persistence in MVP? Proposal: defaults only in MVP; persistence with Phase 2.
- **Phase-2 stitch cost/latency** — `ffmpeg` re-encode is CPU-heavy on Cloud Run; run as a job (like the worker) and notify when ready, or synchronous for short reels? Likely async job for >~30s reels.
- **Season reel scope** — all games, or filter by `eventSeason`/date range from the existing FilterBar? Reuse FilterBar selection.
- **Confidence floor for saves** — is `CONF_FLOOR = 2` right for a *morale* reel (looser) vs a recruiter reel (stricter)? Possibly relax for this feature.
