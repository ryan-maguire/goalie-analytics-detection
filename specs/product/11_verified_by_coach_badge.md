# Verified by Coach ✓ — Trust Badge & Credibility Metric
*Turn the analytic feedback that coaches already give into a visible, aggregatable trust signal.*

## 1. Summary & problem
Goalie Analytics Pro already collects per-clip coach feedback via `POST /api/submit-feedback`
(`feedback_type ∈ "Accurate Analytic" | "Correction Analytic"`). Today that signal is used only to
re-color the thumbs buttons inside `ClipCard` (`isPositiveHighlighted` / `isNegativeHighlighted`) and to
fix data in the saved JSON. It is **invisible everywhere else** — a viewer (a recruiter, a parent, the
goalie) has no way to know that a real human coach looked at a clip and confirmed the engine's read.

This spec makes that human-verification status a **first-class, cross-cutting trust signal**: a small
reusable badge with three states — **✓ Coach-verified**, **✎ Coach-corrected**, **unreviewed** — surfaced
on every clip surface, and rolled up into a credibility metric *"X% of this season's clips are
coach-verified."* That metric is most valuable on the `RecruitingProfile` (where credibility is the entire
point) and in `StatsDashboard`.

This is **not a new page**. It is a derived state + a reusable component reused across existing surfaces. It
derives from data already collected; **no detection-pipeline change** is required.

**Relationship to spec 12 (model confidence triage):** spec 12 surfaces *the engine's* self-rated
confidence (`*_confidence_score`). THIS spec surfaces *a human's* verdict. They are orthogonal trust axes
and must be shown as **two distinct signals**, never merged (a high-confidence clip can still be wrong; a
coach-corrected clip carries human authority that overrides confidence). See §7 for co-display rules.

## 2. Target users
- **Recruiters / college coaches** viewing a shared `RecruitingProfile` — "are these numbers trustworthy?"
- **The goalie & family** — pride/credibility signal; knowing which clips a coach actually blessed.
- **The reviewing coach** — immediate visual confirmation that their review "took" and is reflected publicly.

## 3. Data inputs (exact fields)
Source of truth is the existing feedback document, read back via
`GET get-json?id=<vID>_feedback` → `FeedbackRecord[]`. Current shape (`goalie-analytics-pro-ui/types.ts`):

```ts
export interface FeedbackItem { attribute: string; suggested_value: string; }
export interface FeedbackRecord {
  clip_id: string;
  video_id: string;
  feedback_type: 'Accurate Analytic' | 'Correction Analytic';
  timestamp: string;          // ISO 8601, e.g. new Date().toISOString()
  general_feedback: string;
  feedback_items: FeedbackItem[];
}
```
Clip identity: `ClipDetail.clipID` (`feedbackMap` is keyed by `clip_id` in `App.tsx`).

**Per-clip analysis context (read-only, for co-display with spec 12, NOT inputs to verification):**
`goalie_position_confidence_score`, `coaching_confidence_score`, `beaten_location_confidence`.

**NEW state to introduce (all client-derived; no schema migration required for the MVP):**
- `VerificationState = 'verified' | 'corrected' | 'unreviewed'` (derived per clip, §6).
- `coachVerifiedPct` (derived aggregate per game / per season, §6).
- **OPTIONAL** (Phase 2, see §5): a `coach_id` / `coach_name` field added to new `FeedbackRecord`s to
  support multi-coach attribution and de-duplication. Backward-compatible: absent on legacy records.

**Existing CAVEATS to respect:** ranks (`depth_rank`, etc.) are qualitative; `threat_type` is unreliable —
the badge must never imply those specific fields are "correct," only that *a coach reviewed this clip's
analysis as a whole*. Badge copy uses "analysis," not field-level claims.

## 4. — (folded into §3/§5)

## 5. Pipeline / backend
**No detection-pipeline change.** All inputs already exist.

**Derivation source — two supported paths (MVP uses Path A):**

**Path A — derive from existing feedback records (default, zero new endpoints).**
The verification state of a clip is computed from its `FeedbackRecord`(s) in `<vID>_feedback`.
The submit + read endpoints already exist (`/api/submit-feedback`, `get-json?id=<vID>_feedback`).
> ⚠️ **Known gap to fix:** `App.tsx` builds `feedbackMap[f.clip_id] = f` by last-write-wins in **array
> order**, not by `timestamp`. For a correct "latest action wins" rule (§6) this MUST become a
> timestamp-max reduction (see §6). This is a small frontend fix, not a backend change.

**Path B — dedicated verification doc (optional, Phase 2 hardening).**
If we later need server-authoritative, multi-coach, or signed verification, persist a compact
`<vID>_verification` doc via the existing `POST /api/save-json` (path-addressed; no new route):
```jsonc
{ "video_id": "<vID>",
  "clips": { "<clipID>": { "state": "verified", "coach_id": "...", "timestamp": "..." } } }
```
Public/aggregation read path stays uniform: `GET /api/get-json?id=<vID>_verification`. Path B is **only**
needed if Path A's client derivation proves insufficient (e.g., abuse, attribution). Ship Path A first.

**Read path the public/aggregation uses:** identical to today — `fetchClipFeedback(vID)` in
`services/api.ts` (`GET <API_BASE>?id=<vID>_feedback`). RecruitingProfile/StatsDashboard aggregate over the
already-loaded `clipFeedbackMap` (per game) or fan-out per `vID` for season rollups.

## 6. Computation / logic
**Per-clip state (`deriveVerification(records: FeedbackRecord[]): VerificationState`):**
1. Filter records to `clip_id === clip.clipID`.
2. If none → `'unreviewed'`.
3. Pick the record with the **maximum `timestamp`** (ISO strings compare lexicographically; tie-break:
   later position in array). **Latest coach action wins** — recency is precedence.
4. Map: `'Accurate Analytic'` → `'verified'`; `'Correction Analytic'` → `'corrected'`.

This makes the model robust to the real workflows in §8: corrected-then-reconfirmed → `verified`;
confirmed-then-corrected → `corrected`.

**Aggregate ("X% coach-verified"):**
- Denominator = clips eligible for review. **MVP denominator = all clips in scope** (per game or per
  season). Document the choice in tooltip copy ("of all clips").
- Numerator = clips whose derived state === `'verified'`.
- `coachVerifiedPct = round(100 * verifiedCount / max(totalClips, 1))`.
- Also expose `correctedCount` and `unreviewedCount` for richer copy.
- **Season rollup** = sum of per-game numerators / sum of per-game denominators (**pooled**, not a mean of
  per-game percentages — consistent with the project's pooled-metric convention).

**Precedence / recency rules summary:**
- Newer `timestamp` always overrides older for the same `clip_id`.
- A `'corrected'` clip is *not* counted as verified even if an older `'Accurate Analytic'` exists.
- A re-confirmation (newer `'Accurate Analytic'`) flips a previously corrected clip back to `'verified'`.

Implement as a pure helper in `utils/` (e.g. `utils/verification.ts`) so all surfaces share one definition.

## 7. Frontend
**Reusable component:** `components/VerifiedBadge.tsx`
```ts
type VerifiedBadgeProps = {
  state: VerificationState;
  size?: 'sm' | 'md';
  showLabel?: boolean;     // sm chips may icon-only with title=; md shows text
};
```
Reuse `lucide-react` icons already in the bundle: `BadgeCheck` (✓), `PencilLine`/`Pencil` (✎), and a muted
`Circle`/dash for unreviewed. Match the existing emerald "verified" palette used in `RecruitingProfile`
(`text-emerald-300/500`) so coach-verified reads consistently across the app.

**Badge states — ASCII mockup:**
```
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ ✓ Coach-verified     │   │ ✎ Coach-corrected    │   │ ○ Unreviewed         │
 └──────────────────────┘   └──────────────────────┘   └──────────────────────┘
   emerald bg/border          amber bg/border             slate-400, low emphasis
   tooltip: "A coach          tooltip: "A coach           tooltip: "No coach has
   confirmed this clip's      submitted a correction      reviewed this clip yet."
   analysis."                 to this clip's analysis."
```

**Surfaces:**
- **`ClipCard`** — small chip in the metadata row (near the existing rank badges / duration pill). Derive
  from the `existingFeedback` prop already passed in (extend to consider latest-by-timestamp via the §6
  helper). Distinct from the existing thumbs buttons: thumbs are the *input*; the badge is the *status*.
- **`ClipModal`** — `md` badge in the header, next to clip title, beside the spec-12 model-confidence chip.
  **Co-display rule:** render two separate pills — `[ model: 0.82 ]  [ ✓ Coach-verified ]` — never combine.
- **`StatsDashboard`** — a new stat tile **"% Coach-verified"** (e.g. `73% · 19/26 clips`) with a small
  breakdown line `19 verified · 3 corrected · 4 unreviewed`. Complements (does not replace) any spec-12
  confidence summary.
- **`RecruitingProfile`** — a **trust line** under the existing "Stats & saves verified by Goalie Analytics
  Pro" line (`RecruitingProfile.tsx:156`). Critically **distinguish engine vs human**: the existing
  `BadgeCheck` "Verified by GAP" = engine-derived from full-game video; the new line = human:
  *"✓ 73% of this season's clips independently confirmed by a coach."* Two separate sentences, two distinct
  meanings — do not let the new copy be read as the engine verifying itself.
- **`FilterBar`** — **optional** new boolean filter `verifiedOnly` (Phase 2). Add to `FilterState`
  (alongside `hasFeedbackOnly`) and apply in `App.tsx`'s `filteredClips`. Reuse the existing `BooleanToggle`
  pattern. Label "Coach-verified only."
- **`Leak Finder` / `FilmSession`** (light touch, optional): when a leak/agenda item cites a clip, show the
  badge inline so users know whether the underlying clip is coach-blessed.

**Accessibility:** badge has `aria-label` + `title`; color is never the sole signal (icon + text).

## 8. Edge cases
- **Conflicting / older feedback for one clip** → latest `timestamp` wins (§6). Fix the array-order bug.
- **Multiple coaches on one clip** → MVP: still latest-wins (anonymous). Phase 2 (`coach_id`): if ≥2
  distinct coaches both confirm → "✓ Coach-verified (2)"; if they disagree (one accurate, one correction)
  → show **✎ Coach-corrected** (a correction is the stronger, more cautious signal) and surface count in
  tooltip. Document this tie-break.
- **Corrected-then-reconfirmed** → newest `'Accurate Analytic'` → `'verified'`. Conversely confirmed-then-
  corrected → `'corrected'`.
- **Empty / missing `<vID>_feedback`** (404 / `[]`) → every clip `'unreviewed'`, pct = 0%, no error UI
  (`fetchClipFeedback` already swallows failures to `[]`).
- **Records with empty `clip_id`** → skipped (matches existing `if (f.clip_id)` guard).
- **`timestamp` malformed/missing on legacy records** → treat as epoch-min so any record with a valid newer
  timestamp wins; if all malformed, fall back to array order.
- **Clip count = 0 (no clips in game)** → show "—" not "0%" to avoid implying failure.
- **Aggregate honesty** → tooltip states denominator ("of all clips"); never imply "verified = correct" at
  field level (respects qualitative-rank / `threat_type` caveats).

## 9. Phasing & effort
- **Phase 1 — S.** `VerificationState` + `deriveVerification`/aggregate helpers (`utils/verification.ts`),
  fix the latest-by-timestamp bug in `App.tsx`, `VerifiedBadge` component, wire into `ClipCard` + `ClipModal`
  + a `StatsDashboard` tile + the `RecruitingProfile` trust line.
- **Phase 2 — M.** `verifiedOnly` filter in `FilterBar`/`FilterState`; season-level pooled rollup with
  per-`vID` fan-out; co-display polish with spec 12; LeakFinder/FilmSession inline badges.
- **Phase 3 — M (only if needed).** Path B `<vID>_verification` doc + `coach_id` attribution + multi-coach
  rules.

**Dependencies:** **Spec 07 (Coach Review Queue)** is the primary generator of the `FeedbackRecord`s this
spec visualizes — verification coverage is only meaningful once 07 drives review volume. **Complements
spec 12 (model confidence)** — coordinate the ClipModal/Dashboard co-display so the two trust signals sit
side by side without visual collision.

## 10. Success metrics
- **Coverage:** season-level coach-verified % trends up after launch (proxy for review adoption).
- **Recruiter trust:** RecruitingProfile share→view dwell / contact-CTA rate uplift on profiles with a
  visible verified %.
- **Loop closure:** % of submitted feedback that becomes a visible badge within one page load (target 100%
  — i.e., the array-order bug is fixed and state is always reflected).
- **No regression:** existing thumbs feedback submission rate unchanged or higher (badge reinforces, not
  cannibalizes, the input).

## 11. Open questions
1. Denominator definition: all clips, or only "reviewable" clips (e.g., exclude non-shot clips)? MVP = all.
2. Should `corrected` clips count *against* a separate "needs attention" metric surfaced to the goalie's own
   coach (links to spec 07 queue)?
3. Multi-coach attribution — do we need named coaches publicly on a RecruitingProfile, or is anonymous
   aggregate enough for credibility? (Drives Phase 3 + `coach_id`.)
4. Should a coach correction that the coach *also* marks resolved auto-flip to verified, or require an
   explicit re-confirm? (§6 currently requires a new `'Accurate Analytic'`.)
5. Do we expose the verified % to recruiters as a hard number, or bucketed ("Extensively coach-reviewed")
   to avoid over-indexing on small denominators?
6. Cross-axis copy: how do we phrase a clip that is **high model confidence + coach-corrected** so viewers
   trust the human over the engine without distrusting the product? (coordinate with spec 12).
