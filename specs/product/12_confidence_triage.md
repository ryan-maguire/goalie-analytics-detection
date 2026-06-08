# Confidence Triage — Model Confidence as a First-Class Signal
*Stop hiding the confidence scores. One normalized confidence model, one reusable chip, one global "trust the numbers" toggle — across every analytics surface.*

## Summary & problem
Every clip the pipeline emits already carries model-confidence signals — `goalie_positioning.goalie_position_confidence_score`, `coaching_feedback.coaching_confidence_score`, the v15 `metrics.beaten_location_confidence`, and the `analysis_confidence_caveats[]` array that flags footage limitations. Yet today these are surfaced **inconsistently and ad hoc**:

- `ClipCard`/`ClipModal`/`FavoriteClips` show a single `clip.confidence_score * 100` "AI Confidence %" — a *different, top-level* field that is **not** the per-clip nested scores the analytics actually use.
- `utils/leakAnalysis.ts` weights leaks by `goalie_position_confidence_score`/`coaching_confidence_score` (÷5, default 3/5) and flags `lowConfidence: avgConf < 0.5`.
- `utils/filmSession.ts` gates on `min(coaching, position)` scores ≥ 2, and applies a `0.85` caveat multiplier when `analysis_confidence_caveats` is non-empty.
- `utils/recruitingProfile.ts` gates the highlight reel on `goalie_position_confidence_score ≥ 2` and blends both scores into its reel score.
- `useDangerSplits` ignores confidence entirely; `BeatenMap` reads `beaten_location_confidence` independently.
- `FilterBar` exposes a "confidence" filter built from the *top-level* `clip.confidence_score` (discrete `%` buckets), unrelated to the nested scores.

The result: **four different definitions of "confidence," three default values, two numeric scales (1–5 ints and 0–1 floats), and no way for a user to say "only show me numbers you're sure about."** Families and recruiters cannot tell whether a stat is solid or a low-footage guess.

This spec defines **one** normalized clip-confidence model (`utils/confidence.ts`), a reusable **`ConfidenceChip`** indicator, a **"needs review"** flag on Low-confidence clips (feeding the Coach Review Queue, spec 07), and a **global "high-confidence only" toggle** that every analytics surface honors. It is the **model-confidence** trust layer, complementary to **spec 11** (human coach verification) — model confidence is automatic and per-field; coach verification is a manual override badge.

## Target users
- **Goalies** — know which feedback to act on first; a Low-confidence clip is "tell the camera operator," not "you played badly."
- **Goalie families** — one honest trust signal next to every number; can flip "high-confidence only" before sharing a stat.
- **Goalie coaches** — triage: jump straight to the clips the model is unsure about (review queue), and present only solid numbers in film sessions.
- **Recruiters (secondary)** — assurance that a shared profile's stats were computed on confident footage.

## Data inputs (exact confidence fields + scales)
All per-clip, from the final analysis JSON (`item.type === 'clips'`). Already in `types.ts` except where noted.

| Field | Location | Type / scale | Default when missing | Notes |
|---|---|---|---|---|
| `goalie_position_confidence_score` | `goalie_positioning` | int **1–5** (sometimes string) | `3` (neutral) | positioning ranks confidence |
| `coaching_confidence_score` | `coaching_feedback` | int **1–5** (sometimes string) | `3` (neutral) | coaching feedback confidence |
| `beaten_location_confidence` | `metrics` | float **0.0–1.0** (v15) | `undefined` → excluded | goal clips only |
| `analysis_confidence_caveats` | clip root | `string[]` | `[]` | **non-empty ⇒ footage limited** (penalty, not a score) |
| `confidence_score` | clip root | float 0–1 (legacy top-level) | `undefined` | **distinct** from the above; currently shown as "AI Confidence %". Treated as an optional 4th input (see Edge cases), not the source of truth. |
| `windows_succeeded` / `windows_failed` / `windows_analysed` | `summary.response` | int | — | **coverage** (game-level), surfaced as context; not part of per-clip score |

**Scale caveats (must normalize):** the two `*_confidence_score` fields are **1–5 integers**; `beaten_location_confidence` is a **0–1 float**; caveats are a **boolean penalty** (presence, not magnitude). Positioning *ranks* (`depth_rank`, etc.) are **qualitative** and are **not** confidence — do not conflate. Inputs may arrive as strings (`"4"`); coerce with a finite-number guard (mirroring the existing `toNum` helpers).

## Pipeline / backend changes
**None. Pure-frontend.** All fields above are already emitted by `metrics_seg`/`feedback_seg` and present in the JSON (`beaten_location_confidence` shipped in v15). This spec adds **no** pipeline, schema, or API-gateway change. The entire feature is a new shared util (`goalie-analytics-pro-ui/utils/confidence.ts`) plus a reusable component and a global toggle in `App.tsx`. The UI repo auto-deploys on push to `main`.

(Future, out of scope: if Low-confidence rate is high, a `metrics_seg` prompt note could ask Gemini to populate `analysis_confidence_caveats` more granularly — tracked as an open question, not built here.)

## Computation / logic — the single normalized model
A new module `utils/confidence.ts` is the **sole** definition of clip confidence.

### Normalized inputs (all → 0..1)
```ts
const FIVE = (v: any): number | null => {            // 1–5 int → 0..1
  const n = Number(v);
  return Number.isFinite(n) ? clamp01((n - 1) / 4) : null;   // 1→0.0, 3→0.5, 5→1.0
};
const UNIT = (v: any): number | null => {            // already 0–1
  const n = Number(v);
  return Number.isFinite(n) ? clamp01(n) : null;
};
```
> Mapping rationale: a 1–5 score uses `(n-1)/4` so that **1/5 maps to 0.0** and **5/5 to 1.0**. The legacy ÷5 convention in `leakAnalysis`/`filmSession` (where 3/5 = 0.6) is replaced; those callers migrate to this model. The neutral default is **3/5 = 0.5**, preserved.

### Component scores & weights
```ts
const pos    = FIVE(gp.goalie_position_confidence_score);  // positioning
const coach  = FIVE(cf.coaching_confidence_score);         // feedback
const beaten = UNIT(m.beaten_location_confidence);         // goal clips only
```
Combine **only the present** components as a weighted mean (renormalizing weights over present inputs, so a missing component does not silently mean "low"):
```ts
W = { pos: 0.45, coach: 0.35, beaten: 0.20 };  // beaten only contributes on goal clips
score0 = weightedMean([[pos,W.pos],[coach,W.coach],[beaten,W.beaten]]);
// all components missing → score0 = NEUTRAL = 0.5  (do not penalize for absence)
```

### Caveat penalty
`analysis_confidence_caveats` is a **multiplicative** penalty (footage-limited), matching the existing `0.85` precedent in `filmSession`:
```ts
const caveatPenalty = (caveats?.length ?? 0) > 0 ? 0.85 : 1.0;
const score = clamp01(score0 * caveatPenalty);
```

### Bands (3 tiers, fixed thresholds)
```ts
HIGH   if score >= 0.70
MEDIUM if 0.45 <= score < 0.70
LOW    if score < 0.45
```
Calibration: a clip with pos=4/5, coach=4/5, no caveats → `0.75` → **High**. Both at 3/5 (the default/neutral) → `0.50` → **Medium**. Either ≤ 2/5, or 3/5 with a caveat (`0.5*0.85=0.425`) → **Low**. Thresholds live as exported constants (`CONF_HIGH = 0.70`, `CONF_MED = 0.45`) so they are tunable in one place.

### Public API (consumed everywhere)
```ts
export interface ClipConfidence {
  score: number;                 // 0..1 normalized
  band: 'high' | 'medium' | 'low';
  needsReview: boolean;          // band === 'low'
  hasCaveats: boolean;
  components: { pos: number|null; coach: number|null; beaten: number|null };
  missing: boolean;              // all components absent (score is the 0.5 default)
}
export function clipConfidence(clip: ClipDetail): ClipConfidence;
export function meanConfidence(clips: ClipDetail[]): number;   // for aggregate surfaces (HDSV%, Recruiting)
export function isHighConfidence(clip: ClipDetail): boolean;   // band === 'high'
export const CONF_HIGH = 0.70, CONF_MED = 0.45;
```

### Feeding the global toggle
A single boolean lives in app state (`highConfidenceOnly`, persisted to `localStorage`). When ON, analytics surfaces filter their input clips with `clipConfidence(c).band !== 'low'` (i.e., keep High + Medium; **Low is what gets hidden**, not "only High") before computing. Per-clip displays never hide a clip when the toggle is on — they only suppress aggregate **numbers** computed from Low clips. (See Edge cases: over-hiding.)

## Frontend

### `ConfidenceChip` (new reusable component)
A small pill rendered wherever a clip or aggregate appears. Props: `confidence: ClipConfidence` (or `{score, band}`), `size?: 'sm'|'md'`, `showScore?: boolean`.
- **High** → green dot + "High confidence".
- **Medium** → amber dot + "Medium".
- **Low** → red dot + "Low — needs review" (also a `needsReview` flag icon).
- Tooltip lists present components ("Positioning 4/5 · Coaching 4/5 · Footage limited") and, when `missing`, "Not enough signal — neutral default."
- Replaces the bespoke `clip.confidence_score * 100` "AI Confidence %" rendering in `ClipCard` (line ~122), `ClipModal` (~146), and `FavoriteClips` (~449).

### "Needs review" flag
`needsReview === true` (Low band) renders a flag on `ClipCard` and is the **client-side signal the Coach Review Queue (spec 07) orders by** — spec 07 consumes `clipConfidence(c).score` ascending and `needsReview` as its primary sort/filter, instead of re-deriving low-confidence itself.

### Global "high-confidence only" toggle
A single switch in the app `Header` (next to the game selector), label **"High-confidence only"**, with a tooltip "Hide low-confidence clips from all stats." State in `App.tsx`, persisted. **Every analytics surface respects it:**

| Surface | How it honors the toggle |
|---|---|
| **Leak Finder** (`leakAnalysis`) | pre-filter clips to non-Low before `analyzeLeaks`; drop bespoke `avgConf`/`lowConfidence` derivation in favor of `meanConfidence` |
| **HDSV% / splits** (`useDangerSplits`) | pre-filter input clips; show coverage note when clips dropped |
| **BeatenMap** | filter goal clips by `beaten_location_confidence` via the unified model |
| **Recruiting stats & reel** (`recruitingProfile`) | replace `CONF_FLOOR ≥ 2` gate with `clipConfidence` band; reel excludes Low |
| **Season Report Card / StatsDashboard** | aggregates computed on filtered clips; header shows "computed on N high/medium-confidence clips (M low hidden)" |
| **Film Session** (`filmSession`) | replace `confOf min ≥ 2` + `0.85` caveat with `clipConfidence`; toggle hides Low picks |
| **FilterBar** | the ad-hoc `%`-bucket "confidence" dimension is replaced by a 3-band (High/Med/Low) chip filter driven by `clipConfidence` |

When the toggle hides clips, each surface shows a small, dismissible note ("3 low-confidence clips hidden — show all") so data is never silently dropped.

### UX + ASCII mockup
```
 Header ───────────────────────────────────────────────────────────────────
  Goalie Analytics Pro     [ Game: vs Lightning ▾ ]   ◉ High-confidence only ⓘ
 ───────────────────────────────────────────────────────────────────────────

 StatsDashboard
  HDSV%  .842   (computed on 18 of 21 clips · 3 low-confidence hidden — show all)

 ClipCard ────────────────────────┐     ConfidenceChip variants
  02:14  Goal against             │       ● High confidence
  Depth: Aggressive · Sq: Poor    │       ● Medium
  ┌───────────────────────────┐   │       ● Low — needs review  ⚑
  │ ● Low — needs review  ⚑   │   │
  └───────────────────────────┘   │     Tooltip (Low):
 ─────────────────────────────────┘       Positioning 2/5 · Coaching 3/5
                                           Footage limited (1 caveat)

 Coach Review Queue (spec 07)  →  ordered by clipConfidence ascending, Low first
```

## Edge cases
- **Missing scores → neutral, not Low.** If `goalie_position_confidence_score` and `coaching_confidence_score` are absent, default each to neutral; if **all** components absent, `score = 0.5` (Medium) and `missing = true`. Absence must never be punished as low confidence (otherwise older v14 clips without `beaten_location_confidence` would all read Low).
- **Mixed scales.** `FIVE` vs `UNIT` strictly separate the 1–5 and 0–1 inputs; string inputs (`"4"`) coerced; non-finite → component `null` (excluded from the weighted mean, not treated as 0).
- **`beaten_location_confidence` only on goals.** It contributes only when present; its 0.20 weight is renormalized away on non-goal clips. Never let its absence drag down a clean save clip.
- **Over-hiding data.** The toggle defaults **OFF**. It hides only Low (keeps Medium), never hides per-clip cards (only suppresses aggregate numbers), and always shows a "show all" escape. Surfaces that would drop below their min-sample threshold (e.g., HDSV% `MIN_N`, Leak Finder `minN`/`goalsLeak ≥ 3`) keep their existing guards and report "not enough confident clips" rather than a misleading number.
- **Legacy top-level `confidence_score`.** Kept as an optional, lowest-priority input only if the nested scores are entirely missing; otherwise ignored to avoid double-counting. The FilterBar `%`-bucket UI is removed in favor of band filtering.
- **Coach-verified clips (spec 11).** A coach-verified clip should display **both** its model band and the verified badge; verification does **not** overwrite the model score (they are orthogonal). If spec 11 ships a manual confidence override, it takes display precedence but the underlying `clipConfidence` is unchanged.

## Phasing & effort
**Size: S** — this is a refactor + surface, not new analytics. No pipeline work.
- **P1 (core):** `utils/confidence.ts` (model + thresholds + tests against real JSON), `ConfidenceChip`, swap the bespoke `confidence_score %` displays in `ClipCard`/`ClipModal`/`FavoriteClips`.
- **P2 (toggle):** global `highConfidenceOnly` state + Header switch + `localStorage`; wire `useDangerSplits`, `StatsDashboard`, `leakAnalysis`, `recruitingProfile`, `filmSession`, `BeatenMap`, `FilterBar` to respect it; add "N hidden — show all" notes.
- **P3 (triage):** `needsReview` flag on cards; expose `clipConfidence` ordering to the Coach Review Queue (spec 07).

**Features that must migrate to `utils/confidence.ts`** (delete their local confidence logic): `leakAnalysis.ts` (`confOf`, `avgConf`, `lowConfidence`), `filmSession.ts` (`confOf`, `0.85` caveat), `recruitingProfile.ts` (`CONF_FLOOR`, `posConf`/`coachConf` blend), `FilterBar.tsx` + `App.tsx` (the `%`-bucket confidence filter), `BeatenMap.tsx` (`beaten_location_confidence` read), and the per-card `confidence_score` renders.

## Success metrics
- **Single source of truth:** zero remaining ad-hoc confidence derivations outside `utils/confidence.ts` (grep audit at PR review).
- **Trust adoption:** % of sessions that toggle "high-confidence only" at least once; % of shared Recruiting profiles created with it ON.
- **Triage throughput:** median time-to-first-review for Low/`needsReview` clips in the Coach Review Queue (spec 07) drops after the flag ships.
- **Honesty:** share of displayed aggregate stats accompanied by a confidence band/coverage note (target: 100% of headline numbers).
- **No regressions:** Leak Finder / HDSV% / Recruiting outputs unchanged when the toggle is OFF (snapshot test before/after migration).

## Open questions
1. **Weights (0.45/0.35/0.20)** — calibrate against labeled clips, or expose as remote config? Start fixed.
2. **Thresholds (0.70 / 0.45)** — validate against the distribution of real scores so bands aren't lopsided (most defaults land at 0.5 = Medium; confirm that's desired).
3. **Default ON for recruiters?** Should shared/Recruiting views force `highConfidenceOnly` ON for credibility, with an explicit "showing all" override?
4. **Game-level coverage** (`windows_failed`) — fold into a *game* confidence banner, or keep purely as the per-surface coverage note?
5. **Granular caveats** — should `metrics_seg` populate typed caveat reasons (angle/occlusion/distance) so the chip tooltip can be specific? (Pipeline follow-up, out of scope here.)
6. **Interaction with spec 11** — final precedence rule when a clip is both Low-model-confidence and coach-verified.
