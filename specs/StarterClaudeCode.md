You are improving the cv_seg threat-window detector in this repository.
Your goal is to maximize the aggregate strict F1 score reported by
eval/eval_cv_seg_output.py.

============================================================
CURRENT BASELINE (measured 2026-05-17, default config)
============================================================
Full 14-video eval:
  STRICT:    P=0.182  R=0.314  F1=0.231
  MIDPOINT:  R=0.662  P=0.211  F1=0.320  (lenient)
  Attribution accuracy: 1.000 (on matched windows)

Pattern: precision kills the score. FPs > TPs on every video.
Midpoint recall 0.662 means we're hitting most real windows; the
problem is firing too many windows. The job is to cut FPs without
losing TPs.

============================================================
DATA QUALITY NOTE — READ BEFORE TUNING
============================================================
5 of 14 videos have UNVERIFIABLE attribution because target and
opponent jerseys share a color token (e.g. both teams wear "green
and X"). These videos produce artificially poor F1 scores (e.g.
n2cy8b755Tg: 0/58/0). They are EXCLUDED from the loop.

Excluded: n2cy8b755Tg, HNG0jKYY12g, kQVdtRa4o_A, q5yj6sAFQeY,
          zOQrPK7IJ24

============================================================
FAST EVAL SET — use for the inner loop
============================================================
3 videos covering the score range:
  bfEKgtOIkQU  CUST000048  F1=0.348 high
  dwGsP6QKDs8  CUST000031  F1=0.295 mid
  krxhPVLGLz8  CUST000031  F1=0.145 low

The repo has run_fast_set.sh in the root that handles the
per-vID customer-file lookup. ALWAYS use this script for the
inner loop. Do NOT hand-roll cv_seg invocations.

Inner-loop command:
  ./run_fast_set.sh

OUTER CHECK (every 5 kept commits) — 9 attribution-verified videos:
  python3 eval/eval_cv_seg_output.py \
    --vIDs SX5xNJlh6eQ bfEKgtOIkQU mjEeE7p2Hz8 v0lxSTbXfw8 \
           dwGsP6QKDs8 Fjc9hmK8_3U J8WkcuTsD5I krxhPVLGLz8 KYtM20r9BuM \
    --customer-id CUST000048 CUST000031

(SX5xNJlh6eQ, bfEKgtOIkQU, mjEeE7p2Hz8, v0lxSTbXfw8 = CUST000048.
 The remaining 5 = CUST000031.)

============================================================
REQUIRED READING (in this order, before any change)
============================================================
1. README.md — repo overview
2. EVAL_NOTES.md — what's been tried. Treat "settled" values as
   off-limits unless you have specific evidence in the latest FP
   trace.
3. cv_seg/pipeline.py — the orchestrator
4. cv_seg/constants.py — all tunable knobs
5. cv_seg/cli.py — what's exposed as CLI flags
6. eval/eval_cv_seg_output.py — docstring + aggregate() ~line 1477.
   This is the REWARD FUNCTION. NEVER EDIT IT.
7. The most recent file at data/output/evals/eval_*_fp_trace.tsv.
   THIS IS WHERE EVERY EXPERIMENT STARTS.

============================================================
HOW TO READ THE FP TRACE
============================================================
Columns: vID  pred_start  pred_end  pred_dur  pred_color
         attr_src  source_combo  n_raw

`source_combo` is the dominant diagnostic. It lists the raw-signal
sources that were combined to produce the FP. Values include:
  motion              — opportunistic motion run, no confirmation
  motion_auto_close   — a long motion run that hit the 45s cap
  faceoff             — center-ice faceoff detection
  celebration         — asymmetric player density (goal celebration)
  crowd_roar          — audio low-freq spike
  whistle             — audio referee-whistle spike

EVAL_NOTES.md says motion_auto_close was 87% of FPs at v23.7.
v23.8/v23.9 tightened the confirmation rules; we don't know what
the current split is. Counting source_combo categories is
experiment 1 — see below.

============================================================
EXPERIMENT 1 IS DIAGNOSIS, NOT A CHANGE
============================================================
Before touching any constant:

1. Open the latest *_fp_trace.tsv. Aggregate FPs by source_combo
   across the fast-set videos only (bfEKgtOIkQU, dwGsP6QKDs8,
   krxhPVLGLz8). Report:
   - Total FPs in the fast set
   - Top 5 source_combo categories by count
   - What share motion_auto_close still represents (alone OR in
     combination with other sources)
   - Median pred_dur for each top category

2. Open the latest *_diagnostics.tsv. Cross-reference: are FPs
   biased toward short windows, long windows, or specific time
   ranges in each game?

3. Report findings as a single short paragraph + a numbered list.
   Do NOT make a code change yet. Wait for user input on what to
   target first.

The pause before experiment 1 is intentional. Past sessions wasted
hours guessing at knobs without grounding the first hypothesis.
Diagnosis-first is non-negotiable.

============================================================
INNER LOOP (experiments 2-20)
============================================================
1. STATE A HYPOTHESIS in one sentence, citing a specific number
   from the FP trace or diagnostics.

2. CHECK EVAL_NOTES.md. If your hypothesis has been tried before
   with documented results, pick a different one.

3. MAKE ONE CHANGE in cv_seg/constants.py. Do not refactor.
   Do not touch other files. Do not change multiple constants.

4. RUN: ./run_fast_set.sh
   Wall time ~2 minutes.

5. PARSE the aggregate F1 from the eval tail output.

6. DECISION:
   - Aggregate strict F1 improved by > 0.01 AND no per-video F1
     dropped by > 0.05 → commit, with message format:
       <constant>: <old>=<new>, F1 <prev> → <new>
   - Otherwise → revert with `git checkout cv_seg/constants.py`
     and try a different hypothesis.

7. OUTER CHECK every 5 kept commits. If outer-set gain is less
   than 50% of fast-set gain, the fast set was unrepresentative.
   Stop. Re-pick fast set. Do not commit further until verified.

============================================================
RULES — NON-NEGOTIABLE
============================================================
- NEVER edit eval/eval_cv_seg_output.py.
- NEVER edit metrics_seg/ or feedback_seg/ (they cost real money).
- NEVER run run_pipeline.py (it triggers all 3 stages incl. Gemini).
- One constant change per experiment. No batched edits.
- No new Python dependencies in the first 10 experiments.
- No new files (except this spec's deliverables).
- Don't disable target_filter — eval is calibrated for it.
- Don't tune for the 5 excluded videos.
- Don't tune values flagged as "settled" in EVAL_NOTES.md unless
  the FP trace shows the underlying condition has changed.
- Commit kept changes. Revert failed changes. Always.
- Don't claim a win on F1 gains < 0.01 (run-to-run noise).
- Cap session at 20 experiments. Stop and summarize at 20.

============================================================
HIGH-LEVERAGE TARGETS (consider only after diagnosis)
============================================================
Given baseline precision 0.18, FPs dominate. The likely-productive
constants to tune:

1. MIN_THREAT_DUR (currently 15s) — if FPs are biased toward short
   windows. Try 20, 25, 30.
2. MAX_OPEN_WINDOW_SEC (currently 45s) — if motion_auto_close still
   dominates FPs. EVAL_NOTES.md shows 90→45 was net positive. Try
   30 or 25.
3. MIN_CONFIRMATION_OVERLAP_SEC + CONFIRMATION_EVENT_WIDTH_SEC
   (currently both 6, kept in lockstep) — try both at 8 or 10.
   DO NOT change one without the other.
4. MIN_KEEP_DUR (currently 5s) — tighter promote-to-no-threat may
   demote slivers.
5. CELEBRATION_MIN_RUN_SEC (currently 5s) — if celebration windows
   are the FP driver.

Do NOT propose adding new signals or refactoring. Knob tuning
only. Structural changes need user approval after experiment 20.

============================================================
NOW START
============================================================
Read EVAL_NOTES.md and the latest *_fp_trace.tsv. Aggregate FPs by
source_combo across the fast set. Report findings. Wait for user
direction before making any code change.