#!/usr/bin/env bash
# Validate the Flash pre-filter on the same 3 games we used for the
# Pro-vs-Flash A/B. Uses cv_seg windows. Pro still runs on positives;
# Flash decides skip/escalate per window. Output to
# data/output/runs/metrics_flash_screen/ so we can diff against the
# existing metrics_v13/ baseline.
#
# Expected outcome:
#   - 50-80% Pro calls SAVED (cost drops ~50-70%)
#   - Goal recall MUST NOT regress
#   - Shot F1 should stay within ±0.03 of baseline
#
# Cost estimate: ~$5-8 (177 cv_seg windows × Flash + ~half × Pro).

set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VIDS=(Fjc9hmK8_3U q5yj6sAFQeY KYtM20r9BuM)
CUST=CUST000031

OUT_DIR=data/output/runs/metrics_flash_screen
SEGMENTS_DIR=data/output/runs/cv_seg
VIDEO_DIR=data/videos

mkdir -p "$OUT_DIR" logs

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

log "Flash-screen validation starting"
log "  output: $OUT_DIR"
log "  cv_seg windows: $SEGMENTS_DIR"
echo

for vid in "${VIDS[@]}"; do
    out="$OUT_DIR/gt_metrics_${vid}.json"
    if [ -s "$out" ]; then
        log "  $vid: output already present → skip"
        continue
    fi
    seg_in="$SEGMENTS_DIR/gt_seg_${vid}.json"
    if [ ! -s "$seg_in" ]; then
        log "  $vid: no cv_seg windows at $seg_in → skip"
        continue
    fi
    n_win=$(python3 -c "import json; print(len(json.load(open('$seg_in'))))")
    log "  $vid: running metrics_seg --flash-screen on $n_win cv_seg windows"
    # NOTE: no --model override — Pro path uses default gemini-2.5-pro,
    # flash_screen uses its module-level FLASH_MODEL (gemini-2.5-flash).
    python3 metrics_seg/01_detect_segment_metrics.py \
        --vID "$vid" \
        --customID "$CUST" \
        --flash-screen \
        --segments-dir "$SEGMENTS_DIR" \
        --local-video-dir "$VIDEO_DIR" \
        --output-dir "$OUT_DIR" \
        --no-gcs-upload \
        --workers 4 2>&1 \
        | sed "s/^/    [${vid}-fscreen] /"
done

echo
log "===== validation run complete ====="
for vid in "${VIDS[@]}"; do
    f="$OUT_DIR/gt_metrics_${vid}.json"
    if [ -s "$f" ]; then
        n_total=$(python3 -c "import json; d=json.load(open('$f')); print(len(d))")
        n_skipped=$(python3 -c "import json; d=json.load(open('$f')); print(sum(1 for w in d if (w.get('metrics') or {}).get('_flash_screen_skip')))")
        log "  $vid: $n_total windows, $n_skipped flash-screen skips ($((n_skipped * 100 / n_total))%)"
    else
        log "  $vid: output MISSING"
    fi
done
