#!/usr/bin/env bash
# A/B test gemini-3.5-flash against the existing gemini-2.5-pro
# metrics_seg outputs.
#
# For each vID, runs metrics_seg with --model gemini-3.5-flash against
# the SAME cv_seg windows that the Pro run used (from yesterday's
# 11-vID validation). Writes outputs to data/output/runs/metrics_flash35/.
# Idempotent — skips a vID whose flash output already exists.
#
# Comparison: tools/diff_flash_vs_pro.py walks both dirs and emits a
# per-window diff + per-game aggregate deltas.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VIDS=(Fjc9hmK8_3U q5yj6sAFQeY KYtM20r9BuM)

CUST_MAP_Fjc9hmK8_3U=CUST000031
CUST_MAP_q5yj6sAFQeY=CUST000031
CUST_MAP_KYtM20r9BuM=CUST000031

OUT_DIR=data/output/runs/metrics_flash35
SEGMENTS_DIR=data/output/runs/cv_seg
VIDEO_DIR=data/videos
MODEL=gemini-3.5-flash
# gemini-3.x preview models route through 'global' rather than a
# specific region (us-central1 returns 404 for these IDs).
VERTEX_LOCATION=global

mkdir -p "$OUT_DIR" logs

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

log "A/B test starting: model=$MODEL  vIDs=${VIDS[*]}"
log "  segments source: $SEGMENTS_DIR"
log "  outputs:         $OUT_DIR"
echo

for vid in "${VIDS[@]}"; do
    out="$OUT_DIR/gt_metrics_${vid}.json"
    if [ -s "$out" ]; then
        log "  $vid: flash output already present → skip"
        continue
    fi
    cust_var="CUST_MAP_${vid}"
    cust="${!cust_var}"
    seg_in="$SEGMENTS_DIR/gt_seg_${vid}.json"
    if [ ! -s "$seg_in" ]; then
        log "  $vid: no cv_seg windows at $seg_in → skip"
        continue
    fi
    n_win=$(python3 -c "import json; print(len(json.load(open('$seg_in'))))")
    log "  $vid: running metrics_seg with --model $MODEL on $n_win cv_seg windows (cust=$cust)"
    python3 metrics_seg/01_detect_segment_metrics.py \
        --vID "$vid" \
        --customID "$cust" \
        --model "$MODEL" \
        --vertex-location "$VERTEX_LOCATION" \
        --segments-dir "$SEGMENTS_DIR" \
        --local-video-dir "$VIDEO_DIR" \
        --output-dir "$OUT_DIR" \
        --no-gcs-upload \
        --workers 4 2>&1 \
        | sed "s/^/    [${vid}-flash] /"
done

echo
log "===== A/B run complete ====="
for vid in "${VIDS[@]}"; do
    f="$OUT_DIR/gt_metrics_${vid}.json"
    if [ -s "$f" ]; then
        n=$(python3 -c "import json; d=json.load(open('$f')); print(len(d) if isinstance(d,list) else len(d.get('windows',[])))")
        log "  $vid: flash output OK ($n windows)"
    else
        log "  $vid: flash output MISSING"
    fi
done
