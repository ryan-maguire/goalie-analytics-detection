#!/usr/bin/env bash
# Extend YOLO + audio per-second probs coverage to 11 vIDs that the
# fusion-wide stage 1 cannot yet score. Idempotent — skips any vID
# whose probs TSV already exists.
#
# Pipeline per vID:
#   1. extract audio features  → data/output/audio_features/{vID}.tsv
#   2. YOLO inference on video → runs/yolo_curve_n16/probs/{vID}.tsv
#   3. audio inference         → runs/audio_curve_n16/probs/{vID}.tsv
#
# YOLO inference reads the video directly (no need for cached YOLO
# features TSV). Audio inference reads the cached audio features.

set -u  # don't -e: keep going on per-vID failures

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VIDS=(
    SX5xNJlh6eQ bfEKgtOIkQU n2cy8b755Tg v0lxSTbXfw8
    Fjc9hmK8_3U HNG0jKYY12g kQVdtRa4o_A krxhPVLGLz8
    KYtM20r9BuM q5yj6sAFQeY zOQrPK7IJ24
)

YOLO_WEIGHTS=runs/yolo_curve_n16/work/runs/hockeyai_shot_n16/weights/best.pt
AUDIO_WEIGHTS=runs/audio_curve_n16/work/best.pt
VIDEOS_DIR=data/videos
AUDIO_FEAT_DIR=data/output/audio_features
YOLO_PROBS_DIR=runs/yolo_curve_n16/probs
AUDIO_PROBS_DIR=runs/audio_curve_n16/probs

mkdir -p "$AUDIO_FEAT_DIR" "$YOLO_PROBS_DIR" "$AUDIO_PROBS_DIR" logs

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

resolve_video() {
    # Some videos are named full_<vID>.mp4, others <vID>.mp4
    local v=$1
    for cand in "$VIDEOS_DIR/full_${v}.mp4" "$VIDEOS_DIR/${v}.mp4" \
                "$VIDEOS_DIR/full_${v}.mkv" "$VIDEOS_DIR/${v}.mkv"; do
        if [ -e "$cand" ]; then echo "$cand"; return 0; fi
    done
    # Fallback glob
    ls "$VIDEOS_DIR"/*"${v}"*.mp4 2>/dev/null | head -1
}

log "extend_probs_coverage starting for ${#VIDS[@]} vIDs"
log "yolo_weights:  $YOLO_WEIGHTS"
log "audio_weights: $AUDIO_WEIGHTS"
echo

# Phase 1: audio feature extraction (per vID, cheap)
log "===== PHASE 1 — audio feature extraction ====="
for v in "${VIDS[@]}"; do
    out="$AUDIO_FEAT_DIR/${v}.tsv"
    if [ -s "$out" ]; then
        log "  $v: audio_features already present — skip"
        continue
    fi
    video=$(resolve_video "$v")
    if [ -z "$video" ]; then
        log "  $v: VIDEO NOT FOUND — skip"
        continue
    fi
    log "  $v: extracting audio features from $video"
    python3 util/extract_audio_features.py \
        --video "$video" --out "$out" 2>&1 \
        | sed "s/^/    [${v}-audio_feat] /"
done
echo

# Phase 2: YOLO inference (reads video directly; one process per vID,
# serial because YOLO loads the model once per process; could fork but
# the cost of model load is amortized inside the script over all --vIDs)
log "===== PHASE 2 — YOLO inference ====="
to_do_yolo=()
for v in "${VIDS[@]}"; do
    if [ -s "$YOLO_PROBS_DIR/${v}.tsv" ]; then
        log "  $v: yolo probs already present — skip"
    else
        to_do_yolo+=("$v")
    fi
done
if [ ${#to_do_yolo[@]} -gt 0 ]; then
    log "  running YOLO inference for: ${to_do_yolo[*]}"
    python3 util/predict_shots_yolo.py \
        --weights "$YOLO_WEIGHTS" \
        --videos-dir "$VIDEOS_DIR" \
        --out-dir "$YOLO_PROBS_DIR" \
        --vIDs "${to_do_yolo[@]}" \
        --fps 1.0 2>&1 \
        | sed "s/^/    [yolo] /"
else
    log "  nothing to do"
fi
echo

# Phase 3: audio inference (reads cached audio features)
log "===== PHASE 3 — audio inference ====="
to_do_audio=()
for v in "${VIDS[@]}"; do
    if [ -s "$AUDIO_PROBS_DIR/${v}.tsv" ]; then
        log "  $v: audio probs already present — skip"
    elif [ ! -s "$AUDIO_FEAT_DIR/${v}.tsv" ]; then
        log "  $v: NO AUDIO FEATURES — skip"
    else
        to_do_audio+=("$v")
    fi
done
if [ ${#to_do_audio[@]} -gt 0 ]; then
    log "  running audio inference for: ${to_do_audio[*]}"
    python3 training/audio_shot/infer.py \
        --weights "$AUDIO_WEIGHTS" \
        --features-dir "$AUDIO_FEAT_DIR" \
        --out-dir "$AUDIO_PROBS_DIR" \
        --vIDs "${to_do_audio[@]}" 2>&1 \
        | sed "s/^/    [audio] /"
else
    log "  nothing to do"
fi
echo

# Summary
log "===== SUMMARY ====="
for v in "${VIDS[@]}"; do
    y=$([ -s "$YOLO_PROBS_DIR/${v}.tsv"  ] && echo OK || echo MISS)
    a=$([ -s "$AUDIO_PROBS_DIR/${v}.tsv" ] && echo OK || echo MISS)
    log "  $v  yolo_probs=$y  audio_probs=$a"
done
log "DONE"
