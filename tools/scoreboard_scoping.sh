#!/usr/bin/env bash
# Sample 10 evenly-spaced frames from each of the 14 validated vIDs.
# Output: data/output/scoreboard_scoping/{vID}/frame_NN_tMMSS.jpg
# Cheap (~70 MB on disk, ~3 min wall) — strict scoping pass, no OCR.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VIDS=(mjEeE7p2Hz8 dwGsP6QKDs8 J8WkcuTsD5I
      SX5xNJlh6eQ bfEKgtOIkQU n2cy8b755Tg v0lxSTbXfw8
      Fjc9hmK8_3U HNG0jKYY12g kQVdtRa4o_A krxhPVLGLz8
      KYtM20r9BuM q5yj6sAFQeY zOQrPK7IJ24)

OUT_BASE=data/output/scoreboard_scoping
N_FRAMES=10

mkdir -p "$OUT_BASE"

for vid in "${VIDS[@]}"; do
    src=""
    for cand in "data/videos/full_${vid}.mp4" "data/videos/${vid}.mp4"; do
        [ -e "$cand" ] && src="$cand" && break
    done
    if [ -z "$src" ]; then
        echo "[$vid] NO VIDEO — skip"; continue
    fi
    # Read duration with ffprobe
    dur=$(ffprobe -v error -show_entries format=duration \
                   -of default=noprint_wrappers=1:nokey=1 "$src" 2>/dev/null)
    dur=${dur%.*}
    if [ -z "$dur" ] || [ "$dur" -le 0 ]; then
        echo "[$vid] ffprobe failed"; continue
    fi

    out_dir="$OUT_BASE/$vid"
    mkdir -p "$out_dir"
    rm -f "$out_dir"/*.jpg 2>/dev/null

    # 10 evenly spaced timestamps (10%, 20%, ... 100% of duration)
    # Skip the very edges (intro/outro)
    for i in $(seq 1 $N_FRAMES); do
        t=$(( dur * i * 95 / (N_FRAMES * 100) + dur / 20 ))
        mm=$(( t / 60 ))
        ss=$(( t % 60 ))
        out="$out_dir/frame_$(printf '%02d' $i)_t${mm}m$(printf '%02d' $ss)s.jpg"
        ffmpeg -hide_banner -loglevel error \
            -ss "$t" -i "$src" -frames:v 1 -q:v 3 \
            -y "$out" 2>/dev/null
    done

    n=$(ls "$out_dir"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
    sz=$(du -sh "$out_dir" 2>/dev/null | cut -f1)
    echo "[$vid] extracted $n frames  ($sz) from a ${dur}s video"
done

echo
echo "Total disk: $(du -sh "$OUT_BASE" | cut -f1)"
echo "Frames at: $OUT_BASE/<vID>/"
