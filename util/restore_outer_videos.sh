#!/bin/bash
# Download the 6 outer-only videos from GCS into data/videos/.
# Skips files already present. Creates the {vID}.mp4 symlink that
# run_fast_set.sh / pipeline.py expects (mirrors the existing pattern
# for the 3 fast-set videos).
#
# Bucket + prefix come from cv_seg/constants.py:
#   GCS_BUCKET   = "goalie_video_bucket"
#   VIDEO_PREFIX = "ground_truth_video/full_video"
# Per pipeline.py:189, blob path is "{VIDEO_PREFIX}/full_{vID}.mp4".
#
# Auth: assumes `gcloud auth application-default login` (or service
# account credentials in GOOGLE_APPLICATION_CREDENTIALS) is already
# configured. If not, gsutil will fail with a clear message.
#
# Cost note: ~600-900MB per video, ~4GB total. Run on a fast connection.
#
# Usage:
#   ./util/restore_outer_videos.sh

set -u  # NOT -e — we want to continue past one bad download and report

BUCKET="goalie_video_bucket"
PREFIX="ground_truth_video/full_video"
DEST="data/videos"

VIDS=(
    SX5xNJlh6eQ
    mjEeE7p2Hz8
    v0lxSTbXfw8
    Fjc9hmK8_3U
    J8WkcuTsD5I
    KYtM20r9BuM
)

mkdir -p "$DEST"

declare -a OK_LIST
declare -a FAIL_LIST
declare -a SKIP_LIST

for vid in "${VIDS[@]}"; do
    full="$DEST/full_${vid}.mp4"
    link="$DEST/${vid}.mp4"

    if [ -f "$full" ]; then
        echo "[skip] $full already present"
        SKIP_LIST+=("$vid")
    else
        echo "=== Downloading $vid ==="
        # Use `gcloud storage cp` rather than `gsutil cp`. gsutil runs
        # under Python and segfaults on macOS with "crashed on child
        # side of fork pre-exec" when its workers fork after the
        # parent has touched Apple's Network framework — neither the
        # Python issue 33725 flag nor OBJC_DISABLE_INITIALIZE_FORK_SAFETY
        # consistently resolves it on this machine. gcloud's storage
        # subcommand is the modern replacement and avoids the issue.
        gcloud storage cp "gs://${BUCKET}/${PREFIX}/full_${vid}.mp4" "$full"
        if [ $? -eq 0 ] && [ -f "$full" ]; then
            OK_LIST+=("$vid")
        else
            echo "[fail] $vid — gsutil exit nonzero or file missing"
            FAIL_LIST+=("$vid")
            continue
        fi
    fi

    # Create or refresh the convention symlink (used by --local-video)
    if [ -L "$link" ] || [ -e "$link" ]; then
        :  # exists; leave alone
    else
        (cd "$DEST" && ln -s "full_${vid}.mp4" "${vid}.mp4")
        echo "[link] $link -> full_${vid}.mp4"
    fi
done

echo ""
echo "============================================================"
echo "Summary:"
echo "  downloaded: ${#OK_LIST[@]}  ${OK_LIST[*]:-(none)}"
echo "  skipped:    ${#SKIP_LIST[@]}  ${SKIP_LIST[*]:-(none)}"
echo "  failed:     ${#FAIL_LIST[@]}  ${FAIL_LIST[*]:-(none)}"
echo "============================================================"

# Exit non-zero if any failed, so caller scripts notice
[ ${#FAIL_LIST[@]} -eq 0 ]
