#!/bin/bash
# Fast eval set: 3 videos covering the F1 score range.
# Usage: ./run_fast_set.sh
# Re-runs cv_seg on the fast set, then evaluates and prints aggregate F1.

set -e

declare -A CUSTOMER
CUSTOMER[bfEKgtOIkQU]=CUST000048.json
CUSTOMER[dwGsP6QKDs8]=CUST000031.json
CUSTOMER[krxhPVLGLz8]=CUST000031.json

for vid in bfEKgtOIkQU dwGsP6QKDs8 krxhPVLGLz8; do
    echo "=== cv_seg: $vid (customer=${CUSTOMER[$vid]}) ==="
    python3 -m cv_seg --vID $vid \
        --customID data/customers/${CUSTOMER[$vid]} \
        --local-video data/videos/$vid.mp4 \
        --no-gcs \
        --output-dir data/output/runs/cv_seg/
done

echo ""
echo "=== eval ==="
python3 eval/eval_cv_seg_output.py \
    --vIDs bfEKgtOIkQU dwGsP6QKDs8 krxhPVLGLz8 \
    --customer-id CUST000048 CUST000031 2>&1 | tail -10