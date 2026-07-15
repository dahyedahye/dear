#!/bin/bash
# Train DEAR-r: apply DEAR (Dissect and Prune) to the base Rajan detector.
#
# Paper setting: prune the top 30% + bottom 30% of channels by RAD score
# (keep the robust middle 40%), then refine the linear head on
# (Rajan aligned pairs + diagnostic inpaint data).
#
# Prerequisites:
#   - base checkpoint at checkpoints/rajan/model_best.pth
#   - data/ assembled per docs/DATASET.md (aligned train data + inpaint data)
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH

python pipelines/rajan_mask_gated.py \
    mode=train \
    pipeline_name=dear_r \
    pretrain_path=checkpoints/rajan/model_best.pth \
    exclude_top_ratio=0.3 \
    exclude_bottom_ratio=0.3 \
    num_test_samples=100

echo "DEAR-r training complete -> results/dear_r/model_best.pth"
