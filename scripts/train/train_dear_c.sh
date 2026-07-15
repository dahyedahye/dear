#!/bin/bash
# Train DEAR-c: apply DEAR (Dissect and Prune) to the base Corvi detector.
#
# Paper setting: prune the top 20% + bottom 20% of channels by RAD score
# (keep the robust middle 60%), then refine the linear head on
# (Corvi train data + diagnostic inpaint data).
#
# Prerequisites:
#   - base checkpoint at checkpoints/corvi/model_best.pth
#   - data/ assembled per docs/DATASET.md (train data + inpaint data)
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH

python pipelines/corvi_mask_gated.py \
    mode=train \
    pipeline_name=dear_c \
    pretrain_path=checkpoints/corvi/model_best.pth \
    exclude_top_ratio=0.2 \
    exclude_bottom_ratio=0.2 \
    num_test_samples=100

echo "DEAR-c training complete -> results/dear_c/model_best.pth"
