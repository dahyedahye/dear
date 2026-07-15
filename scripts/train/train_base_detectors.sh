#!/bin/bash
# (Optional) Train the base detectors from scratch.
#
# Most users should download the released base checkpoints instead
# (checkpoints/corvi/model_best.pth, checkpoints/rajan/model_best.pth) and go
# straight to train_dear_c.sh / train_dear_r.sh. This script is provided only
# for full from-scratch reproduction.
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH

# Base Corvi detector (real LSUN/COCO vs. LDM fakes).
python pipelines/corvi.py mode=train pipeline_name=corvi

# Base Rajan detector (aligned real/fake pairs).
python pipelines/rajan.py mode=train pipeline_name=rajan

echo "Base detectors trained -> results/corvi/model_best.pth, results/rajan/model_best.pth"
