#!/bin/bash
# Quick DEAR demo: predict real/fake for a single image or a folder of images.
#
# Usage:
#   bash scripts/inference/run_demo.sh [dear_c|dear_r] <image_or_folder>
#
# Requires the released checkpoint at checkpoints/{model}/model_best.pth.
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH

MODEL=${1:-dear_c}
INPUT=${2:-assets/samples}

python scripts/inference.py --model "$MODEL" --input "$INPUT"
