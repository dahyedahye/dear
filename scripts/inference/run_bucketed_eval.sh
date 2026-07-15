#!/bin/bash
# Reproduce the DEAR benchmark table: batched inference + per-generator metrics
# on both the original and the post-processed test sets.
#
# Usage:
#   bash scripts/inference/run_bucketed_eval.sh [corvi|rajan|dear_c|dear_r] [data_root]
#
# Requires:
#   - released checkpoint at checkpoints/{model}/model_best.pth
#   - data/ assembled per docs/DATASET.md (test + test_processed)
set -e
export PYTHONPATH=$(pwd):$PYTHONPATH

MODEL=${1:-dear_c}
DATA_ROOT=${2:-./data}

# 1) Batched (size-bucketed) inference over the original + post-processed sets.
python scripts/bucketed_inference.py \
    --model "$MODEL" \
    --data_root "$DATA_ROOT" \
    --test_dirnames test test_processed

# 2) Per-generator metrics (AP / AUC / R.Acc / F.Acc / Acc).
for test_dirname in test test_processed; do
    echo "===== $MODEL / $test_dirname ====="
    for fake_type in sd Midjourney kandinsky playground pixelart lcm flux wuerstchen amused Chameleon loki WildRF; do
        python scripts/eval.py \
            --real_csv results/"$MODEL"/"$test_dirname"/real/redcaps.csv \
            --fake_csv results/"$MODEL"/"$test_dirname"/fake/"$fake_type".csv
    done
done

# 3) Summary table (AUC / Acc / R.Acc / F.Acc, averaged over the 9 generators).
python scripts/analyze_results.py \
    --results_dir results --methods "$MODEL" \
    --test_dirs test test_processed
