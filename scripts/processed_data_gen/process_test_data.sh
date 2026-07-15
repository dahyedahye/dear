#!/bin/bash
# Build the post-processed test set (test_processed) used for the robustness
# evaluation, by applying random JPEG compression / resize / color jitter to each
# test image. Run once per generator subset (real and fake).
#
# Credit: the perturbation protocol follows AlignedForensics
# (https://github.com/AniSundar18/AlignedForensics, issue #9).
set -e

# Real subset
python scripts/processed_data_gen/process_test_data.py \
    --input_folder data/test/real/redcaps \
    --output_folder data/test_processed/real/redcaps \
    --seed 42

# Fake subsets
for fake_type in sd Midjourney kandinsky playground pixelart lcm flux wuerstchen amused Chameleon loki WildRF; do
    python scripts/processed_data_gen/process_test_data.py \
        --input_folder data/test/fake/$fake_type \
        --output_folder data/test_processed/fake/$fake_type \
        --seed 42
done
