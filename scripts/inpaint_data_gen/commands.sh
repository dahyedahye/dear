#!/bin/bash
# Generate the DEAR diagnostic inpaint dataset from real LSUN images.
#
# Step 1: sample a random rectangular mask per real image.
# Step 2: inpaint the masked region with Stable Diffusion 1.5 and composite back.
#
# Outputs (consumed by training via train_fake_inpaint_types / _mask_types):
#   data/train/fake/lsun_inpaint_mask/{stem}_mask.png
#   data/train/fake/lsun_inpaint_sd/{stem}_inpaint.png
#
# The released diagnostic dataset is also available on the Hugging Face Hub
# (see docs/DATASET.md) so this step can be skipped.
set -e

# Step 1: masks
python scripts/inpaint_data_gen/gen_mask.py \
    --input_folder data/train/real/lsun \
    --output_folder data/train/fake/lsun_inpaint_mask

# Step 2: SD-1.5 inpainting (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/inpaint_data_gen/gen_inpaint_data.py \
    --num_gpus 8 \
    --input_folder data/train/real/lsun \
    --mask_folder data/train/fake/lsun_inpaint_mask \
    --output_folder data/train/fake/lsun_inpaint_sd
