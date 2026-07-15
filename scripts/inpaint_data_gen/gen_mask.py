#!/usr/bin/env python3
"""Generate random rectangular inpainting masks for the diagnostic dataset.

For each real image, a single random rectangle (2%-20% of the image area) is
marked as the region to be inpainted. Masks are saved as ``{stem}_mask.png`` and
later consumed by ``gen_inpaint_data.py``.

Example:
    python scripts/inpaint_data_gen/gen_mask.py \\
        --input_folder data/train/real/lsun \\
        --output_folder data/train/fake/lsun_inpaint_mask
"""

import os
import argparse
import random

import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_random_rect_mask(width, height,
                              min_area_ratio=0.02,
                              max_area_ratio=0.2,
                              min_aspect=0.5,
                              max_aspect=2.0):
    img_area = width * height
    target_area = random.uniform(min_area_ratio, max_area_ratio) * img_area
    aspect_ratio = random.uniform(min_aspect, max_aspect)

    h = int(round((target_area / aspect_ratio) ** 0.5))
    w = int(round((target_area * aspect_ratio) ** 0.5))
    h = min(h, height)
    w = min(w, width)

    top = random.randint(0, height - h)
    left = random.randint(0, width - w)

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:top + h, left:left + w] = 255
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='data/train/real/lsun',
                        help='Folder of real images to mask.')
    parser.add_argument('--output_folder', type=str, default='data/train/fake/lsun_inpaint_mask',
                        help='Where to save generated masks.')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    image_paths = [
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]
    print(f"Total images found: {len(image_paths)}")

    for image_path in tqdm(image_paths):
        base_name = os.path.basename(image_path)
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        mask_np = generate_random_rect_mask(width=w, height=h)
        mask_img = Image.fromarray(mask_np, mode="L").convert("RGB")

        save_name = base_name.split('.')[0] + "_mask.png"
        mask_img.save(os.path.join(args.output_folder, save_name))

    print("Done!")


if __name__ == "__main__":
    main()
