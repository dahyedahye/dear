#!/usr/bin/env python3
"""Build the post-processed test set (robustness evaluation).

Applies random JPEG compression, resizing, and color jittering to every image in
a folder -- the perturbations used to measure DEAR's robustness in the paper
(Table 1, bottom / "test_processed"). Each output is saved as JPEG.

Transformations (always applied):
    - JPEG quality:  50-95
    - Resize:        clamped to [256, 1024] px
    - Color jitter:  brightness / contrast / sharpness / color in [0.8, 1.2]

Credit: this perturbation protocol follows AlignedForensics
(https://github.com/AniSundar18/AlignedForensics, see issue #9).

Example:
    python scripts/processed_data_gen/process_test_data.py \\
        --input_folder data/test/fake/WildRF \\
        --output_folder data/test_processed/fake/WildRF \\
        --seed 42
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance
from tqdm import tqdm


def apply_random_transformations(image):
    quality = random.randint(50, 95)
    original_width, original_height = image.size

    if random.choice([True, False]):
        scale_factor = random.uniform(1.0, min(1024 / original_width, 1024 / original_height))
    else:
        scale_factor = random.uniform(max(256 / original_width, 256 / original_height), 1.0)

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    new_width = max(256, min(1024, new_width))
    new_height = max(256, min(1024, new_height))
    image = image.resize((new_width, new_height))

    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))

    return image, quality


def process_image(img_path, output_path):
    try:
        image = Image.open(img_path).convert('RGB')
        transformed_image, quality = apply_random_transformations(image)
        output_path = Path(output_path).with_suffix('.jpg')
        transformed_image.save(output_path, format='JPEG', quality=quality)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Build the post-processed test set')
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    img_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    img_files = [f for f in input_folder.iterdir()
                 if f.is_file() and f.suffix.lower() in img_extensions]

    print(f"Processing {len(img_files)} images from {input_folder}")
    print(f"Output folder: {output_folder}")

    success_count = 0
    for img_path in tqdm(img_files, desc="Processing"):
        if process_image(img_path, output_folder / img_path.name):
            success_count += 1

    print(f"Successfully processed {success_count}/{len(img_files)} images")


if __name__ == '__main__':
    main()
