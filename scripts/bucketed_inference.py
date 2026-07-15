#!/usr/bin/env python3
"""Bucketed batched inference over a benchmark, for DEAR (DEAR-c / DEAR-r).

DEAR never resizes images (to preserve generative artifacts), so images come in
many resolutions. This script groups images by size into buckets and runs
DataLoader-batched inference per bucket: results are identical to single-image
inference but throughput is far higher. One logit CSV is written per real/fake
subset; pair it with ``scripts/eval.py`` to reproduce the paper tables.

Expected data layout::

    {data_root}/{test_dirname}/real/{type}/*.png
    {data_root}/{test_dirname}/fake/{type}/*.png

Example:
    python scripts/bucketed_inference.py --model dear_c \\
        --data_root ./data --test_dirnames test test_processed
"""

import argparse
import os
import random
import struct
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dear.dataset.transforms import make_normalize

FAKE_TYPES = ['sd', 'Midjourney', 'kandinsky', 'playground', 'pixelart',
              'lcm', 'flux', 'wuerstchen', 'amused', 'Chameleon', 'loki', 'WildRF']

DEFAULT_CKPTS = {
    'corvi':  'checkpoints/corvi/model_best.pth',
    'rajan':  'checkpoints/rajan/model_best.pth',
    'dear_c': 'checkpoints/dear_c/model_best.pth',
    'dear_r': 'checkpoints/dear_r/model_best.pth',
}


class PathListDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        return {'image': self.transform(image), 'path': self.paths[idx]}


def _get_image_size(path):
    """Fast image-size probe (reads only the PNG header when possible)."""
    ext = str(path).rsplit('.', 1)[-1].lower()
    if ext == 'png':
        with open(path, 'rb') as f:
            f.read(16)
            w = struct.unpack('>I', f.read(4))[0]
            h = struct.unpack('>I', f.read(4))[0]
        return (w, h)
    with Image.open(path) as img:
        return img.size


def _filter_paths(paths, max_images=None, seed=0):
    if max_images is None or max_images >= len(paths):
        return list(paths)
    rng = random.Random(seed)
    return sorted(rng.sample(list(paths), max_images))


def group_by_size(image_dir, max_images=None, seed=0):
    candidates = []
    for p in sorted(Path(image_dir).iterdir()):
        try:
            if p.is_file():
                candidates.append(str(p))
        except (PermissionError, OSError):
            continue
    candidates = _filter_paths(candidates, max_images=max_images, seed=seed)
    buckets = defaultdict(list)
    for p in candidates:
        try:
            buckets[_get_image_size(p)].append(p)
        except Exception:
            continue
    return buckets


def _adapt_batch_size(base_batch_size, image_size):
    """Shrink the batch for large images to stay within GPU memory."""
    pixels = image_size[0] * image_size[1]
    safe_pixels = 1024 * 1024
    safe_batch = 16
    return max(1, min(int(safe_batch * safe_pixels / max(pixels, 1)), base_batch_size))


def run_bucketed_inference(detector, image_dir, transform, batch_size, num_workers, device,
                           max_images=None, seed=0):
    buckets = group_by_size(image_dir, max_images=max_images, seed=seed)
    if not buckets:
        return [], [], 0.0, 0

    all_paths, all_logits = [], []
    total_images = sum(len(v) for v in buckets.values())
    start = time.time()
    pbar = tqdm.tqdm(total=total_images, desc=os.path.basename(image_dir))

    with torch.no_grad():
        for size, paths in sorted(buckets.items(), key=lambda x: -len(x[1])):
            adapted_bs = _adapt_batch_size(batch_size, size)
            loader = DataLoader(PathListDataset(paths, transform), batch_size=adapted_bs,
                                shuffle=False, num_workers=num_workers, pin_memory=True)
            for batch in loader:
                x = batch['image'].to(device)
                logits = detector.predict(x).squeeze(-1)
                all_logits.extend(logits.cpu().tolist())
                all_paths.extend(batch['path'])
                pbar.update(len(batch['path']))

    pbar.close()
    return all_paths, all_logits, time.time() - start, len(buckets)


def save_csv(paths, logits, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame({'path': paths, 'logit': logits}).to_csv(csv_path, index=False)


def build_detector(model_name, ckpt_path, device):
    if model_name == 'dear_c':
        from dear.detector.corvi_mask_gated_detector import CorviMaskGatedDetector
        detector = CorviMaskGatedDetector(device=device)
    elif model_name == 'dear_r':
        from dear.detector.rajan_mask_gated_detector import RajanMaskGatedDetector
        detector = RajanMaskGatedDetector(device=device)
    elif model_name == 'corvi':
        from dear.detector.corvi_detector import CorviDetector
        detector = CorviDetector(device=device, pretrained=False)
    else:  # rajan
        from dear.detector.rajan_detector import RajanDetector
        detector = RajanDetector(device=device, pretrained=False)
    detector.load(ckpt_path)
    detector.eval()
    return detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['corvi', 'rajan', 'dear_c', 'dear_r'], required=True)
    parser.add_argument('--ckpt', default=None,
                        help='Checkpoint path (default: checkpoints/{model}/model_best.pth)')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--pipeline_name', default=None,
                        help='Output subdir under results_dir (default: model name)')
    parser.add_argument('--fake_types', nargs='+', default=None,
                        help='Subset of fake generators to evaluate (default: all).')
    parser.add_argument('--real_types', nargs='+', default=None,
                        help='Real subsets (default: redcaps).')
    parser.add_argument('--test_dirnames', nargs='+', default=['test', 'test_processed'])
    parser.add_argument('--max_images', type=int, default=None,
                        help='Max samples per subset (deterministic random sample).')
    parser.add_argument('--sample_seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--out_suffix', default='')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    ckpt_path = args.ckpt or DEFAULT_CKPTS[args.model]
    pipeline_name = args.pipeline_name or args.model
    fake_types = args.fake_types or FAKE_TYPES
    real_types = args.real_types or ['redcaps']
    device = args.device if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    detector = build_detector(args.model, ckpt_path, device)
    transform = make_normalize('resnet')
    print(f"[{args.model}] {ckpt_path}  device={device}")

    for test_dirname in args.test_dirnames:
        out_dirname = f"{test_dirname}{args.out_suffix}"
        csv_dir = os.path.join(args.results_dir, pipeline_name, out_dirname)
        total_time, total_images = 0.0, 0

        for label, types in [('real', real_types), ('fake', fake_types)]:
            for data_type in types:
                dataset_path = os.path.join(args.data_root, test_dirname, label, data_type)
                if not os.path.isdir(dataset_path):
                    print(f"Skipping {dataset_path} (not found)")
                    continue
                paths, logits, elapsed, n_buckets = run_bucketed_inference(
                    detector, dataset_path, transform, batch_size=args.batch_size,
                    num_workers=args.num_workers, device=device,
                    max_images=args.max_images, seed=args.sample_seed)
                csv_path = os.path.join(csv_dir, label, f'{data_type}.csv')
                save_csv(paths, logits, csv_path)
                total_time += elapsed
                total_images += len(paths)
                print(f"  {label}/{data_type}: {len(paths)} imgs, {n_buckets} size groups, "
                      f"{elapsed:.1f}s ({len(paths)/max(elapsed,1e-6):.1f} img/s)")

        print(f"[{out_dirname}] total {total_images} imgs, {total_time:.1f}s\n")


if __name__ == '__main__':
    main()
