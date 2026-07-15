import os
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from dear.dataset.transforms import (
    _apply_blur,
    _apply_compression,
    _create_cutout_transform,
    _create_noise_transform,
)


class InpaintDataset(Dataset):
    """Load (inpainted image, mask) pairs for the diagnostic dataset.

    The target is decided dynamically from the crop:
    - mask has foreground pixels  -> target=1 (fake), apply_align=True
    - mask is empty after cropping -> target=0 (real), apply_align=False

    ``apply_align`` flags samples whose mask survives the random crop; only those
    are used when computing channel--mask alignment (RAD) scores.
    """

    def __init__(self, inpaint_dir, mask_dir, transform_cfg=None, deterministic_seed=None):
        self.inpaint_dir = inpaint_dir
        self.mask_dir = mask_dir
        self.transform_cfg = transform_cfg
        self.deterministic_seed = deterministic_seed
        self.files = self._collect_files()

    def _collect_files(self):
        # inpaint: {stem}_inpaint.png   mask: {stem}_mask.png
        files = []
        for f in sorted(os.listdir(self.inpaint_dir)):
            if f.endswith('_inpaint.png'):
                stem = f.replace('_inpaint.png', '')
                mask_path = os.path.join(self.mask_dir, f'{stem}_mask.png')
                if os.path.exists(mask_path):
                    files.append((os.path.join(self.inpaint_dir, f), mask_path))
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        inpaint_path, mask_path = self.files[index]

        image = Image.open(inpaint_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image, mask = self.sync_transform(image, mask, index)

        # Target depends on whether mask content survived the crop
        mask_has_content = mask.sum() > 0
        target = 1 if mask_has_content else 0
        apply_align = mask_has_content

        return {
            'image': image,
            'target': target,
            'mask': mask,                              # (1, H, W)
            'apply_align': torch.as_tensor(apply_align),
            'path': inpaint_path,
        }

    def sync_transform(self, image, mask, index=0):
        """Apply matched geometric transforms to image and mask."""
        if self.deterministic_seed is not None:
            SEED = (self.deterministic_seed + index) % (2**32)
        else:
            SEED = np.random.randint(0, 2**32)

        image, mask = self._apply_augmentation(image, mask, SEED)
        image, mask = self._apply_crop(image, mask, SEED)

        image = self._normalize(image)

        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        mask = (mask > 0.5).float().unsqueeze(0)  # binary, (1, H, W)

        return image, mask

    def _apply_augmentation(self, image, mask, seed):
        """Geometric transforms apply to both; color/degradation to image only."""
        cfg = self.transform_cfg
        if cfg is None:
            return image, mask

        # RandomResizedCrop (geometric -> both)
        if getattr(cfg, 'resize_size', 0) > 0 and getattr(cfg, 'resize_prob', 0) > 0:
            random.seed(seed)
            if random.random() < cfg.resize_prob:
                random.seed(seed + 1)
                torch.manual_seed(seed + 1)
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.08, 1.0),
                    ratio=(cfg.resize_ratio, 1.0 / cfg.resize_ratio),
                )
                image = TF.resized_crop(image, i, j, h, w, cfg.resize_size)
                mask = TF.resized_crop(mask, i, j, h, w, cfg.resize_size,
                                       interpolation=TF.InterpolationMode.NEAREST)

        # ColorJitter (image only)
        if getattr(cfg, 'jitter_prob', 0) > 0:
            random.seed(seed + 2)
            if random.random() < cfg.jitter_prob:
                image = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(image)

        # Grayscale (image only)
        if getattr(cfg, 'colordist_prob', 0) > 0:
            random.seed(seed + 3)
            if random.random() < cfg.colordist_prob:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

        # Cutout (image only)
        if getattr(cfg, 'cutout_prob', 0) > 0:
            random.seed(seed + 4)
            image = _create_cutout_transform(cfg.cutout_prob)(image)

        # Noise (image only)
        if getattr(cfg, 'noise_prob', 0) > 0:
            random.seed(seed + 5)
            image = _create_noise_transform(cfg.noise_prob)(image)

        # Blur (image only)
        if getattr(cfg, 'blur_prob', 0) > 0:
            random.seed(seed + 6)
            image = _apply_blur(image, cfg.blur_prob, cfg.blur_sig)

        # Compression (image only)
        if getattr(cfg, 'cmp_prob', 0) > 0:
            random.seed(seed + 7)
            image = _apply_compression(image, cfg.cmp_prob, cfg.cmp_method, cfg.cmp_qual)

        # 90-degree rotation (geometric -> both)
        if getattr(cfg, 'rot90_prob', 0) > 0:
            random.seed(seed + 8)
            if random.random() < cfg.rot90_prob:
                random.seed(seed + 9)
                angle = random.choice([0, 90, 180, 270])
                image = image.rotate(angle, expand=True)
                mask = mask.rotate(angle, expand=True)

        # Horizontal flip (geometric -> both)
        if hasattr(cfg, 'no_flip') and not cfg.no_flip:
            random.seed(seed + 10)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        return image, mask

    def _apply_crop(self, image, mask, seed):
        """Apply the same RandomCrop to image and mask."""
        cfg = self.transform_cfg
        if cfg is None or not hasattr(cfg, 'crop_size') or cfg.crop_size <= 0:
            return image, mask

        random.seed(seed + 100)
        torch.manual_seed(seed + 100)

        w, h = image.size
        if w < cfg.crop_size or h < cfg.crop_size:
            pad_w = max(0, cfg.crop_size - w)
            pad_h = max(0, cfg.crop_size - h)
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
            image = TF.pad(image, padding, padding_mode='symmetric')
            mask = TF.pad(mask, padding, fill=0)

        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(cfg.crop_size, cfg.crop_size)
        )
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        return image, mask

    def _normalize(self, image):
        image = TF.to_tensor(image)
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        return image


def load_inpaint_dataset(
    root,
    train_fake_inpaint_types,
    train_fake_inpaint_mask_types,
    transform_cfg=None,
):
    """Load one or more inpaint diagnostic datasets as a ConcatDataset.

    Args:
        root: Root directory of the dataset.
        train_fake_inpaint_types: Inpaint image subdirs (e.g. ['lsun_inpaint_sd']).
        train_fake_inpaint_mask_types: Matching mask subdirs (e.g. ['lsun_inpaint_mask']).
        transform_cfg: Transform configuration.

    Returns:
        torch.utils.data.ConcatDataset of InpaintDataset(s).
    """
    assert len(train_fake_inpaint_types) == len(train_fake_inpaint_mask_types), \
        "inpaint types and mask types must have the same length"

    datasets = []
    for inpaint_type, mask_type in zip(train_fake_inpaint_types, train_fake_inpaint_mask_types):
        dataset = InpaintDataset(
            inpaint_dir=os.path.join(root, 'train', 'fake', inpaint_type),
            mask_dir=os.path.join(root, 'train', 'fake', mask_type),
            transform_cfg=transform_cfg,
        )
        datasets.append(dataset)
        print(f"  # {inpaint_type} samples = {len(dataset)}")

    inpaint_train_dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"# total inpaint train samples = {len(inpaint_train_dataset)}")

    return inpaint_train_dataset
