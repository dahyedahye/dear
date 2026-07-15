import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import torchvision.transforms as transforms


def make_transform(transform_cfg):
    """Create the training/validation transform pipeline.

    Args:
        transform_cfg: Transform configuration (e.g., args.transform).

    Returns:
        transforms.Compose object.
    """
    transform_list = []

    aug_transforms = make_augmentation(transform_cfg)
    if aug_transforms is not None:
        transform_list.append(aug_transforms)

    post_transforms = make_post_processing(transform_cfg)
    if post_transforms is not None:
        transform_list.append(post_transforms)

    transform_list.append(make_normalize(transform_cfg.norm_type))

    return transforms.Compose(transform_list)


def make_augmentation(cfg):
    """Create data augmentation transforms (compression, blur, jitter, ...)."""
    aug_list = []

    # RandomResizedCrop
    if cfg.resize_size > 0 and cfg.resize_prob > 0:
        aug_list.append(
            transforms.RandomApply(
                [transforms.RandomResizedCrop(
                    size=cfg.resize_size,
                    scale=(0.08, 1.0),
                    ratio=(cfg.resize_ratio, 1.0 / cfg.resize_ratio),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )],
                p=cfg.resize_prob,
            )
        )

    # ColorJitter
    if cfg.jitter_prob > 0:
        aug_list.append(
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=cfg.jitter_prob,
            )
        )

    # Grayscale
    if cfg.colordist_prob > 0:
        aug_list.append(transforms.RandomGrayscale(p=cfg.colordist_prob))

    # Cutout
    if cfg.cutout_prob > 0:
        aug_list.append(_create_cutout_transform(cfg.cutout_prob))

    # Noise
    if cfg.noise_prob > 0:
        aug_list.append(_create_noise_transform(cfg.noise_prob))

    # Blur
    if cfg.blur_prob > 0:
        aug_list.append(
            transforms.Lambda(
                lambda img: _apply_blur(img, cfg.blur_prob, cfg.blur_sig)
            )
        )

    # Compression
    if cfg.cmp_prob > 0:
        aug_list.append(
            transforms.Lambda(
                lambda img: _apply_compression(
                    img, cfg.cmp_prob, cfg.cmp_method, cfg.cmp_qual
                )
            )
        )

    # Rotation (90 degrees)
    if cfg.rot90_prob > 0:
        aug_list.append(
            transforms.Lambda(lambda img: _apply_rotation(img, cfg.rot90_prob))
        )

    # Horizontal flip
    if not cfg.no_flip:
        aug_list.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(aug_list) if aug_list else None


def make_post_processing(cfg):
    """Create post-processing transforms (random crop)."""
    post_list = []

    if cfg.crop_size > 0:
        post_list.append(
            transforms.RandomCrop(
                cfg.crop_size,
                pad_if_needed=True,
                padding_mode='symmetric',
            )
        )

    return transforms.Compose(post_list) if post_list else None


def make_normalize(norm_type):
    """Create the ToTensor + Normalize transform (ImageNet stats)."""
    if norm_type == 'resnet':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------
def _apply_blur(img, prob, blur_sig):
    """Apply Gaussian blur."""
    if random.random() < prob:
        img = np.array(img)
        sig = random.uniform(blur_sig[0], blur_sig[1])
        gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sig)
        gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sig)
        gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sig)
        img = Image.fromarray(img)
    return img


def _apply_compression(img, prob, cmp_methods, cmp_qual):
    """Apply JPEG compression (cv2 or PIL)."""
    if random.random() < prob:
        img = np.array(img)
        method = random.choice(cmp_methods)
        qual = random.randint(cmp_qual[0], cmp_qual[1])

        if method == 'cv2':
            img = _cv2_jpg(img, qual)
        elif method == 'pil':
            img = _pil_jpg(img, qual)

        img = Image.fromarray(img)
    return img


def _apply_rotation(img, prob):
    """Apply a random 90-degree rotation."""
    if random.random() < prob:
        angle = random.choice([0, 90, 180, 270])
        img = img.rotate(angle, expand=True)
    return img


def _cv2_jpg(img, quality):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def _pil_jpg(img, quality):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=quality)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


def _create_cutout_transform(prob):
    """Create a cutout transform using albumentations."""
    try:
        from albumentations.augmentations.dropout.cutout import Cutout
    except Exception:
        from albumentations.augmentations.transforms import Cutout

    aug = Cutout(
        num_holes=1,
        max_h_size=48,
        max_w_size=48,
        fill_value=128,
        always_apply=False,
        p=prob,
    )
    return transforms.Lambda(
        lambda img: Image.fromarray(aug(image=np.array(img))['image'])
    )


def _create_noise_transform(prob, var_limit=(10.0, 50.0)):
    """Create a Gaussian noise transform using albumentations."""
    from albumentations.augmentations.transforms import GaussNoise

    aug = GaussNoise(var_limit=var_limit, always_apply=False, p=prob)
    return transforms.Lambda(
        lambda img: Image.fromarray(aug(image=np.array(img))['image'])
    )
