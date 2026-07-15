import os

import torch

from dear.dataset.folder_dataset import FolderDataset
from dear.dataset.transforms import make_transform


def load_corvi_dataset(
    root, train_real_types, train_fake_types, val_real_types, val_fake_types,
    transform_cfg=None
):
    """Load the Corvi train/val datasets.

    Data layout expected under ``root``::

        root/train/real/{type}/*.png
        root/train/fake/{type}/*.png
        root/val/real/{type}/*.png
        root/val/fake/{type}/*.png

    Args:
        root: Root directory of the dataset.
        train_real_types: Real image subsets for training (e.g. ['coco', 'lsun']).
        train_fake_types: Fake image subsets for training (e.g. ['ldm']).
        val_real_types: Real image subsets for validation.
        val_fake_types: Fake image subsets for validation.
        transform_cfg: Transform configuration (e.g. args.transform).

    Returns:
        (train_dataset, val_dataset)
    """
    transform = make_transform(transform_cfg) if transform_cfg else None

    train_dataset = []
    for train_real_type in train_real_types:
        train_dataset.append(FolderDataset(
            os.path.join(root, 'train', 'real', train_real_type),
            target=0, transform=transform,
        ))
    for train_fake_type in train_fake_types:
        train_dataset.append(FolderDataset(
            os.path.join(root, 'train', 'fake', train_fake_type),
            target=1, transform=transform,
        ))

    val_dataset = []
    for val_real_type in val_real_types:
        val_dataset.append(FolderDataset(
            os.path.join(root, 'val', 'real', val_real_type),
            target=0, transform=transform,
        ))
    for val_fake_type in val_fake_types:
        val_dataset.append(FolderDataset(
            os.path.join(root, 'val', 'fake', val_fake_type),
            target=1, transform=transform,
        ))

    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset)

    print(f"# train samples = {len(train_dataset)}")
    print(f"# val samples = {len(val_dataset)}")

    return train_dataset, val_dataset
