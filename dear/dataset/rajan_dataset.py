import os
import time
import random
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

from dear.dataset.transforms import make_transform
from dear.dataset.folder_dataset import FolderDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RajanDataset(Dataset):
    """Dataset for Rajan (AlignedForensics).

    Supports:
    - use_inversions: pair each real image with its aligned fake counterpart
      (VAE reconstruction), matched by filename.
    - batched_syncing: apply the same augmentation to the real/fake pair so the
      only distinguishing factor is the generative artifact.
    """

    def __init__(
        self,
        real_dir,
        fake_dir,
        transform=None,
        batched_syncing=False,
        use_inversions=False,
        seed=17,
    ):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        self.batched_syncing = batched_syncing
        self.use_inversions = use_inversions

        paths = sorted(os.listdir(real_dir))

        self.files = []
        random.seed(seed)

        if use_inversions:
            # Aligned pairs: real image -> fake counterpart (same stem, .png)
            for path in paths:
                rpath = os.path.join(self.real_dir, path)
                fpath = os.path.join(self.fake_dir, path.replace('.jpg', '.png'))

                self.files.append((rpath, 0))
                if not batched_syncing:
                    self.files.append((fpath, 1))
        else:
            # Random pairing of real and fake
            fake_paths = self._get_all_fakes(self.fake_dir)
            fake_paths = random.sample(fake_paths, len(paths))

            for idx, path in enumerate(paths):
                rpath = os.path.join(self.real_dir, path)
                fpath = fake_paths[idx]
                self.files.append((rpath, 0))
                self.files.append((fpath, 1))

        self.targets = [label for _, label in self.files]

    def _get_all_fakes(self, fake_dir):
        return glob.glob(os.path.join(fake_dir, '*.*'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path, target = self.files[index]
        sample = Image.open(path).convert('RGB')

        if self.batched_syncing:
            # Apply the same augmentation to both real and fake
            SEED = int(time.time() * 1000000) % (2**32)

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            sample = self.transform(sample)

            fname = os.path.basename(path).replace('.jpg', '.png')
            fpath = os.path.join(self.fake_dir, fname)
            if not os.path.exists(fpath):
                fpath = fpath.replace('.png', '.jpg')
            fsample = Image.open(fpath).convert('RGB')

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            fsample = self.transform(fsample)

            return (
                {"image": sample, "target": 0, "path": str(path)},
                {"image": fsample, "target": 1, "path": str(fpath)}
            )
        else:
            if self.transform is not None:
                sample = self.transform(sample)
            return {"image": sample, "target": target, "path": str(path)}


def load_rajan_dataset(
    root,
    train_real_types,
    train_fake_types,
    val_real_types,
    val_fake_types,
    transform_cfg=None,
    use_inversions=False,
    batched_syncing=False,
    seed=17,
):
    """Load the Rajan train/val datasets (aligned real/fake pairs for training).

    Args:
        root: Root directory of the dataset.
        train_real_types: Real subsets for training (e.g. ['coco', 'lsun']).
        train_fake_types: Fake subsets for training (e.g. ['aligned']).
        val_real_types: Real subsets for validation.
        val_fake_types: Fake subsets for validation.
        transform_cfg: Transform configuration.
        use_inversions: Use aligned GAN/VAE inversion pairs.
        batched_syncing: Sync augmentations across each real/fake pair.
        seed: Random seed.

    Returns:
        (train_dataset, val_dataset)
    """
    transform = make_transform(transform_cfg) if transform_cfg else None

    train_datasets = []
    for real_type in train_real_types:
        for fake_type in train_fake_types:
            real_dir = os.path.join(root, 'train', 'real', real_type)
            fake_dir = os.path.join(root, 'train', 'fake', fake_type)
            train_datasets.append(RajanDataset(
                real_dir=real_dir,
                fake_dir=fake_dir,
                transform=transform,
                batched_syncing=batched_syncing,
                use_inversions=use_inversions,
                seed=seed,
            ))

    # Validation: independent real/fake loading (same as Corvi)
    val_datasets = []
    for val_real_type in val_real_types:
        val_datasets.append(FolderDataset(
            os.path.join(root, 'val', 'real', val_real_type),
            target=0, transform=transform,
        ))
    for val_fake_type in val_fake_types:
        val_datasets.append(FolderDataset(
            os.path.join(root, 'val', 'fake', val_fake_type),
            target=1, transform=transform,
        ))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    print(f"# train samples = {len(train_dataset)}")
    print(f"# val samples = {len(val_dataset)}")

    return train_dataset, val_dataset
