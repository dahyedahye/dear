"""Shared helpers for benchmark inference and evaluation (Hydra pipelines)."""

import os

import tqdm
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score

from dear.utils import webp_compress
from dear.dataset.transforms import make_normalize
from dear.dataset.folder_dataset import FolderDataset


def make_test_transform(args):
    """Build the inference transform and a csv-dir suffix.

    Optionally applies WebP compression / resizing to probe robustness; the
    suffix records which perturbation was used (e.g. '_webp80_resize0.5').
    """
    transform_list = []
    csv_suffix = ''

    if args.webp_quality is not None:
        transform_list.append(lambda x: webp_compress(x, args.webp_quality))
        csv_suffix += f'_webp{args.webp_quality}'
    if args.resize_ratio is not None:
        transform_list.append(transforms.Resize(int(args.resize_ratio * 512)))
        csv_suffix += f'_resize{args.resize_ratio}'
    transform_list.append(make_normalize(args.transform.norm_type))

    return transforms.Compose(transform_list), csv_suffix


def _run_inference_on_folder(detector, args, transform, csv_dir, label, data_type):
    """Run inference over one folder and write a per-folder logit CSV."""
    test_dataset = FolderDataset(
        os.path.join(args.data_root, args.test_dirname, label, data_type),
        target=0 if label == 'real' else 1,
        transform=transform,
    )

    num_samples = args.num_test_samples if args.num_test_samples is not None else len(test_dataset)
    paths, logits = [], []

    for i in tqdm.tqdm(range(num_samples), desc=f"Testing {data_type}"):
        sample = test_dataset[i]
        x = sample['image'].to(args.device).unsqueeze(0)
        with torch.no_grad():
            logit = detector.predict(x).squeeze().item()
        paths.append(sample['path'])
        logits.append(logit)
        del x, sample

    del test_dataset
    torch.cuda.empty_cache()

    df = pd.DataFrame({'path': paths, 'logit': logits})
    save_dir = os.path.join(csv_dir, label)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'{data_type}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


def run_inference(detector, args):
    """Load the trained model and run inference over all test subsets.

    Writes one CSV of logits per real/fake subset under
    ``{save_path}/{test_dirname}{suffix}/{real,fake}/{type}.csv``.
    """
    csv_dir = os.path.join(args.save_path, args.test_dirname)
    transform, csv_suffix = make_test_transform(args)
    csv_dir += csv_suffix

    model_path = f"{args.save_path}/model_best.pth"
    print(f"Loading trained model from {model_path}")
    detector.load(model_path)
    detector.eval()
    torch.cuda.empty_cache()

    for test_real_type in args.test_real_types:
        _run_inference_on_folder(detector, args, transform, csv_dir, 'real', test_real_type)

    for test_fake_type in args.test_fake_types:
        _run_inference_on_folder(detector, args, transform, csv_dir, 'fake', test_fake_type)


def compute_test_ap(detector, args):
    """Average Precision over the test set (all fake types pooled). 0-100 scale."""
    detector.eval()
    transform = make_normalize(args.transform.norm_type)

    real_logits = []
    real_dataset = FolderDataset(
        os.path.join(args.data_root, args.test_dirname, 'real', args.test_real_types[0]),
        target=0, transform=transform,
    )
    num_real = args.num_test_samples if args.num_test_samples is not None else len(real_dataset)
    for i in tqdm.tqdm(range(num_real), desc=f"Test real ({args.test_real_types[0]})"):
        sample = real_dataset[i]
        x = sample['image'].to(args.device).unsqueeze(0)
        with torch.no_grad():
            logit = detector.predict(x).squeeze().item()
        real_logits.append(logit)
        del x
    del real_dataset
    torch.cuda.empty_cache()

    fake_logits = []
    for fake_type in args.test_fake_types:
        fake_dataset = FolderDataset(
            os.path.join(args.data_root, args.test_dirname, 'fake', fake_type),
            target=1, transform=transform,
        )
        num_fake = args.num_test_samples if args.num_test_samples is not None else len(fake_dataset)
        for i in tqdm.tqdm(range(num_fake), desc=f"Test fake ({fake_type})"):
            sample = fake_dataset[i]
            x = sample['image'].to(args.device).unsqueeze(0)
            with torch.no_grad():
                logit = detector.predict(x).squeeze().item()
            fake_logits.append(logit)
            del x
        del fake_dataset
        torch.cuda.empty_cache()

    y_true = np.concatenate([np.zeros(len(real_logits)), np.ones(len(fake_logits))])
    y_scores = np.concatenate([real_logits, fake_logits])
    return average_precision_score(y_true, y_scores) * 100
