#!/usr/bin/env python3
"""DEAR single-image / folder inference (quick demo).

Predict real vs. AI-generated for one image or every image in a folder, using a
trained DEAR checkpoint (DEAR-c or DEAR-r). Each image is processed at its native
resolution (ToTensor + ImageNet normalize, no resize/crop) so the fragile
generative artifacts DEAR relies on are preserved.

A positive logit -> FAKE (AI-generated); a negative logit -> REAL.

Examples:
    python scripts/inference.py --model dear_c --input path/to/image.png
    python scripts/inference.py --model dear_r --input path/to/folder/
    python scripts/inference.py --model dear_c \\
        --ckpt checkpoints/dear_c/model_best.pth --input img.jpg
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

DEFAULT_CKPTS = {
    'dear_c': 'checkpoints/dear_c/model_best.pth',
    'dear_r': 'checkpoints/dear_r/model_best.pth',
}


def build_detector(model_name, ckpt_path, device):
    if model_name == 'dear_c':
        from dear.detector.corvi_mask_gated_detector import CorviMaskGatedDetector
        detector = CorviMaskGatedDetector(device=device)
    elif model_name == 'dear_r':
        from dear.detector.rajan_mask_gated_detector import RajanMaskGatedDetector
        detector = RajanMaskGatedDetector(device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    detector.load(ckpt_path)
    detector.eval()
    return detector


def collect_images(input_path):
    p = Path(input_path)
    if p.is_dir():
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXT)
    return [str(p)]


def main():
    parser = argparse.ArgumentParser(description="DEAR single-image/folder inference")
    parser.add_argument('--model', choices=['dear_c', 'dear_r'], default='dear_c')
    parser.add_argument('--input', required=True, help='Image file or a folder of images')
    parser.add_argument('--ckpt', default=None,
                        help='Checkpoint path (default: checkpoints/{model}/model_best.pth)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    ckpt_path = args.ckpt or DEFAULT_CKPTS[args.model]
    device = args.device if torch.cuda.is_available() else 'cpu'

    detector = build_detector(args.model, ckpt_path, device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    images = collect_images(args.input)
    if not images:
        print(f"No images found at {args.input}")
        return

    print(f"[{args.model}] {ckpt_path}  device={device}")
    print(f"{'image':<50} {'logit':>9} {'p(fake)':>9}  verdict")
    print('-' * 82)
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = detector.predict(x).squeeze().item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        verdict = 'FAKE' if logit > 0 else 'REAL'
        print(f"{Path(img_path).name:<50} {logit:>9.4f} {prob:>9.4f}  {verdict}")


if __name__ == '__main__':
    main()
