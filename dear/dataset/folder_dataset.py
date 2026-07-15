from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset


class FolderDataset(Dataset):
    """Load every image in a folder and assign a fixed target label."""

    def __init__(self, root, target, transform=None):
        self.root = root
        self.samples = sorted(list(Path(root).glob('*')))
        self.targets = [target] * len(self.samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        target = self.targets[index]
        image = Image.open(sample_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return {
            'image': image,
            'target': target,
            'path': str(sample_path)
        }
