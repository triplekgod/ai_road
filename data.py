import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

IMAGE_SIZE = (512, 288)  # width, height; original aspect ratio is preserved
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def mask_path(image_path: Path, masks_dir: Path) -> Path:
    return masks_dir / f"{image_path.stem}_mask.bmp"


def find_pairs(root: str | Path):
    root = Path(root)
    images_dir, masks_dir = root / "images", root / "masks"
    pairs = [(p, mask_path(p, masks_dir)) for p in sorted(images_dir.glob("*.bmp"))]
    return [(image, mask) for image, mask in pairs if mask.exists()]


class RoadDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = list(pairs)
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path_ = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path_).convert("L")

        if self.augment:
            if random.random() < 0.5:
                image, mask = F.hflip(image), F.hflip(mask)
            if random.random() < 0.8:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
                image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.25))

        image = F.resize(image, IMAGE_SIZE[::-1], antialias=True)
        mask = F.resize(mask, IMAGE_SIZE[::-1], interpolation=F.InterpolationMode.NEAREST)
        image = F.normalize(F.pil_to_tensor(image).float() / 255.0, MEAN, STD)
        mask = (F.pil_to_tensor(mask).float() > 0.5).float()
        return image, mask, image_path.name


