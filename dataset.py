from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RoadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=192, augment=False):
        self.images_dir, self.masks_dir = Path(images_dir), Path(masks_dir)
        self.image_size, self.augment = image_size, augment
        self.items = [p for p in sorted(self.images_dir.glob("*.bmp"))
                      if (self.masks_dir / f"{p.stem}_mask.bmp").exists()]
        if not self.items:
            raise ValueError("No BMP image/mask pairs found")

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        path = self.items[index]
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_dir / f"{path.stem}_mask.bmp"), cv2.IMREAD_GRAYSCALE)
        if self.augment and np.random.random() < .5:
            image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return (torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.,
                torch.from_numpy((mask > 127).copy()).unsqueeze(0).float())
