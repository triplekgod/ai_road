"""Paired segmentation data with shared geometric and image-only augmentations."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data_tools import discover_pairs, inspect_pair, read_image
from preprocess import prepare_input


def augment_pair(image, mask):
    """Keep masks binary; never apply vertical flips to road scenes."""
    image, mask = image.copy(), mask.copy()
    height, width = mask.shape
    if np.random.random() < .5:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    if np.random.random() < .45:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), np.random.uniform(-8, 8), np.random.uniform(.9, 1.1))
        matrix[:, 2] += [np.random.uniform(-.04, .04) * width, np.random.uniform(-.04, .04) * height]
        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    if np.random.random() < .2:
        source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        destination = source + np.random.uniform(-.05, .05, (4, 2)).astype(np.float32) * [width, height]
        matrix = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
        image = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpPerspective(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    if np.random.random() < .2 and min(height, width) > 4:
        fraction = np.random.uniform(.8, 1.)
        crop_h, crop_w = max(1, int(height * fraction)), max(1, int(width * fraction))
        top = np.random.randint(0, height - crop_h + 1)
        left = np.random.randint(0, width - crop_w + 1)
        image = cv2.resize(image[top:top + crop_h, left:left + crop_w], (width, height))
        mask = cv2.resize(mask[top:top + crop_h, left:left + crop_w], (width, height), interpolation=cv2.INTER_NEAREST)
    working = image.astype(np.float32) / 255.
    if np.random.random() < .7:
        working = np.clip(working * np.random.uniform(.75, 1.25) + np.random.uniform(-.12, .12), 0, 1)
        working = np.power(working, np.random.uniform(.7, 1.4))
    if np.random.random() < .35:
        working *= np.random.uniform(.85, 1.15, (1, 1, 3))
    if np.random.random() < .2:
        working += np.random.normal(0, np.random.uniform(.005, .035), working.shape).astype(np.float32)
    if np.random.random() < .18:
        fog = np.random.uniform(.05, .35) * np.linspace(1., .35, height, dtype=np.float32)[:, None, None]
        tint = np.array([.75, .80, .86], dtype=np.float32) if np.random.random() < .5 else np.ones(3, np.float32)
        working = working * (1 - fog) + tint * fog
    if np.random.random() < .25:
        shadow = np.zeros((height, width), np.float32)
        polygon = np.array([[0, np.random.randint(height)], [width - 1, np.random.randint(height)],
                            [width - 1, height - 1], [0, height - 1]], np.int32)
        cv2.fillPoly(shadow, [polygon], 1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(1, width * .025))
        working *= 1 - shadow[..., None] * np.random.uniform(.2, .55)
    if np.random.random() < .15:
        yy, xx = np.ogrid[:height, :width]
        spot = np.exp(-((xx - np.random.uniform(0, width)) ** 2 + (yy - np.random.uniform(0, height)) ** 2)
                      / (2 * (max(width, height) * np.random.uniform(.1, .3)) ** 2))
        working += spot[..., None] * np.random.uniform(.15, .6)
    image = (np.clip(working, 0, 1) * 255).astype(np.uint8)
    if np.random.random() < .15:
        image = cv2.GaussianBlur(image, (3, 3), np.random.uniform(.3, 1.2))
    if np.random.random() < .12:
        kernel = np.zeros((5, 5), np.float32)
        if np.random.random() < .5:
            kernel[2, :] = .2
        else:
            np.fill_diagonal(kernel, .2)
        image = cv2.filter2D(image, -1, kernel)
    if np.random.random() < .2:
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(np.random.randint(35, 91))])
        if success:
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(image), np.ascontiguousarray((mask > 0).astype(np.uint8))


class RoadDataset(Dataset):
    def __init__(self, images_dir=None, masks_dir=None, image_size=None, augment=False, *,
                 samples=None, input_width=256, input_height=144, roi_top=.20, validate=True):
        if image_size is not None:
            input_width = input_height = image_size
        if input_width <= 0 or input_height <= 0 or input_width % 16 or input_height % 16:
            raise ValueError("Input dimensions must be positive multiples of 16")
        if not 0 <= roi_top < 1:
            raise ValueError("roi_top must be in [0, 1)")
        if samples is None:
            if images_dir is None or masks_dir is None:
                raise ValueError("Provide samples or both image and mask directories")
            samples = discover_pairs(images_dir, masks_dir, "unspecified_source")
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("No image/mask pairs found")
        if validate:
            for sample in self.samples:
                inspect_pair(sample["image"], sample["mask"])
        self.items = [Path(sample["image"]) for sample in self.samples]
        self.input_width, self.input_height = int(input_width), int(input_height)
        self.roi_top, self.augment = float(roi_top), bool(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = read_image(sample["image"])
        mask = (read_image(sample["mask"], cv2.IMREAD_UNCHANGED) > 0).astype(np.uint8)
        if self.augment:
            image, mask = augment_pair(image, mask)
        tensor, transform = prepare_input(image, input_width=self.input_width,
                                          input_height=self.input_height, roi_top=self.roi_top)
        target = transform.apply_mask(mask)
        return torch.from_numpy(np.ascontiguousarray(tensor[0])), torch.from_numpy((target > 0).astype(np.float32)[None])
