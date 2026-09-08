"""Shared BGR -> RGB ROI/letterbox transform for training and every backend."""
from dataclasses import dataclass

import cv2
import numpy as np


def validate_input(input_width, input_height, roi_top, preprocess_version=1):
    if any(not isinstance(v, int) or v < 16 or (preprocess_version == 1 and v % 16) for v in (input_width, input_height)):
        raise ValueError("input_width and input_height must be positive multiples of 16")
    if not np.isfinite(roi_top) or not 0 <= roi_top < 1:
        raise ValueError("roi_top must be in [0, 1)")


@dataclass(frozen=True)
class FrameTransform:
    original_height: int
    original_width: int
    roi_y: int
    input_width: int
    input_height: int
    left: int
    top: int
    content_width: int
    content_height: int

    def unpad(self, array):
        if array.shape[:2] != (self.input_height, self.input_width):
            raise ValueError("Array shape does not match model input")
        return array[self.top:self.top + self.content_height,
                     self.left:self.left + self.content_width].copy()

    def apply_mask(self, mask):
        if mask.shape[:2] != (self.original_height, self.original_width):
            raise ValueError("Image and mask dimensions differ")
        output = np.zeros((self.input_height, self.input_width), dtype=mask.dtype)
        output[self.top:self.top + self.content_height,
               self.left:self.left + self.content_width] = cv2.resize(
            mask[self.roi_y:], (self.content_width, self.content_height),
            interpolation=cv2.INTER_NEAREST)
        return output

    def restore(self, array, interpolation=cv2.INTER_NEAREST, already_unpadded=False):
        small = array if already_unpadded else self.unpad(array)
        if small.shape[:2] != (self.content_height, self.content_width):
            raise ValueError("ROI shape does not match transform")
        shape = (self.original_height, self.original_width) + small.shape[2:]
        output = np.zeros(shape, dtype=small.dtype)
        output[self.roi_y:] = cv2.resize(
            small, (self.original_width, self.original_height - self.roi_y),
            interpolation=interpolation)
        return output


def prepare_input(image_bgr, input_width=256, input_height=144, roi_top=.20,
                  preprocess_version=1):
    validate_input(input_width, input_height, roi_top, preprocess_version)
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0:
        raise ValueError("Expected a nonempty HxWx3 BGR frame")
    if preprocess_version not in (0, 1):
        raise ValueError("Unsupported preprocess_version")
    height, width = image_bgr.shape[:2]
    roi_y = min(height - 1, int(height * roi_top))
    roi = image_bgr[roi_y:]
    if preprocess_version == 0:
        if roi_top != 0:
            raise ValueError("Legacy square preprocessing requires roi_top=0")
        scaled_w, scaled_h = input_width, input_height
    else:
        scale = min(input_width / width, input_height / len(roi))
        scaled_w = max(1, min(input_width, round(width * scale)))
        scaled_h = max(1, min(input_height, round(len(roi) * scale)))
    left, top = (input_width - scaled_w) // 2, (input_height - scaled_h) // 2
    canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    canvas[top:top + scaled_h, left:left + scaled_w] = cv2.resize(
        cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.
    transform = FrameTransform(height, width, roi_y, input_width, input_height,
                               left, top, scaled_w, scaled_h)
    return tensor, transform
