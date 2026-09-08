"""Pixel-micro segmentation metrics; empty-vs-empty overlap scores equal one."""
import cv2
import numpy as np


def boundary_mask(mask, boundary_ratio=.02):
    """Inner boundary band of width round(ratio * image diagonal), minimum 1 px."""
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2 or not np.isfinite(boundary_ratio) or boundary_ratio <= 0:
        raise ValueError("Expected a 2D mask and positive boundary_ratio")
    width = max(1, int(round(boundary_ratio * np.hypot(*mask.shape))))
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=width,
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (mask > 0) & (eroded == 0)


class SegmentationMetrics:
    def __init__(self, boundary_ratio=.02):
        self.boundary_ratio = float(boundary_ratio)
        self.tp = self.fp = self.fn = self.tn = 0
        self.boundary_intersection = self.boundary_union = 0
        self.frames = self.negative_frames = self.negative_pixels = 0
        self.negative_false_pixels = self.negative_false_frames = 0

    def update(self, prediction, target):
        prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
        if prediction.shape != target.shape or prediction.ndim not in (2, 3, 4):
            raise ValueError("Prediction and target need equal HW, NHW or N1HW shapes")
        if prediction.ndim == 4:
            if prediction.shape[1] != 1:
                raise ValueError("Only one road probability channel is supported")
            prediction, target = prediction[:, 0], target[:, 0]
        if prediction.ndim == 2:
            prediction, target = prediction[None], target[None]
        self.tp += int(np.count_nonzero(prediction & target))
        self.fp += int(np.count_nonzero(prediction & ~target))
        self.fn += int(np.count_nonzero(~prediction & target))
        self.tn += int(np.count_nonzero(~prediction & ~target))
        for pred, truth in zip(prediction, target):
            pred_boundary, truth_boundary = boundary_mask(pred, self.boundary_ratio), boundary_mask(truth, self.boundary_ratio)
            self.boundary_intersection += int(np.count_nonzero(pred_boundary & truth_boundary))
            self.boundary_union += int(np.count_nonzero(pred_boundary | truth_boundary))
            self.frames += 1
            if not truth.any():
                false_pixels = int(np.count_nonzero(pred))
                self.negative_frames += 1
                self.negative_pixels += truth.size
                self.negative_false_pixels += false_pixels
                self.negative_false_frames += int(false_pixels > 0)

    def compute(self):
        ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else 1.
        return {"iou": ratio(self.tp, self.tp + self.fp + self.fn),
                "dice_f1": ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
                "precision": ratio(self.tp, self.tp + self.fp),
                "recall": ratio(self.tp, self.tp + self.fn),
                "boundary_iou": ratio(self.boundary_intersection, self.boundary_union),
                "no_road_pixel_fpr": (self.negative_false_pixels / self.negative_pixels if self.negative_pixels else None),
                "no_road_frame_fpr": (self.negative_false_frames / self.negative_frames if self.negative_frames else None),
                "frames": self.frames, "negative_frames": self.negative_frames,
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "boundary_ratio": self.boundary_ratio}
