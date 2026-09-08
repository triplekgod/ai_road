"""Probability EMA, scene resets and connected hysteresis thresholding."""
from collections import deque

import cv2
import numpy as np


def hysteresis(probability, low=.40, high=.65):
    if not 0 <= low <= high <= 1:
        raise ValueError("Require 0 <= low <= high <= 1")
    possible = (probability >= low).astype(np.uint8)
    count, labels = cv2.connectedComponents(possible, connectivity=8)
    accepted = np.zeros(count, dtype=bool)
    accepted[np.unique(labels[probability >= high])] = True
    accepted[0] = False
    return accepted[labels].astype(np.uint8) * 255


class TemporalFilter:
    def __init__(self, mode='ema', alpha=.70, low=.40, high=.65, threshold=.55,
                 window=5, confirmed=4, scene_threshold=.25):
        if mode not in ('ema', 'legacy', 'none'):
            raise ValueError("Unknown temporal mode")
        if not 0 < alpha <= 1 or not 0 <= low <= high <= 1 or not 0 <= threshold <= 1:
            raise ValueError("Invalid temporal thresholds or alpha")
        if not 1 <= confirmed <= window or not 0 < scene_threshold <= 1:
            raise ValueError("Invalid voting or scene reset configuration")
        self.mode, self.alpha = mode, alpha
        self.low, self.high, self.threshold = low, high, threshold
        self.window, self.confirmed, self.scene_threshold = window, confirmed, scene_threshold
        self.reset()

    def reset(self):
        self.previous = None
        self.scene = None
        self.shape = None
        self.history = deque(maxlen=self.window)
        self.was_reset = True

    def update(self, probability, frame):
        if probability.ndim != 2 or not np.all(np.isfinite(probability)):
            raise ValueError("Expected finite HxW probabilities")
        thumbnail = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 18)).astype(np.float32) / 255.
        key = (frame.shape, probability.shape)
        changed = self.shape != key or (self.scene is not None and
                  float(np.mean(np.abs(thumbnail - self.scene))) > self.scene_threshold)
        if changed:
            self.reset()
        self.was_reset = changed
        self.shape, self.scene = key, thumbnail
        if self.mode == 'legacy':
            current = probability >= self.threshold
            self.history.append(current)
            return (current & (np.sum(self.history, axis=0) >= self.confirmed)).astype(np.uint8) * 255
        smoothed = probability if self.previous is None or self.mode == 'none' else (
            self.alpha * probability + (1 - self.alpha) * self.previous)
        self.previous = smoothed.copy()
        return hysteresis(smoothed, self.low, self.high)
