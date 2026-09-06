from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Zones:
    left: float = .15
    center: float = .70
    right: float = .15
    def __post_init__(self):
        if min(self.left, self.center, self.right) < 0 or abs(self.left + self.center + self.right - 1) > 1e-6:
            raise ValueError("Zone shares must be non-negative and sum to 1.0")


def primary_road(binary_mask, bottom_fraction=.42, min_area=500):
    """Keep one component connected to the lower part of the image; reject distant blobs."""
    h, w = binary_mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean)
    candidates = []
    lower_start = int(h * (1 - bottom_fraction))
    for label in range(1, count):
        x, y, cw, ch, area = stats[label]
        touches_lower = y + ch >= lower_start
        if area >= min_area and touches_lower:
            # Prefer large, low components rather than unrelated regions in sky/walls.
            candidates.append((area + 3 * (y + ch), label))
    if not candidates: return np.zeros_like(binary_mask)
    label = max(candidates)[1]
    return np.where(labels == label, 255, 0).astype(np.uint8)


def zone_mask(road, zones: Zones):
    """Split each road row directly; no costly row-segment grouping/polygon reconstruction."""
    output = np.zeros((*road.shape, 3), dtype=np.uint8)
    for y in range(road.shape[0]):
        xs = np.flatnonzero(road[y])
        if not len(xs): continue
        left, right = xs[0], xs[-1] + 1
        width = right - left
        a, b = left + round(width * zones.left), right - round(width * zones.right)
        output[y, left:a] = (0, 255, 255)  # yellow BGR
        output[y, a:b] = (0, 255, 0)       # green BGR
        output[y, b:right] = (0, 255, 255)
    return output
