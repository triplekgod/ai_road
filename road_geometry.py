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


def _row_segments(row):
    """Return separate filled road spans in a row; gaps are branch separators."""
    indices = np.flatnonzero(row)
    if not len(indices): return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, breaks)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _estimated_full_width(segments_by_row, y, anchor, edge, visible_width, frame_width):
    """Estimate a clipped branch width from the closest fully visible branch."""
    candidates = []
    search_radius = min(100, len(segments_by_row) - 1)
    for candidate_y in range(max(0, y - search_radius), min(len(segments_by_row), y + search_radius + 1)):
        for left, right in segments_by_row[candidate_y]:
            if left == 0 or right == frame_width:  # this sample is clipped too
                continue
            reference_edge = right if edge == "left" else left
            # Prefer a nearby row whose visible edge lies on the same branch.
            score = 3 * abs(candidate_y - y) + abs(reference_edge - anchor)
            candidates.append((score, right - left))
    if candidates:
        width = min(candidates, key=lambda item: item[0])[1]
        return max(visible_width, width)
    # No full sample exists: extrapolate conservatively instead of treating
    # the image edge as a real road boundary.
    return max(visible_width, int(visible_width * 1.5), 32)


def zone_mask(road, zones: Zones):
    """Split every visible road branch independently into three colored zones.

    A connected road component may split into two or more disjoint spans on a
    scanline. Treating its outermost pixels as one span painted the empty gap
    and produced wrong zones. Each span is now handled as its own branch.
    """
    height, frame_width = road.shape
    output = np.zeros((height, frame_width, 3), dtype=np.uint8)
    segments_by_row = [_row_segments(road[y]) for y in range(height)]
    for y, segments in enumerate(segments_by_row):
        for left, right in segments:
            visible_width = right - left
            clipped_left, clipped_right = left == 0, right == frame_width
            virtual_left, virtual_right = left, right
            if clipped_left and not clipped_right:
                width = _estimated_full_width(segments_by_row, y, right, "left", visible_width, frame_width)
                virtual_left = right - width
            elif clipped_right and not clipped_left:
                width = _estimated_full_width(segments_by_row, y, left, "right", visible_width, frame_width)
                virtual_right = left + width
            # If both sides are clipped, the road is wider than the frame and
            # its true center is unknowable from one image; retain the visible
            # center rather than invent an asymmetric shift.
            width = virtual_right - virtual_left
            a = virtual_left + round(width * zones.left)
            b = virtual_right - round(width * zones.right)
            # Paint only the part inside the actual image, but calculate all
            # boundaries in the extrapolated (virtual) road interval.
            for start, end, color in ((virtual_left, a, (0, 255, 255)),
                                      (a, b, (0, 255, 0)),
                                      (b, virtual_right, (0, 255, 255))):
                start = max(0, left, start)
                end = min(frame_width, right, end)
                if start < end:
                    output[y, start:end] = color
    return output


def smooth_road_mask(mask):
    """Remove one-pixel noise and soften jagged segmentation edges cheaply."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=1)


def draw_zone_outlines(colored_zones, road):
    """Draw a dark outer road contour and white borders between visible zones."""
    result = colored_zones.copy()
    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 80, 0), 2, cv2.LINE_AA)
    for color in ((0, 255, 255), (0, 255, 0)):
        pixels = cv2.inRange(colored_zones, np.array(color), np.array(color))
        contours, _ = cv2.findContours(pixels, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return result
