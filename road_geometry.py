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


def _row_center_mask(road, center_share):
    """Stable center band for a road that does not actually split in a row."""
    height, frame_width = road.shape
    result = np.zeros_like(road)
    segments_by_row = [_row_segments(road[y]) for y in range(height)]
    for y, segments in enumerate(segments_by_row):
        for left, right in segments:
            visible_width = right - left
            virtual_left, virtual_right = left, right
            if left == 0 and right != frame_width:
                virtual_left = right - _estimated_full_width(segments_by_row, y, right, "left", visible_width, frame_width)
            elif right == frame_width and left != 0:
                virtual_right = left + _estimated_full_width(segments_by_row, y, left, "right", visible_width, frame_width)
            width = virtual_right - virtual_left
            # center_share is symmetric around the road axis; side colors are
            # identical, so only this interval needs to be materialized.
            start = virtual_left + round(width * (1.0 - center_share) / 2.0)
            end = virtual_right - round(width * (1.0 - center_share) / 2.0)
            start, end = max(left, start, 0), min(right, end, frame_width)
            if start < end: result[y, start:end] = 255
    return result


def _has_persistent_split(row_segments, frame_width):
    """Reject tiny/short mask gaps; they are noise, not a road junction."""
    min_gap = max(16, round(frame_width * .04))
    min_branch_width = max(20, round(frame_width * .06))
    longest = current = 0
    for spans in row_segments:
        is_split = any(
            right_left - left_right >= min_gap
            and left_right - left_left >= min_branch_width
            and right_right - right_left >= min_branch_width
            for (left_left, left_right), (right_left, right_right) in zip(spans, spans[1:])
        )
        current = current + 1 if is_split else 0
        longest = max(longest, current)
    # A genuine fork remains visible across multiple adjacent image rows.
    return longest >= max(12, len(row_segments) // 30)


def _connect_confirmed_branches(center, road):
    """Bridge a main center band to branches only at a persistent split."""
    segments = [_row_segments(road[y]) for y in range(road.shape[0])]
    if not _has_persistent_split(segments, road.shape[1]):
        return center
    result = center.copy()
    # Roads in this camera view originate near the bottom. For each split row,
    # join a child center to the nearest one-piece parent directly below it.
    for y, spans in enumerate(segments[:-1]):
        if len(spans) < 2:
            continue
        # Only the final split row needs a connector. Connecting every row of
        # a fork would paint a large triangular green wedge.
        if len(segments[y + 1]) >= 2:
            continue
        parent_y = next((candidate for candidate in range(y + 1, min(len(segments), y + 80)) if len(segments[candidate]) == 1), None)
        if parent_y is None:
            continue
        parent_left, parent_right = segments[parent_y][0]
        parent_x = (parent_left + parent_right) // 2
        for left, right in spans:
            child_x = (left + right) // 2
            connector = np.zeros_like(road)
            thickness = max(3, round(min(right - left, parent_right - parent_left) * .12))
            cv2.line(connector, (parent_x, parent_y), (child_x, y), 255, thickness, cv2.LINE_AA)
            result = cv2.bitwise_or(result, cv2.bitwise_and(connector, road))
    return result


def unified_center_mask(road, center_share):
    """Stable scanline center bands, with bridges only at confirmed forks."""
    return _connect_confirmed_branches(_row_center_mask(road, center_share), road)


def zone_mask(road, zones: Zones):
    """Split every visible road branch independently into three colored zones.

    A connected road component may split into two or more disjoint spans on a
    scanline. Treating its outermost pixels as one span painted the empty gap
    and produced wrong zones. Each span is now handled as its own branch.
    """
    output = np.zeros((*road.shape, 3), dtype=np.uint8)
    output[road > 0] = (0, 255, 255)  # both side zones are yellow
    output[unified_center_mask(road, zones.center) > 0] = (0, 255, 0)
    return output


def smooth_road_mask(mask):
    """Remove one-pixel noise and soften jagged segmentation edges cheaply."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel, iterations=1)


def draw_zone_outlines(colored_zones, road):
    """White internal boundaries and blue external contour, clipped to road."""
    result = colored_zones.copy()
    green = cv2.inRange(colored_zones, (0, 255, 0), (0, 255, 0))
    yellow = cv2.inRange(colored_zones, (0, 255, 255), (0, 255, 255))
    internal = (cv2.dilate(green, np.ones((3, 3), np.uint8)) > 0) & (yellow > 0)
    result[internal] = (255, 255, 255)
    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 0, 0), 1, cv2.LINE_8)
    result[road == 0] = 0
    return result


# The historical functions above retain their scanline behavior for comparison.
# Stateful centerline geometry is the default selected by the video CLI.
from centerline import Branch, CenterlineGeometry, GeometryConfig, GeometryResult  # noqa: E402,F401
