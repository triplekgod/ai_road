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


def _virtual_road(road):
    """Extend one-sided clipped spans outside the image before geometry work."""
    height, frame_width = road.shape
    pad = frame_width
    virtual = np.zeros((height, frame_width + 2 * pad), dtype=np.uint8)
    segments_by_row = [_row_segments(road[y]) for y in range(height)]
    for y, segments in enumerate(segments_by_row):
        for left, right in segments:
            visible_width = right - left
            virtual_left, virtual_right = left, right
            if left == 0 and right != frame_width:
                virtual_left = right - _estimated_full_width(segments_by_row, y, right, "left", visible_width, frame_width)
            elif right == frame_width and left != 0:
                virtual_right = left + _estimated_full_width(segments_by_row, y, left, "right", visible_width, frame_width)
            start, end = max(0, virtual_left + pad), min(virtual.shape[1], virtual_right + pad)
            if start < end: virtual[y, start:end] = 255
    return virtual, pad


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


def _skeletonize(mask):
    """OpenCV-only morphological skeleton; operates on a reduced mask."""
    image = mask.copy()
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        eroded = cv2.erode(image, element)
        edge = cv2.subtract(image, cv2.dilate(eroded, element))
        skeleton = cv2.bitwise_or(skeleton, edge)
        image = eroded
    return skeleton


def unified_center_mask(road, center_share):
    """Create one center-zone tree that follows the road and all its branches."""
    row_segments = [_row_segments(road[y]) for y in range(road.shape[0])]
    # A wide bend still has one continuous road span per row. Morphological
    # skeletons create harmless-looking but wrong side spurs on such shapes;
    # use the stable scanline center unless a real split is observed.
    if max((len(spans) for spans in row_segments), default=0) <= 1:
        return _row_center_mask(road, center_share)
    virtual, pad = _virtual_road(road)
    # Geometry at a bounded resolution keeps this step suitable for live CPU use.
    scale = min(1.0, 960.0 / virtual.shape[1])
    work_size = (max(1, round(virtual.shape[1] * scale)), max(1, round(virtual.shape[0] * scale)))
    work = cv2.resize(virtual, work_size, interpolation=cv2.INTER_NEAREST)
    skeleton = _skeletonize(work)
    if not cv2.countNonZero(skeleton): return np.zeros_like(road)
    edge_distance = cv2.distanceTransform(work, cv2.DIST_L2, 3)
    seed = np.full_like(work, 255); seed[skeleton > 0] = 0
    center_distance = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
    # For a strip: radius = distance-to-edge + distance-to-skeleton. This
    # selects exactly `center_share` of the local width and follows a fork.
    center = ((1.0 - center_share) * center_distance <= center_share * edge_distance)
    center &= work > 0
    crop_start = round(pad * work.shape[1] / virtual.shape[1])
    crop_end = round((pad + road.shape[1]) * work.shape[1] / virtual.shape[1])
    cropped = center[:, crop_start:crop_end].astype(np.uint8) * 255
    result = cv2.resize(cropped, (road.shape[1], road.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(result, road)


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
    """Draw a dark outer road contour and white borders between visible zones."""
    result = colored_zones.copy()
    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 80, 0), 2, cv2.LINE_AA)
    for color in ((0, 255, 255), (0, 255, 0)):
        pixels = cv2.inRange(colored_zones, np.array(color), np.array(color))
        contours, _ = cv2.findContours(pixels, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return result
