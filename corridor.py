from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CorridorState:
    probability: np.ndarray | None = None


def _remove_tiny_components(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(count, bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(12, int(mask.size * 0.0008))
    return keep[labels]


def _morphological_skeleton(mask):
    """Fallback when opencv-contrib ximgproc is unavailable."""
    image = mask.astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image):
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
    return skeleton > 0


def _skeletonize(mask):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        thin = cv2.ximgproc.thinning(
            mask.astype(np.uint8) * 255,
            thinningType=cv2.ximgproc.THINNING_GUOHALL,
        )
        return thin > 0
    return _morphological_skeleton(mask)


def _skeleton_orientation(skeleton):
    """Return a consistently forward-oriented tangent for every skeleton pixel."""
    h, w = skeleton.shape
    line = cv2.GaussianBlur(skeleton.astype(np.float32), (0, 0), 1.8)
    gx = cv2.Sobel(line, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(line, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.5)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.5)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.5)

    normal_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-8)
    tangent_x = -np.sin(normal_angle)
    tangent_y = np.cos(normal_angle)

    yy, xx = np.indices((h, w), dtype=np.float32)
    away_x = xx - 0.55 * w
    away_y = yy - (h - 1)
    reverse = tangent_x * away_x + tangent_y * away_y < 0
    tangent_x[reverse] *= -1
    tangent_y[reverse] *= -1
    return tangent_x, tangent_y


def _nearest_skeleton(skeleton):
    """For every pixel return coordinates of its nearest skeleton pixel."""
    source = np.ones(skeleton.shape, np.uint8)
    source[skeleton] = 0
    _, labels = cv2.distanceTransformWithLabels(
        source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    sy, sx = np.nonzero(skeleton)
    max_label = int(labels.max())
    lookup_x = np.zeros(max_label + 1, np.int32)
    lookup_y = np.zeros(max_label + 1, np.int32)
    skeleton_labels = labels[sy, sx]
    lookup_x[skeleton_labels] = sx
    lookup_y[skeleton_labels] = sy
    return lookup_x[labels], lookup_y[labels]


def build_zones(probability, state=None, threshold=0.5):
    """Return 0=background/obstacle, 1=left, 2=center, 3=right for every branch."""
    if state is None:
        state = CorridorState()
    raw_probability = np.asarray(probability, np.float32)
    if (raw_probability >= threshold).mean() < 0.002:
        state.probability = raw_probability.copy()
        zones = np.zeros(raw_probability.shape, np.uint8)
        return zones, state, {"heading_deg": float("nan"), "confidence": float(1.0 - raw_probability.mean())}

    probability = raw_probability
    if state.probability is not None and state.probability.shape == probability.shape:
        probability = 0.72 * probability + 0.28 * state.probability
    state.probability = raw_probability.copy()

    mask = probability >= threshold
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    mask = _remove_tiny_components(mask)
    skeleton = _skeletonize(mask)
    zones = np.zeros(mask.shape, np.uint8)
    if not skeleton.any():
        return zones, state, {"heading_deg": float("nan"), "confidence": 0.0}

    nearest_x, nearest_y = _nearest_skeleton(skeleton)
    tangent_x, tangent_y = _skeleton_orientation(skeleton)
    local_tx = tangent_x[nearest_y, nearest_x]
    local_ty = tangent_y[nearest_y, nearest_x]

    yy, xx = np.indices(mask.shape, dtype=np.float32)
    dx = xx - nearest_x
    dy = yy - nearest_y
    signed_lateral = local_tx * dy - local_ty * dx
    road_distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    local_radius = np.maximum(road_distance[nearest_y, nearest_x], 2.0)
    normalized_lateral = signed_lateral / local_radius

    zones[mask & (normalized_lateral < -1.0 / 3.0)] = 1
    zones[mask & (np.abs(normalized_lateral) <= 1.0 / 3.0)] = 2
    zones[mask & (normalized_lateral > 1.0 / 3.0)] = 3

    sy, sx = np.nonzero(skeleton)
    lower = sy > mask.shape[0] * 0.55
    if lower.any():
        candidates = np.flatnonzero(lower)
        distance_to_vehicle = (sx[lower] - mask.shape[1] * 0.55) ** 2 + (sy[lower] - mask.shape[0]) ** 2
        point = candidates[int(distance_to_vehicle.argmin())]
    else:
        point = int(np.argmax(sy))
    heading = np.degrees(np.arctan2(tangent_x[sy[point], sx[point]], -tangent_y[sy[point], sx[point]]))
    confidence = float(probability[mask].mean()) if mask.any() else 0.0
    return zones, state, {"heading_deg": float(heading), "confidence": confidence}


def overlay_zones(frame, zones, alpha=0.45):
    colors = np.array([[0, 0, 0], [255, 80, 40], [40, 220, 40], [40, 80, 255]], np.uint8)
    layer = colors[zones]
    output = frame.copy()
    road = zones > 0
    output[road] = cv2.addWeighted(frame, 1 - alpha, layer, alpha, 0)[road]
    return output
