from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CorridorState:
    path: np.ndarray | None = None
    probability: np.ndarray | None = None


def _largest_relevant_component(mask, origin_x):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask
    h, w = mask.shape
    score = np.full(count, -1e9, np.float32)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        bottom = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        score[i] = area + 2.0 * bottom - 0.3 * abs(centers[i, 0] - origin_x)
    return labels == int(score.argmax())


def _remove_tiny_components(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(count, bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(12, int(mask.size * 0.0008))
    return keep[labels]


def _smooth(values, window=17):
    if len(values) < 3:
        return values
    window = min(window, len(values) // 2 * 2 + 1)
    kernel = np.ones(window, np.float32) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _main_path(mask, probability, previous=None):
    """Dynamic-programming path: wide, confident, forward and temporally continuous."""
    h, w = mask.shape
    road_rows = np.where(mask.sum(1) > max(3, w * 0.01))[0]
    if not len(road_rows):
        return None
    y_bottom, y_top = int(road_rows[-1]), int(road_rows[0])
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    distance /= max(float(distance.max()), 1.0)
    inf = np.inf
    costs = np.full((h, w), inf, np.float32)
    parent = np.full((h, w), -1, np.int16)
    xs = np.arange(w)
    start = 0.55 * w
    valid = mask[y_bottom]
    costs[y_bottom, valid] = 0.8 * np.abs(xs[valid] - start) / w - 2.0 * distance[y_bottom, valid] - probability[y_bottom, valid]
    step = max(4, w // 28)

    offsets = np.arange(-step, step + 1, dtype=np.int16)
    for y in range(y_bottom - 1, y_top - 1, -1):
        prev_cost = costs[y + 1]
        choices = np.full((len(offsets), w), inf, np.float32)
        for k, dx in enumerate(offsets):
            if dx < 0:
                choices[k, -dx:] = prev_cost[:w + dx]
            elif dx > 0:
                choices[k, :w - dx] = prev_cost[dx:]
            else:
                choices[k] = prev_cost
            choices[k] += 0.035 * abs(int(dx))
        best_k = choices.argmin(0)
        best = choices[best_k, xs]
        temporal = np.zeros(w, np.float32)
        if previous is not None and y < len(previous) and np.isfinite(previous[y]):
            temporal = 0.6 * np.abs(xs - previous[y]) / w
        row_cost = best - 2.2 * distance[y] - probability[y] + temporal
        row_cost[~mask[y] | ~np.isfinite(best)] = inf
        costs[y] = row_cost
        parent[y] = np.clip(xs + offsets[best_k], 0, w - 1)

    feasible_rows = np.where(np.isfinite(costs).any(1))[0]
    if not len(feasible_rows):
        return None
    end_y = int(feasible_rows[0])
    x = int(np.argmin(costs[end_y]))
    path = np.full(h, np.nan, np.float32)
    for y in range(end_y, y_bottom + 1):
        path[y] = x
        if y < y_bottom:
            x = int(parent[y, x])
            if x < 0:
                break
    known = np.flatnonzero(np.isfinite(path))
    if len(known) < 2:
        return None
    path = np.interp(np.arange(h), known, path[known]).astype(np.float32)
    return _smooth(path, 19).astype(np.float32)


def build_zones(probability, state=None, threshold=0.5):
    """Return labels: 0 background/obstacle, 1 left, 2 center, 3 right."""
    if state is None:
        state = CorridorState()
    probability = np.asarray(probability, np.float32)
    if (probability >= threshold).mean() < 0.002:
        state.probability = probability.copy()
        state.path = None
        zones = np.zeros(probability.shape, np.uint8)
        return zones, state, {"heading_deg": float("nan"), "confidence": float(1.0 - probability.mean())}
    if state.probability is not None and state.probability.shape == probability.shape:
        probability = 0.65 * probability + 0.35 * state.probability
    state.probability = probability.copy()

    mask = probability >= threshold
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    mask = _remove_tiny_components(mask)
    path_mask = _largest_relevant_component(mask, probability.shape[1] * 0.55)
    path = _main_path(path_mask, probability, state.path)
    zones = np.zeros(mask.shape, np.uint8)
    if path is None:
        state.path = None
        return zones, state, {"heading_deg": float("nan"), "confidence": 0.0}
    if state.path is not None and state.path.shape == path.shape:
        path = 0.72 * path + 0.28 * state.path
    state.path = path

    h, w = mask.shape
    span = np.full(h, np.nan, np.float32)
    for y in range(h):
        road_x = np.flatnonzero(path_mask[y])
        if len(road_x):
            span[y] = road_x[-1] - road_x[0] + 1
    known = np.flatnonzero(np.isfinite(span))
    if len(known):
        span = np.interp(np.arange(h), known, span[known]).astype(np.float32)
    else:
        span[:] = w * 0.18
    # One third of the transverse corridor is the center zone. A long smoothing
    # window prevents a side branch or an obstacle from widening/narrowing it abruptly.
    center_half_width = np.maximum(_smooth(span, 31) / 6.0, w * 0.02)

    yy, xx = np.indices(mask.shape)
    offset = xx - path[:, None]
    boundary = center_half_width[:, None]
    zones[mask & (offset < -boundary)] = 1
    zones[mask & (np.abs(offset) <= boundary)] = 2
    zones[mask & (offset > boundary)] = 3

    y1, y2 = int(h * 0.35), int(h * 0.75)
    heading = np.degrees(np.arctan2(path[y1] - path[y2], max(y2 - y1, 1)))
    confidence = float(probability[mask].mean()) if mask.any() else 0.0
    return zones, state, {"heading_deg": float(heading), "confidence": confidence}


def overlay_zones(frame, zones, alpha=0.45):
    colors = np.array([[0, 0, 0], [255, 80, 40], [40, 220, 40], [40, 80, 255]], np.uint8)
    layer = colors[zones]
    output = frame.copy()
    road = zones > 0
    output[road] = cv2.addWeighted(frame, 1 - alpha, layer, alpha, 0)[road]
    return output
