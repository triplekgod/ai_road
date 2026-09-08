"""Small-mask centerline geometry and temporally confirmed outgoing road routes.

Coordinates are always (x, y) in the unpadded ROI mask supplied to ``process``.
A branch is a route from the bottom origin to an outgoing endpoint: a straight
road has one route, a Y or T has two, and a cross has three. Shared trunk pixels
belong to every route. This is deliberately separate from choosing a route for
navigation; ``corridor_mode='all'`` renders all confirmed outgoing routes.
"""

from dataclasses import dataclass, field
import heapq
import math

import cv2
import numpy as np


@dataclass
class GeometryConfig:
    bottom_fraction: float = .20
    min_component_area: float = .003
    close_gap: float = .025
    max_hole_area: float = .004
    min_branch_length: float = .12
    min_branch_width: float = .05
    min_branch_angle: float = 25.
    min_branch_area: float = .002
    confirm_frames: int = 3
    missing_frames: int = 2
    match_distance: float = .18
    corridor_mode: str = "all"

    def __post_init__(self):
        for name in ("bottom_fraction", "min_component_area", "close_gap",
                     "max_hole_area", "min_branch_length", "min_branch_width",
                     "min_branch_area", "match_distance"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite fraction between 0 and 1")
        if self.bottom_fraction == 0 or self.match_distance == 0:
            raise ValueError("bottom_fraction and match_distance must be positive")
        if not math.isfinite(self.min_branch_angle) or not 0 <= self.min_branch_angle < 180:
            raise ValueError("min_branch_angle must be between 0 and 180 degrees")
        if self.confirm_frames < 1 or self.missing_frames < 0:
            raise ValueError("confirm_frames must be positive; missing_frames cannot be negative")
        if self.corridor_mode not in ("all", "main"):
            raise ValueError("corridor_mode must be 'all' or 'main'")


@dataclass
class Branch:
    id: int
    points: np.ndarray
    split_point: tuple
    direction: tuple
    length: float
    width: float
    area: float
    state: str = "candidate"
    confirmation_count: int = 1
    missed_frames: int = 0
    _confirmed: bool = field(default=False, repr=False)

    @property
    def endpoint(self):
        return tuple(float(v) for v in self.points[-1])

    def as_dict(self):
        """JSON-compatible tracking metadata, without duplicating every pixel."""
        return {"id": self.id, "state": self.state,
                "split_point": list(self.split_point), "direction": list(self.direction),
                "endpoint": list(self.endpoint), "length": self.length,
                "width": self.width, "area": self.area,
                "confirmation_count": self.confirmation_count,
                "missed_frames": self.missed_frames}


@dataclass
class GeometryResult:
    road: np.ndarray
    colors: np.ndarray
    center: np.ndarray
    skeleton: np.ndarray
    branches: list
    main_branch_id: int | None


def _clean_road(mask, config):
    height, width = mask.shape
    road = np.where(mask > 0, 255, 0).astype(np.uint8)
    diameter = max(1, round(min(height, width) * config.close_gap))
    diameter += 1 - diameter % 2
    if diameter > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)
    # Fill only small enclosed holes. Large occluders remain background; their
    # skeleton cycles are not interpreted as additional outgoing branches.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - road, connectivity=8)
    border = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    for label in range(1, count):
        if label not in border and stats[label, cv2.CC_STAT_AREA] <= height * width * config.max_hole_area:
            road[labels == label] = 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
    lower = max(0, int(height * (1 - config.bottom_fraction)))
    candidates = []
    for label in np.unique(labels[lower:]):
        if label == 0 or stats[label, cv2.CC_STAT_AREA] < max(4, height * width * config.min_component_area):
            continue
        bottom_y = stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] - 1
        lower_x = np.flatnonzero(np.any(labels[lower:] == label, axis=0))
        anchor_distance = abs(float(np.mean(lower_x)) - (width - 1) / 2) / max(1, width)
        score = (4 * bottom_y / max(1, height) - anchor_distance
                 + math.log1p(float(stats[label, cv2.CC_STAT_AREA])) / 10)
        candidates.append((score, int(label)))
    if not candidates:
        return np.zeros_like(road)
    return np.where(labels == max(candidates)[1], 255, 0).astype(np.uint8)


def _thinning_tables():
    patterns = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.uint8)
    count = patterns.sum(axis=1)
    transitions = ((patterns == 0) & (np.roll(patterns, -1, axis=1) == 1)).sum(axis=1)
    p2, _, p4, _, p6, _, p8, _ = patterns.T
    basic = (count >= 2) & (count <= 6) & (transitions == 1)
    first = basic & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
    second = basic & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
    return first.astype(np.uint8), second.astype(np.uint8)


_THIN_TABLES = _thinning_tables()
_THIN_KERNEL = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], np.float32)


def thin_mask(mask):
    """Zhang-Suen thinning via native 3x3 filters and 256-entry lookup tables.

    This is the usual vectorized thinning rule, with the eight neighbor tests
    precomputed. OpenCV's standard build suffices; contrib/scipy are optional.
    """
    image = (mask > 0).astype(np.uint8)
    if not np.any(image):
        return np.zeros_like(mask)
    for _ in range(max(mask.shape)):
        changed = False
        for table in _THIN_TABLES:
            encoded = cv2.filter2D(image, -1, _THIN_KERNEL, borderType=cv2.BORDER_CONSTANT)
            delete = cv2.bitwise_and(image, cv2.LUT(encoded, table))
            if cv2.countNonZero(delete):
                image -= delete
                changed = True
        if not changed:
            break
    return image * 255


def _pixel_graph(skeleton):
    """Build an eight-neighbor graph, suppressing redundant corner diagonals."""
    ys, xs = np.nonzero(skeleton)
    points = np.column_stack((xs, ys)).astype(np.int32)
    index = np.full(skeleton.shape, -1, np.int32)
    index[ys, xs] = np.arange(len(points))
    adjacency = [[] for _ in points]
    height, width = skeleton.shape
    for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
        tx, ty = xs + dx, ys + dy
        valid = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
        source = np.flatnonzero(valid)
        target = index[ty[valid], tx[valid]]
        keep = target >= 0
        if dx and dy:
            keep &= (index[ys[source], tx[source]] < 0) & (index[ty[source], xs[source]] < 0)
        for first, second in zip(source[keep], target[keep]):
            adjacency[int(first)].append(int(second))
            adjacency[int(second)].append(int(first))
    return points, adjacency


def _prune_spurs(skeleton, distance, min_length):
    for _ in range(6):
        points, graph = _pixel_graph(skeleton)
        remove = set()
        for start, neighbors in enumerate(graph):
            if len(neighbors) != 1:
                continue
            x, y = points[start]
            endpoint_radius = float(distance[y, x])
            edge_distance = min(x, y, skeleton.shape[1] - 1 - x, skeleton.shape[0] - 1 - y)
            if edge_distance <= endpoint_radius * 1.6 + 2:
                # A short segment ending at the camera boundary is an open
                # road continuation, especially below/above a large occluder.
                # Pruning it would detach the route from its bottom origin.
                continue
            path, previous, current = [start], start, neighbors[0]
            length = 0.
            while True:
                length += math.hypot(*(points[current] - points[previous]))
                if len(graph[current]) != 2:
                    break
                path.append(current)
                previous, current = current, next(p for p in graph[current] if p != previous)
            if len(graph[current]) >= 3:
                radius = float(distance[points[current, 1], points[current, 0]])
                if length < max(min_length, 1.6 * radius):
                    remove.update(path)
        if not remove:
            break
        dead = points[list(remove)]
        skeleton[dead[:, 1], dead[:, 0]] = 0
    return skeleton


def _length(points):
    return float(np.linalg.norm(np.diff(points.astype(np.float32), axis=0), axis=1).sum())


def _unit(vector):
    length = float(np.linalg.norm(vector))
    return vector / length if length else np.array([0., -1.])


def _extend_to_border(points, road, distance):
    """Recover clipped endpoint portions lost to finite-image thinning."""
    result = points.copy()
    height, width = road.shape
    for reverse in (True, False):
        ordered = result[::-1] if reverse else result
        if len(ordered) < 2:
            continue
        end = ordered[-1]
        radius = float(distance[end[1], end[0]])
        if min(end[0], end[1], width - 1 - end[0], height - 1 - end[1]) > radius * 1.6 + 2:
            continue
        tangent = _unit(end.astype(float) - ordered[max(0, len(ordered) - max(4, round(radius)))] )
        extension = []
        reaches_border = False
        for step in range(1, max(4, round(radius * 2.5 + 4))):
            point = np.rint(end + tangent * step).astype(np.int32)
            x, y = point
            if not (0 <= x < width and 0 <= y < height) or road[y, x] == 0:
                break
            if not extension or np.any(point != extension[-1]):
                extension.append(point)
            if x in (0, width - 1) or y in (0, height - 1):
                reaches_border = True
                break
        if reaches_border:
            ordered = np.concatenate((ordered, np.asarray(extension)), axis=0)
            result = ordered[::-1] if reverse else ordered
    return result


def _outgoing_routes(skeleton, road, distance, config):
    points, graph = _pixel_graph(skeleton)
    if len(points) < 2:
        return []
    height, width = road.shape
    ends = [index for index, neighbors in enumerate(graph) if len(neighbors) == 1]
    if not ends:
        # A closed skeleton has no outgoing fork. Use the highest reachable
        # point as one destination rather than reporting the hole as a fork.
        ends = [int(np.argmin(points[:, 1])), int(np.argmax(points[:, 1]))]
    lower = [index for index in ends if points[index, 1] >= height * (1 - config.bottom_fraction)]
    root = max(lower or ends, key=lambda index: points[index, 1] - .18 * abs(points[index, 0] - width / 2))
    costs = np.full(len(points), np.inf)
    parents = np.full(len(points), -1, np.int32)
    costs[root] = 0
    queue = [(0., root)]
    while queue:
        cost, current = heapq.heappop(queue)
        if cost > costs[current]:
            continue
        for neighbor in graph[current]:
            step = math.hypot(*(points[neighbor] - points[current]))
            # Prefer the road's wider interior when choosing one path around
            # an occluder. This changes route selection, not measured length.
            radius = distance[points[neighbor, 1], points[neighbor, 0]]
            candidate = cost + step * (1 + .5 / max(1., float(radius)))
            if candidate < costs[neighbor]:
                costs[neighbor], parents[neighbor] = candidate, current
                heapq.heappush(queue, (candidate, neighbor))
    paths = []
    for endpoint in ends:
        if endpoint == root or not np.isfinite(costs[endpoint]):
            continue
        path = [endpoint]
        while path[-1] != root:
            path.append(int(parents[path[-1]]))
        path.reverse()
        if _length(points[path]) >= max(2, config.min_branch_length * height):
            paths.append(path)
    if not paths:
        return []
    # Assign observed road pixels to their nearest graph point. Branch area
    # therefore measures actual mask support, not a length-times-width box
    # that could overstate support near a hole or at a clipped image edge.
    _, owner = cv2.distanceTransformWithLabels(255 - skeleton, cv2.DIST_L2, 5,
                                              labelType=cv2.DIST_LABEL_PIXEL)
    support = np.bincount(owner[road > 0], minlength=int(owner.max()) + 1)
    pixel_support = support[owner[points[:, 1], points[:, 0]]]
    visits = np.zeros(len(points), np.int32)
    for path in paths:
        visits[path] += 1
    routes = []
    for path in paths:
        unique = next((i for i, point in enumerate(path) if visits[point] == 1), 0)
        split = max(0, unique - 1)
        tail = points[path[split:]]
        route_points = _extend_to_border(points[path], road, distance)
        tail_length = _length(tail)
        widths = distance[tail[:, 1], tail[:, 0]] * 2
        mean_width = float(np.mean(widths))
        area = float(pixel_support[path[split:]].sum())
        if (tail_length < config.min_branch_length * height
                or mean_width < config.min_branch_width * width
                or area < config.min_branch_area * height * width):
            continue
        direction = _unit(tail[-1].astype(float) - tail[0])
        routes.append(Branch(0, route_points, tuple(float(v) for v in tail[0]),
                             tuple(float(v) for v in direction), tail_length, mean_width, area))
    # Close, nearly parallel terminal spurs can survive thinning at ragged
    # caps; do not count them as a junction. Keep the best-supported arm.
    routes.sort(key=lambda branch: branch.area, reverse=True)
    accepted = []
    for branch in routes:
        duplicate = any(
            np.linalg.norm(np.subtract(branch.split_point, other.split_point)) < .12 * max(height, width)
            and math.degrees(math.acos(float(np.clip(np.dot(branch.direction, other.direction), -1, 1)))) < config.min_branch_angle
            for other in accepted)
        if not duplicate:
            accepted.append(branch)
    return accepted


def _corridor(road, distance, branches, share):
    centerline = np.zeros_like(road)
    if not branches or share <= 0:
        return centerline
    for branch in branches:
        cv2.polylines(centerline, [np.rint(branch.points).astype(np.int32)], False, 255, 1)
    centerline[road == 0] = 0
    if not np.any(centerline):
        return centerline
    if share >= 1:
        return road.copy()
    # Each pixel receives the local half-width of its nearest centerline
    # point. Euclidean distance to the curve measures the cross-road offset
    # independent of its angle to image rows. Junction corridors are united.
    offset, labels = cv2.distanceTransformWithLabels(255 - centerline, cv2.DIST_L2, 5,
                                                   labelType=cv2.DIST_LABEL_PIXEL)
    widths = np.zeros(int(labels.max()) + 1, np.float32)
    seeds = centerline > 0
    widths[labels[seeds]] = distance[seeds]
    center = (offset <= share * widths[labels]) & (road > 0)
    return np.where(center, 255, 0).astype(np.uint8)


class CenterlineGeometry:
    def __init__(self, zones=None, config=None):
        # Lazy import avoids a cycle when road_geometry re-exports this API.
        if zones is None:
            from road_geometry import Zones
            zones = Zones()
        self.zones = zones
        self.config = config or GeometryConfig()
        self.reset()

    def reset(self):
        self._shape = None
        self._tracks = []
        self._next_id = 1
        self._main_id = None
        self._previous_direction = None

    def _track(self, detections, shape):
        tracks = [track for track in self._tracks if track.state != "removed"]
        unmatched_tracks, unmatched_detections = set(range(len(tracks))), set(range(len(detections)))
        scale = float(max(shape))
        pairs = []
        for ti, track in enumerate(tracks):
            for di, detection in enumerate(detections):
                position = float(np.linalg.norm(np.subtract(track.endpoint, detection.endpoint))) / scale
                angle = 1 - float(np.clip(np.dot(track.direction, detection.direction), -1, 1))
                split = float(np.linalg.norm(np.subtract(track.split_point, detection.split_point))) / scale
                if position <= self.config.match_distance and angle < .8:
                    pairs.append((position + .12 * angle + .08 * split, ti, di))
        for _, ti, di in sorted(pairs):
            if ti not in unmatched_tracks or di not in unmatched_detections:
                continue
            track, detection = tracks[ti], detections[di]
            detection.id = track.id
            detection.confirmation_count = track.confirmation_count + 1 if track.missed_frames == 0 else 1
            detection._confirmed = track._confirmed or detection.confirmation_count >= self.config.confirm_frames
            detection.state = "confirmed" if detection._confirmed else "candidate"
            tracks[ti] = detection
            unmatched_tracks.remove(ti)
            unmatched_detections.remove(di)
        for ti in unmatched_tracks:
            track = tracks[ti]
            track.missed_frames += 1
            track.confirmation_count = 0
            track.state = "removed" if track.missed_frames > self.config.missing_frames else "missing"
        for di in sorted(unmatched_detections):
            detection = detections[di]
            detection.id = self._next_id
            self._next_id += 1
            detection._confirmed = self.config.confirm_frames == 1
            detection.state = "confirmed" if detection._confirmed else "candidate"
            tracks.append(detection)
        self._tracks = tracks
        return tracks

    def _choose_main(self, detections):
        if not detections:
            return next((track for track in self._tracks if track.id == self._main_id and track.state != "removed"), None)
        def score(branch):
            direction = _unit(branch.points[-1].astype(float) - branch.points[0])
            continuity = float(np.dot(direction, self._previous_direction)) if self._previous_direction is not None else -float(direction[1])
            return continuity + (.3 if branch.id == self._main_id else 0) + .05 * math.log1p(branch.area)
        main = max(detections, key=score)
        self._main_id = main.id
        self._previous_direction = _unit(main.points[-1].astype(float) - main.points[0])
        return main

    def process(self, binary_mask):
        mask = np.asarray(binary_mask)
        if mask.ndim != 2 or min(mask.shape) < 2:
            raise ValueError("Geometry expects a two-dimensional unpadded ROI mask of at least 2x2")
        if self._shape != mask.shape:
            self.reset()
            self._shape = mask.shape
        road = _clean_road(mask, self.config)
        if np.all(road):
            # There is no observed lateral boundary in a full-frame mask.
            # Use the visible image center, avoiding OpenCV's infinite DT.
            distance = np.full(mask.shape, mask.shape[1] / 2, np.float32)
            skeleton = np.zeros_like(road)
            skeleton[:, mask.shape[1] // 2] = 255
        else:
            distance = cv2.distanceTransform(road, cv2.DIST_L2, 5)
            skeleton = _prune_spurs(thin_mask(road), distance, self.config.min_branch_length * mask.shape[0])
        detections = _outgoing_routes(skeleton, road, distance, self.config)
        tracks = self._track(detections, mask.shape)
        main = self._choose_main(detections)
        selected = [main] if main is not None else []
        if self.config.corridor_mode == "all":
            selected += [track for track in tracks if track._confirmed and track.state != "removed"
                         and (main is None or track.id != main.id)]
        center = _corridor(road, distance, selected, self.zones.center)
        # A retained missing track can intersect a changed mask in isolated
        # pieces. Keep its visible corridor connected to the current origin.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(center, connectivity=8)
        if count > 2:
            anchor_label = 0
            if main is not None:
                x, y = np.rint(main.points[0]).astype(int)
                anchor_label = int(labels[y, x])
            if anchor_label == 0:
                anchor_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            center[labels != anchor_label] = 0
        colors = np.zeros((*road.shape, 3), np.uint8)
        colors[road > 0] = (0, 255, 255)
        colors[center > 0] = (0, 255, 0)
        # Snapshot mutable tracks so callers can compare past frame results.
        snapshots = [Branch(**{**vars(track), "points": track.points.copy()}) for track in tracks]
        return GeometryResult(road, colors, center, skeleton, snapshots,
                              main.id if main is not None else None)
