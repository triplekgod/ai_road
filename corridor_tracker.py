"""Temporal, camera-anchored road-corridor tracker for unstable quarry scenes.

The segmentation mask says only where road *may* be.  This module keeps a
directional corridor state, so every frame is not free to invent a new centre.
It deliberately works at the neural-network resolution (typically 192x192).
"""
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class CorridorConfig:
    anchor_x: float = .50
    anchor_y: float = .88
    max_step_fraction: float = .065
    predicted_weight: float = 1.8
    smooth_weight: float = .7
    branch_confirm_frames: int = 5
    branch_hold_frames: int = 12
    min_branch_rows: float = .10
    min_branch_gap: float = .06

    def __post_init__(self):
        if not 0 <= self.anchor_x <= 1 or not 0 <= self.anchor_y <= 1:
            raise ValueError("anchor coordinates must be fractions in [0, 1]")
        if self.max_step_fraction <= 0 or self.branch_confirm_frames < 1 or self.branch_hold_frames < 0:
            raise ValueError("invalid tracker configuration")


@dataclass
class BranchState:
    points: np.ndarray
    count: int = 1
    missed: int = 0


@dataclass
class CorridorResult:
    colors: np.ndarray
    center: np.ndarray
    confidence: float
    branch_count: int


def _segments(row):
    xs = np.flatnonzero(row)
    if not len(xs):
        return []
    return [(int(part[0]), int(part[-1])) for part in np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)]


class CorridorTracker:
    def __init__(self, zones, config=None):
        self.zones = zones
        self.config = config or CorridorConfig()
        self.reset()

    def reset(self):
        self.main_x = None
        self.previous_gray = None
        self.branches: Dict[str, BranchState] = {}
        self.last_confidence = 0.

    def _flow_shift(self, gray):
        if self.main_x is None or self.previous_gray is None or self.previous_gray.shape != gray.shape:
            return 0., 0.
        ys = np.linspace(0, len(self.main_x) - 1, 20).astype(np.int32)
        points = np.array([[self.main_x[y], y] for y in ys if np.isfinite(self.main_x[y])], np.float32)
        if len(points) < 4:
            return 0., 0.
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points.reshape(-1, 1, 2), None)
        good = status.ravel().astype(bool)
        if good.sum() < 4:
            return 0., 0.
        delta = next_points.reshape(-1, 2)[good] - points[good]
        return tuple(np.median(delta, axis=0))

    def _prediction(self, height, width, dx, dy):
        if self.main_x is None or len(self.main_x) != height:
            return np.full(height, self.config.anchor_x * (width - 1), np.float32)
        source_y = np.arange(height, dtype=np.float32) - dy
        prediction = np.interp(source_y, np.arange(height), self.main_x,
                               left=self.main_x[0], right=self.main_x[-1]) + dx
        return np.clip(prediction, 0, width - 1)

    def _anchor(self, road, prediction):
        h, w = road.shape
        start = min(h - 1, max(0, round(self.config.anchor_y * (h - 1))))
        target = self.config.anchor_x * (w - 1)
        for y in range(start, max(-1, start - max(8, h // 3)), -1):
            xs = np.flatnonzero(road[y])
            if len(xs):
                return y, int(xs[np.argmin(np.abs(xs - target))])
        ys, xs = np.nonzero(road)
        if not len(xs):
            return start, int(prediction[start])
        index = np.argmin((ys - start) ** 2 + (xs - target) ** 2)
        return int(ys[index]), int(xs[index])

    def _trace_main(self, road, prediction):
        h, w = road.shape
        distance = cv2.distanceTransform(road, cv2.DIST_L2, 3)
        anchor_y, anchor_x = self._anchor(road, prediction)
        path = np.full(h, np.nan, np.float32)
        path[anchor_y] = anchor_x
        previous = float(anchor_x)
        max_step = max(2, round(w * self.config.max_step_fraction))
        observed, support, proximity = 0, [], []
        for y in range(anchor_y, -1, -1):
            xs = np.flatnonzero(road[y])
            if not len(xs):
                continue
            allowed = xs[np.abs(xs - previous) <= max_step]
            if not len(allowed):
                allowed = xs
            # Prefer the local mask centre, but never let it jump far from
            # the predicted/previous direction without strong evidence.
            cost = (-distance[y, allowed]
                    + self.config.predicted_weight * np.abs(allowed - prediction[y])
                    + self.config.smooth_weight * np.abs(allowed - previous))
            x = float(allowed[np.argmin(cost)])
            path[y], previous = x, x
            observed += 1
            support.append(distance[y, int(x)])
            proximity.append(abs(x - prediction[y]) / max(1, w))
        valid = np.flatnonzero(np.isfinite(path))
        if len(valid) >= 2:
            # Median filter avoids rapid zone rotations from one noisy row.
            values = path[valid]
            smoothed = cv2.medianBlur(values.reshape(1, -1).astype(np.float32), 5).ravel()
            path[valid] = smoothed
            path[:valid[0]] = path[valid[0]]
            path[valid[-1] + 1:] = path[valid[-1]]
        else:
            path = prediction
        coverage = observed / max(1, anchor_y + 1)
        width_support = min(1., float(np.mean(support) if support else 0.) / max(1., w * .12))
        continuity = 1. - min(1., float(np.mean(proximity) if proximity else 1.) * 4.)
        confidence = .45 * coverage + .35 * width_support + .20 * continuity
        return path, distance, float(np.clip(confidence, 0., 1.))

    def _branch_candidates(self, road, main_x):
        h, w = road.shape
        minimum_rows = max(8, round(h * self.config.min_branch_rows))
        minimum_gap = max(8, round(w * self.config.min_branch_gap))
        groups = {"left": [], "right": []}
        for y in range(h):
            spans = _segments(road[y])
            if len(spans) < 2:
                continue
            main_index = min(range(len(spans)), key=lambda i: 0 if spans[i][0] <= main_x[y] <= spans[i][1]
                             else min(abs(main_x[y] - spans[i][0]), abs(main_x[y] - spans[i][1])))
            for i, (left, right) in enumerate(spans):
                if i == main_index:
                    continue
                gap = left - spans[main_index][1] if i > main_index else spans[main_index][0] - right
                if gap < minimum_gap:
                    continue
                side = "right" if i > main_index else "left"
                groups[side].append((y, (left + right) / 2))
        candidates = {}
        for side, points in groups.items():
            if not points:
                continue
            points.sort()
            runs, current = [], [points[0]]
            for point in points[1:]:
                if point[0] - current[-1][0] <= 2:
                    current.append(point)
                else:
                    runs.append(current); current = [point]
            runs.append(current)
            run = max(runs, key=len)
            if len(run) < minimum_rows:
                continue
            root_y = max(point[0] for point in run)
            root = (float(main_x[root_y]), float(root_y))
            branch = np.array([root, *[(x, y) for y, x in sorted(run, reverse=True)]], np.float32)
            candidates[side] = branch
        return candidates

    def _update_branches(self, candidates):
        updated = {}
        for side, points in candidates.items():
            old = self.branches.get(side)
            updated[side] = BranchState(points, (old.count + 1) if old else 1, 0)
        for side, old in self.branches.items():
            if side not in updated and old.missed < self.config.branch_hold_frames:
                updated[side] = BranchState(old.points, old.count, old.missed + 1)
        self.branches = updated

    @staticmethod
    def _paint_corridor(road, distance, paths, center_share):
        center = np.zeros_like(road)
        h, w = road.shape
        for path in paths:
            for x, y in np.rint(path).astype(np.int32):
                if 0 <= x < w and 0 <= y < h and road[y, x]:
                    radius = max(1, round(float(distance[y, x]) * center_share))
                    cv2.circle(center, (x, y), radius, 255, -1, cv2.LINE_AA)
        return cv2.bitwise_and(center, road)

    def update(self, road, gray):
        road = np.where(road > 0, 255, 0).astype(np.uint8)
        if road.ndim != 2 or gray.shape != road.shape:
            raise ValueError("road and grayscale frame must have equal two-dimensional shapes")
        dx, dy = self._flow_shift(gray)
        prediction = self._prediction(road.shape[0], road.shape[1], dx, dy)
        if cv2.countNonZero(road):
            main, distance, confidence = self._trace_main(road, prediction)
            candidates = self._branch_candidates(road, main)
            self._update_branches(candidates)
            self.main_x = main
        else:
            main, confidence = prediction, 0.
            distance = np.zeros_like(road, np.float32)
            self._update_branches({})
        self.previous_gray = gray.copy()
        self.last_confidence = confidence
        main_path = np.column_stack((main, np.arange(road.shape[0]))).astype(np.float32)
        paths = [main_path]
        paths += [state.points for state in self.branches.values()
                  if state.count >= self.config.branch_confirm_frames and state.missed == 0]
        center = self._paint_corridor(road, distance, paths, self.zones.center)
        colors = np.zeros((*road.shape, 3), np.uint8)
        colors[road > 0] = (0, 255, 255)
        colors[center > 0] = (0, 255, 0)
        confirmed = sum(state.count >= self.config.branch_confirm_frames for state in self.branches.values())
        return CorridorResult(colors, center, confidence, confirmed)
