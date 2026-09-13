import unittest

import cv2
import numpy as np

from corridor_tracker import CorridorConfig, CorridorTracker
from road_geometry import Zones


class CorridorTrackerTest(unittest.TestCase):
    def setUp(self):
        self.zones = Zones(.15, .70, .15)

    def test_curving_road_keeps_one_longitudinal_center(self):
        road = np.zeros((192, 192), np.uint8)
        for y in range(192):
            center = 96 + round(28 * np.sin(y / 191 * np.pi))
            road[y, center - 32:center + 32] = 255
        tracker = CorridorTracker(self.zones)
        result = tracker.update(road, np.full_like(road, 120))
        self.assertEqual(cv2.connectedComponents(result.center)[0] - 1, 1)
        self.assertGreater(cv2.countNonZero(result.center), cv2.countNonZero(road) * .35)

    def test_branch_requires_multiple_frames(self):
        road = np.zeros((192, 192), np.uint8)
        # Both arms meet the common trunk on the next row.  The tracker must
        # not accept one-frame segmentation noise as a new road direction.
        road[82:, 68:124] = 255
        road[:82, 22:90] = 255
        road[:82, 102:170] = 255
        tracker = CorridorTracker(self.zones, CorridorConfig(branch_confirm_frames=3))
        for _ in range(2):
            result = tracker.update(road, np.full_like(road, 120))
            self.assertEqual(result.branch_count, 0)
        result = tracker.update(road, np.full_like(road, 120))
        self.assertEqual(result.branch_count, 1)
        self.assertGreater(cv2.countNonZero(result.center), 0)


if __name__ == "__main__":
    unittest.main()
