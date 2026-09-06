import unittest
import cv2
import numpy as np

from road_geometry import Zones, zone_mask


class RoadGeometryTest(unittest.TestCase):
    def test_crossroad_has_one_connected_center_and_no_gaps(self):
        """A plus-shaped intersection must keep a continuous 70% center."""
        road = np.zeros((480, 480), dtype=np.uint8)
        road[:, 180:300] = 255       # main road
        road[180:300, :] = 255       # crossing branch

        colors = zone_mask(road, Zones(.15, .70, .15))
        green = cv2.inRange(colors, (0, 255, 0), (0, 255, 0))

        self.assertEqual(cv2.connectedComponents(green)[0] - 1, 1)
        self.assertFalse(np.any((road > 0) & np.all(colors == 0, axis=2)))
        green_share = cv2.countNonZero(green) / cv2.countNonZero(road)
        self.assertAlmostEqual(green_share, .70, delta=.03)

    def test_t_junction_keeps_center_connected(self):
        """A main road and its outgoing branch share one center zone."""
        road = np.zeros((480, 480), dtype=np.uint8)
        road[:, 180:300] = 255
        road[150:270, 180:480] = 255

        colors = zone_mask(road, Zones(.15, .70, .15))
        green = cv2.inRange(colors, (0, 255, 0), (0, 255, 0))
        self.assertEqual(cv2.connectedComponents(green)[0] - 1, 1)


if __name__ == "__main__":
    unittest.main()
