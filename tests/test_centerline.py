import unittest

import cv2
import numpy as np

from road_geometry import CenterlineGeometry, GeometryConfig, Zones


def blank():
    return np.zeros((144, 256), np.uint8)


def stroke(mask, points, width=25):
    cv2.polylines(mask, [np.asarray(points, np.int32)], False, 255, width)
    return mask


def fork():
    mask = stroke(blank(), [(128, 143), (128, 80)])
    stroke(mask, [(128, 80), (43, 5)])
    stroke(mask, [(128, 80), (213, 5)])
    return mask


class CenterlineTest(unittest.TestCase):
    def settled(self, mask, config=None):
        geometry = CenterlineGeometry(config=config)
        for _ in range(geometry.config.confirm_frames):
            result = geometry.process(mask)
        return result

    def assert_zones(self, result):
        self.assertFalse(np.any(result.center[result.road == 0]))
        self.assertFalse(np.any(result.colors[result.road == 0]))
        self.assertTrue(np.all(result.colors[result.road > 0, 1] == 255))
        if np.any(result.center):
            self.assertEqual(cv2.connectedComponents(result.center)[0] - 1, 1)

    def test_empty_and_full_masks(self):
        empty = self.settled(blank())
        self.assertEqual(empty.branches, [])
        self.assertIsNone(empty.main_branch_id)
        self.assertFalse(np.any(empty.colors))
        full = self.settled(np.full_like(blank(), 255))
        self.assertEqual(len(full.branches), 1)
        self.assert_zones(full)

    def test_straight_center_and_visible_width(self):
        mask = blank()
        mask[:, 108:148] = 255
        result = self.settled(mask)
        self.assertEqual(len(result.branches), 1)
        self.assertTrue(np.all(result.center[:, 128] > 0))
        self.assertAlmostEqual(np.count_nonzero(result.center[70]) / 40, .70, delta=.06)
        self.assert_zones(result)

    def test_diagonal_center_follows_perpendicular_width(self):
        mask = stroke(blank(), [(40, 0), (195, 143)], 25)
        result = self.settled(mask)
        self.assertEqual(len(result.branches), 1)
        for y in range(15, 130, 10):
            x = round(40 + 155 * y / 143)
            self.assertGreater(result.center[y, x], 0)
            centers = np.flatnonzero(result.center[y])
            self.assertLess(abs(float(centers.mean()) - x), 4)
        self.assert_zones(result)

    def test_s_curve_no_false_fork(self):
        ys = np.arange(144)
        xs = np.rint(128 + 42 * np.sin(ys * 2 * np.pi / 144)).astype(int)
        mask = stroke(blank(), np.column_stack((xs, ys)), 23)
        result = self.settled(mask)
        self.assertEqual(len(result.branches), 1)
        self.assertGreater(np.mean(result.center[ys[10:-10], xs[10:-10]] > 0), .97)
        self.assert_zones(result)

    def test_sharp_turns(self):
        for end_x in (0, 255):
            with self.subTest(end_x=end_x):
                mask = stroke(blank(), [(128, 143), (128, 55), (end_x, 55)], 25)
                result = self.settled(mask)
                self.assertEqual(len(result.branches), 1)
                self.assertGreater(result.center[100, 128], 0)
                self.assertGreater(result.center[55, 25 if end_x == 0 else 230], 0)
                self.assert_zones(result)

    def test_y_t_and_cross_topology(self):
        t_shape = stroke(blank(), [(128, 143), (128, 65)], 25)
        stroke(t_shape, [(0, 65), (255, 65)], 25)
        cross = stroke(t_shape.copy(), [(128, 65), (128, 0)], 25)
        for name, mask, expected in (("Y", fork(), 2), ("T", t_shape, 2), ("cross", cross, 3)):
            with self.subTest(name=name):
                result = self.settled(mask)
                self.assertEqual(len(result.branches), expected)
                self.assertTrue(all(branch.state == "confirmed" for branch in result.branches))
                self.assert_zones(result)

    def test_different_widths_and_branch_threshold(self):
        mask = stroke(blank(), [(128, 143), (128, 80)], 28)
        stroke(mask, [(128, 80), (43, 5)], 28)
        stroke(mask, [(128, 80), (213, 5)], 15)
        result = self.settled(mask)
        self.assertEqual(len(result.branches), 2)
        widths = sorted(branch.width for branch in result.branches)
        self.assertGreater(widths[1] / widths[0], 1.4)
        self.assert_zones(result)
        narrow_rejected = self.settled(mask, GeometryConfig(min_branch_width=.09))
        self.assertEqual(len(narrow_rejected.branches), 1)

    def test_false_components_are_removed(self):
        mask = stroke(blank(), [(128, 0), (128, 143)], 25)
        cv2.rectangle(mask, (10, 20), (65, 90), 255, -1)
        cv2.circle(mask, (230, 45), 20, 255, -1)
        result = self.settled(mask)
        self.assertEqual(len(result.branches), 1)
        self.assertFalse(np.any(result.road[20:90, 10:65]))
        self.assert_zones(result)

    def test_road_at_either_frame_edge(self):
        for start in (0, 230):
            with self.subTest(start=start):
                mask = blank()
                mask[:, start:start + 26] = 255
                result = self.settled(mask)
                self.assertEqual(len(result.branches), 1)
                self.assertGreater(result.center[70, start + 12], 0)
                self.assert_zones(result)

    def test_short_spur_does_not_create_branch(self):
        mask = stroke(blank(), [(128, 0), (128, 143)], 25)
        stroke(mask, [(128, 75), (151, 75)], 11)
        result = self.settled(mask)
        self.assertEqual(len(result.branches), 1)
        self.assert_zones(result)

    def test_shadow_gap_and_vehicle_hole(self):
        shadow = stroke(blank(), [(128, 0), (128, 143)], 37)
        shadow[70:73] = 0
        result = self.settled(shadow)
        self.assertEqual(len(result.branches), 1)
        self.assertGreater(result.center[71, 128], 0)
        self.assert_zones(result)
        occluded = stroke(blank(), [(128, 0), (128, 143)], 43)
        occluded[58:83, 120:138] = 0
        result = self.settled(occluded)
        self.assertEqual(len(result.branches), 1)
        self.assertEqual(result.road[70, 128], 0)
        self.assertGreaterEqual(result.branches[0].points[0, 1], 142)
        self.assertGreater(result.center[143, 128], 0)
        self.assertGreater(result.center[0, 128], 0)
        self.assert_zones(result)

    def test_confirmation_missing_recovery_and_removal(self):
        geometry = CenterlineGeometry()
        first = geometry.process(fork())
        second = geometry.process(fork())
        third = geometry.process(fork())
        self.assertEqual({branch.state for branch in first.branches}, {"candidate"})
        self.assertEqual({branch.state for branch in second.branches}, {"candidate"})
        self.assertEqual({branch.state for branch in third.branches}, {"confirmed"})
        self.assertEqual([b.confirmation_count for b in first.branches], [1, 1])
        ids = {branch.id for branch in third.branches}
        main_id = third.main_branch_id
        missing = geometry.process(blank())
        self.assertEqual({branch.state for branch in missing.branches}, {"missing"})
        self.assertEqual({branch.id for branch in missing.branches}, ids)
        recovered = geometry.process(fork())
        self.assertEqual({branch.state for branch in recovered.branches}, {"confirmed"})
        self.assertEqual({branch.id for branch in recovered.branches}, ids)
        self.assertEqual(recovered.main_branch_id, main_id)
        geometry.process(blank())
        second_missing = geometry.process(blank())
        self.assertEqual({branch.state for branch in second_missing.branches}, {"missing"})
        removed = geometry.process(blank())
        self.assertEqual({branch.state for branch in removed.branches}, {"removed"})
        self.assertEqual(geometry.process(blank()).branches, [])

    def test_candidate_requires_consecutive_frames(self):
        geometry = CenterlineGeometry()
        geometry.process(fork())
        geometry.process(blank())
        recovered = geometry.process(fork())
        self.assertTrue(all(branch.confirmation_count == 1 for branch in recovered.branches))
        self.assertTrue(all(branch.state == "candidate" for branch in recovered.branches))

    def test_main_route_mode_is_separate_from_all_roads(self):
        all_roads = self.settled(fork())
        main = self.settled(fork(), GeometryConfig(corridor_mode="main"))
        self.assertEqual(len(main.branches), 2)
        self.assertLess(np.count_nonzero(main.center), np.count_nonzero(all_roads.center) * .8)
        self.assert_zones(main)

    def test_resize_and_explicit_reset_clear_history(self):
        geometry = CenterlineGeometry()
        for _ in range(3):
            geometry.process(fork())
        resized = cv2.resize(fork(), (224, 128), interpolation=cv2.INTER_NEAREST)
        result = geometry.process(resized)
        self.assertEqual(result.road.shape, (128, 224))
        self.assertTrue(all(branch.state == "candidate" for branch in result.branches))
        geometry.reset()
        result = geometry.process(fork())
        self.assertTrue(all(branch.confirmation_count == 1 for branch in result.branches))

    def test_zone_extremes_and_bad_configuration(self):
        mask = stroke(blank(), [(128, 0), (128, 143)], 25)
        no_center = CenterlineGeometry(Zones(.5, 0, .5)).process(mask)
        self.assertFalse(np.any(no_center.center))
        all_center = CenterlineGeometry(Zones(0, 1, 0)).process(mask)
        np.testing.assert_array_equal(all_center.center, all_center.road)
        for config in ({"confirm_frames": 0}, {"missing_frames": -1},
                       {"min_branch_width": -1}, {"corridor_mode": "unknown"}):
            with self.assertRaises(ValueError):
                GeometryConfig(**config)


if __name__ == "__main__":
    unittest.main()
