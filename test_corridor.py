import cv2
import numpy as np

from corridor import CorridorState, build_zones


def synthetic_probability():
    mask = np.zeros((240, 400), np.uint8)
    road = np.array([[80, 220], [155, 120], [180, 20], [250, 20], [245, 120], [350, 220]])
    cv2.fillPoly(mask, [road], 255)
    cv2.circle(mask, (220, 155), 20, 0, -1)  # obstacle
    branch = np.array([[190, 105], [80, 25], [135, 20], [230, 115]])
    cv2.fillPoly(mask, [branch], 255)
    return mask.astype(np.float32) / 255.0


def test_partition_and_obstacle():
    probability = synthetic_probability()
    zones, state, info = build_zones(probability, CorridorState(), 0.5)
    assert np.all(zones[probability == 0] == 0), "background/obstacle was colored"
    coverage = float((zones[probability > 0] > 0).mean())
    assert coverage > 0.97, f"too much road was lost: coverage={coverage:.3f}"
    assert set(np.unique(zones)) == {0, 1, 2, 3}
    assert np.isfinite(info["heading_deg"])

    empty, state, _ = build_zones(np.zeros_like(probability), state, 0.5)
    assert not empty.any(), "no-road frame must stay empty even after a road frame"


if __name__ == "__main__":
    test_partition_and_obstacle()
    print("corridor test: OK (fork/obstacle/no-road and three longitudinal zones)")
