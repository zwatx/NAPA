"""Ordinary detection separates supported corner contacts without a recovery pass."""

import numpy as np
import pytest

from napa_vision import NapaDetector
from napa_vision.tags import tag_image
from tests.test_napa_detector import _aprilgrid, _scene


@pytest.mark.parametrize("minimum", [16, 32])
@pytest.mark.parametrize("tag_id", [137, 586])
@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_ordinary_tag_touching_a_filled_corner(minimum, tag_id, turns):
    image, corners = _scene(tag_id, 96)
    image[144:193, 224:273] = 0
    for _ in range(turns):
        width = image.shape[1]
        image = np.rot90(image).copy()
        corners = np.column_stack((corners[:, 1], width - 1 - corners[:, 0]))
    detector = NapaDetector(min_tag_px=minimum)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [tag_id]
    assert observations[0].hamming == 0
    np.testing.assert_allclose(observations[0].corners, corners, atol=0.05)


@pytest.mark.filterwarnings("ignore:a frame filled all:RuntimeWarning")
@pytest.mark.parametrize("minimum", [16, 32])
@pytest.mark.parametrize("angle", [0.0, 12.0])
@pytest.mark.parametrize("capacity", [1, 5, 64])
def test_recovery_preserves_ordinary_board_prefix(minimum, angle, capacity):
    image, _ = _aprilgrid(list(range(100, 136)), columns=6, tag_px=48, angle_deg=angle)
    # Isolated tags make the ordinary prefix nonempty even when the rotated board merges into one blob.
    for tag_id in range(6):
        image[20:100, 20 + tag_id * 120 : 100 + tag_id * 120] = tag_image(tag_id, cell_px=8)
    ordinary_detector = NapaDetector(min_tag_px=minimum, max_detections=capacity, split_touching=False)
    try:
        ordinary = ordinary_detector.detect(image)
    finally:
        ordinary_detector.close()
    assert len(ordinary) >= min(6, capacity)
    if angle == 0:
        assert len(ordinary) == min(42, capacity)
    for shadow, split in [(False, True), (True, False), (True, True)]:
        detector = NapaDetector(
            min_tag_px=minimum, max_detections=capacity, recover_shadows=shadow, split_touching=split
        )
        try:
            recovered = detector.detect(image)
        finally:
            detector.close()
        assert len(recovered) >= len(ordinary)
        for before, after in zip(ordinary, recovered, strict=False):
            assert (before.tag_id, before.hamming, before.decision_margin) == (
                after.tag_id,
                after.hamming,
                after.decision_margin,
            )
            if shadow:  # recovery may find a tag from a coarser level, shifting its corners slightly
                np.testing.assert_allclose(before.corners, after.corners, atol=0.1)
            else:
                np.testing.assert_array_equal(before.corners, after.corners)
