"""Owned projective tag beside a weaker gray attachment, with analytic corners."""

from pathlib import Path

import numpy as np
import pytest

from napa_vision import NapaDetector
from napa_vision.tags import tag_image

FIXTURE = Path(__file__).parent / "fixtures" / "initial-background-2.npz"


def _fixture():
    with np.load(FIXTURE) as data:
        return data["image"].copy(), data["corners"].copy()


def _max_error(actual, expected):
    return min(float(np.linalg.norm(actual - np.roll(expected, k, axis=0), axis=1).max()) for k in range(4))


def _raw(observations):
    return [(o.tag_id, o.hamming, o.decision_margin, o.corners.tolist()) for o in observations]


@pytest.mark.parametrize("turns", range(4))
def test_oblique_tag_beside_weaker_background(turns):
    image, corners = _fixture()
    for _ in range(turns):
        width = image.shape[1]
        image = np.rot90(image).copy()
        corners = np.column_stack((corners[:, 1], width - 1 - corners[:, 0]))
    before = image.copy()
    # The fixture targets the min_tag_px=32 grid; at the default 22 one rotation is missed (see the xfail below).
    detector = NapaDetector(min_tag_px=32)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [2]
    assert observations[0].hamming == 0
    assert _max_error(observations[0].corners, corners) < 0.25
    np.testing.assert_array_equal(image, before)


@pytest.mark.xfail(
    strict=True,
    reason="sampling phase: the stride-2 grid (min_tag_px 16-31) misses this tag at 4 of 16 one-pixel crops, this "
    "one included; min_tag_px=32 finds it at all 16",
)
def test_default_grid_finds_oblique_tag_turned_twice():
    image, corners = _fixture()
    for _ in range(2):
        width = image.shape[1]
        image = np.rot90(image).copy()
        corners = np.column_stack((corners[:, 1], width - 1 - corners[:, 0]))
    detector = NapaDetector()
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [2]
    assert _max_error(observations[0].corners, corners) < 0.25


@pytest.mark.filterwarnings("ignore:a frame filled all:RuntimeWarning")
@pytest.mark.parametrize("capacity", [1, 2, 64])
@pytest.mark.parametrize(("split", "shadows"), [(False, False), (True, False), (False, True), (True, True)])
def test_recovered_outline_follows_established_detection(capacity, split, shadows):
    left, _ = _fixture()
    image = np.full((240, 640), 190, np.uint8)
    image[:, :320] = left
    marker = tag_image(5, cell_px=8)
    image[72 : 72 + marker.shape[0], 432 : 432 + marker.shape[1]] = marker
    # At min_tag_px=32 tag 2 is a recovered outline appended after tag 5; at 22 it is an ordinary detection.
    ordinary = NapaDetector(min_tag_px=32)
    try:
        expected = _raw(ordinary.detect(image))
    finally:
        ordinary.close()
    assert [o[0] for o in expected] == [5, 2]
    detector = NapaDetector(min_tag_px=32, max_detections=capacity, split_touching=split, recover_shadows=shadows)
    try:
        actual = _raw(detector.detect(image))
    finally:
        detector.close()
    assert actual == expected[:capacity]
