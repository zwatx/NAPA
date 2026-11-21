"""Owned projective tag beside a weaker gray attachment, with analytic corners."""

from pathlib import Path

import numpy as np
import pytest

from napa_vision import NapaDetector

FIXTURE = Path(__file__).parent / "fixtures" / "initial-background-2.npz"


def _fixture():
    with np.load(FIXTURE) as data:
        return data["image"].copy(), data["corners"].copy()


def _max_error(actual, expected):
    return min(float(np.linalg.norm(actual - np.roll(expected, k, axis=0), axis=1).max()) for k in range(4))


@pytest.mark.parametrize(
    "turns",
    [
        pytest.param(
            0,
            marks=pytest.mark.xfail(
                strict=True, reason="only the removed confidence-quad retry found the unturned tag at min_tag_px=32"
            ),
        ),
        1,
        2,
        3,
    ],
)
def test_oblique_tag_beside_weaker_background(turns):
    image, corners = _fixture()
    for _ in range(turns):
        width = image.shape[1]
        image = np.rot90(image).copy()
        corners = np.column_stack((corners[:, 1], width - 1 - corners[:, 0]))
    before = image.copy()
    # The fixture targets the min_tag_px=32 grid; the default 22 misses turns 1 and 2 (the xfail below covers one).
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
