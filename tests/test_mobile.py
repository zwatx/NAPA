"""Checks that need only Napa and NumPy, for platforms without OpenCV wheels, such as Android.

The Android wheels run this file on an emulator; see ``[tool.cibuildwheel.android]`` in pyproject.toml.
"""

from __future__ import annotations

import numpy as np
import pytest

from napa_vision import NapaDetector, families
from napa_vision.tags import family_layout, tag_image


@pytest.mark.parametrize("family", families())
def test_detects_drawn_tags_in_every_family(family: str) -> None:
    layout = family_layout(family)
    detector = NapaDetector(family=family)
    for tag_id in sorted({0, layout.ncodes // 2, layout.ncodes - 1}):
        art = tag_image(tag_id, family=family, cell_px=12, quiet_zone_cells=2)
        image = np.full((art.shape[0] + 60, art.shape[1] + 80), 255, np.uint8)
        image[30 : 30 + art.shape[0], 40 : 40 + art.shape[1]] = art
        for turns in range(4):
            [observation] = detector.detect(np.ascontiguousarray(np.rot90(image, turns)))
            assert (observation.tag_id, observation.family) == (tag_id, family)


def test_blank_and_noise_frames_have_no_tags() -> None:
    detector = NapaDetector()
    assert detector.detect(np.full((480, 640), 128, np.uint8)) == []
    rng = np.random.default_rng(1)
    assert detector.detect(rng.integers(0, 256, (480, 640), dtype=np.uint8)) == []
