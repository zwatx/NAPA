from pathlib import Path

import numpy as np
import pytest

from napa_vision import NapaDetector


@pytest.mark.parametrize("filename", ["blur-corner-345.npz", "blur-decode-424.npz"])
def test_wide_motion_blur_keeps_corners_within_one_pixel(filename: str) -> None:
    # Renderer seed 2468, frames 1/11: ID345 had biased corners; ID424 failed the final decode.
    # Frozen pixels and independent projected corners keep this independent of the benchmark renderer.
    with np.load(Path(__file__).parent / "fixtures" / filename) as fixture:
        image = fixture["image"]
        expected = fixture["corners"]
        tag_id = int(fixture["tag_id"])
    detector = NapaDetector()
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [observation.tag_id for observation in observations] == [tag_id]
    errors = np.linalg.norm(observations[0].corners - expected, axis=1)
    assert errors.max() < 1.0
