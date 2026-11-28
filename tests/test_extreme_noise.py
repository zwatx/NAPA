"""Tags in extreme sensor noise."""

import numpy as np
import pytest

from napa_vision import NapaDetector
from tests.test_napa_detector import _scene


@pytest.mark.parametrize("seed", [1, 2])
def test_extreme_noise(seed):
    # Sigma 35 is beyond the stride-two noise floor's ordinary cap: a frame of it must not turn into a candidate per
    # noise blob, and a strong tag in it is still read.
    rng = np.random.default_rng(seed)
    image, corners = _scene(137, 240, angle_deg=20.0, size=(1920, 1080))
    tagged = np.clip(image.astype(np.float64) * 0.8 + 25 + rng.normal(0, 35, image.shape), 0, 255).astype(np.uint8)
    blank = np.clip(216 + rng.normal(0, 35, image.shape), 0, 255).astype(np.uint8)
    detector = NapaDetector()
    try:
        assert detector.detect(blank) == []
        observations = detector.detect(tagged)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [137]
    np.testing.assert_allclose(observations[0].corners, corners, atol=1.0)
