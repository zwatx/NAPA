from pathlib import Path

import numpy as np
import pytest

from napa_vision import NapaDetector


@pytest.mark.parametrize("recovery", [False, True])
def test_clipped_oblique_tag_keeps_corners_within_quarter_pixel(recovery: bool) -> None:
    # Own area-sampled tag rendering, projected corners, and 1.5x exposure clipped to uint8.
    # The oblique cell direction differs from the perpendicular direction used to move the edges.
    with np.load(Path(__file__).parent / "fixtures" / "oblique-clipped-24.npz") as fixture:
        image = fixture["image"]
        expected = fixture["corners"]
        tag_id = int(fixture["tag_id"])
    detector = NapaDetector(min_tag_px=16, recover_shadows=recovery, split_touching=recovery)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [observation.tag_id for observation in observations] == [tag_id]
    errors = np.linalg.norm(observations[0].corners - expected, axis=1)
    assert errors.max() < 0.25
