"""New proposal shapes must not displace established, better corner measurements."""

from pathlib import Path

import numpy as np
import pytest

from napa_vision import NapaDetector


@pytest.mark.parametrize("tag_id", [6, 390, 484])
@pytest.mark.parametrize("split", [False, True])
def test_new_shape_keeps_existing_noisy_tag_corners(tag_id, split):
    # Owned projected tags, seed 2468; frozen crops retain the original sampling phase.
    # Immediate acceptance of new shapes displaced the correct later fit by 12–29 px.
    with np.load(Path(__file__).parent / "fixtures" / f"noisy-corner-{tag_id}.npz") as fixture:
        image, expected = fixture["image"], fixture["corners"]
    detector = NapaDetector(min_tag_px=32, split_touching=split)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [tag_id]
    error = min(
        np.linalg.norm(observations[0].corners - np.roll(expected, shift, axis=0), axis=1).max() for shift in range(4)
    )
    assert error < 0.5
