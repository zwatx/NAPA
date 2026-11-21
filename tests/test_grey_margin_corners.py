from pathlib import Path

import numpy as np
import pytest

from napa_vision import NapaDetector


@pytest.mark.parametrize("seed", [32, 241, 324, 470, 800])
@pytest.mark.parametrize(
    "options", [{}, {"split_touching": True, "recover_shadows": True}], ids=["default", "recovery"]
)
def test_grey_margin_side_ends(seed: int, options: dict) -> None:
    # A quiet zone as grey as the white cells, on a lighter background: specks of the margin can join the border's
    # blob near one corner, and the fitted side then pivots out onto the margin's outer edge at that end while the tag
    # still decodes. Seed 32 is a whole corner out on the margin along both sides. In seeds 324 and 800 one side
    # pivots out, and moving that whole side in a cell would pull its anchored end into the tag. Without the repairs
    # these corners are 3-18 px off. Regenerate with tests/fixtures/generate_grey_margin.py.
    with np.load(Path(__file__).parent / "fixtures" / f"grey-margin-{seed}.npz") as data:
        image, expected, tag_id = data["image"], data["corners"], int(data["tag_id"])
    detector = NapaDetector(**options)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [observation.tag_id for observation in observations] == [tag_id]
    corners = observations[0].corners
    error = min(
        np.linalg.norm(np.roll(candidate, shift, axis=0) - expected, axis=1).max()
        for candidate in (corners, corners[::-1])
        for shift in range(4)
    )
    assert error < 1.0
