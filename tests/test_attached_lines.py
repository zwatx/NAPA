"""Tags with thin dark lines touching them."""

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from tests.test_napa_detector import _scene


def _lines_off_corners(tag_id, angle_deg, corners_with_lines):
    """A 240 px tag with 3 px dark grey lines running 120 px out of the given corners, along its diagonals."""

    image, corners = _scene(tag_id, 240, angle_deg=angle_deg, size=(1280, 720))
    centre = corners.mean(axis=0)
    for k in corners_with_lines:
        out = (corners[k] - centre) / np.linalg.norm(corners[k] - centre)
        start, end = np.round(corners[k] - 4 * out).astype(int), np.round(corners[k] + 120 * out).astype(int)
        cv2.line(image, tuple(int(v) for v in start), tuple(int(v) for v in end), 60, 3)
    return image, corners


@pytest.mark.parametrize("tag_id", [5, 137, 400])
@pytest.mark.parametrize(
    ("angle", "lines"),
    [(0.0, (0,)), (40.0, (0,)), (0.0, (0, 2)), (40.0, (0, 2)), (0.0, (0, 1)), (25.0, (0, 1))],
)
def test_tag_with_lines_off_its_corners(tag_id, angle, lines):
    # The lines carry the blob's extremes off the tag, so its first rough quad fails; the trimmed outline finds it.
    image, corners = _lines_off_corners(tag_id, angle, lines)
    detector = NapaDetector()
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [tag_id]
    assert observations[0].hamming == 0
    np.testing.assert_allclose(observations[0].corners, corners, atol=0.3)
