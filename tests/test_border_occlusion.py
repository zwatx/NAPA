from __future__ import annotations

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from tests.test_napa_detector import _scene


def _crossed(side: int, angle: float, corner: int, level: float, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """A tag with a bar, like a cable, lying across one corner: it covers part of the border on two sides."""

    image, corners = _scene(42, side, angle_deg=angle)
    centre = corners.mean(0)
    start = centre + (corners[corner] - centre) * 1.08
    end = centre + (corners[corner] - centre) * reach
    along = (end - start) / np.linalg.norm(end - start)
    across = np.array([-along[1], along[0]]) * 0.06 * side
    bar = np.array([start - across, end - across, end + across, start + across])
    frame = image.astype(np.float32)
    cv2.fillConvexPoly(frame, np.int32(np.round(bar * 16)), level, cv2.LINE_AA, 4)
    frame = 30 + cv2.GaussianBlur(frame, (0, 0), 0.6) * (190 / 255)  # print and lighting: black 30, white 220
    noise = np.random.default_rng(side + corner).normal(0, 1.5, frame.shape)
    return np.clip(frame + noise, 0, 255).astype(np.uint8), corners


@pytest.mark.parametrize("level", [0.0, 128.0, 255.0])
def test_a_bar_across_the_border_leaves_the_corners_in_place(level: float) -> None:
    # The cells under the bar misread the border's width by much of a cell; they mustn't push the corners out.
    detector = NapaDetector()
    errors = []
    try:
        for side in (120, 200):
            for corner in range(4):
                for reach in (0.6, 0.3):
                    image, expected = _crossed(side, 17.0, corner, level, reach)
                    found = [o for o in detector.detect(image) if o.tag_id == 42]
                    if found:
                        shifts = (
                            np.linalg.norm(np.roll(found[0].corners, s, 0) - expected, axis=1).max() for s in range(4)
                        )
                        errors.append(min(shifts))
    finally:
        detector.close()
    assert len(errors) >= 8
    assert max(errors) < 0.15
