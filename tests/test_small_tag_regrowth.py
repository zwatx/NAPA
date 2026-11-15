"""With regrow_small_tags, small tags that come apart at the finest level's sampling are regrown at full resolution and
read in the same pass."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from tests.test_napa_detector import _scene


def _blurred_small_tag(tag_id: int, side: int, angle: float, quality: int) -> tuple[np.ndarray, np.ndarray]:
    """A small tag smeared by a few pixels of motion, gamma-encoded and JPEG-compressed, as in match footage: sampled
    every other pixel, its thin border comes apart into fragments."""

    image, corners = _scene(tag_id, side, angle_deg=angle)
    image = cv2.filter2D(image, -1, np.full((1, 3), 1 / 3, np.float32))
    image = np.clip(255 * (image / 255.0) ** (1 / 2.2) * 0.7 + 30, 0, 255).astype(np.uint8)
    encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE), corners


@pytest.mark.parametrize(
    ("tag_id", "side", "angle", "quality"), [(7, 20, 25, 95), (137, 24, 12, 60), (300, 22, 12, 95)]
)
@pytest.mark.parametrize("threads", [1, 4])
def test_small_tags_that_come_apart_are_read_in_the_same_pass(
    tag_id: int, side: int, angle: float, quality: int, threads: int
) -> None:
    image, _ = _blurred_small_tag(tag_id, side, angle, quality)
    found = NapaDetector(min_tag_px=22, threads=threads, regrow_small_tags=True).detect(image)
    assert [o.tag_id for o in found] == [tag_id]
    # Corners as close as sampling every pixel gets them.
    whole = NapaDetector(min_tag_px=16).detect(image)
    np.testing.assert_allclose(found[0].corners, whole[0].corners, atol=0.05)


def test_regrowth_finds_nothing_in_noise_or_checkers() -> None:
    rng = np.random.default_rng(5)
    frames = [
        rng.integers(0, 256, (480, 640), dtype=np.uint8),
        cv2.GaussianBlur(rng.integers(0, 256, (480, 640), dtype=np.uint8), (5, 5), 0),
    ]
    for square in (5, 6, 9, 13):
        board = np.kron(
            (np.indices((480 // square + 1, 640 // square + 1)).sum(0) % 2) * 255, np.ones((square, square))
        )
        frames.append(board[:480, :640].astype(np.uint8))
    detector = NapaDetector(min_tag_px=22, regrow_small_tags=True)
    for frame in frames:
        assert detector.detect(frame) == []


def test_regrowth_is_off_by_default() -> None:
    image, _ = _blurred_small_tag(7, 20, 25, 95)
    assert NapaDetector(min_tag_px=22).regrow_small_tags is False
    assert NapaDetector(min_tag_px=22).detect(image) == []


def test_regrow_small_tags_must_be_a_bool() -> None:
    with pytest.raises(ValueError, match="regrow_small_tags"):
        NapaDetector(regrow_small_tags=1)
