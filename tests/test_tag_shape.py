import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from napa_vision import tags as tg


def _narrow(width: float, height: float, tag_id: int = 42) -> np.ndarray:
    # A tag squeezed to `width` x `height` pixels, as a tag turned about the vertical axis looks.
    art = tg.tag_image(tag_id).astype(np.float32)
    n = art.shape[0]
    big = cv2.resize(art, (n * 40, n * 40), interpolation=cv2.INTER_NEAREST)
    cx, cy, hx, hy = 160.3, 120.2, width / 8 * n / 2, height / 8 * n / 2
    target = np.float32([[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]])
    source = np.float32([[0, 0], [n * 40, 0], [n * 40, n * 40], [0, n * 40]])
    warped = cv2.warpPerspective(
        big, cv2.getPerspectiveTransform(source, target), (320, 240), flags=cv2.INTER_AREA, borderValue=255
    )
    return np.clip(30 + cv2.GaussianBlur(warped, (0, 0), 0.6) * 190 / 255, 0, 255).astype(np.uint8)


def _detect(image: np.ndarray, min_tag_px: int = 16, **options):
    detector = NapaDetector(min_tag_px=min_tag_px, **options)
    try:
        return detector.detect(np.ascontiguousarray(image))
    finally:
        detector.close()


@pytest.mark.parametrize("value", ["up", "TALL", 1, True])
def test_rejects_unknown_shapes(value) -> None:
    with pytest.raises(ValueError, match="tag_shape"):
        NapaDetector(min_tag_px=16, tag_shape=value)


def test_needs_fine_sampling() -> None:
    with pytest.raises(ValueError, match="min_tag_px from 8 to 31 for tag36h11"):
        NapaDetector(min_tag_px=32, tag_shape="tall")  # samples every third pixel
    with pytest.raises(ValueError, match="min_tag_px from 8 to 23 for tag16h5"):
        NapaDetector(family="tag16h5", min_tag_px=24, tag_shape="wide")
    assert NapaDetector(min_tag_px=31, tag_shape="tall").tag_shape == "tall"
    assert NapaDetector(tag_shape="tall").tag_shape == "tall"  # the default min_tag_px=22 samples every other pixel


def test_unset_matches_omitting_it() -> None:
    image = _narrow(16, 34)
    plain, unset = _detect(image), _detect(image, tag_shape=None)
    assert [o.tag_id for o in unset] == [o.tag_id for o in plain] == [42]
    np.testing.assert_array_equal(unset[0].corners, plain[0].corners)


def test_tall_keeps_narrow_tall_tags_that_wide_misses() -> None:
    image = _narrow(12, 30)
    assert [o.tag_id for o in _detect(image, tag_shape="tall")] == [42]
    assert _detect(image, tag_shape="wide") == []


def test_turning_the_frame_swaps_the_directions() -> None:
    image = _narrow(12, 30)
    turned = np.rot90(image)  # (x, y) -> (y, width - 1 - x)
    upright, sideways = _detect(image, tag_shape="tall"), _detect(turned, tag_shape="wide")
    assert _detect(turned, tag_shape="tall") == []
    assert [o.tag_id for o in sideways] == [o.tag_id for o in upright] == [42]
    corners = np.asarray(sideways[0].corners)
    back = np.stack([image.shape[1] - 1 - corners[:, 1], corners[:, 0]], axis=1)
    assert min(np.abs(np.roll(back, r, axis=0) - upright[0].corners).max() for r in range(4)) < 0.01


def test_the_coarser_band_keeps_its_direction_too() -> None:
    # At min_tag_px=22 "tall" samples every second pixel across and every fourth down.
    image = _narrow(16, 40)
    found = _detect(image, 22, tag_shape="tall")
    assert [o.tag_id for o in found] == [42]
    turned = _detect(np.rot90(image), 22, tag_shape="wide")
    corners = np.asarray(turned[0].corners)
    back = np.stack([image.shape[1] - 1 - corners[:, 1], corners[:, 0]], axis=1)
    assert min(np.abs(np.roll(back, r, axis=0) - found[0].corners).max() for r in range(4)) < 0.01
