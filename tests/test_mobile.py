"""Checks that need only Napa and NumPy, for platforms without OpenCV wheels, such as Android.

The Android wheels run this file on an emulator; see ``[tool.cibuildwheel.android]`` in pyproject.toml.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from napa_vision import NapaDetector, families
from napa_vision.napa_detector import _grayscale
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


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("layout", ["contiguous", "cropped", "stepped", "reversed", "transposed", "broadcast"])
def test_color_fallback_preserves_rounding_and_strides(monkeypatch, channels: int, layout: str) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    image = np.random.default_rng(7).integers(0, 256, (73, 641, channels), dtype=np.uint8)
    if layout == "cropped":
        image = image[1:-1, 2:-2]
    elif layout == "stepped":
        image = image[::2, ::2]
    elif layout == "reversed":
        image = image[::-1, ::-1, ::-1]
    elif layout == "transposed":
        image = image.transpose(1, 0, 2)
    elif layout == "broadcast":
        image = np.broadcast_to(image[:1, :1], image.shape)
    before = image.copy()
    # The original fixed-point BT.601 formula defines the fallback's exact rounding.
    weighted = sum(image[..., c].astype(np.uint32) * weight for c, weight in enumerate((1868, 9617, 4899)))
    expected = ((weighted + 8192) >> 14).astype(np.uint8)

    actual = _grayscale(image)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)
    assert actual.dtype == np.uint8 and actual.flags.c_contiguous


@pytest.mark.parametrize("shape", [(0, 5, 3), (5, 0, 4), (1, 1, 3)])
def test_color_fallback_handles_empty_and_single_pixel_images(monkeypatch, shape: tuple[int, ...]) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    image = np.full(shape, 255, dtype=np.uint8)
    np.testing.assert_array_equal(_grayscale(image), np.full(shape[:2], 255, dtype=np.uint8))


@pytest.mark.parametrize("channels", [3, 4])
def test_color_fallback_detects_tags_and_preserves_results_after_empty_frame(monkeypatch, channels: int) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    gray = np.pad(tag_image(7, cell_px=12), 20, constant_values=255)
    image = np.repeat(gray[..., None], channels, axis=2)
    if channels == 4:
        image[..., 3] = np.random.default_rng(7).integers(0, 256, gray.shape, dtype=np.uint8)
    detector = NapaDetector()
    [expected] = detector.detect(gray)
    [actual] = detector.detect(image)
    assert actual.tag_id == expected.tag_id == 7
    assert detector.detect(np.full_like(image, 255)) == []
    np.testing.assert_array_equal(actual.corners, expected.corners)
