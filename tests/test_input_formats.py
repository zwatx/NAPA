"""Colour, YUYV and strided frames: the native conversion gives OpenCV's greyscale exactly, and the same detections."""

from __future__ import annotations

import ctypes

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from napa_vision.napa_detector import _grayscale, _library
from napa_vision.tags import tag_image

pytestmark = pytest.mark.skipif(
    not hasattr(_library(), "napa_gray_from_pixels"), reason="requires a native library with napa_gray_from_pixels"
)


def _frame(channels: int, seed: int = 3, size=(203, 517)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (*size, channels), dtype=np.uint8)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("layout", ["contiguous", "cropped", "stepped", "reversed", "bgra-as-bgr", "odd-width"])
@pytest.mark.parametrize("threads", [1, 4])
def test_colour_matches_opencv_exactly(channels: int, layout: str, threads: int) -> None:
    image = _frame(channels, size=(1031, 1201) if threads > 1 else (203, 517))
    if layout == "cropped":
        image = image[3:-2, 5:-7]
    elif layout == "stepped":
        image = image[::2, ::3]
    elif layout == "reversed":
        image = image[::-1, ::-1]
    elif layout == "bgra-as-bgr":
        image = _frame(4)[:, :, :3]  # four bytes per pixel, the last row ending before its final alpha byte
    elif layout == "odd-width":
        image = image[:, :17]
    before = image.copy()
    expected = cv2.cvtColor(
        np.ascontiguousarray(image), cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    )
    detector = NapaDetector(threads=threads)
    actual = _grayscale(image, detector._handle)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_strided_grey_and_yuyv_planes(step: int) -> None:
    base = np.random.default_rng(5).integers(0, 256, (61, 33 * step + 1), dtype=np.uint8)
    view = base[:, ::step]
    np.testing.assert_array_equal(_grayscale(view), np.ascontiguousarray(view))
    yuyv = np.random.default_rng(6).integers(0, 256, (48, 70, 2), dtype=np.uint8)
    np.testing.assert_array_equal(_grayscale(yuyv[..., 0]), yuyv[..., 0])  # YUYV luma
    np.testing.assert_array_equal(_grayscale(yuyv[..., 1]), yuyv[..., 1])  # UYVY luma
    with pytest.raises(ValueError, match="grayscale"):
        _grayscale(yuyv)  # the layout is ambiguous: the caller picks the luma plane


def test_rejects_invalid_native_arguments() -> None:
    library = _library()
    src = (ctypes.c_uint8 * 64)()
    dst = (ctypes.c_uint8 * 64)()
    assert library.napa_gray_from_pixels(None, src, 4, 4, 12, 3, 1, dst, 4) == 0
    assert library.napa_gray_from_pixels(None, src, 4, 4, 11, 3, 1, dst, 4) == -1  # rows overlap
    assert library.napa_gray_from_pixels(None, src, 4, 4, 8, 2, 1, dst, 4) == -1  # colour needs 3+ bytes
    assert library.napa_gray_from_pixels(None, src, 4, 4, 4, 1, 0, dst, 3) == -1  # destination too narrow
    assert library.napa_gray_from_pixels(None, None, 4, 4, 4, 1, 0, dst, 4) == -1


@pytest.mark.parametrize("form", ["bgr", "bgra", "yuyv"])
def test_same_detections_from_every_format(form: str) -> None:
    gray = np.full((480, 640), 255, np.uint8)
    for k, (y, x) in enumerate([(40, 40), (200, 300), (300, 80)]):
        tile = tag_image(k + 3, cell_px=11)
        gray[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
    if form == "yuyv":
        frame = np.dstack([gray, np.full_like(gray, 128)])[..., 0]  # the luma plane of a YUYV frame, as a view
        reference = gray
    else:
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA if form == "bgra" else cv2.COLOR_GRAY2BGR)
        frame[..., 0] = np.clip(frame[..., 0].astype(int) + 7, 0, 255)  # not a pure grey: weights matter
        reference = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY if form == "bgra" else cv2.COLOR_BGR2GRAY)
    detector = NapaDetector()
    expected = [(o.tag_id, o.corners.tobytes()) for o in detector.detect(reference)]
    assert [tag_id for tag_id, _ in expected] == [3, 4, 5]
    assert [(o.tag_id, o.corners.tobytes()) for o in detector.detect(frame)] == expected


@pytest.mark.parametrize("shape", [(0, 10), (10, 0), (0, 10, 3), (10, 0, 4)])
def test_empty_frames_return_no_detections(shape):
    detector = NapaDetector()
    try:
        assert detector.detect(np.empty(shape, np.uint8)) == []
    finally:
        detector.close()


def test_oversized_colour_is_rejected_before_conversion(monkeypatch):
    from napa_vision import napa_detector

    def unexpected(*args, **kwargs):
        pytest.fail("oversized frame reached conversion")

    monkeypatch.setattr(napa_detector, "_native_gray", unexpected)
    detector = NapaDetector()
    try:
        with pytest.raises(ValueError, match="larger than 30000"):
            detector.detect(np.zeros((1, 30001, 3), np.uint8))
    finally:
        detector.close()


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("stride", [2**31, 2**32 + 48])
def test_stride_outside_native_int_range_is_copied(monkeypatch, channels, stride):
    # The synthetic view has no backing storage at its later rows. Intercept the copy before it reads them.
    # Its wrapped C stride would read only the real backing buffer, so a regression also fails safely.
    base = np.full((16, 16, channels), 255, np.uint8)
    shape = (16, 16) if channels == 1 else base.shape
    strides = (stride, 1) if channels == 1 else (stride, channels, 1)
    view = np.lib.stride_tricks.as_strided(base, shape, strides)
    reference = base[..., 0].copy() if channels == 1 else base.copy()
    original_copy = np.array
    copies = []

    def owned(image, *args, **kwargs):
        if image is view:
            copies.append(True)
            return reference
        return original_copy(image, *args, **kwargs)

    monkeypatch.setattr(np, "array", owned)
    actual = _grayscale(view)
    assert actual is not view
    assert copies == [True]
    np.testing.assert_array_equal(actual, np.full((16, 16), 255, np.uint8))


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("stride", [0, -1, 2**31, 2**32 + 48])
def test_singleton_strides_are_normalized_before_native_conversion(channels, axis, stride):
    shape = (1, 16) if axis == 0 else (16, 1)
    if channels != 1:
        shape += (channels,)
    base = np.full(shape, 255, np.uint8)
    strides = list(base.strides)
    strides[axis] = stride
    # A singleton axis never advances its pointer, so this view is entirely backed by base.
    view = np.ndarray(shape, np.uint8, buffer=base, strides=strides)
    assert view.flags.c_contiguous
    np.testing.assert_array_equal(_grayscale(view), base if channels == 1 else base[..., 0])
    detector = NapaDetector()
    try:
        assert detector.detect(view) == []
    finally:
        detector.close()
