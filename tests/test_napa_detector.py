from __future__ import annotations

import cv2
import numpy as np
import pytest

import napa_vision.napa_detector as napa_detector
from napa_vision import NapaDetector, OpenCVArucoFiducialDetector, VisionProcessor, default_detector, native_available
from napa_vision.fiducials import make_apriltag_detector

pytestmark = pytest.mark.skipif(not native_available(), reason="Napa core library is not built")

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def _scene(tag_id: int, side: int, *, angle_deg: float = 0.0, size: tuple[int, int] = (640, 480)):
    """Warp an ideal marker into a frame and return it with exact OpenCV-order corners."""

    width, height = size
    cells = 10
    marker = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, 8 * cells, borderBits=1)
    padded = np.full((10 * cells, 10 * cells), 255, np.uint8)
    padded[cells : 9 * cells, cells : 9 * cells] = marker
    source = np.array([[cells, cells], [9 * cells, cells], [9 * cells, 9 * cells], [cells, 9 * cells]]) - 0.5
    angle = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    unit = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    corners = unit @ rotation.T * side + [width / 2, height / 2]
    homography = cv2.getPerspectiveTransform(source.astype(np.float32), corners.astype(np.float32))
    image = cv2.warpPerspective(padded, homography, (width, height), flags=cv2.INTER_AREA, borderValue=255)
    return image, corners


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_decodes_every_tag_id_with_exact_corner_order(rotation: int) -> None:
    detector = NapaDetector()
    for tag_id in range(0, 587, 1 if rotation == 0 else 7):
        image, corners = _scene(tag_id, 120, angle_deg=90.0 * rotation)
        observations = detector.detect(image)
        assert [observation.tag_id for observation in observations] == [tag_id]
        np.testing.assert_allclose(observations[0].corners, corners, atol=0.35)


@pytest.mark.parametrize("side", [32, 48, 96, 240])
@pytest.mark.parametrize("angle", [0.0, 17.0, 45.0, 71.0])
def test_sub_pixel_corners_across_sizes_and_angles(side: int, angle: float) -> None:
    image, corners = _scene(42, side, angle_deg=angle)
    observations = NapaDetector().detect(image)
    assert [observation.tag_id for observation in observations] == [42]
    assert np.abs(observations[0].corners - corners).max() < 0.6


def test_matches_opencv_ids_and_corner_order() -> None:
    image, _corners = _scene(314, 160, angle_deg=33.0)
    napa = NapaDetector().detect(image)
    opencv = OpenCVArucoFiducialDetector().detect(image)
    assert [observation.tag_id for observation in napa] == [observation.tag_id for observation in opencv] == [314]
    assert np.abs(napa[0].corners - opencv[0].corners).max() < 1.5


def test_accepts_colour_and_strided_frames() -> None:
    image, corners = _scene(5, 100)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bgra = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    wide = np.zeros((image.shape[0], image.shape[1] * 2), np.uint8)
    wide[:, ::2] = image
    padded = np.full((image.shape[0] + 10, image.shape[1] + 30), 255, np.uint8)
    padded[5:-5, 10:-20] = image
    detector = NapaDetector()
    for frame in (bgr, bgra, wide[:, ::2], padded[5:-5, 10:-20]):
        observations = detector.detect(frame)
        assert [observation.tag_id for observation in observations] == [5]
        np.testing.assert_allclose(observations[0].corners, corners, atol=0.35)


def test_rejects_invalid_images_and_parameters() -> None:
    detector = NapaDetector()
    with pytest.raises(ValueError, match="8-bit"):
        detector.detect(np.zeros((64, 64), np.float32))
    with pytest.raises(ValueError, match="grayscale"):
        detector.detect(np.zeros((64, 64, 2), np.uint8))
    assert detector.detect(np.zeros((8, 8), np.uint8)) == []
    with pytest.raises(ValueError, match="30000"):
        detector.detect(np.lib.stride_tricks.as_strided(np.zeros(1, np.uint8), (16, 30001), (0, 0)))
    for kwargs in (
        {"min_tag_px": 4},
        {"max_hamming": 4},
        {"min_contrast": 0},
        {"min_border_cells": 29},
        {"sharpening": 2.0},
        {"max_detections": 0},
        {"min_tag_px": True},
    ):
        with pytest.raises(ValueError):
            NapaDetector(**kwargs)


def test_no_false_positives_on_noise_text_and_flat_frames() -> None:
    rng = np.random.default_rng(7)
    detector = NapaDetector()
    frames = [
        np.zeros((480, 640), np.uint8),
        np.full((480, 640), 255, np.uint8),
        rng.integers(0, 256, (480, 640), dtype=np.uint8),
        cv2.GaussianBlur(rng.integers(0, 256, (480, 640), dtype=np.uint8), (5, 5), 0),
    ]
    text = np.full((480, 640), 230, np.uint8)
    for row in range(12):
        cv2.putText(text, "NAPA 2468 ## []", (10, 40 * row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 3)
    frames.append(text)
    checker = np.kron((np.indices((12, 16)).sum(0) % 2) * 255, np.ones((40, 40))).astype(np.uint8)
    frames.append(checker)
    for frame in frames:
        assert detector.detect(frame) == []


def test_decodes_with_corrupted_payload_bits() -> None:
    image, _corners = _scene(100, 200)
    cell = 200 / 8
    origin = 320 - 100
    # Flip three payload cells; tag36h11's minimum distance of 11 keeps three corrections unambiguous.
    for row, column in ((2, 2), (4, 5), (6, 3)):
        y0 = int(240 - 100 + row * cell + 4)
        x0 = int(origin + column * cell + 4)
        patch = image[y0 : y0 + int(cell) - 8, x0 : x0 + int(cell) - 8]
        patch[...] = 255 - patch
    assert [observation.tag_id for observation in NapaDetector().detect(image)] == [100]
    assert NapaDetector(max_hamming=2).detect(image) == []


def test_multiple_tags_and_processor_default() -> None:
    frame = np.full((480, 640), 255, np.uint8)
    expected = {}
    for index, (tag_id, x, y) in enumerate(((1, 60, 60), (42, 400, 60), (200, 60, 300), (586, 420, 300))):
        side = 72 + 24 * index
        frame[y : y + side, x : x + side] = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, side, borderBits=1)
        expected[tag_id] = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]]) - 0.5
    processor = VisionProcessor()
    assert isinstance(processor.detector, NapaDetector)
    result = processor.process(frame, timestamp=1.0)
    found = {observation.tag_id: observation.corners for observation in result.image_observations}
    assert set(found) == set(expected)
    for tag_id, corners in expected.items():
        np.testing.assert_allclose(found[tag_id], corners, atol=0.2)


def test_default_detector_falls_back_without_native_library(monkeypatch, tmp_path) -> None:
    napa_detector._library.cache_clear()
    monkeypatch.setenv("NAPA_CORE_LIBRARY", str(tmp_path / "missing.so"))
    monkeypatch.setattr(napa_detector, "_LIBRARY_PATTERNS", ())
    try:
        assert not native_available()
        assert isinstance(default_detector(), OpenCVArucoFiducialDetector)
        with pytest.raises(napa_detector.NativeLibraryError, match="NAPA_CORE_LIBRARY"):
            NapaDetector()
    finally:
        napa_detector._library.cache_clear()


def test_minimum_size_guarantee_holds_at_every_sampling_phase() -> None:
    """Tags at min_tag_px are found whatever their sub-pixel phase and rotation relative to the rake."""

    detector = NapaDetector(min_tag_px=32)
    width, height = 320, 240
    source = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], np.float32) - 0.5
    misses = []
    for side in (32, 48):
        for angle in range(0, 90, 9):
            for offset in (0.0, 1.0, 1.75, 2.5):
                tag_id = (side + angle * 7 + int(offset * 4)) % 587
                padded = np.full((100, 100), 255, np.uint8)
                padded[10:90, 10:90] = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, 80, borderBits=1)
                theta = np.deg2rad(angle)
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                unit = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
                corners = unit @ rotation.T * side + [width / 2 + offset, height / 2 + offset / 2]
                # Render at 4x and area-average down, as a camera integrates light over each pixel.
                homography = cv2.getPerspectiveTransform(source, (corners * 4 + 1.5).astype(np.float32))
                large = cv2.warpPerspective(padded, homography, (width * 4, height * 4), borderValue=255)
                image = cv2.GaussianBlur(cv2.resize(large, (width, height), interpolation=cv2.INTER_AREA), (0, 0), 0.8)
                found = [o for o in detector.detect(image) if o.tag_id == tag_id]
                if not found or np.abs(found[0].corners - corners).max() > 0.5:
                    misses.append((side, angle, offset))
    assert misses == []


def test_napa_is_a_named_comparison_candidate() -> None:
    assert isinstance(make_apriltag_detector("napa"), NapaDetector)
    assert isinstance(make_apriltag_detector("baseline"), OpenCVArucoFiducialDetector)
