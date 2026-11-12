"""Small whole outlines supplement ordinary detections without lowering decode confidence."""

import numpy as np
import pytest

from napa_vision import NapaDetector


def _small_scene(shape: tuple[int, int], phase: tuple[int, int] = (0, 0)) -> tuple[np.ndarray, np.ndarray]:
    cv2 = pytest.importorskip("cv2")
    image = np.full(shape, 255, np.uint8)
    x, y = (15, 18) if shape == (60, 60) else (120, 90)
    x, y = x + phase[0], y + phase[1]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image[y : y + 18, x : x + 18] = cv2.aruco.generateImageMarker(dictionary, 424, 18)
    corners = np.array([[x, y], [x + 18, y], [x + 18, y + 18], [x, y + 18]], dtype=float) - 0.5
    return image, corners


def _partial_small_outline(tag_id: int) -> tuple[np.ndarray, np.ndarray]:
    cv2 = pytest.importorskip("cv2")
    image = np.full((480, 640), 255, np.uint8)
    x, y = 213, 162
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image[y : y + 18, x : x + 18] = cv2.aruco.generateImageMarker(dictionary, tag_id, 18)
    corners = np.array([[x, y], [x + 18, y], [x + 18, y + 18], [x, y + 18]], dtype=float) - 0.5
    return image, corners


@pytest.mark.parametrize("tag_id", [42, 137])
@pytest.mark.parametrize("turns", range(4))
def test_small_outline_with_three_sampled_corners(tag_id: int, turns: int) -> None:
    """The sparse grid loses one corner to payload structure; the other three bound its seed."""
    image, expected = _partial_small_outline(tag_id)
    height, width = image.shape
    for _ in range(turns):
        image = np.ascontiguousarray(np.rot90(image))
        expected = np.column_stack([expected[:, 1], width - 1 - expected[:, 0]])
        height, width = width, height
    before = image.copy()
    detector = NapaDetector()
    try:
        [observation] = detector.detect(image)
        assert detector.detect(np.full_like(image, 255)) == []
        [repeated] = detector.detect(image)
    finally:
        detector.close()
    assert (observation.tag_id, observation.hamming) == (tag_id, 0)
    # Ordered corners check orientation as well as spatial localization.
    np.testing.assert_allclose(observation.corners, expected, atol=0.05)
    np.testing.assert_array_equal(repeated.corners, observation.corners)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("shape", [(60, 60), (480, 640)], ids=["terminal", "nonterminal"])
@pytest.mark.parametrize("phase", [(0, 0), (1, 0), (2, 2)])
def test_default_reads_small_whole_outline(shape: tuple[int, int], phase: tuple[int, int]) -> None:
    image, expected = _small_scene(shape, phase)
    before = image.copy()
    detector = NapaDetector()
    try:
        observations = detector.detect(image)
        # A reused detector must clear saved proposals when the next image has none.
        assert detector.detect(np.full(shape, 255, np.uint8)) == []
        repeated = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [424]
    assert observations[0].hamming == 0
    error = min(np.linalg.norm(observations[0].corners - np.roll(expected, k, axis=0), axis=1).max() for k in range(4))
    assert error < 0.01
    np.testing.assert_array_equal(repeated[0].corners, observations[0].corners)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("shadows", [False, True])
@pytest.mark.parametrize("tag_id", [424, 42, 137])
def test_small_additions_follow_established_whole_outputs(split: bool, shadows: bool, tag_id: int) -> None:
    cv2 = pytest.importorskip("cv2")
    image, _ = _small_scene((480, 640)) if tag_id == 424 else _partial_small_outline(tag_id)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    # The new small component comes first in raster order; the old 72px reading keeps priority.
    image[240:312, 300:372] = cv2.aruco.generateImageMarker(dictionary, 5, 72)
    # Small additions are what min_tag_px=32 adds below its grid; at the default 22 these tags are ordinary detections.
    detector = NapaDetector(min_tag_px=32, split_touching=split, recover_shadows=shadows)
    capped = NapaDetector(min_tag_px=32, split_touching=split, recover_shadows=shadows, max_detections=1)
    try:
        observations = detector.detect(image)
        with pytest.warns(RuntimeWarning, match="max_detections"):
            limited = capped.detect(image)
    finally:
        detector.close()
        capped.close()
    assert [o.tag_id for o in observations] == [5, tag_id]
    assert [o.tag_id for o in limited] == [5]
    np.testing.assert_array_equal(limited[0].corners, observations[0].corners)


def test_small_additions_require_exact_payload() -> None:
    image, _ = _small_scene((480, 640))
    # Invert one complete payload cell (row3,col1) in the uneven 18px rendering.
    image[97:99, 123:125] = 255 - image[97:99, 123:125]
    fine = NapaDetector(min_tag_px=16)
    ordinary = NapaDetector(min_tag_px=32)  # a small addition there; at 22 it is ordinary and may correct a bit
    try:
        reference = fine.detect(image)
        observations = ordinary.detect(image)
    finally:
        fine.close()
        ordinary.close()
    assert [(o.tag_id, o.hamming) for o in reference] == [(424, 1)]
    assert observations == []
