"""Finer outlines supplement old detections before they merge at coarser levels."""

from pathlib import Path

import numpy as np
import pytest

from napa_vision import NapaDetector
from tests.test_napa_detector import _scene


def _gray_card(
    gray: int,
    side: int = 80,
    *,
    noise: float = 0,
    shape: tuple[int, int] = (480, 640),
    offset: tuple[int, int] = (120, 64),
) -> tuple[np.ndarray, np.ndarray]:
    cv2 = pytest.importorskip("cv2")
    yy, xx = np.indices(shape)
    image = np.where((xx // 8 + yy // 8) & 1, 220, 40).astype(np.uint8)
    pad = side // 8
    card = np.full((side + 2 * pad, side + 2 * pad), gray, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 137, side)
    card[pad : pad + side, pad : pad + side] = np.rint(marker.astype(float) * gray / 255).astype(np.uint8)
    x, y = offset
    image[y : y + len(card), x : x + len(card)] = card
    if noise:
        samples = np.random.default_rng(9237).normal(0, noise, image.shape)
        image = np.clip(np.rint(image.astype(float) + samples), 0, 255).astype(np.uint8)
    corners = (
        np.array(
            [
                [x + pad, y + pad],
                [x + pad + side, y + pad],
                [x + pad + side, y + pad + side],
                [x + pad, y + pad + side],
            ],
            dtype=float,
        )
        - 0.5
    )
    return image, corners


def _max_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return min(float(np.linalg.norm(actual - np.roll(expected, shift, axis=0), axis=1).max()) for shift in range(4))


@pytest.mark.parametrize("gray", [48, 70, 96])
@pytest.mark.parametrize("split", [False, True])
def test_gray_card_outline_survives_coarse_merging(gray: int, split: bool) -> None:
    image, expected = _gray_card(gray)
    before = image.copy()
    detector = NapaDetector(min_tag_px=16, split_touching=split)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [137]
    assert observations[0].hamming == 0
    assert _max_error(observations[0].corners, expected) < 0.01
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("minimum", [8, 16])
@pytest.mark.parametrize("shadows", [False, True])
def test_new_dense_parent_keeps_existing_small_piece(minimum: int, shadows: bool) -> None:
    # 93x93 dense body/parent; its old 12px child must not inherit the new parent's rejection.
    with np.load(Path(__file__).parent / "fixtures" / "dense-parent-small-tag.npz") as data:
        image, expected = data["image"], data["corners"]
    before = image.copy()
    detector = NapaDetector(min_tag_px=minimum, split_touching=True, recover_shadows=shadows)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [137]
    assert observations[0].hamming == 0
    assert _max_error(observations[0].corners, expected) < 0.1
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("split", [False, True])
def test_finer_additions_follow_existing_whole_outputs(split: bool) -> None:
    left, _ = _gray_card(70)
    right, _ = _scene(5, 96)
    image = np.hstack([left, right])
    detector = NapaDetector(min_tag_px=16, split_touching=split)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [5, 137]
    capped = NapaDetector(min_tag_px=16, split_touching=split, max_detections=1)
    try:
        with pytest.warns(RuntimeWarning, match="max_detections"):
            limited = capped.detect(image)
    finally:
        capped.close()
    assert [o.tag_id for o in limited] == [5]
    np.testing.assert_array_equal(limited[0].corners, observations[0].corners)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("shadows", [False, True])
def test_deferred_fitter_retains_stride_two_grid(split: bool, shadows: bool) -> None:
    # Raw noise hides the finest outline; only the averaged stride-two outline survives.
    with np.load(Path(__file__).parent / "fixtures" / "dense-stride-two.npz") as data:
        image, expected = data["image"], data["corners"]
    before = image.copy()
    detector = NapaDetector(min_tag_px=16, split_touching=split, recover_shadows=shadows)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [137]
    assert observations[0].hamming == 0
    assert _max_error(observations[0].corners, expected) < 0.15
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("side,gray,noise", [(72, 70, 0), (73, 70, 0), (144, 48, 8), (146, 48, 8)])
@pytest.mark.parametrize("split", [False, True])
def test_sampling_band_boundary_readings(side: int, gray: int, noise: float, split: bool) -> None:
    # These outlines span 72/73 samples on the raw or averaged grid respectively.
    image, expected = _gray_card(gray, side, noise=noise)
    before = image.copy()
    detector = NapaDetector(min_tag_px=16, split_touching=split)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [137]
    assert observations[0].hamming == 0
    assert _max_error(observations[0].corners, expected) < 0.2
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize(
    "side,gray,shape,offset,minimum",
    [(80, 70, (143, 256), (64, 20), 16), (160, 255, (480, 640), (120, 64), 32)],
    ids=["terminal-grid", "sparser-grid"],
)
def test_terminal_and_sparser_grids_remain_valid(
    side: int, gray: int, shape: tuple[int, int], offset: tuple[int, int], minimum: int
) -> None:
    image, expected = _gray_card(gray, side, shape=shape, offset=offset)
    detector = NapaDetector(min_tag_px=minimum)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [137]
    assert observations[0].hamming == 0
    assert _max_error(observations[0].corners, expected) < 0.01
