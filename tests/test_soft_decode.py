"""The soft decoder finds tiny tags, keeps the corner order, and stays out of the way when it's off."""

import ctypes

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from napa_vision import tags as tag_art
from napa_vision.napa_detector import _Detection, _library, _Params
from tests.test_napa_detector import _scene

RECOVERY = {"min_tag_px": 16, "split_touching": True, "recover_shadows": True}


def _art(tag_id: int) -> np.ndarray:
    cells = cv2.resize(tag_art.tag_image(tag_id).astype(np.uint8), (320, 320), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(cells, 64, 64, 64, 64, cv2.BORDER_CONSTANT, value=255)


def _tiny_tags(seed: int, quarter_turns: int, side: float = 11) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    width, height, scale = 480, 360, 4
    fine = np.full((height * scale, width * scale), 150, np.uint8)
    truth = []
    for index in range(16):
        tag_id = int(rng.integers(0, 587))
        angle = rng.uniform(-0.25, 0.25) + quarter_turns * np.pi / 2
        turn = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * side @ turn
        corners += [60 + 120 * (index % 4), 45 + 90 * (index // 4)]
        black = np.float32([[96, 96], [352, 96], [352, 352], [96, 352]]) - 0.5
        warp = cv2.getPerspectiveTransform(black, np.float32(corners * scale - 0.5))
        cv2.warpPerspective(
            _art(tag_id),
            warp,
            (width * scale, height * scale),
            dst=fine,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        truth.append((tag_id, corners - 0.5))
    image = cv2.GaussianBlur(
        cv2.resize(fine, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 0.7
    )
    return np.clip(image + rng.normal(0, 1.5, (height, width)), 0, 255).astype(np.uint8), truth


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_soft_tags_report_corners_in_the_native_order(quarter_turns: int) -> None:
    image, truth = _tiny_tags(100 + quarter_turns, quarter_turns)
    found = [o for o in NapaDetector(soft_decode=True, max_detections=64, **RECOVERY).detect(image) if o.soft_decoded]
    assert found
    for observation in found:
        # The artwork's top-left black corner is the third corner reported, as for an ordinary detection.
        expected = [corners for tag_id, corners in truth if tag_id == observation.tag_id]
        assert min(np.linalg.norm(observation.corners - np.roll(c, 2, axis=0), axis=1).max() for c in expected) < 1.2


def test_soft_tags_are_marked_and_are_not_exact_matches() -> None:
    image, _ = _tiny_tags(101, 1)
    found = NapaDetector(soft_decode=True, max_detections=64, **RECOVERY).detect(image)
    soft = [o for o in found if o.soft_decoded]
    assert soft
    assert all(o.hamming is None and o.decision_margin is not None for o in soft)
    assert all(o.hamming is not None for o in found if not o.soft_decoded)


@pytest.mark.parametrize("options", [{}, RECOVERY, {"min_tag_px": 16}, {"min_tag_px": 22, "recover_shadows": False}])
def test_ordinary_results_come_first_and_are_unchanged(options: dict) -> None:
    image, _ = _tiny_tags(103, 3, side=14)
    plain = NapaDetector(max_detections=64, **options).detect(image)
    soft = NapaDetector(soft_decode=True, max_detections=64, **options).detect(image)
    assert not any(o.soft_decoded for o in plain)
    assert not any(o.soft_decoded for o in soft[: len(plain)])
    assert all(o.soft_decoded for o in soft[len(plain) :])
    for a, b in zip(plain, soft, strict=False):
        assert (a.tag_id, a.hamming, a.decision_margin) == (b.tag_id, b.hamming, b.decision_margin)
        assert np.array_equal(a.corners, b.corners)


def test_max_detections_also_limits_soft_tags() -> None:
    image, _ = _tiny_tags(101, 1)
    with pytest.warns(RuntimeWarning, match="max_detections"):
        assert len(NapaDetector(soft_decode=True, max_detections=1, **RECOVERY).detect(image)) == 1


def test_blank_squares_are_not_soft_tags() -> None:
    rng = np.random.default_rng(7)
    image = np.full((360, 480), 170, np.float32)
    for x in range(40, 440, 40):
        for y in range(40, 320, 40):
            cv2.rectangle(image, (x, y), (x + 11, y + 11), 30, -1)
    noisy = np.clip(cv2.GaussianBlur(image, (0, 0), 0.7) + rng.normal(0, 2, image.shape), 0, 255).astype(np.uint8)
    assert NapaDetector(soft_decode=True, max_detections=64, **RECOVERY).detect(noisy) == []


def _random_pattern_squares(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((480, 640), int(rng.integers(150, 230)), np.float32)
    for grid_x in range(40, 600, 55):
        for grid_y in range(40, 440, 55):
            side = int(rng.integers(6, 21))
            x, y = grid_x + rng.integers(-5, 6), grid_y + rng.integers(-5, 6)
            cv2.rectangle(
                image,
                (int(x - side), int(y - side)),
                (int(x + side * 2), int(y + side * 2)),
                int(rng.integers(190, 250)),
                -1,
            )
            square = np.full((80, 80), rng.integers(0, 70), np.float32)
            bits = rng.integers(0, 2, (6, 6))
            for row in range(6):
                for column in range(6):
                    if bits[row, column]:
                        square[(row + 1) * 10 : (row + 2) * 10, (column + 1) * 10 : (column + 2) * 10] = 210
            angle = rng.uniform(-0.5, 0.5)
            turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) * side / 80
            offset = np.array([x, y]) - turn @ np.array([40, 40])
            cv2.warpAffine(
                square,
                np.hstack([turn, offset[:, None]]),
                (640, 480),
                dst=image,
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
    return np.clip(cv2.GaussianBlur(image, (0, 0), 0.7) + rng.normal(0, 2, image.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("seed", [3, 21, 29, 31, 42])
def test_random_patterns_in_dark_squares_are_not_soft_tags(seed: int) -> None:
    # Each of these frames has a square whose picture scores like a code, but whose freely fitted cells contradict it.
    found = NapaDetector(soft_decode=True, max_detections=128, **RECOVERY).detect(_random_pattern_squares(seed))
    assert not [o for o in found if o.soft_decoded]


@pytest.mark.parametrize("family", ["tag16h5", "tag25h9", "tag36h10", "tagStandard41h12"])
def test_other_families_are_refused(family: str) -> None:
    with pytest.raises(ValueError, match="tag36h11"):
        NapaDetector(family=family, soft_decode=True)
    library = _library()
    handle = ctypes.c_void_p(library.napa_create_family(family.encode()))
    results = (_Detection * 4)()
    pixels = np.zeros((64, 64), np.uint8)
    try:
        status = library.napa_detect_soft(
            handle, pixels.ctypes.data, 64, 64, 64, ctypes.byref(_Params(32, 0, 20, 22, 25, 0)), results, 4, 0, 0
        )
    finally:
        library.napa_destroy(handle)
    assert status == -1


def test_soft_decode_must_be_a_bool() -> None:
    with pytest.raises(ValueError, match="soft_decode"):
        NapaDetector(soft_decode=1)


def _blocked_tag(quarter_turns: int) -> tuple[np.ndarray, np.ndarray]:
    image, corners = _scene(42, 96, angle_deg=12.0 + 90.0 * quarter_turns)
    square = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
    to_image = cv2.getPerspectiveTransform(square, np.float32(corners))
    cover = np.float32([[2, 2], [6, 2], [6, 6], [2, 6]]).reshape(-1, 1, 2)
    patch = cv2.perspectiveTransform(cover, to_image).astype(np.int32)
    blocked = image.copy()
    cv2.fillConvexPoly(blocked, patch, 230)
    return blocked, np.asarray(corners, float)


@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_a_flat_patch_over_the_payload_does_not_hide_the_tag(quarter_turns: int) -> None:
    image, expected = _blocked_tag(quarter_turns)
    assert NapaDetector(max_detections=8).detect(image) == []
    found = NapaDetector(soft_decode=True, max_detections=8).detect(image)
    assert [o.tag_id for o in found] == [42]
    assert found[0].soft_decoded and found[0].hamming is None
    assert np.linalg.norm(found[0].corners - expected, axis=1).max() < 0.5
