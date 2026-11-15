import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector, estimate_camera_gamma
from tests.test_corner_bias_correction import _worst_corner_error
from tests.test_napa_detector import _scene


def _encode(light: np.ndarray, gamma: float, black: float = 0.08, span: float = 0.72) -> np.ndarray:
    return np.clip(np.rint(255 * (black + span * light) ** (1 / gamma)), 0, 255).astype(np.uint8)


def _streaked(side: int, direction: float, streak: float, angle: float, gamma: float, **tone: float):
    # A straight motion streak in linear light, `streak` cells long either way, then the camera's curve.
    image, expected = _scene(42, side, angle_deg=angle)
    half = streak * side / 8
    kernel = np.zeros((328, 328), np.float32)
    dx, dy = np.cos(np.deg2rad(direction)) * half * 4, np.sin(np.deg2rad(direction)) * half * 4
    cv2.line(kernel, (int(164 - dx), int(164 - dy)), (int(164 + dx), int(164 + dy)), 1.0, 1, cv2.LINE_AA)
    kernel = cv2.resize(kernel, (41, 41), interpolation=cv2.INTER_AREA)
    sharp = cv2.GaussianBlur(image.astype(np.float32) / 255, (0, 0), 0.5)
    light = cv2.filter2D(sharp, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REPLICATE)
    return _encode(light, gamma, **tone), expected


def _blurred(rng: np.random.Generator, gamma: float) -> np.ndarray:
    image, _ = _scene(42, int(rng.choice((40, 56, 72, 96, 128))), angle_deg=float(rng.uniform(5, 40)))
    return _encode(cv2.GaussianBlur(image.astype(np.float64) / 255, (0, 0), float(rng.choice((0.6, 0.9, 1.3)))), gamma)


def _detect(image: np.ndarray, **options):
    detector = NapaDetector(min_tag_px=16, **options)
    try:
        return detector.detect(image)
    finally:
        detector.close()


@pytest.mark.parametrize("value", [True, 0.9, 4.5, float("nan"), "2.2"])
def test_rejects_invalid_gammas(value) -> None:
    with pytest.raises(ValueError, match="camera_gamma"):
        NapaDetector(camera_gamma=value)


def test_is_tag36h11_only() -> None:
    with pytest.raises(ValueError, match="tag36h11"):
        NapaDetector(family="tag16h5", camera_gamma=2.2)


@pytest.mark.parametrize("gamma", [None, 1.0])
def test_unset_or_linear_leaves_results_unchanged(gamma) -> None:
    image, _ = _streaked(80, 20, 0.9, 25, 2.2)
    plain, other = _detect(image), _detect(image, camera_gamma=gamma)
    assert [o.tag_id for o in other] == [o.tag_id for o in plain] == [42]
    np.testing.assert_array_equal(other[0].corners, plain[0].corners)


@pytest.mark.parametrize("side,direction,streak,angle", [(64, 20, 0.9, 25), (80, 20, 0.9, 25), (80, 60, 0.7, 10)])
def test_matching_gamma_removes_motion_blur_error(side: int, direction: float, streak: float, angle: float) -> None:
    # Motion blur on a gamma-encoded camera shifts corners along the streak; in linear light the fit lands on the edge.
    image, expected = _streaked(side, direction, streak, angle, 2.2)
    plain, linear = _detect(image), _detect(image, camera_gamma=2.2)
    assert [o.tag_id for o in linear] == [42]
    before, after = _worst_corner_error(plain[0].corners, expected), _worst_corner_error(linear[0].corners, expected)
    assert after < 0.15 and after < before / 3


def test_dark_tags_keep_raw_refinement() -> None:
    # Grey levels 20 to 67 hold 1 to 18 once the curve is undone, too little to fit, so it's refined as if unset.
    image, _ = _streaked(80, 20, 0.6, 25, 2.2, black=0.0037, span=0.05)
    plain, linear = _detect(image), _detect(image, camera_gamma=2.2)
    assert [o.tag_id for o in plain] == [42]
    np.testing.assert_array_equal(linear[0].corners, plain[0].corners)


@pytest.mark.parametrize("gamma", [1.0, 2.2])
def test_estimates_the_gamma_of_rendered_frames(gamma: float) -> None:
    rng = np.random.default_rng(20261110)
    estimate = estimate_camera_gamma((_blurred(rng, gamma) for _ in range(40)), min_tag_px=16)
    assert estimate.frames >= 30
    assert type(estimate.gamma) is float
    assert estimate.gamma == pytest.approx(gamma, abs=0.08)
    assert estimate.low <= estimate.gamma <= estimate.high


def test_too_few_frames_give_no_estimate() -> None:
    rng = np.random.default_rng(1)
    estimate = estimate_camera_gamma([_blurred(rng, 2.2) for _ in range(5)], min_tag_px=16)
    assert estimate.gamma is None and estimate.low is None and 0 < estimate.frames < 30
    with pytest.raises(ValueError, match="min_frames"):
        estimate_camera_gamma([], min_frames=0)
