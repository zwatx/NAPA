import cv2
import numpy as np
import pytest

from napa_vision.napa_detector import NapaDetector
from tests.test_napa_detector import _scene


def _worst_corner_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return min(float(np.linalg.norm(np.roll(actual, shift, axis=0) - expected, axis=1).max()) for shift in range(4))


def _blurred(side: int, angle: float) -> tuple[np.ndarray, np.ndarray]:
    image, corners = _scene(42, side, angle_deg=angle)
    return cv2.GaussianBlur(image.astype(np.float64) / 255, (0, 0), 0.8), corners


@pytest.mark.parametrize("side,angle", [(96, 0.0), (96, 20.0), (160, 35.0)])
@pytest.mark.parametrize("gamma", [1.0, 2.2])
def test_corners_stay_accurate_whatever_the_camera_encoding(side: int, angle: float, gamma: float) -> None:
    light, expected = _blurred(side, angle)
    # Levels clear of 0 and 255, so this isn't the separately handled clipped case.
    image = np.clip(np.rint(255 * (0.08 + 0.72 * light) ** (1 / gamma)), 0, 255).astype(np.uint8)
    found = NapaDetector(min_tag_px=16, recover_shadows=False).detect(image)
    assert [o.tag_id for o in found] == [42]
    assert _worst_corner_error(found[0].corners, expected) < 0.2


def _outward_shift(actual: np.ndarray, expected: np.ndarray) -> float:
    best = min((np.roll(actual, shift, axis=0) for shift in range(4)), key=lambda q: float(np.abs(q - expected).sum()))
    center = expected.mean(axis=0)
    outward = (expected - center) / np.linalg.norm(expected - center, axis=1, keepdims=True)
    return float(((best - expected) * outward).sum(axis=1).mean())


@pytest.mark.parametrize("side,angle,sigma", [(64, 15.0, 2.2), (40, 20.0, 1.4), (80, 10.0, 2.6)])
def test_blur_under_a_gamma_curve_does_not_shrink_the_tag(side: int, angle: float, sigma: float) -> None:
    # Blur in linear light, then encode: the fitted edges drift inward by a few percent, a direct range error.
    image, expected = _scene(42, side, angle_deg=angle)
    light = cv2.GaussianBlur(image.astype(np.float64) / 255, (0, 0), sigma)
    encoded = np.clip(np.rint(255 * (0.08 + 0.72 * light) ** (1 / 2.2)), 0, 255).astype(np.uint8)
    found = NapaDetector(min_tag_px=16, recover_shadows=False).detect(encoded)
    assert [o.tag_id for o in found] == [42]
    assert abs(_outward_shift(found[0].corners, expected)) < 0.005 * side


@pytest.mark.parametrize("side,angle,blur", [(64, 15.0, 0.40), (48, 25.0, 0.42), (96, 20.0, 0.40)])
def test_blur_on_a_straight_tone_curve_does_not_grow_the_tag(side: int, angle: float, blur: float) -> None:
    image, expected = _scene(42, side, angle_deg=angle)
    light = cv2.GaussianBlur(image.astype(np.float64) / 255, (0, 0), blur * side / 8)
    encoded = np.clip(np.rint(255 * (0.08 + 0.72 * light)), 0, 255).astype(np.uint8)
    found = NapaDetector(min_tag_px=16, recover_shadows=False).detect(encoded)
    assert [o.tag_id for o in found] == [42]
    assert abs(_outward_shift(found[0].corners, expected)) < 0.005 * side


@pytest.mark.parametrize(
    "side,direction,streak,angle", [(64, 20.0, 0.9, 25.0), (80, 20.0, 0.9, 25.0), (96, 20.0, 0.6, 25.0)]
)
def test_motion_blur_under_a_gamma_curve_keeps_corners_close(
    side: int, direction: float, streak: float, angle: float
) -> None:
    # Motion widens the edges of only some sides, so the sides disagree about the border's width for a reason
    # that has nothing to do with tone; a corner refit that trusts that disagreement pulls corners off.
    image, expected = _scene(42, side, angle_deg=angle)
    half = streak * side / 8
    kernel = np.zeros((41 * 8, 41 * 8), np.float32)
    dx, dy = np.cos(np.deg2rad(direction)) * half * 4, np.sin(np.deg2rad(direction)) * half * 4
    cv2.line(kernel, (int(164 - dx), int(164 - dy)), (int(164 + dx), int(164 + dy)), 1.0, 1, cv2.LINE_AA)
    kernel = cv2.resize(kernel, (41, 41), interpolation=cv2.INTER_AREA)
    light = cv2.GaussianBlur(image.astype(np.float32) / 255, (0, 0), 0.5)
    light = cv2.filter2D(light, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REPLICATE)
    encoded = np.clip(np.rint(255 * (0.08 + 0.72 * light) ** (1 / 2.2)), 0, 255).astype(np.uint8)
    found = NapaDetector(min_tag_px=16, recover_shadows=False).detect(encoded)
    assert [o.tag_id for o in found] == [42]
    assert _worst_corner_error(found[0].corners, expected) < 1.6


def _streaked(side: int, direction: float, streak: float, angle: float) -> tuple[np.ndarray, np.ndarray]:
    image, expected = _scene(42, side, angle_deg=angle)
    half = streak * side / 8
    kernel = np.zeros((41 * 8, 41 * 8), np.float32)
    dx, dy = np.cos(np.deg2rad(direction)) * half * 4, np.sin(np.deg2rad(direction)) * half * 4
    cv2.line(kernel, (int(164 - dx), int(164 - dy)), (int(164 + dx), int(164 + dy)), 1.0, 1, cv2.LINE_AA)
    kernel = cv2.resize(kernel, (41, 41), interpolation=cv2.INTER_AREA)
    light = cv2.filter2D(
        cv2.GaussianBlur(image.astype(np.float32) / 255, (0, 0), 0.5),
        -1,
        kernel / kernel.sum(),
        borderType=cv2.BORDER_REPLICATE,
    )
    return np.clip(np.rint(255 * (0.08 + 0.72 * light) ** (1 / 2.2)), 0, 255).astype(np.uint8), expected


def test_align_corners_is_off_unless_asked_for() -> None:
    image, _ = _streaked(80, 20.0, 0.9, 25.0)
    a = NapaDetector(min_tag_px=16, recover_shadows=False).detect(image)
    b = NapaDetector(min_tag_px=16, recover_shadows=False, align_corners=False).detect(image)
    assert np.array_equal(a[0].corners, b[0].corners)


@pytest.mark.parametrize("side,direction,streak,angle", [(80, 20.0, 0.9, 25.0), (64, 20.0, 0.9, 25.0)])
def test_align_corners_follows_motion_blur(side: int, direction: float, streak: float, angle: float) -> None:
    image, expected = _streaked(side, direction, streak, angle)
    plain = NapaDetector(min_tag_px=16, recover_shadows=False).detect(image)
    aligned = NapaDetector(min_tag_px=16, recover_shadows=False, align_corners=True).detect(image)
    assert [o.tag_id for o in aligned] == [42]
    assert _worst_corner_error(aligned[0].corners, expected) < _worst_corner_error(plain[0].corners, expected)


@pytest.mark.parametrize("family", ["tag16h5", "tag25h9", "tag36h10", "tagStandard41h12"])
def test_align_corners_refuses_other_families(family: str) -> None:
    with pytest.raises(ValueError, match="tag36h11"):
        NapaDetector(family=family, align_corners=True)


def test_align_corners_must_be_a_bool() -> None:
    with pytest.raises(ValueError, match="align_corners"):
        NapaDetector(align_corners=1)


@pytest.mark.parametrize("side,angle,blur,limit", [(28, 0.7, 0.4, 0.56), (30, 0.7, 0.4, 0.57), (40, 0.7, 0.6, 0.42)])
def test_sides_nearly_along_the_pixel_grid_stay_accurate(side: int, angle: float, blur: float, limit: float) -> None:
    # A side that hugs the grid keeps one pixel phase along its whole length, so phase errors don't average out.
    image, expected = _scene(42, side, angle_deg=angle)
    found = NapaDetector(min_tag_px=16, recover_shadows=False).detect(cv2.GaussianBlur(image, (0, 0), blur))
    assert [o.tag_id for o in found] == [42]
    assert _worst_corner_error(found[0].corners, expected) < limit
