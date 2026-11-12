from pathlib import Path

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from napa_vision import tags as tag_art
from tests.test_napa_detector import _scene


@pytest.mark.parametrize("minimum", [16, 32])
@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("radial", [0.0, 0.2], ids=["straight", "curved"])
def test_radially_warped_tag_corners(minimum: int, recovery: bool, radial: float) -> None:
    # Own artwork, 4x4 area sampling, integer pixel centers. Truth uses the analytic forward
    # radial map of the square's vertices; it is independent of the detector's curve fit.
    fixture = Path(__file__).parent / "fixtures" / f"radial-tag23-{radial:.1f}.npz"
    with np.load(fixture) as data:
        image, expected, tag_id = data["image"], data["corners"], int(data["tag_id"])
    detector = NapaDetector(min_tag_px=minimum, recover_shadows=recovery, split_touching=recovery)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [observation.tag_id for observation in observations] == [tag_id]
    # Compare physical vertices independently of the artwork's canonical starting corner.
    error = min(
        np.linalg.norm(observations[0].corners - np.roll(expected, shift, axis=0), axis=1).max() for shift in range(4)
    )
    assert error < (0.1 if radial == 0 else 1.0)


def _warped(side: int, angle: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    image, corners = _scene(42, side, angle_deg=angle)
    height, width = image.shape[:2]
    center, scale = np.array([width / 2, height / 2]), max(width, height) / 2
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    grow = 1 + k * (((xs - center[0]) / scale) ** 2 + ((ys - center[1]) / scale) ** 2)
    light = cv2.remap(
        image.astype(np.float32) / 255,
        np.float32(center[0] + (xs - center[0]) * grow),
        np.float32(center[1] + (ys - center[1]) * grow),
        cv2.INTER_CUBIC,
    )
    light = cv2.GaussianBlur(light.astype(np.float64), (0, 0), 0.8)
    encoded = np.clip(np.rint(255 * (0.08 + 0.72 * light) ** (1 / 2.2)), 0, 255).astype(np.uint8)
    expected = []
    for corner in np.asarray(corners, float):
        point = corner.copy()
        for _ in range(30):
            offset = (point - center) / scale
            point = center + (corner - center) / (1 + k * (offset @ offset))
        expected.append(point)
    return encoded, np.array(expected)


# Single angles: at k 0.1 the worst corner over 2-44 degrees averages 0.20 px and peaks at 0.37, so 0.3 leaves room
# for the angle-to-angle scatter of the final fit's end profiles.
@pytest.mark.parametrize(
    "angle,k,limit",
    [(20.0, 0.1, 0.3), (20.0, 0.2, 0.55), (35.0, 0.2, 0.7)],
)
def test_bowed_gamma_tag_corners_follow_their_edges(angle: float, k: float, limit: float) -> None:
    # Bowed enough that a whole-side fit misses the corners, and gamma-encoded so the half level is off the edge.
    image, expected = _warped(240, angle, k)
    detector = NapaDetector(min_tag_px=32)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [observation.tag_id for observation in observations] == [42]
    error = min(
        np.linalg.norm(observations[0].corners - np.roll(expected, shift, axis=0), axis=1).max() for shift in range(4)
    )
    assert error < limit


def _frame_corners(rng: np.random.Generator, side: int, overhang: float, edge: int) -> np.ndarray:
    width, height = 640, 480
    angle = rng.uniform(0.12, 0.55) * rng.choice([-1, 1])
    turn = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * side @ turn
    corners += rng.uniform(-0.05, 0.05, (4, 2)) * side
    low, high = corners.min(0), corners.max(0)
    along = rng.uniform(0.35, 0.65)
    return (
        corners
        + {
            0: [-overhang - low[0], along * height],
            1: [width + overhang - high[0], along * height],
            2: [along * width, -overhang - low[1]],
            3: [along * width, height + overhang - high[1]],
        }[edge]
    )


def _tag_art(cell: int) -> np.ndarray:
    art = cv2.resize(tag_art.tag_image(57).astype(np.uint8), (10 * cell, 10 * cell), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(art, 2 * cell, 2 * cell, 2 * cell, 2 * cell, cv2.BORDER_CONSTANT, value=255)


def _crossing_frame(rng: np.random.Generator, side: int, overhang: float, edge: int) -> tuple[np.ndarray, np.ndarray]:
    width, height, scale, cell = 640, 480, 4, 32
    corners = _frame_corners(rng, side, overhang, edge)
    black = np.float32([[3, 3], [11, 3], [11, 11], [3, 11]]) * cell - 0.5
    warp = cv2.getPerspectiveTransform(black, np.float32(corners * scale - 0.5))
    fine = cv2.warpPerspective(
        _tag_art(cell), warp, (width * scale, height * scale), flags=cv2.INTER_LINEAR, borderValue=90
    )
    image = cv2.GaussianBlur(
        cv2.resize(fine, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 0.7
    )
    return np.clip(image + rng.normal(0, 2.0, (height, width)), 0, 255).astype(np.uint8), corners - 0.5


@pytest.mark.parametrize("side,overhang,limit", [(80, 1.0, 1.5), (80, 2.0, 2.5), (128, 2.0, 1.5)])
def test_straight_tags_cut_by_the_frame_do_not_extrapolate_a_curve(side: int, overhang: float, limit: float) -> None:
    # A side cut by the frame has no samples at its true end; a curve fitted to the rest bends freely past it.
    rng = np.random.default_rng(20261109)
    detector = NapaDetector(max_detections=16)
    worst = 0.0
    try:
        for trial in range(120):
            image, expected = _crossing_frame(rng, side, overhang, trial % 4)
            found = [observation for observation in detector.detect(image) if observation.tag_id == 57]
            if found:
                errors = min(
                    np.linalg.norm(found[0].corners - np.roll(expected, shift, axis=0), axis=1).max()
                    for shift in range(4)
                )
                worst = max(worst, float(errors))
    finally:
        detector.close()
    assert worst < limit


def _bowed_crossing_frame(
    rng: np.random.Generator, side: int, overhang: float, edge: int, k: float
) -> tuple[np.ndarray, np.ndarray]:
    width, height, scale, cell, pad = 640, 480, 3, 32, 240
    corners = _frame_corners(rng, side, overhang, edge)
    center, radius = np.array([width / 2, height / 2]), max(width, height) / 2

    def bend(points: np.ndarray) -> np.ndarray:
        offset = points - center
        return center + offset * (1 + k * ((offset / radius) ** 2).sum(-1, keepdims=True))

    black = np.float32([[3, 3], [11, 3], [11, 11], [3, 11]]) * cell - 0.5
    warp = cv2.getPerspectiveTransform(black, np.float32((bend(corners) + pad) * scale - 0.5))
    flat = cv2.warpPerspective(
        _tag_art(cell),
        warp,
        ((width + 2 * pad) * scale, (height + 2 * pad) * scale),
        flags=cv2.INTER_LINEAR,
        borderValue=90,
    ).astype(np.float32)
    ys, xs = np.mgrid[0 : height * scale, 0 : width * scale]
    source = bend(np.dstack(((xs + 0.5) / scale, (ys + 0.5) / scale)))
    image = cv2.remap(
        flat,
        np.float32((source[..., 0] + pad) * scale - 0.5),
        np.float32((source[..., 1] + pad) * scale - 0.5),
        cv2.INTER_LINEAR,
    )
    image = cv2.GaussianBlur(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), (0, 0), 0.8)
    return np.clip(image + rng.normal(0, 2.0, (height, width)), 0, 255).astype(np.uint8), corners - 0.5


def test_a_side_cut_by_the_frame_takes_its_bow_from_the_others() -> None:
    # A shared radial term carries the bow of the fully seen sides onto the cut one.
    rng = np.random.default_rng(20261109)
    detector = NapaDetector(max_detections=16)
    errors = []
    try:
        for trial in range(16):
            image, expected = _bowed_crossing_frame(rng, 128, 2.0, trial % 4, 0.1)
            found = [observation for observation in detector.detect(image) if observation.tag_id == 57]
            if found:
                errors.append(
                    min(
                        np.linalg.norm(found[0].corners - np.roll(expected, shift, axis=0), axis=1).max()
                        for shift in range(4)
                    )
                )
    finally:
        detector.close()
    assert len(errors) == 16
    assert float(np.median(errors)) < 0.25
