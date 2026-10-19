"""Napa versus standard AprilTag on a large, seeded corpus of distinct synthetic frames.

Every frame is rendered independently: random tag IDs, positions, sizes, arbitrary in-plane
angles, 3D tilt, backgrounds, and camera degradations. Each tag is rendered with exact sub-pixel
ground truth (4x supersampled, area-averaged like a sensor), so the benchmark scores which tags
were found, which IDs were invented, and how far each corner landed from the truth.

Timing is per frame and interleaved: every backend runs on the same frame, in a shuffled order,
``--repetitions`` times. Rendering, scoring, and report writing are outside the timed calls.
Frames are generated one at a time, so the corpus size is bounded by run time, not memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np

from napa_vision.napa_detector import NapaDetector, native_available

REPORT_SCHEMA_VERSION = 4
DEFAULT_RESOLUTIONS = ((640, 480), (1280, 720), (1920, 1080))
DEFAULT_FRAMES_PER_SCENARIO = 50
DEFAULT_OCCLUSION_CASES = 24
TAG_COUNT = 587
SUPERSAMPLE = 4
OCCLUSION_FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
OCCLUSION_REGIONS = ("payload-center", "border-strip", "corner")
OCCLUSION_VALUES = (255, 127)
DEFAULT_RANGE_CASES = 50
RANGE_SIDES_PX = (12, 14, 16, 20, 24, 28, 32, 40, 48, 64, 96)

Corners = np.ndarray  # (4, 2) float64, black-square corners in OpenCV order, pixel-centre coordinates
Detection = tuple[int, Corners]
DetectFn = Callable[[np.ndarray], list[Detection]]


# ------------------------------------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class TagTruth:
    tag_id: int
    corners: Corners

    @property
    def side_px(self) -> float:
        """Mean edge length of the black square, in pixels."""

        return float(np.mean(np.linalg.norm(self.corners - np.roll(self.corners, -1, axis=0), axis=1)))


@dataclass(slots=True)
class Frame:
    scenario: str
    index: int
    resolution: tuple[int, int]
    image: np.ndarray
    tags: list[TagTruth] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.scenario}-{self.resolution[0]}x{self.resolution[1]}-{self.index:03d}"


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    description: str
    render: Callable[[np.random.Generator, int, int], tuple[np.ndarray, list[TagTruth], dict[str, object]]]


@dataclass(frozen=True, slots=True)
class Backend:
    name: str
    factory: Callable[[], DetectFn]
    config: dict[str, object]


@dataclass(frozen=True, slots=True)
class FrameScore:
    true_positive: int
    false_positive: int
    false_negative: int
    corner_errors: tuple[float, ...]


# -------------------------------------------------------------------------------------- rendering


@dataclass(frozen=True, slots=True)
class Occlusion:
    """An axis-aligned patch in tag cell units, where the black square spans 0.5 to 8.5."""

    x0: float
    y0: float
    x1: float
    y1: float
    value: int

    @property
    def fraction(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0) / 64.0


_PATTERNS: dict[int, np.ndarray] = {}


def tag_pattern(tag_id: int) -> np.ndarray:
    """Return the 10x10 cell pattern: one white quiet-zone cell around the 8x8 black-bordered tag."""

    pattern = _PATTERNS.get(tag_id)
    if pattern is None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        pattern = np.full((10, 10), 255, np.uint8)
        pattern[1:9, 1:9] = cv2.aruco.generateImageMarker(dictionary, tag_id, 8, borderBits=1)
        _PATTERNS[tag_id] = pattern
    return pattern


def render_tag(
    image: np.ndarray,
    tag_id: int,
    corners: Corners,
    *,
    black: float = 30.0,
    white: float = 235.0,
    occlusion: Occlusion | None = None,
) -> None:
    """Composite one tag (with its white quiet zone) into a float32 image, in place.

    ``corners`` are the black square's corners in pixel-centre coordinates. Each output pixel is
    the mean of a 4x4 grid of exact point samples, so edges are integrated like a real sensor.
    """

    height, width = image.shape
    source = np.array([[0.5, 0.5], [8.5, 0.5], [8.5, 8.5], [0.5, 8.5]], np.float64)
    to_image = cv2.getPerspectiveTransform(source.astype(np.float32), np.asarray(corners, np.float32)).astype(
        np.float64
    )
    outer = cv2.perspectiveTransform(
        np.array([[[-0.5, -0.5], [9.5, -0.5], [9.5, 9.5], [-0.5, 9.5]]], np.float64), to_image
    )[0]
    x0 = max(0, int(math.floor(outer[:, 0].min())) - 1)
    y0 = max(0, int(math.floor(outer[:, 1].min())) - 1)
    x1 = min(width, int(math.ceil(outer[:, 0].max())) + 2)
    y1 = min(height, int(math.ceil(outer[:, 1].max())) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    step = 1.0 / SUPERSAMPLE
    xs = x0 - 0.5 + (np.arange((x1 - x0) * SUPERSAMPLE) + 0.5) * step
    ys = y0 - 0.5 + (np.arange((y1 - y0) * SUPERSAMPLE) + 0.5) * step
    grid_x, grid_y = np.meshgrid(xs, ys)
    to_cells = np.linalg.inv(to_image)
    w = to_cells[2, 0] * grid_x + to_cells[2, 1] * grid_y + to_cells[2, 2]
    u = (to_cells[0, 0] * grid_x + to_cells[0, 1] * grid_y + to_cells[0, 2]) / w
    v = (to_cells[1, 0] * grid_x + to_cells[1, 1] * grid_y + to_cells[1, 2]) / w
    column = np.floor(u + 0.5).astype(np.int64)
    row = np.floor(v + 0.5).astype(np.int64)
    inside = (column >= 0) & (column < 10) & (row >= 0) & (row < 10)
    pattern = tag_pattern(tag_id)
    values = np.where(inside, pattern[np.clip(row, 0, 9), np.clip(column, 0, 9)], 255).astype(np.float32)
    values = black + (white - black) * values / 255.0
    if occlusion is not None:
        covered = inside & (u >= occlusion.x0) & (u < occlusion.x1) & (v >= occlusion.y0) & (v < occlusion.y1)
        values[covered] = occlusion.value
    shape = (y1 - y0, SUPERSAMPLE, x1 - x0, SUPERSAMPLE)
    coverage = inside.astype(np.float32).reshape(shape).mean(axis=(1, 3))
    tag = (values * inside).reshape(shape).sum(axis=(1, 3)) / (SUPERSAMPLE * SUPERSAMPLE)
    patch = image[y0:y1, x0:x1]
    patch *= 1.0 - coverage
    patch += tag


def tag_pose(
    rng: np.random.Generator,
    side: float,
    center: tuple[float, float],
    *,
    max_tilt_deg: float = 20.0,
    min_tilt_deg: float = 0.0,
) -> Corners:
    """Corners of a square of roughly ``side`` pixels at any in-plane angle, tilted in 3D."""

    roll = rng.uniform(0.0, 2.0 * math.pi)
    tilt = math.radians(rng.uniform(min_tilt_deg, max_tilt_deg))
    axis = rng.uniform(0.0, 2.0 * math.pi)
    square = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    roll_matrix = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    k = np.array([math.cos(axis), math.sin(axis), 0.0])
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    tilt_matrix = np.eye(3) + math.sin(tilt) * kx + (1 - math.cos(tilt)) * kx @ kx
    points = square @ roll_matrix.T @ tilt_matrix.T
    distance = 2.5  # tag widths from the camera: close enough for visible perspective
    projected = points[:, :2] / (distance + points[:, 2:3]) * distance
    corners = projected * side + np.asarray(center)
    return corners.astype(np.float64)


def _outer_radius(corners: Corners) -> float:
    center = corners.mean(axis=0)
    return float(np.linalg.norm(corners - center, axis=1).max()) * 1.25 + 2.0


def place_tags(
    rng: np.random.Generator,
    width: int,
    height: int,
    count: int,
    side_range: tuple[float, float],
    **pose: float,
) -> list[TagTruth]:
    """Place up to ``count`` non-overlapping tags with distinct random IDs, fully inside the frame."""

    tags: list[TagTruth] = []
    ids = rng.choice(TAG_COUNT, size=count, replace=False)
    for tag_id in ids:
        for _attempt in range(60):
            side = float(rng.uniform(*side_range))
            reach = side * 0.9 + 4
            if 2 * reach >= min(width, height):
                side = (min(width, height) / 2 - 4) / 0.9 - 1
                reach = side * 0.9 + 4
            center = (float(rng.uniform(reach, width - reach)), float(rng.uniform(reach, height - reach)))
            corners = tag_pose(rng, side, center, **pose)
            radius = _outer_radius(corners)
            mean = corners.mean(axis=0)
            if (
                mean[0] - radius < 0
                or mean[1] - radius < 0
                or mean[0] + radius > width - 1
                or mean[1] + radius > height - 1
            ):
                continue
            if all(
                np.linalg.norm(mean - other.corners.mean(axis=0)) > radius + _outer_radius(other.corners)
                for other in tags
            ):
                tags.append(TagTruth(int(tag_id), corners))
                break
    return tags


# ------------------------------------------------------------------------------------ backgrounds


def _uniform(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    return np.full((height, width), rng.uniform(150, 245), np.float32)


def _smooth_texture(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Low-frequency 'natural' texture: blurred noise at a few scales."""

    image = np.zeros((height, width), np.float32)
    for scale, weight in ((64, 40.0), (16, 18.0), (4, 8.0)):
        small = rng.normal(0.0, 1.0, (height // scale + 2, width // scale + 2)).astype(np.float32)
        image += weight * cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return image + rng.uniform(110, 170)


def _shapes(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    # OpenCV draws text and shapes only on 8-bit images.
    image = np.full((height, width), int(rng.uniform(180, 240)), np.uint8)
    for _ in range(int(rng.integers(20, 60))):
        x0, x1 = sorted(rng.integers(0, width, size=2).tolist())
        y0, y1 = sorted(rng.integers(0, height, size=2).tolist())
        value = int(rng.uniform(20, 220))
        thickness = int(rng.choice((1, 2, 3, -1)))
        shape = int(rng.integers(0, 3))
        if shape == 0:
            cv2.rectangle(image, (x0, y0), (x1, y1), value, thickness=thickness)
        elif shape == 1:
            cv2.line(image, (x0, y0), (x1, y1), value, thickness=max(1, thickness))
        else:
            radius = int(rng.integers(5, max(6, min(width, height) // 6)))
            cv2.circle(image, (x0, y0), radius, value, thickness=thickness)
    return image.astype(np.float32)


def _text(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    image = np.full((height, width), int(rng.uniform(200, 240)), np.uint8)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@%&"
    for _ in range(max(10, (width * height) // 12_000)):
        text = "".join(rng.choice(list(alphabet), size=int(rng.integers(3, 12))))
        origin = (int(rng.integers(-40, width)), int(rng.integers(10, height + 20)))
        font = int(rng.choice((cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX)))
        scale = float(rng.uniform(0.5, 2.5))
        cv2.putText(image, text, origin, font, scale, int(rng.uniform(0, 100)), int(rng.integers(1, 4)))
    return image.astype(np.float32)


def _checkers(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Checkerboards and grids: high-contrast square structure that is not a tag."""

    image = np.full((height, width), rng.uniform(170, 240), np.float32)
    for _ in range(int(rng.integers(2, 6))):
        cell = int(rng.integers(6, 40))
        cols, rows = int(rng.integers(3, 12)), int(rng.integers(3, 12))
        board = ((np.indices((rows, cols)).sum(axis=0) % 2) * 200 + 25).astype(np.float32)
        board = cv2.resize(board, (cols * cell, rows * cell), interpolation=cv2.INTER_NEAREST)
        angle = float(rng.uniform(0, 360))
        center = (float(rng.uniform(0, width)), float(rng.uniform(0, height)))
        matrix = cv2.getRotationMatrix2D((cols * cell / 2, rows * cell / 2), angle, 1.0)
        matrix[:, 2] += np.array(center) - np.array([cols * cell / 2, rows * cell / 2])
        mask = cv2.warpAffine(np.ones_like(board), matrix, (width, height), flags=cv2.INTER_LINEAR)
        warped = cv2.warpAffine(board, matrix, (width, height), flags=cv2.INTER_LINEAR)
        image = image * (1 - mask) + warped * mask
    return image


def _decoys(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Black-bordered squares with random payloads: tag-shaped, but (almost always) not tag36h11 codes."""

    image = _uniform(rng, width, height)
    for _ in range(int(rng.integers(4, 12))):
        cells = np.full((10, 10), 255, np.uint8)
        cells[1:9, 1:9] = 0
        cells[2:8, 2:8] = rng.integers(0, 2, (6, 6)) * 255
        side = float(rng.uniform(40, min(160, min(width, height) / 3)))
        center = (float(rng.uniform(side, width - side)), float(rng.uniform(side, height - side)))
        _render_cells(image, cells, tag_pose(rng, side, center))
    return image


def _render_cells(image: np.ndarray, cells: np.ndarray, corners: Corners) -> None:
    """Render an arbitrary 10x10 cell pattern, reusing render_tag's sampler via a temporary pattern."""

    _PATTERNS[-1] = cells
    try:
        render_tag(image, -1, corners)
    finally:
        _PATTERNS.pop(-1, None)


BACKGROUNDS: dict[str, Callable[[np.random.Generator, int, int], np.ndarray]] = {
    "uniform": _uniform,
    "texture": _smooth_texture,
    "shapes": _shapes,
    "text": _text,
    "checkers": _checkers,
}


def _random_background(rng: np.random.Generator, width: int, height: int) -> tuple[np.ndarray, str]:
    name = str(rng.choice(list(BACKGROUNDS)))
    return BACKGROUNDS[name](rng, width, height), name


# ----------------------------------------------------------------------------------- degradations


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def motion_blur(image: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    return cv2.filter2D(image, -1, kernel / kernel.sum())


def lighting(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """A linear gradient in a random direction plus radial vignetting."""

    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = rng.uniform(0, 2 * math.pi)
    ramp = (xx - width / 2) * math.cos(angle) + (yy - height / 2) * math.sin(angle)
    ramp = ramp / max(1.0, float(np.abs(ramp).max()))
    low = rng.uniform(0.3, 0.6)
    gain = low + (1 - low) * (ramp + 1) / 2
    radius = np.hypot((xx - width / 2) / (width / 2), (yy - height / 2) / (height / 2))
    gain *= 1.0 - rng.uniform(0.2, 0.45) * np.clip(radius, 0, 1.5) ** 2
    return image * gain


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", _to_uint8(image), (cv2.IMWRITE_JPEG_QUALITY, quality))
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE).astype(np.float32)


def add_noise(rng: np.random.Generator, image: np.ndarray, sigma: float) -> np.ndarray:
    return image + rng.normal(0.0, sigma, image.shape).astype(np.float32)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


# -------------------------------------------------------------------------------------- scenarios


def _scale(width: int, height: int) -> float:
    """Tag sizes scale gently with resolution, so larger frames don't only add empty pixels."""

    return min(width, height) / 480.0


def _tags_scene(
    rng: np.random.Generator,
    width: int,
    height: int,
    *,
    count: tuple[int, int] = (1, 3),
    side: tuple[float, float] | None = None,
    background: str | None = None,
    black: tuple[float, float] = (15, 60),
    white: tuple[float, float] = (200, 250),
    **pose: float,
) -> tuple[np.ndarray, list[TagTruth], dict[str, object]]:
    scale = _scale(width, height)
    side = side or (48 * scale, 170 * scale)
    if background is None:
        image, background = _random_background(rng, width, height)
    else:
        image = BACKGROUNDS[background](rng, width, height)
    tags = place_tags(rng, width, height, int(rng.integers(count[0], count[1] + 1)), side, **pose)
    for tag in tags:
        render_tag(image, tag.tag_id, tag.corners, black=rng.uniform(*black), white=rng.uniform(*white))
    return image, tags, {"background": background}


def _degraded(
    transform: Callable[[np.random.Generator, np.ndarray, dict[str, object]], np.ndarray], **scene: object
) -> Callable[[np.random.Generator, int, int], tuple[np.ndarray, list[TagTruth], dict[str, object]]]:
    def render(rng: np.random.Generator, width: int, height: int):
        image, tags, details = _tags_scene(rng, width, height, **scene)
        return transform(rng, image, details), tags, details

    return render


def _negative(
    background: str | None,
) -> Callable[[np.random.Generator, int, int], tuple[np.ndarray, list[TagTruth], dict[str, object]]]:
    def render(rng: np.random.Generator, width: int, height: int):
        if background == "decoys":
            image, name = _decoys(rng, width, height), "decoys"
        elif background is None:
            image, name = _random_background(rng, width, height)
            while name == "uniform":
                image, name = _random_background(rng, width, height)
        else:
            image, name = BACKGROUNDS[background](rng, width, height), background
        return add_noise(rng, image, rng.uniform(0.5, 3.0)), [], {"background": name}

    return render


def _with(key: str, value_fn: Callable[[np.random.Generator], object], apply):
    def transform(rng: np.random.Generator, image: np.ndarray, details: dict[str, object]) -> np.ndarray:
        value = value_fn(rng)
        details[key] = value
        return apply(rng, image, value)

    return transform


def _mild_noise(rng: np.random.Generator, image: np.ndarray, details: dict[str, object]) -> np.ndarray:
    return add_noise(rng, image, 1.5)


def _edge_scene(rng: np.random.Generator, width: int, height: int):
    """Tags whose quiet zone touches the frame edge."""

    image, background = _random_background(rng, width, height)
    scale = _scale(width, height)
    tags: list[TagTruth] = []
    for edge in rng.permutation(4)[: int(rng.integers(1, 4))]:
        side = float(rng.uniform(40 * scale, 120 * scale))
        corners = tag_pose(rng, side, (0.0, 0.0), max_tilt_deg=15)
        radius = _outer_radius(corners) - 2.0
        along = float(rng.uniform(radius, (width if edge in (0, 2) else height) - radius))
        center = {
            0: (along, radius),
            1: (width - 1 - radius, along),
            2: (along, height - 1 - radius),
            3: (radius, along),
        }[int(edge)]
        corners = corners + np.asarray(center)
        mean = corners.mean(axis=0)
        if any(np.linalg.norm(mean - t.corners.mean(axis=0)) < radius + _outer_radius(t.corners) for t in tags):
            continue
        tag_id = int(rng.integers(0, TAG_COUNT))
        if any(t.tag_id == tag_id for t in tags):
            continue
        render_tag(image, tag_id, corners, black=rng.uniform(15, 60), white=rng.uniform(200, 250))
        tags.append(TagTruth(tag_id, corners))
    return add_noise(rng, image, 1.5), tags, {"background": background}


def _large_tags(rng: np.random.Generator, width: int, height: int):
    lo, hi = 0.45 * min(width, height) / 1.25, 0.8 * min(width, height) / 1.25
    image, tags, details = _tags_scene(rng, width, height, count=(1, 1), side=(lo, hi), max_tilt_deg=20)
    return add_noise(rng, image, 1.5), tags, details


def _busy(rng: np.random.Generator, width: int, height: int):
    background = str(rng.choice(("text", "shapes", "checkers")))
    image, tags, details = _tags_scene(rng, width, height, background=background)
    return add_noise(rng, image, 1.5), tags, details


def _dark_background(rng: np.random.Generator, width: int, height: int):
    """Tags whose one-cell white margin is all that separates the black border from a dark background."""

    name = str(rng.choice(("uniform", "texture", "shapes")))
    image = BACKGROUNDS[name](rng, width, height)
    image *= float(rng.uniform(15, 120)) / max(1.0, float(image.mean()))
    tags = place_tags(rng, width, height, int(rng.integers(1, 4)), (36.0, 140.0 * _scale(width, height)))
    for tag in tags:
        render_tag(image, tag.tag_id, tag.corners, black=rng.uniform(5, 40), white=rng.uniform(170, 250))
    return add_noise(rng, image, 1.5), tags, {"background": name}


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("empty", "Uniform frame at a random brightness, light noise", _negative("uniform")),
    Scenario("clutter-only", "Shapes, text, checkerboards, or natural texture with no tag", _negative(None)),
    Scenario("decoys", "Black-bordered squares with random payloads; none is a valid code", _negative("decoys")),
    Scenario(
        "one-tag",
        "One tag, any angle, mild tilt, uniform background",
        _degraded(_mild_noise, count=(1, 1), background="uniform"),
    ),
    Scenario(
        "many-tags",
        "2-12 tags of mixed sizes on mixed backgrounds",
        _degraded(_mild_noise, count=(2, 12)),
    ),
    Scenario(
        "small-tags",
        "1-4 tags with a 32-53 px black square",
        _degraded(_mild_noise, count=(1, 4), side=(32.0, 53.0), max_tilt_deg=10),
    ),
    Scenario("large-tags", "One tag filling 45-80% of the frame height", _large_tags),
    Scenario(
        "steep-angle",
        "1-3 tags tilted 40-65 degrees away from the camera",
        _degraded(_mild_noise, min_tilt_deg=40.0, max_tilt_deg=65.0),
    ),
    Scenario(
        "blur",
        "Gaussian defocus, sigma 0.8-2.0 px",
        _degraded(
            _with(
                "blur_sigma",
                lambda rng: round(float(rng.uniform(0.8, 2.0)), 2),
                lambda rng, img, s: add_noise(rng, gaussian_blur(img, s), 1.5),
            )
        ),
    ),
    Scenario(
        "motion-blur",
        "Linear motion blur, 3-11 px (odd lengths) in any direction",
        _degraded(
            _with(
                "motion_px",
                lambda rng: int(rng.integers(1, 6)) * 2 + 1,  # odd: centred kernel, no shift
                lambda rng, img, n: add_noise(rng, motion_blur(img, n, float(rng.uniform(0, 180))), 1.5),
            )
        ),
    ),
    Scenario(
        "noise",
        "Gaussian sensor noise, sigma 4-14 grey levels",
        _degraded(
            _with(
                "noise_sigma",
                lambda rng: round(float(rng.uniform(4, 14)), 1),
                lambda rng, img, s: add_noise(rng, img, s),
            )
        ),
    ),
    Scenario(
        "uneven-light",
        "Lighting gradient (30-60% at the dark end) plus vignetting",
        _degraded(
            _with(
                "lighting",
                lambda rng: "gradient+vignette",
                lambda rng, img, _v: add_noise(rng, lighting(rng, img), 1.5),
            )
        ),
    ),
    Scenario(
        "low-contrast",
        "Black and white only 40-90 grey levels apart",
        _degraded(_mild_noise, background="uniform", black=(90, 120), white=(160, 190)),
    ),
    Scenario(
        "underexposed",
        "Dark frame: everything scaled to 15-35% brightness, then noise",
        _degraded(
            _with(
                "exposure",
                lambda rng: round(float(rng.uniform(0.15, 0.35)), 2),
                lambda rng, img, gain: add_noise(rng, img * gain, 2.0),
            )
        ),
    ),
    Scenario(
        "overexposed",
        "Bright frame: gain 1.3-1.8 with clipping",
        _degraded(
            _with(
                "exposure",
                lambda rng: round(float(rng.uniform(1.3, 1.8)), 2),
                lambda rng, img, gain: add_noise(rng, img * gain + 20, 1.5),
            )
        ),
    ),
    Scenario(
        "jpeg",
        "JPEG compression at quality 25-70",
        _degraded(
            _with(
                "jpeg_quality",
                lambda rng: int(rng.integers(25, 71)),
                lambda rng, img, q: jpeg(add_noise(rng, img, 1.5), q),
            )
        ),
    ),
    Scenario("busy-background", "1-3 tags on text, shapes, or checkerboards", _busy),
    Scenario("frame-edge", "1-3 tags whose quiet zone touches the frame edge", _edge_scene),
    # Every other scene adds at least light sensor noise; this one shows the noise-free baseline.
    Scenario(
        "noise-free",
        "1-3 tags on a plain background with no sensor noise at all",
        lambda rng, width, height: _tags_scene(rng, width, height, background="uniform"),
    ),
    Scenario("dark-background", "1-3 tags on a dark or mid-grey background, 36-140 px", _dark_background),
)

SCENARIO_NAMES = tuple(scenario.name for scenario in SCENARIOS)
_SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def iter_frames(
    *,
    scenarios: Sequence[str] = SCENARIO_NAMES,
    resolutions: Sequence[tuple[int, int]] = DEFAULT_RESOLUTIONS,
    frames_per_scenario: int = DEFAULT_FRAMES_PER_SCENARIO,
    seed: int = 2468,
) -> Iterator[Frame]:
    """Yield distinct frames. Each frame has its own seed, so any one can be re-rendered alone."""

    unknown = set(scenarios) - set(_SCENARIO_BY_NAME)
    if unknown:
        raise ValueError(f"unknown scenarios: {sorted(unknown)}")
    for width, height in resolutions:
        for name in scenarios:
            for index in range(frames_per_scenario):
                yield render_frame(name, index, (width, height), seed)


def render_frame(name: str, index: int, resolution: tuple[int, int], seed: int = 2468) -> Frame:
    """Render one frame. Its seed depends only on its scenario, index, and resolution."""

    width, height = resolution
    rng = np.random.default_rng((seed, SCENARIO_NAMES.index(name), width, height, index))
    image, tags, details = _SCENARIO_BY_NAME[name].render(rng, width, height)
    return Frame(name, index, resolution, _to_uint8(image), tags, details)


# -------------------------------------------------------------------------------------- occlusion


def occlusion_box(fraction: float, region: str) -> tuple[float, float, float, float]:
    """A patch covering ``fraction`` of the black square, in cell units (the square spans 0.5-8.5)."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("occlusion fraction must be between zero and one")
    if region == "payload-center":
        length = math.sqrt(fraction) * 8.0
        start = 4.5 - length / 2
        return start, start, start + length, start + length
    if region == "border-strip":
        return 0.5, 0.5, 0.5 + fraction * 8.0, 8.5
    if region == "corner":
        length = math.sqrt(fraction) * 8.0
        return 0.5, 0.5, 0.5 + length, 0.5 + length
    raise ValueError(f"unknown occlusion region {region!r}")


def iter_occlusion_frames(
    *, resolution: tuple[int, int] = (640, 480), cases: int = DEFAULT_OCCLUSION_CASES, seed: int = 2468
) -> Iterator[Frame]:
    """One tag per frame, a random ID, angle, and size, with a patch hiding part of it."""

    for index in range(len(OCCLUSION_REGIONS) * len(OCCLUSION_FRACTIONS) * len(OCCLUSION_VALUES) * cases):
        yield render_occlusion_frame(index, resolution=resolution, cases=cases, seed=seed)


def render_occlusion_frame(
    index: int, *, resolution: tuple[int, int] = (640, 480), cases: int = DEFAULT_OCCLUSION_CASES, seed: int = 2468
) -> Frame:
    """Obstruction frame ``index``, in region, fraction, patch colour, case order."""

    width, height = resolution
    case = index % cases
    value = OCCLUSION_VALUES[index // cases % len(OCCLUSION_VALUES)]
    fraction_index = index // (cases * len(OCCLUSION_VALUES)) % len(OCCLUSION_FRACTIONS)
    region_index = index // (cases * len(OCCLUSION_VALUES) * len(OCCLUSION_FRACTIONS))
    region, fraction = OCCLUSION_REGIONS[region_index], OCCLUSION_FRACTIONS[fraction_index]
    rng = np.random.default_rng((seed, 999, region_index, fraction_index, value, case))
    image = _uniform(rng, width, height)
    (tag,) = place_tags(rng, width, height, 1, (64.0, 160.0), max_tilt_deg=15)
    occlusion = Occlusion(*occlusion_box(fraction, region), value=value)
    render_tag(
        image, tag.tag_id, tag.corners, black=rng.uniform(15, 60), white=rng.uniform(200, 250), occlusion=occlusion
    )
    details = {"region": region, "fraction": fraction, "mask_value": value}
    return Frame("occlusion", index, resolution, _to_uint8(add_noise(rng, image, 1.5)), [tag], details)


# ------------------------------------------------------------------------------------------ range


def iter_range_frames(
    *, resolution: tuple[int, int] = (640, 480), cases: int = DEFAULT_RANGE_CASES, seed: int = 2468
) -> Iterator[Frame]:
    """One tag per frame at each fixed size in ``RANGE_SIDES_PX``: how far away each detector still sees a tag."""

    for index in range(len(RANGE_SIDES_PX) * cases):
        yield render_range_frame(index, resolution=resolution, cases=cases, seed=seed)


def render_range_frame(
    index: int, *, resolution: tuple[int, int] = (640, 480), cases: int = DEFAULT_RANGE_CASES, seed: int = 2468
) -> Frame:
    """Range frame ``index``, in size, case order: a random ID and angle, up to 15° of tilt, and light noise."""

    width, height = resolution
    case = index % cases
    side_index = index // cases
    side = float(RANGE_SIDES_PX[side_index])
    rng = np.random.default_rng((seed, 777, side_index, case))
    image = _uniform(rng, width, height)
    (tag,) = place_tags(rng, width, height, 1, (side, side), max_tilt_deg=15)
    render_tag(image, tag.tag_id, tag.corners, black=rng.uniform(15, 60), white=rng.uniform(200, 250))
    return Frame("range", index, resolution, _to_uint8(add_noise(rng, image, 1.5)), [tag], {"side_px": side})


# ---------------------------------------------------------------------------------------- scoring


def score_frame(truth: Sequence[TagTruth], detections: Sequence[Detection], *, align_corners: bool) -> FrameScore:
    """Score IDs exactly and corners against ground truth.

    Each expected ID is found at most once; repeated or unexpected IDs are false positives.
    With ``align_corners`` the detector's corner order is matched to the truth (rotation or
    reflection) before scoring, for detectors with a different corner convention.
    """

    expected = {tag.tag_id: tag for tag in truth}
    matched: set[int] = set()
    false_positive = 0
    errors: list[float] = []
    for tag_id, corners in detections:
        tag = expected.get(tag_id)
        if tag is None or tag_id in matched:
            false_positive += 1
            continue
        matched.add(tag_id)
        corners = np.asarray(corners, np.float64).reshape(4, 2)
        if align_corners:
            corners = min(
                (np.roll(order, shift, 0) for order in (corners, corners[::-1]) for shift in range(4)),
                key=lambda candidate: float(np.abs(candidate - tag.corners).max()),
            )
        errors.append(float(np.linalg.norm(corners - tag.corners, axis=1).max()))
    return FrameScore(len(matched), false_positive, len(expected) - len(matched), tuple(errors))


def summarize_latency(samples_ns: Sequence[int]) -> dict[str, float | int]:
    if not samples_ns:
        raise ValueError("latency samples must not be empty")
    values = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    p50, p95, p99 = np.percentile(values, (50, 95, 99))
    return {
        "samples": len(samples_ns),
        "min_ms": float(np.min(values)),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(np.max(values)),
    }


def _error_summary(errors: Sequence[float]) -> dict[str, float | None]:
    if not errors:
        return {"median_px": None, "p95_px": None, "max_px": None}
    values = np.asarray(errors)
    return {
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(values.max()),
    }


# --------------------------------------------------------------------------------------- backends


def build_backends(names: Sequence[str], *, threads: int = 1) -> tuple[list[Backend], dict[str, str]]:
    """Create backend factories. Construction is measured later, separately from detection."""

    available: dict[str, Callable[[], Backend]] = {
        "napa": _napa_backend,
        "apriltag": lambda: _apriltag_backend(2.0, threads),
        "apriltag-full-res": lambda: _apriltag_backend(1.0, threads),
        "opencv-aruco": _opencv_backend,
    }
    backends: list[Backend] = []
    skipped: dict[str, str] = {}
    for name in names:
        if name not in available:
            raise ValueError(f"unknown backend {name!r}; choose from {sorted(available)}")
        try:
            backends.append(available[name]())
        except (ImportError, RuntimeError) as exc:
            skipped[name] = str(exc)
    return backends, skipped


def _napa_backend() -> Backend:
    if not native_available():
        raise RuntimeError("the Napa core library is not built")

    def factory() -> DetectFn:
        detector = NapaDetector()
        return lambda image: [(o.tag_id, o.corners) for o in detector.detect(image)]

    defaults = NapaDetector()
    config = {"min_tag_px": defaults.min_tag_px, "max_hamming": defaults.max_hamming, "corner_order": "native"}
    defaults.close()
    return Backend("napa", factory, config)


def _apriltag_backend(decimate: float, threads: int) -> Backend:
    from robotpy_apriltag import AprilTagDetector

    def factory() -> DetectFn:
        detector = AprilTagDetector()
        if not detector.addFamily("tag36h11", 2):
            raise RuntimeError("AprilTag could not load tag36h11")
        config = detector.getConfig()
        config.numThreads = threads
        config.quadDecimate = decimate
        detector.setConfig(config)
        applied = detector.getConfig()
        if applied.numThreads != threads or applied.quadDecimate != decimate:
            raise RuntimeError("AprilTag did not apply the requested configuration")

        def detect(image: np.ndarray) -> list[Detection]:
            # AprilTag's pixel origin is a pixel's corner, not its centre: shift by -0.5 px.
            return [
                (int(d.getId()), np.array([[d.getCorner(i).x, d.getCorner(i).y] for i in range(4)]) - 0.5)
                for d in detector.detect(image)
            ]

        return detect

    name = "apriltag" if decimate == 2.0 else "apriltag-full-res"
    config = {
        "library": "AprilTag 3 (robotpy-apriltag)",
        "version": _package_version("robotpy-apriltag"),
        "family": "tag36h11",
        "bits_corrected": 2,
        "quad_decimate": decimate,
        "num_threads": threads,
        "corner_order": "aligned to truth",
    }
    return Backend(name, factory, config)


def _opencv_backend() -> Backend:
    def factory() -> DetectFn:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        def detect(image: np.ndarray) -> list[Detection]:
            corners, ids, _rejected = detector.detectMarkers(image)
            if ids is None:
                return []
            return [(int(i), c.reshape(4, 2).astype(np.float64)) for i, c in zip(ids.reshape(-1), corners, strict=True)]

        return detect

    config = {"library": "OpenCV ArUco", "version": cv2.__version__, "corner_refinement": "subpix"}
    return Backend("opencv-aruco", factory, config)


# ---------------------------------------------------------------------------------------- running


def run_matrix(
    frames: Iterator[Frame] | Sequence[Frame],
    *,
    backends: Sequence[Backend],
    repetitions: int = 10,
    warmup: int = 1,
    seed: int = 2468,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, dict], list[dict[str, object]]]:
    """Time and score every backend on every frame; return per-group aggregates and per-frame rows."""

    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup must not be negative")
    if not backends:
        raise ValueError("at least one backend is required")
    lifecycle: dict[str, dict[str, float]] = {}
    instances: dict[str, DetectFn] = {}
    for backend in backends:
        start = perf_counter_ns()
        instances[backend.name] = backend.factory()
        lifecycle[backend.name] = {"init_ms": (perf_counter_ns() - start) / 1e6}
    shuffler = random.Random(seed)
    rows: list[dict[str, object]] = []
    for count, frame in enumerate(frames, start=1):
        if count == 1:
            for backend in backends:
                start = perf_counter_ns()
                instances[backend.name](frame.image)
                lifecycle[backend.name]["first_call_ms"] = (perf_counter_ns() - start) / 1e6
        samples: dict[str, list[int]] = {backend.name: [] for backend in backends}
        results: dict[str, list[Detection]] = {}
        for _ in range(warmup):
            for backend in backends:
                instances[backend.name](frame.image)
        for _ in range(repetitions):
            order = list(backends)
            shuffler.shuffle(order)
            for backend in order:
                start = perf_counter_ns()
                detections = instances[backend.name](frame.image)
                samples[backend.name].append(perf_counter_ns() - start)
                results[backend.name] = detections
        for backend in backends:
            score = score_frame(
                frame.tags, results[backend.name], align_corners=backend.config.get("corner_order") != "native"
            )
            latency = samples[backend.name]
            rows.append(
                {
                    "frame": frame.name,
                    "scenario": frame.scenario,
                    "resolution": f"{frame.resolution[0]}x{frame.resolution[1]}",
                    "backend": backend.name,
                    "tags": len(frame.tags),
                    "min_side_px": round(min((t.side_px for t in frame.tags), default=0.0), 1),
                    "expected_ids": ";".join(str(t.tag_id) for t in frame.tags),
                    "detected_ids": ";".join(str(tag_id) for tag_id, _ in results[backend.name]),
                    "true_positive": score.true_positive,
                    "false_positive": score.false_positive,
                    "false_negative": score.false_negative,
                    "max_corner_error_px": round(max(score.corner_errors), 4) if score.corner_errors else "",
                    "corner_errors_px": ";".join(f"{e:.4f}" for e in score.corner_errors),
                    "median_ms": round(float(np.median(latency)) / 1e6, 5),
                    "min_ms": round(min(latency) / 1e6, 5),
                    "max_ms": round(max(latency) / 1e6, 5),
                    "samples_ns": ";".join(map(str, latency)),
                    **{f"detail_{key}": value for key, value in frame.details.items()},
                }
            )
        if progress is not None:
            progress(count)
    return lifecycle, rows


def aggregate(rows: Sequence[dict[str, object]], *, baseline: str = "napa") -> dict[str, object]:
    """Fold per-frame rows into per-(scenario, resolution, backend) results and per-frame speedups."""

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario"]), str(row["resolution"]), str(row["backend"]))].append(row)
    results = []
    for (scenario, resolution, backend), members in sorted(groups.items()):
        samples = [int(value) for row in members for value in str(row["samples_ns"]).split(";")]
        errors = [float(value) for row in members for value in str(row["corner_errors_px"]).split(";") if value]
        tags = sum(int(row["tags"]) for row in members)
        found = sum(int(row["true_positive"]) for row in members)
        results.append(
            {
                "scenario": scenario,
                "resolution": resolution,
                "backend": backend,
                "frames": len(members),
                "tags": tags,
                "found": found,
                "recall": found / tags if tags else None,
                "false_positives": sum(int(row["false_positive"]) for row in members),
                "frames_with_false_positive": sum(1 for row in members if int(row["false_positive"])),
                "latency": summarize_latency(samples),
                "frame_median_ms": summarize_latency([round(float(row["median_ms"]) * 1e6) for row in members]),
                "corner_error": _error_summary(errors),
            }
        )
    speedups = _speedups(rows, baseline)
    return {"results": results, "speedups": speedups}


def _speedups(rows: Sequence[dict[str, object]], baseline: str) -> list[dict[str, object]]:
    """Per-frame ratio of each backend's median time to the baseline's, on the same frame."""

    by_frame: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_frame[str(row["frame"])][str(row["backend"])] = row
    ratios: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for backends in by_frame.values():
        base = backends.get(baseline)
        if base is None:
            continue
        for name, row in backends.items():
            if name == baseline:
                continue
            ratio = float(row["median_ms"]) / max(float(base["median_ms"]), 1e-9)
            ratios[(str(row["scenario"]), str(row["resolution"]), name)].append(ratio)
            ratios[("all", "all", name)].append(ratio)
    report = []
    for (scenario, resolution, versus), values in sorted(ratios.items()):
        array = np.asarray(values)
        report.append(
            {
                "scenario": scenario,
                "resolution": resolution,
                "versus": versus,
                "frames": len(values),
                "median_speedup": float(np.median(array)),
                "p5_speedup": float(np.percentile(array, 5)),
                "min_speedup": float(array.min()),
                "frames_baseline_slower": int(np.count_nonzero(array < 1.0)),
            }
        )
    return report


def occlusion_report(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float, int], list[int]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["backend"]),
            str(row["detail_region"]),
            float(row["detail_fraction"]),
            int(row["detail_mask_value"]),
        )
        groups[key].append(int(row["true_positive"]))
    return [
        {
            "backend": backend,
            "region": region,
            "fraction": fraction,
            "mask_value": value,
            "cases": len(found),
            "found": sum(found),
        }
        for (backend, region, fraction, value), found in sorted(groups.items())
    ]


def range_report(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for row in rows:
        groups[(str(row["backend"]), float(row["detail_side_px"]))].append(int(row["true_positive"]))
    return [
        {"backend": backend, "side_px": side, "cases": len(found), "found": sum(found)}
        for (backend, side), found in sorted(groups.items())
    ]


def range_summary(entries: Sequence[dict[str, object]]) -> str:
    """Tags found at each black-square size, one row per backend."""

    sides = sorted({float(entry["side_px"]) for entry in entries})
    found = {(entry["backend"], float(entry["side_px"])): entry for entry in entries}
    lines = [
        "\n#### Range: tags found by black-square size\n",
        "| Backend | " + " | ".join(f"{side:g} px" for side in sides) + " |",
        "| --- |" + " ---: |" * len(sides),
    ]
    for backend in dict.fromkeys(entry["backend"] for entry in entries):
        cells = [f"{found[(backend, side)]['found']}/{found[(backend, side)]['cases']}" for side in sides]
        lines.append(f"| {backend} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------- report


def markdown_summary(report: dict[str, object], *, baseline: str = "napa", versus: str = "apriltag") -> str:
    """Tables in the README's format: one row per scenario at each resolution."""

    results = {(r["scenario"], r["resolution"], r["backend"]): r for r in report["results"]}
    speedups = {(s["scenario"], s["resolution"], s["versus"]): s for s in report["speedups"]}
    resolutions = sorted({r["resolution"] for r in report["results"]}, key=lambda s: int(s.split("x")[0]))
    lines: list[str] = []
    for resolution in resolutions:
        lines += [
            f"\n#### {resolution}\n",
            f"| Scene | Frames | Tags | {versus} p50 | {baseline} p50 | Speedup (median / worst frame) "
            f"| Found ({versus} → {baseline}) | False IDs |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for scenario in report["scenarios"]:
            ours = results.get((scenario, resolution, baseline))
            theirs = results.get((scenario, resolution, versus))
            speed = speedups.get((scenario, resolution, versus))
            if ours is None or theirs is None or speed is None:
                continue
            found = "—" if not ours["tags"] else f"{theirs['found']}/{theirs['tags']} → {ours['found']}/{ours['tags']}"
            lines.append(
                f"| {scenario} | {ours['frames']} | {ours['tags']} | {theirs['latency']['p50_ms']:.2f} ms "
                f"| {ours['latency']['p50_ms']:.2f} ms | {speed['median_speedup']:.1f}× / {speed['min_speedup']:.1f}× "
                f"| {found} | {theirs['false_positives']} → {ours['false_positives']} |"
            )
    return "\n".join(lines) + "\n"


def write_reports(report: dict[str, object], rows: Sequence[dict[str, object]], output: Path, csv_output: Path) -> None:
    _atomic_write(output, lambda file: json.dump(report, file, indent=2, sort_keys=True, allow_nan=False))
    fieldnames = sorted({key for row in rows for key in row})

    def write_csv(file) -> None:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(csv_output, write_csv)


def _atomic_write(path: Path, write: Callable[[object], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as file:
            temporary = file.name
            write(file)
            file.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _metadata() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": sys.version,
        "dependencies": {
            name: _package_version(name)
            for name in ("numpy", "opencv-python", "opencv-python-headless", "robotpy-apriltag", "napa-vision")
        },
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip() or None
    except OSError:
        pass
    return platform.processor() or None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# -------------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    today = datetime.now().date().isoformat()
    parser = argparse.ArgumentParser(
        description="Benchmark Napa against standard AprilTag on distinct synthetic frames"
    )
    parser.add_argument("--output", type=Path, default=Path(f"benchmarks/results/{today}_detector_matrix.json"))
    parser.add_argument("--samples-csv", type=Path, help="Per-frame results CSV (defaults beside --output)")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument(
        "--frames",
        type=_nonnegative_int,
        default=DEFAULT_FRAMES_PER_SCENARIO,
        help="Frames per scenario and resolution (0 to run only the sweeps)",
    )
    parser.add_argument(
        "--occlusion-cases",
        type=_nonnegative_int,
        default=DEFAULT_OCCLUSION_CASES,
        help="Tags per occlusion region, fraction, and patch colour (0 to skip)",
    )
    parser.add_argument(
        "--range-cases",
        type=_nonnegative_int,
        default=DEFAULT_RANGE_CASES,
        help=f"Tags at each size in {', '.join(map(str, RANGE_SIDES_PX))} px (0 to skip)",
    )
    parser.add_argument("--repetitions", type=_positive_int, default=10, help="Timed calls per backend per frame")
    parser.add_argument("--warmup", type=_nonnegative_int, default=1, help="Untimed calls per backend per frame")
    parser.add_argument("--threads", type=_positive_int, default=1, help="AprilTag worker threads")
    parser.add_argument(
        "--resolutions", type=_resolution_list, default=DEFAULT_RESOLUTIONS, help="Comma-separated WIDTHxHEIGHT values"
    )
    parser.add_argument(
        "--scenarios",
        type=_name_list,
        default=SCENARIO_NAMES,
        help=f"Comma-separated subset of: {','.join(SCENARIO_NAMES)}",
    )
    parser.add_argument(
        "--backends",
        type=_name_list,
        default=("napa", "apriltag", "apriltag-full-res"),
        help="Comma-separated: napa, apriltag, apriltag-full-res, opencv-aruco",
    )
    parser.add_argument("--quiet", action="store_true", help="Don't print progress or the summary tables")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        backends, skipped = build_backends(args.backends, threads=args.threads)
        cv2.setNumThreads(1)
        total = len(args.scenarios) * len(args.resolutions) * args.frames

        def progress(done: int) -> None:
            if not args.quiet and (done % 50 == 0 or done == total):
                print(f"\r{done}/{total} frames", end="", file=sys.stderr, flush=True)

        frames = iter_frames(
            scenarios=args.scenarios, resolutions=args.resolutions, frames_per_scenario=args.frames, seed=args.seed
        )
        lifecycle, rows = run_matrix(
            frames,
            backends=backends,
            repetitions=args.repetitions,
            warmup=args.warmup,
            seed=args.seed,
            progress=progress,
        )
        if not args.quiet:
            print(file=sys.stderr)
        occlusion_rows: list[dict[str, object]] = []
        if args.occlusion_cases:
            smallest = min(args.resolutions, key=lambda r: r[0] * r[1])
            occlusion_frames = iter_occlusion_frames(resolution=smallest, cases=args.occlusion_cases, seed=args.seed)
            _, occlusion_rows = run_matrix(occlusion_frames, backends=backends, repetitions=1, warmup=0, seed=args.seed)
        range_rows: list[dict[str, object]] = []
        if args.range_cases:
            smallest = min(args.resolutions, key=lambda r: r[0] * r[1])
            range_frames = iter_range_frames(resolution=smallest, cases=args.range_cases, seed=args.seed)
            _, range_rows = run_matrix(range_frames, backends=backends, repetitions=1, warmup=0, seed=args.seed)
        summary = aggregate(rows)
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "scope": (
                "Detector call only, on pre-rendered uint8 grayscale frames. Every frame is distinct. Corner error is "
                "the worst corner's Euclidean distance from exact ground truth; AprilTag corners are shifted -0.5 px "
                "to the pixel-centre convention and re-ordered to the truth before scoring."
            ),
            "metadata": _metadata(),
            "run_config": {
                "seed": args.seed,
                "frames_per_scenario": args.frames,
                "repetitions": args.repetitions,
                "warmup": args.warmup,
                "threads": args.threads,
                "resolutions": [f"{w}x{h}" for w, h in args.resolutions],
                "occlusion_cases": args.occlusion_cases,
                "range_cases": args.range_cases,
            },
            "scenarios": list(args.scenarios),
            "scenario_descriptions": {s.name: s.description for s in SCENARIOS if s.name in args.scenarios},
            "backends": {b.name: {"config": b.config, "lifecycle": lifecycle[b.name]} for b in backends},
            "skipped_backends": skipped,
            "frames": len(rows) // max(1, len(backends)),
            "tags": sum(int(row["tags"]) for row in rows) // max(1, len(backends)),
            **summary,
            "occlusion": occlusion_report(occlusion_rows),
            "range": range_report(range_rows),
        }
        csv_output = args.samples_csv or args.output.with_suffix(".csv")
        write_reports(report, rows + occlusion_rows + range_rows, args.output, csv_output)
        if not args.quiet:
            print(markdown_summary(report))
            if report["range"]:
                print(range_summary(report["range"]))
        print(json.dumps({"report": str(args.output), "samples_csv": str(csv_output)}, indent=2))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"performance-matrix: error: {exc}", file=sys.stderr)
        return 2
    return 0


def _resolution_list(value: str) -> tuple[tuple[int, int], ...]:
    resolutions: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            width_text, height_text = item.lower().split("x", 1)
            width, height = int(width_text), int(height_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("resolutions must use WIDTHxHEIGHT, separated by commas") from exc
        if width < 320 or height < 240:
            raise argparse.ArgumentTypeError("resolutions must be at least 320x240")
        resolutions.append((width, height))
    return tuple(resolutions)


def _name_list(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("at least one name is required")
    return names


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
