"""Synthetic tag36h11 rendering utilities for tests and benchmarks."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .family import TAG36H11, code_to_bits


def render_tag36h11(tag_id: int, *, cell_size: int = 24, margin: bool = True) -> np.ndarray:
    """Render a standard tag36h11 marker as a grayscale image.

    ``margin=True`` produces the common 10x10-cell printable form: one white
    quiet-zone cell around the 8x8 AprilTag marker. ``margin=False`` renders
    the bare 8x8 marker with black border and 6x6 payload.
    """

    if tag_id < 0 or tag_id >= len(TAG36H11.codes):
        raise ValueError(f"tag_id must be in [0, {len(TAG36H11.codes) - 1}]")

    marker = np.full((TAG36H11.width_at_border, TAG36H11.width_at_border), 255, dtype=np.uint8)
    marker[0, :] = 0
    marker[-1, :] = 0
    marker[:, 0] = 0
    marker[:, -1] = 0
    bits = code_to_bits(TAG36H11.codes[tag_id])
    for bit, x, y in zip(bits, TAG36H11.bit_x.astype(int), TAG36H11.bit_y.astype(int)):
        marker[y, x] = 255 if bit else 0

    # AprilTag decoding is rotation invariant. OpenCV's generator publishes
    # this family in the orientation that is 180 degrees from the upstream C
    # codeword order, so render that orientation for easier baseline checks.
    marker = np.rot90(marker, 2).copy()

    if margin:
        grid = np.full((TAG36H11.total_width, TAG36H11.total_width), 255, dtype=np.uint8)
        grid[1:-1, 1:-1] = marker
    else:
        grid = marker
    return np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)


def render_tag36h11_scene(
    tag_id: int,
    *,
    image_size: tuple[int, int] = (480, 640),
    corners: np.ndarray | None = None,
    cell_size: int = 32,
    occluders: Iterable[tuple[int, int, int, int, int]] = (),
    blur_sigma: float = 0.0,
    noise_sigma: float = 0.0,
    brightness: float = 0.0,
) -> np.ndarray:
    """Render one tag onto a white scene.

    ``corners`` are the image-space corners of the black-border marker, not the
    outer white quiet zone. Occluders are ``(x0, y0, x1, y1, gray)`` rectangles
    applied after warping.
    """

    height, width = image_size
    canvas = np.full((height, width), 255, dtype=np.uint8)
    tag = render_tag36h11(tag_id, cell_size=cell_size, margin=True)
    quiet = float(cell_size)
    marker0 = np.array(
        [
            [quiet, quiet],
            [tag.shape[1] - quiet, quiet],
            [tag.shape[1] - quiet, tag.shape[0] - quiet],
            [quiet, tag.shape[0] - quiet],
        ],
        dtype=np.float32,
    )
    if corners is None:
        cx, cy = width * 0.5, height * 0.5
        side = min(width, height) * 0.42
        corners = np.array(
            [
                [cx - side / 2, cy - side / 2],
                [cx + side / 2, cy - side / 2],
                [cx + side / 2, cy + side / 2],
                [cx - side / 2, cy + side / 2],
            ],
            dtype=np.float32,
        )
    else:
        corners = np.asarray(corners, dtype=np.float32)

    h = cv2.getPerspectiveTransform(marker0, corners)
    warped = cv2.warpPerspective(tag, h, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
    mask = cv2.warpPerspective(
        np.full_like(tag, 255),
        h,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    canvas[mask > 0] = warped[mask > 0]

    for x0, y0, x1, y1, gray in occluders:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), int(gray), thickness=-1)

    if brightness:
        canvas = np.clip(canvas.astype(np.float32) + brightness, 0, 255).astype(np.uint8)
    if blur_sigma > 0:
        ksize = max(3, int(blur_sigma * 6 + 1) | 1)
        canvas = cv2.GaussianBlur(canvas, (ksize, ksize), blur_sigma)
    if noise_sigma > 0:
        noise = np.random.default_rng(0).normal(0, noise_sigma, canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return canvas
