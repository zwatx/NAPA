"""Regenerate grey-margin-<seed>.npz: one tag36h11 tag whose quiet zone is as grey as its white cells, on a lighter
background (the margin reads dark against the scene but light against the border), with mild blur and sensor noise.
usage: python tests/fixtures/generate_grey_margin.py SEED [SEED ...]"""

import sys
from pathlib import Path

import cv2
import numpy as np

from napa_vision.tags import tag_cells


def render(seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    tag_id = int(rng.integers(0, 587))
    side, angle = rng.uniform(60, 110), rng.uniform(0, 2 * np.pi)
    center = np.array([320, 240]) + rng.uniform(-60, 60, 2)
    perspective = rng.uniform(-0.15, 0.15, 2)
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * side / 2
    square = square * (1 + square @ perspective / side)[:, None]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = square @ rotation.T + center
    scale, margin = 16, 16  # 16 px per cell in the artwork; a one-cell quiet zone
    artwork = np.ones((10 * scale, 10 * scale))
    artwork[margin : margin + 8 * scale, margin : margin + 8 * scale] = np.kron(
        tag_cells(tag_id, quiet_zone_cells=0).astype(float), np.ones((scale, scale))
    )
    grey, black, background = 132 + rng.uniform(-20, 20), 29 + rng.uniform(-10, 10), 238
    texture = np.where(artwork > 0.5, grey, black).astype(np.float32)
    source = (
        np.float32(
            [
                [margin, margin],
                [margin + 8 * scale, margin],
                [margin + 8 * scale, margin + 8 * scale],
                [margin, margin + 8 * scale],
            ]
        )
        - 0.5
    )
    warp = cv2.getPerspectiveTransform(source, np.float32(corners))
    image = cv2.warpPerspective(texture, warp, (640, 480), flags=cv2.INTER_AREA, borderValue=-1)
    cover = cv2.warpPerspective(np.ones_like(texture), warp, (640, 480), flags=cv2.INTER_LINEAR, borderValue=0)
    image = cover * np.where(image < 0, background, image) + (1 - cover) * background
    image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.4, 0.9)) + rng.normal(0, rng.uniform(1, 4), image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), corners, tag_id


if __name__ == "__main__":
    for seed in map(int, sys.argv[1:]):
        image, corners, tag_id = render(seed)
        np.savez_compressed(
            Path(__file__).with_name(f"grey-margin-{seed}.npz"), image=image, corners=corners, tag_id=tag_id
        )
