"""Owned gray tag with sensor noise: independent pixel-center square corners."""

from pathlib import Path

import cv2
import numpy as np

yy, xx = np.indices((480, 640))
image = np.where((xx // 8 + yy // 8) & 1, 220, 40).astype(np.uint8)
side, gray, pad = 160, 48, 20
card = np.full((side + 2 * pad, side + 2 * pad), gray, np.uint8)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
marker = cv2.aruco.generateImageMarker(dictionary, 137, side)
card[pad : pad + side, pad : pad + side] = np.rint(marker.astype(float) * gray / 255).astype(np.uint8)
x, y = 120, 64
image[y : y + len(card), x : x + len(card)] = card
noise = np.random.default_rng(9237).normal(0, 8, image.shape)
image = np.clip(np.rint(image.astype(float) + noise), 0, 255).astype(np.uint8)
corners = (
    np.array(
        [[x + pad, y + pad], [x + pad + side, y + pad], [x + pad + side, y + pad + side], [x + pad, y + pad + side]],
        float,
    )
    - 0.5
)
np.savez_compressed(Path(__file__).with_name("dense-stride-two.npz"), image=image, corners=corners, tag_id=137)
