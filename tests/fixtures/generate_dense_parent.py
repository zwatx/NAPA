"""Owned dense body attached through an unsupported corner to a small tag."""

from pathlib import Path

import cv2
import numpy as np

image = np.full((384, 448), 255, np.uint8)
side, body, gap = 12, 80, 1
x, y = 216, 192
bx, by = x - gap - body, y - gap - body
block = np.zeros((body, body), np.uint8)
# Small white holes keep local contrast throughout a connected black body.
for j in range(2, body - 2, 4):
    for i in range(2, body - 2, 4):
        block[j : j + 2, i : i + 2] = 255
image[by : by + body, bx : bx + body] = block
cv2.line(image, (bx + body - 1, by + body - 1), (x - 1, y - 1), 0, 1, cv2.LINE_8)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
image[y : y + side, x : x + side] = cv2.aruco.generateImageMarker(dictionary, 137, side)
corners = np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], float) - 0.5
np.savez_compressed(Path(__file__).with_name("dense-parent-small-tag.npz"), image=image, corners=corners, tag_id=137)
