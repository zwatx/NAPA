"""Owned projective tag and gray attachment, with independent pixel-center corners."""

from pathlib import Path

import cv2
import numpy as np


def make_fixture():
    corners = np.array([[140.5, 70.5], [160.5, 84.5], [163.5, 142.5], [142.5, 134.5]])
    scale = 4
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 2, 256)
    card = np.full((320, 320), 230, np.uint8)
    card[32:288, 32:288] = np.where(marker > 0, 230, 100)
    source = np.array([[31.5, 31.5], [287.5, 31.5], [287.5, 287.5], [31.5, 287.5]], np.float32)
    matrix = cv2.getPerspectiveTransform(source, ((corners + 0.5) * scale - 0.5).astype(np.float32))
    clean = cv2.warpPerspective(card, matrix, (320 * scale, 240 * scale), flags=cv2.INTER_LINEAR, borderValue=190)
    mask = np.zeros_like(clean)
    cv2.fillConvexPoly(mask, np.round((corners + 0.5) * scale - 0.5).astype(np.int32), 255)
    image = clean.copy()
    # A weaker edge beside the quiet zone must not displace the printed border.
    x = int(np.ceil(corners[1, 0])) + 2
    cv2.rectangle(image, (x * scale, 86 * scale), ((x + 3) * scale, 104 * scale), 140, -1)
    cv2.rectangle(image, (159 * scale, 86 * scale), ((x + 3) * scale, 89 * scale), 140, -1)
    image[mask > 0] = clean[mask > 0]
    image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    shifted = np.full_like(image, 190)
    shifted[1:] = image[:-1]
    return shifted, corners + [0, 1]


if __name__ == "__main__":
    image, corners = make_fixture()
    np.savez_compressed(Path(__file__).with_name("initial-background-2.npz"), image=image, corners=corners, tag_id=2)
