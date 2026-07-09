"""Compatibility wrapper for OpenCV's AprilTag 36h11 ArUco detector."""

from __future__ import annotations

import cv2
import numpy as np


class OpenCVArucoAprilTag36h11:
    def __init__(self) -> None:
        self.available = (
            hasattr(cv2, "aruco")
            and hasattr(cv2.aruco, "getPredefinedDictionary")
            and hasattr(cv2.aruco, "ArucoDetector")
        )
        if not self.available:
            self.dictionary = None
            self.params = None
            self.detector = None
            return
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, gray: np.ndarray):
        if not self.available:
            return None, None, None
        return self.detector.detectMarkers(gray)


def draw_marker(dictionary, tag_id: int, side_pixels: int, border_bits: int = 1) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, tag_id, side_pixels, borderBits=border_bits)
    return cv2.aruco.drawMarker(dictionary, tag_id, side_pixels, borderBits=border_bits)
