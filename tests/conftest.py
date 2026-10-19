"""Shared test fixtures: a synthetic 1280x720 pinhole camera and tag projection."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from napa_vision.geometry import invert_transform, tag_field_corners, transform_points, wpilib_points_to_opencv


def example_profile() -> SimpleNamespace:
    """A distortion-free 1280x720 camera (900 px focal length) and a 16.51 cm tag."""

    camera = SimpleNamespace(
        calibration_matrix=np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.zeros(5),
    )
    return SimpleNamespace(camera=camera, tag_size_m=0.1651, max_reprojection_error_px=5.0)


def project_field_tag(
    field_to_camera: np.ndarray,
    field_to_tag: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    """Project a tag's corners from world coordinates into the image."""

    camera_to_field = invert_transform(field_to_camera)
    corners_field = tag_field_corners(field_to_tag, tag_size_m)
    corners_camera = wpilib_points_to_opencv(transform_points(camera_to_field, corners_field))
    corners, _ = cv2.projectPoints(corners_camera, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, distortion)
    return corners.reshape(4, 2)
