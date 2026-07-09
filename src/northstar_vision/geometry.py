from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

# OpenCV camera coordinates are +X right, +Y down, +Z forward. WPILib uses
# +X forward, +Y left, +Z up. These matrices convert vector coordinates and
# conjugate full transforms between the two right-handed bases.
WPILIB_TO_OPENCV_ROTATION = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
OPENCV_TO_WPILIB_ROTATION = WPILIB_TO_OPENCV_ROTATION.T


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def transform_matrix(translation: list[float] | np.ndarray, rotation_rpy: list[float] | np.ndarray) -> np.ndarray:
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    r = euler_xyz_to_matrix(float(rotation_rpy[0]), float(rotation_rpy[1]), float(rotation_rpy[2]))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def invert_transform(transform: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    r = transform[:3, :3]
    t = transform[:3, 3]
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    r, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return out


def opencv_pose_to_wpilib_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert an OpenCV pose transform into the equivalent WPILib basis."""

    opencv_transform = rvec_tvec_to_transform(rvec, tvec)
    basis = np.eye(4, dtype=np.float64)
    basis[:3, :3] = OPENCV_TO_WPILIB_ROTATION
    return basis @ opencv_transform @ basis.T


def opencv_translation_to_wpilib(translation: np.ndarray) -> np.ndarray:
    """Convert an OpenCV XYZ translation to WPILib forward/left/up axes."""

    return OPENCV_TO_WPILIB_ROTATION @ np.asarray(translation, dtype=np.float64).reshape(3)


def wpilib_points_to_opencv(points: np.ndarray) -> np.ndarray:
    """Convert one or more WPILib-coordinate points to OpenCV coordinates."""

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (WPILIB_TO_OPENCV_ROTATION @ values.T).T


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (transform @ homogeneous.T).T[:, :3]


@lru_cache(maxsize=16)
def tag_object_points(tag_size_m: float) -> np.ndarray:
    """Return cached IPPE square corners in OpenCV's tag-local convention."""

    half = tag_size_m / 2.0
    points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    points.setflags(write=False)
    return points


def tag_field_corners(field_to_tag: np.ndarray, tag_size_m: float) -> np.ndarray:
    """Return tag corners in the WPILib field basis.

    ``field_to_tag`` is the pose of the tag in the field (a transform from tag
    coordinates to field coordinates). A WPILib tag lies in its local Y/Z
    plane with +X normal to its face. The order matches official AprilRobotics
    tag artwork and upstream Northstar. An unrotated OpenCV-generated marker
    can be warped directly into this ordered corner array; rotating it to look
    like the physical artwork also requires remapping destination corners.
    """

    half = tag_size_m / 2.0
    local_corners = np.array(
        [
            [0.0, half, -half],
            [0.0, -half, -half],
            [0.0, -half, half],
            [0.0, half, half],
        ],
        dtype=np.float64,
    )
    return transform_points(field_to_tag, local_corners)


def rotation_matrix_to_quaternion_wxyz(transform: np.ndarray) -> np.ndarray:
    r = transform[:3, :3]
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    return quaternion if norm == 0.0 else quaternion / norm
