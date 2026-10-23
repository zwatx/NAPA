from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import _import_cv2

# OpenCV camera coordinates are +X right, +Y down, +Z forward. WPILib uses
# +X forward, +Y left, +Z up. These matrices convert vector coordinates and
# conjugate full transforms between the two right-handed bases.
WPILIB_TO_OPENCV_ROTATION = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
OPENCV_TO_WPILIB_ROTATION = WPILIB_TO_OPENCV_ROTATION.T
OPENCV_TO_WPILIB_BASIS = np.eye(4, dtype=np.float64)
OPENCV_TO_WPILIB_BASIS[:3, :3] = OPENCV_TO_WPILIB_ROTATION
OPENCV_TO_WPILIB_BASIS.setflags(write=False)


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


def _rigid_transform(value: np.ndarray, name: str) -> np.ndarray:
    """Return an independent finite rigid transform snapshot."""

    transform = np.asarray(value, dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.all(np.isfinite(transform))
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0.0, atol=1e-6)
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, rtol=0.0, atol=1e-6)
    ):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return transform.copy()


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    r, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return out


def opencv_pose_to_wpilib_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert an OpenCV pose transform into the equivalent WPILib basis."""

    opencv_transform = rvec_tvec_to_transform(rvec, tvec)
    return OPENCV_TO_WPILIB_BASIS @ opencv_transform @ OPENCV_TO_WPILIB_BASIS.T


def opencv_translation_to_wpilib(translation: np.ndarray) -> np.ndarray:
    """Convert an OpenCV XYZ translation to WPILib forward/left/up axes."""

    return OPENCV_TO_WPILIB_ROTATION @ np.asarray(translation, dtype=np.float64).reshape(3)


def wpilib_points_to_opencv(points: np.ndarray) -> np.ndarray:
    """Convert one or more WPILib-coordinate points to OpenCV coordinates."""

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (WPILIB_TO_OPENCV_ROTATION @ values.T).T


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


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


# Upstream AprilTag's tag coordinates for each of Napa's corners. Napa returns corners in OpenCV's order, which is
# upstream's p[1], p[0], p[3], p[2]; upstream maps p[0..3] from (-1, 1), (1, 1), (1, -1), (-1, -1).
_HOMOGRAPHY_TAG_POINTS = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])


def tag_homography(corners: np.ndarray) -> np.ndarray:
    """Return the 3x3 homography from upstream AprilTag's tag coordinates to these image corners.

    Tag coordinates span ``[-1, 1]`` across the black square, as in upstream AprilTag's ``H``, with
    ``H[2, 2] == 1``. The image coordinates are the corners' own (Napa's are pixel-centre).
    """

    points = np.asarray(corners, dtype=np.float64)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError("corners must be a finite 4x2 array")
    u, v = _HOMOGRAPHY_TAG_POINTS[:, 0], _HOMOGRAPHY_TAG_POINTS[:, 1]
    x, y = points[:, 0], points[:, 1]
    zeros, ones = np.zeros(4), np.ones(4)
    system = np.empty((8, 8), dtype=np.float64)
    system[0::2] = np.column_stack((u, v, ones, zeros, zeros, zeros, -u * x, -v * x))
    system[1::2] = np.column_stack((zeros, zeros, zeros, u, v, ones, -u * y, -v * y))
    try:
        solution = np.linalg.solve(system, points.reshape(8))
    except np.linalg.LinAlgError:
        raise ValueError("corners do not form a quadrilateral") from None
    return np.append(solution, 1.0).reshape(3, 3)


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
