"""Pose uncertainty for Kalman filters, linearized from corner reprojection.

A pose's 6x6 covariance is ordered ``[x, y, z, rx, ry, rz]``, as in ROS ``PoseWithCovariance``: the error in
translation, in the parent frame, then a small rotation about the parent frame's axes, applied on the left of the
estimated rotation (``R_true = exp([r]x) @ R``). Covariances come from the Gauss-Newton approximation
``sigma^2 (J^T J)^-1``, where ``J`` is the Jacobian of the projected tag corners with respect to the pose and
``sigma`` is the per-axis corner noise in pixels. The solvers then widen planar targets' covariances to cover their
mirror solution (``ambiguity_covariances``) and cap correlations (``cap_correlations``); docs/api.md explains why.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import _import_cv2
from .geometry import WPILIB_TO_OPENCV_ROTATION, invert_transform

DEFAULT_CORNER_NOISE_PX = 1.0
# Fraction of each variance added back to the diagonal, capping every correlation at 1 / 1.02. Linearization can make
# a combination of axes look almost exactly known (a far wall of tags, a lone tag at range) while the true error
# surface is slightly curved; in Monte Carlo tests that left mean NEES at 50-90 for 6 dimensions with every per-axis
# variance right. With the cap, each axis's standard deviation grows by 1% and those cases come back to about 6.
CORRELATION_SLACK = 0.02


def _skew(vector: np.ndarray) -> np.ndarray:
    """The cross-product matrix ``[v]x`` of a 3-vector."""

    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _block_diagonal(rotation: np.ndarray) -> np.ndarray:
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = rotation
    out[3:, 3:] = rotation
    return out


def _propagate(jacobian: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    out = jacobian @ covariance @ jacobian.T
    return (out + out.T) / 2.0


def cap_correlations(covariance: np.ndarray | None) -> np.ndarray | None:
    """Add ``CORRELATION_SLACK`` of each variance to the diagonal of a reported covariance."""

    if covariance is None:
        return None
    return covariance + CORRELATION_SLACK * np.diag(np.diag(covariance))


def _checked_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.array(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return transform


def validate_corner_noise(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError("corner_noise_px must be finite and greater than zero")
    return float(value)


@dataclass(frozen=True, slots=True)
class Pose2dWithCovariance:
    """A planar pose ``(x, y, yaw)`` and its 3x3 covariance, in metres and radians."""

    x: float
    y: float
    yaw: float
    covariance: np.ndarray

    def std_devs(self) -> np.ndarray:
        """Standard deviations of ``(x, y, yaw)``, with x and y along the parent frame's axes."""

        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    def local_std_devs(self) -> np.ndarray:
        """Standard deviations along the pose's own heading (forward, left) and of yaw.

        Pass these to WPILib's ``addVisionMeasurement``: its estimators apply vision standard deviations to the twist
        from the estimated pose, which is in the robot's frame, not the field's.
        """

        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        rotation = np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        return np.sqrt(np.clip(np.diag(rotation @ self.covariance @ rotation.T), 0.0, None))


@dataclass(frozen=True, slots=True)
class PoseWithCovariance:
    """A 4x4 pose of a child frame in a parent frame, with the 6x6 covariance of its error.

    See the module docstring for the covariance's order and convention. The transforms passed to
    ``transform_by`` and ``relative_to`` are treated as exact.
    """

    transform: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        transform = _checked_transform(self.transform, "transform")
        covariance = np.array(self.covariance, dtype=np.float64)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be a finite 6x6 array")
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "covariance", covariance)

    @property
    def translation(self) -> np.ndarray:
        return self.transform[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        return self.transform[:3, :3].copy()

    def std_devs(self) -> np.ndarray:
        """Standard deviations of ``[x, y, z, rx, ry, rz]``."""

        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    def transform_by(self, transform: np.ndarray) -> PoseWithCovariance:
        """Return ``self.transform @ transform``, as WPILib's ``Pose3d.transformBy``.

        Use it to move to a rigidly attached frame: ``field_to_camera.transform_by(camera_to_robot)`` is the robot's
        pose in the field.
        """

        offset = _checked_transform(transform, "transform")
        rotation = self.transform[:3, :3]
        jacobian = np.eye(6, dtype=np.float64)
        jacobian[:3, 3:] = -_skew(rotation @ offset[:3, 3])
        return PoseWithCovariance(self.transform @ offset, _propagate(jacobian, self.covariance))

    def relative_to(self, origin: np.ndarray) -> PoseWithCovariance:
        """Return this pose in ``origin``'s frame, ``inverse(origin) @ self.transform``, as ``Pose3d.relativeTo``.

        ``origin`` is the new frame's pose in the current parent frame, for example the red alliance's origin in
        the blue-origin field frame.
        """

        new_from_parent = invert_transform(_checked_transform(origin, "origin"))
        jacobian = _block_diagonal(new_from_parent[:3, :3])
        return PoseWithCovariance(new_from_parent @ self.transform, _propagate(jacobian, self.covariance))

    def inverse(self) -> PoseWithCovariance:
        """Return the parent's pose in the child frame, with the covariance expressed in the child frame."""

        rotation_t = self.transform[:3, :3].T
        jacobian = np.zeros((6, 6), dtype=np.float64)
        jacobian[:3, :3] = -rotation_t
        jacobian[:3, 3:] = -rotation_t @ _skew(self.transform[:3, 3])
        jacobian[3:, 3:] = -rotation_t
        return PoseWithCovariance(invert_transform(self.transform), _propagate(jacobian, self.covariance))

    def to_pose2d(self) -> Pose2dWithCovariance:
        """Project onto the parent's X/Y plane: ``(x, y, yaw)`` with yaw counter-clockwise about +Z.

        Yaw is the heading of the child's +X axis. Raises ``ValueError`` when that axis points straight up or down.
        """

        heading = self.transform[:3, 0]
        planar = heading[0] * heading[0] + heading[1] * heading[1]
        if planar < 1e-12:
            raise ValueError("yaw is undefined: the pose's +X axis is vertical")
        jacobian = np.zeros((3, 6), dtype=np.float64)
        jacobian[0, 0] = jacobian[1, 1] = 1.0
        # A small rotation r turns the heading c by r x c; differentiate atan2(c_y, c_x).
        jacobian[2, 3:] = np.array([-heading[0] * heading[2], -heading[1] * heading[2], planar]) / planar
        return Pose2dWithCovariance(
            float(self.transform[0, 3]),
            float(self.transform[1, 3]),
            math.atan2(heading[1], heading[0]),
            _propagate(jacobian, self.covariance),
        )


def change_basis(covariance: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Re-express a pose covariance when the parent frame's axes are rotated by ``rotation`` (new from old)."""

    return _propagate(_block_diagonal(rotation), covariance)


def reprojection_covariances(
    camera_from_points: np.ndarray,
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    corner_noise_px: float,
    *,
    opencv_from_camera: np.ndarray | None = None,
) -> list[np.ndarray | None]:
    """Covariances of poses that map ``points`` into the camera frame, one per pose.

    ``camera_from_points`` is ``(M, 4, 4)`` and ``points`` is ``(K, 3)``, shared by every pose, or ``(M, K, 3)``.
    The camera frame is OpenCV's unless ``opencv_from_camera`` rotates it there. Each covariance is in the camera
    frame; it is ``None`` where the corners don't constrain all six degrees of freedom.
    """

    cv2 = _import_cv2()
    poses = np.asarray(camera_from_points, dtype=np.float64).reshape(-1, 4, 4)
    count = len(poses)
    object_points = np.broadcast_to(np.asarray(points, dtype=np.float64), (count, *np.shape(points)[-2:]))
    rotated = np.einsum("mij,mkj->mki", poses[:, :3, :3], object_points)
    camera_points = rotated + poses[:, None, :3, 3]
    basis = np.eye(3) if opencv_from_camera is None else np.asarray(opencv_from_camera, dtype=np.float64)
    _, jacobian = cv2.projectPoints(
        (camera_points @ basis.T).reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        camera_matrix,
        distortion_coefficients,
    )
    # With a zero pose, d(pixel)/d(tvec) is d(pixel)/d(point).
    pixel_from_point = jacobian[:, 3:6].reshape(count, -1, 2, 3) @ basis
    # A small rotation r moves a point p by r x p, so each pixel row a gains a . (r x p) = (p x a) . r.
    rotation = np.cross(rotated[:, :, None, :], pixel_from_point)
    stacked = np.concatenate((pixel_from_point, rotation), axis=-1).reshape(count, -1, 6)
    information = np.einsum("mki,mkj->mij", stacked, stacked)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    valid = eigenvalues[:, 0] > 1e-12 * eigenvalues[:, -1]
    scale = corner_noise_px * corner_noise_px / np.where(valid[:, None], eigenvalues, 1.0)
    covariances = (eigenvectors * scale[:, None, :]) @ eigenvectors.transpose(0, 2, 1)
    return [covariance if ok else None for covariance, ok in zip(covariances, valid, strict=True)]


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """The rotation vector of a rotation matrix, accurate for tiny angles (OpenCV's ``Rodrigues`` zeroes them)."""

    sine_axis = 0.5 * np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    )
    sine = float(np.linalg.norm(sine_axis))
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    if cosine < 0.0 and sine < 1e-6:
        # Near a half turn the antisymmetric part vanishes; OpenCV handles this case well.
        return _import_cv2().Rodrigues(rotation)[0].reshape(3)
    angle = math.atan2(sine, cosine)
    return sine_axis * (angle / sine if sine > 0.0 else 1.0)


def pose_difference(transform: np.ndarray, other: np.ndarray) -> np.ndarray:
    """``other``'s offset from ``transform`` in the covariance convention: ``[dt, r]`` with ``R_other = exp(r) R``."""

    rotation = _rotation_vector(other[:3, :3] @ transform[:3, :3].T)
    return np.concatenate((other[:3, 3] - transform[:3, 3], rotation))


def reprojection_ssr(
    camera_from_points: np.ndarray,
    points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    *,
    opencv_from_camera: np.ndarray | None = None,
) -> np.ndarray:
    """Sum of squared corner reprojection residuals, px^2, of each pose; arguments as ``reprojection_covariances``.

    ``image_points`` is ``(K, 2)``, shared by every pose, or ``(M, K, 2)``.
    """

    cv2 = _import_cv2()
    poses = np.asarray(camera_from_points, dtype=np.float64).reshape(-1, 4, 4)
    count = len(poses)
    object_points = np.broadcast_to(np.asarray(points, dtype=np.float64), (count, *np.shape(points)[-2:]))
    camera_points = np.einsum("mij,mkj->mki", poses[:, :3, :3], object_points) + poses[:, None, :3, 3]
    basis = np.eye(3) if opencv_from_camera is None else np.asarray(opencv_from_camera, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        (camera_points @ basis.T).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, distortion_coefficients
    )
    corners = object_points.shape[1]
    residuals = projected.reshape(count, corners, 2) - np.asarray(image_points, dtype=np.float64).reshape(
        -1, corners, 2
    )
    return np.sum(residuals * residuals, axis=(1, 2))


def ambiguity_covariances(
    transforms: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray | None],
    squared_residuals_px2: Sequence[float],
    corner_noise_px: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Widen two planar-ambiguity solutions' covariances to cover each other.

    A planar target's two pose solutions are modes of one posterior. Each is weighted by its corner likelihood,
    tempered by 2, ``exp(-SSR / (4 sigma^2))`` with ``SSR`` its sum of squared corner residuals, and each returned
    covariance is the mixture's second moment about that solution. A clearly worse alternative adds nothing; an
    equally good one adds the full spread, which also covers the linearization's failure when the target is seen
    nearly head-on. The tempering allows for the valley between the solutions not being quadratic: untempered, a
    head-on wall of tags was still overconfident in Monte Carlo tests (mean NEES 27 for 6 dimensions; tempered, 8).
    """

    if covariances[0] is None or covariances[1] is None:
        return covariances[0], covariances[1]
    log_ratio = (squared_residuals_px2[1] - squared_residuals_px2[0]) / (4.0 * corner_noise_px * corner_noise_px)
    weight_1 = 1.0 / (1.0 + math.exp(min(max(log_ratio, -700.0), 700.0)))
    weight_0 = 1.0 - weight_1
    difference = pose_difference(transforms[0], transforms[1])
    spread = np.outer(difference, difference)
    return (
        weight_0 * covariances[0] + weight_1 * (covariances[1] + spread),
        weight_1 * covariances[1] + weight_0 * (covariances[0] + spread),
    )


def center_angle_covariances(
    corner_angles: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    corner_noise_px: float,
) -> np.ndarray:
    """2x2 covariances of tags' mean corner angles, from ``(T, 4, 2)`` undistorted corner angles in radians."""

    cv2 = _import_cv2()
    angles = np.asarray(corner_angles, dtype=np.float64).reshape(-1, 4, 2)
    normalized = np.tan(angles).reshape(-1, 2)
    points = np.column_stack((normalized, np.ones(len(normalized))))
    _, jacobian = cv2.projectPoints(
        points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, distortion_coefficients
    )
    # At depth 1, d(pixel)/d(point x, y) is d(pixel)/d(normalized x, y); angle = atan(normalized) per axis.
    angle_from_pixel = np.linalg.inv(jacobian[:, 3:5].reshape(-1, 2, 2)) / (1.0 + normalized**2)[:, :, None]
    per_corner = angle_from_pixel @ angle_from_pixel.transpose(0, 2, 1)
    return corner_noise_px * corner_noise_px * per_corner.reshape(len(angles), 4, 2, 2).sum(axis=1) / 16.0


def camera_pose_covariance(
    field_to_camera: np.ndarray,
    field_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    corner_noise_px: float = DEFAULT_CORNER_NOISE_PX,
) -> PoseWithCovariance | None:
    """Covariance of a camera pose estimated from the image corners of ``field_points``.

    Uses Napa's pose conventions: ``field_to_camera`` is the camera's 4x4 pose in the field, with WPILib camera axes
    (+X forward, +Y left, +Z up), and ``field_points`` is ``(K, 3)`` in the field frame, for example from
    ``geometry.tag_field_corners``. Returns ``None`` when the points don't constrain all six degrees of freedom.
    """

    camera_to_field = invert_transform(_checked_transform(field_to_camera, "field_to_camera"))
    (covariance,) = reprojection_covariances(
        camera_to_field,
        np.asarray(field_points, dtype=np.float64).reshape(-1, 3),
        camera_matrix,
        distortion_coefficients,
        validate_corner_noise(corner_noise_px),
        opencv_from_camera=WPILIB_TO_OPENCV_ROTATION,
    )
    if covariance is None:
        return None
    field_to_camera_covariance = PoseWithCovariance(camera_to_field, covariance).inverse().covariance
    return PoseWithCovariance(field_to_camera, cap_correlations(field_to_camera_covariance))
