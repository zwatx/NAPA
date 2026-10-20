"""Pose uncertainty for Kalman filters, linearized from corner reprojection.

A pose's 6x6 covariance is ordered ``[x, y, z, rx, ry, rz]``, as in ROS ``PoseWithCovariance``: the error in
translation, in the parent frame, then a small rotation about the parent frame's axes, applied on the left of the
estimated rotation (``R_true = exp([r]x) @ R``). Covariances come from the Gauss-Newton approximation
``sigma^2 (J^T J)^-1``, where ``J`` is the Jacobian of the projected tag corners with respect to the pose and
``sigma`` is the per-axis corner noise in pixels. A planar target's pose can also be a weighted set of solutions
(``mixture_pose``), and every reported covariance has its correlations capped (``cap_correlations``); docs/api.md
explains why.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from . import _import_cv2
from .geometry import WPILIB_TO_OPENCV_ROTATION, invert_transform

DEFAULT_CORNER_NOISE_PX = 1.0
# Fraction of each variance added back to the diagonal, capping every correlation at 1 / 1.1. Linearization can make
# a combination of axes look almost exactly known (a far or wide wall of tags, a lone tag at range) while the true
# error surface is curved; in Monte Carlo tests that left mean NEES at 50-90 for 6 dimensions with every per-axis
# variance right. Over 144 random wall geometries, this cap kept at most 9% of errors outside the 95% ellipse (0.02
# left up to 21%), at the cost of 5% on each axis's standard deviation.
CORRELATION_SLACK = 0.1
# A single tag's two solutions are weighted by their corner likelihood tempered by this factor,
# exp(-SSR / (2 T sigma^2)): the error surface between them isn't quadratic, and untempered weights left ambiguous
# poses overconfident in Monte Carlo tests.
LIKELIHOOD_TEMPERING = 2.0
# Steps along a multi-tag pose's mirror path are re-solved against the true points, so their likelihoods need less.
MIRROR_PATH_TEMPERING = 1.0
# Steps along the rotation from a planar target's solved pose (0) to its mirror solution (1), for multi-tag poses.
MIRROR_PATH_STEPS = (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
# The path is skipped when the mirror is both this many standard deviations away along its rotation and fits this
# many sigma^2 worse. Over 4,080 frames of 144 random walls, that skipped 61% of paths, and none of the skipped ones
# would have changed any standard deviation.
MIRROR_SKIP_SIGMAS = 30.0
MIRROR_SKIP_SSR = 100.0

Component = tuple[np.ndarray, np.ndarray, float]  # (4x4 transform, its own 6x6 covariance, weight)


# The generators of rotation: [v]x = sum_k v_k E_k, flattened so stacks of vectors map with one matrix product.
_GENERATORS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


def _skew(vectors: np.ndarray) -> np.ndarray:
    """The cross-product matrices ``[v]x`` of a 3-vector or a stack of them."""

    vectors = np.asarray(vectors, dtype=np.float64)
    return (vectors @ _GENERATORS).reshape(*vectors.shape[:-1], 3, 3)


def _block_diagonal(rotation: np.ndarray) -> np.ndarray:
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = rotation
    out[3:, 3:] = rotation
    return out


def _propagate(jacobian: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """``J C J^T``, symmetrized; either argument may be a stack."""

    out = jacobian @ covariance @ np.swapaxes(jacobian, -1, -2)
    return (out + np.swapaxes(out, -1, -2)) / 2.0


def cap_correlations(covariance: np.ndarray | None) -> np.ndarray | None:
    """Add ``CORRELATION_SLACK`` of each variance to the diagonal of a reported covariance (or a stack of them)."""

    if covariance is None:
        return None
    out = np.array(covariance, dtype=np.float64)
    diagonal = np.arange(out.shape[-1])
    out[..., diagonal, diagonal] *= 1.0 + CORRELATION_SLACK
    return out


def _invert_transforms(transforms: np.ndarray) -> np.ndarray:
    """``invert_transform`` of a stack of rigid transforms."""

    rotations_t = np.swapaxes(transforms[..., :3, :3], -1, -2)
    out = np.zeros_like(transforms)
    out[..., :3, :3] = rotations_t
    out[..., :3, 3] = -(rotations_t @ transforms[..., :3, 3, None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def _frozen(array: np.ndarray) -> np.ndarray:
    out = np.array(array, dtype=np.float64)
    out.setflags(write=False)
    return out


def _checked_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.array(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return transform


def validate_corner_noise(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError("corner_noise_px must be finite and greater than zero")
    return float(value)


def _rotation_vectors(rotations: np.ndarray) -> np.ndarray:
    """Rotation vectors of stacked rotation matrices, accurate for tiny angles (OpenCV's ``Rodrigues`` zeroes them)."""

    rotations = np.asarray(rotations, dtype=np.float64)
    shape = rotations.shape[:-2]
    rotations = rotations.reshape(-1, 3, 3)
    # The axis scaled by the sine is the antisymmetric part's vector: half of each generator's inner product with R.
    sine_axis = 0.5 * rotations.reshape(-1, 9) @ _GENERATORS.T
    sine = np.sqrt(np.einsum("mi,mi->m", sine_axis, sine_axis))
    cosine = (np.trace(rotations, axis1=1, axis2=2) - 1.0) / 2.0
    positive = sine > 0.0
    vectors = sine_axis * np.where(positive, np.arctan2(sine, cosine) / np.where(positive, sine, 1.0), 1.0)[:, None]
    # Near a half turn the antisymmetric part vanishes; OpenCV handles this case well.
    for index in np.flatnonzero((cosine < 0.0) & (sine < 1e-6)):
        vectors[index] = _import_cv2().Rodrigues(rotations[index])[0].reshape(3)
    return vectors.reshape(*shape, 3)


def _rotation_matrices(vectors: np.ndarray) -> np.ndarray:
    """``exp([v]x)`` for each row of an ``(N, 3)`` array, by Rodrigues' formula with stable small-angle terms."""

    vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    angles = np.linalg.norm(vectors, axis=1)
    small = angles < 1e-4
    safe = np.where(small, 1.0, angles)
    # sin(a)/a and (1 - cos(a))/a^2, by their series where the closed forms lose precision.
    first = np.where(small, 1.0 - angles**2 / 6.0, np.sin(safe) / safe)[:, None, None]
    second = np.where(small, 0.5 - angles**2 / 24.0, (1.0 - np.cos(safe)) / (safe * safe))[:, None, None]
    skew = _skew(vectors)
    return np.eye(3) + first * skew + second * skew @ skew


def _rotation_matrix(vector: np.ndarray) -> np.ndarray:
    """``exp([v]x)``, by Rodrigues' formula."""

    return _rotation_matrices(vector)[0]


def pose_difference(transform: np.ndarray, other: np.ndarray) -> np.ndarray:
    """``other``'s offset from ``transform`` in the covariance convention: ``[dt, r]`` with ``R_other = exp(r) R``.

    Either argument may be a stack of transforms; the result broadcasts like them.
    """

    transform, other = np.asarray(transform, dtype=np.float64), np.asarray(other, dtype=np.float64)
    rotation = _rotation_vectors(other[..., :3, :3] @ np.swapaxes(transform[..., :3, :3], -1, -2))
    return np.concatenate((other[..., :3, 3] - transform[..., :3, 3], rotation), axis=-1)


def _perturb(transform: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = transform.copy()
    out[:3, :3] = _rotation_matrix(delta[3:]) @ transform[:3, :3]
    out[:3, 3] += delta[:3]
    return out


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


def _inverse_jacobians(poses: np.ndarray) -> np.ndarray:
    """Jacobians of each inverted pose's error with respect to the pose's error, for a stack of poses."""

    rotations_t = np.swapaxes(poses[:, :3, :3], -1, -2)
    out = np.zeros((len(poses), 6, 6), dtype=np.float64)
    out[:, :3, :3] = out[:, 3:, 3:] = -rotations_t
    out[:, :3, 3:] = -rotations_t @ _skew(poses[:, :3, 3])
    return out


def _planar(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y, yaw)`` of a stack of poses and the Jacobians of those with respect to each pose's error."""

    heading = transforms[:, :3, 0]
    planar = heading[:, 0] * heading[:, 0] + heading[:, 1] * heading[:, 1]
    if np.any(planar < 1e-12):
        raise ValueError("yaw is undefined: the pose's +X axis is vertical")
    jacobians = np.zeros((len(transforms), 3, 6), dtype=np.float64)
    jacobians[:, 0, 0] = jacobians[:, 1, 1] = 1.0
    # A small rotation r turns the heading c by r x c; differentiate atan2(c_y, c_x).
    jacobians[:, 2, 3:] = (
        np.stack((-heading[:, 0] * heading[:, 2], -heading[:, 1] * heading[:, 2], planar), axis=1) / planar[:, None]
    )
    values = np.stack((transforms[:, 0, 3], transforms[:, 1, 3], np.arctan2(heading[:, 1], heading[:, 0])), axis=1)
    return values, jacobians


class _Components(NamedTuple):
    """The weighted solutions a pose was summarized from, stacked: ``(M, 4, 4)``, ``(M, 6, 6)`` and ``(M,)``."""

    transforms: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class PoseWithCovariance:
    """A 4x4 pose of a child frame in a parent frame, with the 6x6 covariance of its error.

    See the module docstring for the covariance's order and convention. The transforms passed to ``transform_by``
    and ``relative_to`` are treated as exact.

    A pose from a planar target can carry the weighted solutions it was summarized from. Then every operation moves
    each solution exactly and recomputes the spread between them, rather than linearizing a spread that may span tens
    of degrees. Otherwise operations propagate the covariance to first order. Both arrays are read-only copies.
    """

    transform: np.ndarray
    covariance: np.ndarray
    # Set only by mixture_pose and its batched form; dataclasses.replace() drops it, so a replaced covariance is never
    # overridden.
    _components: _Components | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        transform = _checked_transform(self.transform, "transform")
        covariance = np.array(self.covariance, dtype=np.float64)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be a finite 6x6 array")
        transform.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "covariance", covariance)

    def __reduce__(self) -> tuple:
        # Rebuild through the constructor, so copies and unpickled poses keep read-only arrays.
        components = None if self._components is None else tuple(np.array(part) for part in self._components)
        return _restored_pose, (np.array(self.transform), np.array(self.covariance), components)

    @property
    def translation(self) -> np.ndarray:
        return self.transform[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        return self.transform[:3, :3].copy()

    def std_devs(self) -> np.ndarray:
        """Standard deviations of ``[x, y, z, rx, ry, rz]``."""

        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    def _map(
        self, transform_fn: Callable[[np.ndarray], np.ndarray], jacobian_fn: Callable[[np.ndarray], np.ndarray]
    ) -> PoseWithCovariance:
        """Apply an exact map of poses and its error Jacobian, both taking a stack of transforms."""

        if self._components is None:
            transform = self.transform[None]
            return PoseWithCovariance(
                transform_fn(transform)[0], _propagate(jacobian_fn(transform), self.covariance[None])[0]
            )
        transforms, covariances, weights = self._components
        return _mixtures(
            transform_fn(transforms)[None], _propagate(jacobian_fn(transforms), covariances)[None], weights[None]
        )[0]

    def transform_by(self, transform: np.ndarray) -> PoseWithCovariance:
        """Return ``self.transform @ transform``, as WPILib's ``Pose3d.transformBy``.

        Use it to move to a rigidly attached frame: ``field_to_camera.transform_by(camera_to_robot)`` is the robot's
        pose in the field.
        """

        offset = _checked_transform(transform, "transform")

        def jacobian(poses: np.ndarray) -> np.ndarray:
            out = np.broadcast_to(np.eye(6), (len(poses), 6, 6)).copy()
            out[:, :3, 3:] = -_skew(poses[:, :3, :3] @ offset[:3, 3])
            return out

        return self._map(lambda poses: poses @ offset, jacobian)

    def relative_to(self, origin: np.ndarray) -> PoseWithCovariance:
        """Return this pose in ``origin``'s frame, ``inverse(origin) @ self.transform``, as ``Pose3d.relativeTo``.

        ``origin`` is the new frame's pose in the current parent frame, for example the red alliance's origin in
        the blue-origin field frame.
        """

        new_from_parent = invert_transform(_checked_transform(origin, "origin"))
        jacobian = _block_diagonal(new_from_parent[:3, :3])
        return self._map(lambda poses: new_from_parent @ poses, lambda poses: jacobian)

    def inverse(self) -> PoseWithCovariance:
        """Return the parent's pose in the child frame, with the covariance expressed in the child frame."""

        return self._map(_invert_transforms, _inverse_jacobians)

    def rebased(self, new_from_old: np.ndarray) -> PoseWithCovariance:
        """The same pose with both frames' axes rotated by ``new_from_old`` (3x3), e.g. from OpenCV to WPILib axes."""

        basis = np.eye(4)
        basis[:3, :3] = new_from_old
        jacobian = _block_diagonal(np.asarray(new_from_old, dtype=np.float64))
        return self._map(lambda poses: basis @ poses @ basis.T, lambda poses: jacobian)

    def to_pose2d(self) -> Pose2dWithCovariance:
        """Project onto the parent's X/Y plane: ``(x, y, yaw)`` with yaw counter-clockwise about +Z.

        Yaw is the heading of the child's +X axis. Raises ``ValueError`` when that axis points straight up or down.
        """

        (value,), (jacobian,) = _planar(self.transform[None])
        if self._components is None:
            covariance = _propagate(jacobian, self.covariance)
        else:
            transforms, covariances, weights = self._components
            values, jacobians = _planar(transforms)
            offsets = values - value
            offsets[:, 2] = (offsets[:, 2] + math.pi) % math.tau - math.pi
            spread = _propagate(jacobians, covariances) + offsets[:, :, None] * offsets[:, None, :]
            covariance = cap_correlations(np.tensordot(weights, spread, axes=1))
        return Pose2dWithCovariance(float(value[0]), float(value[1]), float(value[2]), covariance)


def _restored_pose(
    transform: np.ndarray, covariance: np.ndarray, components: tuple[np.ndarray, ...] | None
) -> PoseWithCovariance:
    pose = PoseWithCovariance(transform, covariance)
    if components is not None:
        object.__setattr__(pose, "_components", _Components(*(_frozen(part) for part in components)))
    return pose


def likelihood_weights(
    squared_residuals_px2: Sequence[float], corner_noise_px: float, tempering: float = LIKELIHOOD_TEMPERING
) -> np.ndarray:
    """Normalized, tempered corner likelihoods ``exp(-SSR / (2 T sigma^2))`` of competing solutions (the last axis)."""

    ssr = np.asarray(squared_residuals_px2, dtype=np.float64)
    log_weights = -(ssr - ssr.min(axis=-1, keepdims=True)) / (2.0 * tempering * corner_noise_px * corner_noise_px)
    weights = np.exp(log_weights)
    return weights / weights.sum(axis=-1, keepdims=True)


def _mixtures(transforms: np.ndarray, covariances: np.ndarray, weights: np.ndarray) -> list[PoseWithCovariance]:
    """``mixture_pose`` of each row of stacked components: ``(N, M, 4, 4)``, ``(N, M, 6, 6)`` and ``(N, M)``."""

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum(axis=1, keepdims=True)
    covariance = np.einsum("nm,nmij->nij", weights, covariances)
    if transforms.shape[1] > 1:
        offsets = pose_difference(transforms[:, :1], transforms[:, 1:])
        covariance += np.swapaxes(offsets * weights[:, 1:, None], 1, 2) @ offsets
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(transforms)):
        raise ValueError("covariance must be a finite 6x6 array")
    summaries = _frozen(cap_correlations(covariance))
    transforms, covariances, weights = _frozen(transforms), _frozen(covariances), _frozen(weights)
    poses = []
    for index, summary in enumerate(summaries):
        # The arrays are already checked, read-only copies, so skip the constructor's validation.
        pose = object.__new__(PoseWithCovariance)
        object.__setattr__(pose, "transform", transforms[index, 0])
        object.__setattr__(pose, "covariance", summary)
        object.__setattr__(pose, "_components", _Components(transforms[index], covariances[index], weights[index]))
        poses.append(pose)
    return poses


def mixture_pose(components: Sequence[Component]) -> PoseWithCovariance:
    """A pose summarizing weighted solutions: the first solution's transform, with the mixture's spread about it.

    Each component is a solution's transform, its own (linearized) covariance, and its weight. The reported
    covariance is the mixture's second moment about the first solution, with correlations capped. A single
    component is a plain linearized pose; carrying it as a component makes every operation cap correlations once,
    in the destination frame, instead of propagating a cap applied in another frame.
    """

    transforms, covariances, weights = zip(*components, strict=True)
    return _mixtures(
        np.asarray(transforms, dtype=np.float64)[None],
        np.asarray(covariances, dtype=np.float64)[None],
        np.asarray(weights, dtype=np.float64)[None],
    )[0]


def _camera_points(
    camera_from_points: np.ndarray, points: np.ndarray, opencv_from_camera: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Each pose's rotated points and its points in the OpenCV camera frame, both ``(M, K, 3)``."""

    poses = np.asarray(camera_from_points, dtype=np.float64).reshape(-1, 4, 4)
    rotated = np.asarray(points, dtype=np.float64) @ poses[:, :3, :3].transpose(0, 2, 1)
    camera_points = rotated + poses[:, None, :3, 3]
    if opencv_from_camera is not None:
        camera_points = camera_points @ np.asarray(opencv_from_camera, dtype=np.float64).T
    return rotated, camera_points


def _projection_terms(
    rotated: np.ndarray,
    camera_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    opencv_from_camera: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Projected pixels ``(M, K, 2)`` and their Jacobian ``(M, 2K, 6)``, from ``_camera_points``."""

    count = len(rotated)
    projected, jacobian = _import_cv2().projectPoints(
        camera_points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, distortion_coefficients
    )
    stacked = np.empty((count, rotated.shape[1], 2, 6))
    # With a zero pose, d(pixel)/d(tvec) is d(pixel)/d(point).
    pixel_from_point = stacked[..., :3]
    pixel_from_point[...] = jacobian[:, 3:6].reshape(count, -1, 2, 3)
    if opencv_from_camera is not None:
        pixel_from_point[...] = pixel_from_point @ np.asarray(opencv_from_camera, dtype=np.float64)
    # A small rotation r moves a point p by r x p, so each pixel row a gains a . (r x p) = (p x a) . r.
    x, y, z = (rotated[..., None, axis] for axis in range(3))
    a, b, c = (pixel_from_point[..., axis] for axis in range(3))
    stacked[..., 3], stacked[..., 4], stacked[..., 5] = y * c - z * b, z * a - x * c, x * b - y * a
    return projected.reshape(count, -1, 2), stacked.reshape(count, -1, 6)


def _reprojection_terms(
    camera_from_points: np.ndarray,
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    opencv_from_camera: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Projected pixels ``(M, K, 2)`` and their Jacobian ``(M, 2K, 6)`` with respect to each pose's error."""

    rotated, camera_points = _camera_points(camera_from_points, points, opencv_from_camera)
    return _projection_terms(rotated, camera_points, camera_matrix, distortion_coefficients, opencv_from_camera)


def _covariances_from_jacobians(jacobians: np.ndarray, corner_noise_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Stacked covariances and whether each is defined (the corners constrain all six degrees of freedom)."""

    information = np.swapaxes(jacobians, 1, 2) @ jacobians
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    valid = eigenvalues[:, 0] > 1e-12 * eigenvalues[:, -1]
    scale = corner_noise_px * corner_noise_px / np.where(valid[:, None], eigenvalues, 1.0)
    return (eigenvectors * scale[:, None, :]) @ eigenvectors.transpose(0, 2, 1), valid


def reprojection_covariances(
    camera_from_points: np.ndarray,
    points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    corner_noise_px: float,
    *,
    opencv_from_camera: np.ndarray | None = None,
) -> list[np.ndarray | None]:
    """Linearized covariances of poses that map ``points`` into the camera frame, one per pose.

    ``camera_from_points`` is ``(M, 4, 4)`` and ``points`` is ``(K, 3)``, shared by every pose, or ``(M, K, 3)``.
    The camera frame is OpenCV's unless ``opencv_from_camera`` rotates it there. Each covariance is in the camera
    frame; it is ``None`` where the corners don't constrain all six degrees of freedom.
    """

    _, jacobians = _reprojection_terms(
        camera_from_points, points, camera_matrix, distortion_coefficients, opencv_from_camera
    )
    covariances, valid = _covariances_from_jacobians(jacobians, corner_noise_px)
    return [covariance if ok else None for covariance, ok in zip(covariances, valid, strict=True)]


def _reprojection_fit(
    camera_from_points: np.ndarray,
    points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    corner_noise_px: float,
    opencv_from_camera: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``reprojection_covariances``, stacked with whether each is defined, and each pose's squared corner residuals.

    ``image_points`` is ``(K, 2)``, shared by every pose, or ``(M, K, 2)``; the residuals are summed in px^2.
    """

    projected, jacobians = _reprojection_terms(
        camera_from_points, points, camera_matrix, distortion_coefficients, opencv_from_camera
    )
    covariances, valid = _covariances_from_jacobians(jacobians, corner_noise_px)
    residuals = projected - np.asarray(image_points, dtype=np.float64).reshape(-1, projected.shape[1], 2)
    return covariances, valid, np.sum(residuals * residuals, axis=(1, 2))


def _pair_mixtures(
    transforms: np.ndarray, covariances: np.ndarray, squared_residuals: np.ndarray, corner_noise_px: float
) -> list[tuple[PoseWithCovariance, PoseWithCovariance]]:
    """Each of a planar target's two solutions as a pose whose covariance also covers the other, for stacked pairs.

    Takes ``(P, 2, 4, 4)`` transforms, ``(P, 2, 6, 6)`` covariances and ``(P, 2)`` squared residuals, and returns each
    pair's two poses. Solutions are weighted by ``likelihood_weights``. A clearly worse alternative adds nothing; an
    equally good one adds the full spread, which also covers the linearization's failure when the target is seen
    nearly head-on.
    """

    weights = likelihood_weights(squared_residuals, corner_noise_px)
    both = _mixtures(
        np.concatenate((transforms, transforms[:, ::-1])),
        np.concatenate((covariances, covariances[:, ::-1])),
        np.concatenate((weights, weights[:, ::-1])),
    )
    return list(zip(both[: len(transforms)], both[len(transforms) :], strict=True))


def mirror_path_pose(
    camera_from_points: np.ndarray,
    mirror: np.ndarray,
    points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    corner_noise_px: float,
    *,
    opencv_from_camera: np.ndarray | None = None,
    iterations: int = 3,
) -> PoseWithCovariance | None:
    """A planar target's pose, with a covariance that follows its error surface toward the mirror solution.

    Steps along the rotation from the solved pose to ``mirror`` (``MIRROR_PATH_STEPS``), re-solving the other five
    degrees of freedom against the true points at each step, and weights each step by its corner likelihood. That
    covers both a separate mirror minimum and the long, flat valley a target has when seen nearly head-on, which a
    single unrefined mirror candidate misjudges. Returns ``None`` when the covariance is undefined.
    """

    intrinsics = (camera_matrix, distortion_coefficients)
    targets = np.asarray(image_points, dtype=np.float64).reshape(-1)
    start = np.asarray(camera_from_points, dtype=np.float64)
    mirror = np.asarray(mirror, dtype=np.float64)
    direction = pose_difference(start, mirror)
    angle = float(np.linalg.norm(direction[3:]))
    projected, jacobians = _reprojection_terms(np.stack((start, mirror)), points, *intrinsics, opencv_from_camera)
    linears, valid = _covariances_from_jacobians(jacobians[:1], corner_noise_px)
    if not valid[0]:
        return None
    linear = linears[0]
    if angle < 1e-9:
        return mixture_pose([(start, linear, 1.0)])
    axis_variance = float(direction[3:] @ linear[3:, 3:] @ direction[3:]) / (angle * angle)
    start_ssr, mirror_ssr = np.sum((projected - targets.reshape(1, -1, 2)) ** 2, axis=(1, 2))
    if (
        angle > MIRROR_SKIP_SIGMAS * math.sqrt(axis_variance)
        and mirror_ssr - start_ssr > MIRROR_SKIP_SSR * corner_noise_px * corner_noise_px
    ):
        return mixture_pose([(start, linear, 1.0)])
    # Free directions: all of translation, and rotation about the two axes perpendicular to the path's.
    axis = direction[3:] / angle
    _, _, frame = np.linalg.svd(axis.reshape(1, 3))
    free = np.zeros((6, 5))
    free[:3, :3] = np.eye(3)
    free[3:, 3:] = frame[1:].T

    def squared_errors(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Squared error, residuals, and Jacobians of each pose; infinite error for poses that can't be projected."""

        rotated, camera_points = _camera_points(poses, points, opencv_from_camera)
        # Projecting points behind the camera gives meaningless pixels.
        valid = np.all(camera_points[..., 2] > 1e-3, axis=1) & np.all(np.isfinite(camera_points), axis=(1, 2))
        if valid.all():
            projected, jacobians = _projection_terms(rotated, camera_points, *intrinsics, opencv_from_camera)
            residuals = targets - projected.reshape(len(poses), -1)
        else:
            residuals = np.zeros((len(poses), len(targets)))
            jacobians = np.zeros((len(poses), len(targets), 6))
            if valid.any():
                projected, jacobians[valid] = _projection_terms(
                    rotated[valid], camera_points[valid], *intrinsics, opencv_from_camera
                )
                residuals[valid] = targets - projected.reshape(int(valid.sum()), -1)
        errors = np.where(valid, np.einsum("mi,mi->m", residuals, residuals), np.inf)
        errors[~np.isfinite(errors)] = np.inf
        return errors, residuals, jacobians

    def perturbed(poses: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        out = poses.copy()
        out[:, :3, :3] = _rotation_matrices(deltas[:, 3:]) @ poses[:, :3, :3]
        out[:, :3, 3] += deltas[:, :3]
        return out

    # Start each step on the straight path between the two solutions, then run damped Gauss-Newton on the free
    # directions for every step at once.
    steps = np.array(sorted(MIRROR_PATH_STEPS, key=lambda step: step != 0.0))  # the solved pose first
    poses = np.repeat(start[None], len(steps), axis=0)
    poses[:, :3, :3] = _rotation_matrices(steps[:, None] * direction[3:]) @ start[:3, :3]
    poses[:, :3, 3] = start[:3, 3] + steps[:, None] * direction[:3]
    errors, residuals, jacobians = squared_errors(poses)
    if not np.isfinite(errors[0]):
        return mixture_pose([(start, linear, 1.0)])
    active = (steps != 0.0) & np.isfinite(errors)  # steps that still improve; the solved pose stays fixed
    for _ in range(iterations):
        if not active.any():
            break
        reduced = jacobians[active] @ free
        normal = np.swapaxes(reduced, 1, 2) @ reduced
        # Levenberg-style damping scaled to each step's own information keeps a near-singular step well posed.
        damping = 1e-9 * np.trace(normal, axis1=1, axis2=2)[:, None, None] / 5.0 + 1e-12
        rhs = np.swapaxes(reduced, 1, 2) @ residuals[active][..., None]
        updates = np.zeros((len(steps), 5))
        updates[active] = np.linalg.solve(normal + damping * np.eye(5), rhs)[..., 0]
        pending = active.copy()
        for scale in (1.0, 0.5, 0.25):
            candidates = perturbed(poses[pending], scale * updates[pending] @ free.T)
            candidate_errors, candidate_residuals, candidate_jacobians = squared_errors(candidates)
            better = candidate_errors < errors[pending]
            indices = np.flatnonzero(pending)[better]
            poses[indices], errors[indices] = candidates[better], candidate_errors[better]
            residuals[indices], jacobians[indices] = candidate_residuals[better], candidate_jacobians[better]
            pending[indices] = False
            if not pending.any():
                break
        active &= ~pending  # a step no damped update improves has converged
    usable = np.isfinite(errors)
    covariances, valid = _covariances_from_jacobians(jacobians[usable], corner_noise_px)
    if not valid[0]:
        return None
    weights = likelihood_weights(errors[usable], corner_noise_px, MIRROR_PATH_TEMPERING)
    return _mixtures(poses[usable][valid][None], covariances[valid][None], weights[valid][None])[0]


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
    return mixture_pose([(camera_to_field, covariance, 1.0)]).inverse()
