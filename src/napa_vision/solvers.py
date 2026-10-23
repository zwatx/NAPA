from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import cv2
import numpy as np

from .calibration import undistorted_corner_angles
from .covariance import (
    DEFAULT_CORNER_NOISE_PX,
    PoseWithCovariance,
    _inverse_jacobians,
    _invert_transforms,
    _mixtures,
    _pair_mixtures,
    _propagate,
    _reprojection_fit,
    _rotation_matrices,
    center_angle_covariances,
    mirror_path_pose,
    mixture_pose,
    reprojection_covariances,
    validate_corner_noise,
)
from .geometry import (
    WPILIB_TO_OPENCV_ROTATION,
    invert_transform,
    opencv_pose_to_wpilib_transform,
    opencv_translation_to_wpilib,
    tag_field_corners,
    tag_object_points,
    wpilib_points_to_opencv,
)
from .types import (
    CameraPoseBounds,
    CameraPoseObservation,
    FiducialImageObservation,
    PoseSolveDiagnostics,
    SingleTagPoseObservation,
    TagAngleObservation,
    fingerprint,
)


class CameraGeometry(Protocol):
    """The image geometry solvers need, independent of capture configuration."""

    @property
    def calibration_matrix(self) -> np.ndarray: ...

    @property
    def distortion_coefficients(self) -> np.ndarray: ...


Distributions = tuple[PoseWithCovariance | None, PoseWithCovariance | None]


@dataclass(frozen=True, slots=True)
class _Intrinsics:
    """A read-only snapshot of a camera's geometry, so deferred covariances don't see later changes to the camera."""

    calibration_matrix: np.ndarray
    distortion_coefficients: np.ndarray

    @classmethod
    def of(cls, camera: CameraGeometry) -> _Intrinsics:
        matrix = np.array(camera.calibration_matrix, dtype=np.float64)
        distortion = np.array(camera.distortion_coefficients, dtype=np.float64)
        matrix.setflags(write=False)
        distortion.setflags(write=False)
        return cls(matrix, distortion)


class _SingleTagCovariances:
    """Covariances of every tag from one ``solve_many`` call, computed together the first time any is read."""

    def __init__(
        self,
        solutions: list[list[tuple[np.ndarray, np.ndarray]]],
        corners: list[np.ndarray],
        object_points: np.ndarray,
        camera: CameraGeometry,
        corner_noise_px: float,
    ) -> None:
        self._inputs: tuple | None = (solutions, corners, object_points, camera, corner_noise_px)
        self._results: list[Distributions] = []

    def __call__(self, index: int) -> Distributions:
        # Read the inputs once: another thread may consume them meanwhile. Results are published before inputs clear.
        inputs = self._inputs
        if inputs is not None:
            self._results = self._compute(*inputs)
            self._inputs = None
        return self._results[index]

    @staticmethod
    def _compute(
        solutions: list[list[tuple[np.ndarray, np.ndarray]]],
        corners: list[np.ndarray],
        object_points: np.ndarray,
        camera: CameraGeometry,
        corner_noise_px: float,
    ) -> list[Distributions]:
        counts = [len(pair) for pair in solutions]
        results: list[Distributions] = [(None, None)] * len(solutions)
        if not any(counts):
            return results
        vectors = np.array([(rvec, tvec) for pair in solutions for rvec, tvec in pair], dtype=np.float64)
        transforms = np.zeros((len(vectors), 4, 4))
        transforms[:, :3, :3] = _rotation_matrices(vectors[:, 0])
        transforms[:, :3, 3] = vectors[:, 1].reshape(-1, 3)
        transforms[:, 3, 3] = 1.0
        image_points = np.stack([image for image, count in zip(corners, counts, strict=True) for _ in range(count)])
        covariances, valid, squared_residuals = _reprojection_fit(
            transforms,
            object_points,
            image_points,
            camera.calibration_matrix,
            camera.distortion_coefficients,
            corner_noise_px,
        )
        firsts = np.cumsum([0, *counts[:-1]])
        # Both solutions defined: each one's pose covers the other. Otherwise a lone defined first solution is plain.
        paired = [tag for tag, count in enumerate(counts) if count == 2 and valid[firsts[tag] : firsts[tag] + 2].all()]
        alone = [tag for tag, count in enumerate(counts) if count and valid[firsts[tag]] and tag not in paired]
        if paired:
            rows = firsts[paired][:, None] + np.arange(2)
            pairs = _pair_mixtures(transforms[rows], covariances[rows], squared_residuals[rows], corner_noise_px)
            for tag, pair in zip(paired, pairs, strict=True):
                results[tag] = pair
        if alone:
            rows = firsts[alone][:, None]
            singles = _mixtures(transforms[rows], covariances[rows], np.ones((len(alone), 1)))
            for tag, single in zip(alone, singles, strict=True):
                results[tag] = (single, None)
        return results


class _CenterAngleCovariances:
    """Centre-angle covariances of every tag from one ``TagAngleSolver.solve_many`` call, computed on first read."""

    def __init__(self, corner_angles: np.ndarray, camera: CameraGeometry, corner_noise_px: float) -> None:
        self._inputs: tuple | None = (corner_angles, camera, corner_noise_px)
        self._results = np.empty((0, 2, 2))

    def __call__(self, index: int) -> np.ndarray:
        inputs = self._inputs  # read once, as in _SingleTagCovariances
        if inputs is not None:
            corner_angles, camera, corner_noise_px = inputs
            results = center_angle_covariances(
                corner_angles, camera.calibration_matrix, camera.distortion_coefficients, corner_noise_px
            )
            results.setflags(write=False)
            self._results = results
            self._inputs = None
        return self._results[index]


def _single_tag_camera_distributions(
    field_to_cameras: tuple[np.ndarray, np.ndarray],
    field_points: np.ndarray,
    corners: np.ndarray,
    camera: CameraGeometry,
    corner_noise_px: float,
) -> Distributions:
    """The camera's two poses in the field, each covering the other; solutions are inverted exactly before mixing."""

    field_to_camera = np.stack(field_to_cameras)
    camera_to_field = _invert_transforms(field_to_camera)
    covariances, valid, squared_residuals = _reprojection_fit(
        camera_to_field,
        field_points,
        corners,
        camera.calibration_matrix,
        camera.distortion_coefficients,
        corner_noise_px,
        WPILIB_TO_OPENCV_ROTATION,
    )
    if not valid.all():
        return None, None
    covariances = _propagate(_inverse_jacobians(camera_to_field), covariances)
    ((first, second),) = _pair_mixtures(
        field_to_camera[None], covariances[None], squared_residuals[None], corner_noise_px
    )
    return first, second


def _multi_tag_distributions(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera: CameraGeometry,
    corner_noise_px: float,
) -> Distributions:
    """The camera's pose in the field, following the error surface toward a planar target's mirror solution."""

    camera_to_field = opencv_pose_to_wpilib_transform(rvec, tvec)
    field_points = object_points @ WPILIB_TO_OPENCV_ROTATION
    intrinsics = (camera.calibration_matrix, camera.distortion_coefficients)
    mirror = CameraPoseSolver._planar_alternative(object_points, image_points, rvec, tvec, camera)
    if mirror is None:
        (covariance,) = reprojection_covariances(
            camera_to_field, field_points, *intrinsics, corner_noise_px, opencv_from_camera=WPILIB_TO_OPENCV_ROTATION
        )
        if covariance is None:
            return None, None
        return mixture_pose([(camera_to_field, covariance, 1.0)]).inverse(), None
    pose = mirror_path_pose(
        camera_to_field,
        opencv_pose_to_wpilib_transform(*mirror),
        field_points,
        image_points,
        *intrinsics,
        corner_noise_px,
        opencv_from_camera=WPILIB_TO_OPENCV_ROTATION,
    )
    return (None if pose is None else pose.inverse()), None


def _solution_error(
    errors: np.ndarray | None,
    index: int,
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera: CameraGeometry,
) -> float:
    if errors is not None and index < len(errors):
        value = float(np.asarray(errors[index]).reshape(-1)[0])
        if np.isfinite(value):
            return value
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera.calibration_matrix,
        camera.distortion_coefficients,
    )
    residuals = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def _valid_solution(rvec: np.ndarray, tvec: np.ndarray, error: float, max_error: float) -> bool:
    return (
        bool(np.isfinite(rvec).all() and np.isfinite(tvec).all()) and math.isfinite(error) and 0.0 <= error <= max_error
    )


class SingleTagPoseSolver:
    """Solve each tag's two planar (IPPE) poses, with covariances for ``corner_noise_px`` of corner noise."""

    def __init__(
        self, max_reprojection_error_px: float = 5.0, *, corner_noise_px: float = DEFAULT_CORNER_NOISE_PX
    ) -> None:
        self.max_reprojection_error_px = max_reprojection_error_px
        self.corner_noise_px = validate_corner_noise(corner_noise_px)

    def solve(
        self,
        observation: FiducialImageObservation,
        camera: CameraGeometry,
        tag_size_m: float,
    ) -> SingleTagPoseObservation | None:
        """Return the tag's pose, or ``None`` when no solution is in front of the camera within the error limit."""

        return self.solve_many([observation], camera, tag_size_m)[0]

    def solve_many(
        self,
        observations: Sequence[FiducialImageObservation],
        camera: CameraGeometry,
        tag_size_m: float,
    ) -> list[SingleTagPoseObservation | None]:
        """Solve every tag. Results match ``observations`` by index.

        Covariances are deferred: the first one read computes every tag's in one batch.
        """

        if not observations:
            return []
        object_points = tag_object_points(tag_size_m)
        solved = [self._ippe_solutions(observation, camera, object_points) for observation in observations]
        covariances = _SingleTagCovariances(
            [[(np.array(rvec), np.array(tvec)) for rvec, tvec, _ in solutions] for solutions in solved],
            [np.array(observation.corners, dtype=np.float64) for observation in observations],
            object_points,
            _Intrinsics.of(camera),
            self.corner_noise_px,
        )
        results: list[SingleTagPoseObservation | None] = []
        for index, (observation, solutions) in enumerate(zip(observations, solved, strict=True)):
            if not solutions:
                results.append(None)
                continue
            (pose_0_rvec, pose_0_tvec, error_0), *alternative = solutions
            pose_1_rvec, pose_1_tvec, error_1 = alternative[0] if alternative else (None, None, None)
            single = SingleTagPoseObservation(
                tag_id=observation.tag_id,
                pose_0_rvec=pose_0_rvec,
                pose_0_tvec=pose_0_tvec,
                error_0=error_0,
                pose_1_rvec=pose_1_rvec,
                pose_1_tvec=pose_1_tvec,
                error_1=error_1,
                distance=float(np.linalg.norm(pose_0_tvec)),
            )
            single._defer(
                partial(covariances, index),
                (fingerprint(pose_0_rvec, pose_0_tvec), fingerprint(pose_1_rvec, pose_1_tvec)),
            )
            results.append(single)
        return results

    def _ippe_solutions(
        self, observation: FiducialImageObservation, camera: CameraGeometry, object_points: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Up to two valid IPPE solutions, best first; empty when the tag can't be solved."""

        corners = np.asarray(observation.corners, dtype=np.float64)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            return []
        try:
            ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                object_points,
                corners,
                camera.calibration_matrix,
                camera.distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            return []
        if not ok or len(rvecs) == 0:
            return []

        solutions: list[tuple[np.ndarray, np.ndarray, float]] = []
        for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs, strict=False)):
            error = _solution_error(errors, index, object_points, corners, rvec, tvec, camera)
            if _valid_solution(rvec, tvec, error, float("inf")) and float(tvec[2][0]) > 0.0:
                solutions.append((rvec, tvec, error))
        solutions.sort(key=lambda item: item[2])
        if not solutions or solutions[0][2] > self.max_reprojection_error_px:
            return []
        return solutions[:2]


class CameraPoseSolver:
    """Estimate a WPILib field pose with tag-level quality diagnostics."""

    def __init__(
        self,
        max_reprojection_error_px: float = 5.0,
        *,
        max_tag_reprojection_error_px: float | None = None,
        bounds: CameraPoseBounds | None = None,
        max_outlier_iterations: int = 8,
        ransac_iterations: int = 100,
        corner_noise_px: float = DEFAULT_CORNER_NOISE_PX,
    ) -> None:
        if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px <= 0.0:
            raise ValueError("max_reprojection_error_px must be finite and greater than zero")
        tag_limit = (
            max_reprojection_error_px if max_tag_reprojection_error_px is None else max_tag_reprojection_error_px
        )
        if not np.isfinite(tag_limit) or tag_limit <= 0.0:
            raise ValueError("max_tag_reprojection_error_px must be finite and greater than zero")
        if isinstance(max_outlier_iterations, bool) or max_outlier_iterations < 0:
            raise ValueError("max_outlier_iterations must be a non-negative integer")
        if isinstance(ransac_iterations, bool) or ransac_iterations < 1:
            raise ValueError("ransac_iterations must be a positive integer")
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.max_tag_reprojection_error_px = float(tag_limit)
        self.bounds = bounds
        self.max_outlier_iterations = max_outlier_iterations
        self.ransac_iterations = ransac_iterations
        self.corner_noise_px = validate_corner_noise(corner_noise_px)
        self.last_diagnostics = PoseSolveDiagnostics()
        self._field_corner_cache: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}

    def solve(
        self,
        observations: Sequence[FiducialImageObservation],
        single_tag_poses: Sequence[SingleTagPoseObservation],
        camera: CameraGeometry,
        tag_size_m: float,
        tag_layout: dict[int, np.ndarray],
    ) -> CameraPoseObservation | None:
        diagnostics = PoseSolveDiagnostics(attempted_tag_ids=[observation.tag_id for observation in observations])
        self.last_diagnostics = diagnostics
        known: list[FiducialImageObservation] = []
        seen_ids: set[int] = set()
        for observation in observations:
            if observation.tag_id not in tag_layout:
                diagnostics.rejected_tag_reasons[observation.tag_id] = "tag is not present in the field layout"
                continue
            try:
                corners = np.asarray(observation.corners, dtype=np.float64)
            except (TypeError, ValueError):
                diagnostics.rejected_tag_reasons[observation.tag_id] = "corners must contain finite numbers"
                continue
            if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
                diagnostics.rejected_tag_reasons[observation.tag_id] = "corners must be a finite 4x2 array"
                continue
            if observation.tag_id in seen_ids:
                diagnostics.rejected_tag_reasons[observation.tag_id] = "duplicate observation ignored"
                continue
            known.append(observation)
            seen_ids.add(observation.tag_id)
        if not known:
            diagnostics.rejection_reason = "no unique observed tags are present in the field layout"
            return None
        if len(known) == 1:
            single_by_id = {pose.tag_id: pose for pose in single_tag_poses}
            return self._solve_single(
                known[0],
                single_by_id.get(known[0].tag_id),
                camera,
                tag_size_m,
                tag_layout,
                diagnostics,
            )
        return self._solve_multi(known, camera, tag_size_m, tag_layout, diagnostics)

    def _solve_single(
        self,
        observation: FiducialImageObservation,
        single: SingleTagPoseObservation | None,
        camera: CameraGeometry,
        tag_size_m: float,
        tag_layout: dict[int, np.ndarray],
        diagnostics: PoseSolveDiagnostics,
    ) -> CameraPoseObservation | None:
        if single is None:
            diagnostics.rejection_reason = "single-tag IPPE solution is unavailable"
            return None
        if single.pose_1_rvec is None or single.pose_1_tvec is None or single.error_1 is None:
            diagnostics.rejection_reason = "single-tag pose does not contain both ambiguity candidates"
            return None
        diagnostics.per_tag_rms_reprojection_error_px[observation.tag_id] = single.error_0
        diagnostics.overall_rms_reprojection_error_px = single.error_0
        if single.error_0 > self.max_reprojection_error_px or single.error_1 > self.max_reprojection_error_px:
            diagnostics.rejection_reason = (
                f"single-tag ambiguity candidate exceeds {self.max_reprojection_error_px:.3f}px reprojection limit"
            )
            return None
        field_to_tag = tag_layout[observation.tag_id]
        camera_to_tag_0 = opencv_pose_to_wpilib_transform(single.pose_0_rvec, single.pose_0_tvec)
        camera_to_tag_1 = opencv_pose_to_wpilib_transform(single.pose_1_rvec, single.pose_1_tvec)
        field_to_camera_0 = field_to_tag @ invert_transform(camera_to_tag_0)
        field_to_camera_1 = field_to_tag @ invert_transform(camera_to_tag_1)
        if self.bounds is not None:
            for index, pose in enumerate((field_to_camera_0, field_to_camera_1)):
                rejection_reason = self.bounds.rejection_reason(pose[:3, 3])
                if rejection_reason is not None:
                    diagnostics.rejection_reason = f"single-tag candidate {index}: {rejection_reason}"
                    return None
        diagnostics.accepted_tag_ids = [observation.tag_id]
        diagnostics.iterations = 1
        confidence = 1.0 / (1.0 + single.error_0 + 0.1 * single.distance)
        result = CameraPoseObservation(
            tag_ids=[observation.tag_id],
            field_to_camera_0=field_to_camera_0,
            error_0=single.error_0,
            field_to_camera_1=field_to_camera_1,
            error_1=single.error_1,
            average_tag_distance=single.distance,
            confidence=confidence,
            diagnostics=diagnostics,
        )
        result._defer(
            partial(
                _single_tag_camera_distributions,
                (field_to_camera_0.copy(), field_to_camera_1.copy()),
                tag_field_corners(field_to_tag, tag_size_m),
                np.array(observation.corners, dtype=np.float64),
                _Intrinsics.of(camera),
                self.corner_noise_px,
            ),
            (fingerprint(field_to_camera_0), fingerprint(field_to_camera_1)),
        )
        return result

    @staticmethod
    def _planar_alternative(
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera: CameraGeometry,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The mirror solution of a (nearly) planar target's pose, as an OpenCV ``rvec``/``tvec``.

        Tags on one wall form one plane, which has the same two-fold ambiguity as a single tag. IPPE on the
        flattened points gives both candidates, and the one farther from the solved pose is returned. It is only a
        rough guide to where the mirror lies: ``mirror_path_pose`` re-solves the path toward it against the true
        points.
        """

        centroid = object_points.mean(axis=0)
        _, singular_values, plane = np.linalg.svd(object_points - centroid, full_matrices=False)
        if singular_values[2] > 0.25 * singular_values[0]:
            return None  # Depth across the target breaks the ambiguity.
        if np.linalg.det(plane) < 0.0:
            plane[2] *= -1.0
        plane_points = (object_points - centroid) @ plane.T
        plane_points[:, 2] = 0.0
        intrinsics = (camera.calibration_matrix, camera.distortion_coefficients)
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(plane_points, image_points, *intrinsics, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return None
        if not ok:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for plane_rvec, plane_tvec in zip(rvecs, tvecs, strict=False):
            candidate_rotation = cv2.Rodrigues(plane_rvec)[0] @ plane
            candidate_rvec = cv2.Rodrigues(candidate_rotation)[0]
            candidate_tvec = (plane_tvec.reshape(3) - candidate_rotation @ centroid).reshape(3, 1)
            if not _valid_solution(candidate_rvec, candidate_tvec, 0.0, 0.0):
                continue
            if not CameraPoseSolver._points_in_front(object_points, candidate_rvec, candidate_tvec):
                continue
            relative = cv2.Rodrigues(candidate_rvec)[0] @ rotation.T
            angle = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
            if best is None or angle > best[0]:
                best = (angle, candidate_rvec, candidate_tvec)
        return None if best is None else (best[1], best[2])

    def _multi_points(
        self,
        observations: Sequence[FiducialImageObservation],
        tag_size_m: float,
        tag_layout: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        object_points = np.empty((len(observations) * 4, 3), dtype=np.float64)
        image_points = np.empty((len(observations) * 4, 2), dtype=np.float64)
        for index, observation in enumerate(observations):
            start = index * 4
            object_points[start : start + 4] = self._cached_field_corners(observation.tag_id, tag_size_m, tag_layout)
            image_points[start : start + 4] = np.asarray(observation.corners, dtype=np.float64).reshape(4, 2)
        return object_points, image_points

    def _cached_field_corners(
        self,
        tag_id: int,
        tag_size_m: float,
        tag_layout: dict[int, np.ndarray],
    ) -> np.ndarray:
        transform = np.asarray(tag_layout[tag_id], dtype=np.float64)
        cached = self._field_corner_cache.get(tag_id)
        if cached is not None and cached[0] == tag_size_m and np.array_equal(cached[1], transform):
            return cached[2]
        corners = wpilib_points_to_opencv(tag_field_corners(transform, tag_size_m))
        self._field_corner_cache[tag_id] = (tag_size_m, transform.copy(), corners)
        return corners

    @staticmethod
    def _points_in_front(object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> bool:
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = (rotation @ object_points.T).T + np.asarray(tvec).reshape(1, 3)
        return bool(np.all(camera_points[:, 2] > 0.0))

    @staticmethod
    def _residuals(
        observations: Sequence[FiducialImageObservation],
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera: CameraGeometry,
    ) -> tuple[dict[int, float], float]:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera.calibration_matrix,
            camera.distortion_coefficients,
        )
        squared = np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)
        tag_errors = np.sqrt(squared.reshape(-1, 4).mean(axis=1)).tolist()
        per_tag = {observation.tag_id: error for observation, error in zip(observations, tag_errors, strict=True)}
        return per_tag, float(np.sqrt(squared.mean()))

    def _generic_candidate(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera: CameraGeometry,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        flags = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_ITERATIVE
        try:
            ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                object_points,
                image_points,
                camera.calibration_matrix,
                camera.distortion_coefficients,
                flags=flags,
            )
        except cv2.error:
            return None
        if not ok or len(rvecs) == 0:
            return None
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for index, (candidate_rvec, candidate_tvec) in enumerate(zip(rvecs, tvecs, strict=False)):
            error = _solution_error(
                errors,
                index,
                object_points,
                image_points,
                candidate_rvec,
                candidate_tvec,
                camera,
            )
            if _valid_solution(candidate_rvec, candidate_tvec, error, float("inf")) and self._points_in_front(
                object_points, candidate_rvec, candidate_tvec
            ):
                candidates.append((error, candidate_rvec, candidate_tvec))
        if not candidates:
            return None
        _, rvec, tvec = min(candidates, key=lambda candidate: candidate[0])
        return rvec, tvec

    def _ransac_candidate(
        self,
        observations: Sequence[FiducialImageObservation],
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera: CameraGeometry,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if len(observations) < 3:
            return None
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera.calibration_matrix,
                camera.distortion_coefficients,
                iterationsCount=self.ransac_iterations,
                reprojectionError=self.max_tag_reprojection_error_px,
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            return None
        if not ok or rvec is None or tvec is None or inliers is None:
            return None
        inlier_counts = [0] * len(observations)
        for point_index in np.asarray(inliers).reshape(-1):
            inlier_counts[int(point_index) // 4] += 1
        if sum(count >= 3 for count in inlier_counts) < 2:
            return None
        if not _valid_solution(rvec, tvec, 0.0, float("inf")) or not self._points_in_front(object_points, rvec, tvec):
            return None
        return rvec, tvec

    def _solve_multi(
        self,
        observations: Sequence[FiducialImageObservation],
        camera: CameraGeometry,
        tag_size_m: float,
        tag_layout: dict[int, np.ndarray],
        diagnostics: PoseSolveDiagnostics,
    ) -> CameraPoseObservation | None:
        active = list(observations)
        outlier_iterations = 0
        while len(active) >= 2:
            diagnostics.iterations += 1
            object_points, image_points = self._multi_points(active, tag_size_m, tag_layout)
            if not np.all(np.isfinite(image_points)):
                diagnostics.rejection_reason = "one or more tag corners are non-finite"
                return None

            candidate = self._generic_candidate(object_points, image_points, camera)
            used_ransac_candidate = False
            if candidate is None:
                candidate = self._ransac_candidate(active, object_points, image_points, camera)
                if candidate is None:
                    diagnostics.rejection_reason = "OpenCV could not produce a physically valid multi-tag pose"
                    return None
                diagnostics.used_ransac = True
                used_ransac_candidate = True
            rvec, tvec = candidate
            per_tag, error = self._residuals(active, object_points, image_points, rvec, tvec, camera)

            # Clean frames need only the all-corner generic solve. If it does
            # not meet the acceptance thresholds, use RANSAC to identify a
            # coherent subset before choosing which tag to remove.
            if not used_ransac_candidate and (
                max(per_tag.values()) > self.max_tag_reprojection_error_px or error > self.max_reprojection_error_px
            ):
                robust = self._ransac_candidate(active, object_points, image_points, camera)
                if robust is not None:
                    diagnostics.used_ransac = True
                    rvec, tvec = robust
                    per_tag, error = self._residuals(active, object_points, image_points, rvec, tvec, camera)

            diagnostics.per_tag_rms_reprojection_error_px.update(per_tag)
            diagnostics.overall_rms_reprojection_error_px = error
            worst_tag_id, worst_error = max(per_tag.items(), key=lambda item: item[1])
            tag_errors_ok = worst_error <= self.max_tag_reprojection_error_px
            overall_error_ok = error <= self.max_reprojection_error_px
            if tag_errors_ok and overall_error_ok:
                camera_to_field = opencv_pose_to_wpilib_transform(rvec, tvec)
                field_to_camera = invert_transform(camera_to_field)
                camera_position = field_to_camera[:3, 3]
                if self.bounds is not None:
                    rejection_reason = self.bounds.rejection_reason(camera_position)
                    if rejection_reason is not None:
                        diagnostics.rejection_reason = rejection_reason
                        return None
                distances = [float(np.linalg.norm(tag_layout[obs.tag_id][:3, 3] - camera_position)) for obs in active]
                average_distance = float(np.mean(distances))
                confidence = 1.0 / (1.0 + error + 0.1 * average_distance)
                diagnostics.accepted_tag_ids = [obs.tag_id for obs in active]
                result = CameraPoseObservation(
                    tag_ids=diagnostics.accepted_tag_ids.copy(),
                    field_to_camera_0=field_to_camera,
                    error_0=error,
                    field_to_camera_1=None,
                    error_1=None,
                    average_tag_distance=average_distance,
                    confidence=confidence,
                    diagnostics=diagnostics,
                )
                result._defer(
                    partial(
                        _multi_tag_distributions,
                        object_points,
                        image_points,
                        rvec,
                        tvec,
                        _Intrinsics.of(camera),
                        self.corner_noise_px,
                    ),
                    (fingerprint(field_to_camera), None),
                )
                return result

            if len(active) <= 2 or outlier_iterations >= self.max_outlier_iterations:
                diagnostics.rejection_reason = (
                    f"multi-tag reprojection error is {error:.3f}px overall; worst tag {worst_tag_id} "
                    f"is {worst_error:.3f}px"
                )
                return None
            if worst_error > self.max_tag_reprojection_error_px:
                reason = (
                    f"reprojection residual {worst_error:.3f}px exceeded {self.max_tag_reprojection_error_px:.3f}px"
                )
            else:
                reason = (
                    f"highest tag residual {worst_error:.3f}px removed because overall residual "
                    f"{error:.3f}px exceeded {self.max_reprojection_error_px:.3f}px"
                )
            diagnostics.rejected_tag_reasons[worst_tag_id] = reason
            active = [observation for observation in active if observation.tag_id != worst_tag_id]
            outlier_iterations += 1

        diagnostics.rejection_reason = "fewer than two inlier tags remain"
        return None


class TagAngleSolver:
    """Per-tag viewing angles, with the centre angles' covariance for ``corner_noise_px`` of corner noise."""

    def __init__(self, *, corner_noise_px: float = DEFAULT_CORNER_NOISE_PX) -> None:
        self.corner_noise_px = validate_corner_noise(corner_noise_px)

    def solve(
        self,
        observation: FiducialImageObservation,
        camera: CameraGeometry,
        single_tag_pose: SingleTagPoseObservation,
    ) -> TagAngleObservation:
        return self.solve_many([observation], camera, [single_tag_pose])[0]

    def solve_many(
        self,
        observations: Sequence[FiducialImageObservation],
        camera: CameraGeometry,
        single_tag_poses: Sequence[SingleTagPoseObservation],
    ) -> list[TagAngleObservation]:
        """Build tag-angle observations with one undistortion call."""

        if len(observations) != len(single_tag_poses):
            raise ValueError("observations and single_tag_poses must have the same length")
        if any(
            observation.tag_id != single.tag_id
            for observation, single in zip(observations, single_tag_poses, strict=True)
        ):
            raise ValueError("each observation must match the paired single_tag_pose tag_id")
        if not observations:
            return []
        all_corners = np.concatenate(
            [np.asarray(observation.corners, dtype=np.float64).reshape(4, 2) for observation in observations]
        )
        all_angles = undistorted_corner_angles(
            all_corners,
            camera.calibration_matrix,
            camera.distortion_coefficients,
        ).reshape(len(observations), 4, 2)
        covariances = _CenterAngleCovariances(all_angles.copy(), _Intrinsics.of(camera), self.corner_noise_px)
        centers = all_angles.mean(axis=1).tolist()
        results = [
            TagAngleObservation(
                tag_id=observation.tag_id,
                corner_angles=corner_angles,
                center_tx=center_tx,
                center_ty=center_ty,
                distance=single.distance,
                camera_to_tag_translation=opencv_translation_to_wpilib(single.pose_0_tvec),
            )
            for observation, single, corner_angles, (center_tx, center_ty) in zip(
                observations, single_tag_poses, all_angles, centers, strict=True
            )
        ]
        for index, result in enumerate(results):
            result._center_source = partial(covariances, index)
        return results
