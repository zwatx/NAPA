from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np

from .calibration import undistorted_corner_angles
from .covariance import (
    DEFAULT_CORNER_NOISE_PX,
    PoseWithCovariance,
    ambiguity_covariances,
    cap_correlations,
    center_angle_covariances,
    reprojection_covariances,
    reprojection_ssr,
    validate_corner_noise,
)
from .geometry import (
    WPILIB_TO_OPENCV_ROTATION,
    invert_transform,
    opencv_pose_to_wpilib_transform,
    opencv_translation_to_wpilib,
    rvec_tvec_to_transform,
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
)


class CameraGeometry(Protocol):
    """The image geometry solvers need, independent of capture configuration."""

    @property
    def calibration_matrix(self) -> np.ndarray: ...

    @property
    def distortion_coefficients(self) -> np.ndarray: ...


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
    return np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)) and np.isfinite(error) and 0.0 <= error <= max_error


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
        """Solve every tag, computing all covariances in one batch. Results match ``observations`` by index."""

        object_points = tag_object_points(tag_size_m)
        solved = [self._ippe_solutions(observation, camera, object_points) for observation in observations]
        transforms = [rvec_tvec_to_transform(rvec, tvec) for solutions in solved for rvec, tvec, _ in solutions]
        if not transforms:
            return [None] * len(observations)
        stacked = np.stack(transforms)
        image_points = np.stack(
            [
                np.asarray(observation.corners, dtype=np.float64)
                for observation, solutions in zip(observations, solved, strict=True)
                for _ in solutions
            ]
        )
        intrinsics = (camera.calibration_matrix, camera.distortion_coefficients)
        covariances = iter(reprojection_covariances(stacked, object_points, *intrinsics, self.corner_noise_px))
        squared_residuals = iter(reprojection_ssr(stacked, object_points, image_points, *intrinsics))
        poses = iter(transforms)
        results: list[SingleTagPoseObservation | None] = []
        for observation, solutions in zip(observations, solved, strict=True):
            if not solutions:
                results.append(None)
                continue
            (pose_0_rvec, pose_0_tvec, error_0), *alternative = solutions
            pose_0, covariance_0, ssr_0 = next(poses), next(covariances), next(squared_residuals)
            pose_1_rvec = pose_1_tvec = error_1 = covariance_1 = None
            if alternative:
                pose_1_rvec, pose_1_tvec, error_1 = alternative[0]
                covariance_0, covariance_1 = ambiguity_covariances(
                    (pose_0, next(poses)),
                    (covariance_0, next(covariances)),
                    (ssr_0, next(squared_residuals)),
                    self.corner_noise_px,
                )
            results.append(
                SingleTagPoseObservation(
                    tag_id=observation.tag_id,
                    pose_0_rvec=pose_0_rvec,
                    pose_0_tvec=pose_0_tvec,
                    error_0=error_0,
                    pose_1_rvec=pose_1_rvec,
                    pose_1_tvec=pose_1_tvec,
                    error_1=error_1,
                    distance=float(np.linalg.norm(pose_0_tvec)),
                    covariance_0=cap_correlations(covariance_0),
                    covariance_1=cap_correlations(covariance_1),
                )
            )
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
        covariance_0, covariance_1 = self._field_to_camera_covariances(
            (field_to_camera_0, field_to_camera_1),
            tag_field_corners(field_to_tag, tag_size_m),
            np.asarray(observation.corners, dtype=np.float64),
            camera,
        )
        return CameraPoseObservation(
            tag_ids=[observation.tag_id],
            field_to_camera_0=field_to_camera_0,
            error_0=single.error_0,
            field_to_camera_1=field_to_camera_1,
            error_1=single.error_1,
            average_tag_distance=single.distance,
            confidence=confidence,
            diagnostics=diagnostics,
            covariance_0=covariance_0,
            covariance_1=covariance_1,
        )

    def _field_to_camera_covariances(
        self,
        field_to_cameras: Sequence[np.ndarray],
        field_points: np.ndarray,
        image_points: np.ndarray,
        camera: CameraGeometry,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Covariances of one pose, or of two planar-ambiguity solutions widened to cover each other."""

        camera_to_fields = np.stack([invert_transform(pose) for pose in field_to_cameras])
        intrinsics = (camera.calibration_matrix, camera.distortion_coefficients)
        covariances = [
            None if covariance is None else PoseWithCovariance(pose, covariance).inverse().covariance
            for pose, covariance in zip(
                camera_to_fields,
                reprojection_covariances(
                    camera_to_fields,
                    field_points,
                    *intrinsics,
                    self.corner_noise_px,
                    opencv_from_camera=WPILIB_TO_OPENCV_ROTATION,
                ),
                strict=True,
            )
        ]
        if len(covariances) == 1:
            return cap_correlations(covariances[0]), None
        squared_residuals = reprojection_ssr(
            camera_to_fields, field_points, image_points, *intrinsics, opencv_from_camera=WPILIB_TO_OPENCV_ROTATION
        )
        covariance_0, covariance_1 = ambiguity_covariances(
            field_to_cameras, covariances, squared_residuals, self.corner_noise_px
        )
        return cap_correlations(covariance_0), cap_correlations(covariance_1)

    @staticmethod
    def _planar_alternative(
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera: CameraGeometry,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The mirror solution of a (nearly) planar target's pose, as an OpenCV ``rvec``/``tvec``.

        Tags on one wall form one plane, which has the same two-fold ambiguity as a single tag. IPPE on the plane
        gives both candidates, and the one farther from the solved pose is returned. They are deliberately not
        refined: near head-on the two minima merge, and refining would collapse the spread the covariance needs.
        """

        centroid = object_points.mean(axis=0)
        _, singular_values, plane = np.linalg.svd(object_points - centroid)
        if singular_values[2] > 0.1 * singular_values[0]:
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
        per_tag = {
            observation.tag_id: float(np.sqrt(np.mean(squared[index * 4 : (index + 1) * 4])))
            for index, observation in enumerate(observations)
        }
        return per_tag, float(np.sqrt(np.mean(squared)))

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
                candidates = [field_to_camera]
                alternative = self._planar_alternative(object_points, image_points, rvec, tvec, camera)
                if alternative is not None:
                    candidates.append(invert_transform(opencv_pose_to_wpilib_transform(*alternative)))
                covariance, _ = self._field_to_camera_covariances(
                    candidates, object_points @ WPILIB_TO_OPENCV_ROTATION, image_points, camera
                )
                diagnostics.accepted_tag_ids = [obs.tag_id for obs in active]
                return CameraPoseObservation(
                    tag_ids=diagnostics.accepted_tag_ids.copy(),
                    field_to_camera_0=field_to_camera,
                    error_0=error,
                    field_to_camera_1=None,
                    error_1=None,
                    average_tag_distance=average_distance,
                    confidence=confidence,
                    diagnostics=diagnostics,
                    covariance_0=covariance,
                )

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
        covariances = center_angle_covariances(
            all_angles, camera.calibration_matrix, camera.distortion_coefficients, self.corner_noise_px
        )
        return [
            TagAngleObservation(
                tag_id=observation.tag_id,
                corner_angles=corner_angles,
                center_tx=float(np.mean(corner_angles[:, 0])),
                center_ty=float(np.mean(corner_angles[:, 1])),
                distance=single.distance,
                camera_to_tag_translation=opencv_translation_to_wpilib(single.pose_0_tvec),
                center_covariance=covariance,
            )
            for observation, single, corner_angles, covariance in zip(
                observations, single_tag_poses, all_angles, covariances, strict=True
            )
        ]
