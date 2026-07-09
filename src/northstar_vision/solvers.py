from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from .calibration import undistorted_corner_angles
from .config import CameraConfig
from .geometry import (
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
    TrigPoseObservation,
)


def _solution_error(
    errors: np.ndarray | None,
    index: int,
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera: CameraConfig,
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
    def __init__(self, max_reprojection_error_px: float = 5.0) -> None:
        self.max_reprojection_error_px = max_reprojection_error_px

    def solve(
        self,
        observation: FiducialImageObservation,
        camera: CameraConfig,
        tag_size_m: float,
    ) -> SingleTagPoseObservation | None:
        object_points = tag_object_points(tag_size_m)
        corners = np.asarray(observation.corners, dtype=np.float64)
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            return None
        try:
            ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                object_points,
                corners,
                camera.calibration_matrix,
                camera.distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            return None
        if not ok or len(rvecs) == 0:
            return None

        solutions: list[tuple[np.ndarray, np.ndarray, float]] = []
        for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs, strict=False)):
            error = _solution_error(errors, index, object_points, corners, rvec, tvec, camera)
            if _valid_solution(rvec, tvec, error, float("inf")) and float(tvec[2][0]) > 0.0:
                solutions.append((rvec, tvec, error))
        if not solutions:
            return None
        solutions.sort(key=lambda item: item[2])
        if solutions[0][2] > self.max_reprojection_error_px:
            return None
        pose_0_rvec, pose_0_tvec, error_0 = solutions[0]
        pose_1_rvec = pose_1_tvec = None
        error_1 = None
        if len(solutions) > 1:
            pose_1_rvec, pose_1_tvec, error_1 = solutions[1]
        return SingleTagPoseObservation(
            tag_id=observation.tag_id,
            pose_0_rvec=pose_0_rvec,
            pose_0_tvec=pose_0_tvec,
            error_0=error_0,
            pose_1_rvec=pose_1_rvec,
            pose_1_tvec=pose_1_tvec,
            error_1=error_1,
            distance=float(np.linalg.norm(pose_0_tvec)),
        )


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
        self.last_diagnostics = PoseSolveDiagnostics()

    def solve(
        self,
        observations: Sequence[FiducialImageObservation],
        single_tag_poses: Sequence[SingleTagPoseObservation],
        camera: CameraConfig,
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
                tag_layout,
                diagnostics,
            )
        return self._solve_multi(known, camera, tag_size_m, tag_layout, diagnostics)

    def _solve_single(
        self,
        observation: FiducialImageObservation,
        single: SingleTagPoseObservation | None,
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
        return CameraPoseObservation(
            tag_ids=[observation.tag_id],
            field_to_camera_0=field_to_camera_0,
            error_0=single.error_0,
            field_to_camera_1=field_to_camera_1,
            error_1=single.error_1,
            average_tag_distance=single.distance,
            confidence=confidence,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _multi_points(
        observations: Sequence[FiducialImageObservation],
        tag_size_m: float,
        tag_layout: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        object_points = np.concatenate(
            [wpilib_points_to_opencv(tag_field_corners(tag_layout[obs.tag_id], tag_size_m)) for obs in observations]
        )
        image_points = np.concatenate([np.asarray(obs.corners, dtype=np.float64).reshape(4, 2) for obs in observations])
        return object_points, image_points

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
        camera: CameraConfig,
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
        camera: CameraConfig,
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
        camera: CameraConfig,
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

class TagAngleSolver:
    def solve(
        self,
        observation: FiducialImageObservation,
        camera: CameraConfig,
        single_tag_pose: SingleTagPoseObservation,
    ) -> TagAngleObservation:
        corner_angles = undistorted_corner_angles(
            observation.corners,
            camera.calibration_matrix,
            camera.distortion_coefficients,
        )
        return TagAngleObservation(
            tag_id=observation.tag_id,
            corner_angles=corner_angles,
            center_tx=float(np.mean(corner_angles[:, 0])),
            center_ty=float(np.mean(corner_angles[:, 1])),
            distance=single_tag_pose.distance,
            camera_to_tag_translation=opencv_translation_to_wpilib(single_tag_pose.pose_0_tvec),
        )


class TrigDistanceSolver:
    def solve(
        self,
        tag_angle: TagAngleObservation,
        camera: CameraConfig,
        tag_layout: dict[int, np.ndarray],
        robot_yaw_rad: float,
    ) -> TrigPoseObservation | None:
        field_to_tag = tag_layout.get(tag_angle.tag_id)
        if field_to_tag is None:
            return None

        camera_to_tag = np.asarray(tag_angle.camera_to_tag_translation, dtype=np.float64).reshape(3)
        robot_to_tag = camera.robot_to_camera[:3, 3] + camera.robot_to_camera[:3, :3] @ camera_to_tag
        cosine, sine = math.cos(robot_yaw_rad), math.sin(robot_yaw_rad)
        field_rotation = np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        field_to_robot_translation = field_to_tag[:3, 3] - field_rotation @ robot_to_tag
        robot_pose = np.array(
            [field_to_robot_translation[0], field_to_robot_translation[1], robot_yaw_rad],
            dtype=np.float64,
        )
        return TrigPoseObservation(
            tag_id=tag_angle.tag_id,
            robot_pose=robot_pose,
            camera_to_tag_translation=camera_to_tag.copy(),
            distance=tag_angle.distance,
        )
