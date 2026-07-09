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
    CameraPoseObservation,
    FiducialImageObservation,
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
    """Estimate a WPILib field pose while preserving single-tag ambiguity."""

    def __init__(self, max_reprojection_error_px: float = 5.0) -> None:
        self.max_reprojection_error_px = max_reprojection_error_px

    def solve(
        self,
        observations: Sequence[FiducialImageObservation],
        single_tag_poses: Sequence[SingleTagPoseObservation],
        camera: CameraConfig,
        tag_size_m: float,
        tag_layout: dict[int, np.ndarray],
    ) -> CameraPoseObservation | None:
        known: list[FiducialImageObservation] = []
        seen_ids: set[int] = set()
        for observation in observations:
            if observation.tag_id in tag_layout and observation.tag_id not in seen_ids:
                known.append(observation)
                seen_ids.add(observation.tag_id)
        if not known:
            return None
        if len(known) == 1:
            single_by_id = {pose.tag_id: pose for pose in single_tag_poses}
            return self._solve_single(known[0], single_by_id.get(known[0].tag_id), tag_layout)
        return self._solve_multi(known, camera, tag_size_m, tag_layout)

    def _solve_single(
        self,
        observation: FiducialImageObservation,
        single: SingleTagPoseObservation | None,
        tag_layout: dict[int, np.ndarray],
    ) -> CameraPoseObservation | None:
        if (
            single is None
            or single.pose_1_rvec is None
            or single.pose_1_tvec is None
            or single.error_1 is None
            or single.error_0 > self.max_reprojection_error_px
            or single.error_1 > self.max_reprojection_error_px
        ):
            return None
        field_to_tag = tag_layout[observation.tag_id]
        camera_to_tag_0 = opencv_pose_to_wpilib_transform(single.pose_0_rvec, single.pose_0_tvec)
        camera_to_tag_1 = opencv_pose_to_wpilib_transform(single.pose_1_rvec, single.pose_1_tvec)
        field_to_camera_0 = field_to_tag @ invert_transform(camera_to_tag_0)
        field_to_camera_1 = field_to_tag @ invert_transform(camera_to_tag_1)
        confidence = 1.0 / (1.0 + single.error_0 + 0.1 * single.distance)
        return CameraPoseObservation(
            tag_ids=[observation.tag_id],
            field_to_camera_0=field_to_camera_0,
            error_0=single.error_0,
            field_to_camera_1=field_to_camera_1,
            error_1=single.error_1,
            average_tag_distance=single.distance,
            confidence=confidence,
        )

    def _solve_multi(
        self,
        observations: Sequence[FiducialImageObservation],
        camera: CameraConfig,
        tag_size_m: float,
        tag_layout: dict[int, np.ndarray],
    ) -> CameraPoseObservation | None:
        object_points = np.concatenate(
            [wpilib_points_to_opencv(tag_field_corners(tag_layout[obs.tag_id], tag_size_m)) for obs in observations]
        )
        image_points = np.concatenate([np.asarray(obs.corners, dtype=np.float64).reshape(4, 2) for obs in observations])
        if not np.all(np.isfinite(image_points)):
            return None
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
            candidate_error = _solution_error(
                errors,
                index,
                object_points,
                image_points,
                candidate_rvec,
                candidate_tvec,
                camera,
            )
            if not _valid_solution(
                candidate_rvec,
                candidate_tvec,
                candidate_error,
                self.max_reprojection_error_px,
            ):
                continue
            rotation, _ = cv2.Rodrigues(candidate_rvec)
            camera_points = (rotation @ object_points.T).T + np.asarray(candidate_tvec).reshape(1, 3)
            if np.all(camera_points[:, 2] > 0.0):
                candidates.append((candidate_error, candidate_rvec, candidate_tvec))
        if not candidates:
            return None
        error, rvec, tvec = min(candidates, key=lambda candidate: candidate[0])

        camera_to_field = opencv_pose_to_wpilib_transform(rvec, tvec)
        field_to_camera = invert_transform(camera_to_field)
        camera_position = field_to_camera[:3, 3]
        distances = [float(np.linalg.norm(tag_layout[obs.tag_id][:3, 3] - camera_position)) for obs in observations]
        average_distance = float(np.mean(distances))
        confidence = 1.0 / (1.0 + error + 0.1 * average_distance)
        return CameraPoseObservation(
            tag_ids=[obs.tag_id for obs in observations],
            field_to_camera_0=field_to_camera,
            error_0=error,
            field_to_camera_1=None,
            error_1=None,
            average_tag_distance=average_distance,
            confidence=confidence,
        )


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
