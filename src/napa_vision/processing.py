"""Synchronous image processing without camera, transport, or device ownership."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from time import perf_counter

import numpy as np

from .calibration import CameraCalibration
from .napa_detector import NapaDetector
from .solvers import CameraPoseSolver, SingleTagPoseSolver, TagAngleSolver
from .types import CameraPoseBounds, FiducialDetector, FrameResult


@dataclass(frozen=True, slots=True)
class _CameraGeometry:
    calibration_matrix: np.ndarray
    distortion_coefficients: np.ndarray


def _rigid_transform(value: np.ndarray, name: str) -> np.ndarray:
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


class VisionProcessor:
    """Process caller-owned images; use one instance per camera stream.

    Without calibration, return IDs and corners only. Calibrated processing also
    requires the physical tag size. A tag layout (tag ID to world-to-tag transform)
    is optional and enables a camera pose; poses use the conventions documented in
    ``geometry``, never a device SDK's world frame.
    Calls are synchronous and must not overlap on one instance. The caller owns
    capture, pixel-format conversion, clocks, scheduling, and result delivery.
    """

    def __init__(
        self,
        *,
        detector: FiducialDetector | None = None,
        calibration: CameraCalibration | None = None,
        tag_size_m: float | None = None,
        tag_layout: dict[int, np.ndarray] | None = None,
        max_reprojection_error_px: float = 5.0,
        max_tag_reprojection_error_px: float | None = None,
        camera_pose_bounds: CameraPoseBounds | None = None,
        max_outlier_iterations: int = 8,
        ransac_iterations: int = 100,
    ) -> None:
        if (calibration is None) != (tag_size_m is None):
            raise ValueError("calibration and tag_size_m must be provided together")
        if tag_size_m is not None and (
            isinstance(tag_size_m, bool)
            or not isinstance(tag_size_m, Real)
            or not math.isfinite(tag_size_m)
            or tag_size_m <= 0.0
        ):
            raise ValueError("tag_size_m must be finite and greater than zero")
        if calibration is None and tag_layout is not None:
            raise ValueError("tag_layout requires calibration")
        if calibration is not None and calibration.image_size is None:
            raise ValueError("calibration must record its image resolution")
        self.calibration = calibration
        self.tag_size_m = tag_size_m
        self.tag_layout: dict[int, np.ndarray] = {}
        for tag_id, transform in ({} if tag_layout is None else tag_layout).items():
            if isinstance(tag_id, bool) or not isinstance(tag_id, int) or tag_id < 0:
                raise ValueError("tag_layout IDs must be non-negative integers")
            self.tag_layout[tag_id] = _rigid_transform(transform, f"tag_layout[{tag_id}]")
        self._camera = (
            None
            if calibration is None
            else _CameraGeometry(calibration.camera_matrix, calibration.distortion_coefficients)
        )
        self._single_solver = SingleTagPoseSolver(max_reprojection_error_px)
        self._camera_solver = CameraPoseSolver(
            max_reprojection_error_px,
            max_tag_reprojection_error_px=max_tag_reprojection_error_px,
            bounds=camera_pose_bounds,
            max_outlier_iterations=max_outlier_iterations,
            ransac_iterations=ransac_iterations,
        )
        self._angle_solver = TagAngleSolver()
        self.detector = detector if detector is not None else NapaDetector()

    def process(
        self,
        image: np.ndarray,
        *,
        timestamp: float,
        capture_monotonic_s: float | None = None,
    ) -> FrameResult:
        """Return a fresh result without retaining the image or publishing it.

        Images must be nonempty uint8 grayscale, BGR, or BGRA arrays. Strided
        arrays are accepted. Timestamp units are seconds; capture_monotonic_s
        belongs to the caller's clock. Robot/world augmentation and device
        coordinate conversion belong to the caller.
        """

        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise ValueError("AprilTag detection requires an 8-bit image")
        if image.ndim not in {2, 3} or (image.ndim == 3 and image.shape[2] not in {3, 4}):
            raise ValueError("AprilTag detection requires a grayscale, BGR, or BGRA image")
        if image.size == 0:
            raise ValueError("AprilTag detection requires a nonempty image")
        if timestamp is None:
            raise ValueError("timestamp must be finite")
        for value, name in (
            (timestamp, "timestamp"),
            (capture_monotonic_s, "capture_monotonic_s"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.calibration is not None:
            self.calibration.validate_for_image_size(image.shape[1], image.shape[0])
        detection_start = perf_counter()
        observations = self.detector.detect(image)
        detection_end = perf_counter()
        result = FrameResult(
            timestamp=timestamp,
            capture_monotonic_s=capture_monotonic_s,
            image_observations=observations,
        )
        if self._camera is not None and self.tag_size_m is not None:
            solved_observations = []
            for observation in observations:
                single = self._single_solver.solve(observation, self._camera, self.tag_size_m)
                if single is not None:
                    result.single_tag_poses.append(single)
                    solved_observations.append(observation)
            result.tag_angles = self._angle_solver.solve_many(
                solved_observations, self._camera, result.single_tag_poses
            )
            result.camera_pose = self._camera_solver.solve(
                observations, result.single_tag_poses, self._camera, self.tag_size_m, self.tag_layout
            )
            result.pose_diagnostics = self._camera_solver.last_diagnostics
        solve_end = perf_counter()
        result.health = {
            "detection_ms": (detection_end - detection_start) * 1000.0,
            "solve_ms": (solve_end - detection_end) * 1000.0,
        }
        return result
