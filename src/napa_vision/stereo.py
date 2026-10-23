"""Joint tag localization for two calibrated, caller-synchronized cameras."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Literal

import cv2
import numpy as np

from .covariance import (
    DEFAULT_CORNER_NOISE_PX,
    PoseWithCovariance,
    _covariances_from_jacobians,
    _perturb,
    _projection_terms,
    likelihood_weights,
    mixture_pose,
    pose_difference,
    validate_corner_noise,
)
from .geometry import (
    WPILIB_TO_OPENCV_ROTATION,
    _rigid_transform,
    invert_transform,
    opencv_pose_to_wpilib_transform,
    tag_field_corners,
    tag_object_points,
    wpilib_points_to_opencv,
)
from .solvers import CameraPoseSolver, SingleTagPoseSolver, _Intrinsics
from .stereo_calibration import StereoCalibration
from .types import FiducialImageObservation

Group = tuple[str, int]


@dataclass(slots=True)
class StereoSolveDiagnostics:
    """Residuals are per camera/tag pair; observing a tag in both cameras is valid."""

    capture_skew_s: float
    attempted_groups: list[Group] = field(default_factory=list)
    accepted_groups: list[Group] = field(default_factory=list)
    rejected_group_reasons: dict[Group, str] = field(default_factory=dict)
    per_group_rms_px: dict[Group, float] = field(default_factory=dict)
    candidate_rms_px: tuple[float, ...] = ()
    iterations: int = 0
    rejection_reason: str | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StereoPoseResult:
    """Rig poses in WPILib field axes, best first, including competing-solution uncertainty.

    Covariance is conditional on exact calibration/layout and independent corner noise. It does not account for
    clock offsets, rolling shutter, mount flex or calibration error. Rejected results have no poses. Monocular
    fallback uses the selected camera's capture time; stereo uses the pair's midpoint.
    """

    timestamp: float
    status: Literal["stereo", "monocular", "rejected"]
    poses: tuple[PoseWithCovariance, ...]
    diagnostics: StereoSolveDiagnostics

    @property
    def pose(self) -> PoseWithCovariance | None:
        return self.poses[0] if self.poses else None


@dataclass(frozen=True, slots=True)
class _View:
    camera_id: str
    camera: _Intrinsics
    rig_to_camera: np.ndarray
    cv_from_rig: np.ndarray
    cv_offset: np.ndarray
    noise: float
    observations: tuple[FiducialImageObservation, ...]
    points: np.ndarray
    pixels: np.ndarray

    @property
    def groups(self) -> list[Group]:
        return [(self.camera_id, observation.tag_id) for observation in self.observations]

    def without(self, group: Group) -> _View:
        keep = [index for index, key in enumerate(self.groups) if key != group]
        return replace(
            self,
            observations=tuple(self.observations[index] for index in keep),
            points=self.points.reshape(-1, 4, 3)[keep].reshape(-1, 3),
            pixels=self.pixels.reshape(-1, 4, 2)[keep].reshape(-1, 2),
        )


@dataclass(frozen=True, slots=True)
class _Fit:
    rig_from_field: np.ndarray
    residuals: np.ndarray
    jacobian: np.ndarray
    errors: np.ndarray
    iterations: int

    @property
    def cost(self) -> float:
        return float(self.residuals @ self.residuals)


def _finite_time(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite seconds on the common capture clock")
    return float(value)


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return float(value)


def _joint_terms(
    rig_from_field: np.ndarray, views: Sequence[_View]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Whitened pixel residuals/Jacobian and raw group RMS; perturbations use rig-frame translation/rotation."""

    residuals, jacobians, errors = [], [], []
    for view in views:
        if not view.observations:
            continue
        rotated = (rig_from_field[:3, :3] @ view.points.T).T
        rig_points = rotated + rig_from_field[:3, 3]
        camera_points = rig_points @ view.cv_from_rig.T + view.cv_offset
        if not np.all(np.isfinite(camera_points)) or np.any(camera_points[:, 2] <= 1e-6):
            return None
        try:
            projected, jacobian = _projection_terms(
                rotated[None],
                camera_points[None],
                view.camera.calibration_matrix,
                view.camera.distortion_coefficients,
                view.cv_from_rig,
            )
        except cv2.error:
            return None
        residual = view.pixels - projected[0]
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
            return None
        residuals.append(residual.reshape(-1) / view.noise)
        jacobians.append(jacobian[0] / view.noise)
        errors.extend(np.sqrt(np.sum(residual.reshape(-1, 4, 2) ** 2, axis=2).mean(axis=1)))
    if not residuals:
        return None
    return np.concatenate(residuals), np.concatenate(jacobians), np.asarray(errors)


class StereoPoseSolver:
    """Solve one rig pose from two timestamped sets of tag corners, with fixed calibration and layout.

    Inputs use raw, unrectified image pixels matching each calibration. Capture times are exposure midpoints in
    seconds on one clock. The caller owns synchronization. ``max_capture_skew_s`` is required because acceptable
    skew depends on motion. A rig pose maps rig coordinates into field coordinates; camera mounts use WPILib axes.
    """

    def __init__(
        self,
        calibration: StereoCalibration,
        tag_layout: dict[int, np.ndarray],
        tag_size_m: float,
        *,
        max_capture_skew_s: float,
        max_reprojection_error_px: float = 5.0,
        max_outlier_iterations: int = 2,
        max_iterations: int = 50,
        corner_noise_px: tuple[float, float] = (DEFAULT_CORNER_NOISE_PX, DEFAULT_CORNER_NOISE_PX),
    ) -> None:
        if not isinstance(calibration, StereoCalibration):
            raise ValueError("calibration must be a StereoCalibration")
        self._calibration = replace(calibration)
        self.tag_size_m = _positive(tag_size_m, "tag_size_m")
        self.max_capture_skew_s = _finite_time(max_capture_skew_s, "max_capture_skew_s")
        if self.max_capture_skew_s < 0:
            raise ValueError("max_capture_skew_s must be non-negative")
        self.max_reprojection_error_px = _positive(max_reprojection_error_px, "max_reprojection_error_px")
        if len(corner_noise_px) != 2:
            raise ValueError("corner_noise_px must contain one value per camera")
        self._noise = tuple(validate_corner_noise(value) for value in corner_noise_px)
        for value, name, minimum in (
            (max_outlier_iterations, "max_outlier_iterations", 0),
            (max_iterations, "max_iterations", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.max_outlier_iterations = max_outlier_iterations
        self.max_iterations = max_iterations
        self._layout = {}
        for tag_id, transform in tag_layout.items():
            if isinstance(tag_id, bool) or not isinstance(tag_id, int) or tag_id < 0:
                raise ValueError("tag_layout IDs must be non-negative integers")
            self._layout[tag_id] = _rigid_transform(transform, f"tag_layout[{tag_id}]")
        self._field_points = {tag_id: tag_field_corners(pose, tag_size_m) for tag_id, pose in self._layout.items()}

    def _view(
        self, side: int, observations: Sequence[FiducialImageObservation], diagnostics: StereoSolveDiagnostics
    ) -> _View:
        calibration = self._calibration
        camera_id = (calibration.left_id, calibration.right_id)[side]
        camera = (calibration.left, calibration.right)[side]
        mount = (calibration.rig_to_left, calibration.rig_to_right)[side]
        normalized, seen = [], set()
        for observation in observations:
            key = (camera_id, observation.tag_id)
            diagnostics.attempted_groups.append(key)
            if observation.tag_id not in self._layout:
                diagnostics.rejected_group_reasons[key] = "tag is not present in the field layout"
                continue
            if observation.tag_id in seen:
                diagnostics.rejected_group_reasons[key] = "duplicate observation ignored"
                continue
            try:
                corners = np.array(observation.corners, dtype=np.float64)
            except (TypeError, ValueError):
                corners = np.empty(0)
            if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
                diagnostics.rejected_group_reasons[key] = "corners must be a finite 4x2 array"
                continue
            normalized.append(FiducialImageObservation(observation.tag_id, corners))
            seen.add(observation.tag_id)
        camera_from_rig = invert_transform(mount)
        return _View(
            camera_id,
            _Intrinsics(camera.camera_matrix, camera.distortion_coefficients),
            mount,
            WPILIB_TO_OPENCV_ROTATION @ camera_from_rig[:3, :3],
            WPILIB_TO_OPENCV_ROTATION @ camera_from_rig[:3, 3],
            self._noise[side],
            tuple(normalized),
            np.array([self._field_points[obs.tag_id] for obs in normalized]).reshape(-1, 3),
            np.array([obs.corners for obs in normalized]).reshape(-1, 2),
        )

    def _seeds(self, views: Sequence[_View]) -> list[np.ndarray]:
        seeds = []
        single_solver = SingleTagPoseSolver(max_reprojection_error_px=math.inf)
        generic_solver = CameraPoseSolver()
        for view in views:
            if not view.observations:
                continue
            # Keep both alternatives from the best single tag, even if a mono acceptance gate would reject it.
            pairs = [
                (
                    observation,
                    single_solver._ippe_solutions(observation, view.camera, tag_object_points(self.tag_size_m)),
                )
                for observation in view.observations
            ]
            pairs = [(obs, pair) for obs, pair in pairs if pair]
            if pairs:
                observation, pair = min(pairs, key=lambda item: item[1][0][2])
                for rvec, tvec, _ in pair:
                    field_to_camera = self._layout[observation.tag_id] @ invert_transform(
                        opencv_pose_to_wpilib_transform(rvec, tvec)
                    )
                    seeds.append(view.rig_to_camera @ invert_transform(field_to_camera))
            if len(view.observations) > 1:
                points = wpilib_points_to_opencv(view.points)
                candidate = generic_solver._generic_candidate(points, view.pixels, view.camera)
                if candidate is not None:
                    candidates = [candidate]
                    mirror = generic_solver._planar_alternative(points, view.pixels, *candidate, view.camera)
                    if mirror is not None:
                        candidates.append(mirror)
                    seeds.extend(view.rig_to_camera @ opencv_pose_to_wpilib_transform(*pose) for pose in candidates)
        return seeds

    def _refine(self, seed: np.ndarray, views: Sequence[_View]) -> _Fit | None:
        pose = seed.copy()
        terms = _joint_terms(pose, views)
        if terms is None:
            return None
        damping = 1e-6
        iterations = 0
        for _ in range(self.max_iterations):
            iterations += 1
            residual, jacobian, _ = terms
            normal = jacobian.T @ jacobian
            diagonal = np.maximum(np.diag(normal), 1.0)
            gradient = jacobian.T @ residual
            if np.max(np.abs(gradient) / np.sqrt(diagonal)) < 1e-8:
                break
            try:
                delta = np.linalg.solve(normal + damping * np.diag(diagonal), gradient)
            except np.linalg.LinAlgError:
                return None
            if not np.all(np.isfinite(delta)):
                return None
            if np.linalg.norm(delta) < 1e-10:
                break
            candidate = _perturb(pose, delta)
            candidate_terms = _joint_terms(candidate, views)
            if candidate_terms is not None and candidate_terms[0] @ candidate_terms[0] < residual @ residual:
                pose, terms = candidate, candidate_terms
                damping = max(damping / 10.0, 1e-12)
            else:
                damping *= 10.0
                if damping > 1e12:
                    break
        residual, jacobian, _ = terms
        scale = np.maximum(np.linalg.norm(jacobian, axis=0), 1.0)
        if np.max(np.abs(jacobian.T @ residual) / scale) > 1e-5:
            return None
        return _Fit(pose, *terms, iterations)

    def _fits(self, views: Sequence[_View]) -> list[_Fit]:
        fits = []
        for seed in self._seeds(views):
            fit = self._refine(seed, views)
            if fit is not None:
                fits.append(fit)
        return sorted(fits, key=lambda fit: fit.cost)

    @staticmethod
    def _poses(fits: Sequence[_Fit]) -> tuple[tuple[PoseWithCovariance, ...], tuple[float, ...]]:
        distinct: list[_Fit] = []
        for fit in fits:
            # Repeated seeds near the same minimum must not give that mode extra probability mass.
            if any(
                np.linalg.norm(other.jacobian @ pose_difference(other.rig_from_field, fit.rig_from_field)) < 1e-3
                for other in distinct
            ):
                continue
            distinct.append(fit)
        if not distinct:
            return (), ()
        covariances, valid = _covariances_from_jacobians(np.stack([fit.jacobian for fit in distinct]), 1.0)
        if not valid[0]:
            return (), ()
        weights = likelihood_weights(np.array([fit.cost for fit in distinct]), 1.0)
        retained = [index for index, ok in enumerate(valid) if ok and weights[index] >= 0.01 * weights[0]]
        components = [(distinct[index].rig_from_field, covariances[index], weights[index]) for index in retained]
        # Each candidate carries the others' spread; inversion moves every mixture component exactly.
        poses = tuple(
            mixture_pose(components[index:] + components[:index]).inverse() for index in range(len(components))
        )
        errors = tuple(float(np.sqrt(np.mean(distinct[index].errors ** 2))) for index in retained)
        return poses, errors

    def solve(
        self,
        left_observations: Sequence[FiducialImageObservation],
        right_observations: Sequence[FiducialImageObservation],
        *,
        left_capture_time_s: float,
        right_capture_time_s: float,
        allow_monocular_fallback: bool = False,
    ) -> StereoPoseResult:
        """Return a fresh result, including diagnostics when rejected. Fallback tries the newest usable view first."""

        times = (
            _finite_time(left_capture_time_s, "left_capture_time_s"),
            _finite_time(right_capture_time_s, "right_capture_time_s"),
        )
        if not isinstance(allow_monocular_fallback, bool):
            raise ValueError("allow_monocular_fallback must be a bool")
        skew = abs(times[1] - times[0])
        diagnostics = StereoSolveDiagnostics(skew)
        views = (self._view(0, left_observations, diagnostics), self._view(1, right_observations, diagnostics))
        timestamp = times[0] / 2 + times[1] / 2
        if skew > self.max_capture_skew_s:
            diagnostics.rejection_reason = "capture skew exceeds max_capture_skew_s"
        elif not all(view.observations for view in views):
            diagnostics.rejection_reason = "stereo requires known tags in both cameras"
        else:
            active = views
            for outlier_iteration in range(self.max_outlier_iterations + 1):
                fits = self._fits(active)
                if not fits:
                    diagnostics.rejection_reason = "no seed produced a converged, physically valid joint pose"
                    break
                best = fits[0]
                groups = [group for view in active for group in view.groups]
                diagnostics.iterations += sum(fit.iterations for fit in fits)
                diagnostics.per_group_rms_px.update(zip(groups, best.errors, strict=True))
                acceptable = [fit for fit in fits if np.all(fit.errors <= self.max_reprojection_error_px)]
                if acceptable:
                    poses, errors = self._poses(acceptable)
                    if poses:
                        diagnostics.rejection_reason = None
                        diagnostics.accepted_groups = groups
                        diagnostics.per_group_rms_px.update(zip(groups, acceptable[0].errors, strict=True))
                        diagnostics.candidate_rms_px = errors
                        return StereoPoseResult(timestamp, "stereo", poses, diagnostics)
                    diagnostics.rejection_reason = "joint pose covariance is undefined"
                    break
                diagnostics.rejection_reason = "joint reprojection residual exceeds the configured limit"
                worst = groups[int(np.argmax(best.errors))]
                remaining = tuple(view.without(worst) for view in active)
                if outlier_iteration == self.max_outlier_iterations or not all(view.observations for view in remaining):
                    break
                diagnostics.rejected_group_reasons[worst] = "reprojection residual exceeded the configured limit"
                active = remaining
        if allow_monocular_fallback:
            return self._fallback(views, times, diagnostics)
        return StereoPoseResult(timestamp, "rejected", (), diagnostics)

    def _fallback(
        self, views: Sequence[_View], times: tuple[float, float], diagnostics: StereoSolveDiagnostics
    ) -> StereoPoseResult:
        for side in sorted(range(2), key=lambda index: times[index], reverse=True):
            view = views[side]
            if not view.observations:
                continue
            singles = SingleTagPoseSolver(self.max_reprojection_error_px, corner_noise_px=view.noise).solve_many(
                view.observations, view.camera, self.tag_size_m
            )
            solver = CameraPoseSolver(self.max_reprojection_error_px, corner_noise_px=view.noise)
            mono = solver.solve(
                view.observations,
                [single for single in singles if single is not None],
                view.camera,
                self.tag_size_m,
                self._layout,
            )
            if mono is None or mono.covariance_0 is None:
                continue
            camera_to_rig = invert_transform(view.rig_to_camera)
            poses = [mono.field_to_camera().transform_by(camera_to_rig)]
            if mono.field_to_camera_1 is not None and mono.covariance_1 is not None:
                poses.append(mono.field_to_camera(1).transform_by(camera_to_rig))
            diagnostics.fallback_reason = diagnostics.rejection_reason
            diagnostics.rejection_reason = None
            diagnostics.accepted_groups = [(view.camera_id, tag_id) for tag_id in mono.tag_ids]
            diagnostics.per_group_rms_px.update(
                ((view.camera_id, tag_id), error)
                for tag_id, error in solver.last_diagnostics.per_tag_rms_reprojection_error_px.items()
            )
            return StereoPoseResult(times[side], "monocular", tuple(poses), diagnostics)
        return StereoPoseResult(times[0] / 2 + times[1] / 2, "rejected", (), diagnostics)
