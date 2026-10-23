"""Validated stereo-camera calibration artifacts and fixed-intrinsic fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import _import_cv2
from .calibration import (
    CalibrationError,
    CameraCalibration,
    _camera_calibration_from_dict,
    _camera_calibration_to_dict,
    _number,
    _positive_integer,
)
from .geometry import OPENCV_TO_WPILIB_BASIS, _rigid_transform, invert_transform
from .provenance import ArtifactProvenance, ProvenanceError, ProvenanceKind


@dataclass(frozen=True, slots=True)
class StereoCalibration:
    """Two calibrated cameras, with each camera's WPILib pose expressed in the rig.

    ``rig_to_left`` and ``rig_to_right`` map coordinates from their respective WPILib camera frame into the WPILib
    rig frame. A stereo fit chooses the left camera as the rig origin.
    """

    left: CameraCalibration
    right: CameraCalibration
    rig_to_left: np.ndarray
    rig_to_right: np.ndarray
    provenance: ArtifactProvenance
    left_id: str = "left"
    right_id: str = "right"
    rms_reprojection_error_px: float | None = None
    views_used: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.left, CameraCalibration) or not isinstance(self.right, CameraCalibration):
            raise CalibrationError("left and right must be CameraCalibration instances")
        if self.left.image_size is None or self.right.image_size is None:
            raise CalibrationError("stereo calibration cameras must record their image resolutions")
        if not isinstance(self.provenance, ArtifactProvenance):
            raise CalibrationError("provenance must be an ArtifactProvenance")
        if not isinstance(self.left_id, str) or not self.left_id.strip():
            raise CalibrationError("left_id must be a non-empty string")
        if not isinstance(self.right_id, str) or not self.right_id.strip():
            raise CalibrationError("right_id must be a non-empty string")
        if self.left_id == self.right_id:
            raise CalibrationError("left_id and right_id must be distinct")
        if self.rms_reprojection_error_px is not None:
            _number(self.rms_reprojection_error_px, "rms_reprojection_error_px", minimum=0.0)
        if self.views_used is not None:
            _positive_integer(self.views_used, "views_used")
        try:
            left_pose = _rigid_transform(self.rig_to_left, "rig_to_left")
            right_pose = _rigid_transform(self.rig_to_right, "rig_to_right")
        except (TypeError, ValueError) as exc:
            raise CalibrationError(str(exc)) from exc
        left_pose.setflags(write=False)
        right_pose.setflags(write=False)
        object.__setattr__(self, "left", replace(self.left))
        object.__setattr__(self, "right", replace(self.right))
        object.__setattr__(self, "rig_to_left", left_pose)
        object.__setattr__(self, "rig_to_right", right_pose)

    def require_production(self) -> None:
        """Require independently measured monocular and stereo fit artifacts."""

        self.left.require_production()
        self.right.require_production()
        try:
            self.provenance.require_kind(ProvenanceKind.MEASURED, "stereo calibration")
        except ProvenanceError as exc:
            raise CalibrationError(str(exc)) from exc
        if self.rms_reprojection_error_px is None or self.views_used is None:
            raise CalibrationError("production stereo calibration must include reprojection error and view count")

    def epipolar_errors(self, left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
        """Symmetric epipolar distances in undistorted pixels for held-out matched corners.

        Small errors check geometric consistency, not baseline scale or capture synchronization. Supply points
        from independent paired captures; a fit's own residuals are not held-out validation.
        """

        points = []
        for value, camera in ((left_points, self.left), (right_points, self.right)):
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 2 or not len(array) or not np.all(np.isfinite(array)):
                raise CalibrationError("epipolar points must be nonempty finite Nx2 arrays")
            undistorted = (
                _import_cv2()
                .undistortPoints(
                    array.reshape(-1, 1, 2),
                    camera.camera_matrix,
                    camera.distortion_coefficients,
                    P=camera.camera_matrix,
                )
                .reshape(-1, 2)
            )
            points.append(np.column_stack((undistorted, np.ones(len(array)))))
        if len(points[0]) != len(points[1]):
            raise CalibrationError("left and right epipolar points must have matching lengths")
        if np.array_equal(self.rig_to_left[:3, 3], self.rig_to_right[:3, 3]):
            raise CalibrationError("epipolar distances require distinct camera centers")
        right_from_left = invert_transform(self.rig_to_right) @ self.rig_to_left
        opencv = OPENCV_TO_WPILIB_BASIS.T @ right_from_left @ OPENCV_TO_WPILIB_BASIS
        essential = np.cross(opencv[:3, 3], opencv[:3, :3].T).T
        fundamental = np.linalg.inv(self.right.camera_matrix).T @ essential @ np.linalg.inv(self.left.camera_matrix)
        scale = np.linalg.norm(fundamental)
        if scale == 0:
            raise CalibrationError("epipolar distances require distinct camera centers")
        fundamental /= scale
        right_lines = points[0] @ fundamental.T
        left_lines = points[1] @ fundamental
        right_norm = np.sum(right_lines[:, :2] ** 2, axis=1)
        left_norm = np.sum(left_lines[:, :2] ** 2, axis=1)
        if np.any(right_norm == 0) or np.any(left_norm == 0):
            raise CalibrationError("epipolar distance is undefined at an epipole")
        residual = np.sum(points[1] * right_lines, axis=1)
        return np.abs(residual) * np.sqrt(0.5 / right_norm + 0.5 / left_norm)


def save_stereo_calibration(path: str | Path, calibration: StereoCalibration) -> None:
    """Write a validated, versioned stereo calibration artifact."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "coordinate_convention": "wpilib_camera_to_rig",
        "left": _camera_calibration_to_dict(calibration.left),
        "right": _camera_calibration_to_dict(calibration.right),
        "rig_to_left": calibration.rig_to_left.tolist(),
        "rig_to_right": calibration.rig_to_right.tolist(),
        "provenance": calibration.provenance.to_dict(),
        "left_id": calibration.left_id,
        "right_id": calibration.right_id,
        "rms_reprojection_error_px": calibration.rms_reprojection_error_px,
        "views_used": calibration.views_used,
    }
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
    except OSError as exc:
        raise CalibrationError(f"could not write stereo calibration {destination}: {exc}") from exc


def load_stereo_calibration(path: str | Path) -> StereoCalibration:
    """Load a strict-schema stereo calibration artifact."""

    source = Path(path).expanduser()
    try:
        with source.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise CalibrationError(f"stereo calibration file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"invalid stereo calibration JSON at line {exc.lineno}: {exc.msg}") from exc
    except OSError as exc:
        raise CalibrationError(f"could not read stereo calibration {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError("stereo calibration must be a JSON object")
    allowed = {
        "schema_version",
        "coordinate_convention",
        "left",
        "right",
        "rig_to_left",
        "rig_to_right",
        "provenance",
        "left_id",
        "right_id",
        "rms_reprojection_error_px",
        "views_used",
    }
    unknown = set(data) - allowed
    missing = allowed - set(data)
    if unknown:
        raise CalibrationError(f"unknown stereo calibration field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise CalibrationError(f"stereo calibration is missing field(s): {', '.join(sorted(missing))}")
    if (
        isinstance(data["schema_version"], bool)
        or not isinstance(data["schema_version"], int)
        or data["schema_version"] != 1
    ):
        raise CalibrationError(f"unsupported stereo calibration schema_version {data['schema_version']!r}; expected 1")
    if data["coordinate_convention"] != "wpilib_camera_to_rig":
        raise CalibrationError("coordinate_convention must be 'wpilib_camera_to_rig'")
    try:
        return StereoCalibration(
            left=_camera_calibration_from_dict(
                data["left"], default_source=str(source), require_schema_version=True, require_complete=True
            ),
            right=_camera_calibration_from_dict(
                data["right"], default_source=str(source), require_schema_version=True, require_complete=True
            ),
            rig_to_left=np.asarray(data["rig_to_left"], dtype=np.float64),
            rig_to_right=np.asarray(data["rig_to_right"], dtype=np.float64),
            provenance=ArtifactProvenance.from_dict(data["provenance"], default_source=str(source)),
            left_id=data["left_id"],
            right_id=data["right_id"],
            rms_reprojection_error_px=(
                None
                if data["rms_reprojection_error_px"] is None
                else _number(data["rms_reprojection_error_px"], "rms_reprojection_error_px", minimum=0.0)
            ),
            views_used=None if data["views_used"] is None else _positive_integer(data["views_used"], "views_used"),
        )
    except ProvenanceError as exc:
        raise CalibrationError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CalibrationError):
            raise
        raise CalibrationError("stereo calibration arrays must contain finite numbers") from exc


def _point_views(values: object, width: int, name: str) -> tuple[np.ndarray, ...]:
    try:
        raw_views = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise CalibrationError(f"{name} must contain one array per view") from exc
    converted: list[np.ndarray] = []
    for index, raw in enumerate(raw_views):
        try:
            points = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"{name}[{index}] must contain finite numeric points") from exc
        if points.ndim == 3 and points.shape[1:] == (1, width):
            points = points.reshape(-1, width)
        if points.ndim != 2 or points.shape[1] != width or len(points) < 4 or not np.all(np.isfinite(points)):
            raise CalibrationError(f"{name}[{index}] must be a finite Nx{width} array with at least 4 points")
        converted.append(np.ascontiguousarray(points, dtype=np.float32))
    return tuple(converted)


def calibrate_stereo_points(
    object_points: object,
    left_image_points: object,
    right_image_points: object,
    left: CameraCalibration,
    right: CameraCalibration,
    *,
    source: str,
) -> StereoCalibration:
    """Fit a left-origin stereo rig from matched checkerboard-corner observations."""

    if not isinstance(left, CameraCalibration) or not isinstance(right, CameraCalibration):
        raise CalibrationError("left and right must be CameraCalibration instances")
    if left.image_size is None or right.image_size is None:
        raise CalibrationError("stereo calibration cameras must record their image resolutions")
    if left.image_size != right.image_size:
        raise CalibrationError("stereo calibration cameras must use the same image resolution")
    if not isinstance(source, str) or not source.strip():
        raise CalibrationError("source must be a non-empty string")
    objects = _point_views(object_points, 3, "object_points")
    left_points = _point_views(left_image_points, 2, "left_image_points")
    right_points = _point_views(right_image_points, 2, "right_image_points")
    if len(objects) < 3 or len(left_points) != len(objects) or len(right_points) != len(objects):
        raise CalibrationError("stereo calibration requires at least 3 matched views")
    if any(
        len(world) != len(left_view) or len(world) != len(right_view)
        for world, left_view, right_view in zip(objects, left_points, right_points, strict=True)
    ):
        raise CalibrationError("each stereo view must contain matching object and image corner counts")

    cv2 = _import_cv2()
    try:
        rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
            list(objects),
            list(left_points),
            list(right_points),
            left.camera_matrix.copy(),
            left.distortion_coefficients.copy(),
            right.camera_matrix.copy(),
            right.distortion_coefficients.copy(),
            left.image_size,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    except cv2.error as exc:
        raise CalibrationError(f"OpenCV stereo calibration failed: {exc}") from exc

    left_to_right_opencv = np.eye(4, dtype=np.float64)
    left_to_right_opencv[:3, :3] = rotation
    left_to_right_opencv[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    left_to_right_wpilib = OPENCV_TO_WPILIB_BASIS @ left_to_right_opencv @ OPENCV_TO_WPILIB_BASIS.T
    return StereoCalibration(
        left=left,
        right=right,
        rig_to_left=np.eye(4, dtype=np.float64),
        rig_to_right=invert_transform(left_to_right_wpilib),
        provenance=ArtifactProvenance(ProvenanceKind.MEASURED, source),
        rms_reprojection_error_px=float(rms),
        views_used=len(objects),
    )
