from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .provenance import ArtifactProvenance, ProvenanceError, ProvenanceKind


class CalibrationError(ValueError):
    """Raised when camera calibration data cannot be measured or validated."""


@dataclass(frozen=True, slots=True)
class CheckerboardSpec:
    """Physical checkerboard definition using the number of inner corners."""

    columns: int
    rows: int
    square_size_m: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.columns, bool)
            or isinstance(self.rows, bool)
            or not isinstance(self.columns, int)
            or not isinstance(self.rows, int)
            or self.columns < 2
            or self.rows < 2
        ):
            raise CalibrationError("checkerboard must have at least 2x2 inner corners")
        if (
            isinstance(self.square_size_m, bool)
            or not isinstance(self.square_size_m, (int, float))
            or not np.isfinite(self.square_size_m)
            or self.square_size_m <= 0.0
        ):
            raise CalibrationError("checkerboard square_size_m must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Validated camera intrinsics together with measurement provenance."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int] | None
    provenance: ArtifactProvenance
    rms_reprojection_error_px: float | None = None
    mean_reprojection_error_px: float | None = None
    views_used: int | None = None
    checkerboard: CheckerboardSpec | None = None
    rejected_images: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ArtifactProvenance):
            raise CalibrationError("provenance must be an ArtifactProvenance")
        if self.checkerboard is not None and not isinstance(self.checkerboard, CheckerboardSpec):
            raise CalibrationError("checkerboard must be a CheckerboardSpec or null")
        if not isinstance(self.rejected_images, (tuple, list)) or not all(
            isinstance(value, str) for value in self.rejected_images
        ):
            raise CalibrationError("rejected_images must contain strings")
        try:
            matrix = np.asarray(self.camera_matrix, dtype=np.float64)
            distortion = np.asarray(self.distortion_coefficients, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise CalibrationError("calibration arrays must contain finite numbers") from exc
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise CalibrationError("camera_matrix must be a finite 3x3 matrix")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or abs(float(np.linalg.det(matrix))) < 1e-12:
            raise CalibrationError("camera_matrix must have positive focal lengths and be invertible")
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-12) or abs(float(matrix[1, 0])) > 1e-12:
            raise CalibrationError("camera_matrix must use the canonical pinhole form")
        if len(distortion) not in {4, 5, 8, 12, 14} or not np.all(np.isfinite(distortion)):
            raise CalibrationError("distortion_coefficients must contain 4, 5, 8, 12, or 14 finite values")
        if self.image_size is not None:
            width, height = self.image_size
            if (
                isinstance(width, bool)
                or isinstance(height, bool)
                or not isinstance(width, int)
                or not isinstance(height, int)
                or width <= 0
                or height <= 0
            ):
                raise CalibrationError("image_size must contain positive integer width and height")
        for value, name in (
            (self.rms_reprojection_error_px, "rms_reprojection_error_px"),
            (self.mean_reprojection_error_px, "mean_reprojection_error_px"),
        ):
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or value < 0.0
                ):
                    raise CalibrationError(f"{name} must be finite and non-negative")
        if self.views_used is not None:
            if isinstance(self.views_used, bool) or not isinstance(self.views_used, int) or self.views_used < 1:
                raise CalibrationError("views_used must be a positive integer")
        matrix = matrix.copy()
        distortion = distortion.copy()
        matrix.setflags(write=False)
        distortion.setflags(write=False)
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)
        object.__setattr__(self, "rejected_images", tuple(self.rejected_images))

    def validate_for_image_size(self, width: int, height: int) -> None:
        if self.image_size is None:
            raise CalibrationError("calibration does not record its image resolution")
        if self.image_size != (width, height):
            raise CalibrationError(
                f"calibration was measured at {self.image_size[0]}x{self.image_size[1]}, not {width}x{height}"
            )

    def require_production(self) -> None:
        try:
            self.provenance.require_kind(ProvenanceKind.MEASURED, "camera calibration")
        except ProvenanceError as exc:
            raise CalibrationError(str(exc)) from exc
        if self.image_size is None:
            raise CalibrationError("production calibration must record its image resolution")
        if self.rms_reprojection_error_px is None or self.views_used is None:
            raise CalibrationError("production calibration must include reprojection error and view count")


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{name} must be a number")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" at least {minimum}" if minimum is not None else " finite"
        raise CalibrationError(f"{name} must be{qualifier}")
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CalibrationError(f"{name} must be a positive integer")
    return value


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        np.asarray(data["camera_matrix"], dtype=np.float64),
        np.asarray(data.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64),
    )


def save_calibration(path: str | Path, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> None:
    payload = {
        "camera_matrix": np.asarray(camera_matrix, dtype=float).tolist(),
        "distortion_coefficients": np.asarray(distortion_coefficients, dtype=float).reshape(-1).tolist(),
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def undistorted_corner_angles(
    corners: np.ndarray, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray
) -> np.ndarray:
    """Convert image corners to horizontal/vertical optical angles.

    Omitting ``P`` makes OpenCV return normalized ideal-image coordinates, so
    this avoids rebuilding and inverting the same camera matrix for every tag.
    """

    normalized = cv2.undistortPoints(
        np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix,
        distortion_coefficients,
    ).reshape(-1, 2)
    return np.arctan(normalized).astype(np.float64, copy=False)
