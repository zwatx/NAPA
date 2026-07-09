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


def load_calibration_data(path: str | Path) -> CameraCalibration:
    """Load and validate a versioned or legacy Northstar calibration file."""

    source = Path(path).expanduser()
    try:
        with source.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise CalibrationError(f"calibration file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"invalid calibration JSON at line {exc.lineno}: {exc.msg}") from exc
    except OSError as exc:
        raise CalibrationError(f"could not read calibration {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError("calibration must be a JSON object")
    allowed = {
        "schema_version",
        "camera_matrix",
        "distortion_coefficients",
        "image_size",
        "rms_reprojection_error_px",
        "mean_reprojection_error_px",
        "views_used",
        "checkerboard",
        "provenance",
        "rejected_images",
    }
    unknown = set(data) - allowed
    if unknown:
        raise CalibrationError(f"unknown calibration field(s): {', '.join(sorted(unknown))}")
    schema_version = data.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        raise CalibrationError(f"unsupported calibration schema_version {schema_version!r}; expected 1")
    if "camera_matrix" not in data:
        raise CalibrationError("calibration is missing camera_matrix")

    image_size_data = data.get("image_size")
    image_size: tuple[int, int] | None = None
    if image_size_data is not None:
        if not isinstance(image_size_data, dict):
            raise CalibrationError("image_size must be a JSON object")
        unknown = set(image_size_data) - {"width", "height"}
        if unknown:
            raise CalibrationError(f"unknown image_size field(s): {', '.join(sorted(unknown))}")
        image_size = (
            _positive_integer(image_size_data.get("width"), "image_size.width"),
            _positive_integer(image_size_data.get("height"), "image_size.height"),
        )

    checkerboard_data = data.get("checkerboard")
    checkerboard = None
    if checkerboard_data is not None:
        if not isinstance(checkerboard_data, dict):
            raise CalibrationError("checkerboard must be a JSON object")
        unknown = set(checkerboard_data) - {"columns", "rows", "square_size_m"}
        if unknown:
            raise CalibrationError(f"unknown checkerboard field(s): {', '.join(sorted(unknown))}")
        checkerboard = CheckerboardSpec(
            columns=_positive_integer(checkerboard_data.get("columns"), "checkerboard.columns"),
            rows=_positive_integer(checkerboard_data.get("rows"), "checkerboard.rows"),
            square_size_m=_number(checkerboard_data.get("square_size_m"), "checkerboard.square_size_m", minimum=0.0),
        )

    rejected = data.get("rejected_images", [])
    if not isinstance(rejected, list) or not all(isinstance(value, str) for value in rejected):
        raise CalibrationError("rejected_images must be an array of strings")
    try:
        provenance = ArtifactProvenance.from_dict(data.get("provenance"), default_source=str(source))
        return CameraCalibration(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            distortion_coefficients=np.asarray(
                data.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64
            ),
            image_size=image_size,
            provenance=provenance,
            rms_reprojection_error_px=(
                None
                if data.get("rms_reprojection_error_px") is None
                else _number(data["rms_reprojection_error_px"], "rms_reprojection_error_px", minimum=0.0)
            ),
            mean_reprojection_error_px=(
                None
                if data.get("mean_reprojection_error_px") is None
                else _number(data["mean_reprojection_error_px"], "mean_reprojection_error_px", minimum=0.0)
            ),
            views_used=(
                None if data.get("views_used") is None else _positive_integer(data["views_used"], "views_used")
            ),
            checkerboard=checkerboard,
            rejected_images=tuple(rejected),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, (CalibrationError, ProvenanceError)):
            raise
        raise CalibrationError("calibration arrays must contain finite numbers") from exc


def save_calibration_data(path: str | Path, calibration: CameraCalibration) -> None:
    """Write a validated, self-describing calibration artifact."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "camera_matrix": calibration.camera_matrix.tolist(),
        "distortion_coefficients": calibration.distortion_coefficients.tolist(),
        "image_size": (
            None
            if calibration.image_size is None
            else {"width": calibration.image_size[0], "height": calibration.image_size[1]}
        ),
        "rms_reprojection_error_px": calibration.rms_reprojection_error_px,
        "mean_reprojection_error_px": calibration.mean_reprojection_error_px,
        "views_used": calibration.views_used,
        "provenance": calibration.provenance.to_dict(),
        "rejected_images": list(calibration.rejected_images),
    }
    if calibration.checkerboard is not None:
        payload["checkerboard"] = {
            "columns": calibration.checkerboard.columns,
            "rows": calibration.checkerboard.rows,
            "square_size_m": calibration.checkerboard.square_size_m,
        }
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
    except OSError as exc:
        raise CalibrationError(f"could not write calibration {destination}: {exc}") from exc


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load camera arrays from a legacy or versioned calibration file."""

    calibration = load_calibration_data(path)
    return calibration.camera_matrix.copy(), calibration.distortion_coefficients.copy()


def save_calibration(path: str | Path, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> None:
    """Save camera arrays as an explicitly non-production example artifact."""

    save_calibration_data(
        path,
        CameraCalibration(
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            image_size=None,
            provenance=ArtifactProvenance(
                ProvenanceKind.EXAMPLE,
                "legacy save_calibration call without measurement metadata",
            ),
        ),
    )


def discover_checkerboard_images(directory: str | Path) -> tuple[Path, ...]:
    """Return common image files in deterministic order for calibration."""

    root = Path(directory).expanduser()
    if not root.is_dir():
        raise CalibrationError(f"checkerboard image directory not found: {root}")
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    return tuple(sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes))


def calibrate_checkerboard_images(
    image_paths: Sequence[str | Path],
    checkerboard: CheckerboardSpec,
    *,
    minimum_views: int = 8,
    calibration_flags: int = 0,
) -> CameraCalibration:
    """Measure intrinsics from checkerboard photographs.

    Images without a complete checkerboard are recorded in ``rejected_images``.
    All accepted images must have the same resolution. ``columns`` and ``rows``
    in ``checkerboard`` are inner-corner counts, not square counts.
    """

    if minimum_views < 3:
        raise CalibrationError("minimum_views must be at least 3")
    paths = tuple(Path(path).expanduser() for path in image_paths)
    if not paths:
        raise CalibrationError("at least one checkerboard image path is required")

    template = np.zeros((checkerboard.columns * checkerboard.rows, 3), dtype=np.float32)
    template[:, :2] = np.mgrid[0 : checkerboard.columns, 0 : checkerboard.rows].T.reshape(-1, 2)
    template *= checkerboard.square_size_m
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted_paths: list[Path] = []
    rejected_paths: list[str] = []
    image_size: tuple[int, int] | None = None
    detection_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rejected_paths.append(f"{path}: unreadable")
            continue
        current_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise CalibrationError(
                f"checkerboard images mix resolutions {image_size[0]}x{image_size[1]} and "
                f"{current_size[0]}x{current_size[1]} ({path})"
            )
        found, corners = cv2.findChessboardCorners(
            image,
            (checkerboard.columns, checkerboard.rows),
            flags=detection_flags,
        )
        if not found or corners is None:
            rejected_paths.append(f"{path}: complete checkerboard not found")
            continue
        refined = cv2.cornerSubPix(
            image,
            np.asarray(corners, dtype=np.float32),
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
        object_points.append(template.copy())
        image_points.append(refined)
        accepted_paths.append(path)

    if image_size is None:
        raise CalibrationError("none of the checkerboard images could be read")
    if len(image_points) < minimum_views:
        raise CalibrationError(
            f"only {len(image_points)} complete checkerboard views were found; at least {minimum_views} are required"
        )
    try:
        rms, camera_matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=calibration_flags,
        )
    except cv2.error as exc:
        raise CalibrationError(f"OpenCV checkerboard calibration failed: {exc}") from exc

    view_errors: list[float] = []
    for world_points, detected, rvec, tvec in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, distortion)
        residual = projected.reshape(-1, 2) - detected.reshape(-1, 2)
        view_errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
    source = f"checkerboard calibration from {len(accepted_paths)} images"
    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        image_size=image_size,
        provenance=ArtifactProvenance(ProvenanceKind.MEASURED, source),
        rms_reprojection_error_px=float(rms),
        mean_reprojection_error_px=float(np.mean(view_errors)),
        views_used=len(image_points),
        checkerboard=checkerboard,
        rejected_images=tuple(rejected_paths),
    )


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
