from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .covariance import PoseWithCovariance, change_basis
from .geometry import OPENCV_TO_WPILIB_ROTATION, opencv_pose_to_wpilib_transform, tag_homography


@dataclass(slots=True)
class FiducialImageObservation:
    """One detected tag: its ID and four corners in pixel-centre coordinates, OpenCV corner order.

    ``hamming`` is the number of payload bits corrected (0 is a perfect read) and
    ``decision_margin`` the smaller of the white and black payload bits' mean distance from the
    decision threshold, in grey levels; both as in upstream AprilTag. Both are ``None`` for
    observations not produced by a detector that reports them.

    ``homography`` and ``center`` are computed from the corners on each access.
    """

    tag_id: int
    corners: np.ndarray
    hamming: int | None = None
    decision_margin: float | None = None

    @property
    def homography(self) -> np.ndarray:
        """3x3 map from upstream AprilTag's tag coordinates (``[-1, 1]`` across the black square) to pixels.

        As upstream's ``H``: ``H[2, 2] == 1`` and the tag centre is ``H @ (0, 0, 1)``. Pixels are in the
        same coordinates as ``corners``.
        """

        return tag_homography(self.corners)

    @property
    def center(self) -> np.ndarray:
        """The tag centre in pixels, ``(x, y)``: the homography's image of the tag origin, as upstream's ``c``.

        This is where the square's diagonals cross, which under perspective differs from the corners' mean.
        """

        return self.homography[:2, 2].copy()


class FiducialDetector(Protocol):
    """Detect tags in uint8 grayscale, BGR, or BGRA images, in the caller's pixel coordinates."""

    def detect(self, image: np.ndarray) -> list[FiducialImageObservation]: ...


def _solution(solution: int) -> int:
    if solution not in (0, 1) or isinstance(solution, bool):
        raise ValueError("solution must be 0 or 1")
    return solution


@dataclass(slots=True)
class SingleTagPoseObservation:
    """Two planar IPPE solutions for a tag, expressed in OpenCV coordinates.

    Each ``rvec``/``tvec`` maps tag-local points into the OpenCV camera frame (+X right, +Y down, +Z forward).
    ``error_*`` is the solution's RMS corner reprojection error in pixels; solution 0 has the lower error.
    ``covariance_*`` is the 6x6 covariance of that camera-to-tag pose in the OpenCV camera frame (see
    ``napa_vision.covariance``), or ``None`` when it wasn't computed.
    """

    tag_id: int
    pose_0_rvec: np.ndarray
    pose_0_tvec: np.ndarray
    error_0: float
    pose_1_rvec: np.ndarray | None
    pose_1_tvec: np.ndarray | None
    error_1: float | None
    distance: float
    covariance_0: np.ndarray | None = None
    covariance_1: np.ndarray | None = None

    @property
    def ambiguity(self) -> float | None:
        """``error_0 / error_1``, as PhotonVision and Limelight report it: near 1 means the two solutions fit
        equally well and solution 0 may be the wrong one. ``None`` without a second solution."""

        if self.error_1 is None:
            return None
        return self.error_0 / self.error_1 if self.error_1 > 0.0 else 1.0

    def camera_to_tag(self, solution: int = 0) -> PoseWithCovariance:
        """The tag's pose in WPILib camera axes (+X forward, +Y left, +Z up), with its covariance.

        The tag frame is WPILib's: its face lies in its local Y/Z plane and +X is the face normal.
        """

        rvec, tvec, covariance = (
            (self.pose_0_rvec, self.pose_0_tvec, self.covariance_0)
            if _solution(solution) == 0
            else (self.pose_1_rvec, self.pose_1_tvec, self.covariance_1)
        )
        if rvec is None or tvec is None:
            raise ValueError(f"solution {solution} is unavailable")
        if covariance is None:
            raise ValueError(f"solution {solution} has no covariance")
        return PoseWithCovariance(
            opencv_pose_to_wpilib_transform(rvec, tvec), change_basis(covariance, OPENCV_TO_WPILIB_ROTATION)
        )


@dataclass(frozen=True, slots=True)
class CameraPoseBounds:
    """Optional physical bounds for accepting a field-to-camera pose."""

    field_length_m: float | None = None
    field_width_m: float | None = None
    field_margin_m: float = 0.0
    min_camera_height_m: float | None = None
    max_camera_height_m: float | None = None

    def __post_init__(self) -> None:
        values = {
            "field_length_m": self.field_length_m,
            "field_width_m": self.field_width_m,
            "field_margin_m": self.field_margin_m,
            "min_camera_height_m": self.min_camera_height_m,
            "max_camera_height_m": self.max_camera_height_m,
        }
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value))
            for value in values.values()
        ):
            raise ValueError("camera pose bounds must contain finite numbers")
        if self.field_length_m is not None and self.field_length_m <= 0.0:
            raise ValueError("field_length_m must be greater than zero")
        if self.field_width_m is not None and self.field_width_m <= 0.0:
            raise ValueError("field_width_m must be greater than zero")
        if self.field_margin_m < 0.0:
            raise ValueError("field_margin_m must be non-negative")
        if (
            self.min_camera_height_m is not None
            and self.max_camera_height_m is not None
            and self.min_camera_height_m > self.max_camera_height_m
        ):
            raise ValueError("min_camera_height_m must not exceed max_camera_height_m")

    def rejection_reason(self, position: np.ndarray) -> str | None:
        x, y, z = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite([x, y, z])):
            return "camera position is not finite"
        margin = self.field_margin_m
        if self.field_length_m is not None and not (-margin <= x <= self.field_length_m + margin):
            return f"camera x={x:.3f}m is outside configured field bounds"
        if self.field_width_m is not None and not (-margin <= y <= self.field_width_m + margin):
            return f"camera y={y:.3f}m is outside configured field bounds"
        if self.min_camera_height_m is not None and z < self.min_camera_height_m:
            return f"camera height {z:.3f}m is below {self.min_camera_height_m:.3f}m"
        if self.max_camera_height_m is not None and z > self.max_camera_height_m:
            return f"camera height {z:.3f}m is above {self.max_camera_height_m:.3f}m"
        return None


@dataclass(slots=True)
class PoseSolveDiagnostics:
    """Explain which tags contributed to a global camera-pose estimate."""

    attempted_tag_ids: list[int] = field(default_factory=list)
    accepted_tag_ids: list[int] = field(default_factory=list)
    rejected_tag_reasons: dict[int, str] = field(default_factory=dict)
    per_tag_rms_reprojection_error_px: dict[int, float] = field(default_factory=dict)
    overall_rms_reprojection_error_px: float | None = None
    iterations: int = 0
    used_ransac: bool = False
    rejection_reason: str | None = None


@dataclass(slots=True)
class CameraPoseObservation:
    """One or two candidate WPILib camera poses in the field frame.

    ``covariance_*`` is the 6x6 covariance of ``field_to_camera_*`` in the field frame (see
    ``napa_vision.covariance``), or ``None`` when it wasn't computed.
    """

    tag_ids: list[int]
    field_to_camera_0: np.ndarray
    error_0: float
    field_to_camera_1: np.ndarray | None
    error_1: float | None
    average_tag_distance: float
    confidence: float
    diagnostics: PoseSolveDiagnostics | None = None
    covariance_0: np.ndarray | None = None
    covariance_1: np.ndarray | None = None

    @property
    def ambiguity(self) -> float | None:
        """``error_0 / error_1`` for a single-tag pose; ``None`` for a multi-tag pose, which has one solution."""

        if self.error_1 is None:
            return None
        return self.error_0 / self.error_1 if self.error_1 > 0.0 else 1.0

    def field_to_camera(self, solution: int = 0) -> PoseWithCovariance:
        """The camera's pose in the field, with its covariance, ready for a filter or ``transform_by``."""

        transform, covariance = (
            (self.field_to_camera_0, self.covariance_0)
            if _solution(solution) == 0
            else (self.field_to_camera_1, self.covariance_1)
        )
        if transform is None:
            raise ValueError(f"solution {solution} is unavailable")
        if covariance is None:
            raise ValueError(f"solution {solution} has no covariance")
        return PoseWithCovariance(transform, covariance)


@dataclass(slots=True)
class TagAngleObservation:
    """Viewing angles of a tag, in radians from the optical axis, corrected for lens distortion.

    ``corner_angles`` is 4x2, ``(horizontal, vertical)`` per corner, positive right and down (OpenCV's image axes).
    ``center_tx``/``center_ty`` are the corners' mean angles. ``center_covariance`` is their 2x2 covariance in
    radians squared, or ``None`` when it wasn't computed. ``distance`` is the camera-to-tag distance in metres
    from solution 0, and ``camera_to_tag_translation`` that solution's translation in WPILib camera axes.
    """

    tag_id: int
    corner_angles: np.ndarray
    center_tx: float
    center_ty: float
    distance: float
    camera_to_tag_translation: np.ndarray
    center_covariance: np.ndarray | None = None


@dataclass(slots=True)
class FrameResult:
    timestamp: float
    capture_monotonic_s: float | None = None
    image_observations: list[FiducialImageObservation] = field(default_factory=list)
    single_tag_poses: list[SingleTagPoseObservation] = field(default_factory=list)
    camera_pose: CameraPoseObservation | None = None
    pose_diagnostics: PoseSolveDiagnostics | None = None
    tag_angles: list[TagAngleObservation] = field(default_factory=list)
    health: dict[str, Any] = field(default_factory=dict)
