from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def health_state_fields(health: Mapping[str, Any]) -> dict[str, Any]:
    """Return top-level health fields that represent immediate state transitions."""

    return {
        key: value
        for key, value in health.items()
        if key in {"connected", "state"} or key.endswith(("_available", "_connected", "_error", "_timed_out"))
    }


@dataclass(slots=True)
class FiducialImageObservation:
    tag_id: int
    corners: np.ndarray


@dataclass(slots=True)
class SingleTagPoseObservation:
    """Two planar IPPE solutions for a tag, expressed in OpenCV coordinates."""

    tag_id: int
    pose_0_rvec: np.ndarray
    pose_0_tvec: np.ndarray
    error_0: float
    pose_1_rvec: np.ndarray | None
    pose_1_tvec: np.ndarray | None
    error_1: float | None
    distance: float


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
    """One or two candidate WPILib camera poses in the field frame."""

    tag_ids: list[int]
    field_to_camera_0: np.ndarray
    error_0: float
    field_to_camera_1: np.ndarray | None
    error_1: float | None
    average_tag_distance: float
    confidence: float
    diagnostics: PoseSolveDiagnostics | None = None


@dataclass(slots=True)
class TagAngleObservation:
    tag_id: int
    corner_angles: np.ndarray
    center_tx: float
    center_ty: float
    distance: float
    camera_to_tag_translation: np.ndarray


@dataclass(slots=True)
class TrigPoseObservation:
    tag_id: int
    robot_pose: np.ndarray
    camera_to_tag_translation: np.ndarray
    distance: float


@dataclass(slots=True)
class FrameResult:
    timestamp: float
    capture_monotonic_s: float | None = None
    image_observations: list[FiducialImageObservation] = field(default_factory=list)
    single_tag_poses: list[SingleTagPoseObservation] = field(default_factory=list)
    camera_pose: CameraPoseObservation | None = None
    pose_diagnostics: PoseSolveDiagnostics | None = None
    tag_angles: list[TagAngleObservation] = field(default_factory=list)
    trig_poses: list[TrigPoseObservation] = field(default_factory=list)
    health: dict[str, Any] = field(default_factory=dict)
