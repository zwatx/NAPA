from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
    tag_angles: list[TagAngleObservation] = field(default_factory=list)
    trig_poses: list[TrigPoseObservation] = field(default_factory=list)
    health: dict[str, Any] = field(default_factory=dict)
