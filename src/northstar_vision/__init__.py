"""AprilTag detection and Northstar-compatible pose publishing."""

from .config import CameraConfig, PlatformProfile, ProfileError, available_profiles, load_profile
from .fiducials import OpenCVArucoFiducialDetector
from .prefilter import BlockSkipPrefilter, Roi, SkipScanConfig
from .solvers import CameraPoseSolver, SingleTagPoseSolver, TagAngleSolver, TrigDistanceSolver
from .types import (
    CameraPoseObservation,
    FiducialImageObservation,
    FrameResult,
    SingleTagPoseObservation,
    TagAngleObservation,
    TrigPoseObservation,
)

__version__ = "0.2.0"

__all__ = [
    "BlockSkipPrefilter",
    "CameraConfig",
    "CameraPoseObservation",
    "CameraPoseSolver",
    "FiducialImageObservation",
    "FrameResult",
    "OpenCVArucoFiducialDetector",
    "PlatformProfile",
    "ProfileError",
    "Roi",
    "SingleTagPoseObservation",
    "SingleTagPoseSolver",
    "SkipScanConfig",
    "TagAngleObservation",
    "TagAngleSolver",
    "TrigDistanceSolver",
    "TrigPoseObservation",
    "available_profiles",
    "load_profile",
]
