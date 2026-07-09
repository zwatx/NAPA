"""AprilTag detection and Northstar-compatible pose publishing."""

from ._version import __version__
from .calibration import (
    CalibrationError,
    CameraCalibration,
    CheckerboardSpec,
    calibrate_checkerboard_images,
    discover_checkerboard_images,
    load_calibration_data,
    save_calibration_data,
)
from .config import CameraConfig, PlatformProfile, ProfileError, available_profiles, load_profile
from .fiducials import OpenCVArucoFiducialDetector
from .field_layout import AprilTagFieldLayout, FieldLayoutError, load_wpilib_apriltag_layout
from .prefilter import BlockSkipPrefilter, Roi, SkipScanConfig
from .provenance import ArtifactProvenance, ProvenanceError, ProvenanceKind, require_production_artifacts
from .solvers import CameraPoseSolver, SingleTagPoseSolver, TagAngleSolver, TrigDistanceSolver
from .types import (
    CameraPoseBounds,
    CameraPoseObservation,
    FiducialImageObservation,
    FrameResult,
    PoseSolveDiagnostics,
    SingleTagPoseObservation,
    TagAngleObservation,
    TrigPoseObservation,
)

__all__ = [
    "AprilTagFieldLayout",
    "ArtifactProvenance",
    "BlockSkipPrefilter",
    "CalibrationError",
    "CameraCalibration",
    "CameraConfig",
    "CameraPoseObservation",
    "CameraPoseBounds",
    "CameraPoseSolver",
    "CheckerboardSpec",
    "FieldLayoutError",
    "FiducialImageObservation",
    "FrameResult",
    "OpenCVArucoFiducialDetector",
    "PlatformProfile",
    "PoseSolveDiagnostics",
    "ProfileError",
    "ProvenanceError",
    "ProvenanceKind",
    "Roi",
    "SingleTagPoseObservation",
    "SingleTagPoseSolver",
    "SkipScanConfig",
    "TagAngleObservation",
    "TagAngleSolver",
    "TrigDistanceSolver",
    "TrigPoseObservation",
    "available_profiles",
    "__version__",
    "calibrate_checkerboard_images",
    "discover_checkerboard_images",
    "load_calibration_data",
    "load_profile",
    "load_wpilib_apriltag_layout",
    "require_production_artifacts",
    "save_calibration_data",
]
