"""Napa: modular AprilTag perception and pose processing."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from ._version import __version__

_OPENCV_INSTALL_HINT = (
    "OpenCV is not installed; install napa-vision[gui] for desktop display support "
    "or napa-vision[headless] for servers and robots"
)

_EXPORT_MODULES = {
    "AprilTagFieldLayout": "field_layout",
    "ArtifactProvenance": "provenance",
    "BlockSkipPrefilter": "prefilter",
    "CalibrationError": "calibration",
    "CameraCalibration": "calibration",
    "CameraConfig": "config",
    "CameraPoseBounds": "types",
    "CameraPoseObservation": "types",
    "CameraPoseSolver": "solvers",
    "CheckerboardSpec": "calibration",
    "FieldLayoutError": "field_layout",
    "FiducialDetector": "fiducials",
    "FiducialImageObservation": "types",
    "FrameResult": "types",
    "OpenCVArucoFiducialDetector": "fiducials",
    "PlatformProfile": "config",
    "PoseSolveDiagnostics": "types",
    "ProfileError": "config",
    "ProvenanceError": "provenance",
    "ProvenanceKind": "provenance",
    "Roi": "prefilter",
    "SingleTagPoseObservation": "types",
    "SingleTagPoseSolver": "solvers",
    "SkipScanConfig": "prefilter",
    "TagAngleObservation": "types",
    "TagAngleSolver": "solvers",
    "TrigDistanceSolver": "solvers",
    "TrigPoseObservation": "types",
    "VisionProcessor": "processing",
    "available_profiles": "config",
    "calibrate_checkerboard_images": "calibration",
    "discover_checkerboard_images": "calibration",
    "load_calibration_data": "calibration",
    "load_profile": "config",
    "load_wpilib_apriltag_layout": "field_layout",
    "NapaDetector": "napa_detector",
    "NativeLibraryError": "napa_detector",
    "default_detector": "fiducials",
    "native_available": "napa_detector",
    "require_production_artifacts": "provenance",
    "save_calibration_data": "calibration",
}

__all__ = [*_EXPORT_MODULES, "__version__"]


def _import_cv2() -> ModuleType:
    try:
        return import_module("cv2")
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            raise RuntimeError(_OPENCV_INSTALL_HINT) from exc
        raise


def __getattr__(name: str) -> Any:
    """Load public vision APIs only when requested."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            raise RuntimeError(_OPENCV_INSTALL_HINT) from exc
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
