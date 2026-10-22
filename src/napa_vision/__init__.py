"""Napa: a fast tag36h11 AprilTag detector with optional pose processing."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from ._version import __version__

_OPENCV_INSTALL_HINT = (
    "OpenCV is not installed; pose processing and calibration need it. Install "
    "napa-vision[headless] (or napa-vision[gui] for desktop display support)"
)

_EXPORT_MODULES = {
    "ArtifactProvenance": "provenance",
    "CalibrationError": "calibration",
    "CameraCalibration": "calibration",
    "CameraPoseBounds": "types",
    "CameraPoseObservation": "types",
    "CameraPoseSolver": "solvers",
    "CheckerboardSpec": "calibration",
    "FiducialDetector": "types",
    "FiducialImageObservation": "types",
    "FrameResult": "types",
    "NapaDetector": "napa_detector",
    "NativeLibraryError": "napa_detector",
    "Pose2dWithCovariance": "covariance",
    "PoseSolveDiagnostics": "types",
    "PoseWithCovariance": "covariance",
    "ProvenanceError": "provenance",
    "ProvenanceKind": "provenance",
    "SingleTagPoseObservation": "types",
    "SingleTagPoseSolver": "solvers",
    "TagAngleObservation": "types",
    "TagAngleSolver": "solvers",
    "VisionProcessor": "processing",
    "calibrate_checkerboard_images": "calibration",
    "camera_pose_covariance": "covariance",
    "discover_checkerboard_images": "calibration",
    "families": "napa_detector",
    "load_calibration_data": "calibration",
    "native_available": "napa_detector",
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
