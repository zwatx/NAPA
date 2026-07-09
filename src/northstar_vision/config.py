from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import transform_matrix
from .prefilter import SkipScanConfig


class ProfileError(ValueError):
    """Raised when a profile is missing, malformed, or internally inconsistent."""


@dataclass
class CameraConfig:
    camera_id: int | str
    width: int
    height: int
    fps: float | None
    buffer_size: int | None
    auto_exposure: float | int | None
    exposure: float | int | None
    gain: float | int | None
    calibration_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    robot_to_camera: np.ndarray


@dataclass
class PlatformProfile:
    name: str
    device_id: str
    platform: str
    replay_path: str | None
    camera: CameraConfig
    tag_size_m: float
    tag_layout: dict[int, np.ndarray]
    skip_scan: SkipScanConfig
    max_reprojection_error_px: float
    output_mode: str
    output_path: str
    nt_server: str


def available_profiles() -> tuple[str, ...]:
    """Return the names of profiles bundled in the installed package."""

    profile_root = resources.files("northstar_vision.profiles")
    return tuple(
        sorted(item.name.removesuffix(".json") for item in profile_root.iterdir() if item.name.endswith(".json"))
    )


def _camera_id(value: Any) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ProfileError("camera.id must be an integer index or device path")
    if isinstance(value, str):
        value = value.strip()
        if value.lstrip("-").isdigit():
            value = int(value)
        elif not value:
            raise ProfileError("camera.id must not be empty")
    if isinstance(value, int) and value < 0:
        raise ProfileError("camera.id index must not be negative")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileError(f"{context} must be a JSON object")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ProfileError(f"Unknown {context} field(s): {', '.join(unknown)}")


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{context} must be a number")
    result = float(value)
    if not np.isfinite(result):
        raise ProfileError(f"{context} must be finite")
    return result


def _positive_number(value: Any, context: str) -> float:
    result = _number(value, context)
    if result <= 0:
        raise ProfileError(f"{context} must be greater than zero")
    return result


def _optional_number(value: Any, context: str) -> float | None:
    return None if value is None else _number(value, context)


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{context} must be a non-empty string")
    return value.strip()


def _integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ProfileError(f"{context} must be at least {minimum}")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ProfileError(f"{context} must be true or false")
    return value


def _vector(value: Any, length: int, context: str) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{context} must contain {length} finite numbers") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ProfileError(f"{context} must contain {length} finite numbers")
    return array.tolist()


def _load_json(source: Any) -> dict[str, Any]:
    try:
        with source.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise ProfileError(f"Profile not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"Invalid JSON in profile {source}: {exc.msg} at line {exc.lineno}") from exc
    except OSError as exc:
        raise ProfileError(f"Could not read profile {source}: {exc}") from exc
    return dict(_mapping(data, "profile"))


def _resolve_profile(profile: str | Path) -> Any:
    raw = Path(profile).expanduser()
    if raw.suffix == ".json" or raw.exists() or raw.parent != Path("."):
        return raw
    return resources.files("northstar_vision.profiles").joinpath(f"{raw.name}.json")


def load_profile(profile: str | Path) -> PlatformProfile:
    """Load and validate a bundled profile name or an external JSON file."""

    source = _resolve_profile(profile)
    data = _load_json(source)
    _reject_unknown(
        data,
        {
            "schema_version",
            "name",
            "device_id",
            "platform",
            "replay_path",
            "camera",
            "tag_size_m",
            "tag_layout",
            "detection",
            "output",
        },
        "profile",
    )
    schema_version = _integer(data.get("schema_version", 1), "schema_version", minimum=1)
    if schema_version != 1:
        raise ProfileError(f"Unsupported profile schema_version {schema_version}; expected 1")

    camera_data = _mapping(data.get("camera"), "camera")
    _reject_unknown(
        camera_data,
        {
            "id",
            "width",
            "height",
            "fps",
            "buffer_size",
            "auto_exposure",
            "exposure",
            "gain",
            "calibration",
            "robot_to_camera",
        },
        "camera",
    )
    width = _integer(camera_data.get("width"), "camera.width", minimum=1)
    height = _integer(camera_data.get("height"), "camera.height", minimum=1)
    fps_value = camera_data.get("fps")
    fps = None if fps_value is None else _positive_number(fps_value, "camera.fps")
    buffer_value = camera_data.get("buffer_size")
    buffer_size = None if buffer_value is None else _integer(buffer_value, "camera.buffer_size", minimum=1)

    calibration = _mapping(camera_data.get("calibration", {}), "camera.calibration")
    _reject_unknown(calibration, {"camera_matrix", "distortion_coefficients"}, "camera.calibration")
    try:
        camera_matrix = np.asarray(
            calibration.get(
                "camera_matrix",
                [[width, 0.0, width / 2.0], [0.0, width, height / 2.0], [0.0, 0.0, 1.0]],
            ),
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ProfileError("camera.calibration.camera_matrix must be a finite 3x3 matrix") from exc
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise ProfileError("camera.calibration.camera_matrix must be a finite 3x3 matrix")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0 or abs(float(np.linalg.det(camera_matrix))) < 1e-12:
        raise ProfileError("camera.calibration.camera_matrix must have positive focal lengths and be invertible")
    if not np.allclose(camera_matrix[2], [0.0, 0.0, 1.0], atol=1e-12) or abs(camera_matrix[1, 0]) > 1e-12:
        raise ProfileError("camera.calibration.camera_matrix must use the canonical pinhole form")
    try:
        distortion = np.asarray(calibration.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(
            -1
        )
    except (TypeError, ValueError) as exc:
        raise ProfileError(
            "camera.calibration.distortion_coefficients must contain 4, 5, 8, 12, or 14 finite values"
        ) from exc
    if len(distortion) not in {4, 5, 8, 12, 14} or not np.all(np.isfinite(distortion)):
        raise ProfileError("camera.calibration.distortion_coefficients must contain 4, 5, 8, 12, or 14 finite values")

    mounting = _mapping(camera_data.get("robot_to_camera", {}), "camera.robot_to_camera")
    _reject_unknown(mounting, {"translation_m", "rotation_rpy_rad"}, "camera.robot_to_camera")
    robot_to_camera = transform_matrix(
        _vector(mounting.get("translation_m", [0.0, 0.0, 0.0]), 3, "camera.robot_to_camera.translation_m"),
        _vector(mounting.get("rotation_rpy_rad", [0.0, 0.0, 0.0]), 3, "camera.robot_to_camera.rotation_rpy_rad"),
    )

    layout_data = _mapping(data.get("tag_layout", {}), "tag_layout")
    _reject_unknown(layout_data, {"tags"}, "tag_layout")
    tags = layout_data.get("tags", [])
    if not isinstance(tags, list):
        raise ProfileError("tag_layout.tags must be an array")
    layout: dict[int, np.ndarray] = {}
    for index, tag_value in enumerate(tags):
        context = f"tag_layout.tags[{index}]"
        tag = _mapping(tag_value, context)
        _reject_unknown(tag, {"id", "pose"}, context)
        tag_id = _integer(tag.get("id"), f"{context}.id", minimum=0)
        if tag_id in layout:
            raise ProfileError(f"Duplicate tag ID {tag_id} in tag_layout")
        pose = _mapping(tag.get("pose", {}), f"{context}.pose")
        _reject_unknown(pose, {"translation_m", "rotation_rpy_rad"}, f"{context}.pose")
        layout[tag_id] = transform_matrix(
            _vector(pose.get("translation_m", [0.0, 0.0, 0.0]), 3, f"{context}.pose.translation_m"),
            _vector(pose.get("rotation_rpy_rad", [0.0, 0.0, 0.0]), 3, f"{context}.pose.rotation_rpy_rad"),
        )

    detection = _mapping(data.get("detection", {}), "detection")
    _reject_unknown(detection, {"skip_scan", "max_reprojection_error_px"}, "detection")
    skip_data = _mapping(detection.get("skip_scan", {}), "detection.skip_scan")
    _reject_unknown(
        skip_data,
        {
            "enabled",
            "stride",
            "min_global_std",
            "min_edge_strength",
            "dilation_px",
            "padding_px",
            "min_roi_size_px",
            "max_rois",
            "max_roi_area_ratio",
        },
        "detection.skip_scan",
    )
    skip_scan = SkipScanConfig(
        enabled=_boolean(skip_data.get("enabled", False), "detection.skip_scan.enabled"),
        stride=_integer(skip_data.get("stride", 4), "detection.skip_scan.stride", minimum=1),
        min_global_std=_positive_number(skip_data.get("min_global_std", 3.0), "detection.skip_scan.min_global_std"),
        min_edge_strength=_positive_number(
            skip_data.get("min_edge_strength", 18.0), "detection.skip_scan.min_edge_strength"
        ),
        dilation_px=_integer(skip_data.get("dilation_px", 56), "detection.skip_scan.dilation_px", minimum=1),
        padding_px=_integer(skip_data.get("padding_px", 48), "detection.skip_scan.padding_px", minimum=0),
        min_roi_size_px=_integer(
            skip_data.get("min_roi_size_px", 48), "detection.skip_scan.min_roi_size_px", minimum=1
        ),
        max_rois=_integer(skip_data.get("max_rois", 8), "detection.skip_scan.max_rois", minimum=1),
        max_roi_area_ratio=_positive_number(
            skip_data.get("max_roi_area_ratio", 0.45), "detection.skip_scan.max_roi_area_ratio"
        ),
    )
    if skip_scan.max_roi_area_ratio > 1.0:
        raise ProfileError("detection.skip_scan.max_roi_area_ratio must be at most 1.0")

    output = _mapping(data.get("output", {}), "output")
    _reject_unknown(output, {"mode", "path", "nt_server"}, "output")
    output_mode = _string(output.get("mode", "jsonl"), "output.mode")
    if output_mode not in {"jsonl", "none", "nt"}:
        raise ProfileError("output.mode must be one of: jsonl, none, nt")
    output_path = _string(output.get("path", "northstar_output.jsonl"), "output.path")
    nt_server = _string(output.get("nt_server", "10.0.0.2"), "output.nt_server")

    camera = CameraConfig(
        camera_id=_camera_id(camera_data.get("id", 0)),
        width=width,
        height=height,
        fps=fps,
        buffer_size=buffer_size,
        auto_exposure=_optional_number(camera_data.get("auto_exposure"), "camera.auto_exposure"),
        exposure=_optional_number(camera_data.get("exposure"), "camera.exposure"),
        gain=_optional_number(camera_data.get("gain"), "camera.gain"),
        calibration_matrix=camera_matrix,
        distortion_coefficients=distortion,
        robot_to_camera=robot_to_camera,
    )
    name = _string(data.get("name", getattr(source, "stem", "profile")), "name")
    device_id = _string(data.get("device_id", "northstar_0"), "device_id")
    platform = _string(data.get("platform", name), "platform")
    replay_path_value = data.get("replay_path")
    if replay_path_value is not None:
        replay_path_value = _string(replay_path_value, "replay_path")

    return PlatformProfile(
        name=name,
        device_id=device_id,
        platform=platform,
        replay_path=replay_path_value,
        camera=camera,
        tag_size_m=_positive_number(data.get("tag_size_m", 0.1651), "tag_size_m"),
        tag_layout=layout,
        skip_scan=skip_scan,
        max_reprojection_error_px=_positive_number(
            detection.get("max_reprojection_error_px", 5.0), "detection.max_reprojection_error_px"
        ),
        output_mode=output_mode,
        output_path=output_path,
        nt_server=nt_server,
    )
