from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import CalibrationError, load_calibration_data
from .field_layout import FieldLayoutError, load_wpilib_apriltag_layout
from .geometry import transform_matrix
from .prefilter import SkipScanConfig
from .provenance import ArtifactProvenance, ProvenanceError, ProvenanceKind
from .recording import JsonlRecorderConfig
from .types import CameraPoseBounds


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
    calibration_provenance: ArtifactProvenance
    calibration_rms_reprojection_error_px: float | None
    calibration_views_used: int | None
    reconnect_initial_s: float
    reconnect_max_s: float
    read_timeout_s: float | None
    shutdown_timeout_s: float


@dataclass
class PlatformProfile:
    name: str
    device_id: str
    platform: str
    replay_path: str | None
    camera: CameraConfig
    tag_size_m: float
    tag_layout: dict[int, np.ndarray]
    tag_layout_provenance: ArtifactProvenance
    field_length_m: float | None
    field_width_m: float | None
    production: bool
    skip_scan: SkipScanConfig
    max_reprojection_error_px: float
    max_tag_reprojection_error_px: float
    max_outlier_iterations: int
    ransac_iterations: int
    camera_pose_bounds: CameraPoseBounds | None
    output_mode: str
    output_path: str
    nt_server: str
    nt_server_port: int
    nt_send_all: bool
    nt_close_drain_s: float
    jsonl_recorder: JsonlRecorderConfig
    robot_yaw_topic: str | None
    yaw_max_age_s: float
    yaw_queue_size: int


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


def _optional_positive_number(value: Any, context: str) -> float | None:
    return None if value is None else _positive_number(value, context)


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


def _provenance(value: Any, context: str, source: Any) -> ArtifactProvenance:
    if value is None:
        return ArtifactProvenance(ProvenanceKind.EXAMPLE, f"inline values in {source}")
    try:
        return ArtifactProvenance.from_dict(value, default_source=str(source))
    except ProvenanceError as exc:
        raise ProfileError(f"Invalid {context}: {exc}") from exc


def _artifact_path(source: Any, value: Any, context: str) -> Path:
    raw = Path(_string(value, context)).expanduser()
    if raw.is_absolute():
        return raw
    if not isinstance(source, Path):
        raise ProfileError(f"{context} must be absolute when used by a bundled profile")
    return source.parent / raw


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
            "production",
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
            "runtime",
        },
        "camera",
    )
    width = _integer(camera_data.get("width"), "camera.width", minimum=1)
    height = _integer(camera_data.get("height"), "camera.height", minimum=1)
    fps_value = camera_data.get("fps")
    fps = None if fps_value is None else _positive_number(fps_value, "camera.fps")
    buffer_value = camera_data.get("buffer_size")
    buffer_size = None if buffer_value is None else _integer(buffer_value, "camera.buffer_size", minimum=1)

    runtime = _mapping(camera_data.get("runtime", {}), "camera.runtime")
    _reject_unknown(
        runtime,
        {"reconnect_initial_s", "reconnect_max_s", "read_timeout_s", "shutdown_timeout_s"},
        "camera.runtime",
    )
    reconnect_initial_s = _positive_number(
        runtime.get("reconnect_initial_s", 0.1), "camera.runtime.reconnect_initial_s"
    )
    reconnect_max_s = _positive_number(runtime.get("reconnect_max_s", 2.0), "camera.runtime.reconnect_max_s")
    if reconnect_max_s < reconnect_initial_s:
        raise ProfileError("camera.runtime.reconnect_max_s must be at least reconnect_initial_s")
    read_timeout_s = _optional_positive_number(runtime.get("read_timeout_s"), "camera.runtime.read_timeout_s")
    shutdown_timeout_s = _number(runtime.get("shutdown_timeout_s", 2.0), "camera.runtime.shutdown_timeout_s")
    if shutdown_timeout_s < 0:
        raise ProfileError("camera.runtime.shutdown_timeout_s must not be negative")

    calibration = _mapping(camera_data.get("calibration", {}), "camera.calibration")
    _reject_unknown(
        calibration,
        {
            "file",
            "camera_matrix",
            "distortion_coefficients",
            "provenance",
            "rms_reprojection_error_px",
            "views_used",
        },
        "camera.calibration",
    )
    calibration_file = calibration.get("file")
    if calibration_file is not None:
        conflicting = set(calibration) & {
            "camera_matrix",
            "distortion_coefficients",
            "provenance",
            "rms_reprojection_error_px",
            "views_used",
        }
        if conflicting:
            raise ProfileError("camera.calibration.file cannot be combined with inline calibration fields")
        path = _artifact_path(source, calibration_file, "camera.calibration.file")
        try:
            calibration_artifact = load_calibration_data(path)
            calibration_artifact.validate_for_image_size(width, height)
        except (CalibrationError, OSError) as exc:
            raise ProfileError(f"Invalid camera calibration artifact: {exc}") from exc
        camera_matrix = calibration_artifact.camera_matrix.copy()
        distortion = calibration_artifact.distortion_coefficients.copy()
        calibration_provenance = calibration_artifact.provenance
        calibration_rms = calibration_artifact.rms_reprojection_error_px
        calibration_views = calibration_artifact.views_used
    else:
        calibration_provenance = _provenance(calibration.get("provenance"), "camera.calibration.provenance", source)
        calibration_rms = _optional_number(
            calibration.get("rms_reprojection_error_px"), "camera.calibration.rms_reprojection_error_px"
        )
        if calibration_rms is not None and calibration_rms < 0:
            raise ProfileError("camera.calibration.rms_reprojection_error_px must not be negative")
        calibration_views = (
            None
            if calibration.get("views_used") is None
            else _integer(calibration.get("views_used"), "camera.calibration.views_used", minimum=1)
        )
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
            distortion = np.asarray(
                calibration.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64
            ).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ProfileError(
                "camera.calibration.distortion_coefficients must contain 4, 5, 8, 12, or 14 finite values"
            ) from exc
        if len(distortion) not in {4, 5, 8, 12, 14} or not np.all(np.isfinite(distortion)):
            raise ProfileError(
                "camera.calibration.distortion_coefficients must contain 4, 5, 8, 12, or 14 finite values"
            )

    mounting = _mapping(camera_data.get("robot_to_camera", {}), "camera.robot_to_camera")
    _reject_unknown(mounting, {"translation_m", "rotation_rpy_rad"}, "camera.robot_to_camera")
    robot_to_camera = transform_matrix(
        _vector(mounting.get("translation_m", [0.0, 0.0, 0.0]), 3, "camera.robot_to_camera.translation_m"),
        _vector(mounting.get("rotation_rpy_rad", [0.0, 0.0, 0.0]), 3, "camera.robot_to_camera.rotation_rpy_rad"),
    )

    layout_data = _mapping(data.get("tag_layout", {}), "tag_layout")
    _reject_unknown(layout_data, {"wpilib_json", "tags", "provenance", "field"}, "tag_layout")
    wpilib_json = layout_data.get("wpilib_json")
    if wpilib_json is not None:
        if "tags" in layout_data or "field" in layout_data:
            raise ProfileError("tag_layout.wpilib_json cannot be combined with inline tags or field dimensions")
        explicit_provenance = (
            None
            if layout_data.get("provenance") is None
            else _provenance(layout_data.get("provenance"), "tag_layout.provenance", source)
        )
        path = _artifact_path(source, wpilib_json, "tag_layout.wpilib_json")
        try:
            layout_artifact = load_wpilib_apriltag_layout(path, provenance=explicit_provenance)
        except (FieldLayoutError, OSError) as exc:
            raise ProfileError(f"Invalid WPILib field-layout artifact: {exc}") from exc
        layout = layout_artifact.as_transform_dict()
        layout_provenance = layout_artifact.provenance
        field_length_m = layout_artifact.field_length_m
        field_width_m = layout_artifact.field_width_m
    else:
        layout_provenance = _provenance(layout_data.get("provenance"), "tag_layout.provenance", source)
        field_data = _mapping(layout_data.get("field", {}), "tag_layout.field")
        _reject_unknown(field_data, {"length_m", "width_m"}, "tag_layout.field")
        field_length_m = _optional_positive_number(field_data.get("length_m"), "tag_layout.field.length_m")
        field_width_m = _optional_positive_number(field_data.get("width_m"), "tag_layout.field.width_m")
        if (field_length_m is None) != (field_width_m is None):
            raise ProfileError("tag_layout.field must provide both length_m and width_m")
        tags = layout_data.get("tags", [])
        if not isinstance(tags, list):
            raise ProfileError("tag_layout.tags must be an array")
        layout = {}
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
    _reject_unknown(
        detection,
        {
            "skip_scan",
            "max_reprojection_error_px",
            "max_tag_reprojection_error_px",
            "max_outlier_iterations",
            "ransac_iterations",
            "pose_bounds",
        },
        "detection",
    )
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

    max_reprojection_error_px = _positive_number(
        detection.get("max_reprojection_error_px", 5.0), "detection.max_reprojection_error_px"
    )
    max_tag_reprojection_error_px = _positive_number(
        detection.get("max_tag_reprojection_error_px", max_reprojection_error_px),
        "detection.max_tag_reprojection_error_px",
    )
    max_outlier_iterations = _integer(
        detection.get("max_outlier_iterations", 8), "detection.max_outlier_iterations", minimum=0
    )
    ransac_iterations = _integer(detection.get("ransac_iterations", 100), "detection.ransac_iterations", minimum=1)
    bounds_data = _mapping(detection.get("pose_bounds", {}), "detection.pose_bounds")
    _reject_unknown(
        bounds_data,
        {"field_margin_m", "min_camera_height_m", "max_camera_height_m"},
        "detection.pose_bounds",
    )
    camera_pose_bounds = None
    if bounds_data:
        try:
            camera_pose_bounds = CameraPoseBounds(
                field_length_m=field_length_m,
                field_width_m=field_width_m,
                field_margin_m=_number(bounds_data.get("field_margin_m", 0.0), "detection.pose_bounds.field_margin_m"),
                min_camera_height_m=_optional_number(
                    bounds_data.get("min_camera_height_m"), "detection.pose_bounds.min_camera_height_m"
                ),
                max_camera_height_m=_optional_number(
                    bounds_data.get("max_camera_height_m"), "detection.pose_bounds.max_camera_height_m"
                ),
            )
        except ValueError as exc:
            raise ProfileError(f"Invalid detection.pose_bounds: {exc}") from exc

    output = _mapping(data.get("output", {}), "output")
    _reject_unknown(
        output,
        {
            "mode",
            "path",
            "nt_server",
            "nt_server_port",
            "nt_delivery",
            "nt_close_drain_s",
            "jsonl",
            "robot_yaw",
        },
        "output",
    )
    output_mode = _string(output.get("mode", "jsonl"), "output.mode")
    if output_mode not in {"jsonl", "none", "nt"}:
        raise ProfileError("output.mode must be one of: jsonl, none, nt")
    output_path = _string(output.get("path", "northstar_output.jsonl"), "output.path")
    nt_server = _string(output.get("nt_server", "10.0.0.2"), "output.nt_server")
    nt_server_port = _integer(output.get("nt_server_port", 0), "output.nt_server_port", minimum=0)
    if nt_server_port > 65535:
        raise ProfileError("output.nt_server_port must be at most 65535")
    nt_delivery = _string(output.get("nt_delivery", "upstream"), "output.nt_delivery")
    if nt_delivery not in {"upstream", "latest"}:
        raise ProfileError("output.nt_delivery must be one of: upstream, latest")
    nt_close_drain_s = _number(output.get("nt_close_drain_s", 0.1), "output.nt_close_drain_s")
    if nt_close_drain_s < 0:
        raise ProfileError("output.nt_close_drain_s must not be negative")

    jsonl_data = _mapping(output.get("jsonl", {}), "output.jsonl")
    _reject_unknown(
        jsonl_data,
        {
            "queue_size",
            "drop_policy",
            "flush_interval_s",
            "fsync_interval_s",
            "max_file_size_bytes",
            "rotation_interval_s",
            "retention_files",
            "close_timeout_s",
        },
        "output.jsonl",
    )
    drop_policy = _string(jsonl_data.get("drop_policy", "drop_oldest"), "output.jsonl.drop_policy")
    if drop_policy not in {"drop_oldest", "drop_newest"}:
        raise ProfileError("output.jsonl.drop_policy must be one of: drop_oldest, drop_newest")
    max_file_size_value = jsonl_data.get("max_file_size_bytes", 64 * 1024 * 1024)
    max_file_size_bytes = (
        None
        if max_file_size_value is None
        else _integer(max_file_size_value, "output.jsonl.max_file_size_bytes", minimum=1)
    )
    rotation_interval_s = _optional_positive_number(
        jsonl_data.get("rotation_interval_s", 3600.0), "output.jsonl.rotation_interval_s"
    )
    fsync_interval_s = _optional_positive_number(
        jsonl_data.get("fsync_interval_s", 5.0), "output.jsonl.fsync_interval_s"
    )
    try:
        jsonl_recorder = JsonlRecorderConfig(
            queue_size=_integer(jsonl_data.get("queue_size", 256), "output.jsonl.queue_size", minimum=1),
            drop_policy=drop_policy,
            flush_interval_s=_number(jsonl_data.get("flush_interval_s", 1.0), "output.jsonl.flush_interval_s"),
            fsync_interval_s=fsync_interval_s,
            max_file_size_bytes=max_file_size_bytes,
            rotation_interval_s=rotation_interval_s,
            retention_files=_integer(jsonl_data.get("retention_files", 8), "output.jsonl.retention_files", minimum=0),
            close_timeout_s=_positive_number(jsonl_data.get("close_timeout_s", 5.0), "output.jsonl.close_timeout_s"),
        )
    except ValueError as exc:
        raise ProfileError(f"Invalid output.jsonl: {exc}") from exc

    yaw_data = _mapping(output.get("robot_yaw", {}), "output.robot_yaw")
    _reject_unknown(yaw_data, {"topic", "max_age_s", "queue_size"}, "output.robot_yaw")
    yaw_topic_value = yaw_data.get("topic")
    robot_yaw_topic = None if yaw_topic_value is None else _string(yaw_topic_value, "output.robot_yaw.topic")
    if robot_yaw_topic is not None and not robot_yaw_topic.startswith("/"):
        raise ProfileError("output.robot_yaw.topic must be an absolute NetworkTables path")
    yaw_max_age_s = _positive_number(yaw_data.get("max_age_s", 0.1), "output.robot_yaw.max_age_s")
    yaw_queue_size = _integer(yaw_data.get("queue_size", 64), "output.robot_yaw.queue_size", minimum=2)

    production = _boolean(data.get("production", False), "production")
    if production:
        if calibration_provenance.kind is not ProvenanceKind.MEASURED:
            raise ProfileError("production profiles require measured camera calibration provenance")
        if calibration_rms is None or calibration_views is None:
            raise ProfileError("production profiles require calibration reprojection error and view count")
        if layout_provenance.kind is not ProvenanceKind.OFFICIAL:
            raise ProfileError("production profiles require official field-layout provenance")
        if field_length_m is None or field_width_m is None:
            raise ProfileError("production profiles require field layout dimensions")

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
        calibration_provenance=calibration_provenance,
        calibration_rms_reprojection_error_px=calibration_rms,
        calibration_views_used=calibration_views,
        reconnect_initial_s=reconnect_initial_s,
        reconnect_max_s=reconnect_max_s,
        read_timeout_s=read_timeout_s,
        shutdown_timeout_s=shutdown_timeout_s,
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
        tag_layout_provenance=layout_provenance,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        production=production,
        skip_scan=skip_scan,
        max_reprojection_error_px=max_reprojection_error_px,
        max_tag_reprojection_error_px=max_tag_reprojection_error_px,
        max_outlier_iterations=max_outlier_iterations,
        ransac_iterations=ransac_iterations,
        camera_pose_bounds=camera_pose_bounds,
        output_mode=output_mode,
        output_path=output_path,
        nt_server=nt_server,
        nt_server_port=nt_server_port,
        nt_send_all=nt_delivery == "upstream",
        nt_close_drain_s=nt_close_drain_s,
        jsonl_recorder=jsonl_recorder,
        robot_yaw_topic=robot_yaw_topic,
        yaw_max_age_s=yaw_max_age_s,
        yaw_queue_size=yaw_queue_size,
    )
