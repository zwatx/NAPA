from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any

import numpy as np

from .geometry import rotation_matrix_to_quaternion_wxyz
from .types import CameraPoseObservation, FrameResult, PoseSolveDiagnostics, SingleTagPoseObservation

if TYPE_CHECKING:
    from .recording import JsonlRecorderConfig

JSON_SCHEMA_VERSION = 1


def _encode_pose(error: float, pose: np.ndarray) -> list[float]:
    quaternion = rotation_matrix_to_quaternion_wxyz(pose)
    return [
        float(error),
        float(pose[0, 3]),
        float(pose[1, 3]),
        float(pose[2, 3]),
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    ]


def encode_observations(result: FrameResult) -> list[float]:
    """Encode the upstream Northstar ``observations`` topic contract.

    The first value is the camera-pose count: 0 for none, 1 for a multi-tag
    pose, and 2 for the ambiguous pair produced by a single tag. Each pose is
    eight values (error, XYZ, quaternion WXYZ), followed by fixed ten-value tag
    records (ID, eight corner angles, distance).
    """

    data: list[float] = [0.0]
    camera_pose = result.camera_pose
    if camera_pose is not None:
        data[0] = 1.0
        data.extend(_encode_pose(camera_pose.error_0, camera_pose.field_to_camera_0))
        if camera_pose.field_to_camera_1 is not None and camera_pose.error_1 is not None:
            data[0] = 2.0
            data.extend(_encode_pose(camera_pose.error_1, camera_pose.field_to_camera_1))
    for observation in result.tag_angles:
        data.append(float(observation.tag_id))
        data.extend(np.asarray(observation.corner_angles, dtype=np.float64).reshape(8).tolist())
        data.append(float(observation.distance))
    return data


def _transform_to_dict(transform: np.ndarray) -> dict[str, list[float]]:
    return {
        "translation_m": np.asarray(transform[:3, 3], dtype=np.float64).tolist(),
        "quaternion_wxyz": rotation_matrix_to_quaternion_wxyz(transform).tolist(),
    }


def _single_tag_pose_to_dict(observation: SingleTagPoseObservation) -> dict[str, Any]:
    candidates = [
        {
            "opencv_rvec": np.asarray(observation.pose_0_rvec).reshape(3).tolist(),
            "opencv_tvec_m": np.asarray(observation.pose_0_tvec).reshape(3).tolist(),
            "reprojection_error_px": observation.error_0,
        }
    ]
    if observation.pose_1_rvec is not None and observation.pose_1_tvec is not None:
        candidates.append(
            {
                "opencv_rvec": np.asarray(observation.pose_1_rvec).reshape(3).tolist(),
                "opencv_tvec_m": np.asarray(observation.pose_1_tvec).reshape(3).tolist(),
                "reprojection_error_px": observation.error_1,
            }
        )
    return {"tag_id": observation.tag_id, "distance_m": observation.distance, "candidates": candidates}


def _camera_pose_to_dict(observation: CameraPoseObservation | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    candidates = [
        {
            **_transform_to_dict(observation.field_to_camera_0),
            "reprojection_error_px": observation.error_0,
        }
    ]
    if observation.field_to_camera_1 is not None:
        candidates.append(
            {
                **_transform_to_dict(observation.field_to_camera_1),
                "reprojection_error_px": observation.error_1,
            }
        )
    return {
        "tag_ids": observation.tag_ids,
        "average_tag_distance_m": observation.average_tag_distance,
        "confidence": observation.confidence,
        "candidates": candidates,
    }


def _pose_diagnostics_to_dict(observation: PoseSolveDiagnostics | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {
        "attempted_tag_ids": observation.attempted_tag_ids,
        "accepted_tag_ids": observation.accepted_tag_ids,
        "rejected_tags": [
            {"tag_id": tag_id, "reason": reason} for tag_id, reason in sorted(observation.rejected_tag_reasons.items())
        ],
        "per_tag_rms_reprojection_error_px": [
            {"tag_id": tag_id, "rms_reprojection_error_px": error}
            for tag_id, error in sorted(observation.per_tag_rms_reprojection_error_px.items())
        ],
        "overall_rms_reprojection_error_px": observation.overall_rms_reprojection_error_px,
        "iterations": observation.iterations,
        "used_ransac": observation.used_ransac,
        "rejection_reason": observation.rejection_reason,
    }


def frame_result_to_dict(result: FrameResult) -> dict[str, Any]:
    """Return a complete, self-describing JSON representation of a frame."""

    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "timestamp_s": result.timestamp,
        "capture_monotonic_s": result.capture_monotonic_s,
        "image_observations": [
            {"tag_id": observation.tag_id, "corners_px": observation.corners.tolist()}
            for observation in result.image_observations
        ],
        "single_tag_poses": [_single_tag_pose_to_dict(observation) for observation in result.single_tag_poses],
        "camera_pose": _camera_pose_to_dict(result.camera_pose),
        "pose_diagnostics": _pose_diagnostics_to_dict(result.pose_diagnostics),
        "tag_angles": [
            {
                "tag_id": observation.tag_id,
                "corner_angles_rad": observation.corner_angles.tolist(),
                "center_tx_rad": observation.center_tx,
                "center_ty_rad": observation.center_ty,
                "distance_m": observation.distance,
                "camera_to_tag_translation_m": observation.camera_to_tag_translation.tolist(),
            }
            for observation in result.tag_angles
        ],
        "trig_poses": [
            {
                "tag_id": observation.tag_id,
                "robot_pose_xy_yaw": observation.robot_pose.tolist(),
                "camera_to_tag_translation_m": observation.camera_to_tag_translation.tolist(),
                "distance_m": observation.distance,
            }
            for observation in result.trig_poses
        ],
        "health": result.health,
    }


class OutputPublisher(ABC):
    @abstractmethod
    def publish(self, result: FrameResult) -> None:
        """Publish a completed frame."""

    @abstractmethod
    def close(self) -> None:
        """Release publisher resources."""

    def telemetry(self) -> dict[str, Any]:
        """Return JSON-compatible output health data."""

        return {}


class NullOutputPublisher(OutputPublisher):
    def publish(self, result: FrameResult) -> None:
        del result

    def close(self) -> None:
        return None


class NTOutputPublisher(OutputPublisher):
    def __init__(
        self,
        device_id: str,
        server: str,
        *,
        server_port: int = 0,
        send_all: bool = True,
        close_drain_s: float = 0.1,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if server_port < 0 or server_port > 65535:
            raise ValueError("server_port must be between 0 and 65535")
        if not math.isfinite(close_drain_s) or close_drain_s < 0:
            raise ValueError("close_drain_s must be finite and non-negative")
        try:
            import ntcore
        except ImportError as exc:
            raise RuntimeError("Install northstar-vision[nt] to use NetworkTables output") from exc
        self.ntcore = ntcore
        self.clock = clock
        self.send_all = send_all
        self.delivery_policy = "upstream_lossless" if send_all else "latest_value"
        self.close_drain_s = close_drain_s
        self.instance = ntcore.NetworkTableInstance.create()
        self.publishers: list[Any] = []
        self.closed = True
        try:
            if server_port:
                self.instance.setServer(server, server_port)
            else:
                self.instance.setServer(server)
            self.instance.startClient4(device_id)
            table = self.instance.getTable(f"/{device_id}/output")
            self.observations = table.getDoubleArrayTopic("observations").publish(
                ntcore.PubSubOptions(periodic=0.01, sendAll=send_all, keepDuplicates=True)
            )
            self.publishers.append(self.observations)
            self.fps = table.getIntegerTopic("fps_apriltags").publish()
            self.publishers.append(self.fps)
            self.latency = table.getDoubleTopic("latency_ms").publish()
            self.publishers.append(self.latency)
            self.health = table.getStringTopic("health").publish()
            self.publishers.append(self.health)
        except BaseException:
            self._close_resources(raise_errors=False, drain=False)
            raise
        self.closed = False

    def publish(self, result: FrameResult) -> None:
        if self.closed:
            raise RuntimeError("Cannot publish after the NetworkTables publisher is closed")
        observation_data = encode_observations(result)
        if result.capture_monotonic_s is None:
            self.observations.set(observation_data)
        else:
            frame_age_us = max(0, math.floor((self.clock() - result.capture_monotonic_s) * 1_000_000))
            timestamp_us = max(1, int(self.ntcore._now()) - frame_age_us)
            self.observations.set(observation_data, timestamp_us)
        self.fps.set(int(result.health.get("fps", 0)))
        self.latency.set(float(result.health.get("latency_ms", 0.0)))
        health = {**result.health, "nt_connected": self.instance.isConnected()}
        self.health.set(json.dumps(health, separators=(",", ":")))

    def telemetry(self) -> dict[str, Any]:
        return {
            "type": "networktables",
            "connected": bool(self.instance.isConnected()),
            "delivery_policy": self.delivery_policy,
            "close_drain_s": self.close_drain_s,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._close_resources(raise_errors=True, drain=True)

    def _close_resources(self, *, raise_errors: bool, drain: bool) -> None:
        errors: list[Exception] = []
        if self.publishers:
            try:
                self.instance.flush()
            except Exception as exc:
                errors.append(exc)
            if drain and self.close_drain_s > 0:
                sleep(self.close_drain_s)
        for publisher in reversed(self.publishers):
            try:
                publisher.close()
            except Exception as exc:  # Continue so the NT client is always destroyed.
                errors.append(exc)
        self.publishers.clear()
        try:
            self.instance.stopClient()
        except Exception as exc:
            errors.append(exc)
        try:
            self.ntcore.NetworkTableInstance.destroy(self.instance)
        except Exception as exc:
            errors.append(exc)
        if errors and raise_errors:
            raise errors[0]


def make_output_publisher(
    mode: str,
    device_id: str,
    nt_server: str,
    *,
    output_path: str | Path = "northstar_output.jsonl",
    nt_server_port: int = 0,
    nt_send_all: bool = True,
    nt_close_drain_s: float = 0.1,
    jsonl_config: JsonlRecorderConfig | None = None,
) -> OutputPublisher:
    if mode == "nt":
        return NTOutputPublisher(
            device_id,
            nt_server,
            server_port=nt_server_port,
            send_all=nt_send_all,
            close_drain_s=nt_close_drain_s,
        )
    if mode == "none":
        return NullOutputPublisher()
    if mode == "jsonl":
        from .recording import AsyncJsonlOutputPublisher

        return AsyncJsonlOutputPublisher(output_path, config=jsonl_config)
    raise ValueError(f"Unknown output mode {mode!r}; expected jsonl, nt, or none")
