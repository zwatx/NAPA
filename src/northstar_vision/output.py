from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .geometry import rotation_matrix_to_quaternion_wxyz
from .types import CameraPoseObservation, FrameResult, SingleTagPoseObservation

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


class NullOutputPublisher(OutputPublisher):
    def publish(self, result: FrameResult) -> None:
        del result

    def close(self) -> None:
        return None


class JsonlOutputPublisher(OutputPublisher):
    def __init__(
        self,
        path: str | Path = "northstar_output.jsonl",
        *,
        flush_interval_s: float = 1.0,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if flush_interval_s < 0:
            raise ValueError("flush_interval_s must not be negative")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", encoding="utf-8")
        self.flush_interval_s = flush_interval_s
        self.clock = clock
        self.next_flush = clock() + flush_interval_s
        self.closed = False

    def publish(self, result: FrameResult) -> None:
        if self.closed:
            raise RuntimeError("Cannot publish after the JSONL publisher is closed")
        self.file.write(json.dumps(frame_result_to_dict(result), separators=(",", ":")) + "\n")
        now = self.clock()
        if self.flush_interval_s == 0 or now >= self.next_flush:
            self.file.flush()
            self.next_flush = now + self.flush_interval_s

    def close(self) -> None:
        if not self.closed:
            self.file.flush()
            self.file.close()
            self.closed = True


class NTOutputPublisher(OutputPublisher):
    def __init__(self, device_id: str, server: str, *, clock: Callable[[], float] = perf_counter) -> None:
        try:
            import ntcore
        except ImportError as exc:
            raise RuntimeError("Install northstar-vision[nt] to use NetworkTables output") from exc
        self.ntcore = ntcore
        self.clock = clock
        self.instance = ntcore.NetworkTableInstance.create()
        self.publishers: list[Any] = []
        self.closed = True
        try:
            self.instance.setServer(server)
            self.instance.startClient4(device_id)
            table = self.instance.getTable(f"/{device_id}/output")
            self.observations = table.getDoubleArrayTopic("observations").publish(
                ntcore.PubSubOptions(periodic=0.01, sendAll=True, keepDuplicates=True)
            )
            self.publishers.append(self.observations)
            self.fps = table.getIntegerTopic("fps_apriltags").publish()
            self.publishers.append(self.fps)
            self.latency = table.getDoubleTopic("latency_ms").publish()
            self.publishers.append(self.latency)
            self.health = table.getStringTopic("health").publish()
            self.publishers.append(self.health)
        except BaseException:
            self._close_resources(raise_errors=False)
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

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._close_resources(raise_errors=True)

    def _close_resources(self, *, raise_errors: bool) -> None:
        errors: list[Exception] = []
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
) -> OutputPublisher:
    if mode == "nt":
        return NTOutputPublisher(device_id, nt_server)
    if mode == "none":
        return NullOutputPublisher()
    if mode == "jsonl":
        return JsonlOutputPublisher(output_path)
    raise ValueError(f"Unknown output mode {mode!r}; expected jsonl, nt, or none")
