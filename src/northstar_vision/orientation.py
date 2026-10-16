from __future__ import annotations

import math
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any


def normalize_angle_radians(value: float) -> float:
    """Normalize an angle to the half-open interval [-pi, pi)."""

    normalized = math.remainder(float(value), 2.0 * math.pi)
    return -math.pi if normalized == math.pi else normalized


@dataclass(frozen=True, slots=True)
class YawSample:
    monotonic_s: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.monotonic_s) or not math.isfinite(self.yaw_rad):
            raise ValueError("Yaw samples must contain finite timestamps and angles")


@dataclass(frozen=True, slots=True)
class YawEstimate:
    yaw_rad: float
    sample_age_s: float
    interpolated: bool


class YawBuffer:
    """Small timestamp-ordered yaw buffer with wrap-safe interpolation."""

    def __init__(self, max_samples: int = 64) -> None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        self.max_samples = max_samples
        self._samples: list[YawSample] = []

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def latest(self) -> YawSample | None:
        return self._samples[-1] if self._samples else None

    def add(self, monotonic_s: float, yaw_rad: float) -> None:
        sample = YawSample(float(monotonic_s), normalize_angle_radians(yaw_rad))
        if not self._samples or sample.monotonic_s > self._samples[-1].monotonic_s:
            self._samples.append(sample)
            if len(self._samples) > self.max_samples:
                del self._samples[0]
            return
        index = bisect_left(self._samples, sample.monotonic_s, key=lambda value: value.monotonic_s)
        if index < len(self._samples) and self._samples[index].monotonic_s == sample.monotonic_s:
            self._samples[index] = sample
        else:
            self._samples.insert(index, sample)
        overflow = len(self._samples) - self.max_samples
        if overflow > 0:
            del self._samples[:overflow]

    def extend(self, samples: Iterable[YawSample]) -> None:
        for sample in samples:
            self.add(sample.monotonic_s, sample.yaw_rad)

    def estimate(self, monotonic_s: float, *, max_age_s: float) -> YawEstimate | None:
        if not math.isfinite(max_age_s) or max_age_s < 0:
            raise ValueError("max_age_s must not be negative")
        if not self._samples or not math.isfinite(monotonic_s):
            return None
        index = bisect_left(self._samples, monotonic_s, key=lambda value: value.monotonic_s)
        if index == 0:
            sample = self._samples[0]
            age = abs(sample.monotonic_s - monotonic_s)
            return None if age > max_age_s else YawEstimate(sample.yaw_rad, age, False)
        if index == len(self._samples):
            sample = self._samples[-1]
            age = monotonic_s - sample.monotonic_s
            return None if age > max_age_s else YawEstimate(sample.yaw_rad, age, False)

        before = self._samples[index - 1]
        after = self._samples[index]
        nearest_age = min(monotonic_s - before.monotonic_s, after.monotonic_s - monotonic_s)
        if nearest_age > max_age_s:
            return None
        span = after.monotonic_s - before.monotonic_s
        fraction = 0.0 if span <= 0.0 else (monotonic_s - before.monotonic_s) / span
        delta = math.remainder(after.yaw_rad - before.yaw_rad, 2.0 * math.pi)
        yaw = normalize_angle_radians(before.yaw_rad + fraction * delta)
        return YawEstimate(yaw, nearest_age, True)


class RobotYawSource(ABC):
    @abstractmethod
    def yaw_at(self, capture_monotonic_s: float) -> YawEstimate | None:
        """Return yaw aligned to a frame capture time, or None when stale."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return source telemetry suitable for FrameResult.health."""

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""


class StaticRobotYawSource(RobotYawSource):
    def __init__(self, yaw_rad: float) -> None:
        if not math.isfinite(yaw_rad):
            raise ValueError("Static robot yaw must be finite")
        self.yaw_rad = normalize_angle_radians(yaw_rad)

    def yaw_at(self, capture_monotonic_s: float) -> YawEstimate | None:
        if not math.isfinite(capture_monotonic_s):
            return None
        return YawEstimate(self.yaw_rad, 0.0, False)

    def health(self) -> dict[str, Any]:
        return {"yaw_source": "static", "yaw_samples": 1, "yaw_connected": True}

    def close(self) -> None:
        return None


class NetworkTablesRobotYawSource(RobotYawSource):
    """Read timestamped yaw samples and align them to frame receive time.

    NetworkTables timestamps and ``ntcore._now()`` share a monotonic microsecond
    time base. Each poll translates samples by their age relative to ``_now`` so
    the buffer remains in Python's ``perf_counter`` time base used by frames.
    """

    def __init__(
        self,
        device_id: str,
        server: str,
        *,
        topic: str = "/RobotState/yaw",
        max_age_s: float = 0.1,
        queue_size: int = 64,
        server_port: int = 0,
        clock: Callable[[], float] = perf_counter,
        ntcore_module: Any | None = None,
        instance: Any | None = None,
    ) -> None:
        if not math.isfinite(max_age_s) or max_age_s < 0:
            raise ValueError("max_age_s must not be negative")
        if queue_size < 2:
            raise ValueError("queue_size must be at least 2")
        if server_port < 0 or server_port > 65535:
            raise ValueError("server_port must be between 0 and 65535")
        if not topic.startswith("/"):
            raise ValueError("NetworkTables yaw topic must be an absolute path")
        if ntcore_module is None:
            try:
                import ntcore as ntcore_module
            except ImportError as exc:
                raise RuntimeError("Install northstar-vision[nt] to use NetworkTables robot yaw") from exc

        self.ntcore = ntcore_module
        self.clock = clock
        self.topic = topic
        self.max_age_s = max_age_s
        self.buffer = YawBuffer(queue_size)
        self.closed = True
        self.owns_instance = instance is None
        self.instance = instance if instance is not None else self.ntcore.NetworkTableInstance.create()
        self.subscriber: Any | None = None
        try:
            if self.owns_instance:
                if server_port:
                    self.instance.setServer(server, server_port)
                else:
                    self.instance.setServer(server)
                self.instance.startClient4(f"{device_id}-yaw")
            options = self.ntcore.PubSubOptions(pollStorage=queue_size, keepDuplicates=True)
            self.subscriber = self.instance.getDoubleTopic(topic).subscribe(math.nan, options)
        except BaseException:
            self._close_resources()
            raise
        self.closed = False

    def poll(self) -> int:
        if self.closed or self.subscriber is None:
            return 0
        samples = self.subscriber.readQueue()
        if not samples:
            return 0
        local_now = self.clock()
        nt_now_us = int(self.ntcore._now())
        accepted = 0
        for sample in samples:
            if int(sample.time) <= 0 or not math.isfinite(float(sample.value)):
                continue
            local_timestamp = local_now + (int(sample.time) - nt_now_us) / 1_000_000.0
            self.buffer.add(local_timestamp, float(sample.value))
            accepted += 1
        return accepted

    def yaw_at(self, capture_monotonic_s: float) -> YawEstimate | None:
        self.poll()
        return self.buffer.estimate(capture_monotonic_s, max_age_s=self.max_age_s)

    def health(self) -> dict[str, Any]:
        latest = self.buffer.latest
        latest_age_ms = None if latest is None else max(0.0, (self.clock() - latest.monotonic_s) * 1000.0)
        return {
            "yaw_source": "networktables",
            "yaw_topic": self.topic,
            "yaw_samples": len(self.buffer),
            "yaw_latest_age_ms": latest_age_ms,
            "yaw_connected": bool(self.instance.isConnected()),
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._close_resources()

    def _close_resources(self) -> None:
        errors: list[Exception] = []
        if self.subscriber is not None:
            try:
                self.subscriber.close()
            except Exception as exc:
                errors.append(exc)
            self.subscriber = None
        if self.owns_instance:
            try:
                self.instance.stopClient()
            except Exception as exc:
                errors.append(exc)
            try:
                self.ntcore.NetworkTableInstance.destroy(self.instance)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
