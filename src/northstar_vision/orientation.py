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
        timestamps = [value.monotonic_s for value in self._samples]
        index = bisect_left(timestamps, sample.monotonic_s)
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
        timestamps = [value.monotonic_s for value in self._samples]
        index = bisect_left(timestamps, monotonic_s)
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


