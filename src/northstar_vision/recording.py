from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from .output import OutputPublisher, frame_result_to_dict
from .types import FrameResult

DropPolicy = Literal["drop_oldest", "drop_newest"]


class JsonlRecorderError(RuntimeError):
    """Base error for asynchronous recorder failures."""


class JsonlRecorderCloseTimeout(JsonlRecorderError):
    """Raised when recorder shutdown reaches its configured hard deadline."""


@dataclass(frozen=True, slots=True)
class JsonlRecorderConfig:
    """Bounds and durability policy for :class:`AsyncJsonlOutputPublisher`.

    ``retention_files`` counts rotated archives; the active ``path`` is not
    included. A value of zero removes an archive as soon as its replacement is
    open. Oversized individual records are written intact rather than split.
    """

    queue_size: int = 256
    drop_policy: DropPolicy = "drop_oldest"
    flush_interval_s: float = 1.0
    fsync_interval_s: float | None = 5.0
    max_file_size_bytes: int | None = 64 * 1024 * 1024
    rotation_interval_s: float | None = 60.0 * 60.0
    retention_files: int = 8
    close_timeout_s: float = 5.0
    include_writer_telemetry: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.queue_size, bool) or not isinstance(self.queue_size, int) or self.queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        if self.drop_policy not in {"drop_oldest", "drop_newest"}:
            raise ValueError("drop_policy must be 'drop_oldest' or 'drop_newest'")
        if not math.isfinite(self.flush_interval_s) or self.flush_interval_s < 0:
            raise ValueError("flush_interval_s must not be negative")
        if self.fsync_interval_s is not None and (
            not math.isfinite(self.fsync_interval_s) or self.fsync_interval_s <= 0
        ):
            raise ValueError("fsync_interval_s must be greater than zero when set")
        if self.max_file_size_bytes is not None and self.max_file_size_bytes < 1:
            raise ValueError("max_file_size_bytes must be at least 1 when set")
        if self.rotation_interval_s is not None and (
            not math.isfinite(self.rotation_interval_s) or self.rotation_interval_s <= 0
        ):
            raise ValueError("rotation_interval_s must be greater than zero when set")
        if (
            isinstance(self.retention_files, bool)
            or not isinstance(self.retention_files, int)
            or self.retention_files < 0
        ):
            raise ValueError("retention_files must not be negative")
        if not math.isfinite(self.close_timeout_s) or self.close_timeout_s <= 0:
            raise ValueError("close_timeout_s must be greater than zero")


