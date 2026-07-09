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


class AsyncJsonlOutputPublisher(OutputPublisher):
    """Write frame records off the processing thread with bounded memory.

    ``publish`` never waits for disk I/O. When the bounded queue is full it
    applies the configured drop policy and exposes the decision through
    :meth:`telemetry`. ``close`` drains accepted records for at most
    ``close_timeout_s``; a stuck writer raises :class:`TimeoutError` and the
    daemon writer is told to discard any work that has not started.
    """

    def __init__(
        self,
        path: str | Path = "northstar_output.jsonl",
        *,
        config: JsonlRecorderConfig | None = None,
        clock: Callable[[], float] = perf_counter,
        _line_encoder: Callable[[FrameResult], str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.config = config or JsonlRecorderConfig()
        self.clock = clock
        self._line_encoder = _line_encoder or self._encode_line
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._queue: deque[FrameResult] = deque()
        self._in_flight = False
        self._accepting = True
        self._force_stop = False
        self.closed = False
        self._state = "running"
        self._writer_error: str | None = None

        self._publish_attempts = 0
        self._enqueued = 0
        self._written = 0
        self._dropped_oldest = 0
        self._dropped_newest = 0
        self._dropped_writer_error = 0
        self._dropped_on_close = 0
        self._queue_high_water_mark = 0
        self._rotations = 0
        self._flushes = 0
        self._fsyncs = 0
        self._last_fsync_duration_ms: float | None = None
        self._last_write_monotonic_s: float | None = None
        self._last_write_duration_ms: float | None = None
        self._total_write_duration_ms = 0.0
        self._max_write_duration_ms = 0.0
        self._archive_sequence = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        try:
            try:
                self._current_size_bytes = self.path.stat().st_size
            except OSError:
                self._current_size_bytes = 0
            now = self.clock()
            self._opened_monotonic_s = now
            self._next_flush_monotonic_s = now + self.config.flush_interval_s
            self._next_fsync_monotonic_s = (
                None if self.config.fsync_interval_s is None else now + self.config.fsync_interval_s
            )
            self._retained_archives = self._prune_archives()
        except BaseException:
            self._file.close()
            raise

        self._thread = threading.Thread(
            target=self._writer_main,
            name=f"northstar-jsonl-{self.path.name}",
            daemon=True,
        )
        self._thread.start()

