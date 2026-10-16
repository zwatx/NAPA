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

    def publish(self, result: FrameResult) -> None:
        with self._condition:
            self._publish_attempts += 1
            if not self._accepting:
                if self._writer_error is not None:
                    raise JsonlRecorderError(f"JSONL writer failed: {self._writer_error}")
                raise RuntimeError("Cannot publish after the JSONL publisher is closing or closed")

            if len(self._queue) >= self.config.queue_size:
                if self.config.drop_policy == "drop_newest":
                    self._dropped_newest += 1
                    return
                self._queue.popleft()
                self._dropped_oldest += 1

            self._queue.append(result)
            self._enqueued += 1
            self._queue_high_water_mark = max(self._queue_high_water_mark, len(self._queue))
            self._condition.notify()

    def telemetry(self) -> dict[str, Any]:
        """Return a thread-safe, JSON-serializable recorder health snapshot."""

        with self._condition:
            dropped_total = (
                self._dropped_oldest + self._dropped_newest + self._dropped_writer_error + self._dropped_on_close
            )
            return {
                "state": self._state,
                "queue_capacity": self.config.queue_size,
                "queue_depth": len(self._queue),
                "queue_high_water_mark": self._queue_high_water_mark,
                "publish_attempts": self._publish_attempts,
                "enqueued_records": self._enqueued,
                "written_records": self._written,
                "dropped_records": dropped_total,
                "dropped_oldest": self._dropped_oldest,
                "dropped_newest": self._dropped_newest,
                "dropped_writer_error": self._dropped_writer_error,
                "dropped_on_close": self._dropped_on_close,
                "rotations": self._rotations,
                "flushes": self._flushes,
                "fsyncs": self._fsyncs,
                "last_fsync_duration_ms": self._last_fsync_duration_ms,
                "retained_archives": self._retained_archives,
                "writer_error": self._writer_error,
                "writer_thread_alive": self._thread.is_alive(),
                "last_write_monotonic_s": self._last_write_monotonic_s,
                "last_write_duration_ms": self._last_write_duration_ms,
                "mean_write_duration_ms": (
                    None if self._written == 0 else self._total_write_duration_ms / self._written
                ),
                "max_write_duration_ms": None if self._written == 0 else self._max_write_duration_ms,
            }

    def health_state(self) -> dict[str, Any]:
        with self._condition:
            return {
                "state": self._state,
                "writer_error": self._writer_error,
                "writer_thread_alive": self._thread.is_alive(),
            }

    def wait_until_idle(self, timeout_s: float = 5.0) -> bool:
        """Wait until all currently queued records finish, primarily for tests/tools."""

        if timeout_s < 0:
            raise ValueError("timeout_s must not be negative")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._queue or self._in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self) -> None:
        with self._close_lock:
            if self.closed:
                return
            with self._condition:
                self._accepting = False
                if self._state == "running":
                    self._state = "closing"
                self._condition.notify_all()

            self._thread.join(self.config.close_timeout_s)
            if self._thread.is_alive():
                with self._condition:
                    self._force_stop = True
                    self._dropped_on_close += len(self._queue)
                    self._queue.clear()
                    self._state = "timed_out"
                    self.closed = True
                    self._condition.notify_all()
                raise JsonlRecorderCloseTimeout(
                    f"JSONL writer did not stop within {self.config.close_timeout_s:g} seconds; "
                    "queued records were discarded"
                )

            with self._condition:
                self.closed = True
                writer_error = self._writer_error
            if writer_error is not None:
                raise JsonlRecorderError(f"JSONL writer failed: {writer_error}")

    def _writer_main(self) -> None:
        try:
            while True:
                result = self._next_result()
                if result is None:
                    break
                write_start = perf_counter()
                try:
                    line = self._line_encoder(result)
                    if not line.endswith("\n"):
                        line += "\n"
                    encoded_size = len(line.encode("utf-8"))
                    if self._should_rotate(encoded_size):
                        self._rotate()
                    self._file.write(line)
                    self._current_size_bytes += encoded_size
                    now = self.clock()
                    if self.config.flush_interval_s == 0 or now >= self._next_flush_monotonic_s:
                        self._flush(now)
                    with self._condition:
                        write_duration_ms = (perf_counter() - write_start) * 1000.0
                        self._written += 1
                        self._last_write_monotonic_s = now
                        self._last_write_duration_ms = write_duration_ms
                        self._total_write_duration_ms += write_duration_ms
                        self._max_write_duration_ms = max(self._max_write_duration_ms, write_duration_ms)
                except Exception as exc:
                    self._record_writer_error(exc)
                    break
                finally:
                    with self._condition:
                        self._in_flight = False
                        self._condition.notify_all()
        finally:
            self._finish_writer()

    def _next_result(self) -> FrameResult | None:
        with self._condition:
            while not self._queue and self._accepting and not self._force_stop:
                timeout = None
                if self.config.flush_interval_s > 0:
                    timeout = max(0.0, self._next_flush_monotonic_s - self.clock())
                self._condition.wait(timeout)
                if (
                    not self._queue
                    and self.config.flush_interval_s > 0
                    and self.clock() >= self._next_flush_monotonic_s
                ):
                    try:
                        self._flush(self.clock())
                    except Exception as exc:
                        self._record_writer_error(exc)
                        return None
            if self._force_stop:
                return None
            if not self._queue:
                return None
            self._in_flight = True
            return self._queue.popleft()

    def _encode_line(self, result: FrameResult) -> str:
        payload = frame_result_to_dict(result)
        if self.config.include_writer_telemetry:
            payload["health"] = dict(payload["health"])
            payload["health"]["jsonl_writer"] = self.telemetry()
        return json.dumps(payload, separators=(",", ":"))

    def _should_rotate(self, next_record_size: int) -> bool:
        if self._current_size_bytes <= 0:
            return False
        if (
            self.config.max_file_size_bytes is not None
            and self._current_size_bytes + next_record_size > self.config.max_file_size_bytes
        ):
            return True
        return bool(
            self.config.rotation_interval_s is not None
            and self.clock() - self._opened_monotonic_s >= self.config.rotation_interval_s
        )

    def _rotate(self) -> None:
        self._flush(self.clock(), force_fsync=True)
        self._file.close()
        archive = self._next_archive_path()
        try:
            os.replace(self.path, archive)
            self._file = self.path.open("a", encoding="utf-8")
        except Exception:
            # Reopen the active path when possible so final cleanup remains safe.
            if self._file.closed:
                self._file = self.path.open("a", encoding="utf-8")
            raise
        self._current_size_bytes = 0
        self._opened_monotonic_s = self.clock()
        with self._condition:
            self._rotations += 1
        retained_archives = self._prune_archives()
        with self._condition:
            self._retained_archives = retained_archives

    def _flush(self, now: float, *, force_fsync: bool = False) -> None:
        self._file.flush()
        with self._condition:
            self._flushes += 1
        should_fsync = self.config.fsync_interval_s is not None and (
            force_fsync or self._next_fsync_monotonic_s is None or now >= self._next_fsync_monotonic_s
        )
        if should_fsync:
            fsync_start = perf_counter()
            os.fsync(self._file.fileno())
            fsync_duration_ms = (perf_counter() - fsync_start) * 1000.0
            with self._condition:
                self._fsyncs += 1
                self._last_fsync_duration_ms = fsync_duration_ms
            self._next_fsync_monotonic_s = now + self.config.fsync_interval_s
        self._next_flush_monotonic_s = now + self.config.flush_interval_s

    def _record_writer_error(self, exc: Exception) -> None:
        with self._condition:
            if self._writer_error is None:
                self._writer_error = f"{type(exc).__name__}: {exc}"
            self._accepting = False
            self._state = "failed"
            self._dropped_writer_error += len(self._queue)
            self._queue.clear()
            self._condition.notify_all()

    def _finish_writer(self) -> None:
        try:
            self._flush(self.clock(), force_fsync=True)
        except Exception as exc:
            self._record_writer_error(exc)
        try:
            self._file.close()
        except Exception as exc:
            self._record_writer_error(exc)
        with self._condition:
            if self._state not in {"failed", "timed_out"}:
                self._state = "closed"
            self._condition.notify_all()

    def _next_archive_path(self) -> Path:
        timestamp_ns = time.time_ns()
        while True:
            self._archive_sequence += 1
            archive = self.path.with_name(
                f"{self.path.stem}.archive-{timestamp_ns}-{self._archive_sequence:04d}{self.path.suffix}"
            )
            if not archive.exists():
                return archive

    def _archive_paths(self) -> list[Path]:
        pattern = f"{self.path.stem}.archive-*{self.path.suffix}"
        return sorted((path for path in self.path.parent.glob(pattern) if path.is_file()), key=lambda path: path.name)

    def _prune_archives(self) -> int:
        archives = self._archive_paths()
        excess = len(archives) - self.config.retention_files
        for archive in archives[: max(0, excess)]:
            archive.unlink(missing_ok=True)
        return min(len(archives), self.config.retention_files)
