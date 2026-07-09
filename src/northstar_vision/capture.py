from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .config import PlatformProfile

LOGGER = logging.getLogger(__name__)


class FrameSource(ABC):
    is_live = False

    @abstractmethod
    def read(self) -> tuple[bool, float, float, np.ndarray | None]:
        """Return success, wall time, monotonic receive time, and a frame."""

    @abstractmethod
    def release(self) -> None:
        """Release source resources."""

    def health(self) -> dict[str, Any]:
        """Return JSON-compatible source health data."""

        return {"source_connected": True}


class VideoCaptureLike(Protocol):
    """The small subset of ``cv2.VideoCapture`` used by the live source."""

    def isOpened(self) -> bool: ...

    def set(self, prop_id: int, value: float) -> bool: ...

    def get(self, prop_id: int) -> float: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[int | str], VideoCaptureLike]


class _FatalCameraConfiguration(RuntimeError):
    pass


class OpenCVCameraSource(FrameSource):
    is_live = True

    def __init__(
        self,
        profile: PlatformProfile,
        *,
        read_attempts: int = 3,
        reconnect_initial_s: float = 0.1,
        reconnect_max_s: float = 2.0,
        read_timeout_s: float | None = None,
        shutdown_timeout_s: float = 2.0,
        capture_factory: CaptureFactory = cv2.VideoCapture,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        camera = profile.camera
        if isinstance(read_attempts, bool) or not isinstance(read_attempts, int) or read_attempts < 1:
            raise ValueError("read_attempts must be at least one")
        if not math.isfinite(reconnect_initial_s) or reconnect_initial_s <= 0:
            raise ValueError("reconnect_initial_s must be greater than zero")
        if not math.isfinite(reconnect_max_s) or reconnect_max_s < reconnect_initial_s:
            raise ValueError("reconnect_max_s must be at least reconnect_initial_s")
        if read_timeout_s is not None and (not math.isfinite(read_timeout_s) or read_timeout_s <= 0):
            raise ValueError("read_timeout_s must be greater than zero")
        if not math.isfinite(shutdown_timeout_s) or shutdown_timeout_s < 0:
            raise ValueError("shutdown_timeout_s must not be negative")

        self.camera = camera
        self.camera_id = camera.camera_id
        self.read_attempts = read_attempts
        self.reconnect_initial_s = reconnect_initial_s
        self.reconnect_max_s = reconnect_max_s
        self.read_timeout_s = read_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self._capture_factory = capture_factory
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock

        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest: tuple[float, float, np.ndarray] | None = None
        self._closed = False
        self._connected = False
        self._fatal_error: RuntimeError | None = None
        self._last_error: str | None = None
        self._frames_captured = 0
        self._frames_delivered = 0
        self._frames_overwritten = 0
        self._open_attempts = 0
        self._open_failures = 0
        self._read_failures = 0
        self._disconnects = 0
        self._reconnects = 0
        self._ever_connected = False
        self._shutdown_timed_out = False
        self._last_frame_monotonic_s: float | None = None
        self._property_specs = self._camera_properties()
        self._accepted_properties: dict[str, bool] = {}
        self._negotiated_properties: dict[str, float | None] = {}
        self._negotiated_backend: str | None = None
        self._negotiated_fourcc: str | None = None
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{self.camera_id}-capture",
            daemon=True,
        )
        self._thread.start()

class ReplaySource(FrameSource):
    def __init__(self, replay_path: str | Path, *, loop: bool = False) -> None:
        self.path = Path(replay_path).expanduser()
        self.loop = loop
        self.video: cv2.VideoCapture | None = None
        self.images: list[Path] = []
        self.index = 0
        if not self.path.exists():
            raise RuntimeError(f"Replay source does not exist: {self.path}")
        if self.path.is_dir():
            self.images = sorted(
                path
                for path in self.path.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if not self.images:
                raise RuntimeError(f"Replay directory contains no PNG or JPEG images: {self.path}")
        else:
            self.video = cv2.VideoCapture(str(self.path))
            if not self.video.isOpened():
                self.video.release()
                raise RuntimeError(f"Could not open replay source {self.path}")

    def read(self) -> tuple[bool, float, float, np.ndarray | None]:
        if self.video is not None:
            ok, frame = self.video.read()
            if ok and frame is not None:
                return True, time.time(), time.perf_counter(), frame
            if self.loop:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.video.read()
                if ok and frame is not None:
                    return True, time.time(), time.perf_counter(), frame
            return False, time.time(), time.perf_counter(), None

        if self.index >= len(self.images):
            if not self.loop:
                return False, time.time(), time.perf_counter(), None
            self.index = 0
        path = self.images[self.index]
        self.index += 1
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not decode replay image {path}")
        return True, time.time(), time.perf_counter(), frame

    def release(self) -> None:
        if self.video is not None:
            self.video.release()


def make_frame_source(
    profile: PlatformProfile,
    replay_path: str | Path | None = None,
    *,
    loop_replay: bool = False,
) -> FrameSource:
    path = replay_path or profile.replay_path
    if path:
        return ReplaySource(path, loop=loop_replay)
    return OpenCVCameraSource(profile)
