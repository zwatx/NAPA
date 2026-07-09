from __future__ import annotations

import itertools
import queue
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np
import pytest

from northstar_vision import load_profile
from northstar_vision.capture import OpenCVCameraSource

CaptureResult = tuple[bool, np.ndarray | None]


class _FakeCapture:
    def __init__(
        self,
        actions: list[CaptureResult | Callable[[], CaptureResult]],
        *,
        opened: bool = True,
        width: float = 1280.0,
        height: float = 720.0,
    ) -> None:
        self.actions: queue.Queue[CaptureResult | Callable[[], CaptureResult]] = queue.Queue()
        for action in actions:
            self.actions.put(action)
        self.opened = opened
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_BUFFERSIZE: 1.0,
            cv2.CAP_PROP_AUTO_EXPOSURE: 0.25,
            cv2.CAP_PROP_EXPOSURE: 5.0,
            cv2.CAP_PROP_GAIN: 2.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }
        self.accepted: dict[int, float] = {}
        self.operation_threads: list[int] = []
        self.read_started = threading.Event()
        self.release_count = 0

    def _record_thread(self) -> None:
        self.operation_threads.append(threading.get_ident())

    def isOpened(self) -> bool:
        self._record_thread()
        return self.opened

    def set(self, prop_id: int, value: float) -> bool:
        self._record_thread()
        self.accepted[prop_id] = value
        return True

    def get(self, prop_id: int) -> float:
        self._record_thread()
        return self.properties.get(prop_id, 0.0)

    def read(self) -> CaptureResult:
        self._record_thread()
        self.read_started.set()
        try:
            action = self.actions.get(timeout=0.05)
        except queue.Empty:
            return False, None
        return action() if callable(action) else action

    def getBackendName(self) -> str:
        self._record_thread()
        return "FAKE"

    def release(self) -> None:
        self._record_thread()
        self.release_count += 1


def _frame(value: int) -> np.ndarray:
    return np.full((720, 1280, 3), value, dtype=np.uint8)


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 1.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while not predicate():
        if time.perf_counter() >= deadline:
            pytest.fail("condition was not reached before timeout")
        time.sleep(0.001)


def test_latest_frame_replaces_stale_frames_and_reports_properties() -> None:
    profile = load_profile("laptop")
    capture = _FakeCapture([(True, _frame(1)), (True, _frame(2)), (True, _frame(3))])
    factory_threads: list[int] = []

    def factory(_camera_id: int | str) -> _FakeCapture:
        factory_threads.append(threading.get_ident())
        return capture

    wall_values = itertools.count(100)
    monotonic_values = itertools.count(200)
    source = OpenCVCameraSource(
        profile,
        capture_factory=factory,
        wall_clock=lambda: float(next(wall_values)),
        monotonic_clock=lambda: float(next(monotonic_values)),
        read_attempts=1,
        reconnect_initial_s=0.01,
        reconnect_max_s=0.02,
    )
    try:
        _wait_until(lambda: source.health()["capture_frames_captured"] == 3)
        ok, timestamp, capture_monotonic_s, frame = source.read()
        assert ok and frame is not None
        assert int(frame[0, 0, 0]) == 3
        assert timestamp == 102.0
        assert capture_monotonic_s == 202.0

        health = source.health()
        assert health["capture_frames_captured"] == 3
        assert health["capture_frames_delivered"] == 1
        assert health["capture_frames_overwritten"] == 2
        assert health["capture_frames_dropped"] == 2
        assert health["capture_requested"]["width"] == 1280
        assert health["capture_negotiated"]["width"] == 1280.0
        assert health["capture_negotiated"]["height"] == 720.0
        assert health["capture_property_accepted"]["width"] is True
        assert health["capture_backend"] == "FAKE"
        assert health["capture_fourcc"] == "MJPG"
    finally:
        source.release()

    worker_thread_ids = set(factory_threads + capture.operation_threads)
    assert len(worker_thread_ids) == 1
    assert threading.get_ident() not in worker_thread_ids
    assert capture.release_count == 1


def test_camera_reconnects_after_read_and_open_failures() -> None:
    profile = load_profile("laptop")
    disconnect = threading.Event()

    def wait_then_disconnect() -> CaptureResult:
        assert disconnect.wait(1.0)
        return False, None

    first = _FakeCapture([(True, _frame(1)), wait_then_disconnect])
    unavailable = _FakeCapture([], opened=False)
    recovered = _FakeCapture([(True, _frame(2))])
    captures: queue.Queue[_FakeCapture] = queue.Queue()
    for capture in (first, unavailable, recovered):
        captures.put(capture)

    def factory(_camera_id: int | str) -> _FakeCapture:
        return captures.get_nowait()

    source = OpenCVCameraSource(
        profile,
        capture_factory=factory,
        read_attempts=1,
        reconnect_initial_s=0.002,
        reconnect_max_s=0.004,
        read_timeout_s=1.0,
    )
    try:
        assert source.read()[3] is not None
        disconnect.set()
        ok, _, _, frame = source.read()
        assert ok and frame is not None
        assert int(frame[0, 0, 0]) == 2
        health = source.health()
        assert health["capture_open_attempts"] >= 3
        assert health["capture_open_failures"] >= 1
        assert health["capture_read_failures"] >= 1
        assert health["capture_disconnects"] >= 1
        assert health["capture_reconnects"] >= 1
    finally:
        source.release()

    assert first.release_count == 1
    assert unavailable.release_count == 1
    assert recovered.release_count == 1


