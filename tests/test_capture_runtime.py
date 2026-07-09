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


