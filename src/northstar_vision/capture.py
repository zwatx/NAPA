from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

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


class OpenCVCameraSource(FrameSource):
    is_live = True

    def __init__(self, profile: PlatformProfile, *, read_attempts: int = 3) -> None:
        camera = profile.camera
        self.camera_id = camera.camera_id
        self.read_attempts = max(1, read_attempts)
        self.cap = cv2.VideoCapture(camera.camera_id)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open camera {camera.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
        if camera.fps is not None:
            self._set_optional(cv2.CAP_PROP_FPS, camera.fps, "fps")
        if camera.buffer_size is not None:
            self._set_optional(cv2.CAP_PROP_BUFFERSIZE, camera.buffer_size, "buffer_size")
        if camera.auto_exposure is not None:
            self._set_optional(cv2.CAP_PROP_AUTO_EXPOSURE, camera.auto_exposure, "auto_exposure")
        if camera.exposure is not None:
            self._set_optional(cv2.CAP_PROP_EXPOSURE, camera.exposure, "exposure")
        if camera.gain is not None:
            self._set_optional(cv2.CAP_PROP_GAIN, camera.gain, "gain")

        actual_width = int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual_width > 0 and actual_height > 0 and (actual_width != camera.width or actual_height != camera.height):
            self.cap.release()
            raise RuntimeError(
                f"Camera negotiated {actual_width}x{actual_height}, but profile calibration requires "
                f"{camera.width}x{camera.height}"
            )

    def _set_optional(self, prop: int, value: float | int, name: str) -> None:
        if not self.cap.set(prop, value):
            LOGGER.warning("Camera %s did not accept %s=%s", self.camera_id, name, value)

    def read(self) -> tuple[bool, float, float, np.ndarray | None]:
        for _ in range(self.read_attempts):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return True, time.time(), time.perf_counter(), frame
        return False, time.time(), time.perf_counter(), None

    def release(self) -> None:
        self.cap.release()


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
