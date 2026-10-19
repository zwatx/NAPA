from __future__ import annotations

import numpy as np

from napa_vision import load_profile
from napa_vision.capture import FrameSource
from napa_vision.orientation import RobotYawSource, YawBuffer, YawEstimate
from napa_vision.output import OutputPublisher
from napa_vision.service import NapaVisionService
from napa_vision.types import FiducialImageObservation, FrameResult


class FrameSourceStub:
    is_live = False

    def __init__(self) -> None:
        self.used = False
        self.released = False

    def read(self) -> tuple[bool, float, float, np.ndarray | None]:
        if self.used:
            return False, 0.0, 0.0, None
        self.used = True
        return True, 1000.0, 10.0, np.full((1, 1, 3), 255, dtype=np.uint8)

    def health(self) -> dict[str, object]:
        return {"source_connected": True, "capture_overwritten_frames": 3}

    def release(self) -> None:
        self.released = True


class OutputStub(OutputPublisher):
    def __init__(self) -> None:
        self.results: list[FrameResult] = []
        self.closed = False

    def publish(self, result: FrameResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.closed = True


class YawSourceStub(RobotYawSource):
    def __init__(self) -> None:
        self.requested_at: list[float] = []
        self.closed = False

    def yaw_at(self, capture_monotonic_s: float) -> YawEstimate | None:
        self.requested_at.append(capture_monotonic_s)
        return YawEstimate(0.5, 0.012, True)

    def health(self) -> dict[str, object]:
        return {"yaw_source": "test", "yaw_connected": True}

    def close(self) -> None:
        self.closed = True


def test_service_uses_yaw_samples_that_arrive_during_detection() -> None:
    class BufferedYawSource(YawSourceStub):
        def __init__(self) -> None:
            super().__init__()
            self.buffer = YawBuffer()
            self.buffer.add(9.99, 0.0)

        def yaw_at(self, capture_monotonic_s: float) -> YawEstimate | None:
            return self.buffer.estimate(capture_monotonic_s, max_age_s=0.1)

    yaw = BufferedYawSource()

    class SampleDeliveringDetector:
        def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
            yaw.buffer.add(10.01, 1.0)
            return []

    profile = load_profile("laptop")
    profile.camera.width = profile.camera.height = 1
    output = OutputStub()
    service = NapaVisionService(
        profile,
        output,
        frame_source=FrameSourceStub(),
        yaw_source=yaw,
        detector=SampleDeliveringDetector(),
    )
    service.run(max_frames=1)

    assert output.results[0].health["yaw_interpolated"] is True
    assert output.results[0].health["yaw_available"] is True


def test_service_aligns_yaw_and_merges_source_health() -> None:
    profile = load_profile("laptop")
    profile.camera.width = profile.camera.height = 1
    source = FrameSourceStub()
    output = OutputStub()
    yaw = YawSourceStub()
    service = NapaVisionService(profile, output, frame_source=source, yaw_source=yaw)
    service.run(max_frames=1)

    assert yaw.requested_at == [10.0]
    assert output.results[0].health["yaw_available"] is True
    assert output.results[0].health["yaw_sample_age_ms"] == 12.0
    assert output.results[0].health["yaw_interpolated"] is True
    assert output.results[0].health["capture_overwritten_frames"] == 3
    assert source.released and yaw.closed and output.closed


def test_service_reports_state_transition_before_detailed_health_refresh() -> None:
    class TransitionSource:
        is_live = False

        def __init__(self) -> None:
            self.reads = 0
            self.health_calls = 0

        def read(self) -> tuple[bool, float, float, np.ndarray | None]:
            self.reads += 1
            if self.reads > 2:
                return False, 0.0, 0.0, None
            return True, float(self.reads), 10.0, np.full((1, 1, 3), 255, dtype=np.uint8)

        def health(self) -> dict[str, object]:
            self.health_calls += 1
            return {"source_connected": True, "capture_backend": "test"}

        def health_state(self) -> dict[str, object]:
            return {"source_connected": self.reads == 1}

        def release(self) -> None:
            return None

    profile = load_profile("laptop")
    profile.camera.width = profile.camera.height = 1
    source = TransitionSource()
    output = OutputStub()
    service = NapaVisionService(profile, output, frame_source=source)

    service.run(max_frames=2)

    assert [result.health["source_connected"] for result in output.results] == [True, False]
    assert source.health_calls == 1
    assert all(result.health["capture_backend"] == "test" for result in output.results)


def test_legacy_frame_source_health_transition_remains_immediate() -> None:
    class LegacySource(FrameSource):
        def __init__(self) -> None:
            self.reads = 0
            self.health_calls = 0

        def read(self) -> tuple[bool, float, float, np.ndarray | None]:
            self.reads += 1
            return True, float(self.reads), 10.0, np.full((1, 1, 3), 255, dtype=np.uint8)

        def health(self) -> dict[str, object]:
            self.health_calls += 1
            return {"source_connected": self.reads == 1, "detail_generation": self.reads}

        def release(self) -> None:
            return None

    profile = load_profile("laptop")
    profile.camera.width = profile.camera.height = 1
    source = LegacySource()
    output = OutputStub()
    service = NapaVisionService(profile, output, frame_source=source)

    assert service.process_next()
    service._next_health_refresh_s = float("inf")
    assert service.process_next()
    service.close()

    assert [result.health["source_connected"] for result in output.results] == [True, False]
    assert [result.health["detail_generation"] for result in output.results] == [1, 1]
    assert source.health_calls == 3
