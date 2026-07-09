from __future__ import annotations

import numpy as np

from northstar_vision import load_profile
from northstar_vision.orientation import RobotYawSource, YawEstimate
from northstar_vision.output import OutputPublisher
from northstar_vision.service import NorthstarVisionService
from northstar_vision.types import FrameResult


class FrameSourceStub:
    is_live = False

    def __init__(self) -> None:
        self.used = False
        self.released = False

    def read(self) -> tuple[bool, float, float, np.ndarray | None]:
        if self.used:
            return False, 0.0, 0.0, None
        self.used = True
        return True, 1000.0, 10.0, np.full((720, 1280, 3), 255, dtype=np.uint8)

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


def test_service_aligns_yaw_and_merges_source_health() -> None:
    profile = load_profile("laptop")
    source = FrameSourceStub()
    output = OutputStub()
    yaw = YawSourceStub()
    service = NorthstarVisionService(profile, output, frame_source=source, yaw_source=yaw)
    service.run(max_frames=1)

    assert yaw.requested_at == [10.0]
    assert output.results[0].health["yaw_available"] is True
    assert output.results[0].health["yaw_sample_age_ms"] == 12.0
    assert output.results[0].health["yaw_interpolated"] is True
    assert output.results[0].health["capture_overwritten_frames"] == 3
    assert source.released and yaw.closed and output.closed
