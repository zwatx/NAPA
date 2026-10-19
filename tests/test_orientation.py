from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from napa_vision.orientation import NetworkTablesRobotYawSource, StaticRobotYawSource, YawBuffer


def test_yaw_buffer_interpolates_across_angle_wrap() -> None:
    buffer = YawBuffer(max_samples=4)
    buffer.add(1.0, math.radians(179.0))
    buffer.add(2.0, math.radians(-179.0))
    estimate = buffer.estimate(1.5, max_age_s=1.0)
    assert estimate is not None
    assert abs(abs(estimate.yaw_rad) - math.pi) < math.radians(0.01)
    assert estimate.interpolated


def test_yaw_buffer_rejects_stale_and_bounds_storage() -> None:
    buffer = YawBuffer(max_samples=3)
    for index in range(6):
        buffer.add(float(index), float(index))
    assert len(buffer) == 3
    assert buffer.estimate(10.0, max_age_s=1.0) is None
    with pytest.raises(ValueError, match="max_age_s"):
        buffer.estimate(5.0, max_age_s=float("nan"))
    assert buffer.latest is not None and buffer.latest.monotonic_s == 5.0


def test_static_yaw_source_is_always_fresh() -> None:
    source = StaticRobotYawSource(2.0 * math.pi + 0.25)
    estimate = source.yaw_at(100.0)
    assert estimate is not None
    assert estimate.yaw_rad == pytest.approx(0.25)
    assert estimate.sample_age_s == 0.0


def test_networktables_yaw_translates_timestamp_and_releases_resources() -> None:
    class FakeSubscriber:
        def __init__(self) -> None:
            self.closed = False
            self.samples = [SimpleNamespace(time=1_950_000, value=1.0), SimpleNamespace(time=2_000_000, value=1.2)]

        def readQueue(self) -> list[SimpleNamespace]:
            samples, self.samples = self.samples, []
            return samples

        def close(self) -> None:
            self.closed = True

    subscriber = FakeSubscriber()

    class FakeTopic:
        def subscribe(self, _default: float, _options: object) -> FakeSubscriber:
            return subscriber

    class FakeInstance:
        def __init__(self) -> None:
            self.stopped = False

        def setServer(self, _server: str) -> None:
            return None

        def startClient4(self, _identity: str) -> None:
            return None

        def getDoubleTopic(self, _topic: str) -> FakeTopic:
            return FakeTopic()

        def isConnected(self) -> bool:
            return True

        def stopClient(self) -> None:
            self.stopped = True

    instance = FakeInstance()

    class FakeNetworkTableInstance:
        destroyed = False

        @staticmethod
        def create() -> FakeInstance:
            return instance

        @staticmethod
        def destroy(_instance: FakeInstance) -> None:
            FakeNetworkTableInstance.destroyed = True

    fake_ntcore = SimpleNamespace(
        NetworkTableInstance=FakeNetworkTableInstance,
        PubSubOptions=lambda **kwargs: kwargs,
        _now=lambda: 2_000_000,
    )
    source = NetworkTablesRobotYawSource(
        "device",
        "server",
        max_age_s=0.1,
        clock=lambda: 10.0,
        ntcore_module=fake_ntcore,
    )
    estimate = source.yaw_at(9.975)
    assert estimate is not None
    assert estimate.interpolated
    assert estimate.yaw_rad == pytest.approx(1.1)
    assert source.health()["yaw_connected"] is True
    source.close()
    assert subscriber.closed and instance.stopped and FakeNetworkTableInstance.destroyed
