from __future__ import annotations

import math
import socket
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from northstar_vision.orientation import NetworkTablesRobotYawSource
from northstar_vision.output import NTOutputPublisher
from northstar_vision.types import FrameResult

ntcore = pytest.importorskip("ntcore")


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_real_ntcore_yaw_timestamp_delivery(tmp_path: Path) -> None:
    port = _unused_tcp_port()
    server = ntcore.NetworkTableInstance.create()
    publisher = None
    source = None
    try:
        server.startServer(str(tmp_path / "networktables.json"), "127.0.0.1", 0, port)
        publisher = server.getDoubleTopic("/RobotState/yaw").publish(
            ntcore.PubSubOptions(keepDuplicates=True, sendAll=True)
        )
        source = NetworkTablesRobotYawSource(
            "integration-test",
            "127.0.0.1",
            server_port=port,
            topic="/RobotState/yaw",
            max_age_s=0.5,
        )
        assert _wait_until(source.instance.isConnected)

        publisher.set(math.radians(179.0))
        server.flush()
        assert _wait_until(lambda: source.poll() >= 1)
        first = source.buffer.latest
        assert first is not None

        publisher.set(math.radians(-179.0))
        server.flush()
        assert _wait_until(lambda: source.poll() >= 1)
        second = source.buffer.latest
        assert second is not None and second.monotonic_s >= first.monotonic_s

        estimate = source.yaw_at((first.monotonic_s + second.monotonic_s) / 2.0)
        assert estimate is not None
        assert abs(abs(estimate.yaw_rad) - math.pi) < math.radians(0.5)
    finally:
        if source is not None:
            source.close()
        if publisher is not None:
            publisher.close()
        server.stopServer()
        ntcore.NetworkTableInstance.destroy(server)


def test_real_ntcore_default_yaw_publisher_duplicate_suppression_ages_out(tmp_path: Path) -> None:
    port = _unused_tcp_port()
    server = ntcore.NetworkTableInstance.create()
    publisher = None
    source = None
    try:
        server.startServer(str(tmp_path / "yaw-duplicate-networktables.json"), "127.0.0.1", 0, port)
        publisher = server.getDoubleTopic("/RobotState/yaw").publish()
        source = NetworkTablesRobotYawSource(
            "duplicate-test",
            "127.0.0.1",
            server_port=port,
            topic="/RobotState/yaw",
            max_age_s=0.05,
        )
        assert _wait_until(source.instance.isConnected)

        publisher.set(1.0)
        server.flush()
        assert _wait_until(lambda: source.poll() >= 1)
        time.sleep(0.08)

        publisher.set(1.0)
        server.flush()
        time.sleep(0.03)
        assert source.poll() == 0
        assert source.yaw_at(time.perf_counter()) is None
    finally:
        if source is not None:
            source.close()
        if publisher is not None:
            publisher.close()
        server.stopServer()
        ntcore.NetworkTableInstance.destroy(server)


def test_real_ntcore_observation_timestamp_and_delivery_policy(tmp_path: Path) -> None:
    port = _unused_tcp_port()
    server = ntcore.NetworkTableInstance.create()
    subscriber = None
    publisher = None
    try:
        server.startServer(str(tmp_path / "observations-networktables.json"), "127.0.0.1", 0, port)
        subscriber = server.getDoubleArrayTopic("/integration/output/observations").subscribe(
            [], ntcore.PubSubOptions(pollStorage=8, keepDuplicates=True)
        )
        publisher = NTOutputPublisher(
            "integration",
            "127.0.0.1",
            server_port=port,
            send_all=False,
        )
        assert _wait_until(publisher.instance.isConnected)

        capture_monotonic_s = time.perf_counter() - 0.05
        publisher.publish(FrameResult(timestamp=time.time(), capture_monotonic_s=capture_monotonic_s))
        publisher.instance.flush()

        samples: list[object] = []

        def received() -> bool:
            samples.extend(subscriber.readQueue())
            return bool(samples)

        assert _wait_until(received)
        sample = samples[-1]
        assert sample.value == [0.0]  # type: ignore[attr-defined]
        observed_age_ms = (int(ntcore._now()) - int(sample.time)) / 1000.0  # type: ignore[attr-defined]
        assert 35.0 <= observed_age_ms <= 500.0
        assert publisher.telemetry()["delivery_policy"] == "latest_value"

        samples.clear()
        publisher.publish(FrameResult(timestamp=time.time(), capture_monotonic_s=time.perf_counter() - 0.02))
        publisher.instance.flush()
        assert _wait_until(received)
        second_sample = samples[-1]
        assert second_sample.value == [0.0]  # type: ignore[attr-defined]
        assert int(second_sample.time) > int(sample.time)  # type: ignore[attr-defined]
    finally:
        if publisher is not None:
            publisher.close()
        if subscriber is not None:
            subscriber.close()
        server.stopServer()
        ntcore.NetworkTableInstance.destroy(server)


def test_real_ntcore_close_flushes_pending_observation(tmp_path: Path) -> None:
    port = _unused_tcp_port()
    server = ntcore.NetworkTableInstance.create()
    subscriber = None
    publisher = None
    try:
        server.startServer(str(tmp_path / "close-networktables.json"), "127.0.0.1", 0, port)
        subscriber = server.getDoubleArrayTopic("/close-test/output/observations").subscribe(
            [], ntcore.PubSubOptions(pollStorage=8, keepDuplicates=True)
        )
        publisher = NTOutputPublisher("close-test", "127.0.0.1", server_port=port, send_all=False)
        assert _wait_until(publisher.instance.isConnected)

        publisher.publish(FrameResult(timestamp=time.time(), capture_monotonic_s=time.perf_counter()))
        publisher.close()
        publisher = None

        samples: list[object] = []

        def received() -> bool:
            samples.extend(subscriber.readQueue())
            return bool(samples)

        assert _wait_until(received)
        assert samples[-1].value == [0.0]  # type: ignore[attr-defined]
    finally:
        if publisher is not None:
            publisher.close()
        if subscriber is not None:
            subscriber.close()
        server.stopServer()
        ntcore.NetworkTableInstance.destroy(server)
