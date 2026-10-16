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


def _start_server(server: object, persistence_path: Path) -> int:
    """Start on an ephemeral port and prove that endpoint belongs to this server."""

    last_error: Exception | None = None
    for attempt in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        try:
            server.startServer(str(persistence_path), "127.0.0.1", 0, port)  # type: ignore[attr-defined]
        except RuntimeError as exc:
            last_error = exc
            server.stopServer()  # type: ignore[attr-defined]
            continue
        nonce = f"{time.time_ns()}-{attempt}"
        if _endpoint_matches_server(server, port, nonce):
            return port
        last_error = RuntimeError(f"NetworkTables server did not become reachable on port {port}")
        server.stopServer()  # type: ignore[attr-defined]
    raise RuntimeError("could not bind a local NetworkTables test server") from last_error


def _endpoint_matches_server(server: object, port: int, nonce: str, timeout_s: float = 1.0) -> bool:
    """Verify a port at the NT protocol level using a server-owned nonce topic."""

    topic_name = f"/__northstar_test_probe/{nonce}"
    publisher = server.getStringTopic(topic_name).publish()  # type: ignore[attr-defined]
    probe = ntcore.NetworkTableInstance.create()
    subscriber = None
    try:
        subscriber = probe.getStringTopic(topic_name).subscribe("")
        probe.setServer("127.0.0.1", port)
        probe.startClient4(f"northstar-port-probe-{nonce}")
        publisher.set(nonce)
        server.flush()  # type: ignore[attr-defined]
        return _wait_until(lambda: probe.isConnected() and subscriber.get() == nonce, timeout_s=timeout_s)
    finally:
        if subscriber is not None:
            subscriber.close()
        publisher.close()
        probe.stopClient()
        ntcore.NetworkTableInstance.destroy(probe)


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_start_server_retries_when_endpoint_verification_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeServer:
        def __init__(self) -> None:
            self.starts = 0
            self.stops = 0

        def startServer(self, _path: str, _address: str, _port3: int, _port4: int) -> None:
            self.starts += 1

        def stopServer(self) -> None:
            self.stops += 1

    outcomes = iter((False, True))
    monkeypatch.setattr(f"{__name__}._endpoint_matches_server", lambda *_args: next(outcomes))
    server = FakeServer()

    assert _start_server(server, tmp_path / "retry-networktables.json") > 0
    assert server.starts == 2
    assert server.stops == 1


def test_real_ntcore_yaw_timestamp_delivery(tmp_path: Path) -> None:
    server = ntcore.NetworkTableInstance.create()
    publisher = None
    source = None
    try:
        port = _start_server(server, tmp_path / "networktables.json")
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
    server = ntcore.NetworkTableInstance.create()
    publisher = None
    source = None
    try:
        port = _start_server(server, tmp_path / "yaw-duplicate-networktables.json")
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
    server = ntcore.NetworkTableInstance.create()
    subscriber = None
    publisher = None
    try:
        port = _start_server(server, tmp_path / "observations-networktables.json")
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
    server = ntcore.NetworkTableInstance.create()
    subscriber = None
    publisher = None
    try:
        port = _start_server(server, tmp_path / "close-networktables.json")
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
