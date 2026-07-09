from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from northstar_vision.recording import (
    AsyncJsonlOutputPublisher,
    JsonlRecorderCloseTimeout,
    JsonlRecorderConfig,
)
from northstar_vision.types import FrameResult


def _frame(index: int) -> FrameResult:
    return FrameResult(timestamp=float(index), health={"frame": index})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"queue_size": 0}, "queue_size"),
        ({"drop_policy": "wait"}, "drop_policy"),
        ({"flush_interval_s": -1.0}, "flush_interval_s"),
        ({"fsync_interval_s": 0.0}, "fsync_interval_s"),
        ({"max_file_size_bytes": 0}, "max_file_size_bytes"),
        ({"rotation_interval_s": 0.0}, "rotation_interval_s"),
        ({"retention_files": -1}, "retention_files"),
        ({"close_timeout_s": 0.0}, "close_timeout_s"),
        ({"close_timeout_s": float("nan")}, "close_timeout_s"),
        ({"flush_interval_s": float("inf")}, "flush_interval_s"),
    ],
)
def test_recorder_config_rejects_unbounded_or_invalid_values(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        JsonlRecorderConfig(**overrides)  # type: ignore[arg-type]


def test_async_recorder_writes_complete_json_and_reports_health(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "frames.jsonl"
    publisher = AsyncJsonlOutputPublisher(
        path,
        config=JsonlRecorderConfig(
            flush_interval_s=0,
            max_file_size_bytes=None,
            rotation_interval_s=None,
        ),
    )

    publisher.publish(_frame(7))
    assert publisher.wait_until_idle()
    running_health = publisher.telemetry()
    assert running_health["written_records"] == 1
    assert running_health["writer_error"] is None
    assert running_health["last_write_duration_ms"] >= 0
    assert running_health["mean_write_duration_ms"] >= 0
    assert running_health["max_write_duration_ms"] >= running_health["last_write_duration_ms"]
    publisher.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["timestamp_s"] == 7.0
    assert payload["health"]["frame"] == 7
    assert payload["health"]["jsonl_writer"]["queue_capacity"] == 256
    assert publisher.telemetry()["state"] == "closed"
    assert publisher.telemetry()["fsyncs"] >= 1


@pytest.mark.parametrize(
    ("drop_policy", "expected_timestamps", "drop_counter"),
    [
        ("drop_newest", [0.0, 1.0, 2.0], "dropped_newest"),
        ("drop_oldest", [0.0, 2.0, 3.0], "dropped_oldest"),
    ],
)
def test_async_recorder_applies_bounded_drop_policy(
    tmp_path: Path,
    drop_policy: str,
    expected_timestamps: list[float],
    drop_counter: str,
) -> None:
    encoder_started = threading.Event()
    release_encoder = threading.Event()

    def controlled_encoder(result: FrameResult) -> str:
        if result.timestamp == 0.0:
            encoder_started.set()
            assert release_encoder.wait(2.0)
        return json.dumps({"timestamp_s": result.timestamp})

    path = tmp_path / f"{drop_policy}.jsonl"
    publisher = AsyncJsonlOutputPublisher(
        path,
        config=JsonlRecorderConfig(
            queue_size=2,
            drop_policy=drop_policy,  # type: ignore[arg-type]
            flush_interval_s=0,
            max_file_size_bytes=None,
            rotation_interval_s=None,
            include_writer_telemetry=False,
        ),
        _line_encoder=controlled_encoder,
    )
    publisher.publish(_frame(0))
    assert encoder_started.wait(1.0)
    publisher.publish(_frame(1))
    publisher.publish(_frame(2))
    publisher.publish(_frame(3))
    health = publisher.telemetry()
    assert health["queue_depth"] == 2
    assert health["queue_high_water_mark"] == 2
    assert health[drop_counter] == 1

    release_encoder.set()
    publisher.close()
    timestamps = [json.loads(line)["timestamp_s"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert timestamps == expected_timestamps


