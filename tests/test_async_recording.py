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


