from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import cv2
import numpy as np

from .fiducials import OpenCVArucoFiducialDetector
from .types import FiducialImageObservation

CORPUS_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 1


class CorpusManifestError(ValueError):
    """Raised when a replay corpus manifest is invalid or not reproducible."""


class FiducialDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[FiducialImageObservation]: ...


@dataclass(frozen=True, slots=True)
class CorpusFrame:
    relative_path: str
    path: Path
    expected_tag_ids: tuple[int, ...]
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    path: Path
    name: str
    description: str
    frames: tuple[CorpusFrame, ...]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    repetitions: int = 1
    warmup_passes: int = 1
    max_failure_samples: int = 25
    verify_integrity: bool = True

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if self.warmup_passes < 0:
            raise ValueError("warmup_passes must not be negative")
        if self.max_failure_samples < 0:
            raise ValueError("max_failure_samples must not be negative")


@dataclass(frozen=True, slots=True)
class BenchmarkThresholds:
    """Optional accuracy and latency gates for a replay run."""

    min_recall: float | None = None
    min_precision: float | None = None
    max_p95_total_ms: float | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.min_recall, "min_recall"), (self.min_precision, "min_precision")):
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_p95_total_ms is not None and (
            not np.isfinite(self.max_p95_total_ms) or self.max_p95_total_ms <= 0.0
        ):
            raise ValueError("max_p95_total_ms must be greater than zero")


def evaluate_benchmark_gate(summary: Mapping[str, Any], thresholds: BenchmarkThresholds) -> dict[str, Any]:
    """Evaluate configured benchmark thresholds without hiding missing metrics."""

    accuracy = summary["accuracy"]
    total_timing = summary["timing"]["total_ms"]
    measured = {
        "recall": accuracy["recall"],
        "precision": accuracy["precision"],
        "p95_total_ms": total_timing["p95_ms"],
    }
    configured = {
        "min_recall": thresholds.min_recall,
        "min_precision": thresholds.min_precision,
        "max_p95_total_ms": thresholds.max_p95_total_ms,
    }
    failures: list[str] = []
    for threshold_name, metric_name in (("min_recall", "recall"), ("min_precision", "precision")):
        threshold = configured[threshold_name]
        value = measured[metric_name]
        if threshold is not None and (value is None or value < threshold):
            failures.append(f"{metric_name} {value!r} is below required {threshold:g}")
    if thresholds.max_p95_total_ms is not None and measured["p95_total_ms"] > thresholds.max_p95_total_ms:
        failures.append(f"p95_total_ms {measured['p95_total_ms']:.3f} exceeds allowed {thresholds.max_p95_total_ms:g}")
    return {"passed": not failures, "thresholds": configured, "measured": measured, "failures": failures}


