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


def load_corpus_manifest(path: str | Path) -> CorpusManifest:
    """Load a strict, portable image-corpus manifest.

    Frame paths must be relative to the manifest and may not escape its
    directory. Optional SHA-256 values protect individual compressed image
    files; :func:`run_replay_benchmark` also computes a whole-corpus digest by
    default.
    """

    manifest_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise CorpusManifestError(f"Could not read corpus manifest {manifest_path}: {exc}") from exc
    try:
        data = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise CorpusManifestError(f"Invalid JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(data, Mapping):
        raise CorpusManifestError("Corpus manifest must be a JSON object")
    _reject_unknown(data, {"schema_version", "name", "description", "frames"}, "manifest")
    schema_version = data.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != CORPUS_SCHEMA_VERSION:
        raise CorpusManifestError(f"schema_version must be {CORPUS_SCHEMA_VERSION}")

    name = _nonempty_string(data.get("name"), "name")
    description_value = data.get("description", "")
    if not isinstance(description_value, str):
        raise CorpusManifestError("description must be a string")
    frame_values = data.get("frames")
    if not isinstance(frame_values, list) or not frame_values:
        raise CorpusManifestError("frames must be a non-empty array")

    root = manifest_path.parent
    frames: list[CorpusFrame] = []
    seen_paths: set[str] = set()
    for index, value in enumerate(frame_values):
        context = f"frames[{index}]"
        if not isinstance(value, Mapping):
            raise CorpusManifestError(f"{context} must be an object")
        _reject_unknown(value, {"path", "expected_tag_ids", "sha256"}, context)
        relative_path = _nonempty_string(value.get("path"), f"{context}.path")
        relative = Path(relative_path)
        if relative.is_absolute():
            raise CorpusManifestError(f"{context}.path must be relative to the manifest")
        resolved = (root / relative).resolve()
        if not resolved.is_relative_to(root):
            raise CorpusManifestError(f"{context}.path must not escape the manifest directory")
        portable_path = relative.as_posix()
        if portable_path in seen_paths:
            raise CorpusManifestError(f"Duplicate frame path {portable_path!r}")
        seen_paths.add(portable_path)

        expected_value = value.get("expected_tag_ids")
        if not isinstance(expected_value, list):
            raise CorpusManifestError(f"{context}.expected_tag_ids must be an array")
        expected: list[int] = []
        for tag_index, tag_id in enumerate(expected_value):
            if isinstance(tag_id, bool) or not isinstance(tag_id, int) or tag_id < 0:
                raise CorpusManifestError(f"{context}.expected_tag_ids[{tag_index}] must be a non-negative integer")
            expected.append(tag_id)
        if len(expected) != len(set(expected)):
            raise CorpusManifestError(f"{context}.expected_tag_ids must not contain duplicates")

        sha_value = value.get("sha256")
        if sha_value is not None:
            if (
                not isinstance(sha_value, str)
                or len(sha_value) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in sha_value)
            ):
                raise CorpusManifestError(f"{context}.sha256 must contain 64 hexadecimal characters")
            sha_value = sha_value.lower()
        frames.append(CorpusFrame(portable_path, resolved, tuple(sorted(expected)), sha_value))

    return CorpusManifest(
        path=manifest_path,
        name=name,
        description=description_value,
        frames=tuple(frames),
        manifest_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


