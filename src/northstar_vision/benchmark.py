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


def run_replay_benchmark(
    manifest: CorpusManifest | str | Path,
    *,
    detector: FiducialDetector | None = None,
    options: BenchmarkOptions | None = None,
) -> dict[str, Any]:
    """Stream an annotated replay corpus and return a compact JSON summary."""

    if not isinstance(manifest, CorpusManifest):
        manifest = load_corpus_manifest(manifest)
    options = options or BenchmarkOptions()
    detector = detector or OpenCVArucoFiducialDetector()
    corpus_sha256 = _verify_corpus(manifest) if options.verify_integrity else None

    for _ in range(options.warmup_passes):
        for frame in manifest.frames:
            detector.detect(_decode_image(frame))

    stages: dict[str, list[float]] = {"load_ms": [], "detect_ms": [], "score_ms": [], "total_ms": []}
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    false_positive_frames = 0
    expected_instances = 0
    detected_instances = 0
    peak_decoded_image_bytes = 0
    failure_count = 0
    failure_samples: list[dict[str, Any]] = []

    for repetition in range(options.repetitions):
        for frame in manifest.frames:
            total_start = perf_counter()
            load_start = perf_counter()
            image = _decode_image(frame)
            load_end = perf_counter()
            peak_decoded_image_bytes = max(peak_decoded_image_bytes, image.nbytes)

            detect_start = perf_counter()
            observations = detector.detect(image)
            detect_end = perf_counter()
            detected_ids = [observation.tag_id for observation in observations]

            score_start = perf_counter()
            expected = Counter(frame.expected_tag_ids)
            detected = Counter(detected_ids)
            frame_true_positives = sum(min(count, detected[tag_id]) for tag_id, count in expected.items())
            frame_false_negatives = sum(max(0, count - detected[tag_id]) for tag_id, count in expected.items())
            frame_false_positives = sum(max(0, count - expected[tag_id]) for tag_id, count in detected.items())
            score_end = perf_counter()

            true_positives += frame_true_positives
            false_negatives += frame_false_negatives
            false_positives += frame_false_positives
            expected_instances += len(frame.expected_tag_ids)
            detected_instances += len(detected_ids)
            if frame_false_positives:
                false_positive_frames += 1
            if frame_false_positives or frame_false_negatives:
                failure_count += 1
                if len(failure_samples) < options.max_failure_samples:
                    failure_samples.append(
                        {
                            "repetition": repetition,
                            "path": frame.relative_path,
                            "expected_tag_ids": list(frame.expected_tag_ids),
                            "detected_tag_ids": sorted(detected_ids),
                            "missing_tag_ids": sorted((expected - detected).elements()),
                            "unexpected_tag_ids": sorted((detected - expected).elements()),
                        }
                    )

            stages["load_ms"].append((load_end - load_start) * 1000.0)
            stages["detect_ms"].append((detect_end - detect_start) * 1000.0)
            stages["score_ms"].append((score_end - score_start) * 1000.0)
            stages["total_ms"].append((perf_counter() - total_start) * 1000.0)

    evaluated_frames = len(manifest.frames) * options.repetitions
    precision_denominator = true_positives + false_positives
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": {
            "name": manifest.name,
            "description": manifest.description,
            "manifest_path": str(manifest.path),
            "manifest_sha256": manifest.manifest_sha256,
            "corpus_sha256": corpus_sha256,
            "frames": len(manifest.frames),
            "positive_frames": sum(bool(frame.expected_tag_ids) for frame in manifest.frames),
            "negative_frames": sum(not frame.expected_tag_ids for frame in manifest.frames),
        },
        "run": {
            "repetitions": options.repetitions,
            "warmup_passes": options.warmup_passes,
            "evaluated_frames": evaluated_frames,
            "integrity_verified": options.verify_integrity,
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "accuracy": {
            "expected_instances": expected_instances,
            "detected_instances": detected_instances,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "recall": None if expected_instances == 0 else true_positives / expected_instances,
            "precision": None if precision_denominator == 0 else true_positives / precision_denominator,
            "false_positive_frames": false_positive_frames,
            "false_positives_per_frame": false_positives / evaluated_frames,
        },
        "timing": {stage: _timing_summary(samples) for stage, samples in stages.items()},
        "memory": {
            "peak_decoded_image_bytes": peak_decoded_image_bytes,
            "per_frame_results_retained": False,
            "failure_sample_limit": options.max_failure_samples,
        },
        "failures": {
            "count": failure_count,
            "sampled": len(failure_samples),
            "omitted": failure_count - len(failure_samples),
            "samples": failure_samples,
        },
    }


def write_benchmark_summary(summary: Mapping[str, Any], path: str | Path, *, pretty: bool = False) -> Path:
    """Atomically write a benchmark summary, returning the resolved path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(summary, temporary, indent=2 if pretty else None, separators=None if pretty else (",", ":"))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def _verify_corpus(manifest: CorpusManifest) -> str:
    corpus_digest = hashlib.sha256()
    for frame in manifest.frames:
        file_digest = hashlib.sha256()
        corpus_digest.update(frame.relative_path.encode("utf-8"))
        corpus_digest.update(b"\0")
        try:
            with frame.path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    file_digest.update(chunk)
                    corpus_digest.update(chunk)
        except OSError as exc:
            raise CorpusManifestError(f"Could not read corpus frame {frame.relative_path}: {exc}") from exc
        actual_digest = file_digest.hexdigest()
        if frame.sha256 is not None and actual_digest != frame.sha256:
            raise CorpusManifestError(
                f"SHA-256 mismatch for {frame.relative_path}: expected {frame.sha256}, got {actual_digest}"
            )
        corpus_digest.update(b"\0")
    return corpus_digest.hexdigest()


def _decode_image(frame: CorpusFrame) -> np.ndarray:
    image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
    if image is None:
        raise CorpusManifestError(f"Could not decode corpus frame {frame.relative_path}")
    return image


def _timing_summary(samples: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    p50, p95, p99 = np.percentile(values, [50.0, 95.0, 99.0])
    return {
        "samples": len(samples),
        "min_ms": float(np.min(values)),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(np.max(values)),
    }


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusManifestError(f"{context} must be a non-empty string")
    return value.strip()


def _reject_unknown(data: Mapping[str, object], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise CorpusManifestError(f"Unknown {context} field(s): {', '.join(unknown)}")


