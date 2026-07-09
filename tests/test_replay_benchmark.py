from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from northstar_vision.benchmark import (
    BenchmarkOptions,
    BenchmarkThresholds,
    CorpusManifestError,
    evaluate_benchmark_gate,
    load_corpus_manifest,
    main,
    run_replay_benchmark,
    write_benchmark_summary,
)


def _marker_scene(tag_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 160, borderBits=1)
    image = np.full((320, 320), 255, dtype=np.uint8)
    image[80:240, 80:240] = marker
    return image


def _write_manifest(tmp_path: Path, frames: list[dict[str, object]], **extra: object) -> Path:
    manifest = {
        "schema_version": 1,
        "name": "small regression corpus",
        "description": "known positive, negative, and mismatch cases",
        "frames": frames,
        **extra,
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_replay_benchmark_reports_accuracy_percentiles_and_bounded_failures(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "tag1.png"), _marker_scene(1))
    cv2.imwrite(str(tmp_path / "missing.png"), np.full((320, 320), 255, dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "unexpected.png"), _marker_scene(2))
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"path": "tag1.png", "expected_tag_ids": [1]},
            {"path": "missing.png", "expected_tag_ids": [3]},
            {"path": "unexpected.png", "expected_tag_ids": []},
        ],
    )

    summary = run_replay_benchmark(
        manifest_path,
        options=BenchmarkOptions(warmup_passes=0, max_failure_samples=1),
    )

    assert summary["accuracy"] == {
        "expected_instances": 2,
        "detected_instances": 2,
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "recall": 0.5,
        "precision": 0.5,
        "false_positive_frames": 1,
        "false_positives_per_frame": 1 / 3,
    }
    assert summary["failures"]["count"] == 2
    assert summary["failures"]["sampled"] == 1
    assert summary["failures"]["omitted"] == 1
    assert summary["memory"]["per_frame_results_retained"] is False
    assert summary["corpus"]["corpus_sha256"]
    for stage in ("load_ms", "detect_ms", "score_ms", "total_ms"):
        timing = summary["timing"][stage]
        assert timing["samples"] == 3
        assert 0 <= timing["p50_ms"] <= timing["p95_ms"] <= timing["p99_ms"] <= timing["max_ms"]
    json.dumps(summary)


def test_manifest_integrity_hash_is_enforced(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((20, 20), dtype=np.uint8))
    actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest_path = _write_manifest(
        tmp_path,
        [{"path": "frame.png", "expected_tag_ids": [], "sha256": actual_hash}],
    )
    manifest = load_corpus_manifest(manifest_path)
    assert manifest.frames[0].sha256 == actual_hash

    image_path.write_bytes(b"changed")
    with pytest.raises(CorpusManifestError, match="SHA-256 mismatch"):
        run_replay_benchmark(manifest, options=BenchmarkOptions(warmup_passes=0))


@pytest.mark.parametrize(
    ("frame", "extra", "message"),
    [
        ({"path": "../outside.png", "expected_tag_ids": []}, {}, "must not escape"),
        ({"path": "frame.png", "expected_tag_ids": [1, 1]}, {}, "must not contain duplicates"),
        ({"path": "frame.png", "expected_tag_ids": [True]}, {}, "non-negative integer"),
        ({"path": "frame.png", "expected_tag_ids": [], "unknown": 1}, {}, "Unknown frames"),
        ({"path": "frame.png", "expected_tag_ids": []}, {"unknown": 1}, "Unknown manifest"),
    ],
)
def test_manifest_validation_is_strict(
    tmp_path: Path,
    frame: dict[str, object],
    extra: dict[str, object],
    message: str,
) -> None:
    path = _write_manifest(tmp_path, [frame], **extra)
    with pytest.raises(CorpusManifestError, match=message):
        load_corpus_manifest(path)


def test_repository_example_manifest_stays_schema_valid() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    manifest = load_corpus_manifest(repository_root / "benchmarks" / "corpus.example.json")
    assert manifest.name == "robot-camera-regression"
    assert manifest.frames[0].expected_tag_ids == (1,)


