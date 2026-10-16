from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import northstar_vision.benchmark as benchmark_module
from northstar_vision.benchmark import (
    BenchmarkOptions,
    BenchmarkThresholds,
    CorpusManifestError,
    evaluate_benchmark_gate,
    load_corpus_manifest,
    main,
    run_replay_benchmark,
    run_replay_detector_comparison,
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


def test_replay_detector_comparison_reuses_one_decoded_frame_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cv2.imwrite(str(tmp_path / "tag9.png"), _marker_scene(9))
    manifest_path = _write_manifest(tmp_path, [{"path": "tag9.png", "expected_tag_ids": [9]}])
    decode_calls = 0
    original_decode = benchmark_module._decode_image

    def counted_decode(frame: benchmark_module.CorpusFrame) -> np.ndarray:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(frame)

    monkeypatch.setattr(benchmark_module, "_decode_image", counted_decode)
    summary = run_replay_detector_comparison(
        manifest_path,
        candidates=("aruco3", "layout-restricted"),
        layout_tag_ids=(2, 9),
        options=BenchmarkOptions(repetitions=2, warmup_passes=1),
    )

    assert decode_calls == 1
    assert summary["run"]["frame_source"] == "decoded-cache"
    assert summary["comparison_run"] == {
        "execution_order": ["baseline", "aruco3", "layout-restricted"],
        "shared_decoded_frames": True,
        "max_decoded_cache_bytes": benchmark_module.DEFAULT_COMPARISON_CACHE_BYTES,
        "production_default_changed": False,
        "requires_physical_hardware_validation": True,
    }
    assert summary["detector_comparisons"]["aruco3"]["assessment"]["accuracy_equal"] is True
    restricted = summary["detector_comparisons"]["layout-restricted"]["results"]
    assert restricted["accuracy"] == summary["accuracy"]
    assert restricted["memory"]["cached_decoded_image_bytes"] > 0


def test_replay_detector_comparison_enforces_decoded_cache_limit(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "tag9.png"), _marker_scene(9))
    manifest_path = _write_manifest(tmp_path, [{"path": "tag9.png", "expected_tag_ids": [9]}])

    with pytest.raises(ValueError, match="decoded comparison cache exceeds"):
        run_replay_detector_comparison(
            manifest_path,
            options=BenchmarkOptions(warmup_passes=0),
            max_decoded_cache_bytes=1,
        )


def test_replay_detector_comparison_handles_zero_candidate_timing() -> None:
    baseline = {"timing": {"detect_ms": {"p50_ms": 1.0, "p95_ms": 2.0}}, "accuracy": {"recall": 1.0}}
    candidate = {"timing": {"detect_ms": {"p50_ms": 0.0, "p95_ms": 0.0}}, "accuracy": {"recall": 1.0}}

    assessment = benchmark_module._detector_comparison(baseline, candidate)

    assert assessment["detect_p50_speedup"] is None
    assert assessment["detect_p95_speedup"] is None
    assert assessment["eligible_for_physical_validation"] is False


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


def test_benchmark_summary_writer_and_console_entrypoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((40, 40), 255, dtype=np.uint8))
    manifest_path = _write_manifest(tmp_path, [{"path": "blank.png", "expected_tag_ids": []}])
    output_path = tmp_path / "reports" / "summary.json"

    exit_code = main(
        [
            str(manifest_path),
            "--warmup-passes",
            "0",
            "--max-failure-samples",
            "0",
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    stdout_summary = json.loads(capsys.readouterr().out)
    file_summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_summary == file_summary
    assert stdout_summary["accuracy"]["false_positives"] == 0

    replacement = {"schema_version": 1, "replacement": True}
    assert write_benchmark_summary(replacement, output_path) == output_path.resolve()
    assert json.loads(output_path.read_text(encoding="utf-8")) == replacement


def test_benchmark_summary_writer_is_durable_by_default_with_explicit_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr(benchmark_module.os, "fsync", fsync_calls.append)

    write_benchmark_summary({"schema_version": 1}, tmp_path / "durable.json")
    assert len(fsync_calls) == 1

    write_benchmark_summary({"schema_version": 1}, tmp_path / "atomic-only.json", durable=False)
    assert len(fsync_calls) == 1


def test_benchmark_options_reject_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="repetitions"):
        BenchmarkOptions(repetitions=0)
    with pytest.raises(ValueError, match="warmup_passes"):
        BenchmarkOptions(warmup_passes=-1)
    with pytest.raises(ValueError, match="max_failure_samples"):
        BenchmarkOptions(max_failure_samples=-1)
    with pytest.raises(ValueError, match="min_recall"):
        BenchmarkThresholds(min_recall=1.1)
    with pytest.raises(ValueError, match="max_p95_total_ms"):
        BenchmarkThresholds(max_p95_total_ms=0.0)


def test_benchmark_gate_reports_failures_and_sets_console_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = {
        "accuracy": {"recall": 0.75, "precision": 0.5},
        "timing": {"total_ms": {"p95_ms": 12.0}},
    }
    gate = evaluate_benchmark_gate(
        summary,
        BenchmarkThresholds(min_recall=0.8, min_precision=0.4, max_p95_total_ms=10.0),
    )
    assert gate["passed"] is False
    assert len(gate["failures"]) == 2

    cv2.imwrite(str(tmp_path / "blank.png"), np.full((40, 40), 255, dtype=np.uint8))
    manifest_path = _write_manifest(tmp_path, [{"path": "blank.png", "expected_tag_ids": [1]}])
    exit_code = main([str(manifest_path), "--warmup-passes", "0", "--min-recall", "1"])
    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["gate"]["passed"] is False
