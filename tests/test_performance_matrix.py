from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.performance_matrix import build_corpus, score_ids, summarize_latency  # noqa: E402


def test_corpus_keeps_occluded_tag_in_ground_truth_and_is_deterministic() -> None:
    first = build_corpus(resolutions=((640, 480),), tag_side_pixels=(96,), seed=7)
    second = build_corpus(resolutions=((640, 480),), tag_side_pixels=(96,), seed=7)

    assert [frame.name for frame in first] == [frame.name for frame in second]
    assert all(np.array_equal(a.image, b.image) for a, b in zip(first, second, strict=True))
    assert any(frame.scenario == "blank" and frame.expected_ids == () for frame in first)
    assert any(frame.scenario == "textured-negative" and frame.expected_ids == () for frame in first)
    assert all(
        len(frame.expected_ids) == len(set(frame.expected_ids)) for frame in first if frame.scenario == "four-tag"
    )
    occluded = [frame for frame in first if frame.scenario == "occlusion"]
    assert occluded
    assert all(len(frame.expected_ids) == 1 for frame in occluded)
    assert {frame.rotation for frame in first if frame.scenario == "clean-one-tag"} == {0, 1, 2, 3}


def test_occlusion_area_records_geometric_raster_and_changed_pixel_fractions() -> None:
    frames = build_corpus(resolutions=((640, 480),), tag_side_pixels=(160,), seed=11)
    occluded = next(
        frame
        for frame in frames
        if frame.scenario == "occlusion"
        and frame.occlusion is not None
        and frame.occlusion["region"] == "border-strip"
        and frame.occlusion["requested_fraction"] == 0.20
        and frame.occlusion["mask_value"] == 127
    )

    assert occluded.occlusion is not None
    measured = occluded.occlusion
    assert measured["geometric_fraction"] == measured["raster_fraction"]
    assert 0.18 <= float(measured["raster_fraction"]) <= 0.22
    assert 0.0 <= float(measured["changed_pixel_fraction"]) <= float(measured["raster_fraction"])


def test_id_metrics_count_exact_ids_and_duplicates() -> None:
    metrics = score_ids((1, 42), (1, 1, 314))

    assert metrics.true_positive == 1
    assert metrics.false_positive == 2
    assert metrics.false_negative == 1
    assert metrics.precision == 1 / 3
    assert metrics.recall == 1 / 2


def test_latency_summary_schema_uses_latency_derived_fps() -> None:
    summary = summarize_latency((1_000_000, 2_000_000, 3_000_000))

    assert set(summary) == {
        "samples",
        "min_ms",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "latency_derived_p50_fps",
    }
    assert summary["samples"] == 3
    assert summary["p50_ms"] == 2.0
    assert summary["latency_derived_p50_fps"] == 500.0


def test_four_tag_scene_has_four_unique_ground_truth_instances() -> None:
    from napa_vision import OpenCVArucoFiducialDetector

    frames = build_corpus(resolutions=((640, 480),), tag_side_pixels=(96,))
    frame = next(frame for frame in frames if frame.scenario == "four-tag")
    observed = OpenCVArucoFiducialDetector().detect(frame.image)
    assert len(frame.expected_ids) == 4
    assert sorted(tag.tag_id for tag in observed) == sorted(frame.expected_ids)


def test_occlusion_sweep_covers_same_ids_and_rotations_at_each_fraction() -> None:
    from benchmarks.performance_matrix import OCCLUSION_FRACTIONS, TAG_IDS

    frames = build_corpus(resolutions=((640, 480), (1280, 720)), tag_side_pixels=(96,))
    for fraction in OCCLUSION_FRACTIONS:
        selected = [
            frame
            for frame in frames
            if frame.occlusion is not None
            and frame.occlusion["requested_fraction"] == fraction
            and frame.occlusion["region"] == "payload-center"
            and frame.occlusion["mask_value"] == 127
        ]
        assert {(frame.expected_ids[0], frame.rotation) for frame in selected} == {
            (tag_id, rotation) for tag_id in TAG_IDS for rotation in range(4)
        }
        assert all(frame.resolution == (640, 480) for frame in selected)


def test_matrix_scores_hidden_tag_misses_and_separate_repetition_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    import numpy as np

    import benchmarks.performance_matrix as matrix

    image = np.zeros((20, 20), dtype=np.uint8)
    frames = [
        matrix.BenchmarkFrame("clean", image, (1,), (20, 20), "clean-one-tag"),
        matrix.BenchmarkFrame(
            "hidden",
            image,
            (42,),
            (20, 20),
            "occlusion",
            tag_side_px=16,
            occlusion={"region": "corner", "requested_fraction": 0.2, "mask_value": 127},
        ),
    ]
    monkeypatch.setattr(
        matrix, "build_backends", lambda **kwargs: ([matrix.Backend("fake", lambda: lambda image: [1], {})], {})
    )
    monkeypatch.setattr(matrix, "run_decoder_microbenchmark", lambda: {"skipped": "not a timing test"})
    monkeypatch.setattr(matrix, "_metadata", lambda: {})
    previous_threads = cv2.getNumThreads()
    try:
        report, rows = matrix.run_matrix(frames, repetitions=3, occlusion_repetitions=2, warmup=0, threads=1)
    finally:
        cv2.setNumThreads(previous_threads)
    assert len(rows) == 5
    assert report["backends"]["fake"]["id_metrics"] == {
        "true_positive": 3,
        "false_positive": 2,
        "false_negative": 2,
        "precision": 3 / 5,
        "recall": 3 / 5,
    }
    assert report["run_config"]["opencv_threads_effective"] == 1
    assert report["occlusion"][0]["id_metrics"]["false_negative"] == 2


def test_occlusion_report_exposes_actual_coverage_and_changed_pixel_ranges() -> None:
    from benchmarks.performance_matrix import IdMetrics, _occlusion_report

    key = ("fake", "640x480", "payload-center", 0.2, "255", 160)
    samples = [
        {
            "stage": "occlusion",
            "backend": "fake",
            "resolution": "640x480",
            "tag_side_px": 160,
            "occlusion_region": "payload-center",
            "occlusion_requested_fraction": 0.2,
            "occlusion_mask_value": 255,
            "occlusion_geometric_fraction": 0.2025,
            "occlusion_raster_fraction": 0.2025,
            "occlusion_changed_pixel_fraction": changed,
        }
        for changed in (0.05, 0.15)
    ]
    report = _occlusion_report({key: IdMetrics(1, 0, 1)}, samples)
    fractions = report[0]["area_fractions"]
    assert fractions["raster_fraction"] == {"min": 0.2025, "mean": 0.2025, "max": 0.2025}
    assert fractions["changed_pixel_fraction"] == {"min": 0.05, "mean": 0.1, "max": 0.15}
