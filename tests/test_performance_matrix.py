from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmarks.performance_matrix as matrix  # noqa: E402
from napa_vision import NapaDetector  # noqa: E402


def test_frames_are_distinct_and_reproducible_one_at_a_time() -> None:
    frames = list(
        matrix.iter_frames(scenarios=("one-tag", "many-tags"), resolutions=((640, 480),), frames_per_scenario=4)
    )
    again = matrix.render_frame("many-tags", 2, (640, 480))

    assert len(frames) == 8
    assert len({frame.name for frame in frames}) == 8
    assert np.array_equal(frames[6].image, again.image)
    assert [tag.tag_id for tag in frames[6].tags] == [tag.tag_id for tag in again.tags]
    # Distinct frames: no two images (or ID sets) repeat.
    assert len({frame.image.tobytes() for frame in frames}) == 8
    assert len({tuple(tag.tag_id for tag in frame.tags) for frame in frames}) == 8


def test_every_scenario_renders_and_negatives_have_no_tags() -> None:
    negatives = {"empty", "clutter-only", "decoys"}
    for name in matrix.SCENARIO_NAMES:
        frame = matrix.render_frame(name, 0, (640, 480))
        assert frame.image.dtype == np.uint8 and frame.image.shape == (480, 640)
        assert (frame.tags == []) == (name in negatives), name
        ids = [tag.tag_id for tag in frame.tags]
        assert len(ids) == len(set(ids))
        for tag in frame.tags:
            # The whole tag, quiet zone included, stays inside the frame.
            assert tag.corners.min() >= 0 and tag.corners[:, 0].max() < 640 and tag.corners[:, 1].max() < 480


def test_small_tags_stay_in_their_size_band() -> None:
    sides = [tag.side_px for i in range(6) for tag in matrix.render_frame("small-tags", i, (1280, 720)).tags]
    assert sides and min(sides) > 30.5 and max(sides) < 54.5


def test_rendered_ground_truth_matches_the_detected_corners() -> None:
    image = np.full((240, 320), 200.0, np.float32)
    corners = matrix.tag_pose(np.random.default_rng(5), 90.0, (160.0, 120.0), max_tilt_deg=25)
    matrix.render_tag(image, 77, corners)
    found = NapaDetector().detect(np.clip(image, 0, 255).astype(np.uint8))

    assert [observation.tag_id for observation in found] == [77]
    assert np.linalg.norm(found[0].corners - corners, axis=1).max() < 0.15


def test_occlusion_patches_cover_the_requested_fraction() -> None:
    for region in matrix.OCCLUSION_REGIONS:
        for fraction in matrix.OCCLUSION_FRACTIONS:
            box = matrix.Occlusion(*matrix.occlusion_box(fraction, region), value=255)
            assert box.fraction == pytest.approx(fraction)
            assert 0.5 <= box.x0 <= box.x1 <= 8.5 and 0.5 <= box.y0 <= box.y1 <= 8.5
    frames = list(matrix.iter_occlusion_frames(cases=2))
    assert len(frames) == len(matrix.OCCLUSION_REGIONS) * len(matrix.OCCLUSION_FRACTIONS) * 2 * 2
    assert all(len(frame.tags) == 1 for frame in frames)


def test_scoring_counts_ids_once_and_aligns_corners_only_when_asked() -> None:
    truth = [
        matrix.TagTruth(1, np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])),
        matrix.TagTruth(42, np.array([[20.0, 0.0], [30.0, 0.0], [30.0, 10.0], [20.0, 10.0]])),
    ]
    rotated = np.roll(truth[0].corners, 1, axis=0)
    detections = [(1, rotated), (1, truth[0].corners), (314, truth[1].corners)]

    strict = matrix.score_frame(truth, detections, align_corners=False)
    aligned = matrix.score_frame(truth, detections, align_corners=True)

    assert (strict.true_positive, strict.false_positive, strict.false_negative) == (1, 2, 1)
    assert strict.corner_errors[0] == pytest.approx(10.0)
    assert aligned.corner_errors == (0.0,)


def test_run_matrix_times_every_backend_on_every_frame_and_aggregates() -> None:
    frames = [matrix.render_frame("one-tag", index, (640, 480)) for index in range(3)]
    detector = NapaDetector()
    backends = [
        matrix.Backend(
            "napa",
            lambda: lambda image: [(o.tag_id, o.corners) for o in detector.detect(image)],
            {"corner_order": "native"},
        ),
        matrix.Backend("blind", lambda: lambda image: [], {}),
    ]

    lifecycle, rows = matrix.run_matrix(frames, backends=backends, repetitions=3, warmup=0)
    summary = matrix.aggregate(rows)

    assert set(lifecycle) == {"napa", "blind"} and "first_call_ms" in lifecycle["napa"]
    assert len(rows) == 6 and all(len(str(row["samples_ns"]).split(";")) == 3 for row in rows)
    results = {row["backend"]: row for row in summary["results"]}
    assert results["napa"]["found"] == results["napa"]["tags"] == 3
    assert results["blind"]["found"] == 0 and results["blind"]["recall"] == 0.0
    overall = next(row for row in summary["speedups"] if row["scenario"] == "all")
    assert overall["versus"] == "blind" and overall["frames"] == 3
    assert "| one-tag | 3 | 3 |" in matrix.markdown_summary({**summary, "scenarios": ["one-tag"]}, versus="blind")


def test_latency_summary_reports_percentiles() -> None:
    summary = matrix.summarize_latency((1_000_000, 2_000_000, 3_000_000))
    assert summary["samples"] == 3 and summary["p50_ms"] == 2.0 and summary["max_ms"] == 3.0
    with pytest.raises(ValueError):
        matrix.summarize_latency(())
