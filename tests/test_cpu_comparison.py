from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmarks.cpu_backends as cb  # noqa: E402
import benchmarks.cpu_comparison as cc  # noqa: E402
import benchmarks.performance_matrix as matrix  # noqa: E402

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def test_scoring_separates_false_ids_duplicates_geometry_and_order() -> None:
    truth = [matrix.TagTruth(1, SQUARE), matrix.TagTruth(2, SQUARE + 20)]
    reflected = np.roll(SQUARE[::-1], 1, axis=0) + 0.3  # same square, another corner order, shifted 0.3 px
    score = cc.score_detail(truth, [(1, reflected), (1, SQUARE), (99, SQUARE)])

    assert (score["true_positive"], score["false_id"], score["duplicate"], score["false_negative"]) == (1, 1, 1, 1)
    match = score["matched"][1]
    assert match["worst_px"] == pytest.approx(0.3 * np.sqrt(2))
    assert match["alignment"] == "rs1" and match["offset"] == pytest.approx([0.3, 0.3])
    assert cc.score_detail(truth, [(2, SQUARE + 20)])["matched"][2]["alignment"] == "s0"


def test_stress_frames_are_reproducible_and_keep_their_bands() -> None:
    tiny = [cc.render_stress("stress-tiny", i, (640, 480)) for i in range(4)]
    assert all(12 * 0.95 < t.side_px < 31 * 1.02 for f in tiny for t in f.tags)
    assert np.array_equal(tiny[2].image, cc.render_stress("stress-tiny", 2, (640, 480)).image)
    assert len(cc.render_stress("stress-dense", 0, (1280, 720)).tags) >= 20
    assert cc.render_stress("stress-clutter-noisy", 0, (640, 480)).tags == []
    assert cc.render_stress("stress-heavy-noise", 0, (640, 480)).details["noise_sigma"] >= 15


def test_validation_truth_follows_a_whole_frame_rotation() -> None:
    frames = cc.validation_frames()
    turned = next(f for f in frames if f.details.get("rot90"))
    found = cb.napa_backend("napa").factory()(turned.image)
    score = cc.score_detail(turned.tags, found)
    assert score["true_positive"] == 1 and score["matched"][586]["alignment"] == "s0"
    assert score["matched"][586]["worst_px"] < 0.2


def test_frame_ratios_compare_the_same_frame() -> None:
    rows = [
        {"frame": "a", "backend": "napa", "scenario": "one-tag", "resolution": "640x480", "median_ms": 1.0},
        {"frame": "a", "backend": "other", "scenario": "one-tag", "resolution": "640x480", "median_ms": 4.0},
        {"frame": "b", "backend": "napa", "scenario": "one-tag", "resolution": "640x480", "median_ms": 2.0},
        {"frame": "b", "backend": "other", "scenario": "one-tag", "resolution": "640x480", "median_ms": 1.0},
    ]
    (entry,) = cc.frame_ratios(rows, "napa", lambda row: "g")
    assert (entry["min"], entry["max"], entry["frames_baseline_slower"]) == (0.5, 4.0, 1)


@pytest.mark.parametrize("name", list(cb.TUNING))
def test_adapter_conventions_on_known_geometry(name: str) -> None:
    """Every backend that can be built: right IDs, pixel origin within 0.15 px, one corner order, no blanks.

    Recall is the benchmark's job, not this test's: RapidTag 0.1.12 and OpenCV 4.7 miss the noise-free,
    near-axis-aligned 120 px tags here (they find them once sensor noise is added), so only most must be found.
    """

    backends, skipped = cb.build_backends([name], threads=1)
    if not backends:
        pytest.skip(skipped[name])
    entry = cc.validate(backends)[name]
    tagged = [case for case in entry["cases"] if case["expected"]]
    found = sum(case["true_positive"] for case in tagged)
    assert found > len(tagged) // 2
    assert entry["false_ids"] == entry["duplicates"] == entry["blank_detections"] == 0
    assert np.all(np.abs(entry["mean_offset_px"]) < 0.15)
    assert entry["order_consistent"]


def test_resource_summary_keeps_skipped_backends_apart_from_measured_ones() -> None:
    run = {
        "backend": "napa",
        "round": 0,
        "skipped": {},
        "config": {},
        "mode": "serial",
        "workers": 1,
        "threads_per_detector": 1,
        "cpus": [2],
        "wall_ms_per_frame": 1.0,
        "cpu_ms_per_frame": 1.0,
        "frames_per_s": 1000.0,
        "mean_busy_cpus": 1.0,
        "latency_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
        "setup": {"wall_ms": 1.0},
        "frame_median_ms": [1.0],
        "threads_end": [{"cpus": [2]}],
        "rss_kib": {"before_backend": 0, "after_construct": 1024, "after_warmup": 2048, "end": 2048, "peak": 2048},
    }
    summary = cc.resources_summary({"results": [run, {"backend": "locus", "round": 0, "skip_reason": "no locus"}]})
    measured, skipped = summary
    assert measured["wall_ms_per_frame"] == 1.0 and measured["rss_peak_over_constructed_mib"] == 1.0
    assert skipped == {"backend": "locus", "failures": [], "skip_reason": ["no locus"]}
    assert "skipped: no locus" in cc.resources_markdown(summary, "t")
