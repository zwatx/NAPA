from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import napa_vision.simulation_artifacts as simulation_artifacts_module
import napa_vision.simulation_cli as simulation_cli_module
from napa_vision.benchmark import load_corpus_manifest
from napa_vision.config import load_profile
from napa_vision.fiducials import OpenCVArucoFiducialDetector
from napa_vision.geometry import transform_matrix
from napa_vision.simulation import SimulationFrameSpec
from napa_vision.simulation_artifacts import write_simulation_corpus
from napa_vision.simulation_benchmark import SimulationRun, run_simulation
from napa_vision.simulation_cli import main
from napa_vision.simulation_metrics import SimulationThresholds, evaluate_simulation_gate
from napa_vision.simulation_renderer import SyntheticAprilTagRenderer
from napa_vision.simulation_scenarios import standard_simulation_frames, synthetic_wall_layout


@pytest.fixture(scope="module")
def standard_run() -> SimulationRun:
    """Shared test-only result; every consumer below intentionally treats it as read-only."""

    return run_simulation(load_profile("laptop"), repetitions=1, warmup_passes=0)


def test_renderer_is_deterministic_and_changes_with_seed() -> None:
    profile = load_profile("laptop")
    layout = synthetic_wall_layout()
    spec = standard_simulation_frames()[3]

    first = SyntheticAprilTagRenderer(profile.camera, profile.tag_size_m, layout, seed=11).render(spec)
    repeated = SyntheticAprilTagRenderer(profile.camera, profile.tag_size_m, layout, seed=11).render(spec)
    changed = SyntheticAprilTagRenderer(profile.camera, profile.tag_size_m, layout, seed=12).render(spec)

    assert np.array_equal(first.image, repeated.image)
    assert not np.array_equal(first.image, changed.image)
    assert first.truth.to_dict() == repeated.truth.to_dict()


def test_renderer_is_isolated_from_caller_layout_mutation() -> None:
    profile = load_profile("laptop")
    layout = synthetic_wall_layout()
    renderer = SyntheticAprilTagRenderer(profile.camera, profile.tag_size_m, layout, seed=11)
    spec = standard_simulation_frames()[0]
    before = renderer.render(spec)

    layout[1][:] = np.eye(4)
    after = renderer.render(spec)

    assert np.array_equal(before.image, after.image)
    assert before.truth.to_dict() == after.truth.to_dict()


def test_renderer_corner_order_matches_hard_coded_golden_projection() -> None:
    profile = load_profile("laptop")
    renderer = SyntheticAprilTagRenderer(profile.camera, profile.tag_size_m, synthetic_wall_layout(), supersample=1)
    frame = renderer.render(standard_simulation_frames()[0])
    observations = {
        observation.tag_id: observation for observation in OpenCVArucoFiducialDetector().detect(frame.image)
    }

    expected_tag_one_corners = np.array(
        [[893.5506, 484.4730], [850.9460, 484.4730], [850.6612, 441.7752], [893.2083, 441.7752]]
    )
    assert set(observations) == {1, 2, 3, 4}
    assert observations[1].corners == pytest.approx(expected_tag_one_corners, abs=1.0)


def test_renderer_culls_back_facing_tag() -> None:
    profile = load_profile("laptop")
    layout = {1: transform_matrix([4.0, 1.5, 1.0], [0.0, 0.0, 0.0])}
    renderer = SyntheticAprilTagRenderer(profile.camera, profile.tag_size_m, layout)
    spec = SimulationFrameSpec(
        0,
        "back-facing",
        transform_matrix([0.5, 1.5, 1.0], [0.0, 0.0, 0.0]),
        (1,),
        expect_pose=False,
    )

    frame = renderer.render(spec)

    assert frame.truth.rendered_tag_ids == ()
    assert OpenCVArucoFiducialDetector().detect(frame.image) == []


def test_standard_simulation_exercises_detection_pose_and_outlier_rejection(standard_run: SimulationRun) -> None:
    gate = evaluate_simulation_gate(standard_run.summary, SimulationThresholds())
    scenarios = {scenario["name"]: scenario for scenario in standard_run.summary["scenarios"]}

    assert gate["passed"] is True
    assert standard_run.summary["accuracy"]["recall"] == 1.0
    assert standard_run.summary["accuracy"]["precision"] == 1.0
    assert standard_run.summary["pose"]["solve_rate"] == 1.0
    assert scenarios["layout-outlier"]["expected_rejection_failures"] == 0
    assert scenarios["single-tag"]["pose_solve_rate"] is None
    assert standard_run.summary["evidence_kind"] == "synthetic"
    assert standard_run.summary["limitations"]


def test_simulation_run_constructor_remains_backward_compatible_and_corpus_reuse_is_validated(
    standard_run: SimulationRun,
) -> None:
    legacy_run = SimulationRun(standard_run.frames, standard_run.tag_layout, standard_run.summary)
    assert legacy_run.corpus is None

    with pytest.raises(ValueError, match="reused simulation corpus does not match"):
        run_simulation(
            load_profile("laptop"),
            seed=2027,
            repetitions=1,
            warmup_passes=0,
            corpus=standard_run.corpus,
        )


def test_simulation_corpus_is_replay_compatible(tmp_path: Path, standard_run: SimulationRun) -> None:
    profile = load_profile("laptop")

    artifacts = write_simulation_corpus(
        standard_run.frames,
        tmp_path,
        profile,
        standard_run.tag_layout,
        suite="standard",
    )
    manifest = load_corpus_manifest(artifacts["manifest"])
    replay_profile = load_profile(artifacts["profile"])

    assert len(manifest.frames) == 9
    assert all(frame.sha256 for frame in manifest.frames)
    assert replay_profile.name == "synthetic-replay"
    assert set(replay_profile.tag_layout) == {1, 2, 3, 4}
    truth = json.loads(artifacts["truth"].read_text(encoding="utf-8"))
    assert truth["evidence_kind"] == "synthetic"


def test_simulation_artifact_bytes_are_durable_by_default_with_explicit_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr(simulation_artifacts_module.os, "fsync", fsync_calls.append)

    simulation_artifacts_module._atomic_write_bytes(tmp_path / "durable.bin", b"durable")
    assert len(fsync_calls) == 1

    simulation_artifacts_module._atomic_write_bytes(tmp_path / "atomic-only.bin", b"atomic", durable=False)
    assert len(fsync_calls) == 1


def test_simulation_cli_writes_report_and_returns_failed_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--repetitions",
            "1",
            "--warmup-passes",
            "0",
            "--max-translation-error-m",
            "0.001",
        ]
    )

    assert exit_code == 1
    report = json.loads((tmp_path / "simulation-report.json").read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is False
    assert str(tmp_path / "simulation-report.json") == report["artifacts"]["report"]
    assert json.loads(capsys.readouterr().out)["gate"]["passed"] is False


def test_simulation_cli_compares_detector_candidates_without_changing_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--output-dir",
            str(tmp_path),
            "--repetitions",
            "1",
            "--warmup-passes",
            "0",
            "--compare-optimizations",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    comparisons = report["optimization_comparisons"]
    assert {"aruco3", "layout-restricted"} <= comparisons.keys()
    assert comparisons["aruco3"]["assessment"]["accuracy_equal"] is True
    assert comparisons["layout-restricted"]["assessment"]["accuracy_equal"] is True


def test_simulation_comparison_handles_exact_zero_metrics() -> None:
    baseline = {
        "timing": {"processing_ms": {"p50": 1.0, "p95": 2.0}},
        "accuracy": {"recall": 1.0},
        "pose": {
            "solved": 1,
            "translation_error_m": {"max": 0.0},
            "rotation_error_deg": {"max": 0.0},
        },
    }
    candidate = {
        "timing": {"processing_ms": {"p50": 0.0, "p95": 0.0}},
        "accuracy": {"recall": 1.0},
        "pose": {
            "solved": 1,
            "translation_error_m": {"max": 0.0},
            "rotation_error_deg": {"max": 0.0},
        },
        "gate": {"passed": True},
    }

    assessment = simulation_cli_module._comparison(baseline, candidate)

    assert assessment["processing_p50_speedup"] is None
    assert assessment["processing_p95_speedup"] is None
    assert assessment["max_translation_error_improvement"] is None
    assert assessment["max_rotation_error_improvement"] is None
    assert assessment["eligible_for_hardware_validation"] is False
