from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from northstar_vision.benchmark import load_corpus_manifest
from northstar_vision.config import load_profile
from northstar_vision.fiducials import OpenCVArucoFiducialDetector
from northstar_vision.geometry import transform_matrix
from northstar_vision.simulation import SimulationFrameSpec
from northstar_vision.simulation_artifacts import write_simulation_corpus
from northstar_vision.simulation_benchmark import run_simulation
from northstar_vision.simulation_cli import main
from northstar_vision.simulation_metrics import SimulationThresholds, evaluate_simulation_gate
from northstar_vision.simulation_renderer import SyntheticAprilTagRenderer
from northstar_vision.simulation_scenarios import standard_simulation_frames, synthetic_wall_layout


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


def test_standard_simulation_exercises_detection_pose_and_outlier_rejection() -> None:
    run = run_simulation(load_profile("laptop"), repetitions=1, warmup_passes=0)
    gate = evaluate_simulation_gate(run.summary, SimulationThresholds())
    scenarios = {scenario["name"]: scenario for scenario in run.summary["scenarios"]}

    assert gate["passed"] is True
    assert run.summary["accuracy"]["recall"] == 1.0
    assert run.summary["accuracy"]["precision"] == 1.0
    assert run.summary["pose"]["solve_rate"] == 1.0
    assert scenarios["layout-outlier"]["expected_rejection_failures"] == 0
    assert scenarios["single-tag"]["pose_solve_rate"] is None
    assert run.summary["evidence_kind"] == "synthetic"
    assert run.summary["limitations"]


def test_simulation_corpus_is_replay_compatible(tmp_path: Path) -> None:
    profile = load_profile("laptop")
    run = run_simulation(profile, repetitions=1, warmup_passes=0)

    artifacts = write_simulation_corpus(run.frames, tmp_path, profile, run.tag_layout, suite="standard")
    manifest = load_corpus_manifest(artifacts["manifest"])
    replay_profile = load_profile(artifacts["profile"])

    assert len(manifest.frames) == 9
    assert all(frame.sha256 for frame in manifest.frames)
    assert replay_profile.name == "synthetic-replay"
    assert set(replay_profile.tag_layout) == {1, 2, 3, 4}
    truth = json.loads(artifacts["truth"].read_text(encoding="utf-8"))
    assert truth["evidence_kind"] == "synthetic"


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
