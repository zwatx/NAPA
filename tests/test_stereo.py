"""Joint stereo geometry, timing, rejection and conditional uncertainty."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from conftest import project_field_tag

from napa_vision.calibration import CameraCalibration
from napa_vision.covariance import _perturb, pose_difference
from napa_vision.geometry import invert_transform, transform_matrix
from napa_vision.provenance import ArtifactProvenance, ProvenanceKind
from napa_vision.stereo import StereoPoseSolver, StereoSolveDiagnostics, _joint_terms
from napa_vision.stereo_calibration import StereoCalibration
from napa_vision.types import FiducialImageObservation

TAG_SIZE = 0.1651


def stereo_scene(*, tags: int = 3, baseline: float = 0.25):
    provenance = ArtifactProvenance(ProvenanceKind.EXAMPLE, "synthetic stereo test")
    left = CameraCalibration(
        np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 360.0], [0.0, 0.0, 1.0]]),
        np.array([-0.08, 0.02, 0.001, -0.001, 0.0]),
        (1280, 720),
        provenance,
    )
    right = CameraCalibration(
        np.array([[880.0, 0.0, 630.0], [0.0, 895.0, 365.0], [0.0, 0.0, 1.0]]),
        np.array([0.03, -0.01, -0.001, 0.001, 0.0]),
        (1280, 720),
        provenance,
    )
    calibration = StereoCalibration(
        left,
        right,
        transform_matrix([0.1, 0.03, 0.02], [0.01, -0.02, 0.03]),
        transform_matrix([0.1, 0.03 - baseline, 0.02], [-0.02, 0.03, -0.04]),
        provenance,
    )
    truth = transform_matrix([0.2, 0.8, 0.9], [0.02, -0.04, 0.08])
    layout = {
        index: transform_matrix([3.5, 0.4 + 0.45 * index, 0.8 + 0.1 * (index % 2)], [0.0, 0.0, math.pi])
        for index in range(tags)
    }
    observations = [
        [
            FiducialImageObservation(
                tag_id,
                project_field_tag(truth @ mount, tag, TAG_SIZE, camera.camera_matrix, camera.distortion_coefficients),
            )
            for tag_id, tag in layout.items()
        ]
        for camera, mount in ((left, calibration.rig_to_left), (right, calibration.rig_to_right))
    ]
    return calibration, layout, truth, observations


def solve(solver, observations, *, left_time=1.0, right_time=1.0, fallback=False):
    return solver.solve(
        *observations, left_capture_time_s=left_time, right_capture_time_s=right_time, allow_monocular_fallback=fallback
    )


@pytest.mark.parametrize("tags", [1, 3])
def test_shared_tags_recover_rig_pose_with_different_intrinsics_and_rotated_mounts(tags):
    calibration, layout, truth, observations = stereo_scene(tags=tags)
    result = solve(StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01), observations)
    assert result.status == "stereo", result.diagnostics
    assert result.pose.transform == pytest.approx(truth, abs=1e-6)
    assert np.linalg.eigvalsh(result.pose.covariance).min() > 0
    assert len(result.diagnostics.accepted_groups) == 2 * tags
    assert result.diagnostics.rejection_reason is None
    assert max(result.diagnostics.per_group_rms_px.values()) < 1e-6


def test_disjoint_tags_still_constrain_one_rig_pose():
    calibration, layout, truth, observations = stereo_scene(tags=3)
    observations = (observations[0][:1], observations[1][1:])
    result = solve(StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01), observations)
    assert result.status == "stereo"
    assert result.pose.transform == pytest.approx(truth, abs=1e-6)
    assert result.diagnostics.accepted_groups == [("left", 0), ("right", 1), ("right", 2)]


def test_joint_jacobian_matches_finite_differences_in_rig_frame():
    calibration, layout, truth, observations = stereo_scene()
    solver = StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01, corner_noise_px=(0.4, 1.1))
    views = [solver._view(side, obs, StereoSolveDiagnostics(0.0)) for side, obs in enumerate(observations)]
    pose = invert_transform(truth)
    _, jacobian, _ = _joint_terms(pose, views)
    numeric = np.empty_like(jacobian)
    for axis in range(6):
        delta = np.zeros(6)
        delta[axis] = 1e-6
        plus = _joint_terms(_perturb(pose, delta), views)[0]
        minus = _joint_terms(_perturb(pose, -delta), views)[0]
        numeric[:, axis] = -(plus - minus) / 2e-6
    np.testing.assert_allclose(jacobian, numeric, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("skew, accepted", [(0.0, True), (0.01, True), (0.010001, False)])
def test_capture_skew_gate_and_output_time(skew, accepted):
    calibration, layout, _, observations = stereo_scene()
    solver = StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01)
    result = solve(solver, observations, left_time=0.0, right_time=skew)
    assert (result.status == "stereo") is accepted
    assert result.timestamp == skew / 2
    if not accepted:
        assert result.pose is None
        assert "skew" in result.diagnostics.rejection_reason
        assert result.diagnostics.iterations == 0


def test_monocular_fallback_is_opt_in_and_uses_available_camera_capture_time():
    calibration, layout, truth, observations = stereo_scene()
    solver = StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01)
    rejected = solve(solver, (observations[0], []), left_time=1.0, right_time=1.1)
    result = solve(solver, (observations[0], []), left_time=1.0, right_time=1.1, fallback=True)
    assert rejected.status == "rejected"
    assert result.status == "monocular"
    assert result.timestamp == 1.0
    assert result.pose.transform == pytest.approx(truth, abs=1e-6)
    assert "skew" in result.diagnostics.fallback_reason
    assert all(camera == "left" for camera, _ in result.diagnostics.accepted_groups)
    assert rejected.diagnostics is not result.diagnostics
    assert rejected.diagnostics.accepted_groups == []


def test_bad_camera_tag_group_can_be_removed_without_removing_other_view_of_tag():
    calibration, layout, truth, observations = stereo_scene(tags=4)
    observations[0][2].corners += [80.0, -60.0]
    solver = StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01, max_reprojection_error_px=2.0)
    result = solve(solver, observations)
    assert result.status == "stereo", result.diagnostics
    assert result.pose.transform == pytest.approx(truth, abs=1e-5)
    assert ("left", 2) not in result.diagnostics.accepted_groups
    assert ("right", 2) in result.diagnostics.accepted_groups
    assert ("left", 2) in result.diagnostics.rejected_group_reasons
    assert result.diagnostics.rejection_reason is None


def test_invalid_unknown_and_duplicate_observations_are_diagnosed_without_counting_twice():
    calibration, layout, _, observations = stereo_scene()
    observations[0] += [
        observations[0][0],
        FiducialImageObservation(99, np.zeros((4, 2))),
        FiducialImageObservation(1, np.full((4, 2), np.nan)),
    ]
    result = solve(StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01), observations)
    assert result.status == "stereo"
    assert len(result.diagnostics.accepted_groups) == 6
    assert "duplicate" in result.diagnostics.rejected_group_reasons[("left", 0)]
    assert "layout" in result.diagnostics.rejected_group_reasons[("left", 99)]
    empty = solve(StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01), ([], []))
    assert empty.status == "rejected" and empty.pose is None


def test_solver_snapshots_layout_and_calibration_and_result_owns_arrays():
    calibration, layout, truth, observations = stereo_scene()
    solver = StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01)
    for transform in layout.values():
        transform[0, 3] += 10
    calibration.rig_to_right.setflags(write=True)
    calibration.rig_to_right[0, 3] = 100
    result = solve(solver, observations)
    for view in observations:
        for obs in view:
            obs.corners[:] = 0
    assert result.status == "stereo"
    assert result.pose.transform == pytest.approx(truth, abs=1e-6)
    assert not result.pose.transform.flags.writeable
    assert not result.pose.covariance.flags.writeable


@pytest.mark.parametrize(
    "keyword, value",
    [
        ("max_capture_skew_s", -0.1),
        ("max_capture_skew_s", True),
        ("max_iterations", 0),
        ("max_iterations", 1.5),
        ("max_outlier_iterations", -1),
        ("corner_noise_px", (1.0, 0.0)),
    ],
)
def test_invalid_configuration_is_rejected(keyword, value):
    calibration, layout, _, _ = stereo_scene()
    kwargs = {"max_capture_skew_s": 0.01, keyword: value}
    with pytest.raises(ValueError):
        StereoPoseSolver(calibration, layout, TAG_SIZE, **kwargs)


@pytest.mark.parametrize("timestamp", [math.nan, math.inf, True, "1.0"])
def test_invalid_capture_times_are_rejected(timestamp):
    calibration, layout, _, observations = stereo_scene()
    solver = StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01)
    with pytest.raises(ValueError, match="finite seconds"):
        solve(solver, observations, left_time=timestamp)


def test_conditional_covariance_scales_with_corner_noise():
    calibration, layout, _, observations = stereo_scene()
    results = [
        solve(
            StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01, corner_noise_px=(noise, noise)),
            observations,
        )
        for noise in (0.5, 1.0)
    ]
    np.testing.assert_allclose(results[1].pose.covariance, 4 * results[0].pose.covariance, rtol=1e-5, atol=1e-10)


def test_joint_covariance_is_consistent_under_independent_pixel_noise():
    calibration, layout, truth, observations = stereo_scene()
    solver = StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01, corner_noise_px=(0.35, 0.35))
    rng = np.random.default_rng(81)
    nees = []
    for _ in range(50):
        noisy = [
            [replace(obs, corners=obs.corners + rng.normal(0.0, 0.35, (4, 2))) for obs in view] for view in observations
        ]
        result = solve(solver, noisy)
        assert result.status == "stereo"
        delta = pose_difference(result.pose.transform, truth)
        nees.append(delta @ np.linalg.solve(result.pose.covariance, delta))
    assert np.mean(np.asarray(nees) > 12.59) < 0.16
    assert 1.0 < np.median(nees) < 9.0


@pytest.mark.parametrize("baseline", [0.0, 1e-5])
def test_weak_baseline_retains_distinct_modes_without_duplicate_seed_weight(baseline):
    calibration, _, _, _ = stereo_scene(tags=1)
    calibration = replace(
        calibration, rig_to_left=np.eye(4), rig_to_right=transform_matrix([0, -baseline, 0], [0, 0, 0])
    )
    layout = {0: transform_matrix([4, 0, 0], [0, 0, np.pi])}
    truth = transform_matrix([0, 0, 0], [0, 0, 0.02])
    rng = np.random.default_rng(10)
    observations = [
        [
            FiducialImageObservation(
                0,
                project_field_tag(
                    truth @ mount, layout[0], TAG_SIZE, camera.camera_matrix, camera.distortion_coefficients
                )
                + rng.normal(0, 0.25, (4, 2)),
            )
        ]
        for camera, mount in (
            (calibration.left, calibration.rig_to_left),
            (calibration.right, calibration.rig_to_right),
        )
    ]
    solver = StereoPoseSolver(calibration, layout, TAG_SIZE, max_capture_skew_s=0.01, corner_noise_px=(0.25, 0.25))
    result = solve(solver, observations)
    assert result.status == "stereo"
    assert len(result.poses) == len(result.diagnostics.candidate_rms_px) == 2
    delta = pose_difference(result.poses[0].transform, result.poses[1].transform)
    assert np.linalg.norm(delta[3:]) > 0.05
    # Seeding the same local minimum twice must not change its mixture weight.
    views = [solver._view(side, obs, StereoSolveDiagnostics(0)) for side, obs in enumerate(observations)]
    fits = solver._fits(views)
    doubled, _ = solver._poses([fits[0], *fits])
    np.testing.assert_allclose(doubled[0].covariance, result.pose.covariance, rtol=1e-8, atol=1e-10)


def test_held_out_epipolar_check_detects_bad_mount_orientation():
    calibration, _, _, observations = stereo_scene()
    left, right = [np.concatenate([obs.corners for obs in view]) for view in observations]
    assert calibration.epipolar_errors(left, right).max() < 1e-5
    shifted = calibration.rig_to_right @ transform_matrix([0, 0, 0], [0.04, 0.02, 0])
    wrong = replace(calibration, rig_to_right=shifted)
    assert np.median(wrong.epipolar_errors(left, right)) > 3
    with pytest.raises(ValueError, match="matching lengths"):
        calibration.epipolar_errors(left, right[:-1])
    coincident = replace(calibration, rig_to_right=calibration.rig_to_left)
    with pytest.raises(ValueError, match="distinct camera centers"):
        coincident.epipolar_errors(left, right)
