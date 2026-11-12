"""Public pose errors and cutoffs use RMS Euclidean corner distance in pixels."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from conftest import example_profile

from napa_vision.geometry import tag_object_points, transform_matrix
from napa_vision.solvers import CameraPoseSolver, SingleTagPoseSolver
from napa_vision.types import FiducialImageObservation


def _noisy_observation(profile: SimpleNamespace) -> FiducialImageObservation:
    projected, _ = cv2.projectPoints(
        tag_object_points(profile.tag_size_m),
        np.array([0.18, -0.24, 0.08]),
        np.array([0.12, -0.07, 2.0]),
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    noise = np.array([[0.8, -0.3], [-0.5, 0.6], [0.2, 0.9], [-0.6, -0.8]])
    return FiducialImageObservation(1, projected.reshape(4, 2) + noise)


def _radial_rms(
    profile: SimpleNamespace, observation: FiducialImageObservation, rvec: np.ndarray, tvec: np.ndarray
) -> float:
    projected, _ = cv2.projectPoints(
        tag_object_points(profile.tag_size_m),
        rvec,
        tvec,
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    delta = projected.reshape(4, 2) - observation.corners
    # Independent Euclidean-distance definition, not OpenCV's returned scalar.
    distances = np.linalg.norm(delta, axis=1)
    return float(np.sqrt(distances @ distances / len(distances)))


def _error_provider(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    original = cv2.solvePnPGeneric

    def solve(*args, **kwargs):
        ok, rotations, translations, errors = original(*args, **kwargs)
        if provider == "missing":
            errors = None
        elif provider == "nonfinite":
            errors = np.full_like(errors, np.nan)
        return ok, rotations, translations, errors

    monkeypatch.setattr(cv2, "solvePnPGeneric", solve)


@pytest.mark.parametrize("provider", ["native", "missing", "nonfinite"])
def test_pose_errors_and_diagnostics_equal_corner_distance_rms(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    profile = example_profile()
    observation = _noisy_observation(profile)
    _error_provider(monkeypatch, provider)
    single = SingleTagPoseSolver(max_reprojection_error_px=100).solve(observation, profile.camera, profile.tag_size_m)
    assert single is not None and single.pose_1_rvec is not None and single.pose_1_tvec is not None
    expected = _radial_rms(profile, observation, single.pose_0_rvec, single.pose_0_tvec)
    assert expected > 0.1
    assert single.error_0 == pytest.approx(expected, rel=1e-10)
    assert single.error_1 == pytest.approx(
        _radial_rms(profile, observation, single.pose_1_rvec, single.pose_1_tvec), rel=1e-10
    )
    layout = {1: transform_matrix([3, 0, 1], [0, 0, np.pi])}
    solver = CameraPoseSolver(max_reprojection_error_px=100)
    result = solver.solve([observation], [single], profile.camera, profile.tag_size_m, layout)
    assert result is not None
    assert result.error_0 == pytest.approx(expected, rel=1e-10)
    assert solver.last_diagnostics.overall_rms_reprojection_error_px == pytest.approx(expected, rel=1e-10)
    assert solver.last_diagnostics.per_tag_rms_reprojection_error_px[1] == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("provider", ["native", "missing", "nonfinite"])
def test_single_pose_cutoff_uses_euclidean_corner_rms(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    profile = example_profile()
    observation = _noisy_observation(profile)
    _error_provider(monkeypatch, provider)
    reference = SingleTagPoseSolver(max_reprojection_error_px=100).solve(
        observation, profile.camera, profile.tag_size_m
    )
    assert reference is not None
    radial = _radial_rms(profile, observation, reference.pose_0_rvec, reference.pose_0_tvec)
    assert radial > 0.1
    solver = SingleTagPoseSolver(max_reprojection_error_px=radial * 0.9)
    assert solver.solve(observation, profile.camera, profile.tag_size_m) is None


@pytest.mark.parametrize("provider", ["native", "missing", "nonfinite"])
def test_camera_pose_cutoff_checks_both_candidates_in_same_units(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    profile = example_profile()
    observation = _noisy_observation(profile)
    _error_provider(monkeypatch, provider)
    single = SingleTagPoseSolver(max_reprojection_error_px=100).solve(observation, profile.camera, profile.tag_size_m)
    assert single is not None and single.pose_1_rvec is not None and single.pose_1_tvec is not None
    worst = max(
        _radial_rms(profile, observation, single.pose_0_rvec, single.pose_0_tvec),
        _radial_rms(profile, observation, single.pose_1_rvec, single.pose_1_tvec),
    )
    solver = CameraPoseSolver(max_reprojection_error_px=worst * 0.9)
    layout = {1: transform_matrix([3, 0, 1], [0, 0, np.pi])}
    assert solver.solve([observation], [single], profile.camera, profile.tag_size_m, layout) is None
    assert solver.last_diagnostics.rejection_reason is not None
    assert "reprojection limit" in solver.last_diagnostics.rejection_reason


def test_generic_candidate_compares_mixed_error_metadata_in_corner_units(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = example_profile()
    square = tag_object_points(profile.tag_size_m)
    points = np.vstack([square, square + [0.3, 0.0, 0.05]])
    rotation = np.zeros((3, 1))
    image, _ = cv2.projectPoints(
        points,
        rotation,
        np.array([0.0, 0.0, 2.0]),
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    better = np.array([[0.001], [0.0], [2.0]])
    worse = np.array([[0.0011], [0.0], [2.0]])
    projected, _ = cv2.projectPoints(
        points, rotation, worse, profile.camera.calibration_matrix, profile.camera.distortion_coefficients
    )
    # The worse candidate has native coordinate RMS; the better one's metadata is missing.
    coordinate_rms = np.linalg.norm(projected - image) / np.sqrt(2 * len(points))
    monkeypatch.setattr(
        cv2,
        "solvePnPGeneric",
        lambda *args, **kwargs: (True, [rotation, rotation], [worse, better], np.array([[coordinate_rms], [np.nan]])),
    )
    candidate = CameraPoseSolver()._generic_candidate(points, image.reshape(-1, 2), profile.camera)
    assert candidate is not None
    np.testing.assert_array_equal(candidate[1], better)
