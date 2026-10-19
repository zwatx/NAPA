"""Pose solvers and coordinate conventions, on exact synthetic projections and real detections."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from conftest import example_profile, project_field_tag

from napa_vision import CameraPoseSolver, NapaDetector, SingleTagPoseSolver, TagAngleSolver
from napa_vision.calibration import undistorted_corner_angles
from napa_vision.geometry import (
    opencv_translation_to_wpilib,
    tag_object_points,
    transform_matrix,
    wpilib_points_to_opencv,
)
from napa_vision.types import FiducialImageObservation


def test_single_tag_pose_returns_ippe_solutions_and_distance() -> None:
    profile = example_profile()
    corners, _ = cv2.projectPoints(
        tag_object_points(profile.tag_size_m),
        np.zeros((3, 1), dtype=np.float64),
        np.array([[0.0], [0.0], [2.0]], dtype=np.float64),
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    observation = FiducialImageObservation(1, corners.reshape(4, 2))
    result = SingleTagPoseSolver().solve(observation, profile.camera, profile.tag_size_m)
    assert result is not None
    assert result.pose_1_tvec is not None
    assert abs(result.distance - 2.0) < 0.05


def test_single_tag_pose_rejects_nonsensical_corners() -> None:
    profile = example_profile()
    corners = np.array([[0.0, 0.0], [1200.0, 20.0], [30.0, 700.0], [1100.0, 650.0]])
    observation = FiducialImageObservation(1, corners)
    assert (
        SingleTagPoseSolver(max_reprojection_error_px=2.0).solve(observation, profile.camera, profile.tag_size_m)
        is None
    )


def test_tag_angle_solver_uses_normalized_coordinates_and_wpilib_translation() -> None:
    profile = example_profile()
    corners, _ = cv2.projectPoints(
        tag_object_points(profile.tag_size_m),
        np.zeros((3, 1), dtype=np.float64),
        np.array([[1.0], [0.0], [1.0]], dtype=np.float64),
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    observation = FiducialImageObservation(1, corners.reshape(4, 2))
    single = SingleTagPoseSolver().solve(observation, profile.camera, profile.tag_size_m)
    assert single is not None
    angles = TagAngleSolver().solve(observation, profile.camera, single)
    assert angles.corner_angles.shape == (4, 2)
    assert angles.center_tx == pytest.approx(math.pi / 4.0, abs=0.01)
    assert angles.camera_to_tag_translation == pytest.approx([1.0, -1.0, 0.0], abs=0.02)


def test_tag_angle_solver_batches_corner_undistortion(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = example_profile()
    corners, _ = cv2.projectPoints(
        tag_object_points(profile.tag_size_m),
        np.zeros((3, 1), dtype=np.float64),
        np.array([[0.0], [0.0], [2.0]], dtype=np.float64),
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    observations = [FiducialImageObservation(tag_id, corners.reshape(4, 2)) for tag_id in (1, 2)]
    singles = [
        SingleTagPoseSolver().solve(observation, profile.camera, profile.tag_size_m) for observation in observations
    ]
    assert all(single is not None for single in singles)
    calls = 0
    original = undistorted_corner_angles

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("napa_vision.solvers.undistorted_corner_angles", counted)

    results = TagAngleSolver().solve_many(observations, profile.camera, singles)  # type: ignore[arg-type]

    assert [result.tag_id for result in results] == [1, 2]
    assert calls == 1


def test_tag_angle_solver_rejects_mismatched_pose_pair() -> None:
    profile = example_profile()
    corners, _ = cv2.projectPoints(
        tag_object_points(profile.tag_size_m),
        np.zeros((3, 1), dtype=np.float64),
        np.array([[0.0], [0.0], [2.0]], dtype=np.float64),
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    observation = FiducialImageObservation(1, corners.reshape(4, 2))
    single = SingleTagPoseSolver().solve(observation, profile.camera, profile.tag_size_m)
    assert single is not None
    single.tag_id = 2

    with pytest.raises(ValueError, match="paired single_tag_pose tag_id"):
        TagAngleSolver().solve_many([observation], profile.camera, [single])


def test_calibration_angle_helper_matches_centered_tag() -> None:
    profile = example_profile()
    corners = np.array([[600.0, 320.0], [680.0, 320.0], [680.0, 400.0], [600.0, 400.0]])
    angles = undistorted_corner_angles(
        corners,
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    assert angles.shape == (4, 2)
    assert float(np.mean(angles[:, 0])) == pytest.approx(0.0, abs=1e-12)
    assert float(np.mean(angles[:, 1])) == pytest.approx(0.0, abs=1e-12)


def test_coordinate_basis_conversion() -> None:
    point = np.array([1.0, 2.0, 3.0])
    opencv = wpilib_points_to_opencv(point)[0]
    assert opencv == pytest.approx([-2.0, -3.0, 1.0])
    assert opencv_translation_to_wpilib(opencv) == pytest.approx(point)


def test_single_known_tag_preserves_two_global_camera_poses() -> None:
    profile = example_profile()
    expected = transform_matrix([4.0, 0.0, 1.0], [0.0, 0.0, math.pi])
    layout = {1: transform_matrix([2.0, 0.0, 1.0], [0.0, 0.0, math.pi])}
    corners = project_field_tag(
        expected,
        layout[1],
        profile.tag_size_m,
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    observation = FiducialImageObservation(1, corners)
    single = SingleTagPoseSolver().solve(observation, profile.camera, profile.tag_size_m)
    assert single is not None
    result = CameraPoseSolver().solve([observation], [single], profile.camera, profile.tag_size_m, layout)
    assert result is not None
    assert result.field_to_camera_1 is not None
    errors = [
        np.linalg.norm(result.field_to_camera_0[:3, 3] - expected[:3, 3]),
        np.linalg.norm(result.field_to_camera_1[:3, 3] - expected[:3, 3]),
    ]
    assert min(errors) < 0.05


def test_multi_tag_solver_recovers_wpilib_field_to_camera_pose() -> None:
    profile = example_profile()
    expected = transform_matrix([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    layout = {
        1: transform_matrix([2.0, -0.5, 1.0], [0.0, 0.0, math.pi]),
        2: transform_matrix([2.0, 0.5, 1.0], [0.0, 0.0, math.pi]),
    }
    observations = [
        FiducialImageObservation(
            tag_id,
            project_field_tag(
                expected,
                field_to_tag,
                profile.tag_size_m,
                profile.camera.calibration_matrix,
                profile.camera.distortion_coefficients,
            ),
        )
        for tag_id, field_to_tag in layout.items()
    ]
    result = CameraPoseSolver().solve(observations, [], profile.camera, profile.tag_size_m, layout)
    assert result is not None
    assert result.field_to_camera_1 is None
    assert set(result.tag_ids) == {1, 2}
    assert np.linalg.norm(result.field_to_camera_0[:3, 3] - expected[:3, 3]) < 0.05
    assert result.error_0 < 0.1


def test_official_artwork_orientation_recovers_real_detector_multi_tag_pose() -> None:
    profile = example_profile()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = np.full((720, 1280), 255, dtype=np.uint8)
    for tag_id, left in ((1, 783), (2, 423)):
        generated = cv2.aruco.generateImageMarker(dictionary, tag_id, 74, borderBits=1)
        official_orientation = cv2.rotate(generated, cv2.ROTATE_180)
        image[323:397, left : left + 74] = official_orientation

    observations = NapaDetector().detect(image)
    layout = {
        1: transform_matrix([2.0, -0.4, 1.0], [0.0, 0.0, math.pi]),
        2: transform_matrix([2.0, 0.4, 1.0], [0.0, 0.0, math.pi]),
    }
    result = CameraPoseSolver().solve(observations, [], profile.camera, profile.tag_size_m, layout)
    assert result is not None
    assert sorted(result.tag_ids) == [1, 2]
    assert result.field_to_camera_0[:3, 3] == pytest.approx([0.0, 0.0, 1.0], abs=0.005)
    assert result.error_0 < 1.0


def test_single_tag_camera_pose_keeps_both_ippe_candidates_and_suppresses_a_bad_one() -> None:
    profile = example_profile()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    generated = cv2.aruco.generateImageMarker(dictionary, 1, 74, borderBits=1)
    image = np.full((720, 1280), 255, dtype=np.uint8)
    image[323:397, 603:677] = cv2.rotate(generated, cv2.ROTATE_180)
    observation = NapaDetector().detect(image)[0]
    single = SingleTagPoseSolver().solve(observation, profile.camera, profile.tag_size_m)
    assert single is not None and single.pose_1_rvec is not None
    layout = {1: transform_matrix([2.0, 0.0, 1.0], [0.0, 0.0, math.pi])}

    # A fronto-parallel tag is ambiguous: when both candidates fit, both are reported. (Which one
    # fits better is decided by sub-pixel noise here, so the errors are set explicitly.)
    single.error_1 = 0.5
    result = CameraPoseSolver().solve([observation], [single], profile.camera, profile.tag_size_m, layout)
    assert result is not None and result.field_to_camera_1 is not None
    assert result.field_to_camera_0[:3, 3] == pytest.approx([0.0, 0.0, 1.0], abs=0.02)

    # A candidate beyond the reprojection limit suppresses the single-tag camera pose.
    single.error_1 = profile.max_reprojection_error_px + 1.0
    assert CameraPoseSolver().solve([observation], [single], profile.camera, profile.tag_size_m, layout) is None
    angle = TagAngleSolver().solve(observation, profile.camera, single)
    assert angle.tag_id == 1
