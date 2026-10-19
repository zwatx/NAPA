"""Pose covariances: exact Jacobians and Monte Carlo consistency."""

from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from conftest import project_field_tag

from napa_vision import (
    CameraPoseSolver,
    FiducialImageObservation,
    Pose2dWithCovariance,
    PoseWithCovariance,
    SingleTagPoseSolver,
    TagAngleSolver,
    VisionProcessor,
    camera_pose_covariance,
)
from napa_vision.covariance import pose_difference
from napa_vision.geometry import (
    opencv_pose_to_wpilib_transform,
    tag_field_corners,
    tag_object_points,
    transform_matrix,
)

TAG_SIZE_M = 0.1651
CAMERA = SimpleNamespace(
    calibration_matrix=np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
    distortion_coefficients=np.array([0.08, -0.15, 0.001, -0.0005, 0.03]),
)
FIELD_TO_CAMERA = transform_matrix([1.0, 2.0, 0.6], [0.02, -0.1, 0.3])
LAYOUT = {
    1: transform_matrix([4.0, 2.2, 0.8], [0.0, 0.0, math.pi]),
    2: transform_matrix([4.0, 3.3, 1.1], [0.0, 0.0, math.pi + 0.3]),
    3: transform_matrix([3.6, 1.0, 0.5], [0.0, 0.0, math.pi - 0.4]),
}


def _project(field_to_camera: np.ndarray, tag_id: int) -> np.ndarray:
    return project_field_tag(
        field_to_camera, LAYOUT[tag_id], TAG_SIZE_M, CAMERA.calibration_matrix, CAMERA.distortion_coefficients
    )


def _perturb(transform: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = transform.copy()
    out[:3, :3] = cv2.Rodrigues(delta[3:])[0] @ transform[:3, :3]
    out[:3, 3] += delta[:3]
    return out


CHI_SQUARED_95 = {2: 5.99, 3: 7.81, 6: 12.59}


def _assert_consistent(samples: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Never overconfident, and not grossly pessimistic.

    The normalized estimation error squared (NEES) averages the dimension when covariances are exact. Napa caps
    correlations (``CORRELATION_SLACK``), which keeps the narrowest combinations of axes conservative, so the mean
    may fall below the dimension; but no more than about 5% of errors may fall outside the 95% ellipse.
    """

    dimension = len(samples[0][0])
    nees = np.array([error @ np.linalg.solve(covariance, error) for error, covariance in samples])
    assert 0.4 * dimension < nees.mean() < 1.15 * dimension
    assert np.mean(nees > CHI_SQUARED_95[dimension]) < 0.09


def _assert_axis_std_devs_match(samples: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Each axis's reported standard deviation matches the actual scatter."""

    errors = np.array([error for error, _ in samples])
    reported = np.sqrt(np.mean([np.diag(covariance) for _, covariance in samples], axis=0))
    assert errors.std(axis=0) / reported == pytest.approx(np.ones(len(reported)), abs=0.15)


def _wall_samples(wall: dict[int, np.ndarray], field_to_camera: np.ndarray, seed: int, count: int = 250):
    rng = np.random.default_rng(seed)
    sigma = 0.5
    exact = {
        tag_id: project_field_tag(
            field_to_camera, pose, TAG_SIZE_M, CAMERA.calibration_matrix, CAMERA.distortion_coefficients
        )
        for tag_id, pose in wall.items()
    }
    solver = CameraPoseSolver(corner_noise_px=sigma)
    samples = []
    for _ in range(count):
        observations = [
            FiducialImageObservation(tag_id, corners + rng.normal(0.0, sigma, (4, 2)))
            for tag_id, corners in exact.items()
        ]
        pose = solver.solve(observations, [], CAMERA, TAG_SIZE_M, wall)
        assert pose is not None and pose.field_to_camera_1 is None
        samples.append((pose_difference(pose.field_to_camera_0, field_to_camera), pose.covariance_0))
    return samples


@pytest.mark.parametrize(
    "operation",
    [
        lambda pose: pose.transform_by(transform_matrix([-0.3, 0.1, -0.5], [0.1, 0.2, -0.4])),
        lambda pose: pose.relative_to(transform_matrix([16.5, 8.0, 0.0], [0.0, 0.0, math.pi])),
        lambda pose: pose.inverse(),
    ],
    ids=["transform_by", "relative_to", "inverse"],
)
def test_pose_operations_propagate_covariance_by_their_jacobian(operation) -> None:
    rng = np.random.default_rng(3)
    root = rng.normal(size=(6, 6))
    pose = PoseWithCovariance(FIELD_TO_CAMERA, root @ root.T * 1e-3)
    result = operation(pose)

    epsilon = 1e-6
    jacobian = np.empty((6, 6))
    for column in range(6):
        step = np.zeros(6)
        step[column] = epsilon
        plus = operation(PoseWithCovariance(_perturb(pose.transform, step), pose.covariance)).transform
        minus = operation(PoseWithCovariance(_perturb(pose.transform, -step), pose.covariance)).transform
        jacobian[:, column] = (pose_difference(result.transform, plus) - pose_difference(result.transform, minus)) / (
            2 * epsilon
        )
    assert result.covariance == pytest.approx(jacobian @ pose.covariance @ jacobian.T, rel=1e-5, abs=1e-12)


def test_pose2d_projection_differentiates_yaw() -> None:
    rng = np.random.default_rng(4)
    root = rng.normal(size=(6, 6))
    pose = PoseWithCovariance(FIELD_TO_CAMERA, root @ root.T * 1e-3)
    planar = pose.to_pose2d()

    epsilon = 1e-6
    jacobian = np.zeros((3, 6))
    jacobian[0, 0] = jacobian[1, 1] = 1.0
    for column in range(3, 6):
        step = np.zeros(6)
        step[column] = epsilon
        yaws = [
            PoseWithCovariance(_perturb(pose.transform, sign * step), pose.covariance).to_pose2d().yaw
            for sign in (1, -1)
        ]
        jacobian[2, column] = (yaws[0] - yaws[1]) / (2 * epsilon)
    assert (planar.x, planar.y) == (FIELD_TO_CAMERA[0, 3], FIELD_TO_CAMERA[1, 3])
    assert planar.yaw == pytest.approx(math.atan2(FIELD_TO_CAMERA[1, 0], FIELD_TO_CAMERA[0, 0]))
    assert planar.covariance == pytest.approx(jacobian @ pose.covariance @ jacobian.T, rel=1e-5)
    assert planar.std_devs() == pytest.approx(np.sqrt(np.diag(planar.covariance)))


def test_multi_tag_covariance_matches_monte_carlo_and_carries_to_the_robot() -> None:
    rng = np.random.default_rng(7)
    sigma = 0.5
    single_solver, camera_solver = SingleTagPoseSolver(corner_noise_px=sigma), CameraPoseSolver(corner_noise_px=sigma)
    camera_to_robot = transform_matrix([-0.3, 0.1, -0.5], [0.0, 0.1, 0.2])
    field_to_robot = FIELD_TO_CAMERA @ camera_to_robot
    true_yaw = math.atan2(field_to_robot[1, 0], field_to_robot[0, 0])
    camera_samples, robot_samples = [], []
    for _ in range(250):
        observations = [
            FiducialImageObservation(tag_id, _project(FIELD_TO_CAMERA, tag_id) + rng.normal(0.0, sigma, (4, 2)))
            for tag_id in LAYOUT
        ]
        singles = single_solver.solve_many(observations, CAMERA, TAG_SIZE_M)
        pose = camera_solver.solve(observations, singles, CAMERA, TAG_SIZE_M, LAYOUT)
        assert pose is not None and pose.covariance_1 is None
        camera_samples.append((pose_difference(pose.field_to_camera_0, FIELD_TO_CAMERA), pose.covariance_0))
        robot = pose.field_to_camera().transform_by(camera_to_robot).to_pose2d()
        yaw_error = (true_yaw - robot.yaw + math.pi) % math.tau - math.pi
        robot_samples.append(
            (np.array([field_to_robot[0, 3] - robot.x, field_to_robot[1, 3] - robot.y, yaw_error]), robot.covariance)
        )
    _assert_consistent(camera_samples)
    _assert_axis_std_devs_match(camera_samples)
    _assert_consistent(robot_samples)
    _assert_axis_std_devs_match(robot_samples)


def test_public_camera_pose_covariance_matches_the_solver() -> None:
    observations = [FiducialImageObservation(tag_id, _project(FIELD_TO_CAMERA, tag_id)) for tag_id in LAYOUT]
    pose = CameraPoseSolver(corner_noise_px=0.7).solve(observations, [], CAMERA, TAG_SIZE_M, LAYOUT)
    assert pose is not None
    field_points = np.concatenate([tag_field_corners(LAYOUT[tag_id], TAG_SIZE_M) for tag_id in LAYOUT])
    helper = camera_pose_covariance(
        pose.field_to_camera_0, field_points, CAMERA.calibration_matrix, CAMERA.distortion_coefficients, 0.7
    )
    assert helper is not None
    assert helper.covariance == pytest.approx(pose.covariance_0, rel=1e-9)
    # Covariance scales with the corner noise variance.
    doubled = camera_pose_covariance(
        pose.field_to_camera_0, field_points, CAMERA.calibration_matrix, CAMERA.distortion_coefficients, 1.4
    )
    assert doubled is not None and doubled.covariance == pytest.approx(4.0 * helper.covariance, rel=1e-9)


def test_single_tag_covariance_and_angles_match_monte_carlo() -> None:
    rng = np.random.default_rng(11)
    sigma = 0.5
    solver, angle_solver = SingleTagPoseSolver(corner_noise_px=sigma), TagAngleSolver(corner_noise_px=sigma)
    camera_to_tag = transform_matrix([0.3, -0.2, 1.5], [0.4, 0.5, 0.1])  # OpenCV axes
    rvec = cv2.Rodrigues(camera_to_tag[:3, :3])[0]
    corners, _ = cv2.projectPoints(
        tag_object_points(TAG_SIZE_M),
        rvec,
        camera_to_tag[:3, 3],
        CAMERA.calibration_matrix,
        CAMERA.distortion_coefficients,
    )
    corners = corners.reshape(4, 2)
    true_pose = opencv_pose_to_wpilib_transform(rvec, camera_to_tag[:3, 3])
    exact = FiducialImageObservation(1, corners)
    true_angles = angle_solver.solve(exact, CAMERA, solver.solve(exact, CAMERA, TAG_SIZE_M))
    pose_samples, angle_samples = [], []
    for _ in range(250):
        observation = FiducialImageObservation(1, corners + rng.normal(0.0, sigma, (4, 2)))
        single = solver.solve(observation, CAMERA, TAG_SIZE_M)
        assert single is not None
        estimate = single.camera_to_tag()
        pose_samples.append((pose_difference(estimate.transform, true_pose), estimate.covariance))
        angles = angle_solver.solve(observation, CAMERA, single)
        error = np.array([true_angles.center_tx - angles.center_tx, true_angles.center_ty - angles.center_ty])
        angle_samples.append((error, angles.center_covariance))
    _assert_consistent(pose_samples)
    _assert_axis_std_devs_match(pose_samples)
    _assert_consistent(angle_samples)
    _assert_axis_std_devs_match(angle_samples)


def test_ambiguous_single_tag_covariance_covers_both_solutions() -> None:
    # A lone tag 3 m away, seen nearly head-on: its yaw is barely observable and the IPPE solutions trade places.
    rng = np.random.default_rng(5)
    sigma = 0.5
    field_to_tag = {1: transform_matrix([5.0, 2.0, 0.8], [0.0, 0.0, math.pi])}
    position = np.array([5.0 - 3.0 * math.cos(0.09), 2.0 + 3.0 * math.sin(0.09), 0.6])
    field_to_camera = transform_matrix(position, [0.0, 0.0, math.atan2(2.0 - position[1], 5.0 - position[0])])
    exact = project_field_tag(
        field_to_camera, field_to_tag[1], TAG_SIZE_M, CAMERA.calibration_matrix, CAMERA.distortion_coefficients
    )
    single_solver, camera_solver = SingleTagPoseSolver(corner_noise_px=sigma), CameraPoseSolver(corner_noise_px=sigma)
    samples = []
    for _ in range(250):
        observation = FiducialImageObservation(1, exact + rng.normal(0.0, sigma, (4, 2)))
        single = single_solver.solve(observation, CAMERA, TAG_SIZE_M)
        pose = camera_solver.solve([observation], [single], CAMERA, TAG_SIZE_M, field_to_tag)
        if pose is not None:
            samples.append((pose_difference(pose.field_to_camera_0, field_to_camera), pose.covariance_0))
            assert pose.ambiguity == pytest.approx(pose.error_0 / pose.error_1)
    assert len(samples) > 200
    # Without the ambiguity term the mean NEES is about 130.
    _assert_consistent(samples)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([5.0, 0.2, 0.8], [5.0, -0.2, 0.8]),  # one plane: a single tag's two-fold ambiguity; plain NEES ~120
        ([8.0, 0.75, 0.8], [8.0, -0.75, 0.8]),  # farther and wider; plain NEES ~50
        ([8.1, 0.2, 0.8], [8.0, -0.2, 0.8]),  # nearly coplanar: a curved error valley; plain NEES ~50
    ],
    ids=["wall-5m", "wall-8m-wide", "near-planar-8m"],
)
def test_head_on_multi_tag_covariance_is_never_overconfident(first: list[float], second: list[float]) -> None:
    wall = {1: transform_matrix(first, [0.0, 0.0, math.pi]), 2: transform_matrix(second, [0.0, 0.0, math.pi])}
    _assert_consistent(_wall_samples(wall, transform_matrix([0.0, 0.0, 0.8], [0.0, 0.0, 0.04]), seed=9, count=300))


def test_local_std_devs_follow_the_heading() -> None:
    covariance = np.diag([0.04, 0.01, 0.0025])
    facing_left = Pose2dWithCovariance(1.0, 2.0, math.pi / 2, covariance)
    assert facing_left.std_devs() == pytest.approx([0.2, 0.1, 0.05])
    # Facing +Y, the robot's forward axis is the field's Y and its left axis the field's -X.
    assert facing_left.local_std_devs() == pytest.approx([0.1, 0.2, 0.05])


def test_processor_attaches_covariances_and_validates_noise() -> None:
    corners = {tag_id: _project(FIELD_TO_CAMERA, tag_id) for tag_id in LAYOUT}

    class Detector:
        def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
            return [FiducialImageObservation(tag_id, value) for tag_id, value in corners.items()]

    calibration = SimpleNamespace(
        camera_matrix=CAMERA.calibration_matrix,
        distortion_coefficients=CAMERA.distortion_coefficients,
        image_size=(1280, 720),
        validate_for_image_size=lambda width, height: None,
    )
    processor = VisionProcessor(
        detector=Detector(), calibration=calibration, tag_size_m=TAG_SIZE_M, tag_layout=LAYOUT, corner_noise_px=0.8
    )
    result = processor.process(np.zeros((720, 1280), dtype=np.uint8), timestamp=0.0)

    assert all(pose.covariance_0 is not None and pose.covariance_1 is not None for pose in result.single_tag_poses)
    assert all(angle.center_covariance is not None for angle in result.tag_angles)
    assert result.camera_pose is not None
    field_to_camera = result.camera_pose.field_to_camera()
    assert field_to_camera.covariance.shape == (6, 6) and np.all(field_to_camera.std_devs() > 0.0)
    with pytest.raises(ValueError, match="solution 1 is unavailable"):
        result.camera_pose.field_to_camera(1)
    with pytest.raises(ValueError, match="solution must be 0 or 1"):
        result.single_tag_poses[0].camera_to_tag(2)
    with pytest.raises(ValueError, match="corner_noise_px"):
        VisionProcessor(calibration=calibration, tag_size_m=TAG_SIZE_M, corner_noise_px=0.0)
    with pytest.raises(ValueError, match="6x6"):
        PoseWithCovariance(np.eye(4), np.eye(3))
