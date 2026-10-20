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
from napa_vision.covariance import _rotation_matrices, mirror_path_pose, pose_difference
from napa_vision.geometry import (
    WPILIB_TO_OPENCV_ROTATION,
    invert_transform,
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


def _assert_consistent(samples: list[tuple[np.ndarray, np.ndarray]], *, max_outside: float = 0.1) -> None:
    """Never overconfident, and not grossly pessimistic.

    With exact covariances, 5% of errors fall outside the 95% ellipse; allow sampling noise but nothing more. Napa
    caps correlations (``CORRELATION_SLACK``) and widens ambiguous poses, so it may be conservative; the median
    normalized error squared (NEES, 5.3 for an exact 6-D covariance) must still show the covariance isn't useless.
    """

    dimension = len(samples[0][0])
    nees = np.array([error @ np.linalg.solve(covariance, error) for error, covariance in samples])
    assert np.mean(nees > CHI_SQUARED_95[dimension]) < max_outside
    assert np.median(nees) > 0.2 * dimension


def _assert_axis_std_devs_match(samples: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """A typical frame's reported standard deviation on each axis is close to the actual scatter."""

    errors = np.array([error for error, _ in samples])
    reported = np.sqrt(np.median([np.diag(covariance) for _, covariance in samples], axis=0))
    ratio = errors.std(axis=0) / reported
    assert np.all((ratio > 0.75) & (ratio < 1.15))


def _wall_samples(
    wall: dict[int, np.ndarray],
    field_to_camera: np.ndarray,
    seed: int,
    count: int = 250,
    sigma: float = 0.5,
    camera: SimpleNamespace = CAMERA,
):
    rng = np.random.default_rng(seed)
    exact = {
        tag_id: project_field_tag(
            field_to_camera, pose, TAG_SIZE_M, camera.calibration_matrix, camera.distortion_coefficients
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
        pose = solver.solve(observations, [], camera, TAG_SIZE_M, wall)
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
    samples, inverse_samples = [], []
    for _ in range(250):
        observation = FiducialImageObservation(1, exact + rng.normal(0.0, sigma, (4, 2)))
        single = single_solver.solve(observation, CAMERA, TAG_SIZE_M)
        pose = camera_solver.solve([observation], [single], CAMERA, TAG_SIZE_M, field_to_tag)
        if pose is not None:
            samples.append((pose_difference(pose.field_to_camera_0, field_to_camera), pose.covariance_0))
            assert pose.ambiguity == pytest.approx(pose.error_0 / pose.error_1)
            # The solutions survive into the pose, so moving it keeps covering both.
            inverse = pose.field_to_camera().inverse()
            assert len(inverse._components.weights) == 2
            inverse_samples.append(
                (pose_difference(inverse.transform, invert_transform(field_to_camera)), inverse.covariance)
            )
    assert len(samples) > 200
    # Without the ambiguity term the mean NEES is about 130.
    _assert_consistent(samples)
    _assert_consistent(inverse_samples)


HEAD_ON = transform_matrix([0.0, 0.0, 0.8], [0.0, 0.0, 0.04])
STRONG_DISTORTION = SimpleNamespace(
    calibration_matrix=CAMERA.calibration_matrix, distortion_coefficients=np.array([-0.22, 0.06, 0.001, -0.001, 0.0])
)


def _wall(*positions: list[float]) -> dict[int, np.ndarray]:
    return {index + 1: transform_matrix(position, [0.0, 0.0, math.pi]) for index, position in enumerate(positions)}


@pytest.mark.parametrize(
    ("wall", "field_to_camera", "sigma", "camera"),
    [
        # One plane: a single tag's two-fold ambiguity. A plain linearization gives mean NEES ~120.
        (_wall([5.0, 0.2, 0.8], [5.0, -0.2, 0.8]), HEAD_ON, 0.5, CAMERA),
        # Farther and wider: ~50.
        (_wall([8.0, 0.75, 0.8], [8.0, -0.75, 0.8]), HEAD_ON, 0.5, CAMERA),
        # Nearly coplanar, a curved error valley: ~50.
        (_wall([8.1, 0.2, 0.8], [8.0, -0.2, 0.8]), HEAD_ON, 0.5, CAMERA),
        # A wide row of three tags through strong distortion: 11, with 22% outside the 95% ellipse.
        (
            _wall([8.0, -1.5, 0.8], [8.0, 0.0, 0.8], [8.0, 1.5, 0.8]),
            transform_matrix([0.0, 0.0, 0.8], [0.0, 0.0, 0.0]),
            1.0,
            STRONG_DISTORTION,
        ),
        # Four tags 6.6 m apart, seen 8 m away at 17 degrees: a 2% correlation cap left 21% outside.
        (
            _wall([10.0, -3.3155, 0.8], [10.0, -1.1052, 0.8], [10.0, 1.1052, 0.8], [10.0, 3.3155, 0.8]),
            transform_matrix([2.4815, -2.2692, 0.6], [0.0, 0.0, 0.2097]),
            0.866,
            STRONG_DISTORTION,
        ),
    ],
    ids=["wall-5m", "wall-8m-wide", "near-planar-8m", "wide-row-distorted", "wide-row-oblique"],
)
def test_planar_multi_tag_covariance_is_never_overconfident(wall, field_to_camera, sigma, camera) -> None:
    _assert_consistent(_wall_samples(wall, field_to_camera, seed=9, count=300, sigma=sigma, camera=camera))


def test_far_oblique_pair_covers_its_mirror_solution() -> None:
    # Two tags 10 m away at 54 degrees: on some frames the solved pose is the mirror, 16 m and 108 degrees off.
    wall = _wall([10.0, -0.4, 0.8], [10.0, 0.4, 0.8])
    position = [10.0 - 9.8 * math.cos(math.radians(-53.9)), 9.8 * math.sin(math.radians(-53.9)), 0.6]
    field_to_camera = transform_matrix(position, [0.0, 0.0, math.atan2(-position[1], 10.0 - position[0])])
    camera = SimpleNamespace(calibration_matrix=CAMERA.calibration_matrix, distortion_coefficients=np.zeros(5))
    # The hardest geometry found: its tags are ~15 px across, below what the default detector reliably finds. A plain
    # linearization put 15% of errors outside the 95% ellipse, thousands of times too confident on those frames.
    samples = _wall_samples(wall, field_to_camera, seed=5, count=300, sigma=0.97, camera=camera)
    _assert_consistent(samples, max_outside=0.15)


def test_inverting_a_single_tag_pose_keeps_its_ambiguity_coverage() -> None:
    # A lone tag 3 m away, nearly head-on. Linearizing the inverse of the widened covariance gave mean NEES ~58.
    rng = np.random.default_rng(3)
    field_to_tag = transform_matrix([5.0, 2.0, 0.8], [0.0, 0.0, math.pi])
    position = np.array([5.0 - 3.0 * math.cos(0.09), 2.0 + 3.0 * math.sin(0.09), 0.6])
    field_to_camera = transform_matrix(position, [0.0, 0.0, math.atan2(2.0 - position[1], 5.0 - position[0])])
    tag_to_camera = invert_transform(field_to_tag) @ field_to_camera
    camera_to_robot = transform_matrix([-0.3, 0.1, -0.4], [0.0, 0.1, 0.2])
    camera = SimpleNamespace(calibration_matrix=CAMERA.calibration_matrix, distortion_coefficients=np.zeros(5))
    exact = project_field_tag(field_to_camera, field_to_tag, TAG_SIZE_M, camera.calibration_matrix, np.zeros(5))
    solver = SingleTagPoseSolver(corner_noise_px=0.5)
    inverse_samples, robot_samples = [], []
    for _ in range(300):
        single = solver.solve(FiducialImageObservation(1, exact + rng.normal(0.0, 0.5, (4, 2))), camera, TAG_SIZE_M)
        assert single is not None
        estimate = single.camera_to_tag().inverse()
        inverse_samples.append((pose_difference(estimate.transform, tag_to_camera), estimate.covariance))
        robot = estimate.transform_by(camera_to_robot)
        robot_samples.append((pose_difference(robot.transform, tag_to_camera @ camera_to_robot), robot.covariance))
    _assert_consistent(inverse_samples)
    _assert_consistent(robot_samples)


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


def test_mirror_path_survives_steps_behind_the_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    # A mirror candidate rotated far enough that the path's steps swing the tags behind the camera. The skip gate
    # would bypass the path for so poor a mirror, so it is disabled to exercise the guards.
    import napa_vision.covariance as covariance

    monkeypatch.setattr(covariance, "MIRROR_SKIP_SIGMAS", math.inf)
    field_points = np.concatenate([tag_field_corners(LAYOUT[tag_id], TAG_SIZE_M) for tag_id in (1, 2)])
    camera_to_field = invert_transform(FIELD_TO_CAMERA)
    image_points = np.concatenate([_project(FIELD_TO_CAMERA, tag_id) for tag_id in (1, 2)])
    mirror = camera_to_field.copy()
    mirror[:3, :3] = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(150.0)]))[0] @ camera_to_field[:3, :3]
    pose = mirror_path_pose(
        camera_to_field,
        mirror,
        field_points,
        image_points,
        CAMERA.calibration_matrix,
        CAMERA.distortion_coefficients,
        0.5,
        opencv_from_camera=WPILIB_TO_OPENCV_ROTATION,
    )
    assert pose is not None
    # The path ran, and the steps behind the camera were dropped.
    assert 1 < len(pose._components.weights) < len(covariance.MIRROR_PATH_STEPS)
    assert np.all(np.isfinite(pose.covariance)) and np.all(np.linalg.eigvalsh(pose.covariance) > 0.0)
    assert pose.transform == pytest.approx(camera_to_field)


def test_editing_a_result_drops_the_stored_solutions() -> None:
    observations = [FiducialImageObservation(tag_id, _project(FIELD_TO_CAMERA, tag_id)) for tag_id in LAYOUT]
    single = SingleTagPoseSolver().solve(observations[0], CAMERA, TAG_SIZE_M)
    pose = CameraPoseSolver().solve(observations, [], CAMERA, TAG_SIZE_M, LAYOUT)
    assert single is not None and pose is not None
    with pytest.raises(ValueError, match="read-only"):
        single.covariance_0 *= 4.0
    original = single.camera_to_tag().covariance
    single.covariance_0 = single.covariance_0 * 4.0
    # A plain pose built from the edited values, not the solver's stored solutions.
    edited = single.camera_to_tag()
    assert edited.covariance == pytest.approx(4.0 * original)
    assert edited.inverse().inverse().covariance == pytest.approx(edited.covariance)
    pose.field_to_camera_0[0, 3] += 1.0
    assert pose.field_to_camera().transform == pytest.approx(pose.field_to_camera_0)


def test_tiny_rotations_keep_their_first_order_term() -> None:
    vector = np.array([1e-13, -2e-13, 3e-13])
    rotated = np.eye(4)
    rotated[:3, :3] = _rotation_matrices(vector)[0]
    assert pose_difference(np.eye(4), rotated)[3:] == pytest.approx(vector, rel=1e-6, abs=1e-20)


def test_covariances_are_computed_only_when_read(monkeypatch: pytest.MonkeyPatch) -> None:
    import pickle

    import napa_vision.solvers as solvers

    calls = {"poses": 0, "angles": 0}
    angles = solvers.center_angle_covariances

    def counted(function):
        def counted_reprojection(*args, **kwargs):
            calls["poses"] += 1
            return function(*args, **kwargs)

        return counted_reprojection

    def counted_angles(*args, **kwargs):
        calls["angles"] += 1
        return angles(*args, **kwargs)

    for name in ("reprojection_covariances", "_reprojection_fit", "mirror_path_pose"):
        monkeypatch.setattr(solvers, name, counted(getattr(solvers, name)))
    monkeypatch.setattr(solvers, "center_angle_covariances", counted_angles)
    observations = [FiducialImageObservation(tag_id, _project(FIELD_TO_CAMERA, tag_id)) for tag_id in LAYOUT]
    singles = SingleTagPoseSolver().solve_many(observations, CAMERA, TAG_SIZE_M)
    tag_angles = TagAngleSolver().solve_many(observations, CAMERA, singles)
    pose = CameraPoseSolver().solve(observations, singles, CAMERA, TAG_SIZE_M, LAYOUT)
    assert pose is not None and calls == {"poses": 0, "angles": 0}

    # Deferred work survives pickling, as for results sent to another process.
    restored = pickle.loads(pickle.dumps((singles, tag_angles, pose)))
    assert restored[2].field_to_camera().covariance == pytest.approx(pose.field_to_camera().covariance)

    # The first tag read computes every tag's covariance in one batch.
    calls["poses"] = 0
    assert singles[0].covariance_0 is not None
    assert calls["poses"] == 1
    assert all(single.covariance_0 is not None for single in singles) and calls["poses"] == 1
    assert tag_angles[0].center_covariance is not None and calls["angles"] == 1
    assert all(angle.center_covariance is not None for angle in tag_angles) and calls["angles"] == 1

    # An assigned covariance replaces the solver's.
    tag_angles[0].center_covariance = np.eye(2)
    assert tag_angles[0].center_covariance == pytest.approx(np.eye(2))
    pose.covariance_0 = np.eye(6)
    assert pose.field_to_camera().covariance == pytest.approx(np.eye(6))


def test_deferred_covariances_ignore_later_changes_to_their_inputs() -> None:
    camera = SimpleNamespace(
        calibration_matrix=CAMERA.calibration_matrix.copy(), distortion_coefficients=CAMERA.distortion_coefficients
    )
    observations = [FiducialImageObservation(tag_id, _project(FIELD_TO_CAMERA, tag_id)) for tag_id in LAYOUT]
    singles = SingleTagPoseSolver().solve_many(observations, camera, TAG_SIZE_M)
    tag_angles = TagAngleSolver().solve_many(observations, camera, singles)
    pose = CameraPoseSolver().solve(observations, singles, camera, TAG_SIZE_M, LAYOUT)
    assert pose is not None
    reference = CameraPoseSolver().solve(observations, singles, CAMERA, TAG_SIZE_M, LAYOUT)
    expected_angle = TagAngleSolver().solve_many(observations, CAMERA, singles)[0].center_covariance
    camera.calibration_matrix[:2, :2] *= 2.0
    observations[0].corners[:] = np.nan
    tag_angles[0].corner_angles[:] = np.nan
    assert pose.covariance_0 == pytest.approx(reference.covariance_0)
    assert tag_angles[0].center_covariance == pytest.approx(expected_angle)
    assert all(single.covariance_0 is not None and np.all(np.isfinite(single.covariance_0)) for single in singles)


def test_copies_and_replacements_keep_uncertainty_consistent() -> None:
    import copy
    import dataclasses
    import pickle

    observations = [FiducialImageObservation(tag_id, _project(FIELD_TO_CAMERA, tag_id)) for tag_id in LAYOUT]
    singles = SingleTagPoseSolver().solve_many(observations, CAMERA, TAG_SIZE_M)
    tag_angles = TagAngleSolver().solve_many(observations, CAMERA, singles)
    pose = CameraPoseSolver().solve(observations, singles, CAMERA, TAG_SIZE_M, LAYOUT)
    assert pose is not None

    # Replacing an unrelated field keeps the uncertainty; replacing the pose drops the stored solutions.
    for observation in (singles[0], pose):
        assert dataclasses.replace(observation).covariance_0 == pytest.approx(observation.covariance_0)
    assert dataclasses.replace(tag_angles[0]).center_covariance == pytest.approx(tag_angles[0].center_covariance)
    moved = dataclasses.replace(pose, field_to_camera_0=pose.field_to_camera_0.copy())
    moved.field_to_camera_0[0, 3] += 1.0
    assert moved.field_to_camera().transform == pytest.approx(moved.field_to_camera_0)

    # A copy's assignment doesn't reach the original.
    duplicate = copy.copy(pose)
    duplicate.covariance_0 = np.eye(6)
    assert not np.allclose(pose.covariance_0, np.eye(6))

    # Copied and unpickled poses stay read-only, and a replaced covariance is never overridden by old solutions.
    field_pose = pose.field_to_camera()
    for clone in (copy.deepcopy(field_pose), pickle.loads(pickle.dumps(field_pose))):
        assert not clone.covariance.flags.writeable
        assert clone.covariance == pytest.approx(field_pose.covariance)
    widened = dataclasses.replace(field_pose, covariance=4.0 * field_pose.covariance)
    assert widened.transform_by(np.eye(4)).covariance == pytest.approx(widened.covariance)
