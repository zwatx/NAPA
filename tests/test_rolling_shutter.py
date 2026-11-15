from __future__ import annotations

import cv2
import numpy as np
import pytest

from napa_vision import (
    ArtifactProvenance,
    CameraCalibration,
    FiducialImageObservation,
    ProvenanceKind,
    RollingShutter,
    VisionProcessor,
    estimate_rolling_shutter,
)
from napa_vision.geometry import tag_field_corners, tag_object_points

WIDTH, HEIGHT, SIZE = 640, 480, 0.2
READOUT, DT = 0.02, 0.02
IMAGE = np.zeros((HEIGHT, WIDTH), np.uint8)
CALIBRATION = CameraCalibration(
    camera_matrix=np.array([[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]]),
    distortion_coefficients=np.zeros(5),
    image_size=(WIDTH, HEIGHT),
    provenance=ArtifactProvenance(ProvenanceKind.EXAMPLE, "synthetic rolling shutter test"),
)
DIRECTIONS = {
    "top_to_bottom": (1, 1.0),
    "bottom_to_top": (1, -1.0),
    "left_to_right": (0, 1.0),
    "right_to_left": (0, -1.0),
}


def _corners(tvec: tuple[float, float, float]) -> np.ndarray:
    points, _ = cv2.projectPoints(
        tag_object_points(SIZE),
        np.array([0.2, -0.1, 0.1]),
        np.array(tvec, float),
        CALIBRATION.camera_matrix,
        CALIBRATION.distortion_coefficients,
    )
    return points.reshape(4, 2)


def _rolled(sharp: np.ndarray, velocity: np.ndarray, direction: str) -> np.ndarray:
    # Each corner sits where the tag was when its line was read, relative to the moment of the middle line.
    axis, sign = DIRECTIONS[direction]
    lines = (WIDTH, HEIGHT)[axis]
    read_after_middle = sign * (sharp[:, axis] - (lines - 1) / 2) / lines * READOUT
    return sharp + velocity * read_after_middle[:, None]


class Scripted:
    def __init__(self) -> None:
        self.frames: list[list[FiducialImageObservation]] = []

    def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
        return self.frames.pop(0)


def _processor(detector: Scripted, shutter: RollingShutter | None, layout=None) -> VisionProcessor:
    return VisionProcessor(
        detector=detector, calibration=CALIBRATION, tag_size_m=SIZE, tag_layout=layout, rolling_shutter=shutter
    )


def _tvec(result) -> np.ndarray:
    return np.asarray(result.single_tag_poses[0].pose_0_tvec).ravel()


def _sequence(direction: str, velocity: np.ndarray, frames: int = 3) -> list[np.ndarray]:
    sharp = _corners((0.1, 0.05, 2.0))
    return [_rolled(sharp + velocity * DT * k, velocity, direction) for k in range(frames)], sharp + velocity * DT * (
        frames - 1
    )


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_sheared_corners_solve_as_if_read_at_the_middle_line(direction: str) -> None:
    velocity = np.array([900.0, 700.0])
    rolled, sharp = _sequence(direction, velocity)
    detector = Scripted()
    detector.frames = [[FiducialImageObservation(42, corners)] for corners in rolled] + [
        [FiducialImageObservation(42, sharp)]
    ]
    processor = _processor(detector, RollingShutter(READOUT, direction=direction))
    results = [processor.process(IMAGE, timestamp=DT * k) for k in range(3)]
    ideal = _processor(detector, None).process(IMAGE, timestamp=0.0)

    assert results[0].uncorrected_tag_ids == (42,)
    assert results[1].uncorrected_tag_ids == results[2].uncorrected_tag_ids == ()
    assert np.linalg.norm(_tvec(results[2]) - _tvec(ideal)) < 1.5e-3
    # The detections themselves are reported as they were found.
    np.testing.assert_array_equal(results[2].image_observations[0].corners, rolled[2])


def test_uses_the_last_steps_acceleration() -> None:
    # Velocity grows each frame; two earlier sightings extrapolate to the current frame's speed.
    sharp = _corners((0.0, 0.0, 2.0))
    speeds = np.array([[600.0, 0.0], [800.0, 0.0], [1000.0, 0.0]])
    positions = [sharp, sharp + speeds[1] * DT - np.array([0.0, 0.0]), sharp + speeds[1] * DT + speeds[2] * DT]
    positions[1] = sharp + 0.5 * (speeds[0] + speeds[1]) * DT
    positions[2] = positions[1] + 0.5 * (speeds[1] + speeds[2]) * DT
    rolled = [_rolled(p, speed, "top_to_bottom") for p, speed in zip(positions, speeds, strict=True)]
    detector = Scripted()
    detector.frames = [[FiducialImageObservation(42, corners)] for corners in rolled] + [
        [FiducialImageObservation(42, positions[2])]
    ]
    processor = _processor(detector, RollingShutter(READOUT))
    results = [processor.process(IMAGE, timestamp=DT * k) for k in range(3)]
    ideal = _processor(detector, None).process(IMAGE, timestamp=0.0)
    assert np.linalg.norm(_tvec(results[2]) - _tvec(ideal)) < 1.5e-3


def _solved_as_detected(result, corners: np.ndarray) -> None:
    detector = Scripted()
    detector.frames = [[FiducialImageObservation(42, corners)]]
    plain = _processor(detector, None).process(IMAGE, timestamp=0.0)
    assert result.uncorrected_tag_ids == (42,)
    np.testing.assert_array_equal(_tvec(result), _tvec(plain))


def test_stale_sightings_are_not_used() -> None:
    velocity = np.array([900.0, 0.0])
    rolled, _ = _sequence("top_to_bottom", velocity)
    detector = Scripted()
    detector.frames = [[FiducialImageObservation(42, corners)] for corners in rolled]
    processor = _processor(detector, RollingShutter(READOUT, max_frame_gap_s=0.05))
    processor.process(IMAGE, timestamp=0.0)
    _solved_as_detected(processor.process(IMAGE, timestamp=0.06), rolled[1])


def test_repeated_or_backward_timestamps_start_over() -> None:
    rolled, _ = _sequence("top_to_bottom", np.array([900.0, 0.0]), frames=2)
    detector = Scripted()
    detector.frames = [[FiducialImageObservation(42, corners)] for corners in rolled]
    processor = _processor(detector, RollingShutter(READOUT))
    processor.process(IMAGE, timestamp=5.0)
    _solved_as_detected(processor.process(IMAGE, timestamp=4.98), rolled[1])


def test_an_id_seen_twice_in_a_frame_is_not_corrected() -> None:
    rolled, _ = _sequence("top_to_bottom", np.array([900.0, 0.0]), frames=2)
    far = rolled[1] + np.array([150.0, 0.0])
    detector = Scripted()
    detector.frames = [
        [FiducialImageObservation(42, rolled[0])],
        [FiducialImageObservation(42, rolled[1]), FiducialImageObservation(42, far)],
    ]
    processor = _processor(detector, RollingShutter(READOUT))
    processor.process(IMAGE, timestamp=0.0)
    assert processor.process(IMAGE, timestamp=DT).uncorrected_tag_ids == (42, 42)


def _layout_and_corners() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    from napa_vision.geometry import wpilib_points_to_opencv

    layout, corners = {}, {}
    for tag_id, offset in ((1, -0.4), (2, 0.0), (3, 0.4)):
        field_to_tag = np.eye(4)
        field_to_tag[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        field_to_tag[:3, 3] = [3.0, offset, offset / 2]
        layout[tag_id] = field_to_tag
        points, _ = cv2.projectPoints(
            wpilib_points_to_opencv(tag_field_corners(field_to_tag, SIZE)),
            np.zeros(3),
            np.zeros(3),
            CALIBRATION.camera_matrix,
            CALIBRATION.distortion_coefficients,
        )
        corners[tag_id] = points.reshape(4, 2)
    return layout, corners


def test_a_tag_without_history_is_left_out_of_the_multi_tag_pose() -> None:
    layout, corners = _layout_and_corners()
    detector = Scripted()
    detector.frames = [
        [FiducialImageObservation(1, corners[1]), FiducialImageObservation(2, corners[2])],
        [
            FiducialImageObservation(1, corners[1]),
            FiducialImageObservation(2, corners[2]),
            FiducialImageObservation(3, corners[3]),
        ],
    ]
    processor = _processor(detector, RollingShutter(READOUT), layout)
    first = processor.process(IMAGE, timestamp=0.0)
    second = processor.process(IMAGE, timestamp=DT)

    # With nothing corrected yet, every tag is used, as without the setting.
    assert sorted(first.camera_pose.tag_ids) == [1, 2]
    assert second.uncorrected_tag_ids == (3,)
    assert sorted(second.camera_pose.tag_ids) == [1, 2]
    assert 3 in [pose.tag_id for pose in second.single_tag_poses]  # still solved on its own


def test_unset_passes_the_detections_straight_to_the_solvers(monkeypatch) -> None:
    rolled, _ = _sequence("top_to_bottom", np.array([900.0, 0.0]))
    detector = Scripted()
    detections = [FiducialImageObservation(42, rolled[0])]
    detector.frames = [detections]
    processor = _processor(detector, None)
    seen = []
    solve = processor._single_solver.solve_many
    monkeypatch.setattr(
        processor._single_solver, "solve_many", lambda obs, *rest: seen.append(obs) or solve(obs, *rest)
    )
    result = processor.process(IMAGE, timestamp=0.0)
    assert seen[0] is detections and result.uncorrected_tag_ids == ()


@pytest.mark.parametrize(
    "arguments",
    [
        {"readout_s": 0.0},
        {"readout_s": -1.0},
        {"readout_s": float("nan")},
        {"readout_s": True},
        {"readout_s": 0.01, "direction": "sideways"},
        {"readout_s": 0.01, "max_frame_gap_s": 0.0},
    ],
)
def test_rejects_invalid_settings(arguments: dict) -> None:
    with pytest.raises(ValueError):
        RollingShutter(**arguments)


def test_requires_calibration_and_the_right_type() -> None:
    with pytest.raises(ValueError, match="calibration"):
        VisionProcessor(rolling_shutter=RollingShutter(0.01))
    with pytest.raises(ValueError, match="RollingShutter"):
        VisionProcessor(calibration=CALIBRATION, tag_size_m=SIZE, rolling_shutter=0.01)


class _Corners:
    def __init__(self) -> None:
        self.frames: list[list[FiducialImageObservation]] = []

    def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
        return self.frames.pop(0)


def _recording(
    direction: str, readout: float, yaw: float, pitch: float, sequences: int = 40, distance: tuple = (0.9, 1.4)
):
    # Each corner solved for the time its own line is read; three consecutive frames per tag.
    axis, sign = DIRECTIONS[direction]
    lines = (WIDTH, HEIGHT)[axis]
    rng = np.random.default_rng(20261110)
    matrix = CALIBRATION.camera_matrix
    frames, detector, start = [], _Corners(), 0.0
    for _ in range(sequences):
        z = rng.uniform(*distance)  # 0.9 to 1.4 m is 70 to 110 px across
        tvec = np.array([rng.uniform(-0.2, 0.2) * z, rng.uniform(-0.12, 0.12) * z, z])
        rvec = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4)])
        rotation = cv2.Rodrigues(rvec)[0]
        points = tag_object_points(SIZE) @ rotation.T + tvec

        def project(at: float, points: np.ndarray = points) -> np.ndarray:
            camera = cv2.Rodrigues(np.array([pitch, yaw, 0.0]) * at)[0]
            projected = (points @ camera) @ matrix.T
            return projected[:, :2] / projected[:, 2:]

        for frame in range(3):
            time = frame * DT
            corners = project(time)
            for _ in range(6):
                times = time + sign * (corners[:, axis] / lines - 0.5) * readout
                corners = np.array([project(times[k])[k] for k in range(4)])
            frames.append((start + time, IMAGE))
            detector.frames.append([FiducialImageObservation(7, corners + rng.normal(0, 0.05, corners.shape))])
        start += 1.0
    return frames, detector


def _estimate(direction: str, readout: float, yaw: float, pitch: float, **options):
    frames, detector = _recording(direction, readout, yaw, pitch)
    return estimate_rolling_shutter(frames, calibration=CALIBRATION, tag_size_m=SIZE, detector=detector, **options)


@pytest.mark.parametrize(
    ("direction", "yaw", "pitch"),
    [
        ("top_to_bottom", 2.0, 0.0),
        ("bottom_to_top", 2.0, 0.0),
        ("left_to_right", 0.0, 2.0),
        ("right_to_left", 0.0, 2.0),
    ],
)
def test_reads_the_readout_and_direction_from_motion(direction: str, yaw: float, pitch: float) -> None:
    estimate = _estimate(direction, 0.015, yaw, pitch)
    assert estimate.rolling_shutter is not None and estimate.rolling_shutter.direction == direction
    assert estimate.readout_s == pytest.approx(0.015, abs=0.002)
    assert estimate.low <= estimate.readout_s <= estimate.high + 1e-3


def test_a_global_shutter_reads_as_one() -> None:
    estimate = _estimate("top_to_bottom", 0.0, 2.0, 0.0)
    assert estimate.readout_s == 0.0 and estimate.rolling_shutter is None and estimate.axes_checked == ("rows",)


def test_a_recording_without_motion_says_nothing() -> None:
    estimate = _estimate("top_to_bottom", 0.015, 0.05, 0.0)
    assert estimate.readout_s is None and estimate.rolling_shutter is None and estimate.axes_checked == ()


def test_small_slow_tags_say_nothing() -> None:
    # Moving enough to pass the speed limit, but the shear across a tag this small is lost in the pose fit's residual.
    frames, detector = _recording("top_to_bottom", 0.03, 0.35, 0.0, distance=(4.0, 5.0))
    estimate = estimate_rolling_shutter(frames, calibration=CALIBRATION, tag_size_m=SIZE, detector=detector)
    assert estimate.readout_s is None and estimate.rolling_shutter is None


def test_tags_under_the_size_floor_say_nothing() -> None:
    # About 40 to 50 px across and moving fast: enough shear, but such tags read 10 to 20% off with a narrow range.
    frames, detector = _recording("top_to_bottom", 0.015, 2.0, 0.0, distance=(2.0, 2.5))
    estimate = estimate_rolling_shutter(frames, calibration=CALIBRATION, tag_size_m=SIZE, detector=detector)
    assert estimate.readout_s is None and estimate.sightings == 0


def test_a_minimum_at_the_end_of_the_search_is_not_an_answer() -> None:
    estimate = _estimate("top_to_bottom", 0.09, 2.0, 0.0)
    assert estimate.rolling_shutter is None and estimate.readout_s is None


def test_too_few_sightings_say_nothing() -> None:
    assert _estimate("top_to_bottom", 0.015, 2.0, 0.0, min_sightings=1000).readout_s is None
    with pytest.raises(ValueError, match="min_sightings"):
        _estimate("top_to_bottom", 0.015, 2.0, 0.0, min_sightings=0)


def test_a_minimum_at_either_end_of_the_search_is_not_an_answer() -> None:
    estimate = _estimate("bottom_to_top", 0.09, 2.0, 0.0)
    assert estimate.rolling_shutter is None and estimate.readout_s is None


def _wall(columns: int, rows: int, distance: float = 6.0) -> dict[int, np.ndarray]:
    # Tags about 17 px across, facing the camera from `distance` along the field's +X.
    layout = {}
    for k in range(columns * rows):
        field_to_tag = np.eye(4)
        field_to_tag[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        y = (k % columns - (columns - 1) / 2) * 1.5
        z = (k // columns - (rows - 1) / 2) * 1.4
        field_to_tag[:3, 3] = [distance + 0.3 * (k % 3), y, z]
        layout[k] = field_to_tag
    return layout


def _layout_recording(layout: dict, direction: str, readout: float, yaw: float, pitch: float, sequences: int = 20):
    from napa_vision.geometry import wpilib_points_to_opencv

    axis, sign = DIRECTIONS[direction]
    lines = (WIDTH, HEIGHT)[axis]
    rng = np.random.default_rng(20261111)
    world = {i: wpilib_points_to_opencv(tag_field_corners(t, SIZE)) for i, t in layout.items()}
    frames, detector, start = [], _Corners(), 0.0
    for _ in range(sequences):
        turn = np.array([pitch, yaw, 0.0]) * rng.choice([-1.0, 1.0])
        offset = rng.uniform(-0.1, 0.1, 3)

        def project(points: np.ndarray, at: float, turn: np.ndarray = turn, offset: np.ndarray = offset) -> np.ndarray:
            rvec = cv2.Rodrigues(cv2.Rodrigues(turn * at)[0] @ cv2.Rodrigues(offset)[0])[0]
            return cv2.projectPoints(points, rvec, np.zeros(3), CALIBRATION.camera_matrix, None)[0].reshape(-1, 2)

        for frame in range(3):
            time = frame * DT
            seen = []
            for tag_id, points in world.items():
                corners = project(points, time)
                for _ in range(6):
                    times = time + sign * (corners[:, axis] / lines - 0.5) * readout
                    corners = np.array([project(points[k : k + 1], times[k])[0] for k in range(4)])
                if corners.min() > 2 and corners[:, 0].max() < WIDTH - 3 and corners[:, 1].max() < HEIGHT - 3:
                    seen.append(FiducialImageObservation(tag_id, corners + rng.normal(0, 0.05, corners.shape)))
            frames.append((start + time, IMAGE))
            detector.frames.append(seen)
        start += 1.0
    return frames, detector


def _layout_estimate(layout: dict, direction: str, readout: float, yaw: float, pitch: float, **options):
    frames, detector = _layout_recording(layout, direction, readout, yaw, pitch)
    return estimate_rolling_shutter(
        frames, calibration=CALIBRATION, tag_size_m=SIZE, detector=detector, tag_layout=layout, **options
    )


@pytest.mark.parametrize(
    ("direction", "yaw", "pitch"),
    [("top_to_bottom", 1.0, 0.0), ("bottom_to_top", 1.0, 0.0), ("left_to_right", 0.0, 1.0)],
)
def test_a_layout_reads_the_readout_from_tags_too_small_on_their_own(direction: str, yaw: float, pitch: float) -> None:
    layout = _wall(3, 3)
    frames, detector = _layout_recording(layout, direction, 0.02, yaw, pitch)
    alone = estimate_rolling_shutter(frames, calibration=CALIBRATION, tag_size_m=SIZE, detector=detector)
    assert alone.readout_s is None
    estimate = _layout_estimate(layout, direction, 0.02, yaw, pitch)
    assert estimate.rolling_shutter is not None and estimate.rolling_shutter.direction == direction
    assert estimate.readout_s == pytest.approx(0.02, abs=0.001)
    assert estimate.low <= estimate.readout_s <= estimate.high + 1e-3


def test_a_layout_reads_a_global_shutter_as_one() -> None:
    estimate = _layout_estimate(_wall(3, 3), "top_to_bottom", 0.0, 1.0, 0.0)
    assert estimate.readout_s == 0.0 and estimate.rolling_shutter is None


def test_tags_along_one_line_say_nothing() -> None:
    # A column of tags lets the pose roll to take up the shear, so no frame of it counts.
    estimate = _layout_estimate(_wall(1, 3), "top_to_bottom", 0.02, 1.0, 0.0)
    assert estimate.readout_s is None and estimate.sightings == 0


def test_rejects_an_invalid_layout() -> None:
    with pytest.raises(ValueError, match="tag_layout"):
        estimate_rolling_shutter([], calibration=CALIBRATION, tag_size_m=SIZE, tag_layout={-1: np.eye(4)})


def test_off_centre_corners_are_corrected_with_their_own_speeds() -> None:
    # Away from the centre a turn moves a tag's corners at different speeds; one speed for the tag would leave ~0.2 px.
    from napa_vision.processing import _ShutterCorrector

    points = tag_object_points(SIZE) @ cv2.Rodrigues(np.array([0.2, -0.4, 0.1]))[0].T + np.array([0.6, 0.1, 1.4])

    def project(at: float) -> np.ndarray:
        turned = (points @ cv2.Rodrigues(np.array([0.0, 2.0, 0.0]) * at)[0]) @ CALIBRATION.camera_matrix.T
        return turned[:, :2] / turned[:, 2:]

    corrector = _ShutterCorrector(RollingShutter(READOUT))
    for frame in range(3):
        time = frame * DT
        corners = project(time)
        for _ in range(8):
            times = time + (corners[:, 1] / HEIGHT - 0.5) * READOUT
            corners = np.array([project(times[k])[k] for k in range(4)])
        corrected, _ = corrector.correct([FiducialImageObservation(1, corners)], time, HEIGHT, WIDTH)
    truth = project(time)
    assert np.abs(corners - truth).max() > 1.0
    assert np.linalg.norm(np.asarray(corrected[0].corners) - truth, axis=1).max() < 0.08
