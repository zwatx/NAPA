from __future__ import annotations

import json
import math
import sys
import time
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import northstar_vision
from northstar_vision import (
    BlockSkipPrefilter,
    CameraPoseObservation,
    CameraPoseSolver,
    OpenCVArucoFiducialDetector,
    ProfileError,
    SingleTagPoseSolver,
    SkipScanConfig,
    TagAngleSolver,
    TrigDistanceSolver,
    available_profiles,
    load_profile,
)
from northstar_vision.calibration import load_calibration, save_calibration, undistorted_corner_angles
from northstar_vision.capture import ReplaySource
from northstar_vision.cli import main
from northstar_vision.geometry import (
    invert_transform,
    opencv_translation_to_wpilib,
    tag_field_corners,
    tag_object_points,
    transform_matrix,
    transform_points,
    wpilib_points_to_opencv,
)
from northstar_vision.output import (
    NTOutputPublisher,
    OutputPublisher,
    encode_observations,
    frame_result_to_dict,
    make_output_publisher,
)
from northstar_vision.recording import AsyncJsonlOutputPublisher, JsonlRecorderConfig
from northstar_vision.service import NorthstarVisionService
from northstar_vision.types import FiducialImageObservation, FrameResult, TagAngleObservation


def test_lazy_public_import_explains_missing_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = ModuleNotFoundError("No module named 'cv2'", name="cv2")

    def fail_import(*_args: object, **_kwargs: object) -> None:
        raise missing

    monkeypatch.setattr(northstar_vision, "import_module", fail_import)

    with pytest.raises(RuntimeError, match=r"northstar-vision\[gui\].*northstar-vision\[headless\]"):
        northstar_vision.__getattr__("OpenCVArucoFiducialDetector")


def make_marker_scene(tag_id: int = 1, *, size: int = 200) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, size, borderBits=1)
    image = np.full((400, 400), 255, dtype=np.uint8)
    offset = (400 - size) // 2
    image[offset : offset + size, offset : offset + size] = marker
    return image


def project_field_tag(
    field_to_camera: np.ndarray,
    field_to_tag: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    camera_to_field = invert_transform(field_to_camera)
    corners_field = tag_field_corners(field_to_tag, tag_size_m)
    corners_camera_wpilib = transform_points(camera_to_field, corners_field)
    corners_camera_opencv = wpilib_points_to_opencv(corners_camera_wpilib)
    corners, _ = cv2.projectPoints(
        corners_camera_opencv,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        distortion,
    )
    return corners.reshape(4, 2)


def test_profiles_are_packaged_and_valid() -> None:
    assert available_profiles() == ("laptop", "raspi")
    laptop = load_profile("laptop")
    raspi = load_profile("raspi")
    assert laptop.platform == "laptop"
    assert raspi.platform == "raspberry_pi"
    assert laptop.camera.width == 1280
    assert laptop.tag_layout
    assert raspi.output_mode == "nt"


def test_cli_lists_and_validates_bundled_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-profiles"]) == 0
    assert capsys.readouterr().out.splitlines() == ["laptop", "raspi"]
    assert main(["--profile", "laptop", "--check-config"]) == 0
    assert "is valid" in capsys.readouterr().out


def test_cli_lists_profiles_without_loading_opencv_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    from northstar_vision import cli

    original_import = cli.import_module

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "cv2":
            raise ModuleNotFoundError("No module named 'cv2'", name="cv2")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(cli, "import_module", guarded_import)
    assert main(["--list-profiles"]) == 0
    assert capsys.readouterr().out.splitlines() == ["laptop", "raspi"]
    assert main(["--check-config"]) == 0
    assert "is valid" in capsys.readouterr().out
    assert not caplog.text


def test_cli_constructor_rollback_closes_output_when_yaw_close_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from northstar_vision import orientation, output, service

    class FailingYaw:
        def close(self) -> None:
            raise RuntimeError("yaw close failed")

    class TrackingOutput(OutputPublisher):
        def __init__(self) -> None:
            self.closed = False

        def publish(self, result: FrameResult) -> None:
            del result

        def close(self) -> None:
            self.closed = True

    tracking_output = TrackingOutput()
    monkeypatch.setattr(output, "make_output_publisher", lambda *_args, **_kwargs: tracking_output)
    monkeypatch.setattr(orientation, "NetworkTablesRobotYawSource", lambda *_args, **_kwargs: FailingYaw())

    def fail_service(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("service setup failed")

    monkeypatch.setattr(service, "NorthstarVisionService", fail_service)

    assert main(["--output", "none", "--robot-yaw-topic", "/RobotState/yaw"]) == 1
    assert tracking_output.closed


def test_profile_validation_rejects_unknown_output_mode(tmp_path: Path) -> None:
    source = resources.files("northstar_vision.profiles").joinpath("laptop.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    data["output"]["mode"] = "typo"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProfileError, match="output.mode"):
        load_profile(path)


def test_profile_validation_rejects_duplicate_tag_ids(tmp_path: Path) -> None:
    source = resources.files("northstar_vision.profiles").joinpath("laptop.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    data["tag_layout"]["tags"].append(data["tag_layout"]["tags"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProfileError, match="Duplicate tag ID"):
        load_profile(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["camera"].__setitem__("id", -1), "must not be negative"),
        (lambda data: data["output"].__setitem__("path", 42), "output.path must be a non-empty string"),
        (
            lambda data: data["camera"]["calibration"].__setitem__(
                "camera_matrix", [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 1.0, 1.0]]
            ),
            "canonical pinhole form",
        ),
        (
            lambda data: data["camera"]["calibration"].__setitem__("distortion_coefficients", ["bad"]),
            "distortion_coefficients",
        ),
    ],
)
def test_profile_validation_rejects_malformed_typed_values(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    source = resources.files("northstar_vision.profiles").joinpath("laptop.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    mutation(data)  # type: ignore[operator]
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProfileError, match=message):
        load_profile(path)


def test_opencv_apriltag_detector_detects_36h11() -> None:
    observations = OpenCVArucoFiducialDetector().detect(make_marker_scene(1))
    assert len(observations) == 1
    assert observations[0].tag_id == 1
    assert observations[0].corners.shape == (4, 2)


def test_detector_validates_image_type() -> None:
    detector = OpenCVArucoFiducialDetector()
    with pytest.raises(ValueError, match="8-bit"):
        detector.detect(np.ones((20, 20), dtype=np.float32))


def test_no_tag_frame_is_empty() -> None:
    image = np.full((400, 400), 255, dtype=np.uint8)
    assert OpenCVArucoFiducialDetector().detect(image) == []


def test_uniformity_gate_does_not_miss_small_tag_between_sample_grid_points() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 1, 18, borderBits=1)
    image = np.full((720, 1280), 255, dtype=np.uint8)
    image[306:324, 601:619] = marker
    assert [observation.tag_id for observation in OpenCVArucoFiducialDetector().detect(image)] == [1]


def test_disabled_skip_scan_falls_back_to_full_frame() -> None:
    prefilter = BlockSkipPrefilter(SkipScanConfig(enabled=False))
    assert prefilter.find_rois(make_marker_scene(1)) is None
    observations = OpenCVArucoFiducialDetector(prefilter=prefilter).detect(make_marker_scene(1))
    assert [observation.tag_id for observation in observations] == [1]


def test_skip_scan_prefilter_skips_blank_frames_and_detects_marker() -> None:
    prefilter = BlockSkipPrefilter(SkipScanConfig(enabled=True))
    assert prefilter.find_rois(np.full((400, 400), 255, dtype=np.uint8)) == []
    observations = OpenCVArucoFiducialDetector(prefilter=prefilter).detect(make_marker_scene(1))
    assert [observation.tag_id for observation in observations] == [1]


def test_single_tag_pose_returns_ippe_solutions_and_distance() -> None:
    profile = load_profile("laptop")
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
    profile = load_profile("laptop")
    corners = np.array([[0.0, 0.0], [1200.0, 20.0], [30.0, 700.0], [1100.0, 650.0]])
    observation = FiducialImageObservation(1, corners)
    assert (
        SingleTagPoseSolver(max_reprojection_error_px=2.0).solve(observation, profile.camera, profile.tag_size_m)
        is None
    )


def test_tag_angle_solver_uses_normalized_coordinates_and_wpilib_translation() -> None:
    profile = load_profile("laptop")
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
    profile = load_profile("laptop")
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

    monkeypatch.setattr("northstar_vision.solvers.undistorted_corner_angles", counted)

    results = TagAngleSolver().solve_many(observations, profile.camera, singles)  # type: ignore[arg-type]

    assert [result.tag_id for result in results] == [1, 2]
    assert calls == 1


def test_tag_angle_solver_rejects_mismatched_pose_pair() -> None:
    profile = load_profile("laptop")
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
    profile = load_profile("laptop")
    corners = np.array([[600.0, 320.0], [680.0, 320.0], [680.0, 400.0], [600.0, 400.0]])
    angles = undistorted_corner_angles(
        corners,
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    assert angles.shape == (4, 2)
    assert float(np.mean(angles[:, 0])) == pytest.approx(0.0, abs=1e-12)
    assert float(np.mean(angles[:, 1])) == pytest.approx(0.0, abs=1e-12)


def test_calibration_round_trip(tmp_path: Path) -> None:
    profile = load_profile("laptop")
    path = tmp_path / "calibration.json"
    save_calibration(path, profile.camera.calibration_matrix, profile.camera.distortion_coefficients)
    camera_matrix, distortion = load_calibration(path)
    assert camera_matrix == pytest.approx(profile.camera.calibration_matrix)
    assert distortion == pytest.approx(profile.camera.distortion_coefficients)


def test_coordinate_basis_conversion() -> None:
    point = np.array([1.0, 2.0, 3.0])
    opencv = wpilib_points_to_opencv(point)[0]
    assert opencv == pytest.approx([-2.0, -3.0, 1.0])
    assert opencv_translation_to_wpilib(opencv) == pytest.approx(point)


def test_single_known_tag_preserves_two_global_camera_poses() -> None:
    profile = load_profile("laptop")
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
    profile = load_profile("laptop")
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
    profile = load_profile("laptop")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = np.full((720, 1280), 255, dtype=np.uint8)
    for tag_id, left in ((1, 783), (2, 423)):
        generated = cv2.aruco.generateImageMarker(dictionary, tag_id, 74, borderBits=1)
        official_orientation = cv2.rotate(generated, cv2.ROTATE_180)
        image[323:397, left : left + 74] = official_orientation

    observations = OpenCVArucoFiducialDetector().detect(image)
    layout = {
        1: transform_matrix([2.0, -0.4, 1.0], [0.0, 0.0, math.pi]),
        2: transform_matrix([2.0, 0.4, 1.0], [0.0, 0.0, math.pi]),
    }
    result = CameraPoseSolver().solve(observations, [], profile.camera, profile.tag_size_m, layout)
    assert result is not None
    assert result.tag_ids == [1, 2]
    assert result.field_to_camera_0[:3, 3] == pytest.approx([0.0, 0.0, 1.0], abs=0.005)
    assert result.error_0 < 1.0


def test_single_tag_camera_pose_suppresses_bad_secondary_ippe_candidate() -> None:
    profile = load_profile("laptop")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    generated = cv2.aruco.generateImageMarker(dictionary, 1, 74, borderBits=1)
    image = np.full((720, 1280), 255, dtype=np.uint8)
    image[323:397, 603:677] = cv2.rotate(generated, cv2.ROTATE_180)
    observation = OpenCVArucoFiducialDetector().detect(image)[0]
    single = SingleTagPoseSolver().solve(observation, profile.camera, profile.tag_size_m)
    assert single is not None
    assert single.error_1 is not None and single.error_1 > profile.max_reprojection_error_px

    layout = {1: transform_matrix([2.0, 0.0, 1.0], [0.0, 0.0, math.pi])}
    result = CameraPoseSolver().solve([observation], [single], profile.camera, profile.tag_size_m, layout)
    assert result is None
    angle = TagAngleSolver().solve(observation, profile.camera, single)
    encoded = encode_observations(FrameResult(timestamp=1.0, camera_pose=result, tag_angles=[angle]))
    assert encoded[0] == 0.0
    assert encoded[1] == 1.0


def test_trig_distance_solver_uses_full_pnp_translation_and_mount_transform() -> None:
    profile = load_profile("laptop")
    profile.camera.robot_to_camera = np.eye(4, dtype=np.float64)
    observation = TagAngleObservation(
        tag_id=1,
        corner_angles=np.zeros((4, 2)),
        center_tx=math.pi / 4.0,
        center_ty=0.0,
        distance=math.sqrt(2.0),
        camera_to_tag_translation=np.array([1.0, -1.0, 0.0]),
    )
    layout = {1: transform_matrix([1.0, -1.0, 0.0], [0.0, 0.0, math.pi])}
    result = TrigDistanceSolver().solve(observation, profile.camera, layout, robot_yaw_rad=0.0)
    assert result is not None
    assert result.robot_pose == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_northstar_output_schema_is_unambiguous_for_multi_tag_pose() -> None:
    camera_pose = CameraPoseObservation(
        tag_ids=[1, 2],
        field_to_camera_0=np.eye(4),
        error_0=0.2,
        field_to_camera_1=None,
        error_1=None,
        average_tag_distance=2.0,
        confidence=0.8,
    )
    angle = TagAngleObservation(1, np.zeros((4, 2)), 0.0, 0.0, 2.0, np.array([2.0, 0.0, 0.0]))
    encoded = encode_observations(FrameResult(timestamp=1.0, camera_pose=camera_pose, tag_angles=[angle]))
    assert encoded[0] == 1.0
    assert len(encoded) == 19
    assert encoded[9] == 1.0


def test_northstar_output_schema_preserves_single_tag_ambiguity() -> None:
    camera_pose = CameraPoseObservation(
        tag_ids=[1],
        field_to_camera_0=np.eye(4),
        error_0=0.1,
        field_to_camera_1=transform_matrix([0.1, 0.0, 0.0], [0.0, 0.0, 0.0]),
        error_1=0.2,
        average_tag_distance=2.0,
        confidence=0.8,
    )
    encoded = encode_observations(FrameResult(timestamp=1.0, camera_pose=camera_pose))
    assert encoded[0] == 2.0
    assert len(encoded) == 17
    assert encoded[1] == pytest.approx(0.1)
    assert encoded[9] == pytest.approx(0.2)


def test_json_output_is_complete_and_buffered(tmp_path: Path) -> None:
    profile = load_profile("laptop")
    corners, _ = cv2.projectPoints(
        tag_object_points(profile.tag_size_m),
        np.zeros((3, 1), dtype=np.float64),
        np.array([[0.0], [0.0], [2.0]], dtype=np.float64),
        profile.camera.calibration_matrix,
        profile.camera.distortion_coefficients,
    )
    observations = [FiducialImageObservation(1, corners.reshape(4, 2))]
    single = SingleTagPoseSolver().solve(observations[0], profile.camera, profile.tag_size_m)
    assert single is not None
    angle = TagAngleSolver().solve(observations[0], profile.camera, single)
    frame = FrameResult(timestamp=1.0, image_observations=observations, single_tag_poses=[single], tag_angles=[angle])
    payload = frame_result_to_dict(frame)
    assert payload["schema_version"] == 1
    assert payload["single_tag_poses"][0]["candidates"]
    assert payload["tag_angles"][0]["camera_to_tag_translation_m"]

    path = tmp_path / "logs" / "frames.jsonl"
    publisher = AsyncJsonlOutputPublisher(
        path,
        config=JsonlRecorderConfig(
            flush_interval_s=60.0,
            fsync_interval_s=None,
            max_file_size_bytes=None,
            rotation_interval_s=None,
        ),
        clock=lambda: 0.0,
    )
    publisher.publish(frame)
    publisher.close()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["schema_version"] == 1
    assert written["image_observations"][0]["tag_id"] == 1


def test_output_factory_rejects_unknown_modes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown output mode"):
        make_output_publisher("typo", "device", "server", output_path=tmp_path / "unexpected.jsonl")
    assert not (tmp_path / "unexpected.jsonl").exists()


def test_networktables_publisher_uses_capture_timestamp_and_releases_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePublisher:
        def __init__(self) -> None:
            self.values: list[tuple[object, ...]] = []
            self.closed = False

        def set(self, *values: object) -> None:
            self.values.append(values)

        def close(self) -> None:
            self.closed = True

    class FakeTopic:
        def __init__(self) -> None:
            self.publisher = FakePublisher()

        def publish(self, *_args: object, **_kwargs: object) -> FakePublisher:
            return self.publisher

    class FakeTable:
        def __init__(self) -> None:
            self.topics: dict[str, FakeTopic] = {}

        def _topic(self, name: str) -> FakeTopic:
            return self.topics.setdefault(name, FakeTopic())

        getDoubleArrayTopic = _topic
        getIntegerTopic = _topic
        getDoubleTopic = _topic
        getStringTopic = _topic

    class FakeInstance:
        def __init__(self) -> None:
            self.table = FakeTable()
            self.stopped = False
            self.flushed = False

        def setServer(self, _server: str) -> None:
            return None

        def startClient4(self, _identity: str) -> None:
            return None

        def getTable(self, _path: str) -> FakeTable:
            return self.table

        def isConnected(self) -> bool:
            return True

        def flush(self) -> None:
            self.flushed = True

        def stopClient(self) -> None:
            self.stopped = True

    instance = FakeInstance()

    class FakeNetworkTableInstance:
        destroyed = False

        @staticmethod
        def create() -> FakeInstance:
            return instance

        @staticmethod
        def destroy(_instance: FakeInstance) -> None:
            FakeNetworkTableInstance.destroyed = True

    fake_ntcore = SimpleNamespace(
        NetworkTableInstance=FakeNetworkTableInstance,
        PubSubOptions=lambda **kwargs: kwargs,
        _now=lambda: 2_000_000,
    )
    monkeypatch.setitem(sys.modules, "ntcore", fake_ntcore)
    publisher = NTOutputPublisher("device", "server", close_drain_s=0.0, clock=lambda: 2.0)
    publisher.publish(
        FrameResult(
            timestamp=1_700_000_000.0,
            capture_monotonic_s=1.25,
            health={"fps": 30.0, "latency_ms": 4.0},
        )
    )
    observations = instance.table.topics["observations"].publisher.values
    assert observations == [([0.0], 1_250_000)]
    publisher.close()
    assert instance.flushed and instance.stopped and FakeNetworkTableInstance.destroyed
    assert all(topic.publisher.closed for topic in instance.table.topics.values())


def test_networktables_constructor_rolls_back_partial_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePublisher:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    first_publisher = FakePublisher()

    class FakeTopic:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail

        def publish(self, *_args: object, **_kwargs: object) -> FakePublisher:
            if self.fail:
                raise RuntimeError("topic setup failed")
            return first_publisher

    class FakeTable:
        def getDoubleArrayTopic(self, _name: str) -> FakeTopic:
            return FakeTopic()

        def getIntegerTopic(self, _name: str) -> FakeTopic:
            return FakeTopic(fail=True)

    class FakeInstance:
        def __init__(self) -> None:
            self.stopped = False

        def setServer(self, _server: str) -> None:
            return None

        def startClient4(self, _identity: str) -> None:
            return None

        def getTable(self, _path: str) -> FakeTable:
            return FakeTable()

        def stopClient(self) -> None:
            self.stopped = True

    instance = FakeInstance()

    class FakeNetworkTableInstance:
        destroyed = False

        @staticmethod
        def create() -> FakeInstance:
            return instance

        @staticmethod
        def destroy(_instance: FakeInstance) -> None:
            FakeNetworkTableInstance.destroyed = True

    fake_ntcore = SimpleNamespace(
        NetworkTableInstance=FakeNetworkTableInstance,
        PubSubOptions=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "ntcore", fake_ntcore)
    with pytest.raises(RuntimeError, match="topic setup failed"):
        NTOutputPublisher("device", "server")
    assert first_publisher.closed
    assert instance.stopped
    assert FakeNetworkTableInstance.destroyed


def test_networktables_health_uses_cadence_and_shared_instance_ownership() -> None:
    class FakePublisher:
        def __init__(self) -> None:
            self.values: list[tuple[object, ...]] = []

        def set(self, *values: object) -> None:
            self.values.append(values)

        def close(self) -> None:
            return None

    class FakeTopic:
        def __init__(self) -> None:
            self.publisher = FakePublisher()

        def publish(self, *_args: object, **_kwargs: object) -> FakePublisher:
            return self.publisher

    class FakeTable:
        def __init__(self) -> None:
            self.topics: dict[str, FakeTopic] = {}

        def topic(self, name: str) -> FakeTopic:
            return self.topics.setdefault(name, FakeTopic())

        getDoubleArrayTopic = topic
        getIntegerTopic = topic
        getDoubleTopic = topic
        getStringTopic = topic

    class FakeInstance:
        def __init__(self) -> None:
            self.table = FakeTable()
            self.stopped = False

        def getTable(self, _path: str) -> FakeTable:
            return self.table

        def isConnected(self) -> bool:
            return True

        def flush(self) -> None:
            return None

        def stopClient(self) -> None:
            self.stopped = True

    destroyed = False

    class FakeNetworkTableInstance:
        @staticmethod
        def destroy(_instance: FakeInstance) -> None:
            nonlocal destroyed
            destroyed = True

    now = [0.0]
    instance = FakeInstance()
    fake_ntcore = SimpleNamespace(
        NetworkTableInstance=FakeNetworkTableInstance,
        PubSubOptions=lambda **kwargs: kwargs,
        _now=lambda: 1_000_000,
    )
    publisher = NTOutputPublisher(
        "device",
        "server",
        clock=lambda: now[0],
        close_drain_s=0.0,
        ntcore_module=fake_ntcore,
        instance=instance,
    )

    publisher.publish(
        FrameResult(
            timestamp=1.0,
            health={"fps": 1, "latency_ms": 1.0, "source_connected": True, "output": {"connected": True}},
        )
    )
    now[0] = 0.1
    publisher.publish(
        FrameResult(
            timestamp=2.0,
            health={"fps": 2, "latency_ms": 2.0, "source_connected": True, "output": {"connected": True}},
        )
    )
    health_values = instance.table.topics["health"].publisher.values
    assert len(health_values) == 1
    assert len(instance.table.topics["fps_apriltags"].publisher.values) == 2

    publisher.publish(
        FrameResult(
            timestamp=3.0,
            health={"fps": 3, "latency_ms": 3.0, "source_connected": True, "output": {"connected": False}},
        )
    )
    assert len(health_values) == 2
    now[0] = 1.2
    publisher.publish(
        FrameResult(
            timestamp=4.0,
            health={"fps": 4, "latency_ms": 4.0, "source_connected": True, "output": {"connected": False}},
        )
    )
    assert len(health_values) == 3
    publisher.close()
    assert not instance.stopped
    assert not destroyed


def test_image_directory_replay_stops_unless_looping(tmp_path: Path) -> None:
    cv2.imwrite(str(tmp_path / "frame.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    source = ReplaySource(tmp_path)
    assert source.read()[0]
    assert not source.read()[0]
    source.release()

    looping = ReplaySource(tmp_path, loop=True)
    assert looping.read()[0]
    assert looping.read()[0]


def test_empty_replay_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="contains no"):
        ReplaySource(tmp_path)


class _StaticFrameSource:
    is_live = False

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.read_count = 0
        self.released = False

    def read(self) -> tuple[bool, float, float, np.ndarray | None]:
        self.read_count += 1
        if self.read_count == 1:
            return True, 1.0, time.perf_counter(), self.frame
        return False, 2.0, time.perf_counter(), None

    def release(self) -> None:
        self.released = True


class _CollectingPublisher(OutputPublisher):
    def __init__(self) -> None:
        self.results: list[FrameResult] = []
        self.closed = False

    def publish(self, result: FrameResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.closed = True


def test_service_can_run_without_camera_and_reports_stage_timing() -> None:
    profile = load_profile("laptop")
    source = _StaticFrameSource(np.full((720, 1280), 255, dtype=np.uint8))
    output = _CollectingPublisher()
    service = NorthstarVisionService(profile, output, frame_source=source)
    service.run()
    assert len(output.results) == 1
    assert output.results[0].health["source_connected"] is True
    assert output.results[0].health["acquire_wait_ms"] >= 0.0
    assert output.results[0].health["processing_ms"] >= 0.0
    assert output.results[0].health["frame_age_ms"] >= 0.0
    assert output.closed and source.released


def test_service_zero_frame_limit_does_not_read() -> None:
    profile = load_profile("laptop")
    source = _StaticFrameSource(np.full((720, 1280), 255, dtype=np.uint8))
    output = _CollectingPublisher()
    NorthstarVisionService(profile, output, frame_source=source).run(max_frames=0)
    assert source.read_count == 0
    assert output.closed and source.released


def test_service_negative_frame_limit_still_releases_resources() -> None:
    profile = load_profile("laptop")
    source = _StaticFrameSource(np.full((720, 1280), 255, dtype=np.uint8))
    output = _CollectingPublisher()
    service = NorthstarVisionService(profile, output, frame_source=source)
    with pytest.raises(ValueError, match="must not be negative"):
        service.run(max_frames=-1)
    assert source.read_count == 0
    assert output.closed and source.released


def test_service_rejects_frames_that_do_not_match_calibration_resolution() -> None:
    profile = load_profile("laptop")
    source = _StaticFrameSource(np.full((480, 640), 255, dtype=np.uint8))
    output = _CollectingPublisher()
    service = NorthstarVisionService(profile, output, frame_source=source)
    with pytest.raises(RuntimeError, match="profile calibration requires"):
        service.run()
    assert output.closed and source.released
