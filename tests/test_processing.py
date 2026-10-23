from __future__ import annotations

import subprocess
import sys

import cv2
import numpy as np
import pytest

from napa_vision import (
    ArtifactProvenance,
    CalibrationError,
    CameraCalibration,
    FiducialImageObservation,
    ProvenanceKind,
    VisionProcessor,
)
from napa_vision.geometry import invert_transform, opencv_pose_to_wpilib_transform, tag_object_points


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.zeros(5),
        image_size=(640, 480),
        provenance=ArtifactProvenance(ProvenanceKind.EXAMPLE, "synthetic processor test"),
    )


class ProjectedDetector:
    """A structural backend with known camera-relative geometry, no camera SDK."""

    def __init__(self, calibration: CameraCalibration) -> None:
        self.corners, _ = cv2.projectPoints(
            tag_object_points(0.2),
            np.array([0.2, -0.1, 0.1]),
            np.array([0.1, 0.05, 2.0]),
            calibration.camera_matrix,
            calibration.distortion_coefficients,
        )
        self.calls = 0

    def __bool__(self) -> bool:
        return False

    def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
        self.calls += 1
        return [FiducialImageObservation(42, self.corners.reshape(4, 2).copy())]


def test_detector_works_without_opencv() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['cv2'] = None; import numpy as np; from napa_vision import NapaDetector; "
            "assert NapaDetector().detect(np.zeros((64, 64, 3), np.uint8)) == []",
        ],
        check=True,
    )


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_detection_only_accepts_strided_images_without_profile_or_calibration(channels: int) -> None:
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 9, 160)
    gray = np.full((240, 320), 255, dtype=np.uint8)
    gray[40:200, 80:240] = marker
    image = gray if channels == 1 else np.repeat(gray[:, :, None], channels, axis=2)
    backing = np.repeat(image, 2, axis=1)
    strided = backing[:, ::2]
    before = backing.copy()

    result = VisionProcessor().process(strided, timestamp=12.5, capture_monotonic_s=3.0)

    assert [observation.tag_id for observation in result.image_observations] == [9]
    assert np.min(result.image_observations[0].corners[:, 0]) == pytest.approx(80, abs=1)
    assert result.timestamp == 12.5 and result.capture_monotonic_s == 3.0
    assert result.single_tag_poses == [] and result.camera_pose is None and result.pose_diagnostics is None
    assert result.tag_angles == []
    assert set(result.health) == {"detection_ms", "solve_ms"}
    assert np.array_equal(backing, before)


def test_calibrated_processing_uses_custom_backend_without_field_layout() -> None:
    calibration = _calibration()
    detector = ProjectedDetector(calibration)
    processor = VisionProcessor(detector=detector, calibration=calibration, tag_size_m=0.2)
    image = np.full((480, 640), 255, dtype=np.uint8)

    result = processor.process(image, timestamp=1.0)

    assert detector.calls == 1
    assert [pose.tag_id for pose in result.single_tag_poses] == [42]
    assert np.allclose(result.single_tag_poses[0].pose_0_tvec.reshape(3), [0.1, 0.05, 2.0], atol=1e-6)
    assert [angle.tag_id for angle in result.tag_angles] == [42]
    assert result.camera_pose is None
    assert result.pose_diagnostics is not None
    assert result.pose_diagnostics.rejected_tag_reasons == {42: "tag is not present in the field layout"}


def test_optional_field_solving_preserves_prior_results() -> None:
    calibration = _calibration()
    layout = np.eye(4)
    layout[:3, 3] = [4.0, 1.0, 1.5]
    processor = VisionProcessor(
        detector=ProjectedDetector(calibration),
        calibration=calibration,
        tag_size_m=0.2,
        tag_layout={42: layout},
    )
    expected_layout = layout.copy()
    # The processor snapshots caller-supplied geometry, so another stream cannot alter it.
    layout[0, 3] = 100.0
    image = np.full((480, 640), 255, dtype=np.uint8)
    first = processor.process(image, timestamp=1.0)
    second = processor.process(image, timestamp=2.0)

    assert first.camera_pose is not None and first.camera_pose.tag_ids == [42]
    expected_pose = expected_layout @ invert_transform(
        opencv_pose_to_wpilib_transform(np.array([0.2, -0.1, 0.1]), np.array([0.1, 0.05, 2.0]))
    )
    assert np.allclose(first.camera_pose.field_to_camera_0, expected_pose, atol=1e-6)
    assert first.pose_diagnostics is not second.pose_diagnostics
    assert first.timestamp == 1.0


def test_empty_calibrated_frame_refreshes_diagnostics_after_tags_disappear(monkeypatch) -> None:
    calibration = _calibration()
    detector = ProjectedDetector(calibration)
    processor = VisionProcessor(detector=detector, calibration=calibration, tag_size_m=0.2, tag_layout={42: np.eye(4)})
    image = np.full((480, 640), 255, dtype=np.uint8)
    first = processor.process(image, timestamp=1.0)
    assert first.camera_pose is not None
    assert first.pose_diagnostics.accepted_tag_ids == [42]
    monkeypatch.setattr(detector, "detect", lambda image: [])

    empty = processor.process(image, timestamp=2.0)

    assert empty.image_observations == empty.single_tag_poses == empty.tag_angles == []
    assert empty.camera_pose is None
    assert empty.pose_diagnostics is not first.pose_diagnostics
    assert empty.pose_diagnostics.attempted_tag_ids == empty.pose_diagnostics.accepted_tag_ids == []
    assert empty.pose_diagnostics.rejected_tag_reasons == {}
    assert empty.pose_diagnostics.rejection_reason == "no unique observed tags are present in the field layout"
    assert first.pose_diagnostics.accepted_tag_ids == [42]


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (np.zeros((20, 20), dtype=np.float32), "8-bit"),
        (np.zeros((20, 20, 2), dtype=np.uint8), "grayscale, BGR, or BGRA"),
        (np.zeros((0, 20), dtype=np.uint8), "nonempty"),
    ],
)
def test_invalid_images_are_rejected_before_backend_calls(image: np.ndarray, message: str) -> None:
    detector = ProjectedDetector(_calibration())
    with pytest.raises(ValueError, match=message):
        VisionProcessor(detector=detector).process(image, timestamp=0.0)
    assert detector.calls == 0


@pytest.mark.parametrize("name", ["timestamp", "capture_monotonic_s"])
def test_nonfinite_metadata_is_rejected(name: str) -> None:
    values = {"timestamp": 1.0, name: float("nan")}
    with pytest.raises(ValueError, match=name):
        VisionProcessor().process(np.zeros((20, 20), dtype=np.uint8), **values)


def test_calibration_resolution_is_checked_before_detection() -> None:
    calibration = _calibration()
    detector = ProjectedDetector(calibration)
    with pytest.raises(CalibrationError, match="640x480, not 320x240"):
        VisionProcessor(detector=detector, calibration=calibration, tag_size_m=0.2).process(
            np.zeros((240, 320), dtype=np.uint8), timestamp=0.0
        )
    assert detector.calls == 0


def test_invalid_pose_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="provided together"):
        VisionProcessor(calibration=_calibration())
    with pytest.raises(ValueError, match="requires calibration"):
        VisionProcessor(tag_layout={42: np.eye(4)})
    with pytest.raises(ValueError, match="greater than zero"):
        VisionProcessor(calibration=_calibration(), tag_size_m=0.0)
    with pytest.raises(ValueError, match="tag_layout\\[42\\].*rigid"):
        VisionProcessor(calibration=_calibration(), tag_size_m=0.2, tag_layout={42: np.zeros((4, 4))})


def test_backend_failure_propagates_to_caller() -> None:
    class FailingDetector:
        def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
            raise RuntimeError("backend unavailable")

    with pytest.raises(RuntimeError, match="backend unavailable"):
        VisionProcessor(detector=FailingDetector()).process(np.zeros((20, 20), dtype=np.uint8), timestamp=1.0)
