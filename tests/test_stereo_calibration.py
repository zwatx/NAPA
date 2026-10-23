from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from napa_vision.calibration import CalibrationError, CameraCalibration
from napa_vision.geometry import OPENCV_TO_WPILIB_BASIS, invert_transform
from napa_vision.provenance import ArtifactProvenance, ProvenanceKind
from napa_vision.stereo_calibration import (
    StereoCalibration,
    calibrate_stereo_points,
    load_stereo_calibration,
    save_stereo_calibration,
)


def _camera(
    *, size: tuple[int, int] = (1280, 720), kind: ProvenanceKind = ProvenanceKind.MEASURED
) -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array([[920.0, 0.0, 640.0], [0.0, 915.0, 360.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.array([-0.08, 0.02, 0.001, -0.0007, 0.003]),
        image_size=size,
        provenance=ArtifactProvenance(kind, "checkerboard run"),
        rms_reprojection_error_px=0.25,
        views_used=12,
    )


def test_stereo_artifact_round_trip_production_and_immutable_snapshots(tmp_path: Path) -> None:
    left = _camera()
    right = _camera()
    right_pose = np.eye(4)
    right_pose[:3, 3] = [0.0, -0.22, 0.03]
    calibration = StereoCalibration(
        left=left,
        right=right,
        rig_to_left=np.eye(4),
        rig_to_right=right_pose,
        provenance=ArtifactProvenance(ProvenanceKind.MEASURED, "paired checkerboard run"),
        left_id="front-left",
        right_id="front-right",
        rms_reprojection_error_px=0.18,
        views_used=11,
    )
    assert calibration.left is not left
    assert calibration.left.camera_matrix is not left.camera_matrix
    assert not calibration.left.camera_matrix.flags.writeable
    assert not calibration.rig_to_right.flags.writeable
    right_pose[1, 3] = 99.0
    assert calibration.rig_to_right[1, 3] == pytest.approx(-0.22)

    path = tmp_path / "stereo.json"
    save_stereo_calibration(path, calibration)
    loaded = load_stereo_calibration(path)
    loaded.require_production()
    assert loaded.left_id == "front-left"
    assert loaded.right_id == "front-right"
    assert loaded.rig_to_right == pytest.approx(calibration.rig_to_right)
    assert loaded.left.distortion_coefficients == pytest.approx(left.distortion_coefficients)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationError, match="unknown stereo calibration field"):
        load_stereo_calibration(path)


def test_stereo_artifact_rejects_invalid_ids_transforms_and_provenance() -> None:
    left = _camera()
    with pytest.raises(CalibrationError, match="distinct"):
        StereoCalibration(
            left=left,
            right=_camera(),
            rig_to_left=np.eye(4),
            rig_to_right=np.eye(4),
            provenance=ArtifactProvenance(ProvenanceKind.MEASURED, "paired run"),
            left_id="camera",
            right_id="camera",
        )
    with pytest.raises(CalibrationError, match="rigid"):
        StereoCalibration(
            left=left,
            right=_camera(),
            rig_to_left=np.eye(4),
            rig_to_right=np.zeros((4, 4)),
            provenance=ArtifactProvenance(ProvenanceKind.MEASURED, "paired run"),
        )
    unmeasured = StereoCalibration(
        left=_camera(kind=ProvenanceKind.EXAMPLE),
        right=_camera(),
        rig_to_left=np.eye(4),
        rig_to_right=np.eye(4),
        provenance=ArtifactProvenance(ProvenanceKind.EXAMPLE, "example rig"),
    )
    with pytest.raises(CalibrationError, match="measured"):
        unmeasured.require_production()


@pytest.mark.parametrize("scope", ["rig", "left", "right"])
def test_stereo_loader_wraps_invalid_provenance_as_calibration_error(tmp_path: Path, scope: str) -> None:
    calibration = StereoCalibration(
        _camera(), _camera(), np.eye(4), np.eye(4), ArtifactProvenance(ProvenanceKind.MEASURED, "paired run")
    )
    path = tmp_path / "rig.json"
    save_stereo_calibration(path, calibration)
    payload = json.loads(path.read_text())
    target = payload if scope == "rig" else payload[scope]
    target["provenance"]["kind"] = "invalid"
    path.write_text(json.dumps(payload))
    with pytest.raises(CalibrationError, match="provenance.kind"):
        load_stereo_calibration(path)


def test_fixed_intrinsic_stereo_fit_recovers_rotated_right_camera_with_distortion() -> None:
    left = _camera()
    right = CameraCalibration(
        camera_matrix=np.array([[905.0, 0.0, 636.0], [0.0, 910.0, 358.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.array([-0.06, 0.012, -0.0008, 0.0005, -0.001]),
        image_size=(1280, 720),
        provenance=ArtifactProvenance(ProvenanceKind.MEASURED, "right checkerboard run"),
        rms_reprojection_error_px=0.24,
        views_used=12,
    )
    grid = np.array([[x * 0.03, y * 0.03, 0.0] for y in range(5) for x in range(7)], dtype=np.float64)
    right_rotation, _ = cv2.Rodrigues(np.array([0.018, -0.031, 0.012]))
    right_translation = np.array([-0.21, 0.006, -0.014])
    object_views: list[np.ndarray] = []
    left_views: list[np.ndarray] = []
    right_views: list[np.ndarray] = []
    for rotation_vector, translation_vector in (
        ([0.08, -0.04, 0.03], [-0.08, 0.03, 1.25]),
        ([-0.05, 0.09, -0.06], [0.06, -0.02, 1.45]),
        ([0.11, 0.03, 0.07], [-0.02, 0.08, 1.7]),
        ([-0.09, -0.06, 0.04], [0.1, 0.04, 1.35]),
        ([0.04, -0.1, -0.03], [-0.12, -0.05, 1.6]),
        ([0.02, 0.06, -0.08], [0.04, -0.08, 1.5]),
    ):
        left_rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector))
        left_translation = np.asarray(translation_vector)
        left_pixels, _ = cv2.projectPoints(
            grid, np.asarray(rotation_vector), left_translation, left.camera_matrix, left.distortion_coefficients
        )
        right_pose_rotation = right_rotation @ left_rotation
        right_pose_translation = right_rotation @ left_translation + right_translation
        right_pixels, _ = cv2.projectPoints(
            grid,
            cv2.Rodrigues(right_pose_rotation)[0],
            right_pose_translation,
            right.camera_matrix,
            right.distortion_coefficients,
        )
        object_views.append(grid)
        left_views.append(left_pixels.reshape(-1, 2))
        right_views.append(right_pixels.reshape(-1, 2))

    fitted = calibrate_stereo_points(object_views, left_views, right_views, left, right, source="synthetic paired run")

    left_to_right_opencv = np.eye(4)
    left_to_right_opencv[:3, :3] = right_rotation
    left_to_right_opencv[:3, 3] = right_translation
    expected_right_pose = invert_transform(OPENCV_TO_WPILIB_BASIS @ left_to_right_opencv @ OPENCV_TO_WPILIB_BASIS.T)
    assert fitted.rig_to_left == pytest.approx(np.eye(4))
    assert fitted.rig_to_right == pytest.approx(expected_right_pose, abs=3e-4)
    assert fitted.rms_reprojection_error_px == pytest.approx(0.0, abs=2e-4)
    assert fitted.views_used == 6


def test_stereo_fit_rejects_unmatched_or_nonfinite_points() -> None:
    camera = _camera()
    world = [np.zeros((4, 3)) for _ in range(3)]
    pixels = [np.zeros((4, 2)) for _ in range(3)]
    pixels[1][0, 0] = np.nan
    with pytest.raises(CalibrationError, match="finite"):
        calibrate_stereo_points(world, pixels, pixels, camera, camera, source="paired run")
    valid_pixels = [np.zeros((4, 2)) for _ in range(3)]
    with pytest.raises(CalibrationError, match="at least 3"):
        calibrate_stereo_points(world[:2], valid_pixels[:2], valid_pixels[:2], camera, camera, source="paired run")
