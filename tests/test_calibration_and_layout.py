from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from napa_vision.calibration import (
    CalibrationError,
    CameraCalibration,
    CheckerboardSpec,
    calibrate_checkerboard_images,
    load_calibration_data,
    save_calibration_data,
)
from napa_vision.field_layout import FieldLayoutError, load_wpilib_apriltag_layout
from napa_vision.provenance import (
    ArtifactProvenance,
    ProvenanceKind,
    require_production_artifacts,
)


def _wpilib_layout() -> dict[str, object]:
    return {
        "field": {"length": 16.54, "width": 8.21},
        "tags": [
            {
                "ID": 1,
                "pose": {
                    "translation": {"x": 1.0, "y": 2.0, "z": 1.4},
                    "rotation": {"quaternion": {"W": 0.0, "X": 0.0, "Y": 0.0, "Z": 1.0}},
                },
            },
            {
                "id": 2,
                "pose": {
                    "translation": {"x": 15.0, "y": 6.0, "z": 1.2},
                    "rotation": {"quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}},
                },
            },
        ],
    }


def test_wpilib_layout_import_preserves_field_transforms_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "field.json"
    path.write_text(json.dumps(_wpilib_layout()), encoding="utf-8")

    imported = load_wpilib_apriltag_layout(path)
    assert imported.field_length_m == pytest.approx(16.54)
    assert imported.field_width_m == pytest.approx(8.21)
    assert imported.tag_poses[1][:3, 3] == pytest.approx([1.0, 2.0, 1.4])
    assert imported.tag_poses[1][:3, :3] == pytest.approx(np.diag([-1.0, -1.0, 1.0]))
    assert imported.provenance.kind is ProvenanceKind.IMPORTED
    with pytest.raises(FieldLayoutError, match="production"):
        imported.require_production()

    official = load_wpilib_apriltag_layout(
        path,
        provenance=ArtifactProvenance(ProvenanceKind.OFFICIAL, "WPILib season resource"),
    )
    official.require_production()
    copied = official.as_transform_dict()
    copied[1][0, 3] = 99.0
    assert official.tag_poses[1][0, 3] == pytest.approx(1.0)


def test_wpilib_layout_rejects_duplicate_ids_and_bad_quaternions(tmp_path: Path) -> None:
    data = _wpilib_layout()
    tags = data["tags"]
    assert isinstance(tags, list)
    tags[1]["id"] = 1
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FieldLayoutError, match="duplicate tag ID 1"):
        load_wpilib_apriltag_layout(path)

    data = _wpilib_layout()
    tags = data["tags"]
    assert isinstance(tags, list)
    tags[0]["pose"]["rotation"]["quaternion"]["W"] = 2.0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FieldLayoutError, match="normalized"):
        load_wpilib_apriltag_layout(path)


def test_versioned_calibration_round_trip_and_production_gate(tmp_path: Path) -> None:
    calibration = CameraCalibration(
        camera_matrix=np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 360.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.zeros(5),
        image_size=(1280, 720),
        provenance=ArtifactProvenance(ProvenanceKind.MEASURED, "camera A checkerboard run"),
        rms_reprojection_error_px=0.31,
        mean_reprojection_error_px=0.27,
        views_used=14,
        checkerboard=CheckerboardSpec(9, 6, 0.024),
    )
    path = tmp_path / "calibration.json"
    save_calibration_data(path, calibration)
    loaded = load_calibration_data(path)

    loaded.validate_for_image_size(1280, 720)
    loaded.require_production()
    assert loaded.camera_matrix == pytest.approx(calibration.camera_matrix)
    assert loaded.distortion_coefficients == pytest.approx(np.zeros(5))
    with pytest.raises(CalibrationError, match="1280x720"):
        loaded.validate_for_image_size(640, 480)

    layout_path = tmp_path / "field.json"
    layout_path.write_text(json.dumps(_wpilib_layout()), encoding="utf-8")
    layout = load_wpilib_apriltag_layout(
        layout_path,
        provenance=ArtifactProvenance(ProvenanceKind.OFFICIAL, "WPILib season resource"),
    )
    require_production_artifacts(calibration=loaded, field_layout=layout)


def test_checkerboard_calibration_reports_insufficient_complete_views(tmp_path: Path) -> None:
    images = []
    for index in range(3):
        path = tmp_path / f"blank-{index}.png"
        assert cv2.imwrite(str(path), np.full((480, 640), 127, dtype=np.uint8))
        images.append(path)

    with pytest.raises(CalibrationError, match="only 0 complete checkerboard views"):
        calibrate_checkerboard_images(images, CheckerboardSpec(7, 5, 0.025), minimum_views=3)


def test_calibration_rejects_invalid_intrinsics() -> None:
    with pytest.raises(CalibrationError, match="positive focal lengths"):
        CameraCalibration(
            camera_matrix=np.array([[0.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
            distortion_coefficients=np.zeros(5),
            image_size=(1280, 720),
            provenance=ArtifactProvenance(ProvenanceKind.MEASURED, "invalid run"),
            rms_reprojection_error_px=math.inf,
            views_used=10,
        )


def test_production_gates_require_artifact_specific_provenance(tmp_path: Path) -> None:
    calibration = CameraCalibration(
        camera_matrix=np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
        distortion_coefficients=np.zeros(5),
        image_size=(1280, 720),
        provenance=ArtifactProvenance(ProvenanceKind.OFFICIAL, "wrong kind"),
        rms_reprojection_error_px=0.2,
        views_used=10,
    )
    with pytest.raises(CalibrationError, match="measured"):
        calibration.require_production()

    layout_path = tmp_path / "field.json"
    layout_path.write_text(json.dumps(_wpilib_layout()), encoding="utf-8")
    layout = load_wpilib_apriltag_layout(
        layout_path,
        provenance=ArtifactProvenance(ProvenanceKind.MEASURED, "wrong kind"),
    )
    with pytest.raises(FieldLayoutError, match="official"):
        layout.require_production()
