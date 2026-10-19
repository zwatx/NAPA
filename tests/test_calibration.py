from __future__ import annotations

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
from napa_vision.provenance import ArtifactProvenance, ProvenanceKind


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


def test_production_gate_requires_measured_calibration() -> None:
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
