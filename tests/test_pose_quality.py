from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from northstar_vision.config import load_profile
from northstar_vision.geometry import (
    invert_transform,
    tag_field_corners,
    transform_matrix,
    transform_points,
    wpilib_points_to_opencv,
)
from northstar_vision.solvers import CameraPoseSolver
from northstar_vision.types import CameraPoseBounds, FiducialImageObservation


def _project_tag(
    field_to_camera: np.ndarray,
    field_to_tag: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    camera_to_field = invert_transform(field_to_camera)
    field_corners = tag_field_corners(field_to_tag, tag_size_m)
    camera_corners = transform_points(camera_to_field, field_corners)
    opencv_corners = wpilib_points_to_opencv(camera_corners)
    projected, _ = cv2.projectPoints(
        opencv_corners,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        distortion,
    )
    return projected.reshape(4, 2)


def _scene() -> tuple[object, np.ndarray, dict[int, np.ndarray], list[FiducialImageObservation]]:
    profile = load_profile("laptop")
    expected = transform_matrix([0.4, 1.0, 1.0], [0.0, 0.0, 0.0])
    layout = {
        1: transform_matrix([3.0, 0.25, 0.7], [0.0, 0.0, math.pi]),
        2: transform_matrix([3.0, 1.00, 1.4], [0.0, 0.0, math.pi]),
        3: transform_matrix([3.0, 1.75, 0.8], [0.0, 0.0, math.pi]),
        4: transform_matrix([3.0, 1.65, 1.5], [0.0, 0.0, math.pi]),
    }
    observations = [
        FiducialImageObservation(
            tag_id,
            _project_tag(
                expected,
                field_to_tag,
                profile.tag_size_m,
                profile.camera.calibration_matrix,
                profile.camera.distortion_coefficients,
            ),
        )
        for tag_id, field_to_tag in layout.items()
    ]
    return profile, expected, layout, observations


def test_multi_tag_solver_rejects_corrupted_tag_and_recovers_pose() -> None:
    profile, expected, layout, observations = _scene()
    observations[-1].corners = observations[-1].corners + np.array([260.0, -180.0])
    bounds = CameraPoseBounds(
        field_length_m=16.54,
        field_width_m=8.21,
        field_margin_m=0.25,
        min_camera_height_m=0.2,
        max_camera_height_m=2.0,
    )
    solver = CameraPoseSolver(
        max_reprojection_error_px=2.0,
        max_tag_reprojection_error_px=3.0,
        bounds=bounds,
    )

    result = solver.solve(observations, [], profile.camera, profile.tag_size_m, layout)

    assert result is not None
    assert result.field_to_camera_1 is None
    assert result.tag_ids == [1, 2, 3]
    assert result.field_to_camera_0[:3, 3] == pytest.approx(expected[:3, 3], abs=0.02)
    assert result.error_0 < 0.1
    assert result.diagnostics is solver.last_diagnostics
    assert result.diagnostics is not None
    assert result.diagnostics.used_ransac
    assert 4 in result.diagnostics.rejected_tag_reasons
    assert result.diagnostics.per_tag_rms_reprojection_error_px[4] > 100.0


def test_multi_tag_solver_reports_bounds_rejection() -> None:
    profile, _, layout, observations = _scene()
    solver = CameraPoseSolver(
        max_reprojection_error_px=2.0,
        bounds=CameraPoseBounds(max_camera_height_m=0.5),
    )

    result = solver.solve(observations[:3], [], profile.camera, profile.tag_size_m, layout)

    assert result is None
    assert solver.last_diagnostics.accepted_tag_ids == []
    assert solver.last_diagnostics.rejection_reason is not None
    assert "camera height" in solver.last_diagnostics.rejection_reason


def test_multi_tag_solver_explains_failure_when_two_tags_disagree() -> None:
    profile, _, layout, observations = _scene()
    observations[1].corners = observations[1].corners + np.array([180.0, 90.0])
    solver = CameraPoseSolver(max_reprojection_error_px=1.0, max_tag_reprojection_error_px=2.0)

    result = solver.solve(observations[:2], [], profile.camera, profile.tag_size_m, layout)

    assert result is None
    assert solver.last_diagnostics.rejection_reason is not None
    assert "reprojection error" in solver.last_diagnostics.rejection_reason
    assert solver.last_diagnostics.overall_rms_reprojection_error_px is not None
