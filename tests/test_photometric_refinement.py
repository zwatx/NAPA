"""Frozen development regression: projected truth is independent of detected corners."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from napa_vision import NapaDetector
from napa_vision.solvers import CameraPoseSolver, SingleTagPoseSolver


def test_blurred_oblique_tag_preserves_camera_pose() -> None:
    # Crop of rendered metric scene 176: 4m, 25 degrees, sigma1.6, dim light.
    # Keep calibration, tag map and camera truth frozen, never select using truth.
    with np.load(Path(__file__).parent / "fixtures" / "photometric-oblique-pose.npz") as fixture:
        image = fixture["image"]
        camera = SimpleNamespace(
            calibration_matrix=fixture["calibration_matrix"], distortion_coefficients=fixture["distortion_coefficients"]
        )
        expected = fixture["field_to_camera"]
        layout = dict(zip(fixture["tag_ids"].tolist(), fixture["field_to_tags"], strict=True))
        size = float(fixture["tag_size_m"])
        tag_id = int(fixture["tag_id"])
        projected = fixture["projected_corners"]
    detector = NapaDetector()
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert [o.tag_id for o in observations] == [tag_id]
    assert np.linalg.norm(observations[0].corners - projected, axis=1).max() < 1.0
    singles = SingleTagPoseSolver(max_reprojection_error_px=5.0).solve_many(observations, camera, size)
    result = CameraPoseSolver(max_reprojection_error_px=5.0).solve(
        observations, [s for s in singles if s is not None], camera, size, layout
    )
    assert result is not None
    # The production residual policy chooses primary; truth only scores it here.
    estimate = result.field_to_camera_0
    assert np.linalg.norm(estimate[:3, 3] - expected[:3, 3]) < 0.1
    cosine = np.clip((np.trace(estimate[:3, :3].T @ expected[:3, :3]) - 1) / 2, -1, 1)
    assert np.degrees(np.arccos(cosine)) < 2.0
