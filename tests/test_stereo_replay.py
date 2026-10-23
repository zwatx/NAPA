from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from napa_vision.covariance import PoseWithCovariance
from napa_vision.stereo import StereoSolveDiagnostics
from napa_vision.types import FiducialImageObservation


@pytest.fixture()
def replay_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "stereo_replay.py"
    spec = importlib.util.spec_from_file_location("stereo_replay_test_module", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(tmp_path: Path, *, truth: bool = False) -> Path:
    image = np.full((40, 60), 255, dtype=np.uint8)
    image[10:30, 20:40] = 0
    assert cv2.imwrite(str(tmp_path / "left.png"), image)
    assert cv2.imwrite(str(tmp_path / "right.png"), image)
    transform = np.eye(4).tolist()
    if truth:
        transform[0][3] = 1.0
    manifest = {
        "schema_version": 1,
        "family": "tag36h11",
        "tag_size_m": 0.16,
        "tag_layout": {"1": transform},
        "pairs": [
            {
                "left": {"camera_id": "left", "path": "left.png", "capture_time_s": 1.0},
                "right": {"camera_id": "right", "path": "right.png", "capture_time_s": 1.002},
                **({"ground_truth": [{"timestamp_s": 1.001, "field_to_rig": transform}]} if truth else {}),
            }
        ],
    }
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class _Camera:
    def validate_for_image_size(self, width: int, height: int) -> None:
        assert (width, height) == (60, 40)


class _Calibration:
    left_id = "left"
    right_id = "right"
    left = _Camera()
    right = _Camera()


def _install_fakes(monkeypatch: pytest.MonkeyPatch, module: object) -> None:
    corners = np.array([[20.0, 10.0], [40.0, 10.0], [40.0, 30.0], [20.0, 30.0]])

    class FakeDetector:
        def __init__(self, *, family: str) -> None:
            assert family == "tag36h11"

        def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
            assert image.shape == (40, 60)
            return [FiducialImageObservation(1, corners)]

        def close(self) -> None:
            pass

    class FakeSolver:
        def __init__(self, *args: object, **kwargs: object) -> None:
            assert kwargs["max_capture_skew_s"] == 0.01

        def solve(self, left: object, right: object, **kwargs: object):
            diagnostics = StereoSolveDiagnostics(abs(kwargs["left_capture_time_s"] - kwargs["right_capture_time_s"]))
            if left and right:
                diagnostics.accepted_groups = [("left", 1), ("right", 1)]
                status = "stereo"
            else:
                diagnostics.fallback_reason = "stereo requires known tags in both cameras"
                status = "monocular"
            transform = np.eye(4)
            transform[0, 3] = 1.0
            return module.StereoPoseResult(
                1.001,
                status,
                (PoseWithCovariance(transform, np.eye(6)),),
                diagnostics,
            )

    monkeypatch.setattr(module, "NapaDetector", FakeDetector)
    monkeypatch.setattr(module, "StereoPoseSolver", FakeSolver)
    monkeypatch.setattr(module, "load_stereo_calibration", lambda path: _Calibration())


def test_paired_replay_reports_unknown_accuracy_without_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay_module: object
) -> None:
    _install_fakes(monkeypatch, replay_module)
    report = replay_module.run_stereo_replay(_write_manifest(tmp_path), tmp_path / "rig.json", max_capture_skew_s=0.01)

    pair = report["pairs"][0]
    assert pair["stereo"]["status"] == "stereo"
    assert pair["stereo"]["translation_error_m"] is None
    assert pair["stereo"]["rotation_error_deg"] is None
    assert pair["monocular"]["left"]["status"] == "monocular"
    assert pair["left"]["detections"] == pair["right"]["detections"] == 1
    assert report["timing"]["detection_ms"]["samples"] == 1
    assert report["timing"]["solver_ms"]["samples"] == 1
    json.dumps(report, allow_nan=False)


def test_paired_replay_scores_exact_synthetic_pose_only_when_truth_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay_module: object
) -> None:
    _install_fakes(monkeypatch, replay_module)
    report = replay_module.run_stereo_replay(
        _write_manifest(tmp_path, truth=True), tmp_path / "rig.json", max_capture_skew_s=0.01
    )

    result = report["pairs"][0]["stereo"]
    assert result["translation_error_m"] == 0.0
    assert result["rotation_error_deg"] == 0.0
    assert result["accepted_groups"] == [["left", 1], ["right", 1]]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value["pairs"][0]["left"].update(path="../outside.png"), "must not escape"),
        (lambda value: value.update(tag_size_m=float("nan")), "must be finite"),
        (lambda value: value["pairs"][0]["left"].update(camera_id="right"), "camera IDs must be distinct"),
    ],
)
def test_paired_manifest_rejects_unsafe_or_nonfinite_values(
    tmp_path: Path, replay_module: object, change: object, message: str
) -> None:
    path = _write_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(replay_module.StereoReplayManifestError, match=message):
        replay_module.load_stereo_replay_manifest(path)


def test_moving_rig_truth_is_scored_only_at_the_result_time(replay_module):
    from napa_vision.stereo import StereoPoseResult

    before = np.eye(4)
    after = np.eye(4)
    after[0, 3] = 0.5
    truth = (replay_module.ReplayTruth(1.0, before), replay_module.ReplayTruth(1.1, after))
    result = StereoPoseResult(1.1, "monocular", (PoseWithCovariance(after, np.eye(6)),), StereoSolveDiagnostics(0.1))
    matched = replay_module._result(result, truth)
    assert matched["translation_error_m"] == 0
    assert matched["conditional_nees"] == 0
    unmatched = replay_module._result(result, truth[:1])
    assert unmatched["translation_error_m"] is None
    assert unmatched["conditional_nees"] is None
    assert not unmatched["ground_truth_time_matched"]


def test_replay_runs_real_detector_calibration_io_and_joint_solver(tmp_path, replay_module):
    from dataclasses import replace

    from conftest import project_field_tag
    from test_stereo import TAG_SIZE, stereo_scene

    from napa_vision.stereo_calibration import save_stereo_calibration
    from napa_vision.tags import tag_image

    calibration, layout, truth, _ = stereo_scene()
    # A planar homography renderer assumes a pinhole image; distortion is covered in the numerical solver tests.
    calibration = replace(
        calibration,
        left=replace(calibration.left, distortion_coefficients=np.zeros(5)),
        right=replace(calibration.right, distortion_coefficients=np.zeros(5)),
    )
    for name, camera, mount in (
        ("left", calibration.left, calibration.rig_to_left),
        ("right", calibration.right, calibration.rig_to_right),
    ):
        # Integrate subpixel edges instead of quantizing each projected corner to a single pixel.
        scale = 4
        image = np.full((720 * scale, 1280 * scale), 255, np.uint8)
        for tag_id, tag in layout.items():
            corners = project_field_tag(
                truth @ mount, tag, TAG_SIZE, camera.camera_matrix, camera.distortion_coefficients
            )
            source = np.array([[40, 40], [360, 40], [360, 360], [40, 360]], np.float32) - 0.5
            destination = (corners[[2, 3, 0, 1]] + 0.5) * scale - 0.5
            warp = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
            drawn = cv2.warpPerspective(
                tag_image(tag_id, cell_px=40), warp, (1280 * scale, 720 * scale), borderValue=255
            )
            image = np.minimum(image, drawn)
        image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
        assert cv2.imwrite(str(tmp_path / f"{name}.png"), image)
    manifest = {
        "schema_version": 1,
        "family": "tag36h11",
        "tag_size_m": TAG_SIZE,
        "tag_layout": {str(key): value.tolist() for key, value in layout.items()},
        "pairs": [
            {
                "left": {"camera_id": "left", "path": "left.png", "capture_time_s": 1.0},
                "right": {"camera_id": "right", "path": "right.png", "capture_time_s": 1.0},
                "ground_truth": [{"timestamp_s": 1.0, "field_to_rig": truth.tolist()}],
            }
        ],
    }
    manifest_path = tmp_path / "pairs.json"
    manifest_path.write_text(json.dumps(manifest))
    calibration_path = tmp_path / "rig.json"
    save_stereo_calibration(calibration_path, calibration)
    report = replay_module.run_stereo_replay(manifest_path, calibration_path, max_capture_skew_s=0.01)
    row = report["pairs"][0]
    assert row["left"]["detections"] == row["right"]["detections"] == len(layout)
    for result in (row["stereo"], row["monocular"]["left"], row["monocular"]["right"], row["independent_pose_fusion"]):
        assert result["status"] != "rejected", result
        assert result["translation_error_m"] < 0.04
        assert result["rotation_error_deg"] < 1.0
    json.dumps(report, allow_nan=False)
