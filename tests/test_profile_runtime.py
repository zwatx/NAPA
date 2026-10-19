from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from napa_vision import (
    ArtifactProvenance,
    CameraCalibration,
    ProfileError,
    ProvenanceKind,
    load_profile,
    save_calibration_data,
)
from napa_vision.output import make_output_publisher
from napa_vision.recording import AsyncJsonlOutputPublisher


def _profile_data() -> dict[str, object]:
    source = resources.files("napa_vision.profiles").joinpath("laptop.json")
    return json.loads(source.read_text(encoding="utf-8"))


def _write_profile(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_bundled_profile_exposes_runtime_quality_and_provenance() -> None:
    profile = load_profile("laptop")
    assert profile.production is False
    assert profile.camera.calibration_provenance.kind is ProvenanceKind.EXAMPLE
    assert profile.tag_layout_provenance.kind is ProvenanceKind.EXAMPLE
    assert profile.camera.reconnect_initial_s == 0.1
    assert profile.camera_pose_bounds is not None
    assert profile.field_length_m == 6.0
    assert profile.jsonl_recorder.queue_size == 256
    assert profile.nt_send_all is True
    assert profile.nt_close_drain_s == pytest.approx(0.1)


def test_production_profile_rejects_example_artifacts(tmp_path: Path) -> None:
    data = _profile_data()
    data["production"] = True
    with pytest.raises(ProfileError, match="measured camera calibration"):
        load_profile(_write_profile(tmp_path, data))


def test_production_profile_accepts_explicit_measured_and_official_artifacts(tmp_path: Path) -> None:
    data = _profile_data()
    data["production"] = True
    calibration = data["camera"]["calibration"]  # type: ignore[index]
    calibration["provenance"] = {"kind": "measured", "source": "camera serial 123 calibration run"}
    calibration["rms_reprojection_error_px"] = 0.2
    calibration["views_used"] = 20
    data["tag_layout"]["provenance"] = {  # type: ignore[index]
        "kind": "official",
        "source": "verified WPILib season layout",
    }
    profile = load_profile(_write_profile(tmp_path, data))
    assert profile.production


def test_output_factory_uses_bounded_async_recorder(tmp_path: Path) -> None:
    profile = load_profile("laptop")
    publisher = make_output_publisher(
        "jsonl",
        profile.device_id,
        profile.nt_server,
        output_path=tmp_path / "frames.jsonl",
        jsonl_config=profile.jsonl_recorder,
    )
    assert isinstance(publisher, AsyncJsonlOutputPublisher)
    publisher.close()


def test_production_profile_loads_relative_calibration_and_wpilib_layout_artifacts(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    save_calibration_data(
        calibration_path,
        CameraCalibration(
            camera_matrix=np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]),
            distortion_coefficients=np.zeros(5),
            image_size=(1280, 720),
            provenance=ArtifactProvenance(ProvenanceKind.MEASURED, "camera serial 123 checkerboard run"),
            rms_reprojection_error_px=0.2,
            views_used=20,
        ),
    )
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(
        json.dumps(
            {
                "field": {"length": 16.5, "width": 8.2},
                "tags": [
                    {
                        "ID": 1,
                        "pose": {
                            "translation": {"x": 1.0, "y": 2.0, "z": 1.5},
                            "rotation": {"quaternion": {"W": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0}},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    data = _profile_data()
    data["production"] = True
    data["camera"]["calibration"] = {"file": calibration_path.name}  # type: ignore[index]
    data["tag_layout"] = {
        "wpilib_json": layout_path.name,
        "provenance": {"kind": "official", "source": "verified WPILib season layout"},
    }
    profile = load_profile(_write_profile(tmp_path, data))
    assert profile.production
    assert profile.field_length_m == 16.5
    assert profile.tag_layout_provenance.kind is ProvenanceKind.OFFICIAL
