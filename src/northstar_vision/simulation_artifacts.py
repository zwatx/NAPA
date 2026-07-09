from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .benchmark import write_benchmark_summary
from .config import PlatformProfile
from .simulation import RenderedFrame

SIMULATION_SCHEMA_VERSION = 1
SIMULATION_LIMITATIONS = (
    "Synthetic images do not validate a physical lens, exposure, rolling shutter, or calibration.",
    "Desktop timings do not predict Raspberry Pi thermal or camera/USB behavior.",
    "OpenCV generates and detects the same tag dictionary, so recall is optimistic.",
    "Tag planes use a homographic distortion approximation; large nonzero lens distortion is not modeled exactly.",
)


def write_simulation_corpus(
    frames: tuple[RenderedFrame, ...],
    output_dir: str | Path,
    profile: PlatformProfile,
    tag_layout: dict[int, np.ndarray],
    *,
    suite: str,
) -> dict[str, Path]:
    """Write replay images, strict corpus metadata, truth, and a replay profile."""

    if not frames:
        raise ValueError("simulation corpus requires at least one frame")
    root = Path(output_dir).expanduser().resolve()
    image_dir = root / "frames"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_frames: list[dict[str, object]] = []
    truth_frames: list[dict[str, object]] = []
    for frame in frames:
        filename = f"{frame.truth.index:03d}-{frame.truth.name}.png"
        encoded = _encode_png(frame.image)
        _atomic_write_bytes(image_dir / filename, encoded)
        manifest_frames.append(
            {
                "path": f"frames/{filename}",
                "expected_tag_ids": list(frame.truth.expected_tag_ids),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        truth_frames.append(frame.truth.to_dict())

    manifest_path = write_benchmark_summary(
        {
            "schema_version": 1,
            "name": f"northstar-synthetic-{suite}",
            "description": "Deterministic synthetic evidence; it is not a substitute for robot-camera testing.",
            "frames": manifest_frames,
        },
        root / "corpus.json",
        pretty=True,
    )
    truth_path = write_benchmark_summary(
        {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "evidence_kind": "synthetic",
            "suite": suite,
            "limitations": list(SIMULATION_LIMITATIONS),
            "frames": truth_frames,
        },
        root / "ground-truth.json",
        pretty=True,
    )
    profile_path = write_benchmark_summary(
        _simulation_profile(profile, tag_layout),
        root / "simulation-profile.json",
        pretty=True,
    )
    return {"manifest": manifest_path, "truth": truth_path, "profile": profile_path, "frames": image_dir}


def _simulation_profile(profile: PlatformProfile, tag_layout: dict[int, np.ndarray]) -> dict[str, object]:
    camera = profile.camera
    tags = []
    for tag_id, transform in sorted(tag_layout.items()):
        tags.append(
            {
                "id": tag_id,
                "pose": {
                    "translation_m": transform[:3, 3].tolist(),
                    "rotation_rpy_rad": list(_matrix_to_rpy(transform[:3, :3])),
                },
            }
        )
    return {
        "schema_version": 1,
        "name": "synthetic-replay",
        "device_id": "northstar_sim",
        "platform": "simulation",
        "production": False,
        "tag_size_m": profile.tag_size_m,
        "camera": {
            "id": 0,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "calibration": {
                "camera_matrix": camera.calibration_matrix.tolist(),
                "distortion_coefficients": camera.distortion_coefficients.tolist(),
                "provenance": {"kind": "example", "source": f"synthetic copy of {profile.name}"},
            },
            "robot_to_camera": {"translation_m": [0.0, 0.0, 0.0], "rotation_rpy_rad": [0.0, 0.0, 0.0]},
        },
        "tag_layout": {
            "provenance": {"kind": "example", "source": "generated four-tag simulation wall"},
            "field": {"length_m": 5.0, "width_m": 3.0},
            "tags": tags,
        },
        "detection": {
            "max_reprojection_error_px": profile.max_reprojection_error_px,
            "max_tag_reprojection_error_px": profile.max_tag_reprojection_error_px,
            "max_outlier_iterations": profile.max_outlier_iterations,
            "ransac_iterations": profile.ransac_iterations,
            "skip_scan": asdict(profile.skip_scan),
        },
        "output": {"mode": "none"},
    }


def _matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
    return roll, pitch, yaw


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("OpenCV could not encode a simulated frame")
    return encoded.tobytes()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary_name = file.name
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
