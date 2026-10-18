from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .geometry import transform_matrix
from .simulation import FrameEffects, SimulationFrameSpec


def synthetic_wall_layout() -> dict[int, np.ndarray]:
    """Return four tags on one wall, facing cameras in the lower-X half-space."""

    return {
        1: transform_matrix([4.0, 0.6, 0.7], [0.0, 0.0, math.pi]),
        2: transform_matrix([4.0, 1.2, 1.5], [0.0, 0.0, math.pi]),
        3: transform_matrix([4.0, 1.8, 0.7], [0.0, 0.0, math.pi]),
        4: transform_matrix([4.0, 2.4, 1.5], [0.0, 0.0, math.pi]),
    }


def _camera_pose(position: Sequence[float], target: Sequence[float] = (4.0, 1.5, 1.1)) -> np.ndarray:
    position_array = np.asarray(position, dtype=np.float64)
    direction = np.asarray(target, dtype=np.float64) - position_array
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    pitch = -math.atan2(float(direction[2]), float(np.linalg.norm(direction[:2])))
    return transform_matrix(position_array, [0.0, pitch, yaw])


def standard_simulation_frames() -> tuple[SimulationFrameSpec, ...]:
    """Small deterministic suite covering normal, degraded, and invalid input."""

    all_tags = (1, 2, 3, 4)
    nominal = _camera_pose([0.5, 1.5, 1.0])
    return (
        SimulationFrameSpec(0, "clean", nominal, all_tags),
        SimulationFrameSpec(1, "left-low-oblique", _camera_pose([0.7, 0.85, 0.75]), all_tags),
        SimulationFrameSpec(2, "right-high-oblique", _camera_pose([0.7, 2.15, 1.25]), all_tags),
        SimulationFrameSpec(
            3,
            "blur-and-noise",
            nominal,
            all_tags,
            effects=FrameEffects(blur_sigma=1.2, noise_sigma=7.0),
        ),
        SimulationFrameSpec(
            4,
            "dark-motion-jpeg",
            _camera_pose([0.6, 1.4, 0.95]),
            all_tags,
            effects=FrameEffects(brightness=-30.0, contrast=0.9, motion_blur_px=3, jpeg_quality=75),
        ),
        SimulationFrameSpec(
            5,
            "partial-occlusion",
            nominal,
            all_tags,
            expected_tag_ids=(1, 3, 4),
            effects=FrameEffects(occluded_tag_id=2, occlusion_fraction=0.72),
        ),
        SimulationFrameSpec(
            6,
            "layout-outlier",
            nominal,
            all_tags,
            effects=FrameEffects(displaced_tag_id=4, displaced_tag_offset_px=(145.0, -95.0)),
            expected_rejected_tag_ids=(4,),
        ),
        SimulationFrameSpec(7, "single-tag", nominal, (2,), expect_pose=False),
        SimulationFrameSpec(8, "blank", nominal, (), expected_tag_ids=(), expect_pose=False),
    )


def pose_sweep_frames(frame_count: int = 31) -> tuple[SimulationFrameSpec, ...]:
    if frame_count < 2:
        raise ValueError("pose sweep requires at least two frames")
    frames: list[SimulationFrameSpec] = []
    for index, progress in enumerate(np.linspace(0.0, 1.0, frame_count)):
        y = 0.9 + 1.2 * float(progress)
        z = 0.9 + 0.18 * math.sin(float(progress) * math.pi)
        frames.append(SimulationFrameSpec(index, f"sweep-{index:03d}", _camera_pose([0.6, y, z]), (1, 2, 3, 4)))
    return tuple(frames)


def simulation_frames(suite: str) -> tuple[SimulationFrameSpec, ...]:
    if suite == "standard":
        return standard_simulation_frames()
    if suite == "sweep":
        return pose_sweep_frames()
    raise ValueError(f"Unknown simulation suite {suite!r}; expected standard or sweep")
