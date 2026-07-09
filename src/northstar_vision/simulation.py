from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from .capture import FrameSource


@dataclass(frozen=True, slots=True)
class FrameEffects:
    """Deterministic image degradations applied after rendering."""

    brightness: float = 0.0
    contrast: float = 1.0
    blur_sigma: float = 0.0
    noise_sigma: float = 0.0
    motion_blur_px: int = 0
    jpeg_quality: int | None = None
    occluded_tag_id: int | None = None
    occlusion_fraction: float = 0.0
    displaced_tag_id: int | None = None
    displaced_tag_offset_px: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        values = (self.brightness, self.contrast, self.blur_sigma, self.noise_sigma, *self.displaced_tag_offset_px)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("simulation effects must contain finite numbers")
        if self.contrast <= 0.0 or self.blur_sigma < 0.0 or self.noise_sigma < 0.0:
            raise ValueError("contrast must be positive and blur/noise must be non-negative")
        if self.motion_blur_px < 0:
            raise ValueError("motion_blur_px must not be negative")
        if self.jpeg_quality is not None and not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 0.0 <= self.occlusion_fraction <= 1.0:
            raise ValueError("occlusion_fraction must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SimulationFrameSpec:
    index: int
    name: str
    field_to_camera: np.ndarray
    tag_ids: tuple[int, ...]
    expected_tag_ids: tuple[int, ...] | None = None
    effects: FrameEffects = field(default_factory=FrameEffects)
    expect_pose: bool = True
    expected_rejected_tag_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        pose = np.asarray(self.field_to_camera, dtype=np.float64)
        if self.index < 0 or not self.name.strip():
            raise ValueError("simulation frame index and name must be valid")
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("field_to_camera must be a finite 4x4 transform")
        if len(self.tag_ids) != len(set(self.tag_ids)):
            raise ValueError("tag_ids must not contain duplicates")
        if self.expected_tag_ids is not None and len(self.expected_tag_ids) != len(set(self.expected_tag_ids)):
            raise ValueError("expected_tag_ids must not contain duplicates")


@dataclass(frozen=True, slots=True)
class SimulationTruth:
    index: int
    name: str
    field_to_camera: np.ndarray
    rendered_tag_ids: tuple[int, ...]
    expected_tag_ids: tuple[int, ...]
    expected_rejected_tag_ids: tuple[int, ...]
    image_corners: dict[int, np.ndarray]
    layout_corners: dict[int, np.ndarray]
    effects: FrameEffects
    expect_pose: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "field_to_camera": self.field_to_camera.tolist(),
            "rendered_tag_ids": list(self.rendered_tag_ids),
            "expected_tag_ids": list(self.expected_tag_ids),
            "expected_rejected_tag_ids": list(self.expected_rejected_tag_ids),
            "image_corners": {str(key): value.tolist() for key, value in self.image_corners.items()},
            "layout_corners": {str(key): value.tolist() for key, value in self.layout_corners.items()},
            "effects": asdict(self.effects),
            "expect_pose": self.expect_pose,
        }


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    image: np.ndarray
    truth: SimulationTruth


class SyntheticFrameSource(FrameSource):
    """Finite in-memory camera source used by the normal vision service."""

    def __init__(self, frames: Sequence[RenderedFrame]) -> None:
        if not frames:
            raise ValueError("synthetic frame source requires at least one frame")
        self.frames = tuple(frames)
        self.index = 0
        self.closed = False

    def read(self) -> tuple[bool, float, float, np.ndarray | None]:
        now_wall, now_monotonic = time.time(), time.perf_counter()
        if self.closed or self.index >= len(self.frames):
            return False, now_wall, now_monotonic, None
        frame = self.frames[self.index].image
        self.index += 1
        return True, now_wall, now_monotonic, frame

    def release(self) -> None:
        self.closed = True

    def health(self) -> dict[str, object]:
        return {
            "source": "synthetic",
            "source_connected": not self.closed,
            "frames_delivered": self.index,
            "frames_available": len(self.frames),
        }
