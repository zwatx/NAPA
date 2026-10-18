from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .provenance import ArtifactProvenance, ProvenanceError, ProvenanceKind


class FieldLayoutError(ValueError):
    """Raised when a WPILib AprilTag field layout is malformed or unsafe."""


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise FieldLayoutError(f"{context} must be a JSON object")
    return value


def _number(value: object, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FieldLayoutError(f"{context} must be a number")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and greater than zero" if positive else "finite"
        raise FieldLayoutError(f"{context} must be {qualifier}")
    return result


def _component(data: Mapping[str, object], lower: str, context: str) -> float:
    upper = lower.upper()
    present = [key for key in (lower, upper) if key in data]
    if len(present) != 1:
        raise FieldLayoutError(f"{context} must contain exactly one of {lower!r} or {upper!r}")
    return _number(data[present[0]], f"{context}.{present[0]}")


def _quaternion_rotation(w: float, x: float, y: float, z: float, context: str) -> np.ndarray:
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or abs(norm - 1.0) > 1e-3:
        raise FieldLayoutError(f"{context} must be a normalized, non-zero quaternion")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _validate_transform(value: object, context: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FieldLayoutError(f"{context} must be a finite rigid 4x4 transform") from exc
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise FieldLayoutError(f"{context} must be a finite rigid 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise FieldLayoutError(f"{context} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-7
    ):
        raise FieldLayoutError(f"{context} rotation must be orthonormal and right-handed")
    result = transform.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class AprilTagFieldLayout:
    """Validated WPILib field dimensions and field-to-tag transforms."""

    tag_poses: Mapping[int, np.ndarray]
    field_length_m: float
    field_width_m: float
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.tag_poses, Mapping):
            raise FieldLayoutError("tag_poses must be a mapping of tag IDs to transforms")
        if not isinstance(self.provenance, ArtifactProvenance):
            raise FieldLayoutError("provenance must be an ArtifactProvenance")
        length = _number(self.field_length_m, "field_length_m", positive=True)
        width = _number(self.field_width_m, "field_width_m", positive=True)
        validated: dict[int, np.ndarray] = {}
        for tag_id, transform in self.tag_poses.items():
            if isinstance(tag_id, bool) or not isinstance(tag_id, int) or tag_id < 0:
                raise FieldLayoutError("tag IDs must be non-negative integers")
            if tag_id in validated:
                raise FieldLayoutError(f"duplicate tag ID {tag_id}")
            validated[tag_id] = _validate_transform(transform, f"tag {tag_id} pose")
        if not validated:
            raise FieldLayoutError("field layout must contain at least one tag")
        object.__setattr__(self, "field_length_m", length)
        object.__setattr__(self, "field_width_m", width)
        object.__setattr__(self, "tag_poses", MappingProxyType(validated))

    def as_transform_dict(self) -> dict[int, np.ndarray]:
        """Return copies in the exact mapping shape consumed by the solvers."""

        return {tag_id: transform.copy() for tag_id, transform in self.tag_poses.items()}

    def validate_tag_bounds(self, *, margin_m: float = 1.0, min_height_m: float = 0.0) -> None:
        """Check surveyed tag origins against plausible field extents."""

        if (
            isinstance(margin_m, bool)
            or not isinstance(margin_m, (int, float))
            or not np.isfinite(margin_m)
            or margin_m < 0.0
        ):
            raise FieldLayoutError("margin_m must be finite and non-negative")
        if (
            isinstance(min_height_m, bool)
            or not isinstance(min_height_m, (int, float))
            or not np.isfinite(min_height_m)
        ):
            raise FieldLayoutError("min_height_m must be finite")
        for tag_id, pose in self.tag_poses.items():
            x, y, z = pose[:3, 3]
            if not (-margin_m <= x <= self.field_length_m + margin_m):
                raise FieldLayoutError(f"tag {tag_id} x={x:.3f}m is outside field length bounds")
            if not (-margin_m <= y <= self.field_width_m + margin_m):
                raise FieldLayoutError(f"tag {tag_id} y={y:.3f}m is outside field width bounds")
            if z < min_height_m:
                raise FieldLayoutError(f"tag {tag_id} z={z:.3f}m is below {min_height_m:.3f}m")

    def require_production(self) -> None:
        try:
            self.provenance.require_kind(ProvenanceKind.OFFICIAL, "AprilTag field layout")
        except ProvenanceError as exc:
            raise FieldLayoutError(str(exc)) from exc


def load_wpilib_apriltag_layout(
    path: str | Path,
    *,
    provenance: ArtifactProvenance | None = None,
    validate_bounds: bool = True,
) -> AprilTagFieldLayout:
    """Import a WPILib AprilTag JSON layout without changing its field axes.

    WPILib JSON translations and quaternions already use the same +X forward,
    +Y left, +Z up field convention as this package. Each returned matrix is a
    field-to-tag transform and can be passed directly to ``CameraPoseSolver``.

    A parsed file defaults to ``IMPORTED`` rather than ``OFFICIAL``. Callers
    must supply explicit official provenance after verifying the file source.
    """

    source = Path(path).expanduser()
    try:
        with source.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise FieldLayoutError(f"field layout not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise FieldLayoutError(f"invalid field-layout JSON at line {exc.lineno}: {exc.msg}") from exc
    except OSError as exc:
        raise FieldLayoutError(f"could not read field layout {source}: {exc}") from exc
    root = _mapping(data, "field layout")
    field = _mapping(root.get("field"), "field")
    field_length = _number(field.get("length"), "field.length", positive=True)
    field_width = _number(field.get("width"), "field.width", positive=True)
    tags = root.get("tags")
    if not isinstance(tags, list):
        raise FieldLayoutError("tags must be a JSON array")

    transforms: dict[int, np.ndarray] = {}
    for index, raw_tag in enumerate(tags):
        context = f"tags[{index}]"
        tag = _mapping(raw_tag, context)
        id_keys = [key for key in ("ID", "id") if key in tag]
        if len(id_keys) != 1:
            raise FieldLayoutError(f"{context} must contain exactly one of 'ID' or 'id'")
        tag_id_value = tag[id_keys[0]]
        if isinstance(tag_id_value, bool) or not isinstance(tag_id_value, int) or tag_id_value < 0:
            raise FieldLayoutError(f"{context}.{id_keys[0]} must be a non-negative integer")
        if tag_id_value in transforms:
            raise FieldLayoutError(f"duplicate tag ID {tag_id_value}")

        pose = _mapping(tag.get("pose"), f"{context}.pose")
        translation = _mapping(pose.get("translation"), f"{context}.pose.translation")
        rotation = _mapping(pose.get("rotation"), f"{context}.pose.rotation")
        quaternion = _mapping(rotation.get("quaternion"), f"{context}.pose.rotation.quaternion")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _quaternion_rotation(
            _component(quaternion, "w", f"{context}.pose.rotation.quaternion"),
            _component(quaternion, "x", f"{context}.pose.rotation.quaternion"),
            _component(quaternion, "y", f"{context}.pose.rotation.quaternion"),
            _component(quaternion, "z", f"{context}.pose.rotation.quaternion"),
            f"{context}.pose.rotation.quaternion",
        )
        transform[:3, 3] = [
            _component(translation, "x", f"{context}.pose.translation"),
            _component(translation, "y", f"{context}.pose.translation"),
            _component(translation, "z", f"{context}.pose.translation"),
        ]
        transforms[tag_id_value] = transform

    layout = AprilTagFieldLayout(
        tag_poses=transforms,
        field_length_m=field_length,
        field_width_m=field_width,
        provenance=(
            provenance if provenance is not None else ArtifactProvenance(ProvenanceKind.IMPORTED, str(source.resolve()))
        ),
    )
    if validate_bounds:
        layout.validate_tag_bounds()
    return layout
