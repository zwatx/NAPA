from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProvenanceError(ValueError):
    """Raised when an example or unverified artifact is used for production."""


class ProvenanceKind(StrEnum):
    """Trust level for a calibration file."""

    EXAMPLE = "example"
    IMPORTED = "imported"
    MEASURED = "measured"
    OFFICIAL = "official"


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Describe where a calibration came from.

    ``IMPORTED`` deliberately remains unverified: reading a file successfully
    does not prove that it was measured for the installed camera. Only measured
    data (or an explicitly official artifact) passes a production readiness check.
    """

    kind: ProvenanceKind
    source: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProvenanceKind):
            raise ProvenanceError("artifact provenance kind must be a ProvenanceKind")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ProvenanceError("artifact provenance source must not be empty")
        if self.notes is not None and not isinstance(self.notes, str):
            raise ProvenanceError("artifact provenance notes must be a string or null")

    @property
    def production_ready(self) -> bool:
        return self.kind in {ProvenanceKind.MEASURED, ProvenanceKind.OFFICIAL}

    def require_production(self, artifact_name: str) -> None:
        if not self.production_ready:
            raise ProvenanceError(
                f"{artifact_name} is marked {self.kind.value!r} from {self.source!r}; "
                "production requires measured calibration or an official layout"
            )

    def require_kind(self, expected_kind: ProvenanceKind, artifact_name: str) -> None:
        """Require the artifact-specific provenance kind, not just generic trust."""

        if self.kind is not expected_kind:
            raise ProvenanceError(
                f"{artifact_name} is marked {self.kind.value!r} from {self.source!r}; "
                f"production requires {expected_kind.value!r} provenance"
            )

    def to_dict(self) -> dict[str, str]:
        data = {"kind": self.kind.value, "source": self.source}
        if self.notes is not None:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, value: object, *, default_source: str) -> ArtifactProvenance:
        if value is None:
            return cls(ProvenanceKind.IMPORTED, default_source)
        if not isinstance(value, dict):
            raise ProvenanceError("provenance must be a JSON object")
        unknown = set(value) - {"kind", "source", "notes"}
        if unknown:
            raise ProvenanceError(f"unknown provenance field(s): {', '.join(sorted(unknown))}")
        try:
            kind = ProvenanceKind(value.get("kind", ProvenanceKind.IMPORTED.value))
        except ValueError as exc:
            choices = ", ".join(item.value for item in ProvenanceKind)
            raise ProvenanceError(f"provenance.kind must be one of: {choices}") from exc
        source = value.get("source", default_source)
        notes = value.get("notes")
        if not isinstance(source, str) or not source.strip():
            raise ProvenanceError("provenance.source must be a non-empty string")
        if notes is not None and not isinstance(notes, str):
            raise ProvenanceError("provenance.notes must be a string or null")
        return cls(kind, source.strip(), notes)
