"""Read-only paired-image evaluation for :class:`napa_vision.stereo.StereoPoseSolver`.

Manifest schema (version 1)::

  {"schema_version": 1, "family": "tag36h11", "tag_size_m": 0.1651,
   "tag_layout": {"1": [[...4 rows...], ...}, "pairs": [
     {"left": {"camera_id": "left", "path": "left.png", "capture_time_s": 1.0},
      "right": {"camera_id": "right", "path": "right.png", "capture_time_s": 1.002},
      "ground_truth": [{"timestamp_s": 1.001, "field_to_rig": [[...4 rows...]]}]}]}

Paths are relative to the manifest. Ground truth is optional; absent accuracy is
reported as ``null``, never as a zero error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from napa_vision.calibration import CalibrationError
from napa_vision.covariance import PoseWithCovariance, _perturb, pose_difference
from napa_vision.geometry import _rigid_transform
from napa_vision.napa_detector import NapaDetector
from napa_vision.stereo import StereoPoseResult, StereoPoseSolver, StereoSolveDiagnostics
from napa_vision.stereo_calibration import load_stereo_calibration


class StereoReplayManifestError(ValueError):
    """Raised for an invalid or unsafe paired replay manifest."""


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    camera_id: str
    path: Path
    relative_path: str
    capture_time_s: float


@dataclass(frozen=True, slots=True)
class ReplayTruth:
    timestamp_s: float
    field_to_rig: np.ndarray


@dataclass(frozen=True, slots=True)
class ReplayPair:
    left: ReplayFrame
    right: ReplayFrame
    ground_truth: tuple[ReplayTruth, ...]


@dataclass(frozen=True, slots=True)
class StereoReplayManifest:
    path: Path
    family: str
    tag_size_m: float
    tag_layout: dict[int, np.ndarray]
    pairs: tuple[ReplayPair, ...]


def load_stereo_replay_manifest(path: str | Path) -> StereoReplayManifest:
    """Load a strict paired-image manifest without reading images or modifying them."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StereoReplayManifestError(f"manifest file not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise StereoReplayManifestError(f"invalid manifest JSON at line {exc.lineno}: {exc.msg}") from exc
    except OSError as exc:
        raise StereoReplayManifestError(f"could not read manifest {manifest_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise StereoReplayManifestError("manifest must be a JSON object")
    _unknown(data, {"schema_version", "family", "tag_size_m", "tag_layout", "pairs"}, "manifest")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise StereoReplayManifestError("schema_version must be 1")
    family = _string(data.get("family"), "family")
    tag_size_m = _positive(data.get("tag_size_m"), "tag_size_m")
    tag_layout_value = data.get("tag_layout")
    if not isinstance(tag_layout_value, Mapping) or not tag_layout_value:
        raise StereoReplayManifestError("tag_layout must be a non-empty object")
    tag_layout: dict[int, np.ndarray] = {}
    for raw_id, raw_transform in tag_layout_value.items():
        if not isinstance(raw_id, str) or not raw_id.isdecimal():
            raise StereoReplayManifestError("tag_layout keys must be non-negative integer strings")
        tag_id = int(raw_id)
        if tag_id in tag_layout:
            raise StereoReplayManifestError(f"duplicate tag_layout ID {tag_id}")
        try:
            tag_layout[tag_id] = _rigid_transform(np.asarray(raw_transform, dtype=np.float64), f"tag_layout[{tag_id}]")
        except (TypeError, ValueError) as exc:
            raise StereoReplayManifestError(f"tag_layout[{tag_id}] must be a finite rigid 4x4 transform") from exc
    raw_pairs = data.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise StereoReplayManifestError("pairs must be a non-empty array")
    root = manifest_path.parent
    pairs: list[ReplayPair] = []
    for index, raw_pair in enumerate(raw_pairs):
        context = f"pairs[{index}]"
        if not isinstance(raw_pair, Mapping):
            raise StereoReplayManifestError(f"{context} must be an object")
        _unknown(raw_pair, {"left", "right", "ground_truth"}, context)
        left = _frame(raw_pair.get("left"), f"{context}.left", root)
        right = _frame(raw_pair.get("right"), f"{context}.right", root)
        if left.camera_id == right.camera_id:
            raise StereoReplayManifestError(f"{context} camera IDs must be distinct")
        truth = raw_pair.get("ground_truth", [])
        if not isinstance(truth, list):
            raise StereoReplayManifestError(f"{context}.ground_truth must be an array of timestamped poses")
        truths = []
        for sample in truth:
            if not isinstance(sample, Mapping):
                raise StereoReplayManifestError("ground-truth samples must be objects")
            _unknown(sample, {"timestamp_s", "field_to_rig"}, "ground_truth")
            timestamp = _finite(sample["timestamp_s"], "ground_truth.timestamp_s")
            if any(item.timestamp_s == timestamp for item in truths):
                raise StereoReplayManifestError("ground-truth timestamps must be unique within a pair")
            try:
                transform = _rigid_transform(sample["field_to_rig"], "ground_truth.field_to_rig")
            except (TypeError, ValueError) as exc:
                raise StereoReplayManifestError(str(exc)) from exc
            truths.append(ReplayTruth(timestamp, transform))
        pairs.append(ReplayPair(left, right, tuple(truths)))
    return StereoReplayManifest(manifest_path, family, tag_size_m, tag_layout, tuple(pairs))


def run_stereo_replay(
    manifest: StereoReplayManifest | str | Path,
    calibration_path: str | Path,
    *,
    max_capture_skew_s: float,
    max_reprojection_error_px: float = 5.0,
) -> dict[str, Any]:
    """Evaluate synchronized frames and return JSON-safe metrics without changing input files."""

    if not isinstance(manifest, StereoReplayManifest):
        manifest = load_stereo_replay_manifest(manifest)
    calibration = load_stereo_calibration(calibration_path)
    detector = NapaDetector(family=manifest.family)
    solver = StereoPoseSolver(
        calibration,
        manifest.tag_layout,
        manifest.tag_size_m,
        max_capture_skew_s=max_capture_skew_s,
        max_reprojection_error_px=max_reprojection_error_px,
    )
    pair_results: list[dict[str, Any]] = []
    detect_times: list[float] = []
    solver_times: list[float] = []
    try:
        for pair in manifest.pairs:
            _validate_camera_ids(pair, calibration.left_id, calibration.right_id)
            left_image = _read_image(pair.left.path)
            right_image = _read_image(pair.right.path)
            calibration.left.validate_for_image_size(left_image.shape[1], left_image.shape[0])
            calibration.right.validate_for_image_size(right_image.shape[1], right_image.shape[0])
            started = perf_counter()
            left_observations = detector.detect(left_image)
            right_observations = detector.detect(right_image)
            detect_times.append((perf_counter() - started) * 1000.0)
            started = perf_counter()
            stereo = solver.solve(
                left_observations,
                right_observations,
                left_capture_time_s=pair.left.capture_time_s,
                right_capture_time_s=pair.right.capture_time_s,
            )
            solver_times.append((perf_counter() - started) * 1000.0)
            left_mono = solver.solve(
                left_observations,
                (),
                left_capture_time_s=pair.left.capture_time_s,
                right_capture_time_s=pair.right.capture_time_s,
                allow_monocular_fallback=True,
            )
            right_mono = solver.solve(
                (),
                right_observations,
                left_capture_time_s=pair.left.capture_time_s,
                right_capture_time_s=pair.right.capture_time_s,
                allow_monocular_fallback=True,
            )
            pair_results.append(
                _pair_result(pair, left_observations, right_observations, stereo, left_mono, right_mono)
            )
    finally:
        detector.close()
    statuses = {
        status: sum(row["stereo"]["status"] == status for row in pair_results)
        for status in ("stereo", "monocular", "rejected")
    }
    return {
        "schema_version": 1,
        "manifest": str(manifest.path),
        "calibration": str(Path(calibration_path).expanduser().resolve()),
        "family": manifest.family,
        "tag_size_m": manifest.tag_size_m,
        "pairs": pair_results,
        "summary": {"pairs": len(pair_results), "stereo_statuses": statuses},
        "timing": {"detection_ms": _percentiles(detect_times), "solver_ms": _percentiles(solver_times)},
    }


def _pair_result(
    pair: ReplayPair,
    left: Sequence[Any],
    right: Sequence[Any],
    stereo: StereoPoseResult,
    left_mono: StereoPoseResult,
    right_mono: StereoPoseResult,
) -> dict[str, Any]:
    return {
        "left": {"camera_id": pair.left.camera_id, "path": pair.left.relative_path, "detections": len(left)},
        "right": {"camera_id": pair.right.camera_id, "path": pair.right.relative_path, "detections": len(right)},
        "stereo": _result(stereo, pair.ground_truth),
        "independent_pose_fusion": _result(_fused(left_mono, right_mono, stereo), pair.ground_truth),
        "monocular": {
            "left": _result(left_mono, pair.ground_truth),
            "right": _result(right_mono, pair.ground_truth),
        },
    }


def _fused(left: StereoPoseResult, right: StereoPoseResult, stereo: StereoPoseResult) -> StereoPoseResult:
    """Independent best-pose Gaussian fusion baseline, conditional on fixed calibration and a static pair."""

    diagnostics = StereoSolveDiagnostics(stereo.diagnostics.capture_skew_s)
    timestamp = left.timestamp / 2 + right.timestamp / 2
    if stereo.diagnostics.rejection_reason == "capture skew exceeds max_capture_skew_s":
        diagnostics.rejection_reason = "capture skew exceeds max_capture_skew_s"
    elif left.pose is None or right.pose is None:
        diagnostics.rejection_reason = "fusion requires two accepted monocular poses"
    else:
        # Both covariances already use field-frame translation and left-applied rotation errors. This baseline
        # linearizes around the first modes; it does not search alternate mode pairings or compensate motion.
        try:
            left_information = np.linalg.inv(left.pose.covariance)
            right_information = np.linalg.inv(right.pose.covariance)
            covariance = np.linalg.inv(left_information + right_information)
            delta = covariance @ right_information @ pose_difference(left.pose.transform, right.pose.transform)
            pose = PoseWithCovariance(_perturb(left.pose.transform, delta), covariance)
        except np.linalg.LinAlgError:
            diagnostics.rejection_reason = "monocular covariance is singular"
        else:
            diagnostics.accepted_groups = left.diagnostics.accepted_groups + right.diagnostics.accepted_groups
            return StereoPoseResult(timestamp, "stereo", (pose,), diagnostics)
    return StereoPoseResult(timestamp, "rejected", (), diagnostics)


def _result(result: StereoPoseResult, truths: tuple[ReplayTruth, ...]) -> dict[str, Any]:
    truth = next(
        (
            item.field_to_rig
            for item in truths
            if math.isclose(item.timestamp_s, result.timestamp, rel_tol=0.0, abs_tol=1e-9)
        ),
        None,
    )
    diagnostics = result.diagnostics
    reason = diagnostics.rejection_reason or diagnostics.fallback_reason
    if reason is None and result.pose is not None:
        reason = "accepted"
    row: dict[str, Any] = {
        "status": result.status,
        "timestamp_s": result.timestamp,
        "reason": reason,
        "accepted_groups": [list(group) for group in diagnostics.accepted_groups],
        "translation_error_m": None,
        "rotation_error_deg": None,
        "ground_truth_time_matched": truth is not None,
        "candidate_count": len(result.poses),
        "conditional_nees": None,
        "std_devs": None if result.pose is None else result.pose.std_devs().tolist(),
        "candidate_rms_px": list(diagnostics.candidate_rms_px),
    }
    if truth is not None and result.pose is not None:
        estimated = result.pose.transform
        row["translation_error_m"] = float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
        cosine = np.clip((np.trace(estimated[:3, :3].T @ truth[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        row["rotation_error_deg"] = float(math.degrees(math.acos(cosine)))
        delta = pose_difference(estimated, truth)
        try:
            row["conditional_nees"] = float(delta @ np.linalg.solve(result.pose.covariance, delta))
        except np.linalg.LinAlgError:
            pass  # A singular reported covariance cannot be scored.
    return row


def _frame(value: object, context: str, root: Path) -> ReplayFrame:
    if not isinstance(value, Mapping):
        raise StereoReplayManifestError(f"{context} must be an object")
    _unknown(value, {"camera_id", "path", "capture_time_s"}, context)
    camera_id = _string(value.get("camera_id"), f"{context}.camera_id")
    relative_path = _string(value.get("path"), f"{context}.path")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise StereoReplayManifestError(f"{context}.path must be relative to the manifest")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise StereoReplayManifestError(f"{context}.path must not escape the manifest directory")
    capture_time_s = _finite(value.get("capture_time_s"), f"{context}.capture_time_s")
    return ReplayFrame(camera_id, resolved, relative.as_posix(), capture_time_s)


def _validate_camera_ids(pair: ReplayPair, left_id: str, right_id: str) -> None:
    if pair.left.camera_id != left_id or pair.right.camera_id != right_id:
        raise StereoReplayManifestError(
            f"recorded camera IDs must be left={left_id!r} and right={right_id!r}, "
            f"got {pair.left.camera_id!r}/{pair.right.camera_id!r}"
        )


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise StereoReplayManifestError(f"could not decode image {path}")
    return image


def _unknown(value: Mapping[str, object], allowed: set[str], context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise StereoReplayManifestError(f"unknown {context} field(s): {', '.join(sorted(unknown))}")
    missing = allowed - set(value)
    if missing - {"ground_truth"}:
        raise StereoReplayManifestError(f"{context} is missing field(s): {', '.join(sorted(missing))}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StereoReplayManifestError(f"{name} must be a non-empty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StereoReplayManifestError(f"{name} must be finite")
    return float(value)


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise StereoReplayManifestError(f"{name} must be greater than zero")
    return result


def _percentiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate recorded synchronized stereo image pairs")
    parser.add_argument("manifest")
    parser.add_argument("--calibration", required=True, help="Stereo calibration artifact JSON")
    parser.add_argument("--max-capture-skew-s", required=True, type=float)
    parser.add_argument("--max-reprojection-error-px", type=float, default=5.0)
    parser.add_argument("--output", help="Write JSON results to this path")
    args = parser.parse_args(argv)
    try:
        report = run_stereo_replay(
            args.manifest,
            args.calibration,
            max_capture_skew_s=args.max_capture_skew_s,
            max_reprojection_error_px=args.max_reprojection_error_px,
        )
    except (CalibrationError, OSError, RuntimeError, StereoReplayManifestError, ValueError) as exc:
        print(f"stereo-replay: error: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
