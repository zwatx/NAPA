from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SimulationThresholds:
    min_recall: float = 1.0
    min_precision: float = 1.0
    min_pose_solve_rate: float = 1.0
    max_corner_rms_px: float = 2.0
    max_translation_error_m: float = 0.08
    max_rotation_error_deg: float = 1.5

    def __post_init__(self) -> None:
        for value, name in (
            (self.min_recall, "min_recall"),
            (self.min_precision, "min_precision"),
            (self.min_pose_solve_rate, "min_pose_solve_rate"),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for value, name in (
            (self.max_corner_rms_px, "max_corner_rms_px"),
            (self.max_translation_error_m, "max_translation_error_m"),
            (self.max_rotation_error_deg, "max_rotation_error_deg"),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be greater than zero")


def evaluate_simulation_gate(summary: Mapping[str, Any], thresholds: SimulationThresholds) -> dict[str, Any]:
    accuracy, pose = summary["accuracy"], summary["pose"]
    measured = {
        "recall": accuracy["recall"],
        "precision": accuracy["precision"],
        "pose_solve_rate": pose["solve_rate"],
        "max_corner_rms_px": summary["corners"]["rms_px"]["max"],
        "max_translation_error_m": pose["translation_error_m"]["max"],
        "max_rotation_error_deg": pose["rotation_error_deg"]["max"],
        "expected_rejection_failures": pose["expected_rejection_failures"],
    }
    configured = {
        "min_recall": thresholds.min_recall,
        "min_precision": thresholds.min_precision,
        "min_pose_solve_rate": thresholds.min_pose_solve_rate,
        "max_corner_rms_px": thresholds.max_corner_rms_px,
        "max_translation_error_m": thresholds.max_translation_error_m,
        "max_rotation_error_deg": thresholds.max_rotation_error_deg,
    }
    failures: list[str] = []
    for threshold_name, metric_name in (
        ("min_recall", "recall"),
        ("min_precision", "precision"),
        ("min_pose_solve_rate", "pose_solve_rate"),
    ):
        value, threshold = measured[metric_name], configured[threshold_name]
        if value is None or value < threshold:
            failures.append(f"{metric_name} {value!r} is below required {threshold:g}")
    for threshold_name, metric_name in (
        ("max_corner_rms_px", "max_corner_rms_px"),
        ("max_translation_error_m", "max_translation_error_m"),
        ("max_rotation_error_deg", "max_rotation_error_deg"),
    ):
        value, threshold = measured[metric_name], configured[threshold_name]
        if value is None or value > threshold:
            failures.append(f"{metric_name} {value!r} exceeds allowed {threshold:g}")
    if measured["expected_rejection_failures"]:
        failures.append(f"{measured['expected_rejection_failures']} expected outlier rejection checks failed")
    return {"passed": not failures, "thresholds": configured, "measured": measured, "failures": failures}


def metric_distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    p50, p95, p99 = np.percentile(array, [50.0, 95.0, 99.0])
    return {
        "samples": len(values),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(np.max(array)),
    }


def rotation_error_deg(expected: np.ndarray, actual: np.ndarray) -> float:
    delta = expected[:3, :3].T @ actual[:3, :3]
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))
