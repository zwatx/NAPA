from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .output import OutputPublisher
from .simulation import RenderedFrame
from .simulation_metrics import metric_distribution, rotation_error_deg
from .types import FrameResult


@dataclass(slots=True)
class _ScenarioSamples:
    expected_tag_ids: tuple[int, ...]
    expect_pose: bool
    detected_tag_ids: tuple[int, ...] = ()
    missing_tag_ids: tuple[int, ...] = ()
    unexpected_tag_ids: tuple[int, ...] = ()
    corner_rms_px: list[float] = field(default_factory=list)
    translation_error_m: list[float] = field(default_factory=list)
    rotation_error_deg: list[float] = field(default_factory=list)
    reprojection_error_px: list[float] = field(default_factory=list)
    detection_ms: list[float] = field(default_factory=list)
    solve_ms: list[float] = field(default_factory=list)
    pose_solved: int = 0
    samples: int = 0
    expected_rejection_failures: int = 0


class SimulationScoringPublisher(OutputPublisher):
    def __init__(self, frames: tuple[RenderedFrame, ...], *, warmup_passes: int) -> None:
        self.frames = frames
        self.warmup_passes = warmup_passes
        self.published = 0
        self.closed = False
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.pose_expected = 0
        self.pose_solved = 0
        self.expected_rejection_failures = 0
        self.corner_rms_px: list[float] = []
        self.translation_error_m: list[float] = []
        self.rotation_error_deg: list[float] = []
        self.reprojection_error_px: list[float] = []
        self.timings: dict[str, list[float]] = {"detection_ms": [], "solve_ms": [], "processing_ms": []}
        self.scenarios = {
            frame.truth.name: _ScenarioSamples(frame.truth.expected_tag_ids, frame.truth.expect_pose)
            for frame in self.frames
        }

    def publish(self, result: FrameResult) -> None:
        frame_index = self.published % len(self.frames)
        pass_index = self.published // len(self.frames)
        truth = self.frames[frame_index].truth
        self.published += 1
        if pass_index < self.warmup_passes:
            return
        scenario = self.scenarios[truth.name]
        expected = Counter(truth.expected_tag_ids)
        detected_ids = tuple(sorted(observation.tag_id for observation in result.image_observations))
        detected = Counter(detected_ids)
        self.true_positives += sum(min(count, detected[tag_id]) for tag_id, count in expected.items())
        self.false_negatives += sum(max(0, count - detected[tag_id]) for tag_id, count in expected.items())
        self.false_positives += sum(max(0, count - expected[tag_id]) for tag_id, count in detected.items())
        scenario.detected_tag_ids = detected_ids
        scenario.missing_tag_ids = tuple(sorted((expected - detected).elements()))
        scenario.unexpected_tag_ids = tuple(sorted((detected - expected).elements()))
        scenario.samples += 1

        observations_by_id = {observation.tag_id: observation for observation in result.image_observations}
        for tag_id in set(observations_by_id).intersection(truth.image_corners):
            residual = observations_by_id[tag_id].corners - truth.image_corners[tag_id]
            corner_error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            self.corner_rms_px.append(corner_error)
            scenario.corner_rms_px.append(corner_error)

        pose = result.camera_pose
        if truth.expect_pose:
            self.pose_expected += 1
            if pose is not None:
                self.pose_solved += 1
                scenario.pose_solved += 1
                translation_error = float(np.linalg.norm(pose.field_to_camera_0[:3, 3] - truth.field_to_camera[:3, 3]))
                rotation_error = rotation_error_deg(truth.field_to_camera, pose.field_to_camera_0)
                self.translation_error_m.append(translation_error)
                self.rotation_error_deg.append(rotation_error)
                self.reprojection_error_px.append(pose.error_0)
                scenario.translation_error_m.append(translation_error)
                scenario.rotation_error_deg.append(rotation_error)
                scenario.reprojection_error_px.append(pose.error_0)

        rejected_ids = set(result.pose_diagnostics.rejected_tag_reasons) if result.pose_diagnostics else set()
        if set(truth.expected_rejected_tag_ids) - rejected_ids:
            self.expected_rejection_failures += 1
            scenario.expected_rejection_failures += 1
        for name in self.timings:
            value = float(result.health[name])
            self.timings[name].append(value)
            if name == "detection_ms":
                scenario.detection_ms.append(value)
            elif name == "solve_ms":
                scenario.solve_ms.append(value)

    def close(self) -> None:
        self.closed = True

    def summary(self) -> dict[str, Any]:
        expected_instances = self.true_positives + self.false_negatives
        detected_instances = self.true_positives + self.false_positives
        scenario_summaries = []
        for name, samples in self.scenarios.items():
            scenario_summaries.append(
                {
                    "name": name,
                    "samples": samples.samples,
                    "expected_tag_ids": list(samples.expected_tag_ids),
                    "detected_tag_ids": list(samples.detected_tag_ids),
                    "missing_tag_ids": list(samples.missing_tag_ids),
                    "unexpected_tag_ids": list(samples.unexpected_tag_ids),
                    "pose_solve_rate": (
                        None
                        if not samples.expect_pose or samples.samples == 0
                        else samples.pose_solved / samples.samples
                    ),
                    "corner_rms_px": metric_distribution(samples.corner_rms_px),
                    "translation_error_m": metric_distribution(samples.translation_error_m),
                    "rotation_error_deg": metric_distribution(samples.rotation_error_deg),
                    "reprojection_error_px": metric_distribution(samples.reprojection_error_px),
                    "detection_ms": metric_distribution(samples.detection_ms),
                    "solve_ms": metric_distribution(samples.solve_ms),
                    "expected_rejection_failures": samples.expected_rejection_failures,
                }
            )
        return {
            "accuracy": {
                "expected_instances": expected_instances,
                "detected_instances": detected_instances,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "recall": None if expected_instances == 0 else self.true_positives / expected_instances,
                "precision": None if detected_instances == 0 else self.true_positives / detected_instances,
            },
            "pose": {
                "expected": self.pose_expected,
                "solved": self.pose_solved,
                "solve_rate": None if self.pose_expected == 0 else self.pose_solved / self.pose_expected,
                "translation_error_m": metric_distribution(self.translation_error_m),
                "rotation_error_deg": metric_distribution(self.rotation_error_deg),
                "reprojection_error_px": metric_distribution(self.reprojection_error_px),
                "expected_rejection_failures": self.expected_rejection_failures,
            },
            "corners": {"rms_px": metric_distribution(self.corner_rms_px)},
            "timing": {name: metric_distribution(values) for name, values in self.timings.items()},
            "scenarios": scenario_summaries,
        }
