from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2

from .benchmark import serialize_benchmark_summary, write_benchmark_summary_text
from .config import PlatformProfile, ProfileError, load_profile
from .fiducials import OpenCVArucoFiducialDetector, make_apriltag_detector
from .prefilter import BlockSkipPrefilter
from .simulation_artifacts import write_simulation_corpus
from .simulation_benchmark import run_simulation
from .simulation_metrics import SimulationThresholds, evaluate_simulation_gate


def _ratio(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic AprilTag simulation through the real pipeline")
    parser.add_argument("--profile", default="laptop", help="Profile supplying resolution, intrinsics, and thresholds")
    parser.add_argument("--suite", choices=["standard", "sweep"], default="standard")
    parser.add_argument("--output-dir", default="simulation-output", help="Generated corpus and report directory")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--supersample", type=_positive_int, default=1)
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument("--warmup-passes", type=_nonnegative_int, default=1)
    parser.add_argument("--compare-optimizations", action="store_true")
    parser.add_argument("--min-recall", type=_ratio, default=1.0)
    parser.add_argument("--min-precision", type=_ratio, default=1.0)
    parser.add_argument("--min-pose-solve-rate", type=_ratio, default=1.0)
    parser.add_argument("--max-corner-rms-px", type=_positive_float, default=2.0)
    parser.add_argument("--max-translation-error-m", type=_positive_float, default=0.08)
    parser.add_argument("--max-rotation-error-deg", type=_positive_float, default=1.5)
    parser.add_argument("--pretty", action="store_true")
    return parser


def _candidate_detector(
    profile: PlatformProfile,
    candidate: str,
    *,
    layout_tag_ids: Sequence[int] | None = None,
) -> OpenCVArucoFiducialDetector:
    if candidate in {"aruco3", "layout-restricted"}:
        return make_apriltag_detector(candidate, layout_tag_ids=layout_tag_ids)
    if candidate == "subpixel-corners":
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return OpenCVArucoFiducialDetector(parameters=parameters)
    if candidate == "skip-scan":
        config = replace(profile.skip_scan, enabled=True)
        return OpenCVArucoFiducialDetector(prefilter=BlockSkipPrefilter(config))
    raise ValueError(f"Unknown optimization candidate {candidate}")


def _comparison(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    base_timing = baseline["timing"]["processing_ms"]
    candidate_timing = candidate["timing"]["processing_ms"]
    accuracy_equal = baseline["accuracy"] == candidate["accuracy"]
    pose_equal = baseline["pose"]["solved"] == candidate["pose"]["solved"]
    candidate_gate_passed = bool(candidate["gate"]["passed"])
    p50_speedup = _safe_ratio(base_timing["p50"], candidate_timing["p50"])
    p95_speedup = _safe_ratio(base_timing["p95"], candidate_timing["p95"])
    useful_latency_gain = (
        p50_speedup is not None
        and p95_speedup is not None
        and (p50_speedup >= 1.05 or (p95_speedup >= 1.20 and p50_speedup >= 0.90))
    )
    baseline_pose = baseline["pose"]
    candidate_pose = candidate["pose"]
    return {
        "accuracy_equal": accuracy_equal,
        "pose_solve_count_equal": pose_equal,
        "candidate_gate_passed": candidate_gate_passed,
        "processing_p50_speedup": p50_speedup,
        "processing_p95_speedup": p95_speedup,
        "max_translation_error_improvement": _safe_ratio(
            baseline_pose["translation_error_m"]["max"],
            candidate_pose["translation_error_m"]["max"],
        ),
        "max_rotation_error_improvement": _safe_ratio(
            baseline_pose["rotation_error_deg"]["max"],
            candidate_pose["rotation_error_deg"]["max"],
        ),
        "eligible_for_hardware_validation": (
            accuracy_equal and pose_equal and candidate_gate_passed and useful_latency_gain
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = SimulationThresholds(
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        min_pose_solve_rate=args.min_pose_solve_rate,
        max_corner_rms_px=args.max_corner_rms_px,
        max_translation_error_m=args.max_translation_error_m,
        max_rotation_error_deg=args.max_rotation_error_deg,
    )
    try:
        profile = load_profile(args.profile)
        baseline = run_simulation(
            profile,
            suite=args.suite,
            seed=args.seed,
            supersample=args.supersample,
            repetitions=args.repetitions,
            warmup_passes=args.warmup_passes,
        )
        baseline.summary["gate"] = evaluate_simulation_gate(baseline.summary, thresholds)
        # This corpus is a replaceable generated batch. Keep each publish atomic, but avoid an fsync barrier per file.
        artifacts = write_simulation_corpus(
            baseline.frames,
            args.output_dir,
            profile,
            baseline.tag_layout,
            suite=args.suite,
            durable=False,
        )
        baseline.summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        if args.compare_optimizations:
            comparisons: dict[str, object] = {}
            for candidate_name in ("aruco3", "layout-restricted", "subpixel-corners", "skip-scan"):
                candidate = run_simulation(
                    profile,
                    suite=args.suite,
                    seed=args.seed,
                    supersample=args.supersample,
                    repetitions=args.repetitions,
                    warmup_passes=args.warmup_passes,
                    detector=_candidate_detector(
                        profile,
                        candidate_name,
                        layout_tag_ids=tuple(baseline.tag_layout),
                    ),
                    corpus=baseline.corpus,
                )
                candidate.summary["gate"] = evaluate_simulation_gate(candidate.summary, thresholds)
                comparisons[candidate_name] = {
                    "assessment": _comparison(baseline.summary, candidate.summary),
                    "results": candidate.summary,
                }
            baseline.summary["optimization_comparisons"] = comparisons
        report_path = Path(args.output_dir).expanduser().resolve() / "simulation-report.json"
        baseline.summary["artifacts"]["report"] = str(report_path)
        report_text = serialize_benchmark_summary(baseline.summary, pretty=True)
        # The report is regenerated with the corpus, so atomic replacement is sufficient here as well.
        write_benchmark_summary_text(report_text, report_path, durable=False)
        stdout_text = report_text if args.pretty else serialize_benchmark_summary(baseline.summary)
        print(stdout_text)
    except (cv2.error, OSError, ProfileError, RuntimeError, ValueError) as exc:
        print(f"northstar-simulate: error: {exc}", file=sys.stderr)
        return 2
    return 0 if baseline.summary["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
