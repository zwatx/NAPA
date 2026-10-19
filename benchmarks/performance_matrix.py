"""Reproducible whole-frame AprilTag detector and occlusion benchmark.

The corpus is deliberately generated in memory before any detector is called.
Timing excludes generation, report writing, and scoring; it includes each
detector wrapper's normal image conversion or uniform-image gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np

from napa_vision.benchmark import write_benchmark_summary
from napa_vision.fiducials import OpenCVArucoFiducialDetector, make_apriltag_detector
from napa_vision.napa_detector import NapaDetector, native_available
from napa_vision.prefilter import BlockSkipPrefilter, SkipScanConfig

REPORT_SCHEMA_VERSION = 2
DEFAULT_RESOLUTIONS = ((640, 480), (1280, 720), (1920, 1080))
DEFAULT_TAG_SIDES = (96, 160)
TAG_IDS = (1, 42, 314)
LAYOUT_TAG_IDS = (1, 42, 200, 314)
OCCLUSION_FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
OCCLUSION_REGIONS = ("payload-center", "border-strip", "corner")
OCCLUSION_VALUES = (255, 127)
STRESS_TAG_ID = 200
CORNER_TAG_SIDES = (32, 48, 64, 96, 160, 240)
CORNER_POSES_PER_SIDE = 20


@dataclass(frozen=True, slots=True)
class BenchmarkFrame:
    name: str
    image: np.ndarray
    expected_ids: tuple[int, ...]
    resolution: tuple[int, int]
    scenario: str
    rotation: int = 0
    tag_side_px: int | None = None
    occlusion: dict[str, object] | None = None


@dataclass(slots=True)
class Backend:
    name: str
    factory: Callable[[], Callable[[np.ndarray], list[int]]]
    config: dict[str, object]
    experimental: bool = False


@dataclass(frozen=True, slots=True)
class IdMetrics:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float | None:
        denominator = self.true_positive + self.false_positive
        return None if denominator == 0 else self.true_positive / denominator

    @property
    def recall(self) -> float | None:
        denominator = self.true_positive + self.false_negative
        return None if denominator == 0 else self.true_positive / denominator


def build_corpus(
    *,
    resolutions: Sequence[tuple[int, int]] = DEFAULT_RESOLUTIONS,
    tag_side_pixels: Sequence[int] = DEFAULT_TAG_SIDES,
    seed: int = 2468,
) -> list[BenchmarkFrame]:
    """Build a seeded, in-memory grayscale corpus shared by every backend."""

    if not resolutions or not tag_side_pixels:
        raise ValueError("resolutions and tag_side_pixels must not be empty")
    if any(width < 160 or height < 120 for width, height in resolutions):
        raise ValueError("each resolution must be at least 160x120")
    if any(side < 32 for side in tag_side_pixels):
        raise ValueError("tag_side_pixels must be at least 32")

    occlusion_resolution = min(resolutions, key=lambda item: item[0] * item[1])
    rng = np.random.default_rng(seed)
    frames: list[BenchmarkFrame] = []
    for width, height in resolutions:
        resolution = (width, height)
        blank = np.full((height, width), 255, dtype=np.uint8)
        frames.append(BenchmarkFrame("blank", blank, (), resolution, "blank"))
        frames.append(
            BenchmarkFrame(
                "textured-negative",
                _textured_negative(width, height, rng),
                (),
                resolution,
                "textured-negative",
            )
        )
        for side in tag_side_pixels:
            if side + 20 >= min(width, height):
                continue
            for rotation in range(4):
                frames.append(
                    _single_tag_frame(
                        name=f"clean-id1-r{rotation}-s{side}",
                        width=width,
                        height=height,
                        tag_id=1,
                        side=side,
                        rotation=rotation,
                        scenario="clean-one-tag",
                    )
                )
            frames.append(_four_tag_frame(width, height, side))
            frames.append(
                _single_tag_frame(
                    name=f"tiny-id314-s{side}",
                    width=width,
                    height=height,
                    tag_id=314,
                    side=max(32, side // 3),
                    rotation=2,
                    scenario="tiny-tag",
                )
            )
            blurred = _single_tag_frame(
                name=f"blurred-id42-s{side}",
                width=width,
                height=height,
                tag_id=42,
                side=side,
                rotation=1,
                scenario="blurred",
            )
            frames.append(
                BenchmarkFrame(
                    blurred.name,
                    cv2.GaussianBlur(blurred.image, (0, 0), sigmaX=1.6),
                    blurred.expected_ids,
                    resolution,
                    blurred.scenario,
                    blurred.rotation,
                    blurred.tag_side_px,
                )
            )
            frames.append(_perspective_frame(width, height, side))
            frames.extend(_stress_frames(width, height, side, rng))
            # Hold ID and rotation fixed across every obstruction fraction.
            # Payload sampling depends on tag size, so test both sizes at the
            # smallest frame resolution rather than retaining gigabytes of copies.
            if resolution == occlusion_resolution:
                for tag_id in TAG_IDS:
                    for rotation in range(4):
                        for fraction in OCCLUSION_FRACTIONS:
                            for region in OCCLUSION_REGIONS:
                                for value in OCCLUSION_VALUES:
                                    frames.append(
                                        _occlusion_frame(
                                            width=width,
                                            height=height,
                                            tag_id=tag_id,
                                            side=side,
                                            rotation=rotation,
                                            fraction=fraction,
                                            region=region,
                                            value=value,
                                        )
                                    )
    return frames


def score_ids(expected: Iterable[int], detected: Iterable[int]) -> IdMetrics:
    """Score IDs exactly; repeated detected IDs count as additional false positives."""

    expected_set = set(expected)
    detected_values = [int(value) for value in detected]
    detected_set = set(detected_values)
    duplicates = len(detected_values) - len(detected_set)
    return IdMetrics(
        true_positive=len(expected_set & detected_set),
        false_positive=len(detected_set - expected_set) + duplicates,
        false_negative=len(expected_set - detected_set),
    )


def summarize_latency(samples_ns: Sequence[int]) -> dict[str, float | int]:
    if not samples_ns:
        raise ValueError("latency samples must not be empty")
    values = np.asarray(samples_ns, dtype=np.float64) / 1_000_000.0
    p50, p95, p99 = np.percentile(values, (50, 95, 99))
    return {
        "samples": len(samples_ns),
        "min_ms": float(np.min(values)),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(np.max(values)),
        "latency_derived_p50_fps": float(1000.0 / p50) if p50 > 0 else float("inf"),
    }


def build_backends(*, threads: int, include_robust_erased10: bool = False) -> tuple[list[Backend], dict[str, str]]:
    """Create factories only. Construction and first-call time are measured later."""

    if threads < 1:
        raise ValueError("threads must be at least 1")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    raw_params = cv2.aruco.DetectorParameters()
    dictionary_config = _dictionary_config(dictionary)
    open_cv_threads = cv2.getNumThreads()
    raw_config = {
        "dictionary": "DICT_APRILTAG_36h11",
        "dictionary_config": dictionary_config,
        "parameters": _opencv_parameters(raw_params),
        "wrapper": "none",
        "opencv_threads_effective": open_cv_threads,
    }
    backends = []
    skipped: dict[str, str] = {}
    if native_available():
        native = NapaDetector()
        backends.append(
            Backend(
                "napa-native",
                _napa_native_detector,
                {
                    "min_tag_px": native.min_tag_px,
                    "max_hamming": native.max_hamming,
                    "implementation": "portable C core via ctypes",
                },
            )
        )
        native.close()
    else:
        skipped["napa-native"] = "unavailable: the Napa core library is not built"
    backends += [
        Backend("opencv-aruco-raw", lambda: _raw_opencv_detector(dictionary, raw_params), raw_config),
        Backend(
            "napa-default",
            _napa_default_detector,
            _napa_config("baseline", dictionary, cv2.aruco.DetectorParameters(), open_cv_threads),
        ),
        Backend(
            "napa-aruco3",
            _napa_aruco3_detector,
            _napa_config("aruco3", dictionary, _aruco3_parameters(), open_cv_threads),
        ),
        Backend(
            "napa-skip-scan",
            _napa_skip_scan_detector,
            _napa_config(
                "skip-scan",
                dictionary,
                cv2.aruco.DetectorParameters(),
                open_cv_threads,
                skip_scan=asdict(SkipScanConfig(enabled=True)),
            ),
        ),
        Backend(
            "napa-layout-restricted",
            lambda: _napa_layout_detector(LAYOUT_TAG_IDS),
            _napa_config(
                "layout-restricted",
                dictionary,
                cv2.aruco.DetectorParameters(),
                open_cv_threads,
                layout_tag_ids=list(LAYOUT_TAG_IDS),
            ),
        ),
    ]
    try:
        import robotpy_apriltag  # noqa: F401
    except ImportError as exc:
        skipped["apriltag3-robotpy-decimate-1"] = f"unavailable: {exc}"
        skipped["apriltag3-robotpy-decimate-2"] = f"unavailable: {exc}"
    else:
        for decimate in (1.0, 2.0):
            config = {
                "family": "tag36h11",
                "bits_corrected": 2,
                "num_threads": threads,
                "quad_decimate": decimate,
            }
            backends.append(
                Backend(
                    f"apriltag3-robotpy-decimate-{int(decimate)}",
                    lambda decimate=decimate, config=config: _robotpy_detector(
                        threads=threads,
                        quad_decimate=decimate,
                        recorded_config=config,
                    ),
                    config,
                )
            )
    try:
        _ensure_repository_root_on_path()
        from experiments.occlusion_detector.robust_apriltag import Robust36h11Detector  # noqa: F401
    except ImportError as exc:
        skipped["experimental-robust36h11-full-search"] = f"unavailable: {exc}"
    else:
        backends.append(
            Backend(
                "experimental-robust36h11-full-search",
                lambda: _robust_detector(max_erased_bits=0),
                {
                    "experimental": True,
                    "candidate_mode": "full-search",
                    "seeded_corners": False,
                    "max_erased_bits": 0,
                },
                experimental=True,
            )
        )
        if include_robust_erased10:
            backends.append(
                Backend(
                    "experimental-robust36h11-full-search-erased10",
                    lambda: _robust_detector(max_erased_bits=10),
                    {
                        "experimental": True,
                        "candidate_mode": "full-search",
                        "seeded_corners": False,
                        "max_erased_bits": 10,
                    },
                    experimental=True,
                )
            )
    return backends, skipped


def run_matrix(
    frames: Sequence[BenchmarkFrame],
    *,
    repetitions: int = 10,
    occlusion_repetitions: int = 1,
    warmup: int = 2,
    seed: int = 2468,
    threads: int = 1,
    include_robust_erased10: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run detectors in a deterministic interleaved order and return report plus raw samples."""

    if repetitions < 1 or occlusion_repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup must not be negative")
    if not frames:
        raise ValueError("frames must not be empty")
    cv2.setNumThreads(threads)
    backends, skipped = build_backends(threads=threads, include_robust_erased10=include_robust_erased10)
    instances: dict[str, Callable[[np.ndarray], list[int]]] = {}
    lifecycle: dict[str, dict[str, object]] = {}
    for backend in backends:
        start_ns = perf_counter_ns()
        instances[backend.name] = backend.factory()
        lifecycle[backend.name] = {"init_ms": (perf_counter_ns() - start_ns) / 1_000_000.0, "first_call_ms": None}

    first_frame = frames[0]
    for backend in backends:
        start_ns = perf_counter_ns()
        instances[backend.name](first_frame.image)
        lifecycle[backend.name]["first_call_ms"] = (perf_counter_ns() - start_ns) / 1_000_000.0

    rng = random.Random(seed)
    pairs = [(backend, frame) for backend in backends for frame in frames]
    for _pass_index in range(warmup):
        order = pairs.copy()
        rng.shuffle(order)
        for backend, frame in order:
            instances[backend.name](frame.image)

    raw_samples: list[dict[str, object]] = []
    totals: dict[str, IdMetrics] = {backend.name: IdMetrics(0, 0, 0) for backend in backends}
    occlusion_totals: dict[tuple[str, str, str, float, str, int], IdMetrics] = {}
    latency: dict[str, list[int]] = defaultdict(list)
    scenario_totals: dict[tuple[str, str, str], IdMetrics] = {}
    scenario_latency: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    def measure(pairs_to_measure: Sequence[tuple[Backend, BenchmarkFrame]], stage: str, count: int) -> None:
        for repetition in range(count):
            order = list(pairs_to_measure)
            rng.shuffle(order)
            for backend, frame in order:
                start_ns = perf_counter_ns()
                detected = instances[backend.name](frame.image)
                elapsed_ns = perf_counter_ns() - start_ns
                metric = score_ids(frame.expected_ids, detected)
                totals[backend.name] = _add_metrics(totals[backend.name], metric)
                latency[backend.name].append(elapsed_ns)
                scenario_key = (backend.name, frame.scenario, f"{frame.resolution[0]}x{frame.resolution[1]}")
                scenario_totals[scenario_key] = _add_metrics(
                    scenario_totals.get(scenario_key, IdMetrics(0, 0, 0)), metric
                )
                scenario_latency[scenario_key].append(elapsed_ns)
                row = {
                    "backend": backend.name,
                    "stage": stage,
                    "repetition": repetition,
                    "frame": frame.name,
                    "scenario": frame.scenario,
                    "resolution": f"{frame.resolution[0]}x{frame.resolution[1]}",
                    "tag_side_px": frame.tag_side_px,
                    "latency_ns": elapsed_ns,
                    "expected_ids": ";".join(map(str, frame.expected_ids)),
                    "detected_ids": ";".join(map(str, detected)),
                    "true_positive": metric.true_positive,
                    "false_positive": metric.false_positive,
                    "false_negative": metric.false_negative,
                }
                if frame.occlusion is not None:
                    row.update({f"occlusion_{key}": value for key, value in frame.occlusion.items()})
                    key = (
                        backend.name,
                        f"{frame.resolution[0]}x{frame.resolution[1]}",
                        str(frame.occlusion["region"]),
                        float(frame.occlusion["requested_fraction"]),
                        str(frame.occlusion["mask_value"]),
                        frame.tag_side_px or 0,
                    )
                    occlusion_totals[key] = _add_metrics(occlusion_totals.get(key, IdMetrics(0, 0, 0)), metric)
                raw_samples.append(row)

    main_pairs = [(backend, frame) for backend, frame in pairs if frame.scenario != "occlusion"]
    occlusion_pairs = [(backend, frame) for backend, frame in pairs if frame.scenario == "occlusion"]
    measure(main_pairs, "main", repetitions)
    measure(occlusion_pairs, "occlusion", occlusion_repetitions)

    backend_reports: dict[str, object] = {}
    for backend in backends:
        metrics = totals[backend.name]
        backend_reports[backend.name] = {
            "experimental_backend": backend.experimental,
            "config": backend.config,
            "lifecycle": lifecycle[backend.name],
            "latency": summarize_latency(latency[backend.name]),
            "id_metrics": _metrics_dict(metrics),
        }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "measurement_scope": {
            "quality": (
                "Exact-ID precision/recall for ALL backends. Corner accuracy against exact ground truth is "
                "reported separately in corner_accuracy; pose accuracy is in the calibrated simulator."
            ),
            "included": "Detector call, selected wrapper conversion/uniformity gate, and per-backend ID extraction.",
            "excluded": "Synthetic rendering, file I/O, report writing, ID scoring, and module imports.",
            "experimental_robust": (
                "Robust36h11 full-search is an experimental ID-only comparison; it receives no "
                "seeded corners and is not a production detector claim."
            ),
            "memory": (
                "No memory metric is reported; tracemalloc would not represent native OpenCV or AprilTag allocations."
            ),
        },
        "metadata": _metadata(),
        "run_config": {
            "seed": seed,
            "repetitions": repetitions,
            "occlusion_repetitions": occlusion_repetitions,
            "warmup": warmup,
            "threads": threads,
            "frames": len(frames),
            "first_call_frame": {"name": first_frame.name, "resolution": list(first_frame.resolution)},
            "opencv_threads_effective": cv2.getNumThreads(),
        },
        "corpus": _corpus_summary(frames),
        "backends": backend_reports,
        "scenario_resolution": _scenario_report(scenario_totals, scenario_latency),
        "skipped_backends": skipped,
        "occlusion": _occlusion_report(occlusion_totals, raw_samples),
        "decoder_microbenchmark": run_decoder_microbenchmark(),
        "corner_accuracy": run_corner_accuracy(seed=seed, threads=threads),
    }
    return report, raw_samples


def run_decoder_microbenchmark() -> dict[str, object]:
    """Compare the compact decoder with an exhaustive reference after exact output checks."""

    try:
        _ensure_repository_root_on_path()
        from experiments.occlusion_detector.robust_apriltag.family import (
            FULL_MASK,
            ROTATED_CODES,
            TAG36H11,
            decode_observed,
            popcount,
            rotate90_code,
        )
    except ImportError as exc:
        return {"skipped": f"robust decoder unavailable: {exc}"}

    queries: list[tuple[int, int, int, int]] = []
    for tag_id in TAG_IDS:
        observed = TAG36H11.codes[tag_id]
        for _rotation in range(4):
            queries.append((observed, FULL_MASK, 2, 36))
            queries.append((observed ^ (1 << 3), FULL_MASK, 2, 36))
            queries.append((observed ^ (1 << 3) ^ (1 << 29), FULL_MASK, 2, 36))
            queries.append((observed ^ (1 << 3), FULL_MASK ^ (1 << 7) ^ (1 << 21), 2, 30))
            observed = rotate90_code(observed)

    def exhaustive(code: int, mask: int, max_hamming: int, min_reliable_bits: int) -> tuple[int, int, int, int] | None:
        if popcount(mask) < min_reliable_bits:
            return None
        best: tuple[int, int, int] | None = None
        second = 10_000
        for family_code, tag_id, rotation in ROTATED_CODES:
            hamming = popcount((code ^ family_code) & mask)
            if best is None or hamming < best[1]:
                if best is not None:
                    second = min(second, best[1])
                best = (tag_id, hamming, rotation)
            elif hamming < second:
                second = hamming
        if best is None or best[1] > max_hamming:
            return None
        margin = 11 - best[1] if mask == FULL_MASK and max_hamming <= 2 else second - best[1]
        return best[0], best[1], best[2], margin

    def compact(query: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
        return decode_observed(query[0], query[1], max_hamming=query[2], min_reliable_bits=query[3])

    expected = [exhaustive(*query) for query in queries]
    actual = [compact(query) for query in queries]
    if actual != expected:
        raise RuntimeError("compact decoder disagrees with exhaustive rotated-word reference")
    loops = 100
    start_ns = perf_counter_ns()
    for _ in range(loops):
        for query in queries:
            exhaustive(*query)
    exhaustive_ns = perf_counter_ns() - start_ns
    start_ns = perf_counter_ns()
    for _ in range(loops):
        for query in queries:
            compact(query)
    compact_ns = perf_counter_ns() - start_ns
    return {
        "separate_from_whole_frame_speed": True,
        "queries": len(queries),
        "verification": "all compact outputs match exhaustive rotated-word reference before timing",
        "exhaustive_ns": exhaustive_ns,
        "compact_ns": compact_ns,
        "speedup": exhaustive_ns / compact_ns if compact_ns else float("inf"),
    }


def write_reports(
    report: dict[str, object],
    samples: Sequence[dict[str, object]],
    output: Path,
    csv_output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    write_benchmark_summary(report, output, pretty=True, durable=True)
    _write_csv_atomically(samples, csv_output)


def build_parser() -> argparse.ArgumentParser:
    today = datetime.now().date().isoformat()
    parser = argparse.ArgumentParser(
        description="Run deterministic AprilTag detector, occlusion, and decoder benchmarks"
    )
    parser.add_argument("--output", type=Path, default=Path(f"benchmarks/results/napa-performance-{today}.json"))
    parser.add_argument("--samples-csv", type=Path, help="Raw detector-call samples CSV (defaults beside --output)")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--repetitions", type=_positive_int, default=10)
    parser.add_argument(
        "--occlusion-repetitions",
        type=_positive_int,
        default=1,
        help="Measured passes over the occlusion sweep, separate from --repetitions",
    )
    parser.add_argument("--warmup", type=_nonnegative_int, default=2)
    parser.add_argument(
        "--threads", type=_positive_int, default=1, help="OpenCV and RobotPy AprilTag3 detector threads"
    )
    parser.add_argument(
        "--resolutions",
        type=_resolution_list,
        default=DEFAULT_RESOLUTIONS,
        help="Comma-separated WIDTHxHEIGHT values",
    )
    parser.add_argument(
        "--tag-side-pixels",
        type=_positive_int_list,
        default=DEFAULT_TAG_SIDES,
        help="Comma-separated marker sides; defaults to 96,160",
    )
    parser.add_argument(
        "--include-robust-erased10",
        action="store_true",
        help="Add experimental Robust36h11 max_erased_bits=10 full-search backend",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frames = build_corpus(resolutions=args.resolutions, tag_side_pixels=args.tag_side_pixels, seed=args.seed)
        report, samples = run_matrix(
            frames,
            repetitions=args.repetitions,
            occlusion_repetitions=args.occlusion_repetitions,
            warmup=args.warmup,
            seed=args.seed,
            threads=args.threads,
            include_robust_erased10=args.include_robust_erased10,
        )
        csv_output = args.samples_csv or args.output.with_suffix(".csv")
        write_reports(report, samples, args.output, csv_output)
        print(json.dumps({"report": str(args.output), "samples_csv": str(csv_output)}, indent=2))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"performance-matrix: error: {exc}", file=sys.stderr)
        return 2
    return 0


def _single_tag_frame(
    *, name: str, width: int, height: int, tag_id: int, side: int, rotation: int, scenario: str
) -> BenchmarkFrame:
    image = np.full((height, width), 255, dtype=np.uint8)
    x = (width - side) // 2
    y = (height - side) // 2
    _place_marker(image, tag_id, side, rotation, x, y)
    return BenchmarkFrame(name, image, (tag_id,), (width, height), scenario, rotation, side)


def _four_tag_frame(width: int, height: int, requested_side: int) -> BenchmarkFrame:
    side = min(requested_side, max(32, min(width, height) // 4))
    image = np.full((height, width), 255, dtype=np.uint8)
    positions = (
        (width // 4 - side // 2, height // 4 - side // 2),
        (3 * width // 4 - side // 2, height // 4 - side // 2),
        (width // 4 - side // 2, 3 * height // 4 - side // 2),
        (3 * width // 4 - side // 2, 3 * height // 4 - side // 2),
    )
    ids = LAYOUT_TAG_IDS
    for index, (x, y) in enumerate(positions):
        _place_marker(image, ids[index], side, index, x, y)
    return BenchmarkFrame(
        f"four-tags-s{requested_side}",
        image,
        ids,
        (width, height),
        "four-tag",
        tag_side_px=side,
    )


def _perspective_frame(width: int, height: int, side: int) -> BenchmarkFrame:
    image = np.full((height, width), 255, dtype=np.uint8)
    marker = _marker(314, side, 3)
    source = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
    cx, cy = width // 2, height // 2
    target = np.array(
        [
            [cx - side * 0.52, cy - side * 0.42],
            [cx + side * 0.45, cy - side * 0.55],
            [cx + side * 0.58, cy + side * 0.43],
            [cx - side * 0.46, cy + side * 0.57],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(marker, transform, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
    mask = cv2.warpPerspective(np.full_like(marker, 255), transform, (width, height), flags=cv2.INTER_NEAREST)
    image[mask > 0] = warped[mask > 0]
    return BenchmarkFrame(f"perspective-id314-s{side}", image, (314,), (width, height), "perspective", 3, side)


def _occlusion_frame(
    *, width: int, height: int, tag_id: int, side: int, rotation: int, fraction: float, region: str, value: int
) -> BenchmarkFrame:
    image = np.full((height, width), 255, dtype=np.uint8)
    x = (width - side) // 2
    y = (height - side) // 2
    marker = _marker(tag_id, side, rotation)
    original = marker.copy()
    x0, y0, x1, y1 = _occlusion_box(side, fraction, region)
    marker[y0:y1, x0:x1] = value
    image[y : y + side, x : x + side] = marker
    raster_pixels = (x1 - x0) * (y1 - y0)
    changed_pixels = int(np.count_nonzero(original[y0:y1, x0:x1] != value))
    occlusion = {
        "region": region,
        "mask_value": value,
        "requested_fraction": fraction,
        "geometric_fraction": raster_pixels / float(side * side),
        "raster_fraction": raster_pixels / float(side * side),
        "changed_pixel_fraction": changed_pixels / float(side * side),
    }
    name = f"occlusion-{region}-f{fraction:.2f}-v{value}-id{tag_id}-r{rotation}-s{side}"
    return BenchmarkFrame(name, image, (tag_id,), (width, height), "occlusion", rotation, side, occlusion)


def _occlusion_box(side: int, fraction: float, region: str) -> tuple[int, int, int, int]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("occlusion fraction must be between zero and one")
    if fraction == 0.0:
        return (side // 2, side // 2, side // 2, side // 2)
    if region == "payload-center":
        payload_start = side // 8
        payload_end = side - payload_start
        length = min(payload_end - payload_start, max(1, round((fraction**0.5) * side)))
        x0 = payload_start + (payload_end - payload_start - length) // 2
        y0 = payload_start + (payload_end - payload_start - length) // 2
        return x0, y0, x0 + length, y0 + length
    if region == "border-strip":
        length = min(side, max(1, round(fraction * side)))
        return 0, 0, length, side
    if region == "corner":
        length = min(side, max(1, round((fraction**0.5) * side)))
        return 0, 0, length, length
    raise ValueError(f"unknown occlusion region {region!r}")


def _place_marker(image: np.ndarray, tag_id: int, side: int, rotation: int, x: int, y: int) -> None:
    image[y : y + side, x : x + side] = _marker(tag_id, side, rotation)


def _marker(tag_id: int, side: int, rotation: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, side, borderBits=1)
    return np.rot90(marker, rotation).copy()


def _textured_negative(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    image = np.full((height, width), 235, dtype=np.uint8)
    for _ in range(36):
        x0, x1 = sorted(rng.integers(0, width, size=2).tolist())
        y0, y1 = sorted(rng.integers(0, height, size=2).tolist())
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        value = int(rng.integers(70, 210))
        if rng.integers(0, 2):
            cv2.rectangle(image, (x0, y0), (x1, y1), value, thickness=2)
        else:
            cv2.line(image, (x0, y0), (x1, y1), value, thickness=2)
    return image


def _stress_frames(width: int, height: int, side: int, rng: np.random.Generator) -> list[BenchmarkFrame]:
    """Camera-like degradations of one tag, each a separate scenario."""

    resolution = (width, height)
    base = _single_tag_frame(
        name="stress", width=width, height=height, tag_id=STRESS_TAG_ID, side=side, rotation=3, scenario="stress"
    ).image.astype(np.float32)
    noisy = base + rng.normal(0.0, 6.0, base.shape)
    gradient = base * np.linspace(0.35, 1.0, width, dtype=np.float32)[None, :]
    kernel = np.zeros((7, 7), np.float32)
    kernel[3, :] = 1.0 / 7.0
    motion = cv2.filter2D(base, -1, kernel)
    clutter = np.full((height, width), 225, np.uint8)
    for _ in range(max(6, (width * height) // 40_000)):
        origin = (int(rng.integers(0, max(1, width - 240))), int(rng.integers(24, height)))
        cv2.putText(clutter, "NAPA 2468 ##", origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, int(rng.integers(0, 90)), 2)
    # The tag keeps its one-cell white quiet zone, as a printed tag does.
    margin = side // 8 + 2
    x = (width - side) // 2
    y = (height - side) // 2
    clutter[y - margin : y + side + margin, x - margin : x + side + margin] = 255
    clutter[y : y + side, x : x + side] = base[y : y + side, x : x + side].astype(np.uint8)
    scenes = (("noisy", noisy), ("lighting-gradient", gradient), ("motion-blur", motion), ("cluttered", clutter))
    return [
        BenchmarkFrame(
            f"{scenario}-id{STRESS_TAG_ID}-s{side}",
            np.clip(image, 0, 255).astype(np.uint8),
            (STRESS_TAG_ID,),
            resolution,
            scenario,
            3,
            side,
        )
        for scenario, image in scenes
    ]


def corner_accuracy_scene(
    tag_id: int,
    side: float,
    angle_rad: float,
    center: tuple[float, float],
    blur_sigma: float,
    noise_sigma: float,
    rng: np.random.Generator,
    size: tuple[int, int] = (1280, 720),
) -> tuple[np.ndarray, np.ndarray]:
    """Render a tag with exact sub-pixel corners: supersample 4x, then area-average like a sensor."""

    width, height = size
    cells = 10
    padded = np.full((10 * cells, 10 * cells), 255, np.uint8)
    padded[cells : 9 * cells, cells : 9 * cells] = _marker(tag_id, 8 * cells, 0)
    source = np.array([[cells, cells], [9 * cells, cells], [9 * cells, 9 * cells], [cells, 9 * cells]], np.float32)
    source -= 0.5
    rotation = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    unit = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) + rng.uniform(-0.06, 0.06, (4, 2))
    corners = unit @ rotation.T * side + np.asarray(center)
    homography = cv2.getPerspectiveTransform(source, (corners * 4 + 1.5).astype(np.float32))
    large = cv2.warpPerspective(padded, homography, (width * 4, height * 4), borderValue=255)
    image = cv2.resize(large, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    image = image * 0.8 + 20 + rng.normal(0.0, noise_sigma, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), corners


def run_corner_accuracy(*, seed: int = 2468, threads: int = 1) -> dict[str, object]:
    """Score detected corners against exact ground truth on seeded sub-pixel renders.

    AprilTag reports corners with a +0.5 px origin and its own corner order; its corners are
    shifted by -0.5 px and aligned to OpenCV order before scoring, so only real error counts.
    """

    rng = np.random.default_rng(seed)
    adapters: dict[str, Callable[[np.ndarray], list[tuple[int, np.ndarray]]]] = {}
    if native_available():
        native = NapaDetector()
        adapters["napa-native"] = lambda image: [(o.tag_id, o.corners) for o in native.detect(image)]
    opencv = OpenCVArucoFiducialDetector()
    adapters["opencv-aruco"] = lambda image: [(o.tag_id, o.corners) for o in opencv.detect(image)]
    try:
        from robotpy_apriltag import AprilTagDetector
    except ImportError:
        pass
    else:
        for decimate in (1.0, 2.0):
            detector = AprilTagDetector()
            detector.addFamily("tag36h11", 2)
            config = detector.getConfig()
            config.numThreads = threads
            config.quadDecimate = decimate
            detector.setConfig(config)

            def apriltag(image: np.ndarray, detector=detector) -> list[tuple[int, np.ndarray]]:
                return [
                    (int(d.getId()), np.array([[d.getCorner(i).x, d.getCorner(i).y] for i in range(4)]) - 0.5)
                    for d in detector.detect(image)
                ]

            adapters[f"apriltag3-robotpy-decimate-{int(decimate)}"] = apriltag
    errors: dict[str, dict[int, list[float]]] = {name: defaultdict(list) for name in adapters}
    scenes = 0
    for side in CORNER_TAG_SIDES:
        for pose in range(CORNER_POSES_PER_SIDE):
            tag_id = int(rng.integers(0, 587))
            image, truth = corner_accuracy_scene(
                tag_id,
                float(side),
                float(rng.uniform(0, 2 * np.pi)),
                (float(rng.uniform(300, 980)), float(rng.uniform(250, 470))),
                (0.0, 0.8)[pose % 2],
                2.0,
                rng,
            )
            scenes += 1
            for name, detect in adapters.items():
                for found_id, corners in detect(image):
                    if found_id != tag_id:
                        continue
                    if name.startswith("apriltag3"):
                        corners = min(
                            (np.roll(order, shift, 0) for order in (corners, corners[::-1]) for shift in range(4)),
                            key=lambda candidate: float(np.abs(candidate - truth).max()),
                        )
                    errors[name][side].append(float(np.linalg.norm(corners - truth, axis=1).max()))
                    break
    report: dict[str, object] = {
        "scope": (
            "Worst corner's Euclidean error per detected tag, against exact homography ground truth, on 1280x720 "
            "renders supersampled 4x with mild perspective, sigma 0/0.8 blur and sigma 2 noise. AprilTag corners "
            "are shifted -0.5 px to OpenCV's pixel-centre convention and re-ordered before scoring."
        ),
        "scenes": scenes,
        "tag_sides_px": list(CORNER_TAG_SIDES),
        "backends": {},
    }
    for name, by_side in errors.items():
        values = [value for side in CORNER_TAG_SIDES for value in by_side[side]]
        report["backends"][name] = {
            "found": len(values),
            "median_px": float(np.median(values)) if values else None,
            "p95_px": float(np.percentile(values, 95)) if values else None,
            "max_px": float(np.max(values)) if values else None,
            "by_side": {
                str(side): {
                    "found": len(by_side[side]),
                    "median_px": float(np.median(by_side[side])) if by_side[side] else None,
                }
                for side in CORNER_TAG_SIDES
            },
        }
    return report


def _raw_opencv_detector(
    dictionary: cv2.aruco.Dictionary, parameters: cv2.aruco.DetectorParameters
) -> Callable[[np.ndarray], list[int]]:
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(image: np.ndarray) -> list[int]:
        _corners, ids, _rejected = detector.detectMarkers(image)
        return [] if ids is None else [int(tag_id) for tag_id in ids.reshape(-1)]

    return detect


def _napa_native_detector() -> Callable[[np.ndarray], list[int]]:
    detector = NapaDetector()
    return lambda image: [observation.tag_id for observation in detector.detect(image)]


def _napa_default_detector() -> Callable[[np.ndarray], list[int]]:
    detector = OpenCVArucoFiducialDetector()
    return lambda image: [observation.tag_id for observation in detector.detect(image)]


def _napa_aruco3_detector() -> Callable[[np.ndarray], list[int]]:
    detector = make_apriltag_detector("aruco3")
    return lambda image: [observation.tag_id for observation in detector.detect(image)]


def _napa_skip_scan_detector() -> Callable[[np.ndarray], list[int]]:
    detector = OpenCVArucoFiducialDetector(prefilter=BlockSkipPrefilter(SkipScanConfig(enabled=True)))
    return lambda image: [observation.tag_id for observation in detector.detect(image)]


def _napa_layout_detector(tag_ids: Sequence[int]) -> Callable[[np.ndarray], list[int]]:
    detector = make_apriltag_detector("layout-restricted", layout_tag_ids=tag_ids)
    return lambda image: [observation.tag_id for observation in detector.detect(image)]


def _robotpy_detector(
    *,
    threads: int,
    quad_decimate: float,
    recorded_config: dict[str, object],
) -> Callable[[np.ndarray], list[int]]:
    from robotpy_apriltag import AprilTagDetector

    detector = AprilTagDetector()
    if not detector.addFamily("tag36h11", 2):
        raise RuntimeError("RobotPy AprilTagDetector could not load tag36h11")
    config = detector.getConfig()
    config.numThreads = threads
    config.quadDecimate = quad_decimate
    detector.setConfig(config)
    applied = detector.getConfig()
    if applied.numThreads != threads or applied.quadDecimate != quad_decimate:
        raise RuntimeError("RobotPy AprilTagDetector did not apply requested thread or decimate configuration")
    recorded_config["applied_config"] = {
        "debug": bool(applied.debug),
        "decode_sharpening": float(applied.decodeSharpening),
        "num_threads": int(applied.numThreads),
        "quad_decimate": float(applied.quadDecimate),
        "quad_sigma": float(applied.quadSigma),
        "refine_edges": bool(applied.refineEdges),
    }
    return lambda image: [int(detection.getId()) for detection in detector.detect(image)]


def _robust_detector(*, max_erased_bits: int) -> Callable[[np.ndarray], list[int]]:
    _ensure_repository_root_on_path()
    from experiments.occlusion_detector.robust_apriltag import Robust36h11Detector

    detector = Robust36h11Detector(refine_pose=False, max_erased_bits=max_erased_bits)
    return lambda image: [int(detection.id) for detection in detector.detect(image)]


def _ensure_repository_root_on_path() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))


def _opencv_parameters(parameters: cv2.aruco.DetectorParameters) -> dict[str, object]:
    names = (
        "useAruco3Detection",
        "minMarkerLengthRatioOriginalImg",
        "cornerRefinementMethod",
        "markerBorderBits",
        "errorCorrectionRate",
    )
    return {name: getattr(parameters, name) for name in names if hasattr(parameters, name)}


def _aruco3_parameters() -> cv2.aruco.DetectorParameters:
    parameters = cv2.aruco.DetectorParameters()
    if not hasattr(parameters, "useAruco3Detection"):
        raise RuntimeError("This OpenCV build does not support ArUco3 detection")
    parameters.useAruco3Detection = True
    parameters.minMarkerLengthRatioOriginalImg = 0.0
    return parameters


def _dictionary_config(dictionary: cv2.aruco.Dictionary) -> dict[str, int]:
    return {
        "marker_size": int(dictionary.markerSize),
        "max_correction_bits": int(dictionary.maxCorrectionBits),
    }


def _napa_config(
    candidate: str,
    dictionary: cv2.aruco.Dictionary,
    parameters: cv2.aruco.DetectorParameters,
    open_cv_threads: int,
    **extra: object,
) -> dict[str, object]:
    return {
        "wrapper": "OpenCVArucoFiducialDetector",
        "candidate": candidate,
        "dictionary": "DICT_APRILTAG_36h11",
        "dictionary_config": _dictionary_config(dictionary),
        "parameters": _opencv_parameters(parameters),
        "opencv_threads_effective": open_cv_threads,
        **extra,
    }


def _metrics_dict(metrics: IdMetrics) -> dict[str, float | int | None]:
    return {
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "precision": metrics.precision,
        "recall": metrics.recall,
    }


def _add_metrics(left: IdMetrics, right: IdMetrics) -> IdMetrics:
    return IdMetrics(
        left.true_positive + right.true_positive,
        left.false_positive + right.false_positive,
        left.false_negative + right.false_negative,
    )


def _corpus_summary(frames: Sequence[BenchmarkFrame]) -> dict[str, object]:
    return {
        "frames": len(frames),
        "resolutions": sorted({f"{frame.resolution[0]}x{frame.resolution[1]}" for frame in frames}),
        "scenarios": sorted({frame.scenario for frame in frames}),
        "expected_ids": sorted({tag_id for frame in frames for tag_id in frame.expected_ids}),
        "in_memory": True,
        "image_dtype": "uint8",
        "image_channels": 1,
    }


def _scenario_report(
    totals: dict[tuple[str, str, str], IdMetrics],
    latency: dict[tuple[str, str, str], list[int]],
) -> list[dict[str, object]]:
    return [
        {
            "backend": backend,
            "scenario": scenario,
            "resolution": resolution,
            "latency": summarize_latency(latency[(backend, scenario, resolution)]),
            "id_metrics": _metrics_dict(metrics),
        }
        for (backend, scenario, resolution), metrics in sorted(totals.items())
    ]


def _occlusion_report(
    totals: dict[tuple[str, str, str, float, str, int], IdMetrics],
    samples: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    area_values: dict[tuple[object, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    area_names = ("geometric_fraction", "raster_fraction", "changed_pixel_fraction")
    for sample in samples:
        if sample.get("stage") != "occlusion":
            continue
        key = (
            sample["backend"],
            sample["resolution"],
            sample["occlusion_region"],
            float(sample["occlusion_requested_fraction"]),
            str(sample["occlusion_mask_value"]),
            int(sample["tag_side_px"]),
        )
        for name in area_names:
            value = sample.get(f"occlusion_{name}")
            if value is not None and value != "":
                area_values[key][name].append(float(value))
    report: list[dict[str, object]] = []
    for (backend, resolution, region, fraction, mask_value, tag_side_px), metrics in sorted(totals.items()):
        report.append(
            {
                "backend": backend,
                "resolution": resolution,
                "region": region,
                "requested_fraction": fraction,
                "mask_value": mask_value,
                "tag_side_px": tag_side_px,
                "area_fractions": {
                    name: {"min": min(values), "mean": float(np.mean(values)), "max": max(values)}
                    for name, values in area_values[
                        (backend, resolution, region, fraction, mask_value, tag_side_px)
                    ].items()
                },
                "id_metrics": _metrics_dict(metrics),
            }
        )
    return report


def _metadata() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "opencv_version": cv2.__version__,
        "opencv_build": cv2.getBuildInformation(),
        "dependencies": {
            name: _package_version(name)
            for name in (
                "numpy",
                "opencv-python",
                "opencv-python-headless",
                "opencv-contrib-python",
                "robotpy-apriltag",
            )
        },
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip() or None
    except OSError:
        pass
    return platform.processor() or None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _write_csv_atomically(rows: Sequence[dict[str, object]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        raise ValueError("CSV rows must not be empty")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", newline="", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as file:
            temporary_name = file.name
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _resolution_list(value: str) -> tuple[tuple[int, int], ...]:
    resolutions: list[tuple[int, int]] = []
    for item in value.split(","):
        try:
            width_text, height_text = item.lower().split("x", 1)
            width, height = int(width_text), int(height_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("resolutions must use WIDTHxHEIGHT, separated by commas") from exc
        if width < 160 or height < 120:
            raise argparse.ArgumentTypeError("resolutions must be at least 160x120")
        resolutions.append((width, height))
    if not resolutions:
        raise argparse.ArgumentTypeError("at least one resolution is required")
    return tuple(resolutions)


def _positive_int_list(value: str) -> tuple[int, ...]:
    values = tuple(_positive_int(item) for item in value.split(","))
    if not values:
        raise argparse.ArgumentTypeError("at least one positive integer is required")
    return values


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


if __name__ == "__main__":
    raise SystemExit(main())
