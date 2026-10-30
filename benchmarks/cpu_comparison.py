"""NAPA against other CPU tag36h11 detectors, on the performance-matrix corpus plus explicit stress cases.

Subcommands:

- ``validate``: check each adapter's pixel origin, corner order, and false detections on known geometry
  (clean tags at known corners, every in-plane quarter turn, tilts, blank frames) before any score is used.
- ``latency``: every backend on every frame in one process, interleaved in a shuffled order per repetition,
  each detector constructed once and warmed up. Times only the detector call (and its adapter's result
  conversion); process CPU time is read around each call. Scores recall, false IDs, duplicates, corner error
  (after aligning corner order) and corner order separately.
- ``corpus``: write a frame subset to ``.npz`` for ``resources``.
- ``resources``: each backend alone in its own pinned process (``cpu_backends.py``), serial or
  frame-parallel, for CPU time per frame, throughput, and peak RSS. Rounds run in shuffled backend order,
  one process at a time.

Rendering and scoring reuse ``performance_matrix``; per-frame speed ratios use its ``_speedups``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns, process_time_ns

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import cpu_backends as cb  # noqa: E402
from benchmarks import performance_matrix as pm  # noqa: E402

SCHEMA_VERSION = 1
GROSS_CORNER_PX = 2.0

# ------------------------------------------------------------------------------------ stress cases


def _stress_tiny(rng: np.random.Generator, width: int, height: int):
    """1-4 tags with a 12-31 px black square: below NAPA's default min_tag_px of 32."""

    image = pm._uniform(rng, width, height) if rng.random() < 0.5 else pm._smooth_texture(rng, width, height)
    tags = pm.place_tags(rng, width, height, int(rng.integers(1, 5)), (12.0, 31.0), max_tilt_deg=10)
    for tag in tags:
        pm.render_tag(image, tag.tag_id, tag.corners, black=rng.uniform(15, 60), white=rng.uniform(200, 250))
    return pm.add_noise(rng, image, 1.5), tags, {"background": "uniform/texture"}


def _stress_heavy_noise(rng: np.random.Generator, width: int, height: int):
    """Sensor noise sigma 15-30 grey levels, beyond the corpus's 4-14."""

    image, tags, details = pm._tags_scene(rng, width, height)
    sigma = round(float(rng.uniform(15, 30)), 1)
    details["noise_sigma"] = sigma
    return pm.add_noise(rng, image, sigma), tags, details


def _stress_dense(rng: np.random.Generator, width: int, height: int):
    """Up to 30-80 non-overlapping tags of 36-80 px (scaled), on mixed backgrounds."""

    scale = pm._scale(width, height)
    image, tags, details = pm._tags_scene(rng, width, height, count=(30, 80), side=(36 * scale, 80 * scale))
    return pm.add_noise(rng, image, 1.5), tags, details


def _stress_clutter_noisy(rng: np.random.Generator, width: int, height: int):
    """No tags: shapes, text, checkerboards, or texture with sigma 6-12 noise."""

    image, name = pm._random_background(rng, width, height)
    while name == "uniform":
        image, name = pm._random_background(rng, width, height)
    sigma = round(float(rng.uniform(6, 12)), 1)
    return pm.add_noise(rng, image, sigma), [], {"background": name, "noise_sigma": sigma}


STRESS: dict[str, Callable] = {
    "stress-tiny": _stress_tiny,
    "stress-heavy-noise": _stress_heavy_noise,
    "stress-dense": _stress_dense,
    "stress-clutter-noisy": _stress_clutter_noisy,
}
STRESS_DESCRIPTIONS = {name: fn.__doc__.strip() for name, fn in STRESS.items()}


def render_stress(name: str, index: int, resolution: tuple[int, int], seed: int = 2468) -> pm.Frame:
    width, height = resolution
    rng = np.random.default_rng((seed, 9000 + list(STRESS).index(name), width, height, index))
    image, tags, details = STRESS[name](rng, width, height)
    return pm.Frame(name, index, resolution, pm._to_uint8(image), tags, details)


def iter_corpus(
    *,
    scenarios: Sequence[str],
    stress: Sequence[str],
    resolutions: Sequence[tuple[int, int]],
    frames: int,
    stress_frames: int,
    range_cases: int,
    seed: int,
    real_images: Sequence[Path] = (),
) -> Iterator[pm.Frame]:
    for resolution in resolutions:
        for name in scenarios:
            for index in range(frames):
                yield pm.render_frame(name, index, resolution, seed)
        for name in stress:
            for index in range(stress_frames):
                yield render_stress(name, index, resolution, seed)
    if range_cases:
        yield from pm.iter_range_frames(
            resolution=min(resolutions, key=lambda r: r[0] * r[1]), cases=range_cases, seed=seed
        )
    yield from real_frames(real_images)


REAL = "real-unannotated"


def frame_by_name(name: str, seed: int = 2468, range_cases: int = 30) -> pm.Frame:
    """Re-render one frame from its name (``scenario-WxH-index``), e.g. to re-time an outlier alone."""

    scenario, size, index = name.rsplit("-", 2)
    resolution = tuple(int(v) for v in size.split("x"))
    if scenario in STRESS:
        return render_stress(scenario, int(index), resolution, seed)
    if scenario == "range":
        return pm.render_range_frame(int(index), resolution=resolution, cases=range_cases, seed=seed)
    return pm.render_frame(scenario, int(index), resolution, seed)


def real_frames(paths: Sequence[Path]) -> Iterator[pm.Frame]:
    """Photos loaded as grayscale. They carry no ground truth: rows keep raw detections, never scores."""

    import cv2

    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"cannot read {path}")
        frame = pm.Frame(
            REAL,
            index,
            (image.shape[1], image.shape[0]),
            np.ascontiguousarray(image),
            [],
            {"path": str(path), "annotated": False},
        )
        yield frame


# ----------------------------------------------------------------------------------------- scoring

# The eight ways to read four cyclic corners: (reflected, shift). (False, 0) is the truth's order, which is
# OpenCV's DICT_APRILTAG_36h11 marker order (top-left of the marker image first, clockwise).
ALIGNMENTS = tuple((reflected, shift) for reflected in (False, True) for shift in range(4))


def _aligned(corners: np.ndarray, reflected: bool, shift: int) -> np.ndarray:
    return np.roll(corners[::-1] if reflected else corners, shift, axis=0)


def score_detail(truth: Sequence[pm.TagTruth], detections: Sequence[cb.Detection]) -> dict[str, object]:
    """IDs and corners against ground truth, keeping geometry and corner order separate.

    A detection whose ID isn't in the frame is a false ID; a second detection of an ID already matched is a
    duplicate. Corner error is measured after the corner order that fits best (geometry only); that order is
    reported as the alignment, and "canonical" means the detector's own order was already the truth's.
    """

    expected = {tag.tag_id: tag for tag in truth}
    matched: dict[int, dict[str, object]] = {}
    false_ids = duplicates = 0
    for tag_id, corners in detections:
        tag = expected.get(tag_id)
        if tag is None:
            false_ids += 1
            continue
        if tag_id in matched:
            duplicates += 1
            continue
        corners = np.asarray(corners, np.float64).reshape(4, 2)
        best = min(ALIGNMENTS, key=lambda a: float(np.linalg.norm(_aligned(corners, *a) - tag.corners, axis=1).max()))
        distances = np.linalg.norm(_aligned(corners, *best) - tag.corners, axis=1)
        matched[tag_id] = {
            "worst_px": float(distances.max()),
            "rms_px": float(np.sqrt(np.mean(distances**2))),
            "alignment": f"{'r' if best[0] else ''}s{best[1]}",
            "side_px": tag.side_px,
            "offset": (_aligned(corners, *best) - tag.corners).mean(axis=0).tolist(),
        }
    return {
        "true_positive": len(matched),
        "false_id": false_ids,
        "duplicate": duplicates,
        "false_negative": len(expected) - len(matched),
        "matched": matched,
    }


def _canonical_ids(detections: Sequence[cb.Detection]) -> list[tuple[int, tuple[float, ...]]]:
    return sorted((int(i), tuple(np.round(np.asarray(c, np.float64).ravel(), 6))) for i, c in detections)


# ------------------------------------------------------------------------------------------ runner


def run_interleaved(
    frames: Iterator[pm.Frame] | Sequence[pm.Frame],
    *,
    backends: Sequence[cb.Backend],
    repetitions: int,
    warmup: int,
    seed: int,
    progress: Callable[[int], None] | None = None,
) -> tuple[dict[str, dict], list[dict[str, object]], dict[str, object]]:
    """Time and score every backend on every frame, interleaved in a shuffled order per repetition."""

    lifecycle: dict[str, dict[str, float]] = {}
    instances: dict[str, cb.DetectFn] = {}
    for backend in backends:
        start = perf_counter_ns()
        instances[backend.name] = backend.factory()
        lifecycle[backend.name] = {"init_ms": (perf_counter_ns() - start) / 1e6}
    shuffler = random.Random(seed)
    rows: list[dict[str, object]] = []
    observed: dict[str, object] = {}
    gc.disable()  # collections run between frames, never inside a timed call
    try:
        for count, frame in enumerate(frames, start=1):
            image = frame.image
            if count == 1:
                for backend in backends:
                    start = perf_counter_ns()
                    instances[backend.name](image)
                    lifecycle[backend.name]["first_call_ms"] = (perf_counter_ns() - start) / 1e6
                observed["threads_after_first_frame"] = cb.thread_report()
            for _ in range(warmup):
                order = list(backends)
                shuffler.shuffle(order)
                for backend in order:
                    instances[backend.name](image)
            samples: dict[str, list[int]] = {b.name: [] for b in backends}
            cpu: dict[str, list[int]] = {b.name: [] for b in backends}
            first: dict[str, list[cb.Detection]] = {}
            stable: dict[str, bool] = {b.name: True for b in backends}
            for _ in range(repetitions):
                order = list(backends)
                shuffler.shuffle(order)
                for backend in order:
                    detect = instances[backend.name]
                    c0 = process_time_ns()
                    t0 = perf_counter_ns()
                    detections = detect(image)
                    t1 = perf_counter_ns()
                    c1 = process_time_ns()
                    samples[backend.name].append(t1 - t0)
                    cpu[backend.name].append(c1 - c0)
                    if backend.name not in first:
                        first[backend.name] = detections
                    elif stable[backend.name] and _canonical_ids(detections) != _canonical_ids(first[backend.name]):
                        stable[backend.name] = False
            for backend in backends:
                score = score_detail(frame.tags, first[backend.name])
                matched = score["matched"]
                latency = samples[backend.name]
                rows.append(
                    {
                        "frame": frame.name,
                        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest()[:16],
                        "scenario": frame.scenario,
                        "resolution": f"{frame.resolution[0]}x{frame.resolution[1]}",
                        "backend": backend.name,
                        "tags": len(frame.tags),
                        "min_side_px": round(min((t.side_px for t in frame.tags), default=0.0), 1),
                        "expected_ids": ";".join(str(t.tag_id) for t in frame.tags),
                        "detected_ids": ";".join(str(i) for i, _ in first[backend.name]),
                        "true_positive": score["true_positive"],
                        "false_id": score["false_id"],
                        "duplicate": score["duplicate"],
                        "false_negative": score["false_negative"],
                        "tp_ids": ";".join(map(str, matched)),
                        "corner_errors_px": ";".join(f"{m['worst_px']:.4f}" for m in matched.values()),
                        "corner_rms_px": ";".join(f"{m['rms_px']:.4f}" for m in matched.values()),
                        "alignments": ";".join(str(m["alignment"]) for m in matched.values()),
                        "offsets_px": ";".join(f"{m['offset'][0]:.4f},{m['offset'][1]:.4f}" for m in matched.values()),
                        "stable_across_repetitions": stable[backend.name],
                        "median_ms": round(float(np.median(latency)) / 1e6, 5),
                        "min_ms": round(min(latency) / 1e6, 5),
                        "max_ms": round(max(latency) / 1e6, 5),
                        "samples_ns": ";".join(map(str, latency)),
                        "cpu_ns": ";".join(map(str, cpu[backend.name])),
                        "detections_json": ""
                        if frame.scenario != REAL
                        else json.dumps([[i, np.round(np.asarray(c), 4).tolist()] for i, c in first[backend.name]]),
                        **{f"detail_{key}": value for key, value in frame.details.items()},
                    }
                )
            gc.collect()
            if progress is not None:
                progress(count)
    finally:
        gc.enable()
    observed["threads_end"] = cb.thread_report()
    return lifecycle, rows, observed


def _split(value: object, cast=float) -> list:
    return [cast(v) for v in str(value).split(";") if v != ""]


def aggregate(rows: Sequence[dict[str, object]], baseline: str = "napa") -> dict[str, object]:
    """Per (scenario, resolution, backend) quality and latency, per-resolution totals, and per-frame ratios."""

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["scenario"]), str(row["resolution"]), str(row["backend"]))
        groups[key].append(row)
        if row["scenario"] in pm.SCENARIO_NAMES:
            groups[("all-corpus", str(row["resolution"]), str(row["backend"]))].append(row)
        elif str(row["scenario"]).startswith("stress"):
            groups[("all-stress", str(row["resolution"]), str(row["backend"]))].append(row)
    results = []
    for (scenario, resolution, backend), members in sorted(groups.items()):
        samples = [v for row in members for v in _split(row["samples_ns"], int)]
        cpu = [v for row in members for v in _split(row["cpu_ns"], int)]
        errors = [v for row in members for v in _split(row["corner_errors_px"])]
        alignments = Counter(a for row in members for a in str(row["alignments"]).split(";") if a)
        tags = sum(int(row["tags"]) for row in members)
        found = sum(int(row["true_positive"]) for row in members)
        modal = alignments.most_common(1)[0] if alignments else None
        if scenario == REAL:  # no truth: only timing and what was reported
            results.append(
                {
                    "scenario": scenario,
                    "resolution": resolution,
                    "backend": backend,
                    "frames": len(members),
                    "annotated": False,
                    "detections": sum(len(_split(r["detected_ids"], int)) for r in members),
                    "unstable_frames": sum(1 for r in members if str(r["stable_across_repetitions"]) != "True"),
                    "latency": pm.summarize_latency(samples),
                    "frame_median_ms": pm.summarize_latency([round(float(r["median_ms"]) * 1e6) for r in members]),
                    "cpu_ms_mean": float(np.mean(cpu)) / 1e6,
                    "cpu_over_wall": float(np.sum(cpu)) / max(1.0, float(np.sum(samples))),
                }
            )
            continue
        results.append(
            {
                "scenario": scenario,
                "resolution": resolution,
                "backend": backend,
                "frames": len(members),
                "tags": tags,
                "found": found,
                "recall": found / tags if tags else None,
                "false_ids": sum(int(row["false_id"]) for row in members),
                "duplicates": sum(int(row["duplicate"]) for row in members),
                "frames_with_false_id": sum(1 for row in members if int(row["false_id"])),
                "unstable_frames": sum(1 for row in members if str(row["stable_across_repetitions"]) != "True"),
                "latency": pm.summarize_latency(samples),
                "frame_median_ms": pm.summarize_latency([round(float(r["median_ms"]) * 1e6) for r in members]),
                "cpu_ms_mean": float(np.mean(cpu)) / 1e6,
                "cpu_over_wall": float(np.sum(cpu)) / max(1.0, float(np.sum(samples))),
                "corner_error": pm._error_summary(errors),
                "gross_corner_errors": sum(1 for e in errors if e > GROSS_CORNER_PX),
                "corner_order": {
                    "alignments": dict(alignments),
                    "modal": modal[0] if modal else None,
                    "modal_fraction": modal[1] / sum(alignments.values()) if modal else None,
                    "canonical_fraction": alignments.get("s0", 0) / sum(alignments.values()) if alignments else None,
                },
            }
        )
    return {
        "results": results,
        "speedups": pm._speedups(rows, baseline),
        "common_corners": common_corners(rows, baseline),
    }


def common_corners(rows: Sequence[dict[str, object]], baseline: str) -> list[dict[str, object]]:
    """Corner error on the tags both the baseline and each other backend found, per resolution."""

    by_frame: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    resolution_of: dict[str, str] = {}
    for row in rows:
        by_frame[str(row["frame"])][str(row["backend"])] = dict(
            zip(_split(row["tp_ids"], int), _split(row["corner_errors_px"]), strict=True)
        )
        resolution_of[str(row["frame"])] = str(row["resolution"])
    pairs: dict[tuple[str, str], tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for frame, backends in by_frame.items():
        base = backends.get(baseline)
        if base is None:
            continue
        for name, found in backends.items():
            if name == baseline:
                continue
            for tag_id in base.keys() & found.keys():
                for key in ((resolution_of[frame], name), ("all", name)):
                    pairs[key][0].append(base[tag_id])
                    pairs[key][1].append(found[tag_id])
    return [
        {
            "resolution": resolution,
            "versus": name,
            "common_tags": len(ours),
            "baseline": pm._error_summary(ours),
            "other": pm._error_summary(theirs),
        }
        for (resolution, name), (ours, theirs) in sorted(pairs.items())
    ]


# --------------------------------------------------------------------------------------- validation


def validation_frames() -> list[pm.Frame]:
    """Known geometry: one clean 120 px tag, every quarter turn plus odd angles, flat and tilted; blank frames."""

    frames = []
    for index, (roll_deg, tilt_deg, tag_id) in enumerate(
        [
            (0, 0, 0),
            (90, 0, 1),
            (180, 0, 2),
            (270, 0, 3),
            (30, 0, 586),
            (217, 0, 300),
            (0, 25, 42),
            (90, 30, 43),
            (135, 35, 44),
            (300, 20, 45),
        ]
    ):
        image = np.full((480, 640), 200.0, np.float32)
        rng = np.random.default_rng(index)
        corners = _pose(120.0, (320.0 + rng.uniform(-3, 3), 240.0 + rng.uniform(-3, 3)), roll_deg, tilt_deg)
        pm.render_tag(image, tag_id, corners, black=30.0, white=230.0)
        frames.append(
            pm.Frame(
                "known-geometry",
                index,
                (640, 480),
                pm._to_uint8(image),
                [pm.TagTruth(tag_id, corners)],
                {"roll_deg": roll_deg, "tilt_deg": tilt_deg},
            )
        )
    # A whole frame turned by numpy: the tag's corner order must turn with it.
    base = frames[4]
    turned = np.ascontiguousarray(np.rot90(base.image))  # 90 degrees counter-clockwise: (x, y) -> (y, W-1-x)
    corners = np.stack([base.tags[0].corners[:, 1], 639 - base.tags[0].corners[:, 0]], axis=1)
    frames.append(
        pm.Frame("known-geometry", len(frames), (480, 640), turned, [pm.TagTruth(586, corners)], {"rot90": True})
    )
    for index, value in enumerate((0, 128, 255)):
        frames.append(pm.Frame("blank", index, (640, 480), np.full((480, 640), value, np.uint8), [], {"value": value}))
    noise = np.random.default_rng(7).normal(128, 20, (720, 1280))
    frames.append(pm.Frame("blank", 3, (1280, 720), np.clip(noise, 0, 255).astype(np.uint8), [], {"noise": 20}))
    return frames


def _pose(side: float, center: tuple[float, float], roll_deg: float, tilt_deg: float) -> np.ndarray:
    """Like performance_matrix.tag_pose but with fixed angles (tilt about the x axis)."""

    roll, tilt = np.radians(roll_deg), np.radians(tilt_deg)
    square = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    points = square @ rz.T @ rx.T
    projected = points[:, :2] / (2.5 + points[:, 2:3]) * 2.5
    return projected * side + np.asarray(center)


def validate(backends: Sequence[cb.Backend]) -> dict[str, object]:
    frames = validation_frames()
    report = {}
    for backend in backends:
        detect = backend.factory()
        cases = []
        for frame in frames:
            detections = detect(frame.image)
            score = score_detail(frame.tags, detections)
            cases.append(
                {
                    "frame": frame.name,
                    "details": frame.details,
                    "expected": [t.tag_id for t in frame.tags],
                    "detected": [i for i, _ in detections],
                    **{k: v for k, v in score.items() if k != "matched"},
                    "matched": score["matched"],
                }
            )
        tagged = [c for c in cases if c["expected"]]
        matched = [m for c in tagged for m in c["matched"].values()]
        offsets = np.asarray([m["offset"] for m in matched]) if matched else np.zeros((0, 2))
        alignments = Counter(m["alignment"] for m in matched)
        report[backend.name] = {
            "config": backend.config,
            "known_tags_found": f"{sum(c['true_positive'] for c in tagged)}/{len(tagged)}",
            "false_ids": sum(c["false_id"] for c in cases),
            "duplicates": sum(c["duplicate"] for c in cases),
            "blank_detections": sum(len(c["detected"]) for c in cases if not c["expected"]),
            "worst_corner_px": max((m["worst_px"] for m in matched), default=None),
            "mean_offset_px": offsets.mean(axis=0).tolist() if len(offsets) else None,
            "alignments": dict(alignments),
            "order_consistent": len(alignments) == 1,
            "cases": cases,
        }
    return report


# ---------------------------------------------------------------------------------------- metadata


def machine_metadata() -> dict[str, object]:
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    cpus = sorted(os.sched_getaffinity(0))
    return {
        "platform": platform.platform(),
        "cpu_model": pm._cpu_model(),
        "logical_cpus": os.cpu_count(),
        "affinity": cpus,
        "p_cores": read("/sys/devices/cpu_core/cpus"),
        "e_cores": read("/sys/devices/cpu_atom/cpus"),
        "governor": {c: read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor") for c in cpus},
        "epp": {c: read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/energy_performance_preference") for c in cpus},
        "max_freq_khz": {c: read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq") for c in cpus},
        "core_id": {c: read(f"/sys/devices/system/cpu/cpu{c}/topology/core_id") for c in cpus},
        "smt_siblings": {c: read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") for c in cpus},
        "no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "loadavg": os.getloadavg(),
        "python": sys.version,
        "numpy": np.__version__,
        "env": {
            k: os.environ.get(k)
            for k in (
                "RAYON_NUM_THREADS",
                "OMP_NUM_THREADS",
                "RAPIDTAG_CORES",
                "NAPA_CORE_LIBRARY",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "temperatures_c": _temperatures(),
    }


def _temperatures() -> dict[str, float]:
    temps = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            temps[(zone / "type").read_text().strip()] = int((zone / "temp").read_text()) / 1000
        except (OSError, ValueError):
            continue
    return temps


# -------------------------------------------------------------------------------------------- CLI


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(cb.public_text(buffer.getvalue()), encoding="utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cb.public_text(json.dumps(data, indent=2, sort_keys=True, default=str)) + "\n")


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backends", type=pm._name_list, required=True)
    parser.add_argument("--threads", type=int, default=1, help="Threads inside each detector")
    parser.add_argument("--seed", type=int, default=2468)


def _corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenarios", type=pm._name_list, default=pm.SCENARIO_NAMES)
    parser.add_argument("--stress", type=lambda v: () if v == "none" else pm._name_list(v), default=tuple(STRESS))
    parser.add_argument("--resolutions", type=pm._resolution_list, default=pm.DEFAULT_RESOLUTIONS)
    parser.add_argument("--frames", type=pm._nonnegative_int, default=10, help="Frames per corpus scenario")
    parser.add_argument("--stress-frames", type=pm._nonnegative_int, default=10)
    parser.add_argument("--range-cases", type=pm._nonnegative_int, default=0)
    parser.add_argument(
        "--real-images",
        type=Path,
        nargs="*",
        default=(),
        help="Real photos with no ground truth: timed, detections recorded, never scored",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser("validate")
    _common_arguments(validate_parser)
    validate_parser.add_argument("--output", type=Path, required=True)

    latency = commands.add_parser("latency")
    _common_arguments(latency)
    _corpus_arguments(latency)
    latency.add_argument("--repetitions", type=pm._positive_int, default=5)
    latency.add_argument("--warmup", type=pm._nonnegative_int, default=1)
    latency.add_argument("--only-frames", type=pm._name_list, help="Re-time just these frame names")
    latency.add_argument(
        "--order-seed", type=int, help="Seed for the shuffled backend order (default: --seed, the render seed)"
    )
    latency.add_argument("--output", type=Path, required=True)

    report_parser = commands.add_parser("report")
    report_parser.add_argument("latency_json", type=Path)
    report_parser.add_argument("--groups", type=pm._name_list, default=("all-corpus", "all-stress"))
    report_parser.add_argument("--output", type=Path)

    published = commands.add_parser("published")
    published.add_argument("latency_json", type=Path)
    published.add_argument("--output", type=Path)

    agreement_parser = commands.add_parser("real-agreement")
    agreement_parser.add_argument("latency_json", type=Path)
    agreement_parser.add_argument("--output", type=Path, required=True)

    resources_report = commands.add_parser("resources-report")
    resources_report.add_argument("inputs", type=Path, nargs="+")
    resources_report.add_argument("--output", type=Path)

    corpus = commands.add_parser("corpus")
    _corpus_arguments(corpus)
    corpus.add_argument("--seed", type=int, default=2468)
    corpus.add_argument("--output", type=Path, required=True)

    resources = commands.add_parser("resources")
    _common_arguments(resources)
    resources.add_argument("--frames-npz", type=Path, required=True)
    resources.add_argument("--cpus", required=True, help="taskset CPU list for every worker process")
    resources.add_argument("--mode", choices=("serial", "parallel"), default="serial")
    resources.add_argument("--workers", type=int, default=1)
    resources.add_argument("--passes", type=int, default=3)
    resources.add_argument("--rounds", type=int, default=3)
    resources.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "corpus":
        frames = list(
            iter_corpus(
                scenarios=args.scenarios,
                stress=args.stress,
                resolutions=args.resolutions,
                frames=args.frames,
                stress_frames=args.stress_frames,
                range_cases=args.range_cases,
                seed=args.seed,
                real_images=args.real_images,
            )
        )
        arrays = {f"image_{i:05d}": f.image for i, f in enumerate(frames)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, **arrays, names=np.array([f.name for f in frames]))
        print(json.dumps({"frames": len(frames), "output": str(args.output)}))
        return 0

    if args.command == "resources":
        return run_resources(args)

    if args.command == "published":
        with args.latency_json.with_suffix(".csv").open(newline="", encoding="utf-8") as file:
            text = published_tables(list(csv.DictReader(file)))
        if args.output:
            args.output.write_text(text)
        print(text)
        return 0

    if args.command == "real-agreement":
        with args.latency_json.with_suffix(".csv").open(newline="", encoding="utf-8") as file:
            agreement = real_agreement(list(csv.DictReader(file)))
        _write_json(args.output, agreement)
        for name, entry in agreement["backends"].items():
            times = [round(p["median_ms"], 2) for p in entry["pairs"]]
            print(
                f"{name:22s} detections {entry['detections']:3d}  matched {entry['matched']:3d}/"
                f"{entry['reference_tags']}  worst corner vs ref median "
                f"{entry['median_worst_corner_vs_reference_px'] or 0:.2f} px  median ms per image {times}"
            )
        return 0

    if args.command == "resources-report":
        text = ""
        for path in args.inputs:
            data = json.loads(path.read_text())
            config = data["run_config"]
            title = (
                f"{path.stem}: {config['mode']}, {config['workers']} worker(s) x {config['threads']} thread(s), "
                f"CPUs {config['cpus']}, {config['passes']} passes x {config['rounds']} rounds"
            )
            text += resources_markdown(resources_summary(data), title)
        if args.output:
            args.output.write_text(text)
        print(text)
        return 0

    if args.command == "report":
        report = json.loads(args.latency_json.read_text())
        with args.latency_json.with_suffix(".csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        text = markdown_report(report, all_frame_ratios(rows), args.groups)
        if args.output:
            args.output.write_text(text)
        print(text)
        return 0

    cpus = sorted(os.sched_getaffinity(0))
    env = cb.configure_threads(args.threads, cpus)
    import cv2  # the renderer's OpenCV; its pool must not run beside a timed call

    cv2.setNumThreads(args.threads)
    backends, skipped = cb.build_backends(list(args.backends), args.threads)
    if not backends:
        print(f"no backend could be built: {skipped}", file=sys.stderr)
        return 2
    header = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "machine": machine_metadata(),
        "thread_env": env,
        "renderer_opencv": cv2.__version__,
        "backends": {b.name: b.config for b in backends},
        "skipped_backends": skipped,
        "napa_core_loaded": cb.loaded_libraries("napa_core"),
    }

    if args.command == "validate":
        report = validate(backends)
        _write_json(args.output, {**header, "validation": report})
        for name, entry in report.items():
            print(
                f"{name:22s} found {entry['known_tags_found']:>6s}  false {entry['false_ids']}  "
                f"dup {entry['duplicates']}  blank {entry['blank_detections']}  "
                f"worst {entry['worst_corner_px'] or 0:.3f} px  "
                f"offset {np.round(entry['mean_offset_px'] or [0, 0], 3).tolist()}  order {entry['alignments']}"
            )
        return 0

    frames = iter_corpus(
        scenarios=args.scenarios,
        stress=args.stress,
        resolutions=args.resolutions,
        frames=args.frames,
        stress_frames=args.stress_frames,
        range_cases=args.range_cases,
        seed=args.seed,
        real_images=args.real_images,
    )
    total = len(args.resolutions) * (len(args.scenarios) * args.frames + len(args.stress) * args.stress_frames)
    total += args.range_cases * len(pm.RANGE_SIDES_PX) + len(args.real_images)
    if args.only_frames:
        frames = [frame_by_name(name, args.seed, args.range_cases or 30) for name in args.only_frames]
        total = len(frames)

    def progress(done: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"\r{done}/{total} frames  load {os.getloadavg()[0]:.2f}", end="", file=sys.stderr, flush=True)

    started = datetime.now(UTC).isoformat()
    lifecycle, rows, observed = run_interleaved(
        frames,
        backends=backends,
        repetitions=args.repetitions,
        warmup=args.warmup,
        seed=args.seed if args.order_seed is None else args.order_seed,
        progress=progress,
    )
    print(file=sys.stderr)
    summary = aggregate(rows)
    report = {
        **header,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "machine_end": machine_metadata(),
        "run_config": {
            k: ([str(x) for x in v] if isinstance(v, (tuple, list)) else v)
            for k, v in vars(args).items()
            if k != "output"
        },
        "scenario_descriptions": {**{s.name: s.description for s in pm.SCENARIOS}, **STRESS_DESCRIPTIONS},
        "lifecycle": lifecycle,
        "threads_observed": observed,
        "gross_corner_px": GROSS_CORNER_PX,
        **summary,
    }
    _write_json(args.output, report)
    _write_rows(args.output.with_suffix(".csv"), rows)
    print(json.dumps({"report": str(args.output), "rows": str(args.output.with_suffix(".csv"))}))
    return 0


def _fmt(value: float | None, digits: int = 3, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def markdown_report(
    report: dict[str, object], ratios: dict[tuple[str, str, str], dict], scenarios: Sequence[str]
) -> str:
    """One table per (scenario or group, resolution): latency, per-frame ratio to NAPA, and quality."""

    results = {(r["scenario"], r["resolution"], r["backend"]): r for r in report["results"]}
    resolutions = sorted({r["resolution"] for r in report["results"]}, key=lambda s: int(s.split("x")[0]))
    backends = list(report["backends"])
    lines = []
    for scenario in scenarios:
        for resolution in resolutions:
            rows = [results.get((scenario, resolution, b)) for b in backends]
            if not any(rows):
                continue
            lines += [
                f"\n#### {scenario} @ {resolution}\n",
                "| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) "
                "| Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
            for backend, row in zip(backends, rows, strict=True):
                if row is None:
                    continue
                latency, error, order = row["latency"], row["corner_error"], row["corner_order"]
                ratio = ratios.get((scenario, resolution, backend))
                speed = f"{ratio['median']:.2f}× ({ratio['min']:.2f}–{ratio['max']:.2f})" if ratio else "—"
                recall = "—" if row["recall"] is None else f"{row['found']}/{row['tags']} ({100 * row['recall']:.1f}%)"
                lines.append(
                    f"| {backend} | {report['backends'][backend].get('tuning', '')} | {latency['p50_ms']:.3f} "
                    f"| {latency['p95_ms']:.3f} | {latency['p99_ms']:.3f} | {row['cpu_over_wall']:.2f} | {speed} "
                    f"| {recall} | {row['false_ids']} | {row['duplicates']} | {_fmt(error['median_px'])} / "
                    f"{_fmt(error['p95_px'])} / {_fmt(error['max_px'], 2)} | {row['gross_corner_errors']} "
                    f"| {order['modal'] or '—'} {_fmt(order['modal_fraction'], 3)} |"
                )
    return "\n".join(lines) + "\n"


def all_frame_ratios(rows: Sequence[dict[str, object]], baseline: str = "napa") -> dict[tuple[str, str, str], dict]:
    """Per-frame ratios keyed (scenario or group, resolution, backend), as markdown_report expects."""

    def group_of(kind: str) -> Callable[[dict], str | None]:
        def name(row: dict) -> str | None:
            scenario = str(row["scenario"])
            if kind == "scenario":
                return f"{scenario}|{row['resolution']}"
            if kind == "corpus" and scenario in pm.SCENARIO_NAMES:
                return f"all-corpus|{row['resolution']}"
            if kind == "stress" and scenario.startswith("stress"):
                return f"all-stress|{row['resolution']}"
            return None

        return name

    keyed = {}
    for kind in ("scenario", "corpus", "stress"):
        for entry in frame_ratios(rows, baseline, group_of(kind)):
            scenario, resolution = entry["group"].split("|")
            keyed[(scenario, resolution, entry["versus"])] = entry
    return keyed


def frame_ratios(rows: Sequence[dict[str, object]], baseline: str, group: Callable[[dict], str | None]) -> list[dict]:
    """Per-frame ratio (backend median ÷ baseline median on the same frame) over arbitrary frame groups."""

    by_frame: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_frame[str(row["frame"])][str(row["backend"])] = row
    ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    for backends in by_frame.values():
        base = backends.get(baseline)
        if base is None or (name := group(base)) is None:
            continue
        for backend, row in backends.items():
            if backend != baseline:
                ratios[(name, backend)].append(float(row["median_ms"]) / max(float(base["median_ms"]), 1e-9))
    return [
        {
            "group": name,
            "versus": backend,
            "frames": len(values),
            "median": float(np.median(values)),
            "p5": float(np.percentile(values, 5)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "frames_baseline_slower": int(np.sum(np.asarray(values) < 1.0)),
        }
        for (name, backend), values in sorted(ratios.items())
    ]


def _peak(result: dict[str, object]) -> int:
    """Peak RSS in KiB: VmHWM, or a sampled RSS when that reads higher (VmHWM can lag by a few pages)."""

    rss = result["rss_kib"]
    return max(rss.get("vmhwm", rss["peak"]), rss["peak"], rss["after_warmup"], rss["end"])


def resources_summary(data: dict[str, object], baseline: str = "napa") -> list[dict[str, object]]:
    """Per backend over rounds: medians of throughput and CPU, worst peak RSS, and per-frame ratios to NAPA
    taken within the same round (same frames, adjacent in time)."""

    ok = [r for r in data["results"] if "error" not in r and "skip_reason" not in r]
    by_round = defaultdict(dict)
    for result in ok:
        by_round[result["round"]][result["backend"]] = result
    summary = []
    for backend in dict.fromkeys(r["backend"] for r in data["results"]):
        runs = [r for r in ok if r["backend"] == backend]
        failures = [r["error"] for r in data["results"] if r["backend"] == backend and "error" in r]
        skips = [r["skip_reason"] for r in data["results"] if r["backend"] == backend and "skip_reason" in r]
        if not runs:
            summary.append({"backend": backend, "failures": failures, "skip_reason": skips[:1]})
            continue
        ratios = []
        for rounds in by_round.values():
            if backend in rounds and baseline in rounds and backend != baseline:
                ours, theirs = rounds[baseline]["frame_median_ms"], rounds[backend]["frame_median_ms"]
                ratios += [t / max(o, 1e-9) for o, t in zip(ours, theirs, strict=True)]
        median = lambda key, runs=runs: float(np.median([r[key] for r in runs]))  # noqa: E731
        summary.append(
            {
                "backend": backend,
                "tuning": runs[0]["config"].get("tuning"),
                "rounds": len(runs),
                "failures": failures,
                "mode": runs[0]["mode"],
                "workers": runs[0]["workers"],
                "threads_per_detector": runs[0]["threads_per_detector"],
                "cpus": runs[0]["cpus"],
                "wall_ms_per_frame": median("wall_ms_per_frame"),
                "wall_ms_per_frame_range": [
                    min(r["wall_ms_per_frame"] for r in runs),
                    max(r["wall_ms_per_frame"] for r in runs),
                ],
                "cpu_ms_per_frame": median("cpu_ms_per_frame"),
                "frames_per_s": median("frames_per_s"),
                "mean_busy_cpus": median("mean_busy_cpus"),
                "latency_p50_ms": float(np.median([r["latency_ms"]["p50"] for r in runs])),
                "latency_p95_ms": float(np.median([r["latency_ms"]["p95"] for r in runs])),
                "latency_p99_ms": float(np.median([r["latency_ms"]["p99"] for r in runs])),
                "setup_ms": float(np.median([r["setup"]["wall_ms"] for r in runs])),
                "rss_import_construct_mib": max(
                    r["rss_kib"]["after_construct"] - r["rss_kib"]["before_backend"] for r in runs
                )
                / 1024,
                "rss_peak_over_baseline_mib": max(_peak(r) - r["rss_kib"]["before_backend"] for r in runs) / 1024,
                "rss_peak_over_constructed_mib": max(_peak(r) - r["rss_kib"]["after_construct"] for r in runs) / 1024,
                "os_threads": max(len(r["threads_end"]) for r in runs),
                "thread_cpus": sorted({c for r in runs for t in r["threads_end"] for c in t["cpus"]}),
                "per_frame_ratio_to_napa": None
                if not ratios
                else {
                    "median": float(np.median(ratios)),
                    "min": float(np.min(ratios)),
                    "max": float(np.max(ratios)),
                    "frames": len(ratios),
                    "frames_napa_slower": int(np.sum(np.asarray(ratios) < 1.0)),
                },
            }
        )
    return summary


def resources_markdown(summary: Sequence[dict[str, object]], title: str) -> str:
    lines = [
        f"\n#### {title}\n",
        "| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s "
        "| p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB "
        "| Peak RSS over construct MiB | OS threads | Thread CPUs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in summary:
        if "wall_ms_per_frame" not in s:
            status = f"skipped: {s['skip_reason'][0]}" if s.get("skip_reason") else f"failed: {s['failures'][:1]}"
            lines.append(f"| {s['backend']} | {status} |" + " |" * 10)
            continue
        ratio = s["per_frame_ratio_to_napa"]
        ratio = f"{ratio['median']:.2f}× ({ratio['min']:.2f}–{ratio['max']:.2f})" if ratio else "—"
        low, high = s["wall_ms_per_frame_range"]
        lines.append(
            f"| {s['backend']} | {s['tuning']} | {s['wall_ms_per_frame']:.3f} ({low:.3f}–{high:.3f}) "
            f"| {s['cpu_ms_per_frame']:.3f} | {s['mean_busy_cpus']:.2f} | {s['frames_per_s']:.0f} "
            f"| {s['latency_p50_ms']:.3f} / {s['latency_p95_ms']:.3f} / {s['latency_p99_ms']:.3f} | {ratio} "
            f"| {s['rss_import_construct_mib']:.1f} | {s['rss_peak_over_constructed_mib']:.1f} | {s['os_threads']} "
            f"| {','.join(map(str, s['thread_cpus']))} |"
        )
    return "\n".join(lines) + "\n"


PUBLISHED_LABELS = {
    "wpilib-d2": "WPILib AprilTag, decimate 2",
    "apriltag3-d2": "AprilTag, decimate 2",
    "apriltag3-d1": "AprilTag, decimate 1",
    "locus": "Locus",
    "opencv": "OpenCV ArUco",
    "aruco-nano": "ArUco Nano",
    "rapidtag": "RapidTag",
    "napa": "NAPA",
}


def _times(value: float) -> str:
    return f"{value:.0f}×" if value >= 10 else f"{value:.1f}×"


def published_tables(rows: Sequence[dict[str, object]], baseline: str = "napa") -> str:
    """The README and benchmark-guide tables, from one ``latency`` run's rows (corpus, range sweep)."""

    present = {str(r["backend"]) for r in rows}
    backends = [b for b in PUBLISHED_LABELS if b in present]
    others = [b for b in backends if b != baseline]
    corpus = [r for r in rows if r["scenario"] in pm.SCENARIO_NAMES]
    ranged = [r for r in rows if r["scenario"] == "range"]
    resolutions = sorted({str(r["resolution"]) for r in corpus}, key=lambda v: int(v.split("x")[0]))
    by = defaultdict(list)
    for row in corpus:
        by[str(row["backend"])].append(row)
    frames = len(by[baseline])
    if not frames:
        raise ValueError("published tables need the corpus scenarios in the run")
    lines = [
        "#### Summary\n",
        "| | " + " | ".join(PUBLISHED_LABELS[b] for b in backends) + " |",
        "| --- |" + " ---: |" * len(backends),
    ]

    def found(members):
        return sum(int(r["true_positive"]) for r in members), sum(int(r["tags"]) for r in members)

    smallest = {}
    if ranged:
        counts = defaultdict(lambda: [0, 0])
        for row in ranged:
            key = (str(row["backend"]), float(row["detail_side_px"]))
            counts[key][0] += int(row["true_positive"])
            counts[key][1] += 1
        sides = sorted({side for _, side in counts})
        for backend in backends:
            misses = {s: counts[(backend, s)][1] - counts[(backend, s)][0] for s in sides}
            exact = next((s for s in sides if not any(misses[t] for t in sides if t >= s)), None)
            if exact is not None:
                smallest[backend] = f"{exact:g} px"
                continue
            # Not one size from which every tag was found: say so, unless a single missed tag is what prevents it.
            near = next((s for s in sides if sum(misses[t] for t in sides if t >= s) == 1), None)
            missed = next((t for t in sides if near is not None and t >= near and misses[t]), None)
            smallest[backend] = f"{near:g} px (one {missed:g} px tag missed)" if near is not None else "—"
    cells = {
        "Tags found": [f"{100 * found(by[b])[0] / found(by[b])[1]:.1f}%" for b in backends],
        "Smallest tag found every time (range test)": [smallest[b] for b in backends] if ranged else None,
        "Median error of a tag's worst corner": [
            f"{np.median([e for r in by[b] for e in _split(r['corner_errors_px'])]):.3f} px" for b in backends
        ],
        f"Wrong IDs in {frames:,} frames": [
            str(sum(int(r["false_id"]) + int(r["duplicate"]) for r in by[b])) for b in backends
        ],
    }
    lines += [f"| {name} | " + " | ".join(values) + " |" for name, values in cells.items() if values]

    def ratio_groups(group: Callable[[dict], str | None]) -> dict[tuple[str, str], dict]:
        return {(e["group"], e["versus"]): e for e in frame_ratios(corpus, baseline, group)}

    per_resolution = ratio_groups(lambda r: str(r["resolution"]))
    overall = ratio_groups(lambda r: "all")
    lines += [
        "\n#### NAPA's speedup per frame: median (slowest frame)\n",
        "| Resolution | " + " | ".join(PUBLISHED_LABELS[b] for b in others) + " |",
        "| --- |" + " ---: |" * len(others),
    ]
    for name, table in [(res, per_resolution) for res in resolutions] + [("all", overall)]:
        label = "All frames" if name == "all" else name.replace("x", "×")
        values = [table[(name, b)] for b in others]
        lines.append(f"| {label} | " + " | ".join(f"{_times(v['median'])} ({_times(v['min'])})" for v in values) + " |")
    slower = {b: overall[("all", b)]["frames_baseline_slower"] for b in others}
    lines.append(f"\nFrames where NAPA was slower: {slower}\n")

    results = defaultdict(list)
    for row in corpus:
        results[(str(row["scenario"]), str(row["resolution"]), str(row["backend"]))].append(row)
    scene_ratios = ratio_groups(lambda r: f"{r['scenario']}|{r['resolution']}")
    for resolution in resolutions:
        lines += [
            f"\n#### {resolution}: median call (NAPA's per-frame speedup, median)\n",
            "| Scene | Frames | Tags | " + " | ".join(PUBLISHED_LABELS[b] for b in backends) + " |",
            "| --- | ---: | ---: |" + " ---: |" * len(backends),
        ]
        for scenario in pm.SCENARIO_NAMES:
            members = results.get((scenario, resolution, baseline))
            if not members:
                continue
            values = []
            for b in backends:
                samples = [v for r in results[(scenario, resolution, b)] for v in _split(r["samples_ns"], int)]
                p50 = float(np.median(samples)) / 1e6
                ratio = scene_ratios.get((f"{scenario}|{resolution}", b))
                values.append(f"{p50:.2f} ms" + (f" ({_times(ratio['median'])})" if ratio else ""))
            tags = sum(int(r["tags"]) for r in members)
            lines.append(f"| {scenario} | {len(members)} | {tags} | " + " | ".join(values) + " |")
        lines += [
            f"\n#### {resolution}: tags found (wrong IDs)\n",
            "| Scene | Tags | " + " | ".join(PUBLISHED_LABELS[b] for b in backends) + " |",
            "| --- | ---: |" + " ---: |" * len(backends),
        ]
        for scenario in pm.SCENARIO_NAMES:
            members = results.get((scenario, resolution, baseline))
            if not members:
                continue
            tags = sum(int(r["tags"]) for r in members)
            values = []
            for b in backends:
                got, _ = found(results[(scenario, resolution, b)])
                wrong = sum(int(r["false_id"]) + int(r["duplicate"]) for r in results[(scenario, resolution, b)])
                text = "—" if not tags else str(got)
                values.append(text + (f" ({wrong})" if wrong else ""))
            lines.append(f"| {scenario} | {tags} | " + " | ".join(values) + " |")

    if ranged:
        sides = sorted({side for _, side in counts})
        groups: list[list[float]] = []
        for side in sides:
            same = groups and all(counts[(b, groups[-1][-1])][0] == counts[(b, side)][0] for b in backends)
            groups[-1].append(side) if same else groups.append([side])
        cases = counts[(baseline, sides[0])][1]
        header = [f"{g[0]:g}" if len(g) == 1 else f"{g[0]:g}–{g[-1]:g}" for g in groups]
        lines += [
            f"\n#### Range: tags found out of {cases} at each black-square size (px)\n",
            "| Detector | " + " | ".join(header) + " |",
            "| --- |" + " ---: |" * len(groups),
        ]
        for b in backends:
            lines.append(f"| {PUBLISHED_LABELS[b]} | " + " | ".join(str(counts[(b, g[0])][0]) for g in groups) + " |")
    return "\n".join(lines) + "\n"


REAL_MATCH_CENTRE_PX = 3.0


def apriltag_reference(path: Path) -> list[tuple[int, np.ndarray]]:
    """Upstream AprilTag's regression output beside a test photo: ``id, (x y), ...`` per line, +0.5 centres."""

    import re

    tags = []
    for line in path.read_text().splitlines():
        if line.strip():
            points = re.findall(r"\(([-\d.]+) ([-\d.]+)\)", line)
            tags.append((int(line.split(",")[0]), np.array(points, np.float64) - 0.5))
    return tags


def real_agreement(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Agreement of each backend's detections with the reference output, not accuracy.

    Rule: per image, pairs with the same ID whose centres are within REAL_MATCH_CENTRE_PX, matched one-to-one
    greedily by centre distance; corner distance is the worst corner after the best corner-order alignment.
    """

    summary: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["scenario"] != REAL:
            continue
        path = Path(str(row["detail_path"]))
        reference = apriltag_reference(path.with_suffix(".txt"))
        detections = [(int(i), np.asarray(c, np.float64)) for i, c in json.loads(str(row["detections_json"]))]
        pairs = sorted(
            (float(np.linalg.norm(d.mean(axis=0) - r.mean(axis=0))), di, ri)
            for di, (did, d) in enumerate(detections)
            for ri, (rid, r) in enumerate(reference)
            if did == rid
        )
        used_d, used_r, matches = set(), set(), []
        for distance, di, ri in pairs:
            if distance > REAL_MATCH_CENTRE_PX or di in used_d or ri in used_r:
                continue
            used_d.add(di), used_r.add(ri)
            corners = detections[di][1]
            worst = min(
                float(np.linalg.norm(_aligned(corners, *a) - reference[ri][1], axis=1).max()) for a in ALIGNMENTS
            )
            matches.append({"detection": di, "reference": ri, "centre_px": distance, "worst_corner_px": worst})
        entry = summary.setdefault(
            str(row["backend"]), {"images": 0, "detections": 0, "reference_tags": 0, "matched": 0, "pairs": []}
        )
        entry["images"] += 1
        entry["detections"] += len(detections)
        entry["reference_tags"] += len(reference)
        entry["matched"] += len(matches)
        entry["pairs"].append({"image": path.name, "matches": matches, "median_ms": float(row["median_ms"])})
    for entry in summary.values():
        worst = [m["worst_corner_px"] for image in entry["pairs"] for m in image["matches"]]
        entry["median_worst_corner_vs_reference_px"] = float(np.median(worst)) if worst else None
    return {"rule": real_agreement.__doc__.strip(), "centre_px": REAL_MATCH_CENTRE_PX, "backends": summary}


def run_resources(args: argparse.Namespace) -> int:
    """One process per backend per round, never overlapping, in a shuffled order each round."""

    shuffler = random.Random(args.seed)
    results = []
    worker = ROOT / "benchmarks" / "cpu_backends.py"
    env = {k: v for k, v in os.environ.items() if k not in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "RAPIDTAG_CORES")}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
    # NumPy starts OpenBLAS's pool at import, before the worker can configure anything: limit it here.
    env["OPENBLAS_NUM_THREADS"] = "1"
    for round_index in range(args.rounds):
        order = list(args.backends)
        shuffler.shuffle(order)
        for name in order:
            out = args.output.parent / "workers" / f"{args.output.stem}-r{round_index}-{name}.json"
            command = [
                "taskset",
                "-c",
                args.cpus,
                sys.executable,
                str(worker),
                "--backend",
                name,
                "--frames",
                str(args.frames_npz),
                "--threads",
                str(args.threads),
                "--mode",
                args.mode,
                "--workers",
                str(args.workers),
                "--passes",
                str(args.passes),
                "--output",
                str(out),
            ]
            completed = subprocess.run(command, env=env, capture_output=True, text=True)
            if completed.returncode == cb.SKIPPED_EXIT:
                reason = completed.stderr.strip().splitlines()[-1:] or ["unavailable"]
                results.append({"backend": name, "round": round_index, "skip_reason": reason[0]})
                print(f"{name}: skipped ({reason[0]})", file=sys.stderr)
                continue
            if completed.returncode != 0:
                results.append({"backend": name, "round": round_index, "error": completed.stderr[-2000:]})
                print(f"{name}: failed\n{completed.stderr[-500:]}", file=sys.stderr)
                continue
            result = json.loads(out.read_text())
            result["round"] = round_index
            result["command"] = command
            results.append(result)
            print(
                f"r{round_index} {name:20s} {result['wall_ms_per_frame']:8.3f} ms/frame wall  "
                f"{result['cpu_ms_per_frame']:8.3f} ms/frame CPU  {result['frames_per_s']:8.1f} fps  "
                f"peak +{result['rss_kib']['peak_minus_before'] / 1024:6.1f} MiB  "
                f"threads {len(result['threads_end'])}",
                file=sys.stderr,
            )
    _write_json(
        args.output,
        {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "machine": machine_metadata(),
            "run_config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "results": results,
        },
    )
    data = json.loads(args.output.read_text())
    data["summary"] = resources_summary(data)
    _write_json(args.output, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
