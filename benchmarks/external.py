"""Pinned independent Locus renders: correct-ID spatial recall, precision and corner error.

Optional benchmark-only dependency: pyarrow. No input images are generated or modified.
Run ``python -m benchmarks.external --help`` for download and evaluation options.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import cpu_backends as cb  # noqa: E402
from napa_vision.napa_detector import NapaDetector, _library  # noqa: E402

REPO = "NoeFontana/locus-tag-bench"
REVISION = "5346835a9ed350828472a4c83d6e3999cf94fefb"
EVALUATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CONFIGS = (
    *(f"locus_v1_tag36h11_{size}" for size in ("640x480", "1280x720", "1920x1080", "3840x2160")),
    "locus_v1_tag16h5_1920x1080",
    *(f"locus_v1_{kind}_1920x1080" for kind in ("high_iso", "low_key", "raw_pipeline")),
    *(
        f"aprilgrid_{kind}_1920x1080"
        for kind in ("golden_v1", "distortion_brown_conrady_v1", "distortion_kannala_brandt_v1")
    ),
)
FILES = ("train-00000-of-00001.parquet", "rich_truth.json", "manifest.json", "checksums.json")


def files_for(config: str) -> tuple[str, ...]:
    if config.endswith("3840x2160"):
        return ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet", *FILES[1:])
    return FILES


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(root: Path, configs: list[str]) -> None:
    """Fetch immutable data, atomically; never execute dataset code."""
    for config in configs:
        for name in files_for(config):
            path = root / config / name
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{config}/{name}"
            for attempt in range(4):
                try:
                    with (
                        urllib.request.urlopen(url, timeout=60) as response,
                        path.with_suffix(".tmp").open("wb") as out,
                    ):
                        while block := response.read(1024 * 1024):
                            out.write(block)
                    path.with_suffix(".tmp").replace(path)
                    break
                except OSError:
                    if attempt == 3:
                        raise
            print(f"downloaded {config}/{name}", flush=True)


def corner_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geometry only: cyclic/reflected alignment, never ID-only localization credit."""
    options = [
        np.linalg.norm(np.roll(a[::-1] if rev else a, k, axis=0) - b, axis=1) for rev in (False, True) for k in range(4)
    ]
    return min(options, key=lambda d: float(np.sum(d * d)))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    a, b = (cv2.convexHull(np.asarray(q, np.float32)) for q in (a, b))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
    return float(intersection / union) if union > 0 else 0.0


def score(truth: list[dict], detections: list[cb.Detection]) -> dict:
    """One-to-one ID + IoU>=0.5 matching; partial truths can absorb detections only.

    Complete truths have priority. Duplicate detections and correct IDs at the wrong
    location are false positives. Incomplete truths never inflate recall or precision.
    """
    truth = sorted(truth, key=lambda t: not t["eval_complete"])
    remaining = set(range(len(detections)))
    matched, errors, ignored, localized = [], [], 0, 0
    for tag in truth:
        candidates = [
            (overlap(np.asarray(c), np.asarray(tag["corners"])), j)
            for j, (i, c) in enumerate(detections)
            if j in remaining and i == tag["tag_id"]
        ]
        best_iou, j = max(candidates, default=(0, -1))
        if best_iou < 0.5:
            continue
        remaining.remove(j)
        if not tag["eval_complete"]:
            ignored += 1
            continue
        d = corner_distances(np.asarray(detections[j][1]), np.asarray(tag["corners"]))
        matched.append(tag["tag_id"])
        errors.extend(d.tolist())
        localized += int(d.max() <= 1.0)
    expected = sum(bool(t["eval_complete"]) for t in truth)
    return {
        "expected": expected,
        "tp": len(matched),
        "fn": expected - len(matched),
        "fp": len(remaining),
        "ignored": ignored,
        "matched_ids": matched,
        "localized_1px": localized,
        "corner_errors_px": errors,
    }


def icra_frames(root: Path, config: str):
    """ICRA 2020 CSV corners; both published image variants, no detector labels."""
    directory = root / config.removeprefix("icra-")
    truths = defaultdict(dict)
    with (directory / "tags.csv").open() as stream:
        for row in csv.DictReader(stream):
            tag_id, corner = int(row["tag_id"]), int(row["corner"])
            tag = truths[row["image"]].setdefault(
                tag_id, {"tag_id": tag_id, "corners": [None] * 4, "eval_complete": row["tag_fully_visible"] == "1"}
            )
            if not 0 <= corner < 4 or tag["corners"][corner] is not None:
                raise ValueError("invalid or duplicate ICRA corner")
            tag["corners"][corner] = [float(row["ground_truth_x"]) - 0.5, float(row["ground_truth_y"]) - 0.5]
    for tags in truths.values():
        for tag in tags.values():
            if any(c is None for c in tag["corners"]) or not np.isfinite(tag["corners"]).all():
                raise ValueError("missing or invalid ICRA corners")
    for kind in ("pure_tags_images", "checkerboard_corners_images"):
        paths = sorted((directory / kind).glob("*.png"))
        if {p.name for p in paths} != set(truths):
            raise ValueError(f"incomplete ICRA image set: {kind}")
        for path in paths:
            if path.name not in truths:
                raise ValueError(f"missing ICRA truth: {path.name}")
            data = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"unreadable ICRA image: {path}")
            yield f"{kind}/{path.name}", digest(data), image, list(truths[path.name].values())


def frames(root: Path, config: str):
    import pyarrow.parquet as pq

    directory = root / config
    rich = json.loads((directory / "rich_truth.json").read_text())
    checksums = json.loads((directory / "checksums.json").read_text())["files"]
    if checksums["rich_truth.json"] != digest((directory / "rich_truth.json").read_bytes()):
        raise ValueError("truth checksum mismatch")
    truths = defaultdict(list)
    for tag in rich["records"]:
        if tag["record_type"] == "TAG":
            corners = np.asarray(tag["corners"])
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                raise ValueError("invalid external truth corners")
            if tag["eval_complete"] != all(v == 2 for v in tag["corners_visibility"]):
                raise ValueError("inconsistent visibility")
            tag["corners"] = (corners - 0.5).tolist()
            truths[tag["image_id"]].append(tag)
    expected_rows = 150 if config.startswith("aprilgrid_golden") else 100 if "tag16h5" in config else 50
    parquet_files = [directory / f for f in files_for(config) if f.endswith(".parquet")]
    if sum(pq.ParquetFile(p).metadata.num_rows for p in parquet_files) != expected_rows:
        raise ValueError("incomplete Locus image set")
    seen = set()
    for batch in (
        batch
        for file in files_for(config)
        if file.endswith(".parquet")
        for batch in pq.ParquetFile(directory / file).iter_batches(batch_size=1)
    ):
        row = batch.to_pylist()[0]
        name = row["image_id"]
        if name in seen:
            raise ValueError(f"duplicate image {name}")
        seen.add(name)
        image_bytes = row["image"]["bytes"]
        image_hash = digest(image_bytes)
        if checksums[f"images/{name}.png"] != image_hash:
            raise ValueError(f"image checksum mismatch: {config}/{name}")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"unreadable image {name}")
        yield name, image_hash, image, truths[name]
    if set(truths) - seen:
        raise ValueError("truths reference missing images")


def summarize(rows: list[dict]) -> dict:
    counts = {k: sum(row[k] for row in rows) for k in ("expected", "tp", "fn", "fp", "ignored", "localized_1px")}
    tp, fp, expected = (counts[k] for k in ("tp", "fp", "expected"))
    errors = [e for row in rows for e in row["corner_errors_px"]]
    times = [t for row in rows for t in row["times_ms"]]
    return {
        "frames": len(rows),
        **counts,
        "recall": tp / expected if expected else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "corner_accuracy_at_1px": counts["localized_1px"] / expected if expected else None,
        "corner_px_p50_p95": np.percentile(errors, [50, 95]).tolist() if errors else None,
        "latency_ms_p50_p95": np.percentile(times, [50, 95]).tolist(),
    }


def evaluate(root: Path, configs: list[str], names: list[str], repeats: int) -> dict:
    rows, settings, provenance = [], {}, {}
    rng = random.Random(2468)
    for config in configs:
        family = "tag16h5" if "tag16h5" in config else "tag36h11"
        detectors = {}
        owned = []
        for name in names:
            if name in ("napa", "napa16"):
                det = NapaDetector(family=family, min_tag_px=16 if name == "napa16" else 32, max_detections=1024)
                owned.append(det)
                detectors[name] = lambda image, det=det: [(o.tag_id, o.corners) for o in det.detect(image)]
                settings[f"{config}/{name}"] = {
                    "family": family,
                    "min_tag_px": det._params.min_tag_px,
                    "max_detections": 1024,
                }
            elif family == "tag36h11":
                backends, skipped = cb.build_backends([name], 1)
                if skipped:
                    raise RuntimeError(str(skipped))
                backend = backends[0]
                detectors[name] = backend.factory()
                settings[f"{config}/{name}"] = backend.config
        if config.startswith("icra-"):
            provenance[config] = {"tags.csv": digest((root / config[5:] / "tags.csv").read_bytes())}
            loader = icra_frames
        else:
            provenance[config] = {f: digest((root / config / f).read_bytes()) for f in files_for(config)}
            loader = frames
        try:
            for image_id, image_hash, image, truth in loader(root, config):
                for det in detectors.values():
                    det(image)
                samples = {name: [] for name in detectors}
                outputs = {}
                for _ in range(repeats):
                    order = list(detectors)
                    rng.shuffle(order)
                    for name in order:
                        start = time.perf_counter_ns()
                        found = detectors[name](image)
                        samples[name].append((time.perf_counter_ns() - start) / 1e6)
                        serialized = [(i, c.tolist()) for i, c in found]
                        if name in outputs and outputs[name] != serialized:
                            raise RuntimeError("nondeterministic detector output")
                        outputs[name] = serialized
                for name in detectors:
                    found = [(i, np.asarray(c)) for i, c in outputs[name]]
                    rows.append(
                        {
                            "config": config,
                            "image_id": image_id,
                            "image_sha256": image_hash,
                            "backend": name,
                            **score(truth, found),
                            "detections": outputs[name],
                            "times_ms": samples[name],
                        }
                    )
        finally:
            for det in owned:
                det.close()
        print(f"evaluated {config}", flush=True)
    grouped = defaultdict(list)
    for row in rows:
        subset = row["image_id"].split("/")[0] + "/" if row["config"].startswith("icra-") else ""
        grouped[f"{row['config']}/{subset}{row['backend']}"].append(row)
    libraries = [str(Path(_library()._name).resolve())]
    return {
        "dataset": "ICRA2020" if configs[0].startswith("icra-") else REPO,
        "revision": "6c7653385d480351fe8235d010dedc18fbc0b992" if configs[0].startswith("icra-") else REVISION,
        "protocol": "correct ID and IoU >= 0.5; complete TAG truths",
        "pixel_convention": "integer pixel centres; dataset truth shifted by -0.5 on both axes",
        "evaluator_sha256": EVALUATOR_SHA256,
        "adapter_sha256": cb.sha256(Path(cb.__file__)),
        "repeats": repeats,
        "warmup": 1,
        "settings": settings,
        "input_hashes": provenance,
        "core_libraries": {cb.public_text(p): cb.sha256(Path(p)) for p in libraries},
        "working_tree_core_source_sha256": cb.sha256(ROOT / "src/napa_vision/_native/napa_core.c"),
        "summary": {k: summarize(v) for k, v in grouped.items()},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("work/external/locus"))
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=(*CONFIGS, "icra-forward", "icra-circle", "icra-rotation", "icra-random"),
        default=list(CONFIGS),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backends", nargs="+", default=["napa", "napa16", "apriltag3-d1"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if any(c.startswith("icra-") for c in args.configs) and not all(c.startswith("icra-") for c in args.configs):
        parser.error("run ICRA and Locus datasets separately")
    if args.download and args.configs[0].startswith("icra-"):
        parser.error("ICRA archives must be downloaded and extracted separately; see benchmarks/README.md")
    if args.download:
        download(args.root, args.configs)
    if args.output:
        if args.output.exists():
            parser.error("refusing to overwrite results")
        result = evaluate(args.root, args.configs, args.backends, args.repeats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
