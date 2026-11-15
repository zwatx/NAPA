"""The two benchmark sets of bdcudatag (github.com/cobouch/bd-cudatag), scored by that project's own rules.

The sets are regenerated from its repository, as described in ``benchmarks/README.md``. A tag is found when a detection
has its ID and a centre within 0.75 tag sides. Corner error is the mean corner distance of a matched tag under the best
corner order, reported as a median over tags. A wrong ID is a detection whose ID isn't in the frame.

Run ``python -m benchmarks.bdcudatag --help`` for options.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import cpu_backends as cb  # noqa: E402
from napa_vision import NapaDetector  # noqa: E402

COMMIT = "4ff7a31a5a7ced787825a5e226580b3cbaef07f6"
SCENE_RESOLUTIONS = ("640x480", "1280x720", "1280x800", "1920x1080")
SCENES_PER_RESOLUTION = 32
SCENE_TAGS = SCENES_PER_RESOLUTION * 8
HARD_FRAMES = 780
HARD_TAGS = 2389
WARM_UP = 20
TIMED = 200
ORDERS = [list(order) for order in itertools.permutations(range(4))]
NAPA_SETTINGS = {
    "napa": {},
    "napa16": {"min_tag_px": 16},
    "napa-recall": {"min_tag_px": 16, "split_touching": True, "recover_shadows": True},
}

Tag = tuple[int, np.ndarray]


def score(truth: list[Tag], found: list[Tag]) -> tuple[int, list[float]]:
    """Return the number of wrong IDs and the corner error of each truth tag that was found."""

    known = {tag for tag, _ in truth}
    wrong = sum(tag not in known for tag, _ in found)
    errors = []
    for tag, corners in truth:
        side = 0.5 * (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[1]))
        near = [q for i, q in found if i == tag and np.linalg.norm(q.mean(0) - corners.mean(0)) <= 0.75 * side]
        if near:
            errors.append(
                min(float(np.linalg.norm(q[order] - corners, axis=1).mean()) for q in near for order in ORDERS)
            )
    return wrong, errors


def _gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(image)


def _summary(name: str, tags: int, wrong: int, errors: list[float], times: list[float]) -> dict:
    return {
        "group": name,
        "tags": tags,
        "found": len(errors),
        "wrong_ids": wrong,
        "corner_px_median": float(np.median(errors)) if errors else None,
        "ms_median": float(np.median(times)),
        "ms_p95": float(np.percentile(times, 95)),
    }


def evaluate_scenes(root: Path, detect) -> list[dict]:
    """Timed like the source project: warm-up frames, then 200 frames cycling through the scenes of one resolution."""

    rows = []
    for resolution in SCENE_RESOLUTIONS:
        entries = json.loads((root / f"scenes_{resolution}.json").read_text())["scenes"]
        tags = sum(len(entry["tags"]) for entry in entries)
        if len(entries) != SCENES_PER_RESOLUTION or tags != SCENE_TAGS:
            raise ValueError(f"{resolution}: expected {SCENES_PER_RESOLUTION} scenes and {SCENE_TAGS} tags")
        images = [_gray(root / entry["file"]) for entry in entries]
        for index in range(WARM_UP):
            detect(images[index % len(images)])
        times = []
        for index in range(TIMED):
            start = time.perf_counter()
            detect(images[index % len(images)])
            times.append(1e3 * (time.perf_counter() - start))
        wrong, errors = 0, []
        for entry, image in zip(entries, images, strict=True):
            truth = [(tag["id"], np.array(tag["corners"], float)) for tag in entry["tags"]]
            frame_wrong, frame_errors = score(truth, detect(image))
            wrong, errors = wrong + frame_wrong, errors + frame_errors
        rows.append(_summary(resolution, tags, wrong, errors, times))
    return rows


def evaluate_hard(root: Path, detect) -> list[dict]:
    """Per scenario and resolution, as the source project reports it."""

    records = [json.loads(line) for line in (root / "metadata.jsonl").read_text().splitlines()]
    tags = sum(record["num_tags"] for record in records)
    if len(records) != HARD_FRAMES or tags != HARD_TAGS:
        raise ValueError(f"expected {HARD_FRAMES} frames and {HARD_TAGS} tags, found {len(records)} and {tags}")
    groups: dict[str, dict] = {}
    for index, record in enumerate(records):
        image = _gray(root / record["file_name"])
        if index < WARM_UP:
            detect(image)
        start = time.perf_counter()
        found = detect(image)
        elapsed = 1e3 * (time.perf_counter() - start)
        truth = [
            (tag, np.array(corners, float)) for tag, corners in zip(record["tag_ids"], record["corners"], strict=True)
        ]
        wrong, errors = score(truth, found)
        name = f"{record['scenario']} {record['width']}x{record['height']}"
        group = groups.setdefault(name, {"tags": 0, "wrong": 0, "errors": [], "times": []})
        group["tags"] += record["num_tags"]
        group["wrong"] += wrong
        group["errors"] += errors
        group["times"].append(elapsed)
    return [_summary(name, g["tags"], g["wrong"], g["errors"], g["times"]) for name, g in groups.items()]


def detector(name: str):
    if name in NAPA_SETTINGS:
        napa = NapaDetector(family="tag36h11", max_detections=1024, **NAPA_SETTINGS[name])
        return lambda image: [(o.tag_id, np.asarray(o.corners, float)) for o in napa.detect(image)]
    backends, skipped = cb.build_backends([name], 1)
    if not backends:
        raise SystemExit(f"{name} is unavailable: {skipped}")
    call = backends[0].factory()
    return lambda image: [(int(tag), np.asarray(q, float).reshape(4, 2)) for tag, q in call(image)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("work/external/bdcudatag"))
    parser.add_argument("--sets", nargs="+", choices=("scenes", "hard"), default=["scenes", "hard"])
    parser.add_argument("--backends", nargs="+", default=["napa", "napa16", "apriltag3-d1"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; results are not overwritten")
    results = []
    for name in args.backends:
        detect = detector(name)
        for dataset in args.sets:
            rows = (
                evaluate_scenes(args.root / "scenes", detect)
                if dataset == "scenes"
                else evaluate_hard(args.root / "synthetic-hard" / "data" / "hard", detect)
            )
            results.append({"backend": name, "set": dataset, "rows": rows})
            tags, found = sum(r["tags"] for r in rows), sum(r["found"] for r in rows)
            print(f"{dataset} {name}: {found} of {tags}, wrong IDs {sum(r['wrong_ids'] for r in rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source_commit": COMMIT, "results": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
