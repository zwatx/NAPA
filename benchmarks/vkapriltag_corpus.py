"""vkapriltag's benchmark corpus (github.com/yojobama/vkapriltag), scored by that project's own rules.

Its ``apriltags_vulkan/tools/napa_corpus/gen_corpus.py`` renders the frames with exact corner truth; see
``benchmarks/README.md``.
A detection matches a tag with the same ID when its worst corner is under a quarter of the tag's side. False detections
are wrong IDs and detections with no tag. Corner error is each found tag's worst corner.

Run ``python -m benchmarks.vkapriltag_corpus --help`` for options.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks import cpu_backends as cb  # noqa: E402
from napa_vision.napa_detector import NapaDetector, _library  # noqa: E402

COMMIT = "956d502943306b1d408ab90104ba0a64f23dfe19"
SCENARIOS = 26
MAIN_TAGS = 9991
FRAMES = 4313
SHORT_FAMILY_SET = "family16h5"
NAPA_SETTINGS = {
    "napa": {},
    "napa16": {"min_tag_px": 16},
    "napa-recall": {"min_tag_px": 16, "split_touching": True, "recover_shadows": True},
}
VULKAN = {"vkapriltag-d1": 1, "vkapriltag-d2": 2}

Found = list[tuple[int, np.ndarray]]


def worst_corner(truth: np.ndarray, corners: np.ndarray) -> float:
    return min(float(np.linalg.norm(np.roll(corners, shift, axis=0) - truth, axis=1).max()) for shift in range(4))


def score_frame(truth: list[dict], found: Found) -> tuple[list[float], int]:
    """Return the worst-corner error of each tag found and the number of false detections."""

    remaining = list(found)
    errors = []
    for tag in truth:
        corners = np.array(tag["corners"], float)
        candidates = [(worst_corner(corners, q), k) for k, (i, q) in enumerate(remaining) if i == tag["id"]]
        candidates = [c for c in candidates if c[0] < 0.25 * tag["size_px"]]
        if candidates:
            error, index = min(candidates)
            errors.append(error)
            remaining.pop(index)
    return errors, len(remaining)


def signed_area(corners: np.ndarray) -> float:
    return 0.5 * float(np.sum(corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1]))


def to_corpus(corners: np.ndarray, sign: float) -> np.ndarray:
    """From integer-centred pixels to the corpus's [i, i + 1) pixels, in the truth's winding."""

    converted = np.asarray(corners, float).reshape(4, 2) + 0.5
    return converted if np.sign(signed_area(converted)) == sign else converted[::-1]


def detector(name: str, family: str) -> Callable[[np.ndarray], Found] | None:
    if name in NAPA_SETTINGS:
        napa = NapaDetector(family=family, max_detections=1024, **NAPA_SETTINGS[name])
        return lambda image: [(o.tag_id, o.corners) for o in napa.detect(image)]
    if name in VULKAN:
        call = cb.vkapriltag_backend(name, VULKAN[name], family).factory()
    elif family != "tag36h11":
        return None  # the other adapters are tag36h11 only
    else:
        backends, skipped = cb.build_backends([name], 1)
        if not backends:
            raise SystemExit(f"{name} is unavailable: {skipped}")
        call = backends[0].factory()
    return lambda image: [(int(i), np.asarray(q, float)) for i, q in call(image)]


def evaluate(root: Path, name: str) -> dict[str, dict]:
    scenarios = sorted(path.name for path in root.iterdir() if (path / "gt.json").exists())
    if len(scenarios) != SCENARIOS:
        raise ValueError(f"expected {SCENARIOS} scenarios, found {len(scenarios)}")
    out, frames, main_tags = {}, 0, 0
    detectors: dict[str, Callable[[np.ndarray], Found] | None] = {}  # one per family: vkapriltag's is a process
    for scenario in scenarios:
        family = "tag16h5" if scenario == SHORT_FAMILY_SET else "tag36h11"
        truth = json.loads((root / scenario / "gt.json").read_text())
        frames += len(truth)
        if family == "tag36h11":
            main_tags += sum(len(entry["tags"]) for entry in truth.values())
        if family not in detectors:
            detectors[family] = detector(name, family)
        detect = detectors[family]
        if detect is None:
            continue
        sign = next(
            (np.sign(signed_area(np.array(t["corners"], float))) for v in truth.values() for t in v["tags"]), 1.0
        )
        tags, false, errors = 0, 0, []
        for frame in sorted(truth):
            image = cv2.imread(str(root / scenario / frame), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(root / scenario / frame)
            found = [(i, to_corpus(q, sign)) for i, q in detect(np.ascontiguousarray(image))]
            frame_errors, frame_false = score_frame(truth[frame]["tags"], found)
            tags += len(truth[frame]["tags"])
            false += frame_false
            errors += frame_errors
        out[scenario] = {
            "tags": tags,
            "found": len(errors),
            "false": false,
            "worst_corner_median_px": float(np.median(errors)) if errors else None,
            "worst_corner_p95_px": float(np.percentile(errors, 95)) if errors else None,
        }
    if frames != FRAMES or main_tags != MAIN_TAGS:
        raise ValueError(f"expected {FRAMES} frames and {MAIN_TAGS} tag36h11 tags, found {frames} and {main_tags}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("work/external/vkapriltag-corpus"))
    parser.add_argument("--backends", nargs="+", default=["napa", "napa16", "apriltag3-d1"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; results are not overwritten")
    results = {}
    for name in args.backends:
        results[name] = evaluate(args.root, name)
        main_sets = {k: v for k, v in results[name].items() if k != SHORT_FAMILY_SET}
        print(
            f"{name}: {sum(v['found'] for v in main_sets.values())} of {sum(v['tags'] for v in main_sets.values())}, "
            f"false {sum(v['false'] for v in main_sets.values())}",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    core = Path(_library()._name).resolve()
    record = {"source_commit": COMMIT, "core_library": cb.public_text(str(core)), "core_sha256": cb.sha256(core)}
    args.output.write_text(json.dumps({**record, "results": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
