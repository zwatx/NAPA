"""Compare guarded and pre-guard core outputs exactly on the cached diagnostic images."""

import argparse
import json
from pathlib import Path

import numpy as np
from rerun import ROOT, backend, cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="work/optimization/pre-roi.so")
    parser.add_argument("--after", default="work/optimization/libnapa_core_candidate.so")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("terminal_guard_outputs.json"))
    args = parser.parse_args()
    frames = cached(ROOT / f"work/optimization/frames-{args.seed}.npz", False, args.seed)
    paths = (args.before, args.after)
    differences = []
    configs = []
    calls = 0
    for minimum in (16, 22, 32, 43, 64):
        pair = [
            backend(f"{label}-{minimum}", ROOT / path, minimum)
            for label, path in zip(("pre-guard", "guarded"), paths, strict=True)
        ]
        configs.extend(b.config for b in pair)
        before, after = [b.factory() for b in pair]
        for frame in frames:
            left = before(frame.image)
            right = after(frame.image)
            equal = len(left) == len(right) and all(
                a[0] == b[0] and np.array_equal(a[1], b[1]) for a, b in zip(left, right, strict=True)
            )
            if not equal:
                differences.append({"frame": frame.name, "min_tag_px": minimum})
            calls += 1
    result = {"frame_configuration_pairs": calls, "differences": differences, "cores": configs}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(calls, "frame/configuration pairs;", len(differences), "different outputs")
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
