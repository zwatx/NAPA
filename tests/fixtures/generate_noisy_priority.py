"""Regenerate owned projected-tag fixtures from the repository benchmark renderer."""

from pathlib import Path

import numpy as np

from benchmarks.cpu_comparison import frame_by_name


def main():
    cases = [
        ("stress-heavy-noise-1280x720-048", 6, 192, 96, 576, 576),
        ("stress-heavy-noise-1920x1080-005", 390, 1248, 0, 576, 480),
        ("stress-heavy-noise-1920x1080-044", 484, 384, 480, 576, 480),
    ]
    for name, tag_id, x, y, width, height in cases:
        frame = frame_by_name(name, seed=2468, range_cases=50)
        truth = next(tag for tag in frame.tags if tag.tag_id == tag_id)
        np.savez_compressed(
            Path(__file__).parent / f"noisy-corner-{tag_id}.npz",
            image=frame.image[y : y + height, x : x + width],
            corners=np.asarray(truth.corners) - [x, y],
            tag_id=tag_id,
        )


if __name__ == "__main__":
    main()
