"""Public NAPA adapter versus a direct ctypes C call; neither includes scoring or rendering."""

import ctypes
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from benchmarks import cpu_comparison as cc  # noqa: E402
from napa_vision import NapaDetector  # noqa: E402


def main():
    rng = random.Random(1039)
    detector = NapaDetector(min_tag_px=16)
    output = []
    for name in ("empty-1280x720-000", "one-tag-1280x720-000", "stress-dense-1280x720-000"):
        frame = cc.frame_by_name(name)
        image = frame.image

        def native(image=image):
            return detector._library.napa_detect(
                detector._handle,
                image.ctypes.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                ctypes.byref(detector._params),
                detector._results,
                len(detector._results),
            )

        def public(image=image):
            return [(o.tag_id, o.corners) for o in detector.detect(image)]

        for _ in range(5):
            native()
            public()
        native_count = native()
        assert native_count == len(public())
        samples = {"direct_ctypes": [], "public_adapter": []}
        gc.disable()
        for _ in range(200):
            order = [("direct_ctypes", native), ("public_adapter", public)]
            rng.shuffle(order)
            for label, call in order:
                start = perf_counter_ns()
                call()
                samples[label].append(perf_counter_ns() - start)
        gc.enable()
        output.append(
            {
                "frame": name,
                "detections": native_count,
                "p50_ms": {k: float(np.median(v)) / 1e6 for k, v in samples.items()},
                "samples_ns": samples,
            }
        )
    path = Path(os.environ.get("NAPA_BENCH_OUTPUT_DIR", Path(__file__).parent)) / "adapter_overhead.json"
    path.write_text(
        json.dumps(
            {
                "affinity": sorted(os.sched_getaffinity(0)),
                "core_sha256": hashlib.sha256(Path(detector._library._name).read_bytes()).hexdigest(),
                "frames": output,
                "note": "Direct ctypes retains call overhead; differences estimate Python input/result handling.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
