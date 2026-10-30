"""Task-local targeted A/B: reuse CPU comparison frames, scoring and timing boundaries."""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks import cpu_backends as cb  # noqa: E402
from benchmarks import cpu_comparison as cc  # noqa: E402
from benchmarks import performance_matrix as pm  # noqa: E402
from napa_vision import napa_detector as nd  # noqa: E402


def backend(label, path, mtp):
    path = Path(path).resolve()

    def factory():
        previous = os.environ.get("NAPA_CORE_LIBRARY")
        os.environ["NAPA_CORE_LIBRARY"] = str(path)
        nd._library.cache_clear()
        try:
            detector = nd.NapaDetector(min_tag_px=mtp, max_detections=1024)
            if Path(detector._library._name).resolve() != path or str(path) not in cb.loaded_libraries(path.name):
                raise RuntimeError(f"requested core not loaded: {path}")
        finally:
            nd._library.cache_clear()
            if previous is None:
                os.environ.pop("NAPA_CORE_LIBRARY", None)
            else:
                os.environ["NAPA_CORE_LIBRARY"] = previous
        return lambda image: [(o.tag_id, o.corners) for o in detector.detect(image)]

    return cb.Backend(
        label,
        factory,
        {
            "path": str(path),
            "sha256": cb.sha256(path),
            "min_tag_px": mtp,
            "max_detections": 1024,
            "threads": 1,
            "max_hamming": 3,
            "min_contrast": 20,
            "min_border_cells": 22,
            "sharpening": 0.25,
        },
    )


def frames(full=False, seed=2468):
    if full:
        yield from cc.iter_corpus(
            scenarios=pm.SCENARIO_NAMES,
            stress=tuple(cc.STRESS),
            resolutions=pm.DEFAULT_RESOLUTIONS,
            frames=15,
            stress_frames=15,
            range_cases=30,
            seed=seed,
        )
        return
    names = {f"{s}-{w}x{h}-000" for w, h in pm.DEFAULT_RESOLUTIONS for s in (*pm.SCENARIO_NAMES, *cc.STRESS)}
    for s in ("stress-heavy-noise", "stress-clutter-noisy", "stress-tiny", "low-contrast", "motion-blur"):
        names.update(f"{s}-1920x1080-{i:03d}" for i in (1, 3, 5, 8, 12))
    # All prior stride-one losses against Locus d2 and Nano are deliberately included.
    import csv

    with (ROOT / "benchmarks/results/2026-10-28_cpu_comparison/single_thread_latency.csv").open() as f:
        rows = list(csv.DictReader(f))
    dense = {r["frame"]: float(r["median_ms"]) for r in rows if r["backend"] == "napa-mtp16"}
    names.update(
        r["frame"]
        for r in rows
        if r["backend"] in ("locus-d2", "aruco-nano")
        and float(r["median_ms"]) < dense.get(r["frame"], 0)
        and r["scenario"] != "range"
    )
    for name in sorted(names):
        yield cc.frame_by_name(name, seed=seed)


def cached(path, full, seed):
    if full:
        return frames(True, seed)
    if not path.exists():
        fs = list(frames(False, seed))
        meta = [
            dict(
                scenario=f.scenario,
                index=f.index,
                resolution=f.resolution,
                details=f.details,
                tags=[[t.tag_id, t.corners.tolist()] for t in f.tags],
            )
            for f in fs
        ]
        np.savez(path, metadata=json.dumps(meta), **{f"image_{i}": f.image for i, f in enumerate(fs)})
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata"]))
    return [
        pm.Frame(
            m["scenario"],
            m["index"],
            tuple(m["resolution"]),
            data[f"image_{i}"],
            [pm.TagTruth(t[0], np.asarray(t[1])) for t in m["tags"]],
            m["details"],
        )
        for i, m in enumerate(meta)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--core", action="append", default=[], help="label=path")
    p.add_argument("--mtp", default="16,32")
    p.add_argument("--competitors", default="")
    p.add_argument("--full", action="store_true")
    p.add_argument("--only", default="")
    p.add_argument("--seed", type=int, default=2468)
    p.add_argument("--order-seed", type=int, default=8712)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cb.configure_threads(1, list(os.sched_getaffinity(0)))
    fs = cached(ROOT / f"work/optimization/frames-{a.seed}.npz", a.full, a.seed)
    if a.only:
        have = {f.name: f for f in fs}
        fs = [have[name] if name in have else cc.frame_by_name(name, seed=a.seed) for name in a.only.split(",")]
    bs = [
        backend(f"{label}-mtp{mtp}", path, mtp)
        for entry in a.core
        for label, path in [entry.split("=", 1)]
        for mtp in map(int, a.mtp.split(","))
    ]
    if a.profile or a.trace:
        output = []
        for b in bs:
            detect = b.factory()
            lib = ctypes.CDLL(b.config["path"])
            fn = lib.napa_profile_read if a.profile else lib.napa_trace_read
            typ = ctypes.c_uint64 if a.profile else ctypes.c_int
            buf = (typ * (16 if a.profile else 6 * 65536))()
            fn.argtypes = [ctypes.POINTER(typ), ctypes.c_int]
            for f in fs:
                fn(buf, 16 if a.profile else 65536)
                detections = detect(f.image)
                n = fn(buf, 16 if a.profile else 65536)
                rec = dict(backend=b.name, frame=f.name, score=cc.score_detail(f.tags, detections))
                rec["profile_ns" if a.profile else "trace"] = (
                    list(buf[:8]) if a.profile else np.asarray(buf[: n * 6]).reshape(n, 6).tolist()
                )
                output.append(rec)
        a.output.write_text(json.dumps(output, indent=2))
        return
    competitors, errors = cb.build_backends(a.competitors.split(",") if a.competitors else [], 1)
    if errors:
        raise RuntimeError(errors)
    bs += competitors
    life, rows, observed = cc.run_interleaved(
        fs,
        backends=bs,
        repetitions=a.repetitions,
        warmup=0 if a.full else 1,
        seed=a.order_seed,
        progress=lambda n: print(f"{n} frames", flush=True) if n % 100 == 0 else None,
    )
    cc._write_rows(a.output.with_suffix(".csv"), rows)
    cc._write_json(
        a.output,
        dict(
            config=vars(a),
            backends={b.name: b.config for b in bs},
            lifecycle=life,
            observed=observed,
            machine=cc.machine_metadata(),
            summary=cc.aggregate(rows, bs[0].name),
        ),
    )
    for b in bs:
        rs = [r for r in rows if r["backend"] == b.name]
        es = [v for r in rs for v in cc._split(r["corner_errors_px"])]
        print(
            b.name,
            "frames",
            len(rs),
            "TP",
            sum(r["true_positive"] for r in rs),
            "FN",
            sum(r["false_negative"] for r in rs),
            "FP",
            sum(r["false_id"] + r["duplicate"] for r in rs),
            "corner p50/p95/max",
            np.percentile(es, [50, 95, 100]).round(4),
            flush=True,
        )


if __name__ == "__main__":
    main()
