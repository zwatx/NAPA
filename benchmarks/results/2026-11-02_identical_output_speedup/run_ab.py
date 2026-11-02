"""Baseline and candidate NAPA cores in the repo's own timing loop, on the 2026-10-30 fixed corpus.

Uses benchmarks/cpu_comparison.py unchanged: iter_corpus with that run's arguments (seed 2468; 20 scenes,
4 stress cases, 50 frames each at 640x480, 1280x720, 1920x1080; 18 x 50 range frames) and run_interleaved
(one warm-up and ten timed calls per frame and backend, backend order shuffled with order seed 2468).
Four backends go through the public NapaDetector adapter: each core at min_tag_px 32 and 16, every
detector bound to its own explicitly loaded library. Writes the per-frame rows (CSV) and a summary (JSON).

    run_ab.py BASELINE.so CANDIDATE.so OUT_PREFIX

The process pins itself to CPU $AB_CPU (default 4). OUT_PREFIX.csv gets cpu_comparison's per-frame rows and
OUT_PREFIX.json the summary: pooled percentiles of every timed call per group and backend, and per-frame median
time ratios (baseline / candidate).
"""

import ctypes
import json
import os
import sys
import weakref
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from benchmarks import cpu_backends as cb  # noqa: E402
from benchmarks import cpu_comparison as cc  # noqa: E402
from benchmarks import performance_matrix as pm  # noqa: E402
from napa_vision import napa_detector as nd  # noqa: E402


def load(path: str) -> ctypes.CDLL:
    os.environ[nd._LIBRARY_ENV] = path
    return nd._library.__wrapped__()  # the uncached loader: one configured CDLL per path


def backend(name: str, lib: ctypes.CDLL, mtp: int) -> cb.Backend:
    def factory():
        det = nd.NapaDetector(min_tag_px=mtp)
        det._finalizer()  # release the default library's handle, then bind this one
        det._library = lib
        det._handle = ctypes.c_void_p(lib.napa_create_family(b"tag36h11"))
        det._finalizer = weakref.finalize(det, lib.napa_destroy, det._handle)
        return lambda image: [(o.tag_id, o.corners) for o in det.detect(image)]

    return cb.Backend(name, factory, {"library": lib._name, "min_tag_px": mtp})


def main() -> None:
    base, cand, prefix = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    os.sched_setaffinity(0, {int(os.environ.get("AB_CPU", "4"))})
    libs = {"base": load(base), "cand": load(cand)}
    backends = [backend(f"{k}-mtp{m}", lib, m) for m in (32, 16) for k, lib in libs.items()]
    frames = cc.iter_corpus(
        scenarios=pm.SCENARIO_NAMES,
        stress=tuple(cc.STRESS),
        resolutions=pm.DEFAULT_RESOLUTIONS,
        frames=50,
        stress_frames=50,
        range_cases=50,
        seed=2468,
    )
    total = [0]

    def progress(done: int) -> None:
        total[0] = done
        if done % 100 == 0:
            print(f"{done}/4500 frames  load {os.getloadavg()[0]:.2f}", file=sys.stderr, flush=True)

    _, rows, _ = cc.run_interleaved(frames, backends=backends, repetitions=10, warmup=1, seed=2468, progress=progress)
    cc._write_rows(prefix.with_suffix(".csv"), rows)
    groups = {"main": set(pm.SCENARIO_NAMES), "stress": set(cc.STRESS), "range": {"range"}}
    summary = {}
    for g, scenes in groups.items():
        sel = [r for r in rows if r["scenario"] in scenes]
        for b in backends:
            mine = [r for r in sel if r["backend"] == b.name]
            ns = np.array([int(v) for r in mine for v in str(r["samples_ns"]).split(";")]) / 1e6
            cpu = np.array([int(v) for r in mine for v in str(r["cpu_ns"]).split(";")]) / 1e6
            summary[f"{g}/{b.name}"] = {
                "frames": len(mine),
                "calls": int(ns.size),
                "p50_ms": float(np.percentile(ns, 50)),
                "p95_ms": float(np.percentile(ns, 95)),
                "p99_ms": float(np.percentile(ns, 99)),
                "mean_ms": float(ns.mean()),
                "cpu_mean_ms": float(cpu.mean()),
                "found": int(sum(r["true_positive"] for r in mine)),
                "tags": int(sum(r["tags"] for r in mine)),
                "wrong_ids": int(sum(r["false_id"] for r in mine)),
                "unstable_frames": int(sum(not r["stable_across_repetitions"] for r in mine)),
            }
        for m in (32, 16):
            a = {r["frame"]: r["median_ms"] for r in sel if r["backend"] == f"base-mtp{m}"}
            b = {r["frame"]: r["median_ms"] for r in sel if r["backend"] == f"cand-mtp{m}"}
            ratio = np.array([a[f] / b[f] for f in a])
            same = sum(
                r1["detected_ids"] == r2["detected_ids"] and r1["corner_errors_px"] == r2["corner_errors_px"]
                for r1 in sel
                if r1["backend"] == f"base-mtp{m}"
                for r2 in [next(r for r in sel if r["backend"] == f"cand-mtp{m}" and r["frame"] == r1["frame"])]
            )
            summary[f"{g}/ratio-mtp{m}"] = {
                "median": float(np.median(ratio)),
                "min": float(ratio.min()),
                "geomean": float(np.exp(np.log(ratio).mean())),
                "candidate_slower_frames": int((ratio < 1).sum()),
                "frames": int(ratio.size),
                "frames_with_identical_ids_and_corner_errors": int(same),
            }
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(k, json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}))


if __name__ == "__main__":
    main()
