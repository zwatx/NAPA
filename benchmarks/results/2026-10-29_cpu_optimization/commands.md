# Reproducing this optimization comparison

This is a follow-up to the pinned CPU builds in [the original comparison](../2026-10-28_cpu_comparison/commands.md).
Use only its source-clone, venv, library-build and shim-build setup blocks. Do not run its older measurement scripts: they write to the preserved historical results directory. The new script generates missing resource frame sets itself.
Add WPILib and Ruff to the same benchmark environment (the earlier setup did not require them):

```bash
work/venv/bin/python -m pip install robotpy-apriltag==2026.2.2 ruff==0.17.1
```

No system-wide dependencies, GPU backends or detector process per frame are used.

Build the baseline, intermediate snapshots and final core before running. From the repository root:

```bash
mkdir -p work/optimization
D=benchmarks/results/2026-10-29_cpu_optimization
git show 674c91b:src/napa_vision/_native/napa_core.c > work/optimization/baseline.c
compile_core() {
  cc -fno-strict-overflow -DNDEBUG -g -O2 -Wall -Wextra -fPIC -O3 -std=c99 \
    -fvisibility=hidden -fno-wrapv -shared -Isrc/napa_vision/_native -o "$1" "$2" -lm
}
compile_core work/optimization/baseline.so work/optimization/baseline.c
compile_core work/optimization/pre-roi.so "$D/diagnostic_napa_core.c"
compile_core work/optimization/unpacked.so "$D/unpacked_napa_core.c"
compile_core work/optimization/libnapa_core_candidate.so src/napa_vision/_native/napa_core.c
bash "$D/run.sh"
```

Replays write to `work/optimization/reproduction` by default, preserving committed raw measurements.
Set `NAPA_BENCH_OUTPUT_DIR` to another fresh directory to choose a destination. `run.sh` includes the final
follow-up and quality replay, then regenerates the report. A fresh source manifest is written in the new output directory; historical source manifests are not relabeled. To regenerate tables from the committed data alone:

```bash
work/venv/bin/python benchmarks/results/2026-10-29_cpu_optimization/make_report.py
```

`source_manifest.json` identifies the tested source and binaries. Debug paths can make file hashes differ between
checkouts. `implementation.patch` records the native changes and regression tests against the base commit.

`rerun.py` reuses the existing CPU comparison renderer, scoring, timing and adapters. It loads each requested
NAPA core through the public Python API, checks the actual mapped path, and records its binary hash. Factory
creation is sequential; the temporary library override never runs inside a timed call. Detector instances and
output buffers are built once. Each call includes native work and Python result conversion.

The 149-frame cache is diagnostic: every scenario/resolution plus historical losses to Locus d2 and ArUco Nano.
Seed 7319 changes the images under the same selection rule; it is not a statistically random camera corpus.
The full 1,620-frame NAPA replay is a correctness check with one call per frame and no per-frame warm-up; do not
use its timings as the main latency result. No scoring rules or image content depend on candidate output.

The NAPA A/B harness allows 1,024 output detections, consistently for baseline and candidate, to avoid a capacity
limit in dense scenes. The existing resource and real-photo adapters retain their 64-output default.

The first diagnostic run loaded the core before the terminal-small-crop guards were added. These guards affect
terminal stride-one/two levels; the tag36h11 benchmark's 640×480 and larger frames do not take those paths.
Its original hash remains in the diagnostic configuration; `diagnostic_napa_core.c` preserves that exact source snapshot. Compile it with the same flags and native include directory to reproduce that version. Subsequent processes use the guarded core.
The exact pre/post checks are in `terminal_guard_outputs.json` and `packed_outputs_2468.json`/`packed_outputs_7319.json`. The final-prefixed latency, quality and resource files use the packed final core. Earlier measurements retain their original hashes.

Core affinity and thermal/load observations are in `monitor.tsv`; actual thread settings, versions and affinities
are recorded in each JSON. Resource workers are isolated and sequential, with phase-separated memory readouts.
Real-photo results are unannotated timing and detector agreement, not ground-truth accuracy.

The final storage-only change was checked for exact ordered IDs and corner coordinates on both cached seeds and
all five strides. The final core then replayed all 1,620 quality frames independently. `safety.c` is a separate
ASan/UBSan harness; sanitizer execution times are never used as performance measurements.
