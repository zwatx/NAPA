# Merged-branch CPU comparison

This run measures the consolidated detector from source commit
`2489605efc2f07043222b52e3a69132d2e5dee6d`. It does not modify the October measurements.
The source hashes, compiler flags, native binary hash and configuration recorded before timing are in
[protocol.json](protocol.json).

## Inputs and environment

The existing fixed renderer supplies all 4,500 frames, with seed 2468: 3,000 main frames, 600 stress frames and
900 range frames. The renderer source has the same SHA256 as the October 30 protocol. No frame is selected by a
detector's output. Main, stress and range are reported separately.

Use the existing third-party setup described in the
[original CPU comparison](../2026-10-28_cpu_comparison/commands.md). The measured environment uses Python 3.14.4,
NumPy 2.5.4, OpenCV 5.0.0, RobotPy AprilTag 2026.2.2, Locus 0.10.0 and RapidTag 0.1.12, plus the AprilTag and
ArUco Nano native shims. Exact effective settings are retained in `matrix.json`.

Each NAPA row uses its named sampling minimum, `split_touching="auto"`, one thread, a 64-output capacity and the
other public defaults. In particular, shadow recovery, small-tag regrowth, soft decoding, corner alignment,
camera gamma and calibrated lens fitting are off. The previous headline default was 32; this run's default is 22.

## Commands

Run from the measured source checkout. Supply paths to the existing benchmark environment and third-party builds,
and choose a new output directory. Do not overwrite this run or historical measurements.

```bash
export NAPA_COMPARE_PYTHON=/path/to/benchmark/venv/bin/python
export NAPA_BENCH_SHIMS=/path/to/shared/work/build/shims
export NAPA_BENCH_THIRD_PARTY=/path/to/shared/work/thirdparty
export NAPA_COMPARE_OUTPUT=work/merged-cpu-reproduction
mkdir -p "$NAPA_COMPARE_OUTPUT"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 RAPIDTAG_CORES=2
export PYTHONPATH="$PWD/src:$PWD"
export NAPA_CORE_LIBRARY="$PWD/$NAPA_COMPARE_OUTPUT/napa_core.so"
cc -O3 -std=c99 -fvisibility=hidden -fno-wrapv -pthread -fPIC -shared \
  -Isrc/napa_vision/_native src/napa_vision/_native/napa_core.c -lm -o "$NAPA_CORE_LIBRARY"

NAPA_COMPARE_BACKENDS=napa,napa-mtp16,napa-mtp32,wpilib-d2,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,opencv-aruco3-half16,aruco-nano,rapidtag,rapidtag-subpix

taskset -c 2 "$NAPA_COMPARE_PYTHON" -B benchmarks/cpu_comparison.py validate \
  --backends "$NAPA_COMPARE_BACKENDS" --threads 1 --seed 2468 \
  --output "$NAPA_COMPARE_OUTPUT/validate.json"

taskset -c 2 "$NAPA_COMPARE_PYTHON" -B benchmarks/cpu_comparison.py latency \
  --backends "$NAPA_COMPARE_BACKENDS" --threads 1 --seed 2468 --order-seed 2468 \
  --resolutions 640x480,1280x720,1920x1080 --frames 50 --stress-frames 50 --range-cases 50 \
  --warmup 1 --repetitions 10 --output "$NAPA_COMPARE_OUTPUT/matrix.json"
```

CPU2 was verified as a performance core on the measured i7-12700H; CPU3 is its SMT sibling. Adapt affinity and
`RAPIDTAG_CORES` to another host rather than assuming those CPU numbers identify equivalent cores everywhere.
The run's [monitor](monitor.tsv) records CPU2 and CPU3 utilization, frequency and package temperatures every five
seconds. Background applications and system settings were left as they were.

## Regenerate tables

After the measurements have completed, regenerate the report from its saved data:

```bash
"$NAPA_COMPARE_PYTHON" benchmarks/results/2026-11-15_merged_cpu_comparison/make_report.py
```

The generator reads the measurements and checks coverage, source hashes, settings and output capacity. It writes
`report.md` and `analysis.json`; it does not run a detector, rewrite the READMEs or publish a package.
Keep the measured source files unchanged when regenerating. A reproduction is a separate run and needs its own
protocol, timestamps and binary hashes; it must not be relabelled as these measurements.
