#!/bin/bash
# The published corpus (20 scenarios x 50 frames x 3 resolutions, seed 2468) and the 18-size range sweep (50 tags
# each), with every library at its defaults, interleaved on one P-core: 1 warm-up + 10 timed calls per frame.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=benchmarks/results/2026-10-29_cpu_detector_matrix
export NAPA_CORE_LIBRARY=$PWD/work/build/napa/libnapa_core.so
export OPENBLAS_NUM_THREADS=1
benchmarks/cpu_comparison_shims/monitor.sh $R/monitor.tsv 2 3 &
MON=$!
trap 'kill $MON' EXIT
taskset -c 2 work/venv/bin/python benchmarks/cpu_comparison.py latency \
  --backends napa,wpilib-d2,apriltag3-d2,apriltag3-d1,locus,opencv,aruco-nano,rapidtag --threads 1 \
  --frames 50 --stress none --range-cases 50 --repetitions 10 --warmup 1 --output $R/matrix.json
