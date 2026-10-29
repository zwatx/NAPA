#!/bin/bash
# Range sweep on the current renderer (18 black-square sizes, 30 tags each, 640x480), one P-core, interleaved.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=benchmarks/results/2026-10-28_cpu_comparison
export NAPA_CORE_LIBRARY=$PWD/work/build/napa/libnapa_core.so
export OPENBLAS_NUM_THREADS=1
ALL=napa,napa-mtp16,napa-mtp22,napa-mtp43,napa-mtp64,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,opencv-aruco3-half16,aruco-nano,rapidtag,rapidtag-subpix
taskset -c 2 work/venv/bin/python benchmarks/cpu_comparison.py latency --backends $ALL --threads 1 \
  --frames 0 --stress-frames 0 --range-cases 30 --resolutions 640x480 --repetitions 7 --warmup 1 \
  --output $R/range_sweep_latency.json
