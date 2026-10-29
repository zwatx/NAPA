#!/bin/bash
# Single-thread comparison on one verified P-core (CPU 2: core 1, P-core list 0-11, SMT sibling CPU 3).
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=benchmarks/results/2026-10-28_cpu_comparison
export NAPA_CORE_LIBRARY=$PWD/work/build/napa/libnapa_core.so
ALL=napa,napa-mtp16,napa-mtp22,napa-mtp43,napa-mtp64,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,aruco-nano,rapidtag,rapidtag-subpix
benchmarks/cpu_comparison_shims/monitor.sh $R/monitor_single_thread.tsv 2 3 &
MON=$!
trap 'kill $MON' EXIT
taskset -c 2 work/venv/bin/python benchmarks/cpu_comparison.py latency --backends $ALL --threads 1 \
  --frames 15 --stress-frames 15 --range-cases 30 --repetitions 7 --warmup 1 \
  --output $R/single_thread_latency.json
