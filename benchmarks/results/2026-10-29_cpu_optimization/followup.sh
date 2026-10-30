#!/bin/bash
# Focused checks and one short final cached cohort after the component-memory optimization.
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=benchmarks/results/2026-10-29_cpu_optimization
R=${NAPA_BENCH_OUTPUT_DIR:-work/optimization/reproduction}
mkdir -p "$R"
export NAPA_BENCH_OUTPUT_DIR="$R"
export OPENBLAS_NUM_THREADS=1
export NAPA_CORE_LIBRARY=$PWD/work/optimization/libnapa_core_candidate.so
export PYTHONPATH=$PWD/src
benchmarks/cpu_comparison_shims/monitor.sh "$R/monitor_followup.tsv" 0 1 2 3 4 6 8 10 &
MON=$!
trap 'kill "$MON"' EXIT
for seed in 2468 7319; do
  taskset -c 2 work/venv/bin/python "$D/compare_cores.py" --before work/optimization/unpacked.so --after work/optimization/libnapa_core_candidate.so --seed "$seed" --output "$R/packed_outputs_$seed.json"
done
# Three repeats on the existing 149 cached frames, not a regenerated full matrix.
ALL=apriltag3-d2,apriltag3-d1,wpilib-d2,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,opencv-aruco3-half16,aruco-nano,rapidtag,rapidtag-subpix
taskset -c 2 work/venv/bin/python "$D/rerun.py" --core candidate=work/optimization/libnapa_core_candidate.so --mtp 16,32 --competitors "$ALL" --repetitions 3 --order-seed 18391 --output "$R/final_latency.json"
FRAMES=$(paste -sd, "$D/retime_frames.txt")
taskset -c 2 work/venv/bin/python "$D/rerun.py" --core candidate=work/optimization/libnapa_core_candidate.so --mtp 16,32 --competitors locus-d2,aruco-nano,opencv-aruco3-min32,opencv-aruco3-half16,wpilib-d2 --only "$FRAMES" --repetitions 20 --order-seed 19103 --output "$R/outlier_retime.json"
# Remeasure NAPA, nearest competitors, and WPILib; reuse already measured unchanged slower backends.
CLOSE=napa,napa-mtp16,aruco-nano,locus-d2,opencv-aruco3-min32,wpilib-d2
for res in 640x480 1280x720 1920x1080; do
  work/venv/bin/python benchmarks/cpu_comparison.py resources --backends "$CLOSE" --threads 1 --cpus 2 --mode serial --frames-npz "work/frames/$res.npz" --passes 2 --rounds 2 --output "$R/final_resources_1t_$res.json"
  work/venv/bin/python benchmarks/cpu_comparison.py resources --backends "$CLOSE" --threads 6 --cpus 0,2,4,6,8,10 --mode serial --frames-npz "work/frames/$res.npz" --passes 2 --rounds 2 --output "$R/final_resources_6t_$res.json"
done
taskset -c 2 work/venv/bin/python benchmarks/cpu_comparison.py latency --backends napa,napa-mtp16,apriltag3-d1,aruco-nano,locus --threads 1 --frames 0 --stress-frames 0 --range-cases 0 --real-images work/thirdparty/apriltag/test/data/*.jpg --repetitions 30 --warmup 3 --output "$R/final_real_photos.json"
work/venv/bin/python benchmarks/cpu_comparison.py real-agreement "$R/final_real_photos.json" --output "$R/final_real_agreement.json"
taskset -c 2 work/venv/bin/python "$D/adapter_overhead.py"
# Sanitizers are separate from timing. Never use their execution times as benchmark data.
cc -O1 -g -std=c99 -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc/napa_vision/_native -o work/optimization/safety "$D/safety.c" src/napa_vision/_native/napa_core.c -lm
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 taskset -c 2 work/optimization/safety > "$R/sanitizers.txt" 2>&1
work/venv/bin/python -m pytest -q -rs > "$R/tests.txt" 2>&1
.venv/bin/ruff check . > "$R/lint.txt" 2>&1
