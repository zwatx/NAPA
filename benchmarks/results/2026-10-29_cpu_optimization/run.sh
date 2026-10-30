#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
D=benchmarks/results/2026-10-29_cpu_optimization
R=${NAPA_BENCH_OUTPUT_DIR:-work/optimization/reproduction}
mkdir -p "$R"
export NAPA_BENCH_OUTPUT_DIR="$R"
export OPENBLAS_NUM_THREADS=1
export NAPA_CORE_LIBRARY=$PWD/work/optimization/libnapa_core_candidate.so
# Generate only the resource inputs when absent; never execute the older result-writing scripts.
for res in 640x480 1280x720 1920x1080; do
  if [[ ! -f "work/frames/$res.npz" ]]; then
    taskset -c 2 work/venv/bin/python benchmarks/cpu_comparison.py corpus --resolutions "$res" --frames 4 --stress-frames 4 --seed 2468 --output "work/frames/$res.npz"
  fi
done
work/venv/bin/python "$D/record_sources.py"
ALL=apriltag3-d2,apriltag3-d1,wpilib-d2,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,opencv-aruco3-half16,aruco-nano,rapidtag,rapidtag-subpix
benchmarks/cpu_comparison_shims/monitor.sh "$R/monitor.tsv" 0 1 2 3 4 6 8 10 &
MON=$!
trap 'kill "$MON"' EXIT
# Cached diagnostic corpus: all scenarios/resolutions and every former loss to the closest competitors.
taskset -c 2 work/venv/bin/python "$D/rerun.py" --core baseline=work/optimization/baseline.so --core candidate=work/optimization/libnapa_core_candidate.so --mtp 16,22,32,43,64 --competitors "$ALL" --repetitions 7 --output "$R/diagnostic.json"
# Same frame-selection rule on another renderer seed; timing is against the closest competitors.
taskset -c 2 work/venv/bin/python "$D/rerun.py" --core baseline=work/optimization/baseline.so --core candidate=work/optimization/libnapa_core_candidate.so --mtp 16,32 --competitors locus-d2,aruco-nano --seed 7319 --repetitions 7 --output "$R/second_seed.json"
# Correctness-only replay: no competitor timing sweep and no per-frame warmup.
taskset -c 2 work/venv/bin/python "$D/rerun.py" --core baseline=work/optimization/baseline.so --core candidate=work/optimization/libnapa_core_candidate.so --full --repetitions 1 --output "$R/quality.json"
# Real images are explicitly unannotated. These are timing and agreement, never accuracy results.
taskset -c 2 work/venv/bin/python benchmarks/cpu_comparison.py latency --backends "napa,napa-mtp16,$ALL" --threads 1 --frames 0 --stress-frames 0 --range-cases 0 --real-images work/thirdparty/apriltag/test/data/*.jpg --repetitions 30 --warmup 3 --output "$R/real_photos.json"
work/venv/bin/python benchmarks/cpu_comparison.py real-agreement "$R/real_photos.json" --output "$R/real_agreement.json"
# Resource workers run strictly one at a time. Reuse the previous identical saved 96-frame sets.
MC=napa,napa-mtp16,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,opencv-aruco3-half16,aruco-nano,rapidtag,rapidtag-subpix
for res in 640x480 1280x720 1920x1080; do
  work/venv/bin/python benchmarks/cpu_comparison.py resources --backends "$MC" --threads 1 --cpus 2 --mode serial --frames-npz "work/frames/$res.npz" --passes 2 --rounds 2 --output "$R/resources_1t_$res.json"
  work/venv/bin/python benchmarks/cpu_comparison.py resources --backends "$MC" --threads 6 --cpus 0,2,4,6,8,10 --mode serial --frames-npz "work/frames/$res.npz" --passes 2 --rounds 2 --output "$R/resources_6t_$res.json"
done

kill "$MON"
wait "$MON" || true
trap - EXIT
bash "$D/followup.sh"
taskset -c 2 work/venv/bin/python "$D/rerun.py" --core candidate=work/optimization/libnapa_core_candidate.so --mtp 16,32 --full --repetitions 1 --output "$R/final_quality.json"
work/venv/bin/python "$D/make_report.py"
