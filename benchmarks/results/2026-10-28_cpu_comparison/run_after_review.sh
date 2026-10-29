#!/bin/bash
# Re-runs after the independent review: identical-image outlier retime, the corrected ArUco3 half-scale config,
# and every isolated resource / multicore / throughput run with phase-separated memory and limited BLAS pools.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=benchmarks/results/2026-10-28_cpu_comparison
PY=work/venv/bin/python
export NAPA_CORE_LIBRARY=$PWD/work/build/napa/libnapa_core.so
export OPENBLAS_NUM_THREADS=1
ALL=napa,napa-mtp16,napa-mtp22,napa-mtp43,napa-mtp64,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,opencv-aruco3-half16,aruco-nano,rapidtag,rapidtag-subpix
MC=napa,napa-mtp16,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3-min32,opencv-aruco3-half16,aruco-nano
PCORES=0,2,4,6,8,10
# The backend list of the main single-thread run, for the outlier retime
MAIN=napa,napa-mtp16,napa-mtp22,napa-mtp43,napa-mtp64,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,aruco-nano,rapidtag,rapidtag-subpix
benchmarks/cpu_comparison_shims/monitor.sh $R/monitor_after_review.tsv 0 2 3 4 6 8 10 &
MON=$!
trap 'kill $MON' EXIT

echo "== outlier retime on the original images (render seed 2468, order seed 1357)"
taskset -c 2 $PY benchmarks/cpu_comparison.py latency --backends $MAIN \
  --threads 1 --only-frames "$(cat $R/outlier_frames.txt)" --range-cases 30 --repetitions 30 --warmup 3 \
  --seed 2468 --order-seed 1357 --output $R/outlier_retime.json

echo "== supplementary interleaved: corrected ArUco3 half-scale configs vs NAPA, main corpus"
taskset -c 2 $PY benchmarks/cpu_comparison.py latency --backends napa,napa-mtp16,opencv-aruco3-min32,opencv-aruco3-half16 \
  --threads 1 --frames 15 --stress-frames 15 --range-cases 30 --repetitions 7 --warmup 1 \
  --output $R/aruco3_half_scale_latency.json

for res in 640x480 1280x720 1920x1080; do
  echo "== single-thread resources $res"
  $PY benchmarks/cpu_comparison.py resources --backends $ALL --threads 1 --cpus 2 --mode serial \
    --frames-npz work/frames/$res.npz --passes 3 --rounds 3 --output $R/resources_1thread_$res.json
done
for res in 640x480 1280x720 1920x1080; do
  echo "== intra-frame 6 threads $res"
  $PY benchmarks/cpu_comparison.py resources --backends $MC,rapidtag --threads 6 --cpus $PCORES --mode serial \
    --frames-npz work/frames/$res.npz --passes 3 --rounds 3 --output $R/resources_6threads_$res.json
done
for res in 640x480 1280x720 1920x1080; do
  echo "== frame-parallel throughput $res"
  $PY benchmarks/cpu_comparison.py resources --backends $MC --threads 1 --cpus $PCORES --mode parallel \
    --workers 6 --frames-npz work/frames/$res.npz --passes 3 --rounds 3 --output $R/throughput_6workers_$res.json
  $PY benchmarks/cpu_comparison.py resources --backends napa,rapidtag --threads 6 --cpus $PCORES --mode parallel \
    --workers 6 --frames-npz work/frames/$res.npz --passes 3 --rounds 3 --output $R/throughput_6workers_rapidtag_$res.json
done
echo "== done"
