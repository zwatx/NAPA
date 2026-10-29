#!/bin/bash
# Real photos (interleaved, CPU 2), then isolated per-backend processes: single-thread on CPU 2, 6 threads per
# detector and 6 frame-parallel workers on one logical CPU per P-core (0,2,4,6,8,10). Strictly sequential.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=benchmarks/results/2026-10-28_cpu_comparison
PY=work/venv/bin/python
export NAPA_CORE_LIBRARY=$PWD/work/build/napa/libnapa_core.so
ALL=napa,napa-mtp16,napa-mtp22,napa-mtp43,napa-mtp64,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,aruco-nano,rapidtag,rapidtag-subpix
MC=napa,napa-mtp16,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3-min32,aruco-nano,rapidtag
PCORES=0,2,4,6,8,10
REAL=$(ls work/thirdparty/apriltag/test/data/*.jpg)
benchmarks/cpu_comparison_shims/monitor.sh $R/monitor_resources_multicore.tsv 0 1 2 3 4 6 8 10 &
MON=$!
trap 'kill $MON' EXIT

echo "== real photos (unannotated), interleaved, CPU 2"
taskset -c 2 $PY benchmarks/cpu_comparison.py latency --backends $ALL --threads 1 --frames 0 --stress-frames 0 \
  --real-images $REAL --repetitions 30 --warmup 3 --output $R/real_photos_latency.json

echo "== frame sets"
mkdir -p work/frames
for res in 640x480 1280x720 1920x1080; do
  $PY benchmarks/cpu_comparison.py corpus --resolutions $res --frames 4 --stress-frames 4 --output work/frames/$res.npz
done

for res in 640x480 1280x720 1920x1080; do
  echo "== single-thread resources $res"
  $PY benchmarks/cpu_comparison.py resources --backends $ALL --threads 1 --cpus 2 --mode serial \
    --frames-npz work/frames/$res.npz --passes 3 --rounds 3 --output $R/resources_1thread_$res.json
done
for res in 640x480 1280x720 1920x1080; do
  echo "== multicore intra-frame latency $res (6 threads per detector; NAPA has no intra-frame threads)"
  $PY benchmarks/cpu_comparison.py resources --backends $MC --threads 6 --cpus $PCORES --mode serial \
    --frames-npz work/frames/$res.npz --passes 3 --rounds 3 --output $R/resources_6threads_$res.json
done
for res in 640x480 1280x720 1920x1080; do
  echo "== frame-parallel throughput $res (6 workers x 1 thread; RapidTag shares one 6-thread Rayon pool)"
  $PY benchmarks/cpu_comparison.py resources --backends ${MC%,rapidtag} --threads 1 --cpus $PCORES --mode parallel \
    --workers 6 --frames-npz work/frames/$res.npz --passes 3 --rounds 3 --output $R/throughput_6workers_$res.json
  $PY benchmarks/cpu_comparison.py resources --backends napa,rapidtag --threads 6 --cpus $PCORES --mode parallel \
    --workers 6 --frames-npz work/frames/$res.npz --passes 3 --rounds 3 --output $R/throughput_6workers_rapidtag_$res.json
done
echo "== done"
