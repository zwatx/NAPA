#!/bin/bash
# Fixed shared corpus: 3000 ordinary frames + 600 stress frames + 900 range frames.
# NAPA32/default and NAPA16 are ordinary backend options in the same interleaved run.
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=${NAPA_FULL_OUTPUT_DIR:-benchmarks/results/2026-10-30_cpu_full_corpus}
mkdir -p "$R" work/full-corpus
if [[ -e "$R/matrix.json" || -e "$R/matrix.csv" ]]; then
  echo "Refusing to overwrite completed measurements; choose a fresh NAPA_FULL_OUTPUT_DIR." >&2
  exit 1
fi
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export RAPIDTAG_CORES=2
export NAPA_CORE_LIBRARY=$PWD/work/full-corpus/libnapa_core.so
# Compile the frozen source with setup.py-equivalent release flags; no architecture-specific targets.
cc -fno-strict-overflow -DNDEBUG -g -O2 -Wall -Wextra -fPIC -O3 -std=c99 \
  -fvisibility=hidden -fno-wrapv -shared -Isrc/napa_vision/_native \
  -o "$NAPA_CORE_LIBRARY" src/napa_vision/_native/napa_core.c -lm
work/venv/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
p = json.loads(Path('benchmarks/results/2026-10-30_cpu_full_corpus/protocol.json').read_text())
for name, sha in p['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
assert hashlib.sha256(Path(os.environ['NAPA_CORE_LIBRARY']).read_bytes()).hexdigest() == p['napa_binary_sha256']
print('Frozen source, benchmark code and native binary hashes verified.', flush=True)
PY
ALL=napa,napa-mtp16,wpilib-d2,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,opencv-aruco3-half16,aruco-nano,rapidtag,rapidtag-subpix
benchmarks/cpu_comparison_shims/monitor.sh "$R/monitor.tsv" 0 1 2 3 4 6 8 10 &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
taskset -c 2 work/venv/bin/python benchmarks/cpu_comparison.py latency \
  --backends "$ALL" --threads 1 --seed 2468 --order-seed 2468 \
  --resolutions 640x480,1280x720,1920x1080 \
  --frames 50 --stress-frames 50 --range-cases 50 \
  --repetitions 10 --warmup 1 --output "$R/matrix.json"
kill "$MON"
wait "$MON" || true
trap - EXIT
