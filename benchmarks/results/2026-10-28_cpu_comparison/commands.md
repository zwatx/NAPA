# Reproducing the 2026-10-28 CPU comparison

All third-party code is built under `work/` (untracked); nothing is installed system-wide.

```bash
# Sources, at the commits these runs measured
mkdir -p work/thirdparty && cd work/thirdparty
git clone https://github.com/AprilRobotics/apriltag.git && git -C apriltag checkout b7c0ebe9aa20f82ec7a828579004f9e706bfecd9
git clone https://github.com/rmsalinas/aruco_nano.git && git -C aruco_nano checkout c99989bcab795ffa65d9e12df9de15f27677a835
git clone --depth 1 --branch 4.14.0 https://github.com/opencv/opencv.git opencv-4.14.0
cd ../..

# Isolated venv: same NumPy/OpenCV as the repo venv, released Locus and RapidTag wheels, build tools
python3 -m venv work/venv
work/venv/bin/pip install numpy==2.5.4 opencv-python-headless==5.0.0.93 locus-tag==0.10.0 rapidtag==0.1.12 \
  cmake ninja pytest psutil
export PATH=$PWD/work/venv/bin:$PATH

# Upstream AprilTag 3 (Release, -O3)
cmake -S work/thirdparty/apriltag -B work/build/apriltag -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON_WRAPPER=OFF -DCMAKE_INSTALL_PREFIX=$PWD/work/install/apriltag
cmake --build work/build/apriltag && cmake --install work/build/apriltag

# Minimal OpenCV 4.14 C++ for ArUco Nano (OpenCV 5 removed calib3d, which aruco_nano.h includes)
cmake -S work/thirdparty/opencv-4.14.0 -B work/build/opencv -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/work/install/opencv -DBUILD_LIST=core,imgproc,calib3d,features2d,flann,objdetect \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF \
  -DBUILD_JAVA=OFF -DWITH_IPP=ON -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_OPENCL=OFF -DWITH_CUDA=OFF \
  -DWITH_VULKAN=OFF -DWITH_EIGEN=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
  -DWITH_V4L=OFF -DWITH_PROTOBUF=OFF -DWITH_ADE=OFF -DWITH_ITT=OFF
cmake --build work/build/opencv -j6 && cmake --install work/build/opencv

# Shims and NAPA's core, rebuilt from the checked-out source (setup.py-equivalent flags)
benchmarks/cpu_comparison_shims/build.sh work/build/shims work/install/apriltag work/thirdparty/apriltag \
  work/install/opencv work/thirdparty/aruco_nano
# setup.py's flags on top of this Python's CFLAGS
mkdir -p work/build/napa
cc -fno-strict-overflow -DNDEBUG -g -O2 -Wall -Wextra -fPIC -O3 -std=c99 -fvisibility=hidden -fno-wrapv -shared \
  -Isrc/napa_vision/_native -o work/build/napa/libnapa_core.so src/napa_vision/_native/napa_core.c -lm
export NAPA_CORE_LIBRARY=$PWD/work/build/napa/libnapa_core.so

# Validation, then the runs (each script runs its jobs one at a time; run scripts one at a time)
R=benchmarks/results/2026-10-28_cpu_comparison
ALL=napa,napa-mtp16,napa-mtp22,napa-mtp43,napa-mtp64,apriltag3-d2,apriltag3-d1,locus,locus-d2,opencv,opencv-subpix,opencv-aruco3,opencv-aruco3-min32,opencv-aruco3-half16,aruco-nano,rapidtag,rapidtag-subpix
taskset -c 2 work/venv/bin/python benchmarks/cpu_comparison.py validate --backends $ALL --output $R/validate.json
benchmarks/results/2026-10-28_cpu_comparison/run_single_thread.sh
benchmarks/results/2026-10-28_cpu_comparison/run_resources_multicore.sh   # real photos (its resource steps are superseded)
benchmarks/results/2026-10-28_cpu_comparison/run_after_review.sh          # outlier retime, ArUco3 half16, all resources
benchmarks/results/2026-10-28_cpu_comparison/run_range_sweep.sh           # range sweep on the current 18 sizes
work/venv/bin/python benchmarks/cpu_comparison.py real-agreement benchmarks/results/2026-10-28_cpu_comparison/real_photos_latency.json \
  --output benchmarks/results/2026-10-28_cpu_comparison/real_photos_agreement.json

# Tables
work/venv/bin/python benchmarks/cpu_comparison.py report benchmarks/results/2026-10-28_cpu_comparison/single_thread_latency.json \
  --groups all-corpus,all-stress,stress-tiny,stress-heavy-noise,stress-dense,stress-clutter-noisy,range
work/venv/bin/python benchmarks/cpu_comparison.py resources-report benchmarks/results/2026-10-28_cpu_comparison/resources_*.json \
  benchmarks/results/2026-10-28_cpu_comparison/throughput_6workers_*.json
NAPA_CORE_LIBRARY=$NAPA_CORE_LIBRARY PYTHONPATH=src work/venv/bin/python -m pytest tests/test_cpu_comparison.py
```

`outlier_frames.txt` (17 frames: per rival the 2 closest per-frame ratios to NAPA, plus NAPA's 4 slowest):
Runs replaced after the review are described in report.md (Corrections).
