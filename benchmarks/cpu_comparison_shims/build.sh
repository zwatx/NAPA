#!/bin/bash
# build.sh OUT_DIR APRILTAG_PREFIX APRILTAG_SRC OPENCV_PREFIX ARUCO_NANO_SRC
#
# Builds the two C ABI shims used by benchmarks/cpu_comparison.py:
#   OUT_DIR/libapriltag_shim.so   against an installed upstream AprilTag 3 (libapriltag.so + headers)
#   OUT_DIR/libaruco_nano_shim.so against an installed OpenCV 4.x C++ and aruco_nano.h
# Both use -O3 with no -march flags, like the libraries' own Release builds. rpaths point at the given
# prefixes, so nothing is installed system-wide.
set -euo pipefail
out=$(realpath -m "$1"); apriltag=$(realpath "$2"); apriltag_src=$(realpath "$3"); opencv=$(realpath "$4")
nano=$(realpath "$5")
here=$(dirname "$(realpath "$0")")
mkdir -p "$out"
atlib=$(dirname "$(find "$apriltag" -name 'libapriltag.so' | head -1)")
cc -O3 -DNDEBUG -std=gnu99 -fPIC -shared -I"$apriltag/include/apriltag" -I"$apriltag_src" \
   -o "$out/libapriltag_shim.so" "$here/apriltag_shim.c" -L"$atlib" -lapriltag -Wl,-rpath,"$atlib"
cvlib=$(dirname "$(find "$opencv" -name 'libopencv_core.so' | head -1)")
c++ -O3 -DNDEBUG -std=c++17 -fPIC -shared -I"$opencv/include/opencv4" -I"$nano" \
    -o "$out/libaruco_nano_shim.so" "$here/aruco_nano_shim.cpp" \
    -L"$cvlib" -lopencv_objdetect -lopencv_calib3d -lopencv_imgproc -lopencv_core -Wl,-rpath,"$cvlib"
echo "$out/libapriltag_shim.so"
echo "$out/libaruco_nano_shim.so"
