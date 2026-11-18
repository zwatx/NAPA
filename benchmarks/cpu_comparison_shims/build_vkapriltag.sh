#!/bin/bash
# build_vkapriltag.sh OUT_DIR VKAPRILTAG_PREFIX VULKAN_HEADERS_INCLUDE
#
# Builds OUT_DIR/vkapriltag_worker, the optional vkapriltag backend, against a released vkapriltag package (its .deb
# can be unpacked with `dpkg-deb -x`, so nothing is installed). Needs a Vulkan loader and device at run time.
set -euo pipefail
out=$(realpath -m "$1"); prefix=$(realpath "$2"); headers=$(realpath "$3")
here=$(dirname "$(realpath "$0")")
mkdir -p "$out"
c++ -O3 -DNDEBUG -std=c++20 -no-pie -I"$prefix/include" -I"$prefix/include/apriltag" -I"$headers" \
    -o "$out/vkapriltag_worker" "$here/vkapriltag_worker.cpp" "$prefix/lib/libvkapriltag.a" \
    -L"$prefix/lib" -lapriltag -l:libvulkan.so.1 -Wl,-rpath,"$prefix/lib"
echo "$out/vkapriltag_worker"
