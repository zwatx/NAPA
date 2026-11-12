# Platforms

`pip install napa-vision` installs a prebuilt wheel on:

- Linux, glibc, x86-64 and ARM64. That covers a Raspberry Pi or Orange Pi running a 64-bit OS.
- macOS, Apple silicon and Intel.
- Windows x64.
- Android, ARM64 and x86-64.

Anywhere else, pip compiles the C core from source. That needs a C99 compiler. Without one the install fails. It
does not leave you with a package that has no detector.

## C and C++

The whole C API is in `src/napa_vision/_native/napa_core.h`. Create a detector, call `napa_detect` on a greyscale
buffer with a stride, and destroy the detector. `napa_create()` reads tag36h11. `napa_create_family("tag25h9")`
reads another family. `napa_family_count`, `napa_family_name`, and `napa_family_info` describe the nine families,
and `napa_family_code` returns a code for drawing tags. The library needs nothing beyond the C standard library and
`libm`.

Initialise `napa_params` with designated initialisers or zeros, as `examples/c/detect_pgm.c` does. Fields added in
later versions (such as `split_touching`) then start at 0, which is off. Rebuild against the new header when you
update the library.

The experimental options have their own entry points, so the original ABI never changes:

| Entry point | Adds |
| --- | --- |
| `napa_detect` | Ordinary detection |
| `napa_detect_shadows` | Shadow recovery at full sensitivity. Same arguments and error codes as `napa_detect`. |
| `napa_detect_shadows_configured` | A final `shadow_min_tag_px` argument. 0 keeps full sensitivity. Otherwise a value from `params->min_tag_px` to 30000 trades sensitivity for speed. |
| `napa_detect_soft`, `napa_detect_options` | Soft decoding, and with `NAPA_OPTION_ALIGN_CORNERS`, corner alignment. tag36h11 only. Soft-decoded tags carry `hamming == NAPA_SOFT_DECODED`. |
| `napa_set_camera_gamma` | Gamma-aware corner fitting for every later call on the detector. tag36h11 only. |
| `napa_set_tag_shape` | Tall or wide search sampling for every later call on the detector. |

Calling one of the newer symbols needs a newer library. From Python, asking for a feature an old library lacks
raises `NativeLibraryError` and tells you to rebuild. The header documents each function's return values. In short,
detect calls return the number of detections written, -1 for invalid input or parameters, and -2 when allocation
fails. A detector handle is not thread-safe. Use one per thread or camera stream.

Build and install with CMake, without Python:

```bash
cmake -S . -B build && cmake --build build && cmake --install build --prefix /usr/local
```

```cmake
find_package(napa REQUIRED)
target_link_libraries(your_app PRIVATE napa::napa)
```

The build is static by default. Pass `-DBUILD_SHARED_LIBS=ON` for a shared library. Build with
`-DNAPA_BUILD_EXAMPLES=ON` to get `examples/c/detect_pgm.c` as a runnable program that detects tags in a PGM image
and prints them.

## Android and Meta Quest

Each [GitHub release](https://github.com/zwatx/NAPA/releases) carries the core built for Android ARM64 (`arm64-v8a`,
which includes Meta Quest headsets) as `napa-<version>-android-arm64-v8a.zip`. It holds `lib/libnapa.so`,
`include/napa_core.h`, and a CMake package. It needs Android 7.0 (API 24) or newer and supports 16 KB memory pages.

- **NDK apps.** Point CMake at the unzipped folder with `-Dnapa_DIR=<folder>/lib/cmake/napa`, then
  `find_package(napa)` and link `napa::napa`. Package `libnapa.so` with the app. Gradle picks it up from
  `src/main/jniLibs/arm64-v8a/`.
- **Unity.** Put `libnapa.so` in `Assets/Plugins/Android/arm64-v8a/` and call it with `[DllImport("napa")]`,
  following `napa_core.h`.
- **Python on Android**, for example with Chaquopy. `pip install napa-vision` finds the `android_24_arm64_v8a`
  wheel on PyPI. PyPI has no Android NumPy, so install NumPy from your Python distribution's own package index.

To build the Android library yourself, use the NDK's CMake toolchain:

```bash
cmake -S . -B build-android -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=24 -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON -DBUILD_SHARED_LIBS=ON
cmake --build build-android && cmake --install build-android --prefix napa-android --strip
```

## Using your own build of the core

Set `NAPA_CORE_LIBRARY` to the path of a core library you built, and the Python package loads it instead of the
bundled one. An explicit path that fails to load raises an error. It never falls back silently, so a benchmark
cannot quietly time the wrong build.

## Building and releasing

Python loads the core with `ctypes` rather than as an extension module. Each wheel is therefore tagged
`py3-none-<platform>` and serves every supported Python version, 3.11 and newer.

```bash
python -m pip install -e '.[headless,dev,benchmark]'
ruff format --check . && ruff check .
pytest --cov=napa_vision
python -m build && python -m twine check dist/*
```

The Wheels workflow builds and tests wheels with cibuildwheel on native Linux x86-64 and ARM64, macOS ARM and Intel,
and Windows x64 runners. Android wheels are cross-compiled with the NDK, and the x86-64 one is tested on an emulator
(`tests/test_mobile.py`). Run by hand, the workflow uploads the wheels and sdist as artifacts. On a `v*` tag it also
publishes them to PyPI through trusted publishing and creates the GitHub release with the Android library attached.

CI runs on Python 3.11 to 3.14 and on OpenCV 4.7 through the current release. It builds the C core on Linux, macOS,
and Windows, builds it again with CMake and runs the C example, runs a fresh wheel's detector (including BGR
conversion) without OpenCV installed, and smoke-tests the command-line tools.
