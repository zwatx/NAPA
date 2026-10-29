# Platforms

`pip install napa-vision` installs a prebuilt wheel on Linux (glibc; x86-64, ARM64), macOS (Apple silicon, Intel),
Windows x64, and Android (ARM64, x86-64). That covers a Raspberry Pi or Orange Pi running a 64-bit OS. Anywhere else,
pip compiles the C core from source, which needs a C99 compiler. Without one, the install fails instead of leaving you
without a detector.

## C and C++

`src/napa_vision/_native/napa_core.h` is the whole C API: create a detector, call `napa_detect` on a greyscale buffer
with a stride, and destroy the detector. `napa_create()` reads tag36h11, `napa_create_family("tag25h9")` reads another
family, and `napa_family_info()` describes each. The library needs nothing beyond the C standard library and `libm`.

Build and install it with CMake, without Python:

```bash
cmake -S . -B build && cmake --build build && cmake --install build --prefix /usr/local
```

```cmake
find_package(napa REQUIRED)
target_link_libraries(your_app PRIVATE napa::napa)
```

The build is static by default. Pass `-DBUILD_SHARED_LIBS=ON` for a shared library. `examples/c/detect_pgm.c` (built
with `-DNAPA_BUILD_EXAMPLES=ON`) detects tags in a PGM image and prints them.

## Android and Meta Quest

Each [GitHub release](https://github.com/zwatx/NAPA/releases) has the core built for Android ARM64 (`arm64-v8a`, which
includes Meta Quest headsets) as `napa-<version>-android-arm64-v8a.zip`. It holds `lib/libnapa.so`,
`include/napa_core.h`, and a CMake package. It needs Android 7.0 (API 24) or newer and supports 16 KB memory pages.

- **NDK apps:** point CMake at the unzipped folder (`-Dnapa_DIR=<folder>/lib/cmake/napa`), then `find_package(napa)`
  and link `napa::napa`. Package `libnapa.so` with the app: Gradle picks it up from `src/main/jniLibs/arm64-v8a/`.
- **Unity:** put `libnapa.so` in `Assets/Plugins/Android/arm64-v8a/`, and call it with `[DllImport("napa")]`,
  following `napa_core.h`.
- **Python on Android**, for example with Chaquopy: `pip install napa-vision` finds the `android_24_arm64_v8a` wheel
  on PyPI. PyPI has no Android NumPy, so install NumPy from your Python distribution's own package index.

To build the Android library yourself, use the NDK's CMake toolchain:

```bash
cmake -S . -B build-android -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=24 -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON -DBUILD_SHARED_LIBS=ON
cmake --build build-android && cmake --install build-android --prefix napa-android --strip
```

## Using your own build

Set `NAPA_CORE_LIBRARY` to the path of a core library you built, and the Python package loads it instead of the
bundled one.

## Building and releasing

Python loads the core with `ctypes` rather than as an extension module, so each wheel is tagged `py3-none-<platform>`
and serves every supported Python version (3.11 and newer).

```bash
python -m pip install -e '.[headless,dev,benchmark]'
ruff format --check . && ruff check .
pytest --cov=napa_vision
python -m build && python -m twine check dist/*
```

The Wheels workflow builds and tests wheels with cibuildwheel on native Linux x86-64 and ARM64, macOS ARM and Intel,
and Windows x64 runners. Android wheels are cross-compiled with the NDK, and the x86-64 one is tested on an emulator
(`tests/test_mobile.py`). Run by hand, the workflow uploads the wheels and sdist as artifacts. On a `v*` tag, it also
publishes them to PyPI through trusted publishing and creates the GitHub release with the Android library attached.

CI runs on Python 3.11–3.14 and on OpenCV 4.7 through the current release. It builds the C core on Linux, macOS, and
Windows, builds it again with CMake and runs the C example, runs a fresh wheel's detector, including BGR conversion,
without OpenCV installed, and smoke-tests the command-line tools.
