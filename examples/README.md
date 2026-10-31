# Runnable examples

Run these commands from the repository root with Python 3.11 or newer.

| Example | What it shows | Install |
| --- | --- | --- |
| [quickstart.py](python/quickstart.py) | Generate a tag and detect its ID and corners, without a camera or image files | `python -m pip install -e .` |
| [detect_image.py](python/detect_image.py) | Detect tags in a photo, print JSON, and save an annotated copy | `python -m pip install -e ".[headless]"` |
| [detect_video.py](python/detect_video.py) | Process a webcam or recording with one detector reused across frames | `python -m pip install -e ".[headless]"`; use `.[gui]` for `--display` |
| [estimate_pose.py](python/estimate_pose.py) | Read a camera calibration and report both pose candidates, ambiguity, and covariance | `python -m pip install -e ".[headless]"` |
| [detect_pgm.c](c/detect_pgm.c) | Use the native C API without Python or OpenCV | C99 compiler and CMake |

Use one OpenCV package in an environment: choose the `gui` extra instead of `headless` when you need windows.
All Python examples use the public NAPA API and can also run against `pip install napa-vision` with the same extras.

## Start without hardware

```bash
python -m pip install -e .
python examples/python/quickstart.py
```

This creates a white 640×480 frame containing tag36h11 ID 7, then prints its ID, decode quality, and four corners.
Coordinates are `(x, y)` in pixels, with `(0, 0)` at the centre of the top-left pixel. Corner indices follow the tag's
orientation, so corner 0 isn't necessarily the top-left of the image.

## Detect an image

Generate a PNG to try immediately, then replace it with your own photograph:

```bash
python -m pip install -e ".[headless]"
napa-tag 7 --format png --cell-px 32 --output work/example-tags
python examples/python/detect_image.py work/example-tags/tag36h11-007.png --output work/detected.png
python examples/python/detect_image.py photo.jpg --min-tag-px 16
```

The script prints a JSON array, including corners, corrected-bit count, and decision margin. No tags produces `[]`.
`--output` writes outlines and IDs without opening a window. Its parent directory must exist.
Use `--family tag16h5` (or any [supported family](../docs/detection.md#tag-families)) for other tags.

## Use a webcam or recording

```bash
python -m pip install -e ".[gui]"
python examples/python/detect_video.py --camera 0 --display
python examples/python/detect_video.py --video recording.mp4 --display --min-tag-px 16
```

Press **q** or **Escape** in the window, or **Ctrl+C** in the terminal. The capture and detector are released on exit.
The script processes recordings as fast as it can, without pacing playback.

For a coprocessor without a desktop, install `.[headless]` and leave off `--display`:

```bash
python examples/python/detect_video.py --video recording.mp4 > work/detections.jsonl
python examples/python/detect_video.py --camera 0 --max-frames 300 > work/camera-detections.jsonl
```

Each output line has a zero-based frame index, tags with their pixel corners, and detection time in milliseconds.
That time covers only `detector.detect(frame)`, excluding capture, JSON output, and display. Frame indices and
processing times aren't exposure timestamps; use your capture system's timestamps when feeding a robot pose filter.
The camera example requests the camera's default resolution.

## Estimate poses with your calibration

First [calibrate your camera](../docs/calibration.md), then take a photo at the calibrated resolution. Measure the tag's
black square for classic families such as tag36h11, or the inner white square for reversed-border families.

```bash
python examples/python/estimate_pose.py photo.jpg --calibration camera.json --tag-size-m 0.1651
```

The size above is 165.1 mm; use your tag's actual size. The script requires a measured calibration and rejects a
resolution mismatch. It uses the raw image and the calibration's distortion coefficients; don't undistort it first.

JSON output includes both planar pose candidates where available, best reprojection error first. Translation is in
metres along the camera's **forward, left, up** axes. Each 4×4 `camera_to_tag` matrix places the tag in that camera
frame. Covariance is ordered `[x, y, z, rx, ry, rz]`, with metres and radians. Ambiguity near 1 means the candidates fit
equally well; selecting solution 0 doesn't resolve that ambiguity. A detected tag whose pose fails the solver's
quality checks appears in `unsolved_tag_ids`. No detections produces empty pose and unsolved-ID lists.

For camera or robot localization using a field layout, see [poses and filters](../docs/poses.md); for paired cameras,
see [stereo rigs](../docs/stereo.md).

## Use the C API

```bash
cmake -S . -B build -DNAPA_BUILD_EXAMPLES=ON
cmake --build build --config Release
./build/napa_detect_pgm frame.pgm
```

On multi-configuration generators such as Visual Studio, the executable is under `build/Release/`.
The example accepts an 8-bit binary PGM (`P5`) with a simple header without comments. To convert an image after
installing the `headless` extra:

```bash
python -c "import cv2; image = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE); assert image is not None; assert cv2.imwrite('frame.pgm', image)"
```

See [platforms](../docs/platforms.md#c-and-c) for linking NAPA into your own C or C++ project.
