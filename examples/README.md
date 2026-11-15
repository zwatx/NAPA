# Examples

Five small programs. Run them from the repository root with Python 3.11 or newer. The Python examples use only the
public API, so they also work against a released `pip install napa-vision` with the same extras.

| Example | What it does | Install |
| --- | --- | --- |
| [quickstart.py](python/quickstart.py) | Generates a tag and detects it. No camera, no image files, no OpenCV. | `python -m pip install -e .` |
| [detect_image.py](python/detect_image.py) | Detects tags in a photo, prints JSON, can save an annotated copy | `python -m pip install -e ".[headless]"` |
| [detect_video.py](python/detect_video.py) | Detects tags in every frame of a webcam or recording, reusing one detector | `.[headless]`, or `.[gui]` for `--display` |
| [estimate_pose.py](python/estimate_pose.py) | Turns detections into calibrated poses, with both candidates, ambiguity, and covariance | `python -m pip install -e ".[headless]"` |
| [detect_pgm.c](c/detect_pgm.c) | Calls the native C API, without Python or OpenCV | A C99 compiler and CMake |

The `gui` and `headless` extras install conflicting OpenCV packages. Install only one per environment. Pick `gui`
if you want windows.

## Start without hardware

```bash
python -m pip install -e .
python examples/python/quickstart.py
```

The script draws tag36h11 ID 7 on a white 640x480 frame, detects it, and prints the family, ID, corrected bits,
decision margin, and four corners. Coordinates are (x, y) in pixels, with (0, 0) at the centre of the top-left
pixel. Corner order follows the tag's orientation, so corner 0 is not always the top-left one in the image.

## Detect tags in an image

Generate a test PNG first, then try your own photo:

```bash
python -m pip install -e ".[headless]"
napa-tag 7 --format png --cell-px 32 --output work/example-tags
python examples/python/detect_image.py work/example-tags/tag36h11-007.png --output work/detected.png
python examples/python/detect_image.py photo.jpg --min-tag-px 16
```

Output is a JSON array. Each entry has the tag's family, ID, corners, corrected bits (`hamming`), and decision
margin. It prints `[]` when it finds no tags.

`--output` saves a copy with each tag outlined and labelled with its ID. It opens no window, and the output
directory must already exist. For other families, pass `--family`, for example `--family tag16h5`. See
[Tag families](../docs/detection.md#tag-families).

## Use a webcam or recording

```bash
python -m pip install -e ".[gui]"
python examples/python/detect_video.py --camera 0 --display
python examples/python/detect_video.py --video recording.mp4 --display --min-tag-px 16
```

Press q or Escape in the window to stop, or Ctrl+C in the terminal. The script releases the camera and the detector
either way. Recordings run as fast as the detector can process them, not at their original frame rate.

On a coprocessor without a desktop, install `.[headless]` and drop `--display`:

```bash
python examples/python/detect_video.py --video recording.mp4 > work/detections.jsonl
python examples/python/detect_video.py --camera 0 --max-frames 300 > work/camera-detections.jsonl
```

Each output line is a JSON object with a zero-based `frame_index`, a `detection_ms`, and the frame's `tags` with
their pixel corners. `detection_ms` times only `detector.detect(frame)`. Capture, output, and display are outside
it.

Neither field is an exposure timestamp. If you feed these detections to a pose filter, take timestamps from your
capture system. The script does not set a camera resolution, so the camera uses its default.

## Estimate poses with your calibration

First [calibrate your camera](../docs/calibration.md). Then take a photo at the calibrated resolution and run:

```bash
python examples/python/estimate_pose.py photo.jpg --calibration camera.json --tag-size-m 0.1651
```

For `--tag-size-m`, measure the tag-size square: the black square on classic families such as tag36h11, or the inner
white square on reversed-border families. 0.1651 m (165.1 mm, 6.5 in) is the FRC tag size. Replace it with your own
measurement.

The script refuses calibrations that were not measured, and images whose resolution does not match the calibration.
Pass the raw photo. The script applies the calibration's distortion coefficients itself, so undistorting first would
correct the image twice.

For each tag, the JSON lists up to two planar pose candidates, lowest reprojection error first. Translation is in
metres along the camera's forward, left, and up axes. Each 4x4 `camera_to_tag` matrix places the tag in that camera
frame. Covariance is ordered `[x, y, z, rx, ry, rz]`, in metres and radians.

Check `ambiguity` before you trust solution 0. Near 1, both candidates fit about equally well and either may be the
real pose. Each candidate's covariance is already widened to cover the other. To drop those measurements instead,
discard single-tag poses with `ambiguity > 0.2`. See [Ambiguity](../docs/poses.md#ambiguity).

Tags that were detected but failed the solver's quality checks appear in `unsolved_tag_ids`. With no detections,
both lists are empty.

For camera or robot localization with a field layout, see [Poses and filters](../docs/poses.md). For paired cameras,
see [Stereo rigs](../docs/stereo.md).

## Use the C API

```bash
cmake -S . -B build -DNAPA_BUILD_EXAMPLES=ON
cmake --build build --config Release
./build/napa_detect_pgm frame.pgm
```

Multi-configuration generators such as Visual Studio put the executable in `build/Release/`. The example reads 8-bit
binary PGM (`P5`) files whose header has no comments. To convert a photo, install the `headless` extra and run:

```bash
python -c "import cv2; image = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE); assert image is not None; assert cv2.imwrite('frame.pgm', image)"
```

See [Platforms](../docs/platforms.md#c-and-c) for linking NAPA into your own C or C++ project.
