# Napa

**Not Another Pixel Algorithm**: a fast AprilTag detector that skips most of the pixels in each frame.

Give Napa a frame and it returns tag IDs and sub-pixel corners, plus metric poses with covariances if you supply a
camera calibration. The detector is a small C library with no dependencies, and the Python package needs only numpy.
It opens no cameras, starts no threads, and publishes nothing, so a webcam, a Raspberry Pi, a Meta Quest, recorded
video, or a camera SDK can all feed it the same way.

```bash
python -m pip install napa-vision
```

## Why Napa

Compared with upstream AprilTag 3 at its default settings, both single-threaded on the same 3,000 synthetic frames
(5,521 tags with exact ground truth):

- **About 21× faster** on realistic, noisy frames (median per frame), and faster on every one of the 3,000 frames.
  At 1280×720, Napa's median call stays under about 0.5 ms; AprilTag's is 10–15 ms in most noisy scenes.
- **4× the range.** In a tag-size sweep, Napa found every tag down to 24 px across the black square, where AprilTag
  needed 96 px. A 6.5-inch FRC tag seen by a 1280×720, 70° camera is still 24 px across at 6.3 m.
- **More tags found:** 99.8%, against AprilTag's 86.2%.
- **About 5× more accurate corners:** a median worst-corner error of 0.020 px, against AprilTag's 0.099 px.
- **Every AprilTag family**, including tag16h5 without its flood of false IDs (581 for AprilTag, 0 for Napa in the
  family benchmark).

These are synthetic frames on one laptop CPU. Measure on your own camera and hardware before relying on a number:
`napa-benchmark` scores Napa on your own annotated images. [The README](https://github.com/zwatx/NAPA#readme) has
the full results, scene by scene, and explains how Napa gets its speed.

## Quick start

```python
from napa_vision import NapaDetector

detector = NapaDetector()  # tag36h11, the FRC family; or NapaDetector(family="tagStandard41h12")
for tag in detector.detect(image):  # uint8 grayscale, BGR, or BGRA
    print(tag.tag_id, tag.corners)
```

Each detection also has `tag.center` and `tag.homography`, as upstream AprilTag reports them. Use one detector per
camera stream. The native call releases the GIL, so separate streams can run on separate threads.

### Poses

Install the `headless` extra (or `gui`, for OpenCV with windows) for poses, calibration, and `napa-benchmark`:

```bash
python -m pip install "napa-vision[headless]"
```

```python
from napa_vision import VisionProcessor, load_calibration_data

processor = VisionProcessor(calibration=load_calibration_data("camera.json"), tag_size_m=0.1651)
result = processor.process(frame, timestamp=capture_time)
result.single_tag_poses  # both planar solutions for each tag, each with a 6×6 covariance
```

Pass a tag layout to also get the camera's pose in your world frame, solved jointly from every visible tag. Every
pose carries a covariance, so it can go straight into a Kalman filter, including WPILib's pose estimators.

### Switching from pupil_apriltags or dt-apriltags

Change the import and keep the rest of your code:

```python
from napa_vision.compat import Detector  # was: from pupil_apriltags import Detector

detector = Detector(families="tag36h11")
detections = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=0.1651)
```

Results keep upstream's corner order, pixel origin, and pose frames.

## Command-line tools

| Command | Purpose |
| --- | --- |
| `napa-tag` | Write tags to print, at an exact printed size |
| `napa-calibrate` | Measure camera intrinsics from checkerboard photographs |
| `napa-benchmark` | Score Napa on your own annotated images, with optional recall, precision, and latency gates |

## Platforms

Requires Python 3.11 or newer. Prebuilt wheels cover Linux (x86-64, ARM64), macOS (Apple silicon, Intel), Windows
x64, and Android (ARM64, x86-64); elsewhere, pip builds from source, which needs a C99 compiler. Other languages can
call the C library's plain C API directly, and C and C++ projects can build it with CMake. Android apps, including
Meta Quest ones, can use the prebuilt ARM64 library attached to each
[release](https://github.com/zwatx/NAPA/releases).

## Documentation

- [Guide](https://github.com/zwatx/NAPA/blob/master/docs/guide.md): tuning, poses, Kalman filters, coordinate
  conventions, calibration, and using Napa from C or Android
- [API reference](https://github.com/zwatx/NAPA/blob/master/docs/api.md): every class, method, and field
- [Benchmark guide](https://github.com/zwatx/NAPA/blob/master/benchmarks/README.md): how each test scene is built

## License

Copyright © 2026 Zachary Wilson. Napa is free for noncommercial use under the
[PolyForm Noncommercial License 1.0.0](https://github.com/zwatx/NAPA/blob/master/LICENSE): personal projects,
research, education, student robotics teams, and other noncommercial organizations can use, modify, and share it.
Copies, forks, and works based on Napa must include the license and its notices crediting Zachary Wilson.
Commercial use, including selling Napa or products built on it, needs a separate license; ask through
[the repository](https://github.com/zwatx/NAPA). The tag families' codewords come from
[AprilTag](https://github.com/AprilRobotics/apriltag) under its BSD license; see
[THIRD_PARTY_NOTICES](https://github.com/zwatx/NAPA/blob/master/THIRD_PARTY_NOTICES).
