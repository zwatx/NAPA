# NAPA

NAPA is an AprilTag detector. It samples a sparse grid of pixels to find likely tags, then does full-resolution work
only on the tags themselves. Give it a frame and it returns tag IDs and sub-pixel corners. Give it a camera
calibration and it also returns poses, each with a covariance.

NAPA reads all nine AprilTag families. The default is tag36h11, the family FRC has used since 2024. Detection needs
only NumPy. Poses and calibration need OpenCV.

```bash
python -m pip install napa-vision          # detection
python -m pip install "napa-vision[headless]"  # detection + poses
```

Requires Python 3.11 or newer. Full documentation is in the
[repository](https://github.com/zwatx/NAPA). This page is the short version, with absolute links, because PyPI cannot
follow the repository's relative ones.

## Quick start

```python
from napa_vision import NapaDetector

detector = NapaDetector()
for tag in detector.detect(frame):  # uint8 greyscale, BGR, or BGRA
    print(tag.tag_id, tag.corners)
```

For poses:

```python
from napa_vision import VisionProcessor, load_calibration_data

processor = VisionProcessor(
    calibration=load_calibration_data("camera.json"),
    tag_size_m=0.1651,
    tag_layout=layout,  # {tag_id: field_to_tag}, for a camera pose on the field
)
result = processor.process(frame, timestamp=capture_time)
```

Code written for `pupil_apriltags` or `dt-apriltags` keeps working with one import changed:

```python
from napa_vision.compat import Detector  # was: from pupil_apriltags import Detector
```

## How it compares

Median time per frame on one Raspberry Pi 5 core, from the [November 14
run](https://github.com/zwatx/NAPA/blob/feat/consolidated-detector-review/benchmarks/results/2026-11-14_auto_split/report.md):

| Dataset | Frames | Tags matched, or detections returned | Median ms |
| --- | --- | --- | --- |
| Robot-camera captures | 297 | 684 returned (no labels) | 1.226 |
| FRC footage | 188 | 346 returned (no labels) | 1.668 |
| Locus AprilGrid renders | 200 | 2,768 of 2,916 matched | 5.743 |
| ICRA forward renders | 50 | 4,258 of 7,700 matched | 9.882 |

The first two rows count detections returned, not recall. The rendered rows are checked against labelled corners.

On a laptop P-core, NAPA found 10,952 of 10,998 tags in 4,500 synthetic frames, with 1 wrong ID. For detected tags,
median worst-corner error was 0.017 px; p95 was 0.089 px.

The [November 11
comparison](https://github.com/zwatx/NAPA/blob/feat/consolidated-detector-review/benchmarks/results/2026-11-11_full_benchmark/report.md)
used the same 297 robot-camera frames on one Pi 5 core, with an older NAPA build and the recall preset:

| Detector / setting | Median ms | p99 ms |
| --- | --- | --- |
| NAPA recall preset (16, split, shadows 24) | 6.631 | 7.810 |
| AprilTag 3, decimate 2 | 16.716 | 19.015 |
| AprilTag 3, full resolution | 57.468 | 62.643 |
| AprilTag, WPILib defaults | 13.809 | 15.710 |
| ArUco Nano | 14.646 | 15.310 |

The November 14 settings were `min_tag_px=22`, `split_touching="auto"`, shadows off, and `max_detections=2048` (the API
default is 64). The two runs used different builds. The [benchmark
guide](https://github.com/zwatx/NAPA/blob/feat/consolidated-detector-review/benchmarks/README.md#latest-results) has the
full method and per-dataset results.

## Documentation

- [Examples](https://github.com/zwatx/NAPA/blob/master/examples/README.md). Runnable scripts, starting with one that
  needs no camera.
- [Detection](https://github.com/zwatx/NAPA/blob/master/docs/detection.md). Choosing settings, tuning, families,
  printing tags, and switching from `pupil_apriltags`.
- [Poses and filters](https://github.com/zwatx/NAPA/blob/master/docs/poses.md). Metric poses, covariances, WPILib
  pose estimators, and rolling-shutter cameras.
- [Camera calibration](https://github.com/zwatx/NAPA/blob/master/docs/calibration.md).
- [Stereo rigs](https://github.com/zwatx/NAPA/blob/master/docs/stereo.md). One pose from two calibrated cameras.
- [Platforms](https://github.com/zwatx/NAPA/blob/master/docs/platforms.md). The C and C++ API, Android and Meta
  Quest, and building from source.
- [Measuring on your own footage](https://github.com/zwatx/NAPA/blob/master/docs/measuring.md).
- [How NAPA works](https://github.com/zwatx/NAPA/blob/master/docs/how-it-works.md). The pipeline, obstructions, and
  limits.
- [API reference](https://github.com/zwatx/NAPA/blob/master/docs/api.md). Every class, method, and field.

## Command-line tools

| Command | Purpose |
| --- | --- |
| `napa-tag` | Write tags to print, as SVG at an exact printed size or as PNG |
| `napa-calibrate` | Measure camera intrinsics from checkerboard photographs |
| `napa-benchmark` | Score NAPA on your own labeled images, with optional recall, precision, and latency gates |

## License

Copyright 2026 Zachary Wilson. NAPA is open source under the
[GNU General Public License v3.0](https://github.com/zwatx/NAPA/blob/master/LICENSE). Anyone can use, modify, and
share it, including in other GPL projects such as PhotonVision. Software that includes NAPA and is shared with others
must also be released under the GPL and must credit Zachary Wilson as NAPA's author
([NOTICE](https://github.com/zwatx/NAPA/blob/master/NOTICE)). For a closed-source product, ask about a commercial
license through [the repository](https://github.com/zwatx/NAPA).

The tag families' codewords come from [AprilTag](https://github.com/AprilRobotics/apriltag) under its BSD license.
See [THIRD_PARTY_NOTICES](https://github.com/zwatx/NAPA/blob/master/THIRD_PARTY_NOTICES).
