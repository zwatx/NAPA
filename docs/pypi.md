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

Single-core performance on an Intel i7-12700H, across 3,000 labelled synthetic images:

| Detector | Median ms | p95 ms | Tags found | Wrong IDs |
| --- | --- | --- | --- | --- |
| NAPA defaults (`min_tag_px=22`) | 0.315 | 2.052 | 5519/5521 | 1 |
| NAPA, `min_tag_px=16` | 0.569 | 3.885 | 5521/5521 | 1 |
| AprilTag, WPILib defaults | 10.747 | 39.018 | 4759/5521 | 0 |
| AprilTag, decimate 2 | 14.050 | 48.315 | 5503/5521 | 3 |
| AprilTag, full resolution | 54.595 | 193.457 | 5517/5521 | 3 |
| ArUco Nano | 1.803 | 8.020 | 5517/5521 | 0 |

NAPA defaults are `min_tag_px=22`, `split_touching="auto"`, shadows off, and `max_detections=64`. On matched tags
in the main corpus, its median/p95 worst-corner error was 0.017/0.080 px. Matches are scored by ID; corner error
uses the best corner ordering and counts only matched tags.

Including the 600 stress and 900 range frames, NAPA defaults found 10,952 of 10,998 tags, with one wrong ID.
The [full report](https://github.com/zwatx/NAPA/blob/feat/consolidated-detector-review/benchmarks/results/2026-11-15_merged_cpu_comparison/report.md) has all 16 settings, per-scene
results, and the timing method. The [measurement audit](https://github.com/zwatx/NAPA/blob/feat/consolidated-detector-review/benchmarks/results/2026-11-15_merged_cpu_comparison/audit.md)
covers frequency and temperature variation during the run.

The [benchmark guide](https://github.com/zwatx/NAPA/blob/feat/consolidated-detector-review/benchmarks/README.md#latest-results) has the full method.
For results on your camera and hardware, [measure your own footage](https://github.com/zwatx/NAPA/blob/feat/consolidated-detector-review/docs/measuring.md).

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
