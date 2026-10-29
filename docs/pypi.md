# NAPA

**Not Another Pixel Algorithm**: NAPA skips most of the pixels in each frame.

NAPA is a fast AprilTag detector. Give it a greyscale frame and it returns tag IDs and sub-pixel
corners, plus poses with covariances if you give it a camera calibration. It reads every AprilTag family, with
tag36h11, the FRC family, as the default.

```bash
python -m pip install napa-vision
```

## Compared with AprilTag

These results compare NAPA with upstream AprilTag 3 at decimate 2 (WPILib's default) and decimate 1 (full
resolution, PhotonVision's default). The benchmark is 3,000 rendered frames holding 5,521 tags, each with exact
ground truth, and every detector ran single-threaded on the same CPU core.

| | AprilTag, decimate 2 | AprilTag, decimate 1 | NAPA |
| --- | ---: | ---: | ---: |
| Tags found | 86.2% | 99.4% | 99.8% |
| Smallest tag found every time ([range test](https://github.com/zwatx/NAPA/blob/master/benchmarks/README.md#range)) | 80 px | 36 px | 22 px |
| Median error of a tag's worst corner | 0.099 px | 0.098 px | 0.021 px |
| Wrong IDs in 3,000 frames | 0 | 3 | 1 |

NAPA was faster than both on every frame:

| Resolution | vs. decimate 2, median (slowest frame) | vs. decimate 1, median (slowest frame) |
| --- | ---: | ---: |
| 640×480 | 13× (3.6×) | 62× (12×) |
| 1280×720 | 30× (4.2×) | 142× (14×) |
| 1920×1080 | 60× (4.5×) | 273× (13×) |
| All frames | 19× (3.6×) | 87× (12×) |

Range scales with the inverse of the smallest tag found, so on the same camera NAPA sees about 3.6× as far as
decimate 2 and 1.6× as far as decimate 1.

NAPA's one wrong ID was a tag-shaped decoy whose random pattern fell within 2 bits of a real code. On tag16h5, the
2023 FRC family, AprilTag at its default two-bit correction reported 581 wrong IDs in the family benchmark, and NAPA
reported none.

The frames are synthetic. To check NAPA on your own camera, [score it on your own footage](https://github.com/zwatx/NAPA/blob/master/docs/measuring.md). The
[benchmark guide](https://github.com/zwatx/NAPA/blob/master/benchmarks/README.md) has the results for every scene, resolution, and tag family.

## How it works

NAPA finds likely tags on a sparse grid of pixels and only does full-resolution work on the tags themselves.
[How NAPA works](https://github.com/zwatx/NAPA/blob/master/docs/how-it-works.md) explains each step.

## Quick start

Requires Python 3.11 or newer.

```python
from napa_vision import NapaDetector

detector = NapaDetector()
for tag in detector.detect(image):  # uint8 greyscale, BGR, or BGRA
    print(tag.tag_id, tag.corners)
```

Code written for `pupil_apriltags` or `dt-apriltags` works after changing one import:

```python
from napa_vision.compat import Detector  # was: from pupil_apriltags import Detector
```

For poses, install OpenCV with the `headless` extra and give NAPA a calibration:

```bash
python -m pip install "napa-vision[headless]"
```

```python
from napa_vision import VisionProcessor, load_calibration_data

processor = VisionProcessor(calibration=load_calibration_data("camera.json"), tag_size_m=0.1651)
result = processor.process(frame, timestamp=capture_time)
```

## Documentation

- [Detection](https://github.com/zwatx/NAPA/blob/master/docs/detection.md): tuning, tag families, printing tags, and switching from `pupil_apriltags`
- [Poses and filters](https://github.com/zwatx/NAPA/blob/master/docs/poses.md): metric poses, coordinate frames, and WPILib pose estimators
- [Camera calibration](https://github.com/zwatx/NAPA/blob/master/docs/calibration.md)
- [Stereo rigs](https://github.com/zwatx/NAPA/blob/master/docs/stereo.md): one pose from two calibrated cameras
- [Platforms](https://github.com/zwatx/NAPA/blob/master/docs/platforms.md): C and C++, Android and Meta Quest, and building from source
- [Measuring on your own footage](https://github.com/zwatx/NAPA/blob/master/docs/measuring.md)
- [How NAPA works](https://github.com/zwatx/NAPA/blob/master/docs/how-it-works.md): the pipeline, obstructions, and limits
- [API reference](https://github.com/zwatx/NAPA/blob/master/docs/api.md): every class, method, and field

## Command-line tools

| Command | Purpose |
| --- | --- |
| `napa-tag` | Write tags to print: SVG at an exact printed size, or PNG |
| `napa-calibrate` | Measure camera intrinsics from checkerboard photographs |
| `napa-benchmark` | Score NAPA on your own labeled images, with optional recall, precision, and latency gates |

## License

Copyright © 2026 Zachary Wilson. NAPA is free for noncommercial use under the
[PolyForm Noncommercial License 1.0.0](https://github.com/zwatx/NAPA/blob/master/LICENSE): personal projects, research, education, student robotics teams,
and other noncommercial organizations can use, modify, and share it. Copies, forks, and works based on NAPA must
include the license and its notices crediting Zachary Wilson. Commercial use, including selling NAPA or products
built on it, needs a separate license. Ask through [the repository](https://github.com/zwatx/NAPA).

The tag families' codewords come from [AprilTag](https://github.com/AprilRobotics/apriltag) under its BSD license. See
[THIRD_PARTY_NOTICES](https://github.com/zwatx/NAPA/blob/master/THIRD_PARTY_NOTICES).
