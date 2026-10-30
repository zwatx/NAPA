# NAPA

NAPA is a fast AprilTag detector that skips most of the pixels in each frame. Give it a greyscale frame and it returns
tag IDs and sub-pixel corners, plus poses with covariances if you give it a camera calibration. It reads every
AprilTag family, with tag36h11, the FRC family, as the default.

```bash
python -m pip install napa-vision
```

## Compared with other detectors

These results compare NAPA with AprilTag 3 as WPILib runs it (decimate 2 with WPILib's defaults), upstream AprilTag 3
at decimate 2 and at decimate 1 (full resolution), [Locus](https://github.com/NoeFontana/locus-tag), OpenCV's ArUco
module, [ArUco Nano](https://github.com/rmsalinas/aruco_nano), and [RapidTag](https://github.com/SujithChristopher/rapidtag),
each at its own defaults. The benchmark is 3,000 rendered frames holding 5,521 tags, each with exact ground truth,
and every detector ran single-threaded on the same CPU core.

| | WPILib AprilTag, decimate 2 | AprilTag, decimate 2 | AprilTag, decimate 1 | Locus | OpenCV ArUco | ArUco Nano | RapidTag | NAPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tags found | 86.2% | 99.7% | 99.9% | 97.7% | 99.4% | 99.9% | 75.5% | 99.8% |
| Smallest tag found every time ([range test](benchmarks/README.md#range)) | 80 px | 24 px | 14 px | 20 px | 18 px¹ | 16 px | — | 22 px |
| Median error of a tag's worst corner | 0.099 px | 0.043 px | 0.042 px | 0.044 px | 0.987 px | 0.044 px | 1.000 px | 0.021 px |
| Wrong IDs in 3,000 frames | 0 | 3 | 3 | 1 | 1 | 0 | 19 | 1 |

¹ Except one 96 px tag.

WPILib's build keeps only candidate regions of 300 pixels or more (upstream keeps 24), so at the same decimation it
drops small and steeply tilted tags that upstream AprilTag finds.

NAPA was faster than every one of them on every frame:

| Resolution | WPILib AprilTag, decimate 2 | AprilTag, decimate 2 | AprilTag, decimate 1 | Locus | OpenCV ArUco | ArUco Nano | RapidTag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 640×480 | 17× (3.8×) | 24× (4.4×) | 88× (11×) | 19× (6.6×) | 45× (6.6×) | 4.7× (1.8×) | 10× (4.6×) |
| 1280×720 | 37× (4.9×) | 47× (7.3×) | 199× (19×) | 31× (7.4×) | 45× (9.9×) | 6.9× (2.2×) | 17× (5.7×) |
| 1920×1080 | 72× (5.0×) | 89× (9.7×) | 359× (21×) | 44× (7.3×) | 44× (9.9×) | 8.8× (2.4×) | 23× (5.9×) |
| All frames | 25× (3.8×) | 33× (4.4×) | 125× (11×) | 27× (6.6×) | 44× (6.6×) | 6.9× (1.8×) | 16× (4.6×) |

Each cell is NAPA's speedup on the median frame, then on the frame where it was closest.

A tag's size in the image falls with distance, so the smallest tag found suggests relative range: under this
test's conditions (light noise, up to 15° of tilt), NAPA's default would see about 3.6× as far as WPILib's AprilTag
and slightly farther than upstream AprilTag at decimate 2. Upstream at decimate 1, Locus, OpenCV, and ArUco Nano find
smaller tags than NAPA's default does. Setting `min_tag_px=16` found every tag down to 12 px in a separate run; on
the median frame it took two to four times NAPA's default time, and much longer on very noisy frames.

NAPA's one wrong ID was a tag-shaped decoy whose random pattern fell within 2 bits of a real code. On tag16h5, the
2023 FRC family, AprilTag at its default two-bit correction reported 581 wrong IDs in the family benchmark, and NAPA
reported none.

The frames are synthetic. To check NAPA on your own camera, [score it on your own footage](docs/measuring.md). The
[benchmark guide](benchmarks/README.md) has the results for every scene, resolution, and tag family, and the
[detector comparison](benchmarks/results/2026-10-28_cpu_comparison/report.md) covers multicore runs, memory, and the
other libraries' faster settings.

The [native-core optimization follow-up](benchmarks/results/2026-10-29_cpu_optimization/report.md) measures the
updated source at `min_tag_px=16`, with raw timings, CPU and memory measurements, full synthetic quality results,
and the remaining per-image, tail-latency and corner-error losses. The tables above describe the earlier run.

## How it works

![NAPA on one frame, stage by stage: it samples every third pixel, skips flat tiles, turns dark samples into runs and blobs, drops shapes that can't be tags, and fits edges and decodes only the tags, at full resolution.](docs/napa-pipeline.webp)

NAPA finds likely tags on a sparse grid of pixels and only does full-resolution work on the tags themselves. [How NAPA
works](docs/how-it-works.md) explains each step.

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

- [Detection](docs/detection.md): tuning, tag families, printing tags, and switching from `pupil_apriltags`
- [Poses and filters](docs/poses.md): metric poses, coordinate frames, and WPILib pose estimators
- [Camera calibration](docs/calibration.md)
- [Stereo rigs](docs/stereo.md): one pose from two calibrated cameras
- [Platforms](docs/platforms.md): C and C++, Android and Meta Quest, and building from source
- [Measuring on your own footage](docs/measuring.md)
- [How NAPA works](docs/how-it-works.md): the pipeline, obstructions, and limits
- [API reference](docs/api.md): every class, method, and field

## Command-line tools

| Command | Purpose |
| --- | --- |
| `napa-tag` | Write tags to print: SVG at an exact printed size, or PNG |
| `napa-calibrate` | Measure camera intrinsics from checkerboard photographs |
| `napa-benchmark` | Score NAPA on your own labeled images, with optional recall, precision, and latency gates |

## Development

```bash
python -m pip install -e ".[headless,dev,benchmark]"
ruff format --check . && ruff check .
pytest --cov=napa_vision
```

## License

Copyright © 2026 Zachary Wilson. NAPA is free for noncommercial use under the [PolyForm Noncommercial License
1.0.0](LICENSE): personal projects, research, education, student robotics teams, and other noncommercial organizations
can use, modify, and share it. Copies, forks, and works based on NAPA must include the license and its notices
crediting Zachary Wilson. Commercial use, including selling NAPA or products built on it, needs a separate license.
Ask through [the repository](https://github.com/zwatx/NAPA).

The tag families' codewords come from [AprilTag](https://github.com/AprilRobotics/apriltag) under its BSD license. See
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
