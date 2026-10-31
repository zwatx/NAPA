# NAPA

NAPA is a fast AprilTag detector that skips most of the pixels in each frame. Give it a greyscale frame and it returns
tag IDs and sub-pixel corners, plus poses with covariances if you give it a camera calibration. It reads every
AprilTag family, with tag36h11, the FRC family, as the default.

```bash
python -m pip install napa-vision
```

## Compared with other detectors

The benchmark is 3,000 rendered frames holding 5,521 tags at 640×480, 1280×720, and 1920×1080, each with exact ground
truth. Every detector ran single-threaded on the same CPU core, on the same frames.

| Detector | Speed vs. WPILib's defaults (median frame) | Tags found | Worst-corner error (median) | Wrong IDs |
| --- | ---: | ---: | ---: | ---: |
| **NAPA** | **25×** | **99.9%** | **0.020 px** | 1 |
| AprilTag, WPILib's defaults | 1× | 86.2% | 0.099 px | 0 |
| AprilTag, decimate 2 | 0.79× | 99.7% | 0.043 px | 3 |
| AprilTag, full resolution | 0.21× | 99.9% | 0.042 px | 3 |
| Locus | 1.3× | 97.7% | 0.044 px | 1 |
| Locus, decimate 2 | 2.4× | 98.2% | 0.044 px | 1 |
| OpenCV ArUco, subpixel | 0.55× | 99.4% | 0.289 px | 1 |
| ArUco3 | 1.2× | 97.8% | 0.986 px | 1 |
| ArUco3, tags over 64 px | 3.1× | 80.0% | 0.282 px | 0 |
| ArUco3, tags over 32 px | 2.2× | 96.1% | 0.284 px | 1 |
| ArUco Nano | 4.8× | 99.9% | 0.044 px | 0 |
| RapidTag, subpixel | 2.0× | 75.5% | 0.294 px | 19 |

NAPA was faster than every other detector on every one of the 3,000 frames. The closest was ArUco Nano, and NAPA's
smallest lead over it on any frame was 1.8×. NAPA's one wrong ID was a tag-shaped decoy with a random pattern. OpenCV
ArUco and RapidTag without subpixel refinement run at the same speed as above, with corners about a pixel off.

NAPA's lead over AprilTag grows with resolution:

| Resolution | vs. WPILib's defaults, median (slowest frame) | vs. full resolution, median (slowest frame) |
| --- | ---: | ---: |
| 640×480 | 18× (3.3×) | 93× (9.7×) |
| 1280×720 | 36× (5.1×) | 198× (19×) |
| 1920×1080 | 74× (5.1×) | 375× (20×) |

**Range.** A tag looks smaller the farther away it is, so the smallest tag a detector finds sets how far it can see. In
the [range sweep](benchmarks/README.md#range), NAPA found every tag from 22 px up, AprilTag at WPILib's defaults from 80
px, and AprilTag at full resolution from 14 px. On the same camera, NAPA sees about 3.6× as far as WPILib's defaults.

**Smaller tags.** `min_tag_px=16` found every tag in the range sweep, down to 12 px, and 5,519 of the 5,521 tags above.
It takes about 3× as long as the default, which still makes it 12× faster than WPILib's defaults. ArUco Nano, ArUco3,
and Locus at decimate 2 were faster than it on 43 of the 3,000 frames.

**Corners.** NAPA's median worst-corner error is less than half of full-resolution AprilTag's and about a fifth of
WPILib's defaults, and on 95% of tags it was within 0.09 px. Heavy motion blur is the exception: NAPA's largest error
was 3.4 px, against 2.2 px for full-resolution AprilTag on the same tags.

**Stress.** A separate set of 600 harder frames has tags under 32 px, very heavy noise, and up to 80 tags per frame.
NAPA found 96% of those tags with its default settings, and all of them with `min_tag_px=16`.

**Calibration boards.** In an AprilGrid board a black square touches each tag at its corners, which hides most of the
board from the default detector. With `split_touching=True` and `min_tag_px=16`, NAPA found 96.9% of the tags in the
independent [Locus](benchmarks/results/2026-10-31_split_touching/report.md) AprilGrid renders, against 98.0% for
AprilTag at full resolution, and 96.2% under lens distortion, against 96.1%, in about a sixth of the time. Fisheye
boards remain harder: 80% against AprilTag's 94%. Splitting roughly doubles NAPA's time (up to 4× on noisy frames), so
it is off by default.

On tag16h5, the 2023 FRC family, AprilTag at its default two-bit correction reported 581 wrong IDs in the [family
benchmark](benchmarks/README.md#tag-families), and NAPA reported none.

The frames are synthetic, and the timings come from one laptop CPU (Intel i7-12700H). To check NAPA on your own camera,
[score it on your own footage](docs/measuring.md). The [full
report](benchmarks/results/2026-10-30_cpu_full_corpus/report.md) has every configuration's settings, latency
percentiles, and raw data, and the [benchmark guide](benchmarks/README.md) has results for every scene.

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

- [Runnable examples](examples/README.md): a hardware-free quick start, images, webcam and video, calibrated poses, and the C API
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

Copyright © 2026 Zachary Wilson. NAPA is open source under the [GNU General Public License v3.0](LICENSE):
anyone can use, modify, and share it, including in other GPL projects such as PhotonVision. Software that
includes NAPA and is shared with others must also be released under the GPL and credit Zachary Wilson as NAPA's
author ([NOTICE](NOTICE)). For a closed-source product, get a commercial license: ask through
[the repository](https://github.com/zwatx/NAPA).

The tag families' codewords come from [AprilTag](https://github.com/AprilRobotics/apriltag) under its BSD license. See
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
