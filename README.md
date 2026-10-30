# NAPA

NAPA is a fast AprilTag detector that skips most of the pixels in each frame. Give it a greyscale frame and it returns
tag IDs and sub-pixel corners, plus poses with covariances if you give it a camera calibration. It reads every
AprilTag family, with tag36h11, the FRC family, as the default.

```bash
python -m pip install napa-vision
```

## Compared with other detectors

The current native source was compared with 13 CPU detector configurations on one performance core of an Intel
Core i7-12700H. These are **149 deliberately selected diagnostic frames**, including earlier losses to Nano and
Locus, at 640×480, 1280×720 and 1920×1080. They are not a random camera sample or the earlier 3,000-frame run.
Each detector is constructed once, warmed once per frame, then timed three times in shuffled order on identical
grayscale tag36h11 images. The table pools all detector calls; rendering, scoring and initialization are excluded.

**NAPA32 is the unchanged default (`min_tag_px=32`); NAPA16 is tuned (`min_tag_px=16`).** Both retain full-resolution
corner refinement. AprilTag d1/d2 denote decimation one/two; Locus d2 is its standard profile at decimation two.
OpenCV and RapidTag defaults do not refine corners; SUBPIX variants are separate. ArUco3 minimum sizes below are
effective image sizes, accounting for half-scale detection. See the [configuration details and pinned versions](benchmarks/results/2026-10-29_cpu_optimization/report.md#scope-and-timing).

| Detector / configuration | p50 ms | p95 ms | p99 ms | Tags found | Wrong IDs | Duplicates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NAPA32 (default) | 0.419 | 1.938 | 2.577 | 310/319 | 0 | 0 |
| NAPA16 (tuned) | 1.115 | 6.442 | 10.545 | 319/319 | 0 | 0 |
| WPILib AprilTag d2 | 14.201 | 42.795 | 51.072 | 259/319 | 0 | 0 |
| AprilTag d2 | 19.505 | 57.670 | 68.842 | 311/319 | 0 | 0 |
| AprilTag d1 | 72.189 | 219.958 | 258.445 | 319/319 | 0 | 0 |
| Locus standard | 12.701 | 32.652 | 36.665 | 307/319 | 0 | 0 |
| Locus d2 | 5.849 | 12.740 | 14.821 | 305/319 | 0 | 0 |
| OpenCV ArUco | 26.821 | 118.516 | 150.296 | 311/319 | 0 | 0 |
| OpenCV SUBPIX | 26.578 | 117.878 | 142.303 | 311/319 | 0 | 0 |
| ArUco3 full | 14.964 | 93.649 | 109.496 | 296/319 | 0 | 0 |
| ArUco3 min≈64 px | 4.765 | 17.983 | 21.857 | 244/319 | 0 | 0 |
| ArUco3 min≈32 px | 6.668 | 20.635 | 25.568 | 291/319 | 0 | 0 |
| ArUco Nano | 3.144 | 19.145 | 22.132 | 318/319 | 0 | 0 |
| RapidTag | 9.462 | 84.623 | 92.825 | 230/319 | 8 | 0 |
| RapidTag SUBPIX | 9.547 | 86.672 | 92.618 | 230/319 | 8 | 0 |

NAPA16 has lower pooled p50/p95/p99 than every competitor configuration. It does not win every image. Each speedup below
is computed from paired per-frame median times (competitor ÷ NAPA16); values below one are NAPA losses.

| Competitor | Median speedup of NAPA16 | Smallest speedup | NAPA16 slower frames / 149 |
| --- | ---: | ---: | ---: |
| WPILib AprilTag d2 | 10.52× | 1.713× | 0 |
| AprilTag d2 | 16.10× | 3.157× | 0 |
| AprilTag d1 | 58.94× | 7.004× | 0 |
| Locus standard | 7.10× | 2.190× | 0 |
| Locus d2 | 3.45× | 0.992× | 1 |
| OpenCV ArUco | 19.78× | 3.048× | 0 |
| OpenCV SUBPIX | 19.98× | 3.053× | 0 |
| ArUco3 full | 12.47× | 2.669× | 0 |
| ArUco3 min≈64 px | 3.56× | 1.066× | 0 |
| ArUco3 min≈32 px | 4.36× | 1.203× | 0 |
| ArUco Nano | 2.71× | 0.741× | 5 |
| RapidTag | 7.63× | 1.583× | 0 |
| RapidTag SUBPIX | 7.56× | 1.925× | 0 |

A separate 20-repeat check of 28 selected outlier frames confirmed five losses to Nano and one to Locus d2.

The full **1,620-frame synthetic quality replay** found **3,486/3,486 tags** with NAPA16 and **3,334/3,486** with
NAPA32, with zero wrong IDs or duplicates for both. This is a different scope from the 149-frame table above.
In the 30-tags-per-size range sweep, NAPA16 found every tag at all tested sizes from 12–96 px. Full-quality results
for unchanged competitors are reused from the verified earlier corpus, as documented in the report.

With one or six physical P-cores available to each backend, NAPA16 also had lower mean wall time, CPU time and
working memory than every tested competitor. NAPA itself still uses one execution thread per frame. Values below
average two isolated process rounds on identical 96-frame mixed sets at each resolution. Working memory is peak
RSS above the constructed detector, including warm-up and detection; it is not total process RSS.

| P-cores allowed | Resolution | NAPA16 mean ms | Fastest peer mean ms | NAPA16 working MiB | Smallest peer working MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 640×480 | 0.479 | ArUco Nano: 1.075 | 0.84 | Locus d2: 2.38 |
| 1 | 1280×720 | 1.214 | ArUco Nano: 2.661 | 2.71 | ArUco Nano: 4.80 |
| 1 | 1920×1080 | 2.732 | ArUco Nano: 6.922 | 6.45 | ArUco Nano: 7.63 |
| 6 | 640×480 | 0.487 | ArUco3 min≈64 px: 1.054 | 0.83 | ArUco Nano: 3.18 |
| 6 | 1280×720 | 1.166 | ArUco3 min≈64 px: 2.054 | 2.71 | ArUco Nano: 4.95 |
| 6 | 1920×1080 | 2.950 | ArUco3 min≈64 px: 3.775 | 6.45 | ArUco Nano: 7.72 |

At 1080p with six cores available, NAPA16's mean-of-round p99 was 10.755 ms, behind ArUco3 min≈32 px at 10.031 ms,
ArUco3 min≈64 px at 10.072 ms and Locus d2 at 10.377 ms; those peers found fewer tags. On common synthetic tags,
NAPA16 had lower median/p95 corner
error than upstream AprilTag d1, but a worse maximum: 3.57 versus 2.35 px. Three unannotated real JPEGs provide
only timing and detector agreement, not an accuracy result.

These are finite synthetic measurements on one machine. [Score your own camera footage](docs/measuring.md).
The [benchmark guide](benchmarks/README.md) has per-scene and range tables; the
[full report](benchmarks/results/2026-10-29_cpu_optimization/report.md) links the raw data, provenance and
reproduction commands. [Earlier measurements](benchmarks/results/2026-10-29_cpu_detector_matrix/matrix.json)
remain available unchanged.

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
