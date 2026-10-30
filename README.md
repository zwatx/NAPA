# NAPA

NAPA is a fast AprilTag detector that skips most of the pixels in each frame. Give it a greyscale frame and it returns
tag IDs and sub-pixel corners, plus poses with covariances if you give it a camera calibration. It reads every
AprilTag family, with tag36h11, the FRC family, as the default.

```bash
python -m pip install napa-vision
```

## Compared with other detectors

These results use the **full fixed 3,000-frame corpus**, with 5,521 expected tags at 640×480, 1280×720 and
1920×1080: all 20 scenarios, 50 independent renders per scenario/resolution, seed 2468. NAPA32 (the default,
`min_tag_px=32`) and NAPA16 (`min_tag_px=16`) are normal backend options in the **same run on the same frames**
as all 13 competitor configurations. No detector-dependent frame selection is used.

Every backend is configured for one worker on the same verified P-core of an i7-12700H. Detectors are constructed
once, warmed per image and timed ten times in shuffled order. Percentiles pool all 30,000 timed calls per backend.
Rendering, scoring and initialization are excluded; required internal work, input copies and result conversion
remain timed. The current native source was frozen before the run. This does not describe a new PyPI release.

| Detector / configuration | p50 ms | p95 ms | p99 ms | CPU mean ms | Found / tags | Wrong IDs | Dup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NAPA32 (default) | 0.255 | 1.479 | 2.081 | 0.405 | 5513/5521 | 1 | 0 |
| NAPA16 (tuned) | 0.741 | 5.204 | 6.518 | 1.297 | 5519/5521 | 1 | 0 |
| WPILib AprilTag d2 | 11.306 | 41.374 | 48.317 | 14.816 | 4759/5521 | 0 | 0 |
| AprilTag d2 | 14.990 | 51.563 | 59.852 | 19.010 | 5503/5521 | 3 | 0 |
| AprilTag d1 | 57.010 | 204.199 | 233.535 | 72.147 | 5517/5521 | 3 | 0 |
| Locus standard | 8.695 | 26.021 | 34.121 | 10.857 | 5393/5521 | 1 | 0 |
| Locus d2 | 4.287 | 11.562 | 14.093 | 5.148 | 5419/5521 | 1 | 0 |
| OpenCV ArUco | 14.745 | 90.758 | 152.710 | 26.333 | 5486/5521 | 1 | 0 |
| OpenCV SUBPIX | 14.773 | 90.632 | 153.459 | 26.364 | 5486/5521 | 1 | 0 |
| ArUco3 full | 7.960 | 30.161 | 67.950 | 11.333 | 5402/5521 | 1 | 0 |
| ArUco3 min≈64 px | 2.787 | 10.137 | 15.263 | 3.811 | 4415/5521 | 0 | 0 |
| ArUco3 min≈32 px | 3.715 | 14.699 | 21.352 | 5.360 | 5308/5521 | 1 | 0 |
| ArUco Nano | 1.925 | 8.574 | 13.881 | 2.757 | 5517/5521 | 0 | 0 |
| RapidTag | 4.682 | 20.446 | 39.907 | 6.801 | 4167/5521 | 19 | 0 |
| RapidTag SUBPIX | 4.698 | 20.462 | 40.067 | 6.806 | 4167/5521 | 19 | 0 |

Both NAPA options have lower pooled main-corpus p50/p95/p99 than every competitor configuration. NAPA32 has lower per-frame medians on every main-corpus image versus every competitor. NAPA16 slower-frame counts are Locus d2: 1; ArUco3 min≈64 px: 18; ArUco3 min≈32 px: 3; ArUco Nano: 24. These are observed comparisons, not significance tests; near-ties remain included.

Each speedup below is competitor time ÷ NAPA time, from the ten-sample median on each of the same 3,000 frames.
Values below one are NAPA losses; the near-tie is retained without a statistical-significance claim.

| Competitor | Vs NAPA32: median (minimum) | NAPA32 slower / 3000 | Vs NAPA16: median (minimum) | NAPA16 slower / 3000 |
| --- | ---: | ---: | ---: | ---: |
| WPILib AprilTag d2 | 24.98× (3.27607×) | 0 | 12.09× (1.54548×) | 0 |
| AprilTag d2 | 33.59× (3.75824×) | 0 | 16.37× (2.90373×) | 0 |
| AprilTag d1 | 127.56× (9.65688×) | 0 | 63.61× (6.58941×) | 0 |
| Locus standard | 27.07× (7.04917×) | 0 | 10.36× (1.99490×) | 0 |
| Locus d2 | 13.79× (3.26014×) | 0 | 6.12× (0.99987×) | 1 |
| OpenCV ArUco | 45.86× (6.62928×) | 0 | 16.54× (2.27181×) | 0 |
| OpenCV SUBPIX | 45.67× (6.65748×) | 0 | 16.60× (2.23070×) | 0 |
| ArUco3 full | 26.13× (6.12180×) | 0 | 10.32× (2.09818×) | 0 |
| ArUco3 min≈64 px | 9.61× (1.96210×) | 0 | 3.55× (0.70896×) | 18 |
| ArUco3 min≈32 px | 12.30× (2.78592×) | 0 | 4.53× (0.74182×) | 3 |
| ArUco Nano | 7.06× (1.83068×) | 0 | 2.52× (0.83909×) | 24 |
| RapidTag | 16.61× (4.59048×) | 0 | 6.36× (1.50660×) | 0 |
| RapidTag SUBPIX | 16.51× (4.61710×) | 0 | 6.36× (1.50410×) | 0 |

NAPA16 misses two low-contrast tags and reports one wrong ID. NAPA32 misses eight tags and reports one wrong ID;
its latency advantage comes with lower recall. Nano misses four tags and reports no wrong IDs. On common tags,
NAPA's median/p95 corner errors are lower than upstream AprilTag d1, but its maximum is worse: 3.57 versus 2.35 px
for NAPA16, and 3.39 versus 2.16 px for NAPA32 on their respective common tags. Both NAPA settings retain
full-resolution corner refinement.

The same run also includes **600 stress frames** and **900 range frames**, reported separately rather than pooled
into the headline: NAPA16 finds 4,577/4,577 stress tags and 900/900 range tags, with no wrong IDs in either group.
NAPA32 finds 4,390/4,577 and 748/900 respectively. All peer results in these tables are newly measured on the same
frames, not reused from an earlier run.

AprilTag d1/d2 means decimation one/two; Locus d2 is its standard profile at decimation two. OpenCV and RapidTag
defaults do not refine corners; SUBPIX variants are separate. ArUco3 minimum-size labels account for half-scale
detection. The [full report](benchmarks/results/2026-10-30_cpu_full_corpus/report.md) records exact settings, corner comparisons, stress/range results,
CPU time, thermal/load variation, raw data and the protocol recorded before timing.

The [benchmark guide](benchmarks/README.md) has per-scene and range tables. Older
[isolated resource measurements](benchmarks/results/2026-10-29_cpu_optimization/report.md#cpu-and-working-memory)
use separate 96-frame sets; this new interleaved run does not measure isolated RSS. These are finite synthetic
measurements on one host. [Score your own camera footage](docs/measuring.md) before relying on a specific result.

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
