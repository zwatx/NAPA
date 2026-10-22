# Napa

**Not Another Pixel Algorithm**: Napa skips most of the pixels in each frame.

Napa is a fast AprilTag detector for any camera. It reads every AprilTag family, and `tag36h11`, the FRC family, by
default. Give it a frame and it returns tag IDs and sub-pixel
corners, plus poses if you supply a calibration. It opens no cameras, starts no threads, and publishes nothing, so a
webcam, a Raspberry Pi, a headset, recorded video, or a camera SDK can all feed it the same way.

The detector is a small C library with no dependencies, plus a Python binding that needs only numpy.

## Compared with standard AprilTag

"Standard AprilTag" here means the upstream AprilTag 3 library at its default settings (`quad_decimate=2`). Both
detectors ran single-threaded on one pinned core of an Intel Core Ultra 7 256V laptop (Windows 11, Ubuntu-like Linux
under WSL2, best-performance power mode), taking turns on the same grayscale frames. Earlier published results came
from a desktop Intel i7-12700H, so absolute times differ from those.

The benchmark has **3,000 distinct frames holding 5,521 tags**. Every frame is rendered separately, with random tag
IDs, positions, sizes, angles, 3D tilt, backgrounds, and camera degradations, and every tag has exact sub-pixel
ground truth. [The benchmark guide](benchmarks/README.md) describes each scene.

**Tags found.** Napa found **99.8%** of tags. Standard AprilTag found 86.2%, and AprilTag at full resolution
(`quad_decimate=1`) 99.4%. Napa finds the small tags that standard AprilTag's decimation loses, and in every one of
the 60 scene and resolution combinations it found at least as many tags as standard AprilTag.

**Range.** Napa sees tags from much farther away. A tag looks smaller the farther it is, so the smallest tag a
detector can find sets its range. A separate sweep rendered 50 tags at each of eleven sizes, from 12 to 96 px across
the black square, with light noise and up to 15° of tilt, and counted what each detector found:

| Smallest black square found | Standard AprilTag | AprilTag at full resolution | Napa |
| --- | ---: | ---: | ---: |
| Every tag | 96 px | 40 px | **24 px** |
| Most tags | 64 px (36 of 50) | 32 px (39 of 50) | **20 px (46 of 50)** |

That is **4× the range of standard AprilTag** and 1.7× the range of AprilTag at full resolution. For example, a
6.5-inch FRC tag seen by a 1280×720 camera with a 70° field of view is 24 px across at 6.3 m, where standard
AprilTag needs it within 1.6 m. To see farther still, lower `min_tag_px` (see [the guide](docs/guide.md#detection)).

**Speed.** Napa was faster on every one of the 3,000 frames:

- **Noise-free frames: 5.7× to 12.6× faster** (median per frame, by resolution). The slowest single frame was
  still 4.8× faster.
- **Realistic frames: about 21× faster** (median). Every other scene includes at least light sensor noise
  (σ ≈ 1.5 grey levels), which a real sensor always has. Standard AprilTag's time climbs steeply with noise: on a
  1280×720 frame its median is 1.35 ms with none, and 10–15 ms in most noisy scenes (overexposed and JPEG frames,
  whose clipping and compression smooth the noise, take 2–2.6 ms). Napa stays under about 0.5 ms. The slowest single
  frame here was 3.6× faster.
- Against AprilTag at full resolution, Napa was 87× faster (median) and at least 11.8× faster on every frame.

These are the 1280×720 results; the guide has every resolution:

| Scene, 1280×720 | Tags | AprilTag | Napa | Speedup (median) | Slowest frame | Tags found (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Noise-free | 104 | 1.35 ms | 0.16 ms | 8× | 6.9× | 104 → 104 |
| One tag | 50 | 13.1 ms | 0.20 ms | 65× | 46× | 50 → 50 |
| 2–12 tags | 354 | 9.64 ms | 0.49 ms | 20× | 8.0× | 354 → 354 |
| Small tags (32–53 px) | 120 | 11.6 ms | 0.29 ms | 41× | 11.6× | 0 → 120 |
| Large tags | 50 | 10.6 ms | 0.29 ms | 37× | 13.4× | 50 → 50 |
| Steep angle (40–65°) | 112 | 10.6 ms | 0.35 ms | 31× | 11.6× | 109 → 111 |
| Defocus blur | 89 | 10.2 ms | 0.32 ms | 32× | 9.2× | 89 → 89 |
| Motion blur | 94 | 10.2 ms | 0.38 ms | 17× | 9.1× | 94 → 94 |
| Heavy noise (σ 4–14) | 104 | 9.66 ms | 0.52 ms | 19× | 6.0× | 104 → 104 |
| Uneven lighting | 103 | 11.0 ms | 0.37 ms | 29× | 12.0× | 103 → 103 |
| Low contrast | 100 | 13.9 ms | 0.23 ms | 66× | 39× | 95 → 99 |
| Underexposed | 99 | 11.3 ms | 0.33 ms | 35× | 13.6× | 99 → 99 |
| Overexposed | 112 | 2.04 ms | 0.26 ms | 7× | 5.4× | 112 → 112 |
| JPEG artefacts | 100 | 2.58 ms | 0.25 ms | 8× | 4.2× | 100 → 100 |
| Busy background | 87 | 10.9 ms | 0.30 ms | 38× | 11.3× | 87 → 87 |
| At the frame edge | 105 | 10.8 ms | 0.32 ms | 35× | 12.2× | 103 → 105 |
| Dark background | 105 | 10.3 ms | 0.29 ms | 38× | 12.0× | 91 → 105 |
| No tag: empty / clutter / decoys | — | 9–15 ms | 0.12–0.31 ms | 19–101× | 5.1× | — |

Times are the median detector call. Speedups are per frame: AprilTag's time divided by Napa's on the same frame.
"Slowest frame" is the smallest of those ratios.

**Corners.** For each tag, the benchmark measured the worst corner's distance from the true corner. On the 4,754
tags both detectors found, Napa's median error was **0.020 px** against AprilTag's 0.099 px, and its 95th percentile
**0.088 px** against 0.303 px. Napa was more accurate at the median and 95th percentile in every scene. The exception
is the single worst case under heavy motion blur: with 11 px of blur on a tag about 50 px across, one of Napa's
corners was 2.5 px off, where AprilTag's worst in that scene was 0.98 px.

**False detections.** In 3,000 frames, Napa reported one wrong ID: a tag-shaped decoy whose random payload happened
to fall within 2 bits of a real code, the same tolerance standard AprilTag uses. Standard AprilTag reported none,
and full-resolution AprilTag 3. On a separate test of 80,000 such decoys, Napa reported none.

## Every tag family

Napa reads all nine of upstream AprilTag's families: tag36h11 (the default), tag16h5, tag25h9, tag36h10,
tagStandard41h12, tagStandard52h13, tagCircle21h7, tagCircle49h12, and tagCustom48h12. Pass the family when you create
a detector: `NapaDetector(family="tagStandard41h12")`. tag36h11's detection is unchanged by the other families: it
gives exactly the same results as before they were added.

A family-by-family run over the same 20 scenes (each with its own family's tags, spaced by their printed size) gave:

| Family | Tags | Found: AprilTag → Napa | False IDs: AprilTag → Napa | Median corner error: AprilTag → Napa | Speedup: median (slowest frame) |
| --- | ---: | ---: | ---: | ---: | ---: |
| tag36h11 | 289 | 288 → 287 | 0 → 0 | 0.097 → 0.022 px | 40× (7.9×) |
| tag16h5 | 282 | 282 → 280 | **581 → 0** | 0.107 → 0.022 px | 38× (5.5×) |
| tag25h9 | 287 | 286 → 286 | 3 → 0 | 0.098 → 0.021 px | 36× (7.2×) |
| tagStandard41h12 | 232 | 232 → 231 | 0 → 0 | 0.097 → 0.025 px | 42× (8.5×) |
| tagCircle21h7 | 231 | 231 → 230 | 9 → 0 | 0.097 → 0.025 px | 42× (9.3×) |

Frames were 640×480 and 1280×720. Speedups are per frame, from a separate timing pass over 80 frames per family on
one pinned core.

Every tag that only AprilTag found was small (40–60 px) and steeply tilted, where the white quiet zone narrows to a
pixel or two: either its narrow side was under `min_tag_px`, or a dark line in the background touched the tag and
joined its border (see [Limits](#limits)). On tagStandard52h13, tagCircle49h12, tagCustom48h12, and tag36h10 (which the reference
Python binding doesn't include), Napa found 99.2–99.6% of tags with no false IDs. Upstream's decode table for the
three big families needs several gigabytes, which is one more reason to prefer Napa for them: its tables take
megabytes.

tag16h5's 30 codes are only 5 bits apart, so a decoder that corrects two wrong bits, as AprilTag does by default,
accepts about one in four random tag-shaped patterns. Napa sets each family's default correction so that chance
matches stay as rare as they are for tag36h11, and adds a "soft" correction for families like tag16h5: a few bits
may still be corrected, but only if they're exactly the payload's least certain readings. [The API
reference](docs/api.md#tag-families) has the details.

These are synthetic frames on one laptop CPU. Measure on your own camera and hardware before relying on a specific
number: `napa-benchmark` scores Napa on your own annotated images (see [the guide](docs/guide.md)).

## Why it's faster

Standard AprilTag treats each frame as a large image-processing job. It thresholds every pixel, segments the whole
image into regions, and fits quadrilaterals to every region's boundary before it asks whether anything is a tag.
Napa asks that question as early and as cheaply as it can.

![Napa on one frame, stage by stage: the rake keeps every third pixel, flat tiles are skipped, dark samples become runs and blobs, shapes that can't be tags are dropped, and only the tags get edge fitting and decoding at full resolution.](docs/napa-pipeline.webp)

*One synthetic frame, stage by stage. Every overlay is computed from the frame by a port of Napa's pipeline.*

### 1. The rake: sample sparsely, with a guarantee

A tag's black border is one cell wide, and a tag is 8 cells across. Sample every few pixels and you're guaranteed to
land on the border of every tag above a minimum size, so skipping the pixels in between can't skip a tag.

Napa drags a "rake" across the frame, taking one sample every ¾ of the smallest cell in each direction. Blur greys a
cell's edges but leaves its core dark, and ¾-cell spacing always puts a sample in that core. For the default 32 px
minimum tag, that's one pixel in nine. Coarser copies of the rake (2×, 4×, 8×, …) catch larger tags; each is built
from the one before, not from the image, and together they add only a third again to the cost.

Each tag is handled at the level where it's between about 16 and 64 samples across. Below that, its one-cell white
margin can fall between samples and the tag merges with a dark background; above it, the middle of its border is
too far from any white to stand out. Every tag size has such a level.

### 2. String search on a matrix: skip what can't match

Fast string search (Boyer–Moore) doesn't compare every character. One cheap check can prove that no match starts in
the next several positions, so the search jumps past them.

Napa applies the same idea to the image. It splits the raked samples into small tiles and records each tile's
darkest and brightest value. A tile without contrast can't contain a tag's edge, so Napa skips it and does no work
on its pixels. The contrast needed rises automatically with sensor noise, measured from the frame itself and region
by region, so a noisy or textured part of the frame doesn't blind the rest. On a very noisy frame, each rake sample
averages the block of pixels it stands for instead, which divides the noise by the stride.

The same idea speeds up decoding. A tag's 36 payload bits must be matched against 587 codes in 4 rotations, allowing
up to 3 wrong bits. Any code within 3 bits of the payload must match at least one of four 9-bit blocks *exactly* (the
pigeonhole principle). So Napa looks those blocks up in a table and compares only the handful of codes that share
one, instead of all 2,348.

### 3. Old-school vector math

Every pass over the samples is written as whole-row vector operations, the classic Apple AltiVec/vDSP approach. One
SSE2 or NEON instruction compares 16 samples with their thresholds and packs the results into a bit mask. Runs of
dark samples then fall out of the masks with a count-trailing-zeros instruction per run, not a test per pixel.
Runs that touch from row to row are joined into blobs with union-find, so the cost follows the number of runs, not
the number of dark pixels. Gathering every third pixel uses a byte shuffle (SSSE3, chosen at run time) or NEON's
de-interleaving load.

### 4. Reject early, measure carefully

Each blob of plausible size gets a rough quadrilateral from its extreme points. A blob can't be a tag if it's darker
overall than any real code allows: every tag36h11 code has at least 10 white cells. Then Napa:

1. Fits each side to its edge at full resolution.
2. Decodes straight away with a loose tolerance, reading the border first. Text, texture, and checkerboards are
   thrown out here, before any expensive refinement.
3. Refines each side coarse to fine, then with a final pass that measures the edge every 1.5 px along it. Each
   measurement is the centre of the edge's gradient, which is exact for real, pixel-integrated edges. Where blur is
   wider than about half a cell, the edge is located instead where the image crosses halfway between the black and
   white on either side.
4. Intersects the four lines to get the corners, and decodes the payload.

Some images fool a plain edge fit, so Napa checks for them:

- **A grey quiet zone on a bright background** makes the margin look dark, so the first outline is the margin's.
  If the decode fails, Napa tries the outline's inner 8/10, where the black square would be.
- **A side fitted one cell out**, on the margin's outer edge, shows up as border cells that read white. Napa moves
  that side in by a cell and refits it.
- **Clipped (saturated) exposure** shrinks every black region by the same amount. The black border must be exactly
  one cell wide, so Napa measures that shrinkage on the tag itself and undoes it.

## Obstructions

Napa corrects up to 3 wrong payload bits (standard AprilTag defaults to 2), so a hand, cable, or glare spot over part
of a tag often still decodes. It accepts a third corrected bit only when the three wrong cells touch, as they do
under a real obstruction: a random pattern that happens to fall within three bits of a code has its errors
scattered, so this keeps chance matches out. In the benchmark's obstruction sweep (864 tags, each partly covered by
a white or grey patch), Napa recovered 357 tags and standard AprilTag 330.

No AprilTag detector can find a tag whose black border or corners are hidden, because the border is found before any
bits are read. Obstruction tolerance is best-effort, with no guaranteed percentage.

## Limits

- Tags smaller than `min_tag_px` (default 32 px across the black square) may be missed. Lower it to find smaller tags,
  at some cost in speed. A steeply tilted tag counts by its narrowest side.
- Tags need their white quiet zone, as printed tags have. A tag whose quiet zone runs straight into dark, busy
  texture can merge with it and be missed. Steep tilt makes this likelier, because it narrows the quiet zone: a small
  tag tilted 60° can join a dark line that touches it.
- Tags with less than `min_contrast` (default 20) grey levels between their black and white are skipped. Lower it for
  washed-out scenes, at a cost in speed on every frame.
- Motion blur longer than a cell (an eighth of the tag) is the hardest case for corner accuracy: most corners stay
  within a tenth of a pixel, but the worst can be off by a few pixels.
- Prebuilt wheels cover Linux (x86-64, ARM64), macOS (Apple silicon, Intel), and Windows x64. Elsewhere, pip builds
  from source, which needs a C99 compiler.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m pip install napa-vision              # the detector: needs only numpy
python -m pip install "napa-vision[headless]"  # adds OpenCV, for poses, calibration, and napa-benchmark
```

```python
from napa_vision import NapaDetector

detector = NapaDetector()
for tag in detector.detect(image):  # uint8 grayscale, BGR, or BGRA
    print(tag.tag_id, tag.corners)
```

Each detection also has `tag.center` and `tag.homography`, as upstream AprilTag reports them.

Already using `pupil_apriltags` or `dt-apriltags`? Change the import and keep the rest of your code:

```python
from napa_vision.compat import Detector  # was: from pupil_apriltags import Detector

detector = Detector(families="tag36h11")
detections = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=0.1651)
```

Results keep upstream's conventions (corner order, pixel origin, and pose frames), so code written against those
libraries works unchanged.

For metric poses, use `VisionProcessor(calibration=load_calibration_data("camera.json"), tag_size_m=...)`. Every pose
comes with a covariance, so it can go straight into a Kalman filter, including WPILib's pose estimators. Use one
detector per camera stream. The native call releases the GIL, so separate streams can run on separate threads.

The C library has a plain C API (`src/napa_vision/_native/napa_core.h`), so other languages can call it directly.
C and C++ projects can build and install it with CMake and link `napa::napa`; see
[the guide](docs/guide.md#from-c-or-another-language).
[The guide](docs/guide.md) covers tuning, poses, Kalman filters, coordinate conventions, and calibration, and
[the API reference](docs/api.md) lists every class, method, and field.

## Command-line tools

| Command | Purpose |
| --- | --- |
| `napa-tag` | Write tags to print: SVG at an exact printed size, or PNG |
| `napa-calibrate` | Measure camera intrinsics from checkerboard photographs |
| `napa-benchmark` | Score Napa on your own annotated images, with optional recall, precision, and latency gates |

## Development

```bash
python -m pip install -e ".[headless,dev,benchmark]"
ruff format --check . && ruff check .
pytest --cov=napa_vision
```

## License

Copyright © 2026 Zachary Wilson. Napa is free for noncommercial use under the
[PolyForm Noncommercial License 1.0.0](LICENSE): personal projects, research, education, student robotics teams,
and other noncommercial organizations can use, modify, and share it. Copies, forks, and works based on Napa must
include the license and its notices crediting Zachary Wilson. Commercial use, including selling Napa or products
built on it, needs a separate license; ask through [the repository](https://github.com/zwatx/NAPA).

The tag families' codewords come from [AprilTag](https://github.com/AprilRobotics/apriltag) under its BSD license; see
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
