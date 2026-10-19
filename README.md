# Napa

**Not Another Pixel Algorithm**: Napa skips most of the pixels in each frame.

Napa is a fast AprilTag (`tag36h11`) detector for any camera. Give it a frame and it returns tag IDs and sub-pixel
corners, plus poses if you supply a calibration. It opens no cameras, starts no threads, and publishes nothing, so a
webcam, a Raspberry Pi, a headset, recorded video, or a camera SDK can all feed it the same way.

The detector is a small C library with no dependencies, plus a Python binding that needs only numpy.

## Compared with standard AprilTag

"Standard AprilTag" here means the upstream AprilTag 3 library at its default settings (`quad_decimate=2`). Both
detectors ran single-threaded on one pinned core of an Intel i7-12700H, taking turns on the same grayscale frames.

The benchmark has **2,850 distinct frames holding 5,213 tags**. Every frame is rendered separately, with random tag
IDs, positions, sizes, angles, 3D tilt, backgrounds, and camera degradations, and every tag has exact sub-pixel
ground truth. [The benchmark guide](benchmarks/README.md) describes each scene.

**Speed.** Napa was faster on every one of the 2,850 frames:

- **Noise-free frames: 4.6× to 6.1× faster** (median per frame, by resolution). The slowest single frame was still
  3.4× faster.
- **Realistic frames: about 18× faster** (median). Every other scene includes at least light sensor noise
  (σ ≈ 1.5 grey levels), which a real sensor always has. Standard AprilTag's time climbs steeply with noise: on a
  1280×720 frame it takes about 1.2 ms with none, 8 ms at σ = 1, and 13 ms at σ = 1.5. Napa stays near 0.2 ms.
  The slowest single frame here was 3.2× faster.
- Against AprilTag at full resolution (`quad_decimate=1`), Napa was 74× faster (median) and at least 10× faster on
  every frame.

**Tags found.** Napa found **95.4%** of tags, standard AprilTag 86.5%, and full-resolution AprilTag 99.3%. Napa
finds small tags that standard AprilTag's decimation loses, but it misses more tags than AprilTag in a few hard
scenes. These are the 1280×720 results; the guide has every resolution:

| Scene, 1280×720 | Tags | AprilTag | Napa | Speedup (median) | Slowest frame | Tags found (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Noise-free | 104 | 1.54 ms | 0.27 ms | 5.9× | 4.5× | 104 → 104 |
| One tag | 50 | 15.4 ms | 0.27 ms | 56× | 39× | 50 → 50 |
| 2–12 tags | 354 | 11.4 ms | 0.77 ms | 15× | 6.2× | 354 → 342 |
| Small tags (32–53 px) | 120 | 13.6 ms | 0.48 ms | 28× | 6.7× | 0 → 120 |
| Large tags | 50 | 11.8 ms | 0.54 ms | 21× | 7.3× | 50 → 50 |
| Steep angle (40–65°) | 112 | 12.1 ms | 0.59 ms | 20× | 6.9× | 109 → 80 |
| Defocus blur | 89 | 11.5 ms | 0.56 ms | 20× | 6.4× | 89 → 87 |
| Motion blur | 94 | 11.4 ms | 0.88 ms | 12× | 6.7× | 94 → 93 |
| Heavy noise (σ 4–14) | 104 | 10.9 ms | 0.82 ms | 15× | 6.1× | 104 → 101 |
| Uneven lighting | 103 | 12.5 ms | 0.64 ms | 19× | 7.3× | 103 → 99 |
| Low contrast | 100 | 15.3 ms | 0.29 ms | 53× | 29× | 95 → 87 |
| Underexposed | 99 | 12.9 ms | 0.48 ms | 25× | 9.1× | 99 → 93 |
| Overexposed | 112 | 1.99 ms | 0.35 ms | 6× | 4.0× | 112 → 110 |
| JPEG artefacts | 100 | 2.96 ms | 0.49 ms | 6× | 3.9× | 100 → 97 |
| Busy background | 87 | 12.1 ms | 0.51 ms | 23× | 8.4× | 87 → 85 |
| At the frame edge | 105 | 12.4 ms | 0.57 ms | 19× | 7.0× | 103 → 102 |
| No tag: empty / clutter / decoys | — | 11–17 ms | 0.17–0.79 ms | 13–100× | 4.0× | — |

Times are the median detector call. Speedups are per frame: AprilTag's time divided by Napa's on the same frame.
"Slowest frame" is the smallest of those ratios.

**False detections.** Napa reported one wrong ID in 2,850 frames: a tag-shaped decoy whose random payload happened
to fall within Napa's 3-bit error correction of a real code. Standard AprilTag, which corrects 2 bits, reported none.

**Corners.** For every tag found, the benchmark measured the worst corner's distance from the true corner. Napa's
median error was **0.063 px**, against 0.10 px for AprilTag. At the 95th percentile the two were level (0.38 px).
Napa has the worse outliers: in one low-contrast frame it fitted a side to the edge of the tag's white margin
instead of its black border, putting a corner 41 px off. AprilTag's worst was 7.6 px.

These are synthetic frames on a desktop CPU. Measure on your own camera and hardware before relying on a specific
number: `napa-benchmark` scores Napa on your own annotated images (see [the guide](docs/guide.md)).

## Why it's faster

Standard AprilTag treats each frame as a large image-processing job. It thresholds every pixel, segments the whole
image into regions, and fits quadrilaterals to every region's boundary before it asks whether anything is a tag.
Napa asks that question as early and as cheaply as it can.

### 1. The rake: sample sparsely, with a guarantee

A tag's black border is one cell wide, and a tag is 8 cells across. Sample every few pixels and you're guaranteed to
land on the border of every tag above a minimum size, so skipping the pixels in between can't skip a tag.

Napa drags a "rake" across the frame, taking one sample every ¾ of the smallest cell in each direction. Blur greys a
cell's edges but leaves its core dark, and ¾-cell spacing always puts a sample in that core. For the default 32 px
minimum tag, that's one pixel in nine. Coarser copies of the rake (2×, 4×, 8×, …) catch larger tags and add only a
third again to the cost.

### 2. String search on a matrix: skip what can't match

Fast string search (Boyer–Moore) doesn't compare every character. One cheap check can prove that no match starts in
the next several positions, so the search jumps past them.

Napa applies the same idea to the image. It splits the raked samples into small tiles and records each tile's
darkest and brightest value. A tile without contrast can't contain a tag's edge, so Napa skips it and does no work
on its pixels. The contrast needed rises automatically with sensor noise, measured from the frame itself, which is
why noise barely changes Napa's time.

The same idea speeds up decoding. A tag's 36 payload bits must be matched against 587 codes in 4 rotations, allowing
up to 3 wrong bits. Any code within 3 bits of the payload must match at least one of four 9-bit blocks *exactly* (the
pigeonhole principle). So Napa looks those blocks up in a table and compares only the handful of codes that share
one, instead of all 2,348.

### 3. Old-school vector math

Every pass over the samples is a straight loop over contiguous rows: fixed-width tiles, minimums, maximums, and
comparisons. Compilers turn those loops into SIMD instructions (SSE/AVX on x86, NEON on ARM), so each instruction
handles many pixels. It's the classic Apple AltiVec/vDSP approach. Blob labelling avoids unpredictable branches, so
noisy frames don't stall the CPU.

### What happens to a candidate

Each dark blob of plausible size gets a rough quadrilateral from its extreme points. Then Napa:

1. Fits each side to the strongest edge at full resolution, searching outward from the blob.
2. Decodes straight away with a loose tolerance. This throws out text and texture before any expensive refinement.
3. Refines each side to the centre of its edge gradient, which is exact for real, pixel-integrated edges at any blur.
4. Intersects the four lines to get the corners, and decodes the payload.

In short: rake, skip flat tiles, find blobs, fit and reject early, refine, decode.

## Obstructions

Napa corrects up to 3 wrong payload bits (standard AprilTag defaults to 2), so a hand, cable, or glare spot over part
of a tag often still decodes. In the benchmark's obstruction sweep (864 tags, each partly covered by a white or grey
patch), Napa recovered 369 tags and standard AprilTag 330.

No AprilTag detector can find a tag whose black border or corners are hidden, because the border is found before any
bits are read. Obstruction tolerance is best-effort, with no guaranteed percentage.

## Limits

- Tags smaller than `min_tag_px` (default 32 px across the black square) may be missed. Lower it to find smaller tags,
  at some cost in speed.
- Steeply tilted tags are Napa's weakest case: at 40–65° of tilt it found 71–76% of tags at 720p and 1080p, where
  standard AprilTag found 92–97%.
- In low light and low contrast, Napa finds a few percent fewer tags than standard AprilTag.
- Tags need their white quiet zone, as printed tags have. If the margin is darker than the background around it,
  Napa can mistake the margin's outer edge for the tag's.
- On heavily blurred tags near the minimum size (σ ≈ 1.2 px at 32 px), corner error grows to about 0.2 px median,
  because the border is barely wider than the blur.
- Installing needs a C99 compiler, since Napa is its C core.

## Quick start

Requires Python 3.11 or newer and a C compiler.

```bash
python -m pip install -e .              # the detector: needs only numpy
python -m pip install -e ".[headless]"  # adds OpenCV, for poses, calibration, and napa-benchmark
```

```python
from napa_vision import NapaDetector

detector = NapaDetector()
for tag in detector.detect(image):  # uint8 grayscale, BGR, or BGRA
    print(tag.tag_id, tag.corners)
```

For metric poses, use `VisionProcessor(calibration=load_calibration_data("camera.json"), tag_size_m=...)`. Use one
detector per camera stream. The native call releases the GIL, so separate streams can run on separate threads.

The C library has a plain C API (`src/napa_vision/_native/napa_core.h`), so other languages can call it directly.
[The guide](docs/guide.md) covers tuning, poses, coordinate conventions, and calibration.

## Command-line tools

| Command | Purpose |
| --- | --- |
| `napa-calibrate` | Measure camera intrinsics from checkerboard photographs |
| `napa-benchmark` | Score Napa on your own annotated images, with optional recall, precision, and latency gates |

## Development

```bash
python -m pip install -e ".[headless,dev,benchmark]"
ruff format --check . && ruff check .
pytest --cov=napa_vision
```
