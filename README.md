# Napa

Napa is a fast AprilTag (`tag36h11`) detector for any camera. Give it a frame and get back tag IDs and sub-pixel
corners, plus poses if you provide calibration. It opens no cameras, starts no threads, and publishes nothing, so a
webcam, a Raspberry Pi, a headset, recorded video, or a camera SDK can all feed it the same way.

The detector is a small, dependency-free C library with a Python binding. On the benchmark below it is **3.5× to 21×
faster than standard AprilTag on every scene and resolution tested**. It finds more tags and returns corners about
twice as accurate.

## Compared with standard AprilTag

Standard AprilTag means the upstream AprilTag 3 library at its default settings (`quad_decimate=2`). Both detectors
ran on the same generated grayscale frames, interleaved, single-threaded, on one pinned core of an Intel
i7-12700H. Times are median detector-call latency.

| Scene, 1280×720 | AprilTag | Napa | Speedup | Tags found (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: |
| Empty frame | 1.54 ms | 0.24 ms | 6.5× | — |
| Textured, no tag | 4.81 ms | 0.39 ms | 12.3× | — |
| One tag | 1.72 ms | 0.31 ms | 5.5× | 100% → 100% |
| Four tags | 2.19 ms | 0.46 ms | 4.8× | 100% → 100% |
| Small tags (32–53 px) | 1.56 ms | 0.29 ms | 5.3× | 0% → 100% |
| Blurred | 1.75 ms | 0.31 ms | 5.7× | 100% → 100% |
| Perspective | 1.75 ms | 0.32 ms | 5.4× | 100% → 100% |
| Sensor noise | 5.29 ms | 0.34 ms | 15.4× | 100% → 100% |
| Uneven lighting | 1.72 ms | 0.31 ms | 5.5× | 100% → 100% |
| Motion blur | 1.71 ms | 0.31 ms | 5.6× | 100% → 100% |
| Text clutter | 3.04 ms | 0.53 ms | 5.8× | 100% → 100% |

Across all 33 scene/resolution combinations (640×480 to 1920×1080), Napa's median and p95 latencies were never
less than 3.5× faster. The smallest margin was four tags at 640×480; the largest were noise and uneven lighting
(20–21×). Napa also beat AprilTag at full resolution (`quad_decimate=1`) by at least 9× on every combination.
Neither detector reported a false ID.

**Corners**, scored against exact ground truth on 120 sub-pixel renders (32–240 px tags, any rotation, mild
perspective, blur, and noise). AprilTag's corners were shifted by its half-pixel origin before scoring.

| Worst corner per tag | Napa | AprilTag (decimate 1) | AprilTag (default) |
| --- | ---: | ---: | ---: |
| Tags found | 120 / 120 | 112 / 120 | 76 / 120 |
| Median error | 0.037 px | 0.065 px | 0.060 px |
| 95th percentile | 0.078 px | 0.139 px | 0.144 px |
| Maximum | 0.102 px | 0.227 px | 0.181 px |

See [the benchmark guide](benchmarks/README.md) for every scene, resolution, percentile, and raw sample, and how to
reproduce them. These are synthetic frames on a desktop CPU; measure on your own camera and hardware before relying
on a specific number.

## Why it's faster

Standard AprilTag treats the frame as a big image-processing job. It thresholds every pixel, segments the whole
image into regions, and fits quads to every region boundary before it ever asks whether something is a tag. Napa
asks that question as early and as cheaply as possible.

### 1. The rake: sample sparsely, with a guarantee

A tag's black border is one cell wide (a tag is 8 cells across). If you sample every few pixels, you're guaranteed to
hit the border of every tag above a minimum size, so you can't miss a tag by skipping the pixels in between.

Napa drags a "rake" across the frame: one sample every ¾ of the smallest cell, in both directions. Blur turns a
cell's edges grey but leaves its core dark, and ¾-cell spacing always lands a sample in that core. For the default
32 px minimum tag, that's one pixel in nine. Coarser copies of the rake (2×, 4×, 8× …) handle large tags, and cost a
third as much again in total.

### 2. String search on a matrix: skip what can't match

Fast string search (Boyer–Moore) doesn't compare every character. A single cheap check can prove a match can't
start in the next several positions, so the search jumps past them.

Napa applies the same idea to the image matrix. It splits the raked samples into small tiles and records each tile's
darkest and brightest value. A tile with no contrast can't contain a tag edge, so it's skipped outright, and no work
is done on its pixels. The contrast bar rises automatically with sensor noise, measured from the frame itself. In the
benchmark, sensor noise added 7–13% to Napa's time and 2.5–3.4× to standard AprilTag's.

The same idea decodes the tag. Its 36 payload bits are matched against 587 codes in 4 rotations, allowing up to 3
wrong bits. Any code within 3 bits must match at least one of four 9-bit blocks of the payload *exactly* (the
pigeonhole principle), so Napa looks up those blocks in a table and compares only the handful of codes that share
one. It doesn't scan all 2,348.

### 3. Old-school vector math

Every pass over the samples is a straight loop over contiguous rows: fixed-width tiles, min/max, and compares. The
compiler turns those loops into SIMD instructions (SSE/AVX on x86, NEON on ARM). This is the classic Apple
AltiVec/vDSP approach: express the work as whole-row operations so each instruction handles many pixels. Blob
labelling avoids unpredictable branches, so noisy frames don't stall the CPU.

### What happens to a candidate

Each dark blob of plausible size gets a rough quadrilateral from its extreme points. Napa then:

1. Fits each side to the strongest edge at full resolution, searching outward from the blob.
2. Decodes immediately with a loose tolerance, which discards text and texture before any refinement.
3. Refines each side to the centre of its edge gradient, which is exact for real (pixel-integrated) edges at any blur.
4. Intersects the four lines for corners and decodes the payload.

The flow is: rake, then skip flat tiles, then find blobs, then fit and early-reject, then refine, then decode.

## Obstructions

Napa corrects up to 3 wrong payload bits (standard AprilTag defaults to 2), so a hand, cable, or glare spot over
part of the payload often still decodes. In the benchmark's obstruction sweep, Napa recovered 410 of 864 obstructed
tags versus 336 for standard AprilTag.

No AprilTag detector can find a tag whose black border or corners are hidden, because that happens before any bits
are read. Obstruction tolerance is best-effort; there's no guaranteed percentage.

## Limits

- Tags smaller than `min_tag_px` (default 32 px across the black square) may be missed. Lower it to find smaller tags
  at some cost in speed.
- Tags need their white quiet zone, as printed tags have.
- On heavily blurred (σ ≈ 1.2 px), 32 px tags, corner error grows to about 0.2 px median, since the border is barely
  wider than the blur.
- The fast path needs a C compiler at install time. Without one, Napa installs anyway and falls back to OpenCV's
  ArUco detector (`napa_vision.native_available()` reports which is active).

## Quick start

Requires Python 3.11 or newer and a C compiler.

```bash
python -m pip install -e ".[headless]"
```

```python
from napa_vision import VisionProcessor

processor = VisionProcessor()  # uses Napa's native detector
result = processor.process(image, timestamp=frame_timestamp_s)  # uint8 grayscale, BGR, or BGRA
for tag in result.image_observations:
    print(tag.tag_id, tag.corners)
```

For metric poses, pass `calibration=load_calibration_data("camera.json")` and `tag_size_m=` to `VisionProcessor`.
To tune detection, pass `detector=NapaDetector(min_tag_px=24)`. Use one processor per camera stream; the native
call releases the GIL, so separate streams can run on separate threads.

The C library has a plain C API (`src/napa_vision/_native/napa_core.h`), so other languages can call it directly.

## Command-line tools

| Command | Purpose |
| --- | --- |
| `napa-vision` | Run on a live camera or replayed frames |
| `napa-calibrate` | Measure camera intrinsics from checkerboard images |
| `napa-benchmark` | Score detection on an annotated image set; `--compare-detectors napa` measures it against OpenCV |
| `napa-simulate` | Render synthetic scenes and compare detector options |

See [operations](docs/operations.md) for calibration, coordinate frames, and deployment.

## Development

```bash
python -m pip install -e ".[headless,dev,nt,benchmark]"
ruff format --check . && ruff check .
pytest --cov=napa_vision
```
