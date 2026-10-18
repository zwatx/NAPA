# Napa

Napa is a fast AprilTag (`tag36h11`) detection algorithm that works with any camera. Give it a frame and get back tag IDs,
corners, and, if you provide calibration, poses. It doesn't open cameras, start threads, or publish anything,
so a webcam, a Raspberry Pi, a headset, recorded video, or a camera SDK can all feed it the same way.

Standard AprilTag treats every frame the same way. It thresholds the whole image, segments it, fits quads, then
decodes each one. Napa does much less work: it rejects most of a frame cheaply before the detector runs.

## How Napa saves work

### 1. Skip-ahead search on a matrix

String search algorithms like Boyer–Moore are fast because they don't compare every character. One cheap
check can prove the pattern can't start in the next several positions, so the search jumps past them.

Napa uses the same idea on a 2-D pixel matrix. An AprilTag is a dense square of sharp black/white transitions,
so a region with no strong edges can't contain one, and every pixel in it can be skipped. The skip-scan
prefilter (`BlockSkipPrefilter`):

1. Samples the frame on a stride (every 4th pixel in each direction by default, so 1/16 of the pixels).
2. Exits immediately if the sampled frame is flat. Blank, dark, or washed-out frames cost almost nothing.
3. Computes a gradient on the sampled grid and keeps only high-transition cells.
4. Grows those cells into a few padded candidate windows and runs the full decoder only inside them.

If the scene is so busy that the candidates would cover most of the frame, Napa falls back to a single
full-frame pass rather than many small ones, so a cluttered frame costs only slightly more than the plain detector.

### 2. Rake sampling with whole-matrix vector math

The strided sample works like a rake dragged across the image. Its tines sit at a fixed spacing, cover the whole
frame in one pass, and leave everything between them untouched. This is the classic Apple vector-math approach
(AltiVec/vDSP): express the work as operations on whole rows and matrices so the CPU's SIMD lanes process many
pixels per instruction, and never loop over pixels one at a time.

Every prefilter stage (strided view, statistics, blur, Sobel gradient, threshold, morphology, connected
components) is a single array operation, which OpenCV and NumPy run with SSE/AVX on x86 and NEON on ARM.
The strided view itself copies nothing. The next step is a recursive rake: rake finer only inside the cells
the coarse pass marked. It isn't implemented yet.

### 3. Fast payload lookup

Decoding compares each candidate's 36 payload bits against every code in the family, in all four rotations.
The [experimental decoder](experiments/occlusion_detector/README.md) splits each code into blocks and indexes
them, so one exact block match narrows the search to a few candidates before the full Hamming-distance check.
It's 3.2× faster than scanning the whole family, and it doesn't precompute a table of every corrupted codeword.

## Compared with standard AprilTag

These are detector-only timings for the same generated grayscale images, on one CPU thread (Intel i7-12700H).
**Standard AprilTag** is the upstream AprilTag 3 library at its default settings (`quad_decimate=2`).

| 1280×720 frame | Standard AprilTag | Napa |
| --- | ---: | ---: |
| Empty frame (no tag) | 1.18 ms | **0.06 ms** (~20× faster) |
| Small tags (32–53 px), recall | 0% | **100%** |
| Obstructed tags recovered (864 cases) | 336 | **390** (411 with ArUco3 mode) |
| Clean frame with one large tag | **1.30 ms** | 2.83 ms (known IDs), 4.72 ms (skip-scan) |

Napa wins most clearly on empty frames, and it finds small tags that standard AprilTag misses at its default
settings. Standard AprilTag is faster on a clean frame with one large tag and on busy frames with no tag
(3.2 ms versus 9.1 ms), because its decimation trades away the small-tag recall shown above.

For raw timings, p95/p99 latency, recall, and false positives at three resolutions, see the
[benchmarks](benchmarks/README.md).

## Obstructions

Napa tolerates partial obstruction of the tag's payload. Its decoder corrects up to three wrong payload bits;
standard AprilTag defaults to two. In the benchmark, every tag with about 10% of the payload centre covered was
still identified.

No AprilTag detector can find a tag whose black border or corners are hidden, because that happens before any
bits are read. A white patch over about 5% of a corner hid the tag in every test case, but a grey patch the same
size didn't. Treat obstruction tolerance as best-effort; there's no guaranteed percentage.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m pip install -e ".[headless]"
```

```python
from napa_vision import BlockSkipPrefilter, OpenCVArucoFiducialDetector, SkipScanConfig, VisionProcessor

detector = OpenCVArucoFiducialDetector(prefilter=BlockSkipPrefilter(SkipScanConfig(enabled=True)))
processor = VisionProcessor(detector=detector)

result = processor.process(image, timestamp=frame_timestamp_s)  # uint8 grayscale, BGR, or BGRA
for tag in result.image_observations:
    print(tag.tag_id, tag.corners)
```

For metric poses, add `calibration=load_calibration_data("camera.json")` and `tag_size_m=` to `VisionProcessor`.
Use one processor per camera stream.

A custom `detector=` only needs `detect(image) -> list[FiducialImageObservation]`, so you can swap in a
different backend without changing the rest of your code.

## Command-line tools

| Command | Purpose |
| --- | --- |
| `napa-vision` | Run on a live camera or replayed frames |
| `napa-calibrate` | Measure camera intrinsics from checkerboard images |
| `napa-benchmark` | Score detection on an annotated image set |
| `napa-simulate` | Render synthetic scenes and compare detector options |

See [operations](docs/operations.md) for calibration, coordinate frames, and deployment.

## Development

```bash
python -m pip install -e ".[headless,dev,nt,benchmark]"
ruff format --check . && ruff check .
pytest --cov=napa_vision
```
