# Napa benchmarks

`performance_matrix.py` compares Napa with standard AprilTag (AprilTag 3 through `robotpy-apriltag`, at its default
`quad_decimate=2` and at full resolution) on a large set of generated grayscale frames. It measures detector calls
only. Camera delivery, colour conversion, and pose solving are outside the timed region.

## What's in the corpus

Every frame is rendered on its own: no image is reused, and none is a copy of another with a different label. Each
tag gets a random ID from all 587, a random position, any in-plane angle, a random 3D tilt (and therefore
perspective), and random black and white levels. Tags never overlap, and each keeps its white quiet zone inside the
frame.

Tags are rendered with exact sub-pixel ground truth: every pixel is the mean of a 4×4 grid of point samples, as a
sensor integrates light. So the benchmark scores whether each tag was found, any invented ID, and how far each
corner landed from where it really is.

| Scenario | What varies |
| --- | --- |
| `empty` | A uniform frame at random brightness, with light noise. No tags. |
| `clutter-only` | Shapes, text, checkerboards, or smooth natural texture. No tags. |
| `decoys` | Black-bordered squares with random payloads: tag-shaped, but not valid codes. |
| `one-tag` | One tag on a plain background. |
| `many-tags` | 2–12 tags of mixed sizes on mixed backgrounds. |
| `small-tags` | 1–4 tags with a 32–53 px black square, at and just above Napa's default `min_tag_px`. |
| `large-tags` | One tag filling 45–80% of the frame height. |
| `steep-angle` | Tags tilted 40–65° away from the camera. |
| `blur` | Gaussian defocus, σ 0.8–2.0 px. |
| `motion-blur` | Linear motion blur of 3–11 px (odd lengths, so the kernel is centred) in any direction. |
| `noise` | Gaussian sensor noise, σ 4–14 grey levels. |
| `uneven-light` | A lighting gradient (30–60% at the dark end) plus vignetting. |
| `low-contrast` | Tag black and white only 40–90 grey levels apart. |
| `underexposed` | Everything scaled to 15–35% brightness, then noise. |
| `overexposed` | Gain of 1.3–1.8, with clipping. |
| `jpeg` | JPEG compression at quality 25–70. |
| `busy-background` | Tags on text, shapes, or checkerboards. |
| `frame-edge` | Tags whose quiet zone touches the edge of the frame. |
| `noise-free` | 1–3 tags on a plain background with no noise at all: the best case for both detectors. |
| `dark-background` | Tags whose one-cell white margin is all that separates them from a dark or mid-grey background. |

Unless a scenario says otherwise, positive frames hold 1–3 tags on a random background, and tag size scales with
the frame (48–170 px at 480 lines). Every scenario except `noise-free` adds at least light sensor noise
(σ ≈ 1.5 grey levels), as a real sensor would. The default run renders 50 frames per scenario at each of 640×480,
1280×720, and 1920×1080: 3,000 frames in all.

A separate obstruction sweep hides part of one tag per frame with a white or grey patch. The patch sits over the
payload centre, along one border, or over a corner, and covers 0–40% of the black square. There are 24 random tags
for each combination (864 frames), at the smallest resolution.

## Timing

Backends take turns on each frame in a shuffled order, ten timed calls each after one untimed warm-up call. The
report gives every backend's latency percentiles per scenario and resolution. It also gives Napa's speedup **per
frame**: AprilTag's median time on a frame divided by Napa's median time on the same frame. "Worst frame" is the
smallest of those ratios, so a single slow frame can't hide behind a good median.

## Running it

```bash
python -m pip install -e ".[headless,dev,benchmark]"

# Pin to one idle core so other processes don't skew timings (Linux shown). About 90 minutes.
taskset -c 2 python benchmarks/performance_matrix.py --output benchmarks/results/$(date +%F)_detector_matrix.json
```

Useful options:

- `--scenarios noise,jpeg`, `--resolutions 1280x720`, and `--frames 10` select a quick subset.
- `--backends napa,apriltag,apriltag-full-res,opencv-aruco` chooses what to compare (OpenCV is off by default).
- `--occlusion-cases 0` skips the obstruction sweep.
- `--seed` changes every frame. Each frame's seed depends only on the run seed, its scenario, resolution, and
  index, so any frame can be re-rendered on its own with `render_frame()`.

The script prints the summary tables used in the main README and writes two files:

- `*.json`: run configuration, machine details, per-scenario results (frames, tags, found, false IDs, latency
  percentiles, corner error), per-frame speedup statistics, and the obstruction sweep.
- `*.csv`: one row per frame and backend, with expected and detected IDs, corner errors, every raw timing sample,
  and the scenario's random parameters (`detail_*` columns). Use it to find and re-render any frame a detector
  missed.

## Latest results

Standard AprilTag (`quad_decimate=2`) against Napa, from the checked-in run on one pinned core of an Intel i7-12700H.
p50 is the median detector call across all frames and repetitions in a row.

#### 640x480

| Scene | Frames | Tags | AprilTag p50 | Napa p50 | Speedup (median / worst frame) | Found (AprilTag → Napa) | False IDs (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 5.34 ms | 0.03 ms | 148.1× / 15.6× | — | 0 → 0 |
| clutter-only | 50 | 0 | 3.84 ms | 0.17 ms | 18.0× / 6.5× | — | 0 → 0 |
| decoys | 50 | 0 | 3.79 ms | 0.15 ms | 25.1× / 7.2× | — | 0 → 1 |
| one-tag | 50 | 50 | 4.82 ms | 0.13 ms | 37.6× / 26.2× | 46/50 → 50/50 | 0 → 0 |
| many-tags | 50 | 249 | 3.67 ms | 0.34 ms | 11.2× / 6.3× | 178/249 → 249/249 | 0 → 0 |
| small-tags | 50 | 131 | 3.83 ms | 0.21 ms | 17.8× / 8.9× | 0/131 → 131/131 | 0 → 0 |
| large-tags | 50 | 50 | 3.53 ms | 0.18 ms | 19.4× / 8.1× | 50/50 → 50/50 | 0 → 0 |
| steep-angle | 50 | 109 | 3.93 ms | 0.22 ms | 17.1× / 7.2× | 71/109 → 104/109 | 0 → 0 |
| blur | 50 | 96 | 3.50 ms | 0.22 ms | 14.6× / 6.5× | 82/96 → 96/96 | 0 → 0 |
| motion-blur | 50 | 110 | 3.65 ms | 0.23 ms | 13.7× / 6.8× | 86/110 → 109/110 | 0 → 0 |
| noise | 50 | 110 | 3.84 ms | 0.30 ms | 12.2× / 4.1× | 94/110 → 110/110 | 0 → 0 |
| uneven-light | 50 | 98 | 3.68 ms | 0.23 ms | 14.9× / 7.5× | 82/98 → 98/98 | 0 → 0 |
| low-contrast | 50 | 96 | 5.14 ms | 0.16 ms | 33.3× / 20.1× | 70/96 → 95/96 | 0 → 0 |
| underexposed | 50 | 110 | 3.87 ms | 0.24 ms | 15.9× / 9.3× | 96/110 → 110/110 | 0 → 0 |
| overexposed | 50 | 100 | 1.31 ms | 0.20 ms | 5.2× / 3.6× | 82/100 → 100/100 | 0 → 0 |
| jpeg | 50 | 104 | 1.32 ms | 0.21 ms | 6.8× / 3.5× | 92/104 → 104/104 | 0 → 0 |
| busy-background | 50 | 102 | 3.66 ms | 0.22 ms | 15.9× / 6.1× | 89/102 → 102/102 | 0 → 0 |
| frame-edge | 50 | 102 | 3.54 ms | 0.19 ms | 17.1× / 8.9× | 74/102 → 102/102 | 0 → 0 |
| noise-free | 50 | 98 | 0.61 ms | 0.12 ms | 5.4× / 4.4× | 83/98 → 98/98 | 0 → 0 |
| dark-background | 50 | 96 | 3.60 ms | 0.22 ms | 17.4× / 7.7× | 71/96 → 96/96 | 0 → 0 |

#### 1280x720

| Scene | Frames | Tags | AprilTag p50 | Napa p50 | Speedup (median / worst frame) | Found (AprilTag → Napa) | False IDs (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 17.25 ms | 0.09 ms | 164.2× / 20.7× | — | 0 → 0 |
| clutter-only | 50 | 0 | 11.52 ms | 0.35 ms | 18.6× / 5.0× | — | 0 → 0 |
| decoys | 50 | 0 | 15.06 ms | 0.26 ms | 54.3× / 9.2× | — | 0 → 0 |
| one-tag | 50 | 50 | 15.20 ms | 0.19 ms | 77.6× / 58.6× | 50/50 → 50/50 | 0 → 0 |
| many-tags | 50 | 354 | 11.24 ms | 0.61 ms | 18.3× / 7.3× | 354/354 → 354/354 | 0 → 0 |
| small-tags | 50 | 120 | 13.60 ms | 0.30 ms | 45.0× / 11.5× | 0/120 → 120/120 | 0 → 0 |
| large-tags | 50 | 50 | 12.34 ms | 0.38 ms | 41.9× / 13.8× | 50/50 → 50/50 | 0 → 0 |
| steep-angle | 50 | 112 | 12.14 ms | 0.35 ms | 33.2× / 11.8× | 109/112 → 108/112 | 0 → 0 |
| blur | 50 | 89 | 11.53 ms | 0.33 ms | 32.6× / 8.7× | 89/89 → 89/89 | 0 → 0 |
| motion-blur | 50 | 94 | 11.30 ms | 0.43 ms | 16.9× / 8.6× | 94/94 → 94/94 | 0 → 0 |
| noise | 50 | 104 | 10.95 ms | 0.65 ms | 19.3× / 5.3× | 104/104 → 104/104 | 0 → 0 |
| uneven-light | 50 | 103 | 12.53 ms | 0.40 ms | 32.1× / 11.6× | 103/103 → 103/103 | 0 → 0 |
| low-contrast | 50 | 100 | 15.43 ms | 0.22 ms | 70.0× / 43.6× | 95/100 → 99/100 | 0 → 0 |
| underexposed | 50 | 99 | 12.48 ms | 0.33 ms | 39.1× / 13.1× | 99/99 → 99/99 | 0 → 0 |
| overexposed | 50 | 112 | 2.08 ms | 0.26 ms | 7.4× / 4.9× | 112/112 → 112/112 | 0 → 0 |
| jpeg | 50 | 100 | 2.99 ms | 0.29 ms | 8.5× / 4.0× | 100/100 → 100/100 | 0 → 0 |
| busy-background | 50 | 87 | 12.14 ms | 0.31 ms | 40.5× / 10.7× | 87/87 → 87/87 | 0 → 0 |
| frame-edge | 50 | 105 | 12.17 ms | 0.32 ms | 37.0× / 12.2× | 103/105 → 105/105 | 0 → 0 |
| noise-free | 50 | 104 | 1.58 ms | 0.19 ms | 8.6× / 7.0× | 104/104 → 104/104 | 0 → 0 |
| dark-background | 50 | 105 | 11.96 ms | 0.27 ms | 44.4× / 12.5× | 91/105 → 105/105 | 0 → 0 |

#### 1920x1080

| Scene | Frames | Tags | AprilTag p50 | Napa p50 | Speedup (median / worst frame) | Found (AprilTag → Napa) | False IDs (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 43.40 ms | 0.16 ms | 240.8× / 19.3× | — | 0 → 0 |
| clutter-only | 50 | 0 | 28.67 ms | 0.34 ms | 52.3× / 5.6× | — | 0 → 0 |
| decoys | 50 | 0 | 38.06 ms | 0.37 ms | 89.5× / 10.4× | — | 0 → 0 |
| one-tag | 50 | 50 | 37.66 ms | 0.30 ms | 119.8× / 93.3× | 50/50 → 50/50 | 0 → 0 |
| many-tags | 50 | 303 | 26.57 ms | 0.92 ms | 33.1× / 11.2× | 303/303 → 303/303 | 0 → 0 |
| small-tags | 50 | 122 | 30.36 ms | 0.48 ms | 67.5× / 13.4× | 0/122 → 121/122 | 0 → 0 |
| large-tags | 50 | 50 | 30.35 ms | 0.46 ms | 69.0× / 14.5× | 50/50 → 50/50 | 0 → 0 |
| steep-angle | 50 | 96 | 30.19 ms | 0.52 ms | 59.3× / 13.1× | 88/96 → 94/96 | 0 → 0 |
| blur | 50 | 97 | 29.45 ms | 0.56 ms | 52.4× / 9.3× | 97/97 → 97/97 | 0 → 0 |
| motion-blur | 50 | 95 | 31.04 ms | 0.54 ms | 62.4× / 10.7× | 95/95 → 95/95 | 0 → 0 |
| noise | 50 | 100 | 30.73 ms | 1.68 ms | 18.8× / 6.2× | 100/100 → 100/100 | 0 → 0 |
| uneven-light | 50 | 101 | 30.67 ms | 0.66 ms | 50.9× / 14.3× | 101/101 → 101/101 | 0 → 0 |
| low-contrast | 50 | 113 | 36.86 ms | 0.41 ms | 90.4× / 62.7× | 108/113 → 111/113 | 0 → 0 |
| underexposed | 50 | 95 | 35.30 ms | 0.49 ms | 74.6× / 14.6× | 95/95 → 95/95 | 0 → 0 |
| overexposed | 50 | 100 | 9.44 ms | 0.54 ms | 8.8× / 4.6× | 100/100 → 100/100 | 0 → 0 |
| jpeg | 50 | 100 | 8.15 ms | 0.57 ms | 9.6× / 4.0× | 100/100 → 100/100 | 0 → 0 |
| busy-background | 50 | 97 | 30.52 ms | 0.52 ms | 59.8× / 12.9× | 97/97 → 97/97 | 0 → 0 |
| frame-edge | 50 | 88 | 31.28 ms | 0.49 ms | 63.9× / 11.8× | 88/88 → 88/88 | 0 → 0 |
| noise-free | 50 | 108 | 3.22 ms | 0.31 ms | 10.4× / 9.1× | 108/108 → 108/108 | 0 → 0 |
| dark-background | 50 | 107 | 30.84 ms | 0.46 ms | 70.1× / 15.6× | 89/107 → 107/107 | 0 → 0 |

The JSON report also holds full-resolution AprilTag (`quad_decimate=1`), the 95th/99th percentiles, corner error by
scene, and the obstruction sweep.

## Reading the results

- Compare scenario rows at the same resolution rather than overall averages: the scenarios differ a lot in cost.
- Corner error is the worst corner's distance from the truth, for each tag found. AprilTag's corners are shifted by
  −0.5 px to the pixel-centre convention, and its corner order is aligned to the truth before scoring. Napa's are
  scored as returned.
- A "false ID" is any detection whose ID isn't in the frame, or a second detection of the same tag.
- Recall in `small-tags` and `steep-angle` depends on apparent size. A steeply tilted tag can look narrower than
  `min_tag_px` even when its long side is well above it.
- The frames are synthetic and the timings come from one desktop CPU. Measure on your own camera and hardware
  (see `napa-benchmark` in [the guide](../docs/guide.md)) before relying on a specific number.
