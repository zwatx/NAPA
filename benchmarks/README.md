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

# Pin to one idle core so other processes don't skew timings (Linux shown). About 45 minutes.
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

Standard AprilTag (`quad_decimate=2`) against Napa, from the checked-in 2026-10-18 runs on one pinned core of an
Intel i7-12700H. p50 is the median detector call across all frames and repetitions in a row.

#### 640x480

| Scene | Frames | Tags | AprilTag p50 | Napa p50 | Speedup (median / worst frame) | Found (AprilTag → Napa) | False IDs (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 5.39 ms | 0.05 ms | 87.1× / 9.1× | — | 0 → 0 |
| clutter-only | 50 | 0 | 3.65 ms | 0.27 ms | 11.9× / 4.8× | — | 0 → 0 |
| decoys | 50 | 0 | 3.97 ms | 0.22 ms | 16.6× / 4.9× | — | 0 → 1 |
| one-tag | 50 | 50 | 4.70 ms | 0.14 ms | 34.1× / 22.6× | 46/50 → 50/50 | 0 → 0 |
| many-tags | 50 | 249 | 3.42 ms | 0.31 ms | 11.3× / 4.9× | 178/249 → 239/249 | 0 → 0 |
| small-tags | 50 | 131 | 4.18 ms | 0.33 ms | 11.6× / 6.1× | 0/131 → 131/131 | 0 → 0 |
| large-tags | 50 | 50 | 3.41 ms | 0.25 ms | 14.2× / 5.9× | 50/50 → 50/50 | 0 → 0 |
| steep-angle | 50 | 109 | 4.03 ms | 0.28 ms | 12.5× / 5.4× | 71/109 → 81/109 | 0 → 0 |
| blur | 50 | 96 | 3.55 ms | 0.27 ms | 11.3× / 5.4× | 82/96 → 95/96 | 0 → 0 |
| motion-blur | 50 | 110 | 3.77 ms | 0.27 ms | 12.4× / 5.6× | 86/110 → 108/110 | 0 → 0 |
| noise | 50 | 110 | 4.06 ms | 0.34 ms | 12.0× / 5.0× | 94/110 → 107/110 | 0 → 0 |
| uneven-light | 50 | 98 | 3.68 ms | 0.30 ms | 12.0× / 5.9× | 82/98 → 95/98 | 0 → 0 |
| low-contrast | 50 | 96 | 4.89 ms | 0.15 ms | 34.6× / 17.5× | 70/96 → 84/96 | 0 → 0 |
| underexposed | 50 | 110 | 3.80 ms | 0.28 ms | 13.1× / 7.8× | 96/110 → 107/110 | 0 → 0 |
| overexposed | 50 | 100 | 1.27 ms | 0.23 ms | 5.0× / 3.5× | 82/100 → 98/100 | 0 → 0 |
| jpeg | 50 | 104 | 1.39 ms | 0.30 ms | 5.2× / 3.2× | 92/104 → 101/104 | 0 → 0 |
| busy-background | 50 | 102 | 3.84 ms | 0.30 ms | 12.1× / 6.1× | 89/102 → 97/102 | 0 → 0 |
| frame-edge | 50 | 102 | 3.51 ms | 0.23 ms | 14.3× / 6.3× | 74/102 → 98/102 | 0 → 0 |
| noise-free | 50 | 98 | 0.68 ms | 0.15 ms | 4.6× / 3.4× | 83/98 → 98/98 | 0 → 0 |

#### 1280x720

| Scene | Frames | Tags | AprilTag p50 | Napa p50 | Speedup (median / worst frame) | Found (AprilTag → Napa) | False IDs (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 17.28 ms | 0.17 ms | 100.3× / 7.0× | — | 0 → 0 |
| clutter-only | 50 | 0 | 11.18 ms | 0.79 ms | 13.3× / 4.0× | — | 0 → 0 |
| decoys | 50 | 0 | 14.85 ms | 0.37 ms | 38.8× / 5.3× | — | 0 → 0 |
| one-tag | 50 | 50 | 15.43 ms | 0.27 ms | 55.8× / 39.1× | 50/50 → 50/50 | 0 → 0 |
| many-tags | 50 | 354 | 11.35 ms | 0.77 ms | 15.2× / 6.2× | 354/354 → 342/354 | 0 → 0 |
| small-tags | 50 | 120 | 13.61 ms | 0.48 ms | 27.9× / 6.7× | 0/120 → 120/120 | 0 → 0 |
| large-tags | 50 | 50 | 11.80 ms | 0.54 ms | 21.3× / 7.3× | 50/50 → 50/50 | 0 → 0 |
| steep-angle | 50 | 112 | 12.10 ms | 0.59 ms | 20.0× / 6.9× | 109/112 → 80/112 | 0 → 0 |
| blur | 50 | 89 | 11.51 ms | 0.56 ms | 20.0× / 6.4× | 89/89 → 87/89 | 0 → 0 |
| motion-blur | 50 | 94 | 11.35 ms | 0.88 ms | 12.0× / 6.7× | 94/94 → 93/94 | 0 → 0 |
| noise | 50 | 104 | 10.93 ms | 0.82 ms | 15.2× / 6.1× | 104/104 → 101/104 | 0 → 0 |
| uneven-light | 50 | 103 | 12.46 ms | 0.64 ms | 18.7× / 7.3× | 103/103 → 99/103 | 0 → 0 |
| low-contrast | 50 | 100 | 15.34 ms | 0.29 ms | 52.9× / 29.1× | 95/100 → 87/100 | 0 → 0 |
| underexposed | 50 | 99 | 12.85 ms | 0.48 ms | 24.8× / 9.1× | 99/99 → 93/99 | 0 → 0 |
| overexposed | 50 | 112 | 1.99 ms | 0.35 ms | 5.9× / 4.0× | 112/112 → 110/112 | 0 → 0 |
| jpeg | 50 | 100 | 2.96 ms | 0.49 ms | 6.0× / 3.9× | 100/100 → 97/100 | 0 → 0 |
| busy-background | 50 | 87 | 12.10 ms | 0.51 ms | 22.7× / 8.4× | 87/87 → 85/87 | 0 → 0 |
| frame-edge | 50 | 105 | 12.35 ms | 0.57 ms | 19.2× / 7.0× | 103/105 → 102/105 | 0 → 0 |
| noise-free | 50 | 104 | 1.54 ms | 0.27 ms | 5.9× / 4.5× | 104/104 → 104/104 | 0 → 0 |

#### 1920x1080

| Scene | Frames | Tags | AprilTag p50 | Napa p50 | Speedup (median / worst frame) | Found (AprilTag → Napa) | False IDs (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 44.17 ms | 0.35 ms | 119.1× / 9.1× | — | 0 → 0 |
| clutter-only | 50 | 0 | 29.05 ms | 0.83 ms | 23.8× / 4.5× | — | 0 → 0 |
| decoys | 50 | 0 | 38.08 ms | 0.60 ms | 60.4× / 6.7× | — | 0 → 0 |
| one-tag | 50 | 50 | 37.09 ms | 0.49 ms | 73.9× / 53.5× | 50/50 → 50/50 | 0 → 0 |
| many-tags | 50 | 303 | 26.48 ms | 1.28 ms | 22.3× / 6.8× | 303/303 → 289/303 | 0 → 0 |
| small-tags | 50 | 122 | 30.31 ms | 0.87 ms | 35.9× / 8.0× | 0/122 → 121/122 | 0 → 0 |
| large-tags | 50 | 50 | 29.82 ms | 0.89 ms | 35.3× / 8.9× | 50/50 → 50/50 | 0 → 0 |
| steep-angle | 50 | 96 | 31.05 ms | 0.99 ms | 32.2× / 7.5× | 88/96 → 73/96 | 0 → 0 |
| blur | 50 | 97 | 28.94 ms | 0.98 ms | 29.5× / 7.1× | 97/97 → 95/97 | 0 → 0 |
| motion-blur | 50 | 95 | 30.78 ms | 0.93 ms | 35.0× / 7.3× | 95/95 → 94/95 | 0 → 0 |
| noise | 50 | 100 | 30.10 ms | 1.93 ms | 16.3× / 6.6× | 100/100 → 97/100 | 0 → 0 |
| uneven-light | 50 | 101 | 30.34 ms | 1.06 ms | 28.9× / 8.5× | 101/101 → 99/101 | 0 → 0 |
| low-contrast | 50 | 113 | 37.58 ms | 0.62 ms | 59.9× / 43.1× | 108/113 → 94/113 | 0 → 0 |
| underexposed | 50 | 95 | 37.59 ms | 0.82 ms | 44.4× / 10.4× | 95/95 → 89/95 | 0 → 0 |
| overexposed | 50 | 100 | 9.53 ms | 0.92 ms | 6.8× / 4.1× | 100/100 → 99/100 | 0 → 0 |
| jpeg | 50 | 100 | 8.12 ms | 1.07 ms | 6.0× / 3.8× | 100/100 → 96/100 | 0 → 0 |
| busy-background | 50 | 97 | 30.99 ms | 0.98 ms | 31.3× / 11.0× | 97/97 → 96/97 | 0 → 0 |
| frame-edge | 50 | 88 | 31.73 ms | 0.81 ms | 42.6× / 7.3× | 88/88 → 85/88 | 0 → 0 |
| noise-free | 50 | 108 | 3.33 ms | 0.55 ms | 6.1× / 4.8× | 108/108 → 108/108 | 0 → 0 |

The JSON reports also hold full-resolution AprilTag (`quad_decimate=1`), the 95th/99th percentiles, corner error
by scene, and the obstruction sweep.

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
