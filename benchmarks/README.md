# NAPA benchmarks

`performance_matrix.py` compares NAPA with AprilTag 3 as WPILib builds and configures it (`robotpy-apriltag`, at its
default `quad_decimate=2` and at full resolution) on a large set of generated grayscale frames, and
`cpu_comparison.py` runs the same frames through upstream AprilTag and other CPU detectors (see
[Other CPU detectors](#other-cpu-detectors)). Both measure detector calls only. Camera delivery, colour conversion,
and pose solving are outside the timed region.

The tables below use the current native source from the
[CPU optimization comparison](results/2026-10-29_cpu_optimization/report.md). NAPA32 is the unchanged default
(`min_tag_px=32`); NAPA16 is tuned (`min_tag_px=16`). Both keep full-resolution corner refinement. The detailed
report records all versions, effective settings, per-frame losses and the six single-core/multicore resource
comparisons. The [root README](../README.md#compared-with-other-detectors) summarizes latency and resources.

## What's in the corpus

Every frame is rendered on its own: no image is reused, and none is a copy of another with a different label. Each tag
gets a random ID from all 587, a random position, any in-plane angle, a random 3D tilt (and therefore perspective),
and random black and white levels. Tags never overlap, and each keeps its white quiet zone inside the frame.

Tags are rendered with exact sub-pixel ground truth: every pixel is the mean of a 4×4 grid of point samples, as a
sensor integrates light. So the benchmark scores whether each tag was found, any invented ID, and how far each corner
landed from where it really is.

| Scenario | What varies |
| --- | --- |
| `empty` | A uniform frame at random brightness, with light noise. No tags. |
| `clutter-only` | Shapes, text, checkerboards, or smooth natural texture. No tags. |
| `decoys` | Black-bordered squares with random payloads: tag-shaped, but not valid codes. |
| `one-tag` | One tag on a plain background. |
| `many-tags` | 2–12 tags of mixed sizes on mixed backgrounds. |
| `small-tags` | 1–4 tags with a 32–53 px black square, at and just above NAPA's default `min_tag_px`. |
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

Unless a scenario says otherwise, positive frames hold 1–3 tags on a random background, and tag size scales with the
frame (48–170 px at 480 lines). Every scenario except `noise-free` adds at least light sensor noise (σ ≈ 1.5 grey
levels), as a real sensor would. The default run renders 50 frames per scenario at each of 640×480, 1280×720, and
1920×1080: 3,000 frames in all.

A separate obstruction sweep hides part of one tag per frame with a white or grey patch. The patch sits over the
payload centre, along one border, or over a corner, and covers 0–40% of the black square. There are 24 random tags for
each combination (864 frames), at the smallest resolution.

A range sweep measures how small a tag each detector can still find, which sets how far away it can see one. It
renders one tag per frame with a black square of exactly 12 to 96 px, at a random ID and angle, up to 15° of tilt, and
light noise (σ ≈ 1.5). The default sweep uses 50 tags at each size, at the smallest resolution; the
[Range table below](#range) uses a separate 30-tags-per-size replay.

## Timing

Backends take turns on each frame in a shuffled order, ten timed calls each after one untimed warm-up call. The report
gives every backend's latency percentiles per scenario and resolution. It also gives NAPA's speedup **per frame**:
AprilTag's median time on a frame divided by NAPA's median time on the same frame. "Worst frame" is the smallest of
those ratios, so a single slow frame can't hide behind a good median.

## Running it

```bash
python -m pip install -e ".[headless,dev,benchmark]"

# Pin to one idle core so other processes don't skew timings (Linux shown). About 90 minutes.
taskset -c 2 python benchmarks/performance_matrix.py --output benchmarks/results/$(date +%F)_detector_matrix.json
```

Useful options:

- `--scenarios noise,jpeg`, `--resolutions 1280x720`, and `--frames 10` select a quick subset.
- `--backends napa,apriltag,apriltag-full-res,opencv-aruco` chooses what to compare (OpenCV is off by default).
- `--occlusion-cases 0` skips the obstruction sweep, and `--range-cases 0` the range sweep.
- `--frames 0` runs only the sweeps. The range results came from
  `--frames 0 --resolutions 640x480 --occlusion-cases 0`. They count tags found, not time, so any machine gives the
  same numbers.
- `--seed` changes every frame. Each frame's seed depends only on the run seed, its scenario, resolution, and index,
  so any frame can be re-rendered on its own with `render_frame()`.

The script prints the per-scene tables below and writes two files:

- `*.json`: run configuration, machine details, per-scenario results (frames, tags, found, false IDs, latency
  percentiles, corner error), per-frame speedup statistics, and the obstruction sweep.
- `*.csv`: one row per frame and backend, with expected and detected IDs, corner errors, every raw timing sample, and
  the scenario's random parameters (`detail_*` columns). Use it to find and re-render any frame a detector missed.

## Latest results

From [final_latency.csv](results/2026-10-29_cpu_optimization/final_latency.csv): 149 cached diagnostic frames on
one verified P-core (CPU2) of an Intel Core i7-12700H, Ubuntu 26.04. Each backend has one warm-up and three shuffled
timed calls per frame. The set covers all 20 corpus scenarios and four stress cases at three resolutions, with
extra historical NAPA16 losses deliberately included. Scenario counts are small and unequal: these are diagnostic
measurements, not a new full-corpus latency run. Full-quality and range scopes are described separately below.

All detectors receive the same grayscale tag36h11 images. NAPA32 is default; NAPA16 is tuned. Other labels match
those in the root README. OpenCV/RapidTag defaults do not refine corners; SUBPIX variants are separate. ArUco3's
minimum-size labels account for half-scale detection. WPILib d2 keeps larger candidate regions and a stricter
quad-angle cutoff than upstream AprilTag d2; it consequently misses more small and tilted tags.

For each timing cell, the first value is the median of all detector calls for that scenario/resolution. Brackets
show the median paired per-frame speedup of **NAPA16** versus that detector. NAPA32 also has a speedup in brackets,
so values below one there mean the default is faster than NAPA16. Geometric scoring is unchanged; corner order
and pixel origins are normalized by the existing adapters. Counts below cover only these same diagnostic frames.

### 640x480: median call (NAPA16 speedup in brackets)

| Scene | Frames | Tags | NAPA32 (default) | NAPA16 (tuned) | WPILib AprilTag d2 | AprilTag d2 | AprilTag d1 | Locus standard | Locus d2 | OpenCV ArUco | OpenCV SUBPIX | ArUco3 full | ArUco3 min≈64 px | ArUco3 min≈32 px | ArUco Nano | RapidTag | RapidTag SUBPIX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blur | 1 | 2 | 0.422 ms (0.48×) | 0.878 ms | 3.957 ms (4.51×) | 5.700 ms (6.49×) | 19.515 ms (22.23×) | 3.660 ms (4.17×) | 2.704 ms (3.08×) | 12.242 ms (13.95×) | 14.101 ms (16.07×) | 5.332 ms (6.07×) | 1.592 ms (1.81×) | 5.198 ms (5.92×) | 1.356 ms (1.55×) | 2.229 ms (2.54×) | 2.576 ms (2.94×) |
| busy-background | 1 | 2 | 0.381 ms (0.54×) | 0.701 ms | 3.240 ms (4.62×) | 4.925 ms (7.03×) | 15.392 ms (21.96×) | 3.582 ms (5.11×) | 2.479 ms (3.54×) | 34.085 ms (48.64×) | 34.338 ms (49.00×) | 4.558 ms (6.50×) | 1.245 ms (1.78×) | 2.330 ms (3.32×) | 1.103 ms (1.57×) | 2.782 ms (3.97×) | 2.662 ms (3.80×) |
| clutter-only | 1 | 0 | 0.257 ms (0.36×) | 0.711 ms | 3.256 ms (4.58×) | 5.737 ms (8.07×) | 15.440 ms (21.72×) | 3.386 ms (4.76×) | 1.725 ms (2.43×) | 12.944 ms (18.21×) | 12.983 ms (18.26×) | 2.235 ms (3.14×) | 0.895 ms (1.26×) | 0.978 ms (1.38×) | 0.725 ms (1.02×) | 2.249 ms (3.16×) | 2.306 ms (3.24×) |
| dark-background | 1 | 1 | 0.187 ms (0.22×) | 0.832 ms | 2.655 ms (3.19×) | 3.661 ms (4.40×) | 10.955 ms (13.17×) | 4.514 ms (5.43×) | 1.810 ms (2.18×) | 3.835 ms (4.61×) | 3.874 ms (4.66×) | 3.068 ms (3.69×) | 0.961 ms (1.16×) | 1.489 ms (1.79×) | 1.380 ms (1.66×) | 2.763 ms (3.32×) | 2.763 ms (3.32×) |
| decoys | 1 | 0 | 0.109 ms (0.55×) | 0.197 ms | 4.164 ms (21.15×) | 4.762 ms (24.19×) | 19.560 ms (99.36×) | 2.307 ms (11.72×) | 1.133 ms (5.75×) | 9.072 ms (46.09×) | 8.977 ms (45.60×) | 4.281 ms (21.75×) | 4.186 ms (21.26×) | 4.245 ms (21.56×) | 1.609 ms (8.17×) | 1.408 ms (7.15×) | 1.396 ms (7.09×) |
| empty | 1 | 0 | 0.029 ms (0.40×) | 0.072 ms | 5.993 ms (82.83×) | 7.852 ms (108.51×) | 28.896 ms (399.34×) | 1.636 ms (22.62×) | 0.779 ms (10.77×) | 0.969 ms (13.39×) | 0.910 ms (12.57×) | 1.018 ms (14.07×) | 0.287 ms (3.97×) | 0.340 ms (4.70×) | 0.477 ms (6.59×) | 0.786 ms (10.86×) | 0.792 ms (10.95×) |
| frame-edge | 1 | 2 | 0.151 ms (0.56×) | 0.268 ms | 4.109 ms (15.31×) | 6.210 ms (23.14×) | 19.418 ms (72.35×) | 3.139 ms (11.69×) | 2.646 ms (9.86×) | 81.545 ms (303.82×) | 81.722 ms (304.48×) | 3.715 ms (13.84×) | 0.915 ms (3.41×) | 1.844 ms (6.87×) | 0.718 ms (2.68×) | 1.648 ms (6.14×) | 1.653 ms (6.16×) |
| jpeg | 1 | 3 | 0.151 ms (0.64×) | 0.237 ms | 0.945 ms (3.98×) | 1.129 ms (4.76×) | 2.815 ms (11.87×) | 1.997 ms (8.42×) | 1.716 ms (7.23×) | 6.548 ms (27.60×) | 6.678 ms (28.16×) | 1.943 ms (8.19×) | 1.020 ms (4.30×) | 1.017 ms (4.29×) | 0.541 ms (2.28×) | 1.420 ms (5.99×) | 1.435 ms (6.05×) |
| large-tags | 1 | 1 | 0.253 ms (0.27×) | 0.941 ms | 2.654 ms (2.82×) | 3.544 ms (3.76×) | 13.034 ms (13.84×) | 5.411 ms (5.75×) | 2.354 ms (2.50×) | 11.676 ms (12.40×) | 11.918 ms (12.66×) | 4.794 ms (5.09×) | 1.485 ms (1.58×) | 1.677 ms (1.78×) | 2.152 ms (2.29×) | 3.667 ms (3.90×) | 3.710 ms (3.94×) |
| low-contrast | 1 | 1 | 0.070 ms (0.46×) | 0.153 ms | 4.551 ms (29.83×) | 6.195 ms (40.61×) | 27.262 ms (178.70×) | 2.798 ms (18.34×) | 1.067 ms (6.99×) | 1.400 ms (9.17×) | 1.376 ms (9.02×) | 1.438 ms (9.43×) | 0.329 ms (2.16×) | 0.867 ms (5.68×) | 0.433 ms (2.84×) | 0.886 ms (5.80×) | 0.903 ms (5.92×) |
| many-tags | 1 | 4 | 0.240 ms (0.57×) | 0.421 ms | 3.821 ms (9.08×) | 5.346 ms (12.70×) | 17.946 ms (42.65×) | 4.015 ms (9.54×) | 2.771 ms (6.59×) | 77.305 ms (183.71×) | 76.911 ms (182.78×) | 3.237 ms (7.69×) | 1.828 ms (4.34×) | 2.115 ms (5.03×) | 0.864 ms (2.05×) | 2.074 ms (4.93×) | 2.032 ms (4.83×) |
| motion-blur | 1 | 3 | 0.237 ms (0.66×) | 0.360 ms | 4.576 ms (12.69×) | 6.065 ms (16.83×) | 21.501 ms (59.65×) | 3.400 ms (9.43×) | 2.886 ms (8.01×) | 6.832 ms (18.95×) | 6.841 ms (18.98×) | 4.029 ms (11.18×) | 1.340 ms (3.72×) | 3.300 ms (9.15×) | 0.582 ms (1.62×) | 1.373 ms (3.81×) | 1.450 ms (4.02×) |
| noise | 5 | 9 | 0.274 ms (0.47×) | 0.531 ms | 3.907 ms (5.44×) | 7.024 ms (9.13×) | 22.176 ms (32.66×) | 3.928 ms (4.79×) | 1.911 ms (3.27×) | 18.260 ms (35.46×) | 18.294 ms (35.11×) | 11.889 ms (22.71×) | 2.735 ms (3.93×) | 2.913 ms (4.15×) | 2.371 ms (4.68×) | 9.371 ms (18.49×) | 9.498 ms (18.95×) |
| noise-free | 1 | 2 | 0.113 ms (0.64×) | 0.177 ms | 0.668 ms (3.78×) | 0.758 ms (4.29×) | 1.804 ms (10.20×) | 1.424 ms (8.05×) | 1.209 ms (6.84×) | 4.732 ms (26.75×) | 4.708 ms (26.62×) | 1.947 ms (11.01×) | 1.258 ms (7.12×) | 1.272 ms (7.19×) | 0.502 ms (2.84×) | 1.119 ms (6.33×) | 1.106 ms (6.25×) |
| one-tag | 1 | 1 | 0.084 ms (0.49×) | 0.172 ms | 4.456 ms (25.97×) | 5.031 ms (29.32×) | 18.904 ms (110.18×) | 3.017 ms (17.58×) | 1.423 ms (8.29×) | 1.161 ms (6.77×) | 1.157 ms (6.74×) | 1.202 ms (7.01×) | 0.487 ms (2.84×) | 0.488 ms (2.84×) | 0.425 ms (2.48×) | 1.026 ms (5.98×) | 1.099 ms (6.41×) |
| overexposed | 1 | 3 | 0.144 ms (0.69×) | 0.210 ms | 0.587 ms (2.80×) | 0.662 ms (3.16×) | 1.726 ms (8.23×) | 1.621 ms (7.73×) | 1.530 ms (7.30×) | 3.892 ms (18.56×) | 3.873 ms (18.47×) | 2.927 ms (13.96×) | 0.373 ms (1.78×) | 2.211 ms (10.54×) | 0.430 ms (2.05×) | 1.078 ms (5.14×) | 1.037 ms (4.95×) |
| small-tags | 1 | 3 | 0.359 ms (0.46×) | 0.786 ms | 3.704 ms (4.71×) | 6.583 ms (8.37×) | 18.869 ms (24.00×) | 3.487 ms (4.44×) | 2.503 ms (3.18×) | 27.299 ms (34.73×) | 26.578 ms (33.81×) | 4.363 ms (5.55×) | 1.102 ms (1.40×) | 1.882 ms (2.39×) | 1.083 ms (1.38×) | 2.832 ms (3.60×) | 2.855 ms (3.63×) |
| steep-angle | 1 | 3 | 0.304 ms (0.32×) | 0.962 ms | 2.836 ms (2.95×) | 3.842 ms (3.99×) | 11.785 ms (12.25×) | 5.620 ms (5.84×) | 2.652 ms (2.76×) | 5.945 ms (6.18×) | 5.906 ms (6.14×) | 4.754 ms (4.94×) | 1.946 ms (2.02×) | 2.935 ms (3.05×) | 1.795 ms (1.87×) | 3.523 ms (3.66×) | 3.526 ms (3.67×) |
| stress-clutter-noisy | 5 | 0 | 0.165 ms (0.35×) | 0.453 ms | 4.584 ms (10.86×) | 7.331 ms (15.87×) | 26.560 ms (57.62×) | 2.790 ms (5.45×) | 1.144 ms (2.55×) | 17.876 ms (42.01×) | 18.047 ms (42.42×) | 10.151 ms (23.03×) | 1.445 ms (2.50×) | 1.686 ms (3.99×) | 2.086 ms (4.78×) | 7.282 ms (15.80×) | 7.358 ms (15.96×) |
| stress-dense | 1 | 21 | 0.578 ms (0.50×) | 1.153 ms | 2.536 ms (2.20×) | 4.358 ms (3.78×) | 9.702 ms (8.42×) | 8.342 ms (7.24×) | 6.317 ms (5.48×) | 23.595 ms (20.47×) | 23.822 ms (20.66×) | 14.377 ms (12.47×) | 1.276 ms (1.11×) | 11.573 ms (10.04×) | 2.423 ms (2.10×) | 4.234 ms (3.67×) | 4.308 ms (3.74×) |
| stress-heavy-noise | 5 | 9 | 0.310 ms (0.46×) | 0.590 ms | 2.432 ms (4.69×) | 3.832 ms (5.23×) | 9.616 ms (17.94×) | 4.238 ms (6.83×) | 1.812 ms (3.07×) | 25.396 ms (32.19×) | 25.360 ms (31.93×) | 13.909 ms (26.00×) | 2.825 ms (4.98×) | 3.457 ms (5.19×) | 2.453 ms (4.54×) | 10.524 ms (20.44×) | 10.523 ms (20.16×) |
| stress-tiny | 1 | 1 | 0.214 ms (0.20×) | 1.054 ms | 3.093 ms (2.94×) | 4.066 ms (3.86×) | 12.522 ms (11.88×) | 5.704 ms (5.41×) | 1.971 ms (1.87×) | 5.370 ms (5.10×) | 5.388 ms (5.11×) | 3.317 ms (3.15×) | 1.259 ms (1.20×) | 1.409 ms (1.34×) | 1.843 ms (1.75×) | 3.483 ms (3.31×) | 3.482 ms (3.30×) |
| underexposed | 1 | 1 | 0.109 ms (0.73×) | 0.150 ms | 5.746 ms (38.27×) | 6.742 ms (44.90×) | 28.280 ms (188.34×) | 3.187 ms (21.23×) | 1.419 ms (9.45×) | 3.090 ms (20.58×) | 3.191 ms (21.25×) | 1.269 ms (8.45×) | 0.571 ms (3.80×) | 0.562 ms (3.74×) | 0.633 ms (4.22×) | 0.986 ms (6.56×) | 1.044 ms (6.95×) |
| uneven-light | 1 | 3 | 0.466 ms (0.46×) | 1.016 ms | 4.038 ms (3.98×) | 7.696 ms (7.58×) | 24.939 ms (24.56×) | 5.780 ms (5.69×) | 3.684 ms (3.63×) | 25.301 ms (24.91×) | 25.704 ms (25.31×) | 4.613 ms (4.54×) | 2.884 ms (2.84×) | 2.791 ms (2.75×) | 1.645 ms (1.62×) | 3.326 ms (3.27×) | 3.748 ms (3.69×) |

### 640x480: tags found (wrong IDs in brackets)

| Scene | Tags | NAPA32 (default) | NAPA16 (tuned) | WPILib AprilTag d2 | AprilTag d2 | AprilTag d1 | Locus standard | Locus d2 | OpenCV ArUco | OpenCV SUBPIX | ArUco3 full | ArUco3 min≈64 px | ArUco3 min≈32 px | ArUco Nano | RapidTag | RapidTag SUBPIX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blur | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 |
| busy-background | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| clutter-only | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dark-background | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| decoys | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| empty | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| frame-edge | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 |
| jpeg | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| large-tags | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| low-contrast | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| many-tags | 4 | 4 | 4 | 2 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 4 | 4 | 4 | 4 |
| motion-blur | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | 3 |
| noise | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 8 | 8 | 9 | 8 | 8 |
| noise-free | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 0 | 0 |
| one-tag | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| overexposed | 3 | 3 | 3 | 1 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 0 | 3 | 3 | 3 | 3 |
| small-tags | 3 | 3 | 3 | 0 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 0 | 2 | 3 | 2 | 2 |
| steep-angle | 3 | 3 | 3 | 1 | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 1 | 3 | 3 | 3 | 3 |
| stress-clutter-noisy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (2) | 0 (2) |
| stress-dense | 21 | 21 | 21 | 2 | 21 | 21 | 21 | 21 | 21 | 21 | 20 | 0 | 20 | 21 | 15 | 15 |
| stress-heavy-noise | 9 | 9 | 9 | 9 | 9 | 9 | 7 | 9 | 9 | 9 | 9 | 8 | 9 | 9 | 9 | 9 |
| stress-tiny | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| underexposed | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| uneven-light | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

### 1280x720: median call (NAPA16 speedup in brackets)

| Scene | Frames | Tags | NAPA32 (default) | NAPA16 (tuned) | WPILib AprilTag d2 | AprilTag d2 | AprilTag d1 | Locus standard | Locus d2 | OpenCV ArUco | OpenCV SUBPIX | ArUco3 full | ArUco3 min≈64 px | ArUco3 min≈32 px | ArUco Nano | RapidTag | RapidTag SUBPIX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blur | 2 | 5 | 0.878 ms (0.39×) | 2.216 ms | 11.196 ms (5.22×) | 16.320 ms (7.55×) | 52.564 ms (24.45×) | 9.876 ms (4.47×) | 6.525 ms (3.03×) | 26.672 ms (12.94×) | 26.787 ms (12.85×) | 9.420 ms (4.40×) | 3.431 ms (1.58×) | 3.890 ms (1.76×) | 2.597 ms (1.22×) | 6.163 ms (2.85×) | 6.202 ms (2.93×) |
| busy-background | 1 | 3 | 0.316 ms (0.45×) | 0.700 ms | 10.928 ms (15.61×) | 15.662 ms (22.37×) | 59.338 ms (84.77×) | 9.977 ms (14.25×) | 4.959 ms (7.08×) | 153.160 ms (218.80×) | 140.585 ms (200.84×) | 15.817 ms (22.60×) | 3.869 ms (5.53×) | 7.430 ms (10.61×) | 2.211 ms (3.16×) | 4.825 ms (6.89×) | 5.129 ms (7.33×) |
| clutter-only | 1 | 0 | 0.106 ms (0.25×) | 0.429 ms | 4.810 ms (11.20×) | 7.777 ms (18.11×) | 26.381 ms (61.45×) | 7.114 ms (16.57×) | 3.504 ms (8.16×) | 8.492 ms (19.78×) | 8.463 ms (19.71×) | 8.656 ms (20.16×) | 6.056 ms (14.11×) | 6.098 ms (14.20×) | 1.122 ms (2.61×) | 3.824 ms (8.91×) | 3.805 ms (8.86×) |
| dark-background | 1 | 3 | 0.440 ms (0.51×) | 0.864 ms | 8.844 ms (10.23×) | 11.642 ms (13.47×) | 49.317 ms (57.06×) | 12.730 ms (14.73×) | 5.105 ms (5.91×) | 5.643 ms (6.53×) | 5.681 ms (6.57×) | 4.435 ms (5.13×) | 4.097 ms (4.74×) | 4.011 ms (4.64×) | 1.937 ms (2.24×) | 3.105 ms (3.59×) | 3.023 ms (3.50×) |
| decoys | 1 | 0 | 0.198 ms (0.37×) | 0.534 ms | 6.839 ms (12.80×) | 9.996 ms (18.71×) | 35.404 ms (66.26×) | 7.641 ms (14.30×) | 3.784 ms (7.08×) | 18.242 ms (34.14×) | 18.355 ms (34.35×) | 14.380 ms (26.91×) | 6.177 ms (11.56×) | 9.024 ms (16.89×) | 3.566 ms (6.67×) | 3.998 ms (7.48×) | 3.831 ms (7.17×) |
| empty | 1 | 0 | 0.062 ms (0.24×) | 0.256 ms | 3.131 ms (12.25×) | 4.243 ms (16.60×) | 13.816 ms (54.05×) | 7.784 ms (30.45×) | 2.714 ms (10.62×) | 2.423 ms (9.48×) | 2.443 ms (9.56×) | 2.625 ms (10.27×) | 0.822 ms (3.22×) | 0.791 ms (3.09×) | 0.591 ms (2.31×) | 2.114 ms (8.27×) | 2.231 ms (8.73×) |
| frame-edge | 1 | 1 | 0.169 ms (0.33×) | 0.506 ms | 14.201 ms (28.04×) | 16.599 ms (32.77×) | 71.085 ms (140.35×) | 7.983 ms (15.76×) | 3.048 ms (6.02×) | 4.252 ms (8.40×) | 4.149 ms (8.19×) | 4.253 ms (8.40×) | 1.647 ms (3.25×) | 1.616 ms (3.19×) | 1.245 ms (2.46×) | 3.599 ms (7.11×) | 3.778 ms (7.46×) |
| jpeg | 2 | 3 | 0.444 ms (0.31×) | 1.467 ms | 3.131 ms (2.94×) | 5.670 ms (4.68×) | 12.275 ms (10.62×) | 5.740 ms (5.25×) | 3.519 ms (3.77×) | 32.870 ms (36.00×) | 32.731 ms (35.81×) | 8.940 ms (10.70×) | 2.435 ms (2.55×) | 4.029 ms (5.24×) | 1.518 ms (1.56×) | 5.840 ms (5.35×) | 5.679 ms (5.21×) |
| large-tags | 1 | 1 | 0.131 ms (0.32×) | 0.413 ms | 14.942 ms (36.15×) | 18.369 ms (44.44×) | 70.808 ms (171.29×) | 8.385 ms (20.28×) | 3.887 ms (9.40×) | 10.191 ms (24.65×) | 9.966 ms (24.11×) | 8.284 ms (20.04×) | 1.615 ms (3.91×) | 4.489 ms (10.86×) | 1.314 ms (3.18×) | 3.136 ms (7.59×) | 2.849 ms (6.89×) |
| low-contrast | 1 | 2 | 0.197 ms (0.39×) | 0.509 ms | 16.616 ms (32.66×) | 19.421 ms (38.17×) | 73.051 ms (143.58×) | 8.718 ms (17.14×) | 4.412 ms (8.67×) | 7.364 ms (14.47×) | 7.315 ms (14.38×) | 4.650 ms (9.14×) | 2.192 ms (4.31×) | 2.220 ms (4.36×) | 1.377 ms (2.71×) | 3.219 ms (6.33×) | 3.099 ms (6.09×) |
| many-tags | 1 | 10 | 0.513 ms (0.51×) | 1.002 ms | 11.674 ms (11.65×) | 14.272 ms (14.25×) | 57.908 ms (57.81×) | 9.536 ms (9.52×) | 6.466 ms (6.46×) | 18.848 ms (18.81×) | 19.052 ms (19.02×) | 11.237 ms (11.22×) | 6.556 ms (6.54×) | 10.524 ms (10.51×) | 2.361 ms (2.36×) | 5.900 ms (5.89×) | 5.910 ms (5.90×) |
| motion-blur | 1 | 2 | 0.937 ms (0.40×) | 2.367 ms | 9.654 ms (4.08×) | 15.217 ms (6.43×) | 50.755 ms (21.44×) | 9.638 ms (4.07×) | 4.784 ms (2.02×) | 21.929 ms (9.26×) | 21.913 ms (9.26×) | 6.874 ms (2.90×) | 2.818 ms (1.19×) | 4.137 ms (1.75×) | 2.608 ms (1.10×) | 7.492 ms (3.16×) | 7.242 ms (3.06×) |
| noise | 3 | 5 | 0.419 ms (0.34×) | 1.187 ms | 11.524 ms (9.71×) | 18.586 ms (15.19×) | 72.109 ms (58.94×) | 7.423 ms (6.14×) | 3.322 ms (2.80×) | 38.061 ms (31.11×) | 38.214 ms (31.23×) | 30.658 ms (25.06×) | 5.288 ms (4.62×) | 8.676 ms (7.09×) | 6.468 ms (5.29×) | 25.245 ms (20.63×) | 24.939 ms (20.38×) |
| noise-free | 1 | 3 | 0.202 ms (0.42×) | 0.483 ms | 1.849 ms (3.82×) | 1.871 ms (3.87×) | 5.429 ms (11.23×) | 3.411 ms (7.06×) | 3.098 ms (6.41×) | 9.858 ms (20.40×) | 9.820 ms (20.32×) | 8.016 ms (16.58×) | 2.915 ms (6.03×) | 3.848 ms (7.96×) | 2.003 ms (4.14×) | 3.284 ms (6.80×) | 3.351 ms (6.93×) |
| one-tag | 1 | 1 | 0.143 ms (0.38×) | 0.377 ms | 14.489 ms (38.43×) | 17.119 ms (45.41×) | 63.862 ms (169.39×) | 5.599 ms (14.85×) | 3.432 ms (9.10×) | 8.693 ms (23.06×) | 8.854 ms (23.49×) | 4.276 ms (11.34×) | 1.268 ms (3.36×) | 1.300 ms (3.45×) | 1.286 ms (3.41×) | 2.877 ms (7.63×) | 2.777 ms (7.37×) |
| overexposed | 1 | 2 | 0.154 ms (0.46×) | 0.336 ms | 1.440 ms (4.28×) | 1.321 ms (3.93×) | 4.351 ms (12.93×) | 2.666 ms (7.92×) | 2.387 ms (7.09×) | 5.534 ms (16.45×) | 5.564 ms (16.53×) | 4.600 ms (13.67×) | 2.702 ms (8.03×) | 2.699 ms (8.02×) | 0.833 ms (2.47×) | 2.546 ms (7.57×) | 2.461 ms (7.31×) |
| small-tags | 1 | 2 | 0.226 ms (0.37×) | 0.616 ms | 12.335 ms (20.03×) | 16.027 ms (26.02×) | 67.719 ms (109.95×) | 8.000 ms (12.99×) | 3.365 ms (5.46×) | 7.765 ms (12.61×) | 7.790 ms (12.65×) | 5.819 ms (9.45×) | 1.891 ms (3.07×) | 4.556 ms (7.40×) | 2.285 ms (3.71×) | 5.192 ms (8.43×) | 5.052 ms (8.20×) |
| steep-angle | 1 | 3 | 0.295 ms (0.37×) | 0.797 ms | 11.784 ms (14.79×) | 15.318 ms (19.23×) | 64.369 ms (80.81×) | 9.784 ms (12.28×) | 5.000 ms (6.28×) | 163.615 ms (205.39×) | 153.564 ms (192.78×) | 41.175 ms (51.69×) | 3.544 ms (4.45×) | 38.201 ms (47.96×) | 1.959 ms (2.46×) | 5.065 ms (6.36×) | 5.326 ms (6.69×) |
| stress-clutter-noisy | 4 | 0 | 0.361 ms (0.24×) | 1.372 ms | 15.953 ms (12.02×) | 21.210 ms (15.49×) | 91.023 ms (64.34×) | 7.396 ms (5.30×) | 3.269 ms (2.26×) | 37.798 ms (25.10×) | 38.446 ms (24.69×) | 31.920 ms (21.99×) | 5.621 ms (3.91×) | 6.460 ms (4.79×) | 7.518 ms (5.39×) | 25.396 ms (16.18×) | 24.645 ms (15.52×) |
| stress-dense | 1 | 27 | 1.129 ms (0.42×) | 2.659 ms | 7.881 ms (2.96×) | 10.366 ms (3.90×) | 28.871 ms (10.86×) | 18.867 ms (7.10×) | 12.146 ms (4.57×) | 45.563 ms (17.14×) | 45.948 ms (17.28×) | 23.210 ms (8.73×) | 11.301 ms (4.25×) | 19.010 ms (7.15×) | 5.395 ms (2.03×) | 10.623 ms (4.00×) | 11.040 ms (4.15×) |
| stress-heavy-noise | 13 | 25 | 0.671 ms (0.40×) | 1.309 ms | 11.514 ms (7.68×) | 18.623 ms (8.99×) | 72.168 ms (32.81×) | 11.311 ms (7.23×) | 4.929 ms (2.81×) | 56.209 ms (40.72×) | 55.812 ms (40.38×) | 43.850 ms (32.16×) | 8.843 ms (5.81×) | 9.165 ms (6.59×) | 7.704 ms (6.43×) | 35.627 ms (25.46×) | 36.164 ms (25.41×) |
| stress-tiny | 1 | 1 | 0.473 ms (0.19×) | 2.536 ms | 8.175 ms (3.22×) | 10.004 ms (3.95×) | 35.320 ms (13.93×) | 14.857 ms (5.86×) | 5.052 ms (1.99×) | 11.076 ms (4.37×) | 11.057 ms (4.36×) | 8.542 ms (3.37×) | 3.035 ms (1.20×) | 3.338 ms (1.32×) | 4.399 ms (1.73×) | 8.842 ms (3.49×) | 9.070 ms (3.58×) |
| underexposed | 1 | 1 | 0.239 ms (0.41×) | 0.583 ms | 16.809 ms (28.84×) | 18.075 ms (31.01×) | 88.410 ms (151.67×) | 9.778 ms (16.77×) | 3.486 ms (5.98×) | 14.115 ms (24.21×) | 14.243 ms (24.43×) | 11.273 ms (19.34×) | 9.721 ms (16.68×) | 8.647 ms (14.83×) | 1.926 ms (3.30×) | 5.898 ms (10.12×) | 4.969 ms (8.52×) |
| uneven-light | 1 | 3 | 0.272 ms (0.50×) | 0.541 ms | 17.018 ms (31.46×) | 19.917 ms (36.82×) | 81.502 ms (150.66×) | 10.050 ms (18.58×) | 5.095 ms (9.42×) | 9.238 ms (17.08×) | 10.312 ms (19.06×) | 5.902 ms (10.91×) | 3.260 ms (6.03×) | 3.478 ms (6.43×) | 1.561 ms (2.89×) | 3.489 ms (6.45×) | 3.447 ms (6.37×) |

### 1280x720: tags found (wrong IDs in brackets)

| Scene | Tags | NAPA32 (default) | NAPA16 (tuned) | WPILib AprilTag d2 | AprilTag d2 | AprilTag d1 | Locus standard | Locus d2 | OpenCV ArUco | OpenCV SUBPIX | ArUco3 full | ArUco3 min≈64 px | ArUco3 min≈32 px | ArUco Nano | RapidTag | RapidTag SUBPIX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blur | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| busy-background | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 |
| clutter-only | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dark-background | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 | 2 | 3 | 0 | 0 |
| decoys | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| empty | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| frame-edge | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| jpeg | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| large-tags | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| low-contrast | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| many-tags | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 8 | 8 |
| motion-blur | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| noise | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 4 | 4 |
| noise-free | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 0 | 0 |
| one-tag | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| overexposed | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| small-tags | 2 | 2 | 2 | 0 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 0 | 2 | 2 | 2 | 2 |
| steep-angle | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 |
| stress-clutter-noisy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| stress-dense | 27 | 27 | 27 | 20 | 27 | 27 | 27 | 27 | 27 | 27 | 27 | 15 | 26 | 27 | 23 | 23 |
| stress-heavy-noise | 25 | 25 | 25 | 25 | 25 | 25 | 23 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 22 | 22 |
| stress-tiny | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| underexposed | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| uneven-light | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 0 | 0 |

### 1920x1080: median call (NAPA16 speedup in brackets)

| Scene | Frames | Tags | NAPA32 (default) | NAPA16 (tuned) | WPILib AprilTag d2 | AprilTag d2 | AprilTag d1 | Locus standard | Locus d2 | OpenCV ArUco | OpenCV SUBPIX | ArUco3 full | ArUco3 min≈64 px | ArUco3 min≈32 px | ArUco Nano | RapidTag | RapidTag SUBPIX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blur | 1 | 1 | 0.191 ms (0.18×) | 1.086 ms | 33.946 ms (31.25×) | 37.979 ms (34.97×) | 155.508 ms (143.18×) | 20.578 ms (18.95×) | 8.017 ms (7.38×) | 86.249 ms (79.41×) | 81.104 ms (74.67×) | 7.945 ms (7.32×) | 3.031 ms (2.79×) | 3.117 ms (2.87×) | 2.243 ms (2.07×) | 5.646 ms (5.20×) | 5.757 ms (5.30×) |
| busy-background | 1 | 2 | 1.769 ms (0.39×) | 4.502 ms | 28.345 ms (6.30×) | 41.334 ms (9.18×) | 143.415 ms (31.85×) | 20.710 ms (4.60×) | 11.443 ms (2.54×) | 58.772 ms (13.05×) | 60.865 ms (13.52×) | 18.787 ms (4.17×) | 7.681 ms (1.71×) | 10.471 ms (2.33×) | 5.088 ms (1.13×) | 18.085 ms (4.02×) | 17.880 ms (3.97×) |
| clutter-only | 1 | 0 | 1.013 ms (0.16×) | 6.335 ms | 20.443 ms (3.23×) | 25.805 ms (4.07×) | 103.160 ms (16.28×) | 36.463 ms (5.76×) | 12.578 ms (1.99×) | 27.342 ms (4.32×) | 27.900 ms (4.40×) | 24.868 ms (3.93×) | 6.942 ms (1.10×) | 7.622 ms (1.20×) | 8.642 ms (1.36×) | 24.751 ms (3.91×) | 24.718 ms (3.90×) |
| dark-background | 1 | 2 | 0.326 ms (0.27×) | 1.208 ms | 31.325 ms (25.92×) | 40.777 ms (33.75×) | 169.917 ms (140.62×) | 17.620 ms (14.58×) | 6.842 ms (5.66×) | 32.641 ms (27.01×) | 32.832 ms (27.17×) | 29.315 ms (24.26×) | 14.580 ms (12.07×) | 19.403 ms (16.06×) | 2.860 ms (2.37×) | 8.870 ms (7.34×) | 9.024 ms (7.47×) |
| decoys | 1 | 0 | 0.331 ms (0.34×) | 0.973 ms | 35.858 ms (36.84×) | 45.119 ms (46.35×) | 183.642 ms (188.66×) | 12.141 ms (12.47×) | 7.717 ms (7.93×) | 24.643 ms (25.32×) | 24.230 ms (24.89×) | 18.590 ms (19.10×) | 9.897 ms (10.17×) | 11.859 ms (12.18×) | 4.547 ms (4.67×) | 6.434 ms (6.61×) | 6.595 ms (6.77×) |
| empty | 1 | 0 | 0.084 ms (0.15×) | 0.549 ms | 2.529 ms (4.61×) | 2.309 ms (4.21×) | 8.396 ms (15.29×) | 19.322 ms (35.19×) | 4.225 ms (7.69×) | 5.421 ms (9.87×) | 5.553 ms (10.11×) | 5.849 ms (10.65×) | 1.789 ms (3.26×) | 1.830 ms (3.33×) | 1.316 ms (2.40×) | 4.794 ms (8.73×) | 4.909 ms (8.94×) |
| frame-edge | 1 | 2 | 0.346 ms (0.38×) | 0.921 ms | 30.759 ms (33.41×) | 36.780 ms (39.96×) | 150.438 ms (163.43×) | 15.620 ms (16.97×) | 8.377 ms (9.10×) | 63.064 ms (68.51×) | 63.256 ms (68.72×) | 14.030 ms (15.24×) | 3.509 ms (3.81×) | 6.183 ms (6.72×) | 2.583 ms (2.81×) | 6.618 ms (7.19×) | 6.442 ms (7.00×) |
| jpeg | 3 | 5 | 1.995 ms (0.35×) | 5.903 ms | 10.443 ms (1.80×) | 21.796 ms (3.69×) | 43.881 ms (7.60×) | 16.056 ms (2.75×) | 10.479 ms (1.93×) | 59.650 ms (11.32×) | 59.371 ms (11.29×) | 26.659 ms (4.68×) | 6.874 ms (1.23×) | 9.043 ms (1.53×) | 5.579 ms (0.94×) | 19.414 ms (3.29×) | 19.554 ms (3.32×) |
| large-tags | 1 | 1 | 1.419 ms (0.38×) | 3.744 ms | 29.770 ms (7.95×) | 43.236 ms (11.55×) | 169.043 ms (45.15×) | 18.983 ms (5.07×) | 10.346 ms (2.76×) | 68.325 ms (18.25×) | 68.448 ms (18.28×) | 24.724 ms (6.60×) | 6.336 ms (1.69×) | 9.757 ms (2.61×) | 5.797 ms (1.55×) | 16.361 ms (4.37×) | 15.827 ms (4.23×) |
| low-contrast | 6 | 13 | 0.334 ms (0.34×) | 0.992 ms | 39.453 ms (37.78×) | 46.210 ms (44.74×) | 192.709 ms (197.79×) | 20.483 ms (21.33×) | 8.916 ms (9.26×) | 13.863 ms (12.48×) | 12.759 ms (12.15×) | 10.533 ms (11.24×) | 4.230 ms (3.94×) | 4.933 ms (4.58×) | 3.011 ms (2.87×) | 7.439 ms (7.53×) | 7.318 ms (7.48×) |
| many-tags | 1 | 6 | 1.037 ms (0.22×) | 4.748 ms | 18.155 ms (3.82×) | 20.703 ms (4.36×) | 83.162 ms (17.52×) | 31.037 ms (6.54×) | 12.826 ms (2.70×) | 47.194 ms (9.94×) | 46.339 ms (9.76×) | 24.071 ms (5.07×) | 11.080 ms (2.33×) | 11.695 ms (2.46×) | 10.632 ms (2.24×) | 19.858 ms (4.18×) | 19.927 ms (4.20×) |
| motion-blur | 6 | 12 | 0.336 ms (0.30×) | 1.078 ms | 35.107 ms (31.82×) | 42.489 ms (38.62×) | 171.352 ms (157.08×) | 19.032 ms (18.41×) | 8.584 ms (8.63×) | 17.047 ms (15.41×) | 17.694 ms (15.30×) | 13.400 ms (11.65×) | 4.510 ms (4.20×) | 6.693 ms (5.73×) | 2.986 ms (2.77×) | 7.532 ms (7.11×) | 7.381 ms (7.36×) |
| noise | 10 | 21 | 0.939 ms (0.35×) | 2.420 ms | 35.012 ms (14.71×) | 49.768 ms (20.92×) | 198.330 ms (80.25×) | 14.875 ms (6.20×) | 7.681 ms (3.08×) | 70.479 ms (27.80×) | 69.043 ms (27.88×) | 61.978 ms (23.06×) | 9.034 ms (3.29×) | 10.867 ms (4.03×) | 14.118 ms (6.45×) | 44.146 ms (16.19×) | 44.914 ms (16.18×) |
| noise-free | 1 | 2 | 0.218 ms (0.25×) | 0.868 ms | 3.302 ms (3.80×) | 3.070 ms (3.54×) | 9.636 ms (11.10×) | 5.392 ms (6.21×) | 4.710 ms (5.42×) | 9.685 ms (11.15×) | 9.848 ms (11.34×) | 10.090 ms (11.62×) | 4.600 ms (5.30×) | 6.423 ms (7.40×) | 1.980 ms (2.28×) | 5.845 ms (6.73×) | 5.671 ms (6.53×) |
| one-tag | 1 | 1 | 0.225 ms (0.31×) | 0.722 ms | 44.162 ms (61.19×) | 52.080 ms (72.16×) | 217.117 ms (300.83×) | 18.961 ms (26.27×) | 7.476 ms (10.36×) | 9.168 ms (12.70×) | 8.950 ms (12.40×) | 7.335 ms (10.16×) | 2.163 ms (3.00×) | 2.182 ms (3.02×) | 1.756 ms (2.43×) | 5.483 ms (7.60×) | 5.279 ms (7.31×) |
| overexposed | 1 | 1 | 1.564 ms (0.23×) | 6.690 ms | 22.538 ms (3.37×) | 29.159 ms (4.36×) | 91.108 ms (13.62×) | 35.246 ms (5.27×) | 12.522 ms (1.87×) | 26.693 ms (3.99×) | 27.081 ms (4.05×) | 24.239 ms (3.62×) | 9.407 ms (1.41×) | 11.505 ms (1.72×) | 12.778 ms (1.91×) | 25.864 ms (3.87×) | 24.863 ms (3.72×) |
| small-tags | 1 | 1 | 0.214 ms (0.28×) | 0.779 ms | 54.630 ms (70.15×) | 62.274 ms (79.97×) | 225.609 ms (289.71×) | 22.978 ms (29.51×) | 9.227 ms (11.85×) | 7.849 ms (10.08×) | 7.608 ms (9.77×) | 8.257 ms (10.60×) | 2.709 ms (3.48×) | 3.611 ms (4.64×) | 2.057 ms (2.64×) | 6.920 ms (8.89×) | 5.468 ms (7.02×) |
| steep-angle | 1 | 2 | 1.194 ms (0.18×) | 6.649 ms | 22.230 ms (3.34×) | 26.440 ms (3.98×) | 90.209 ms (13.57×) | 38.676 ms (5.82×) | 13.400 ms (2.02×) | 31.504 ms (4.74×) | 31.742 ms (4.77×) | 26.302 ms (3.96×) | 10.323 ms (1.55×) | 10.680 ms (1.61×) | 13.142 ms (1.98×) | 26.064 ms (3.92×) | 25.202 ms (3.79×) |
| stress-clutter-noisy | 10 | 0 | 0.843 ms (0.25×) | 2.667 ms | 35.837 ms (12.11×) | 49.517 ms (17.84×) | 204.479 ms (72.13×) | 16.277 ms (5.28×) | 8.391 ms (2.32×) | 81.898 ms (27.00×) | 82.203 ms (27.08×) | 71.917 ms (23.98×) | 11.621 ms (2.54×) | 12.510 ms (2.50×) | 14.637 ms (5.23×) | 55.091 ms (14.50×) | 55.445 ms (14.45×) |
| stress-dense | 1 | 26 | 1.189 ms (0.57×) | 2.098 ms | 22.200 ms (10.58×) | 25.423 ms (12.12×) | 115.566 ms (55.09×) | 25.813 ms (12.30×) | 16.462 ms (7.85×) | 41.444 ms (19.76×) | 41.920 ms (19.98×) | 24.401 ms (11.63×) | 15.505 ms (7.39×) | 17.613 ms (8.40×) | 7.363 ms (3.51×) | 11.222 ms (5.35×) | 11.076 ms (5.28×) |
| stress-heavy-noise | 12 | 22 | 0.971 ms (0.39×) | 2.416 ms | 20.320 ms (6.65×) | 41.098 ms (8.78×) | 133.234 ms (31.85×) | 24.290 ms (9.34×) | 8.377 ms (2.98×) | 115.879 ms (44.39×) | 115.118 ms (44.45×) | 104.899 ms (39.65×) | 20.179 ms (7.82×) | 20.709 ms (8.21×) | 18.894 ms (7.34×) | 88.516 ms (33.73×) | 89.598 ms (33.83×) |
| stress-tiny | 6 | 16 | 0.226 ms (0.22×) | 0.819 ms | 34.820 ms (47.18×) | 41.252 ms (57.87×) | 171.212 ms (246.13×) | 19.430 ms (20.58×) | 7.683 ms (9.63×) | 7.212 ms (8.04×) | 7.214 ms (8.18×) | 6.242 ms (7.83×) | 2.070 ms (2.39×) | 2.017 ms (2.42×) | 1.909 ms (2.37×) | 5.594 ms (6.86×) | 5.243 ms (6.61×) |
| underexposed | 1 | 3 | 1.252 ms (0.22×) | 5.671 ms | 29.086 ms (5.13×) | 33.096 ms (5.84×) | 179.869 ms (31.72×) | 29.987 ms (5.29×) | 12.561 ms (2.21×) | 17.285 ms (3.05×) | 17.317 ms (3.05×) | 15.140 ms (2.67×) | 6.044 ms (1.07×) | 7.312 ms (1.29×) | 7.764 ms (1.37×) | 8.977 ms (1.58×) | 10.918 ms (1.93×) |
| uneven-light | 1 | 2 | 0.446 ms (0.35×) | 1.259 ms | 40.291 ms (32.01×) | 51.092 ms (40.59×) | 198.711 ms (157.87×) | 19.392 ms (15.41×) | 8.830 ms (7.02×) | 25.937 ms (20.61×) | 25.429 ms (20.20×) | 20.021 ms (15.91×) | 6.338 ms (5.04×) | 8.594 ms (6.83×) | 3.391 ms (2.69×) | 10.867 ms (8.63×) | 10.935 ms (8.69×) |

### 1920x1080: tags found (wrong IDs in brackets)

| Scene | Tags | NAPA32 (default) | NAPA16 (tuned) | WPILib AprilTag d2 | AprilTag d2 | AprilTag d1 | Locus standard | Locus d2 | OpenCV ArUco | OpenCV SUBPIX | ArUco3 full | ArUco3 min≈64 px | ArUco3 min≈32 px | ArUco Nano | RapidTag | RapidTag SUBPIX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| blur | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| busy-background | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| clutter-only | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dark-background | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 0 | 0 |
| decoys | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| empty | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| frame-edge | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 |
| jpeg | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| large-tags | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| low-contrast | 13 | 13 | 13 | 12 | 12 | 13 | 11 | 12 | 13 | 13 | 13 | 13 | 13 | 13 | 0 | 0 |
| many-tags | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 6 | 4 | 4 |
| motion-blur | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| noise | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 20 | 21 | 20 | 21 | 17 | 17 |
| noise-free | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 0 | 0 |
| one-tag | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| overexposed | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| small-tags | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| steep-angle | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 1 | 1 |
| stress-clutter-noisy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (5) | 0 (5) |
| stress-dense | 26 | 26 | 26 | 26 | 26 | 26 | 26 | 26 | 26 | 26 | 26 | 26 | 26 | 26 | 17 | 17 |
| stress-heavy-noise | 22 | 22 | 22 | 22 | 22 | 22 | 20 | 22 | 22 | 22 | 22 | 20 | 20 | 21 | 20 (1) | 20 (1) |
| stress-tiny | 16 | 9 | 16 | 0 | 9 | 16 | 15 | 8 | 10 | 10 | 0 | 0 | 0 | 16 | 0 | 0 |
| underexposed | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| uneven-light | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |

There were no duplicate detections in these diagnostic frames. Full per-call p95/p99 and corner errors are in the linked CSV/JSON report.

The separate full-quality replay covers 1,620 frames (900 corpus, 180 stress and 540 range), with 3,486 placed tags.
NAPA16 found all 3,486; NAPA32 found 3,334. Neither produced a wrong ID or duplicate. Its one-call-per-frame timings
are not used in the latency tables above.

## Range

Tags found out of **30** at each black-square size, in pixels. NAPA rows use the final full-quality replay
([final_quality.csv](results/2026-10-29_cpu_optimization/final_quality.csv)); unchanged competitors reuse the
[earlier range replay](results/2026-10-28_cpu_comparison/range_sweep_latency.csv). All 540 range-image hashes match.
WPILib was not included in that 30-case peer replay, so it has no row here. Its earlier 50-case results remain in
[the archived matrix](results/2026-10-29_cpu_detector_matrix/matrix.json).

| Detector | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 28 | 32 | 36 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NAPA32 (default) | 0 | 0 | 7 | 21 | 29 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| NAPA16 (tuned) | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| AprilTag d2 | 4 | 6 | 8 | 13 | 17 | 25 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| AprilTag d1 | 29 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Locus standard | 26 | 29 | 28 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| Locus d2 | 0 | 0 | 8 | 9 | 16 | 23 | 30 | 30 | 30 | 29 | 30 | 30 | 30 | 30 | 30 | 29 | 30 | 30 |
| OpenCV ArUco | 26 | 28 | 29 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 29 |
| OpenCV SUBPIX | 26 | 28 | 29 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 29 |
| ArUco3 full | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 24 | 30 | 30 | 30 | 30 | 30 | 30 | 29 |
| ArUco3 min≈64 px | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 26 | 30 | 29 |
| ArUco3 min≈32 px | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 24 | 30 | 29 | 30 | 30 | 30 | 30 | 29 |
| ArUco Nano | 21 | 27 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| RapidTag | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 15 | 21 | 17 | 17 | 17 | 18 | 14 | 16 |
| RapidTag SUBPIX | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 15 | 21 | 17 | 17 | 17 | 18 | 14 | 16 |

NAPA16 found all tags at every tested size from 12–96 px. NAPA32 trades small-tag recall for lower latency.
These images use light noise and at most 15° tilt; they do not establish real-camera detection range.

## Tag families

These earlier family results predate the native-core optimization and have not been rerun.

A separate family-by-family run, not part of `performance_matrix.py`, compared NAPA with upstream AprilTag only (most
of the other detectors above read only a few of these families). It used the same scenes with each family's own
tags, spaced by their printed size, at 640×480 and 1280×720. Speedups are against AprilTag at decimate 2, per frame,
from a timing pass over 80 frames per family on one pinned core.

| Family | Tags | Found: AprilTag → NAPA | Wrong IDs: AprilTag → NAPA | Median corner error: AprilTag → NAPA | Speedup: median (slowest frame) |
| --- | ---: | ---: | ---: | ---: | ---: |
| tag36h11 | 289 | 288 → 287 | 0 → 0 | 0.097 → 0.022 px | 40× (7.9×) |
| tag16h5 | 282 | 282 → 280 | **581 → 0** | 0.107 → 0.022 px | 38× (5.5×) |
| tag25h9 | 287 | 286 → 286 | 3 → 0 | 0.098 → 0.021 px | 36× (7.2×) |
| tagStandard41h12 | 232 | 232 → 231 | 0 → 0 | 0.097 → 0.025 px | 42× (8.5×) |
| tagCircle21h7 | 231 | 231 → 230 | 9 → 0 | 0.097 → 0.025 px | 42× (9.3×) |

Every tag that only AprilTag found was small (40–60 px) and steeply tilted, which narrows the white margin to a pixel
or two. Either its narrow side was under `min_tag_px`, or a dark line in the background touched the tag and joined its
border. On tagStandard52h13, tagCircle49h12, tagCustom48h12, and tag36h10, which the reference Python binding doesn't
include, NAPA found 99.2–99.6% of tags with no wrong IDs. Upstream's decode tables for the three largest families need
several gigabytes, where NAPA's take megabytes.

tag16h5's 30 codes are only 5 bits apart, so a decoder that corrects two wrong bits, as AprilTag does by default,
accepts about one in four random tag-shaped patterns. [Tag families](../docs/api.md#tag-families) in the API reference
explains how NAPA sets each family's correction.

## Reading the results

- Compare scenario rows at the same resolution rather than overall averages: the scenarios differ a lot in cost.
- Corner error is the worst corner's distance from the truth, for each tag found, after each detector's corner order
  is aligned to the truth. AprilTag's and Locus's corners are shifted by −0.5 px to the pixel-centre convention, as
  their sources document. Compare errors on common tags, so a detector cannot improve its score by missing hard
  tags. The [current report](results/2026-10-29_cpu_optimization/report.md#quality) gives median, p95 and maximum
  errors separately. OpenCV and RapidTag do not refine corners by default.
- Wrong IDs and duplicate detections are counted separately. Per-scene brackets contain wrong IDs; there were
  no duplicates in the diagnostic comparison.
- Recall in `small-tags` and `steep-angle` depends on apparent size. A steeply tilted tag can look narrower than
  `min_tag_px` even when its long side is well above it.
- The frames are synthetic and the timings come from one laptop CPU. Measure on your own camera and hardware (see
  [Measuring on your own footage](../docs/measuring.md)) before relying on a specific number.

## Other CPU detectors

`cpu_comparison.py` runs the same corpus, plus stress cases (tags under 32 px, noise σ 15–30, 30–80 tags per frame,
noisy clutter) and the range sweep, through other CPU tag36h11 detectors: upstream AprilTag 3 (C library),
[Locus](https://github.com/NoeFontana/locus-tag), OpenCV ArUco (including `useAruco3Detection`),
[ArUco Nano](https://github.com/rmsalinas/aruco_nano), and [RapidTag](https://github.com/SujithChristopher/rapidtag),
each at its defaults and in labelled tuned configurations. `cpu_backends.py` holds one adapter per library, which
converts every library's pixel origin and reports its corner order; the C shims for AprilTag and ArUco Nano are in
`cpu_comparison_shims/`.

- `validate` checks each adapter against known geometry (quarter turns, tilts, a rotated frame, blank frames) before
  any score is used.
- `latency` runs every backend on every frame in one process, interleaved in a shuffled order. It keeps false IDs,
  duplicates, corner error after aligning corner order, and the corner order itself as separate results.
- `resources` runs each backend alone in its own pinned process, for CPU time per frame, peak memory, and threading:
  several threads per frame, or one detector per worker thread processing frames in parallel.

The libraries are not dependencies of this package: build them in a scratch directory, as
`results/2026-10-28_cpu_comparison/commands.md` shows. That directory holds the first comparison: the raw timings,
the configuration and machine details, and a report with its methods, tradeoffs, and open questions.

## Stereo paired replay

`stereo_replay.py` runs the [stereo solver](../docs/stereo.md) over recorded image pairs, alongside each camera on its
own and a simple fusion of the two, and scores them against ground truth when you have it:

```bash
python benchmarks/stereo_replay.py corpus/pairs.json \
  --calibration rig.json --max-capture-skew-s 0.005 --output stereo-results.json
```

The manifest holds the field layout (field-to-tag transforms in metres, in WPILib axes) and each frame's exposure
midpoint on a shared clock. This one places tag 1 at field x = 3 m, facing the origin:

```json
{
  "schema_version": 1,
  "family": "tag36h11",
  "tag_size_m": 0.1651,
  "tag_layout": {"1": [[-1,0,0,3],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]},
  "pairs": [{
    "left": {"camera_id": "left", "path": "left.png", "capture_time_s": 1.0},
    "right": {"camera_id": "right", "path": "right.png", "capture_time_s": 1.002},
    "ground_truth": [{
      "timestamp_s": 1.001,
      "field_to_rig": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }]
  }]
}
```

Image paths must stay inside the manifest's folder, camera IDs must match the calibration, and resolutions must match
the intrinsics. `ground_truth` is optional. To score a moving rig, give a truth sample at each camera's capture time
and at the pair's midpoint: results are scored only against a truth sample within 1 ns of their timestamp, with no
interpolation.

For each pair, the output has the stereo result, each camera's own result, and `independent_pose_fusion`, which
combines each camera's best pose weighted by their covariances. It assumes the rig didn't move between the two frames,
uses the same capture-skew limit, and is a baseline to compare against, not a public solver. Each result records
whether it was accepted and why, the observations used, the RMS error, the translation and rotation error, six
standard deviations, and NEES, the squared pose error weighted by the inverse covariance. Timing reports p50 and p95
separately for detecting both images and for the joint solve, measured once per pair without warm-up. File loading and
the baseline solves aren't timed.
