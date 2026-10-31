# NAPA benchmarks

`performance_matrix.py` compares NAPA with AprilTag 3 as WPILib builds and configures it (`robotpy-apriltag`, at its
default `quad_decimate=2` and at full resolution) on a large set of generated grayscale frames, and
`cpu_comparison.py` runs the same frames through upstream AprilTag and other CPU detectors (see
[Other CPU detectors](#other-cpu-detectors)). Both measure detector calls only. Camera delivery, colour conversion,
and pose solving are outside the timed region.

The latest results are from the [2026-10-30 run](results/2026-10-30_cpu_full_corpus/report.md), which ran NAPA and 13 configurations of other detectors on the same frames.

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
[Range table below](#range) uses that same 50-tags-per-size sweep from the fixed full run.

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
- `--frames 0` runs only the sweeps. A range-only `performance_matrix.py` sweep uses
  `--frames 0 --resolutions 640x480 --occlusion-cases 0`. The latest comparison tables instead come from the
  [fixed shared run](results/2026-10-30_cpu_full_corpus/commands.md), including its range frames.
- `--seed` changes every frame. Each frame's seed depends only on the run seed, its scenario, resolution, and index,
  so any frame can be re-rendered on its own with `render_frame()`.

The script prints the per-scene tables below and writes two files:

- `*.json`: run configuration, machine details, per-scenario results (frames, tags, found, false IDs, latency
  percentiles, corner error), per-frame speedup statistics, and the obstruction sweep.
- `*.csv`: one row per frame and backend, with expected and detected IDs, corner errors, every raw timing sample, and
  the scenario's random parameters (`detail_*` columns). Use it to find and re-render any frame a detector missed.

## Latest results

These tables come from the [2026-10-30 run](results/2026-10-30_cpu_full_corpus/report.md), which gave every detector the
same 3,000 frames: 50 per scene at each resolution. Each detector had one warm-up and ten timed calls per frame, in
shuffled order, on one performance core of an Intel i7-12700H. Each time is the median of a scene's timed calls.

AprilTag at WPILib's defaults is `robotpy-apriltag` as WPILib configures it: decimate 2, a 300 px minimum cluster, and a
45° critical angle. The other AprilTag rows use the upstream C library's defaults. OpenCV ArUco and RapidTag refine
corners only in their subpixel rows. The two limited ArUco3 rows detect at half scale, which sets the smallest tag they
can find. The [report](results/2026-10-30_cpu_full_corpus/report.md) lists every configuration's exact settings, and
compares NAPA's corners with each detector's on the tags both found.

### 640x480: median detector call

| Scene | Frames | Tags | NAPA | NAPA, `min_tag_px=16` | AprilTag, WPILib's defaults | AprilTag, decimate 2 | AprilTag, full resolution | Locus | Locus, decimate 2 | OpenCV ArUco | OpenCV ArUco, subpixel | ArUco3 | ArUco3, tags over 64 px | ArUco3, tags over 32 px | ArUco Nano | RapidTag | RapidTag, subpixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 0.035 ms | 0.089 ms | 5.661 ms | 7.246 ms | 28.407 ms | 2.744 ms | 1.084 ms | 0.918 ms | 0.907 ms | 0.987 ms | 0.329 ms | 0.330 ms | 0.434 ms | 0.821 ms | 0.826 ms |
| clutter-only | 50 | 0 | 0.151 ms | 0.475 ms | 3.760 ms | 5.271 ms | 19.628 ms | 3.063 ms | 1.552 ms | 13.991 ms | 14.075 ms | 2.953 ms | 0.904 ms | 1.354 ms | 0.897 ms | 2.435 ms | 2.445 ms |
| decoys | 50 | 0 | 0.137 ms | 0.270 ms | 3.871 ms | 5.047 ms | 19.854 ms | 2.550 ms | 1.243 ms | 13.473 ms | 13.518 ms | 7.169 ms | 4.500 ms | 5.727 ms | 1.888 ms | 1.962 ms | 1.945 ms |
| one-tag | 50 | 50 | 0.100 ms | 0.172 ms | 4.919 ms | 6.059 ms | 24.457 ms | 3.124 ms | 1.511 ms | 2.642 ms | 2.648 ms | 1.732 ms | 0.938 ms | 1.006 ms | 0.503 ms | 1.055 ms | 1.064 ms |
| many-tags | 50 | 249 | 0.280 ms | 0.482 ms | 3.754 ms | 5.281 ms | 18.526 ms | 4.228 ms | 3.023 ms | 18.158 ms | 18.099 ms | 4.882 ms | 2.047 ms | 3.678 ms | 1.071 ms | 2.304 ms | 2.311 ms |
| small-tags | 50 | 131 | 0.217 ms | 0.440 ms | 4.120 ms | 6.217 ms | 20.207 ms | 3.604 ms | 2.115 ms | 15.366 ms | 15.358 ms | 3.537 ms | 0.870 ms | 2.209 ms | 0.870 ms | 2.371 ms | 2.353 ms |
| large-tags | 50 | 50 | 0.169 ms | 0.339 ms | 3.670 ms | 5.225 ms | 19.478 ms | 3.342 ms | 1.998 ms | 11.189 ms | 11.736 ms | 2.992 ms | 1.264 ms | 1.633 ms | 0.878 ms | 2.090 ms | 2.134 ms |
| steep-angle | 50 | 109 | 0.203 ms | 0.390 ms | 4.075 ms | 5.753 ms | 19.275 ms | 3.632 ms | 2.130 ms | 15.047 ms | 14.932 ms | 3.683 ms | 1.652 ms | 2.141 ms | 0.877 ms | 2.274 ms | 2.291 ms |
| blur | 50 | 96 | 0.220 ms | 0.432 ms | 3.660 ms | 5.180 ms | 17.622 ms | 3.958 ms | 2.459 ms | 6.737 ms | 6.679 ms | 3.397 ms | 1.414 ms | 2.016 ms | 0.993 ms | 2.158 ms | 2.156 ms |
| motion-blur | 50 | 110 | 0.215 ms | 0.446 ms | 3.833 ms | 4.949 ms | 17.558 ms | 3.897 ms | 2.366 ms | 8.443 ms | 8.456 ms | 3.698 ms | 1.556 ms | 2.252 ms | 1.000 ms | 2.247 ms | 2.289 ms |
| noise | 50 | 110 | 0.279 ms | 0.537 ms | 4.141 ms | 6.636 ms | 23.518 ms | 3.299 ms | 2.247 ms | 17.370 ms | 17.389 ms | 11.008 ms | 2.216 ms | 2.836 ms | 2.285 ms | 7.241 ms | 7.188 ms |
| uneven-light | 50 | 98 | 0.218 ms | 0.410 ms | 3.981 ms | 5.556 ms | 21.142 ms | 3.781 ms | 1.941 ms | 12.314 ms | 12.520 ms | 3.386 ms | 1.580 ms | 1.959 ms | 0.897 ms | 2.125 ms | 2.144 ms |
| low-contrast | 50 | 96 | 0.128 ms | 0.216 ms | 5.135 ms | 6.523 ms | 24.948 ms | 3.361 ms | 1.822 ms | 4.009 ms | 4.038 ms | 2.343 ms | 1.135 ms | 1.553 ms | 0.607 ms | 1.201 ms | 1.191 ms |
| underexposed | 50 | 110 | 0.225 ms | 0.423 ms | 4.115 ms | 5.840 ms | 22.294 ms | 4.351 ms | 2.144 ms | 9.325 ms | 9.331 ms | 3.544 ms | 1.440 ms | 1.968 ms | 0.976 ms | 1.797 ms | 1.805 ms |
| overexposed | 50 | 100 | 0.227 ms | 0.451 ms | 1.390 ms | 2.461 ms | 5.543 ms | 2.281 ms | 1.771 ms | 11.896 ms | 11.903 ms | 3.119 ms | 1.499 ms | 1.953 ms | 0.781 ms | 2.125 ms | 2.171 ms |
| jpeg | 50 | 104 | 0.201 ms | 0.409 ms | 1.422 ms | 2.443 ms | 5.478 ms | 2.528 ms | 1.966 ms | 9.398 ms | 9.553 ms | 3.964 ms | 1.612 ms | 2.207 ms | 0.902 ms | 2.624 ms | 2.612 ms |
| busy-background | 50 | 102 | 0.203 ms | 0.390 ms | 3.847 ms | 5.581 ms | 19.219 ms | 3.503 ms | 2.123 ms | 21.845 ms | 21.921 ms | 3.505 ms | 1.513 ms | 1.972 ms | 0.859 ms | 2.197 ms | 2.217 ms |
| frame-edge | 50 | 102 | 0.196 ms | 0.379 ms | 3.826 ms | 5.615 ms | 19.237 ms | 3.547 ms | 2.042 ms | 10.631 ms | 10.419 ms | 3.468 ms | 1.547 ms | 2.205 ms | 0.885 ms | 2.236 ms | 2.264 ms |
| noise-free | 50 | 98 | 0.107 ms | 0.180 ms | 0.666 ms | 0.708 ms | 1.832 ms | 1.417 ms | 1.350 ms | 4.486 ms | 4.525 ms | 1.984 ms | 1.049 ms | 1.264 ms | 0.488 ms | 1.085 ms | 1.101 ms |
| dark-background | 50 | 96 | 0.179 ms | 0.324 ms | 3.417 ms | 4.519 ms | 18.109 ms | 3.850 ms | 1.897 ms | 8.126 ms | 8.155 ms | 4.402 ms | 1.781 ms | 2.570 ms | 0.952 ms | 1.912 ms | 1.921 ms |

### 640x480: tags found (wrong IDs in brackets)

| Scene | Tags | NAPA | NAPA, `min_tag_px=16` | AprilTag, WPILib's defaults | AprilTag, decimate 2 | AprilTag, full resolution | Locus | Locus, decimate 2 | OpenCV ArUco | OpenCV ArUco, subpixel | ArUco3 | ArUco3, tags over 64 px | ArUco3, tags over 32 px | ArUco Nano | RapidTag | RapidTag, subpixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| clutter-only | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| decoys | 0 | 0 (1) | 0 (1) | 0 | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 (1) | 0 | 0 (1) | 0 | 0 (1) | 0 (1) |
| one-tag | 50 | 50 | 50 | 46 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 39 | 50 | 50 | 27 | 27 |
| many-tags | 249 | 249 | 249 | 178 | 249 | 249 | 249 | 249 | 248 | 248 | 247 | 125 | 246 | 249 | 200 | 200 |
| small-tags | 131 | 131 | 131 | 0 | 131 | 131 | 131 | 131 | 131 | 131 | 106 | 0 | 100 | 131 | 84 | 84 |
| large-tags | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 43 | 43 |
| steep-angle | 109 | 105 | 109 | 71 | 109 | 109 | 109 | 106 | 109 | 109 | 109 | 66 | 109 | 109 | 97 | 97 |
| blur | 96 | 96 | 96 | 82 | 96 | 96 | 96 | 96 | 96 | 96 | 96 | 69 | 94 | 96 | 84 | 84 |
| motion-blur | 110 | 109 | 110 | 86 | 110 | 110 | 107 | 106 | 101 | 101 | 103 | 67 | 104 | 108 | 95 | 95 |
| noise | 110 | 110 | 110 | 94 | 110 | 110 | 110 | 110 | 109 | 109 | 106 | 72 | 103 | 110 | 87 (1) | 87 (1) |
| uneven-light | 98 | 98 | 98 | 82 | 98 | 98 | 96 | 95 | 98 | 98 | 98 | 73 | 98 | 98 | 94 | 94 |
| low-contrast | 96 | 95 | 96 | 70 | 88 | 92 | 91 | 92 | 96 | 96 | 96 | 65 | 87 | 94 | 47 | 47 |
| underexposed | 110 | 110 | 110 | 96 | 110 | 110 | 79 | 89 | 110 | 110 | 110 | 78 | 110 | 110 | 110 | 110 |
| overexposed | 100 | 100 | 100 | 82 | 100 | 100 | 92 | 92 | 100 | 100 | 100 | 67 | 100 | 100 | 100 | 100 |
| jpeg | 104 | 104 | 104 | 92 | 104 | 104 | 104 | 104 | 104 | 104 | 103 | 78 | 102 | 104 | 91 | 91 |
| busy-background | 102 | 102 | 102 | 89 | 102 | 102 | 102 | 102 | 102 | 102 | 102 | 71 | 102 | 102 | 96 (1) | 96 (1) |
| frame-edge | 102 | 102 | 102 | 74 | 102 | 102 | 102 | 102 | 97 | 97 | 96 | 58 | 96 | 102 | 89 | 89 |
| noise-free | 98 | 98 | 98 | 83 | 98 | 98 | 98 | 98 | 97 | 97 | 97 | 71 | 97 | 98 | 41 | 41 |
| dark-background | 96 | 96 | 96 | 71 | 96 | 96 | 96 | 96 | 96 | 96 | 95 | 48 | 86 | 96 | 13 | 13 |

### 1280x720: median detector call

| Scene | Frames | Tags | NAPA | NAPA, `min_tag_px=16` | AprilTag, WPILib's defaults | AprilTag, decimate 2 | AprilTag, full resolution | Locus | Locus, decimate 2 | OpenCV ArUco | OpenCV ArUco, subpixel | ArUco3 | ArUco3, tags over 64 px | ArUco3, tags over 32 px | ArUco Nano | RapidTag | RapidTag, subpixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 0.064 ms | 0.251 ms | 17.807 ms | 21.880 ms | 89.307 ms | 8.181 ms | 3.156 ms | 2.686 ms | 2.664 ms | 2.866 ms | 0.867 ms | 0.869 ms | 0.917 ms | 2.331 ms | 2.311 ms |
| clutter-only | 50 | 0 | 0.301 ms | 1.321 ms | 11.778 ms | 15.660 ms | 65.708 ms | 8.973 ms | 4.506 ms | 23.487 ms | 23.525 ms | 9.412 ms | 2.993 ms | 3.902 ms | 2.661 ms | 6.991 ms | 6.989 ms |
| decoys | 50 | 0 | 0.210 ms | 0.571 ms | 15.378 ms | 19.642 ms | 77.880 ms | 7.876 ms | 3.371 ms | 18.585 ms | 18.519 ms | 14.338 ms | 7.313 ms | 11.328 ms | 3.631 ms | 3.882 ms | 3.841 ms |
| one-tag | 50 | 50 | 0.140 ms | 0.378 ms | 15.554 ms | 18.674 ms | 76.021 ms | 8.623 ms | 3.649 ms | 4.643 ms | 4.691 ms | 3.824 ms | 1.591 ms | 1.660 ms | 1.078 ms | 2.700 ms | 2.685 ms |
| many-tags | 50 | 354 | 0.513 ms | 1.021 ms | 11.566 ms | 15.672 ms | 57.329 ms | 10.628 ms | 6.670 ms | 25.683 ms | 25.627 ms | 12.083 ms | 5.634 ms | 7.240 ms | 2.812 ms | 6.061 ms | 6.105 ms |
| small-tags | 50 | 120 | 0.242 ms | 0.656 ms | 13.309 ms | 18.317 ms | 65.717 ms | 9.146 ms | 4.239 ms | 12.820 ms | 12.694 ms | 7.721 ms | 2.215 ms | 3.822 ms | 1.644 ms | 4.707 ms | 4.680 ms |
| large-tags | 50 | 50 | 0.255 ms | 0.699 ms | 12.575 ms | 17.363 ms | 64.780 ms | 9.427 ms | 4.904 ms | 16.382 ms | 16.569 ms | 9.373 ms | 2.961 ms | 4.315 ms | 1.958 ms | 5.265 ms | 5.313 ms |
| steep-angle | 50 | 112 | 0.328 ms | 0.790 ms | 12.680 ms | 17.361 ms | 63.248 ms | 9.493 ms | 5.189 ms | 24.199 ms | 24.235 ms | 10.022 ms | 3.522 ms | 4.693 ms | 2.008 ms | 5.677 ms | 5.807 ms |
| blur | 50 | 89 | 0.318 ms | 0.813 ms | 12.433 ms | 16.234 ms | 62.430 ms | 9.802 ms | 4.876 ms | 12.209 ms | 12.189 ms | 8.210 ms | 2.963 ms | 3.837 ms | 2.443 ms | 5.138 ms | 5.111 ms |
| motion-blur | 50 | 94 | 0.398 ms | 1.765 ms | 12.159 ms | 17.255 ms | 60.417 ms | 9.897 ms | 5.199 ms | 17.675 ms | 17.503 ms | 9.031 ms | 3.377 ms | 4.333 ms | 2.471 ms | 6.605 ms | 6.616 ms |
| noise | 50 | 104 | 0.511 ms | 1.316 ms | 12.085 ms | 18.349 ms | 73.786 ms | 8.547 ms | 4.712 ms | 40.884 ms | 40.901 ms | 33.210 ms | 5.983 ms | 7.052 ms | 6.433 ms | 20.814 ms | 20.725 ms |
| uneven-light | 50 | 103 | 0.366 ms | 0.857 ms | 13.097 ms | 17.734 ms | 69.028 ms | 9.930 ms | 4.622 ms | 17.125 ms | 17.382 ms | 8.473 ms | 3.223 ms | 3.864 ms | 2.009 ms | 5.718 ms | 5.700 ms |
| low-contrast | 50 | 100 | 0.199 ms | 0.451 ms | 16.584 ms | 20.136 ms | 81.012 ms | 9.077 ms | 4.137 ms | 6.125 ms | 6.122 ms | 4.586 ms | 2.069 ms | 2.272 ms | 1.288 ms | 3.029 ms | 3.084 ms |
| underexposed | 50 | 99 | 0.306 ms | 0.762 ms | 13.459 ms | 19.895 ms | 75.965 ms | 10.962 ms | 4.703 ms | 16.084 ms | 16.097 ms | 7.682 ms | 2.806 ms | 3.819 ms | 2.227 ms | 3.966 ms | 4.003 ms |
| overexposed | 50 | 112 | 0.279 ms | 0.733 ms | 2.520 ms | 4.397 ms | 10.602 ms | 4.734 ms | 3.840 ms | 17.506 ms | 17.801 ms | 6.592 ms | 2.991 ms | 3.800 ms | 1.442 ms | 3.868 ms | 3.870 ms |
| jpeg | 50 | 100 | 0.270 ms | 0.718 ms | 3.213 ms | 4.337 ms | 10.994 ms | 5.392 ms | 3.781 ms | 17.463 ms | 17.494 ms | 9.694 ms | 3.223 ms | 4.262 ms | 1.708 ms | 5.243 ms | 5.265 ms |
| busy-background | 50 | 87 | 0.293 ms | 0.736 ms | 12.920 ms | 17.946 ms | 66.060 ms | 9.402 ms | 4.750 ms | 37.885 ms | 37.898 ms | 9.355 ms | 3.161 ms | 4.770 ms | 1.966 ms | 5.471 ms | 5.477 ms |
| frame-edge | 50 | 105 | 0.298 ms | 0.731 ms | 12.932 ms | 17.088 ms | 61.486 ms | 9.097 ms | 4.827 ms | 16.357 ms | 16.465 ms | 8.883 ms | 3.216 ms | 3.997 ms | 1.994 ms | 5.314 ms | 5.300 ms |
| noise-free | 50 | 104 | 0.177 ms | 0.430 ms | 1.750 ms | 1.747 ms | 5.066 ms | 3.080 ms | 2.807 ms | 7.000 ms | 7.028 ms | 4.766 ms | 2.207 ms | 2.416 ms | 1.119 ms | 2.998 ms | 2.981 ms |
| dark-background | 50 | 105 | 0.259 ms | 0.612 ms | 12.666 ms | 15.535 ms | 63.041 ms | 10.463 ms | 4.402 ms | 8.818 ms | 8.738 ms | 7.387 ms | 3.486 ms | 3.956 ms | 1.696 ms | 3.825 ms | 3.777 ms |

### 1280x720: tags found (wrong IDs in brackets)

| Scene | Tags | NAPA | NAPA, `min_tag_px=16` | AprilTag, WPILib's defaults | AprilTag, decimate 2 | AprilTag, full resolution | Locus | Locus, decimate 2 | OpenCV ArUco | OpenCV ArUco, subpixel | ArUco3 | ArUco3, tags over 64 px | ArUco3, tags over 32 px | ArUco Nano | RapidTag | RapidTag, subpixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| clutter-only | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (1) | 0 (1) |
| decoys | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| one-tag | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 48 | 48 | 50 | 26 | 26 |
| many-tags | 354 | 354 | 354 | 354 | 354 | 354 | 354 | 354 | 354 | 354 | 354 | 335 | 353 | 354 | 299 | 299 |
| small-tags | 120 | 120 | 120 | 0 | 120 | 120 | 120 | 120 | 120 | 120 | 93 | 0 | 92 | 120 | 68 | 68 |
| large-tags | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 49 | 49 | 49 | 49 | 49 | 50 | 46 | 46 |
| steep-angle | 112 | 111 | 112 | 109 | 112 | 112 | 112 | 112 | 112 | 112 | 111 | 98 | 110 | 112 | 101 (1) | 101 (1) |
| blur | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 86 | 86 | 89 | 77 | 77 |
| motion-blur | 94 | 94 | 94 | 94 | 94 | 94 | 94 | 92 | 93 | 93 | 93 | 92 | 92 | 94 | 83 | 83 |
| noise | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 102 | 103 | 100 | 104 | 93 (5) | 93 (5) |
| uneven-light | 103 | 103 | 103 | 103 | 103 | 103 | 102 | 101 | 103 | 103 | 103 | 99 | 102 | 103 | 91 | 91 |
| low-contrast | 100 | 100 | 98 | 95 | 95 | 100 | 93 | 93 | 100 | 100 | 100 | 97 | 99 | 100 | 57 | 57 |
| underexposed | 99 | 99 | 99 | 99 | 99 | 99 | 69 | 79 | 99 | 99 | 99 | 95 | 99 | 99 | 99 | 99 |
| overexposed | 112 | 112 | 112 | 112 | 112 | 112 | 107 | 112 | 112 | 112 | 112 | 111 | 112 | 112 | 112 | 112 |
| jpeg | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 97 | 99 | 100 | 83 | 83 |
| busy-background | 87 | 87 | 87 | 87 | 87 (1) | 87 | 87 | 86 | 87 | 87 | 87 | 83 | 86 | 87 | 74 | 74 |
| frame-edge | 105 | 105 | 105 | 103 | 105 | 105 | 105 | 105 | 96 | 96 | 95 | 79 | 94 | 105 | 81 (2) | 81 (2) |
| noise-free | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 100 | 103 | 104 | 39 | 39 |
| dark-background | 105 | 105 | 105 | 91 | 105 | 105 | 105 | 105 | 105 | 105 | 104 | 80 | 92 | 105 | 7 | 7 |

### 1920x1080: median detector call

| Scene | Frames | Tags | NAPA | NAPA, `min_tag_px=16` | AprilTag, WPILib's defaults | AprilTag, decimate 2 | AprilTag, full resolution | Locus | Locus, decimate 2 | OpenCV ArUco | OpenCV ArUco, subpixel | ArUco3 | ArUco3, tags over 64 px | ArUco3, tags over 32 px | ArUco Nano | RapidTag | RapidTag, subpixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 0.108 ms | 0.583 ms | 44.197 ms | 54.449 ms | 213.348 ms | 18.539 ms | 7.031 ms | 6.157 ms | 6.135 ms | 6.509 ms | 1.917 ms | 1.922 ms | 2.144 ms | 5.262 ms | 5.249 ms |
| clutter-only | 50 | 0 | 0.310 ms | 1.248 ms | 31.041 ms | 43.294 ms | 155.774 ms | 19.701 ms | 9.279 ms | 51.963 ms | 51.749 ms | 19.008 ms | 6.317 ms | 8.228 ms | 4.895 ms | 10.782 ms | 10.789 ms |
| decoys | 50 | 0 | 0.321 ms | 0.999 ms | 39.183 ms | 48.549 ms | 193.584 ms | 18.745 ms | 7.522 ms | 21.207 ms | 21.685 ms | 18.661 ms | 9.455 ms | 12.707 ms | 5.248 ms | 7.045 ms | 7.068 ms |
| one-tag | 50 | 50 | 0.200 ms | 0.746 ms | 38.495 ms | 44.977 ms | 189.118 ms | 19.267 ms | 7.778 ms | 8.766 ms | 8.815 ms | 7.825 ms | 2.705 ms | 2.901 ms | 2.157 ms | 5.758 ms | 5.741 ms |
| many-tags | 50 | 303 | 0.750 ms | 1.821 ms | 27.438 ms | 35.245 ms | 137.670 ms | 22.376 ms | 11.839 ms | 31.102 ms | 31.010 ms | 23.585 ms | 8.683 ms | 11.740 ms | 5.393 ms | 12.626 ms | 12.139 ms |
| small-tags | 50 | 122 | 0.366 ms | 1.196 ms | 31.554 ms | 42.447 ms | 157.704 ms | 19.924 ms | 8.880 ms | 24.736 ms | 24.701 ms | 18.423 ms | 6.462 ms | 8.888 ms | 3.256 ms | 9.748 ms | 9.753 ms |
| large-tags | 50 | 50 | 0.343 ms | 1.171 ms | 30.992 ms | 39.178 ms | 152.303 ms | 20.440 ms | 9.226 ms | 25.787 ms | 26.099 ms | 18.251 ms | 6.301 ms | 8.590 ms | 3.099 ms | 9.134 ms | 9.201 ms |
| steep-angle | 50 | 96 | 0.418 ms | 1.294 ms | 31.836 ms | 40.893 ms | 153.857 ms | 20.028 ms | 9.305 ms | 27.413 ms | 27.804 ms | 19.447 ms | 7.231 ms | 8.807 ms | 3.492 ms | 10.313 ms | 10.413 ms |
| blur | 50 | 97 | 0.429 ms | 1.286 ms | 29.377 ms | 39.499 ms | 149.376 ms | 20.773 ms | 10.222 ms | 22.792 ms | 23.147 ms | 15.254 ms | 5.504 ms | 7.262 ms | 4.189 ms | 9.488 ms | 9.509 ms |
| motion-blur | 50 | 95 | 0.436 ms | 1.340 ms | 32.448 ms | 40.566 ms | 152.269 ms | 20.328 ms | 9.532 ms | 23.683 ms | 24.041 ms | 16.313 ms | 5.582 ms | 7.955 ms | 3.710 ms | 9.594 ms | 9.509 ms |
| noise | 50 | 100 | 1.009 ms | 2.824 ms | 32.138 ms | 46.500 ms | 184.526 ms | 19.355 ms | 9.320 ms | 84.207 ms | 83.198 ms | 72.678 ms | 12.202 ms | 14.763 ms | 14.264 ms | 46.170 ms | 46.196 ms |
| uneven-light | 50 | 101 | 0.545 ms | 1.516 ms | 32.893 ms | 42.943 ms | 165.597 ms | 21.609 ms | 9.675 ms | 35.907 ms | 35.645 ms | 18.982 ms | 6.522 ms | 9.075 ms | 3.994 ms | 10.675 ms | 10.777 ms |
| low-contrast | 50 | 113 | 0.293 ms | 0.900 ms | 38.835 ms | 45.598 ms | 181.457 ms | 19.871 ms | 8.481 ms | 11.745 ms | 11.700 ms | 9.720 ms | 3.741 ms | 4.398 ms | 2.535 ms | 6.642 ms | 6.573 ms |
| underexposed | 50 | 95 | 0.391 ms | 1.169 ms | 36.592 ms | 46.983 ms | 188.330 ms | 23.425 ms | 9.173 ms | 21.439 ms | 21.429 ms | 15.491 ms | 4.577 ms | 6.624 ms | 3.845 ms | 8.422 ms | 8.432 ms |
| overexposed | 50 | 100 | 0.455 ms | 1.386 ms | 10.089 ms | 12.260 ms | 40.662 ms | 9.246 ms | 6.699 ms | 28.651 ms | 29.060 ms | 17.636 ms | 6.592 ms | 9.058 ms | 3.716 ms | 9.699 ms | 9.672 ms |
| jpeg | 50 | 100 | 0.445 ms | 1.466 ms | 8.246 ms | 9.905 ms | 29.551 ms | 12.004 ms | 7.903 ms | 30.114 ms | 30.102 ms | 23.089 ms | 7.125 ms | 9.323 ms | 3.886 ms | 12.667 ms | 12.543 ms |
| busy-background | 50 | 97 | 0.393 ms | 1.248 ms | 30.525 ms | 41.126 ms | 144.688 ms | 19.831 ms | 9.394 ms | 62.018 ms | 61.877 ms | 20.925 ms | 6.991 ms | 10.120 ms | 3.295 ms | 15.040 ms | 15.554 ms |
| frame-edge | 50 | 88 | 0.353 ms | 1.137 ms | 31.700 ms | 39.871 ms | 148.177 ms | 19.738 ms | 8.951 ms | 26.514 ms | 26.520 ms | 17.070 ms | 4.975 ms | 7.166 ms | 3.023 ms | 9.717 ms | 9.523 ms |
| noise-free | 50 | 108 | 0.238 ms | 0.810 ms | 3.471 ms | 3.349 ms | 10.481 ms | 5.733 ms | 5.025 ms | 10.743 ms | 10.732 ms | 9.446 ms | 3.462 ms | 4.442 ms | 1.911 ms | 5.977 ms | 6.009 ms |
| dark-background | 50 | 107 | 0.348 ms | 1.142 ms | 31.914 ms | 38.759 ms | 159.123 ms | 20.977 ms | 8.403 ms | 17.098 ms | 17.207 ms | 15.864 ms | 6.528 ms | 7.659 ms | 3.269 ms | 8.321 ms | 8.322 ms |

### 1920x1080: tags found (wrong IDs in brackets)

| Scene | Tags | NAPA | NAPA, `min_tag_px=16` | AprilTag, WPILib's defaults | AprilTag, decimate 2 | AprilTag, full resolution | Locus | Locus, decimate 2 | OpenCV ArUco | OpenCV ArUco, subpixel | ArUco3 | ArUco3, tags over 64 px | ArUco3, tags over 32 px | ArUco Nano | RapidTag | RapidTag, subpixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| clutter-only | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 (1) | 0 (1) |
| decoys | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| one-tag | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 49 | 50 | 28 | 28 |
| many-tags | 303 | 303 | 303 | 303 | 303 | 303 | 303 | 303 | 300 | 300 | 300 | 300 | 298 | 303 | 219 | 219 |
| small-tags | 122 | 121 | 122 | 0 | 122 | 122 | 122 | 122 | 122 | 122 | 103 | 0 | 99 | 122 | 81 (1) | 81 (1) |
| large-tags | 50 | 50 | 50 | 50 | 50 (1) | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 46 | 46 |
| steep-angle | 96 | 96 | 96 | 88 | 96 | 96 (1) | 96 | 96 | 96 | 96 | 96 | 96 | 95 | 96 | 83 (1) | 83 (1) |
| blur | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 96 | 97 | 81 | 81 |
| motion-blur | 95 | 95 | 95 | 95 | 95 | 95 | 95 | 92 | 95 | 95 | 95 | 95 | 94 | 95 | 83 | 83 |
| noise | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 100 | 92 | 100 | 80 (2) | 80 (2) |
| uneven-light | 101 | 101 | 101 | 101 | 101 | 101 (1) | 101 | 101 | 101 | 101 | 101 | 101 | 100 | 101 | 96 | 96 |
| low-contrast | 113 | 113 | 113 | 108 | 108 | 113 | 104 | 104 | 113 | 113 | 113 | 113 | 113 | 113 | 40 | 40 |
| underexposed | 95 | 95 | 95 | 95 | 95 | 95 | 81 | 93 | 95 | 95 | 95 | 95 | 95 | 95 | 95 | 95 |
| overexposed | 100 | 100 | 100 | 100 | 100 | 100 | 87 | 87 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| jpeg | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 100 | 89 | 89 |
| busy-background | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 87 (1) | 87 (1) |
| frame-edge | 88 | 88 | 88 | 88 | 88 | 88 | 88 | 88 | 84 | 84 | 84 | 82 | 81 | 88 | 73 (1) | 73 (1) |
| noise-free | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 45 | 45 |
| dark-background | 107 | 107 | 107 | 89 | 107 | 107 | 107 | 107 | 107 | 107 | 105 | 82 | 92 | 107 | 7 | 7 |

No detector reported the same tag twice. The same run's 600 stress frames (tags under 32 px, noise σ 15–30, 30–80 tags
per frame, and noisy clutter) are scored separately:

| Detector | Median ms | p95 ms | p99 ms | Tags found | Wrong IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| NAPA | 0.600 | 2.201 | 2.728 | 4390/4577 | 0 |
| NAPA, `min_tag_px=16` | 1.561 | 6.597 | 10.638 | 4577/4577 | 0 |
| AprilTag, WPILib's defaults | 11.301 | 40.119 | 45.502 | 3011/4577 | 0 |
| AprilTag, decimate 2 | 15.831 | 53.908 | 62.904 | 4443/4577 | 1 |
| AprilTag, full resolution | 58.495 | 204.379 | 233.922 | 4577/4577 | 1 |
| Locus | 13.389 | 33.434 | 37.747 | 4525/4577 | 0 |
| Locus, decimate 2 | 6.214 | 18.127 | 21.016 | 4402/4577 | 0 |
| OpenCV ArUco | 34.578 | 120.712 | 138.094 | 4533/4577 | 0 |
| OpenCV ArUco, subpixel | 34.551 | 120.577 | 138.781 | 4533/4577 | 0 |
| ArUco3 | 20.755 | 108.398 | 118.193 | 4070/4577 | 0 |
| ArUco3, tags over 64 px | 5.209 | 22.094 | 26.384 | 2330/4577 | 0 |
| ArUco3, tags over 32 px | 8.314 | 24.931 | 30.872 | 4043/4577 | 0 |
| ArUco Nano | 4.759 | 18.619 | 20.433 | 4561/4577 | 0 |
| RapidTag | 11.795 | 83.681 | 95.962 | 3354/4577 | 26 |
| RapidTag, subpixel | 11.820 | 84.273 | 96.223 | 3354/4577 | 26 |

## Range

Tags found out of 50 at each black-square size, from the same run: 900 frames at 640×480, with light noise and up to 15°
of tilt.

| Detector | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 28 | 32 | 36 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NAPA | 0 | 0 | 14 | 37 | 47 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| NAPA, `min_tag_px=16` | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| AprilTag, WPILib's defaults | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 39 | 48 | 50 | 50 | 50 |
| AprilTag, decimate 2 | 4 | 8 | 9 | 20 | 33 | 45 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| AprilTag, full resolution | 49 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| Locus | 44 | 48 | 46 | 49 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| Locus, decimate 2 | 0 | 0 | 10 | 15 | 29 | 37 | 50 | 50 | 50 | 49 | 50 | 49 | 50 | 50 | 50 | 49 | 50 | 50 |
| OpenCV ArUco | 40 | 44 | 49 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 49 |
| OpenCV ArUco, subpixel | 40 | 44 | 49 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 49 |
| ArUco3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31 | 40 | 50 | 50 | 50 | 50 | 50 | 50 | 49 |
| ArUco3, tags over 64 px | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 44 | 48 | 49 |
| ArUco3, tags over 32 px | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 40 | 50 | 49 | 50 | 50 | 50 | 49 | 48 |
| ArUco Nano | 36 | 46 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| RapidTag | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 | 25 | 35 | 29 | 30 | 31 | 26 | 24 | 26 |
| RapidTag, subpixel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 | 25 | 35 | 29 | 30 | 31 | 26 | 24 | 26 |

With its default `min_tag_px=32`, NAPA found every tag from 22 px up. With `min_tag_px=16`, it found every tag at every
size tested. Real cameras add blur, noise, and steeper angles, so measure range on your own.

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
  tags. The [current report](results/2026-10-30_cpu_full_corpus/report.md#corner-quality-on-the-main-corpus) gives median, p95 and maximum
  errors separately. OpenCV and RapidTag do not refine corners by default.
- Wrong IDs and duplicate detections are counted separately. Per-scene brackets contain wrong IDs; there were
  no duplicates in the full main-corpus comparison.
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

## Independent datasets

`external.py` evaluates unchanged third-party images, independently of our renderer. It pins
[Locus-Tag Bench](https://huggingface.co/datasets/NoeFontana/locus-tag-bench) to revision
`5346835a9ed350828472a4c83d6e3999cf94fefb`: all 700 images in its eleven AprilTag configurations.
ChArUco is excluded because its marker dictionary is not an AprilTag family. The runner checks image and truth
checksums, reads authoritative `rich_truth.json` TAG records, and rejects incomplete Parquet shards.

Install the optional reader in your benchmark environment (`python -m pip install pyarrow`), alongside the existing
headless benchmark dependencies. From the repository root:

```bash
# Download pinned data once, into ignored work/.
python -m benchmarks.external --download
# NAPA and upstream AprilTag d1; construct once, warm up per image, shuffle three timed calls.
# Set NAPA_CORE_LIBRARY to the exact build being evaluated, as in the CPU comparison workflow.
python -m benchmarks.external --output work/external/locus-results.json
# Ordinary NAPA settings only, without the optional peer build:
python -m benchmarks.external --backends napa napa16 --output work/external/napa-results.json
```

Scoring requires the correct ID **and quad IoU ≥ 0.5**, with one-to-one matches. Duplicate detections and unmatched
predictions count as false positives under this annotation protocol. Only complete, in-margin ground-truth tags
enter the recall denominator; detections matching partial truth are ignored. Unmatched predictions can include
partially visible tags omitted from the source annotations, so inspect them before interpreting them as invented
IDs. Geometry is aligned cyclically (including reflected order); this does not validate canonical corner order.
Both datasets' truth coordinates are shifted by −0.5 to our integer pixel-centre convention.

Each configuration reports recall, precision, corner-error median/p95, and `corner_accuracy_at_1px`: the fraction
of **all eligible truths**, including missed tags, detected with all four corners within one pixel. These are
separate metrics, not an undefined aggregate “accuracy.” Timings include the detector adapter and exclude image
loading. NAPA output capacity is 1,024 to avoid truncating dense boards. Results preserve per-image detections,
checksums, effective settings, the loaded core hash, and evaluator hash. Files are not overwritten.

The same runner reads the [ICRA 2020 AprilTag accuracy dataset](https://huggingface.co/datasets/NoeFontana/apriltag-validation-data)
(Kallwies, Forkel and Wuensche, *Determining and Improving the Localization Accuracy of AprilTag Detection*, ICRA 2020).
Download an archive from mirror revision `6c7653385d480351fe8235d010dedc18fbc0b992` and extract its scenario directory
under `work/external/icra/`, retaining `tags.csv`, `pure_tags_images/`, and `checkerboard_corners_images/`. Then:

```bash
python -m benchmarks.external --root work/external/icra --configs icra-forward \
  --backends napa napa16 apriltag3-d1 --output work/external/icra-forward.json
```

`icra-circle`, `icra-random`, and `icra-rotation` use the same layout. Missing images are errors; the mirror documents
one removed corrupt checkerboard image in `circle`, so that archive requires an explicit, recorded exclusion protocol
before it can be evaluated by this strict loader. Do not silently reduce denominators. The 2026-10-31 measurements
cover **forward only**, both variants, not the full ICRA dataset.

Detector-generated labels, including the FRC-heavy
[willtoth collection](https://github.com/willtoth/apriltag_dataset), establish reference agreement rather than
independent ground-truth accuracy. Keep those results separate. Current measured gaps and rejected experiments are
in the [independent baseline report](results/2026-10-31_external_baseline/report.md).
