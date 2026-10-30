# NAPA benchmarks

`performance_matrix.py` compares NAPA with AprilTag 3 as WPILib builds and configures it (`robotpy-apriltag`, at its
default `quad_decimate=2` and at full resolution) on a large set of generated grayscale frames, and
`cpu_comparison.py` runs the same frames through upstream AprilTag and other CPU detectors (see
[Other CPU detectors](#other-cpu-detectors)). Both measure detector calls only. Camera delivery, colour conversion,
and pose solving are outside the timed region.

The latest native-core tuning is measured in the
[CPU optimization report](results/2026-10-29_cpu_optimization/report.md). With `min_tag_px=16`, NAPA had the lowest
pooled single-core p50, p95 and p99 on a 149-frame diagnostic set against 13 competitor configurations. It also had
the lowest mean wall time, CPU time and working memory at all three resolutions, with one or six P-cores available
to each backend; NAPA itself uses one thread per frame. It found all 3,486 tags in the full synthetic replay with no
wrong IDs or duplicates. The default `min_tag_px=32` is unchanged and finds 3,334.

The diagnostic frames were chosen to include NAPA's earlier losses, so they are not a random sample. NAPA still
loses some individual frames to ArUco Nano and Locus. At 1080p with six cores available, its p99 trails two tuned
detectors that find fewer tags, and its worst corner error is higher than upstream AprilTag's. These are synthetic
results from one machine. The tables below come from earlier runs and were not changed.

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
light noise (σ ≈ 1.5). There are 50 tags at each size, at the smallest resolution. See [Range](#range) for the
results.

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

From `results/2026-10-29_cpu_detector_matrix/matrix.json`: the default corpus (seed 2468) and the range sweep, run
with `cpu_comparison.py latency` on one pinned performance core of an Intel Core i7-12700H laptop (Ubuntu 26.04,
Python 3.14, gcc 15). Every detector ran at its own defaults, single-threaded, in a shuffled order on each frame with
one warm-up and ten timed calls: napa-vision 0.4.3; WPILib's AprilTag through robotpy-apriltag 2026.2.2 (decimate 2);
upstream AprilTag 3.4.5 at commit b7c0ebe (decimate 2 and 1); Locus 0.10.0 (`standard` profile); OpenCV 5.0.0's ArUco
module (`DICT_APRILTAG_36h11`, no corner refinement, as it defaults); ArUco Nano v10 at c99989b; RapidTag 0.1.12. p50
is the median detector call across all frames and repetitions in a row. Earlier checked-in results came from other
machines, so their absolute times differ.

WPILib's AprilTag keeps only candidate regions of at least 300 pixels, and rejects quads with a corner sharper than
45° or wider than 135° (upstream: 24 pixels, and 10° or 170°). That is why it finds fewer small and steeply tilted
tags than upstream at the same decimation.

#### 640x480: median call, and NAPA's median per-frame speedup in brackets

| Scene | Frames | Tags | WPILib AprilTag, decimate 2 | AprilTag, decimate 2 | AprilTag, decimate 1 | Locus | OpenCV ArUco | ArUco Nano | RapidTag | NAPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 5.47 ms (128×) | 7.00 ms (164×) | 28.01 ms (644×) | 2.66 ms (71×) | 0.91 ms (27×) | 0.46 ms (11×) | 0.80 ms (21×) | 0.04 ms |
| clutter-only | 50 | 0 | 3.76 ms (20×) | 5.53 ms (31×) | 19.76 ms (106×) | 3.05 ms (25×) | 14.02 ms (87×) | 0.88 ms (6.7×) | 2.32 ms (16×) | 0.15 ms |
| decoys | 50 | 0 | 3.83 ms (28×) | 4.87 ms (36×) | 19.13 ms (149×) | 2.55 ms (19×) | 13.53 ms (105×) | 1.84 ms (14×) | 1.95 ms (14×) | 0.14 ms |
| one-tag | 50 | 50 | 4.72 ms (45×) | 5.80 ms (55×) | 22.81 ms (223×) | 3.05 ms (30×) | 2.56 ms (25×) | 0.49 ms (4.7×) | 1.05 ms (10×) | 0.10 ms |
| many-tags | 50 | 249 | 3.58 ms (13×) | 5.13 ms (18×) | 18.22 ms (61×) | 4.32 ms (16×) | 17.97 ms (57×) | 1.03 ms (3.9×) | 2.27 ms (7.7×) | 0.28 ms |
| small-tags | 50 | 131 | 4.05 ms (17×) | 6.08 ms (25×) | 19.39 ms (100×) | 3.54 ms (19×) | 15.41 ms (55×) | 0.88 ms (4.2×) | 2.33 ms (9.0×) | 0.22 ms |
| large-tags | 50 | 50 | 3.66 ms (22×) | 5.12 ms (32×) | 19.16 ms (123×) | 3.35 ms (21×) | 12.37 ms (60×) | 0.90 ms (5.3×) | 2.08 ms (12×) | 0.17 ms |
| steep-angle | 50 | 109 | 4.02 ms (19×) | 5.79 ms (27×) | 19.26 ms (103×) | 3.62 ms (21×) | 14.45 ms (62×) | 0.89 ms (4.4×) | 2.31 ms (10×) | 0.20 ms |
| blur | 50 | 96 | 3.68 ms (14×) | 4.97 ms (21×) | 17.49 ms (75×) | 3.91 ms (19×) | 6.56 ms (27×) | 1.01 ms (5.0×) | 2.15 ms (8.5×) | 0.22 ms |
| motion-blur | 50 | 110 | 3.76 ms (14×) | 5.11 ms (20×) | 17.59 ms (68×) | 3.93 ms (19×) | 9.05 ms (39×) | 0.96 ms (4.3×) | 2.23 ms (8.9×) | 0.22 ms |
| noise | 50 | 110 | 4.08 ms (14×) | 6.41 ms (22×) | 22.84 ms (82×) | 3.31 ms (12×) | 16.94 ms (54×) | 2.26 ms (8.2×) | 7.10 ms (23×) | 0.28 ms |
| uneven-light | 50 | 98 | 3.92 ms (16×) | 5.58 ms (23×) | 21.02 ms (89×) | 3.82 ms (17×) | 11.78 ms (57×) | 0.87 ms (3.6×) | 2.08 ms (9.1×) | 0.23 ms |
| low-contrast | 50 | 96 | 4.92 ms (39×) | 6.25 ms (49×) | 23.80 ms (184×) | 3.28 ms (24×) | 3.76 ms (29×) | 0.59 ms (4.5×) | 1.17 ms (8.8×) | 0.13 ms |
| underexposed | 50 | 110 | 4.00 ms (18×) | 5.67 ms (24×) | 21.56 ms (93×) | 4.24 ms (19×) | 9.76 ms (37×) | 0.96 ms (4.4×) | 1.77 ms (8.1×) | 0.23 ms |
| overexposed | 50 | 100 | 1.34 ms (5.6×) | 2.33 ms (9.9×) | 5.28 ms (22×) | 2.22 ms (11×) | 11.30 ms (51×) | 0.74 ms (3.4×) | 2.07 ms (8.7×) | 0.21 ms |
| jpeg | 50 | 104 | 1.41 ms (6.8×) | 2.47 ms (11×) | 5.55 ms (24×) | 2.64 ms (13×) | 9.14 ms (46×) | 0.93 ms (4.6×) | 2.69 ms (12×) | 0.22 ms |
| busy-background | 50 | 102 | 3.96 ms (17×) | 5.75 ms (24×) | 20.14 ms (83×) | 3.65 ms (16×) | 23.81 ms (89×) | 0.88 ms (3.9×) | 2.27 ms (9.2×) | 0.23 ms |
| frame-edge | 50 | 102 | 3.80 ms (18×) | 5.43 ms (25×) | 18.51 ms (100×) | 3.52 ms (21×) | 10.41 ms (45×) | 0.82 ms (4.6×) | 2.23 ms (11×) | 0.19 ms |
| noise-free | 50 | 98 | 0.66 ms (6.1×) | 0.70 ms (6.5×) | 1.82 ms (17×) | 1.41 ms (13×) | 4.43 ms (39×) | 0.48 ms (4.4×) | 1.09 ms (10×) | 0.11 ms |
| dark-background | 50 | 96 | 3.53 ms (19×) | 4.56 ms (25×) | 18.12 ms (110×) | 3.74 ms (22×) | 8.03 ms (43×) | 0.92 ms (5.5×) | 1.83 ms (10×) | 0.19 ms |

#### 640x480: tags found, and wrong IDs in brackets

| Scene | Tags | WPILib AprilTag, decimate 2 | AprilTag, decimate 2 | AprilTag, decimate 1 | Locus | OpenCV ArUco | ArUco Nano | RapidTag | NAPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 0 | — | — | — | — | — | — | — | — |
| clutter-only | 0 | — | — | — | — | — | — | — | — |
| decoys | 0 | — | — (1) | — (1) | — (1) | — (1) | — | — (1) | — (1) |
| one-tag | 50 | 46 | 50 | 50 | 50 | 50 | 50 | 27 | 50 |
| many-tags | 249 | 178 | 249 | 249 | 249 | 248 | 249 | 200 | 249 |
| small-tags | 131 | 0 | 131 | 131 | 131 | 131 | 131 | 84 | 131 |
| large-tags | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 43 | 50 |
| steep-angle | 109 | 71 | 109 | 109 | 109 | 109 | 109 | 97 | 105 |
| blur | 96 | 82 | 96 | 96 | 96 | 96 | 96 | 84 | 96 |
| motion-blur | 110 | 86 | 110 | 110 | 107 | 101 | 108 | 95 | 109 |
| noise | 110 | 94 | 110 | 110 | 110 | 109 | 110 | 87 (1) | 110 |
| uneven-light | 98 | 82 | 98 | 98 | 96 | 98 | 98 | 94 | 98 |
| low-contrast | 96 | 70 | 88 | 92 | 91 | 96 | 94 | 47 | 95 |
| underexposed | 110 | 96 | 110 | 110 | 79 | 110 | 110 | 110 | 110 |
| overexposed | 100 | 82 | 100 | 100 | 92 | 100 | 100 | 100 | 100 |
| jpeg | 104 | 92 | 104 | 104 | 104 | 104 | 104 | 91 | 104 |
| busy-background | 102 | 89 | 102 | 102 | 102 | 102 | 102 | 96 (1) | 102 |
| frame-edge | 102 | 74 | 102 | 102 | 102 | 97 | 102 | 89 | 102 |
| noise-free | 98 | 83 | 98 | 98 | 98 | 97 | 98 | 41 | 98 |
| dark-background | 96 | 71 | 96 | 96 | 96 | 96 | 96 | 13 | 96 |

#### 1280x720: median call, and NAPA's median per-frame speedup in brackets

| Scene | Frames | Tags | WPILib AprilTag, decimate 2 | AprilTag, decimate 2 | AprilTag, decimate 1 | Locus | OpenCV ArUco | ArUco Nano | RapidTag | NAPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 17.48 ms (242×) | 21.47 ms (282×) | 87.85 ms (1169×) | 8.13 ms (118×) | 2.70 ms (42×) | 0.90 ms (13×) | 2.36 ms (36×) | 0.07 ms |
| clutter-only | 50 | 0 | 11.04 ms (23×) | 15.26 ms (34×) | 64.46 ms (113×) | 9.01 ms (34×) | 23.08 ms (67×) | 2.49 ms (9.0×) | 6.99 ms (22×) | 0.30 ms |
| decoys | 50 | 0 | 15.28 ms (72×) | 19.48 ms (90×) | 77.65 ms (353×) | 7.77 ms (36×) | 18.81 ms (90×) | 3.57 ms (17×) | 3.87 ms (19×) | 0.21 ms |
| one-tag | 50 | 50 | 15.42 ms (103×) | 18.74 ms (122×) | 75.23 ms (505×) | 8.51 ms (57×) | 4.52 ms (31×) | 1.08 ms (7.3×) | 2.71 ms (18×) | 0.15 ms |
| many-tags | 50 | 354 | 11.16 ms (23×) | 14.90 ms (28×) | 56.16 ms (118×) | 10.30 ms (21×) | 24.44 ms (51×) | 2.65 ms (5.3×) | 5.87 ms (11×) | 0.48 ms |
| small-tags | 50 | 120 | 13.81 ms (57×) | 18.30 ms (73×) | 66.50 ms (278×) | 9.11 ms (36×) | 12.68 ms (38×) | 1.60 ms (6.9×) | 4.73 ms (19×) | 0.24 ms |
| large-tags | 50 | 50 | 12.13 ms (52×) | 16.85 ms (67×) | 61.78 ms (278×) | 9.16 ms (36×) | 17.18 ms (57×) | 2.05 ms (7.7×) | 5.23 ms (18×) | 0.26 ms |
| steep-angle | 50 | 112 | 12.55 ms (38×) | 17.26 ms (49×) | 62.75 ms (207×) | 9.30 ms (30×) | 23.97 ms (53×) | 2.01 ms (5.7×) | 5.50 ms (16×) | 0.33 ms |
| blur | 50 | 89 | 11.82 ms (38×) | 15.96 ms (49×) | 60.61 ms (199×) | 9.89 ms (32×) | 12.36 ms (33×) | 2.33 ms (7.7×) | 5.27 ms (16×) | 0.31 ms |
| motion-blur | 50 | 94 | 12.39 ms (20×) | 16.78 ms (24×) | 60.14 ms (83×) | 9.78 ms (31×) | 18.16 ms (34×) | 2.55 ms (5.9×) | 6.53 ms (14×) | 0.38 ms |
| noise | 50 | 104 | 11.23 ms (22×) | 17.70 ms (34×) | 71.05 ms (140×) | 8.62 ms (16×) | 39.34 ms (68×) | 6.18 ms (12×) | 20.12 ms (32×) | 0.51 ms |
| uneven-light | 50 | 103 | 12.87 ms (37×) | 17.59 ms (47×) | 67.83 ms (202×) | 9.79 ms (29×) | 17.54 ms (38×) | 2.19 ms (5.6×) | 5.66 ms (13×) | 0.37 ms |
| low-contrast | 50 | 100 | 15.90 ms (81×) | 19.31 ms (95×) | 78.62 ms (390×) | 8.81 ms (44×) | 5.91 ms (31×) | 1.26 ms (6.5×) | 3.04 ms (15×) | 0.20 ms |
| underexposed | 50 | 99 | 14.00 ms (44×) | 19.10 ms (59×) | 78.22 ms (242×) | 11.10 ms (36×) | 15.18 ms (41×) | 2.24 ms (6.9×) | 4.52 ms (13×) | 0.31 ms |
| overexposed | 50 | 112 | 2.60 ms (8.2×) | 4.46 ms (12×) | 10.48 ms (33×) | 4.62 ms (16×) | 17.54 ms (49×) | 1.45 ms (4.8×) | 3.88 ms (13×) | 0.26 ms |
| jpeg | 50 | 100 | 3.07 ms (11×) | 4.14 ms (14×) | 10.46 ms (34×) | 5.10 ms (19×) | 18.05 ms (55×) | 1.64 ms (6.4×) | 5.26 ms (18×) | 0.26 ms |
| busy-background | 50 | 87 | 12.75 ms (47×) | 18.00 ms (59×) | 66.07 ms (243×) | 9.25 ms (30×) | 37.77 ms (78×) | 1.95 ms (6.2×) | 5.39 ms (16×) | 0.29 ms |
| frame-edge | 50 | 105 | 12.97 ms (43×) | 16.75 ms (55×) | 62.37 ms (224×) | 9.09 ms (32×) | 16.18 ms (39×) | 2.02 ms (6.3×) | 5.28 ms (16×) | 0.30 ms |
| noise-free | 50 | 104 | 1.73 ms (9.9×) | 1.71 ms (9.8×) | 4.94 ms (28×) | 3.02 ms (17×) | 6.84 ms (41×) | 1.10 ms (6.4×) | 2.97 ms (17×) | 0.18 ms |
| dark-background | 50 | 105 | 12.15 ms (52×) | 15.09 ms (62×) | 62.79 ms (266×) | 10.16 ms (39×) | 8.21 ms (31×) | 1.73 ms (6.8×) | 3.62 ms (16×) | 0.25 ms |

#### 1280x720: tags found, and wrong IDs in brackets

| Scene | Tags | WPILib AprilTag, decimate 2 | AprilTag, decimate 2 | AprilTag, decimate 1 | Locus | OpenCV ArUco | ArUco Nano | RapidTag | NAPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 0 | — | — | — | — | — | — | — | — |
| clutter-only | 0 | — | — | — | — | — | — | — (1) | — |
| decoys | 0 | — | — | — | — | — | — | — | — |
| one-tag | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 26 | 50 |
| many-tags | 354 | 354 | 354 | 354 | 354 | 354 | 354 | 299 | 354 |
| small-tags | 120 | 0 | 120 | 120 | 120 | 120 | 120 | 68 | 120 |
| large-tags | 50 | 50 | 50 | 50 | 50 | 49 | 50 | 46 | 50 |
| steep-angle | 112 | 109 | 112 | 112 | 112 | 112 | 112 | 101 (1) | 111 |
| blur | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 77 | 89 |
| motion-blur | 94 | 94 | 94 | 94 | 94 | 93 | 94 | 83 | 94 |
| noise | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 93 (5) | 104 |
| uneven-light | 103 | 103 | 103 | 103 | 102 | 103 | 103 | 91 | 103 |
| low-contrast | 100 | 95 | 95 | 100 | 93 | 100 | 100 | 57 | 99 |
| underexposed | 99 | 99 | 99 | 99 | 69 | 99 | 99 | 99 | 99 |
| overexposed | 112 | 112 | 112 | 112 | 107 | 112 | 112 | 112 | 112 |
| jpeg | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 83 | 100 |
| busy-background | 87 | 87 | 87 (1) | 87 | 87 | 87 | 87 | 74 | 87 |
| frame-edge | 105 | 103 | 105 | 105 | 105 | 96 | 105 | 81 (2) | 105 |
| noise-free | 104 | 104 | 104 | 104 | 104 | 104 | 104 | 39 | 104 |
| dark-background | 105 | 91 | 105 | 105 | 105 | 105 | 105 | 7 | 105 |

#### 1920x1080: median call, and NAPA's median per-frame speedup in brackets

| Scene | Frames | Tags | WPILib AprilTag, decimate 2 | AprilTag, decimate 2 | AprilTag, decimate 1 | Locus | OpenCV ArUco | ArUco Nano | RapidTag | NAPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 44.39 ms (314×) | 54.52 ms (409×) | 212.09 ms (1612×) | 18.44 ms (153×) | 6.44 ms (55×) | 2.13 ms (19×) | 5.65 ms (46×) | 0.12 ms |
| clutter-only | 50 | 0 | 28.83 ms (69×) | 42.10 ms (95×) | 148.77 ms (365×) | 19.36 ms (53×) | 50.77 ms (50×) | 4.95 ms (10×) | 11.26 ms (26×) | 0.32 ms |
| decoys | 50 | 0 | 39.02 ms (104×) | 48.67 ms (130×) | 193.51 ms (518×) | 18.45 ms (55×) | 22.45 ms (67×) | 5.22 ms (16×) | 7.09 ms (22×) | 0.33 ms |
| one-tag | 50 | 50 | 38.27 ms (172×) | 46.84 ms (200×) | 191.29 ms (815×) | 19.02 ms (88×) | 8.98 ms (40×) | 2.18 ms (10×) | 5.88 ms (27×) | 0.22 ms |
| many-tags | 50 | 303 | 27.24 ms (41×) | 35.00 ms (48×) | 134.24 ms (207×) | 22.76 ms (32×) | 30.09 ms (39×) | 5.25 ms (7.8×) | 11.58 ms (18×) | 0.76 ms |
| small-tags | 50 | 122 | 31.31 ms (86×) | 42.18 ms (107×) | 155.53 ms (441×) | 19.86 ms (50×) | 25.20 ms (36×) | 3.09 ms (8.2×) | 9.51 ms (22×) | 0.38 ms |
| large-tags | 50 | 50 | 31.88 ms (90×) | 39.36 ms (107×) | 153.94 ms (441×) | 20.57 ms (53×) | 24.97 ms (44×) | 3.13 ms (8.9×) | 9.25 ms (23×) | 0.36 ms |
| steep-angle | 50 | 96 | 31.15 ms (72×) | 39.71 ms (86×) | 150.13 ms (357×) | 19.89 ms (47×) | 29.83 ms (39×) | 3.42 ms (9.0×) | 10.20 ms (22×) | 0.43 ms |
| blur | 50 | 97 | 31.34 ms (71×) | 41.07 ms (90×) | 153.98 ms (379×) | 22.53 ms (46×) | 23.71 ms (34×) | 4.53 ms (9.4×) | 10.07 ms (21×) | 0.47 ms |
| motion-blur | 50 | 95 | 31.04 ms (80×) | 38.59 ms (91×) | 148.69 ms (370×) | 19.89 ms (46×) | 22.43 ms (36×) | 3.57 ms (7.8×) | 9.35 ms (20×) | 0.43 ms |
| noise | 50 | 100 | 31.38 ms (33×) | 44.54 ms (49×) | 181.86 ms (190×) | 19.23 ms (17×) | 82.05 ms (63×) | 13.85 ms (14×) | 46.82 ms (34×) | 1.02 ms |
| uneven-light | 50 | 101 | 33.25 ms (66×) | 42.64 ms (78×) | 168.94 ms (320×) | 22.29 ms (40×) | 35.95 ms (50×) | 4.35 ms (6.9×) | 13.15 ms (18×) | 0.56 ms |
| low-contrast | 50 | 113 | 37.83 ms (122×) | 44.93 ms (144×) | 177.42 ms (586×) | 19.52 ms (63×) | 11.91 ms (37×) | 2.49 ms (8.6×) | 6.64 ms (21×) | 0.31 ms |
| underexposed | 50 | 95 | 36.10 ms (96×) | 46.26 ms (121×) | 187.11 ms (506×) | 22.49 ms (61×) | 22.39 ms (44×) | 3.88 ms (9.5×) | 8.35 ms (21×) | 0.38 ms |
| overexposed | 50 | 100 | 9.29 ms (11×) | 11.10 ms (15×) | 38.40 ms (44×) | 8.67 ms (20×) | 26.85 ms (39×) | 3.25 ms (6.2×) | 9.32 ms (18×) | 0.39 ms |
| jpeg | 50 | 100 | 7.79 ms (13×) | 9.46 ms (14×) | 27.24 ms (45×) | 11.49 ms (23×) | 29.55 ms (40×) | 3.62 ms (7.4×) | 12.16 ms (22×) | 0.43 ms |
| busy-background | 50 | 97 | 30.38 ms (78×) | 41.29 ms (96×) | 145.04 ms (380×) | 19.52 ms (46×) | 61.28 ms (81×) | 3.44 ms (7.7×) | 9.94 ms (21×) | 0.41 ms |
| frame-edge | 50 | 88 | 31.91 ms (86×) | 39.91 ms (104×) | 152.63 ms (423×) | 19.67 ms (47×) | 26.99 ms (40×) | 3.21 ms (8.4×) | 14.51 ms (25×) | 0.39 ms |
| noise-free | 50 | 108 | 3.49 ms (14×) | 3.38 ms (14×) | 10.56 ms (43×) | 5.74 ms (23×) | 10.88 ms (47×) | 1.89 ms (8.3×) | 6.13 ms (25×) | 0.25 ms |
| dark-background | 50 | 107 | 30.89 ms (91×) | 37.50 ms (110×) | 156.60 ms (466×) | 20.63 ms (57×) | 16.90 ms (41×) | 3.23 ms (8.9×) | 8.36 ms (23×) | 0.35 ms |

#### 1920x1080: tags found, and wrong IDs in brackets

| Scene | Tags | WPILib AprilTag, decimate 2 | AprilTag, decimate 2 | AprilTag, decimate 1 | Locus | OpenCV ArUco | ArUco Nano | RapidTag | NAPA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 0 | — | — | — | — | — | — | — | — |
| clutter-only | 0 | — | — | — | — | — | — | — (1) | — |
| decoys | 0 | — | — | — | — | — | — | — | — |
| one-tag | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 28 | 50 |
| many-tags | 303 | 303 | 303 | 303 | 303 | 300 | 303 | 219 | 303 |
| small-tags | 122 | 0 | 122 | 122 | 122 | 122 | 122 | 81 (1) | 121 |
| large-tags | 50 | 50 | 50 (1) | 50 | 50 | 50 | 50 | 46 | 50 |
| steep-angle | 96 | 88 | 96 | 96 (1) | 96 | 96 | 96 | 83 (1) | 96 |
| blur | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 81 | 97 |
| motion-blur | 95 | 95 | 95 | 95 | 95 | 95 | 95 | 83 | 95 |
| noise | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 80 (2) | 100 |
| uneven-light | 101 | 101 | 101 | 101 (1) | 101 | 101 | 101 | 96 | 101 |
| low-contrast | 113 | 108 | 108 | 113 | 104 | 113 | 113 | 40 | 111 |
| underexposed | 95 | 95 | 95 | 95 | 81 | 95 | 95 | 95 | 95 |
| overexposed | 100 | 100 | 100 | 100 | 87 | 100 | 100 | 100 | 100 |
| jpeg | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 89 | 100 |
| busy-background | 97 | 97 | 97 | 97 | 97 | 97 | 97 | 87 (1) | 97 |
| frame-edge | 88 | 88 | 88 | 88 | 88 | 84 | 88 | 73 (1) | 88 |
| noise-free | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 45 | 108 |
| dark-background | 107 | 89 | 107 | 107 | 107 | 107 | 107 | 7 | 107 |

The JSON report also holds the 95th and 99th percentiles, CPU time per call, and corner error by scene; the CSV holds
every timing sample and detection.

## Range

From the same run: tags found out of 50 at each black-square size, in pixels.

| Detector | 12 | 14 | 16 | 18 | 20 | 22 | 24–32 | 36 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WPILib AprilTag, decimate 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 39 | 48 | 50 | 50 | 50 |
| AprilTag, decimate 2 | 4 | 8 | 9 | 20 | 33 | 45 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| AprilTag, decimate 1 | 49 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| Locus | 44 | 48 | 46 | 49 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| OpenCV ArUco | 40 | 44 | 49 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 49 |
| ArUco Nano | 36 | 46 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| RapidTag | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 21 | 25 | 35 | 29 | 30 | 31 | 26 | 24 | 26 |
| NAPA | 0 | 0 | 14 | 37 | 47 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |

NAPA's default found every tag from 22 px, upstream AprilTag at decimate 2 from 24 px, Locus from 20 px, ArUco Nano
from 16 px, and upstream AprilTag at decimate 1 from 14 px. WPILib's AprilTag at decimate 2 needed 80 px. OpenCV
found every tag from 18 px except one at 96 px, and RapidTag never found all of any size. A tag's size in the image
is inversely proportional to its distance, so under this test's conditions (light noise, up to 15° of tilt) NAPA's
default would see about 3.6× as far as WPILib's AprilTag, and detectors that find smaller tags would see farther.
With `min_tag_px=16`, NAPA found every tag down to 12 px (`results/2026-10-28_cpu_comparison/range_sweep_latency.json`,
30 tags per size).

## Tag families

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
  their sources document. Over the 3,000 frames, NAPA's median was 0.021 px and its 95th percentile 0.093 px, against
  0.099 px and 0.303 px for WPILib's AprilTag, 0.043 px and 0.305 px for upstream AprilTag at decimate 2, and 0.044 px
  and 0.316 px for ArUco Nano. OpenCV and RapidTag don't refine corners by default (about 1 px).
- A wrong ID is any detection whose ID isn't in the frame, or a second detection of the same tag.
- Recall in `small-tags` and `steep-angle` depends on apparent size. A steeply tilted tag can look narrower than
  `min_tag_px` even when its long side is well above it.
- NAPA's one wrong ID was a tag-shaped decoy whose random payload fell within 2 bits of a real code, the same
  tolerance AprilTag uses by default. In a separate test of 80,000 such decoys, NAPA reported none.
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
