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

A range sweep measures how small a tag each detector can still find, which sets how far away it can see one. It
renders one tag per frame with a black square of exactly 12, 14, 16, 20, 24, 28, 32, 40, 48, 64, or 96 px, at a
random ID and angle, up to 15° of tilt, and light noise (σ ≈ 1.5). There are 50 tags at each size (550 frames), at
the smallest resolution. Results are in `benchmarks/results/2026-10-19_range_sweep.json`.

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
- `--occlusion-cases 0` skips the obstruction sweep, and `--range-cases 0` the range sweep.
- `--frames 0` runs only the sweeps. The range results in the main README came from
  `--frames 0 --resolutions 640x480 --occlusion-cases 0`. They count tags found, not time, so any machine gives the
  same numbers.
- `--seed` changes every frame. Each frame's seed depends only on the run seed, its scenario, resolution, and
  index, so any frame can be re-rendered on its own with `render_frame()`.

The script prints the summary tables used in the main README and writes two files:

- `*.json`: run configuration, machine details, per-scenario results (frames, tags, found, false IDs, latency
  percentiles, corner error), per-frame speedup statistics, and the obstruction sweep.
- `*.csv`: one row per frame and backend, with expected and detected IDs, corner errors, every raw timing sample,
  and the scenario's random parameters (`detail_*` columns). Use it to find and re-render any frame a detector
  missed.

## Latest results

Standard AprilTag (`quad_decimate=2`) against Napa, from the checked-in run
(`results/2026-10-19_detector_matrix.json`) on one pinned core of an Intel Core Ultra 7 256V laptop: Windows 11 with
the best-performance power mode, the benchmark running under WSL2 (Ubuntu-like Linux, Python 3.13, gcc 15). Earlier
checked-in results came from a desktop Intel i7-12700H (Ubuntu), so their absolute times differ. p50 is the median
detector call across all frames and repetitions in a row.

#### 640x480

| Scene | Frames | Tags | AprilTag p50 | Napa p50 | Speedup (median / worst frame) | Found (AprilTag → Napa) | False IDs (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 4.66 ms | 0.07 ms | 53.9× / 12.1× | — | 0 → 0 |
| clutter-only | 50 | 0 | 3.08 ms | 0.16 ms | 14.7× / 7.0× | — | 0 → 0 |
| decoys | 50 | 0 | 3.20 ms | 0.15 ms | 19.7× / 7.4× | — | 0 → 1 |
| one-tag | 50 | 50 | 3.86 ms | 0.13 ms | 27.5× / 20.9× | 46/50 → 50/50 | 0 → 0 |
| many-tags | 50 | 249 | 2.96 ms | 0.26 ms | 10.7× / 6.0× | 178/249 → 249/249 | 0 → 0 |
| small-tags | 50 | 131 | 3.32 ms | 0.22 ms | 12.8× / 7.8× | 0/131 → 131/131 | 0 → 0 |
| large-tags | 50 | 50 | 3.00 ms | 0.18 ms | 15.0× / 7.0× | 50/50 → 50/50 | 0 → 0 |
| steep-angle | 50 | 109 | 3.40 ms | 0.21 ms | 15.5× / 6.7× | 71/109 → 105/109 | 0 → 0 |
| blur | 50 | 96 | 3.01 ms | 0.22 ms | 13.0× / 5.9× | 82/96 → 96/96 | 0 → 0 |
| motion-blur | 50 | 110 | 3.19 ms | 0.23 ms | 12.2× / 6.0× | 86/110 → 109/110 | 0 → 0 |
| noise | 50 | 110 | 3.35 ms | 0.29 ms | 11.4× / 3.8× | 94/110 → 110/110 | 0 → 0 |
| uneven-light | 50 | 98 | 3.17 ms | 0.23 ms | 12.9× / 6.9× | 82/98 → 98/98 | 0 → 0 |
| low-contrast | 50 | 96 | 4.14 ms | 0.16 ms | 25.2× / 16.1× | 70/96 → 95/96 | 0 → 0 |
| underexposed | 50 | 110 | 3.32 ms | 0.23 ms | 13.7× / 7.9× | 96/110 → 110/110 | 0 → 0 |
| overexposed | 50 | 100 | 1.09 ms | 0.18 ms | 5.3× / 3.9× | 82/100 → 100/100 | 0 → 0 |
| jpeg | 50 | 104 | 1.10 ms | 0.18 ms | 6.6× / 3.6× | 92/104 → 104/104 | 0 → 0 |
| busy-background | 50 | 102 | 3.15 ms | 0.22 ms | 13.3× / 7.3× | 89/102 → 102/102 | 0 → 0 |
| frame-edge | 50 | 102 | 3.05 ms | 0.21 ms | 13.7× / 7.8× | 74/102 → 102/102 | 0 → 0 |
| noise-free | 50 | 98 | 0.50 ms | 0.09 ms | 5.7× / 4.8× | 83/98 → 98/98 | 0 → 0 |
| dark-background | 50 | 96 | 2.71 ms | 0.19 ms | 15.5× / 6.6× | 71/96 → 96/96 | 0 → 0 |

#### 1280x720

| Scene | Frames | Tags | AprilTag p50 | Napa p50 | Speedup (median / worst frame) | Found (AprilTag → Napa) | False IDs (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 14.89 ms | 0.12 ms | 100.6× / 12.1× | — | 0 → 0 |
| clutter-only | 50 | 0 | 9.16 ms | 0.31 ms | 18.8× / 5.1× | — | 0 → 0 |
| decoys | 50 | 0 | 12.74 ms | 0.23 ms | 52.1× / 8.5× | — | 0 → 0 |
| one-tag | 50 | 50 | 13.07 ms | 0.20 ms | 65.4× / 46.0× | 50/50 → 50/50 | 0 → 0 |
| many-tags | 50 | 354 | 9.64 ms | 0.49 ms | 19.7× / 8.0× | 354/354 → 354/354 | 0 → 0 |
| small-tags | 50 | 120 | 11.59 ms | 0.29 ms | 40.5× / 11.6× | 0/120 → 120/120 | 0 → 0 |
| large-tags | 50 | 50 | 10.56 ms | 0.29 ms | 37.2× / 13.4× | 50/50 → 50/50 | 0 → 0 |
| steep-angle | 50 | 112 | 10.59 ms | 0.35 ms | 30.6× / 11.6× | 109/112 → 111/112 | 0 → 0 |
| blur | 50 | 89 | 10.19 ms | 0.32 ms | 31.5× / 9.2× | 89/89 → 89/89 | 0 → 0 |
| motion-blur | 50 | 94 | 10.20 ms | 0.38 ms | 17.0× / 9.1× | 94/94 → 94/94 | 0 → 0 |
| noise | 50 | 104 | 9.66 ms | 0.52 ms | 18.7× / 6.0× | 104/104 → 104/104 | 0 → 0 |
| uneven-light | 50 | 103 | 10.95 ms | 0.37 ms | 29.2× / 12.0× | 103/103 → 103/103 | 0 → 0 |
| low-contrast | 50 | 100 | 13.88 ms | 0.23 ms | 66.0× / 38.9× | 95/100 → 99/100 | 0 → 0 |
| underexposed | 50 | 99 | 11.27 ms | 0.33 ms | 35.4× / 13.6× | 99/99 → 99/99 | 0 → 0 |
| overexposed | 50 | 112 | 2.04 ms | 0.26 ms | 7.2× / 5.4× | 112/112 → 112/112 | 0 → 0 |
| jpeg | 50 | 100 | 2.58 ms | 0.25 ms | 8.0× / 4.2× | 100/100 → 100/100 | 0 → 0 |
| busy-background | 50 | 87 | 10.87 ms | 0.30 ms | 37.7× / 11.3× | 87/87 → 87/87 | 0 → 0 |
| frame-edge | 50 | 105 | 10.85 ms | 0.32 ms | 35.0× / 12.2× | 103/105 → 105/105 | 0 → 0 |
| noise-free | 50 | 104 | 1.35 ms | 0.16 ms | 8.1× / 6.9× | 104/104 → 104/104 | 0 → 0 |
| dark-background | 50 | 105 | 10.35 ms | 0.29 ms | 38.1× / 12.0× | 91/105 → 105/105 | 0 → 0 |

#### 1920x1080

| Scene | Frames | Tags | AprilTag p50 | Napa p50 | Speedup (median / worst frame) | Found (AprilTag → Napa) | False IDs (AprilTag → Napa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| empty | 50 | 0 | 38.10 ms | 0.16 ms | 162.0× / 18.0× | — | 0 → 0 |
| clutter-only | 50 | 0 | 25.54 ms | 0.42 ms | 50.3× / 6.1× | — | 0 → 0 |
| decoys | 50 | 0 | 34.13 ms | 0.34 ms | 88.8× / 12.1× | — | 0 → 0 |
| one-tag | 50 | 50 | 33.29 ms | 0.25 ms | 131.2× / 85.2× | 50/50 → 50/50 | 0 → 0 |
| many-tags | 50 | 303 | 23.72 ms | 0.74 ms | 37.0× / 12.7× | 303/303 → 303/303 | 0 → 0 |
| small-tags | 50 | 122 | 26.47 ms | 0.46 ms | 65.8× / 13.3× | 0/122 → 121/122 | 0 → 0 |
| large-tags | 50 | 50 | 26.91 ms | 0.41 ms | 73.7× / 16.3× | 50/50 → 50/50 | 0 → 0 |
| steep-angle | 50 | 96 | 26.62 ms | 0.47 ms | 63.2× / 13.9× | 88/96 → 96/96 | 0 → 0 |
| blur | 50 | 97 | 25.24 ms | 0.53 ms | 57.3× / 10.2× | 97/97 → 97/97 | 0 → 0 |
| motion-blur | 50 | 95 | 27.15 ms | 0.48 ms | 67.7× / 12.0× | 95/95 → 95/95 | 0 → 0 |
| noise | 50 | 100 | 27.00 ms | 0.97 ms | 29.8× / 6.0× | 100/100 → 100/100 | 0 → 0 |
| uneven-light | 50 | 101 | 27.59 ms | 0.60 ms | 54.4× / 16.1× | 101/101 → 101/101 | 0 → 0 |
| low-contrast | 50 | 113 | 32.71 ms | 0.33 ms | 100.7× / 60.7× | 108/113 → 111/113 | 0 → 0 |
| underexposed | 50 | 95 | 31.68 ms | 0.42 ms | 85.7× / 15.2× | 95/95 → 95/95 | 0 → 0 |
| overexposed | 50 | 100 | 8.24 ms | 0.44 ms | 9.9× / 5.7× | 100/100 → 100/100 | 0 → 0 |
| jpeg | 50 | 100 | 7.06 ms | 0.52 ms | 11.8× / 4.5× | 100/100 → 100/100 | 0 → 0 |
| busy-background | 50 | 97 | 26.80 ms | 0.45 ms | 69.0× / 14.1× | 97/97 → 97/97 | 0 → 0 |
| frame-edge | 50 | 88 | 27.27 ms | 0.42 ms | 70.5× / 14.0× | 88/88 → 88/88 | 0 → 0 |
| noise-free | 50 | 108 | 2.96 ms | 0.24 ms | 12.6× / 10.3× | 108/108 → 108/108 | 0 → 0 |
| dark-background | 50 | 107 | 27.57 ms | 0.40 ms | 77.2× / 16.7× | 89/107 → 107/107 | 0 → 0 |

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

## Stereo paired replay

Run the joint solver on recorded, caller-paired images using a `StereoCalibration` artifact:

```bash
python benchmarks/stereo_replay.py corpus/pairs.json \
  --calibration rig.json --max-capture-skew-s 0.005 --output stereo-results.json
```

The strict version-1 manifest embeds the field layout (rigid WPILib field-to-tag transforms in metres) and records
each exposure midpoint on a common clock. For example, this layout places tag 1 at field x=3, facing the origin:

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

Image paths must remain inside the manifest directory, camera IDs must match the artifact, and resolutions must
match the intrinsics. `ground_truth` is optional. Provide truth samples at each monocular capture time and the
pair midpoint when scoring a moving rig. Errors and conditional NEES are `null` unless a truth timestamp matches
the result within 1 ns; the evaluator does not interpolate or compare poses from different times.

Each pair reports stereo, left/right monocular, and `independent_pose_fusion` results. The fusion baseline combines
the best monocular poses using a local Gaussian information-weighted update in field axes. It assumes independent
conditional uncertainties and a static pair, does not search alternate-mode combinations or compensate motion,
and rejects pairs exceeding the same skew gate. It is a comparison baseline, not a second public solver API.

Outputs include acceptance/rejection reasons, accepted camera/tag groups, candidate count and joint RMS values,
translation/rotation errors, six conditional standard deviations, and conditional NEES (squared pose error weighted
by inverse covariance). Calibration error and clock error are excluded from those uncertainty estimates. Timing
reports p50/p95 for detecting both images and for joint solving; file loading and baseline solves are excluded.
Calls are measured once per pair without warmup, so this is replay latency, not a controlled microbenchmark.

Tests cover synthetic noisy geometry, ambiguity, calibration perturbations, timing mismatches, and rendered tags
through the real native detector. No real-camera paired corpus is included, so these tests establish integration
and numerical behavior rather than a measured stereo accuracy improvement.
