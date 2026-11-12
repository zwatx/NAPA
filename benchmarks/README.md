# Benchmarks

Two scripts compare NAPA with other detectors on the same generated frames:

- `performance_matrix.py` compares NAPA with AprilTag 3 as WPILib builds and configures it (`robotpy-apriltag`, at
  its default `quad_decimate=2` and at full resolution).
- `cpu_comparison.py` runs the same frames through upstream AprilTag and other CPU detectors (see
  [Other CPU detectors](#other-cpu-detectors)).

Both time detector calls only. Camera delivery, colour conversion, and pose solving sit outside the timed region.

The headline results come from the [2026-10-30 run](results/2026-10-30_cpu_full_corpus/report.md), which ran NAPA and
13 configurations of other detectors on the same 3,000 frames. Per-scene numbers are in that run's
`matrix.json` and `matrix.csv`, next to the report.

For scoring NAPA on your own camera footage instead, see
[Measuring on your own footage](../docs/measuring.md).

## The corpus

Every frame is rendered on its own. No image is reused, and none is a copy of another with a different label. Each
tag gets a random ID from all 587, a random position, any in-plane angle, a random 3D tilt (and therefore
perspective), and random black and white levels. Tags never overlap, and each keeps its white quiet zone inside the
frame.

Tags are rendered with exact sub-pixel ground truth. Every pixel is the mean of a 4x4 grid of point samples, as a
sensor integrates light. The benchmark therefore scores three things: whether each tag was found, any invented ID,
and how far each corner landed from where it really is.

| Scenario | What varies |
| --- | --- |
| `empty` | A uniform frame at random brightness, with light noise. No tags. |
| `clutter-only` | Shapes, text, checkerboards, or smooth natural texture. No tags. |
| `decoys` | Black-bordered squares with random payloads. Tag-shaped, but not valid codes. |
| `one-tag` | One tag on a plain background |
| `many-tags` | 2 to 12 tags of mixed sizes on mixed backgrounds |
| `small-tags` | 1 to 4 tags with a 32 to 53 px black square, at and just above `min_tag_px=32` |
| `large-tags` | One tag filling 45 to 80% of the frame height |
| `steep-angle` | Tags tilted 40 to 65 degrees away from the camera |
| `blur` | Gaussian defocus, sigma 0.8 to 2.0 px |
| `motion-blur` | Linear motion blur of 3 to 11 px (odd lengths, so the kernel is centred) in any direction |
| `noise` | Gaussian sensor noise, sigma 4 to 14 grey levels |
| `uneven-light` | A lighting gradient (30 to 60% at the dark end) plus vignetting |
| `low-contrast` | Tag black and white only 40 to 90 grey levels apart |
| `underexposed` | Everything scaled to 15 to 35% brightness, then noise |
| `overexposed` | Gain of 1.3 to 1.8, with clipping |
| `jpeg` | JPEG compression at quality 25 to 70 |
| `busy-background` | Tags on text, shapes, or checkerboards |
| `frame-edge` | Tags whose quiet zone touches the edge of the frame |
| `noise-free` | 1 to 3 tags on a plain background with no noise at all. The best case for every detector. |
| `dark-background` | Tags whose one-cell white margin is all that separates them from a dark or mid-grey background |

Unless a scenario says otherwise, positive frames hold 1 to 3 tags on a random background, and tag size scales with
the frame (48 to 170 px at 480 lines). Every scenario except `noise-free` adds at least light sensor noise (sigma
about 1.5 grey levels), as a real sensor would. The default run renders 50 frames per scenario at each of 640x480,
1280x720, and 1920x1080: 3,000 frames in all.

Two sweeps sit beside the main corpus:

- **Obstruction.** One tag per frame, partly hidden by a white or grey patch over the payload centre, along one
  border, or over a corner. The patch covers 0 to 40% of the black square. There are 24 random tags for each
  combination, 864 frames at the smallest resolution.
- **Range.** One tag per frame with a black square of exactly 12 to 96 px, at a random ID and angle, up to 15 degrees
  of tilt, with light noise. The default sweep uses 50 tags at each size at the smallest resolution. The
  [Range](#range) table below comes from that sweep in the fixed full run.

## Timing

Backends take turns on each frame in a shuffled order: ten timed calls each, after one untimed warm-up call. The
report gives every backend's latency percentiles per scenario and resolution.

It also gives NAPA's speedup per frame: the other detector's median time on a frame divided by NAPA's median time on
the same frame. "Worst frame" is the smallest of those ratios. That way a single slow frame cannot hide behind a
good median.

## Running it

```bash
python -m pip install -e ".[headless,dev,benchmark]"

# Pin to one idle core so other processes do not skew timings (Linux shown). About 90 minutes.
taskset -c 2 python benchmarks/performance_matrix.py --output benchmarks/results/$(date +%F)_detector_matrix.json
```

Useful options:

- `--scenarios noise,jpeg`, `--resolutions 1280x720`, and `--frames 10` select a quick subset.
- `--backends napa,apriltag,apriltag-full-res,opencv-aruco` chooses what to compare. OpenCV is off by default.
- `--occlusion-cases 0` skips the obstruction sweep. `--range-cases 0` skips the range sweep.
- `--frames 0` runs only the sweeps. A range-only run is
  `--frames 0 --resolutions 640x480 --occlusion-cases 0`. The published tables instead come from the
  [fixed shared run](results/2026-10-30_cpu_full_corpus/commands.md), including its range frames.
- `--seed` changes every frame. Each frame's seed depends only on the run seed, its scenario, resolution, and index,
  so any frame can be re-rendered on its own with `render_frame()`.

The script prints summary tables and writes two files:

- `*.json`: run configuration, machine details, per-scenario results (frames, tags, found, false IDs, latency
  percentiles, corner error), per-frame speedup statistics, and the obstruction sweep.
- `*.csv`: one row per frame and backend, with expected and detected IDs, corner errors, every raw timing sample,
  and the scenario's random parameters (`detail_*` columns). Use it to find, and re-render, any frame a detector
  missed.

## Latest results

The [2026-10-30 run](results/2026-10-30_cpu_full_corpus/report.md) gave every detector the same 3,000 frames: 50 per
scene at each resolution, one warm-up and ten timed calls per frame in shuffled order, on one performance core of an
Intel i7-12700H. "NAPA" there is `min_tag_px=32`, the default when it ran. The current default is `min_tag_px=22`
without the shadow pass (see [Choosing settings](../docs/detection.md#choosing-settings)).

AprilTag at WPILib's defaults is `robotpy-apriltag` as WPILib configures it: decimate 2, a 300 px minimum cluster,
and a 45-degree critical angle. The other AprilTag rows use the upstream C library's defaults. OpenCV ArUco and
RapidTag refine corners only in their subpixel rows. The two limited ArUco3 rows detect at half scale, which sets the
smallest tag they can find.

The report has the full picture. Every configuration's exact settings, latency percentiles, recall and corner quality
per scene and resolution, and NAPA's corners against each detector's on the tags both found. The root
[README](../README.md#how-it-compares) carries the summary table. Two results from the same run are worth keeping
here:

**Stress frames.** A separate 600 frames hold tags under 32 px, noise at sigma 15 to 30, 30 to 80 tags per frame, and
noisy clutter:

| Detector | Median ms | p95 ms | p99 ms | Tags found | Wrong IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| NAPA | 0.600 | 2.201 | 2.728 | 4390/4577 | 0 |
| NAPA, `min_tag_px=16` | 1.561 | 6.597 | 10.638 | 4577/4577 | 0 |
| AprilTag, WPILib defaults | 11.301 | 40.119 | 45.502 | 3011/4577 | 0 |
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

No detector reported the same tag twice on the main corpus.

## Range

Tags found out of 50 at each black-square size, from the same run. 900 frames at 640x480, with light noise and up to 15
degrees of tilt:

| Detector | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 28 | 32 | 36 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NAPA | 0 | 0 | 14 | 37 | 47 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| NAPA, `min_tag_px=16` | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| AprilTag, WPILib defaults | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 39 | 48 | 50 | 50 | 50 |
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

At `min_tag_px=32`, then the default, NAPA found every tag from 22 px up. At `min_tag_px=16` it found every tag at
every size tested. In the [2026-11-11 full benchmark](results/2026-11-11_full_benchmark/report.md), on the range
sweep's frames, the current default (`min_tag_px=22`) found 893 of the 900 tags, against 824 for `min_tag_px=32` with
that run's build.

Real cameras add blur, noise, and steeper angles. Measure range on your own.

## Tag families

These family results predate the native-core optimization work and have not been rerun. Treat the speedups as rough.

A separate family-by-family run, not part of `performance_matrix.py`, compared NAPA with upstream AprilTag only. Most
of the other detectors above read only a few of these families. It used the same scenes with each family's own tags,
spaced by their printed size, at 640x480 and 1280x720. Speedups are against AprilTag at decimate 2, per frame, from a
timing pass over 80 frames per family on one pinned core.

| Family | Tags | Found: AprilTag to NAPA | Wrong IDs: AprilTag to NAPA | Median corner error: AprilTag to NAPA | Speedup: median (slowest frame) |
| --- | ---: | ---: | ---: | ---: | ---: |
| tag36h11 | 289 | 288 to 287 | 0 to 0 | 0.097 to 0.022 px | 40x (7.9x) |
| tag16h5 | 282 | 282 to 280 | 581 to 0 | 0.107 to 0.022 px | 38x (5.5x) |
| tag25h9 | 287 | 286 to 286 | 3 to 0 | 0.098 to 0.021 px | 36x (7.2x) |
| tagStandard41h12 | 232 | 232 to 231 | 0 to 0 | 0.097 to 0.025 px | 42x (8.5x) |
| tagCircle21h7 | 231 | 231 to 230 | 9 to 0 | 0.097 to 0.025 px | 42x (9.3x) |

Every tag that only AprilTag found was small (40 to 60 px) and steeply tilted, which narrows the white margin to a pixel
or two. Either its narrow side was under `min_tag_px`, or a dark line in the background touched the tag and joined its
border. On tagStandard52h13, tagCircle49h12, tagCustom48h12, and tag36h10, which the reference Python binding does not
include, NAPA found 99.2 to 99.6% of tags with no wrong IDs. Upstream's decode tables for the three largest families need
several gigabytes. NAPA's take megabytes.

tag16h5's 30 codes are only 5 bits apart. A decoder that corrects two wrong bits, as AprilTag does by default, accepts
about one in four random tag-shaped patterns. [Tag families](../docs/api.md#tag-families) explains how NAPA sets each
family's correction.

## Reading the results

- Compare scenario rows at the same resolution rather than overall averages. The scenarios differ a lot in cost.
- Corner error is the worst corner's distance from the truth, for each tag found, after each detector's corner order is
  aligned to the truth. AprilTag's and Locus's corners are shifted by -0.5 px into the pixel-centre convention, as their
  sources document. Compare errors on tags both detectors found, so a detector cannot improve its score by missing hard
  tags. The [current report](results/2026-10-30_cpu_full_corpus/report.md#corner-quality-on-the-main-corpus) gives median,
  p95, and maximum errors separately. OpenCV and RapidTag do not refine corners by default.
- Wrong IDs and duplicate detections are counted separately. There were no duplicates in the full main-corpus comparison.
- Recall in `small-tags` and `steep-angle` depends on apparent size. A steeply tilted tag can look narrower than
  `min_tag_px` even when its long side is well above it.
- The frames are synthetic and the timings come from one laptop CPU. Measure on your own camera and hardware before you
  rely on a specific number.

## Other CPU detectors

`cpu_comparison.py` runs the same corpus, plus the stress cases and the range sweep, through other CPU tag36h11
detectors: upstream AprilTag 3 (C library), [Locus](https://github.com/NoeFontana/locus-tag), OpenCV ArUco (including
`useAruco3Detection`), [ArUco Nano](https://github.com/rmsalinas/aruco_nano), and
[RapidTag](https://github.com/SujithChristopher/rapidtag), each at its defaults and in labelled tuned configurations.
`cpu_backends.py` holds one adapter per library. Each adapter converts that library's pixel origin and reports its corner
order. The C shims for AprilTag and ArUco Nano live in `cpu_comparison_shims/`.

Three subcommands:

- `validate` checks each adapter against known geometry (quarter turns, tilts, a rotated frame, blank frames) before any
  score is used.
- `latency` runs every backend on every frame in one process, interleaved in shuffled order. It keeps false IDs,
  duplicates, corner error after aligning corner order, and the corner order itself as separate results.
- `resources` runs each backend alone in its own pinned process. It measures CPU time per frame, peak memory, and
  threading, both several threads per frame and one detector per worker thread processing frames in parallel.

The libraries are not dependencies of this package. Build them in a scratch directory, as
`results/2026-10-28_cpu_comparison/commands.md` shows. That directory holds the first comparison: raw timings,
configuration and machine details, and a report covering methods, tradeoffs, and open questions.

## Stereo paired replay

`stereo_replay.py` scores the [stereo solver](../docs/stereo.md) against each camera alone and a covariance-weighted
fusion of the two, on recorded pairs:

```bash
python benchmarks/stereo_replay.py corpus/pairs.json \
  --calibration rig.json --max-capture-skew-s 0.005 --output stereo-results.json
```

The manifest holds the tag layout (WPILib axes, metres), paired image paths, exposure-midpoint capture times on one
clock, and optional ground truth. Image paths must stay in the manifest folder, camera IDs and resolutions must match
the calibration, and poses score only against truth within 1 ns of their timestamp. Output records acceptance, RMS
error, pose error vs truth, std devs, NEES, and separate detect/solve timings. The fusion baseline assumes the rig did
not move between frames. It is a yardstick, not a public solver.

## Independent datasets

`external.py` scores unchanged third-party images instead of our renders. Scoring here needs the right ID **and** quad
IoU ≥ 0.5, one-to-one. Duplicates and unmatched detections count as false positives. Only complete ground-truth tags
count toward recall, so unmatched detections may be unlabelled partial tags, not inventions. It reports recall,
precision, corner error, and `corner_accuracy_at_1px` separately. No single "accuracy" number.

**Locus-Tag Bench** (700 images, 11 configs, pinned to `5346835a`):

```bash
python -m pip install pyarrow
python -m benchmarks.external --download
python -m benchmarks.external --output work/external/locus-results.json
python -m benchmarks.external --backends napa napa16 --output work/external/napa-results.json  # NAPA only
```

`napa-split`/`napa16-split` add `split_touching=True` for its AprilGrid boards
([report](results/2026-10-31_split_touching/report.md)).

**ICRA 2020 accuracy dataset.** Extract a scenario archive (mirror `6c765338`) under `work/external/icra/`, then:

```bash
python -m benchmarks.external --root work/external/icra --configs icra-forward \
  --backends napa napa16 apriltag3-d1 --output work/external/icra-forward.json
```

Published measurements cover forward only. One corrupt image is missing from `circle`; scoring it needs a recorded
exclusion, never a silent smaller denominator.

**bdcudatag sets.** Regenerate into `work/` (scene generator at `4ff7a31a`, 32 scenes, 30 hard frames), then
`python -m benchmarks.bdcudatag --backends napa napa16 apriltag3-d1 --output work/external/bdcudatag.json`. Scored by
their rules (right ID, centre within 0.75 tag sides). On the 2,389-tag hard corpus NAPA finds 98.5% at the default vs
96.0% in bdcudatag's published table. Their truth cannot rank corner accuracy, and their GPU timings are not
like-for-like. [Report](results/2026-11-11_bdcudatag/report.md).

**vkapriltag corpus.** A Vulkan detector's rendered corpus with exact corner truth: 4,313 frames, 9,991 tags. Generate
with its `gen_corpus.py` (vkapriltag `956d5029`, AprilTag sources `b7c0ebe9`), then
`python -m benchmarks.vkapriltag_corpus --backends napa napa16 apriltag3-d1 --output work/external/vkapriltag-results/napa.json`.
Optional `vkapriltag-d1/d2` backends need a Vulkan device and report accuracy only. NAPA finds 98.3% at the default vs
vkapriltag's 97.8/98.0% and full-resolution AprilTag's 96.8%, with no false detections from any of them. Corners are a
wash on sharp frames; NAPA is better under blur. [Report](results/2026-11-11_vkapriltag/report.md).

Detector-labelled sets (e.g. the willtoth FRC collection) measure agreement with that detector, not accuracy. Keep them
separate. See the [independent baseline report](results/2026-10-31_external_baseline/report.md).
