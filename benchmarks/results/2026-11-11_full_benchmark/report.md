# Full benchmark: every set, every detector, laptop and Raspberry Pi 5 (2026-11-11)

This run is meant to inform the choice of Napa's default and the recall preset.

**Decision (after this report):** the maintainers chose `min_tag_px=22` as the default with the shadow pass off by
default (opt-in), and kept the recall preset. That differs from this report's first recommendation, which kept the
shadow pass on; see [Is the shadow pass worth keeping on at min 22?](#is-the-shadow-pass-worth-keeping-on-at-min-22). The full tables are in [tables.md](tables.md), the chart is [chart.html](chart.html), and the machine-readable
numbers are in [summary.json](summary.json) and `scores/`.

## Build, machines and method

- **Napa core:** source SHA-256 `9f9fe2881ba4…` (full hash in `environment.json`). It includes the border cells, the polish gate,
  the guarded side margin and `tag_shape`. Every detector ran from the same packed grey PNG frames on both machines.
- **"After fix" rows:** these use core source `c986ca720e3a…`, which is the run's core plus the `payload_bias` all-lines fix,
  applied after the run (see [After the run: the payload_bias fix](#after-the-run-the-payload_bias-fix)).
- **Laptop:** Intel Core i7-12700H, x86-64. Timing runs on one P-core per lane (CPUs 2/4/6/8, `intel_pstate` powersave
  governor). Scoring runs on a separate core.
- **Raspberry Pi 5:** 64-bit, `ondemand` governor at 2.4 GHz, one core per run.
  - Throttle flags read `0x0` at the start of every set. Temperature readings stayed between 51 and 75 °C.
  - Large sets ran at a frame step on the Pi (list in `environment.json`). For example, ICRA forward used every second frame,
    and the 1080p Tango camera every fourth.
- **Timing:** each detector is built once. Each frame gets a first call (not timed for p50) and then a warm call, which is timed.
  All detectors run interleaved frame by frame. "Time vs default" is the geometric mean, across every set, of the ratio of
  p50 times against Napa's default on the same machine.
- **Other detectors, at their defaults or labelled settings:**
  - AprilTag 3 (full resolution, and decimate 2);
  - WPILib AprilTag;
  - Locus 0.10;
  - OpenCV ArUco (plain, subpixel, ArUco3). The laptop has OpenCV 5.0 and the Pi has 4.10. This explains why OpenCV finds
    86% of Basler tags on the laptop but 72% on the Pi;
  - ArUco Nano;
  - RapidTag (plain and subpixel);
  - vkapriltag (laptop GPU only; its rows are for accuracy, not speed).
- **bdcudatag** needs CUDA, so it appears only through its published numbers, labelled as such, in
  `2026-11-11_bdcudatag/`.
- **Scoring:**
  - Sets with exact truth (own synthetic, ICRA ×4, Locus, vkapriltag corpus, stress, range sweep) are scored on matched
    tags and worst-corner error.
  - Drive and Roboflow are scored on matched boxes. bdcudatag sets use bdcudatag's own rules.
  - **Drive is synthetic.** It is the public `syntheticAprilTagRIODetector` archive: rendered frames, 543 distinct tag IDs in
    1,097 validation frames. Locus is rendered too. These are the two sets where the shadow pass adds the most.
  - GoPro and the PennCOSYVIO cameras count sightings of surveyed IDs.
  - The Basler, real FRC and "more FRC" sets have no hand truth. They are scored against a consensus: a tag counts if at
    least three other detector configurations report it in that frame. Other Napa settings are among those voters, which
    widens the gaps between Napa settings. [tables.md](tables.md) also scores the key settings against an independent
    reference: a tag counts only when at least two non-Napa engine families report it in the frame. The order of settings is
    the same, but on Basler the gap between min 32 and min 22 is about 2 points (96.2% → 98.3%), not 7.
  - Backgrounds (`fresh`, `indoor`, Basler backgrounds) count every output as false.
- **Added after the run started (marked \* in the tables):**
  - 1,000 more public FRC frames;
  - three PennCOSYVIO cameras: the 640×480 fisheye Tango, the 1080p colour rolling-shutter Tango, and the 752×480 VI Sensor;
  - the range sweep, the ViSP reference images and the render groups;
  - an extra pass with the shadow pass off at min 22 and min 16. That pass ran in one process with the m32, m22 and m16
    anchors, so its times are relative to those anchors.

### Held-back Basler frames

The 288 held-back Basler positive frames (718 consensus tags) were used here for the first time, as their own row. Their pixel
hashes were checked against the curated list. The earlier settings sweep used the Basler test split, not these frames.

## The Pi speed-against-accuracy view

**Basler test frames: p50 time on one Pi 5 core against consensus recall**

The new default's row was timed directly after the run, in one process with min 32 and min 22 as anchors (min 32 read
1.18 / 1.81 ms there, as in the main run). Its "time vs default" is from the extra pass. Four-core throughput comes from
the same later run (min 32: 2,721 frames/s there). With the current core (`c986ca720e3a`) the new default reads
1.44 / 2.25 ms and 2,380 frames/s, against 1.17 / 1.80 ms and 2,712 frames/s for min 32 in the same runs.

| Setting | Pi Basler p50 / p99 ms | Time vs default (all sets) | Basler test | Held-back | Real FRC | More FRC\* | Drive boxes (synthetic) | Locus (rendered) | Tango fisheye\* | VI Sensor\* | 4-core fps, 1280×800 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Min 32, no shadows | 0.81 / 1.50 | 0.77 | 86.9% | 88.3% | 80.7% | 76.6% | 58.4% | 52.5% | 94 | 303 | 3945 |
| **Default (min 32)** | 1.18 / 1.86 | 1.00 | 89.8% | 91.2% | 81.5% | 77.3% | 76.1% | 66.4% | 96 | 313 | 2653 |
| Min 22 tall | 1.25 / 1.88 | 1.15 | 91.7% | 92.8% | 82.8% | 79.1% | 75.8% | 64.1% | 147 | 363 | 2171 |
| **Min 22, no shadows (new default)** | 1.39 / 2.24 | 1.20 | 96.3% | 97.1% | 88.9% | 88.9% | 64.1% | 81.3% | 242 | 465 | 2343 |
| **Min 22** | 1.77 / 2.67 | 1.43 | 96.8% | 97.2% | 89.5% | 89.2% | 78.5% | 87.1% | 243 | 473 | 1761 |
| Min 16 tall | 2.28 / 3.21 | 1.90 | 98.7% | 99.2% | 95.6% | 92.3% | 81.3% | 89.7% | 369 | 639 | 1335 |
| Min 16 | 4.30 / 5.18 | 2.85 | 98.9% | 99.2% | 97.7% | 98.4% | 82.3% | 98.9% | 424 | 772 | 826 |
| Min 16 tall, shadows 22 | 4.10 / 5.10 | 2.81 | 99.0% | 99.2% | 96.9% | 94.6% | 86.6% | 96.2% | 377 | 743 | 792 |
| Min 16, shadows 22 | 5.94 / 7.03 | 3.58 | 99.3% | 99.2% | 98.2% | 99.1% | 87.2% | 99.4% | 426 | 862 | 592 |
| **Preset (16, split, shadows 24)** | 6.63 / 7.81 | 3.95 | 99.7% | 99.3% | 98.7% | 99.2% | 87.3% | 99.6% | 434 | 867 | 548 |
| Min 16, shadows 16 | 11.76 / 13.03 | 4.96 | 99.3% | 99.2% | 98.5% | 99.1% | 88.8% | 99.6% | 441 | 901 | — |
| AprilTag 3, decimate 2 | 16.72 / 19.01 | 19.9 | 89.1% | 89.0% | 79.7% | 73.9% | 63.8% | 87.8% | 194 | 317 | 226 |
| AprilTag 3, full resolution | 57.47 / 62.64 | 63.9 | 92.0% | 93.6% | 93.6% | 91.9% | 77.6% | 96.8% | 379 | 557 | 64 |
| ArUco Nano | 14.65 / 15.31 | 9.5 | 87.3% | 86.9% | 83.5% | 77.7% | 82.8% | 95.2% | 101 | 349 | 290 |
| Locus | 12.48 / 14.36 | 17.9 | 0.0% | 0.0% | 89.2% | 86.8% | 61.9% | 97.8% | 314 | 570 | — |

On the laptop, one P-core with no other load, Basler test p50 is:

| Setting | Basler p50 | vs default |
| --- | --- | --- |
| Default | 0.554 ms | 1.00 |
| Min 22 tall | 0.536 ms | 0.97 |
| Min 22, no shadows | 0.656 ms | 1.18 |
| Min 22 | 0.820 ms | 1.48 |
| Min 16 tall | 0.995 ms | 1.79 |
| Preset | 3.24 ms | 5.84 |

On Roboflow 640×360 frames, the default runs at 0.161 ms and min 22 at 0.233 ms (1.45×).

Notes on the table:

- **Napa false detections:** no Napa setting reported anything on any background frame. Napa had no wrong IDs on the Pi's
  truth sets. On the laptop it had one on the synthetic set for every setting, and the min 22 tall settings had one more on
  the vkapriltag corpus.
- **Napa's only unmatched outputs are on Locus.** Most are tags at the image edge that Locus's annotation leaves out.
- **Tall and wide on Locus:** in Locus frame 576, tag 4 is a large (334 px) bowed tag. Tall and wide sampling report it with a
  15–20 px worst-corner error. The square settings do not report it at all.
- **Locus reported nothing on any Basler frame,** on both machines. This is a problem with that backend on those frames, not
  a recall figure.
- **vkapriltag d2 crashes on frames with odd widths or heights.** It crashed on 376 frames: 108 in `fresh` and 268 in
  `indoor`. The harness restarts it, capped at three restarts in a row, and those frames count as finding nothing.

## Recommendation

### (a) Default: min_tag_px 22

Change the default from min 32 to min 22, with the shadow pass on as it is today.

**What it costs:** about 1.43× the default's time on the Pi across all sets. On Basler frames that is 1.77 ms instead of
1.18 ms per frame on one core, and 1,761 instead of 2,653 frames/s on four cores. On a laptop P-core it is 1.44–1.48×.

**Accuracy:** it gains or ties on every set we have.

| Set | Default (min 32) | Min 22 |
| --- | --- | --- |
| Basler test | 89.8% | 96.8% |
| Held-back Basler | 91.2% | 97.2% |
| Real FRC | 81.5% | 89.5% |
| More FRC | 77.3% | 89.2% |
| Roboflow boxes | 31.3% | 38.8% |
| Drive boxes (synthetic) | 76.1% | 78.5% |
| Locus (rendered) | 66.4% | 87.1% |
| Range sweep | 91.1% | 99.3% |
| ICRA forward (laptop, full) | 32.8% | 43.3% |
| bdcudatag hard | 96.8% | 98.5% |
| GoPro sightings | 1,835 | 2,589 |
| Tango fisheye sightings | 96 | 243 |
| VI Sensor sightings | 313 | 473 |

Against the independent reference (two or more non-Napa engine families), the gains are smaller on Basler but the same in
direction everywhere:

| Set | Default (min 32) | Min 22 |
| --- | --- | --- |
| Basler test | 96.2% | 98.3% |
| Held-back Basler | 96.5% | 99.4% |
| Real FRC | 85.6% | 92.2% |
| More FRC | 82.8% | 93.1% |
| GoPro | 61.7% | 82.6% |
| Roboflow valid | 74.6% | 87.4% |

Tags that only Napa reports carry IDs from each set's own ID list (on Basler, all among the 13 IDs other detectors see), so
there is no sign of false IDs at any setting.

**Min 22 is not a strict superset of min 32.** It misses 9 and 5 tags that min 32 finds on the two Basler sets (1.4% and 0.8%
of min 32's tags), 17 on more FRC and 194 on GoPro (3.5%). All are confirmed by another detector and are 22–37 px, and about
half are found again at min 16. They look like tags that the min 32 sampling grid lands on and the min 22 grid does not. The
net gain is still large, but this is worth closing if min 22 becomes the default.

**What does not get worse:** min 22 adds no false detections on backgrounds and no wrong IDs. Jitter is about the same: 0.68%
of readings jump more than 0.3 px, against 0.63% for the default. More rendered pose scenes are accepted: 459 of 576 against
444.

**Why the median corner error rises on ICRA forward:** it goes from 0.076 to 0.081 px, and p95 goes from 0.28 to 0.38 px.
This is because min 22 finds smaller tags that the default misses. On the 4,893 tags both settings find, 99% of corners
are within 0.05 px of each other.

**Even after the change, min 22 is still faster than every other CPU detector we measured.** On the Pi it is about 9× faster
than AprilTag 3 at decimate 2 (16.7 ms). It also beats AprilTag 3 at decimate 2 on recall on the Basler, FRC, Drive and Penn
sets, but not on ICRA forward (43.3% against 49.1%) or Locus (87.1% against 87.8%).

**If speed matters more than these gains, keep min 32.**

#### Is there a medium setting cheaper than min 22 that still beats the default everywhere?

No, not among the settings measured.

- **Min 22 with the shadow pass off** is the cheaper medium setting: 1.20× on the Pi and 1.18× on the laptop.
  - It keeps nearly all of min 22's gain on Basler (96.3%), held-back (97.1%), real FRC (88.9%) and more FRC (88.9%).
  - It loses on the synthetic Drive set: 64.1%, against 76.1% for the default and 78.5% for min 22. It also loses on the
    rendered Locus set: 81.3% against 87.1% for min 22.
  - On real footage, the shadow pass at min 22 adds little. Against the independent reference it adds 0.2–0.5 points on
    Basler and FRC, 1.5 points on the Tango 1080p camera and 2.6 points on GoPro.
  - See [Is the shadow pass worth keeping on at min 22?](#is-the-shadow-pass-worth-keeping-on-at-min-22) below.
- **Min 22 tall is not a safe default.** It costs 1.15× and gains only a little on Basler (91.7%) and FRC. It loses to the
  default on:
  - ICRA forward: 28.6% against 32.8%;
  - Locus: 64.1% against 66.4%;
  - bdcudatag hard: 94.8% against 96.8%;
  - own synthetic: 97.1% against 98.0%;
  - stress: 2,191 against 2,215 tags;
  - range sweep: 86.9% against 91.1%.

  It also reports the bowed Locus tag with a 20 px error.
- **Min 16 tall sits between min 22 and min 16** at 1.90× on the Pi.
  - It is close to min 16 on Basler (98.7%), held-back (99.2%) and the low-resolution Penn cameras (369 / 639 sightings).
  - It falls behind min 16 on more FRC (92.3% against 98.4%) and Locus (89.7% against 98.9%).
  - It is an option for upright cameras that see tags at a yaw angle, not a default.

**The low-resolution Penn cameras show the clearest middle ground.** Tango fisheye sightings, from the default through the
preset, are 96 → 147 (min 22 tall) → 243 (min 22) → 369 (min 16 tall) → 424 (min 16) → 434 (preset). VI Sensor sightings are
313 → 363 → 473 → 639 → 772 → 867. Cameras like these gain the most from the smaller settings.

### (b) Recall preset: keep min 16 + split_touching + shadows 24

Keep the current preset (min 16, split_touching, shadows down to 24 px).

**Evidence:** among Napa settings, it is the highest, or within 0.1 points of the highest, on every set except Drive and the
PennCOSYVIO sightings (GoPro and the three added cameras). Min 16 with shadows down to 16 px finds more there, for example
Drive 88.8% against 87.3%, at 1.26× the preset's time. Against the other detectors, only Locus finds more, and only on ICRA.

- **ICRA forward:** 68.3%, against 70.6% for Locus, 49.8% for AprilTag 3 and 53.7% for min 16 with shadows 22.
- **Splitting** is what keeps the touching tags on ICRA's boards.

**Cost:** 3.95× the default on the Pi. That is 6.63 ms per Basler frame on one core, and 548 frames/s on four cores. It is
still faster than every other CPU detector we measured, and it finds more.

**Cheaper options to document beside it, not replacing it:**

- **Min 16, shadows 22, no split.** It costs 3.58× (10% less than the preset).
  - It matches the preset within 0.5 points on Basler, FRC, Drive and Locus, and within 2% of its Penn sightings.
  - It gives up ICRA forward: 53.7% against 68.3%, because it has no split.
  - It suits cameras that never see tags touching.
- **Min 16 tall, shadows 22.** It costs 2.81×.
  - It suits upright cameras that see tags at a yaw angle.
  - It is close on Basler (99.0%) and Drive (86.6%).
  - It loses more FRC (94.6%), Locus (96.2%) and ICRA forward (50.0%).
  - Adding split (3.00×) recovers ICRA forward to 56.5%.

### Is the shadow pass worth keeping on at min 22?

The shadow pass at min 22 costs 0.41 ms per Basler frame on one Pi core: 1.80 ms against 1.39 ms, in the same process.
That is about 1.18× across all sets. What it buys:

- **On the synthetic and rendered sets, a lot:** Drive boxes 64.1% → 78.5%, Locus 81.3% → 87.1%.
- **On real footage, little:**

  | Set | Without the pass | With the pass |
  | --- | --- | --- |
  | Basler test | 97.8% | 98.3% |
  | Held-back Basler | 99.2% | 99.4% |
  | Real FRC | 91.7% | 92.2% |
  | More FRC | 92.6% | 93.1% |
  | Tango 1080p | 86.2% | 87.7% |
  | GoPro | 80.0% | 82.6% |

  These are scored against the independent reference. In absolute terms, the pass adds 4 tags over the 585 Basler
  frames and 14 over 1,188 FRC frames.
- **The real tags it does recover look like the intended case.** At min 32, the pass recovers 21 tags per Basler split,
  402 in 312 GoPro frames and 232 in 193 Tango 1080p frames. Nearly all are confirmed by a non-Napa detector. Crops of
  a sample show white margins in shade that merge into grey walls, and tags backlit against bright windows on the
  PennCOSYVIO indoor recordings.
- **At min 22, most of those are found anyway.** The finer grid separates the border from the background on its own.

Per extra Pi millisecond, moving from min 32 to min 22 adds 2 to 8 times as many tags as the shadow pass does, on
every real set. So the shadow pass is better as an opt-in for dim or backlit venues than as a default cost. The docs
describe it that way, with the evidence's limits.

## One-frame dropouts on consecutive frames (added after the run)

A dropout is a tag seen in frames t−1 and t+1 but missed in frame t. On video it shows up as a flickering detection,
which a robot's pose filter or a tracker has to bridge. Each figure is dropouts out of chances, where a chance is a tag
seen in both neighbours. It is pooled over consecutive segments:

- **GoPro and Tango 1080p:** PennCOSYVIO, 500 frames starting one third and two thirds into each of the four recordings,
  so 4,000 frames per camera. Only surveyed wall IDs count.
- **Basler:** four bursts of tagged frames from the robot recording, 1,276 frames read. The held-back frames are never
  read, and only triples with all three frames read count, which leaves about 170 chances per detector. So the Basler
  column is a small sample.

Read the rate together with the sightings. A detector that sees few tags has few chances and can show a low rate:
WPILib's defaults saw about 1,800 GoPro tags, against 21,600 for the new default.

| Detector | GoPro: sightings, dropouts | Tango 1080p: sightings, dropouts | Basler: sightings, dropouts |
| --- | --- | --- | --- |
| Napa min 32, no shadow pass (0.5.1 speed) | 15,066, 8.5% | 11,545, 9.4% | 608, 3.8% |
| Napa min 32, coarse shadow pass (previous default) | 16,227, 8.2% | 12,503, 9.4% | 629, 3.1% |
| **Napa min 22, no shadow pass (new default)** | 21,595, 5.1% | 16,255, 6.0% | 672, 0.0% |
| Napa min 22, coarse shadow pass | 22,216, 5.1% | 16,777, 6.0% | 675, 0.0% |
| Napa min 22 tall | 18,517, 6.7% | 14,997, 6.6% | 616, 1.3% |
| Napa min 16 | 27,443, 2.1% | 18,621, 4.5% | 688, 1.1% |
| Napa preset | 29,325, 2.4% | 19,301, 3.9% | 698, 1.7% |
| AprilTag 3, full resolution | 24,751, 1.9% | 17,253, 4.4% | 641, 1.2% |
| AprilTag 3, decimate 2 | 15,490, 6.0% | 13,169, 10.6% | 626, 4.3% |
| AprilTag, WPILib defaults | 1,831, 1.6% | 2,249, 3.8% | 73, 0.0% |
| Locus | 24,164, 4.9% | 12,390, 11.6% | 0, — |
| OpenCV ArUco, subpixel | 21,731, 4.7% | 14,280, 5.5% | 507, 4.2% |
| ArUco Nano | 19,375, 4.0% | 13,789, 6.8% | 613, 3.1% |
| RapidTag, subpixel | 10,696, 2.1% | 13,312, 5.7% | 469, 0.0% |

- **Cause.** Most of the flicker is sampling phase, not a fault in any one stage. A coarse search grid lands on a tag
  differently as it moves by a pixel. Cropping a frame by one pixel changes which tags min 32 finds by 3–18% of its tags,
  depending on the set; at min 22 it is 3–9%, and at min 16 it is 1–2%.
- **The new default halves the flicker.** Against min 32 it cuts dropouts from 8.5% to 5.1% on GoPro, from 9.4% to 6.0%
  on Tango 1080p, and from 3.8% to 0 of 171 on Basler. It also finds more tags.
- **The shadow pass does not change dropouts.**
- **Finer settings flicker less.** Min 16 and the preset sit at about AprilTag 3 full-resolution levels (2–4.5%).
- **One known case.** The min 22 test fixture that misses a tag (an xfail in the tests) is the same effect: min 22 finds
  that tag at 12 of 16 one-pixel crops, and min 32 at all 16.

## Since v0.5.1 (published)

The published release is v0.5.1. Its source and the current tree (core `c986ca720e3a`) were both built with the
project's flags and run alternately on one core. Each pair of runs on the same pack was adjacent, with three
repetitions, and the median ratio is reported. The ranges across repetitions were within ±0.05 almost everywhere.

**Time, current over v0.5.1** (geometric mean over 8 sets: Basler test, real FRC, more FRC, Roboflow valid, Drive
valid, GoPro, Tango fisheye, ICRA rotation):

| Current setting against v0.5.1's default (min 32) | Pi 5 | Laptop P-core |
| --- | --- | --- |
| Same settings, min 32 with the coarse shadow pass (the unpublished automatic default) | 1.42× | 1.37× |
| `min_tag_px=32, recover_shadows=False` | 1.08× | 1.02× |
| **New default: `min_tag_px=22`, no shadow pass** | **1.66×** | **1.58×** |

| Same setting in both versions | Pi 5 | Laptop P-core |
| --- | --- | --- |
| Min 22, current with the coarse shadow pass | 1.15× | 1.10× |
| Min 16, current with the coarse shadow pass | 0.85× | 0.86× |

**Per set on the Pi:**

- **New default against v0.5.1's default:**
  - Basler: 1.75×;
  - real FRC: 1.65×;
  - more FRC: 1.69×;
  - Roboflow: 2.20×;
  - Drive: 1.63×;
  - GoPro: 1.92×;
  - Tango fisheye: 1.52×;
  - ICRA rotation: 1.10×.
- **`min_tag_px=32, recover_shadows=False`:**
  - Basler: 1.02×;
  - real FRC and more FRC: 1.07×;
  - Drive: 1.01×;
  - GoPro: 1.14×;
  - Tango fisheye: 0.90×;
  - ICRA rotation: 0.99×;
  - Roboflow: 1.52×.

**Small frames.** Roboflow's 640×360 frames are the one set where the current tree is clearly slower at the same
settings. Per-frame cost is unchanged: empty frames take the same time. The per-tag cost has about doubled, from 0.043
to 0.094 ms per tag on one laptop core. Most of that arrived with the per-tag refinement work, before the 2026-11-08
recovery-speed series.

**Tags reported on the same frames,** v0.5.1 default → current `min_tag_px=32, recover_shadows=False` → new default:

| Set | v0.5.1 default | Current, min 32, no shadows | New default |
| --- | --- | --- | --- |
| Basler test | 611 | 616 | 684 |
| Real FRC | 288 | 314 | 346 |
| More FRC | 758 | 830 | 967 |
| Roboflow valid | 256 | 318 | 387 |
| Drive valid (synthetic) | 1,803 | 1,835 | 2,049 |
| GoPro sightings | 1,360 | 1,706 | 2,513 |
| Tango fisheye | 89 | 141 | 370 |
| ICRA rotation | 9,394 | 9,394 | 9,394 |

These are raw output counts. Scored against truth or consensus, current is at or above v0.5.1 on every set at every
shared setting, with no output on background frames from either version. The one exception is Tango fisheye at min 16
(664 → 636 sightings), below.

**Corners.** Worst-corner p95, default to default: ICRA random 0.620 → 0.065 px, ICRA rotation 0.312 → 0.063 px, ICRA
circle 0.446 → 0.146 px, Locus 1.049 → 0.250 px.

**The Tango fisheye exception at min 16** comes from the finest-level contrast floor, a speed change that requires a
local range of 2.4 × `min_contrast` at the finest sampling level. On this dim, low-contrast fisheye camera it drops some
14–22 px tags:

| Floor | Min 16 | Min 22 | Min 32 |
| --- | --- | --- | --- |
| Floor removed | 687 | 397 | 151 |
| Current floor | 636 | 372 | 143 |

Lowering the floor to 1.6× recovers nearly all of them. The cost on one core is +11% at min 32 on Basler frames and +37%
at min 16, for 1–2 more tags on Basler and real FRC. The floor is unchanged.

**Reversed families under gamma.** The tag families read on the inverted image (tagStandard41h12, tagStandard52h13,
tagCircle21h7, tagCircle49h12, tagCustom48h12) come out too big under a gamma tone curve with blur, by about half the blur
sigma (+0.63 px at sigma 1.3). That is the 0.73 px p95 in the family table. v0.5.1 has the same error with the opposite
sign on the classic families too. The current border correction fixes the classic families only. A sign-aware
correction for the reversed families is a likely fix, not yet tried.

## After the run: the payload_bias fix

After the run, the fix was applied to the tree (core `c986ca720e3a`). It fits all payload lines when estimating the border bias,
instead of only the outer ones. The ICRA sets were then rerun with the fix ("after") and with the run's core ("before"), in the
same session on the same machine.

**Pi 5, ICRA subsets plus Basler test:**

- **Found tags:** identical for every setting and set.
- **Basler outputs:** identical.
- **Worst-corner p95 on ICRA random and rotation:** about 0.12–0.25 px before, about 0.064 px after, for every setting.
- **ICRA circle p95:** 0.22–0.33 → 0.15–0.30 px.
- **ICRA forward:** unchanged (0.39 → 0.39 px at min 22). On tall 22 it improves from 0.165 to 0.117 px.
- **Median error:** improves by 0.001–0.009 px everywhere.
- **Time on Basler:** unchanged (−2% to +3%, within noise).
- **Time on ICRA frames** (large, dense tag boards): 0% to +8%.

**Laptop, all four ICRA sets in full, all 21 Napa settings:**

- **Found tags:** identical everywhere.
- **Worst-corner p95:**
  - ICRA random: 0.11–0.13 → 0.067 px;
  - ICRA rotation: 0.25 → 0.073 px;
  - ICRA circle: 0.22–0.31 → 0.15–0.28 px;
  - ICRA forward at the default: 0.279 → 0.264 px.
- **Worst-corner maximum:** ICRA random 1.38 → 0.35 px; ICRA circle at min 16 3.29 → 1.78 px.
- **One small worsening:** the ICRA forward maximum at min 16 and the preset goes from 0.96 to 1.06 px.
- **Time:** in the sequential rows, the fixed core looked 8–20% slower on ICRA. Those rows ran one after the other on a
  machine doing other work. Three alternating reruns on one quiet P-core showed no consistent difference on ICRA forward or
  Basler. At the default, ICRA forward took 0.82–0.95 ms with either core, and Basler 0.58–0.70 ms. Any cost is under that
  ±10% noise. The Pi's 0% to +8% on ICRA is the better estimate.

**Other checks on the fixed core:**

- 240 rendered tags in each of nine families. No family changed by more than 0.035 px on any tag, and found counts are
  identical.
- AddressSanitizer is clean over 1,427 calls.
- 719 unit tests pass.

## Other measurements

These are in [tables.md](tables.md):

- recall by tag-size band;
- corner error by set;
- jitter and rendered-pose results;
- the other tag families;
- the ViSP reference images. At the default, the large-cell families (circle, standard, custom) find only 1–3 of 5 tags. Min
  22 finds 4–5 of 5;
- the render groups (motion and isotropic blur under three tone curves, plus composites);
- the Pi four-core throughput.

## Loose end: rolling-shutter correction

Before the run, VisionProcessor's rolling-shutter correction was changed to use each corner's own velocity instead of the
tag-mean velocity. This was done because simulation showed lower corrected-corner error in 14 of 14 cases. For example, an
off-centre 100 px tag went from 0.44 to 0.16 px mean error.

The docs table in `docs/poses.md` was rerun with the new correction. One sentence there was measured with tag-mean velocity and
has not been rerun: a readout 30% off removes about two thirds of the error, and the wrong direction doubles it. This does not
affect any detector number in this report.

## Files

- `tables.md`: every table, both machines.
- `chart.html`: Pi Basler time against consensus recall.
- `summary.json`: the numbers behind the tables.
- `scores/`: per-set scores for each machine and pass.
- `environment.json`: core hashes, detector versions and builds, Pi frame steps, and health.
