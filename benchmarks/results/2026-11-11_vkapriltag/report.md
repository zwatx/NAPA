# vkapriltag's corpus

[vkapriltag](https://github.com/yojobama/vkapriltag) is a Vulkan AprilTag detector. The image stages run on the GPU, and
quad fitting and decoding run on the CPU through libapriltag. Its repository includes a corpus generator with exact corner
truth, and it compares itself with NAPA 0.5.1 in its `PERFORMANCE.md`. This report runs that corpus, regenerated from commit
`956d502` of its repository, through Napa, vkapriltag and the other CPU detectors in this repository. Scoring follows its
own rules. The runner is `benchmarks/vkapriltag_corpus.py`, and `results.json` holds the numbers below.

**Accuracy is from our runs; timing is not.** vkapriltag ran here, from its released v1.10.1 package, on a laptop's
integrated GPU (Intel Iris Xe). Its times there say nothing about the GPUs it targets, so the only times quoted below are the
ones it publishes.

## Method

- Corpus: `gen_corpus.py` at commit `956d502`: 26 scenarios, 4,313 frames at 640×480, 1280×720 and 1920×1080. 9,991 are
  tag36h11 tags. A further 385 tag16h5 tags come with decoys. Three scenarios (empty, clutter-only, decoys) contain no
  tags. Each pixel averages a 4×4 grid of samples, so corner truth is exact, with no offset: Napa's mean signed corner
  error on it is 0.000 px.
- A tag is found when a detection has its ID and its worst corner is within a quarter of the tag's side. Each detection
  matches at most one tag. Every other detection is false. Corner error is each found tag's worst corner, in the corpus's
  `[i, i + 1)` pixels.
- Napa: core source `87788b0fadcf`, `max_detections=1024`. The detector was given each frame as stored, in grey.
- vkapriltag: v1.10.1 package, configured as its validation tool is: `refine_edges` on and its other defaults, at
  decimation 1 and decimation 2 (its default).
- The other detectors use the adapters in `cpu_backends.py`. They detect tag36h11 only, so they skip the tag16h5 set.

## Tags found

| Detector | tag36h11 found of 9,991 | False | tag16h5 found of 385 | False |
| --- | ---: | ---: | ---: | ---: |
| Napa `min_tag_px=16`, split, shadows | 9,839 (98.5%) | 0 | 385 | 0 |
| Napa `min_tag_px=16` | 9,835 (98.4%) | 0 | 385 | 0 |
| Napa default | 9,813 (98.2%) | 0 | 385 | 0 |
| vkapriltag, decimation 2 | 9,789 (98.0%) | 0 | 385 | 3 |
| vkapriltag, decimation 1 | 9,772 (97.8%) | 0 | 385 | 0 |
| AprilTag 3, full resolution | 9,676 (96.8%) | 0 | not run | |
| ArUco Nano | 9,545 (95.5%) | 0 | not run | |
| AprilTag 3, decimate 2 | 9,533 (95.4%) | 2 | not run | |
| Locus | 9,355 (93.6%) | 0 | not run | |

The detectors differ on a few scenarios:

- **border-occlusion** (230 tags, each crossed at a corner by a bar, like a cable, that covers part of two border sides): Napa default 132, `min_tag_px=16` 115,
  vkapriltag 54 at decimation 1 and 99 at decimation 2. Napa at `min_tag_px=16` finds 17 fewer here than at its default.
- **range** (375 tags, one per frame, 12 to 96 px): Napa default 342, because it looks for tags of 32 px and up. Napa at
  `min_tag_px=16` and vkapriltag at decimation 1 find all 375; vkapriltag at decimation 2 finds 351.
- **obstruction** (281 tags with up to 12% of the payload covered at the centre, border or corner): Napa 240 at every setting, vkapriltag 242.
- **grey-margin** (315 tags): Napa 314 at its default and 315 at `min_tag_px=16`; vkapriltag 312 and 309.

## Worst-corner error

Median / 95th percentile in pixels, for the tags each detector found:

| Scenario | Tags | Napa default | vkapriltag d1 | vkapriltag d2 |
| --- | ---: | ---: | ---: | ---: |
| noise-free | 299 | 0.013 / 0.048 | 0.009 / 0.040 | 0.009 / 0.043 |
| one-tag | 150 | 0.013 / 0.034 | 0.010 / 0.044 | 0.010 / 0.038 |
| large-tags | 101 | 0.012 / 0.055 | 0.010 / 0.042 | 0.009 / 0.043 |
| small-tags | 380 | 0.025 / 0.073 | 0.020 / 0.075 | 0.021 / 0.080 |
| range | 375 | 0.021 / 0.087 | 0.017 / 0.130 | 0.016 / 0.101 |
| many-tags | 216 | 0.013 / 0.048 | 0.010 / 0.043 | 0.009 / 0.042 |
| dense-tags | 4,213 | 0.024 / 0.071 | 0.018 / 0.074 | 0.018 / 0.073 |
| steep-angle | 302 | 0.018 / 0.049 | 0.012 / 0.051 | 0.014 / 0.055 |
| frame-edge | 150 | 0.013 / 0.040 | 0.010 / 0.050 | 0.010 / 0.049 |
| busy-background | 290 | 0.014 / 0.049 | 0.010 / 0.054 | 0.010 / 0.054 |
| dark-background | 297 | 0.014 / 0.055 | 0.010 / 0.048 | 0.010 / 0.051 |
| uneven-light | 309 | 0.016 / 0.051 | 0.011 / 0.052 | 0.012 / 0.051 |
| low-contrast | 291 | 0.023 / 0.062 | 0.017 / 0.053 | 0.018 / 0.051 |
| grey-margin | 315 | 0.018 / 0.049 | 0.013 / 0.047 | 0.014 / 0.042 |
| underexposed | 292 | 0.088 / 0.182 | 0.060 / 0.139 | 0.065 / 0.143 |
| overexposed | 295 | 0.069 / 0.275 | 0.067 / 0.294 | 0.069 / 0.294 |
| noise | 304 | 0.043 / 0.098 | 0.031 / 0.069 | 0.034 / 0.078 |
| jpeg | 290 | 0.036 / 0.082 | 0.030 / 0.067 | 0.031 / 0.069 |
| blur | 304 | 0.020 / 0.056 | 0.039 / 0.137 | 0.044 / 0.160 |
| motion-blur | 307 | 0.018 / 0.083 | 0.066 / 0.431 | 0.059 / 0.452 |
| obstruction | 281 | 0.011 / 0.050 | 0.010 / 0.039 | 0.009 / 0.040 |
| border-occlusion | 230 | 0.013 / 0.051 | 0.453 / 1.640 | 0.204 / 1.090 |
| tag16h5 set | 385 | 0.015 / 0.045 | 0.011 / 0.050 | 0.011 / 0.053 |

Napa's other settings have medians within 0.002 px of its default in every row. Their 95th percentile differs most on
range (0.112 against 0.087 px), where they find 33 more small tags. The other detectors are in `results.json`.

vkapriltag's median is lower than Napa's or equal in every scenario except blur, motion blur and border-occlusion. The
difference is mostly under 0.01 px, and largest on underexposed (0.028 px) and noisy frames (0.012 px). Its corner
refinement is adapted from Napa's, with edge profiles that run along almost the whole side. Napa's errors are about half
of vkapriltag's under blur and a quarter to a fifth under motion blur. On tags crossed by a bar, its 95th percentile is
0.051 px against 1.1 to 1.6 px. In the other scenarios the two 95th percentiles are within 0.05 px of each other.
Before core `87788b0fadcf`, Napa's 95th percentile on border-occlusion was 1.24 px; that build measures the border on the
cells that agree with each other, so a cut or covered border no longer pulls the corners.

## Other sets, measured separately

These figures come from a separate harness, which is not in this directory. It runs Napa with 2,048 detections and Napa
core `ca98a38e84cc`, and counts a detection as wrong when its ID is not on the board or its corners are more than 8 px
off. vkapriltag figures are given at decimation 1 / 2.

| Set | vkapriltag | Napa |
| --- | --- | --- |
| [ICRA 2020](https://huggingface.co/datasets/NoeFontana/apriltag-validation-data) circle, 27,862 tags | found 84.8% / 81.6%, wrong 0 / 0, worst-corner median 0.100 / 0.097 px | found 79.3% default, 85.5% `min_tag_px=16` |
| ICRA 2020 random | found 100% / 100%, wrong 80 / 72, median 0.320 / 0.321 px | wrong 0, median 0.023 px |
| ICRA 2020 rotation | found 100% / 100%, wrong 1 / 0, median 0.256 / 0.256 px | median 0.033 px |
| [PennCOSYVIO](https://daniilidis-group.github.io/penncosyvio/) GoPro C2, every tenth frame, 1,560 frames | 9,041 / 6,249 sightings, 2 / 0 with IDs not on the surveyed wall | 5,488 default, 9,712 `min_tag_px=16`, 10,362 split and shadows; 0 off the wall |

The ICRA artwork has cells of irregular width, so a correction that assumes a one-cell border misfires on it. vkapriltag's
0.26 and 0.32 px on rotation and random are the same as Napa's before core `ed1a48ab1f33`, which added a check against the
payload's own edges. Its own corpus does not show this, because its tags have true one-cell borders.

## Published timing

These are vkapriltag's figures, from sections 8ar to 8at of
[its `PERFORMANCE.md`](https://github.com/yojobama/vkapriltag/blob/956d502943306b1d408ab90104ba0a64f23dfe19/apriltags_vulkan/PERFORMANCE.md),
on an earlier version of its corpus (4,003 frames). They compare it with NAPA 0.5.1, so they predate every Napa change in
this report. We have not reproduced them.

| Machine, all frames, median / 95th percentile (ms) | NAPA 0.5.1 | vkapriltag d1 | vkapriltag d2 |
| --- | ---: | ---: | ---: |
| Desktop with an RX 9060 XT; NAPA built with gcc -O3, one thread (8at) | 0.090 / 0.370 | 0.393 / 0.731 | 0.328 / 0.596 |
| Orange Pi 5 Plus, Mali GPU (8ar, with 8as's accuracy defaults for vkapriltag) | 0.174 / 0.422 | 1.169 / 2.480 | 0.795 / 1.568 |

There, NAPA at `min_tag_px=16` took 0.432 / 1.424 ms on the Pi. The project reports that a chain of dependent GPU dispatches
puts a floor of about 0.3 ms under each vkapriltag frame on the desktop and about 0.75 ms on the Mali. It also reports that
vkapriltag's time stays flat on busy, noisy and dense frames, where NAPA's grows; on dense-tags it was ahead (0.535 against
0.639 ms).
