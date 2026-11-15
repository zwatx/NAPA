# bdcudatag's benchmark sets

[bdcudatag](https://github.com/cobouch/bd-cudatag) is a CUDA AprilTag detector for tag36h11 and tag16h5. This report runs its
two benchmark sets, regenerated from commit `4ff7a31` of its repository, through the CPU detectors in this repository, scored by
its own rules. The runner is `benchmarks/bdcudatag.py`; `results.json` holds the numbers below.

That project describes itself as a team-internal tool and says its hard corpus "is not a cross-detector leaderboard". This is a
comparison on its data, not a ranking of the two projects.

**bdcudatag's own numbers are the ones its README publishes** (Ryzen 9 7900X, RTX 3060, Windows, NAPA 0.5.1 as its reference). It
could not be run here: this machine has no CUDA. Its recall figures can be set beside ours because the sets and rules are the
same. Its timings can't: it runs on a GPU, on another computer.

## Method

- A tag is found when a detection has its ID and a centre within 0.75 tag sides. A wrong ID is a detection whose ID isn't in the frame.
- Scenes: 32 deterministic scenes at each of four resolutions, eight tag36h11 tags each, with two random-code decoys on a quarter of them. Timed over 200 frames after 20 warm-up frames.
- Hard corpus: `gen_synthetic.py --frames 30`, seed 20261103: 780 frames, 2,389 tags, 13 scenarios at 640×480 and 1280×720. The regenerated set has the frame and tag counts its README gives.
- Our detectors: core `ca98a38e84cc`, i7-12700H, one thread. Napa timings were taken alone on one performance core.

## Hard corpus: tags found (wrong IDs)

| Detector | Tags found of 2,389 | Wrong IDs |
| --- | ---: | ---: |
| Napa `min_tag_px=16`, split, shadows | 2364 (99.0%) | 0 |
| Napa `min_tag_px=16` | 2363 (98.9%) | 0 |
| Napa default | 2318 (97.0%) | 0 |
| AprilTag 3, full resolution | 2317 (97.0%) | 0 |
| ArUco Nano | 2298 (96.2%) | 0 |
| bdcudatag, published | 2293 (96.0%) | 0 |
| OpenCV ArUco, subpixel | 2249 (94.1%) | 0 |
| AprilTag 3, decimate 2 | 2196 (91.9%) | 0 |
| Locus | 2036 (85.2%) | 1 |
| RapidTag, subpixel | 1586 (66.4%) | 0 |
| AprilTag, WPILib's defaults | 1058 (44.3%) | 0 |

Found, per scenario, at 640×480 / 1280×720 (percent):

| Scenario | bdcudatag, published | Napa default | Napa `min_tag_px=16` | AprilTag 3, full resolution |
| --- | ---: | ---: | ---: | ---: |
| tiny | 100.0 / 100.0 | 65.9 / 100.0 | 100.0 / 100.0 | 95.6 / 100.0 |
| lowcontrast | 45.0 / 68.6 | 96.7 / 100.0 | 100.0 / 100.0 | 61.7 / 94.1 |
| dark-noisy | 98.2 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| noise | 100.0 / 97.7 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 97.7 |
| underexposed | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| blur | 80.7 / 100.0 | 80.7 / 100.0 | 86.0 / 100.0 | 77.2 / 100.0 |
| motion | 68.9 / 96.5 | 73.8 / 98.2 | 73.8 / 100.0 | 67.2 / 96.5 |
| jpeg | 100.0 / 98.5 | 100.0 / 98.5 | 100.0 / 100.0 | 100.0 / 98.5 |
| shadow | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| glare | 98.1 / 92.7 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| edge | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| dense | 99.8 / 99.8 | 99.8 / 100.0 | 100.0 / 100.0 | 99.8 / 100.0 |
| mixed | 93.5 / 100.0 | 91.3 / 100.0 | 97.8 / 100.0 | 95.7 / 100.0 |

At its default, Napa looks for tags of 32 pixels and up, so it misses part of the two 640×480 scenarios with smaller tags: tiny (65.9% against 100%) and mixed (91.3% against 93.5%). With `min_tag_px=16` it matches or beats the published figure in every scenario at both resolutions. What it still misses there is 26 tags, all at 640×480, in the blur, motion and mixed scenarios.

## Scenes

Tags found, of 1,024: Napa default 1,023 (one 21-pixel tag), Napa `min_tag_px=16` 1,024. bdcudatag published 1,024. No detector here reported a wrong ID.

| Resolution | Napa default, median / p95 ms | Napa `min_tag_px=16`, median / p95 ms | bdcudatag, published, median / p95 ms |
| --- | ---: | ---: | ---: |
| 640×480 | 0.24 / 0.27 | 0.36 / 0.45 | 0.32 / 0.38 |
| 1280×720 | 0.37 / 0.42 | 0.65 / 0.75 | 0.42 / 0.44 |
| 1280×800 | 0.38 / 0.45 | 0.63 / 0.72 | 0.49 / 0.60 |
| 1920×1080 | 0.50 / 0.59 | 0.99 / 1.15 | 0.70 / 0.75 |

The Napa column is one laptop CPU core. The bdcudatag column is that project's published figures on an RTX 3060 and a different computer, so this table isn't a like-for-like speed comparison.

## Corner error isn't reported here

On both sets the truth, more than the detectors, sets the corner error. On the hard corpus the truth is offset by about 0.12 px in x and y: Napa and AprilTag 3 show the same shift, and every accurate detector reads about 0.27 px. With each tag's own shift and size removed, Napa is at 0.008–0.03 px. On the scenes, Napa and AprilTag 3 agree to 0.04–0.05 px but sit 0.10–0.23 px from the truth, growing with resolution, on the same floor as bdcudatag's published 0.13–0.23 px. Neither set can rank detectors by corner accuracy.
