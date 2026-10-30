# CPU optimization comparison — 2026-10-29

NAPA with `min_tag_px=16` has lower single-core p50/p95/p99 and lower mean wall time, CPU time and working memory than every tested competitor configuration on these workloads.
It does **not** win every image, every multicore tail statistic, or every corner.
These are synthetic benchmark results on one i7-12700H, not proof of general real-camera accuracy.

## Changes

- Preserve tiny borders on the original stride-one image; suppress noise with a higher local floor and recover low contrast on an averaged stride-two grid.
- Keep overlapping proposal bands for large low-contrast tags; retain narrow tilted and small blurred tags on finer grids.
- Reject implausible nonterminal proposals with a cheap border screen; preserve terminal-crop behavior and final decoding/refinement.
- Pack component metadata by dense component index, preserving raster order and reducing sparse buffer touches.
- Defaults remain `min_tag_px=32`. Use `NapaDetector(min_tag_px=16)` for the high-recall comparison below. Corner refinement remains enabled; error correction is unchanged.

## Scope and timing

- Final latency: 149 cached diagnostic frames, all 20 scenarios and four stress cases at 640×480, 1280×720 and 1920×1080; one warm-up plus three shuffled timed calls per frame/backend.
- The diagnostic set deliberately includes historical losses to Nano/Locus. A second render seed uses the same selection rule (149 frames, seven repetitions); it is corroboration, not an unbiased camera sample.
- Full quality replay: 900 corpus + 180 stress + 540 range frames (18 sizes × 30), 3,486 tags. Maximum placed tags in a frame is 33; the resource adapter’s 64-output limit is nonbinding.
- Resources: identical saved 96-frame mixed sets per resolution; isolated processes, import/construction/warm-up/timed phases separated, one warm-up pass and two timed passes × two shuffled rounds. One allowed P-core (CPU2), then six physical P-cores (0,2,4,6,8,10). NAPA still uses one execution thread per frame.
- Detectors are constructed once. Timing includes required internal processing, input copies and Python result conversion, but excludes rendering, scoring and process startup. NumPy/OpenBLAS is limited to one thread; all other pools and actual affinities are recorded.
- Existing grayscale renderer/scoring and tag36h11 are unchanged. AprilTag/Locus origins are normalized by −0.5px. Geometric corner alignment and canonical ordering remain separate.
- External Chrome load and thermal/frequency drift were left visible in both monitor TSVs. First/closest/worst frames were retimed separately, not discarded.

Configuration legend: Locus d2 is the standard profile with decimation two. OpenCV/RapidTag defaults do not refine corners; SUBPIX variants are separate.
ArUco3-min32 is a historical name: half-scale detection makes its effective minimum about64px; half16 is about32px. Full-scale ArUco3 performs no refinement at its default ratio despite forcing the SUBPIX flag.
Pinned versions: upstream AprilTag3.4.5/b7c0ebe; WPILib2026.2.2; Locus0.10.0; OpenCV Python5.0.0.93; Nano c99989b with OpenCV4.14.0; RapidTag0.1.12/fad141c. Exact build flags/effective settings/actual threads are in each JSON and the linked original build commands.

## Final diagnostic latency

Pooled detector-call percentiles, milliseconds. The 149-frame mixture is deliberately difficult and unbalanced across resolutions; compare the resource tables for an equal scenario mixture.

| Configuration | p50 | p95 | p99 | Tags found / 319 | Wrong IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate-mtp16 | 1.115 | 6.442 | 10.545 | 319 | 0 |
| candidate-mtp32 | 0.419 | 1.938 | 2.577 | 310 | 0 |
| apriltag3-d2 | 19.505 | 57.670 | 68.842 | 311 | 0 |
| apriltag3-d1 | 72.189 | 219.958 | 258.445 | 319 | 0 |
| wpilib-d2 | 14.201 | 42.795 | 51.072 | 259 | 0 |
| locus | 12.701 | 32.652 | 36.665 | 307 | 0 |
| locus-d2 | 5.849 | 12.740 | 14.821 | 305 | 0 |
| opencv | 26.821 | 118.516 | 150.296 | 311 | 0 |
| opencv-subpix | 26.578 | 117.878 | 142.303 | 311 | 0 |
| opencv-aruco3 | 14.964 | 93.649 | 109.496 | 296 | 0 |
| opencv-aruco3-min32 | 4.765 | 17.983 | 21.857 | 244 | 0 |
| opencv-aruco3-half16 | 6.668 | 20.635 | 25.568 | 291 | 0 |
| aruco-nano | 3.144 | 19.145 | 22.132 | 318 | 0 |
| rapidtag | 9.462 | 84.623 | 92.825 | 230 | 8 |
| rapidtag-subpix | 9.547 | 86.672 | 92.618 | 230 | 8 |

Per-frame median time ratios (competitor ÷ NAPA16), on identical images. Values below one are real NAPA losses.

| Configuration | Median ratio | Smallest ratio | NAPA slower frames / 149 |
| --- | ---: | ---: | ---: |
| apriltag3-d2 | 16.10× | 3.157× | 0 |
| apriltag3-d1 | 58.94× | 7.004× | 0 |
| wpilib-d2 | 10.52× | 1.713× | 0 |
| locus | 7.10× | 2.190× | 0 |
| locus-d2 | 3.45× | 0.992× | 1 |
| opencv | 19.78× | 3.048× | 0 |
| opencv-subpix | 19.98× | 3.053× | 0 |
| opencv-aruco3 | 12.47× | 2.669× | 0 |
| opencv-aruco3-min32 | 3.56× | 1.066× | 0 |
| opencv-aruco3-half16 | 4.36× | 1.203× | 0 |
| aruco-nano | 2.71× | 0.741× | 5 |
| rapidtag | 7.63× | 1.583× | 0 |
| rapidtag-subpix | 7.56× | 1.925× | 0 |

The 28-frame, 20-repeat retime finds Nano faster on 5 selected frames and Locus d2 on 1. Smallest repeated ratios are 0.786× and 0.975× respectively. Losses remain visible.

## CPU and working memory

Means of two process rounds. Working memory is peak RSS over the constructed detector, including warm-up and timed detection; image arrays were loaded before the baseline. Memory values average the two process peaks. RSS is approximate and includes allocator retention.
NAPA/closest/WPILib entries use final packed-core runs; unchanged slower backends retain the earlier isolated measurements. Every row records its source JSON in `analysis.json`.

| Cores allowed | Resolution | NAPA16 wall / CPU ms | Closest competitor mean wall | NAPA16 working MiB | Smallest peer working MiB |
| --- | --- | ---: | --- | ---: | --- |
| 1 | 640x480 | 0.479 / 0.479 | aruco-nano 1.075 ms | 0.84 | locus-d2 2.38 |
| 1 | 1280x720 | 1.214 / 1.214 | aruco-nano 2.661 ms | 2.71 | aruco-nano 4.80 |
| 1 | 1920x1080 | 2.732 / 2.731 | aruco-nano 6.922 ms | 6.45 | aruco-nano 7.63 |
| 6 | 640x480 | 0.487 / 0.487 | opencv-aruco3-min32 1.054 ms | 0.83 | aruco-nano 3.18 |
| 6 | 1280x720 | 1.166 / 1.165 | opencv-aruco3-min32 2.054 ms | 2.71 | aruco-nano 4.95 |
| 6 | 1920x1080 | 2.950 / 2.949 | opencv-aruco3-min32 3.775 ms | 6.45 | aruco-nano 7.72 |

At 1080p, mean-of-round multicore p99 is 10.755ms for NAPA16, 10.377ms for Locus d2 and 10.072ms for ArUco3 half-scale/min≈64px. These tuned peers have lower recall; their latency tradeoff stays visible.
NAPA32 has a lower tail but misses more small tags. Frame-parallel throughput was not rerun and is not claimed here.

## Quality

Full replay source: `final_quality.csv`.

| NAPA configuration | Found / 3486 | Wrong IDs | Duplicates |
| --- | ---: | ---: | ---: |
| candidate-mtp16 | 3486 | 0 | 0 |
| candidate-mtp32 | 3334 | 0 | 0 |

For the unchanged peers, the previous same-corpus measurements found AprilTag d1 3,483/3,486, Nano 3,464,
Locus standard 3,419 and Locus d2 3,273. These are reused measurements, not new full peer replays. All 1,080
baseline corpus/stress truth, ID, corner-error and ordering rows match; all 540 range-image hashes match.
The corpus renderer differs from the earlier commit only in the expanded range-size list, for which the separate 18-size replay is used.

Common-tag corner errors below reuse that verified reference scope; geometry is aligned before scoring.

| Peer | Common tags | NAPA median / p95 / max px | Peer median / p95 / max px |
| --- | ---: | ---: | ---: |
| apriltag3-d1 | 3483 | 0.0223 / 0.1180 / 3.5720 | 0.0424 / 0.1926 / 2.3490 |
| apriltag3-d2 | 3339 | 0.0213 / 0.1045 / 3.5720 | 0.0421 / 0.1973 / 2.1516 |
| aruco-nano | 3464 | 0.0221 / 0.1144 / 3.5720 | 0.0429 / 0.2063 / 18.6881 |
| locus | 3419 | 0.0219 / 0.1114 / 2.7777 | 0.0450 / 0.6751 / 8.6687 |
| opencv-subpix | 3452 | 0.0221 / 0.1135 / 0.7006 | 0.2999 / 0.9268 / 28.5027 |

NAPA retains the motion-blur worst-corner weakness against upstream AprilTag. Its median and p95 are better on common tags; that does not erase the worse maximum.

## Real images and limits

The 3 AprilTag test JPEGs have reference regression output, not annotated truth. NAPA16 takes 1.23/2.09/0.92ms median per image and agrees with 44/47 reference detections (48 detections total). AprilTag d1 agrees with 46/47, Nano 31/47. No false-positive/recall claim is made for these images. No suitable annotated camera recording was available.

The benchmark is finite, synthetic, and developed in this repository. A new seed does not make it independent
of the renderer. No public performance table was changed. Measure on actual camera footage before relying on a specific latency or range.

## Adapter overhead

For empty, one-tag and 27-tag 720p examples, additional Python input/result handling over a direct ctypes call is approximately 0.8/4.2/18.9 microseconds at p50. This is not subtracted from reported detector times. The direct ctypes call still has its own call overhead. RapidTag copies each input frame; that required copy remains timed. See adapter_overhead.json for the paired samples.

## Provenance and checks

- Final timed core hash is in `final_latency.json`; source/build hashes and intermediate snapshots are in `source_manifest.json`.
- Earlier diagnostic timings preceded terminal-crop guards and metadata packing. Their original hashes remain intact. Final timing/resource runs use the final core; exact pre/post comparisons cover both seeds and all five strides.
- 292 tests pass with all optional backends installed; Ruff passes. AddressSanitizer/UndefinedBehaviorSanitizer pass 384 native calls over odd/padded shapes, sample strides and three families.
- Exact ordered outputs matched on two seeds and all five strides after component packing. Reproduced small-crop, quiet-zone and tilted-tag regressions have focused tests.
- Reproduction commands, pinned dependency setup and raw CSV/JSON files are linked from `commands.md`. No per-frame executable launches, detector initialization, GPU processing or corner-refinement disabling supply the reported wins.
