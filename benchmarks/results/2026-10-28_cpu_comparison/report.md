# NAPA vs CPU tag36h11 detectors (2026-10-28/29)

Question: does NAPA give better CPU latency and resource use at comparable detection quality? CPU only; no GPU
implementations. Public performance claims are unchanged by this report. Revised after an independent read-only
review (GPT-6 Astra); the corrections are listed at the end.

## Setup

- Machine: an Intel Core i7-12700H laptop, Ubuntu 26.04, kernel 7.0.0-38, 14 GB. Verified topology: P-cores =
  CPUs 0-11 (6 cores, SMT pairs (0,1)…(10,11)), E-cores = CPUs 12-19. Governor `powersave`, EPP
  `balance_performance`, turbo on. Single-thread runs pinned to CPU 2 (P-core 1; its SMT sibling CPU 3 stayed
  0-1% busy). Multicore runs on CPUs 0,2,4,6,8,10 (one logical CPU per P-core). During the main run CPU 2
  averaged 3.89 GHz (first quarter) and 4.06 GHz (last quarter) at a 91 °C package (max 96 °C), below the
  4.6 GHz single-core turbo. An unrelated background process periodically raised load to ~8; it was not
  stopped. Monitors: `monitor_*.tsv`.
- Timed region: the detector call, including each adapter's result conversion and any copy the library's API
  makes (RapidTag copies every frame). Detectors are built once and warmed up; rendering, scoring and process
  start are excluded. Process CPU time is read around each call; GC is off inside timed loops.

| Backend | Version / build | Effective settings |
| --- | --- | --- |
| NAPA | f47448b, core rebuilt from source with setup.py's flags (`commands.md`), loaded via `NAPA_CORE_LIBRARY`; `/proc/self/maps` confirmed `work/build/napa/libnapa_core.so` (sha256 a6b2c425…) | `min_tag_px` 32 (stride 3), max Hamming 3, min contrast 20, edge-fit corners always, no decimation, 1 thread (no intra-frame threading). Sweep: 16/22/43/64 → stride 1/2/4/6 |
| AprilTag 3 | AprilRobotics/apriltag b7c0ebe (3.4.5), CMake Release `-O3`, ctypes shim (zero-copy) | decimate 2 (default) and 1; sigma 0, refine_edges on, sharpening 0.25, 2 bits corrected, min_white_black_diff 5; nthreads 1 or 6. Corners −0.5 px (source: pixel centres at +0.5) |
| Locus | locus-tag 0.10.0 PyPI wheel (plain `--release`; source tag v0.10.0) | `standard` profile (Erf + corner_subpix, min contrast 20, Hamming 2), decimation 1; speed config decimation 2; scoped Rayon pool `threads`. Corners −0.5 px (documented) |
| OpenCV ArUco | opencv-python-headless 5.0.0.93, DICT_APRILTAG_36h11 | default (no refinement, errorCorrectionRate 0.6 → 3 bits, min perimeter 0.03·max side); SUBPIX; `useAruco3Detection` defaults (min side 32 px; **no refinement actually runs**: the forced SUBPIX loops over zero pyramid levels at full-scale detection, `aruco_detector.cpp` 5.0.0 `findCornerInPyrImage`); two half-scale ArUco3 speed configs: `opencv-aruco3-min32` (historical name: canonical 32 px, **effective minimum ≈64 px**) and `opencv-aruco3-half16` (canonical 16 px, effective minimum ≈32 px). `cv2.setNumThreads` 1 or 6 |
| ArUco Nano | rmsalinas/aruco_nano c99989b (v10), g++ `-O3` against OpenCV 4.14.0 (minimal Release build, IPP-ICV, pthreads), ctypes shim (zero-copy) | defaults, 36h11 only: box filter 15, thres 3, minSize 10, errorCorrectionRate 0, line-fit corner refinement always; `cv::setNumThreads` 1 or 6 (only refinement is parallel) |
| RapidTag | rapidtag 0.1.12 PyPI wheel (v0.1.12 = fad141c, portable) | defaults (no refinement, error correction 0.6, min side 32 px always applied) and SUBPIX. Copies the frame on every call. Global Rayon pool; **auto-pins workers to CPUs 0-11 unless `RAPIDTAG_CORES` is set**, which every run set to the permitted CPUs |

Thread control: `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `RAPIDTAG_CORES` and OpenCV thread counts set per run;
`OPENBLAS_NUM_THREADS=1` in each worker's environment before NumPy is imported (NumPy otherwise starts one
OpenBLAS thread per allowed CPU at import). Every OS thread's name and affinity is recorded. Single-thread runs:
all threads on CPU 2, CPU/wall = 1.00 for every backend; the single-CPU affinity also kept OpenBLAS at one thread
in the interleaved runs.

## Adapter validation (`validate.json`)

Known geometry (11 clean 120 px tags: quarter turns, odd angles, tilts up to 35°, a numpy-rotated frame) and
4 blank frames. After the documented offsets every refined detector's mean corner offset is < 0.03 px (a wrong
convention would show ≈0.5 px); corner order is consistent per backend (AprilTag's native order is a fixed
reflection/rotation of OpenCV's; the others match OpenCV's). No blank-frame detections. RapidTag found 7/11: it
misses noise-free, near-axis-aligned tags ≥120 px (found once σ 1.5 noise is added). Corner error is scored
after best-order alignment (geometry); order is reported separately — every backend was 100% consistent.

## Single thread, one P-core (`single_thread_latency.*`, `tables_single_thread.md`)

Corpus: the existing 20 scenarios × 15 frames × {640×480, 1280×720, 1920×1080} (seed 2468) + 4 stress cases ×
15 × 3 resolutions + the range sweep. All 16 configurations interleaved on every frame in a shuffled
order, 1 warm-up + 7 timed calls. Per-frame ratio = backend median ÷ NAPA median on the same frame.

The range sweep in this run used the renderer's earlier 11 sizes (12-96 px); after the renderer moved to 18 sizes
it was rerun on its own (`range_sweep_latency.*`, `run_range_sweep.sh`, 18 sizes × 30 tags, all configurations
interleaved). Range figures below come from that rerun.

Corpus, 1280×720 (300 frames, 543 tags):

| Backend | p50 / p99 ms | Time ÷ NAPA: median (min) | Recall | False IDs | Corner median / p95 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| NAPA | 0.23 / 1.08 | — | 542/543 | 0 | 0.019 / 0.081 |
| NAPA min_tag_px 16 | 0.62 / 4.45 | 2.87× (1.66) | 543/543 | 0 | 0.019 / 0.083 |
| ArUco Nano | 1.85 / 7.79 | 8.0× (2.27) | 543/543 | 0 | 0.044 / 0.305 |
| OpenCV ArUco3 half scale, min ≈64 px | 2.87 / 13.1 | 11.5× (3.14) | 468/543 | 0 | 0.285 / 0.698 |
| Locus decimation 2 | 4.16 / 8.17 | 17.2× (4.39) | 532/543 | 0 | 0.044 / 0.263 |
| RapidTag | 4.36 / 27.0 | 19.2× (5.96) | 409/543 | 5 | 0.988 / 1.758 |
| OpenCV ArUco3 (defaults) | 8.10 / 40.7 | 31.1× (7.67) | 527/543 | 0 | 0.973 / 1.591 |
| Locus standard | 8.78 / 17.4 | 33.6× (8.71) | 528/543 | 0 | 0.044 / 0.267 |
| OpenCV default | 13.5 / 159 | 51.1× (10.2) | 541/543 | 0 | 0.974 / 1.535 |
| OpenCV SUBPIX | 13.5 / 159 | 51.1× (10.1) | 541/543 | 0 | 0.284 / 0.718 |
| AprilTag dec 2 | 16.2 / 28.6 | 57× (8.94) | 543/543 | 0 | 0.043 / 0.278 |
| AprilTag dec 1 | 63.6 / 108 | 236× (23.1) | 543/543 | 0 | 0.043 / 0.228 |

Supplementary run (`aruco3_half_scale_latency.*`): the corrected ArUco3 half-scale config with a ≈32 px minimum
(`half16`) is 13.5× / 16.4× / 18.8× NAPA's median time at 640/720/1080 (minimum 3.4-4.3×), recall 494/523,
518/543, 508/526 — slower than the ≈64 px config and still unable to find tags under 36 px (range sweep 251/540).

640×480 and 1920×1080 follow the same order (NAPA p50 0.16 and 0.33 ms; ArUco Nano 5.3× and 9.8× median, 2.1×
and 2.6× minimum). NAPA's default had the lower median time than every non-NAPA configuration on all 900 corpus
frames, all 180 stress frames and all 540 range-sweep frames. Re-timing the 17 closest-call and slowest frames alone
on the same images (render seed 2468, 30 repetitions, `outlier_retime.*`) kept NAPA ahead on every frame; most
per-frame ratios moved by less than 10% (169 of 187 comparisons; the full range was −12% to +30%). The closest
call on any frame is 1.38-1.43× across the three runs that timed it (ArUco3 half scale ≈64 px on
`stress-dense-640x480-005`, where it found 0/23 tags and NAPA 23/23); ArUco Nano's closest frame in the main run
was 2.1×.

Recall: comparable overall but not uniformly better. On the corpus NAPA found 1,588/1,592 tags; ArUco Nano
1,589, AprilTag dec 1 1,590, AprilTag dec 2 1,587, Locus 1,551. NAPA's four corpus misses were found by others:
two large low-contrast tags at 1080p (372 and 381 px, `low-contrast-1920x1080-003/-012`, found by ArUco Nano,
AprilTag and Locus; NAPA at min_tag_px 16 misses them too), a 49 px tag under 11 px motion blur and a 65 px
steeply tilted tag (both found at min_tag_px 16).

Corner error on tags both found (all resolutions, ~3,100 tags): median NAPA 0.021 px vs AprilTag 0.040-0.042,
ArUco Nano 0.041, Locus 0.043, OpenCV SUBPIX 0.295, RapidTag SUBPIX 0.291, unrefined OpenCV/RapidTag ~0.95. p95
NAPA 0.09-0.10 vs AprilTag 0.14-0.17, ArUco Nano 0.17, Locus 0.29-0.33, OpenCV SUBPIX 0.82, RapidTag SUBPIX
0.78, unrefined 1.41-1.49. **Worst case: NAPA 3.39 px vs AprilTag dec 1 2.16 px / dec 2 1.55 px** (NAPA's known
motion-blur weakness); ArUco Nano, Locus, OpenCV and RapidTag have worse maxima (8.7-28.6 px).

### Stress and size (where the default loses)

| Case (1280×720 unless noted) | NAPA default | NAPA mtp16 | ArUco Nano | Locus | Locus d2 | AprilTag d1 | AprilTag d2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny tags 12-31 px: recall | 17/35 | 35/35 | 33/35 | 35/35 | 18/35 | 35/35 | 25/35 |
| tiny: p50 ms | 0.38 | 2.43 | 4.39 | 14.6 | 5.06 | 35.2 | 10.7 |
| heavy noise σ 15-30: p50 ms (recall 31/31 except Locus 27) | 0.49 | 5.28 | 7.94 | 11.3 | 4.41 | 63.5 | 18.0 |
| dense (411 tags): p50 ms (recall 411 except Locus d2 410) | 1.04 | 2.47 | 3.86 | 15.9 | 11.7 | 50.3 | 14.1 |
| noisy clutter, no tags: p50 ms (0 false IDs except RapidTag 4-5) | 0.70 | 4.12 | 6.13 | 10.1 | 4.63 | 76.3 | 18.1 |
| range sweep 640×480: found at 12/14/16/18/20/22 px (of 30) | 0/0/7/21/29/30 | all 30 | 21/27/30/30/30/30 | 26/29/28/30/30/30 | 0/0/8/9/16/23 | 29/30/30/30/30/30 | 4/6/8/13/17/25 |
| range sweep, all 18 sizes (of 540) | 447 | 540 | 528 | 533 | 414 | 539 | 433 |

- NAPA's default (min_tag_px 32) does not see tags much under 32 px: by design, but a real recall loss.
- Recovering them costs speed. At min_tag_px 16 (stride 1, so no noise-averaging re-rake) NAPA is 2-6× slower
  than its default, ~10× slower on heavy noise. Its recall is then at least that of the others on tiny tags and
  the range sweep, and it still wins the median (ArUco Nano 2.5× corpus / 1.8× stress; Locus 9-15× / 4-5×;
  AprilTag dec 1 42-127×), but no longer every frame: Locus d2 was faster on 15/60 stress frames at 1080p (mostly
  heavy noise, where Locus d2 keeps recall), ArUco Nano on 4/60, AprilTag dec 2 on 1/60, and Locus d2 / ArUco
  Nano on 3-9 corpus frames per resolution. It still misses the two large low-contrast tags above, and at 1080p
  its corpus recall (521/526) is below its default's (524/526).
- Larger min_tag_px trades recall for speed: 43 → 0.79-0.88× time, 504/523 at 640×480; 64 → 0.64-0.69×,
  454/523.

## Isolated processes: CPU and memory (`resources_1thread_*`, `tables_resources.md`)

Each backend alone in its own pinned process, 96 frames per resolution (4 per corpus scenario + 4 per stress
case), 3 passes × 3 rounds in shuffled order. Memory phases are separated: import + probe, construction, then the
peak (VmHWM reset at the constructed state, or the largest sampled RSS when that reads higher) over warm-up and
the timed passes. RSS figures are approximate (page granularity, allocator retention). Mean wall ms per frame
(CPU ms is identical: 1.00 busy CPUs):

| Backend | 640×480 | 1280×720 | 1920×1080 | Import RSS | Peak over constructed, 720p / 1080p |
| --- | ---: | ---: | ---: | ---: | ---: |
| NAPA | 0.22 | 0.34 | 0.77 | 0.6-0.8 MiB | 0.5 / 1.1 MiB (0.2 at 640×480) |
| NAPA mtp16 | 0.59 | 1.45 | 3.52 | 0.6-0.7 MiB | 5.9 / 14.9 MiB |
| ArUco Nano | 1.12 | 2.67 | 5.83 | 5.1 MiB | 5.6 / 8.5 MiB |
| OpenCV ArUco3 half scale, ≈64 px | 1.54 | 3.94 | 6.86 | 20.2-20.4 MiB | 10.5 / 14.5 MiB |
| OpenCV ArUco3 half scale, ≈32 px | 3.22 | 6.04 | 8.45 | 20.2-20.4 MiB | 10.1 / 13.7 MiB |
| Locus d2 | 2.37 | 4.35 | 11.0 | 13.0-13.3 MiB | 5.3 / 10.4 MiB |
| RapidTag | 2.76 | 7.11 | 17.8 | 5.9-6.1 MiB | 17.3 / 20.2 MiB |
| Locus | 3.78 | 9.13 | 23.1 | 13.0-13.3 MiB | 12.8 / 27.8 MiB |
| AprilTag dec 2 | 4.84 | 14.7 | 38.3 | 0.3 MiB | 8.0 / 16.3 MiB |
| OpenCV default | 15.0 | 26.5 | 38.3 | 20.2-20.4 MiB | 17.5 / 31.2 MiB |
| AprilTag dec 1 | 17.8 | 55.6 | 139 | 0.3 MiB | 27.3 / 60.8 MiB |

(Locus's construction adds 0.2 MiB; every other construction ≤ 0.05 MiB.) The isolated ratios agree with the
interleaved run in ranking and approximate magnitude (e.g. ArUco Nano ≈8× at 720p in both). That agreement does
not prove the interleaved run had no cross-backend effects (with all threads on one CPU, a pool's spin-wait would
show as wall time, not extra CPU), only that any such effect did not change the comparison.

## Multicore

Intra-frame threads (`resources_6threads_*`): 6 threads per detector on CPUs 0,2,4,6,8,10. NAPA has no
intra-frame threading and stays on 1 CPU (1 OS thread, 1.00 busy CPUs); it is still the lowest latency at every
resolution (mean 0.21 / 0.38 / 0.94 ms at 640/720/1080; next: OpenCV ArUco3 half scale ≈64 px 1.18 / 2.22 /
4.20, Locus d2 1.43 / 2.25 / 5.53, ArUco Nano 1.28 / 3.22 / 7.14). Threading cuts the others' wall time but
raises CPU per frame (busy CPUs 1.3-3.95; AprilTag dec 1 75 ms CPU per 720p frame) and peak memory (RapidTag
+72-90 MiB, AprilTag dec 1 +72 / +125 MiB at 720p / 1080p, OpenCV +52 / +108 MiB).

Frame-parallel throughput (`throughput_6workers_*`): 6 workers on the same CPUs, each building and warming up
its own 1-thread detector in its own thread before a common start (RapidTag: 6 callers sharing a 6-thread Rayon
pool). Frames/s at 640×480 / 720p / 1080p, and peak RSS over the 6 constructed detectors at 1080p:

| Backend | 640×480 | 1280×720 | 1920×1080 | Peak over constructed, 1080p |
| --- | ---: | ---: | ---: | ---: |
| NAPA | 23,379 | 12,626 | 4,509 | 6.6 MiB |
| NAPA mtp16 | 7,468 | 2,787 | 1,175 | 88.5 MiB |
| ArUco Nano | 3,563 | 1,491 | 581 | 33.4 MiB |
| OpenCV ArUco3 half scale ≈64 px | 2,588 | 867 | 535 | 49.7 MiB |
| Locus d2 | 2,000 | 857 | 399 | 68.6 MiB |
| OpenCV ArUco3 half scale ≈32 px | 1,409 | 576 | 398 | 48.8 MiB |
| RapidTag | 1,254 | 394 | 174 | 119.9 MiB |
| Locus | 1,150 | 407 | 183 | 172.8 MiB |
| AprilTag dec 2 | 796 | 250 | 112 | 95.7 MiB |
| OpenCV default | 260 | 141 | 105 | 161.8 MiB |
| AprilTag dec 1 | 230 | 73 | 29 | 360.7 MiB |

Per-frame latencies under parallel load are noisy, so throughput, not per-frame ratio, is the metric here. NAPA's
own throughput varies between runs (8,234-12,626 frames/s at 720p across the two files), which the external load
plausibly explains.

## Real photos, unannotated (`real_photos_latency.*`, `real_photos_agreement.json`)

Upstream AprilTag's 3 test photos (NASA Swarmathon, CC BY-SA, 799×533 JPEG), 30 repetitions on CPU 2. No ground
truth: the only reference is AprilTag's own regression output (decimate 1, refine_edges off, 47 tags of
~20-30 px), so this is timing and agreement, not accuracy. Agreement rule: same ID, centres within 3 px,
one-to-one greedy by distance (pairs saved). NAPA default 0.31-0.33 ms per photo but agreed on 12/47; NAPA mtp16
0.85-2.2 ms, 44/47 (48 detections); ArUco Nano 1.7-2.2 ms, 31; Locus 7.1-8.7 ms, 39 (49 detections); AprilTag
dec 1 11.6-20.8 ms, 46; OpenCV 46-64 ms, 36; RapidTag 4.9-5.8 ms, 7.

## Conclusions

Where NAPA wins (measured), with comparable recall for tags ≥ ~32 px (1,588/1,592 corpus tags vs 1,587-1,590
for ArUco Nano and AprilTag) and no false IDs or duplicates:

- Per-frame latency against other libraries: NAPA's default had a lower median time than every non-NAPA
  configuration on every synthetic frame, at every resolution (closest 1.38-1.43× against a config that found
  none of that frame's tags; ArUco Nano's closest was 2.1×). NAPA's own larger min_tag_px settings are faster
  still, at a recall cost. One NAPA core also beat every other detector given 6 threads.
- Resources, per resolution over a 96-frame mixed workload (not per scenario): the lowest CPU per frame, the
  highest frame-parallel throughput, and approximate working memory of 0.2-1.1 MiB per detector at its default
  (≤15 MiB at min_tag_px 16) against 2.5-61 MiB single-threaded for the others.
- Corner quality, in aggregate: the lowest median and p95 error on tags found by both (median 0.021 px vs
  0.040-0.043 for AprilTag, ArUco Nano and Locus). Not in every subset: on the 15 large 1080p tags AprilTag dec 1
  is marginally better (median/p95 0.0122/0.0386 vs NAPA 0.0130/0.0413 px).
- Deterministic output: identical detections across repetitions on every frame.

Where it loses or pays: (1) tags under ~32 px at default settings (stress-tiny, range sweep, and the real photos,
where it agreed on 12/47); (2) recovering them (min_tag_px 16) costs 2-6× (10× under heavy noise) and gives up
the every-frame win — Locus d2 and ArUco Nano win some frames, and at 1080p its corpus recall dips to 521/526;
(3) two large low-contrast 1080p tags that ArUco Nano, AprilTag and Locus found; (4) worst-case corner error
(3.39 px) is worse than AprilTag's (2.16 px). The other detectors' fast settings lose tags (ArUco3 half scale,
Locus d2 on small tags, RapidTag throughout) or corner quality (OpenCV/RapidTag without refinement ≈1 px); those
tradeoffs are in the tables.

Unresolved: (a) the corpus comes from NAPA's own renderer, against which NAPA was developed (different seed, same
image model), so the size of the lead on independent data is unknown; (b) no annotated real-camera data was
available, so real-world accuracy is unmeasured; (c) one CPU model, laptop thermals near 91 °C, and periodic
external load; (d) NAPA cannot use more than one core per frame, so frame latency on images much larger than
1080p with many cores is untested; (e) the other libraries do more (families, boards, pose) and some are pre-1.0.

## Corrections after review

1. The first outlier rerun rendered different images (its `--seed` set both render and order seeds); its "tail
   inflated" claim is withdrawn. Rerun on the same images; `--order-seed` now separates the two.
2. "Equal or better recall for tags ≥ 32 px" was wrong (four corpus misses above); now "comparable overall".
3. Memory phases were mixed (peak reset before import; parallel "construct" included warm-up). Re-measured with
   separated phases; all resource, 6-thread and throughput results above are from the rerun (the earlier files
   were discarded).
4. NumPy's OpenBLAS pool (6 threads) contaminated multi-CPU workers (one NAPA round showed 1.29 busy CPUs).
   Fixed by limiting it in the child environment; single-CPU runs were never affected.
5. `opencv-aruco3-min32` is half-scale detection with an effective ≈64 px minimum, not 32 px; relabelled, and a
   ≈32 px config (`half16`) was added and measured.
6. Corner-summary rows split (OpenCV default vs SUBPIX p95); p95 ranges now name every backend.
7. Resource workers now disable GC while timing; the interleaving claim is softened to agreement.
8. Real-photo agreement now has a saved, reproducible matching rule (counts changed slightly: AprilTag dec 1
   46, OpenCV 36, ArUco Nano 31).

A second read-only check (GPT-6 Astra) verified fixes 1-8 and asked for the scoped conclusions above, the
qualified outlier wording, peak RSS taken as the larger of VmHWM and sampled RSS (NAPA's 6-worker 1080p figure
6.5 → 6.6 MiB; summaries recomputed from the saved raw readings), and stricter worker error handling (timed-call
failures now fail the run instead of shortening it; no saved run was affected).

## Reproduce

`commands.md`, `run_single_thread.sh`, `run_resources_multicore.sh` (real photos; its resource steps were
superseded), `run_after_review.sh` (current resource, multicore, throughput, outlier and ArUco3 runs),
`run_range_sweep.sh`. The NAPA core measured here is machine-code identical (stripped binaries byte-identical) to
the one built from the repository at the time of publishing; the only source change since is a header comment.
