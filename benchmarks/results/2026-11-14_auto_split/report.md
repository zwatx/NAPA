# Automatic splitting: saved November 14 results

This report reuses the completed `out-auto2` (25 laptop sets) and `out-pi-auto2` (12 Pi sets) runs. It was assembled on
November 15 without running a detector again. Dates refer to America/Chicago; the machine-readable summary retains UTC
timestamps. These saved measurements predate the final consolidated frame-edge merge; they are not a rerun of this
branch.

## Build and method

Core source SHA256: `b176874f0a7417eb715408ba8a18c99f0a7b526aeaf4b18735909f9db4277d99`. Both settings use tag36h11,
minimum 22 px, no shadow pass, and `max_detections=2048`. The two settings differ in `split_touching`: `"auto"` or
`True`. The benchmark keeps the detector alive and runs each image twice: a first call followed by a timed warm call.
Backend order alternates by frame. Rendering, disk loading and scoring are outside the timed call; Python result
conversion is included.

Laptop chains run concurrently on CPUs 4, 8 and 12 of an Intel i7-12700H. CPUs 4/8 are P-cores; CPU12 is an E-core. Each
detector uses one core, but the laptop rows are not all P-core measurements. Pi chains use CPUs 2/3 of a Raspberry Pi 5,
one core per detector. Use each row's frame count: the Pi board/large-footage packs are subsets of the laptop packs.

The original harness was `fb.py PACK --repo <perf-checkout> --only
napa-m22,napa-m22-auto,napa-m22-split,napa-m16,napa-m16-auto,napa-preset,napa-preset-auto`. This summary retains the
explicitly configured auto/True rows only. Other options and sampled pixels are unchanged within each saved pack.

## Representative Pi results

| Dataset | Frames | Matched tags / labels, or outputs | Median ms, one Pi 5 core |
| --- | --- | --- | --- |
| Robot-camera captures | 297 | 684 outputs (no hand truth) | 1.226 |
| FRC footage | 188 | 346 outputs (no hand truth) | 1.668 |
| Locus AprilGrid renders | 200 | 2,768 / 2,916 | 5.743 |
| ICRA forward renders | 50 | 4,258 / 7,700 | 9.882 |

Output counts on unlabelled footage are not recall or proof of correct IDs. Locus annotations omit some visible edge
tags; unmatched Locus outputs are therefore not all known false IDs. The saved auto Locus row has 14 unmatched outputs
on the Pi and 53 on the laptop. Do not combine ground-truth matches, box matches, sightings and unlabelled outputs into
one accuracy score.

## Saved per-set results

### Laptop

| Pack | Frames | CPU | Auto matches / outputs | Auto p50 ms | Always-split matches / outputs | Always-split p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| basler-bg | 300 | 4 | 0 false outputs (negative frames) | 0.406 | 0 false outputs (negative frames) | 0.630 |
| Robot-camera captures | 297 | 12 | 684 outputs (no hand truth) | 0.873 | 691 outputs (no hand truth) | 1.290 |
| bd-hard | 780 | 12 | 2,352 / 2,389 | 0.294 | 2,353 / 2,389 | 0.328 |
| bd-scenes | 128 | 8 | 1,024 / 1,024 | 0.471 | 1,024 / 1,024 | 0.500 |
| drive-test | 194 | 12 | 361 / 573 boxes | 0.474 | 373 / 573 boxes | 0.599 |
| drive-valid | 1097 | 12 | 2,049 / 3,186 boxes | 0.496 | 2,132 / 3,186 boxes | 0.619 |
| More FRC footage | 1000 | 12 | 1,949 outputs (no hand truth) | 1.294 | 1,959 outputs (no hand truth) | 1.536 |
| fresh | 120 | 4 | 0 false outputs (negative frames) | 0.168 | 0 false outputs (negative frames) | 0.221 |
| gopro | 1560 | 12 | 7,566 surveyed-ID sightings | 2.204 | 7,769 surveyed-ID sightings | 2.880 |
| ICRA forward renders | 100 | 4 | 8,620 / 15,400 | 5.373 | 8,914 / 15,400 | 5.738 |
| icra-circle | 181 | 4 | 23,241 / 27,862 | 5.604 | 23,241 / 27,862 | 5.951 |
| icra-random | 400 | 8 | 38,300 / 38,300 | 4.452 | 38,300 / 38,300 | 4.763 |
| icra-rotation | 361 | 12 | 55,594 / 55,594 | 7.689 | 55,594 / 55,594 | 8.187 |
| indoor | 268 | 4 | 0 false outputs (negative frames) | 0.110 | 0 false outputs (negative frames) | 0.166 |
| Locus AprilGrid renders | 600 | 12 | 8,148 / 8,743 | 4.098 | 8,148 / 8,743 | 4.598 |
| Fixed synthetic corpus | 4500 | 8 | 10,952 / 10,998 | 0.211 | 10,969 / 10,998 | 0.245 |
| penn-tango-bottom | 1563 | 8 | 6,136 surveyed-ID sightings | 0.997 | 6,210 surveyed-ID sightings | 1.228 |
| penn-tango-top | 1563 | 8 | 735 surveyed-ID sightings | 0.100 | 774 surveyed-ID sightings | 0.139 |
| penn-visensor | 1489 | 12 | 948 surveyed-ID sightings | 0.275 | 966 surveyed-ID sightings | 0.349 |
| Range subset | 900 | 12 | 893 / 900 | 0.083 | 893 / 900 | 0.084 |
| FRC footage | 188 | 4 | 346 outputs (no hand truth) | 0.934 | 350 outputs (no hand truth) | 1.204 |
| rf-test | 258 | 8 | 434 / 1,039 boxes | 0.253 | 437 / 1,039 boxes | 0.323 |
| rf-valid | 259 | 8 | 384 / 1,069 boxes | 0.240 | 384 / 1,069 boxes | 0.320 |
| Stress subset | 600 | 12 | 4,540 / 4,577 | 0.964 | 4,555 / 4,577 | 1.255 |
| vk | 4113 | 4 | 9,818 / 9,991 | 0.148 | 9,827 / 9,991 | 0.157 |

### Pi

| Pack | Frames | CPU | Auto matches / outputs | Auto p50 ms | Always-split matches / outputs | Always-split p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| basler-bg | 300 | 2 | 0 false outputs (negative frames) | 0.675 | 0 false outputs (negative frames) | 1.002 |
| Robot-camera captures | 297 | 3 | 684 outputs (no hand truth) | 1.226 | 691 outputs (no hand truth) | 1.767 |
| drive-test | 194 | 2 | 361 / 573 boxes | 0.616 | 373 / 573 boxes | 0.806 |
| More FRC footage | 500 | 3 | 968 outputs (no hand truth) | 1.668 | 975 outputs (no hand truth) | 2.004 |
| fresh | 120 | 3 | 0 false outputs (negative frames) | 0.294 | 0 false outputs (negative frames) | 0.359 |
| ICRA forward renders | 50 | 3 | 4,258 / 7,700 | 9.882 | 4,363 / 7,700 | 10.721 |
| icra-random | 67 | 3 | 6,877 / 6,877 | 10.362 | 6,877 / 6,877 | 11.009 |
| indoor | 268 | 2 | 0 false outputs (negative frames) | 0.175 | 0 false outputs (negative frames) | 0.253 |
| Locus AprilGrid renders | 200 | 2 | 2,768 / 2,916 | 5.743 | 2,768 / 2,916 | 6.268 |
| FRC footage | 188 | 2 | 346 outputs (no hand truth) | 1.668 | 350 outputs (no hand truth) | 2.082 |
| rf-test | 258 | 2 | 434 / 1,039 boxes | 0.410 | 437 / 1,039 boxes | 0.508 |
| rf-valid | 259 | 2 | 384 / 1,069 boxes | 0.398 | 384 / 1,069 boxes | 0.512 |

## Integrity and scope

The original score files and every saved compressed per-frame result were read directly. Every pack had a completion
footer; row counts, output totals and warm-call medians were checked against the score summaries for both
configurations. [summary.json](summary.json) retains the selected settings, core hashes, source-result hashes, frame
counts and scoring metrics. No images or private source paths are included. The original full result archives remain in
the local benchmark cache; the committed summary is not a replacement for those raw archives.

The fixed `own` pack includes main, stress and range, totalling 4,500 frames/10,998 tags; the separately listed
stress/range packs overlap it and must not be added to it. Corner errors condition on detected tags. The 2048-output
benchmark capacity differs from the default API capacity of 64. These figures do not measure later
regrowth/thread-prepass/frame-edge changes, multicore scaling, or real-camera pose accuracy.

[November 11 head-to-head report](../2026-11-11_full_benchmark/report.md) and [October 30 fixed-corpus
comparison](../2026-10-30_cpu_full_corpus/report.md) retain their original builds, settings and measurements. No
cross-run timing ratios were calculated here.

## Earlier peer comparison (November 11)

These rows reuse the committed `scores/pi-scores.json` Basler-test results from the November 11 full comparison, on 297
frames. They are a separate build and run from the auto results above; the NAPA row is the explicitly configured recall
preset (minimum16, always split, shadows24), not the current default. The source file hash and selected settings/metrics
are in `summary.json`.

| Detector / setting | Median ms, one Pi 5 core | p99 ms |
| --- | --- | --- |
| NAPA recall preset (16, split, shadows 24) | 6.631 | 7.810 |
| AprilTag 3, decimate 2 | 16.716 | 19.015 |
| AprilTag 3, full resolution | 57.468 | 62.643 |
| AprilTag, WPILib defaults | 13.809 | 15.710 |
| ArUco Nano | 14.646 | 15.310 |
