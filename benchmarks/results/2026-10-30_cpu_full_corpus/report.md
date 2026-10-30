# Fixed full-corpus CPU comparison — 2026-10-30

One fixed shared corpus, recorded in protocol.json before timing, with 15 ordinary backend options: NAPA32/default (`napa`), NAPA16 (`napa-mtp16`) and 13 peer configurations. No frame filters or selection based on detector outputs. Main: 20 scenarios × 50 renders × three resolutions = 3,000 frames. Stress: four × 50 × three = 600 frames. Range: 18 sizes × 50 = 900 frames. Groups are reported separately. Each detector was constructed once and had one warm-up plus ten shuffled timed calls on every image; the first image also has an untimed lifecycle probe. Seed/order seed 2468. Same grayscale tag36h11 images, existing scoring, pixel-origin normalization and corner conventions. Required input copies and result conversion remain timed; construction, rendering, scoring and startup are excluded.

## Machine, build and effective settings

Intel Core i7-12700H laptop, Ubuntu 26.04 x86_64. CPU2 is a verified P-core, with SMT sibling CPU3 monitored. Each backend is configured for one worker; every observed OS thread is confined to CPU2. Frozen native source commit: `f7a7adfe27ca701aee75561be924f64b83bdf442`; rebuilt core SHA256 `270a246d250237f36313cf66a6fc94db7afbba19d3d7fc9612d0499adbfff1d7`. Both NAPA options use the public adapter and retain refinement and correction settings. Maximum placed/detected NAPA tags per frame are both 35, below the 64-output capacity.

AprilTag d1/d2 denotes decimation one/two. Locus d2 is its standard profile with decimation two. OpenCV/RapidTag defaults do not refine corners; SUBPIX variants are separate. ArUco3 full-scale does not refine at its default ratio despite the SUBPIX flag; its two half-scale configurations have effective minimum sizes of approximately 64 and 32 px. Exact versions, builds and effective settings are recorded in matrix.json. Peer binary hashes are not a complete inventory.

## Main corpus: 3,000 frames

Each backend has 30,000 timed calls and the same 5,521 expected tags. CPU mean is process CPU time per detector call. Detection counts are scored once per image, not multiplied by repetitions.

| Detector / configuration | p50 ms | p95 ms | p99 ms | CPU mean ms | Found / tags | Wrong IDs | Dup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NAPA32 (default) | 0.255 | 1.479 | 2.081 | 0.405 | 5513/5521 | 1 | 0 |
| NAPA16 (tuned) | 0.741 | 5.204 | 6.518 | 1.297 | 5519/5521 | 1 | 0 |
| WPILib AprilTag d2 | 11.306 | 41.374 | 48.317 | 14.816 | 4759/5521 | 0 | 0 |
| AprilTag d2 | 14.990 | 51.563 | 59.852 | 19.010 | 5503/5521 | 3 | 0 |
| AprilTag d1 | 57.010 | 204.199 | 233.535 | 72.147 | 5517/5521 | 3 | 0 |
| Locus standard | 8.695 | 26.021 | 34.121 | 10.857 | 5393/5521 | 1 | 0 |
| Locus d2 | 4.287 | 11.562 | 14.093 | 5.148 | 5419/5521 | 1 | 0 |
| OpenCV ArUco | 14.745 | 90.758 | 152.710 | 26.333 | 5486/5521 | 1 | 0 |
| OpenCV SUBPIX | 14.773 | 90.632 | 153.459 | 26.364 | 5486/5521 | 1 | 0 |
| ArUco3 full | 7.960 | 30.161 | 67.950 | 11.333 | 5402/5521 | 1 | 0 |
| ArUco3 min≈64 px | 2.787 | 10.137 | 15.263 | 3.811 | 4415/5521 | 0 | 0 |
| ArUco3 min≈32 px | 3.715 | 14.699 | 21.352 | 5.360 | 5308/5521 | 1 | 0 |
| ArUco Nano | 1.925 | 8.574 | 13.881 | 2.757 | 5517/5521 | 0 | 0 |
| RapidTag | 4.682 | 20.446 | 39.907 | 6.801 | 4167/5521 | 19 | 0 |
| RapidTag SUBPIX | 4.698 | 20.462 | 40.067 | 6.806 | 4167/5521 | 19 | 0 |

Both NAPA options have lower pooled main-corpus p50/p95/p99 than every competitor configuration. NAPA32 has lower per-frame medians on every main-corpus image versus every competitor. NAPA16 slower-frame counts are Locus d2: 1; ArUco3 min≈64 px: 18; ArUco3 min≈32 px: 3; ArUco Nano: 24. These are observed comparisons, not significance tests; near-ties remain included.

| Competitor | Vs NAPA32: median (minimum) | NAPA32 slower / 3000 | Vs NAPA16: median (minimum) | NAPA16 slower / 3000 |
| --- | ---: | ---: | ---: | ---: |
| WPILib AprilTag d2 | 24.98× (3.27607×) | 0 | 12.09× (1.54548×) | 0 |
| AprilTag d2 | 33.59× (3.75824×) | 0 | 16.37× (2.90373×) | 0 |
| AprilTag d1 | 127.56× (9.65688×) | 0 | 63.61× (6.58941×) | 0 |
| Locus standard | 27.07× (7.04917×) | 0 | 10.36× (1.99490×) | 0 |
| Locus d2 | 13.79× (3.26014×) | 0 | 6.12× (0.99987×) | 1 |
| OpenCV ArUco | 45.86× (6.62928×) | 0 | 16.54× (2.27181×) | 0 |
| OpenCV SUBPIX | 45.67× (6.65748×) | 0 | 16.60× (2.23070×) | 0 |
| ArUco3 full | 26.13× (6.12180×) | 0 | 10.32× (2.09818×) | 0 |
| ArUco3 min≈64 px | 9.61× (1.96210×) | 0 | 3.55× (0.70896×) | 18 |
| ArUco3 min≈32 px | 12.30× (2.78592×) | 0 | 4.53× (0.74182×) | 3 |
| ArUco Nano | 7.06× (1.83068×) | 0 | 2.52× (0.83909×) | 24 |
| RapidTag | 16.61× (4.59048×) | 0 | 6.36× (1.50660×) | 0 |
| RapidTag SUBPIX | 16.51× (4.61710×) | 0 | 6.36× (1.50410×) | 0 |

NAPA32 found 5,513/5,521 tags; NAPA16 found 5,519/5,521. Both reported one wrong ID on `decoys-640x480-042`. Nano found 5,517 with no wrong IDs. NAPA16 missed two tags on `low-contrast-1280x720-046` and `-047`; the two NAPA options do not have strictly nested detection sets.

## Corner quality on the main corpus

Worst-corner distance per matched tag after geometric alignment. These medians condition on tags found; compare common-tag tables for a matched quality comparison. Native canonical corner order is recorded separately in matrix.json/CSV and is not the geometric error metric.

| Detector | Median worst-corner px | p95 px | Maximum px |
| --- | ---: | ---: | ---: |
| NAPA32 (default) | 0.0204 | 0.0919 | 3.3907 |
| NAPA16 (tuned) | 0.0201 | 0.0918 | 3.5720 |
| WPILib AprilTag d2 | 0.0986 | 0.3029 | 7.5512 |
| AprilTag d2 | 0.0434 | 0.3051 | 2.0335 |
| AprilTag d1 | 0.0418 | 0.2456 | 2.3490 |
| Locus standard | 0.0438 | 0.2879 | 16.5750 |
| Locus d2 | 0.0439 | 0.2811 | 20.8264 |
| OpenCV ArUco | 0.9867 | 1.6012 | 13.6308 |
| OpenCV SUBPIX | 0.2889 | 0.7398 | 13.6308 |
| ArUco3 full | 0.9859 | 1.6156 | 13.6308 |
| ArUco3 min≈64 px | 0.2820 | 0.6809 | 17.4480 |
| ArUco3 min≈32 px | 0.2841 | 1.0140 | 21.1179 |
| ArUco Nano | 0.0437 | 0.3162 | 15.8934 |
| RapidTag | 0.9997 | 1.7698 | 13.6308 |
| RapidTag SUBPIX | 0.2936 | 0.8139 | 10.6301 |

### Common tags: NAPA32 (default)

| Peer | Common tags | NAPA median / p95 / max px | Peer median / p95 / max px |
| --- | ---: | ---: | ---: |
| AprilTag d1 | 5509 | 0.0203 / 0.0919 / 3.3907 | 0.0418 / 0.2453 / 2.1595 |
| AprilTag d2 | 5495 | 0.0203 / 0.0921 / 3.3907 | 0.0434 / 0.3050 / 1.6336 |
| ArUco Nano | 5509 | 0.0203 / 0.0911 / 2.8837 | 0.0437 / 0.3153 / 15.8934 |
| Locus standard | 5386 | 0.0200 / 0.0886 / 2.8837 | 0.0438 / 0.2817 / 16.5750 |
| Locus d2 | 5412 | 0.0201 / 0.0896 / 2.8837 | 0.0439 / 0.2798 / 20.8264 |
| OpenCV ArUco | 5479 | 0.0204 / 0.0906 / 2.4973 | 0.9866 / 1.6003 / 13.6308 |
| ArUco3 full | 5396 | 0.0202 / 0.0910 / 2.4973 | 0.9859 / 1.6130 / 13.6308 |
| ArUco3 min≈32 px | 5302 | 0.0202 / 0.0917 / 2.4973 | 0.2840 / 0.9976 / 21.1179 |
| ArUco3 min≈64 px | 4414 | 0.0188 / 0.0882 / 1.0029 | 0.2820 / 0.6809 / 17.4480 |
| OpenCV SUBPIX | 5479 | 0.0204 / 0.0906 / 2.4973 | 0.2888 / 0.7379 / 13.6308 |
| RapidTag | 4164 | 0.0214 / 0.1098 / 2.4973 | 0.9997 / 1.7700 / 13.6308 |
| RapidTag SUBPIX | 4164 | 0.0214 / 0.1098 / 2.4973 | 0.2935 / 0.8128 / 10.6301 |
| WPILib AprilTag d2 | 4757 | 0.0195 / 0.0888 / 2.4973 | 0.0986 / 0.3029 / 7.5512 |

### Common tags: NAPA16 (tuned)

| Peer | Common tags | NAPA median / p95 / max px | Peer median / p95 / max px |
| --- | ---: | ---: | ---: |
| AprilTag d1 | 5515 | 0.0200 / 0.0918 / 3.5720 | 0.0418 / 0.2458 / 2.3490 |
| AprilTag d2 | 5501 | 0.0200 / 0.0919 / 3.5720 | 0.0434 / 0.3051 / 2.0335 |
| ArUco Nano | 5515 | 0.0200 / 0.0910 / 3.5720 | 0.0437 / 0.3164 / 15.8934 |
| Locus standard | 5391 | 0.0197 / 0.0870 / 2.7777 | 0.0438 / 0.2880 / 16.5750 |
| Locus d2 | 5417 | 0.0198 / 0.0882 / 2.7777 | 0.0439 / 0.2812 / 20.8264 |
| OpenCV ArUco | 5484 | 0.0201 / 0.0893 / 2.3088 | 0.9866 / 1.6012 / 13.6308 |
| ArUco3 full | 5400 | 0.0199 / 0.0904 / 2.3088 | 0.9857 / 1.6157 / 13.6308 |
| ArUco3 min≈32 px | 5306 | 0.0199 / 0.0917 / 2.3088 | 0.2841 / 1.0151 / 21.1179 |
| ArUco3 min≈64 px | 4413 | 0.0185 / 0.0861 / 0.9789 | 0.2820 / 0.6810 / 17.4480 |
| OpenCV SUBPIX | 5484 | 0.0201 / 0.0893 / 2.3088 | 0.2889 / 0.7398 / 13.6308 |
| RapidTag | 4167 | 0.0211 / 0.1097 / 2.3088 | 0.9997 / 1.7698 / 13.6308 |
| RapidTag SUBPIX | 4167 | 0.0211 / 0.1097 / 2.3088 | 0.2936 / 0.8139 / 10.6301 |
| WPILib AprilTag d2 | 4757 | 0.0192 / 0.0872 / 2.3088 | 0.0986 / 0.3029 / 7.5512 |

## Stress: 600 frames

| Detector / configuration | p50 ms | p95 ms | p99 ms | CPU mean ms | Found / tags | Wrong IDs | Dup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NAPA32 (default) | 0.600 | 2.201 | 2.728 | 0.780 | 4390/4577 | 0 | 0 |
| NAPA16 (tuned) | 1.561 | 6.597 | 10.638 | 2.366 | 4577/4577 | 0 | 0 |
| WPILib AprilTag d2 | 11.301 | 40.119 | 45.502 | 14.798 | 3011/4577 | 0 | 0 |
| AprilTag d2 | 15.831 | 53.908 | 62.904 | 20.309 | 4443/4577 | 1 | 0 |
| AprilTag d1 | 58.495 | 204.379 | 233.922 | 75.260 | 4577/4577 | 1 | 0 |
| Locus standard | 13.389 | 33.434 | 37.747 | 14.649 | 4525/4577 | 0 | 0 |
| Locus d2 | 6.214 | 18.127 | 21.016 | 7.256 | 4402/4577 | 0 | 0 |
| OpenCV ArUco | 34.578 | 120.712 | 138.094 | 45.178 | 4533/4577 | 0 | 0 |
| OpenCV SUBPIX | 34.551 | 120.577 | 138.781 | 45.223 | 4533/4577 | 0 | 0 |
| ArUco3 full | 20.755 | 108.398 | 118.193 | 32.358 | 4070/4577 | 0 | 0 |
| ArUco3 min≈64 px | 5.209 | 22.094 | 26.384 | 7.775 | 2330/4577 | 0 | 0 |
| ArUco3 min≈32 px | 8.314 | 24.931 | 30.872 | 10.302 | 4043/4577 | 0 | 0 |
| ArUco Nano | 4.759 | 18.619 | 20.433 | 6.539 | 4561/4577 | 0 | 0 |
| RapidTag | 11.795 | 83.681 | 95.962 | 22.400 | 3354/4577 | 26 | 0 |
| RapidTag SUBPIX | 11.820 | 84.273 | 96.223 | 22.424 | 3354/4577 | 26 | 0 |

## Range: 900 frames

| Detector / configuration | p50 ms | p95 ms | p99 ms | CPU mean ms | Found / tags | Wrong IDs | Dup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NAPA32 (default) | 0.076 | 0.123 | 0.149 | 0.078 | 748/900 | 0 | 0 |
| NAPA16 (tuned) | 0.141 | 0.199 | 0.231 | 0.147 | 900/900 | 0 | 0 |
| WPILib AprilTag d2 | 5.219 | 6.798 | 7.594 | 5.328 | 263/900 | 0 | 0 |
| AprilTag d2 | 6.544 | 8.573 | 9.600 | 6.647 | 719/900 | 0 | 0 |
| AprilTag d1 | 25.331 | 32.300 | 35.537 | 25.333 | 899/900 | 0 | 0 |
| Locus standard | 2.933 | 3.715 | 4.038 | 2.923 | 887/900 | 0 | 0 |
| Locus d2 | 1.327 | 1.723 | 1.943 | 1.341 | 688/900 | 0 | 0 |
| OpenCV ArUco | 1.653 | 3.711 | 5.315 | 1.896 | 882/900 | 0 | 0 |
| OpenCV SUBPIX | 1.655 | 3.715 | 5.356 | 1.899 | 882/900 | 0 | 0 |
| ArUco3 full | 1.108 | 2.017 | 2.500 | 1.259 | 420/900 | 0 | 0 |
| ArUco3 min≈64 px | 0.357 | 1.165 | 1.454 | 0.470 | 173/900 | 0 | 0 |
| ArUco3 min≈32 px | 0.420 | 1.345 | 1.740 | 0.633 | 416/900 | 0 | 0 |
| ArUco Nano | 0.412 | 0.569 | 0.662 | 0.425 | 882/900 | 0 | 0 |
| RapidTag | 0.869 | 1.126 | 1.271 | 0.891 | 247/900 | 0 | 0 |
| RapidTag SUBPIX | 0.869 | 1.125 | 1.261 | 0.892 | 247/900 | 0 | 0 |

## Measurement conditions and limits

The fixed run lasted approximately 2h 56m. The 2,086 monitor samples show CPU2 frequency 2.799–4.578 GHz (median 3.952 GHz); package temperature median 91°C, maximum 97°C. CPU3 busy fraction was median 0%, p95 3%, maximum 38%. Background applications were retained. Quarter-wise median frequencies stayed approximately 3.92–3.99 GHz; the observations do not by themselves establish throttling. All outliers and all preselected frames remain included.

This interleaved single-core run measures call latency and process CPU, not backend-isolated RSS or multicore performance. Earlier [isolated resource measurements](../2026-10-29_cpu_optimization/report.md#cpu-and-working-memory) use separate 96-frame sets and retain their original provenance; they are not full-corpus resource results. The earlier 149-frame set is a development diagnostic only and is superseded for headline latency/quality. This synthetic renderer and one host do not establish real-camera accuracy or a universal winner.

## Integrity and reproduction

Independent read-only audit approved methodology and completed data: 67,500 rows; 4,500 shared images per backend; complete scenario/index coverage; equal hashes, truth and frame properties; ten positive wall/CPU samples each; no skips or unstable rows; frozen native/benchmark hashes; matching preflight settings. The monitor is not a continuous process inventory, so absolute absence of other jobs throughout cannot be proved from it.

[Protocol](protocol.json), [commands](commands.md), [raw JSON](matrix.json), [raw CSV](matrix.csv), [thermal/load monitor](monitor.tsv), [adapter validation](validate.json), [analysis](analysis.json). Run `work/venv/bin/python benchmarks/results/2026-10-30_cpu_full_corpus/make_report.py` to regenerate this report without detecting or loading a native library. Add `--write-docs` to refresh README tables.
