# CPU comparison — 2026-11-15

2026-11-15: 3,000 main frames, 600 stress frames and 900 range frames, shared by all 16 configurations. Each frame has 1 warm-up and 10 timed calls per detector, seed/order seed 2468/2468, single-threaded on CPU2 of 12th Gen Intel(R) Core(TM) i7-12700H. Timing includes detector calls and adapter conversion, excluding construction, rendering and scoring. Latency percentiles pool timed calls; quality counts each frame once. Speedup is the median of paired per-frame median WPILib time divided by detector time. Found counts match IDs without a geometric gate. Corner order is aligned over cyclic shifts and reflections before measuring error. Corner error is the worst corner per matched tag; its distribution conditions on detection. NAPA's current default is min_tag_px=22, split_touching=auto.

## Main: 3,000 frames

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.315 | 2.052 | 2.971 | 5519/5521 | 1 | 0 |
| NAPA, min_tag_px=16 | 0.569 | 3.885 | 5.042 | 5521/5521 | 1 | 0 |
| NAPA, min_tag_px=32 | 0.248 | 1.424 | 2.105 | 5512/5521 | 1 | 0 |
| AprilTag, WPILib defaults | 10.747 | 39.018 | 44.497 | 4759/5521 | 0 | 0 |
| AprilTag, decimate 2 | 14.050 | 48.315 | 55.551 | 5503/5521 | 3 | 0 |
| AprilTag, full resolution | 54.595 | 193.457 | 213.104 | 5517/5521 | 3 | 0 |
| Locus standard | 8.317 | 24.018 | 31.832 | 5393/5521 | 1 | 0 |
| Locus, decimate 2 | 4.025 | 10.954 | 12.513 | 5419/5521 | 1 | 0 |
| OpenCV ArUco | 14.023 | 84.889 | 141.076 | 5486/5521 | 1 | 0 |
| OpenCV ArUco, subpixel | 14.038 | 85.451 | 140.408 | 5486/5521 | 1 | 0 |
| ArUco3 full | 7.642 | 28.925 | 63.899 | 5402/5521 | 1 | 0 |
| ArUco3, minimum ≈64 px | 2.620 | 9.311 | 14.230 | 4415/5521 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.523 | 13.635 | 19.852 | 5308/5521 | 1 | 0 |
| ArUco Nano | 1.803 | 8.020 | 13.039 | 5517/5521 | 0 | 0 |
| RapidTag | 4.384 | 19.261 | 37.145 | 4167/5521 | 19 | 0 |
| RapidTag, subpixel | 4.396 | 19.247 | 37.089 | 4167/5521 | 19 | 0 |

| Detector | Median paired speedup vs WPILib | Corner median / p95 / max px |
| --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 23.14× | 0.017 / 0.080 / 0.764 |
| NAPA, min_tag_px=16 | 13.18× | 0.017 / 0.079 / 0.678 |
| NAPA, min_tag_px=32 | 27.26× | 0.017 / 0.078 / 1.560 |
| AprilTag, WPILib defaults | 1.00× | 0.099 / 0.303 / 7.551 |
| AprilTag, decimate 2 | 0.80× | 0.043 / 0.305 / 2.034 |
| AprilTag, full resolution | 0.21× | 0.042 / 0.246 / 2.349 |
| Locus standard | 1.29× | 0.044 / 0.288 / 16.575 |
| Locus, decimate 2 | 2.41× | 0.044 / 0.281 / 20.826 |
| OpenCV ArUco | 0.55× | 0.987 / 1.601 / 13.631 |
| OpenCV ArUco, subpixel | 0.55× | 0.289 / 0.740 / 13.631 |
| ArUco3 full | 1.16× | 0.986 / 1.616 / 13.631 |
| ArUco3, minimum ≈64 px | 3.15× | 0.282 / 0.681 / 17.448 |
| ArUco3, minimum ≈32 px | 2.23× | 0.284 / 1.014 / 21.118 |
| ArUco Nano | 4.82× | 0.044 / 0.316 / 15.893 |
| RapidTag | 1.96× | 1.000 / 1.770 / 13.631 |
| RapidTag, subpixel | 1.96× | 0.294 / 0.814 / 10.630 |

## Stress: 600 frames

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.738 | 3.097 | 3.982 | 4540/4577 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.300 | 5.377 | 7.860 | 4577/4577 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.561 | 2.266 | 2.714 | 4442/4577 | 0 | 0 |
| AprilTag, WPILib defaults | 10.551 | 38.396 | 41.369 | 3011/4577 | 0 | 0 |
| AprilTag, decimate 2 | 15.095 | 50.759 | 58.795 | 4443/4577 | 1 | 0 |
| AprilTag, full resolution | 54.500 | 196.243 | 212.410 | 4577/4577 | 1 | 0 |
| Locus standard | 12.796 | 32.607 | 36.412 | 4525/4577 | 0 | 0 |
| Locus, decimate 2 | 5.840 | 17.200 | 19.991 | 4402/4577 | 0 | 0 |
| OpenCV ArUco | 33.194 | 117.602 | 127.038 | 4533/4577 | 0 | 0 |
| OpenCV ArUco, subpixel | 33.281 | 117.667 | 127.133 | 4533/4577 | 0 | 0 |
| ArUco3 full | 20.130 | 106.067 | 111.734 | 4070/4577 | 0 | 0 |
| ArUco3, minimum ≈64 px | 4.845 | 21.480 | 23.758 | 2330/4577 | 0 | 0 |
| ArUco3, minimum ≈32 px | 7.962 | 23.718 | 28.283 | 4043/4577 | 0 | 0 |
| ArUco Nano | 4.469 | 18.009 | 19.229 | 4561/4577 | 0 | 0 |
| RapidTag | 10.993 | 81.283 | 89.012 | 3354/4577 | 26 | 0 |
| RapidTag, subpixel | 11.022 | 81.126 | 89.229 | 3354/4577 | 26 | 0 |

| Detector | Median paired speedup vs WPILib | Corner median / p95 / max px |
| --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 11.02× | 0.017 / 0.090 / 1.587 |
| NAPA, min_tag_px=16 | 5.67× | 0.017 / 0.090 / 0.341 |
| NAPA, min_tag_px=32 | 16.96× | 0.017 / 0.090 / 0.549 |
| AprilTag, WPILib defaults | 1.00× | 0.113 / 0.226 / 0.472 |
| AprilTag, decimate 2 | 0.74× | 0.041 / 0.128 / 2.534 |
| AprilTag, full resolution | 0.20× | 0.042 / 0.136 / 0.595 |
| Locus standard | 0.88× | 0.043 / 0.483 / 9.694 |
| Locus, decimate 2 | 1.74× | 0.042 / 0.308 / 9.500 |
| OpenCV ArUco | 0.30× | 0.936 / 1.290 / 37.062 |
| OpenCV ArUco, subpixel | 0.30× | 0.294 / 0.820 / 39.242 |
| ArUco3 full | 0.41× | 0.934 / 1.286 / 37.062 |
| ArUco3, minimum ≈64 px | 2.00× | 0.268 / 0.446 / 17.000 |
| ArUco3, minimum ≈32 px | 1.60× | 0.268 / 0.489 / 17.000 |
| ArUco Nano | 1.92× | 0.042 / 0.135 / 18.688 |
| RapidTag | 0.73× | 0.937 / 1.293 / 37.062 |
| RapidTag, subpixel | 0.73× | 0.287 / 0.698 / 39.242 |

## Range: 900 frames

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.092 | 0.152 | 0.189 | 893/900 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.119 | 0.180 | 0.218 | 900/900 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.083 | 0.144 | 0.182 | 824/900 | 0 | 0 |
| AprilTag, WPILib defaults | 4.915 | 5.858 | 6.373 | 263/900 | 0 | 0 |
| AprilTag, decimate 2 | 6.110 | 7.429 | 8.081 | 719/900 | 0 | 0 |
| AprilTag, full resolution | 23.355 | 27.854 | 29.741 | 899/900 | 0 | 0 |
| Locus standard | 2.782 | 3.056 | 3.320 | 887/900 | 0 | 0 |
| Locus, decimate 2 | 1.245 | 1.466 | 1.563 | 688/900 | 0 | 0 |
| OpenCV ArUco | 1.536 | 3.451 | 5.186 | 882/900 | 0 | 0 |
| OpenCV ArUco, subpixel | 1.536 | 3.449 | 5.150 | 882/900 | 0 | 0 |
| ArUco3 full | 0.953 | 1.864 | 2.006 | 420/900 | 0 | 0 |
| ArUco3, minimum ≈64 px | 0.333 | 1.109 | 1.301 | 173/900 | 0 | 0 |
| ArUco3, minimum ≈32 px | 0.366 | 1.259 | 1.422 | 416/900 | 0 | 0 |
| ArUco Nano | 0.377 | 0.486 | 0.575 | 882/900 | 0 | 0 |
| RapidTag | 0.811 | 0.957 | 1.038 | 247/900 | 0 | 0 |
| RapidTag, subpixel | 0.811 | 0.959 | 1.027 | 247/900 | 0 | 0 |

| Detector | Median paired speedup vs WPILib | Corner median / p95 / max px |
| --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 53.38× | 0.029 / 0.108 / 0.448 |
| NAPA, min_tag_px=16 | 40.90× | 0.029 / 0.113 / 0.451 |
| NAPA, min_tag_px=32 | 58.40× | 0.027 / 0.097 / 0.289 |
| AprilTag, WPILib defaults | 1.00× | 0.103 / 0.210 / 0.280 |
| AprilTag, decimate 2 | 0.80× | 0.042 / 0.203 / 2.152 |
| AprilTag, full resolution | 0.21× | 0.050 / 0.269 / 0.687 |
| Locus standard | 1.78× | 0.068 / 0.933 / 1.446 |
| Locus, decimate 2 | 3.94× | 0.050 / 1.199 / 2.209 |
| OpenCV ArUco | 3.20× | 0.918 / 1.241 / 1.494 |
| OpenCV ArUco, subpixel | 3.20× | 0.399 / 1.179 / 1.719 |
| ArUco3 full | 4.86× | 0.916 / 1.235 / 1.493 |
| ArUco3, minimum ≈64 px | 14.80× | 0.273 / 0.368 / 0.420 |
| ArUco3, minimum ≈32 px | 13.17× | 0.265 / 0.369 / 5.149 |
| ArUco Nano | 13.00× | 0.049 / 0.240 / 14.665 |
| RapidTag | 6.02× | 0.904 / 1.233 / 1.493 |
| RapidTag, subpixel | 6.02× | 0.304 / 0.891 / 1.468 |

## Range by size

| Detector / black-square px | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 28 | 32 | 36 | 40 | 48 | 56 | 64 | 72 | 80 | 88 | 96 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 44/50 | 49/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| NAPA, min_tag_px=16 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| NAPA, min_tag_px=32 | 8/50 | 37/50 | 38/50 | 43/50 | 48/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| AprilTag, WPILib defaults | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 26/50 | 39/50 | 48/50 | 50/50 | 50/50 | 50/50 |
| AprilTag, decimate 2 | 4/50 | 8/50 | 9/50 | 20/50 | 33/50 | 45/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| AprilTag, full resolution | 49/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| Locus standard | 44/50 | 48/50 | 46/50 | 49/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| Locus, decimate 2 | 0/50 | 0/50 | 10/50 | 15/50 | 29/50 | 37/50 | 50/50 | 50/50 | 50/50 | 49/50 | 50/50 | 49/50 | 50/50 | 50/50 | 50/50 | 49/50 | 50/50 | 50/50 |
| OpenCV ArUco | 40/50 | 44/50 | 49/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 49/50 |
| OpenCV ArUco, subpixel | 40/50 | 44/50 | 49/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 49/50 |
| ArUco3 full | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 31/50 | 40/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 49/50 |
| ArUco3, minimum ≈64 px | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 32/50 | 44/50 | 48/50 | 49/50 |
| ArUco3, minimum ≈32 px | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 30/50 | 40/50 | 50/50 | 49/50 | 50/50 | 50/50 | 50/50 | 49/50 | 48/50 |
| ArUco Nano | 36/50 | 46/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 | 50/50 |
| RapidTag | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 21/50 | 25/50 | 35/50 | 29/50 | 30/50 | 31/50 | 26/50 | 24/50 | 26/50 |
| RapidTag, subpixel | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 0/50 | 21/50 | 25/50 | 35/50 | 29/50 | 30/50 | 31/50 | 26/50 | 24/50 | 26/50 |

## Per-scene main results

### empty, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.029 | 0.059 | 0.076 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.047 | 0.075 | 0.095 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.026 | 0.048 | 0.065 | 0/0 | 0 | 0 |
| AprilTag, WPILib defaults | 5.298 | 6.555 | 6.840 | 0/0 | 0 | 0 |
| AprilTag, decimate 2 | 6.833 | 8.357 | 8.883 | 0/0 | 0 | 0 |
| AprilTag, full resolution | 27.436 | 30.904 | 32.935 | 0/0 | 0 | 0 |
| Locus standard | 2.558 | 2.839 | 3.221 | 0/0 | 0 | 0 |
| Locus, decimate 2 | 1.016 | 1.124 | 1.202 | 0/0 | 0 | 0 |
| OpenCV ArUco | 0.838 | 1.325 | 1.445 | 0/0 | 0 | 0 |
| OpenCV ArUco, subpixel | 0.839 | 1.324 | 1.440 | 0/0 | 0 | 0 |
| ArUco3 full | 0.900 | 1.413 | 1.499 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈64 px | 0.300 | 0.353 | 0.386 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈32 px | 0.295 | 0.357 | 0.402 | 0/0 | 0 | 0 |
| ArUco Nano | 0.418 | 0.735 | 0.821 | 0/0 | 0 | 0 |
| RapidTag | 0.756 | 0.991 | 1.063 | 0/0 | 0 | 0 |
| RapidTag, subpixel | 0.752 | 1.003 | 1.099 | 0/0 | 0 | 0 |

### clutter-only, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.203 | 0.432 | 0.512 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.421 | 0.736 | 0.776 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.132 | 0.331 | 0.383 | 0/0 | 0 | 0 |
| AprilTag, WPILib defaults | 3.640 | 4.866 | 5.133 | 0/0 | 0 | 0 |
| AprilTag, decimate 2 | 5.158 | 7.450 | 8.193 | 0/0 | 0 | 0 |
| AprilTag, full resolution | 18.743 | 24.994 | 26.036 | 0/0 | 0 | 0 |
| Locus standard | 2.987 | 5.312 | 6.000 | 0/0 | 0 | 0 |
| Locus, decimate 2 | 1.471 | 2.013 | 2.185 | 0/0 | 0 | 0 |
| OpenCV ArUco | 13.839 | 91.628 | 124.152 | 0/0 | 0 | 0 |
| OpenCV ArUco, subpixel | 13.816 | 91.600 | 123.996 | 0/0 | 0 | 0 |
| ArUco3 full | 2.745 | 21.670 | 28.380 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈64 px | 0.850 | 2.554 | 9.266 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.191 | 11.815 | 12.976 | 0/0 | 0 | 0 |
| ArUco Nano | 0.821 | 1.470 | 1.528 | 0/0 | 0 | 0 |
| RapidTag | 2.260 | 3.409 | 3.780 | 0/0 | 0 | 0 |
| RapidTag, subpixel | 2.254 | 3.401 | 3.794 | 0/0 | 0 | 0 |

### decoys, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.127 | 0.199 | 0.232 | 0/0 | 1 | 0 |
| NAPA, min_tag_px=16 | 0.238 | 0.338 | 0.389 | 0/0 | 1 | 0 |
| NAPA, min_tag_px=32 | 0.117 | 0.184 | 0.216 | 0/0 | 1 | 0 |
| AprilTag, WPILib defaults | 3.692 | 4.998 | 5.220 | 0/0 | 0 | 0 |
| AprilTag, decimate 2 | 4.825 | 6.493 | 6.779 | 0/0 | 1 | 0 |
| AprilTag, full resolution | 18.659 | 24.539 | 27.404 | 0/0 | 1 | 0 |
| Locus standard | 2.419 | 2.910 | 3.119 | 0/0 | 1 | 0 |
| Locus, decimate 2 | 1.147 | 1.533 | 1.763 | 0/0 | 1 | 0 |
| OpenCV ArUco | 13.133 | 21.102 | 22.519 | 0/0 | 1 | 0 |
| OpenCV ArUco, subpixel | 13.061 | 21.129 | 22.789 | 0/0 | 1 | 0 |
| ArUco3 full | 6.952 | 11.196 | 13.578 | 0/0 | 1 | 0 |
| ArUco3, minimum ≈64 px | 4.321 | 6.442 | 8.255 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈32 px | 5.460 | 10.061 | 11.074 | 0/0 | 1 | 0 |
| ArUco Nano | 1.766 | 2.720 | 3.691 | 0/0 | 0 | 0 |
| RapidTag | 1.821 | 2.279 | 2.377 | 0/0 | 1 | 0 |
| RapidTag, subpixel | 1.821 | 2.256 | 2.405 | 0/0 | 1 | 0 |

### one-tag, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.095 | 0.137 | 0.157 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.136 | 0.188 | 0.208 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.089 | 0.127 | 0.144 | 50/50 | 0 | 0 |
| AprilTag, WPILib defaults | 4.510 | 5.554 | 5.885 | 46/50 | 0 | 0 |
| AprilTag, decimate 2 | 5.522 | 6.907 | 7.141 | 50/50 | 0 | 0 |
| AprilTag, full resolution | 21.827 | 27.372 | 28.096 | 50/50 | 0 | 0 |
| Locus standard | 2.915 | 3.150 | 3.259 | 50/50 | 0 | 0 |
| Locus, decimate 2 | 1.411 | 1.550 | 1.638 | 50/50 | 0 | 0 |
| OpenCV ArUco | 2.483 | 5.750 | 6.582 | 50/50 | 0 | 0 |
| OpenCV ArUco, subpixel | 2.492 | 5.733 | 6.603 | 50/50 | 0 | 0 |
| ArUco3 full | 1.632 | 2.713 | 3.211 | 50/50 | 0 | 0 |
| ArUco3, minimum ≈64 px | 0.812 | 1.358 | 1.444 | 39/50 | 0 | 0 |
| ArUco3, minimum ≈32 px | 0.943 | 1.466 | 2.034 | 50/50 | 0 | 0 |
| ArUco Nano | 0.452 | 0.727 | 0.856 | 50/50 | 0 | 0 |
| RapidTag | 0.969 | 1.110 | 1.180 | 27/50 | 0 | 0 |
| RapidTag, subpixel | 0.984 | 1.121 | 1.207 | 27/50 | 0 | 0 |

### many-tags, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.302 | 0.511 | 0.626 | 249/249 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.446 | 0.787 | 0.910 | 249/249 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.285 | 0.451 | 0.497 | 249/249 | 0 | 0 |
| AprilTag, WPILib defaults | 3.468 | 4.865 | 5.145 | 178/249 | 0 | 0 |
| AprilTag, decimate 2 | 4.914 | 6.400 | 7.312 | 249/249 | 0 | 0 |
| AprilTag, full resolution | 17.156 | 23.376 | 25.838 | 249/249 | 0 | 0 |
| Locus standard | 4.039 | 5.463 | 5.857 | 249/249 | 0 | 0 |
| Locus, decimate 2 | 2.853 | 3.746 | 3.984 | 249/249 | 0 | 0 |
| OpenCV ArUco | 17.853 | 74.198 | 106.527 | 248/249 | 0 | 0 |
| OpenCV ArUco, subpixel | 17.888 | 74.044 | 106.335 | 248/249 | 0 | 0 |
| ArUco3 full | 4.691 | 9.611 | 22.566 | 247/249 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.964 | 3.216 | 3.441 | 125/249 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.441 | 6.348 | 13.028 | 246/249 | 0 | 0 |
| ArUco Nano | 0.978 | 1.823 | 2.131 | 249/249 | 0 | 0 |
| RapidTag | 2.120 | 3.332 | 3.533 | 200/249 | 0 | 0 |
| RapidTag, subpixel | 2.138 | 3.338 | 3.561 | 200/249 | 0 | 0 |

### small-tags, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.272 | 0.524 | 0.584 | 131/131 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.447 | 0.770 | 0.819 | 131/131 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.220 | 0.420 | 0.473 | 131/131 | 0 | 0 |
| AprilTag, WPILib defaults | 3.844 | 5.130 | 5.365 | 0/131 | 0 | 0 |
| AprilTag, decimate 2 | 5.985 | 7.166 | 7.811 | 131/131 | 0 | 0 |
| AprilTag, full resolution | 18.612 | 25.185 | 26.642 | 131/131 | 0 | 0 |
| Locus standard | 3.359 | 5.239 | 5.683 | 131/131 | 0 | 0 |
| Locus, decimate 2 | 1.974 | 2.858 | 3.206 | 131/131 | 0 | 0 |
| OpenCV ArUco | 14.984 | 101.073 | 110.997 | 131/131 | 0 | 0 |
| OpenCV ArUco, subpixel | 14.833 | 100.975 | 110.865 | 131/131 | 0 | 0 |
| ArUco3 full | 3.357 | 9.254 | 27.671 | 106/131 | 0 | 0 |
| ArUco3, minimum ≈64 px | 0.834 | 1.859 | 4.263 | 0/131 | 0 | 0 |
| ArUco3, minimum ≈32 px | 2.037 | 4.766 | 7.663 | 100/131 | 0 | 0 |
| ArUco Nano | 0.808 | 1.624 | 1.743 | 131/131 | 0 | 0 |
| RapidTag | 2.230 | 3.203 | 3.342 | 84/131 | 0 | 0 |
| RapidTag, subpixel | 2.218 | 3.175 | 3.338 | 84/131 | 0 | 0 |

### large-tags, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.178 | 0.445 | 0.510 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.314 | 0.741 | 0.810 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.160 | 0.360 | 0.420 | 50/50 | 0 | 0 |
| AprilTag, WPILib defaults | 3.442 | 4.486 | 4.975 | 50/50 | 0 | 0 |
| AprilTag, decimate 2 | 4.868 | 6.379 | 6.760 | 50/50 | 0 | 0 |
| AprilTag, full resolution | 18.298 | 23.197 | 24.232 | 50/50 | 0 | 0 |
| Locus standard | 3.121 | 4.503 | 5.417 | 50/50 | 0 | 0 |
| Locus, decimate 2 | 1.869 | 2.517 | 2.942 | 50/50 | 0 | 0 |
| OpenCV ArUco | 10.174 | 71.044 | 124.833 | 50/50 | 0 | 0 |
| OpenCV ArUco, subpixel | 10.295 | 70.844 | 124.512 | 50/50 | 0 | 0 |
| ArUco3 full | 2.860 | 10.708 | 18.910 | 50/50 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.212 | 2.974 | 6.255 | 50/50 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.531 | 8.188 | 11.364 | 50/50 | 0 | 0 |
| ArUco Nano | 0.807 | 1.646 | 1.759 | 50/50 | 0 | 0 |
| RapidTag | 1.959 | 2.995 | 3.333 | 43/50 | 0 | 0 |
| RapidTag, subpixel | 1.986 | 3.002 | 3.352 | 43/50 | 0 | 0 |

### steep-angle, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.222 | 0.484 | 0.551 | 108/109 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.361 | 0.748 | 0.786 | 109/109 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.201 | 0.367 | 0.416 | 106/109 | 0 | 0 |
| AprilTag, WPILib defaults | 3.865 | 4.902 | 5.187 | 71/109 | 0 | 0 |
| AprilTag, decimate 2 | 5.479 | 7.058 | 7.403 | 109/109 | 0 | 0 |
| AprilTag, full resolution | 18.484 | 23.019 | 25.790 | 109/109 | 0 | 0 |
| Locus standard | 3.408 | 5.210 | 5.336 | 109/109 | 0 | 0 |
| Locus, decimate 2 | 1.975 | 3.291 | 3.484 | 106/109 | 0 | 0 |
| OpenCV ArUco | 14.242 | 91.514 | 123.810 | 109/109 | 0 | 0 |
| OpenCV ArUco, subpixel | 14.259 | 91.324 | 123.464 | 109/109 | 0 | 0 |
| ArUco3 full | 3.533 | 16.288 | 25.004 | 109/109 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.434 | 3.911 | 5.779 | 66/109 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.998 | 6.675 | 16.656 | 109/109 | 0 | 0 |
| ArUco Nano | 0.804 | 1.607 | 1.701 | 109/109 | 0 | 0 |
| RapidTag | 2.156 | 3.226 | 3.335 | 97/109 | 0 | 0 |
| RapidTag, subpixel | 2.181 | 3.204 | 3.283 | 97/109 | 0 | 0 |

### blur, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.311 | 0.632 | 0.704 | 96/96 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.451 | 0.906 | 0.958 | 96/96 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.260 | 0.574 | 0.630 | 96/96 | 0 | 0 |
| AprilTag, WPILib defaults | 3.395 | 4.370 | 4.828 | 82/96 | 0 | 0 |
| AprilTag, decimate 2 | 4.791 | 6.063 | 6.297 | 96/96 | 0 | 0 |
| AprilTag, full resolution | 16.815 | 21.561 | 22.989 | 96/96 | 0 | 0 |
| Locus standard | 3.664 | 4.743 | 5.004 | 96/96 | 0 | 0 |
| Locus, decimate 2 | 2.296 | 2.951 | 3.114 | 96/96 | 0 | 0 |
| OpenCV ArUco | 6.189 | 45.761 | 170.524 | 96/96 | 0 | 0 |
| OpenCV ArUco, subpixel | 6.193 | 45.616 | 169.917 | 96/96 | 0 | 0 |
| ArUco3 full | 3.048 | 7.797 | 14.801 | 96/96 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.341 | 2.758 | 3.473 | 69/96 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.874 | 4.560 | 5.504 | 94/96 | 0 | 0 |
| ArUco Nano | 0.917 | 1.529 | 1.805 | 96/96 | 0 | 0 |
| RapidTag | 1.964 | 2.738 | 2.876 | 84/96 | 0 | 0 |
| RapidTag, subpixel | 1.972 | 2.712 | 2.865 | 84/96 | 0 | 0 |

### motion-blur, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.304 | 0.586 | 0.682 | 110/110 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.463 | 0.831 | 0.925 | 110/110 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.279 | 0.516 | 0.735 | 110/110 | 0 | 0 |
| AprilTag, WPILib defaults | 3.669 | 4.804 | 4.930 | 86/110 | 0 | 0 |
| AprilTag, decimate 2 | 4.808 | 6.322 | 6.679 | 110/110 | 0 | 0 |
| AprilTag, full resolution | 16.925 | 22.267 | 24.020 | 110/110 | 0 | 0 |
| Locus standard | 3.686 | 4.958 | 5.114 | 107/110 | 0 | 0 |
| Locus, decimate 2 | 2.272 | 3.009 | 3.162 | 106/110 | 0 | 0 |
| OpenCV ArUco | 8.503 | 35.452 | 64.181 | 101/110 | 0 | 0 |
| OpenCV ArUco, subpixel | 8.590 | 35.385 | 64.029 | 101/110 | 0 | 0 |
| ArUco3 full | 3.492 | 8.874 | 12.596 | 103/110 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.453 | 2.459 | 2.627 | 67/110 | 0 | 0 |
| ArUco3, minimum ≈32 px | 2.101 | 5.277 | 7.123 | 104/110 | 0 | 0 |
| ArUco Nano | 0.918 | 1.483 | 1.580 | 108/110 | 0 | 0 |
| RapidTag | 2.116 | 2.882 | 3.066 | 95/110 | 0 | 0 |
| RapidTag, subpixel | 2.154 | 2.856 | 3.063 | 95/110 | 0 | 0 |

### noise, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.263 | 0.631 | 0.737 | 110/110 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.447 | 1.143 | 1.216 | 110/110 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.235 | 0.498 | 0.595 | 110/110 | 0 | 0 |
| AprilTag, WPILib defaults | 3.927 | 5.165 | 5.479 | 94/110 | 0 | 0 |
| AprilTag, decimate 2 | 6.067 | 7.849 | 8.410 | 110/110 | 0 | 0 |
| AprilTag, full resolution | 22.356 | 28.622 | 29.826 | 110/110 | 0 | 0 |
| Locus standard | 3.104 | 5.157 | 5.361 | 110/110 | 0 | 0 |
| Locus, decimate 2 | 2.077 | 2.934 | 3.178 | 110/110 | 0 | 0 |
| OpenCV ArUco | 16.461 | 24.039 | 42.080 | 109/110 | 0 | 0 |
| OpenCV ArUco, subpixel | 16.557 | 23.842 | 42.537 | 109/110 | 0 | 0 |
| ArUco3 full | 10.158 | 13.527 | 14.531 | 106/110 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.080 | 3.548 | 4.419 | 72/110 | 0 | 0 |
| ArUco3, minimum ≈32 px | 2.593 | 4.473 | 9.765 | 103/110 | 0 | 0 |
| ArUco Nano | 2.175 | 2.627 | 2.704 | 110/110 | 0 | 0 |
| RapidTag | 6.691 | 9.592 | 10.069 | 87/110 | 1 | 0 |
| RapidTag, subpixel | 6.765 | 9.582 | 10.081 | 87/110 | 1 | 0 |

### uneven-light, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.217 | 0.444 | 0.528 | 98/98 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.347 | 0.700 | 0.755 | 98/98 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.190 | 0.367 | 0.419 | 98/98 | 0 | 0 |
| AprilTag, WPILib defaults | 3.687 | 4.853 | 5.676 | 82/98 | 0 | 0 |
| AprilTag, decimate 2 | 5.278 | 6.887 | 7.225 | 98/98 | 0 | 0 |
| AprilTag, full resolution | 20.268 | 24.406 | 25.526 | 98/98 | 0 | 0 |
| Locus standard | 3.562 | 4.617 | 4.728 | 96/98 | 0 | 0 |
| Locus, decimate 2 | 1.838 | 2.566 | 2.778 | 95/98 | 0 | 0 |
| OpenCV ArUco | 11.154 | 110.682 | 144.465 | 98/98 | 0 | 0 |
| OpenCV ArUco, subpixel | 11.145 | 113.680 | 144.568 | 98/98 | 0 | 0 |
| ArUco3 full | 3.213 | 8.553 | 29.078 | 98/98 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.490 | 2.510 | 3.065 | 73/98 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.842 | 4.021 | 12.107 | 98/98 | 0 | 0 |
| ArUco Nano | 0.820 | 1.293 | 1.379 | 98/98 | 0 | 0 |
| RapidTag | 2.015 | 2.587 | 2.763 | 94/98 | 0 | 0 |
| RapidTag, subpixel | 1.988 | 2.608 | 2.778 | 94/98 | 0 | 0 |

### low-contrast, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.132 | 0.205 | 0.244 | 96/96 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.185 | 0.282 | 0.312 | 96/96 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.124 | 0.196 | 0.219 | 95/96 | 0 | 0 |
| AprilTag, WPILib defaults | 4.778 | 5.741 | 6.047 | 70/96 | 0 | 0 |
| AprilTag, decimate 2 | 6.066 | 7.442 | 7.947 | 88/96 | 0 | 0 |
| AprilTag, full resolution | 22.463 | 26.992 | 27.677 | 92/96 | 0 | 0 |
| Locus standard | 3.160 | 3.640 | 3.836 | 91/96 | 0 | 0 |
| Locus, decimate 2 | 1.684 | 2.115 | 2.223 | 92/96 | 0 | 0 |
| OpenCV ArUco | 3.648 | 6.482 | 7.340 | 96/96 | 0 | 0 |
| OpenCV ArUco, subpixel | 3.668 | 6.469 | 7.349 | 96/96 | 0 | 0 |
| ArUco3 full | 2.050 | 3.836 | 4.540 | 96/96 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.058 | 1.939 | 2.148 | 65/96 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.392 | 4.181 | 4.845 | 87/96 | 0 | 0 |
| ArUco Nano | 0.546 | 0.912 | 1.043 | 94/96 | 0 | 0 |
| RapidTag | 1.096 | 1.396 | 1.527 | 47/96 | 0 | 0 |
| RapidTag, subpixel | 1.110 | 1.422 | 1.497 | 47/96 | 0 | 0 |

### underexposed, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.210 | 0.416 | 0.501 | 110/110 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.357 | 0.650 | 0.714 | 110/110 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.187 | 0.337 | 0.376 | 109/110 | 0 | 0 |
| AprilTag, WPILib defaults | 3.904 | 5.150 | 6.117 | 96/110 | 0 | 0 |
| AprilTag, decimate 2 | 5.602 | 7.022 | 7.396 | 110/110 | 0 | 0 |
| AprilTag, full resolution | 20.743 | 25.656 | 27.126 | 110/110 | 0 | 0 |
| Locus standard | 4.107 | 4.801 | 5.562 | 79/110 | 0 | 0 |
| Locus, decimate 2 | 2.019 | 2.411 | 2.496 | 89/110 | 0 | 0 |
| OpenCV ArUco | 9.109 | 58.065 | 114.312 | 110/110 | 0 | 0 |
| OpenCV ArUco, subpixel | 9.148 | 58.148 | 114.429 | 110/110 | 0 | 0 |
| ArUco3 full | 3.311 | 12.055 | 13.168 | 110/110 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.382 | 3.208 | 3.666 | 78/110 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.897 | 5.175 | 8.675 | 110/110 | 0 | 0 |
| ArUco Nano | 0.927 | 1.342 | 1.527 | 110/110 | 0 | 0 |
| RapidTag | 1.700 | 2.676 | 2.838 | 110/110 | 0 | 0 |
| RapidTag, subpixel | 1.717 | 2.699 | 2.844 | 110/110 | 0 | 0 |

### overexposed, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.247 | 0.468 | 0.546 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.421 | 0.834 | 0.895 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.220 | 0.414 | 0.481 | 100/100 | 0 | 0 |
| AprilTag, WPILib defaults | 1.311 | 2.696 | 3.823 | 82/100 | 0 | 0 |
| AprilTag, decimate 2 | 2.287 | 4.027 | 4.729 | 100/100 | 0 | 0 |
| AprilTag, full resolution | 5.140 | 10.971 | 17.856 | 100/100 | 0 | 0 |
| Locus standard | 2.177 | 4.211 | 4.359 | 92/100 | 0 | 0 |
| Locus, decimate 2 | 1.629 | 2.532 | 3.140 | 92/100 | 0 | 0 |
| OpenCV ArUco | 11.249 | 95.716 | 127.805 | 100/100 | 0 | 0 |
| OpenCV ArUco, subpixel | 11.622 | 96.441 | 128.258 | 100/100 | 0 | 0 |
| ArUco3 full | 2.891 | 10.960 | 43.235 | 100/100 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.443 | 2.697 | 3.108 | 67/100 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.901 | 4.548 | 17.404 | 100/100 | 0 | 0 |
| ArUco Nano | 0.725 | 1.480 | 1.565 | 100/100 | 0 | 0 |
| RapidTag | 2.026 | 3.079 | 3.175 | 100/100 | 0 | 0 |
| RapidTag, subpixel | 2.038 | 3.070 | 3.199 | 100/100 | 0 | 0 |

### jpeg, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.248 | 0.465 | 0.553 | 104/104 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.412 | 0.744 | 0.833 | 104/104 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.210 | 0.349 | 0.404 | 104/104 | 0 | 0 |
| AprilTag, WPILib defaults | 1.335 | 2.679 | 2.807 | 92/104 | 0 | 0 |
| AprilTag, decimate 2 | 2.357 | 3.440 | 3.763 | 104/104 | 0 | 0 |
| AprilTag, full resolution | 5.420 | 9.911 | 10.163 | 104/104 | 0 | 0 |
| Locus standard | 2.446 | 4.323 | 4.904 | 104/104 | 0 | 0 |
| Locus, decimate 2 | 1.871 | 2.615 | 2.722 | 104/104 | 0 | 0 |
| OpenCV ArUco | 8.723 | 60.069 | 64.693 | 104/104 | 0 | 0 |
| OpenCV ArUco, subpixel | 8.721 | 59.318 | 64.689 | 104/104 | 0 | 0 |
| ArUco3 full | 3.796 | 14.784 | 21.597 | 103/104 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.540 | 3.322 | 3.648 | 78/104 | 0 | 0 |
| ArUco3, minimum ≈32 px | 2.109 | 6.795 | 11.253 | 102/104 | 0 | 0 |
| ArUco Nano | 0.845 | 1.632 | 1.991 | 104/104 | 0 | 0 |
| RapidTag | 2.465 | 3.213 | 3.315 | 91/104 | 0 | 0 |
| RapidTag, subpixel | 2.470 | 3.213 | 3.359 | 91/104 | 0 | 0 |

### busy-background, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.232 | 0.534 | 0.601 | 102/102 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.374 | 0.806 | 0.876 | 102/102 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.204 | 0.400 | 0.471 | 102/102 | 0 | 0 |
| AprilTag, WPILib defaults | 3.678 | 4.627 | 4.806 | 89/102 | 0 | 0 |
| AprilTag, decimate 2 | 5.272 | 6.873 | 8.230 | 102/102 | 0 | 0 |
| AprilTag, full resolution | 18.523 | 23.961 | 24.467 | 102/102 | 0 | 0 |
| Locus standard | 3.380 | 4.071 | 4.226 | 102/102 | 0 | 0 |
| Locus, decimate 2 | 2.058 | 2.676 | 3.043 | 102/102 | 0 | 0 |
| OpenCV ArUco | 20.826 | 85.256 | 104.467 | 102/102 | 0 | 0 |
| OpenCV ArUco, subpixel | 20.856 | 85.860 | 104.202 | 102/102 | 0 | 0 |
| ArUco3 full | 3.131 | 8.665 | 18.188 | 102/102 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.445 | 2.760 | 4.170 | 71/102 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.853 | 4.987 | 6.577 | 102/102 | 0 | 0 |
| ArUco Nano | 0.793 | 1.152 | 1.442 | 102/102 | 0 | 0 |
| RapidTag | 2.112 | 2.674 | 2.784 | 96/102 | 1 | 0 |
| RapidTag, subpixel | 2.127 | 2.673 | 2.855 | 96/102 | 1 | 0 |

### frame-edge, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.216 | 0.495 | 0.538 | 102/102 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.345 | 0.746 | 0.818 | 102/102 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.187 | 0.392 | 0.427 | 102/102 | 0 | 0 |
| AprilTag, WPILib defaults | 3.586 | 4.905 | 5.987 | 74/102 | 0 | 0 |
| AprilTag, decimate 2 | 5.270 | 6.570 | 7.372 | 102/102 | 0 | 0 |
| AprilTag, full resolution | 18.348 | 24.219 | 25.502 | 102/102 | 0 | 0 |
| Locus standard | 3.360 | 5.170 | 5.406 | 102/102 | 0 | 0 |
| Locus, decimate 2 | 1.945 | 2.631 | 2.771 | 102/102 | 0 | 0 |
| OpenCV ArUco | 9.607 | 82.298 | 113.169 | 97/102 | 0 | 0 |
| OpenCV ArUco, subpixel | 9.663 | 82.434 | 113.071 | 97/102 | 0 | 0 |
| ArUco3 full | 3.191 | 7.759 | 14.715 | 96/102 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.467 | 3.160 | 4.236 | 58/102 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.986 | 6.571 | 12.697 | 96/102 | 0 | 0 |
| ArUco Nano | 0.779 | 1.592 | 1.708 | 102/102 | 0 | 0 |
| RapidTag | 2.130 | 3.186 | 3.410 | 89/102 | 0 | 0 |
| RapidTag, subpixel | 2.141 | 3.206 | 3.326 | 89/102 | 0 | 0 |

### noise-free, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.119 | 0.165 | 0.202 | 98/98 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.173 | 0.242 | 0.277 | 98/98 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.115 | 0.162 | 0.196 | 98/98 | 0 | 0 |
| AprilTag, WPILib defaults | 0.649 | 0.820 | 0.896 | 83/98 | 0 | 0 |
| AprilTag, decimate 2 | 0.686 | 0.847 | 0.902 | 98/98 | 0 | 0 |
| AprilTag, full resolution | 1.772 | 2.132 | 2.208 | 98/98 | 0 | 0 |
| Locus standard | 1.365 | 1.806 | 1.904 | 98/98 | 0 | 0 |
| Locus, decimate 2 | 1.294 | 1.704 | 1.816 | 98/98 | 0 | 0 |
| OpenCV ArUco | 4.328 | 10.222 | 10.474 | 97/98 | 0 | 0 |
| OpenCV ArUco, subpixel | 4.355 | 10.227 | 10.457 | 97/98 | 0 | 0 |
| ArUco3 full | 1.925 | 3.111 | 3.497 | 97/98 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.018 | 2.034 | 2.177 | 71/98 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.235 | 2.320 | 2.485 | 97/98 | 0 | 0 |
| ArUco Nano | 0.460 | 1.081 | 1.192 | 98/98 | 0 | 0 |
| RapidTag | 1.057 | 1.333 | 1.419 | 41/98 | 0 | 0 |
| RapidTag, subpixel | 1.067 | 1.331 | 1.429 | 41/98 | 0 | 0 |

### dark-background, 640x480

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.167 | 0.289 | 0.320 | 96/96 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.268 | 0.489 | 0.625 | 96/96 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.152 | 0.229 | 0.253 | 96/96 | 0 | 0 |
| AprilTag, WPILib defaults | 3.103 | 4.832 | 5.116 | 71/96 | 0 | 0 |
| AprilTag, decimate 2 | 4.097 | 6.449 | 6.850 | 96/96 | 0 | 0 |
| AprilTag, full resolution | 16.955 | 25.498 | 27.387 | 96/96 | 0 | 0 |
| Locus standard | 3.366 | 4.730 | 5.031 | 96/96 | 0 | 0 |
| Locus, decimate 2 | 1.723 | 2.204 | 2.436 | 96/96 | 0 | 0 |
| OpenCV ArUco | 7.405 | 30.248 | 35.232 | 96/96 | 0 | 0 |
| OpenCV ArUco, subpixel | 7.432 | 30.395 | 35.156 | 96/96 | 0 | 0 |
| ArUco3 full | 3.912 | 9.803 | 13.366 | 95/96 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.674 | 4.581 | 5.344 | 48/96 | 0 | 0 |
| ArUco3, minimum ≈32 px | 2.237 | 8.063 | 11.755 | 86/96 | 0 | 0 |
| ArUco Nano | 0.843 | 1.485 | 1.649 | 96/96 | 0 | 0 |
| RapidTag | 1.717 | 2.812 | 3.177 | 13/96 | 0 | 0 |
| RapidTag, subpixel | 1.688 | 2.820 | 3.145 | 13/96 | 0 | 0 |

### empty, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.072 | 0.113 | 0.130 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.131 | 0.184 | 0.222 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.053 | 0.096 | 0.116 | 0/0 | 0 | 0 |
| AprilTag, WPILib defaults | 16.759 | 20.352 | 22.239 | 0/0 | 0 | 0 |
| AprilTag, decimate 2 | 20.655 | 25.475 | 28.460 | 0/0 | 0 | 0 |
| AprilTag, full resolution | 85.545 | 96.615 | 109.960 | 0/0 | 0 | 0 |
| Locus standard | 7.972 | 8.620 | 9.560 | 0/0 | 0 | 0 |
| Locus, decimate 2 | 3.073 | 3.652 | 4.291 | 0/0 | 0 | 0 |
| OpenCV ArUco | 2.566 | 3.822 | 4.550 | 0/0 | 0 | 0 |
| OpenCV ArUco, subpixel | 2.559 | 3.771 | 4.731 | 0/0 | 0 | 0 |
| ArUco3 full | 2.730 | 3.978 | 5.020 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈64 px | 0.831 | 0.941 | 1.036 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈32 px | 0.834 | 0.943 | 1.031 | 0/0 | 0 | 0 |
| ArUco Nano | 0.843 | 1.858 | 2.124 | 0/0 | 0 | 0 |
| RapidTag | 2.233 | 2.828 | 3.193 | 0/0 | 0 | 0 |
| RapidTag, subpixel | 2.226 | 2.878 | 3.216 | 0/0 | 0 | 0 |

### clutter-only, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.475 | 1.171 | 1.322 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.056 | 2.122 | 2.244 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.303 | 0.829 | 0.929 | 0/0 | 0 | 0 |
| AprilTag, WPILib defaults | 10.715 | 17.888 | 18.594 | 0/0 | 0 | 0 |
| AprilTag, decimate 2 | 14.342 | 24.195 | 25.335 | 0/0 | 0 | 0 |
| AprilTag, full resolution | 60.605 | 88.832 | 91.138 | 0/0 | 0 | 0 |
| Locus standard | 8.470 | 16.036 | 16.420 | 0/0 | 0 | 0 |
| Locus, decimate 2 | 4.209 | 5.665 | 5.962 | 0/0 | 0 | 0 |
| OpenCV ArUco | 21.983 | 133.452 | 160.723 | 0/0 | 0 | 0 |
| OpenCV ArUco, subpixel | 22.217 | 134.648 | 160.290 | 0/0 | 0 | 0 |
| ArUco3 full | 8.734 | 17.009 | 34.651 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.804 | 6.604 | 8.433 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.558 | 12.565 | 35.000 | 0/0 | 0 | 0 |
| ArUco Nano | 2.381 | 4.303 | 4.470 | 0/0 | 0 | 0 |
| RapidTag | 6.595 | 10.925 | 11.289 | 0/0 | 1 | 0 |
| RapidTag, subpixel | 6.612 | 10.931 | 11.407 | 0/0 | 1 | 0 |

### decoys, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.252 | 0.352 | 0.409 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.465 | 0.609 | 0.676 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.179 | 0.252 | 0.323 | 0/0 | 0 | 0 |
| AprilTag, WPILib defaults | 14.461 | 18.279 | 20.379 | 0/0 | 0 | 0 |
| AprilTag, decimate 2 | 18.333 | 23.530 | 25.462 | 0/0 | 0 | 0 |
| AprilTag, full resolution | 74.780 | 90.448 | 97.083 | 0/0 | 0 | 0 |
| Locus standard | 7.551 | 8.489 | 8.823 | 0/0 | 0 | 0 |
| Locus, decimate 2 | 3.200 | 3.826 | 4.240 | 0/0 | 0 | 0 |
| OpenCV ArUco | 17.909 | 29.639 | 31.711 | 0/0 | 0 | 0 |
| OpenCV ArUco, subpixel | 17.778 | 29.676 | 33.132 | 0/0 | 0 | 0 |
| ArUco3 full | 13.538 | 18.741 | 19.566 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈64 px | 6.966 | 12.779 | 13.582 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈32 px | 9.994 | 15.773 | 17.277 | 0/0 | 0 | 0 |
| ArUco Nano | 3.441 | 4.627 | 5.081 | 0/0 | 0 | 0 |
| RapidTag | 3.659 | 4.744 | 5.204 | 0/0 | 0 | 0 |
| RapidTag, subpixel | 3.651 | 4.644 | 5.154 | 0/0 | 0 | 0 |

### one-tag, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.165 | 0.234 | 0.258 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.269 | 0.365 | 0.389 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.147 | 0.211 | 0.247 | 50/50 | 0 | 0 |
| AprilTag, WPILib defaults | 14.800 | 17.794 | 18.874 | 50/50 | 0 | 0 |
| AprilTag, decimate 2 | 17.629 | 21.807 | 22.619 | 50/50 | 0 | 0 |
| AprilTag, full resolution | 72.514 | 86.109 | 88.067 | 50/50 | 0 | 0 |
| Locus standard | 8.393 | 8.797 | 9.561 | 50/50 | 0 | 0 |
| Locus, decimate 2 | 3.542 | 3.883 | 4.333 | 50/50 | 0 | 0 |
| OpenCV ArUco | 4.414 | 8.796 | 10.858 | 50/50 | 0 | 0 |
| OpenCV ArUco, subpixel | 4.399 | 8.793 | 11.050 | 50/50 | 0 | 0 |
| ArUco3 full | 3.725 | 5.384 | 7.032 | 50/50 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.507 | 1.970 | 2.812 | 48/50 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.620 | 2.837 | 3.825 | 48/50 | 0 | 0 |
| ArUco Nano | 1.028 | 1.367 | 1.703 | 50/50 | 0 | 0 |
| RapidTag | 2.586 | 2.945 | 3.104 | 26/50 | 0 | 0 |
| RapidTag, subpixel | 2.592 | 2.963 | 3.232 | 26/50 | 0 | 0 |

### many-tags, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.571 | 1.507 | 1.767 | 354/354 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.880 | 2.277 | 2.439 | 354/354 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.497 | 1.104 | 1.230 | 354/354 | 0 | 0 |
| AprilTag, WPILib defaults | 10.930 | 15.351 | 15.991 | 354/354 | 0 | 0 |
| AprilTag, decimate 2 | 14.334 | 18.493 | 19.323 | 354/354 | 0 | 0 |
| AprilTag, full resolution | 55.053 | 73.142 | 77.560 | 354/354 | 0 | 0 |
| Locus standard | 10.167 | 14.025 | 14.992 | 354/354 | 0 | 0 |
| Locus, decimate 2 | 6.181 | 8.170 | 8.588 | 354/354 | 0 | 0 |
| OpenCV ArUco | 24.172 | 106.787 | 109.734 | 354/354 | 0 | 0 |
| OpenCV ArUco, subpixel | 24.234 | 106.911 | 110.315 | 354/354 | 0 | 0 |
| ArUco3 full | 11.504 | 19.026 | 35.827 | 354/354 | 0 | 0 |
| ArUco3, minimum ≈64 px | 5.337 | 7.875 | 8.772 | 335/354 | 0 | 0 |
| ArUco3, minimum ≈32 px | 6.658 | 12.961 | 15.734 | 353/354 | 0 | 0 |
| ArUco Nano | 2.585 | 4.743 | 5.013 | 354/354 | 0 | 0 |
| RapidTag | 5.691 | 9.162 | 9.391 | 299/354 | 0 | 0 |
| RapidTag, subpixel | 5.711 | 9.274 | 9.846 | 299/354 | 0 | 0 |

### small-tags, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.326 | 1.286 | 1.441 | 120/120 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.565 | 2.021 | 2.308 | 120/120 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.254 | 0.904 | 1.084 | 120/120 | 0 | 0 |
| AprilTag, WPILib defaults | 12.693 | 17.334 | 18.914 | 0/120 | 0 | 0 |
| AprilTag, decimate 2 | 17.181 | 21.342 | 23.873 | 120/120 | 0 | 0 |
| AprilTag, full resolution | 62.392 | 85.522 | 100.284 | 120/120 | 0 | 0 |
| Locus standard | 8.636 | 14.807 | 15.652 | 120/120 | 0 | 0 |
| Locus, decimate 2 | 3.870 | 5.760 | 6.713 | 120/120 | 0 | 0 |
| OpenCV ArUco | 12.075 | 106.000 | 128.325 | 120/120 | 0 | 0 |
| OpenCV ArUco, subpixel | 12.081 | 105.952 | 128.129 | 120/120 | 0 | 0 |
| ArUco3 full | 7.544 | 15.304 | 19.747 | 93/120 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.015 | 7.011 | 9.817 | 0/120 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.676 | 11.307 | 12.431 | 92/120 | 0 | 0 |
| ArUco Nano | 1.530 | 4.452 | 4.660 | 120/120 | 0 | 0 |
| RapidTag | 4.257 | 9.184 | 9.448 | 68/120 | 0 | 0 |
| RapidTag, subpixel | 4.291 | 9.282 | 9.815 | 68/120 | 0 | 0 |

### large-tags, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.329 | 1.129 | 1.210 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.583 | 1.802 | 2.016 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.254 | 0.800 | 0.896 | 50/50 | 0 | 0 |
| AprilTag, WPILib defaults | 11.782 | 15.493 | 16.580 | 50/50 | 0 | 0 |
| AprilTag, decimate 2 | 16.310 | 19.842 | 20.834 | 50/50 | 0 | 0 |
| AprilTag, full resolution | 60.981 | 78.335 | 80.424 | 50/50 | 0 | 0 |
| Locus standard | 8.834 | 13.731 | 14.559 | 50/50 | 0 | 0 |
| Locus, decimate 2 | 4.516 | 5.542 | 6.259 | 50/50 | 0 | 0 |
| OpenCV ArUco | 15.910 | 78.852 | 91.462 | 49/50 | 0 | 0 |
| OpenCV ArUco, subpixel | 15.938 | 79.055 | 91.079 | 49/50 | 0 | 0 |
| ArUco3 full | 8.667 | 15.210 | 27.822 | 49/50 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.758 | 4.652 | 5.198 | 49/50 | 0 | 0 |
| ArUco3, minimum ≈32 px | 4.020 | 7.962 | 12.850 | 49/50 | 0 | 0 |
| ArUco Nano | 1.819 | 4.095 | 4.703 | 50/50 | 0 | 0 |
| RapidTag | 4.918 | 8.601 | 9.089 | 46/50 | 0 | 0 |
| RapidTag, subpixel | 4.850 | 8.557 | 9.123 | 46/50 | 0 | 0 |

### steep-angle, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.397 | 1.323 | 1.421 | 112/112 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.670 | 2.099 | 2.186 | 112/112 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.327 | 0.922 | 0.980 | 111/112 | 0 | 0 |
| AprilTag, WPILib defaults | 11.960 | 16.187 | 16.947 | 109/112 | 0 | 0 |
| AprilTag, decimate 2 | 16.713 | 20.678 | 21.640 | 112/112 | 0 | 0 |
| AprilTag, full resolution | 59.120 | 78.184 | 82.541 | 112/112 | 0 | 0 |
| Locus standard | 9.045 | 14.400 | 15.823 | 112/112 | 0 | 0 |
| Locus, decimate 2 | 4.944 | 6.262 | 6.576 | 112/112 | 0 | 0 |
| OpenCV ArUco | 23.505 | 148.807 | 156.585 | 112/112 | 0 | 0 |
| OpenCV ArUco, subpixel | 23.365 | 148.888 | 155.762 | 112/112 | 0 | 0 |
| ArUco3 full | 9.360 | 28.052 | 40.158 | 111/112 | 0 | 0 |
| ArUco3, minimum ≈64 px | 3.405 | 5.379 | 11.639 | 98/112 | 0 | 0 |
| ArUco3, minimum ≈32 px | 4.229 | 18.346 | 37.723 | 110/112 | 0 | 0 |
| ArUco Nano | 1.895 | 4.457 | 4.834 | 112/112 | 0 | 0 |
| RapidTag | 5.195 | 9.234 | 9.861 | 101/112 | 1 | 0 |
| RapidTag, subpixel | 5.246 | 9.228 | 9.533 | 101/112 | 1 | 0 |

### blur, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.447 | 1.491 | 1.636 | 89/89 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.706 | 2.311 | 2.441 | 89/89 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.350 | 1.098 | 1.190 | 89/89 | 0 | 0 |
| AprilTag, WPILib defaults | 11.519 | 17.383 | 18.237 | 89/89 | 0 | 0 |
| AprilTag, decimate 2 | 15.011 | 21.423 | 22.698 | 89/89 | 0 | 0 |
| AprilTag, full resolution | 57.134 | 85.617 | 90.085 | 89/89 | 0 | 0 |
| Locus standard | 9.337 | 12.889 | 13.260 | 89/89 | 0 | 0 |
| Locus, decimate 2 | 4.721 | 6.489 | 7.038 | 89/89 | 0 | 0 |
| OpenCV ArUco | 11.977 | 74.787 | 120.273 | 89/89 | 0 | 0 |
| OpenCV ArUco, subpixel | 11.993 | 74.846 | 119.117 | 89/89 | 0 | 0 |
| ArUco3 full | 8.111 | 14.433 | 24.610 | 89/89 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.837 | 6.611 | 9.305 | 86/89 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.744 | 13.327 | 27.053 | 86/89 | 0 | 0 |
| ArUco Nano | 2.279 | 3.849 | 4.263 | 89/89 | 0 | 0 |
| RapidTag | 4.902 | 7.880 | 8.295 | 77/89 | 0 | 0 |
| RapidTag, subpixel | 4.900 | 7.844 | 8.276 | 77/89 | 0 | 0 |

### motion-blur, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.618 | 1.683 | 1.836 | 94/94 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.222 | 2.559 | 2.744 | 94/94 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.367 | 1.168 | 1.240 | 94/94 | 0 | 0 |
| AprilTag, WPILib defaults | 11.362 | 17.313 | 17.783 | 94/94 | 0 | 0 |
| AprilTag, decimate 2 | 16.220 | 21.527 | 22.305 | 94/94 | 0 | 0 |
| AprilTag, full resolution | 54.603 | 81.003 | 84.700 | 94/94 | 0 | 0 |
| Locus standard | 9.289 | 13.625 | 14.139 | 94/94 | 0 | 0 |
| Locus, decimate 2 | 4.716 | 6.426 | 7.762 | 92/94 | 0 | 0 |
| OpenCV ArUco | 16.436 | 54.701 | 87.097 | 93/94 | 0 | 0 |
| OpenCV ArUco, subpixel | 16.577 | 54.932 | 87.166 | 93/94 | 0 | 0 |
| ArUco3 full | 8.212 | 13.321 | 23.599 | 93/94 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.981 | 5.253 | 8.563 | 92/94 | 0 | 0 |
| ArUco3, minimum ≈32 px | 4.068 | 8.172 | 15.421 | 92/94 | 0 | 0 |
| ArUco Nano | 2.379 | 4.312 | 4.676 | 94/94 | 0 | 0 |
| RapidTag | 6.299 | 8.769 | 8.983 | 83/94 | 0 | 0 |
| RapidTag, subpixel | 6.451 | 8.730 | 9.025 | 83/94 | 0 | 0 |

### noise, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.451 | 1.597 | 1.782 | 104/104 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.018 | 3.264 | 3.742 | 104/104 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.423 | 1.265 | 1.381 | 104/104 | 0 | 0 |
| AprilTag, WPILib defaults | 11.079 | 17.644 | 18.692 | 104/104 | 0 | 0 |
| AprilTag, decimate 2 | 16.407 | 24.879 | 26.119 | 104/104 | 0 | 0 |
| AprilTag, full resolution | 66.728 | 90.289 | 93.369 | 104/104 | 0 | 0 |
| Locus standard | 8.031 | 13.376 | 13.800 | 104/104 | 0 | 0 |
| Locus, decimate 2 | 4.350 | 6.941 | 7.544 | 104/104 | 0 | 0 |
| OpenCV ArUco | 39.182 | 52.101 | 59.434 | 104/104 | 0 | 0 |
| OpenCV ArUco, subpixel | 39.288 | 52.105 | 59.626 | 104/104 | 0 | 0 |
| ArUco3 full | 30.850 | 42.451 | 52.351 | 102/104 | 0 | 0 |
| ArUco3, minimum ≈64 px | 5.579 | 10.334 | 13.642 | 103/104 | 0 | 0 |
| ArUco3, minimum ≈32 px | 6.807 | 15.432 | 21.120 | 100/104 | 0 | 0 |
| ArUco Nano | 5.997 | 7.497 | 7.835 | 104/104 | 0 | 0 |
| RapidTag | 19.336 | 29.194 | 30.156 | 93/104 | 5 | 0 |
| RapidTag, subpixel | 19.310 | 29.160 | 29.778 | 93/104 | 5 | 0 |

### uneven-light, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.412 | 1.196 | 1.315 | 103/103 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.724 | 1.913 | 2.060 | 103/103 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.333 | 0.919 | 1.031 | 103/103 | 0 | 0 |
| AprilTag, WPILib defaults | 12.328 | 16.558 | 17.358 | 103/103 | 0 | 0 |
| AprilTag, decimate 2 | 17.056 | 20.288 | 20.870 | 103/103 | 0 | 0 |
| AprilTag, full resolution | 65.702 | 76.990 | 79.056 | 103/103 | 0 | 0 |
| Locus standard | 9.532 | 12.638 | 12.844 | 102/103 | 0 | 0 |
| Locus, decimate 2 | 4.488 | 5.635 | 6.106 | 101/103 | 0 | 0 |
| OpenCV ArUco | 16.779 | 93.492 | 151.955 | 103/103 | 0 | 0 |
| OpenCV ArUco, subpixel | 16.665 | 95.035 | 151.773 | 103/103 | 0 | 0 |
| ArUco3 full | 8.084 | 13.899 | 23.649 | 103/103 | 0 | 0 |
| ArUco3, minimum ≈64 px | 3.104 | 5.591 | 8.558 | 99/103 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.689 | 10.352 | 15.180 | 102/103 | 0 | 0 |
| ArUco Nano | 1.877 | 3.815 | 4.023 | 103/103 | 0 | 0 |
| RapidTag | 5.459 | 7.827 | 8.413 | 91/103 | 0 | 0 |
| RapidTag, subpixel | 5.443 | 7.756 | 8.157 | 91/103 | 0 | 0 |

### low-contrast, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.224 | 0.317 | 0.381 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.357 | 0.489 | 0.543 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.193 | 0.283 | 0.328 | 100/100 | 0 | 0 |
| AprilTag, WPILib defaults | 15.387 | 17.928 | 19.595 | 95/100 | 0 | 0 |
| AprilTag, decimate 2 | 18.563 | 22.013 | 23.446 | 95/100 | 0 | 0 |
| AprilTag, full resolution | 74.850 | 87.936 | 96.425 | 100/100 | 0 | 0 |
| Locus standard | 8.687 | 9.534 | 10.593 | 93/100 | 0 | 0 |
| Locus, decimate 2 | 3.911 | 4.660 | 4.815 | 93/100 | 0 | 0 |
| OpenCV ArUco | 5.748 | 11.267 | 13.057 | 100/100 | 0 | 0 |
| OpenCV ArUco, subpixel | 5.724 | 11.326 | 13.013 | 100/100 | 0 | 0 |
| ArUco3 full | 4.320 | 6.631 | 10.527 | 100/100 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.970 | 3.173 | 3.365 | 97/100 | 0 | 0 |
| ArUco3, minimum ≈32 px | 2.143 | 3.697 | 4.040 | 99/100 | 0 | 0 |
| ArUco Nano | 1.206 | 1.695 | 1.887 | 100/100 | 0 | 0 |
| RapidTag | 2.891 | 3.493 | 3.818 | 57/100 | 0 | 0 |
| RapidTag, subpixel | 2.868 | 3.544 | 3.845 | 57/100 | 0 | 0 |

### underexposed, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.286 | 1.004 | 1.072 | 98/99 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.555 | 1.644 | 1.719 | 99/99 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.253 | 0.751 | 0.829 | 99/99 | 0 | 0 |
| AprilTag, WPILib defaults | 12.446 | 17.947 | 18.622 | 99/99 | 0 | 0 |
| AprilTag, decimate 2 | 18.486 | 22.306 | 24.322 | 99/99 | 0 | 0 |
| AprilTag, full resolution | 67.421 | 87.905 | 90.612 | 99/99 | 0 | 0 |
| Locus standard | 10.205 | 12.992 | 14.484 | 69/99 | 0 | 0 |
| Locus, decimate 2 | 4.386 | 5.378 | 5.673 | 79/99 | 0 | 0 |
| OpenCV ArUco | 14.108 | 53.178 | 90.151 | 99/99 | 0 | 0 |
| OpenCV ArUco, subpixel | 14.067 | 53.266 | 90.001 | 99/99 | 0 | 0 |
| ArUco3 full | 7.417 | 14.787 | 51.321 | 99/99 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.674 | 5.497 | 9.374 | 95/99 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.635 | 14.915 | 16.659 | 99/99 | 0 | 0 |
| ArUco Nano | 1.970 | 3.159 | 3.481 | 99/99 | 0 | 0 |
| RapidTag | 3.863 | 7.542 | 7.856 | 99/99 | 0 | 0 |
| RapidTag, subpixel | 3.823 | 7.565 | 7.936 | 99/99 | 0 | 0 |

### overexposed, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.314 | 1.250 | 1.375 | 112/112 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.609 | 2.365 | 2.596 | 112/112 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.254 | 0.995 | 1.174 | 112/112 | 0 | 0 |
| AprilTag, WPILib defaults | 2.199 | 7.608 | 8.503 | 112/112 | 0 | 0 |
| AprilTag, decimate 2 | 3.861 | 10.533 | 12.256 | 112/112 | 0 | 0 |
| AprilTag, full resolution | 9.303 | 32.401 | 33.723 | 112/112 | 0 | 0 |
| Locus standard | 4.245 | 10.144 | 12.932 | 107/112 | 0 | 0 |
| Locus, decimate 2 | 3.513 | 5.194 | 6.018 | 112/112 | 0 | 0 |
| OpenCV ArUco | 17.272 | 138.559 | 207.771 | 112/112 | 0 | 0 |
| OpenCV ArUco, subpixel | 17.159 | 138.919 | 208.233 | 112/112 | 0 | 0 |
| ArUco3 full | 6.467 | 12.762 | 25.671 | 112/112 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.664 | 4.656 | 4.999 | 111/112 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.610 | 7.248 | 18.613 | 112/112 | 0 | 0 |
| ArUco Nano | 1.296 | 3.912 | 4.574 | 112/112 | 0 | 0 |
| RapidTag | 3.539 | 8.860 | 9.677 | 112/112 | 0 | 0 |
| RapidTag, subpixel | 3.554 | 8.921 | 9.938 | 112/112 | 0 | 0 |

### jpeg, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.339 | 1.431 | 1.701 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.594 | 2.290 | 2.759 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.265 | 0.997 | 1.208 | 100/100 | 0 | 0 |
| AprilTag, WPILib defaults | 2.988 | 7.634 | 8.599 | 100/100 | 0 | 0 |
| AprilTag, decimate 2 | 3.956 | 10.213 | 10.917 | 100/100 | 0 | 0 |
| AprilTag, full resolution | 9.983 | 30.660 | 31.702 | 100/100 | 0 | 0 |
| Locus standard | 4.987 | 11.297 | 11.634 | 100/100 | 0 | 0 |
| Locus, decimate 2 | 3.539 | 5.594 | 6.447 | 100/100 | 0 | 0 |
| OpenCV ArUco | 16.844 | 86.336 | 168.940 | 100/100 | 0 | 0 |
| OpenCV ArUco, subpixel | 16.821 | 86.727 | 169.438 | 100/100 | 0 | 0 |
| ArUco3 full | 9.260 | 26.604 | 75.750 | 100/100 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.960 | 5.023 | 9.273 | 97/100 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.945 | 13.008 | 29.262 | 99/100 | 0 | 0 |
| ArUco Nano | 1.572 | 4.441 | 4.914 | 100/100 | 0 | 0 |
| RapidTag | 5.023 | 9.163 | 9.880 | 83/100 | 0 | 0 |
| RapidTag, subpixel | 5.071 | 9.140 | 11.164 | 83/100 | 0 | 0 |

### busy-background, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.359 | 1.313 | 1.472 | 87/87 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.634 | 2.046 | 2.206 | 87/87 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.288 | 0.941 | 1.012 | 87/87 | 0 | 0 |
| AprilTag, WPILib defaults | 12.023 | 16.227 | 17.954 | 87/87 | 0 | 0 |
| AprilTag, decimate 2 | 16.830 | 21.233 | 22.246 | 87/87 | 1 | 0 |
| AprilTag, full resolution | 61.002 | 82.218 | 84.982 | 87/87 | 0 | 0 |
| Locus standard | 8.863 | 10.224 | 11.023 | 87/87 | 0 | 0 |
| Locus, decimate 2 | 4.494 | 6.032 | 6.359 | 86/87 | 0 | 0 |
| OpenCV ArUco | 36.389 | 136.492 | 198.759 | 87/87 | 0 | 0 |
| OpenCV ArUco, subpixel | 36.463 | 136.096 | 199.393 | 87/87 | 0 | 0 |
| ArUco3 full | 8.921 | 24.132 | 43.044 | 87/87 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.897 | 7.933 | 10.686 | 83/87 | 0 | 0 |
| ArUco3, minimum ≈32 px | 4.477 | 14.971 | 21.824 | 86/87 | 0 | 0 |
| ArUco Nano | 1.804 | 2.829 | 3.139 | 87/87 | 0 | 0 |
| RapidTag | 4.915 | 7.787 | 8.270 | 74/87 | 0 | 0 |
| RapidTag, subpixel | 4.926 | 7.821 | 8.357 | 74/87 | 0 | 0 |

### frame-edge, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.363 | 1.218 | 1.332 | 105/105 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.640 | 1.965 | 2.099 | 105/105 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.303 | 0.899 | 1.000 | 105/105 | 0 | 0 |
| AprilTag, WPILib defaults | 12.159 | 16.212 | 16.766 | 103/105 | 0 | 0 |
| AprilTag, decimate 2 | 16.566 | 20.003 | 21.531 | 105/105 | 0 | 0 |
| AprilTag, full resolution | 59.486 | 78.890 | 84.275 | 105/105 | 0 | 0 |
| Locus standard | 8.768 | 14.573 | 14.983 | 105/105 | 0 | 0 |
| Locus, decimate 2 | 4.370 | 5.893 | 6.140 | 105/105 | 0 | 0 |
| OpenCV ArUco | 15.237 | 97.236 | 173.190 | 96/105 | 0 | 0 |
| OpenCV ArUco, subpixel | 15.179 | 97.495 | 173.479 | 96/105 | 0 | 0 |
| ArUco3 full | 8.631 | 17.277 | 40.422 | 95/105 | 0 | 0 |
| ArUco3, minimum ≈64 px | 3.067 | 6.976 | 11.964 | 79/105 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.910 | 10.725 | 12.730 | 94/105 | 0 | 0 |
| ArUco Nano | 1.909 | 4.749 | 5.191 | 105/105 | 0 | 0 |
| RapidTag | 4.821 | 9.310 | 9.511 | 81/105 | 2 | 0 |
| RapidTag, subpixel | 4.847 | 9.297 | 9.553 | 81/105 | 2 | 0 |

### noise-free, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.193 | 0.272 | 0.308 | 104/104 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.333 | 0.442 | 0.489 | 104/104 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.165 | 0.234 | 0.269 | 104/104 | 0 | 0 |
| AprilTag, WPILib defaults | 1.606 | 2.006 | 2.141 | 104/104 | 0 | 0 |
| AprilTag, decimate 2 | 1.586 | 2.066 | 2.165 | 104/104 | 0 | 0 |
| AprilTag, full resolution | 4.671 | 5.538 | 5.884 | 104/104 | 0 | 0 |
| Locus standard | 2.823 | 3.528 | 3.714 | 104/104 | 0 | 0 |
| Locus, decimate 2 | 2.566 | 3.251 | 3.441 | 104/104 | 0 | 0 |
| OpenCV ArUco | 6.540 | 13.675 | 14.094 | 104/104 | 0 | 0 |
| OpenCV ArUco, subpixel | 6.577 | 13.759 | 14.186 | 104/104 | 0 | 0 |
| ArUco3 full | 4.443 | 7.830 | 8.505 | 104/104 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.874 | 2.954 | 3.415 | 100/104 | 0 | 0 |
| ArUco3, minimum ≈32 px | 2.119 | 3.717 | 6.556 | 103/104 | 0 | 0 |
| ArUco Nano | 1.014 | 1.916 | 2.671 | 104/104 | 0 | 0 |
| RapidTag | 2.757 | 3.301 | 3.537 | 39/104 | 0 | 0 |
| RapidTag, subpixel | 2.757 | 3.292 | 3.518 | 39/104 | 0 | 0 |

### dark-background, 1280x720

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.273 | 0.672 | 0.774 | 105/105 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.469 | 1.176 | 1.285 | 105/105 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.232 | 0.421 | 0.479 | 105/105 | 0 | 0 |
| AprilTag, WPILib defaults | 11.703 | 17.063 | 18.272 | 91/105 | 0 | 0 |
| AprilTag, decimate 2 | 14.628 | 20.286 | 21.733 | 105/105 | 0 | 0 |
| AprilTag, full resolution | 60.298 | 83.457 | 89.927 | 105/105 | 0 | 0 |
| Locus standard | 9.466 | 13.432 | 14.055 | 105/105 | 0 | 0 |
| Locus, decimate 2 | 4.045 | 5.190 | 5.598 | 105/105 | 0 | 0 |
| OpenCV ArUco | 8.136 | 26.333 | 44.526 | 105/105 | 0 | 0 |
| OpenCV ArUco, subpixel | 8.137 | 26.490 | 44.365 | 105/105 | 0 | 0 |
| ArUco3 full | 6.844 | 17.807 | 28.836 | 104/105 | 0 | 0 |
| ArUco3, minimum ≈64 px | 3.117 | 10.794 | 16.067 | 80/105 | 0 | 0 |
| ArUco3, minimum ≈32 px | 3.763 | 13.971 | 26.256 | 92/105 | 0 | 0 |
| ArUco Nano | 1.621 | 3.830 | 4.130 | 105/105 | 0 | 0 |
| RapidTag | 3.403 | 7.764 | 8.488 | 7/105 | 0 | 0 |
| RapidTag, subpixel | 3.419 | 7.852 | 8.578 | 7/105 | 0 | 0 |

### empty, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.151 | 0.233 | 0.283 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.284 | 0.404 | 0.499 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.096 | 0.151 | 0.181 | 0/0 | 0 | 0 |
| AprilTag, WPILib defaults | 42.496 | 47.276 | 48.488 | 0/0 | 0 | 0 |
| AprilTag, decimate 2 | 51.862 | 58.939 | 61.263 | 0/0 | 0 | 0 |
| AprilTag, full resolution | 208.453 | 220.354 | 244.504 | 0/0 | 0 | 0 |
| Locus standard | 17.801 | 19.243 | 20.881 | 0/0 | 0 | 0 |
| Locus, decimate 2 | 6.871 | 8.251 | 9.207 | 0/0 | 0 | 0 |
| OpenCV ArUco | 5.889 | 9.989 | 10.974 | 0/0 | 0 | 0 |
| OpenCV ArUco, subpixel | 5.877 | 9.714 | 10.768 | 0/0 | 0 | 0 |
| ArUco3 full | 6.264 | 10.363 | 11.331 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈64 px | 1.868 | 2.063 | 2.271 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈32 px | 1.873 | 2.060 | 2.272 | 0/0 | 0 | 0 |
| ArUco Nano | 1.978 | 4.585 | 4.827 | 0/0 | 0 | 0 |
| RapidTag | 5.127 | 6.680 | 7.464 | 0/0 | 0 | 0 |
| RapidTag, subpixel | 5.109 | 6.721 | 7.159 | 0/0 | 0 | 0 |

### clutter-only, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.441 | 2.655 | 2.817 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.937 | 4.525 | 4.706 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.287 | 1.845 | 1.961 | 0/0 | 0 | 0 |
| AprilTag, WPILib defaults | 28.307 | 45.029 | 45.678 | 0/0 | 0 | 0 |
| AprilTag, decimate 2 | 41.382 | 56.401 | 60.566 | 0/0 | 0 | 0 |
| AprilTag, full resolution | 148.037 | 213.566 | 217.831 | 0/0 | 0 | 0 |
| Locus standard | 18.654 | 35.863 | 36.908 | 0/0 | 0 | 0 |
| Locus, decimate 2 | 8.400 | 11.913 | 12.820 | 0/0 | 0 | 0 |
| OpenCV ArUco | 50.446 | 134.974 | 198.454 | 0/0 | 0 | 0 |
| OpenCV ArUco, subpixel | 50.754 | 134.896 | 198.595 | 0/0 | 0 | 0 |
| ArUco3 full | 18.404 | 26.659 | 34.576 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈64 px | 6.084 | 7.945 | 11.582 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈32 px | 7.782 | 11.623 | 27.775 | 0/0 | 0 | 0 |
| ArUco Nano | 4.740 | 9.767 | 10.787 | 0/0 | 0 | 0 |
| RapidTag | 10.205 | 23.551 | 24.603 | 0/0 | 1 | 0 |
| RapidTag, subpixel | 10.162 | 23.407 | 24.892 | 0/0 | 1 | 0 |

### decoys, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.385 | 0.556 | 0.625 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.725 | 1.012 | 1.093 | 0/0 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.268 | 0.399 | 0.459 | 0/0 | 0 | 0 |
| AprilTag, WPILib defaults | 37.846 | 44.799 | 46.983 | 0/0 | 0 | 0 |
| AprilTag, decimate 2 | 46.751 | 55.171 | 58.098 | 0/0 | 0 | 0 |
| AprilTag, full resolution | 188.528 | 210.025 | 217.513 | 0/0 | 0 | 0 |
| Locus standard | 18.099 | 19.108 | 22.920 | 0/0 | 0 | 0 |
| Locus, decimate 2 | 7.162 | 8.143 | 8.643 | 0/0 | 0 | 0 |
| OpenCV ArUco | 19.593 | 32.698 | 33.682 | 0/0 | 0 | 0 |
| OpenCV ArUco, subpixel | 19.619 | 32.499 | 33.246 | 0/0 | 0 | 0 |
| ArUco3 full | 18.177 | 27.333 | 29.594 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈64 px | 8.958 | 16.691 | 17.174 | 0/0 | 0 | 0 |
| ArUco3, minimum ≈32 px | 12.099 | 18.997 | 19.901 | 0/0 | 0 | 0 |
| ArUco Nano | 4.945 | 7.809 | 8.557 | 0/0 | 0 | 0 |
| RapidTag | 6.716 | 9.133 | 9.787 | 0/0 | 0 | 0 |
| RapidTag, subpixel | 6.711 | 9.142 | 9.601 | 0/0 | 0 | 0 |

### one-tag, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.272 | 0.373 | 0.423 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.480 | 0.624 | 0.697 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.205 | 0.286 | 0.351 | 50/50 | 0 | 0 |
| AprilTag, WPILib defaults | 36.680 | 44.442 | 46.716 | 50/50 | 0 | 0 |
| AprilTag, decimate 2 | 42.371 | 53.738 | 60.925 | 50/50 | 0 | 0 |
| AprilTag, full resolution | 177.197 | 210.375 | 232.667 | 50/50 | 0 | 0 |
| Locus standard | 18.652 | 19.652 | 21.359 | 50/50 | 0 | 0 |
| Locus, decimate 2 | 7.568 | 8.070 | 9.033 | 50/50 | 0 | 0 |
| OpenCV ArUco | 8.349 | 11.749 | 13.957 | 50/50 | 0 | 0 |
| OpenCV ArUco, subpixel | 8.289 | 12.032 | 13.966 | 50/50 | 0 | 0 |
| ArUco3 full | 7.482 | 11.360 | 12.094 | 50/50 | 0 | 0 |
| ArUco3, minimum ≈64 px | 2.569 | 3.903 | 4.233 | 50/50 | 0 | 0 |
| ArUco3, minimum ≈32 px | 2.721 | 5.420 | 7.807 | 49/50 | 0 | 0 |
| ArUco Nano | 2.059 | 2.538 | 2.808 | 50/50 | 0 | 0 |
| RapidTag | 5.630 | 6.449 | 6.905 | 28/50 | 0 | 0 |
| RapidTag, subpixel | 5.567 | 6.423 | 6.856 | 28/50 | 0 | 0 |

### many-tags, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.821 | 2.544 | 2.797 | 303/303 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.335 | 4.078 | 4.385 | 303/303 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.689 | 1.900 | 2.084 | 303/303 | 0 | 0 |
| AprilTag, WPILib defaults | 26.115 | 34.759 | 40.411 | 303/303 | 0 | 0 |
| AprilTag, decimate 2 | 33.509 | 43.633 | 48.493 | 303/303 | 0 | 0 |
| AprilTag, full resolution | 132.043 | 181.166 | 194.030 | 303/303 | 0 | 0 |
| Locus standard | 21.099 | 32.109 | 32.710 | 303/303 | 0 | 0 |
| Locus, decimate 2 | 10.870 | 14.153 | 14.913 | 303/303 | 0 | 0 |
| OpenCV ArUco | 28.730 | 77.117 | 159.984 | 300/303 | 0 | 0 |
| OpenCV ArUco, subpixel | 29.151 | 77.430 | 159.534 | 300/303 | 0 | 0 |
| ArUco3 full | 21.655 | 31.947 | 45.700 | 300/303 | 0 | 0 |
| ArUco3, minimum ≈64 px | 8.067 | 13.005 | 14.945 | 300/303 | 0 | 0 |
| ArUco3, minimum ≈32 px | 10.715 | 18.802 | 20.877 | 298/303 | 0 | 0 |
| ArUco Nano | 4.844 | 10.100 | 10.693 | 303/303 | 0 | 0 |
| RapidTag | 10.718 | 20.356 | 21.364 | 219/303 | 0 | 0 |
| RapidTag, subpixel | 10.833 | 20.280 | 20.748 | 219/303 | 0 | 0 |

### small-tags, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.512 | 2.804 | 2.965 | 122/122 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.947 | 4.493 | 4.669 | 122/122 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.368 | 1.895 | 2.033 | 121/122 | 0 | 0 |
| AprilTag, WPILib defaults | 29.781 | 42.861 | 45.137 | 0/122 | 0 | 0 |
| AprilTag, decimate 2 | 40.473 | 54.169 | 55.407 | 122/122 | 0 | 0 |
| AprilTag, full resolution | 149.211 | 205.138 | 216.897 | 122/122 | 0 | 0 |
| Locus standard | 18.966 | 32.817 | 33.424 | 122/122 | 0 | 0 |
| Locus, decimate 2 | 8.193 | 11.992 | 12.816 | 122/122 | 0 | 0 |
| OpenCV ArUco | 23.756 | 91.344 | 129.726 | 122/122 | 0 | 0 |
| OpenCV ArUco, subpixel | 23.662 | 91.783 | 129.773 | 122/122 | 0 | 0 |
| ArUco3 full | 17.713 | 24.940 | 32.135 | 103/122 | 0 | 0 |
| ArUco3, minimum ≈64 px | 6.161 | 10.577 | 11.669 | 0/122 | 0 | 0 |
| ArUco3, minimum ≈32 px | 8.359 | 13.223 | 17.295 | 99/122 | 0 | 0 |
| ArUco Nano | 3.016 | 9.964 | 10.223 | 122/122 | 0 | 0 |
| RapidTag | 9.262 | 20.681 | 21.119 | 81/122 | 1 | 0 |
| RapidTag, subpixel | 9.268 | 20.633 | 21.127 | 81/122 | 1 | 0 |

### large-tags, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.476 | 2.254 | 2.439 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.923 | 3.740 | 4.028 | 50/50 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.351 | 1.706 | 1.842 | 50/50 | 0 | 0 |
| AprilTag, WPILib defaults | 29.357 | 37.321 | 43.025 | 50/50 | 0 | 0 |
| AprilTag, decimate 2 | 37.633 | 45.784 | 52.144 | 50/50 | 1 | 0 |
| AprilTag, full resolution | 146.755 | 184.473 | 213.379 | 50/50 | 0 | 0 |
| Locus standard | 19.442 | 31.280 | 31.845 | 50/50 | 0 | 0 |
| Locus, decimate 2 | 8.640 | 11.459 | 12.217 | 50/50 | 0 | 0 |
| OpenCV ArUco | 24.051 | 77.618 | 92.453 | 50/50 | 0 | 0 |
| OpenCV ArUco, subpixel | 24.043 | 78.154 | 92.423 | 50/50 | 0 | 0 |
| ArUco3 full | 17.796 | 30.169 | 33.935 | 50/50 | 0 | 0 |
| ArUco3, minimum ≈64 px | 5.988 | 13.531 | 16.826 | 50/50 | 0 | 0 |
| ArUco3, minimum ≈32 px | 8.039 | 16.357 | 19.054 | 50/50 | 0 | 0 |
| ArUco Nano | 2.878 | 8.962 | 9.440 | 50/50 | 0 | 0 |
| RapidTag | 8.672 | 19.159 | 19.817 | 46/50 | 0 | 0 |
| RapidTag, subpixel | 8.654 | 19.042 | 19.916 | 46/50 | 0 | 0 |

### steep-angle, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.549 | 2.670 | 3.049 | 96/96 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.001 | 4.359 | 4.694 | 96/96 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.398 | 1.879 | 2.144 | 96/96 | 0 | 0 |
| AprilTag, WPILib defaults | 29.081 | 41.097 | 42.331 | 88/96 | 0 | 0 |
| AprilTag, decimate 2 | 37.831 | 50.141 | 51.394 | 96/96 | 0 | 0 |
| AprilTag, full resolution | 142.444 | 199.472 | 205.443 | 96/96 | 1 | 0 |
| Locus standard | 19.087 | 32.362 | 36.291 | 96/96 | 0 | 0 |
| Locus, decimate 2 | 8.782 | 12.175 | 13.685 | 96/96 | 0 | 0 |
| OpenCV ArUco | 25.855 | 129.888 | 173.794 | 96/96 | 0 | 0 |
| OpenCV ArUco, subpixel | 25.738 | 130.062 | 173.363 | 96/96 | 0 | 0 |
| ArUco3 full | 19.193 | 34.572 | 40.326 | 96/96 | 0 | 0 |
| ArUco3, minimum ≈64 px | 7.057 | 14.617 | 26.572 | 96/96 | 0 | 0 |
| ArUco3, minimum ≈32 px | 8.429 | 21.238 | 30.622 | 95/96 | 0 | 0 |
| ArUco Nano | 3.230 | 9.838 | 10.800 | 96/96 | 0 | 0 |
| RapidTag | 9.579 | 20.390 | 20.877 | 83/96 | 1 | 0 |
| RapidTag, subpixel | 9.569 | 20.631 | 21.821 | 83/96 | 1 | 0 |

### blur, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.583 | 3.084 | 3.272 | 97/97 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.027 | 4.776 | 5.062 | 97/97 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.460 | 2.211 | 2.432 | 97/97 | 0 | 0 |
| AprilTag, WPILib defaults | 27.831 | 39.767 | 41.155 | 97/97 | 0 | 0 |
| AprilTag, decimate 2 | 37.052 | 47.346 | 48.893 | 97/97 | 0 | 0 |
| AprilTag, full resolution | 143.893 | 187.220 | 193.436 | 97/97 | 0 | 0 |
| Locus standard | 19.755 | 29.300 | 32.180 | 97/97 | 0 | 0 |
| Locus, decimate 2 | 9.514 | 11.859 | 13.042 | 97/97 | 0 | 0 |
| OpenCV ArUco | 21.284 | 76.735 | 182.258 | 97/97 | 0 | 0 |
| OpenCV ArUco, subpixel | 21.447 | 76.440 | 182.293 | 97/97 | 0 | 0 |
| ArUco3 full | 15.033 | 24.114 | 27.798 | 97/97 | 0 | 0 |
| ArUco3, minimum ≈64 px | 5.263 | 8.711 | 9.681 | 97/97 | 0 | 0 |
| ArUco3, minimum ≈32 px | 6.886 | 11.958 | 20.056 | 96/97 | 0 | 0 |
| ArUco Nano | 4.155 | 9.050 | 9.895 | 97/97 | 0 | 0 |
| RapidTag | 9.273 | 18.332 | 19.024 | 81/97 | 0 | 0 |
| RapidTag, subpixel | 9.367 | 18.417 | 19.906 | 81/97 | 0 | 0 |

### motion-blur, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.567 | 2.989 | 3.128 | 95/95 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.034 | 4.739 | 5.130 | 95/95 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.420 | 2.246 | 2.402 | 95/95 | 0 | 0 |
| AprilTag, WPILib defaults | 30.145 | 42.192 | 43.159 | 95/95 | 0 | 0 |
| AprilTag, decimate 2 | 37.491 | 51.763 | 53.442 | 95/95 | 0 | 0 |
| AprilTag, full resolution | 142.716 | 201.038 | 209.093 | 95/95 | 0 | 0 |
| Locus standard | 19.303 | 27.110 | 31.029 | 95/95 | 0 | 0 |
| Locus, decimate 2 | 9.055 | 12.297 | 13.196 | 92/95 | 0 | 0 |
| OpenCV ArUco | 21.576 | 52.406 | 59.256 | 95/95 | 0 | 0 |
| OpenCV ArUco, subpixel | 21.570 | 52.414 | 59.338 | 95/95 | 0 | 0 |
| ArUco3 full | 15.152 | 24.603 | 26.733 | 95/95 | 0 | 0 |
| ArUco3, minimum ≈64 px | 5.133 | 7.792 | 11.791 | 95/95 | 0 | 0 |
| ArUco3, minimum ≈32 px | 7.500 | 13.240 | 16.491 | 94/95 | 0 | 0 |
| ArUco Nano | 3.446 | 7.715 | 9.203 | 95/95 | 0 | 0 |
| RapidTag | 8.912 | 17.192 | 19.382 | 83/95 | 0 | 0 |
| RapidTag, subpixel | 8.945 | 17.267 | 19.307 | 83/95 | 0 | 0 |

### noise, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.785 | 3.550 | 3.780 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=16 | 2.065 | 7.027 | 8.104 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.728 | 2.580 | 2.772 | 100/100 | 0 | 0 |
| AprilTag, WPILib defaults | 29.778 | 40.768 | 43.464 | 100/100 | 0 | 0 |
| AprilTag, decimate 2 | 42.850 | 57.859 | 60.831 | 100/100 | 0 | 0 |
| AprilTag, full resolution | 172.261 | 211.904 | 221.732 | 100/100 | 0 | 0 |
| Locus standard | 17.445 | 30.826 | 34.820 | 100/100 | 0 | 0 |
| Locus, decimate 2 | 8.861 | 13.582 | 14.257 | 100/100 | 0 | 0 |
| OpenCV ArUco | 76.954 | 104.127 | 165.455 | 100/100 | 0 | 0 |
| OpenCV ArUco, subpixel | 76.902 | 104.029 | 165.195 | 100/100 | 0 | 0 |
| ArUco3 full | 69.820 | 89.966 | 95.249 | 99/100 | 0 | 0 |
| ArUco3, minimum ≈64 px | 10.828 | 16.156 | 16.873 | 100/100 | 0 | 0 |
| ArUco3, minimum ≈32 px | 13.490 | 19.263 | 20.341 | 92/100 | 0 | 0 |
| ArUco Nano | 13.497 | 17.112 | 17.730 | 100/100 | 0 | 0 |
| RapidTag | 41.878 | 62.898 | 68.032 | 80/100 | 2 | 0 |
| RapidTag, subpixel | 41.207 | 62.878 | 67.996 | 80/100 | 2 | 0 |

### uneven-light, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.645 | 2.409 | 2.560 | 101/101 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.152 | 3.915 | 4.090 | 101/101 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.475 | 1.674 | 1.818 | 101/101 | 0 | 0 |
| AprilTag, WPILib defaults | 29.973 | 39.526 | 41.984 | 101/101 | 0 | 0 |
| AprilTag, decimate 2 | 40.894 | 47.630 | 50.615 | 101/101 | 0 | 0 |
| AprilTag, full resolution | 154.029 | 180.792 | 187.913 | 101/101 | 1 | 0 |
| Locus standard | 20.236 | 28.067 | 29.010 | 101/101 | 0 | 0 |
| Locus, decimate 2 | 9.264 | 10.983 | 11.574 | 101/101 | 0 | 0 |
| OpenCV ArUco | 34.177 | 148.059 | 180.108 | 101/101 | 0 | 0 |
| OpenCV ArUco, subpixel | 34.218 | 147.752 | 180.688 | 101/101 | 0 | 0 |
| ArUco3 full | 18.114 | 36.453 | 41.031 | 101/101 | 0 | 0 |
| ArUco3, minimum ≈64 px | 6.062 | 9.345 | 12.889 | 101/101 | 0 | 0 |
| ArUco3, minimum ≈32 px | 8.536 | 17.770 | 31.452 | 100/101 | 0 | 0 |
| ArUco Nano | 3.804 | 7.577 | 8.018 | 101/101 | 0 | 0 |
| RapidTag | 9.965 | 16.968 | 17.907 | 96/101 | 0 | 0 |
| RapidTag, subpixel | 10.006 | 16.881 | 17.650 | 96/101 | 0 | 0 |

### low-contrast, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.380 | 0.532 | 0.621 | 113/113 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.662 | 0.851 | 0.935 | 113/113 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.289 | 0.406 | 0.471 | 113/113 | 0 | 0 |
| AprilTag, WPILib defaults | 35.943 | 41.982 | 44.629 | 108/113 | 0 | 0 |
| AprilTag, decimate 2 | 42.608 | 51.398 | 54.174 | 108/113 | 0 | 0 |
| AprilTag, full resolution | 170.392 | 202.088 | 209.474 | 113/113 | 0 | 0 |
| Locus standard | 19.143 | 20.441 | 21.916 | 104/113 | 0 | 0 |
| Locus, decimate 2 | 8.157 | 8.989 | 9.429 | 104/113 | 0 | 0 |
| OpenCV ArUco | 11.226 | 15.876 | 19.112 | 113/113 | 0 | 0 |
| OpenCV ArUco, subpixel | 11.287 | 15.836 | 19.003 | 113/113 | 0 | 0 |
| ArUco3 full | 9.436 | 13.256 | 14.902 | 113/113 | 0 | 0 |
| ArUco3, minimum ≈64 px | 3.499 | 5.366 | 6.051 | 113/113 | 0 | 0 |
| ArUco3, minimum ≈32 px | 4.188 | 7.155 | 8.097 | 113/113 | 0 | 0 |
| ArUco Nano | 2.388 | 2.976 | 3.187 | 113/113 | 0 | 0 |
| RapidTag | 6.381 | 7.458 | 7.998 | 40/113 | 0 | 0 |
| RapidTag, subpixel | 6.411 | 7.455 | 7.989 | 40/113 | 0 | 0 |

### underexposed, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.427 | 2.057 | 2.585 | 95/95 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.831 | 3.567 | 4.222 | 95/95 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.317 | 1.399 | 1.836 | 93/95 | 0 | 0 |
| AprilTag, WPILib defaults | 34.759 | 42.282 | 43.026 | 95/95 | 0 | 0 |
| AprilTag, decimate 2 | 44.758 | 51.736 | 53.361 | 95/95 | 0 | 0 |
| AprilTag, full resolution | 181.733 | 205.479 | 218.399 | 95/95 | 0 | 0 |
| Locus standard | 21.333 | 27.676 | 29.512 | 81/95 | 0 | 0 |
| Locus, decimate 2 | 8.565 | 11.203 | 12.360 | 93/95 | 0 | 0 |
| OpenCV ArUco | 20.085 | 104.367 | 148.906 | 95/95 | 0 | 0 |
| OpenCV ArUco, subpixel | 20.401 | 104.117 | 149.448 | 95/95 | 0 | 0 |
| ArUco3 full | 14.700 | 27.128 | 31.502 | 95/95 | 0 | 0 |
| ArUco3, minimum ≈64 px | 4.278 | 12.428 | 16.369 | 95/95 | 0 | 0 |
| ArUco3, minimum ≈32 px | 6.257 | 15.660 | 22.891 | 95/95 | 0 | 0 |
| ArUco Nano | 3.606 | 6.741 | 7.773 | 95/95 | 0 | 0 |
| RapidTag | 7.903 | 16.587 | 17.277 | 95/95 | 0 | 0 |
| RapidTag, subpixel | 8.009 | 16.481 | 17.150 | 95/95 | 0 | 0 |

### overexposed, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.576 | 2.863 | 3.088 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.136 | 5.135 | 5.286 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.432 | 2.134 | 2.256 | 100/100 | 0 | 0 |
| AprilTag, WPILib defaults | 9.428 | 18.764 | 21.023 | 100/100 | 0 | 0 |
| AprilTag, decimate 2 | 11.272 | 25.434 | 27.208 | 100/100 | 0 | 0 |
| AprilTag, full resolution | 39.316 | 76.968 | 79.175 | 100/100 | 0 | 0 |
| Locus standard | 8.782 | 26.478 | 29.799 | 87/100 | 0 | 0 |
| Locus, decimate 2 | 6.352 | 10.932 | 11.291 | 87/100 | 0 | 0 |
| OpenCV ArUco | 27.201 | 115.606 | 175.261 | 100/100 | 0 | 0 |
| OpenCV ArUco, subpixel | 26.744 | 115.550 | 174.707 | 100/100 | 0 | 0 |
| ArUco3 full | 17.309 | 26.821 | 37.901 | 100/100 | 0 | 0 |
| ArUco3, minimum ≈64 px | 6.259 | 9.491 | 11.074 | 100/100 | 0 | 0 |
| ArUco3, minimum ≈32 px | 9.025 | 13.236 | 22.599 | 100/100 | 0 | 0 |
| ArUco Nano | 3.279 | 9.499 | 10.220 | 100/100 | 0 | 0 |
| RapidTag | 9.376 | 19.777 | 21.638 | 100/100 | 0 | 0 |
| RapidTag, subpixel | 9.288 | 20.016 | 21.973 | 100/100 | 0 | 0 |

### jpeg, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.625 | 3.352 | 3.589 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=16 | 1.196 | 5.329 | 5.618 | 100/100 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.470 | 2.283 | 2.470 | 100/100 | 0 | 0 |
| AprilTag, WPILib defaults | 7.939 | 17.820 | 20.436 | 100/100 | 0 | 0 |
| AprilTag, decimate 2 | 9.861 | 23.198 | 25.253 | 100/100 | 0 | 0 |
| AprilTag, full resolution | 27.575 | 66.226 | 71.168 | 100/100 | 0 | 0 |
| Locus standard | 11.409 | 24.440 | 25.585 | 100/100 | 0 | 0 |
| Locus, decimate 2 | 7.385 | 12.224 | 12.763 | 100/100 | 0 | 0 |
| OpenCV ArUco | 29.292 | 83.633 | 104.033 | 100/100 | 0 | 0 |
| OpenCV ArUco, subpixel | 29.675 | 83.728 | 104.477 | 100/100 | 0 | 0 |
| ArUco3 full | 22.327 | 35.080 | 40.313 | 100/100 | 0 | 0 |
| ArUco3, minimum ≈64 px | 7.007 | 9.421 | 11.355 | 100/100 | 0 | 0 |
| ArUco3, minimum ≈32 px | 9.017 | 13.943 | 16.355 | 99/100 | 0 | 0 |
| ArUco Nano | 3.732 | 9.941 | 10.331 | 100/100 | 0 | 0 |
| RapidTag | 12.249 | 20.443 | 21.309 | 89/100 | 0 | 0 |
| RapidTag, subpixel | 12.249 | 20.442 | 20.986 | 89/100 | 0 | 0 |

### busy-background, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.534 | 2.709 | 2.869 | 97/97 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.991 | 4.324 | 4.597 | 97/97 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.402 | 1.927 | 2.129 | 97/97 | 0 | 0 |
| AprilTag, WPILib defaults | 29.603 | 39.503 | 41.667 | 97/97 | 0 | 0 |
| AprilTag, decimate 2 | 39.936 | 49.171 | 50.717 | 97/97 | 0 | 0 |
| AprilTag, full resolution | 141.472 | 195.790 | 201.289 | 97/97 | 0 | 0 |
| Locus standard | 19.284 | 21.620 | 22.335 | 97/97 | 0 | 0 |
| Locus, decimate 2 | 9.124 | 12.165 | 12.364 | 97/97 | 0 | 0 |
| OpenCV ArUco | 59.724 | 115.801 | 148.676 | 97/97 | 0 | 0 |
| OpenCV ArUco, subpixel | 59.870 | 115.299 | 148.594 | 97/97 | 0 | 0 |
| ArUco3 full | 20.613 | 44.395 | 54.424 | 97/97 | 0 | 0 |
| ArUco3, minimum ≈64 px | 6.664 | 12.596 | 19.352 | 97/97 | 0 | 0 |
| ArUco3, minimum ≈32 px | 9.909 | 27.993 | 32.493 | 97/97 | 0 | 0 |
| ArUco Nano | 3.195 | 5.868 | 6.149 | 97/97 | 0 | 0 |
| RapidTag | 9.133 | 17.453 | 18.257 | 87/97 | 1 | 0 |
| RapidTag, subpixel | 9.311 | 17.572 | 18.144 | 87/97 | 1 | 0 |

### frame-edge, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.479 | 2.621 | 2.855 | 88/88 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.932 | 4.321 | 4.543 | 88/88 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.359 | 1.821 | 1.974 | 88/88 | 0 | 0 |
| AprilTag, WPILib defaults | 30.507 | 39.567 | 41.210 | 88/88 | 0 | 0 |
| AprilTag, decimate 2 | 38.114 | 47.996 | 50.390 | 88/88 | 0 | 0 |
| AprilTag, full resolution | 144.698 | 197.084 | 203.074 | 88/88 | 0 | 0 |
| Locus standard | 19.072 | 31.858 | 33.065 | 88/88 | 0 | 0 |
| Locus, decimate 2 | 8.267 | 12.123 | 12.728 | 88/88 | 0 | 0 |
| OpenCV ArUco | 25.729 | 91.544 | 123.398 | 84/88 | 0 | 0 |
| OpenCV ArUco, subpixel | 25.684 | 91.709 | 123.295 | 84/88 | 0 | 0 |
| ArUco3 full | 15.687 | 29.122 | 34.202 | 84/88 | 0 | 0 |
| ArUco3, minimum ≈64 px | 4.917 | 9.344 | 14.141 | 82/88 | 0 | 0 |
| ArUco3, minimum ≈32 px | 6.994 | 12.120 | 18.962 | 81/88 | 0 | 0 |
| ArUco Nano | 2.873 | 10.288 | 10.601 | 88/88 | 0 | 0 |
| RapidTag | 8.955 | 20.323 | 22.543 | 73/88 | 1 | 0 |
| RapidTag, subpixel | 9.091 | 20.475 | 21.734 | 73/88 | 1 | 0 |

### noise-free, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.332 | 0.441 | 0.517 | 108/108 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.594 | 0.752 | 0.800 | 108/108 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.237 | 0.334 | 0.389 | 108/108 | 0 | 0 |
| AprilTag, WPILib defaults | 3.412 | 4.038 | 4.173 | 108/108 | 0 | 0 |
| AprilTag, decimate 2 | 3.287 | 4.003 | 4.185 | 108/108 | 0 | 0 |
| AprilTag, full resolution | 10.517 | 11.818 | 12.256 | 108/108 | 0 | 0 |
| Locus standard | 5.628 | 6.653 | 6.852 | 108/108 | 0 | 0 |
| Locus, decimate 2 | 4.969 | 5.882 | 6.202 | 108/108 | 0 | 0 |
| OpenCV ArUco | 10.658 | 14.830 | 19.808 | 108/108 | 0 | 0 |
| OpenCV ArUco, subpixel | 10.656 | 14.814 | 19.731 | 108/108 | 0 | 0 |
| ArUco3 full | 9.383 | 13.050 | 20.101 | 108/108 | 0 | 0 |
| ArUco3, minimum ≈64 px | 3.302 | 5.397 | 6.176 | 108/108 | 0 | 0 |
| ArUco3, minimum ≈32 px | 4.326 | 7.482 | 8.840 | 108/108 | 0 | 0 |
| ArUco Nano | 1.872 | 2.548 | 3.288 | 108/108 | 0 | 0 |
| RapidTag | 5.904 | 6.692 | 7.003 | 45/108 | 0 | 0 |
| RapidTag, subpixel | 5.883 | 6.613 | 6.910 | 45/108 | 0 | 0 |

### dark-background, 1920x1080

| Detector | p50 ms | p95 ms | p99 ms | Found / tags | Wrong IDs | Duplicates |
| --- | --- | --- | --- | --- | --- | --- |
| NAPA (default, min_tag_px=22, split_touching=auto) | 0.439 | 1.455 | 1.640 | 107/107 | 0 | 0 |
| NAPA, min_tag_px=16 | 0.855 | 2.441 | 2.709 | 107/107 | 0 | 0 |
| NAPA, min_tag_px=32 | 0.323 | 0.876 | 0.988 | 107/107 | 0 | 0 |
| AprilTag, WPILib defaults | 30.752 | 41.096 | 45.539 | 89/107 | 0 | 0 |
| AprilTag, decimate 2 | 37.170 | 50.366 | 54.916 | 107/107 | 0 | 0 |
| AprilTag, full resolution | 155.219 | 202.379 | 217.995 | 107/107 | 0 | 0 |
| Locus standard | 20.204 | 31.682 | 39.425 | 107/107 | 0 | 0 |
| Locus, decimate 2 | 8.140 | 10.780 | 11.990 | 107/107 | 0 | 0 |
| OpenCV ArUco | 16.918 | 37.829 | 38.734 | 107/107 | 0 | 0 |
| OpenCV ArUco, subpixel | 16.827 | 37.893 | 38.960 | 107/107 | 0 | 0 |
| ArUco3 full | 15.519 | 29.586 | 33.002 | 105/107 | 0 | 0 |
| ArUco3, minimum ≈64 px | 6.423 | 14.744 | 16.111 | 82/107 | 0 | 0 |
| ArUco3, minimum ≈32 px | 7.682 | 19.374 | 21.473 | 92/107 | 0 | 0 |
| ArUco Nano | 3.154 | 8.495 | 8.925 | 107/107 | 0 | 0 |
| RapidTag | 8.090 | 17.727 | 18.923 | 7/107 | 0 | 0 |
| RapidTag, subpixel | 8.171 | 17.804 | 18.502 | 7/107 | 0 | 0 |

## Provenance and limitations

These synthetic images and single-host timings do not establish real-camera accuracy or a universal winner. This run does not measure multicore throughput or isolated RSS. CPU frequency varied and package temperatures reached 97°C; no throttle counters were captured. See the [measurement audit](audit.md) for monitor observations and limits.

Platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.43`. Full machine metadata and effective backend versions/settings are in matrix.json. Frame pixel-hash prefixes and annotations were also checked against the original fixed run; the hashes cover 64-bit SHA256 prefixes, not full pixel digests.

Source commit: `2489605efc2f07043222b52e3a69132d2e5dee6d`. NAPA binary SHA256: `ab299c868122f5ce775ec0f2a0f7ac49c56b82f645ffd27afed67b360da534cc`. Maximum truth/NAPA detections: 35/35, below 64. Source hashes and recorded NAPA settings were checked against protocol.json; peer binary hashes are not a complete inventory of loaded native code. Binary available and hash-verified locally: True.

[Protocol](protocol.json), [raw metadata/settings](matrix.json), [per-frame measurements](matrix.csv.gz), [analysis](analysis.json), [commands](commands.md), [adapter validation](validate.json), [monitor](monitor.tsv).
