
#### all-corpus @ 640x480

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.155 | 0.361 | 0.422 | 1.00 | — | 522/523 (99.8%) | 0 | 0 | 0.026 / 0.104 / 3.39 | 2 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.315 | 1.060 | 1.965 | 1.00 | 2.19× (1.29–12.77) | 523/523 (100.0%) | 0 | 0 | 0.026 / 0.103 / 3.57 | 3 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 2.065 | 7.902 | 14.011 | 1.00 | 13.48× (3.40–223.08) | 494/523 (94.5%) | 0 | 0 | 0.284 / 1.613 / 8.48 | 20 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 1.392 | 4.228 | 5.854 | 1.00 | 8.91× (2.11–84.01) | 316/523 (60.4%) | 0 | 0 | 0.285 / 0.835 / 8.48 | 11 | s0 1.000 |

#### all-corpus @ 1280x720

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.196 | 0.803 | 0.975 | 1.00 | — | 542/543 (99.8%) | 0 | 0 | 0.019 / 0.081 / 0.40 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.547 | 2.834 | 4.215 | 1.00 | 2.83× (1.60–17.56) | 543/543 (100.0%) | 0 | 0 | 0.019 / 0.083 / 0.40 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 3.824 | 14.048 | 18.409 | 1.00 | 16.39× (4.25–308.04) | 518/543 (95.4%) | 0 | 0 | 0.286 / 1.330 / 13.72 | 18 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 2.809 | 8.428 | 12.787 | 1.00 | 12.87× (3.21–76.43) | 468/543 (86.2%) | 0 | 0 | 0.285 / 0.698 / 13.72 | 7 | s0 1.000 |

#### all-corpus @ 1920x1080

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.292 | 1.795 | 2.521 | 1.00 | — | 524/526 (99.6%) | 0 | 0 | 0.018 / 0.085 / 0.29 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.951 | 7.514 | 10.009 | 1.00 | 3.43× (1.71–20.83) | 521/526 (99.0%) | 0 | 0 | 0.018 / 0.084 / 0.29 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 7.454 | 17.298 | 23.852 | 1.00 | 18.79× (3.43–143.21) | 508/526 (96.6%) | 0 | 0 | 0.279 / 0.590 / 4.58 | 9 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 5.282 | 12.241 | 17.215 | 1.00 | 15.26× (2.93–93.94) | 487/526 (92.6%) | 0 | 0 | 0.274 / 0.534 / 2.40 | 1 | s0 1.000 |

#### all-stress @ 640x480

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.291 | 0.640 | 0.707 | 1.00 | — | 365/388 (94.1%) | 0 | 0 | 0.027 / 0.133 / 0.28 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 1.135 | 2.511 | 3.406 | 1.00 | 4.89× (1.37–15.78) | 388/388 (100.0%) | 0 | 0 | 0.028 / 0.137 / 0.32 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 2.870 | 13.233 | 14.681 | 1.00 | 11.77× (3.09–40.66) | 317/388 (81.7%) | 0 | 0 | 0.286 / 0.720 / 10.75 | 10 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 1.451 | 4.488 | 5.260 | 1.00 | 5.90× (1.38–38.65) | 47/388 (12.1%) | 0 | 0 | 0.328 / 3.099 / 10.75 | 3 | s0 1.000 |

#### all-stress @ 1280x720

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.609 | 1.245 | 1.321 | 1.00 | — | 459/477 (96.2%) | 0 | 0 | 0.020 / 0.085 / 0.22 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 2.610 | 7.099 | 10.365 | 1.00 | 5.69× (1.48–25.51) | 477/477 (100.0%) | 0 | 0 | 0.021 / 0.095 / 0.38 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 7.523 | 18.896 | 19.409 | 1.00 | 12.85× (3.97–42.96) | 440/477 (92.2%) | 0 | 0 | 0.269 / 0.670 / 4.83 | 8 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 6.280 | 11.066 | 11.449 | 1.00 | 8.22× (3.53–32.32) | 220/477 (46.1%) | 0 | 0 | 0.270 / 0.411 / 0.62 | 0 | s0 1.000 |

#### all-stress @ 1920x1080

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 1.075 | 2.506 | 2.824 | 1.00 | — | 473/489 (96.7%) | 0 | 0 | 0.015 / 0.103 / 0.18 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 6.369 | 16.647 | 26.816 | 1.00 | 5.86× (1.58–35.48) | 489/489 (100.0%) | 0 | 0 | 0.016 / 0.111 / 0.43 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 15.234 | 28.380 | 37.030 | 1.00 | 12.93× (4.66–35.78) | 450/489 (92.0%) | 0 | 0 | 0.269 / 0.405 / 11.74 | 2 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 13.161 | 23.723 | 29.289 | 1.00 | 12.55× (2.98–29.54) | 442/489 (90.4%) | 0 | 0 | 0.270 / 0.406 / 11.74 | 2 | s0 1.000 |

#### range @ 640x480

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.039 | 0.075 | 0.101 | 1.01 | — | 244/330 (73.9%) | 0 | 0 | 0.031 / 0.093 / 0.20 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.086 | 0.142 | 0.181 | 1.00 | 2.23× (1.53–5.21) | 330/330 (100.0%) | 0 | 0 | 0.041 / 0.205 / 0.55 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 0.301 | 1.416 | 1.858 | 1.00 | 13.09× (6.23–38.75) | 110/330 (33.3%) | 0 | 0 | 0.268 / 0.355 / 0.39 | 0 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 0.300 | 0.864 | 1.535 | 1.00 | 7.89× (4.70–21.61) | 30/330 (9.1%) | 0 | 0 | 0.251 / 0.332 / 0.39 | 0 | s0 1.000 |

#### stress-tiny @ 640x480

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.180 | 0.298 | 0.325 | 1.00 | — | 17/40 (42.5%) | 0 | 0 | 0.040 / 0.119 / 0.20 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.930 | 1.222 | 1.372 | 1.00 | 4.48× (1.70–6.20) | 40/40 (100.0%) | 0 | 0 | 0.052 / 0.213 / 0.28 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 1.201 | 1.520 | 1.686 | 1.00 | 6.31× (3.66–7.81) | 0/40 (0.0%) | 0 | 0 | — / — / — | 0 | — — |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 1.059 | 1.310 | 1.438 | 1.00 | 5.28× (3.51–6.76) | 0/40 (0.0%) | 0 | 0 | — / — / — | 0 | — — |

#### stress-tiny @ 1280x720

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.353 | 0.514 | 0.522 | 1.00 | — | 17/35 (48.6%) | 0 | 0 | 0.048 / 0.157 / 0.16 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 2.334 | 2.483 | 2.535 | 1.00 | 5.17× (2.28–7.21) | 35/35 (100.0%) | 0 | 0 | 0.053 / 0.226 / 0.38 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 3.235 | 3.378 | 3.413 | 1.00 | 8.71× (6.38–14.13) | 0/35 (0.0%) | 0 | 0 | — / — / — | 0 | — — |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 2.886 | 3.036 | 3.108 | 1.00 | 7.71× (5.65–14.16) | 0/35 (0.0%) | 0 | 0 | — / — / — | 0 | — — |

#### stress-tiny @ 1920x1080

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.162 | 0.873 | 0.917 | 1.00 | — | 19/35 (54.3%) | 0 | 0 | 0.046 / 0.147 / 0.17 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.494 | 5.405 | 5.538 | 1.00 | 5.84× (2.96–7.11) | 35/35 (100.0%) | 0 | 0 | 0.059 / 0.146 / 0.43 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 1.842 | 7.566 | 7.687 | 1.00 | 10.95× (8.20–29.48) | 0/35 (0.0%) | 0 | 0 | — / — / — | 0 | — — |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 1.842 | 6.697 | 6.828 | 1.00 | 10.99× (7.26–29.54) | 0/35 (0.0%) | 0 | 0 | — / — / — | 0 | — — |

#### stress-dense @ 640x480

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.575 | 0.702 | 0.730 | 1.00 | — | 318/318 (100.0%) | 0 | 0 | 0.024 / 0.058 / 0.17 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 1.089 | 1.224 | 1.258 | 1.00 | 1.73× (1.37–2.08) | 318/318 (100.0%) | 0 | 0 | 0.024 / 0.060 / 0.17 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 11.867 | 14.664 | 14.830 | 1.00 | 20.78× (14.19–25.95) | 287/318 (90.3%) | 0 | 0 | 0.280 / 0.409 / 3.13 | 5 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 1.726 | 3.934 | 4.046 | 1.00 | 2.87× (1.38–6.72) | 24/318 (7.5%) | 0 | 0 | 0.281 / 0.354 / 0.39 | 0 | s0 1.000 |

#### stress-dense @ 1280x720

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 1.011 | 1.299 | 1.375 | 1.00 | — | 411/411 (100.0%) | 0 | 0 | 0.018 / 0.049 / 0.18 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 2.409 | 2.687 | 2.749 | 1.00 | 2.01× (1.48–2.56) | 411/411 (100.0%) | 0 | 0 | 0.018 / 0.052 / 0.18 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 17.318 | 19.393 | 19.583 | 1.00 | 17.81× (8.75–22.58) | 409/411 (99.5%) | 0 | 0 | 0.264 / 0.719 / 4.83 | 8 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 9.099 | 11.405 | 11.488 | 1.00 | 8.33× (5.78–14.15) | 189/411 (46.0%) | 0 | 0 | 0.263 / 0.376 / 0.45 | 0 | s0 1.000 |

#### stress-dense @ 1920x1080

| Backend | Tuning | p50 ms | p95 ms | p99 ms | CPU/wall | Time ÷ NAPA per frame: median (min–max) | Recall | False IDs | Dup | Corner err median / p95 / max px | >2 px | Order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 1.594 | 2.748 | 2.937 | 1.00 | — | 426/426 (100.0%) | 0 | 0 | 0.014 / 0.046 / 0.14 | 0 | s0 1.000 |
| napa-mtp16 | tuned: min_tag_px=16 | 2.556 | 6.395 | 6.802 | 1.00 | 1.73× (1.58–3.04) | 426/426 (100.0%) | 0 | 0 | 0.014 / 0.047 / 0.15 | 0 | s0 1.000 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 23.503 | 36.566 | 38.001 | 1.00 | 16.04× (9.24–23.26) | 424/426 (99.5%) | 0 | 0 | 0.266 / 0.382 / 0.48 | 0 | s0 1.000 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 20.780 | 29.168 | 30.966 | 1.00 | 15.21× (8.14–18.04) | 416/426 (97.7%) | 0 | 0 | 0.268 / 0.382 / 0.48 | 0 | s0 1.000 |
