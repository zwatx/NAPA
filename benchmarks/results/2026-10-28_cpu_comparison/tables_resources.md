
#### resources_1thread_1280x720: serial, 1 worker(s) x 1 thread(s), CPUs 2, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opencv-subpix | tuned: CORNER_REFINE_SUBPIX | 28.209 (26.667–30.676) | 28.200 | 1.00 | 35 | 13.840 / 90.628 / 159.203 | 54.33× (14.57–1377.85) | 20.3 | 18.8 | 1 | 2 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 6.038 (5.989–6.198) | 6.036 | 1.00 | 166 | 4.108 / 16.405 / 21.588 | 16.71× (4.64–154.33) | 20.4 | 10.1 | 1 | 2 |
| napa | default | 0.336 (0.335–0.338) | 0.336 | 1.00 | 2978 | 0.209 / 1.009 / 1.197 | — | 0.8 | 0.5 | 1 | 2 |
| apriltag3-d2 | default (quad_decimate=2) | 14.679 (13.515–19.295) | 14.624 | 1.00 | 68 | 15.590 / 24.174 / 26.274 | 54.48× (8.54–747.10) | 0.3 | 8.0 | 1 | 2 |
| locus | default (standard profile) | 9.126 (8.943–10.376) | 9.123 | 1.00 | 110 | 8.578 / 15.569 / 18.711 | 36.62× (9.96–224.57) | 13.4 | 12.8 | 2 | 2 |
| rapidtag-subpix | tuned: CORNER_REFINE_SUBPIX | 7.869 (7.175–8.316) | 7.867 | 1.00 | 127 | 4.787 / 27.115 / 38.171 | 21.59× (6.39–135.91) | 6.1 | 16.2 | 2 | 2 |
| opencv | default (no corner refinement) | 26.542 (26.539–28.372) | 26.535 | 1.00 | 38 | 13.115 / 88.038 / 154.518 | 54.18× (14.18–1343.79) | 20.4 | 17.5 | 1 | 2 |
| napa-mtp16 | tuned: min_tag_px=16 | 1.446 (1.435–1.449) | 1.445 | 1.00 | 692 | 0.538 / 5.133 / 8.181 | 2.73× (1.47–25.13) | 0.7 | 5.9 | 1 | 2 |
| opencv-aruco3 | tuned: useAruco3Detection (defaults; no refinement runs at full scale) | 11.794 (11.792–11.914) | 11.790 | 1.00 | 85 | 8.277 / 33.940 / 40.847 | 33.58× (7.59–164.57) | 20.4 | 17.8 | 1 | 2 |
| apriltag3-d1 | tuned: quad_decimate=1 (full resolution) | 55.617 (54.540–57.118) | 55.595 | 1.00 | 18 | 60.904 / 94.392 / 111.222 | 216.12× (23.84–2412.99) | 0.3 | 27.3 | 1 | 2 |
| napa-mtp43 | tuned: min_tag_px=43 | 0.286 (0.284–0.287) | 0.286 | 1.00 | 3495 | 0.201 / 0.843 / 1.000 | 0.94× (0.63–1.14) | 0.7 | 0.3 | 1 | 2 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 3.943 (3.938–3.987) | 3.942 | 1.00 | 254 | 2.974 / 10.573 / 13.873 | 13.05× (3.09–75.20) | 20.4 | 10.5 | 1 | 2 |
| napa-mtp22 | tuned: min_tag_px=22 | 0.534 (0.528–0.704) | 0.534 | 1.00 | 1874 | 0.284 / 1.482 / 2.077 | 1.60× (1.07–7.03) | 0.7 | 1.1 | 1 | 2 |
| napa-mtp64 | tuned: min_tag_px=64 | 0.197 (0.194–0.197) | 0.196 | 1.00 | 5089 | 0.155 / 0.468 / 0.838 | 0.65× (0.25–1.04) | 0.7 | 0.2 | 1 | 2 |
| aruco-nano | default (36h11 dictionary only) | 2.668 (2.650–3.242) | 2.668 | 1.00 | 375 | 1.899 / 6.651 / 7.249 | 9.35× (2.78–42.07) | 5.1 | 5.6 | 1 | 2 |
| rapidtag | default (no corner refinement) | 7.105 (7.054–8.177) | 7.104 | 1.00 | 141 | 4.378 / 25.353 / 34.583 | 21.30× (6.24–124.92) | 6.1 | 17.3 | 2 | 2 |
| locus-d2 | tuned for speed: standard + decimation=2 | 4.350 (4.327–5.230) | 4.349 | 1.00 | 230 | 3.966 / 6.430 / 11.776 | 17.71× (4.38–80.56) | 13.5 | 5.3 | 2 | 2 |

#### resources_1thread_1920x1080: serial, 1 worker(s) x 1 thread(s), CPUs 2, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opencv-subpix | tuned: CORNER_REFINE_SUBPIX | 38.028 (37.489–42.423) | 38.015 | 1.00 | 26 | 25.451 / 97.241 / 111.371 | 49.83× (14.30–516.34) | 20.3 | 33.3 | 1 | 2 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 8.452 (8.397–10.176) | 8.450 | 1.00 | 118 | 7.827 / 18.489 / 19.831 | 15.85× (3.50–76.69) | 20.2 | 13.7 | 1 | 2 |
| napa | default | 0.770 (0.769–0.776) | 0.770 | 1.00 | 1298 | 0.403 / 2.018 / 2.287 | — | 0.6 | 1.1 | 1 | 2 |
| apriltag3-d2 | default (quad_decimate=2) | 38.345 (34.793–40.024) | 38.332 | 1.00 | 26 | 40.394 / 66.393 / 75.757 | 48.44× (9.69–724.42) | 0.3 | 16.3 | 1 | 2 |
| locus | default (standard profile) | 23.129 (20.793–24.220) | 23.116 | 1.00 | 43 | 21.492 / 38.894 / 41.837 | 43.41× (7.31–224.20) | 13.3 | 27.8 | 2 | 2 |
| rapidtag-subpix | tuned: CORNER_REFINE_SUBPIX | 16.562 (16.340–16.599) | 16.556 | 1.00 | 60 | 8.626 / 61.189 / 75.711 | 23.51× (7.98–89.80) | 5.9 | 20.2 | 2 | 2 |
| opencv | default (no corner refinement) | 38.261 (37.785–41.246) | 38.246 | 1.00 | 26 | 26.061 / 97.391 / 115.822 | 46.79× (14.23–462.95) | 20.2 | 31.2 | 1 | 2 |
| napa-mtp16 | tuned: min_tag_px=16 | 3.524 (3.304–4.010) | 3.523 | 1.00 | 284 | 1.245 / 11.329 / 14.853 | 3.41× (1.61–25.08) | 0.6 | 14.9 | 1 | 2 |
| opencv-aruco3 | tuned: useAruco3Detection (defaults; no refinement runs at full scale) | 22.548 (22.512–22.875) | 22.541 | 1.00 | 44 | 18.084 / 75.104 / 86.081 | 36.63× (10.78–112.18) | 20.2 | 31.9 | 1 | 2 |
| apriltag3-d1 | tuned: quad_decimate=1 (full resolution) | 139.390 (137.671–140.862) | 139.343 | 1.00 | 7 | 149.361 / 217.229 / 257.403 | 195.86× (20.47–2766.10) | 0.3 | 60.8 | 1 | 2 |
| napa-mtp43 | tuned: min_tag_px=43 | 0.584 (0.581–0.591) | 0.584 | 1.00 | 1713 | 0.348 / 1.360 / 1.613 | 0.87× (0.59–1.11) | 0.6 | 0.7 | 1 | 2 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 6.856 (6.806–7.956) | 6.852 | 1.00 | 146 | 6.364 / 15.884 / 20.108 | 13.08× (3.07–63.26) | 20.2 | 14.5 | 1 | 2 |
| napa-mtp22 | tuned: min_tag_px=22 | 1.238 (1.222–1.459) | 1.237 | 1.00 | 808 | 0.548 / 3.349 / 3.557 | 1.58× (1.04–3.30) | 0.6 | 3.0 | 1 | 2 |
| napa-mtp64 | tuned: min_tag_px=64 | 0.395 (0.390–0.396) | 0.395 | 1.00 | 2531 | 0.274 / 0.972 / 1.220 | 0.62× (0.29–0.98) | 0.6 | 0.3 | 1 | 2 |
| aruco-nano | default (36h11 dictionary only) | 5.833 (5.799–5.940) | 5.831 | 1.00 | 171 | 4.942 / 15.048 / 17.452 | 9.50× (2.52–43.88) | 5.1 | 8.5 | 1 | 2 |
| rapidtag | default (no corner refinement) | 17.841 (16.234–18.962) | 17.835 | 1.00 | 56 | 9.513 / 67.510 / 80.621 | 24.21× (8.07–120.02) | 5.9 | 20.2 | 2 | 2 |
| locus-d2 | tuned for speed: standard + decimation=2 | 10.992 (9.128–11.272) | 10.986 | 1.00 | 91 | 10.363 / 17.521 / 22.295 | 21.23× (4.37–125.08) | 13.3 | 10.4 | 2 | 2 |

#### resources_1thread_640x480: serial, 1 worker(s) x 1 thread(s), CPUs 2, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opencv-subpix | tuned: CORNER_REFINE_SUBPIX | 16.296 (14.899–17.785) | 16.290 | 1.00 | 61 | 9.543 / 54.852 / 93.130 | 46.85× (12.64–1548.26) | 20.2 | 11.0 | 1 | 2 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 3.223 (2.796–3.400) | 3.220 | 1.00 | 310 | 2.321 / 9.539 / 13.912 | 12.20× (2.86–284.26) | 20.2 | 8.1 | 1 | 2 |
| napa | default | 0.224 (0.178–0.234) | 0.224 | 1.00 | 4463 | 0.187 / 0.495 / 0.696 | — | 0.6 | 0.2 | 1 | 2 |
| apriltag3-d2 | default (quad_decimate=2) | 4.843 (4.777–5.854) | 4.842 | 1.00 | 206 | 5.187 / 7.378 / 7.832 | 23.06× (4.37–373.75) | 0.3 | 2.7 | 1 | 2 |
| locus | default (standard profile) | 3.784 (3.404–4.043) | 3.783 | 1.00 | 264 | 3.579 / 6.393 / 8.420 | 19.10× (7.85–140.19) | 13.2 | 4.7 | 2 | 2 |
| rapidtag-subpix | tuned: CORNER_REFINE_SUBPIX | 2.739 (2.732–2.859) | 2.738 | 1.00 | 365 | 2.181 / 8.555 / 12.603 | 12.23× (4.06–60.34) | 5.9 | 18.3 | 2 | 2 |
| opencv | default (no corner refinement) | 15.045 (14.854–15.377) | 15.040 | 1.00 | 66 | 9.315 / 53.671 / 88.453 | 43.90× (12.47–1516.54) | 20.2 | 9.6 | 1 | 2 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.594 (0.583–0.634) | 0.593 | 1.00 | 1685 | 0.389 / 1.898 / 2.333 | 2.09× (1.04–9.52) | 0.6 | 2.3 | 1 | 2 |
| opencv-aruco3 | tuned: useAruco3Detection (defaults; no refinement runs at full scale) | 5.158 (5.139–5.769) | 5.156 | 1.00 | 194 | 3.505 / 14.505 / 17.121 | 21.45× (6.05–524.10) | 20.2 | 9.8 | 1 | 2 |
| apriltag3-d1 | tuned: quad_decimate=1 (full resolution) | 17.762 (16.553–19.030) | 17.756 | 1.00 | 56 | 18.750 / 29.286 / 31.371 | 77.51× (9.73–1381.71) | 0.3 | 9.3 | 1 | 2 |
| napa-mtp43 | tuned: min_tag_px=43 | 0.153 (0.138–0.176) | 0.153 | 1.00 | 6526 | 0.128 / 0.332 / 0.529 | 0.74× (0.29–1.21) | 0.7 | 0.1 | 1 | 2 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 1.544 (1.542–1.576) | 1.543 | 1.00 | 648 | 1.304 / 3.694 / 4.282 | 7.22× (1.71–63.88) | 20.2 | 8.0 | 1 | 2 |
| napa-mtp22 | tuned: min_tag_px=22 | 0.245 (0.243–0.285) | 0.244 | 1.00 | 4088 | 0.216 / 0.556 / 0.701 | 1.14× (0.71–3.43) | 0.6 | 0.5 | 1 | 2 |
| napa-mtp64 | tuned: min_tag_px=64 | 0.106 (0.105–0.146) | 0.106 | 1.00 | 9438 | 0.084 / 0.221 / 0.394 | 0.57× (0.18–1.24) | 0.7 | 0.1 | 1 | 2 |
| aruco-nano | default (36h11 dictionary only) | 1.115 (1.057–1.226) | 1.113 | 1.00 | 897 | 0.860 / 2.364 / 2.796 | 5.43× (1.64–26.74) | 5.1 | 4.0 | 1 | 2 |
| rapidtag | default (no corner refinement) | 2.762 (2.746–2.763) | 2.761 | 1.00 | 362 | 2.202 / 8.604 / 12.548 | 12.07× (3.93–60.26) | 5.9 | 18.3 | 2 | 2 |
| locus-d2 | tuned for speed: standard + decimation=2 | 2.372 (2.274–2.409) | 2.371 | 1.00 | 422 | 2.104 / 4.001 / 7.856 | 12.03× (4.35–65.07) | 13.2 | 2.5 | 2 | 2 |

#### resources_6threads_1280x720: serial, 1 worker(s) x 6 thread(s), CPUs 0,2,4,6,8,10, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| aruco-nano | default (36h11 dictionary only) | 3.219 (3.130–3.219) | 4.876 | 1.53 | 311 | 2.273 / 7.848 / 9.006 | 8.76× (2.74–37.87) | 5.1 | 5.8 | 6 | 0,2,4,6,8,10 |
| locus-d2 | tuned for speed: standard + decimation=2 | 2.254 (2.238–2.941) | 6.911 | 3.05 | 444 | 2.082 / 3.651 / 4.916 | 8.54× (1.57–31.06) | 13.6 | 10.0 | 7 | 0,2,4,6,8,10 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 3.019 (2.967–3.720) | 11.009 | 3.65 | 331 | 2.753 / 6.131 / 7.816 | 8.92× (2.70–49.83) | 20.4 | 17.3 | 6 | 0,2,4,6,8,10 |
| napa | default | 0.377 (0.361–0.413) | 0.377 | 1.00 | 2652 | 0.231 / 1.152 / 1.331 | — | 0.6 | 0.6 | 1 | 0,2,4,6,8,10 |
| opencv | default (no corner refinement) | 10.348 (10.255–10.899) | 39.425 | 3.81 | 97 | 7.525 / 27.919 / 40.691 | 24.33× (7.19–298.33) | 20.4 | 52.4 | 6 | 0,2,4,6,8,10 |
| apriltag3-d1 | tuned: quad_decimate=1 (full resolution) | 21.877 (20.808–24.733) | 74.926 | 3.42 | 46 | 22.275 / 40.139 / 44.239 | 64.82× (8.12–812.69) | 0.3 | 71.6 | 7 | 0,2,4,6,8,10 |
| napa-mtp16 | tuned: min_tag_px=16 | 1.677 (1.599–1.851) | 1.676 | 1.00 | 596 | 0.623 / 6.012 / 9.491 | 2.79× (1.48–26.17) | 0.7 | 5.9 | 1 | 0,2,4,6,8,10 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 2.223 (2.139–2.615) | 7.769 | 3.49 | 450 | 1.909 / 4.576 / 5.127 | 7.54× (1.45–44.09) | 20.3 | 17.6 | 6 | 0,2,4,6,8,10 |
| apriltag3-d2 | default (quad_decimate=2) | 5.683 (5.662–6.855) | 19.041 | 3.33 | 176 | 6.207 / 8.309 / 8.840 | 18.95× (3.23–171.75) | 0.3 | 24.5 | 7 | 0,2,4,6,8,10 |
| rapidtag | default (no corner refinement) | 3.852 (3.801–3.872) | 9.001 | 2.34 | 260 | 2.423 / 11.607 / 18.222 | 10.42× (2.91–56.52) | 8.1 | 85.8 | 7 | 0,2,4,6,8,10 |
| opencv-subpix | tuned: CORNER_REFINE_SUBPIX | 11.580 (10.407–12.643) | 42.514 | 3.67 | 86 | 8.191 / 32.596 / 48.987 | 26.02× (7.27–367.94) | 20.3 | 53.2 | 6 | 0,2,4,6,8,10 |
| locus | default (standard profile) | 5.009 (4.989–5.073) | 13.833 | 2.75 | 200 | 4.891 / 7.547 / 13.821 | 16.20× (3.63–90.21) | 13.5 | 19.6 | 7 | 0,2,4,6,8,10 |

#### resources_6threads_1920x1080: serial, 1 worker(s) x 6 thread(s), CPUs 0,2,4,6,8,10, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| aruco-nano | default (36h11 dictionary only) | 7.144 (7.115–7.176) | 9.293 | 1.30 | 140 | 5.645 / 18.412 / 22.265 | 9.86× (2.44–53.83) | 5.0 | 8.6 | 6 | 0,2,4,6,8,10 |
| locus-d2 | tuned for speed: standard + decimation=2 | 5.532 (5.441–6.115) | 15.895 | 2.76 | 181 | 5.343 / 8.601 / 13.302 | 9.83× (1.51–55.22) | 13.4 | 16.8 | 7 | 0,2,4,6,8,10 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 4.953 (4.740–5.348) | 15.554 | 3.12 | 202 | 4.945 / 9.173 / 11.576 | 7.61× (1.84–28.82) | 20.2 | 30.5 | 6 | 0,2,4,6,8,10 |
| napa | default | 0.944 (0.790–0.979) | 0.944 | 1.00 | 1059 | 0.486 / 2.506 / 2.975 | — | 0.6 | 1.1 | 1 | 0,2,4,6,8,10 |
| opencv | default (no corner refinement) | 15.732 (15.687–16.801) | 51.806 | 3.29 | 64 | 13.511 / 42.463 / 52.996 | 21.06× (6.04–121.17) | 20.2 | 107.7 | 6 | 0,2,4,6,8,10 |
| apriltag3-d1 | tuned: quad_decimate=1 (full resolution) | 46.939 (45.853–51.011) | 185.572 | 3.95 | 21 | 49.883 / 77.964 / 95.726 | 55.19× (5.23–666.10) | 0.3 | 125.0 | 7 | 0,2,4,6,8,10 |
| napa-mtp16 | tuned: min_tag_px=16 | 3.954 (3.554–4.184) | 3.952 | 1.00 | 253 | 1.441 / 12.589 / 14.542 | 3.16× (1.43–23.76) | 0.6 | 14.9 | 1 | 0,2,4,6,8,10 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 4.200 (3.695–4.211) | 13.211 | 3.20 | 238 | 4.001 / 8.770 / 10.861 | 5.89× (1.50–31.76) | 20.2 | 30.5 | 6 | 0,2,4,6,8,10 |
| apriltag3-d2 | default (quad_decimate=2) | 12.766 (12.695–14.455) | 46.519 | 3.64 | 78 | 13.804 / 18.686 / 20.162 | 16.57× (2.98–197.21) | 0.3 | 47.8 | 7 | 0,2,4,6,8,10 |
| rapidtag | default (no corner refinement) | 8.548 (8.510–9.838) | 20.685 | 2.42 | 117 | 4.837 / 28.723 / 39.436 | 11.41× (3.16–39.45) | 7.9 | 89.9 | 7 | 0,2,4,6,8,10 |
| opencv-subpix | tuned: CORNER_REFINE_SUBPIX | 15.869 (15.818–16.730) | 52.612 | 3.30 | 63 | 13.177 / 42.559 / 54.583 | 20.99× (6.20–122.41) | 20.2 | 107.0 | 6 | 0,2,4,6,8,10 |
| locus | default (standard profile) | 11.583 (10.905–12.848) | 30.954 | 2.67 | 86 | 11.094 / 18.436 / 22.612 | 18.88× (2.29–129.45) | 13.5 | 37.8 | 7 | 0,2,4,6,8,10 |

#### resources_6threads_640x480: serial, 1 worker(s) x 6 thread(s), CPUs 0,2,4,6,8,10, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| aruco-nano | default (36h11 dictionary only) | 1.282 (1.275–1.312) | 2.905 | 2.21 | 780 | 0.972 / 2.782 / 3.440 | 6.29× (2.25–26.12) | 5.1 | 4.1 | 6 | 0,2,4,6,8,10 |
| locus-d2 | tuned for speed: standard + decimation=2 | 1.430 (1.105–1.567) | 3.788 | 2.86 | 699 | 1.277 / 2.661 / 4.339 | 6.86× (2.21–47.55) | 13.4 | 3.6 | 7 | 0,2,4,6,8,10 |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 1.678 (1.656–1.804) | 5.771 | 3.44 | 596 | 1.489 / 3.827 / 4.434 | 7.40× (2.13–96.24) | 20.2 | 11.4 | 6 | 0,2,4,6,8,10 |
| napa | default | 0.209 (0.197–0.216) | 0.209 | 1.00 | 4796 | 0.175 / 0.473 / 0.626 | — | 0.6 | 0.3 | 1 | 0,2,4,6,8,10 |
| opencv | default (no corner refinement) | 6.015 (6.003–6.671) | 22.939 | 3.80 | 166 | 4.771 / 14.168 / 21.797 | 24.36× (8.33–338.66) | 20.2 | 21.1 | 6 | 0,2,4,6,8,10 |
| apriltag3-d1 | tuned: quad_decimate=1 (full resolution) | 6.608 (6.432–9.187) | 23.170 | 3.49 | 151 | 7.088 / 10.005 / 10.990 | 33.46× (6.13–768.37) | 0.3 | 29.8 | 7 | 0,2,4,6,8,10 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.672 (0.647–0.744) | 0.672 | 1.00 | 1488 | 0.457 / 2.127 / 2.638 | 2.44× (1.29–10.48) | 0.6 | 2.3 | 1 | 0,2,4,6,8,10 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 1.178 (1.133–1.204) | 3.706 | 3.23 | 849 | 0.994 / 2.747 / 2.963 | 5.51× (1.03–38.50) | 20.2 | 11.1 | 6 | 0,2,4,6,8,10 |
| apriltag3-d2 | default (quad_decimate=2) | 2.137 (2.076–3.064) | 7.365 | 3.45 | 468 | 2.241 / 3.080 / 4.104 | 10.60× (2.74–181.13) | 0.3 | 9.4 | 7 | 0,2,4,6,8,10 |
| rapidtag | default (no corner refinement) | 1.508 (1.464–1.707) | 3.426 | 2.27 | 663 | 1.179 / 4.069 / 5.936 | 7.79× (2.69–35.31) | 8.0 | 72.3 | 7 | 0,2,4,6,8,10 |
| opencv-subpix | tuned: CORNER_REFINE_SUBPIX | 6.015 (5.946–6.043) | 22.939 | 3.83 | 166 | 4.816 / 14.050 / 21.449 | 23.63× (9.07–332.28) | 20.2 | 22.2 | 6 | 0,2,4,6,8,10 |
| locus | default (standard profile) | 2.034 (1.982–2.557) | 5.870 | 2.89 | 492 | 1.984 / 3.475 / 4.705 | 12.01× (3.69–101.70) | 13.4 | 6.2 | 7 | 0,2,4,6,8,10 |

#### throughput_6workers_1280x720: parallel, 6 worker(s) x 1 thread(s), CPUs 0,2,4,6,8,10, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 1.736 (1.573–2.183) | 9.711 | 5.59 | 576 | 6.545 / 27.913 / 44.857 | 19.70× (0.83–331.27) | 20.5 | 27.2 | 1 | 0,2,4,6,8,10 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 1.154 (1.025–1.155) | 6.403 | 5.82 | 867 | 4.867 / 17.588 / 24.233 | 15.12× (0.90–90.14) | 20.5 | 27.2 | 1 | 0,2,4,6,8,10 |
| napa | default | 0.079 (0.077–0.092) | 0.444 | 5.75 | 12626 | 0.287 / 1.335 / 1.632 | — | 1.7 | 2.8 | 1 | 0,2,4,6,8,10 |
| locus-d2 | tuned for speed: standard + decimation=2 | 1.167 (1.119–1.457) | 6.744 | 5.75 | 857 | 6.235 / 11.744 / 18.298 | 19.54× (3.05–107.99) | 16.1 | 32.2 | 1 | 0,2,4,6,8,10 |
| apriltag3-d1 | tuned: quad_decimate=1 (full resolution) | 13.630 (13.319–14.550) | 80.321 | 5.90 | 73 | 89.937 / 135.381 / 143.677 | 232.16× (27.38–2442.59) | 1.0 | 161.2 | 1 | 0,2,4,6,8,10 |
| opencv | default (no corner refinement) | 7.077 (7.066–7.558) | 41.450 | 5.85 | 141 | 20.502 / 137.571 / 242.549 | 62.61× (3.37–1574.41) | 20.5 | 78.4 | 1 | 0,2,4,6,8,10 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.359 (0.358–0.431) | 2.079 | 5.79 | 2787 | 0.772 / 7.161 / 13.768 | 3.15× (0.21–34.39) | 1.8 | 33.0 | 1 | 0,2,4,6,8,10 |
| apriltag3-d2 | default (quad_decimate=2) | 3.999 (3.539–4.960) | 21.609 | 5.62 | 250 | 24.778 / 39.135 / 66.878 | 66.41× (9.58–816.47) | 1.0 | 44.4 | 1 | 0,2,4,6,8,10 |
| aruco-nano | default (36h11 dictionary only) | 0.671 (0.662–0.703) | 3.963 | 5.90 | 1491 | 2.816 / 9.981 / 11.511 | 9.60× (0.76–38.10) | 5.6 | 17.2 | 1 | 0,2,4,6,8,10 |
| opencv-subpix | tuned: CORNER_REFINE_SUBPIX | 7.011 (6.928–7.022) | 40.978 | 5.85 | 143 | 20.084 / 134.339 / 236.907 | 60.81× (3.33–1557.11) | 20.5 | 79.4 | 1 | 0,2,4,6,8,10 |
| locus | default (standard profile) | 2.455 (2.349–2.725) | 13.901 | 5.66 | 407 | 13.219 / 23.644 / 28.925 | 41.58× (7.79–206.53) | 16.2 | 77.4 | 1 | 0,2,4,6,8,10 |

#### throughput_6workers_1920x1080: parallel, 6 worker(s) x 1 thread(s), CPUs 0,2,4,6,8,10, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 2.515 (2.228–2.786) | 13.693 | 5.69 | 398 | 12.812 / 33.107 / 39.687 | 16.93× (3.23–99.65) | 20.4 | 48.8 | 1 | 0,2,4,6,8,10 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 1.870 (1.737–2.012) | 10.787 | 5.82 | 535 | 9.778 / 27.136 / 32.089 | 13.36× (2.77–63.75) | 20.4 | 49.7 | 1 | 0,2,4,6,8,10 |
| napa | default | 0.222 (0.186–0.222) | 1.157 | 5.26 | 4509 | 0.705 / 3.191 / 4.025 | — | 1.6 | 6.6 | 1 | 0,2,4,6,8,10 |
| locus-d2 | tuned for speed: standard + decimation=2 | 2.505 (2.380–3.078) | 14.155 | 5.65 | 399 | 13.688 / 24.693 / 29.540 | 22.23× (3.90–129.43) | 16.1 | 68.6 | 1 | 0,2,4,6,8,10 |
| apriltag3-d1 | tuned: quad_decimate=1 (full resolution) | 34.006 (33.705–35.703) | 200.872 | 5.91 | 29 | 218.419 / 314.407 / 322.874 | 172.20× (16.90–2511.75) | 1.0 | 360.7 | 1 | 0,2,4,6,8,10 |
| opencv | default (no corner refinement) | 9.558 (9.539–10.629) | 56.236 | 5.88 | 105 | 36.791 / 148.513 / 172.960 | 49.43× (11.97–527.94) | 20.4 | 161.8 | 1 | 0,2,4,6,8,10 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.851 (0.835–0.928) | 4.970 | 5.84 | 1175 | 1.682 / 16.400 / 19.558 | 3.45× (1.63–24.38) | 1.7 | 88.5 | 1 | 0,2,4,6,8,10 |
| apriltag3-d2 | default (quad_decimate=2) | 8.939 (8.931–9.967) | 53.008 | 5.93 | 112 | 58.995 / 83.299 / 89.859 | 47.99× (6.98–720.80) | 1.0 | 95.7 | 1 | 0,2,4,6,8,10 |
| aruco-nano | default (36h11 dictionary only) | 1.720 (1.470–1.894) | 8.590 | 4.83 | 581 | 7.332 / 26.804 / 41.371 | 10.19× (2.11–94.52) | 5.6 | 33.4 | 1 | 0,2,4,6,8,10 |
| opencv-subpix | tuned: CORNER_REFINE_SUBPIX | 10.282 (9.592–10.797) | 60.139 | 5.85 | 97 | 42.137 / 154.694 / 179.315 | 51.33× (12.94–532.27) | 20.4 | 167.5 | 1 | 0,2,4,6,8,10 |
| locus | default (standard profile) | 5.455 (5.401–5.963) | 31.797 | 5.81 | 183 | 30.187 / 51.538 / 57.308 | 41.24× (6.52–237.60) | 16.0 | 172.8 | 1 | 0,2,4,6,8,10 |

#### throughput_6workers_640x480: parallel, 6 worker(s) x 1 thread(s), CPUs 0,2,4,6,8,10, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| opencv-aruco3-half16 | tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px) | 0.710 (0.687–0.718) | 4.138 | 5.82 | 1409 | 2.959 / 12.347 / 19.024 | 12.85× (2.27–252.32) | 20.3 | 14.2 | 1 | 0,2,4,6,8,10 |
| opencv-aruco3-min32 | tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px) | 0.386 (0.365–0.387) | 2.253 | 5.88 | 2588 | 1.876 / 5.431 / 6.335 | 7.87× (1.94–60.11) | 20.3 | 13.9 | 1 | 0,2,4,6,8,10 |
| napa | default | 0.043 (0.042–0.088) | 0.244 | 5.57 | 23379 | 0.206 / 0.561 / 0.754 | — | 1.6 | 1.4 | 1 | 0,2,4,6,8,10 |
| locus-d2 | tuned for speed: standard + decimation=2 | 0.500 (0.472–0.526) | 2.790 | 5.68 | 2000 | 2.619 / 5.632 / 9.163 | 11.54× (3.46–64.84) | 15.9 | 13.1 | 1 | 0,2,4,6,8,10 |
| apriltag3-d1 | tuned: quad_decimate=1 (full resolution) | 4.349 (4.258–5.112) | 25.209 | 5.85 | 230 | 27.301 / 42.017 / 44.751 | 96.80× (6.92–2845.63) | 1.0 | 54.3 | 1 | 0,2,4,6,8,10 |
| opencv | default (no corner refinement) | 3.847 (3.789–4.707) | 22.709 | 5.90 | 260 | 14.232 / 80.581 / 133.181 | 54.35× (5.60–1736.74) | 20.3 | 28.6 | 1 | 0,2,4,6,8,10 |
| napa-mtp16 | tuned: min_tag_px=16 | 0.134 (0.129–0.178) | 0.767 | 5.72 | 7468 | 0.487 / 2.458 / 2.941 | 2.17× (0.44–11.34) | 1.6 | 12.8 | 1 | 0,2,4,6,8,10 |
| apriltag3-d2 | default (quad_decimate=2) | 1.256 (1.241–1.287) | 7.321 | 5.83 | 796 | 7.805 / 11.512 / 13.540 | 27.11× (2.48–471.80) | 1.0 | 15.4 | 1 | 0,2,4,6,8,10 |
| aruco-nano | default (36h11 dictionary only) | 0.281 (0.260–0.341) | 1.537 | 5.48 | 3563 | 1.173 / 3.839 / 5.816 | 5.87× (1.29–29.23) | 5.6 | 9.0 | 1 | 0,2,4,6,8,10 |
| opencv-subpix | tuned: CORNER_REFINE_SUBPIX | 4.016 (3.776–4.185) | 22.759 | 5.82 | 249 | 14.438 / 79.928 / 137.797 | 53.66× (5.52–1741.99) | 20.3 | 29.8 | 1 | 0,2,4,6,8,10 |
| locus | default (standard profile) | 0.869 (0.855–1.205) | 5.075 | 5.90 | 1150 | 4.872 / 8.479 / 12.260 | 21.51× (3.80–159.60) | 16.0 | 28.4 | 1 | 0,2,4,6,8,10 |

#### throughput_6workers_rapidtag_1280x720: parallel, 6 worker(s) x 6 thread(s), CPUs 0,2,4,6,8,10, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.121 (0.108–0.126) | 0.480 | 3.95 | 8234 | 0.335 / 2.535 / 4.670 | — | 1.7 | 2.8 | 1 | 0,2,4,6,8,10 |
| rapidtag | default (no corner refinement) | 2.537 (2.425–2.589) | 10.711 | 4.35 | 394 | 11.415 / 39.704 / 55.704 | 30.12× (2.44–255.76) | 8.3 | 102.2 | 7 | 0,2,4,6,8,10 |

#### throughput_6workers_rapidtag_1920x1080: parallel, 6 worker(s) x 6 thread(s), CPUs 0,2,4,6,8,10, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.202 (0.180–0.232) | 1.098 | 5.63 | 4958 | 0.740 / 3.156 / 3.514 | — | 1.7 | 6.7 | 1 | 0,2,4,6,8,10 |
| rapidtag | default (no corner refinement) | 5.735 (5.041–6.421) | 25.222 | 4.63 | 174 | 22.800 / 94.354 / 151.789 | 34.72× (4.62–410.89) | 8.1 | 119.9 | 7 | 0,2,4,6,8,10 |

#### throughput_6workers_rapidtag_640x480: parallel, 6 worker(s) x 6 thread(s), CPUs 0,2,4,6,8,10, 3 passes x 3 rounds

| Backend | Tuning | Wall ms/frame (range over rounds) | CPU ms/frame | Busy CPUs | Frames/s | p50 / p95 / p99 ms | Time ÷ NAPA per frame: median (min–max) | RSS import+construct MiB | Peak RSS over construct MiB | OS threads | Thread CPUs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| napa | default | 0.040 (0.040–0.040) | 0.229 | 5.72 | 25201 | 0.193 / 0.519 / 0.706 | — | 1.7 | 1.3 | 1 | 0,2,4,6,8,10 |
| rapidtag | default (no corner refinement) | 0.797 (0.759–0.820) | 3.876 | 4.83 | 1254 | 3.649 / 11.243 / 16.964 | 20.20× (4.08–192.73) | 8.1 | 103.0 | 7 | 0,2,4,6,8,10 |
