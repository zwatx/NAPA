# Measurement completion and monitor audit

The fresh measurement runner exited 0, including its final checks of all six recorded source hashes, the
NAPA binary and both native shim hashes. The report generator exited 0 without flags or modifications.
Timing ran from 2026-11-15T20:19:21.382485+00:00 to 2026-11-15T23:01:08.979231+00:00 (2h 41m 47.597s).

All 16 configurations completed all 4,500 frames: 3,000 main, 600 stress and 900 range. The raw CSV retains
72,000 rows and 720,000 positive wall-time samples, plus 720,000 positive CPU-time samples. Every row has ten
timed samples and stable outputs across repetitions. No frames or timing outliers were removed. The largest
individual retained wall-time sample is 274.600632 ms. No backend was skipped. All preflight adapters had zero
wrong IDs, duplicates and blank detections; RapidTag variants found 7/11 known tags and the other 14
configurations found 11/11. Those misses were retained, not treated as skips.

All backends received the same pixels and truth per frame. All 4,500 frame pixel-hash prefixes and annotation
identities match the October 30 reference; the stored pixel digests are 64-bit SHA256 prefixes, not full digests.
Maximum truth and NAPA detections are both 35, below the output capacity of 64. NAPA settings were 22/16/32,
auto splitting, one worker, no shadow recovery or small-tag regrowth. All four observed process threads were
restricted to CPU2 at both observation points; each backend's configured worker count was one. This does not
mean the process contained only one OS thread.

## Monitor observations

The monitor retained 1,926 samples, with intervals of 5–6 seconds (median 5 seconds), from Unix timestamp
1794773966 through 1794783665. Its stderr log is empty. Samples cover the timed run apart from the initial
roughly five seconds and final roughly four seconds.

| Signal | Minimum | Median | p95 | Maximum |
| --- | --- | --- | --- | --- |
| CPU2 frequency, MHz | 2711 | 4200 | 4388 | 4600 |
| CPU2 busy, % | 100 | 100 | 100 | 100 |
| CPU3 sibling busy, % | 0 | 0 | 0 | 16 |
| Package temperature, °C | 75 | 91 | 94 | 97 |
| TCPU temperature, °C | 83 | 91 | 91 | 91 |
| System 1-minute load | 0.59 | 2.35 | 3.97 | 4.76 |

CPU2 frequency mean was 4167.79 MHz; its 5th percentile was 3865.25 MHz. CPU3 busy mean was 0.0561%; one
sample exceeded 10%, and none exceeded 50%. The recorded governor was powersave, EPP balance_performance,
and turbo enabled. Frequency was variable and package temperatures were high. These samples do not contain
throttle counters, per-call frequency, power or per-process sibling attribution: they cannot establish absence
of thermal or power throttling, nor attribute frequency variation to a particular cause. Five-second snapshots
can miss short excursions. Other system activity was present, and this was not an isolated thermal experiment.
Keep these limitations with any interpretation of the local i7-12700H timings. No new Raspberry Pi measurement
was performed.

The main NAPA configurations each returned one wrong ID on the same tag-free decoy frame
`decoys-640x480-042` (ID 323); no duplicates. Stress and range had zero NAPA wrong IDs or duplicates.
Quality counts each frame once. Timing percentiles pool all timed calls; speed ratios use medians of paired
per-frame median WPILib/detector ratios. Conditional corner errors are not an unconditional accuracy metric.

See [report](report.md), [analysis](analysis.json), [raw CSV](matrix.csv.gz), [raw metadata](matrix.json),
[protocol](protocol.json), [preflight](validate.json) and [monitor](monitor.tsv).
