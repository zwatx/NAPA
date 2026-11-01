# split_touching on independent benchmarks — 2026-10-31

`split_touching=True` finds tags whose border touches another dark shape at a corner, as in AprilGrid boards and
tags set in a checkerboard. It is off by default. With it off, detection is unchanged (same output, same speed).
With it on, every ordinary detection is still returned first, unchanged, and the new tags are appended.

A frame first splits only where shapes touch at a corner. Once it finds a tag with dark shapes just beyond its
corners, as every tag on a board has, it starts over and also splits one-pixel necks, which anti-aliasing leaves at
such corners and which cost far more to split. Frames without touching tags, most FRC frames, never take that step.

## Method

- Source: branch `feat/split-touching-tags`, evaluator `benchmarks/external.py` SHA-256 `fb28380e…`, library SHA-256
  `3b76310f…` (all in the raw files). The raw files record core source `6855eca7…`; later comment-only edits give
  the committed `d63c153f…`, whose library has identical code and data sections (only debug information and the
  build ID differ). GCC 15.2, `setup.py` flags, one thread pinned to one core
  of an Intel i7-12700H.
- Data and scoring as in the [baseline report](../2026-10-31_external_baseline/report.md): all 700 Locus-Tag Bench
  images (revision `5346835`) and the 100 ICRA 2020 forward scenes (pure tags and checkerboard), correct ID and
  quad IoU ≥ 0.5, one-to-one, complete truths only. "Within 1 px" counts all four corners within a pixel, over all
  eligible truths including misses. Latency is the evaluator's three shuffled calls per image after a warm-up.
- `napa16` is `min_tag_px=16`; `-split` adds `split_touching=True`. AprilTag 3 runs at full resolution
  (`quad_decimate=1`), as in the baseline report.
- Raw results: `locus.json.gz`, `icra-forward.json.gz` (per-image detections, timings, and hashes).

## Boards

Recall (within 1 px), protocol false positives, and median latency.

| Configuration | napa16 | napa16-split | AprilTag 3, full resolution |
| --- | ---: | ---: | ---: |
| AprilGrid, golden | 81.8% (81.7%), 18 FP, 5.8 ms | **96.9% (96.8%)**, 27 FP, 15.0 ms | 98.0% (97.9%), 31 FP, 75 ms |
| AprilGrid, Brown–Conrady | 79.8% (79.4%), 5 FP, 5.5 ms | **96.2% (95.6%)**, 9 FP, 14.9 ms | 96.1% (89.4%), 10 FP, 73 ms |
| AprilGrid, Kannala–Brandt | 41.8% (16.7%), 7 FP, 3.3 ms | 80.6% (27.1%), 15 FP, 8.6 ms | 94.1% (18.3%), 18 FP, 47 ms |
| ICRA checkerboard | 0.0%, 0 FP, 5.1 ms | **65.0% (65.0%)**, 0 FP, 14.3 ms | 31.5% (31.5%), 0 FP, 115 ms |

| Configuration | napa | napa-split |
| --- | ---: | ---: |
| AprilGrid, golden | 54.5% (54.5%), 1.6 ms | 86.3% (86.3%), 3.9 ms |
| AprilGrid, Brown–Conrady | 40.3% (40.2%), 1.5 ms | 81.0% (80.7%), 3.8 ms |
| AprilGrid, Kannala–Brandt | 23.3% (8.5%), 0.8 ms | 69.0% (22.2%), 2.9 ms |
| ICRA checkerboard | 0.0%, 1.1 ms | 31.2% (31.2%), 1.8 ms |

The eight single-tag Locus configurations and ICRA's pure-tag scenes are unchanged in recall, false positives and
corners at both settings. ICRA's tags are mostly under 10 px, so `min_tag_px=16` is the relevant setting there.

**False positives.** Every added protocol false positive (40 over both settings) decodes with no bit errors and a
decision margin of at least 168, has an ID absent from that image's annotation, and overlaps no annotated tag.
Inspected examples are fully visible tags at the board's edge that the dataset doesn't annotate. They are counted as
false positives above anyway.

**Fisheye.** Kannala–Brandt boards remain the weak case: corners are fitted with straight sides, so even found tags
are rarely within a pixel, and recall trails AprilTag's.

## Safety

- **Random-payload boards.** `grid_decoys.py` renders AprilGrid boards (3×3 to 6×6 tags, up to 45° tilt, blur and
  noise) whose payloads are at least 5 bits from every tag36h11 code. 1,200 boards, 25,835 decoys: no detections at
  either setting, with or without the option. The same boards with real codes (`--real`, 120 boards, 2,517 tags):
  1,864 found at `min_tag_px=16` with the option against 243 without, and 1,209 against 139 at 32, all correct.
  Allowing pieces three corrected bits produced one false ID in 25,835; pieces are therefore held to two.
- **Natural negatives.** 268 Indoor67 photographs and 120 fresh negative frames (six photographs in four
  transforms, 96 procedural frames), every family, both settings: no detection beyond those of the default detector
  (its two are a known tag16h5 background match). The images are not redistributed.
- **Short codes.** Splitting proposes many more blobs. A family's pieces may correct only as many bits as keep chance
  matches no likelier than tag36h11's at two (2.3e-5 per pattern). tag16h5 and tagCircle21h7 aren't split: an earlier
  version found two new chance matches among Locus's tag16h5 images.

## Speed and the FRC set

FRC set: the fixed 188-image selection of the baseline report. Its labels come from another detector, so they
measure agreement, not accuracy. With the option on, agreement rises from 286 to 288 of 363 at the default and from
358 to 360 at 16. On 900 frames of the synthetic corpus, 6 more tags are found at the default and none are lost.

Paired timing of the final build, interleaved per frame, median of seven calls, over the 188 FRC images and 288
synthetic frames. An identical copy of the default library is the noise control.

| Setting | Default library | Identical copy | Option off | Option on |
| --- | ---: | ---: | ---: | ---: |
| FRC, `min_tag_px=32` | 0.43 ms/frame | 1.03× | 0.99× | 1.35× (0.58 ms) |
| FRC, `min_tag_px=16` | 2.14 ms/frame | 1.00× | 0.99× | 1.21× (2.59 ms) |
| Synthetic, `min_tag_px=32` | 0.38 ms/frame | 1.04× | 1.01× | 1.29× |
| Synthetic, `min_tag_px=16` | 1.54 ms/frame | 1.01× | 0.99× | 1.25× |

Splitting every frame at necks as well, as an earlier version did, cost 2.10× and 1.71× on the FRC set for one more
FRC agreement at 16, and the same board recall. A frame that finds touching tags runs twice and can take up to about
5× as long; board frames above take 2.6–2.8× napa16's time. Waiting for touching tags before splitting at stride one too
was tried: it saved another 0.08× at 16 but lost two FRC agreements and 565 checkerboard tags, whose frames hold only
tiny tags.
