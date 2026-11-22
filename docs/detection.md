# Detection

```python
from napa_vision import NapaDetector

detector = NapaDetector()
for tag in detector.detect(frame):  # uint8 greyscale, BGR, or BGRA; strided views are fine
    print(tag.tag_id, tag.corners)  # corners: 4x2 float64
```

Detection needs only NumPy. Poses and calibration need OpenCV.

Greyscale frames are cheapest. Colour frames are converted in the native core, with exactly OpenCV's `BGR2GRAY`
values, which on one Raspberry Pi 5 core costs about 0.6 ms for a 1280x800 frame. For a YUYV camera, pass the luma
plane as a view, `frame[..., 0]` (`frame[..., 1]` for UYVY): it is copied natively in a fraction of that.

Use one detector per camera stream. Do not call the same one from two threads at once. Separate detectors run in
parallel. `close()` frees the buffers early.

## What a detection holds

- `tag_id`, `family`: the ID within its family.
- `corners`: the four sub-pixel corners, OpenCV AprilTag order, pixel-centre coordinates. The centre of the top-left
  pixel is (0, 0). Upstream AprilTag uses that pixel's top-left corner, so its values are 0.5 px larger.
- `hamming`: payload bits corrected. 0 is a perfect read. `None` for a [soft-decoded](#soft-decoding) tag.
- `soft_decoded`: true for a soft-decoded tag.
- `decision_margin`: how far the payload bits sit from the black/white threshold, in grey levels.
- `center`, `homography`: the tag centre, and the map from tag coordinates ([-1, 1] across the tag square) to pixels.

`hamming` and `decision_margin` match upstream AprilTag, so filters written for it carry over. Exact definitions are
in the [API reference](api.md#fiducialimageobservation).

## Choosing settings

Start with the default. It fits most cameras.

```python
detector = NapaDetector()  # min_tag_px=22, no recovery passes
```

On a Raspberry Pi 5, one core, that takes 1.44 ms per 1280x960 frame at the median, 2.25 ms at p99.

| You want | Use |
| --- | --- |
| Most cameras | `NapaDetector()` |
| The speed of NAPA 0.5.1 | `NapaDetector(min_tag_px=32, recover_shadows=False)` |
| Tags in shade or against a bright window | `NapaDetector(recover_shadows=True, shadow_min_tag_px=64)` |
| Calibration boards, tags touching other tags | `NapaDetector(min_tag_px=16, split_touching=True, max_detections=256)` |
| The most tags per frame | The preset below |

Two things the default gives up. It misses 1-3.5% of the 22-37 px tags `min_tag_px=32` finds (the finer grid falls on
them differently), though it still finds more tags overall on every set measured. And it misses more tags whose border
merges into a dim background. That is what the shadow pass in the table is for.

The 0.5.1 settings above keep 0.5.1's speed (1.08x its time on a Pi 5, 1.5x on small crowded frames) with the newer
corner fitting, and found at least as many tags as 0.5.1 everywhere.

### High-accuracy preset

```python
detector = NapaDetector(
    min_tag_px=16,
    split_touching=True,
    recover_shadows=True,
    shadow_min_tag_px=24,
    max_detections=256,
)
```

Finds tags down to about 16 px, touching tags, and tags merged into dim backgrounds. Costs 3-5x the default (6.6 ms
per Pi 5 frame above), more on dense boards. Found up to 17 points more tags than `min_tag_px=16` alone on boards and
real footage. Dropping `shadow_min_tag_px` finds ~2 points more for 1.2-1.8x more time.

Cheaper near-equivalents: drop `split_touching` and use `shadow_min_tag_px=22` (10% faster, equal except on boards),
or add `tag_shape="tall"` instead (30% faster for side-turned tags; see below).

Recovered tags can have looser corners, especially through distorted lenses. Check on your camera before using them
for poses.

## Tuning

| Parameter | Default | Effect |
| --- | ---: | --- |
| `min_tag_px` | 22 | Smallest tag square the search is guaranteed to sample. Lower for distant tags. Cost grows ~1/`min_tag_px`². |
| `max_hamming` | Family's (3) | Bits the decoder may correct, 0-3. A third bit needs the three wrong cells to touch. |
| `min_contrast` | 20 | Smallest black-to-white grey gap. NAPA raises it in noise. Lowering finds fainter tags, slower. |
| `min_border_cells` | 22 | Of 28 border cells (tag36h11), how many must read dark before decoding. |
| `sharpening` | 0.25 | Cell sharpening before decoding. |
| `max_detections` | 64 | Cap per frame. Overflow warns and drops the rest. |
| `split_touching` | `"auto"` | Split frames where an ordinary tag reveals touching shapes; `True` splits every frame. See [Calibration boards](#calibration-boards). |
| `recover_shadows` | `False` | Second low-threshold search for tags in dim backgrounds. Experimental. |
| `shadow_min_tag_px` | `None` | Sampling floor for that search only. Bigger = faster, less sensitive. |
| `soft_decode` | `False` | Read tiny or patched-over tags as whole pictures. tag36h11, experimental. |
| `align_corners` | `False` | Re-fit blurred tags' corners onto their picture. tag36h11, experimental. |
| `camera_gamma` | `None` | Fit corners in linear light for gamma-encoded cameras. tag36h11 only. |
| `tag_shape` | `None` | `"tall"`/`"wide"`: halve sampled rows/columns for turned tags. Needs `min_tag_px` ≤ 31. |
| `regrow_small_tags` | `False` | Regrow small fragmented candidates at full resolution and try them again. Bounded to 32 attempts per pass. |
| `camera` | `None` | The lens, from the camera's calibration: corners follow distortion. See [Lens distortion](#lens-distortion). |
| `threads` | 1 | Threads per `detect` call, for lower latency. Same results. See [Threads](#threads). |

`min_tag_px` sets the sampling step, nothing more. Values giving the same step behave identically (for tag36h11:
16-21, 22-31, 32-42, …). It does not guarantee detection, does not reject smaller tags, and results need not nest
when you change it.

Tag size always means the square whose corners are reported: the black square on classic families, the white square
inside the frame on tagStandard, tagCircle, and tagCustom. That side is also what pose solvers take.

## Calibration boards

On AprilGrid boards a black square touches each tag at its corners, so tag and neighbours form one blob and the
ordinary search drops it. Simple point contacts are separated automatically. The default, `split_touching="auto"`,
runs the ordinary pass first and repeats with splitting when a found tag shows dark shapes beyond three of its
corners. A board where every tag merges has no such cue. To split every frame, including those boards:

```python
detector = NapaDetector(min_tag_px=16, split_touching=True, max_detections=256)
```

Split tags come after ordinary ones. tag16h5 and tagCircle21h7 never split (their codes match random pieces too
often). With `True`, cost is +15-30% on frames with no touching tags and 2.5-3x on boards at `min_tag_px=16`.
Auto keeps the ordinary results when no cue is found; `False` disables splitting.

On the [Locus AprilGrid renders](../benchmarks/results/2026-10-31_split_touching/report.md) this found 96.9% of tags
(AprilTag full resolution: 98.0%), 96.2% vs 96.1% under distortion at a fifth of the time, and 80% vs 94% on fisheye.
Use `True` when auto misses a board; otherwise leave the default.

## Lens distortion

A lens bends a tag's straight sides, most for large tags far from the image centre. Without a calibration, NAPA fits
each side in pixels and corrects the bend afterwards, which leaves a few tenths of a pixel of error on strongly
distorted tags. Given the calibration, it fits each side as a straight line through the lens instead and returns the
corners in the frame's own pixels. The image is never remapped.

```python
detector = NapaDetector(camera=calibration)  # a CameraCalibration
detector = NapaDetector(camera=(camera_matrix, distortion))  # OpenCV's standard model
detector = NapaDetector(camera=(camera_matrix, distortion, "fisheye"))  # cv2.fisheye's model
```

Use the calibration made at the frame's resolution; a `CameraCalibration` that records it rejects frames of any other
size. `VisionProcessor` gives its calibration to the detector it creates.

On rendered frames with exact truth, a fisheye lens's worst-corner error fell from 0.17 to 0.04 px (median) and from
0.59 to 0.16 px (95th percentile), and multi-tag camera positions came out about three times closer, through both a
fisheye and a strong standard lens. A calibration with 20% error in `k1`, or 10% in focal length, kept most of the gain;
a badly wrong one was about as good as none. Tags the lens leaves straight to within 0.02 px are fitted exactly as
without a camera. Bowed tags cost a little more to fit (about 4% of frame time on frames full of large tags through a
strong standard lens); through a fisheye lens, where the after-the-fact corrections it replaces ran on most tags,
detection got faster.

## Threads

`NapaDetector(threads=4)` lets one `detect` call use four threads, the caller's included, for lower latency on each
frame. The detections are identical at every thread count. The search is split into horizontal bands that are joined
exactly. Each candidate's decode and refinement also run ahead on every thread, and the candidates are then taken in
their usual order, so a frame with several tags no longer refines them one after another.

On a Raspberry Pi 5 with 1920x1080 FRC frames, four threads took the default from 1.64 to 1.02 ms per frame and
`min_tag_px=16` from 3.39 to 1.99 ms. Search levels below about 100,000 samples, and `split_touching`'s labelling, run
on one thread. Decoding ahead needs at least 8 candidates on a level. It is off with `camera_gamma`, a camera lens or
`soft_decode`, and with `regrow_small_tags` until the frame's regrowth budget is spent. With several cameras already
keeping the cores busy, leave it at 1 and run one detector per camera instead.

## Tag families

One detector reads one family: `NapaDetector(family="tagStandard41h12")`. `napa_vision.families()` lists all nine.
Use one detector per family to read several. Families with close-together codes correct fewer bits by default, so
random patterns stay as unlikely to decode as in tag36h11. Sizes and defaults:
[API reference](api.md#tag-families).

## Tall or wide tags

Tags turned about the vertical axis look narrow and tall but keep their height, so the search can sample half as many
rows:

```python
detector = NapaDetector(min_tag_px=16, tag_shape="tall")
```

`"wide"` is the same idea for a camera on its side. Fitting and decoding still read the whole frame.

On a Pi 5 (99 robot-camera frames, 1280x960, with the coarse shadow pass): `min_tag_px=16, "tall"` found 231 tags in
2.20 ms/frame, vs 232 in 4.24 ms plain and 226 in 1.73 ms at the default. At `min_tag_px=22`, `"tall"` roughly matches
`min_tag_px=32` cost and finds more narrow tags, fewer overall.

Warnings: the wrong direction is worse than unset (6-11% fewer tags on rotated footage, and slower). A 45° rolled
camera loses little either way. Steeply pitched cameras are untested. The smallest tags suffer: 60% found at 12-16 px
vs 100% at plain `min_tag_px=16`.

## Experimental options

All four are off by default, tag36h11-oriented, and leave ordinary detections first and unchanged. Each can also go
wrong in ways ordinary detection cannot. Test on your footage before relying on one.

### Shadow recovery

`recover_shadows=True` repeats the search with a lower threshold so a black border can separate from a dim background.
Fitting and decode checks do not change, and added tags must match a code exactly.

- `shadow_min_tag_px=64`: cheap pass for larger tags, ~1.2-1.5x total time. Start here.
- No `shadow_min_tag_px`: full sensitivity, roughly 2-3x total time.
- tag16h5 and tagCircle21h7 skip the pass. Recovered corners can be worse (one sweep: worst 0.80 → 2.54 px at
  recovery floor 24). Not a false-positive guarantee.

Evidence is mostly synthetic (64% → 78% of boxes found on rendered backlit frames). On real footage the gain at
default size was small: a handful of tags across ~1,800 FRC and robot frames, ~3% more sightings in window-lit rooms.

### Soft decoding

`soft_decode=True` reads tags as whole pictures when blur leaves too few pixels per cell to call bits one by one. It
targets 8-19 px tags, so use it near `min_tag_px=16`. Results have `soft_decoded=True`, `hamming=None`.

Gains measured (with split and shadow recovery on): rendered small tags 2,231 → 2,435 of 3,080; photo composites 521 →
565 of 783; one real test 514 → 523 of 1,039, another unchanged. No wrong IDs in those runs.

But it can invent tags: 1 and 4 false reports across two sets of ~70,000 random-pattern squares (ordinary decoding:
2 each). Do not act on a soft detection without a cross-check, e.g. expected IDs or a stable position over frames.

It also reads tags with a flat patch over part of the payload, at any size (43% → 62% found on patched renders; a
pattern built to fool it still produced ~7 false tags per 100,000 squares). Real-camera corner accuracy is unmeasured.
Typical cost under 0.1 ms/frame, up to ~56 ms on frames full of tiny tags.

### Corner alignment

`align_corners=True` re-fits blurred tags' corners after decoding, using each tag's own blur and tone. Counts do not
change. It mainly helps motion blur: worst-corner p95 on rendered gamma/sRGB motion blur fell about 1.0 → 0.7 px and
1.9 → 0.8 px. No change on linear images. Cost ~4% on typical FRC frames.

Real-camera accuracy is unmeasured. On 637 real tags it moved 8 by up to 0.5 px, with no truth to judge by.

### Camera gamma

Webcams and phones store brightness on a curve (~2.2), so blur makes black borders read narrow and every tag reads a
few percent small: a direct range error. `NapaDetector(camera_gamma=2.2)` fits corners in linear light instead.

Only set this for a gamma-encoded camera. On a linear (machine-vision) camera it makes corners much worse. Unsure?
Measure:

```python
from napa_vision import NapaDetector, estimate_camera_gamma

estimate = estimate_camera_gamma(frames)  # 30+ frames with a tag36h11 tag; a little blur helps
if estimate.gamma is not None and estimate.gamma > 1.0:
    detector = NapaDetector(camera_gamma=round(estimate.gamma, 1))
```

The estimate is rough: within 0.04 on renders, about 10% low on real optics, and unable to tell a steep curve from
compression or in-camera sharpening. Treat it as a range of a few tenths. `gamma` 1.0 means linear: leave the option
unset.

Payoff, worst-corner p95 on rendered motion-blur tags: 1.1-1.9 px unset → 0.1-0.3 px at the true gamma. Half a point
off still beats unset. On re-encoded real frames, mean worst-corner distance fell 0.93 → 0.08 px. Dark tags and
strongly lens-bowed tags benefit less, and bowed tags occasionally get worse. Cost is small (±10% of frame time).

## Switching from pupil_apriltags

```python
from napa_vision.compat import Detector  # was: from pupil_apriltags import Detector

detector = Detector(families="tag36h11")
detections = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=0.1651)
```

Same constructor, same `Detection` attributes, upstream conventions (corner order, +0.5 px origin, `pose_R/t/err`).
`decode_sharpening` works, and `nthreads` sets the worker count (1–16). `quad_decimate`, `quad_sigma`, `refine_edges`,
`debug`, and `searchpath` are accepted and ignored. Extra keyword arguments go to `NapaDetector`; an explicit `threads`
overrides `nthreads`. Poses need OpenCV and ignore lens distortion, as upstream does.

## Printing tags

```bash
napa-tag 0-7 --size-mm 165.1 --output tags/   # SVG, exact print size (165.1 mm = FRC)
napa-tag 1,4,9 --format png --cell-px 40
napa-tag 0-3 --family tagStandard41h12 --size-mm 100
```

`--size-mm` is the tag-size square. Print at 100% and measure what came out.

From Python: `napa_vision.tags.tag_image()`, `tag_svg()`, `tag_cells()`. Artwork follows the official orientation.
OpenCV's `DICT_APRILTAG_36h11` draws tags turned 180°.
