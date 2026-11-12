# Detection

```python
from napa_vision import NapaDetector

detector = NapaDetector()
for tag in detector.detect(frame):  # uint8 greyscale, BGR, or BGRA; strided views are fine
    print(tag.tag_id, tag.corners)  # corners: 4x2 float64
```

Detection needs only NumPy. If OpenCV is installed it converts BGR frames. Poses and calibration need OpenCV.

One detector reuses its buffers between frames. Use one per camera stream, and do not call the same one from two
threads at once. The native call releases the GIL, so separate detectors run in parallel. Call `close()` to free
the buffers early.

## What a detection holds

| Field | Meaning |
| --- | --- |
| `tag_id` | The ID within the family |
| `corners` | The four sub-pixel corners, in OpenCV's AprilTag order and pixel-centre coordinates. The centre of the top-left pixel is (0, 0). Upstream AprilTag puts (0, 0) at that pixel's top-left corner, so its corners are 0.5 px larger in both axes. |
| `hamming` | Payload bits that were corrected. 0 is a perfect read. `None` for a tag from [soft decoding](#soft-decoding), where no bits were counted. |
| `soft_decoded` | `True` for a tag from soft decoding |
| `decision_margin` | The smaller of the white and black payload bits' mean distance from the decision threshold, in grey levels |
| `family` | The tag family, `"tag36h11"` unless the detector reads another |
| `center` | Where the tag square's diagonals cross, in pixels. Under perspective this differs from the corners' mean. |
| `homography` | The map from upstream tag coordinates ([-1, 1] across the tag square) to pixels |

`hamming` and `decision_margin` are computed the way upstream AprilTag computes them, so filters written for
AprilTag carry over. The [API reference](api.md#fiducialimageobservation) defines every field exactly.

## Choosing settings

Most cameras need nothing beyond the default:

```python
detector = NapaDetector()
```

That is `min_tag_px=22` with no recovery passes. On a Raspberry Pi 5, one core, 1280x960 robot-camera frames take
1.44 ms at the median and 2.25 ms at the 99th percentile. Four processes on four cores run 2,380 frames per second
at 1280x800.

| You want | Use |
| --- | --- |
| The default for most cameras | `NapaDetector()` |
| The speed of NAPA 0.5.1 | `NapaDetector(min_tag_px=32, recover_shadows=False)` |
| Tags in shade or against a bright window | `NapaDetector(recover_shadows=True, shadow_min_tag_px=64)` |
| Calibration boards, where tags touch | `NapaDetector(min_tag_px=16, split_touching=True, max_detections=256)` |
| The most tags per frame | The [high-accuracy preset](#high-accuracy-preset) below |

Two honest warnings about the default. First, it misses about 1 to 3.5% of the tags `min_tag_px=32` finds at 22 to
37 px wide, because the finer search grid falls on them differently. About half of those come back at
`min_tag_px=16`. On every set measured, 22 still found more tags in total. Second, without the shadow pass it finds
fewer tags whose border merges into a dim background. On a synthetic set built around such scenes it found 64% of
the boxes, against 76% for `min_tag_px=32` with the coarse shadow pass.

### The speed of 0.5.1

NAPA 0.5.1 defaulted to `min_tag_px=32` with no shadow pass. Those settings keep that speed and gain the newer
corner fitting. On a Pi 5, over eight sets of camera frames and tag boards, they took 1.08x as long as 0.5.1's
default, and 1.02x on one laptop core. Small frames with many small tags are the exception. Public 640x360 FRC
images took 1.5x as long. They found at least as many tags as 0.5.1 on every set.

### Dim or backlit venues

A tag in shade, or in front of a bright window, can have a white margin as dark as the wall behind it. Its black
border then merges into the background and the ordinary search misses it. The shadow pass searches again with a
lower threshold to separate them.

```python
detector = NapaDetector(recover_shadows=True, shadow_min_tag_px=64)
```

With `shadow_min_tag_px=64` this is a cheap coarse pass for larger tags. On a Pi 5 it adds about 0.4 ms per
1280x960 frame at the default size. Leave `shadow_min_tag_px` out for the full pass, which costs more. Details and
limits are under [Shadow recovery](#shadow-recovery).

Most of the evidence for this pass is synthetic. On rendered FRC frames built with such scenes it raised boxes
found at the default size from 64% to 78%. On real footage it adds little at the default size: 4 tags over 585
robot-camera frames, 14 over 1,188 FRC frames, and about 3% more sightings on indoor recordings with windows. Try
it on your own venue's footage before you turn it on.

### High-accuracy preset

For the most tags per frame at a still-low cost, combine the recovery options:

```python
detector = NapaDetector(
    min_tag_px=16,
    split_touching=True,
    recover_shadows=True,
    shadow_min_tag_px=24,
    max_detections=256,
)
```

This finds tags down to about 16 px wide, including tags that touch other dark shapes and tags whose border merges
into a dim background. Ordinary detections come first. Recoveries follow. On one core it takes about 1.7 to 2.1x
as long as `min_tag_px=16` alone, and about 3 to 5x as long as the default, more on dense boards. On a Pi 5 that
is 6.6 ms per 1280x960 robot-camera frame. On dense boards and real footage it found up to 17 percentage points
more tags than `min_tag_px=16` alone, and it added nothing on frames without tags. Every tag `min_tag_px=16` found
in those tests was still found.

Dropping `shadow_min_tag_px` finds up to about 2 percentage points more and takes about 1.2 to 1.8x as long again.

Two cheaper variants find nearly as much:

- `NapaDetector(min_tag_px=16, recover_shadows=True, shadow_min_tag_px=22)` takes about 10% less time. It matches
  the preset on robot-camera and FRC footage, within half a percentage point. It finds fewer tags on calibration
  boards, where tags touch.
- `NapaDetector(min_tag_px=16, tag_shape="tall", recover_shadows=True, shadow_min_tag_px=22)` takes about 30% less
  time. It suits an upright camera that sees tags turned to the side. It finds fewer tags on public FRC images and
  boards. See [Tall or wide tags](#tall-or-wide-tags).

Recovered tags can have looser corners than ordinary ones, especially through strongly distorted lenses. Check the
preset on your own camera before you use it for poses.

## Tuning

| Parameter | Default | What it does |
| --- | ---: | --- |
| `min_tag_px` | 22 | Smallest side of the tag-size square, in pixels, that the search is guaranteed to sample. Lower it for distant tags. Work grows roughly with 1 / `min_tag_px` squared. |
| `max_hamming` | Family's (3 for tag36h11) | Payload bits the decoder will correct, 0 to 3. A third bit is corrected only when the three wrong cells touch, as they do under a finger, cable, or glare spot. Lower it to reject more. |
| `min_contrast` | 20 | Smallest grey-level gap between the tag's black and its white margin. NAPA raises it on its own where the frame is noisy or textured. Lowering it finds fainter tags in washed-out scenes and costs speed on every frame, because sensor noise starts passing the threshold too. |
| `min_border_cells` | 22 | How many of tag36h11's 28 border cells must read dark before a candidate is decoded. Scaled for other families. |
| `sharpening` | 0.25 | Cell sharpening before decoding, as upstream's `decode_sharpening` |
| `max_detections` | 64 | Results returned per frame. A frame with more tags warns (`RuntimeWarning`) and drops the rest. |
| `split_touching` | `False` | Also decode tags whose border touches another dark shape. Slower. See [Calibration boards](#calibration-boards-and-touching-tags). |
| `recover_shadows` | `False` | Add the shadow pass for tags merged into dim backgrounds. Experimental. See [Shadow recovery](#shadow-recovery). |
| `shadow_min_tag_px` | `None` | Sampling minimum for the shadow pass only, from `min_tag_px` to 30000. Requires `recover_shadows=True`. Larger is faster and less sensitive. |
| `soft_decode` | `False` | tag36h11 only. Add tiny tags read as a whole picture, and tags with a flat patch over part of the payload. Experimental. See [Soft decoding](#soft-decoding). |
| `align_corners` | `False` | tag36h11 only. Move blurred tags' corners onto their own picture after decoding. Experimental. See [Corner alignment](#corner-alignment). |
| `camera_gamma` | `None` | tag36h11 only. The gamma of a camera that stores gamma-encoded video. Leave unset for a linear camera. See [Camera gamma](#camera-gamma). |
| `tag_shape` | `None` | `"tall"` halves the rows the search samples, for tags turned about the vertical axis. `"wide"` halves the columns, for a camera on its side. Needs `min_tag_px` of 31 or less for tag36h11. See [Tall or wide tags](#tall-or-wide-tags). |

`min_tag_px` sets the sampling grid. It does not guarantee detection, and it does not filter out every smaller tag.
Only the sampling step matters, so values that give the same step behave the same. For tag36h11, 16-21, 22-31,
32-42, 43-53, and 54-63 each give one result. A tag above the size can still be missed when its margin merges into
a background, or when fitting and decoding fail. Lowering the minimum changes proposals and fits, so outputs need
not nest. A larger `min_tag_px` can find a tag a smaller one misses.

The tag-size square is the black square for the classic families. For tagStandard, tagCircle, and tagCustom it is
the white square inside the black frame. Its side is also the tag size you give the pose solvers.

## Calibration boards and touching tags

In an AprilGrid board, and in tags set in a checkerboard, a black square touches each tag's border at its corners.
The tag and its neighbours then form one blob, which is not tag-shaped, so the ordinary search drops it.

Ordinary detection separates a point contact for free when nearby samples support two filled dark corners with
light samples between them. Contacts without that support stay joined, as do all overlapping runs.

`split_touching=True` goes further. It splits ambiguous contacts and narrow necks left by anti-aliasing, blur, and
perspective, then decodes the pieces:

```python
detector = NapaDetector(min_tag_px=16, split_touching=True, max_detections=256)
```

Those tags come after the ordinary ones, which are unchanged. Pieces may correct at most two bits, and fewer for
families whose codes sit closer together. tag16h5 and tagCircle21h7 are never split, because random pieces match
their short codes too often.

Costs vary. A frame splits only at corners until it finds a tag with dark shapes just beyond three of its corners.
Then it starts over and also splits one-pixel necks, which costs much more. On FRC frames without touching tags the
option adds about 15 to 30% to the time. At `min_tag_px=16`, boards take 2.5 to 3x as long as without it.

Measured on the independent [Locus AprilGrid renders](../benchmarks/results/2026-10-31_split_touching/report.md),
with `split_touching=True` at `min_tag_px=16`, NAPA found 96.9% of the tags against 98.0% for AprilTag at full
resolution. Under lens distortion it found 96.2% against 96.1%, in about a fifth of the time. Fisheye boards remain
harder, at 80% against AprilTag's 94%.

Leave it off unless your tags touch something dark.

## Tag families

A detector reads one family. Pass another name to read a different one:

```python
detector = NapaDetector(family="tagStandard41h12")
```

`napa_vision.families()` lists all nine families upstream AprilTag defines. To read several, use one detector per
family.

Families whose codes sit close together get less error correction by default, so a random pattern is no likelier to
decode as a tag than it is in tag36h11. The API reference lists [each family's size and default
correction](api.md#tag-families).

## Tall or wide tags

A tag turned about the vertical axis, as field tags look from the side, appears narrow and tall. It keeps its
height, so the search can sample half as many rows and keep full sampling across the narrow side:

```python
detector = NapaDetector(min_tag_px=16, tag_shape="tall")
```

Use `"wide"` for a camera mounted on its side, where the same tags look short and wide. Fitting and decoding still
read the whole frame. The option needs `min_tag_px` of 31 or less for tag36h11. At 16 the search samples every
pixel across and every other row. At 22 it samples every other pixel across and every fourth row.

On a Pi 5, over 99 frames from one robot camera at 1280x960, all with the coarse shadow pass:

| Setting | Time per frame (median) | Tags found |
| --- | ---: | ---: |
| `min_tag_px=32` | 1.11 ms | 207 |
| `min_tag_px=22` | 1.73 ms | 226 |
| `min_tag_px=22, tag_shape="tall"` | 1.21 ms | 216 |
| `min_tag_px=16, tag_shape="tall"` | 2.20 ms | 231 |
| `min_tag_px=16, tag_shape="wide"` (wrong direction here) | 2.38 ms | 226 |
| `min_tag_px=16` | 4.24 ms | 232 |

On one laptop core over 13 sets, `min_tag_px=16` with `"tall"` took 1.75x as long as `min_tag_px=32`, against 1.40x
for `min_tag_px=22` and 2.71x for plain `min_tag_px=16`. It found 63 to 75% of what `min_tag_px=16` adds over 22 on
GoPro recordings and FRC robot footage, and 34 to 36% on public FRC images. Corners matched `min_tag_px=16` and 22
on rendered tags with exact positions. It reported no wrong IDs and no detections on frames without tags. The one
measured loss is the smallest tags. Of tags 12 to 16 px across in a range sweep, it found 60% against 100% at
`min_tag_px=16`.

Three cautions:

- The direction matters. On frames turned 90 degrees, the wrong direction found 6 to 11% fewer tags than
  `min_tag_px=22` on GoPro footage and FRC images, while taking longer.
- A rolled camera loses little. With frames rolled 45 degrees, both directions found about as many tags as
  `min_tag_px=22`, or more.
- A camera pitched steeply up or down at the tags has not been tested.

At `min_tag_px=22`, `"tall"` costs about what `min_tag_px=32` does (1.21 ms against 1.14 ms in the same run, with a
lower 99th percentile) and finds more narrow tags, but fewer tags overall than the plain default. It is off by
default and not part of the high-accuracy preset.

## Experimental options

Four options are off by default and labelled experimental. Each one appends to, or refines, the ordinary result
without changing what ordinary detection returns. Each one also has a failure mode the ordinary path does not. Read
the risk before you enable one, and check it on your own footage.

### Shadow recovery

`recover_shadows=True` runs ordinary detection first, then searches again with a proposal threshold one quarter of
the local intensity range above its minimum. That can separate a black border from a dim background that joins it
at the ordinary midpoint threshold. The pass does not change the image used for fitting or decoding, and it does
not relax the contrast, border, or decode checks. Added tags must match a code exactly, with zero corrected bits.

Guarantees and limits:

- Ordinary detections stay first and are exactly what the same settings find without recovery. Recovery appends up
  to `max_detections`. A full ordinary result buffer skips recovery.
- With `split_touching=True`, ordinary split results keep priority. The shadow pass itself does not split.
- An allocation failure raises an error instead of returning a partial success.
- tag16h5 and tagCircle21h7 skip the pass. Their exact-code chance-match rate is too high.
- Recovery is incomplete. Severe shadows, blur, and payload errors still prevent detection. It is also no guarantee
  against false positives.

Cost depends on `shadow_min_tag_px`. At 64 it adds about 1.2 to 1.5x the time of ordinary detection. At full
sensitivity (`shadow_min_tag_px=None`, or a value equal to `min_tag_px`) single-core tests took roughly 2 to 3x as
long. An explicit value changes only the recovery sampling. The schedule is quantized and family-dependent, so
some adjacent values behave the same, while others shift the sampling phase. Recovered tags can be gained, lost,
or get different corners as it changes. It is not a hard size filter, and recovered sets need not nest. In an
early synthetic shadow sweep, a recovery minimum of 24 at ordinary 16 raised the worst recovered corner error from
0.80 to 2.54 px and lost some recoveries. Use full sensitivity when that trade is wrong for you.

### Soft decoding

Tiny tags often cannot be read cell by cell. Blur leaves too few pixels per cell to call each bit. With
`soft_decode=True` (tag36h11 only), each kept decode attempt is compared as a whole picture against every tag36h11
code and against patterns tags are often confused with. A winner has its corners aligned to the code's picture, is
scored again from scratch, and must come back as the same code in the same rotation. Its payload cells are then
fitted freely, allowing for blur, and the fit must agree with the code.

These detections follow the ordinary ones, which are unchanged. They carry `soft_decoded=True` and `hamming=None`.
No bits were counted, so they are not exact matches. `decision_margin` is measured on the aligned corners without
sharpening. The option targets tags of about 8 to 19 px, so it only helps near `min_tag_px=16`. At `min_tag_px=32`
it rarely fires. It was tested with `split_touching` and `recover_shadows` on:

| Frames | Without | With soft decoding |
| --- | ---: | ---: |
| Rendered small-tag sequences, tags found of 3,080 | 2,231 | 2,435, no false detections |
| 53 photo composites, tags found of 783 | 521 | 565, none wrong or false |
| Real frames, boxes matched of 1,039 (Roboflow test) | 514 | 523, none unmatched |
| Real frames, boxes matched of 573 (Drive test) | 517 | 517, none unmatched |

It can report tags that are not there. Dark squares filled with random patterns are the hardest case tested. Across
two fresh sets of about 70,000 of them, soft decoding reported one and four false tags. Ordinary decoding reported
two on each set. Eight structured patterns (blank squares, two-tone squares, logos) produced none on 320 frames,
and none appeared on 388 tag-free indoor frames. Those checks come from one pattern generator, so they do not
bound the rate on your camera. Check soft detections against something else before you act on them, such as the
expected tag IDs or a position that holds still over several frames.

It also reads tags with part of the payload covered, at any size. A failed read is accepted when one code explains
every wrong cell with a single rectangle of at most 16 cells whose samples are nearly flat, and no other code can
do the same. The corners then get the ordinary side fit, and the fitted tag must explain itself the same way. On
renders with flat patches over part of a tag, tags found rose from 43% to 62%, with no wrong IDs. Worst-corner
error was 0.015 px at the median and 0.15 px at worst. The test obstructions were flat patches, so textured ones
such as fingers or cables will do worse. A pattern built to fool this rule (random cells with a flat rectangle)
produced about 7 false tags per 100,000 dark squares. Plain random patterns produced none in 528,000. Real
tag-free frames produced none, though that sample is small.

No camera frames with known tag positions were available, so corner and pose accuracy on real cameras is
unmeasured. On composited tags the worst-corner error was 0.19 px at the median and 0.38 px at worst. The option
adds under 0.1 ms to typical frames at `min_tag_px=16`, and about 7 ms on average to frames full of tiny tags, up
to about 56 ms. The first frame that needs it builds about 700 KB of tables.

### Corner alignment

`align_corners=True` (tag36h11 only) moves the corners of blurred tags after decoding. It measures each tag's blur
along both axes and the camera's tone curve from the tag itself, then pulls the corners onto the picture that blur
and tone predict. The new corners stay only if the match improves and the tag still decodes to the same ID. Tag
counts do not change.

It mainly helps motion blur, which a fit along each whole side cannot follow. On rendered tags with motion blur,
the worst corner's 95th percentile fell from about 1.05 to 0.72 px on gamma-encoded images and from about 1.95 to
0.79 px on sRGB ones. Motion blur on linear images did not change. On tags composited onto photos, the median
worst corner fell from about 0.23 to 0.16 px. Sharp tags are skipped. Tags too crowded by neighbours to show their
white ring are left alone. In rendered sequences run through the pose solvers, it cut wrong-branch single-tag
poses from about 6% of frames to 3.5 to 5% and lowered median multi-tag translation error by 17 to 30%. Those
renders' blur and tone match what the alignment models, so they flatter it.

It costs time on blurred tags. At the default settings it adds about 4% to typical FRC frames (about 0.02 ms), and
about 25% on one set of small frames. Its accuracy on a real camera is unmeasured. On 637 tags from one real camera
it moved 8 of them by up to about 0.5 px, and there is no ground truth to say whether those moves helped.

### Camera gamma

Most USB webcams, phones, and action cameras store brightness on a curve (gamma about 2.2) instead of in proportion
to light. Blur mixes light, not stored values. On these cameras a blurred black border reads narrower than it is,
so corners move inward and every tag reads a few percent small. That is a direct range error, and it is largest
along motion streaks.

`NapaDetector(camera_gamma=2.2)` (tag36h11 only) undoes the curve on a copy of the pixels around each tag, then
fits the corners there. If that fit or decode fails, the tag is refined from the raw pixels as usual. Leaving the
option unset gives exactly the same results as before.

Set it only for a camera that stores gamma-encoded video. Machine-vision cameras in linear (raw) mode need it unset.
A value on a linear camera moves corners off their edges, far worse than leaving it unset. If you are not sure,
measure it:

```python
from napa_vision import NapaDetector, estimate_camera_gamma

estimate = estimate_camera_gamma(frames)  # frames from one camera at its usual settings
if estimate.gamma is not None and estimate.gamma > 1.0:
    detector = NapaDetector(camera_gamma=round(estimate.gamma, 1))
```

`estimate_camera_gamma` needs at least 30 frames showing a tag36h11 tag (`min_frames`). A little blur helps. It
returns `gamma=None` with fewer usable frames, and 1.0 when the camera reads as linear, which means leave the
option unset. Otherwise it returns the gamma with `low` and `high`, the middle 80% of estimates from resampled
frames. That range covers frame-to-frame scatter only. On a real camera the estimate can sit about 10% below the
true value, outside that range. It runs about eight detections per frame. Tags blurred over more than a cell are
left out, because their border reads narrow at any gamma.

How accurate is the estimate? On rendered frames it read gammas of 1.3 to 3.0 to within 0.04, and called 1.12
linear. Real optics read lower. Frames from a linear machine-vision camera, re-encoded with known curves, read
1.61, 1.97, and 2.32 for true gammas of 1.8, 2.2, and 2.6. The original frames correctly read as linear. Two real
camera recordings read 2.42 and 2.39 on one camera and 2.51 and 2.56 on the other, with spreads of about 0.1.
That is steeper than sRGB's 2.2. The estimate cannot tell a steep curve from video compression or in-camera
sharpening, which also narrow the border. Take the answer as a range of a few tenths, not an exact value.

Measured on rendered tags with exact corners, across two seeds. Values are the 95th percentile of each tag's
worst-corner error. Rendered rows have 20 tags per seed. Composite rows have 100 to 280 tags per seed on indoor
photos.

| Frames | Unset | Set to the true 2.2 | Set 0.4 off (1.8 / 2.6) |
| --- | ---: | ---: | ---: |
| Motion blur, gamma 2.2 | 1.08-1.19 px | 0.13-0.19 px | 0.29-0.40 / 0.48-0.57 px |
| Motion blur, sRGB | 1.65-1.87 px | 0.13-0.29 px | 0.30-0.61 / 0.33-0.47 px |
| Even blur, gamma 2.2 | 0.38-0.39 px | 0.18-0.22 px | 0.32-0.41 / 0.44-0.47 px |
| Composites, gamma 2.2 | 0.49-0.72 px | 0.32-0.48 px | 0.35-0.36 / 0.48-0.55 px |
| Motion blur, linear camera | 0.09-0.12 px | 1.60-1.94 px | 1.18-1.67 / 1.94-2.33 px |
| Composites, linear camera | 0.37-0.38 px | 1.13-1.18 px | 0.93-0.97 / 1.23-1.35 px |

Tag counts did not change, apart from one extra composite tag at 1.8. Being 0.4 off still beats unset on motion
blur. On evenly blurred tags, a value too high is slightly worse than none.

Real frames give a check without exact corners. Frames from a linear camera were re-encoded with a known curve,
and each tag's corners were compared with the detector's own corners on the original linear frame. Distances are
each tag's worst corner, over 297 frames from one camera and session:

| Curve applied | Unset | `camera_gamma=2.2` |
| --- | ---: | ---: |
| Gamma 2.2 | 0.93 px mean, 2.77 p99 | 0.08 px mean, 0.49 p99 |
| sRGB | 1.02 px mean, 2.78 p99 | 0.09 px mean, 0.50 p99 |
| Gamma 1.8 (setting 0.4 too high) | 0.73 px mean, 2.31 p99 | 0.29 px mean, 0.77 p99 |

Limits:

- Very dark tags get no benefit. A tag is refined as if unset when undoing the curve leaves less than
  `min_contrast` between its black and white.
- Lens-bowed tags are mixed. Near the frame edge, median errors fell, often a lot, but the tails went both ways.
  With strong bowing, 71 of 479 tags improved by more than 0.3 px, 22 got worse, and the worst got 2.1 px worse.
- Every accuracy number above comes from renders or re-encoded frames. No camera frames with known tag positions
  were available.

The option costs little time. Set to 2.2, mean time per frame across four sets ranged from 11% slower (+0.025 ms)
to 6% faster. A detector with it set also keeps a buffer of one byte per pixel of the frame, allocated on first
use. `estimate_camera_gamma` took about 10 ms per 640x480 frame.

## Switching from pupil_apriltags or dt-apriltags

`napa_vision.compat.Detector` has the same constructor and `detect` method as `pupil_apriltags.Detector`. It
returns `Detection` objects with the same attributes, in upstream's conventions: corners in upstream order, (0, 0)
at the top-left pixel's corner, and poses (`pose_R`, `pose_t`, `pose_err`) from upstream's tag frame into the
OpenCV camera frame.

```python
from napa_vision.compat import Detector  # was: from pupil_apriltags import Detector

detector = Detector(families="tag36h11")
detections = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=0.1651)
```

`families` takes any of upstream's families, space-separated, as upstream does. `decode_sharpening` is used.
`nthreads`, `quad_decimate`, `quad_sigma`, `refine_edges`, `debug`, and `searchpath` are accepted and ignored,
because NAPA never decimates and always refines. NAPA's own parameters, such as `min_tag_px=24`, pass through as
extra keyword arguments. Poses need OpenCV and, like upstream's, ignore lens distortion.

## Printing tags

`napa-tag` writes tags to print. For SVG, `--size-mm` is the side of the tag-size square, the same measurement the
pose solvers take as the tag size:

```bash
napa-tag 0-7 --size-mm 165.1 --output tags/      # tags 0-7 as SVG, 6.5 in squares (the FRC size)
napa-tag 1,4,9 --format png --cell-px 40         # PNG, 40 px per cell
napa-tag 0-3 --family tagStandard41h12 --size-mm 100
```

Print SVGs at 100% scale, then measure what came out. Measure the black square for classic families, or the white
square inside the black frame for tagStandard, tagCircle, and tagCustom.

From Python, `napa_vision.tags.tag_image(tag_id, cell_px=...)` returns a tag as a `uint8` array, `tag_svg` returns
an SVG string, and `tag_cells` returns a boolean grid. Tags follow the official artwork, in the orientation field
layouts assume. Note that OpenCV's `DICT_APRILTAG_36h11` draws them turned 180 degrees.
