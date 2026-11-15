# Detection

```python
from napa_vision import NapaDetector

detector = NapaDetector()
for tag in detector.detect(frame):  # uint8 greyscale, BGR, or BGRA; strided views are fine
    print(tag.tag_id, tag.corners)  # corners: 4x2 float64
```

Detection needs only numpy. OpenCV converts BGR frames when it's installed, and is required for poses and calibration.

## What each detection holds

- **`tag_id`** and **`corners`**: the ID within the family, and the four sub-pixel corners. Corners come back in
  OpenCV's AprilTag order, which is upstream's `p[1], p[0], p[3], p[2]`, in pixel-centre coordinates: the centre of
  the top-left pixel is `(0, 0)`. Upstream AprilTag puts `(0, 0)` at that pixel's top-left corner, so its corners are
  half a pixel larger in both axes.
- **`hamming`**: the payload bits that were corrected, where 0 is a perfect read.
- **`decision_margin`**: the smaller of the white and black payload bits' mean distance from the decision threshold,
  in grey levels.
- **`center`** and **`homography`**: upstream's `c` and `H`, in NAPA's pixel coordinates. The homography maps
  upstream's tag coordinates, `[-1, 1]` across the black square, to pixels. The centre is where the diagonals cross,
  which under perspective isn't the corners' mean.

`hamming` and `decision_margin` are computed as upstream computes them, so filters written for AprilTag carry over.

## Tuning

| Parameter | Default | What it does |
| --- | ---: | --- |
| `min_tag_px` | 32 | Smallest side of the tag-size square (the black square for classic families), in pixels, that NAPA is guaranteed to sample. Lower it for distant tags. The work grows roughly with `1 / min_tag_px²`. |
| `max_hamming` | family's (3 for tag36h11) | Payload bits NAPA will correct, 0–3. A third bit is corrected only when the three wrong cells touch, as they do under a finger, cable, or glare spot. Lower it to reject more aggressively. |
| `min_contrast` | 20 | Minimum grey-level difference between the tag's black and its white margin. NAPA raises it automatically where the frame is noisy or textured. Lowering it finds fainter tags in washed-out scenes, but costs speed on every frame, because sensor noise starts passing the threshold too. |
| `min_border_cells` | 22 | How many of tag36h11's 28 border cells must read dark before a candidate is decoded, scaled for other families |
| `sharpening` | 0.25 | Cell sharpening before decoding, as upstream's `decode_sharpening` |
| `max_detections` | 64 | Results returned per frame. A frame with more tags warns (`RuntimeWarning`) and drops the rest |
| `split_touching` | `"auto"` | Find tags whose border touches another dark shape at a corner, as in AprilGrid boards. `"auto"` costs time only on boards; `True` is slower on every frame. See below |
| `regrow_small_tags` | `False` | Find small tags, up to about 28 px, whose thin border comes apart when the search samples every other pixel. A candidate that doesn't fit or read is regrown from the full-resolution frame and tried again, at most 32 times a frame. In the 2026-11-15 benchmark it found 2–10% more small tags in blurred match footage (FRC footage 1949 → 1996 outputs) for about 2% more time, with no false detections |

Between the default and `min_tag_px=16` there's a middle step: `min_tag_px=16, tag_shape="tall"` searches every
other row at full resolution. On one Raspberry Pi 5 core (2026-11-14):

| Frames | `min_tag_px=22` | `16`, `tag_shape="tall"` | `16` |
| --- | ---: | ---: | ---: |
| FRC match footage, 188 frames | 1.60 ms, 346 detections | 2.18 ms, 372 | 3.45 ms, 380 |
| More FRC footage, 1000 frames | 1.56 ms, 968 | 2.11 ms, 1001 | 3.37 ms, 1067 |
| Robot camera (1280x960), 297 frames | 1.33 ms, 684 | 1.68 ms, 697 | 2.91 ms, 700 |

It costs about 1.3x the default, against about 2.2x for `min_tag_px=16`, and finds a third to three quarters of the
extra tags; it suits tags seen at an angle, which look tall and narrow. None of the three found anything in frames
without tags.

A detector reuses its buffers between frames. Use one per camera stream, and don't call one from two threads at once.
The native call releases the GIL, so separate detectors run in parallel. `close()` frees the buffers early.

## Calibration boards and touching tags

In an AprilGrid calibration board, and in tags set in a checkerboard, a black square touches each tag's border at
its corners. NAPA groups dark pixels that touch, including at a corner, so the tags and squares form one large blob
that isn't tag-shaped, and most of the board would be missed. `split_touching` splits such blobs where shapes touch
only at a corner or through a one-pixel neck, and decodes the pieces. By default (`"auto"`) it does so only on frames
that show a board; `True` splits every frame:

```python
detector = NapaDetector(min_tag_px=16, split_touching=True, max_detections=256)
```

The tags found this way come after the others, which are exactly what the detector returns without the option.
Pieces may correct at most two bits, fewer for families whose codes are closer together; tag16h5 and tagCircle21h7
aren't split, because random pieces match their short codes too often.

A frame splits only at corners until it finds a tag with dark shapes just beyond three of its corners. Then it starts
over and also splits one-pixel necks, which anti-aliasing leaves where tags meet squares and which cost much more. On
frames without touching tags, such as most FRC footage, `True` costs about 1.15–1.3× the time; frames of boards take
about 2.5–3×. With `split_touching=False`, nothing is split.

The default, `split_touching="auto"`, runs the ordinary pass first and splits only frames where a tag it found has
dark shapes just beyond three of its corners. Those frames are detected again with splitting, necks included from the start, and give
exactly what `True` gives. Other frames cost what they cost with the option off. In the 2026-11-14 benchmark (25 sets, laptop and Raspberry Pi 5),
auto found the same tags as `True` on the Locus AprilGrid renders (93% at `min_tag_px=22`, against 81% off) and
ran as fast as off on every set without boards, with the same results, including no false detections on the sets of
frames without tags. Auto can't split a board where every tag merges with its squares, because then no tag shows them,
and it leaves alone tags that touch other dark shapes off a board; `True` finds both.

## Tag families

A detector reads one family: tag36h11 unless you pass another, such as `NapaDetector(family="tagStandard41h12")`.
`napa_vision.families()` lists all nine of upstream AprilTag's families. To read several, use one detector per family.

Families whose codes are close together, such as tag16h5, get less error correction by default, so that random
patterns are no likelier to decode as tags than they are in tag36h11. [Tag families](api.md#tag-families) in the API
reference lists each family's tag size and default correction.

## Switching from pupil_apriltags or dt-apriltags

`napa_vision.compat.Detector` has the same constructor and `detect` method as `pupil_apriltags.Detector`, and returns
`Detection` objects with the same attributes, in upstream's conventions: corners in upstream order, `(0, 0)` at the
top-left pixel's corner, and poses (`pose_R`, `pose_t`, `pose_err`) from upstream's tag frame into the OpenCV camera
frame.

```python
from napa_vision.compat import Detector  # was: from pupil_apriltags import Detector

detector = Detector(families="tag36h11")
detections = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=0.1651)
```

`families` takes any of upstream's families, space-separated, as upstream does. `decode_sharpening` is used.
`nthreads`, `quad_decimate`, `quad_sigma`, `refine_edges`, `debug`, and `searchpath` are accepted and ignored, because
NAPA never decimates and always refines. NAPA's own parameters, such as `min_tag_px=24`, can be passed as extra
keyword arguments. Poses need OpenCV and, like upstream's, ignore lens distortion.

## Printing tags

`napa-tag` writes tags to print. SVGs print at an exact size: `--size-mm` is the side of the square that pose solvers
use as the tag size.

```bash
napa-tag 0-7 --size-mm 165.1 --output tags/      # tags 0-7 as SVG, 6.5 in black squares (the FRC size)
napa-tag 1,4,9 --format png --cell-px 40         # PNG, 40 px per cell
napa-tag 0-3 --family tagStandard41h12 --size-mm 100
```

Print SVGs at 100% scale, then measure the printed square: the black square for classic families, or the white square
inside the black frame for tagStandard, tagCircle, and tagCustom.

From Python, `napa_vision.tags.tag_image(tag_id, cell_px=...)` returns a tag as a `uint8` array, `tag_svg` as an SVG
string, and `tag_cells` as a boolean grid. Tags are drawn as the official artwork, in the orientation field layouts
assume. OpenCV's `DICT_APRILTAG_36h11` draws them turned 180°.
