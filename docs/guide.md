# Napa guide

This guide covers everything past "detect tags in a frame": tuning the detector, getting metric poses, calibrating a
camera, and measuring Napa on your own footage.

## Detection

```python
from napa_vision import NapaDetector

detector = NapaDetector()
for tag in detector.detect(frame):  # uint8 grayscale, BGR, or BGRA; strided views are fine
    print(tag.tag_id, tag.corners)  # corners: 4x2 float64
```

Each observation also carries `hamming`, the number of payload bits that were corrected (0 is a
perfect read), and `decision_margin`: the smaller of the white and black payload bits' mean distance
from the decision threshold, in grey levels. Both are computed as upstream AprilTag computes them, so
filters written for AprilTag carry over.

Corners come back in OpenCV's AprilTag order, in pixel-centre coordinates: the centre of the top-left pixel is
`(0, 0)`. Upstream AprilTag puts `(0, 0)` at that pixel's top-left corner, so its corners are half a pixel larger in
both axes.

Detection needs only numpy. OpenCV is used for BGR conversion when it's installed, and is required for pose
processing and calibration.

| Parameter | Default | What it does |
| --- | ---: | --- |
| `min_tag_px` | 32 | Smallest black-square side, in pixels, that Napa guarantees to sample. Lower it for distant tags; the work grows roughly with `1 / min_tag_px²`. |
| `max_hamming` | 3 | Payload bits Napa will correct (0–3). A third bit is corrected only when the three wrong cells touch, as they do under a finger, cable, or glare spot; scattered errors look like chance matches and are rejected. Lower it to reject more aggressively. |
| `min_contrast` | 20 | Minimum grey-level difference between tag black and the white quiet zone. Napa raises this automatically, region by region, where the frame is noisy or textured. Lowering it finds fainter tags in washed-out scenes, but costs speed on every frame, because sensor noise then passes the threshold too. |
| `min_border_cells` | 22 | How many of the 28 border cells must read as dark before a candidate is decoded. |
| `sharpening` | 0.25 | Cell sharpening before decoding, as upstream AprilTag's `decode_sharpening`. |
| `max_detections` | 64 | Results returned per frame. |

A detector reuses its buffers between frames. Use one per camera stream, and don't call one from two threads at
once. The native call releases the GIL, so separate detectors can run in parallel. Call `close()` to free the
buffers early; a closed detector raises instead of detecting.

### From C or another language

`src/napa_vision/_native/napa_core.h` is the whole API: create a detector, call `napa_detect` on a grayscale buffer
with a stride, and destroy it. The library has no dependencies beyond the C standard library and `libm`.

## Pose

`VisionProcessor` wraps a detector and, given a calibration and the tag size, adds poses:

```python
from napa_vision import VisionProcessor, load_calibration_data

processor = VisionProcessor(calibration=load_calibration_data("camera.json"), tag_size_m=0.1651)
result = processor.process(frame, timestamp=t)
result.image_observations  # IDs and corners
result.single_tag_poses  # both planar (IPPE) solutions for each tag, in OpenCV camera coordinates
result.tag_angles  # per-tag viewing angles, distance, and camera-to-tag translation
```

Pass `tag_layout={tag_id: world_to_tag}` (4×4 rigid transforms) to also get `result.camera_pose`, the camera's pose
in your world frame. With one visible tag you get both ambiguity candidates. With several, Napa solves them jointly
(RANSAC once three or more are visible), drops outliers, and records why any tag was rejected in
`result.pose_diagnostics`. Pass `camera_pose_bounds` to reject physically impossible poses.

### Coordinate conventions

World and camera-pose transforms use a right-handed frame with +X forward, +Y left, and +Z up, with positive yaw
counter-clockwise seen from above. Single-tag poses stay in OpenCV's camera frame (+X right, +Y down, +Z forward).
`napa_vision.geometry` converts between the two.

A tag's face lies in its local Y/Z plane, and local +X is its face normal. The corner order matches the official
AprilTag artwork. OpenCV's `generateImageMarker()` raster is rotated 180° from that artwork: when you warp a generated
marker onto the corners from `geometry.tag_field_corners()`, map the raster's top-left, top-right, bottom-right, and
bottom-left directly onto corners 0–3. That order already accounts for the rotation, so don't rotate the raster too.

## Camera calibration

Photograph a checkerboard at the exact resolution, focus, and zoom you'll use, from varied positions and angles, then
run:

```bash
napa-calibrate calibration-images/ \
  --output camera.json \
  --columns 9 --rows 6 --square-size-m 0.024 \
  --minimum-views 12
```

`columns` and `rows` count inner corners. The output records the image size, reprojection error, the number of views
used, and which images were rejected. `VisionProcessor` refuses frames at any other resolution.

Each calibration file records its provenance: `measured` for `napa-calibrate` output, `imported` for anything loaded
without a provenance block, and `example` for placeholders. `CameraCalibration.require_production()` passes only for
measured files.

## Measuring on your own footage

Synthetic benchmarks can't stand in for your camera, lighting, and tags. `napa-benchmark` scores Napa on a folder of
annotated images.

Copy `benchmarks/corpus.example.json`, list each image with the tag IDs it contains, and optionally freeze each file
with a SHA-256:

```json
{
  "schema_version": 1,
  "name": "my-camera",
  "frames": [
    {"path": "frames/0001.png", "expected_tag_ids": [1, 7]},
    {"path": "frames/0002.png", "expected_tag_ids": []}
  ]
}
```

Paths are relative to the manifest and must stay inside its directory. Then run:

```bash
napa-benchmark my-corpus/manifest.json \
  --repetitions 5 --warmup-passes 1 \
  --min-recall 0.99 --min-precision 1.0 --max-p95-total-ms 30 \
  --output report.json --pretty
```

The report gives recall, precision, missed and unexpected IDs (with up to 25 examples), and p50/p95/p99 times for
loading, detection, scoring, and the total. It exits with `1` if a `--min-*`/`--max-*` gate fails, and `2` on bad
input. Use the gates in CI to catch regressions on footage you trust.

To compare Napa with standard AprilTag on synthetic scenes, see [the benchmark guide](../benchmarks/README.md).

## Building and packaging

Installing from source compiles the C core (`src/napa_vision/_native`), so it needs a C99 compiler. Without one, the
install fails rather than leaving you without a detector. To use a library you built yourself, set
`NAPA_CORE_LIBRARY` to its path.

```bash
python -m pip install -e '.[headless,dev]'
ruff format --check . && ruff check .
pytest --cov=napa_vision
python -m build && python -m twine check dist/*
```

CI runs on Python 3.11–3.14 and on OpenCV 4.7 through the current release. It builds the C core on Linux, macOS, and
Windows, checks that a fresh wheel detects tags without OpenCV installed, and smoke-tests the command-line tools.
