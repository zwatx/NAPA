# Napa guide

This guide covers everything past "detect tags in a frame": tuning the detector, getting metric poses, feeding them
to a Kalman filter, calibrating a camera, and measuring Napa on your own footage. [The API reference](api.md) lists
every class, method, and field.

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
both axes. OpenCV's order is upstream's `p[1], p[0], p[3], p[2]`.

`tag.center` and `tag.homography` match upstream's `c` and `H`, in Napa's pixel coordinates. The homography maps
upstream's tag coordinates, `[-1, 1]` across the black square, to pixels, with `H[2, 2] == 1`. The centre is that
map's image of `(0, 0)`, where the diagonals cross, which under perspective isn't the corners' mean. Both are
computed from the corners when you read them.

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

Every pose and tag angle comes with a covariance; see [Kalman filters](#kalman-filters).

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

## Kalman filters

Every pose Napa reports carries a 6×6 covariance, and every tag angle a 2×2 one. They come from the geometry of
that frame: how far the tags are, how steeply they're seen, how many there are, and how widely they're spread. A
filter weighted by them trusts a close multi-tag pose and discounts a lone distant tag, with no distance-based
standard-deviation table to tune.

The covariance is ordered `[x, y, z, rx, ry, rz]` (ROS order), with translation error and a small rotation, both in
the parent frame. [The API reference](api.md#uncertainty) defines it exactly and explains how it's computed.

### WPILib pose estimators

`to_pose2d()` gives the `(x, y, yaw)`, and `local_std_devs()` the standard deviations, that `addVisionMeasurement`
takes:

```python
from napa_vision.geometry import invert_transform, transform_matrix

# The camera's mount on the robot: forward, left, up (m); roll, pitch, yaw (rad).
robot_to_camera = transform_matrix([0.30, 0.0, 0.25], [0.0, -0.35, 0.0])
camera_to_robot = invert_transform(robot_to_camera)

result = processor.process(frame, timestamp=capture_time)
if result.camera_pose is not None:
    robot = result.camera_pose.field_to_camera().transform_by(camera_to_robot).to_pose2d()
    pose_estimator.addVisionMeasurement(
        Pose2d(robot.x, robot.y, Rotation2d(robot.yaw)), capture_time, tuple(robot.local_std_devs())
    )
```

Pass the frame's capture time in the estimator's clock, not the time processing finished. When Napa runs on a
coprocessor, send the pose, the three standard deviations, and the capture time to the robot together.

Use `local_std_devs()`, not `std_devs()`. WPILib applies vision standard deviations along the robot's own forward and
left axes, while `std_devs()` and `covariance` are along the field's. A tag constrains the camera's distance and its
sideways position very differently, so the two differ whenever the robot isn't facing along a field axis. WPILib
also takes
only standard deviations, so it drops the correlations between x, y, and yaw. A filter that accepts a full
covariance can use `robot.covariance`.

### Your own filter

For a 6-D update, use the full covariance as the measurement noise. Compute the innovation with `pose_difference` so
that it follows the same convention:

```python
from napa_vision.covariance import pose_difference

measured = result.camera_pose.field_to_camera().transform_by(camera_to_robot)  # field_to_robot
innovation = pose_difference(predicted_field_to_robot, measured.transform)  # [dx, dy, dz, rx, ry, rz]
measurement_noise = measured.covariance
```

To aim at a tag rather than localize, filter its bearing: `(center_tx, center_ty)` with `center_covariance`, in
radians. The tag's distance and its variance come from the single-tag pose:

```python
pose = result.single_tag_poses[0].camera_to_tag()
direction = pose.translation / np.linalg.norm(pose.translation)
distance_variance = direction @ pose.covariance[:3, :3] @ direction
```

### Ambiguity and tuning

A single tag has two candidate poses (see `ambiguity`), and so do several tags on one wall. Napa widens each
candidate's covariance to cover the other, in proportion to how well the other fits. A filter fed solution 0 is therefore not misled when the tag is genuinely
ambiguous: it just gets a measurement it will mostly ignore. You can still discard single-tag poses with
`ambiguity > 0.2` if you'd rather not spend the filter's attention on them.

Covariances assume `corner_noise_px` (default 1.0) of independent noise on each corner coordinate. That is
deliberately pessimistic for clean frames, because blur, rolling shutter, and calibration error all look like corner
noise. To tune it, log poses while the robot stands still and compare their scatter with the reported standard
deviations. Calibration error repeats from frame to frame, which no per-frame covariance can capture, so don't push
the value much below what you measure. Set it on the processor rather than rescaling covariances afterwards: the
ambiguity widening doesn't scale with it.

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

Wheels need no compiler. Python loads the core with `ctypes` rather than as an extension module, so each wheel is
tagged `py3-none-<platform>` and serves every Python 3 version. The Wheels workflow builds and tests them with
cibuildwheel on native Linux x86-64 and ARM64, macOS ARM and Intel, and Windows x64 runners. It runs on version tags
or by hand, and uploads the wheels and sdist as artifacts; it doesn't publish them.

```bash
python -m pip install -e '.[headless,dev]'
ruff format --check . && ruff check .
pytest --cov=napa_vision
python -m build && python -m twine check dist/*
```

CI runs on Python 3.11–3.14 and on OpenCV 4.7 through the current release. It builds the C core on Linux, macOS, and
Windows, checks that a fresh wheel detects tags without OpenCV installed, and smoke-tests the command-line tools.
