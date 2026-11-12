# Poses and filters

Poses need OpenCV. Install `napa-vision[headless]`, or `napa-vision[gui]` if you also want OpenCV's windows.

## Getting poses

`VisionProcessor` runs a detector and, given a [calibration](calibration.md) and the tag size, solves poses:

```python
from napa_vision import VisionProcessor, load_calibration_data

processor = VisionProcessor(calibration=load_calibration_data("camera.json"), tag_size_m=0.1651)
result = processor.process(frame, timestamp=t)
result.image_observations  # IDs and corners
result.single_tag_poses  # both planar (IPPE) solutions per tag, in OpenCV camera coordinates
result.tag_angles  # each tag's viewing angles, distance, and camera-to-tag translation
```

Pass `tag_layout={tag_id: field_to_tag}` (4x4 rigid transforms) to also get `result.camera_pose`, the camera's pose
on the field. With one visible tag you get both candidate poses. With several, NAPA solves all their corners
together, falls back to RANSAC when three or more tags disagree, drops outliers, and records why any tag was
rejected in `result.pose_diagnostics`. Pass `camera_pose_bounds` to reject poses the robot cannot physically be in.

Pose errors and rejection limits are RMS Euclidean corner distances in pixels. The x and y residuals are combined
per corner before averaging. Single-tag and multi-tag diagnostics use the same units. The
[API reference](api.md#cameraposeobservation) explains how this differs from older builds and what that means for
limits you tuned there.

Use one processor per camera stream.

## Coordinate frames

Field and camera poses use WPILib's frame: +X forward, +Y left, +Z up, with positive yaw counter-clockwise seen
from above. Single-tag poses stay in OpenCV's camera frame: +X right, +Y down, +Z forward. `napa_vision.geometry`
converts between the two, and [Conventions](api.md#conventions) defines every frame.

A tag's face lies in its local Y/Z plane, and local +X is its face normal. Corner order matches the official
AprilTag artwork. OpenCV's `generateImageMarker()` draws markers rotated 180 degrees from that artwork. To warp a
generated marker onto the corners from `geometry.tag_field_corners()`, map the marker image's top-left, top-right,
bottom-right, and bottom-left onto corners 0 to 3 directly, without rotating it.

## Uncertainty

Every pose NAPA reports carries a 6x6 covariance. Every tag angle carries a 2x2 one. Each is computed from that
frame's geometry: how far away the tags are, how steeply they are seen, how many there are, and how widely they are
spread. A filter weighted by them trusts a close multi-tag pose and discounts a lone distant tag, with no
hand-tuned distance table.

The covariance is ordered `[x, y, z, rx, ry, rz]`, as in ROS. The translation error and a small rotation are both
expressed in the parent frame. [Uncertainty](api.md#uncertainty) defines the convention exactly and explains how
the numbers are computed, including why planar targets need special handling.

The covariance does not cover bias. On a camera that stores gamma-encoded video, blur makes every tag read slightly
small, so it reads slightly far, by the same amount in every frame. Averaging does not remove that, and neither
does `corner_noise_px`. Setting [`camera_gamma`](detection.md#camera-gamma) does. On 297 frames from one linear
camera, re-encoded with a gamma 2.2 curve, tags read 1.5% small on average and 2.7% small below 35 px. That is about
15 cm at 5 m. With `camera_gamma=2.2` the error was under 0.1%. At 1.97, the value the estimation helper read, it
was 0.3%. A setting 0.4 too high made tags read 0.6% large. Those numbers compare against the detector's own
corners on the original linear frames, from one camera and one session.

## Rolling-shutter cameras

A rolling-shutter camera reads its lines one after another. A moving camera therefore sees each tag sheared. The
bottom of the tag was captured later than its top. The covariance does not include this. In simulation, a 15 ms
readout at 2 rad/s of turn moved a single tag's position by about 38 mm and tilted it 9 degrees. A pose from four
tags moved by 45 mm and 1.1 degrees. A global-shutter camera has none of this, so do not set it for one.

```python
from napa_vision import RollingShutter, VisionProcessor

processor = VisionProcessor(
    calibration=calibration,
    tag_size_m=0.1651,
    tag_layout=layout,
    rolling_shutter=RollingShutter(readout_s=0.015, direction="top_to_bottom"),
)
```

Each corner's speed in the image comes from its own movement between calls, using the `timestamp` you pass. No gyro
and no motion model are needed. Corners get separate speeds because, away from the image centre, a turn moves a
tag's corners at different rates. Corners are moved to where they were when the image's middle line was read, so
every tag in a frame describes the same moment. The corrected corners feed the pose solvers, in raw pixels before
undistortion. `image_observations` stay as detected.

What to know before you enable it:

- The pose belongs to the moment the middle line was read. A `timestamp` taken at the start of readout is early by
  half the readout.
- `readout_s` and `direction` come from the sensor's data sheet or from a measurement, for the frame as delivered.
  `direction` also covers a sensor mounted upside down or a quarter turn. Being 30% off in `readout_s` still
  removes about two thirds of the error. The wrong direction doubles it.
- A tag needs a recent earlier sighting. Its speed uses the last two frames, or the last three when available.
  `max_frame_gap_s` (default 0.1 s) discards older sightings. A tag without one is listed in
  `FrameResult.uncorrected_tag_ids` and solved as detected. It is left out of the multi-tag pose while other tags
  in that frame were corrected, because it describes a different moment. A tag ID seen twice in a frame is not
  corrected. A timestamp that repeats or goes backwards starts the history over.

Simulated results, medians. The setup is 1280x800 with focal length 900 px, 16.5 cm tags at 1.5 to 5 m, 50 frames
per second, and each image row rendered at its own time. Each cell shows error before correction, then after:

| Readout and motion | One tag, position | One tag, tilt | Four-tag camera pose |
| --- | ---: | ---: | ---: |
| 15 ms, 0.5 rad/s | 7.5 to 1.5 mm | 2.2 to 0.3 deg | 10.8 to 0.2 mm, 0.31 to 0.004 deg |
| 15 ms, 2 rad/s | 38 to 1.6 mm | 9.0 to 0.3 deg | 45 to 0.2 mm, 1.11 to 0.004 deg |
| 15 ms, 5 rad/s | 89 to 1.3 mm | 16.6 to 0.4 deg | 141 to 0.9 mm, 3.9 to 0.02 deg |
| 30 ms, 2 rad/s | 64 to 1.0 mm | 13.1 to 0.3 deg | 106 to 0.5 mm, 2.6 to 0.008 deg |
| 15 ms, sideways at 3 m/s | 16 to 1.1 mm | 7.1 to 0.2 deg | 20 to 0.2 mm, 0.57 to 0.004 deg |

Those are simulations, with no lens distortion and a static scene where only the camera moves. One real recording
supports the correction. On four sequences from a GoPro with 170 surveyed tags, moving each tag's corners to the
middle line reduced how badly frames with four or more tags fit one camera pose. The best fit sat at an assumed
readout of 28 to 33 ms, and the fit got worse for the opposite direction. That check runs against a survey accurate
to about 10 cm, so it confirms the effect exists. It does not confirm the size of the improvement above.

### Reading the readout from a recording

If you have no data sheet, `estimate_rolling_shutter` reads the readout time and direction from consecutive frames.
A sheared square is not a valid view of a square, so its pose fit leaves a residual. The answer is the readout that
removes the most of it.

```python
from napa_vision import estimate_rolling_shutter

estimate = estimate_rolling_shutter(recording, calibration=calibration, tag_size_m=0.1651)
processor = VisionProcessor(calibration=calibration, tag_size_m=0.1651, rolling_shutter=estimate.rolling_shutter)
```

Here `recording` yields `(timestamp, frame)` pairs. The helper needs at least 30 sightings (`min_sightings`) of
tags 60 px or more across, in consecutive frames, while the camera moves at 0.3 rad/s or more across the read axis.
A read along rows needs sideways motion, like turning. A read along columns needs up-and-down motion. Otherwise
`readout_s` is `None`. It is 0.0 when the camera reads as a global shutter (under 2.5 ms) on the axes the recording
could check, which are listed in `axes_checked`. A recording with only turning cannot see a sensor mounted a quarter
turn. It reports only `"rows"`, and a 0.0 there does not rule out a read along columns. `low` and `high` show only
the scatter between sightings. The estimate uses each corner's own motion between frames. With the tag's average
speed instead, tags toward the image edge read up to 12% low.

In simulation, with the real detector on images rendered row by row, tags 60 to 80 px across, readouts of 8 to 30 ms,
and 1 to 2 rad/s of turn, the answer landed within about 1 ms at 60 px and 0.3 ms at 80 px, across the image. Reads
along columns were tested only on computed corners. Slower turning needs more sightings. Before the per-corner
change, a calibration missing the lens distortion (one lens model, k1 = -0.3), or a focal length 8% off, moved the
answer by about 1 to 1.5 ms. None of this has been checked against a real camera's known readout.

One hard limit. The tags must be large in the image. A tag counts only if it is at least 60 px across and a 15 ms
readout would shear it by at least 0.3 px. Smaller tags read wrong while looking certain. In the same simulation,
tags of 25 to 50 px read 10 to 20% off, and once more than double, with a range that left out the true value. On FRC
footage most tags are smaller than that. One robot camera had a median tag of 46 px with 15% at 60 px or more. Two
public FRC image sets had medians of 39 and 45 px. Expect `None` from such footage unless it runs long enough to
collect 30 sightings of near tags. For small tags, use the data sheet or the layout method below. The helper reports
0.0 only when the resampled answers all sit under 5 ms.

### With a known tag layout

Given `tag_layout`, the same mapping `VisionProcessor` takes, the helper fits one camera pose to all the tags in
each frame and finds the readout that lets them fit best. The tags' spread across the image gives the shear away
even when each tag is small:

```python
estimate = estimate_rolling_shutter(recording, calibration=calibration, tag_size_m=0.1651, tag_layout=layout)
```

A frame counts as one sighting when it has three or more layout tags with a recent earlier sighting, spanning at
least a quarter of the image both ways, while the camera moves across the read axis. Tags along a single line let
the pose absorb most of the shear by rolling, so those frames are left out.

- In simulation it was exact where the single-tag helper was not. On walls of 6 to 45 tags about 36 px across,
  readouts of 8 to 30 ms read within 0.3 ms in all four directions, at 0.5 to 2 rad/s. On images rendered row by row
  and run through the detector, with 12 tags 30 to 50 px across, it read 8, 15, and 30 ms within 0.2 ms, including
  with 4 and 8 ms exposures and sideways motion.
- It depends on the layout and the calibration. Layout errors repeat in every frame, so they can move the answer
  further than `low` and `high` show. On a 45-tag wall 4 m away, survey errors of 2 to 10 cm per tag moved a 15 ms
  answer by 0.3 to 2 ms, or made it return `None`. On walls of 6 to 9 tags, 2 cm errors gave an answer in 3 of 9
  cases, each within its range, and `None` in the rest. Errors of 5 cm always gave `None`. An unmodelled lens
  (k1 = -0.1) moved the answer by up to 0.6 ms. A 5 to 8% focal error moved it by under 0.1 ms.
- On real footage it agrees with the earlier check. On the four GoPro sequences above, using the dataset's survey
  (good to about 10 cm), it read 26 to 32 ms top to bottom on every sequence. The single-tag helper returned `None`
  on all four. That run used OpenCV's distortion model fitted to the dataset's fisheye calibration, 0.76 px off at
  the median. The dataset's own lens model gave 28 to 33 ms in a separate prototype. The camera's true readout is
  not known.

## WPILib pose estimators

`to_pose2d()` gives the `(x, y, yaw)` that `addVisionMeasurement` takes. `local_std_devs()` gives its standard
deviations:

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

Pass the frame's capture time in the estimator's clock, not the time processing finished. When NAPA runs on a
coprocessor, send the pose, the three standard deviations, and the capture time to the robot together.

Use `local_std_devs()`, not `std_devs()`. WPILib applies vision standard deviations along the robot's forward and
left axes. `std_devs()` and `covariance` run along the field's axes. A tag pins down the camera's distance and its
sideways position very differently, so the two disagree whenever the robot is not facing along a field axis. WPILib
takes only standard deviations, so it drops the correlations between x, y, and yaw. A filter that accepts a full
covariance can use `robot.covariance`.

## Your own filter

For a 6D update, use the full covariance as the measurement noise. Compute the innovation with `pose_difference`
so it follows the same convention:

```python
from napa_vision.covariance import pose_difference

measured = result.camera_pose.field_to_camera().transform_by(camera_to_robot)  # field_to_robot
innovation = pose_difference(predicted_field_to_robot, measured.transform)  # [dx, dy, dz, rx, ry, rz]
measurement_noise = measured.covariance
```

To aim at a tag instead of localizing, filter its bearing. Use `(center_tx, center_ty)` with `center_covariance`,
in radians. The tag's distance and its variance come from the single-tag pose:

```python
pose = result.single_tag_poses[0].camera_to_tag()
direction = pose.translation / np.linalg.norm(pose.translation)
distance_variance = direction @ pose.covariance[:3, :3] @ direction
```

## Ambiguity

A single tag has two candidate poses. Several tags on one wall do too. NAPA widens each candidate's covariance to
cover the other, in proportion to how well the other fits. A filter fed solution 0 of a genuinely ambiguous tag
gets a measurement it will mostly ignore, instead of a confident wrong one. To skip those measurements entirely,
discard single-tag poses with `ambiguity > 0.2`.

Reprojection error and `ambiguity` describe how the candidates fit the observed corners. Biased corners can favour
the wrong planar solution even with a small reprojection error. Similar corner-error sizes can also produce very
different pose errors, depending on which corners move and in which directions.

## Tuning how much to trust vision

Covariances assume `corner_noise_px` (default 1.0) of independent noise on each corner coordinate. That is
deliberately pessimistic for clean frames. Blur, rolling shutter, and calibration error all look like corner noise
to the solver.

To tune it, log poses while the robot stands still and compare their scatter with the reported standard deviations.
Calibration error repeats from frame to frame, and no per-frame covariance can capture it, so do not push the value
much below what you measure. Set it on the processor rather than rescaling covariances afterwards, because the
ambiguity widening does not scale with it.

Stationary scatter measures repeatability. Absolute pose accuracy needs independently measured reference poses,
reliable calibration, and measured tag placement and camera mounts.
