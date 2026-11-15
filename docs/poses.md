# Poses and filters

Poses need OpenCV: `napa-vision[headless]` (or `[gui]` for windows).

## Getting poses

```python
from napa_vision import VisionProcessor, load_calibration_data

processor = VisionProcessor(calibration=load_calibration_data("camera.json"), tag_size_m=0.1651)
result = processor.process(frame, timestamp=t)
result.image_observations  # IDs and corners
result.single_tag_poses  # both planar (IPPE) solutions per tag, OpenCV camera coordinates
result.tag_angles  # viewing angles, distance, camera-to-tag translation
```

Add `tag_layout={tag_id: field_to_tag}` for `result.camera_pose`, the camera's pose on the field. One tag gives both
candidate poses. Several are solved together, with RANSAC fallback and outlier dropping when they disagree.
`result.pose_diagnostics` says why any tag was rejected. `camera_pose_bounds` rejects physically impossible poses.

Pose errors and limits are RMS per-corner distances in pixels, everywhere. Older builds underreported single-tag
errors by sqrt(2); the [API reference](api.md#cameraposeobservation) has the conversion.

Use one processor per camera stream.

## Frames

Field and camera poses use WPILib axes (+X forward, +Y left, +Z up). Single-tag `rvec`/`tvec` stay in OpenCV's
(+X right, +Y down, +Z forward). `napa_vision.geometry` converts. Every frame is defined in
[Conventions](api.md#conventions).

Watch the artwork: OpenCV's `generateImageMarker()` draws tags 180° from the official artwork NAPA assumes. Map a
generated marker's corners directly onto `geometry.tag_field_corners()` corners 0-3, without rotating.

## Uncertainty

Every pose carries a 6x6 covariance, every tag angle a 2x2 one, computed from that frame's geometry: distance, view
angle, tag count, spread. A filter weighted by them trusts close multi-tag poses and discounts lone distant tags. No
hand-tuned distance table needed.

Order is `[x, y, z, rx, ry, rz]`, in the parent frame. [Uncertainty](api.md#uncertainty) defines it exactly.

Covariance covers scatter, not bias. On gamma-encoded cameras, blur makes tags read a few percent small (about 15 cm
at 5 m in one test), the same every frame. Averaging cannot fix that. [`camera_gamma`](detection.md#camera-gamma)
can: in that test it cut the error to under 0.1%.

## Rolling-shutter cameras

Rolling shutters read line by line, so a moving camera sees tags sheared. Covariance ignores this. In simulation a
15 ms readout at 2 rad/s moved a single-tag pose by 38 mm and 9°, a four-tag pose by 45 mm and 1.1°. Global-shutter
cameras do not have this problem. Do not enable it for one.

```python
from napa_vision import RollingShutter, VisionProcessor

processor = VisionProcessor(
    calibration=calibration,
    tag_size_m=0.1651,
    tag_layout=layout,
    rolling_shutter=RollingShutter(readout_s=0.015, direction="top_to_bottom"),
)
```

Each corner's image speed comes from its movement between your `timestamp`ed calls. No gyro needed. Corners are
shifted to the moment the middle line was read, then solved. `image_observations` stay as detected.

- The pose timestamp is that middle-line moment. A start-of-readout timestamp is early by half the readout.
- Get `readout_s` and `direction` from the data sheet or measure them (below). 30% off still removes ~2/3 of the
  error. The wrong direction doubles it.
- Tags need a recent previous sighting (`max_frame_gap_s`, default 0.1 s). Tags without one appear in
  `uncorrected_tag_ids`, are solved as detected, and stay out of that frame's multi-tag pose.

In simulation (1280x800, 16.5 cm tags at 1.5-5 m, 50 fps), correction took single-tag error from 38 mm to under 2 mm
at 15 ms and 2 rad/s, and held up to 5 rad/s and sideways motion. One real check agrees on the effect, not the size:
on GoPro sequences with surveyed tags, multi-tag fit was best at an assumed 28-33 ms readout and worse in the opposite
direction.

### Measuring the readout

No data sheet? `estimate_rolling_shutter` finds the readout that best un-shears recorded tags:

```python
from napa_vision import estimate_rolling_shutter

estimate = estimate_rolling_shutter(recording, calibration=calibration, tag_size_m=0.1651)
processor = VisionProcessor(calibration=calibration, tag_size_m=0.1651, rolling_shutter=estimate.rolling_shutter)
```

`recording` yields `(timestamp, frame)` pairs. Without a layout it needs 30+ sightings of tags at least 60 px across
while the camera moves 0.3 rad/s or more across the read axis. Smaller tags read wrong while looking certain, and
most FRC footage is smaller, so expect `readout_s=None` there. 0.0 means global shutter on the axes checked
(`axes_checked`); turning-only footage cannot rule out a quarter-turned sensor.

With `tag_layout=layout` it fits one pose per frame across all layout tags instead, which works with small tags
(~36 px tags read within 0.3 ms in simulation). The price is dependence on layout accuracy: 2-10 cm survey errors
moved answers by up to 2 ms or produced `None`.

Neither variant has been checked against a camera with a known readout. On the GoPro sequences the layout variant
read 26-32 ms on every sequence; the single-tag variant returned `None`.

## WPILib pose estimators

```python
from napa_vision.geometry import invert_transform, transform_matrix

# Camera mount on the robot: forward, left, up (m); roll, pitch, yaw (rad).
robot_to_camera = transform_matrix([0.30, 0.0, 0.25], [0.0, -0.35, 0.0])
camera_to_robot = invert_transform(robot_to_camera)

result = processor.process(frame, timestamp=capture_time)
if result.camera_pose is not None:
    robot = result.camera_pose.field_to_camera().transform_by(camera_to_robot).to_pose2d()
    pose_estimator.addVisionMeasurement(
        Pose2d(robot.x, robot.y, Rotation2d(robot.yaw)), capture_time, tuple(robot.local_std_devs())
    )
```

Use the capture time, in the estimator's clock. On a coprocessor, send pose, std devs, and capture time together.

Use `local_std_devs()`, not `std_devs()`. WPILib wants deviations along the robot's axes; `std_devs()` is along the
field's. They differ whenever the robot is not axis-aligned, which is most of the time.

## Your own filter

```python
from napa_vision.covariance import pose_difference

measured = result.camera_pose.field_to_camera().transform_by(camera_to_robot)  # field_to_robot
innovation = pose_difference(predicted_field_to_robot, measured.transform)  # [dx, dy, dz, rx, ry, rz]
measurement_noise = measured.covariance
```

To aim at a tag instead, filter its bearing: `(center_tx, center_ty)` with `center_covariance`. Distance variance
comes from the single-tag pose:

```python
pose = result.single_tag_poses[0].camera_to_tag()
direction = pose.translation / np.linalg.norm(pose.translation)
distance_variance = direction @ pose.covariance[:3, :3] @ direction
```

## Ambiguity

One tag (or several on one wall) has two plausible poses. NAPA widens each covariance to cover the other, so your
filter mostly ignores a genuinely ambiguous reading instead of trusting a confident wrong one. To skip them entirely,
drop single-tag poses with `ambiguity > 0.2`.

Careful: biased corners can make the wrong solution fit best with tiny reprojection error. Small error alone is not
proof of the right branch.

## How much to trust vision

Covariances assume `corner_noise_px=1.0` of noise per corner coordinate. That is deliberately pessimistic. Real blur,
shutter shear, and calibration error all look like corner noise to the solver.

Tune it by logging poses with the robot still and comparing scatter to reported std devs. Set it on the processor,
not by rescaling results (ambiguity widening does not rescale). Remember that still-robot scatter measures
repeatability, not absolute accuracy. That needs measured tag placement, mounts, and reference poses.
