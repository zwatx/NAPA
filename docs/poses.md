# Poses and filters

Poses need OpenCV: install `napa-vision[headless]`, or `napa-vision[gui]` if you also want OpenCV's windows.

## Getting poses

`VisionProcessor` runs a detector and, given a [calibration](calibration.md) and the tag size, solves poses:

```python
from napa_vision import VisionProcessor, load_calibration_data

processor = VisionProcessor(calibration=load_calibration_data("camera.json"), tag_size_m=0.1651)
result = processor.process(frame, timestamp=t)
result.image_observations  # IDs and corners
result.single_tag_poses  # both planar (IPPE) solutions for each tag, in OpenCV camera coordinates
result.tag_angles  # each tag's viewing angles, distance, and camera-to-tag translation
```

Pass `tag_layout={tag_id: field_to_tag}` (4×4 rigid transforms) to also get `result.camera_pose`, the camera's pose on
the field. With one visible tag you get both candidate poses. With several, NAPA solves all their corners together,
falls back to RANSAC when three or more tags disagree, drops outliers, and records why any tag was rejected in
`result.pose_diagnostics`. Pass `camera_pose_bounds` to reject poses the robot can't physically be in.

Use one processor per camera stream.

## Coordinate frames

Field and camera poses use WPILib's frame: +X forward, +Y left, +Z up, with positive yaw counter-clockwise seen from
above. Single-tag poses stay in OpenCV's camera frame: +X right, +Y down, +Z forward. `napa_vision.geometry` converts
between the two, and [Conventions](api.md#conventions) in the API reference defines every frame.

A tag's face lies in its local Y/Z plane, and local +X is its face normal. The corner order matches the official
AprilTag artwork. OpenCV's `generateImageMarker()` draws markers rotated 180° from that artwork. To warp a generated
marker onto the corners from `geometry.tag_field_corners()`, map the marker image's top-left, top-right, bottom-right,
and bottom-left onto corners 0–3 directly, without rotating it.

## Uncertainty

Every pose NAPA reports carries a 6×6 covariance, and every tag angle a 2×2 one. Each is computed from that frame's
geometry: how far away the tags are, how steeply they're seen, how many there are, and how widely they're spread. A
filter weighted by them trusts a close multi-tag pose and discounts a lone distant tag, without a hand-tuned distance
table.

The covariance is ordered `[x, y, z, rx, ry, rz]`, as in ROS, with the translation error and a small rotation both in
the parent frame. [Uncertainty](api.md#uncertainty) in the API reference defines it exactly and explains how it's
computed.

## WPILib pose estimators

`to_pose2d()` gives the `(x, y, yaw)` that `addVisionMeasurement` takes, and `local_std_devs()` its standard
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

Use `local_std_devs()`, not `std_devs()`. WPILib applies vision standard deviations along the robot's forward and left
axes, while `std_devs()` and `covariance` are along the field's axes. A tag pins down the camera's distance and its
sideways position very differently, so the two disagree whenever the robot isn't facing along a field axis. WPILib
takes only standard deviations, so it drops the correlations between x, y, and yaw. A filter that accepts a full
covariance can use `robot.covariance`.

## Your own filter

For a 6-D update, use the full covariance as the measurement noise, and compute the innovation with `pose_difference`
so that it uses the same convention:

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

## Ambiguity

A single tag has two candidate poses (see `ambiguity`), and so do several tags on one wall. NAPA widens each
candidate's covariance to cover the other, in proportion to how well the other fits. A filter fed solution 0 of a
genuinely ambiguous tag gets a measurement it will mostly ignore, rather than a confident wrong one. To skip those
measurements entirely, discard single-tag poses with `ambiguity > 0.2`.

## Tuning how much to trust vision

Covariances assume `corner_noise_px` (default 1.0) of independent noise on each corner coordinate. That's deliberately
pessimistic for clean frames, because blur, rolling shutter, and calibration error all look like corner noise to the
solver.

To tune it, log poses while the robot stands still and compare their scatter with the reported standard deviations.
Calibration error repeats from frame to frame, and no per-frame covariance can capture it, so don't push the value
much below what you measure. Set it on the processor rather than rescaling covariances afterwards, because the
ambiguity widening doesn't scale with it.
