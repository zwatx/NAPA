# Napa API reference

Every class, method, and field Napa exposes. [The guide](guide.md) explains how to use them together; this page is
the lookup. Everything listed under `napa_vision` imports from the package root:

```python
from napa_vision import NapaDetector, VisionProcessor, PoseWithCovariance
```

Detection, `center`, `homography`, the result types, and covariance propagation (`PoseWithCovariance`) need only
numpy. Solving poses and covariances from corners, calibrating, and `napa-benchmark` need OpenCV: install
`napa-vision[headless]` or `napa-vision[gui]`. Without it, the calls that need it raise a `RuntimeError` saying so.

- [Conventions](#conventions)
- [Detection](#detection): `NapaDetector`, `FiducialImageObservation`, `FiducialDetector`, `families`,
  `native_available`
- [Pose processing](#pose-processing): `VisionProcessor`, `FrameResult`, and the observations it holds
- [Uncertainty](#uncertainty): `PoseWithCovariance`, `Pose2dWithCovariance`, `camera_pose_covariance`
- [Solvers](#solvers): `SingleTagPoseSolver`, `CameraPoseSolver`, `TagAngleSolver`
- [Calibration](#calibration) and [provenance](#provenance)
- [Geometry helpers](#geometry-helpers): `napa_vision.geometry`
- [Drawing tags](#drawing-tags): `napa_vision.tags`
- [pupil_apriltags compatibility](#pupil_apriltags-compatibility): `napa_vision.compat`
- [Benchmarking](#benchmarking): `napa_vision.benchmark`
- [Command-line tools](#command-line-tools)

## Conventions

**Units.** Metres, radians, and pixels throughout. Matrices are `float64` numpy arrays.

**Pixels.** Pixel-centre coordinates: the centre of the top-left pixel is `(0, 0)`, +x right, +y down. Upstream
AprilTag puts `(0, 0)` at that pixel's top-left corner, so its coordinates are 0.5 px larger in both axes.

**Corner order.** OpenCV's ArUco/AprilTag order, which is upstream AprilTag's `p[1], p[0], p[3], p[2]`. The order is
fixed to the tag's pattern, not to the image, so it rotates with the tag.

**Transforms.** A 4×4 rigid transform named `a_to_b` is frame *b*'s pose in frame *a*: it maps coordinates in *b*
to coordinates in *a*. So `field_to_camera` is the camera's pose on the field, and `field_to_camera[:3, 3]` is the
camera's position.

**Frames.**

| Frame | Axes | Used by |
| --- | --- | --- |
| World / field | +X forward, +Y left, +Z up (WPILib). Positive yaw is counter-clockwise seen from above. | `tag_layout`, `CameraPoseObservation`, `PoseWithCovariance` |
| WPILib camera | +X forward (out of the lens), +Y left, +Z up | `field_to_camera_*`, `camera_to_tag()`, `camera_to_tag_translation` |
| OpenCV camera | +X right, +Y down, +Z forward | `SingleTagPoseObservation` `rvec`/`tvec` and `covariance_*` |
| WPILib tag | Face in the local Y/Z plane, +X the face normal pointing out of the printed side | `tag_layout` values, `camera_to_tag()` |
| OpenCV tag | Face in the local X/Y plane, corners at `(±s/2, ±s/2, 0)` (`geometry.tag_object_points`) | `SingleTagPoseObservation` `rvec`/`tvec` |

**Covariances.** A pose's 6×6 covariance is ordered `[x, y, z, rx, ry, rz]`, the same as ROS
`PoseWithCovariance`. The first three are the translation error, in the parent frame. The last three are a small
rotation about the parent frame's axes, applied on the left: `R_true = exp([r]×) · R_estimated`. "Parent frame" is
the first frame in the transform's name: the field for `field_to_camera`, the camera for `camera_to_tag`. See
[Uncertainty](#uncertainty) for how they're computed.

## Detection

### `NapaDetector`

```python
NapaDetector(*, family="tag36h11", min_tag_px=32, max_hamming=None, min_contrast=20, min_border_cells=22,
             sharpening=0.25, max_detections=64)
```

The native detector for one tag family. [The guide](guide.md#detection) explains each parameter. Invalid values
raise `ValueError`.

| Parameter | Range | Meaning |
| --- | --- | --- |
| `family` | `families()` | The tag family. The default, tag36h11, is the FRC family. To read several families, use one detector per family. |
| `min_tag_px` | 8–30000 | Smallest tag-size square side, in pixels, that is guaranteed to be sampled (see [Tag families](#tag-families)) |
| `max_hamming` | `None`, 0–3 | Payload bits that may be corrected. `None` uses the family's default (see [Tag families](#tag-families)); larger values are capped at what the family's code distance allows. Read back the value in use from `max_hamming`. |
| `min_contrast` | 1–255 | Minimum grey-level difference between tag black and its white quiet zone |
| `min_border_cells` | 0–28 | Border cells that must read dark before decoding, out of tag36h11's 28; scaled for other families |
| `sharpening` | 0–1 | Cell sharpening before decoding |
| `max_detections` | ≥ 1 | Results returned per frame |

**`detect(image) -> list[FiducialImageObservation]`**: detects tags in a `uint8` grayscale (H×W), BGR (H×W×3), or BGRA
(H×W×4) array. Strided views are accepted. Frames smaller than 16 px on a side return `[]`; frames larger than
30000 px raise `ValueError`.

**`close()`**: frees the native buffers now. Later `detect` calls raise `RuntimeError`.

One instance reuses buffers between calls. Don't call one instance from two threads at once; use one per camera
stream. The native call releases the GIL, so separate instances run in parallel.

### `FiducialImageObservation`

One detected tag. A mutable dataclass.

| Field | Type | Meaning |
| --- | --- | --- |
| `tag_id` | `int` | ID within the family |
| `corners` | `ndarray (4, 2)` | Sub-pixel corners, in the order and coordinates described under [Conventions](#conventions) |
| `hamming` | `int \| None` | Payload bits corrected; 0 is a perfect read. As upstream AprilTag. |
| `decision_margin` | `float \| None` | The smaller of the white and black payload bits' mean distance from the decision threshold, in grey levels. As upstream. |
| `family` | `str` | The tag family, `"tag36h11"` unless the detector reads another |

`hamming` and `decision_margin` are `None` when a custom detector doesn't report them.

| Property | Type | Meaning |
| --- | --- | --- |
| `homography` | `ndarray (3, 3)` | Map from upstream AprilTag's tag coordinates to pixels, with `H[2, 2] == 1`. Tag coordinates span `[-1, 1]` across the black square, +x right and +y down on the official artwork. Same as upstream's `H`, apart from the half-pixel offset. |
| `center` | `ndarray (2,)` | The tag centre in pixels, `H @ (0, 0, 1)`: where the square's diagonals cross. Same as upstream's `c`. Under perspective it differs from the corners' mean. |

Both are computed from `corners` each time you read them, so they stay consistent if you edit `corners`. Corners
that don't form a quadrilateral raise `ValueError`. To map any tag point, for example to find where a cell lands,
use `H @ [u, v, 1]` and divide by the third component. To compute a homography from your own corner array, use
`geometry.tag_homography(corners)`.

### `families() -> tuple[str, ...]`

The families Napa detects, tag36h11 first. Also importable from the package root.

### Tag families

Napa reads every family upstream AprilTag defines. Each has a square whose corners Napa reports; its side is the tag
size to give pose solvers, and `min_tag_px` is measured on it. Corners are in the same order for every family (see
[Conventions](#conventions)).

| Family | Codes | Min. distance | Square, cells | Printed, cells | Default `max_hamming` | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tag36h11` | 587 | 11 | 8 | 10 | 3 | FRC since 2024; the default |
| `tag16h5` | 30 | 5 | 6 | 8 | 0 | FRC 2023. Codes are close: see below. |
| `tag25h9` | 35 | 9 | 7 | 9 | 0 | |
| `tag36h10` | 2320 | 10 | 8 | 10 | 2 | |
| `tagStandard41h12` | 2115 | 12 | 5 | 9 | 3 | Upstream's recommended family |
| `tagStandard52h13` | 48714 | 13 | 6 | 10 | 3 | |
| `tagCircle21h7` | 38 | 7 | 5 | 9 | 0 | |
| `tagCircle49h12` | 65535 | 12 | 5 | 11 | 3 | |
| `tagCustom48h12` | 42211 | 12 | 6 | 10 | 3 | |

**Classic families** (tag16h5, tag25h9, tag36h10, tag36h11) are a black square, with a one-cell black border around
the payload and a white quiet zone around it. The square is the tag size. **tagStandard, tagCircle, and tagCustom**
tags are the reverse: a white square inside a one-cell black frame, with more data cells outside the frame. The white
square is the tag size. Napa reads them on the inverted image, where they look like a classic tag.

**Default corrections.** Each family's default `max_hamming` is the most bits it can correct while a random
tag-shaped pattern stays no likelier to decode than it is in tag36h11 at 3 bits (about 1 in 10,000 per candidate).
Families whose codes are close together get fewer: tag16h5's 30 codes are only 5 bits apart, so correcting even one
bit would let about 1 in 30 random patterns through. Those families also get a *soft* correction: up to three bits
(at most what the code distance allows) may be corrected, but only when they are exactly the payload's least certain
readings, the samples nearest the black/white threshold, as blur and compression leave them. A random pattern's
wrong bits are its least certain ones only rarely, so this adds about as many false IDs as exact decoding has.

### `FiducialDetector`

A `typing.Protocol`: anything with `detect(image) -> list[FiducialImageObservation]`. `VisionProcessor` and
`napa-benchmark` accept any implementation, so you can feed them another detector's results.

### `native_available() -> bool`

Whether the compiled C core loads on this platform. When it doesn't, `NapaDetector()` raises `NativeLibraryError`
(a `RuntimeError`) explaining why. Set `NAPA_CORE_LIBRARY` to load a library you built yourself.

## Pose processing

### `VisionProcessor`

```python
VisionProcessor(*, detector=None, calibration=None, tag_size_m=None, tag_layout=None,
                max_reprojection_error_px=5.0, max_tag_reprojection_error_px=None,
                camera_pose_bounds=None, max_outlier_iterations=8, ransac_iterations=100,
                corner_noise_px=1.0)
```

Runs a detector and, given a calibration, solves poses. Use one instance per camera stream. It opens no camera,
starts no thread, and keeps no image.

| Parameter | Meaning |
| --- | --- |
| `detector` | Any `FiducialDetector`. Defaults to a new `NapaDetector()`. |
| `calibration` | A `CameraCalibration` that records its `image_size`. Without one, `process` returns detections only. |
| `tag_size_m` | Black-square side in metres. Required with `calibration`, and only with it. |
| `tag_layout` | `{tag_id: field_to_tag}`, 4×4 WPILib tag poses in your world frame. Enables `camera_pose`. Copied at construction. |
| `max_reprojection_error_px` | Largest accepted RMS corner reprojection error for a solution, and for a multi-tag pose overall |
| `max_tag_reprojection_error_px` | Largest accepted per-tag RMS error in a multi-tag pose. Defaults to `max_reprojection_error_px`. |
| `camera_pose_bounds` | A `CameraPoseBounds`; camera poses outside it are rejected |
| `max_outlier_iterations` | How many tags a multi-tag solve may drop as outliers |
| `ransac_iterations` | RANSAC iterations once three or more tags are visible |
| `corner_noise_px` | Assumed standard deviation of each corner coordinate, in pixels, for every covariance. See [Uncertainty](#uncertainty). |

Inconsistent arguments raise `ValueError`.

**`process(image, *, timestamp, capture_monotonic_s=None) -> FrameResult`**: detects and solves one frame.
`timestamp` (seconds, any clock) and `capture_monotonic_s` are copied into the result unchanged; use the frame's
capture time so a filter can align it with odometry. Raises `ValueError` for a non-`uint8`, empty, or wrongly shaped
image or non-finite times, and `CalibrationError` if the frame's size isn't the calibration's. Detector exceptions
propagate.

### `FrameResult`

| Field | Type | Present |
| --- | --- | --- |
| `timestamp` | `float` | Always |
| `capture_monotonic_s` | `float \| None` | As passed |
| `image_observations` | `list[FiducialImageObservation]` | Always |
| `single_tag_poses` | `list[SingleTagPoseObservation]` | With a calibration: one per tag that could be solved |
| `tag_angles` | `list[TagAngleObservation]` | With a calibration: one per entry of `single_tag_poses` |
| `camera_pose` | `CameraPoseObservation \| None` | With a `tag_layout`, when a pose was accepted |
| `pose_diagnostics` | `PoseSolveDiagnostics \| None` | With a calibration |
| `health` | `dict` | `detection_ms` and `solve_ms`: wall time spent detecting and solving poses. Covariances are computed later, when first read, so `solve_ms` doesn't include them. |

### `SingleTagPoseObservation`

A tag's two planar (IPPE) pose solutions. A flat tag seen through noisy corners has two poses that both fit, mirror
images about the line of sight. Solution 0 fits better, but when the two errors are close, either may be the true
one.

| Field | Type | Meaning |
| --- | --- | --- |
| `tag_id` | `int` | |
| `pose_0_rvec`, `pose_0_tvec` | `ndarray (3, 1)` | Solution 0: maps OpenCV tag points into the OpenCV camera frame, as `cv2.solvePnP` returns |
| `error_0` | `float` | Solution 0's RMS corner reprojection error, px |
| `pose_1_rvec`, `pose_1_tvec`, `error_1` | as above, or `None` | Solution 1, when OpenCV returns a valid second solution in front of the camera |
| `distance` | `float` | Solution 0's camera-to-tag distance, m |
| `covariance_0`, `covariance_1` | `ndarray (6, 6) \| None` | Each solution's covariance in the OpenCV camera frame, including the [ambiguity term](#ambiguity). Computed when first read (see [Cost](#cost)); read-only, but you can assign a replacement. |

| Member | Returns |
| --- | --- |
| `camera_to_tag(solution=0)` | `PoseWithCovariance`: the tag's pose in WPILib camera axes and the WPILib tag frame, with the covariance rotated to match. Raises `ValueError` if that solution or its covariance is missing. |
| `ambiguity` | `error_0 / error_1`, as PhotonVision and Limelight define it: 0 means unambiguous, values near 1 mean either solution may be right. `None` without solution 1. |

### `CameraPoseObservation`

The camera's pose in the field, from every accepted tag.

| Field | Type | Meaning |
| --- | --- | --- |
| `tag_ids` | `list[int]` | Tags used in the pose |
| `field_to_camera_0` | `ndarray (4, 4)` | The camera's pose (WPILib camera axes) |
| `error_0` | `float` | RMS corner reprojection error, px |
| `field_to_camera_1`, `error_1` | as above, or `None` | With one tag, the other IPPE solution; `None` for multi-tag poses |
| `average_tag_distance` | `float` | Mean camera-to-tag distance, m |
| `confidence` | `float` | Heuristic `1 / (1 + error + 0.1 · distance)`, kept for compatibility. Prefer the covariance. |
| `diagnostics` | `PoseSolveDiagnostics \| None` | The solve that produced this pose |
| `covariance_0`, `covariance_1` | `ndarray (6, 6) \| None` | Each solution's covariance in the field frame, including the [ambiguity term](#ambiguity) for one tag or for tags that all lie in one plane. Computed when first read; read-only, but you can assign a replacement. |

| Member | Returns |
| --- | --- |
| `field_to_camera(solution=0)` | `PoseWithCovariance` of the camera in the field. Raises `ValueError` if that solution or its covariance is missing. |
| `ambiguity` | `error_0 / error_1` for a single-tag pose, `None` for a multi-tag pose |

With one tag, a pose is reported only when both IPPE solutions are within `max_reprojection_error_px`. With two or
more, Napa solves all corners jointly (SQPnP), falls back to RANSAC when three or more tags disagree, and drops the
worst tag and retries until every tag is within the limits.

### `PoseSolveDiagnostics`

Why the camera pose came out as it did, also reported when no pose was accepted.

| Field | Meaning |
| --- | --- |
| `attempted_tag_ids` | Every detected tag |
| `accepted_tag_ids` | Tags in the final pose |
| `rejected_tag_reasons` | `{tag_id: reason}` for tags left out: not in the layout, duplicate, bad corners, or outlier |
| `per_tag_rms_reprojection_error_px` | Each tag's RMS reprojection error under the last solve |
| `overall_rms_reprojection_error_px` | All corners' RMS error under the last solve |
| `iterations` | Solve attempts, including outlier rounds |
| `used_ransac` | Whether RANSAC produced the pose |
| `rejection_reason` | Why no pose was accepted, or `None` |

### `TagAngleObservation`

Viewing angles, corrected for lens distortion, in radians from the optical axis. Positive is right and down
(OpenCV's image axes), so negate `center_ty` for Limelight's up-positive `ty`.

| Field | Type | Meaning |
| --- | --- | --- |
| `tag_id` | `int` | |
| `corner_angles` | `ndarray (4, 2)` | `(horizontal, vertical)` angle of each corner |
| `center_tx`, `center_ty` | `float` | Mean of the corner angles |
| `center_covariance` | `ndarray (2, 2) \| None` | Covariance of `(center_tx, center_ty)`, rad². Computed when first read. |
| `distance` | `float` | Solution 0's camera-to-tag distance, m |
| `camera_to_tag_translation` | `ndarray (3,)` | Solution 0's translation in WPILib camera axes |

### `CameraPoseBounds`

```python
CameraPoseBounds(
    field_length_m=None, field_width_m=None, field_margin_m=0.0, min_camera_height_m=None, max_camera_height_m=None
)
```

Physical limits for a camera pose: x in `[-margin, length + margin]`, y in `[-margin, width + margin]`, and z
between the height limits. `None` disables a limit. **`rejection_reason(position) -> str | None`** explains why a
position fails, or returns `None`.

## Uncertainty

Module `napa_vision.covariance`. Napa attaches a covariance to every pose and tag angle it computes, so a Kalman
filter can weight each measurement by how well it is actually constrained. A distant tag, a steep view, or a lone
tag gets a large covariance; many well-spread tags get a small one. You don't have to tune standard deviations
against distance by hand.

### `PoseWithCovariance`

```python
PoseWithCovariance(transform, covariance)
```

A frozen dataclass: a 4×4 pose and its 6×6 covariance in the [convention above](#conventions). Both arrays are
copied and validated; bad shapes or non-finite values raise `ValueError`.

| Member | Returns |
| --- | --- |
| `translation`, `rotation` | Copies of the translation `(3,)` and rotation `(3, 3)` |
| `std_devs()` | `sqrt(diag(covariance))`: `[x, y, z, rx, ry, rz]` |
| `transform_by(transform)` | `self.transform @ transform`, as WPILib's `Pose3d.transformBy`. Moves to a rigidly attached frame, for example `field_to_camera.transform_by(camera_to_robot)`. |
| `relative_to(origin)` | `inverse(origin) @ self.transform`, as `Pose3d.relativeTo`. Re-expresses the pose in another world frame, such as the red alliance's origin. |
| `inverse()` | The parent's pose in the child frame, with the covariance in the child frame |
| `rebased(new_from_old)` | The same pose with both frames' axes rotated by a 3×3 rotation, for example `geometry.OPENCV_TO_WPILIB_ROTATION` |
| `to_pose2d()` | `Pose2dWithCovariance`: the pose projected onto the parent's X/Y plane. Raises `ValueError` if the child's +X axis is vertical. |

Transforms passed to `transform_by` and `relative_to` are treated as exact: measure your camera mount well.

A pose you construct yourself is propagated to first order. A pose from Napa's solvers for a planar target, one tag
or several on one wall, also carries the weighted solutions its covariance summarizes (see
[Ambiguity](#ambiguity)). Every operation then moves each solution exactly and recomputes the spread between them in
the new frame. That matters because the spread can cover tens of degrees, which a first-order propagation would
distort badly: inverting a lone tag's pose that way, for example, left it 10× overconfident in tests. Copies and
unpickled poses keep the solutions; `dataclasses.replace()` drops them, so a replaced covariance is used as given.

### `Pose2dWithCovariance`

A frozen dataclass with `x`, `y`, `yaw` (counter-clockwise, the heading of the child's +X axis) and a 3×3
`covariance`, with x and y along the parent's axes.

- **`std_devs()`** returns `[σx, σy, σyaw]` along the parent's axes.
- **`local_std_devs()`** returns them along the pose's own forward and left axes. This is what WPILib's
  `addVisionMeasurement` expects: its estimators apply vision standard deviations in the robot's frame.

### `camera_pose_covariance`

```python
camera_pose_covariance(field_to_camera, field_points, camera_matrix, distortion_coefficients,
                       corner_noise_px=1.0) -> PoseWithCovariance | None
```

The covariance of a camera pose you solved yourself, from the `(K, 3)` field points whose image corners it was
solved from (`geometry.tag_field_corners` gives a tag's four). Returns `None` when the points don't constrain all
six degrees of freedom. It gives the linearized covariance with the [correlation cap](#curved-error-surfaces), but it
can't know about a second solution, so it leaves out the [ambiguity widening](#ambiguity). For one tag, or tags on
one wall seen nearly head-on, prefer the covariances Napa's solvers attach.

### `pose_difference`

```python
napa_vision.covariance.pose_difference(transform, other) -> ndarray (6,)
```

`other`'s offset from `transform` in the covariance convention: `[t_other - t, r]` with `R_other = exp([r]×) R`. Use
it as the innovation in a 6-D filter, so that residuals and covariances agree.

### How covariances are computed

Each corner coordinate is modelled as independent Gaussian noise with standard deviation `corner_noise_px`. The
pose covariance starts from the Gauss–Newton approximation `σ² (JᵀJ)⁻¹`, where `J` is the Jacobian of the projected
corners, lens distortion included, with respect to the pose, evaluated at the reported pose. That is the linearized
Cramér–Rao bound: the scatter an efficient solver reaches when the noise model is right.

Tag angles use the same noise, propagated through undistortion and `atan` to each corner's angle, then averaged.

A linearization alone isn't safe to give a filter, because planar targets seen from far away or nearly head-on have
error distributions that no single Gaussian at the estimate describes. Napa makes the two adjustments below. Monte
Carlo tests checked the result on single tags at 1–4 m and 5–40°, and on 144 random walls of 1–4 tags, 2–10 m away,
seen head-on to 60°, with 0.3–1 px of noise, with and without lens distortion. A correct covariance leaves 5% of
errors outside its 95% ellipse. In the median geometry, 1% fell outside; in the worst, a pair of tags 10 m away at
54° and only about 15 px across (smaller than the default detector reliably finds), 9–12% did.
In the typical geometry each axis's reported standard deviation is about 5% wider than the actual scatter. Where a
mirror solution is plausible, frames on which it fits well get deliberately wide covariances.

#### Ambiguity

A planar target, whether one tag or several on one wall, has two pose solutions that fit its corners, mirror
images about the line of sight. Seen nearly head-on or from far away, its tilt is barely observable, and the two can
trade places under noise. A linearized covariance about one of them then understates the error badly: in Napa's
tests, a lone tag 2–4 m away within a few degrees of head-on had normalized errors 20–30 times what it allowed.

So Napa describes such a pose as a weighted set of solutions, and reports the set's spread about the solved pose.
Each solution is weighted by how well it fits the corners, by its tempered likelihood `exp(-SSR / 2Tσ²)`, where SSR
is its sum of squared corner residuals. An unambiguous target is unaffected; a truly ambiguous one's covariance
grows to cover both solutions, so a filter isn't misled by the wrong one.

- **One tag.** The set is its two IPPE solutions, tempered by T = 2. The error surface between them isn't
  quadratic, and untempered weights left ambiguous tags overconfident.
- **Several tags in (nearly) one plane.** Napa finds the plane's mirror solution, then steps along the rotation from
  the solved pose to it and beyond: −¼, ¼, ½, ¾, 1, and 1¼ of the way. At each step it re-solves the other five
  degrees of freedom against the true corners, so every step's fit is honest. The steps, untempered, cover both a
  separate mirror solution and the long, flat valley a wall has when seen head-on, which a single mirror candidate
  misjudges in either direction. Tags at clearly different depths break the ambiguity and get no set. Napa skips the
  steps when the mirror is both more than 30 standard deviations away and fits far worse (by 100σ²). Over 4,080
  frames of random walls, that skipped 61% of them, and none of the skipped steps would have changed a standard
  deviation.

#### Curved error surfaces

Even with one clear solution, the error surface of a distant or nearly flat target is curved. The linearization can
then report some combination of axes, typically a rotation and the translation it mimics, as far more certain than
it is, even when every individual axis is right: in tests, that left mean NEES at 50–90 for 6 dimensions, against 6
expected. Napa therefore adds 10% of each variance back to the diagonal of every reported pose covariance
(`covariance.CORRELATION_SLACK`). That caps every correlation at 0.91 and widens each axis's standard deviation by
5%. A 2% cap still left up to 21% of errors outside the 95% ellipse on wide walls of tags.

#### Cost

Napa computes covariances lazily: `process()` returns poses without them, and a covariance is computed the first time
you read it, through the field or through `field_to_camera()` and `camera_to_tag()`. Reading one tag's covariance
computes every tag's from that frame in one batch. Results that are never read cost nothing, and results remain
picklable with their deferred work. The deferred work keeps its own copies of what it needs, so changing the camera
or the corners afterwards doesn't affect it. Results may be read from several threads: two first reads at once may
both compute, but neither sees a partial result. If you edit a pose or assign a covariance, the pose methods use your
values as a plain pose.

On one Apple-silicon core, solving poses took about 0.03–0.06 ms with one tag, 0.15 ms with three, and 0.3 ms with
eight. Reading the camera pose's covariance, as a Kalman filter does, added about 0.1–0.2 ms; reading every tag's
covariance and angle covariance as well, another 0.1–0.2 ms. A wall of tags whose mirror solution is plausible
costs more, about 0.6 ms extra with three tags, for its mirror steps.

#### Choosing `corner_noise_px`

The default of 1 px is deliberately conservative. Napa's corners are far more accurate than that on clean synthetic
frames, but a real camera adds motion blur, rolling shutter, calibration error, and an imperfectly flat or
mismeasured tag. All of these look like corner noise to the solver. Calibration error is also correlated from frame
to frame, which no per-frame covariance can capture. To tune, log poses of a stationary robot and compare the
scatter with the reported standard deviations. Scaling `corner_noise_px` by *k* scales tag-angle covariances, and
pose covariances without ambiguity widening, by *k²*. Widened ones change differently, because their weights depend
on it. Set it on the processor rather than rescaling results.

## Solvers

`VisionProcessor` composes these. Use them directly to solve corners from another source. Each takes a
`camera` with `calibration_matrix` (3×3) and `distortion_coefficients` attributes. That is the `CameraGeometry`
protocol, which any object with those two attributes satisfies, for example
`SimpleNamespace(calibration_matrix=K, distortion_coefficients=D)`.

### `SingleTagPoseSolver`

```python
SingleTagPoseSolver(max_reprojection_error_px=5.0, *, corner_noise_px=1.0)
```

- **`solve(observation, camera, tag_size_m) -> SingleTagPoseObservation | None`** returns `None` for non-finite corners
  or when no solution lies in front of the camera within `max_reprojection_error_px`.
- **`solve_many(observations, camera, tag_size_m) -> list[SingleTagPoseObservation | None]`** does the same for each
  observation, by index, and computes all the covariances in one batch. Prefer it for more than one tag.

### `CameraPoseSolver`

```python
CameraPoseSolver(max_reprojection_error_px=5.0, *, max_tag_reprojection_error_px=None, bounds=None,
                 max_outlier_iterations=8, ransac_iterations=100, corner_noise_px=1.0)
```

**`solve(observations, single_tag_poses, camera, tag_size_m, tag_layout) -> CameraPoseObservation | None`**.
`single_tag_poses` is needed only for the one-tag case. Check `last_diagnostics`, a `PoseSolveDiagnostics`, to see
why a call returned `None`.

### `TagAngleSolver`

```python
TagAngleSolver(*, corner_noise_px=1.0)
```

- **`solve(observation, camera, single_tag_pose) -> TagAngleObservation`**.
- **`solve_many(observations, camera, single_tag_poses) -> list[TagAngleObservation]`** takes paired lists with
  matching tag IDs and raises `ValueError` otherwise.

## Calibration

### `CheckerboardSpec(columns, rows, square_size_m)`

A checkerboard's inner-corner `columns` and `rows`, and its square size in metres.

### `CameraCalibration`

A frozen dataclass; arrays are copied and made read-only. Invalid values raise `CalibrationError` (a `ValueError`).

| Field | Meaning |
| --- | --- |
| `camera_matrix` | 3×3 pinhole matrix with positive focal lengths |
| `distortion_coefficients` | OpenCV distortion: 4, 5, 8, 12, or 14 values |
| `image_size` | `(width, height)` it was measured at, or `None`. `VisionProcessor` requires it. |
| `provenance` | An `ArtifactProvenance` |
| `rms_reprojection_error_px`, `mean_reprojection_error_px`, `views_used` | Calibration quality, or `None` |
| `checkerboard` | The `CheckerboardSpec` used, or `None` |
| `rejected_images` | Images in which the board wasn't found |

- **`validate_for_image_size(width, height)`** raises `CalibrationError` unless the size matches `image_size`.
- **`require_production()`** raises unless the calibration is `measured` and records its size, error, and view count.

### Functions

| Function | Does |
| --- | --- |
| `load_calibration_data(path) -> CameraCalibration` | Reads a calibration JSON file. A file without a `provenance` block is marked `imported`. |
| `save_calibration_data(path, calibration)` | Writes one |
| `discover_checkerboard_images(directory) -> tuple[Path, ...]` | Lists a directory's BMP, JPEG, PNG, and TIFF files, sorted. Raises `CalibrationError` if the directory doesn't exist. |
| `calibrate_checkerboard_images(image_paths, checkerboard, *, minimum_views=8, calibration_flags=0) -> CameraCalibration` | Calibrates with OpenCV, marking the result `measured`. Raises `CalibrationError` with fewer than `minimum_views` (at least 3) complete views or with images of mixed sizes. |

The file format is JSON with `schema_version` (1), `camera_matrix`, `distortion_coefficients`,
`image_size: {width, height}`, and optionally `rms_reprojection_error_px`, `mean_reprojection_error_px`,
`views_used`, `checkerboard: {columns, rows, square_size_m}`, `rejected_images`, and
`provenance: {kind, source, notes}`.

## Provenance

`ArtifactProvenance(kind, source, notes=None)` records where a calibration came from. `kind` is a `ProvenanceKind`:
`EXAMPLE`, `IMPORTED`, `MEASURED`, or `OFFICIAL`. Its `production_ready` property is true for `MEASURED` and
`OFFICIAL`. `require_production(name)` and `require_kind(kind, name)` raise `ProvenanceError` otherwise.
`to_dict()` and `from_dict(value, *, default_source)` convert to and from the JSON block.

## Geometry helpers

Module `napa_vision.geometry`: plain numpy functions, apart from the two marked as needing OpenCV.

| Function | Does |
| --- | --- |
| `transform_matrix(translation, rotation_rpy)` | Builds a 4×4 transform from a translation and roll, pitch, yaw (applied as `Rz · Ry · Rx`) |
| `euler_xyz_to_matrix(roll, pitch, yaw)` | The 3×3 rotation `Rz · Ry · Rx` |
| `invert_transform(transform)` | Inverse of a rigid transform |
| `transform_points(transform, points)` | Applies a transform to `(N, 3)` points |
| `rvec_tvec_to_transform(rvec, tvec)` | OpenCV `rvec`/`tvec` to 4×4 (needs OpenCV) |
| `opencv_pose_to_wpilib_transform(rvec, tvec)` | The same pose re-expressed in WPILib axes on both sides (needs OpenCV) |
| `opencv_translation_to_wpilib(translation)` | One vector from OpenCV to WPILib axes |
| `wpilib_points_to_opencv(points)` | `(N, 3)` points from WPILib to OpenCV axes |
| `tag_object_points(tag_size_m)` | A tag's corners in the OpenCV tag frame, in Napa's corner order (read-only, cached) |
| `tag_field_corners(field_to_tag, tag_size_m)` | A tag's corners in the world frame, in Napa's corner order |
| `tag_homography(corners)` | The upstream-convention homography of four corners (see `FiducialImageObservation.homography`) |
| `rotation_matrix_to_quaternion_wxyz(transform)` | Unit quaternion `(w, x, y, z)` of a transform's rotation |

`WPILIB_TO_OPENCV_ROTATION` and `OPENCV_TO_WPILIB_ROTATION` are the 3×3 basis changes, and `OPENCV_TO_WPILIB_BASIS`
is the 4×4 form.

## Drawing tags

Module `napa_vision.tags`, behind the `napa-tag` command. Needs only numpy. Tags are drawn as the official
AprilRobotics artwork and printed FRC tags are, which is what field layouts and Napa's pose solvers assume. OpenCV's
`DICT_APRILTAG_36h11` draws the same tags turned 180°. Invalid arguments raise `ValueError`.

| Name | Returns |
| --- | --- |
| `TAG36H11_COUNT` | 587, the number of tag36h11 IDs |
| `family_layout(family="tag36h11")` | A frozen `FamilyLayout`: `name`, `ncodes`, `nbits`, `width` (cells across the tag-size square), `total_width` (cells across the printed tag), `reversed`, `min_distance`, `cells` (each code bit's cell, in the core's orientation), and `codes` |
| `tag_cells(tag_id, *, family="tag36h11", quiet_zone_cells=1)` | Boolean grid, `True` for white: the printed pattern inside `quiet_zone_cells` of white. A classic family's pattern is its black square (8 cells for tag36h11), so the default is the official 10×10 image; a reversed family's pattern is the whole printed tag. Cells that carry nothing, such as tagCircle tags' corners, are white, as on the official images. |
| `tag_image(tag_id, *, family="tag36h11", cell_px=1, quiet_zone_cells=1)` | The grid as a `uint8` image, 0 black and 255 white, `cell_px` pixels per cell |
| `tag_svg(tag_id, *, family="tag36h11", size_mm=165.1, quiet_zone_cells=1, label=True)` | An SVG document whose tag-size square prints `size_mm` wide. `label` adds the family, ID, and size in grey below the quiet zone. |
| `png_bytes(image)` | A `uint8` grayscale image encoded as PNG, with no imaging library |

## pupil_apriltags compatibility

Module `napa_vision.compat`: the `pupil_apriltags` / `dt-apriltags` interface on Napa's detector. Its results use
upstream AprilTag's conventions, not those under [Conventions](#conventions).

### `Detector`

```python
Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=2.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
    searchpath=None,
    **napa_options,
)
```

`families` is a space-separated list of any of `families()`, as upstream; an unknown family raises `ValueError`.
Napa runs one detector per family. `decode_sharpening` sets Napa's `sharpening`.
The other upstream parameters are kept in `params` but have no effect. `napa_options` go to `NapaDetector`, for
example `min_tag_px=24`.

**`detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None) -> list[Detection]`**: detects tags in a
`uint8` grayscale image. With `estimate_tag_pose=True`, `camera_params` is `(fx, fy, cx, cy)` and `tag_size` the side
of the tag-size square in metres (see [Tag families](#tag-families)); leaving either out raises `ValueError`. Poses need
OpenCV.

### `Detection`

| Attribute | Type | Meaning |
| --- | --- | --- |
| `tag_family` | `bytes` | The family, such as `b"tag36h11"` |
| `tag_id`, `hamming`, `decision_margin` | | As `FiducialImageObservation` |
| `corners` | `ndarray (4, 2)` | Upstream's `p`: Napa's corners in the order `[1, 0, 3, 2]` (counter-clockwise in the image from the official artwork's bottom-left corner), plus 0.5 px, since `(0, 0)` is the top-left pixel's corner |
| `center` | `ndarray (2,)` | Upstream's `c`, in the same coordinates |
| `homography` | `ndarray (3, 3)` | Upstream's `H`, in the same coordinates, with `H[2, 2] == 1` |
| `pose_R`, `pose_t` | `ndarray (3, 3)`, `(3, 1)` | With `estimate_tag_pose`: maps upstream's tag frame (+x right and +y down on the official artwork, +z into the tag) into the OpenCV camera frame. `None` otherwise. |
| `pose_err` | `float` | Upstream's object-space error of that pose, m². `None` without `estimate_tag_pose`. |

Poses follow upstream's `estimate_tag_pose`: both planar solutions (from OpenCV's IPPE), each refined by orthogonal
iteration on upstream's object-space error, and the one with the lower error kept. Lens distortion is ignored, as
upstream ignores it.

## Benchmarking

Module `napa_vision.benchmark`, behind the `napa-benchmark` command. [The guide](guide.md#measuring-on-your-own-footage)
describes the manifest.

| Name | Does |
| --- | --- |
| `load_corpus_manifest(path) -> CorpusManifest` | Reads and validates a manifest. Raises `CorpusManifestError`. |
| `run_replay_benchmark(manifest, *, detector=None, options=None) -> dict` | Runs a detector (`NapaDetector()` by default) over the corpus and returns the summary |
| `BenchmarkOptions(repetitions=1, warmup_passes=1, max_failure_samples=25, verify_integrity=True)` | Run settings |
| `BenchmarkThresholds(min_recall=None, min_precision=None, max_p95_total_ms=None)` | Pass/fail gates |
| `evaluate_benchmark_gate(summary, thresholds) -> dict` | Applies the gates to a summary |
| `serialize_benchmark_summary(summary, *, pretty=False) -> str` | Summary as JSON |
| `write_benchmark_summary(summary, path, *, pretty=False, durable=True) -> Path` | Writes the summary atomically (fsynced when `durable`) |

## Command-line tools

| Command | Arguments |
| --- | --- |
| `napa-tag IDS... [--family NAME] [--output DIR] [--format svg\|png] [--size-mm MM] [--cell-px N] [--quiet-zone N] [--no-label]` | Writes `FAMILY-NNN.svg` or `.png` files (default family tag36h11); IDs like `3`, `0-9`, or `1,4,7-8`. Exits 2 on bad input. |
| `napa-calibrate IMAGES --output FILE --columns N --rows N --square-size-m M [--minimum-views N]` | Calibrates from a directory of checkerboard photographs |
| `napa-benchmark MANIFEST [--repetitions N] [--warmup-passes N] [--max-failure-samples N] [--no-verify-integrity] [--min-recall R] [--min-precision P] [--max-p95-total-ms T] [--output FILE] [--pretty]` | Scores Napa on an annotated corpus; exits 1 on a failed gate and 2 on bad input |
