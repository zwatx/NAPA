# API reference

Every class, method, and field NAPA exposes. The [guides](README.md) explain how to use them together. This page is
the lookup.

Everything here imports from the package root unless a module is named:

```python
from napa_vision import NapaDetector, VisionProcessor, PoseWithCovariance
```

Detection, `center` and `homography`, the result types, and covariance propagation (`PoseWithCovariance`) need only
NumPy. Solving poses and covariances from corners, calibrating, and `napa-benchmark` need OpenCV. Install
`napa-vision[headless]` or `napa-vision[gui]`. Without OpenCV, the calls that need it raise a `RuntimeError` that
says so.

Contents:

- [Conventions](#conventions)
- [Detection](#detection). `NapaDetector`, `FiducialImageObservation`, `families`, `native_available`,
  `estimate_camera_gamma`, `FiducialDetector`
- [Pose processing](#pose-processing). `VisionProcessor`, `FrameResult`, and the observations it holds
- [Uncertainty](#uncertainty). `PoseWithCovariance`, `Pose2dWithCovariance`, `camera_pose_covariance`,
  `pose_difference`
- [Solvers](#solvers). `SingleTagPoseSolver`, `CameraPoseSolver`, `TagAngleSolver`
- [Stereo](#stereo). `StereoCalibration`, `StereoPoseSolver`, results, and paired calibration
- [Calibration](#calibration) and [provenance](#provenance)
- [Geometry helpers](#geometry-helpers). `napa_vision.geometry`
- [Drawing tags](#drawing-tags). `napa_vision.tags`
- [pupil_apriltags compatibility](#pupil_apriltags-compatibility). `napa_vision.compat`
- [Benchmarking](#benchmarking). `napa_vision.benchmark`
- [Command-line tools](#command-line-tools)

## Conventions

**Units.** Metres, radians, and pixels throughout. Matrices are `float64` NumPy arrays.

**Pixels.** Pixel-centre coordinates. The centre of the top-left pixel is (0, 0), +x right, +y down. Upstream
AprilTag puts (0, 0) at that pixel's top-left corner, so its coordinates are 0.5 px larger in both axes.

**Corner order.** OpenCV's ArUco and AprilTag order, which is upstream AprilTag's `p[1], p[0], p[3], p[2]`. The order
is fixed to the tag's pattern, not to the image, so it rotates with the tag.

**Transforms.** A 4x4 rigid transform named `a_to_b` is frame b's pose in frame a. It maps coordinates in b to
coordinates in a. So `field_to_camera` is the camera's pose on the field, and `field_to_camera[:3, 3]` is the
camera's position.

**Frames.**

| Frame | Axes | Used by |
| --- | --- | --- |
| World / field | +X forward, +Y left, +Z up (WPILib). Positive yaw is counter-clockwise seen from above. | `tag_layout`, `CameraPoseObservation`, `PoseWithCovariance` |
| WPILib camera | +X forward (out of the lens), +Y left, +Z up | `field_to_camera_*`, `camera_to_tag()`, `camera_to_tag_translation` |
| OpenCV camera | +X right, +Y down, +Z forward | `SingleTagPoseObservation` `rvec`/`tvec` and `covariance_*` |
| WPILib tag | Face in the local Y/Z plane, +X the face normal pointing out of the printed side | `tag_layout` values, `camera_to_tag()` |
| OpenCV tag | Face in the local X/Y plane, corners at (+-s/2, +-s/2, 0) (`geometry.tag_object_points`) | `SingleTagPoseObservation` `rvec`/`tvec` |

**Covariances.** A pose's 6x6 covariance is ordered `[x, y, z, rx, ry, rz]`, the same as ROS
`PoseWithCovariance`. The first three entries are the translation error, in the parent frame. The last three are a
small rotation about the parent frame's axes, applied on the left: `R_true = exp([r]x) @ R_estimated`. The parent
frame is the first frame in the transform's name. For `field_to_camera` it is the field. For `camera_to_tag` it is
the camera. [Uncertainty](#uncertainty) explains how the numbers are computed.

## Detection

### `NapaDetector`

```text
NapaDetector(*, family="tag36h11", min_tag_px=22, max_hamming=None, min_contrast=20, min_border_cells=22,
             sharpening=0.25, max_detections=64, split_touching=False, recover_shadows=False,
             shadow_min_tag_px=None, soft_decode=False, align_corners=False, camera_gamma=None, tag_shape=None)
```

The native detector for one tag family. [Choosing settings](detection.md#choosing-settings) and
[Tuning](detection.md#tuning) explain when to change each one. Invalid values raise `ValueError`. Asking for a
feature the loaded native library does not have raises `NativeLibraryError`.

| Parameter | Range | Meaning |
| --- | --- | --- |
| `family` | `families()` | The tag family. The default, tag36h11, is the FRC family. One detector per family. |
| `min_tag_px` | 8 to 30000 | Smallest tag-size square side, in pixels, that the search is guaranteed to sample. Default 22. The value 32 with `recover_shadows=False` runs at about NAPA 0.5.1's speed. |
| `max_hamming` | `None`, 0 to 3 | Payload bits that may be corrected. `None` uses the family's default (see [Tag families](#tag-families)). Larger values are capped at what the family's code distance allows. Read the value in use from `detector.max_hamming`. |
| `min_contrast` | 1 to 255 | Smallest grey-level gap between tag black and its white quiet zone |
| `min_border_cells` | 0 to 28 | Border cells that must read dark before decoding, out of tag36h11's 28. Scaled for other families. |
| `sharpening` | 0 to 1 | Cell sharpening before decoding |
| `max_detections` | 1 or more | Results returned per frame. A frame with more tags warns (`RuntimeWarning`) and drops the rest. |
| `split_touching` | `bool` | Also decode blobs split where they touch only at a corner or a narrow neck, for AprilGrid boards and checkerboards. Those tags come after the ordinary ones, which are unchanged. Slower. |
| `recover_shadows` | `bool` | Experimental. Add the lower-threshold second search for tags merged into dim backgrounds. Appends exact-code recoveries after ordinary results. |
| `shadow_min_tag_px` | `int` or `None` | Sampling minimum for recovery only, from `min_tag_px` to 30000. Requires `recover_shadows=True`. `None` keeps full sensitivity. |
| `soft_decode` | `bool` | Experimental, tag36h11 only. Add tiny tags (about 8 to 19 px) read as a whole picture, and tags with a flat patch over part of the payload. They carry `soft_decoded=True` and `hamming=None`. Non-tags can pass. |
| `align_corners` | `bool` | Experimental, tag36h11 only. Move blurred tags' corners onto their own picture after decoding. Costs time on blurred tags. |
| `camera_gamma` | `None`, 1 to 4 | Experimental, tag36h11 only. The gamma of a camera that stores gamma-encoded video. Corners are then fitted in linear light. Leave unset for a linear camera. `None` and 1.0 change nothing. |
| `tag_shape` | `None`, `"tall"`, `"wide"` | `"tall"` halves the rows the search samples, for tags turned about the vertical axis. `"wide"` halves the columns, for a camera on its side. Fitting and decoding read the whole frame. Needs `min_tag_px` of 31 or less for tag36h11. The exact limit depends on the family's width and is reported in the `ValueError`. |

**`detect(image) -> list[FiducialImageObservation]`** detects tags in a `uint8` greyscale (HxW), BGR (HxWx3), or
BGRA (HxWx4) array. Strided views are accepted. Frames smaller than 16 px on a side return `[]`. A side larger than
30000 px raises `ValueError`.

**`close()`** frees the native buffers now. Later `detect` calls raise `RuntimeError`.

One instance reuses buffers between calls. Use one instance per camera stream, and do not call it from two threads
at once. The native call releases the GIL, so separate instances run in parallel. The settings in force are readable
as attributes (`family`, `min_tag_px`, `max_hamming`, `split_touching`, `recover_shadows`, `shadow_min_tag_px`,
`soft_decode`, `align_corners`, `camera_gamma`, `tag_shape`).

### `FiducialImageObservation`

One detected tag. A mutable dataclass.

| Field | Type | Meaning |
| --- | --- | --- |
| `tag_id` | `int` | ID within the family |
| `corners` | `ndarray (4, 2)` | Sub-pixel corners, in the order and coordinates under [Conventions](#conventions) |
| `hamming` | `int` or `None` | Payload bits corrected. 0 is a perfect read, as upstream AprilTag. `None` when `soft_decoded` is true, or when a custom detector does not report it. |
| `soft_decoded` | `bool` | True for a tag from `soft_decode=True`, read from its whole picture rather than bit by bit |
| `decision_margin` | `float` or `None` | The smaller of the white and black payload bits' mean distance from the decision threshold, in grey levels, as upstream. `None` when a custom detector does not report it. |
| `family` | `str` | The tag family. `"tag36h11"` unless the detector reads another. |

| Property | Type | Meaning |
| --- | --- | --- |
| `homography` | `ndarray (3, 3)` | Map from upstream AprilTag's tag coordinates to pixels, with `H[2, 2] == 1`. Tag coordinates span [-1, 1] across the tag-size square, +x right and +y down on the official artwork. The same as upstream's `H`, apart from the half-pixel offset. |
| `center` | `ndarray (2,)` | The tag centre in pixels, `H @ (0, 0, 1)`. Where the square's diagonals cross, as upstream's `c`. Under perspective it differs from the corners' mean. |

Both properties are computed from `corners` each time you read them, so they stay consistent if you edit `corners`.
Corners that do not form a quadrilateral raise `ValueError`. To map any tag point, for example to find where a cell
lands, use `H @ [u, v, 1]` and divide by the third component. To compute a homography from your own corner array,
use `geometry.tag_homography(corners)`.

### `families() -> tuple[str, ...]`

The families NAPA detects, tag36h11 first.

### Tag families

NAPA reads every family upstream AprilTag defines. Each family has a square whose corners NAPA reports. Its side is
the tag size to give pose solvers, and `min_tag_px` is measured on it. Corner order is the same for every family.

| Family | Codes | Min distance | Square, cells | Printed, cells | Default `max_hamming` | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tag36h11` | 587 | 11 | 8 | 10 | 3 | FRC since 2024. The default. |
| `tag16h5` | 30 | 5 | 6 | 8 | 0 | FRC 2023. Codes are close. See below. |
| `tag25h9` | 35 | 9 | 7 | 9 | 0 | |
| `tag36h10` | 2320 | 10 | 8 | 10 | 2 | |
| `tagStandard41h12` | 2115 | 12 | 5 | 9 | 3 | Upstream's recommended family |
| `tagStandard52h13` | 48714 | 13 | 6 | 10 | 3 | |
| `tagCircle21h7` | 38 | 7 | 5 | 9 | 0 | |
| `tagCircle49h12` | 65535 | 12 | 5 | 11 | 3 | |
| `tagCustom48h12` | 42211 | 12 | 6 | 10 | 3 | |

The classic families (tag16h5, tag25h9, tag36h10, tag36h11) are a black square: a one-cell black border around the
payload, with a white quiet zone around it. That square is the tag size. The tagStandard, tagCircle, and tagCustom
families are the reverse: a white square inside a one-cell black frame, with more data cells outside the frame. The
white square is the tag size. NAPA reads them on the inverted image, where they look like a classic tag.

Each family's default `max_hamming` is the most bits it can correct while a random tag-shaped pattern stays no
likelier to decode than it is in tag36h11 at 3 bits (about 1 in 10,000 per candidate). Families whose codes sit close
together get fewer. tag16h5's 30 codes are only 5 bits apart, so correcting even one bit would let about 1 in 30
random patterns through.

Those families also get a soft correction. Up to three bits (at most what the code distance allows) may be corrected,
but only when they are exactly the payload's least certain readings, the samples nearest the black/white threshold,
as blur and compression leave them. A random pattern's wrong bits are its least certain ones only rarely, so this
adds about as many false IDs as exact decoding has. This is separate from the experimental `soft_decode` option,
despite the similar name.

### `estimate_camera_gamma(frames, *, min_frames=30, **detector_options) -> CameraGammaEstimate`

Reads a camera's gamma from frames that show tag36h11 tags, for `NapaDetector(camera_gamma=...)`. Each frame is
detected with gammas from 1.0 to 3.1 undone. The result is where the black border's width bias crosses zero.
`detector_options` go to the `NapaDetector` it uses. See [Camera gamma](detection.md#camera-gamma) for accuracy and
limits.

`CameraGammaEstimate` is a frozen dataclass:

| Field | Meaning |
| --- | --- |
| `gamma` | `None` if fewer than `min_frames` frames showed a usable tag. 1.0 if the camera reads as linear, which means leave `camera_gamma` unset. Otherwise the estimate. |
| `low`, `high` | Middle 80% of estimates from frames resampled with replacement, or `None`. Frame-to-frame scatter only. On a real camera the estimate can sit about 10% below the true value. |
| `frames`, `tags` | Frames and tag sightings used |
| `bias` | Mean border-width bias in cells at each assumed gamma |

### `FiducialDetector`

A `typing.Protocol`: anything with `detect(image) -> list[FiducialImageObservation]`. `VisionProcessor` and
`napa-benchmark` accept any implementation, so you can feed them another detector's results.

### `native_available() -> bool`

Whether the compiled C core loads on this platform. When it does not, `NapaDetector()` raises `NativeLibraryError`
(a `RuntimeError`) explaining why. Set `NAPA_CORE_LIBRARY` to load a library you built yourself. An explicit path
that fails to load raises instead of falling back.

## Pose processing

### `VisionProcessor`

```text
VisionProcessor(*, detector=None, calibration=None, tag_size_m=None, tag_layout=None,
                max_reprojection_error_px=5.0, max_tag_reprojection_error_px=None,
                camera_pose_bounds=None, max_outlier_iterations=8, ransac_iterations=100,
                corner_noise_px=1.0, rolling_shutter=None)
```

Runs a detector and, given a calibration, solves poses. Use one instance per camera stream. It opens no camera,
starts no thread, and keeps no image.

| Parameter | Meaning |
| --- | --- |
| `detector` | Any `FiducialDetector`. Defaults to a new `NapaDetector()`. |
| `calibration` | A `CameraCalibration` that records its `image_size`. Without one, `process` returns detections only. |
| `tag_size_m` | Tag-size square side in metres. Required with `calibration`, and only with it. |
| `tag_layout` | `{tag_id: field_to_tag}`, 4x4 WPILib tag poses in your world frame. Enables `camera_pose`. Copied at construction. |
| `max_reprojection_error_px` | Largest accepted RMS corner reprojection error for a solution, and for a multi-tag pose overall |
| `max_tag_reprojection_error_px` | Largest accepted per-tag RMS error in a multi-tag pose. Defaults to `max_reprojection_error_px`. |
| `camera_pose_bounds` | A `CameraPoseBounds`. Camera poses outside it are rejected. |
| `max_outlier_iterations` | How many tags a multi-tag solve may drop as outliers |
| `ransac_iterations` | RANSAC iterations once three or more tags are visible |
| `corner_noise_px` | Assumed standard deviation of each corner coordinate, in pixels, for every covariance. See [Uncertainty](#uncertainty). |
| `rolling_shutter` | A `RollingShutter`, unset by default. Corrects the corners that feed the pose solvers. Requires `calibration`. See [Rolling-shutter cameras](poses.md#rolling-shutter-cameras). |

Inconsistent arguments raise `ValueError`.

**`process(image, *, timestamp, capture_monotonic_s=None) -> FrameResult`** detects and solves one frame.
`timestamp` (seconds, any clock) and `capture_monotonic_s` are copied into the result unchanged. Use the frame's
capture time so a filter can align it with odometry. Raises `ValueError` for a non-`uint8`, empty, or wrongly shaped
image or for non-finite times, and `CalibrationError` if the frame's size is not the calibration's. Detector
exceptions propagate.

### `RollingShutter(readout_s, direction="top_to_bottom", max_frame_gap_s=0.1)`

A frozen dataclass. `readout_s` is the time from reading the first line to the last, for the frame as delivered.
`direction` is `"top_to_bottom"`, `"bottom_to_top"`, `"left_to_right"`, or `"right_to_left"`, in the delivered image.
`max_frame_gap_s` is the longest gap between sightings of a tag that still gives a usable speed.

### `estimate_rolling_shutter(frames, *, calibration, tag_size_m, tag_layout=None, detector=None, min_sightings=30, max_frame_gap_s=0.1) -> RollingShutterEstimate`

Reads a rolling shutter's readout time and direction from `(timestamp, image)` pairs of a moving camera. Without a
layout, only tags 60 px or more across count. With `tag_layout` (tag ID to field-to-tag transform), it fits one
camera pose per frame across the layout's tags instead of each tag on its own, which works for small tags. A sighting
is then a frame. See [Reading the readout from a recording](poses.md#measuring-the-readout).

`RollingShutterEstimate` is a frozen dataclass:

| Field | Meaning |
| --- | --- |
| `rolling_shutter` | A `RollingShutter` to pass to `VisionProcessor`, or `None` |
| `readout_s` | `None` if the recording cannot tell. 0.0 if it reads as a global shutter on the axes checked. Otherwise the estimate. |
| `low`, `high` | Middle 80% of estimates from the sightings resampled in half-second blocks, or `None`. Scatter only. |
| `sightings` | Sightings used: tags, or frames when `tag_layout` is given |
| `axes_checked` | `"rows"` and/or `"columns"`. The read axes the recording's motion could show. |

### `FrameResult`

| Field | Type | Present |
| --- | --- | --- |
| `timestamp` | `float` | Always |
| `capture_monotonic_s` | `float` or `None` | As passed |
| `image_observations` | `list[FiducialImageObservation]` | Always |
| `single_tag_poses` | `list[SingleTagPoseObservation]` | With a calibration: one per tag that could be solved |
| `tag_angles` | `list[TagAngleObservation]` | With a calibration: one per entry of `single_tag_poses` |
| `camera_pose` | `CameraPoseObservation` or `None` | With a `tag_layout`, when a pose was accepted |
| `pose_diagnostics` | `PoseSolveDiagnostics` or `None` | With a calibration |
| `uncorrected_tag_ids` | `tuple[int, ...]` | With `rolling_shutter`: tags with no usable speed, solved as detected. Empty otherwise. |
| `health` | `dict` | `detection_ms` and `solve_ms`, the wall time spent detecting and solving. Covariances are computed later, when first read, so `solve_ms` excludes them. |

### `SingleTagPoseObservation`

A tag's two planar (IPPE) pose solutions. A flat tag seen through noisy corners has two poses that both fit, mirror
images about the line of sight. Solution 0 fits better, but when the two errors are close, either may be the true one.

| Field | Type | Meaning |
| --- | --- | --- |
| `tag_id` | `int` | |
| `pose_0_rvec`, `pose_0_tvec` | `ndarray (3, 1)` | Solution 0. Maps OpenCV tag points into the OpenCV camera frame, as `cv2.solvePnP` returns. |
| `error_0` | `float` | Solution 0's RMS corner reprojection error, px |
| `pose_1_rvec`, `pose_1_tvec`, `error_1` | As above, or `None` | Solution 1, when OpenCV returns a valid second solution in front of the camera |
| `distance` | `float` | Solution 0's camera-to-tag distance, m |
| `covariance_0`, `covariance_1` | `ndarray (6, 6)` or `None` | Each solution's covariance in the OpenCV camera frame, including the [ambiguity term](#how-covariances-are-computed). Computed when first read (see [How covariances are computed](#how-covariances-are-computed)). Read-only, but you can assign a replacement. |

| Member | Returns |
| --- | --- |
| `camera_to_tag(solution=0)` | `PoseWithCovariance`: the tag's pose in WPILib camera axes and the WPILib tag frame, with the covariance rotated to match. Raises `ValueError` if that solution or its covariance is missing. |
| `ambiguity` | `error_0 / error_1`, as PhotonVision and Limelight define it. 0 means unambiguous. Values near 1 mean either solution may be right. `None` without solution 1. |

### `CameraPoseObservation`

The camera's pose in the field, from every accepted tag.

| Field | Type | Meaning |
| --- | --- | --- |
| `tag_ids` | `list[int]` | Tags used in the pose |
| `field_to_camera_0` | `ndarray (4, 4)` | The camera's pose, in WPILib camera axes |
| `error_0` | `float` | RMS corner reprojection error, px |
| `field_to_camera_1`, `error_1` | As above, or `None` | With one tag, the other IPPE solution. `None` for multi-tag poses. |
| `average_tag_distance` | `float` | Mean camera-to-tag distance, m |
| `confidence` | `float` | Heuristic `1 / (1 + error + 0.1 * distance)`, kept for compatibility. Prefer the covariance. |
| `diagnostics` | `PoseSolveDiagnostics` or `None` | The solve that produced this pose |
| `covariance_0`, `covariance_1` | `ndarray (6, 6)` or `None` | Each solution's covariance in the field frame, including the [ambiguity term](#how-covariances-are-computed) for one tag or for tags that all lie in one plane. Computed when first read. Read-only, but you can assign a replacement. |

| Member | Returns |
| --- | --- |
| `field_to_camera(solution=0)` | `PoseWithCovariance` of the camera in the field. Raises `ValueError` if that solution or its covariance is missing. |
| `ambiguity` | `error_0 / error_1` for a single-tag pose. `None` for a multi-tag pose. |

With one tag, a pose is reported only when both IPPE solutions fall within `max_reprojection_error_px`. With two or
more, NAPA solves all corners jointly (SQPnP), falls back to RANSAC when three or more tags disagree, and drops the
worst tag and retries until every tag is within the limits.

**Error units.** Pose errors and rejection limits use the RMS Euclidean distance per corner,
`sqrt(mean(dx**2 + dy**2))`. Single-tag errors, multi-tag errors, and diagnostics all use it. OpenCV's returned
scalar averages over x and y separately, so NAPA converts it. Earlier builds used that smaller scalar for finite
single-tag errors, which understated them by a factor of sqrt(2). With an unchanged numeric limit, the corrected
solver can reject poses earlier builds accepted. Multiplying an old single-tag limit by sqrt(2) restores its previous
effective tolerance. Scaling a shared processor or camera-solver limit also relaxes multi-tag rejection, so leave
shared limits unchanged if you want consistent corner-distance rejection. The compatibility `confidence` heuristic
changes with the corrected error, and stereo's monocular fallback uses the corrected limits too. None of this fixes
planar ambiguity.

### `PoseSolveDiagnostics`

Why the camera pose came out as it did. Also reported when no pose was accepted.

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

Viewing angles, corrected for lens distortion, in radians from the optical axis. Positive is right and down (OpenCV's
image axes), so negate `center_ty` for Limelight's up-positive `ty`.

| Field | Type | Meaning |
| --- | --- | --- |
| `tag_id` | `int` | |
| `corner_angles` | `ndarray (4, 2)` | (horizontal, vertical) angle of each corner |
| `center_tx`, `center_ty` | `float` | Mean of the corner angles |
| `center_covariance` | `ndarray (2, 2)` or `None` | Covariance of `(center_tx, center_ty)`, rad squared. Computed when first read. |
| `distance` | `float` | Solution 0's camera-to-tag distance, m |
| `camera_to_tag_translation` | `ndarray (3,)` | Solution 0's translation in WPILib camera axes |

### `CameraPoseBounds`

```python
CameraPoseBounds(
    field_length_m=None, field_width_m=None, field_margin_m=0.0, min_camera_height_m=None, max_camera_height_m=None
)
```

Physical limits for a camera pose. x must lie in `[-margin, length + margin]`, y in `[-margin, width + margin]`, and z
between the height limits. `None` disables a limit. **`rejection_reason(position) -> str | None`** explains why a
position fails, or returns `None`.

## Uncertainty

Module `napa_vision.covariance`. NAPA attaches a covariance to every pose and tag angle it computes, so a Kalman
filter can weight each measurement by how well it is actually constrained. A distant tag, a steep view, or a lone tag
gets a large covariance. Many well-spread tags get a small one.

### `PoseWithCovariance`

```python
PoseWithCovariance(transform, covariance)
```

A frozen dataclass: a 4x4 pose and its 6x6 covariance in the [convention above](#conventions). Both arrays are copied
and validated. Bad shapes or non-finite values raise `ValueError`.

| Member | Returns |
| --- | --- |
| `translation`, `rotation` | Copies of the translation (3,) and rotation (3, 3) |
| `std_devs()` | `sqrt(diag(covariance))`: `[x, y, z, rx, ry, rz]` |
| `transform_by(transform)` | `self.transform @ transform`, as WPILib's `Pose3d.transformBy`. Moves to a rigidly attached frame, for example `field_to_camera.transform_by(camera_to_robot)`. |
| `relative_to(origin)` | `inverse(origin) @ self.transform`, as `Pose3d.relativeTo`. Re-expresses the pose in another world frame, such as the red alliance's origin. |
| `inverse()` | The parent's pose in the child frame, with the covariance in the child frame |
| `rebased(new_from_old)` | The same pose with both frames' axes rotated by a 3x3 rotation, for example `geometry.OPENCV_TO_WPILIB_ROTATION` |
| `to_pose2d()` | `Pose2dWithCovariance`: the pose projected onto the parent's X/Y plane. Raises `ValueError` if the child's +X axis is vertical. |

Transforms passed to `transform_by` and `relative_to` are treated as exact. Measure your camera mount well.

Poses from NAPA's solvers for planar targets carry their solution set (see [Ambiguity](#how-covariances-are-computed)), so transforms move
each solution exactly instead of propagating to first order. That matters: first-order inversion of a lone tag's pose was
10x overconfident in tests. Poses you construct propagate to first order. Copies keep the set; `dataclasses.replace()`
drops it.

### `Pose2dWithCovariance`

A frozen dataclass with `x`, `y`, `yaw` (counter-clockwise, the heading of the child's +X axis) and a 3x3 `covariance`,
with x and y along the parent's axes.

- **`std_devs()`** returns `[sigma_x, sigma_y, sigma_yaw]` along the parent's axes.
- **`local_std_devs()`** returns them along the pose's own forward and left axes. This is what WPILib's
  `addVisionMeasurement` expects, because its estimators apply vision standard deviations in the robot's frame.

### `camera_pose_covariance`

```text
camera_pose_covariance(field_to_camera, field_points, camera_matrix, distortion_coefficients,
                       corner_noise_px=1.0) -> PoseWithCovariance | None
```

Covariance for a pose you solved yourself, from the `(K, 3)` field points it was solved from
(`geometry.tag_field_corners` gives a tag's four). `None` if under-constrained. Linearized only, with the correlation cap
but no [ambiguity widening](#how-covariances-are-computed). For single tags or walls seen head-on, prefer solver covariances.

### `pose_difference`

```text
napa_vision.covariance.pose_difference(transform, other) -> ndarray (6,)
```

`other`'s offset from `transform` in the covariance convention: `[t_other - t, r]` with `R_other = exp([r]x) @ R`. Use
it as the innovation in a 6D filter, so residuals and covariances agree.

### How covariances are computed

Corners are modelled as independent Gaussian noise, std dev `corner_noise_px` (default 1.0, deliberately conservative).
The base covariance is the Gauss-Newton `sigma^2 * (J^T J)^-1` at the solved pose, distortion included. Two fixes keep
it honest on planar targets:

- **Ambiguity.** A flat target has two mirror pose solutions. Head-on or far away they can swap under noise, and a
  single Gaussian then understates error by 20-30x in tests. So planar poses report the spread of a fit-weighted set of
  solutions (both IPPE solutions for one tag; stepped mirror candidates for a wall). Unambiguous poses are unaffected.
  Ambiguous ones get covariances wide enough to cover both branches.
- **Curved error surfaces.** Distant flat targets can look falsely certain along combined axes. Every reported
  covariance gets 10% of each variance added to its diagonal (`CORRELATION_SLACK`), capping correlations at 0.91.

Monte Carlo checks (single tags 1-4 m, walls to 10 m, 0.3-1 px noise): a correct 95% ellipse excludes 5% of errors.
NAPA excluded 1% in median geometry, 9-12% in the worst (tiny distant tags). Typical std devs run ~5% wide.

Covariances are lazy: computed on first read, batched per frame, free if never read. Results stay picklable and are
safe to read from several threads. Rough cost on one Apple-silicon core: solving 0.03-0.3 ms for 1-8 tags, plus
0.1-0.4 ms if you read the covariances.

Tune `corner_noise_px` on the processor, not by rescaling output. Scaling it by k scales unwidened covariances by k².
Measure it from a stationary robot's pose scatter, and remember calibration error repeats every frame, which no
per-frame covariance captures.

## Solvers

`VisionProcessor` composes these. Use them directly to solve corners from another source. Each takes a `camera` with
`calibration_matrix` (3x3) and `distortion_coefficients` attributes. That is the `CameraGeometry` protocol, which any
object with those two attributes satisfies. Note the name: `CameraCalibration` calls its matrix `camera_matrix`, so it
does not satisfy the protocol on its own. Adapt it like this:

```python
from types import SimpleNamespace

camera = SimpleNamespace(
    calibration_matrix=calibration.camera_matrix,
    distortion_coefficients=calibration.distortion_coefficients,
)
```

### `SingleTagPoseSolver`

```text
SingleTagPoseSolver(max_reprojection_error_px=5.0, *, corner_noise_px=1.0)
```

- **`solve(observation, camera, tag_size_m) -> SingleTagPoseObservation | None`** returns `None` for non-finite corners or
  when no solution lies in front of the camera within `max_reprojection_error_px`.
- **`solve_many(observations, camera, tag_size_m) -> list[SingleTagPoseObservation | None]`** does the same for each
  observation, by index, and computes all covariances in one batch. Prefer it for more than one tag.

### `CameraPoseSolver`

```text
CameraPoseSolver(max_reprojection_error_px=5.0, *, max_tag_reprojection_error_px=None, bounds=None,
                 max_outlier_iterations=8, ransac_iterations=100, corner_noise_px=1.0)
```

**`solve(observations, single_tag_poses, camera, tag_size_m, tag_layout) -> CameraPoseObservation | None`**.
`single_tag_poses` is needed only for the one-tag case. Check `last_diagnostics`, a `PoseSolveDiagnostics`, to see why a
call returned `None`.

### `TagAngleSolver`

```text
TagAngleSolver(*, corner_noise_px=1.0)
```

- **`solve(observation, camera, single_tag_pose) -> TagAngleObservation`**
- **`solve_many(observations, camera, single_tag_poses) -> list[TagAngleObservation]`** takes paired lists with matching
  tag IDs and raises `ValueError` otherwise.

## Stereo

[Stereo rigs](stereo.md) explains timing, calibration, and what the covariance covers. Loading a rig calibration needs only
NumPy. Fitting one, checking epipolar errors, and solving poses need OpenCV.

### `StereoCalibration`

```python
StereoCalibration(
    left,
    right,
    rig_to_left,
    rig_to_right,
    provenance,
    left_id="left",
    right_id="right",
    rms_reprojection_error_px=None,
    views_used=None,
)
```

`left` and `right` are `CameraCalibration` objects with recorded resolutions. `rig_to_left` and `rig_to_right` are 4x4
camera poses in the rig, in WPILib axes, not OpenCV's left-to-right extrinsics. Camera IDs must be distinct, non-empty
strings. All inputs are copied and stored read-only.

| Member | Does |
| --- | --- |
| `require_production()` | Raises unless both cameras' calibrations and the rig's are measured, and the fit's RMS error and view count are recorded |
| `epipolar_errors(left_points, right_points)` | One symmetric epipolar distance per matched point, in undistorted pixels. Both inputs are Nx2. |

`save_stereo_calibration(path, calibration)` and `load_stereo_calibration(path)` write and read version-1 JSON with
`coordinate_convention="wpilib_camera_to_rig"`. Invalid files raise `CalibrationError`.

`calibrate_stereo_points(object_points, left_image_points, right_image_points, left, right, *, source)` returns a
`StereoCalibration` with both intrinsics fixed and the left camera as the rig's origin. Inputs are matched lists of Nx3
target points (metres) and Nx2 image points (pixels), with at least three views and four points per view. Both cameras must
have the same recorded resolution.

### `StereoPoseSolver`

```text
StereoPoseSolver(calibration, tag_layout, tag_size_m, *, max_capture_skew_s, max_reprojection_error_px=5.0,
                 max_outlier_iterations=2, max_iterations=50, corner_noise_px=(1.0, 1.0))
solver.solve(left_observations, right_observations, *, left_capture_time_s, right_capture_time_s,
             allow_monocular_fallback=False) -> StereoPoseResult
```

| Parameter | Meaning |
| --- | --- |
| `tag_layout` | `{tag_id: field_to_tag}`, 4x4 rigid transforms. Every tag is `tag_size_m` across. |
| `max_capture_skew_s` | Largest accepted gap between the two capture times. Required, with no default. |
| `max_reprojection_error_px` | Largest accepted RMS corner error for each camera/tag observation |
| `max_outlier_iterations` | How many camera/tag observations may be dropped as outliers |
| `max_iterations` | Refinement iterations allowed for each starting pose |
| `corner_noise_px` | Corner noise standard deviation for each camera, in pixels |

Observations are `FiducialImageObservation` lists from unrectified frames at the calibrated resolutions. Capture times are
exposure midpoints, in seconds, on one clock.

### `StereoPoseResult` and `StereoSolveDiagnostics`

| Result field | Meaning |
| --- | --- |
| `status` | `"stereo"`, `"rejected"`, or `"monocular"` when fallback is allowed |
| `timestamp` | The pair's midpoint, or the chosen camera's capture time for a monocular result |
| `poses` | Every distinct candidate kept, as `PoseWithCovariance`, best first. Empty when rejected. |
| `pose` | The best candidate, or `None` |
| `diagnostics` | A `StereoSolveDiagnostics` for this solve |

Each pose maps rig coordinates into field coordinates, with a covariance in the convention under [Conventions](#conventions).
It covers the spread between candidates plus independent corner noise, and treats the calibration, mounts, and layout as
exact.

Diagnostics hold `capture_skew_s`, `attempted_groups`, `accepted_groups`, `rejected_group_reasons`, and `per_group_rms_px`,
keyed by `(camera_id, tag_id)`. `candidate_rms_px` lists each kept candidate's joint RMS error, and `iterations` counts the
refinement iterations of fits that converged. `rejection_reason` explains a rejected result, and `fallback_reason` explains
why a monocular result could not use stereo.

## Calibration

### `CheckerboardSpec(columns, rows, square_size_m)`

A checkerboard's inner-corner `columns` and `rows`, and its square size in metres.

### `CameraCalibration`

A frozen dataclass. Arrays are copied and made read-only. Invalid values raise `CalibrationError` (a `ValueError`).

| Field | Meaning |
| --- | --- |
| `camera_matrix` | 3x3 pinhole matrix with positive focal lengths |
| `distortion_coefficients` | OpenCV distortion: 4, 5, 8, 12, or 14 values |
| `image_size` | `(width, height)` it was measured at, or `None`. `VisionProcessor` requires it. |
| `provenance` | An `ArtifactProvenance` |
| `rms_reprojection_error_px`, `mean_reprojection_error_px`, `views_used` | Calibration quality, or `None` |
| `checkerboard` | The `CheckerboardSpec` used, or `None` |
| `rejected_images` | Images in which the board was not found |

- **`validate_for_image_size(width, height)`** raises `CalibrationError` unless the size matches `image_size`.
- **`require_production()`** raises unless the calibration is `measured` and records its size, error, and view count. Note
  that this is stricter than `ArtifactProvenance.production_ready`, which also accepts `official`.

### Functions

| Function | Does |
| --- | --- |
| `load_calibration_data(path) -> CameraCalibration` | Reads a calibration JSON file. A file without a `provenance` block is marked `imported`. |
| `save_calibration_data(path, calibration)` | Writes one |
| `discover_checkerboard_images(directory) -> tuple[Path, ...]` | Lists a directory's BMP, JPEG, PNG, and TIFF files, sorted. Raises `CalibrationError` if the directory does not exist. |
| `calibrate_checkerboard_images(image_paths, checkerboard, *, minimum_views=8, calibration_flags=0) -> CameraCalibration` | Calibrates with OpenCV and marks the result `measured`. Raises `CalibrationError` with fewer than `minimum_views` (at least 3) complete views, or with images of mixed sizes. |

The file format is JSON with `schema_version` (1), `camera_matrix`, `distortion_coefficients`, and
`image_size: {width, height}`. Optional fields are `rms_reprojection_error_px`, `mean_reprojection_error_px`, `views_used`,
`checkerboard: {columns, rows, square_size_m}`, `rejected_images`, and `provenance: {kind, source, notes}`.

## Provenance

`ArtifactProvenance(kind, source, notes=None)` records where a calibration came from. `kind` is a `ProvenanceKind`:
`EXAMPLE`, `IMPORTED`, `MEASURED`, or `OFFICIAL`. Its `production_ready` property is true for `MEASURED` and `OFFICIAL`.
`require_production(name)` and `require_kind(kind, name)` raise `ProvenanceError` otherwise. `to_dict()` and
`from_dict(value, *, default_source)` convert to and from the JSON block.

## Geometry helpers

Module `napa_vision.geometry`. Plain NumPy functions, apart from the two marked as needing OpenCV.

| Function | Does |
| --- | --- |
| `transform_matrix(translation, rotation_rpy)` | Builds a 4x4 transform from a translation and roll, pitch, yaw (applied as Rz @ Ry @ Rx) |
| `euler_xyz_to_matrix(roll, pitch, yaw)` | The 3x3 rotation Rz @ Ry @ Rx |
| `invert_transform(transform)` | Inverse of a rigid transform |
| `transform_points(transform, points)` | Applies a transform to (N, 3) points |
| `rvec_tvec_to_transform(rvec, tvec)` | OpenCV `rvec`/`tvec` to 4x4 (needs OpenCV) |
| `opencv_pose_to_wpilib_transform(rvec, tvec)` | The same pose re-expressed in WPILib axes on both sides (needs OpenCV) |
| `opencv_translation_to_wpilib(translation)` | One vector from OpenCV to WPILib axes |
| `wpilib_points_to_opencv(points)` | (N, 3) points from WPILib to OpenCV axes |
| `tag_object_points(tag_size_m)` | A tag's corners in the OpenCV tag frame, in NAPA's corner order (read-only, cached) |
| `tag_field_corners(field_to_tag, tag_size_m)` | A tag's corners in the world frame, in NAPA's corner order |
| `tag_homography(corners)` | The upstream-convention homography of four corners (see `FiducialImageObservation.homography`) |
| `rotation_matrix_to_quaternion_wxyz(transform)` | Unit quaternion (w, x, y, z) of a transform's rotation |

`WPILIB_TO_OPENCV_ROTATION` and `OPENCV_TO_WPILIB_ROTATION` are the 3x3 basis changes, and `OPENCV_TO_WPILIB_BASIS` is the
4x4 form.

## Drawing tags

Module `napa_vision.tags`, behind the `napa-tag` command. Needs only NumPy. Tags follow the official AprilRobotics artwork,
which is what field layouts and NAPA's pose solvers assume. OpenCV's `DICT_APRILTAG_36h11` draws the same tags turned 180
degrees. Invalid arguments raise `ValueError`.

| Name | Returns |
| --- | --- |
| `TAG36H11_COUNT` | 587, the number of tag36h11 IDs |
| `family_layout(family="tag36h11")` | A frozen `FamilyLayout`: `name`, `ncodes`, `nbits`, `width` (cells across the tag-size square), `total_width` (cells across the printed tag), `reversed`, `min_distance`, `cells` (each code bit's cell, in the core's orientation), and `codes` |
| `tag_cells(tag_id, *, family="tag36h11", quiet_zone_cells=1)` | Boolean grid, `True` for white: the printed pattern inside `quiet_zone_cells` of white. A classic family's pattern is its black square (8 cells for tag36h11), so the default is the official 10x10 image. A reversed family's pattern is the whole printed tag. Cells that carry nothing, such as tagCircle tags' corners, are white, as on the official images. |
| `tag_image(tag_id, *, family="tag36h11", cell_px=1, quiet_zone_cells=1)` | The grid as a `uint8` image, 0 black and 255 white, `cell_px` pixels per cell |
| `tag_svg(tag_id, *, family="tag36h11", size_mm=165.1, quiet_zone_cells=1, label=True)` | An SVG document whose tag-size square prints `size_mm` wide. `label` adds the family, ID, and size in grey below the quiet zone. |
| `png_bytes(image)` | A `uint8` greyscale image encoded as PNG, with no imaging library |

## pupil_apriltags compatibility

Module `napa_vision.compat`: the `pupil_apriltags` and `dt-apriltags` interface on NAPA's detector. Its results use upstream
AprilTag's conventions, not the ones under [Conventions](#conventions).

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

`families` is a space-separated list of any of `families()`, as upstream. An unknown family raises `ValueError`. NAPA runs one
detector per family. `decode_sharpening` sets NAPA's `sharpening`. The other upstream parameters are kept in `params` but have
no effect. `napa_options` go to `NapaDetector`, for example `min_tag_px=24`.

**`detect(img, estimate_tag_pose=False, camera_params=None, tag_size=None) -> list[Detection]`** detects tags in a `uint8`
greyscale image. With `estimate_tag_pose=True`, `camera_params` is `(fx, fy, cx, cy)` and `tag_size` is the side of the
tag-size square in metres (see [Tag families](#tag-families)). Leaving either out raises `ValueError`. Poses need OpenCV.

### `Detection`

| Attribute | Type | Meaning |
| --- | --- | --- |
| `tag_family` | `bytes` | The family, such as `b"tag36h11"` |
| `tag_id`, `hamming`, `decision_margin` | | As `FiducialImageObservation` |
| `corners` | `ndarray (4, 2)` | Upstream's `p`: NAPA's corners in the order [1, 0, 3, 2] (counter-clockwise in the image from the official artwork's bottom-left corner), plus 0.5 px, since (0, 0) is the top-left pixel's corner |
| `center` | `ndarray (2,)` | Upstream's `c`, in the same coordinates |
| `homography` | `ndarray (3, 3)` | Upstream's `H`, in the same coordinates, with `H[2, 2] == 1` |
| `pose_R`, `pose_t` | `ndarray (3, 3)`, `(3, 1)` | With `estimate_tag_pose`: maps upstream's tag frame (+x right and +y down on the official artwork, +z into the tag) into the OpenCV camera frame. `None` otherwise. |
| `pose_err` | `float` | Upstream's object-space error of that pose, m squared. `None` without `estimate_tag_pose`. |

Poses follow upstream's `estimate_tag_pose`: both planar solutions (from OpenCV's IPPE), each refined by orthogonal iteration
on upstream's object-space error, with the lower-error one kept. Lens distortion is ignored, as upstream ignores it.

## Benchmarking

Module `napa_vision.benchmark`, behind the `napa-benchmark` command. [Measuring on your own footage](measuring.md) describes
the manifest.

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
| `napa-tag IDS... [--family NAME] [--output DIR] [--format svg\|png] [--size-mm MM] [--cell-px N] [--quiet-zone N] [--no-label]` | Writes `FAMILY-NNN.svg` or `.png` files. The default family is tag36h11. IDs look like `3`, `0-9`, or `1,4,7-8`. Exits 2 on bad input. |
| `napa-calibrate IMAGES --output FILE --columns N --rows N --square-size-m M [--minimum-views N]` | Calibrates from a directory of checkerboard photographs |
| `napa-benchmark MANIFEST [--repetitions N] [--warmup-passes N] [--max-failure-samples N] [--no-verify-integrity] [--min-recall R] [--min-precision P] [--max-p95-total-ms T] [--output FILE] [--pretty]` | Scores NAPA on an annotated corpus. Exits 1 on a failed gate and 2 on bad input. |
