# Stereo rigs

`StereoPoseSolver` fits one rig pose to the tag corners from two calibrated cameras at once. Each camera must see
at least one tag from the layout. The two can see different tags. It needs OpenCV, so install `napa-vision[headless]`.

```python
from napa_vision import NapaDetector, StereoPoseSolver, load_stereo_calibration

calibration = load_stereo_calibration("rig.json")
calibration.require_production()
solver = StereoPoseSolver(calibration, tag_layout, tag_size_m=0.1651, max_capture_skew_s=0.005)

detector = NapaDetector()
result = solver.solve(
    detector.detect(left_image),
    detector.detect(right_image),
    left_capture_time_s=left_exposure_midpoint_s,
    right_capture_time_s=right_exposure_midpoint_s,
)
if result.status == "stereo":
    field_to_rig = result.pose.transform
    covariance = result.pose.covariance
else:
    print(result.diagnostics.rejection_reason)
```

## Inputs

- **Images.** Raw, unrectified frames at each camera's calibrated resolution. Check sizes before detecting with
  `calibration.left.validate_for_image_size(width, height)`, and the same for `calibration.right`.
- **Capture times.** Each frame's exposure midpoint, in seconds, on one clock shared by both cameras. The solver
  rejects pairs further apart than `max_capture_skew_s`. That parameter has no default, because the right value
  depends on how fast your robot moves and how your cameras are triggered.
- **Pairing.** The solver does not synchronize cameras, pair frames, estimate clock offsets, or compensate for
  motion or rolling shutter. Capture, queues, threads, and filtering over time are yours.

The solver copies the calibration and the layout when you create it. Later edits to your objects do not affect it.

## Results

`result.status` is `"stereo"`, `"rejected"`, or `"monocular"` if you allow the fallback described below. A stereo
result is timestamped at the pair's midpoint. Its pose maps rig coordinates into field coordinates. In the
calibration, `rig_to_left` and `rig_to_right` are the cameras' poses in the rig, in WPILib axes.

The solver starts from each camera's own solutions, fits all the corners together, and keeps every distinct
plausible pose it finds in `result.poses`, best first. The search is local, so it is not guaranteed to find every
candidate. `result.pose` is the best one.

Each pose's covariance includes the spread to the other candidates. It assumes the calibration, the mounts, and
the field layout are exact. It does not include calibration error, clock error, or mount flex. A small covariance
alone does not prove stereo improved your accuracy.

Diagnostics identify each observation by `(camera_id, tag_id)`, with its residual and any rejection reason. By
default the solver can drop up to two observations that fail the reprojection limit, as long as both cameras keep
at least one. Rejected results have no pose. If a noisy pair fails to converge, raising `max_iterations` lets each
starting pose refine longer, at some cost in time.

Set `allow_monocular_fallback=True` to accept a one-camera pose when stereo fails. The status is then
`"monocular"`, the timestamp is that camera's capture time, and `fallback_reason` says why stereo failed. The
camera with the newer frame is tried first.

## Calibrating a rig

Calibrate each camera on its own first (see [Camera calibration](calibration.md)). Then, with the intrinsics fixed
and the cameras rigidly mounted, collect target corners that both cameras see at the same moment:

```python
from napa_vision import calibrate_stereo_points, save_stereo_calibration

rig = calibrate_stereo_points(
    object_points,
    left_image_points,
    right_image_points,
    left_calibration,
    right_calibration,
    source="paired checkerboard session",
)
save_stereo_calibration("rig.json", rig)
errors_px = rig.epipolar_errors(held_out_left_corners, held_out_right_corners)
```

`calibrate_stereo_points` takes target points in metres and image points in pixels, with at least three views and
four points per view. It keeps both cameras' intrinsics fixed and makes the left camera the rig's origin. It needs
both cameras at the same resolution. That limit belongs to this fitting helper only. The solver itself accepts
different resolutions. You can also build a `StereoCalibration` directly from separately measured camera mounts.

The saved file embeds both cameras' calibrations, their IDs, the mounts, the fit's RMS error, and the number of
views. `require_production()` checks that the calibration was measured and has fit data. It does not check that the
calibration is accurate.

Before you rely on a rig, do three checks the fit cannot do for you:

1. Measure epipolar errors on held-out frames the fit never saw.
2. Measure the baseline physically. Epipolar errors cannot reveal a wrong baseline length.
3. Confirm the timestamps line up. Nothing in the fit detects a timing offset.

## Checking a rig on recorded frames

The [paired replay evaluator](../benchmarks/README.md#stereo-paired-replay) runs the stereo solver, each camera on
its own, and a simple fusion of the two over recorded pairs. It compares all three against ground truth when you
have it.

Stereo has only been tested on synthetic frames so far. There is no measured claim yet that it improves accuracy
on real cameras. Run the replay on your own rig before you decide it helps.
