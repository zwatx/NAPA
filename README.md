# Northstar Vision

`northstar-vision` is an alpha-quality AprilTag 36h11 camera service for FRC-style coprocessors. It detects tags with
OpenCV, estimates single- and multi-tag poses, publishes the upstream Northstar `observations` packet over
NetworkTables, and can write complete frame records to JSONL for offline analysis.

## Install

Python 3.11 or newer and OpenCV 4.7 or newer are required. Python 3.9 support was removed because that release is
end-of-life, and the detector now uses the modern `cv2.aruco.ArucoDetector` API exclusively.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Install NetworkTables support on a robot coprocessor with:

```bash
python -m pip install '.[nt]'
```

The `nt` extra installs RobotPy's `pyntcore` bindings. Current releases provide Python 3.11–3.14 wheels for the
common desktop platforms and 64-bit Linux ARM coprocessors.

The package installs both a console command and a module entry point:

```bash
northstar-vision --help
python -m northstar_vision --help
```

## Run

Laptop camera with a debug window:

```bash
northstar-vision --profile laptop --show
```

Raspberry Pi profile (defaults to NetworkTables output):

```bash
northstar-vision --profile raspi
```

One-shot replay of a video or image directory:

```bash
northstar-vision --profile laptop --replay path/to/frames --output none
```

Add `--loop-replay` to loop intentionally. Useful configuration checks do not open a camera:

```bash
northstar-vision --list-profiles
northstar-vision --profile raspi --check-config
```

## Profiles and calibration

The `laptop` and `raspi` profiles are bundled package resources, so named profiles also work from an installed wheel.
Pass a JSON path to use an external device profile. Profiles are strictly validated: unknown fields, duplicate tag IDs,
invalid camera matrices, impossible thresholds, and unknown output modes fail at startup.

The bundled calibration matrices and tag layouts are examples, not measurements for a real camera or field. Replace
them before using pose output on a robot. Camera intrinsics are resolution-specific; startup fails when a live camera
negotiates a different resolution rather than silently producing incorrect poses.

All field and mounting transforms use WPILib axes:

- +X forward
- +Y left
- +Z up
- positive yaw counter-clockwise when viewed from above

`camera.robot_to_camera` is the pose of the camera in the robot frame. Tag poses are poses of tag-local coordinates in
the field; the tag face lies in local Y/Z and local +X is the face normal. OpenCV's right/down/forward camera basis is
converted explicitly before any field pose is returned.

The corner convention matches official AprilRobotics/FRC tag artwork and upstream Northstar. OpenCV's
`generateImageMarker()` renders the equivalent dictionary image rotated 180 degrees relative to that artwork, so
rotate generated fixtures or printable markers 180 degrees before using them with these field poses. Detection alone
will still succeed with the other orientation, but its decoded corner order will make the field transform incorrect.

## Outputs

### NetworkTables

The Raspberry Pi profile publishes to `/<device_id>/output` and uses the upstream Northstar topic names. The
`observations` array is timestamped in NetworkTables' synchronized monotonic time base, translated from the frame's
local receive time, and contains:

1. Pose count/type: `0` for none, `1` for one multi-tag pose, `2` for two ambiguous single-tag poses.
2. Eight values per pose: reprojection error, XYZ, quaternion WXYZ.
3. Ten values per detected tag: ID, eight undistorted corner angles in radians, distance in meters.

Additional `fps_apriltags`, `latency_ms`, and `health` topics provide runtime telemetry. A one-tag field estimate keeps
both IPPE solutions when both satisfy the configured reprojection limit so the robot can disambiguate using its current
heading. If either candidate fails, the global single-tag field pose is suppressed while the tag-angle record remains;
the upstream packet has no safe one-candidate single-tag discriminator. A multi-tag estimate is only produced from at
least two unique known tags.

### JSONL

JSONL is self-describing and includes image corners, both per-tag IPPE candidates, global camera-pose candidates,
tag-angle data, trig poses, quality values, and stage timing. Choose its destination explicitly when useful:

```bash
northstar-vision --profile laptop --output jsonl --output-path recordings/run.jsonl
```

Writes are buffered and flushed periodically instead of forcing an SD-card flush for every frame. Rotation and
retention are not implemented yet, so long-running deployments should prefer NetworkTables or `--output none`.

## Detection and performance

The optional skip-scan prefilter samples the image, rejects flat frames, detects on a small set of merged regions, and
falls back to full-frame detection for busy scenes. It never decodes a tag itself. It is disabled in the bundled
profiles because the optimized full-frame path is faster in the current laptop benchmark; enable it only after replay
testing on the target camera and Raspberry Pi. Use `--enable-skip-scan` for that experiment. To force the full-frame
path during comparisons:

```bash
northstar-vision --profile laptop --disable-skip-scan
```

The hot path avoids deprecated ArUco functions, repeated camera-matrix inversion, redundant full-frame contrast scans,
and per-frame JSONL flushes. Health output separates acquisition wait, post-receive processing, frame age, detection,
solve, and previous-publish timing.

## Development

```bash
python -m pip install -e '.[dev,nt]'
ruff format --check .
ruff check .
pytest --cov=northstar_vision
python -m build
```

The earlier occlusion-tolerant detector remains in the source repository under `experiments/occlusion_detector/` as
unsupported research code; it is not included in wheels or source distributions. Its legacy camera demo, benchmark
launcher, fallback test runner, and runtime re-export were removed from the supported package surface.

## Current limitations

- Live capture is still sequential; a bounded latest-frame capture worker and automatic camera reconnect are planned.
- `--robot-yaw` is static for a run. Production trig localization should ingest timestamped robot orientation.
- Pose rejection uses reprojection error and cheirality, but does not yet use multi-tag RANSAC or field-bound checks.
- The repository does not yet declare a redistribution license; choose one before publishing a package.
