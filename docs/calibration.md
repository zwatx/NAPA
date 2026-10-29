# Camera calibration

Photograph a checkerboard at the exact resolution, focus, and zoom you'll use, from varied positions and angles, then
run:

```bash
napa-calibrate calibration-images/ \
  --output camera.json \
  --columns 9 --rows 6 --square-size-m 0.024 \
  --minimum-views 12
```

`--columns` and `--rows` count inner corners. The output records the image size, the reprojection error, the number of
views used, and which images were rejected. `VisionProcessor` refuses frames at any other resolution.

Each calibration file records where it came from: `measured` for `napa-calibrate` output, `imported` for a file
without that record, and `example` for placeholders. `CameraCalibration.require_production()` passes only for measured
files, so a robot can refuse to run on a placeholder.

To calibrate two cameras as one rig, see [Stereo rigs](stereo.md#calibrating-a-rig).
