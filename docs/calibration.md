# Camera calibration

Poses are only as good as the calibration behind them. Calibrate each camera at the resolution, focus, and zoom you
will run it at. Changing any of those afterwards changes the numbers.

Photograph a checkerboard from varied positions and angles. Fill the frame, tilt the board, and cover the corners
of the image, where distortion is strongest. Then run:

```bash
napa-calibrate calibration-images/ \
  --output camera.json \
  --columns 9 --rows 6 --square-size-m 0.024 \
  --minimum-views 12
```

`--columns` and `--rows` count inner corners, not squares. `--square-size-m` is the side of one square in metres.
Measure the printed board. Do not trust the file you printed it from.

The output file records the camera matrix, distortion coefficients, image size, reprojection error, number of views
used, and which images were rejected. `VisionProcessor` refuses frames at any other resolution.

## Provenance

Every calibration file records where it came from:

| Kind | Meaning |
| --- | --- |
| `measured` | Produced by `napa-calibrate` or `calibrate_checkerboard_images` |
| `official` | Supplied by the camera or field vendor |
| `imported` | A file with no provenance block, loaded from elsewhere |
| `example` | A placeholder. Never run a robot on one. |

`CameraCalibration.require_production()` passes only for a `measured` calibration that also records its image size,
reprojection error, and view count. An `official` file does not pass it, even though provenance treats `official`
as production-ready in general. Calibrations are stricter: they must have been measured on the camera in question. Call it at startup if a placeholder reaching the field would be a real problem:

```python
from napa_vision import load_calibration_data

calibration = load_calibration_data("camera.json")
calibration.require_production()
```

From Python, `calibrate_checkerboard_images(paths, CheckerboardSpec(columns, rows, square_size_m))` does what the
command does, and `save_calibration_data` writes the result. The [API reference](api.md#calibration) describes the
file format.

To calibrate two cameras as one rig, see [Stereo rigs](stereo.md#calibrating-a-rig).
