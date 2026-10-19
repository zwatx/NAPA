# Napa measurements

These benchmarks measure processing on generated images. They do not measure camera delivery, Raspberry Pi
thermal behavior, a headset runtime, or robot localization accuracy in the field.

Install the optional upstream AprilTag baseline from a repository checkout:

```bash
python -m pip install -e ".[headless,dev,benchmark]"
python benchmarks/performance_matrix.py --help
```

The detector matrix compares the same pre-rendered grayscale images, with construction and first calls separate
from steady-state timing. Keep rendering, scoring, file writes, and other benchmarks outside the measured runs.
Report recall and false positives with speed: skipping work can also skip a tag. Layout restriction is valid only
when the allowed tag IDs are known in advance. The archived custom detector is an ID-only experiment, not a
calibrated pose backend. No supplied corners are used in end-to-end comparisons.

Pin the run to one idle core so other processes don't skew timings (Linux shown):

```bash
taskset -c 2 python benchmarks/performance_matrix.py \
  --repetitions 100 --occlusion-repetitions 1 --warmup 1 --threads 1 \
  --resolutions 640x480,1280x720,1920x1080 --tag-side-pixels 96,160 \
  --include-robust-erased10 --output benchmarks/results/2026-10-18_detector_matrix.json
```

The matrix has 78 main frames and 864 obstruction cases. Obstructions use IDs 1, 42, and 314 at all four rotations,
both marker sizes, white and grey patches, and six requested covered fractions from 0 to 40%. They run only at the
smallest frame resolution to keep the in-memory corpus bounded. At a fixed marker size, changing the surrounding
frame resolution mainly changes search cost. Main scenes also include four distinct tag IDs, two negative scenes,
small tags, blur, perspective, and four camera-like degradations of one tag: Gaussian noise (sigma 6), a 0.35-1.0
lighting gradient, 7-pixel horizontal motion blur, and text clutter around a tag that keeps its white quiet zone. The
four-tag scene caps marker size where necessary to keep tags separated.

`napa-native` is Napa's C core. The `napa-default`, `napa-aruco3`, `napa-skip-scan`, and `napa-layout-restricted`
rows are the earlier OpenCV-based wrappers, kept for comparison.

The report's `corner_accuracy` section scores corners against exact ground truth on 120 seeded 1280x720 renders
(tag sides 32-240 px, any rotation, mild perspective, sigma 0/0.8 blur, sigma 2 noise), supersampled 4x and
area-averaged like a sensor. AprilTag reports corners with a +0.5 px origin and its own corner order; the report
shifts and re-orders them before scoring, so the comparison counts only real error.

Each reported detector-call time includes its adapter's ID extraction as well as the selected detector's own
wrapper work. Initialization measures object construction after imports, and the separately recorded first call
uses a blank 640×480 image; neither is a full process startup measurement. Compare scenario/resolution rows rather
than the aggregate across differently weighted main and obstruction stages. Obstruction rows test ID recovery,
not pose or corner reconstruction. One repeat per obstruction case supports a deterministic case count, not a
statistical confidence interval for physical obstructions.

## Calibrated pose-processing simulator

Use one OpenCV thread to reproduce the checked-in pose reports. Run each suite separately and use the same
repetition and warm-up settings for every comparison:

```bash
python -c "import cv2; cv2.setNumThreads(1); from napa_vision.simulation_cli import main; raise SystemExit(main())" \
  --profile laptop --suite standard --repetitions 30 --warmup-passes 3 \
  --compare-optimizations --output-dir simulation-output/napa-standard-threads1
python -c "import cv2; cv2.setNumThreads(1); from napa_vision.simulation_cli import main; raise SystemExit(main())" \
  --profile laptop --suite sweep --repetitions 30 --warmup-passes 3 \
  --compare-optimizations --output-dir simulation-output/napa-sweep-threads1
```

The checked-in simulator reports add `measurement_environment` to record the thread setting, CPU, and timed
stage, because the simulator's own metadata does not record its OpenCV thread count. Each candidate uses a fixed
execution block, so the matrix's interleaved detector-only results are preferable for close speed comparisons.
These stage timings include detection and pose processing, and must not be compared directly to detector-only FPS.

The standard simulator excludes its heavily blocked tag from expected detections. Use the dedicated matrix's
occlusion sweep, which retains the hidden tag in the ground truth, to assess obstruction tolerance. Geometric
covered area and pixels actually changed are different quantities, and neither equals the number of bad payload
samples. Read the [source evidence](notes/apriltag_occlusion_sources.md) before interpreting an occlusion score.

Raw measurements and environment/configuration metadata are in [results](results). Timing-derived FPS is the
reciprocal of processing latency for reused images; it is not a measured camera frame rate. Repetition counts
represent repeated measurements of a fixed scene, not independent real-world accuracy examples.
