# Changelog

## Unreleased — 0.3.0

- Renamed the project to Napa, the Python package to `napa_vision`, and the command prefix to `napa-`.
- Renamed the camera service class to `NapaVisionService`; explicit device IDs and the Northstar observation packet remain compatible.
- Added a reproducible detector/occlusion matrix with RobotPy AprilTag 3 baselines, raw timings, and explicit ID-only accuracy scoring.
- Added latest-frame camera capture, bounded reconnects, negotiated-camera telemetry, and deterministic shutdown.
- Added bounded asynchronous JSONL recording with rotation, retention, drop policies, periodic durable syncs, and health counters.
- Added versioned calibration artifacts, checkerboard calibration, strict WPILib field-layout import, and production provenance gates.
- Added robust multi-tag pose solving with RANSAC, iterative outlier rejection, per-tag residuals, physical bounds, and rejection diagnostics.
- Added timestamped NetworkTables robot-yaw interpolation and stale-yaw suppression for trig localization.
- Added an integrity-checked replay benchmark with accuracy and latency percentiles.
- Added deterministic camera-free AprilTag rendering, pose ground truth, replay artifacts, gates, and optimization comparisons.
- Removed a duplicate two-tag PnP solve, cached field-corner geometry, and reduced skip-scan statistics overhead.
- Added real local NetworkTables integration coverage and command/package smoke tests.
- Removed the obsolete synchronous JSONL writer from the runtime surface.

Publication remains blocked until the repository owner selects a license and supplies release metadata.
