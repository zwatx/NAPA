# Archived Occlusion Detector Experiment

This folder preserves the earlier custom `tag36h11` occlusion-tolerant detector.
It is no longer the runtime architecture for the main project.

The current direction is Northstar-style:

- OpenCV AprilTag detection is the detector of record.
- The project focuses on camera service structure, calibration, pose solvers,
  tag-angle outputs, trig-distance pose estimates, and telemetry.
- Any future corner repair should be lightweight and candidate-gated by OpenCV.
