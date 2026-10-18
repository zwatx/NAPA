# AprilTag occlusion evidence and benchmark guardrails

This note separates what the upstream decoders guarantee from this repository's
experimental detector configuration. It does not report timing or an occlusion
success rate.

## What ordinary AprilTag error correction covers

`tag36h11` is a 587-code, 36-payload-bit family whose upstream metadata sets
`h = 11`, where `h` is explicitly the *minimum Hamming distance between any two
codes*. [The family source](https://github.com/AprilRobotics/apriltag/blob/master/tag36h11.c)
sets `h = 11`, `ncodes = 587`, and `nbits = 36`; [the family definition](https://github.com/AprilRobotics/apriltag/blob/master/apriltag.h)
defines `h` as the minimum distance. In coding-theory terms, this gives a
unique-nearest-codeword radius of `floor((11 - 1) / 2) = 5` for **payload-bit
substitutions**. It is a property of the code family, not a promise that a
camera detector will localize or decode a tag with five bad samples.

The ordinary AprilRobotics detector samples a binary codeword and accepts it
only when its Hamming distance to a family code is at most the configured
`bits_corrected`; the [decode loop](https://github.com/AprilRobotics/apriltag/blob/master/apriltag.c)
compares `popcount(correct_code ^ rcode)` to `maxhamming`. The convenience
`apriltag_detector_add_family()` configures **2** corrections, and its public
header says values at least 3 have prohibitive memory cost; the current quick
decoder rejects values above 3. The same header documents the delivered
`hamming` count as corrected error bits and warns that large correction counts
increase false positives. Therefore the supported wording is: *standard
AprilTag can correct a bounded number of observed payload-bit errors when a
quad is found; the normal upstream configuration is two bits, not a general
occlusion percentage.*

There is no erasure/unknown-bit input in that normal decoder path: it accepts a
complete sampled codeword plus an integer correction radius. An occluder that
makes a payload cell read as the wrong colour can sometimes be handled as a
payload error, but an occluder is not inherently a correctable error and has no
family-level percentage guarantee.

## Why border and corner occlusion are different

Payload correction runs only after localization. Upstream AprilTag first fits
four lines to an edge cluster and rejects candidates that cannot form a valid
quad, including clusters with fewer than four maxima or excessive line-fit
error ([quad-threshold source](https://github.com/AprilRobotics/apriltag/blob/master/apriltag_quad_thresh.c)).
The black border supplies that square/edge evidence, so border or corner
occlusion can stop detection before payload decoding and Hamming correction
occur. A corner normally constrains two adjacent edges; losing it changes the
geometry problem, whereas an interior payload cell mainly changes a decoded
bit after a usable homography exists. This explains the different failure mode;
it does **not** establish a universal border- or corner-occlusion threshold.

OpenCV makes the same staging visible: it checks the marker border before
passing the inner cells to `Dictionary::identify()`
([implementation](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/aruco_detector.cpp)).
OpenCV's `maxErroneousBitsInBorderRate` is a separate border gate and is not
payload error correction.

## OpenCV AprilTag36h11 semantics

In current OpenCV 4.x source, predefined `DICT_APRILTAG_36h11` is constructed
with `maxCorrectionBits = (11 - 1) / 2`, i.e. **5**
([dictionary source](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/aruco_dictionary.cpp)).
`Dictionary::identify()` computes `int(maxCorrectionBits * maxCorrectionRate)`
and accepts the nearest dictionary entry only within that integer limit. The
detector passes `DetectorParameters.errorCorrectionRate` as that rate
([identification call](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/aruco_detector.cpp));
its documented default is **0.6**
([parameter declaration](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/include/opencv2/objdetect/aruco_detector.hpp)).
Thus the current default payload limit for this dictionary is `int(5 * 0.6) =
3` bits. This is a rate that scales a dictionary's stored correction cap, not a
fraction of tag area or a fraction of occlusion. OpenCV's official tutorial
describes the same multiplication and says reducing the rate helps avoid false
positives ([tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)).

Record both the OpenCV version and the effective integer limit in benchmark
metadata, because either the dictionary definition or parameters can change
the result.

## This repository's experimental defaults

`Robust36h11Detector` defaults to `max_hamming=2`,
`max_occlusion_ratio=0.25`, and **`max_erased_bits=0`**
([detector constructor](../../experiments/occlusion_detector/robust_apriltag/detector.py)).
The solve path rejects any unreliable payload bit when the default zero-erasure
limit is active, then computes `occlusion_score` from erased border samples,
erased payload samples, and corrected payload bits before comparing it to 0.25
([solve path](../../experiments/occlusion_detector/robust_apriltag/detector.py)).
The local masked decoder does support a mask when a non-default configuration
permits erasures: it compares only `(observed_code ^ family_code) &
observed_mask` and requires a minimum reliable-bit count
([masked decode](../../experiments/occlusion_detector/robust_apriltag/family.py)).

Consequently, the default `max_occlusion_ratio=.25` is an **acceptance score
threshold**, not evidence that the default detector "allows 25% occlusion."
At its defaults it permits no *classified-unreliable payload bits*, and the
score is not geometric pixel/area occlusion. It may still accept some covered
payload cells if they remain confidently classified and fit the two-bit Hamming
budget; that is an outcome to measure, not a default guarantee.

`detect(..., candidate_corners=...)` labels input candidates as
`"seeded-corners"`, bypassing candidate generation, and then permits a
seeded-corner repair step. Any experiment using that argument measures
decode/repair conditional on a supplied localization hypothesis, not end-to-end
tag detection. State this prominently for corner-occlusion experiments.

## Fair RobotPy AprilTag3 timing baseline

Use the official binding's persistent detector API:

```python
from robotpy_apriltag import AprilTagDetector

detector = AprilTagDetector()
detector.addFamily("tag36h11", bitsCorrected=2)
config = detector.getConfig()
config.numThreads = 1
config.quadDecimate = 1.0  # accuracy-oriented baseline
detector.setConfig(config)
detections = detector.detect(gray_u8)
```

RobotPy documents that the detector is expensive to create/tear down, `detect`
takes an 8-bit grayscale image, and `add_family(..., bits_corrected=2)` sets the
maximum corrected bits ([API](https://robotpy.readthedocs.io/projects/robotpy/en/latest/robotpy_apriltag/AprilTagDetector.html)).
The older, still official RobotPy AprilTag API documents `numThreads` (default
one) and `quadDecimate` (default 2.0, with a speed/pose-accuracy/detection-rate tradeoff
while payload decoding stays full resolution)
([configuration API](https://robotpy.readthedocs.io/projects/apriltag/en/latest/robotpy_apriltag/AprilTagDetector.html)).
The pinned benchmark dependency `robotpy-apriltag==2026.2.2` was inspected locally
and exposes the camelCase API shown above (`addFamily`, `getConfig`, `setConfig`,
`numThreads`, `quadDecimate`). The moving latest documentation also describes a
snake_case API; use the installed version's spelling and record the pin.

For a fair timing report, keep the constructed detector outside the timed loop,
use one grayscale frame buffer per equivalent case, pin `numThreads=1`, state
`bitsCorrected=2`, and report `quadDecimate` exactly. `quadDecimate=1.0` is the
like-for-like full-resolution quad baseline; `2.0` is a separately reported
throughput-oriented configuration, not a speedup claim without measurements.
