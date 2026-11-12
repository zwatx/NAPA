"""Synchronous image processing without camera, transport, or device ownership."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from numbers import Real
from time import perf_counter

import numpy as np

from . import _import_cv2
from .calibration import CameraCalibration
from .covariance import DEFAULT_CORNER_NOISE_PX
from .geometry import _rigid_transform, tag_field_corners, tag_object_points, wpilib_points_to_opencv
from .napa_detector import NapaDetector
from .solvers import CameraPoseSolver, SingleTagPoseSolver, TagAngleSolver
from .types import CameraPoseBounds, FiducialDetector, FiducialImageObservation, FrameResult, RollingShutter


@dataclass(frozen=True, slots=True)
class _CameraGeometry:
    calibration_matrix: np.ndarray
    distortion_coefficients: np.ndarray


def _unsheared(corners: np.ndarray, velocity: np.ndarray, axis: int, lines: int, signed_readout: float) -> np.ndarray:
    """Corners as they were when the middle line was read; ``signed_readout`` is negative for a backward read."""

    read_after_middle = (corners[:, axis] - (lines - 1) / 2.0) / lines * signed_readout
    return corners - velocity * read_after_middle[:, None]


class _ShutterCorrector:
    """Moves each tag's corners to where they were when the image's middle line was read.

    Each corner's own motion between frames gives its velocity in the image, whatever the camera is doing. Away from
    the image's centre a turn moves a tag's corners at different speeds, so one speed per tag would leave error.
    Every tag is taken to one line, not its own, so that all the tags of a frame describe a single instant.
    """

    def __init__(self, shutter: RollingShutter, per_corner: bool = True) -> None:
        self._shutter = shutter
        self._per_corner = per_corner
        self._axis = 1 if shutter.direction in ("top_to_bottom", "bottom_to_top") else 0
        self._sign = 1.0 if shutter.direction in ("top_to_bottom", "left_to_right") else -1.0
        self._sightings: dict[tuple[str, int], list[tuple[float, np.ndarray]]] = {}
        self._last_timestamp: float | None = None

    def _velocity(self, key: tuple[str, int], timestamp: float, center: np.ndarray) -> np.ndarray | None:
        """Pixels per second now: from the last step, extrapolated with the step before when there is one."""

        seen = self._sightings.get(key, [])
        if not seen:
            return None
        last_time, last_center = seen[-1]
        step = timestamp - last_time
        if not 0.0 < step <= self._shutter.max_frame_gap_s:
            return None
        velocity = (center - last_center) / step
        if len(seen) == 2:
            first_time, first_center = seen[0]
            earlier_step = last_time - first_time
            if 0.0 < earlier_step <= self._shutter.max_frame_gap_s:
                earlier = (last_center - first_center) / earlier_step
                velocity = velocity + (velocity - earlier) * step / (step + earlier_step)
        return velocity

    def velocities(self, observations: list[FiducialImageObservation], timestamp: float) -> list[np.ndarray | None]:
        """Each tag's speed in pixels per second, or ``None`` without usable history, and remember this frame.

        With ``per_corner`` (the default), each corner's own speed (4 x 2); otherwise the tag's mean, which fitting
        several tags to one pose tolerates.
        """

        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            self._sightings.clear()  # the clock went backwards or repeated: nothing earlier can be trusted
        self._last_timestamp = timestamp
        keys = [(observation.family, observation.tag_id) for observation in observations]
        velocities: list[np.ndarray | None] = []
        centers = {}
        for key, observation in zip(keys, observations, strict=True):
            corners = np.asarray(observation.corners, dtype=np.float64)
            center = corners if self._per_corner else corners.mean(axis=0)
            unique = keys.count(key) == 1
            velocities.append(self._velocity(key, timestamp, center) if unique else None)
            if unique:
                centers[key] = center
        sightings = {
            key: seen
            for key, seen in self._sightings.items()
            if timestamp - seen[-1][0] <= self._shutter.max_frame_gap_s and key not in centers
        }
        for key, center in centers.items():
            sightings[key] = [*self._sightings.get(key, [])[-1:], (timestamp, center)]
        self._sightings = sightings
        return velocities

    def correct(
        self, observations: list[FiducialImageObservation], timestamp: float, height: int, width: int
    ) -> tuple[list[FiducialImageObservation], tuple[int, ...]]:
        """Return the observations with corrected corners, and the IDs of those with no usable velocity."""

        lines = height if self._axis == 1 else width
        corrected, uncorrected = [], []
        velocities = self.velocities(observations, timestamp)
        for observation, velocity in zip(observations, velocities, strict=True):
            if velocity is None:
                corrected.append(observation)
                uncorrected.append(observation.tag_id)
                continue
            corners = _unsheared(
                np.asarray(observation.corners, dtype=np.float64),
                velocity,
                self._axis,
                lines,
                self._sign * self._shutter.readout_s,
            )
            corrected.append(replace(observation, corners=corners))
        return corrected, tuple(uncorrected)


@dataclass(frozen=True, slots=True)
class RollingShutterEstimate:
    """A camera's rolling shutter read from a recording.

    ``rolling_shutter`` is the value to pass to ``VisionProcessor``, or ``None``. ``readout_s`` is ``None`` when the
    recording can't tell, and 0.0 when it reads as a global shutter along every axis it could check; in both cases
    ``rolling_shutter`` is ``None``. ``low`` and ``high`` bound the middle 80% of estimates from sightings resampled in
    half-second blocks: scatter only, not any bias from calibration or a layout. ``sightings`` is the number used: tags,
    or frames when a tag layout was given. ``axes_checked`` holds ``"rows"`` and/or ``"columns"``; a read axis needs
    motion across it to show.
    """

    rolling_shutter: RollingShutter | None
    readout_s: float | None
    low: float | None
    high: float | None
    sightings: int
    axes_checked: tuple[str, ...]


_SHUTTER_LIMIT_S = 0.05
_SHUTTER_GRID = np.linspace(-_SHUTTER_LIMIT_S, _SHUTTER_LIMIT_S, 41)
_SHUTTER_MIN_RATE = 0.3  # radians per second across the read axis, below which a tag's shear says little
_SHUTTER_MIN_SHEAR_PX = 0.3  # shear across a tag at a 15 ms readout: less is lost in the fit's residual on real footage
_SHUTTER_MIN_TAG_PX = 60.0  # mean side: smaller tags read 10 to 20% off with a narrow range in simulation
_SHUTTER_GLOBAL_S = 0.0025  # a shorter readout can't be told from a global shutter, and hardly matters
_SHUTTER_MAX_USED = 400
_SHUTTER_BLOCK_S = 0.5  # resampled together: frames this close share one motion, so they aren't independent
_SHUTTER_RESIDUAL_CAP = 2.0  # pixels: one bad corner mustn't choose the answer
_SHUTTER_DIRECTIONS_BY_AXIS = {1: ("top_to_bottom", "bottom_to_top"), 0: ("left_to_right", "right_to_left")}
_LAYOUT_MIN_TAGS = 3
_LAYOUT_MIN_SPREAD = 0.25  # of the image, both ways: tags close together are read at nearly the same moment
_LAYOUT_RANGE_SHARE = 0.3  # of the answer: a wider range than this usually means a survey error, not scatter
_LAYOUT_CHANGE_CAP = 5.0  # pixels of misfit change over the search: one bad frame mustn't choose the answer


def _curve_minimum(curve: np.ndarray) -> tuple[float, float]:
    """The signed readout at a curve's minimum, by a parabola through the lowest grid point, and the drop from zero."""

    best = int(np.argmin(curve))
    position = float(_SHUTTER_GRID[best])
    if 0 < best < len(curve) - 1:
        bend = curve[best - 1] - 2 * curve[best] + curve[best + 1]
        if bend > 0:
            position += 0.5 * (curve[best - 1] - curve[best + 1]) / bend * (_SHUTTER_GRID[1] - _SHUTTER_GRID[0])
    return float(position), float(curve[len(curve) // 2] - curve[best])


def estimate_rolling_shutter(
    frames: Iterable[tuple[float, np.ndarray]],
    *,
    calibration: CameraCalibration,
    tag_size_m: float,
    tag_layout: dict[int, np.ndarray] | None = None,
    detector: FiducialDetector | None = None,
    min_sightings: int = 30,
    max_frame_gap_s: float = 0.1,
) -> RollingShutterEstimate:
    """Estimate a camera's readout time and direction from ``(timestamp, image)`` frames of a moving camera.

    A sheared square isn't a valid view of a square, so its planar pose fit leaves a residual. The result is the readout
    that removes the most residual over the recording, searched along the image's rows and columns. It needs
    consecutive frames of tags while the camera moves across the read axis, and at least ``min_sightings`` such tags.
    Tags under 60 px across don't count: on their own they read wrong while looking certain.

    With ``tag_layout`` (tag ID to field-to-tag transform, as for ``VisionProcessor``), each frame's tags must instead
    fit one camera pose together, which works with tags too small to show shear on their own. A frame counts as a
    sighting when three or more layout tags with a recent earlier sighting span a quarter of the image both ways.
    It relies on the calibration, and on the layout's accuracy; see the pose guide.
    """

    cv2 = _import_cv2()
    if isinstance(min_sightings, bool) or not isinstance(min_sightings, int) or min_sightings < 1:
        raise ValueError("min_sightings must be a positive integer")
    if calibration.image_size is None:
        raise ValueError("calibration must record its image resolution")
    layout: dict[int, np.ndarray] | None = None
    if tag_layout is not None:
        layout = {}
        for tag_id, transform in tag_layout.items():
            if isinstance(tag_id, bool) or not isinstance(tag_id, int) or tag_id < 0:
                raise ValueError("tag_layout IDs must be non-negative integers")
            field_to_tag = _rigid_transform(transform, f"tag_layout[{tag_id}]")
            layout[tag_id] = wpilib_points_to_opencv(tag_field_corners(field_to_tag, tag_size_m))
    detector = detector if detector is not None else NapaDetector()
    tracker = _ShutterCorrector(RollingShutter(1.0, max_frame_gap_s=max_frame_gap_s), per_corner=layout is None)
    matrix, distortion = calibration.camera_matrix, calibration.distortion_coefficients
    points = tag_object_points(tag_size_m)
    focal = float(matrix[0, 0])
    width, height = calibration.image_size
    moving: dict[int, list[tuple]] = {0: [], 1: []}
    for timestamp, image in frames:
        observations = detector.detect(image)
        velocities = tracker.velocities(observations, timestamp)
        if layout is not None:
            pairs = zip(observations, velocities, strict=True)
            seen = [(o, v) for o, v in pairs if v is not None and o.tag_id in layout]
            if len(seen) < _LAYOUT_MIN_TAGS:
                continue
            tags = [np.asarray(o.corners, dtype=np.float64) for o, _ in seen]
            speeds = np.array([v for _, v in seen])
            world = np.concatenate([layout[o.tag_id] for o, _ in seen])
            stacked = np.concatenate(tags)
            # Tags along one line only, either way, let the pose take up most of the shear by rolling.
            spread = np.ptp(stacked, axis=0) / np.array([width, height])
            if spread.min() < _LAYOUT_MIN_SPREAD:
                continue
            for axis, across in ((1, 0), (0, 1)):
                lines = height if axis == 1 else width
                rate = float(np.mean(np.abs(speeds[:, across])))
                if rate >= _SHUTTER_MIN_RATE * focal and rate * spread[axis] * 0.015 >= _SHUTTER_MIN_SHEAR_PX:
                    moving[axis].append((timestamp, tags, speeds, world))
            continue
        for observation, velocity in zip(observations, velocities, strict=True):
            if velocity is None:
                continue
            corners = np.asarray(observation.corners, dtype=np.float64)
            if np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).mean() < _SHUTTER_MIN_TAG_PX:
                continue
            speed = velocity.mean(axis=0)
            # Motion along a read axis only stretches the tag, which a tilt can imitate; motion across it shears.
            for axis, across in ((1, 0), (0, 1)):
                lines = height if axis == 1 else width
                shear = abs(speed[across]) * float(np.ptp(corners[:, axis])) / lines * 0.015
                if abs(speed[across]) >= _SHUTTER_MIN_RATE * focal and shear >= _SHUTTER_MIN_SHEAR_PX:
                    moving[axis].append((timestamp, corners, velocity))

    def residual(corners: np.ndarray) -> float:
        try:
            ok, _, _, errors = cv2.solvePnPGeneric(points, corners, matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return _SHUTTER_RESIDUAL_CAP**2
        return min(float(np.min(errors)), _SHUTTER_RESIDUAL_CAP) ** 2 if ok else _SHUTTER_RESIDUAL_CAP**2

    def layout_curve(tags: list[np.ndarray], speeds: np.ndarray, world: np.ndarray, axis: int, lines: int) -> list:
        """One camera pose for all of a frame's tags at each readout: its RMS misfit, above the frame's least.

        A layout's own error leaves every frame some misfit; only the change with the readout is about the shutter.
        """

        worst = [0.0] * len(_SHUTTER_GRID)
        image = np.concatenate(tags)
        try:
            ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(world, image, matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return worst
        if not ok or len(rvecs) == 0:
            return worst
        best = int(np.argmin(np.asarray(errors).ravel()))
        start = (rvecs[best].copy(), tvecs[best].copy())
        curve = []
        for t in _SHUTTER_GRID:
            image = np.concatenate([_unsheared(c, v, axis, lines, float(t)) for c, v in zip(tags, speeds, strict=True)])
            guess = (start[0].copy(), start[1].copy())
            try:
                ok, rvec, tvec = cv2.solvePnP(world, image, matrix, distortion, *guess, True, cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                ok = False
            if not ok:
                return worst
            projected = cv2.projectPoints(world, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
            curve.append(float(np.sqrt(np.mean(np.sum((projected - image) ** 2, axis=1)))))
        return list(np.minimum(np.array(curve) - min(curve), _LAYOUT_CHANGE_CAP))

    curves: dict[int, np.ndarray] = {}
    blocks: dict[int, list[np.ndarray]] = {}
    for axis, seen in moving.items():
        if len(seen) >= min_sightings:
            used = seen[:: max(1, len(seen) // _SHUTTER_MAX_USED)][:_SHUTTER_MAX_USED]
            lines = height if axis == 1 else width
            block = np.array([int(sighting[0] // _SHUTTER_BLOCK_S) for sighting in used])
            blocks[axis] = [np.flatnonzero(block == b) for b in np.unique(block)]
            if layout is not None:
                curves[axis] = np.array([layout_curve(*sighting[1:], axis, lines) for sighting in used])
                continue
            curves[axis] = np.array(
                [[residual(_unsheared(c, v, axis, lines, float(t))) for t in _SHUTTER_GRID] for _, c, v in used]
            )
    checked = tuple("rows" if axis == 1 else "columns" for axis in curves)
    count = max((len(c) for c in curves.values()), default=max(len(moving[0]), len(moving[1])))
    if not curves:
        return RollingShutterEstimate(None, None, None, None, count, ())

    def pick(means: dict[int, np.ndarray]) -> tuple[int, float]:
        found = {axis: _curve_minimum(curve) for axis, curve in means.items()}
        axis = max(found, key=lambda a: found[a][1])
        return axis, found[axis][0]

    axis, signed = pick({a: c.mean(axis=0) for a, c in curves.items()})
    rng = np.random.default_rng(0)

    def resampled(a: int) -> np.ndarray:
        chosen = rng.integers(0, len(blocks[a]), len(blocks[a]))
        return curves[a][np.concatenate([blocks[a][i] for i in chosen])].mean(axis=0)

    draws = [pick({a: resampled(a) for a in curves}) for _ in range(200)]
    sign = 1.0 if signed >= 0 else -1.0
    same = [value for a, value in draws if a == axis and value * sign > 0]
    low, high = (float(v) for v in np.percentile([abs(v) for v in same], [10, 90])) if same else (None, None)
    readout = float(abs(signed))
    count = len(curves[axis])
    if readout >= _SHUTTER_LIMIT_S - 1e-9:  # the search ran off its end: there's no minimum to report
        return RollingShutterEstimate(None, None, low, high, count, checked)
    if readout < _SHUTTER_GLOBAL_S:
        # Only when the resamples agree: a wide range around zero is a recording that can't tell.
        if low is None or high > 2 * _SHUTTER_GLOBAL_S:
            return RollingShutterEstimate(None, None, low, high, count, checked)
        return RollingShutterEstimate(None, 0.0, low, high, count, checked)
    # Disagreement between resamples, or a range wider than the answer's tolerance, means the recording can't say.
    # A layout's own errors differ between the frames that see different tags, so there the range must be narrower.
    tolerance = _LAYOUT_RANGE_SHARE if layout is not None else 0.5
    if len(same) < 0.9 * len(draws) or low is None or high - low > tolerance * readout:
        return RollingShutterEstimate(None, None, low, high, count, checked)
    direction = _SHUTTER_DIRECTIONS_BY_AXIS[axis][0 if signed > 0 else 1]
    shutter = RollingShutter(readout, direction=direction, max_frame_gap_s=max_frame_gap_s)
    return RollingShutterEstimate(shutter, readout, low, high, count, checked)


class VisionProcessor:
    """Process caller-owned images; use one instance per camera stream.

    Without calibration, return IDs and corners only. Calibrated processing also
    requires the physical tag size. A tag layout (tag ID to world-to-tag transform)
    is optional and enables a camera pose; poses use the conventions documented in
    ``geometry``, never a device SDK's world frame. Poses and tag angles carry
    covariances for ``corner_noise_px`` of independent per-axis corner noise.
    Calls are synchronous and must not overlap on one instance. The caller owns
    capture, pixel-format conversion, clocks, scheduling, and result delivery.

    ``rolling_shutter`` (a ``RollingShutter``, unset by default) corrects the corners that feed the pose solvers for
    a camera that reads its lines one after another, using each tag's motion between frames and the ``timestamp`` of
    each call. ``image_observations`` stay as detected. See the pose guide.
    """

    def __init__(
        self,
        *,
        detector: FiducialDetector | None = None,
        calibration: CameraCalibration | None = None,
        tag_size_m: float | None = None,
        tag_layout: dict[int, np.ndarray] | None = None,
        max_reprojection_error_px: float = 5.0,
        max_tag_reprojection_error_px: float | None = None,
        camera_pose_bounds: CameraPoseBounds | None = None,
        max_outlier_iterations: int = 8,
        ransac_iterations: int = 100,
        corner_noise_px: float = DEFAULT_CORNER_NOISE_PX,
        rolling_shutter: RollingShutter | None = None,
    ) -> None:
        if (calibration is None) != (tag_size_m is None):
            raise ValueError("calibration and tag_size_m must be provided together")
        if tag_size_m is not None and (
            isinstance(tag_size_m, bool)
            or not isinstance(tag_size_m, Real)
            or not math.isfinite(tag_size_m)
            or tag_size_m <= 0.0
        ):
            raise ValueError("tag_size_m must be finite and greater than zero")
        if calibration is None and tag_layout is not None:
            raise ValueError("tag_layout requires calibration")
        if calibration is not None and calibration.image_size is None:
            raise ValueError("calibration must record its image resolution")
        if rolling_shutter is not None and not isinstance(rolling_shutter, RollingShutter):
            raise ValueError("rolling_shutter must be a RollingShutter or None")
        if rolling_shutter is not None and calibration is None:
            raise ValueError("rolling_shutter requires calibration")
        self._shutter = None if rolling_shutter is None else _ShutterCorrector(rolling_shutter)
        self.calibration = calibration
        self.tag_size_m = tag_size_m
        self.tag_layout: dict[int, np.ndarray] = {}
        for tag_id, transform in ({} if tag_layout is None else tag_layout).items():
            if isinstance(tag_id, bool) or not isinstance(tag_id, int) or tag_id < 0:
                raise ValueError("tag_layout IDs must be non-negative integers")
            self.tag_layout[tag_id] = _rigid_transform(transform, f"tag_layout[{tag_id}]")
        self._camera = (
            None
            if calibration is None
            else _CameraGeometry(calibration.camera_matrix, calibration.distortion_coefficients)
        )
        self._single_solver = SingleTagPoseSolver(max_reprojection_error_px, corner_noise_px=corner_noise_px)
        self._camera_solver = CameraPoseSolver(
            max_reprojection_error_px,
            max_tag_reprojection_error_px=max_tag_reprojection_error_px,
            bounds=camera_pose_bounds,
            max_outlier_iterations=max_outlier_iterations,
            ransac_iterations=ransac_iterations,
            corner_noise_px=corner_noise_px,
        )
        self._angle_solver = TagAngleSolver(corner_noise_px=corner_noise_px)
        self.detector = detector if detector is not None else NapaDetector()

    def process(
        self,
        image: np.ndarray,
        *,
        timestamp: float,
        capture_monotonic_s: float | None = None,
    ) -> FrameResult:
        """Return a fresh result without retaining the image or publishing it.

        Images must be nonempty uint8 grayscale, BGR, or BGRA arrays. Strided
        arrays are accepted. Timestamp units are seconds; capture_monotonic_s
        belongs to the caller's clock. Robot/world augmentation and device
        coordinate conversion belong to the caller.
        """

        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise ValueError("AprilTag detection requires an 8-bit image")
        if image.ndim not in {2, 3} or (image.ndim == 3 and image.shape[2] not in {3, 4}):
            raise ValueError("AprilTag detection requires a grayscale, BGR, or BGRA image")
        if image.size == 0:
            raise ValueError("AprilTag detection requires a nonempty image")
        if timestamp is None:
            raise ValueError("timestamp must be finite")
        for value, name in (
            (timestamp, "timestamp"),
            (capture_monotonic_s, "capture_monotonic_s"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.calibration is not None:
            self.calibration.validate_for_image_size(image.shape[1], image.shape[0])
        detection_start = perf_counter()
        observations = self.detector.detect(image)
        detection_end = perf_counter()
        result = FrameResult(
            timestamp=timestamp,
            capture_monotonic_s=capture_monotonic_s,
            image_observations=observations,
        )
        if self._camera is not None and self.tag_size_m is not None:
            solving = observations
            if self._shutter is not None:
                solving, result.uncorrected_tag_ids = self._shutter.correct(
                    observations, timestamp, image.shape[0], image.shape[1]
                )
            solved_observations = []
            singles = self._single_solver.solve_many(solving, self._camera, self.tag_size_m)
            for observation, single in zip(solving, singles, strict=True):
                if single is not None:
                    result.single_tag_poses.append(single)
                    solved_observations.append(observation)
            result.tag_angles = self._angle_solver.solve_many(
                solved_observations, self._camera, result.single_tag_poses
            )
            joining, joined_singles = solving, result.single_tag_poses
            if result.uncorrected_tag_ids and len(result.uncorrected_tag_ids) < len(solving):
                # An uncorrected tag describes a different instant from the corrected ones, so it would bend their pose.
                skip = set(result.uncorrected_tag_ids)
                joining = [observation for observation in solving if observation.tag_id not in skip]
                joined_singles = [single for single in result.single_tag_poses if single.tag_id not in skip]
            result.camera_pose = self._camera_solver.solve(
                joining, joined_singles, self._camera, self.tag_size_m, self.tag_layout
            )
            result.pose_diagnostics = self._camera_solver.last_diagnostics
        solve_end = perf_counter()
        result.health = {
            "detection_ms": (detection_end - detection_start) * 1000.0,
            "solve_ms": (solve_end - detection_end) * 1000.0,
        }
        return result
