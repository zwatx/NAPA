from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from .prefilter import BlockSkipPrefilter, Roi
from .types import FiducialImageObservation


class OpenCVArucoFiducialDetector:
    def __init__(
        self,
        dictionary_id: int | None = None,
        *,
        prefilter: BlockSkipPrefilter | None = None,
        parameters: cv2.aruco.DetectorParameters | None = None,
        dictionary: cv2.aruco.Dictionary | None = None,
        tag_id_map: Sequence[int] | None = None,
    ) -> None:
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
            raise RuntimeError("OpenCV >= 4.7 with ArucoDetector support is required")
        if dictionary_id is not None and dictionary is not None:
            raise ValueError("dictionary_id and dictionary are mutually exclusive")
        dictionary_id = dictionary_id if dictionary_id is not None else cv2.aruco.DICT_APRILTAG_36h11
        self.dictionary = dictionary if dictionary is not None else cv2.aruco.getPredefinedDictionary(dictionary_id)
        if tag_id_map is not None and len(tag_id_map) != len(self.dictionary.bytesList):
            raise ValueError("tag_id_map must contain one original ID per dictionary row")
        self.tag_id_map = None if tag_id_map is None else tuple(tag_id_map)
        self.prefilter = prefilter
        self.params = parameters if parameters is not None else cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise ValueError("AprilTag detection requires an 8-bit image")
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] in {3, 4}:
            conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(image, conversion)
        else:
            raise ValueError("AprilTag detection requires a grayscale, BGR, or BGRA image")
        gray = np.ascontiguousarray(gray)
        if self.prefilter is not None:
            rois = self.prefilter.find_rois(gray)
            if rois == []:
                return []
            if rois is not None:
                return self._detect_rois(gray, rois)
            return self._detect_gray(gray)
        minimum, maximum, _min_location, _max_location = cv2.minMaxLoc(gray)
        if maximum - minimum < 2.0:
            return []
        return self._detect_gray(gray)

    def _detect_gray(self, gray: np.ndarray) -> list[FiducialImageObservation]:
        corners, ids, _rejected = self.detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return []
        observations: list[FiducialImageObservation] = []
        for row_id, tag_corners in zip(ids.reshape(-1), corners, strict=True):
            tag_id = int(row_id) if self.tag_id_map is None else self.tag_id_map[int(row_id)]
            observations.append(FiducialImageObservation(tag_id, tag_corners.reshape(4, 2).astype(np.float64)))
        return observations

    def _detect_rois(self, gray: np.ndarray, rois: list[Roi]) -> list[FiducialImageObservation]:
        observations: list[FiducialImageObservation] = []
        centers_by_id: dict[int, list[np.ndarray]] = {}
        for roi in rois:
            crop = gray[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            if crop.size == 0:
                continue
            for observation in self._detect_gray(crop):
                shifted = observation.corners.copy()
                shifted[:, 0] += roi.x
                shifted[:, 1] += roi.y
                candidate = FiducialImageObservation(observation.tag_id, shifted)
                center = np.mean(candidate.corners, axis=0)
                prior_centers = centers_by_id.setdefault(candidate.tag_id, [])
                if not _has_duplicate_center(center, prior_centers):
                    observations.append(candidate)
                    prior_centers.append(center)
        return observations


def make_apriltag_detector(
    candidate: str = "baseline",
    *,
    layout_tag_ids: Iterable[int] | None = None,
) -> OpenCVArucoFiducialDetector:
    """Build an opt-in detector candidate without changing runtime defaults."""

    if candidate == "baseline":
        return OpenCVArucoFiducialDetector()
    if candidate == "aruco3":
        parameters = cv2.aruco.DetectorParameters()
        if not hasattr(parameters, "useAruco3Detection"):
            raise RuntimeError("This OpenCV build does not support ArUco3 detection")
        parameters.useAruco3Detection = True
        parameters.minMarkerLengthRatioOriginalImg = 0.0
        return OpenCVArucoFiducialDetector(parameters=parameters)
    if candidate == "layout-restricted":
        original_ids = _validated_layout_tag_ids(layout_tag_ids)
        source = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        invalid_ids = [tag_id for tag_id in original_ids if tag_id >= len(source.bytesList)]
        if invalid_ids:
            raise ValueError(f"layout tag IDs are outside AprilTag 36h11: {invalid_ids}")
        dictionary = cv2.aruco.Dictionary(
            source.bytesList[np.asarray(original_ids, dtype=np.intp)].copy(),
            source.markerSize,
            source.maxCorrectionBits,
        )
        return OpenCVArucoFiducialDetector(dictionary=dictionary, tag_id_map=original_ids)
    raise ValueError(f"Unknown detector candidate {candidate!r}")


def _validated_layout_tag_ids(layout_tag_ids: Iterable[int] | None) -> tuple[int, ...]:
    if layout_tag_ids is None:
        raise ValueError("layout_tag_ids are required for the layout-restricted detector")
    original_ids = tuple(layout_tag_ids)
    if not original_ids:
        raise ValueError("layout_tag_ids must not be empty")
    if any(isinstance(tag_id, bool) or not isinstance(tag_id, int) or tag_id < 0 for tag_id in original_ids):
        raise ValueError("layout_tag_ids must contain non-negative integers")
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("layout_tag_ids must not contain duplicates")
    return tuple(sorted(original_ids))


def _has_duplicate_center(
    candidate_center: np.ndarray,
    observation_centers: list[np.ndarray],
    *,
    center_tolerance_px: float = 8.0,
) -> bool:
    for center in observation_centers:
        if float(np.linalg.norm(candidate_center - center)) <= center_tolerance_px:
            return True
    return False
