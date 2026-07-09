from __future__ import annotations

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
    ) -> None:
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
            raise RuntimeError("OpenCV >= 4.7 with ArucoDetector support is required")
        dictionary_id = dictionary_id if dictionary_id is not None else cv2.aruco.DICT_APRILTAG_36h11
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
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
        for tag_id, tag_corners in zip(ids.flatten().tolist(), corners, strict=True):
            observations.append(FiducialImageObservation(int(tag_id), tag_corners.reshape(4, 2).astype(np.float64)))
        return observations

    def _detect_rois(self, gray: np.ndarray, rois: list[Roi]) -> list[FiducialImageObservation]:
        observations: list[FiducialImageObservation] = []
        for roi in rois:
            crop = gray[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            if crop.size == 0:
                continue
            for observation in self._detect_gray(crop):
                shifted = observation.corners.copy()
                shifted[:, 0] += roi.x
                shifted[:, 1] += roi.y
                candidate = FiducialImageObservation(observation.tag_id, shifted)
                if not _has_duplicate(candidate, observations):
                    observations.append(candidate)
        return observations


def _has_duplicate(
    candidate: FiducialImageObservation,
    observations: list[FiducialImageObservation],
    *,
    center_tolerance_px: float = 8.0,
) -> bool:
    candidate_center = np.mean(candidate.corners, axis=0)
    for observation in observations:
        if observation.tag_id != candidate.tag_id:
            continue
        center = np.mean(observation.corners, axis=0)
        if float(np.linalg.norm(candidate_center - center)) <= center_tolerance_px:
            return True
    return False
