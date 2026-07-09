from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        np.asarray(data["camera_matrix"], dtype=np.float64),
        np.asarray(data.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64),
    )


def save_calibration(path: str | Path, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> None:
    payload = {
        "camera_matrix": np.asarray(camera_matrix, dtype=float).tolist(),
        "distortion_coefficients": np.asarray(distortion_coefficients, dtype=float).reshape(-1).tolist(),
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def undistorted_corner_angles(
    corners: np.ndarray, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray
) -> np.ndarray:
    """Convert image corners to horizontal/vertical optical angles.

    Omitting ``P`` makes OpenCV return normalized ideal-image coordinates, so
    this avoids rebuilding and inverting the same camera matrix for every tag.
    """

    normalized = cv2.undistortPoints(
        np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix,
        distortion_coefficients,
    ).reshape(-1, 2)
    return np.arctan(normalized).astype(np.float64, copy=False)
