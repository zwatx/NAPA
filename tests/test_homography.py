"""Tag homographies and centres, as upstream AprilTag reports them."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from napa_vision import FiducialImageObservation, NapaDetector


def test_homography_maps_upstream_tag_coordinates_to_corners() -> None:
    corners = np.array([[149.3, 99.2], [419.3, 139.2], [399.4, 379.4], [169.4, 329.4]])
    observation = FiducialImageObservation(1, corners)
    homography = observation.homography

    assert homography[2, 2] == 1.0
    # Napa's corners 0-3 are upstream AprilTag's p[1], p[0], p[3], p[2].
    for corner, tag_point in zip(corners, [(1, 1), (-1, 1), (-1, -1), (1, -1)], strict=True):
        projected = homography @ [*tag_point, 1.0]
        assert projected[:2] / projected[2] == pytest.approx(corner)

    # The centre is where the diagonals cross, not the corners' mean.
    (s, _), *_ = np.linalg.lstsq(
        np.column_stack((corners[2] - corners[0], corners[1] - corners[3])), corners[1] - corners[0], rcond=None
    )
    assert observation.center == pytest.approx(corners[0] + s * (corners[2] - corners[0]))
    assert not np.allclose(observation.center, corners.mean(axis=0), atol=0.5)


def test_homography_of_an_upright_official_tag_is_upstreams() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = np.full((480, 640), 255, dtype=np.uint8)
    image[200:280, 300:380] = cv2.rotate(cv2.aruco.generateImageMarker(dictionary, 5, 80, borderBits=1), cv2.ROTATE_180)
    (observation,) = NapaDetector().detect(image)

    # Upstream's tag frame has +x right and +y down on the official artwork, 2 units across the black square.
    expected = np.array([[40.0, 0.0, 339.5], [0.0, 40.0, 239.5], [0.0, 0.0, 1.0]])
    assert observation.homography == pytest.approx(expected, abs=0.1)
    assert observation.center == pytest.approx([339.5, 239.5], abs=0.05)


def test_homography_rejects_degenerate_corners() -> None:
    with pytest.raises(ValueError, match="quadrilateral"):
        _ = FiducialImageObservation(1, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])).homography
