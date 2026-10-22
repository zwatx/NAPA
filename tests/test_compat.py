from __future__ import annotations

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from napa_vision.compat import Detector
from napa_vision.tags import tag_image

FX = FY = 900.0
CX, CY = 639.5, 359.5
CAMERA = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
TAG_SIZE = 0.1651
# The official artwork's corners (top-left, top-right, bottom-right, bottom-left) in upstream's tag frame:
# +x right and +y down on the artwork, +z into the tag.
ARTWORK_CORNERS = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float) * TAG_SIZE / 2


def _scene(tag_id: int, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    projected = (CAMERA @ (ARTWORK_CORNERS @ rotation.T + translation).T).T
    projected = projected[:, :2] / projected[:, 2:]
    artwork = tag_image(tag_id, cell_px=40)  # black square from 40 to 360
    source = np.array([[40, 40], [360, 40], [360, 360], [40, 360]], np.float32) - 0.5
    warp = cv2.getPerspectiveTransform(source, projected.astype(np.float32))
    return cv2.warpPerspective(artwork, warp, (1280, 720), flags=cv2.INTER_AREA, borderValue=255)


def test_matches_upstream_conventions() -> None:
    rotation = cv2.Rodrigues(np.array([0.3, -0.4, 2.0]))[0]
    image = _scene(77, rotation, np.array([0.1, -0.05, 1.2]))
    [detection] = Detector().detect(image)
    [native] = NapaDetector().detect(image)
    assert detection.tag_family == b"tag36h11" and detection.tag_id == 77
    assert detection.hamming == 0 and detection.decision_margin > 50
    # Upstream order is Napa's p[1], p[0], p[3], p[2], half a pixel further along each axis.
    np.testing.assert_allclose(detection.corners, native.corners[[1, 0, 3, 2]] + 0.5)
    np.testing.assert_allclose(detection.center, native.center + 0.5)
    tag_points = np.array([[-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1]], float)
    mapped = (detection.homography @ tag_points.T).T
    np.testing.assert_allclose(mapped[:, :2] / mapped[:, 2:], detection.corners, atol=1e-9)
    assert detection.homography[2, 2] == pytest.approx(1.0)
    assert detection.pose_R is None and detection.pose_t is None and detection.pose_err is None
    assert "tag_id = 77" in repr(detection)


@pytest.mark.parametrize("seed", range(6))
def test_pose_matches_the_true_pose(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rotation = cv2.Rodrigues(np.array([*rng.normal(0, 0.4, 2), rng.uniform(-np.pi, np.pi)]))[0]
    distance = rng.uniform(0.8, 2.5)
    translation = np.array([rng.uniform(-0.2, 0.2) * distance, rng.uniform(-0.1, 0.1) * distance, distance])
    image = _scene(int(rng.integers(587)), rotation, translation)
    [detection] = Detector().detect(image, estimate_tag_pose=True, camera_params=(FX, FY, CX, CY), tag_size=TAG_SIZE)
    assert detection.pose_t.shape == (3, 1) and detection.pose_R.shape == (3, 3)
    np.testing.assert_allclose(detection.pose_R @ detection.pose_R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(detection.pose_R) == pytest.approx(1.0)
    np.testing.assert_allclose(detection.pose_t.ravel(), translation, atol=0.01 * distance)
    np.testing.assert_allclose(detection.pose_R, rotation, atol=0.03)
    assert 0 <= detection.pose_err < 1e-6


def test_reads_several_families_and_rejects_unknown_ones() -> None:
    with pytest.raises(ValueError, match="tag99h1"):
        Detector(families="tag36h11 tag99h1")
    image = np.full((260, 520), 255, np.uint8)
    image[30:230, 30:230] = tag_image(4, cell_px=20)
    image[30:190, 290:450] = tag_image(9, family="tag16h5", cell_px=20)
    found = {(d.tag_family, d.tag_id) for d in Detector(families="tag36h11 tag16h5").detect(image)}
    assert found == {(b"tag36h11", 4), (b"tag16h5", 9)}


def test_rejects_missing_pose_inputs() -> None:
    detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, min_tag_px=24)
    image = np.full((64, 64), 255, np.uint8)
    assert detector.detect(image) == []
    with pytest.raises(ValueError, match="camera_params"):
        detector.detect(image, estimate_tag_pose=True, tag_size=0.1)
    with pytest.raises(ValueError, match="tag_size"):
        detector.detect(image, estimate_tag_pose=True, camera_params=(1, 1, 0, 0))
    with pytest.raises(ValueError):
        detector.detect(np.zeros((64, 64, 3), np.uint8))
