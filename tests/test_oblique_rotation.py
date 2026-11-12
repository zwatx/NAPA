"""Rotated foreshortened tags must survive proposal filtering with accurate corners."""

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from tests.test_napa_detector import DICTIONARY


@pytest.mark.parametrize("ids", [(137, 424), (137, 137)])
@pytest.mark.parametrize(
    "minimum,angle",
    [(16, angle) for angle in range(0, 180, 15)] + [(32, angle) for angle in [30, 45, 120, 135, 150]],
)
def test_rotated_oblique_tags_keep_ids_and_corners(minimum, angle, ids):
    image = np.full((480, 640), 255, np.uint8)
    truth = []
    for x, tag_id in zip([160, 224], ids, strict=True):
        marker = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, 48, borderBits=1)
        image[96:384, x : x + 48] = cv2.resize(marker, (48, 288), interpolation=cv2.INTER_NEAREST)
        truth.append(np.array([[x, 96], [x + 48, 96], [x + 48, 384], [x, 384]], dtype=float) - 0.5)
    matrix = cv2.getRotationMatrix2D((320, 240), angle, 1)
    matrix[:, 2] += [160, 240]
    image = cv2.warpAffine(image, matrix, (960, 960), flags=cv2.INTER_LINEAR, borderValue=255)
    truth = [np.c_[q, np.ones(4)] @ matrix.T for q in truth]
    before = image.copy()
    detector = NapaDetector(min_tag_px=minimum)
    try:
        ordinary = detector.detect(image)
    finally:
        detector.close()
    assert len(ordinary) == 2
    remaining = list(ordinary)
    for tag_id, corners in zip(ids, truth, strict=True):
        nearest = min(
            range(len(remaining)), key=lambda i: np.linalg.norm(remaining[i].corners.mean(0) - corners.mean(0))
        )
        observation = remaining.pop(nearest)
        assert (observation.tag_id, observation.hamming) == (tag_id, 0)
        np.testing.assert_allclose(observation.corners, corners, atol=0.1)
    for shadow, split in [(False, True), (True, False), (True, True)]:
        detector = NapaDetector(min_tag_px=minimum, recover_shadows=shadow, split_touching=split)
        try:
            recovered = detector.detect(image)
        finally:
            detector.close()
        assert len(recovered) == 2
        if shadow:  # recovery searches coarser levels first, so tags can come back in another order
            centres = [o.corners.mean(0) for o in ordinary]
            recovered.sort(key=lambda d: int(np.argmin([np.linalg.norm(d.corners.mean(0) - c) for c in centres])))
        for old, new in zip(ordinary, recovered, strict=True):
            assert (old.tag_id, old.hamming, old.decision_margin) == (new.tag_id, new.hamming, new.decision_margin)
            if shadow:  # recovery may find a tag from a coarser level, shifting its corners slightly
                np.testing.assert_allclose(old.corners, new.corners, atol=0.1)
            else:
                np.testing.assert_array_equal(old.corners, new.corners)
    np.testing.assert_array_equal(image, before)


# These angles exercise the new shape route. The full12-angle sweep is retained
# separately: misses at15/75/105 also occur before the stricter border policy.
@pytest.mark.parametrize("angle", [30, 45, 60, 120, 135, 150])
def test_rotated_short_code_with_complete_border_is_retained(angle):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    image = np.full((480, 640), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(dictionary, 14, 48, borderBits=1)
    image[96:384, 160:208] = cv2.resize(marker, (48, 288), interpolation=cv2.INTER_NEAREST)
    corners = np.array([[160, 96], [208, 96], [208, 384], [160, 384]], dtype=float) - 0.5
    matrix = cv2.getRotationMatrix2D((320, 240), angle, 1)
    matrix[:, 2] += [160, 240]
    image = cv2.warpAffine(image, matrix, (960, 960), flags=cv2.INTER_LINEAR, borderValue=255)
    corners = np.c_[corners, np.ones(4)] @ matrix.T
    detector = NapaDetector(family="tag16h5", min_tag_px=16)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert len(observations) == 1
    assert (observations[0].tag_id, observations[0].hamming) == (14, 0)
    np.testing.assert_allclose(observations[0].corners, corners, atol=0.1)


@pytest.mark.parametrize("minimum", [16, 32])
def test_new_sparse_parent_does_not_block_existing_split_piece(minimum):
    image = np.full((480, 640), 255, np.uint8)
    cv2.line(image, (159, 63), (319, 223), 0, 2, lineType=cv2.LINE_8)
    image[224:272, 320:368] = cv2.aruco.generateImageMarker(DICTIONARY, 137, 48, borderBits=1)
    corners = np.array([[320, 224], [368, 224], [368, 272], [320, 272]], dtype=float) - 0.5
    detector = NapaDetector(min_tag_px=minimum, split_touching=True)
    try:
        observations = detector.detect(image)
    finally:
        detector.close()
    assert len(observations) == 1
    assert (observations[0].tag_id, observations[0].hamming) == (137, 0)
    np.testing.assert_allclose(observations[0].corners, corners, atol=0.01)
