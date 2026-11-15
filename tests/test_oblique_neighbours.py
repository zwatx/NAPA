"""Nearby foreshortened tags stay distinct, including two copies of one ID."""

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from tests.test_napa_detector import DICTIONARY


@pytest.mark.parametrize("minimum", [16, 32])
@pytest.mark.parametrize("ids", [(137, 424), (137, 137)])
@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_nearby_oblique_tags_keep_both_physical_detections(minimum, ids, turns):
    image = np.full((480, 640), 255, np.uint8)
    corners = []
    singles = []
    for x, tag_id in zip([160, 224], ids, strict=True):
        marker = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, 48, borderBits=1)
        single = np.full_like(image, 255)
        single[96:384, x : x + 48] = cv2.resize(marker, (48, 288), interpolation=cv2.INTER_NEAREST)
        image = np.minimum(image, single)
        singles.append(single)
        corners.append(np.array([[x, 96], [x + 48, 96], [x + 48, 384], [x, 384]], dtype=float) - 0.5)
    for _ in range(turns):
        width = image.shape[1]
        image = np.rot90(image).copy()
        singles = [np.rot90(single).copy() for single in singles]
        corners = [np.column_stack((q[:, 1], width - 1 - q[:, 0])) for q in corners]
    before = image.copy()
    detector = NapaDetector(min_tag_px=minimum)
    try:
        for single, tag_id, q in zip(singles, ids, corners, strict=True):
            observations = detector.detect(single)
            assert [o.tag_id for o in observations] == [tag_id]
            np.testing.assert_allclose(observations[0].corners, q, atol=0.05)
        ordinary = detector.detect(image)
        assert len(ordinary) == 2
        remaining = list(ordinary)
        for tag_id, q in zip(ids, corners, strict=True):
            nearest = min(range(len(remaining)), key=lambda i: np.linalg.norm(remaining[i].corners.mean(0) - q.mean(0)))
            observation = remaining.pop(nearest)
            assert (observation.tag_id, observation.hamming) == (tag_id, 0)
            np.testing.assert_allclose(observation.corners, q, atol=0.05)
        # Reuse after two outputs must not retain a stale second footprint.
        assert [o.tag_id for o in detector.detect(singles[0])] == [ids[0]]
    finally:
        detector.close()
    for shadow, split in [(False, True), (True, False), (True, True)]:
        detector = NapaDetector(min_tag_px=minimum, recover_shadows=shadow, split_touching=split)
        try:
            recovered = detector.detect(image)
        finally:
            detector.close()
        assert len(recovered) == 2
        for old, new in zip(ordinary, recovered, strict=True):
            assert (old.tag_id, old.hamming, old.decision_margin) == (new.tag_id, new.hamming, new.decision_margin)
            np.testing.assert_array_equal(old.corners, new.corners)
    np.testing.assert_array_equal(image, before)
