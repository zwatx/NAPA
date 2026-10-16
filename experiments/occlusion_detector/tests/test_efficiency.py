from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from robust_apriltag.family import (  # noqa: E402
    FULL_MASK,
    TAG36H11,
    decode_observed,
    popcount,
    rotate90_code,
)
from robust_apriltag import Robust36h11Detector, render_tag36h11_scene  # noqa: E402


def _reference_decode(
    observed_code: int,
    observed_mask: int,
    *,
    max_hamming: int = 2,
) -> tuple[int, int, int, int]:
    best: tuple[int, int, int] | None = None
    second_hamming = 10_000
    code = observed_code
    mask = observed_mask
    for rotation in range(4):
        for tag_id, family_code in enumerate(TAG36H11.codes):
            hamming = popcount((code ^ family_code) & mask)
            if best is None or hamming < best[1]:
                if best is not None:
                    second_hamming = min(second_hamming, best[1])
                best = (tag_id, hamming, rotation)
            elif hamming < second_hamming:
                second_hamming = hamming
        code = rotate90_code(code)
        mask = rotate90_code(mask)
    assert best is not None
    margin = (
        11 - best[1]
        if observed_mask == FULL_MASK and max_hamming <= 2
        else second_hamming - best[1]
    )
    return best[0], best[1], best[2], margin


def test_decode_preserves_all_rotation_outputs() -> None:
    for tag_id in range(len(TAG36H11.codes)):
        observed = TAG36H11.codes[tag_id]
        for _rotation in range(4):
            expected = _reference_decode(observed, FULL_MASK)
            assert decode_observed(observed, FULL_MASK, max_hamming=2, min_reliable_bits=36) == expected
            observed = rotate90_code(observed)


def test_decode_preserves_one_and_two_bit_error_outputs() -> None:
    error_masks = (1 << 7, (1 << 3) | (1 << 29))
    for tag_id, family_code in enumerate(TAG36H11.codes):
        observed = family_code
        for rotation in range(4):
            for error_mask in error_masks:
                corrupted = observed ^ error_mask
                expected = _reference_decode(corrupted, FULL_MASK)
                assert decode_observed(
                    corrupted,
                    FULL_MASK,
                    max_hamming=2,
                    min_reliable_bits=36,
                ) == expected, (tag_id, rotation, error_mask)
            observed = rotate90_code(observed)


def test_decode_preserves_erased_mask_exhaustive_fallback() -> None:
    observed = TAG36H11.codes[42] ^ (1 << 3)
    mask = FULL_MASK ^ (1 << 7) ^ (1 << 21)
    for _rotation in range(4):
        expected = _reference_decode(observed, mask)
        assert decode_observed(observed, mask, max_hamming=2, min_reliable_bits=30) == expected
        observed = rotate90_code(observed)
        mask = rotate90_code(mask)


def test_decode_preserves_larger_hamming_exhaustive_fallback() -> None:
    observed = TAG36H11.codes[314] ^ (1 << 2) ^ (1 << 13) ^ (1 << 31)
    expected = _reference_decode(observed, FULL_MASK, max_hamming=3)
    assert decode_observed(observed, FULL_MASK, max_hamming=3, min_reliable_bits=36) == expected


def test_bit_centers_are_cached() -> None:
    assert TAG36H11.bit_centers is TAG36H11.bit_centers
    assert TAG36H11.bit_centers.dtype == np.float32


def test_batched_projection_and_edge_search_preserve_detection() -> None:
    image = render_tag36h11_scene(42, image_size=(240, 320), cell_size=16)
    detections = Robust36h11Detector(refine_pose=False).detect(image)
    assert [(detection.id, detection.hamming) for detection in detections] == [(42, 0)]
