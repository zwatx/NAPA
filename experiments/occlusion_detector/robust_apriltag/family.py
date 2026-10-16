"""Helpers for the static tag36h11 family data."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from . import tag36h11


@dataclass(frozen=True)
class Tag36h11Family:
    name: str
    codes: tuple[int, ...]
    bit_x: np.ndarray
    bit_y: np.ndarray
    width_at_border: int
    total_width: int
    nbits: int

    @cached_property
    def bit_centers(self) -> np.ndarray:
        return np.column_stack((self.bit_x + 0.5, self.bit_y + 0.5)).astype(np.float32)


TAG36H11 = Tag36h11Family(
    name=tag36h11.NAME,
    codes=tag36h11.CODES,
    bit_x=np.asarray(tag36h11.BIT_X, dtype=np.float32),
    bit_y=np.asarray(tag36h11.BIT_Y, dtype=np.float32),
    width_at_border=tag36h11.WIDTH_AT_BORDER,
    total_width=tag36h11.TOTAL_WIDTH,
    nbits=tag36h11.NBITS,
)

FULL_MASK = (1 << tag36h11.NBITS) - 1


def popcount(value: int) -> int:
    return int(value).bit_count()


def code_to_bits(code: int, nbits: int = tag36h11.NBITS) -> np.ndarray:
    """Return MSB-first bits in the same order used by AprilTag decoding."""

    return np.asarray([(code >> (nbits - 1 - i)) & 1 for i in range(nbits)], dtype=np.uint8)


def rotate90_code(code: int, nbits: int = tag36h11.NBITS) -> int:
    """Rotate an AprilTag codeword by 90 degrees, matching upstream AprilTag."""

    p = nbits
    l = 0
    if nbits % 4 == 1:
        p = nbits - 1
        l = 1
    rotated = ((code >> l) << (p // 4 + l)) | (code >> (3 * p // 4 + l) << l) | (code & l)
    return rotated & ((1 << nbits) - 1)


def _rotated_codes() -> tuple[tuple[int, int, int], ...]:
    """Return the 2,348 rotated family words instead of a 1.5M-entry near-code table."""

    variants: list[tuple[int, int, int]] = []
    # Preserve the old decoder's iteration and reported rotation semantics: it
    # rotated the observation clockwise, so the equivalent family word is
    # rotated in the opposite direction.
    for rotation in range(4):
        family_rotations = (-rotation) % 4
        for tag_id, family_code in enumerate(TAG36H11.codes):
            code = family_code
            for _ in range(family_rotations):
                code = rotate90_code(code)
            variants.append((code, tag_id, rotation))
    return tuple(variants)


ROTATED_CODES = _rotated_codes()

_BLOCK_BITS = 12
_BLOCK_MASK = (1 << _BLOCK_BITS) - 1
_BLOCK_SHIFTS = (0, _BLOCK_BITS, 2 * _BLOCK_BITS)


def _multi_index() -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Index each rotated code by three disjoint 12-bit blocks."""

    blocks: list[list[list[int]]] = [
        [[] for _ in range(1 << _BLOCK_BITS)] for _ in _BLOCK_SHIFTS
    ]
    for candidate_index, (code, _tag_id, _rotation) in enumerate(ROTATED_CODES):
        for block, shift in zip(blocks, _BLOCK_SHIFTS, strict=True):
            block[(code >> shift) & _BLOCK_MASK].append(candidate_index)
    return tuple(tuple(tuple(bucket) for bucket in block) for block in blocks)


_MULTI_INDEX = _multi_index()


def _full_mask_candidates(observed_code: int) -> list[int]:
    """Return ordered candidates guaranteed to include every <=2-bit match."""

    candidates: set[int] = set()
    for block, shift in zip(_MULTI_INDEX, _BLOCK_SHIFTS, strict=True):
        candidates.update(block[(observed_code >> shift) & _BLOCK_MASK])
    return sorted(candidates)


def make_code_from_bits(bits: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int]:
    """Pack observed MSB-first bits and an optional reliability mask."""

    code = 0
    packed_mask = 0
    if mask is None:
        mask = np.ones(len(bits), dtype=bool)
    for bit, keep in zip(bits, mask):
        code <<= 1
        packed_mask <<= 1
        if int(bit):
            code |= 1
        if bool(keep):
            packed_mask |= 1
    return code, packed_mask


def decode_observed(
    observed_code: int,
    observed_mask: int,
    *,
    max_hamming: int,
    min_reliable_bits: int,
) -> tuple[int, int, int, int] | None:
    """Decode a possibly-erased observation.

    Returns ``(tag_id, hamming, rotation, decision_margin)``. The observation
    and mask are rotated together, so uncertain bits remain uncertain in every
    orientation.
    """

    if popcount(observed_mask) < min_reliable_bits:
        return None

    best: tuple[int, int, int] | None = None
    second_hamming = 10_000
    if observed_mask == FULL_MASK and max_hamming <= 2:
        candidate_indices = _full_mask_candidates(observed_code)
    else:
        candidate_indices = range(len(ROTATED_CODES))
    for candidate_index in candidate_indices:
        family_code, tag_id, rotation = ROTATED_CODES[candidate_index]
        hamming = popcount((observed_code ^ family_code) & observed_mask)
        if best is None or hamming < best[1]:
            if best is not None:
                second_hamming = min(second_hamming, best[1])
            best = (tag_id, hamming, rotation)
        elif hamming < second_hamming:
            second_hamming = hamming

    if best is None or best[1] > max_hamming:
        return None
    if observed_mask == FULL_MASK and max_hamming <= 2:
        return best[0], best[1], best[2], tag36h11.H - best[1]
    margin = second_hamming - best[1] if second_hamming < 10_000 else TAG36H11.nbits - best[1]
    return best[0], best[1], best[2], margin
