"""Helpers for the static tag36h11 family data."""

from __future__ import annotations

from dataclasses import dataclass

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

    @property
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
_NEAR_CODE_LOOKUP: dict[int, tuple[int, int, int]] | None = None


def popcount(value: int) -> int:
    return bin(int(value)).count("1")


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


def _flip_combinations(nbits: int) -> list[int]:
    masks = [0]
    masks.extend(1 << i for i in range(nbits))
    for i in range(nbits):
        for j in range(i + 1, nbits):
            masks.append((1 << i) | (1 << j))
    return masks


def near_code_lookup() -> dict[int, tuple[int, int, int]]:
    global _NEAR_CODE_LOOKUP
    if _NEAR_CODE_LOOKUP is not None:
        return _NEAR_CODE_LOOKUP

    lookup: dict[int, tuple[int, int, int]] = {}
    flip_masks = _flip_combinations(tag36h11.NBITS)
    for tag_id, family_code in enumerate(TAG36H11.codes):
        code = family_code
        for rotation in range(4):
            for flip_mask in flip_masks:
                observed = code ^ flip_mask
                hamming = popcount(flip_mask)
                existing = lookup.get(observed)
                if existing is None or hamming < existing[1]:
                    lookup[observed] = (tag_id, hamming, rotation)
            code = rotate90_code(code)
    _NEAR_CODE_LOOKUP = lookup
    return lookup


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

    if observed_mask == FULL_MASK and max_hamming <= 2:
        match = near_code_lookup().get(observed_code)
        if match is None:
            return None
        tag_id, hamming, rotation = match
        return tag_id, hamming, rotation, tag36h11.H - hamming

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

    if best is None or best[1] > max_hamming:
        return None
    margin = second_hamming - best[1] if second_hamming < 10_000 else TAG36H11.nbits - best[1]
    return best[0], best[1], best[2], margin
