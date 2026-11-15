from __future__ import annotations

import cv2
import numpy as np

from napa_vision import NapaDetector
from napa_vision.tags import tag_cells


def _frame_with_dim_white_cells(tag_id: int) -> tuple[np.ndarray, int]:
    """A tag whose white payload cells with four white neighbours are dim: still white, but sharpening flips them."""

    cells = np.asarray(tag_cells(tag_id)).astype(int)
    size, cell = cells.shape[0], 16
    values = np.where(cells, 255.0, 0.0)
    dimmed = 0
    for row in range(2, size - 2):
        for column in range(2, size - 2):
            around = cells[row - 1, column] + cells[row + 1, column] + cells[row, column - 1] + cells[row, column + 1]
            if cells[row, column] and around == 4:
                values[row, column] = 150.0
                dimmed += 1
    art = cv2.resize(values.astype(np.uint8), (size * cell, size * cell), interpolation=cv2.INTER_NEAREST)
    frame = np.full((art.shape[0] + 80, art.shape[1] + 80), 255, np.uint8)
    frame[40 : 40 + art.shape[0], 40 : 40 + art.shape[1]] = art
    return frame, dimmed


def test_cells_flipped_by_sharpening_still_decode_exactly() -> None:
    detector = NapaDetector(max_hamming=0, recover_shadows=False)
    found = total = 0
    try:
        for tag_id in range(0, 587, 5):
            frame, dimmed = _frame_with_dim_white_cells(tag_id)
            if dimmed:
                total += 1
                found += [(o.tag_id, o.hamming) for o in detector.detect(frame)] == [(tag_id, 0)]
    finally:
        detector.close()
    assert total >= 30 and found >= 0.8 * total
