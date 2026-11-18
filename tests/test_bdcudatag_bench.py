from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks import bdcudatag

SQUARE = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]])


def test_a_tag_is_found_by_id_and_a_nearby_centre() -> None:
    truth = [(1, SQUARE)]
    assert bdcudatag.score(truth, [(1, SQUARE + [29.0, 0.0])])[1] != []  # centre within 0.75 of the 40 px side
    assert bdcudatag.score(truth, [(1, SQUARE + [31.0, 0.0])])[1] == []
    assert bdcudatag.score(truth, [(2, SQUARE)]) == (1, [])


def test_corner_error_uses_the_best_corner_order() -> None:
    wrong, errors = bdcudatag.score([(1, SQUARE)], [(1, np.roll(SQUARE, 2, axis=0) + [0.3, 0.4])])
    assert wrong == 0
    assert errors == pytest.approx([0.5])


def test_an_incomplete_hard_corpus_is_an_error(tmp_path) -> None:
    (tmp_path / "metadata.jsonl").write_text(json.dumps({"num_tags": 1}) + "\n")
    with pytest.raises(ValueError, match="780 frames"):
        bdcudatag.evaluate_hard(tmp_path, lambda image: [])
