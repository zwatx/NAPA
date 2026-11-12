from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks import vkapriltag_corpus as corpus

SQUARE = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]])


def _tag(tag_id: int = 1) -> dict:
    return {"id": tag_id, "corners": SQUARE.tolist(), "size_px": 40.0}


def test_a_tag_is_found_by_id_and_worst_corner_under_a_quarter_side() -> None:
    assert corpus.score_frame([_tag()], [(1, SQUARE + [9.9, 0.0])]) == (pytest.approx([9.9]), 0)
    assert corpus.score_frame([_tag()], [(1, SQUARE + [10.1, 0.0])]) == ([], 1)
    assert corpus.score_frame([_tag()], [(2, SQUARE)]) == ([], 1)


def test_each_detection_counts_once() -> None:
    errors, false = corpus.score_frame([_tag()], [(1, SQUARE + [0.3, 0.4]), (1, SQUARE)])
    assert errors == [0.0] and false == 1


def test_corners_move_to_the_corpus_pixels_in_the_truth_winding() -> None:
    sign = np.sign(corpus.signed_area(SQUARE))
    converted = corpus.to_corpus(SQUARE[::-1] - 0.5, sign)
    assert np.sign(corpus.signed_area(converted)) == sign
    assert corpus.worst_corner(SQUARE, converted) == 0.0


def test_an_incomplete_corpus_is_an_error(tmp_path) -> None:
    (tmp_path / "one-tag").mkdir()
    (tmp_path / "one-tag" / "gt.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="26 scenarios"):
        corpus.evaluate(tmp_path, "napa")
