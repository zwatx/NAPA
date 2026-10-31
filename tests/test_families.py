from __future__ import annotations

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector, families
from napa_vision.tags import family_layout, tag_cells, tag_image

# AprilRobotics' official images for ID 0 ("#" black; blank cells of the circle tag are white paper).
OFFICIAL_ID_0 = {
    "tag16h5": ["........", ".######.", ".###.##.", ".###..#.", ".####.#.", ".#.#..#.", ".######.", "........"],
    "tagStandard41h12": [
        "..#....##",
        "#########",
        ".#.....##",
        "##.....#.",
        "##.##..##",
        "##.#.#.#.",
        ".#.....##",
        "#########",
        "..#.##.##",
    ],
    "tagCircle21h7": [
        "....#....",
        ".#######.",
        ".#.....#.",
        "##.#...##",
        "##.....#.",
        ".#.#.#.#.",
        ".#.....#.",
        ".#######.",
        "...###...",
    ],
}


def test_every_upstream_family_is_available() -> None:
    assert families()[0] == "tag36h11"
    assert set(families()) == {
        "tag36h11",
        "tag16h5",
        "tag25h9",
        "tag36h10",
        "tagStandard41h12",
        "tagStandard52h13",
        "tagCircle21h7",
        "tagCircle49h12",
        "tagCustom48h12",
    }


@pytest.mark.parametrize("family", sorted(OFFICIAL_ID_0))
def test_tags_match_the_official_artwork(family: str) -> None:
    official = np.array([[cell == "." for cell in row] for row in OFFICIAL_ID_0[family]])
    quiet = 0 if family_layout(family).reversed else 1
    np.testing.assert_array_equal(tag_cells(0, family=family, quiet_zone_cells=quiet), official)


@pytest.mark.parametrize(
    ("family", "dictionary"),
    [("tag16h5", "DICT_APRILTAG_16h5"), ("tag25h9", "DICT_APRILTAG_25h9"), ("tag36h10", "DICT_APRILTAG_36h10")],
)
def test_classic_families_match_opencv_turned_180_degrees(family: str, dictionary: str) -> None:
    layout = family_layout(family)
    opencv = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    for tag_id in range(0, layout.ncodes, max(1, layout.ncodes // 50)):
        drawn = cv2.aruco.generateImageMarker(opencv, tag_id, layout.width, borderBits=1) > 127
        np.testing.assert_array_equal(tag_cells(tag_id, family=family, quiet_zone_cells=0), np.rot90(drawn, 2))


@pytest.mark.parametrize("family", families())
def test_detects_every_family_with_napa_corner_order(family: str) -> None:
    """Napa's corners are the official artwork's bottom-right, bottom-left, top-left, and top-right corners
    of the tag-size square, in every family."""

    layout = family_layout(family)
    detector = NapaDetector(family=family)
    cell = 14
    for tag_id in sorted({0, layout.ncodes // 2, layout.ncodes - 1}):
        art = tag_image(tag_id, family=family, cell_px=cell, quiet_zone_cells=2)
        image = np.full((art.shape[0] + 40, art.shape[1] + 40), 255, np.uint8)
        image[20 : 20 + art.shape[0], 20 : 20 + art.shape[1]] = art
        # The tag-size square's first and last pixel, in image coordinates.
        margin = (art.shape[0] // cell - layout.width) // 2
        first = 20 + margin * cell
        last = first + layout.width * cell - 1
        expected = np.array([[last, last], [first, last], [first, first], [last, first]], float)
        expected += [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]  # out to the pixels' outer edges
        for turns in range(4):
            rotated = np.ascontiguousarray(np.rot90(image, turns))
            [observation] = detector.detect(rotated)
            assert (observation.tag_id, observation.family) == (tag_id, family)
            # np.rot90 turns the image counter-clockwise: (x, y) -> (y, width - 1 - x).
            width = image.shape[1]
            turned = expected.copy()
            for _ in range(turns):
                turned = np.column_stack([turned[:, 1], width - 1 - turned[:, 0]])
                width = image.shape[0] if width == image.shape[1] else image.shape[1]
            np.testing.assert_allclose(observation.corners, turned, atol=0.15)


@pytest.mark.parametrize(
    ("family", "recommended", "maximum"),
    [
        ("tag36h11", 3, 3),
        ("tag16h5", 0, 2),
        ("tag25h9", 0, 3),
        ("tag36h10", 2, 3),
        ("tagStandard41h12", 3, 3),
        ("tagStandard52h13", 3, 3),
        ("tagCircle21h7", 0, 3),
        ("tagCircle49h12", 3, 3),
        ("tagCustom48h12", 3, 3),
    ],
)
def test_family_correction_limits(family: str, recommended: int, maximum: int) -> None:
    detector = NapaDetector(family=family)
    assert detector.max_hamming == recommended
    detector.close()
    detector = NapaDetector(family=family, max_hamming=3)
    assert detector.max_hamming == maximum
    detector.close()


def test_tag16h5_rejects_random_patterns() -> None:
    """tag16h5's codes are only 5 bits apart: random black-bordered patterns must not decode."""

    rng = np.random.default_rng(3)
    detector = NapaDetector(family="tag16h5")
    layout = family_layout("tag16h5")
    valid = {tuple(tag_cells(tag_id, family="tag16h5", quiet_zone_cells=0).ravel()) for tag_id in range(30)}
    false_ids = 0
    for _ in range(40):
        image = np.full((300, 600), 255, np.uint8)
        for k in range(4):
            cells = np.zeros((layout.width, layout.width), bool)
            cells[1:-1, 1:-1] = rng.random((layout.width - 2, layout.width - 2)) < 0.5
            if any(tuple(np.rot90(cells, r).ravel()) in valid for r in range(4)):
                continue
            pattern = np.repeat(np.repeat(np.where(cells, 255, 0).astype(np.uint8), 16, 0), 16, 1)
            image[50 : 50 + pattern.shape[0], 30 + 140 * k : 30 + 140 * k + pattern.shape[1]] = pattern
        false_ids += len(detector.detect(image))
    assert false_ids <= 2


def _grid(family: str, ids: list[int], *, columns: int = 4, cell: int = 8, angle_deg: float = 0.0) -> np.ndarray:
    """A classic family's tags in an AprilGrid layout: two-cell gaps with a black square at every crossing,
    touching the four tags around it at their corners."""

    width = family_layout(family).width
    pitch = (width + 2) * cell
    rows = len(ids) // columns
    board = np.full((rows * pitch + 2 * cell, columns * pitch + 2 * cell), 255, np.uint8)
    for row in range(rows + 1):
        for column in range(columns + 1):
            board[row * pitch : row * pitch + 2 * cell, column * pitch : column * pitch + 2 * cell] = 0
    for index, tag_id in enumerate(ids):
        y, x = 2 * cell + (index // columns) * pitch, 2 * cell + (index % columns) * pitch
        board[y : y + width * cell, x : x + width * cell] = tag_image(
            tag_id, family=family, cell_px=cell, quiet_zone_cells=0
        )
    board = np.pad(board, 6 * cell, constant_values=255)
    if angle_deg:
        centre = (board.shape[1] / 2, board.shape[0] / 2)
        board = cv2.warpAffine(
            board,
            cv2.getRotationMatrix2D(centre, angle_deg, 1.0),
            board.shape[::-1],
            flags=cv2.INTER_AREA,
            borderValue=255,
        )
    return board


@pytest.mark.parametrize("family", ["tag36h11", "tag25h9", "tag36h10"])
def test_split_touching_finds_grids_of_classic_families(family: str) -> None:
    ids = list(range(1, 13))
    for angle in (0.0, 10.0):
        image = _grid(family, ids, angle_deg=angle)
        assert len(NapaDetector(family=family, min_tag_px=16, max_detections=64).detect(image)) <= 3
        found = NapaDetector(family=family, min_tag_px=16, max_detections=64, split_touching=True).detect(image)
        assert {observation.tag_id for observation in found} <= set(ids)
        assert len(found) >= (12 if angle == 0.0 else 9)


@pytest.mark.parametrize("family", ["tag16h5", "tagCircle21h7"])
def test_split_touching_skips_families_whose_short_codes_match_pieces(family: str) -> None:
    """Exact chance matches of 16- and 21-bit codes are too likely among the many extra pieces."""

    image = _grid("tag16h5", list(range(1, 13))) if family == "tag16h5" else _grid("tag36h11", list(range(1, 13)))
    rng = np.random.default_rng(5)
    noisy = np.clip(image + rng.normal(0, 6, image.shape), 0, 255).astype(np.uint8)
    for frame in (image, noisy):
        ordinary = NapaDetector(family=family, min_tag_px=16).detect(frame)
        split = NapaDetector(family=family, min_tag_px=16, split_touching=True).detect(frame)
        assert [(o.tag_id, o.corners.tolist()) for o in split] == [(o.tag_id, o.corners.tolist()) for o in ordinary]


@pytest.mark.parametrize("family", families())
def test_split_touching_leaves_isolated_tags_unchanged(family: str) -> None:
    layout = family_layout(family)
    art = tag_image(layout.ncodes // 3, family=family, cell_px=12, quiet_zone_cells=2)
    image = np.full((art.shape[0] + 60, art.shape[1] + 80), 255, np.uint8)
    image[30 : 30 + art.shape[0], 40 : 40 + art.shape[1]] = art
    for min_tag_px in (16, 32):
        ordinary = NapaDetector(family=family, min_tag_px=min_tag_px).detect(image)
        split = NapaDetector(family=family, min_tag_px=min_tag_px, split_touching=True).detect(image)
        assert [(o.tag_id, o.corners.tolist()) for o in split] == [(o.tag_id, o.corners.tolist()) for o in ordinary]
        assert [o.tag_id for o in ordinary] == [layout.ncodes // 3]
