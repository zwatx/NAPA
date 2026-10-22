from __future__ import annotations

import zlib

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from napa_vision.tags import TAG36H11_COUNT, png_bytes, tag_cells, tag_image, tag_svg
from napa_vision.workflows import tag_main

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


# AprilRobotics' official tag36h11 image for ID 0 (tag36_11_00000.png), "#" black.
OFFICIAL_TAG_0 = [
    "..........",
    ".########.",
    ".#..#.#.#.",
    ".##...#.#.",
    ".##..####.",
    ".#.#.####.",
    ".##.#..##.",
    ".####.###.",
    ".########.",
    "..........",
]


def test_tags_are_drawn_as_the_official_artwork() -> None:
    official = np.array([[cell == "." for cell in row] for row in OFFICIAL_TAG_0])
    np.testing.assert_array_equal(tag_cells(0), official)
    # OpenCV draws every tag turned 180 degrees from the official artwork.
    for tag_id in range(TAG36H11_COUNT):
        opencv = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, 8, borderBits=1) > 127
        np.testing.assert_array_equal(tag_cells(tag_id, quiet_zone_cells=0), np.rot90(opencv, 2), err_msg=f"{tag_id}")


def test_drawn_tags_decode() -> None:
    detector = NapaDetector()
    for tag_id in range(0, TAG36H11_COUNT, 5):
        image = np.pad(tag_image(tag_id, cell_px=8, quiet_zone_cells=2), 16, constant_values=255)
        assert [observation.tag_id for observation in detector.detect(image)] == [tag_id]


def test_image_layout() -> None:
    image = tag_image(0, cell_px=3, quiet_zone_cells=2)
    assert image.shape == (36, 36) and image.dtype == np.uint8
    assert set(np.unique(image)) == {0, 255}
    assert (image[:6] == 255).all() and (image[6:9, 6:30] == 0).all()
    assert tag_image(0).shape == (10, 10)


@pytest.mark.parametrize("bad", [-1, TAG36H11_COUNT, 1.0, True])
def test_rejects_bad_ids(bad) -> None:
    with pytest.raises(ValueError):
        tag_cells(bad)


def test_svg_prints_the_black_square_at_the_requested_size() -> None:
    svg = tag_svg(7, size_mm=80.0, label=False)
    assert 'width="100.0000mm"' in svg and 'viewBox="0 0 10 10"' in svg
    assert "<text" not in svg
    labelled = tag_svg(7, size_mm=80.0)
    assert 'height="110.0000mm"' in labelled and "ID 7" in labelled
    with pytest.raises(ValueError):
        tag_svg(7, size_mm=0)


def test_png_round_trips() -> None:
    image = tag_image(12, cell_px=4)
    data = png_bytes(image)
    assert data.startswith(b"\x89PNG")
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(decoded, image)
    assert zlib.crc32(data[12:29]) == int.from_bytes(data[29:33], "big")  # IHDR checksum


def test_tag_command_writes_files(tmp_path) -> None:
    assert tag_main(["0-2,7", "2", "--format", "png", "--cell-px", "5", "--output", str(tmp_path)]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "tag36h11-000.png",
        "tag36h11-001.png",
        "tag36h11-002.png",
        "tag36h11-007.png",
    ]
    assert tag_main(["586", "--output", str(tmp_path)]) == 0
    assert (tmp_path / "tag36h11-586.svg").read_text().startswith("<?xml")
    assert tag_main(["587", "--output", str(tmp_path)]) == 2
    with pytest.raises(SystemExit):
        tag_main(["5-2"])


def test_a_printed_tag_in_the_field_yields_the_field_layout_corners() -> None:
    """A tag drawn by napa-tag, mounted as field layouts assume, gives the corners the pose solvers expect."""

    from conftest import example_profile, project_field_tag

    from napa_vision.geometry import transform_matrix

    profile = example_profile()
    camera = profile.camera
    field_to_camera = transform_matrix([0.0, 0.0, 0.5], [0.0, 0.0, 0.0])
    # The tag 2 m ahead, a little left and above the camera, facing it (yawed 180 degrees) and turned slightly.
    field_to_tag = transform_matrix([2.0, 0.3, 0.8], [0.0, 0.0, np.pi + 0.3])
    expected = project_field_tag(
        field_to_camera, field_to_tag, profile.tag_size_m, camera.calibration_matrix, camera.distortion_coefficients
    )
    artwork = tag_image(3, cell_px=40)  # black square from pixel 40 to 360
    source = np.array([[40, 40], [360, 40], [360, 360], [40, 360]], np.float32) - 0.5
    # tag_field_corners lists the official artwork's bottom-right, bottom-left, top-left, top-right.
    destination = expected[[2, 3, 0, 1]].astype(np.float32)
    warp = cv2.getPerspectiveTransform(source, destination)
    image = cv2.warpPerspective(artwork, warp, (1280, 720), flags=cv2.INTER_AREA, borderValue=255)
    [observation] = NapaDetector().detect(image)
    assert observation.tag_id == 3
    np.testing.assert_allclose(observation.corners, expected, atol=0.3)
