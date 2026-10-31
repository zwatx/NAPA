"""Run the examples as users do, against real image files and the native detector."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from napa_vision import ArtifactProvenance, CameraCalibration, ProvenanceKind, save_calibration_data
from napa_vision.tags import tag_image

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "python"


def run_example(name: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name), *map(str, args)], capture_output=True, text=True, timeout=30
    )


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    image = np.full((480, 640), 255, np.uint8)
    image[120:360, 200:440] = tag_image(7, cell_px=24)
    path = tmp_path / "tag.png"
    assert cv2.imwrite(str(path), image)
    return path


def test_quickstart_without_opencv() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; sys.modules['cv2'] = None; runpy.run_path(sys.argv[1], run_name='__main__')",
            str(EXAMPLES / "quickstart.py"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "ID 7, hamming=0" in result.stdout


def test_image_json_and_annotation(photo: Path, tmp_path: Path) -> None:
    output = tmp_path / "annotated.png"
    result = run_example("detect_image.py", photo, "--output", output)
    assert result.returncode == 0, result.stderr
    tags = json.loads(result.stdout)
    assert [tag["tag_id"] for tag in tags] == [7]
    assert np.asarray(tags[0]["corners_px"]).shape == (4, 2)
    assert not np.array_equal(cv2.imread(str(output)), cv2.imread(str(photo)))


def test_image_no_tags_and_errors(tmp_path: Path, photo: Path) -> None:
    blank = tmp_path / "blank.png"
    assert cv2.imwrite(str(blank), np.full((100, 100), 255, np.uint8))
    result = run_example("detect_image.py", blank)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []
    missing = run_example("detect_image.py", tmp_path / "missing.png")
    assert missing.returncode != 0 and "could not read image" in missing.stderr
    unwritable = run_example("detect_image.py", photo, "--output", tmp_path / "absent" / "out.png")
    assert unwritable.returncode != 0 and "could not write image" in unwritable.stderr


def test_recorded_video_json_and_frame_limit(photo: Path, tmp_path: Path) -> None:
    video = tmp_path / "tags.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build has no MJPEG video writer")
    try:
        writer.write(cv2.imread(str(photo)))
        writer.write(np.full((480, 640, 3), 255, np.uint8))
    finally:
        writer.release()
    result = run_example("detect_video.py", "--video", video)
    assert result.returncode == 0, result.stderr
    frames = [json.loads(line) for line in result.stdout.splitlines()]
    assert [frame["frame_index"] for frame in frames] == [0, 1]
    assert [tag["tag_id"] for tag in frames[0]["tags"]] == [7]
    assert frames[1]["tags"] == []
    limited = run_example("detect_video.py", "--video", video, "--max-frames", 1)
    assert limited.returncode == 0, limited.stderr
    assert len(limited.stdout.splitlines()) == 1
    invalid = run_example("detect_video.py", "--video", video, "--max-frames", 0)
    assert invalid.returncode != 0 and "--max-frames must be positive" in invalid.stderr
    missing = run_example("detect_video.py", "--video", tmp_path / "missing.avi")
    assert missing.returncode != 0 and "could not open capture source" in missing.stderr


def test_pose_candidates_and_resolution_check(photo: Path, tmp_path: Path) -> None:
    # Synthetic intrinsics for this test only; the public example supplies no pretend camera calibration.
    calibration = CameraCalibration(
        camera_matrix=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]]),
        distortion_coefficients=np.zeros(5),
        image_size=(640, 480),
        provenance=ArtifactProvenance(ProvenanceKind.MEASURED, "synthetic example test fixture"),
        rms_reprojection_error_px=0.0,
        views_used=8,
    )
    path = tmp_path / "camera.json"
    save_calibration_data(path, calibration)
    args = ("--calibration", path, "--tag-size-m", 0.1651)
    result = run_example("estimate_pose.py", photo, *args)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["unsolved_tag_ids"] == []
    assert [pose["tag_id"] for pose in data["poses"]] == [7]
    solutions = data["poses"][0]["solutions"]
    assert len(solutions) == 2
    assert solutions[0]["translation_m"][0] == pytest.approx(500 * 0.1651 / 192, abs=0.01)
    assert np.asarray(solutions[0]["covariance"]).shape == (6, 6)
    small = tmp_path / "small.png"
    assert cv2.imwrite(str(small), np.zeros((100, 100), np.uint8))
    mismatch = run_example("estimate_pose.py", small, *args)
    assert mismatch.returncode != 0 and "not 100x100" in mismatch.stderr
    invalid = run_example("estimate_pose.py", photo, "--calibration", path, "--tag-size-m", -1)
    assert invalid.returncode != 0 and "tag_size_m must be finite and greater than zero" in invalid.stderr
