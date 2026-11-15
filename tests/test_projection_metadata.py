from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import cpu_comparison as cc  # noqa: E402
from benchmarks import export_projection_metadata as export  # noqa: E402
from benchmarks import performance_matrix as pm  # noqa: E402


@pytest.mark.parametrize(
    "name",
    [
        "many-tags-640x480-003",
        "frame-edge-640x480-000",
        "range-640x480-005",
        "stress-dense-640x480-000",
        "decoys-640x480-042",
    ],
)
def test_replay_preserves_pixels_randomness_and_projects_canonical_corners(name):
    baseline = cc.frame_by_name(name, seed=2468, range_cases=50)
    with export.capture_poses() as poses:
        frame = cc.frame_by_name(name, seed=2468, range_cases=50)
    assert np.array_equal(baseline.image, frame.image)
    assert [tag.tag_id for tag in baseline.tags] == [tag.tag_id for tag in frame.tags]
    for tag, original in zip(frame.tags, baseline.tags, strict=True):
        assert np.array_equal(tag.corners, original.corners)
        p = export.projection_for_tag(tag.corners, poses)
        # Independent OpenCV projection of the recorded camera and rigid pose.
        import cv2

        rvec, _ = cv2.Rodrigues(np.asarray(p["rotation_matrix"]))
        points, _ = cv2.projectPoints(
            export.SQUARE,
            rvec,
            np.asarray(p["translation_tag_widths"]),
            np.asarray(p["camera_matrix"]),
            np.asarray(p["distortion_coefficients"]),
        )
        np.testing.assert_allclose(points[:, 0], tag.corners, rtol=0, atol=1e-9)
    assert pm.tag_pose.__name__ == "tag_pose"


def test_capture_restores_renderer_after_failure():
    original = pm.tag_pose
    with pytest.raises(RuntimeError), export.capture_poses():
        raise RuntimeError("interrupted render")
    assert pm.tag_pose is original


def test_missing_or_ambiguous_pose_fails_closed():
    with export.capture_poses() as poses:
        corners = pm.tag_pose(np.random.default_rng(1), 100, (200, 150))
    with pytest.raises(ValueError, match="found 0"):
        export.projection_for_tag(corners, [])
    with pytest.raises(ValueError, match="found 2"):
        export.projection_for_tag(corners, poses * 2)
    altered = corners.copy()
    altered[0, 0] += 0.1
    with pytest.raises(ValueError, match="found 0"):
        export.projection_for_tag(altered, poses)


def test_corrupt_hash_or_truth_is_rejected():
    with export.capture_poses() as poses:
        frame = cc.frame_by_name("one-tag-640x480-000")
    row = dict(
        frame_id=frame.name,
        image_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
        tag_ids=[tag.tag_id for tag in frame.tags],
        corners=[tag.corners.tolist() for tag in frame.tags],
        num_tags=len(frame.tags),
        width=640,
        height=480,
    )
    enriched, _ = export.annotate(row, frame, poses)
    assert len(enriched["projections"]) == len(frame.tags)
    assert all(enriched[key] == value for key, value in row.items())
    with pytest.raises(ValueError, match="pixel hash mismatch"):
        export.annotate({**row, "image_sha256": "0" * 64}, frame, poses)
    with pytest.raises(ValueError, match="annotation mismatch"):
        export.annotate({**row, "tag_ids": []}, frame, poses)


def test_export_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        export.export(tmp_path / "missing", tmp_path, "revision")


@pytest.mark.parametrize("failure", ["renderer", "metadata", "counts"])
def test_export_failure_leaves_no_completion_manifest(tmp_path, failure):
    import json

    import cv2

    existing, output = tmp_path / "existing", tmp_path / "output"
    existing.mkdir()
    frame = cc.frame_by_name("empty-640x480-000")
    row = dict(
        frame_id=frame.name,
        image_sha256=hashlib.sha256(frame.image.tobytes()).hexdigest(),
        tag_ids=[],
        corners=[],
        num_tags=0,
        width=640,
        height=480,
        seed=2468,
        range_cases=50,
    )
    files = {}
    for split in export.SPLITS:
        relative = f"data/{split}/metadata.jsonl"
        path = existing / relative
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row) + "\n" if split == "main" else "")
        files[relative] = export.sha256(path)
    manifest = dict(
        renderer_source_sha256={},
        export_environment=dict(numpy=np.__version__, opencv=cv2.__version__),
        files_sha256=files,
        counts=dict(main=2, stress=0, range=0),
        tags=dict(main=0, stress=0, range=0),
    )
    if failure == "renderer":
        manifest["renderer_source_sha256"]["benchmarks/performance_matrix.py"] = "bad"
    elif failure == "metadata":
        manifest["files_sha256"]["data/main/metadata.jsonl"] = "bad"
    (existing / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(
        ValueError,
        match={"renderer": "source mismatch", "metadata": "checksum mismatch", "counts": "counts differ"}[failure],
    ):
        export.export(existing, output, "revision")
    assert not (output / "manifest.json").exists()
