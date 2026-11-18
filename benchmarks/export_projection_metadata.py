"""Backfill per-tag cameras in a local copy of the published NAPA dataset metadata.

No PNGs are written. The frozen renderer is observed without changing its random
stream; every replay must match the published full pixel hash and exact corners.
Run from a NAPA checkout with the frozen renderer and original NumPy/OpenCV versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from benchmarks import cpu_comparison as cc  # noqa: E402
from benchmarks import performance_matrix as pm  # noqa: E402

SQUARE = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
SPLITS = ("main", "stress", "range")
TOLERANCE_PX = 1e-9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def capture_poses():
    """Observe each attempted placement, including rejected attempts and decoys.

    Copy the generator before calling the original function, then read its three
    pose draws from the copy. The original implementation still produces pixels.
    This deliberately targets the hash-pinned renderer, not arbitrary renderers.
    """
    poses = []
    original = pm.tag_pose

    def observe(rng, side, center, *, max_tilt_deg=20.0, min_tilt_deg=0.0):
        replay = deepcopy(rng)
        corners = original(rng, side, center, max_tilt_deg=max_tilt_deg, min_tilt_deg=min_tilt_deg)
        roll = replay.uniform(0, 2 * math.pi)
        tilt = math.radians(replay.uniform(min_tilt_deg, max_tilt_deg))
        axis = replay.uniform(0, 2 * math.pi)
        roll_matrix = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
        k = np.array([math.cos(axis), math.sin(axis), 0])
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        rotation = (np.eye(3) + math.sin(tilt) * kx + (1 - math.cos(tilt)) * kx @ kx) @ roll_matrix
        poses.append((corners.copy(), float(side), np.asarray(center), rotation))
        return corners

    with patch.object(pm, "tag_pose", observe):
        yield poses


def projection_for_tag(corners: np.ndarray, poses: list) -> dict:
    """Match a retained tag to one attempted pose, allowing final edge translation."""
    matches = []
    for original, side, center, rotation in poses:
        shift = (corners - original).mean(axis=0)
        if np.max(np.abs(corners - original - shift)) <= TOLERANCE_PX:
            matches.append((side, center + shift, rotation))
    if len(matches) != 1:
        raise ValueError(f"expected one sampled pose for tag, found {len(matches)}")
    side, center, rotation = matches[0]
    camera = np.array([[2.5 * side, 0, center[0]], [0, 2.5 * side, center[1]], [0, 0, 1]])
    translation = np.array([0.0, 0.0, 2.5])
    projected = (SQUARE @ rotation.T + translation) @ camera.T
    predicted = projected[:, :2] / projected[:, 2:]
    error = float(np.max(np.abs(predicted - corners)))
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0) or error > TOLERANCE_PX:
        raise ValueError(f"sampled pinhole pose does not reproduce corners: {error} px")
    homography = camera @ np.column_stack((rotation[:, :2], translation))
    homography /= homography[2, 2]
    return {
        "model": "pinhole",
        "camera_matrix": camera.tolist(),
        "distortion_coefficients": [0.0] * 5,
        "rotation_matrix": rotation.tolist(),
        "translation_tag_widths": translation.tolist(),
        "homography": homography.tolist(),
    }


def annotate(row: dict, frame: pm.Frame, poses: list) -> tuple[dict, float]:
    if hashlib.sha256(frame.image.tobytes()).hexdigest() != row["image_sha256"]:
        raise ValueError(f"published pixel hash mismatch: {frame.name}")
    ids = [tag.tag_id for tag in frame.tags]
    corners = [tag.corners.tolist() for tag in frame.tags]
    if (
        row["frame_id"] != frame.name
        or row["tag_ids"] != ids
        or row["corners"] != corners
        or row["num_tags"] != len(ids)
        or (row["width"], row["height"]) != frame.resolution
    ):
        raise ValueError(f"published annotation mismatch: {frame.name}")
    projections = [projection_for_tag(tag.corners, poses) for tag in frame.tags]
    worst = 0.0
    for tag, projection in zip(frame.tags, projections, strict=True):
        h = np.asarray(projection["homography"])
        q = np.column_stack((SQUARE[:, :2], np.ones(4))) @ h.T
        error = float(np.max(np.abs(q[:, :2] / q[:, 2:] - tag.corners)))
        if error > TOLERANCE_PX:
            raise ValueError(f"homography does not reproduce corners: {frame.name}")
        worst = max(worst, error)
    return {**row, "projections": projections}, worst


def export(existing: Path, output: Path, source_revision: str) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads((existing / "manifest.json").read_text())
    for name, expected in manifest["renderer_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"frozen renderer source mismatch: {name}")
    environment = manifest["export_environment"]
    if (np.__version__, cv2.__version__) != (environment["numpy"], environment["opencv"]):
        raise ValueError("NumPy/OpenCV versions differ from the original export")
    # Validate original metadata file checksums before trusting its full pixel hashes.
    for split in SPLITS:
        name = f"data/{split}/metadata.jsonl"
        if sha256(existing / name) != manifest["files_sha256"][name]:
            raise ValueError(f"published metadata checksum mismatch: {name}")
    output.mkdir(parents=True)
    counts, tags = {}, {}
    max_error = 0.0
    seen = set()
    for split in SPLITS:
        relative = Path(f"data/{split}/metadata.jsonl")
        destination = output / relative
        destination.parent.mkdir(parents=True)
        counts[split] = tags[split] = 0
        with (existing / relative).open() as source, destination.open("w") as target:
            for line in source:
                row = json.loads(line)
                if row["frame_id"] in seen:
                    raise ValueError(f"duplicate frame: {row['frame_id']}")
                seen.add(row["frame_id"])
                with capture_poses() as poses:
                    frame = cc.frame_by_name(row["frame_id"], seed=row["seed"], range_cases=row["range_cases"])
                enriched, error = annotate(row, frame, poses)
                target.write(json.dumps(enriched, separators=(",", ":"), allow_nan=False) + "\n")
                counts[split] += 1
                tags[split] += len(frame.tags)
                max_error = max(max_error, error)
                if len(seen) % 100 == 0:
                    print(f"Verified {len(seen)} frames", flush=True)
        manifest["files_sha256"][relative.as_posix()] = sha256(destination)
    if counts != manifest["counts"] or tags != manifest["tags"]:
        raise ValueError("replay counts differ from published corpus")
    reproduction = output / "reproduction" / Path(__file__).name
    reproduction.parent.mkdir()
    shutil.copyfile(__file__, reproduction)
    manifest["files_sha256"][reproduction.relative_to(output).as_posix()] = sha256(reproduction)
    manifest["version"] = "1.1.0"
    manifest["projection_metadata"] = {
        "version": 1,
        "source_dataset_revision": source_revision,
        "frames_verified_full_sha256": len(seen),
        "tags_verified": sum(tags.values()),
        "max_homography_coordinate_error_px": max_error,
        "reprojection_tolerance_px": TOLERANCE_PX,
        "camera_scope": "per-tag; no shared image camera",
        "translation_units": "black-square tag widths",
        "unchanged_image_files": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["projection_metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True, help="original manifest and metadata directory")
    parser.add_argument("--output", type=Path, required=True, help="new directory for metadata-only update")
    parser.add_argument("--source-revision", required=True, help="immutable Hub commit of original metadata")
    args = parser.parse_args()
    print(json.dumps(export(args.existing, args.output, args.source_revision), indent=2))


if __name__ == "__main__":
    main()
