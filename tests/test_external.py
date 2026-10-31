"""Independent data scoring must not hide false positives or spatial errors."""

import numpy as np
import pytest

pytest.importorskip("cv2")
from benchmarks.external import corner_distances, score, summarize


def tag(tag_id=1, complete=True, x=0):
    return {
        "tag_id": tag_id,
        "eval_complete": complete,
        "corners": np.array([[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]], dtype=float),
    }


def test_wrong_location_and_id_do_not_count_as_recall():
    truth = tag()
    result = score([truth], [(1, tag(x=100)["corners"]), (2, truth["corners"])])
    assert (result["tp"], result["fp"], result["fn"]) == (0, 2, 1)


def test_duplicates_are_false_positives():
    t = tag()
    result = score([t], [(1, t["corners"]), (1, t["corners"])])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)


def test_partial_truth_is_ignored_only_at_matching_location():
    t = tag(complete=False)
    result = score([t], [(1, t["corners"]), (1, tag(x=100)["corners"])])
    assert (result["tp"], result["fp"], result["ignored"], result["expected"]) == (0, 1, 1, 0)


def test_repeated_ids_at_distinct_locations_match_separately():
    a, b = tag(), tag(x=100)
    result = score([a, b], [(1, b["corners"]), (1, a["corners"])])
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)


def test_complete_truth_has_priority_over_partial():
    a, b = tag(complete=False), tag()
    result = score([a, b], [(1, b["corners"])])
    assert (result["tp"], result["ignored"]) == (1, 0)


def test_alignment_and_empty_metrics():
    a = tag()["corners"]
    assert np.all(corner_distances(np.roll(a[::-1], 1, axis=0), a) == 0)
    row = {**score([], []), "times_ms": [1.0]}
    assert summarize([row])["precision"] is None
    assert summarize([row])["recall"] is None
    assert summarize([row])["corner_px_p50_p95"] is None


def test_icra_loader_normalizes_origin_and_rejects_missing_images(tmp_path):
    import cv2

    from benchmarks.external import icra_frames

    root = tmp_path / "forward"
    root.mkdir()
    (root / "tags.csv").write_text(
        "image,tag_id,corner,ground_truth_x,ground_truth_y,corner_visible,tag_fully_visible\n"
        + "".join(f"0000.png,1,{k},{x + 0.5},{y + 0.5},1,1\n" for k, (x, y) in enumerate(tag()["corners"]))
    )
    for kind in ("pure_tags_images", "checkerboard_corners_images"):
        (root / kind).mkdir()
        cv2.imwrite(str(root / kind / "0000.png"), np.zeros((16, 16), np.uint8))
    rows = list(icra_frames(tmp_path, "icra-forward"))
    np.testing.assert_array_equal(rows[0][3][0]["corners"], tag()["corners"])
    (root / "pure_tags_images" / "0000.png").unlink()
    with pytest.raises(ValueError, match="incomplete ICRA"):
        list(icra_frames(tmp_path, "icra-forward"))


def test_corner_accuracy_counts_misses_in_denominator():
    t, missing = tag(), tag(2, x=100)
    row = {**score([t, missing], [(1, t["corners"])]), "times_ms": [1.0]}
    assert summarize([row])["corner_accuracy_at_1px"] == 0.5


def test_locus_loader_checks_truth_integrity_and_pixel_origin(tmp_path):
    import json

    import cv2

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from benchmarks.external import FILES, digest, frames

    config = "locus_v1_tag36h11_640x480"
    root = tmp_path / config
    root.mkdir()
    image_bytes = cv2.imencode(".png", np.zeros((16, 16), np.uint8))[1].tobytes()
    rows, records, hashes = [], [], {}
    for i in range(50):
        name = f"scene_{i:04d}_cam_0000"
        rows.append({"image_id": name, "image": {"bytes": image_bytes}})
        records.append(
            {
                **tag(),
                "corners": (tag()["corners"] + 0.5).tolist(),
                "record_type": "TAG",
                "image_id": name,
                "corners_visibility": [2] * 4,
            }
        )
        hashes[f"images/{name}.png"] = digest(image_bytes)
    pq.write_table(pa.Table.from_pylist(rows), root / FILES[0])
    (root / "rich_truth.json").write_text(json.dumps({"records": records}))
    hashes["rich_truth.json"] = digest((root / "rich_truth.json").read_bytes())
    (root / "checksums.json").write_text(json.dumps({"files": hashes}))
    loaded = list(frames(tmp_path, config))
    assert len(loaded) == 50
    np.testing.assert_array_equal(loaded[0][3][0]["corners"], tag()["corners"])
    (root / "rich_truth.json").write_text(json.dumps({"records": []}))
    with pytest.raises(ValueError, match="truth checksum"):
        list(frames(tmp_path, config))
