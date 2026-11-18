"""External benchmark records describe the detectors and truth inputs actually used."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
from benchmarks import bdcudatag
from benchmarks import cpu_backends as cb
from benchmarks import vkapriltag_corpus as corpus
from napa_vision.napa_detector import NapaDetector, _library


def _assert_napa(config, family, min_tag_px, split_touching, recover_shadows=False):
    core = Path(_library()._name).resolve()
    assert config["core_library_loaded"] == [cb.public_text(str(core))]
    assert config["core_library_sha256"] == [cb.sha256(core)]
    assert config["family"] == family
    assert config["min_tag_px"] == min_tag_px
    assert config["split_touching"] == split_touching
    assert config["recover_shadows"] is recover_shadows
    assert config["max_detections"] == 1024
    assert config["threads"] == 1
    detector = NapaDetector(family=family)
    try:
        assert config["max_hamming"] == detector._params.max_hamming
        assert config["min_contrast"] == detector._params.min_contrast
        assert config["min_border_cells"] == detector._params.min_border_cells
        assert config["sharpening"] == detector._params.sharpening_x100 / 100
    finally:
        detector.close()
    for flag in ("regrow_small_tags", "soft_decode", "align_corners"):
        assert config[flag] is False
    for option in ("camera_gamma", "tag_shape", "shadow_min_tag_px"):
        assert config[option] is None


def _track_napa(monkeypatch, module):
    detectors = []

    def construct(**kwargs):
        detector = NapaDetector(**kwargs)
        detectors.append(detector)
        return detector

    monkeypatch.setattr(module, "NapaDetector", construct)
    return detectors


def test_napa_config_reads_effective_custom_parameters_and_family_stride():
    detector = NapaDetector(
        family="tag16h5",
        min_tag_px=32,
        max_hamming=1,
        min_contrast=31,
        min_border_cells=16,
        sharpening=0.333,
        split_touching=False,
        threads=2,
    )
    try:
        config = cb.napa_config(detector)
    finally:
        detector.close()
    assert config["family"] == "tag16h5"
    assert config["min_tag_px"] == 32
    assert config["max_hamming"] == 1
    assert config["min_contrast"] == 31
    assert config["min_border_cells"] == 16
    assert config["sharpening"] == 0.33
    assert config["split_touching"] is False
    assert config["threads"] == 2
    assert config["max_detections"] == 64
    assert config["base_sampling_stride_px"] == 4  # tag16h5's detected square is six cells wide


def test_bdcudatag_main_records_effective_settings_core_and_selected_inputs(tmp_path, monkeypatch):
    root, output = tmp_path / "inputs", tmp_path / "result.json"
    paths = [root / "scenes" / f"scenes_{resolution}.json" for resolution in bdcudatag.SCENE_RESOLUTIONS]
    paths.append(root / "synthetic-hard" / "data" / "hard" / "metadata.jsonl")
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fixture": index}))
    peer_config = {"library": "test peer", "threads": 1, "quad_decimate": 1.0}
    built = []

    def build(names, threads):
        built.append((names, threads))
        return [cb.Backend("peer", lambda: lambda image: [], peer_config)], {}

    monkeypatch.setattr(cb, "build_backends", build)
    detectors = _track_napa(monkeypatch, bdcudatag)

    def evaluate(root, detect):
        assert detect(np.zeros((64, 64), np.uint8)) == []
        return [{"tags": 0, "found": 0, "wrong_ids": 0}]

    monkeypatch.setattr(bdcudatag, "evaluate_scenes", evaluate)
    monkeypatch.setattr(bdcudatag, "evaluate_hard", evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bdcudatag",
            "--root",
            str(root),
            "--output",
            str(output),
            "--backends",
            "napa",
            "napa16",
            "napa-recall",
            "peer",
        ],
    )
    bdcudatag.main()
    record = json.loads(output.read_text())
    assert record["source_commit"] == bdcudatag.COMMIT
    assert record["repository_commit"] == cb._git_head(cb.ROOT)
    assert record["input_sha256"] == {str(path.relative_to(root)): cb.sha256(path) for path in paths}
    _assert_napa(record["settings"]["napa"], "tag36h11", 22, "auto")
    _assert_napa(record["settings"]["napa16"], "tag36h11", 16, "auto")
    _assert_napa(record["settings"]["napa-recall"], "tag36h11", 16, True, True)
    assert record["settings"]["peer"] == {"family": "tag36h11", **peer_config}
    assert built == [(["peer"], 1)]
    assert len(record["results"]) == 8
    assert len(detectors) == 3
    assert all(not detector._finalizer.alive for detector in detectors)


def test_vulkan_corpus_main_records_actual_family_settings_without_extra_adapters(tmp_path, monkeypatch):
    root, output = tmp_path / "inputs", tmp_path / "result.json"
    paths = [root / name / "gt.json" for name in ("one-tag", corpus.SHORT_FAMILY_SET)]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"frame.png": {"tags": []}}))
    monkeypatch.setattr(corpus, "SCENARIOS", 2)
    monkeypatch.setattr(corpus, "FRAMES", 2)
    monkeypatch.setattr(corpus, "MAIN_TAGS", 0)
    monkeypatch.setattr(corpus.cv2, "imread", lambda *args: np.zeros((64, 64), np.uint8))
    peer_config = {"library": "test peer", "threads": 1, "quad_decimate": 1.0}
    built, workers = [], []

    def build(names, threads):
        built.append((names, threads))
        return [cb.Backend("peer", lambda: lambda image: [], peer_config)], {}

    def vulkan(name, decimation, family):
        def factory():
            workers.append(family)
            return lambda image: []

        return cb.Backend(name, factory, {"library": "test Vulkan", "decimation": decimation})

    monkeypatch.setattr(cb, "build_backends", build)
    monkeypatch.setattr(cb, "vkapriltag_backend", vulkan)
    detectors = _track_napa(monkeypatch, corpus)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vkapriltag_corpus",
            "--root",
            str(root),
            "--output",
            str(output),
            "--backends",
            "napa",
            "napa16",
            "napa-recall",
            "peer",
            "vkapriltag-d1",
        ],
    )
    corpus.main()
    record = json.loads(output.read_text())
    assert record["source_commit"] == corpus.COMMIT
    assert record["repository_commit"] == cb._git_head(cb.ROOT)
    assert record["input_sha256"] == {str(path.relative_to(root)): cb.sha256(path) for path in paths}
    for family in ("tag36h11", "tag16h5"):
        _assert_napa(record["settings"]["napa"][family], family, 22, "auto")
        _assert_napa(record["settings"]["napa16"][family], family, 16, "auto")
        _assert_napa(record["settings"]["napa-recall"][family], family, 16, True, True)
        assert record["settings"]["vkapriltag-d1"][family] == {
            "family": family,
            "library": "test Vulkan",
            "decimation": 1,
        }
    assert record["settings"]["peer"] == {"tag36h11": {"family": "tag36h11", **peer_config}}
    assert list(record["results"]["peer"]) == ["one-tag"]
    assert sorted(workers) == ["tag16h5", "tag36h11"]
    assert built == [(["peer"], 1)]
    assert len(detectors) == 6
    assert all(not detector._finalizer.alive for detector in detectors)


def test_peer_only_corpus_does_not_load_napa_for_provenance(tmp_path, monkeypatch):
    from napa_vision import napa_detector

    path = tmp_path / "one-tag" / "gt.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"frame.png": {"tags": []}}))
    monkeypatch.setattr(corpus, "SCENARIOS", 1)
    monkeypatch.setattr(corpus, "FRAMES", 1)
    monkeypatch.setattr(corpus, "MAIN_TAGS", 0)
    monkeypatch.setattr(corpus.cv2, "imread", lambda *args: np.zeros((64, 64), np.uint8))

    def unexpected_load():
        raise AssertionError("a peer-only run must not load the NAPA core")

    monkeypatch.setattr(napa_detector, "_library", unexpected_load)
    monkeypatch.setattr(
        cb, "build_backends", lambda *args: ([cb.Backend("peer", lambda: lambda image: [], {"threads": 1})], {})
    )
    output = tmp_path / "out.json"
    monkeypatch.setattr(
        sys, "argv", ["vkapriltag_corpus", "--root", str(tmp_path), "--backends", "peer", "--output", str(output)]
    )
    corpus.main()
    assert json.loads(output.read_text())["settings"] == {"peer": {"tag36h11": {"family": "tag36h11", "threads": 1}}}


@pytest.mark.parametrize("module", [bdcudatag, corpus])
def test_napa_is_closed_when_evaluation_fails(tmp_path, monkeypatch, module):
    detectors = _track_napa(monkeypatch, module)
    if module is bdcudatag:
        path = tmp_path / "synthetic-hard" / "data" / "hard" / "metadata.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")

        def fail(root, detect):
            raise ValueError("bad input")

        monkeypatch.setattr(module, "evaluate_hard", fail)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "bdcudatag",
                "--root",
                str(tmp_path),
                "--sets",
                "hard",
                "--backends",
                "napa",
                "--output",
                str(tmp_path / "out.json"),
            ],
        )
        with pytest.raises(ValueError, match="bad input"):
            module.main()
    else:
        path = tmp_path / "one-tag" / "gt.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"missing.png": {"tags": []}}))
        monkeypatch.setattr(module, "SCENARIOS", 1)
        monkeypatch.setattr(module.cv2, "imread", lambda *args: None)
        with pytest.raises(FileNotFoundError):
            module.evaluate(tmp_path, "napa")
    assert len(detectors) == 1
    assert not detectors[0]._finalizer.alive
