from __future__ import annotations

import io
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmarks.cpu_backends as cb  # noqa: E402
import benchmarks.cpu_comparison as cc  # noqa: E402
import benchmarks.performance_matrix as matrix  # noqa: E402

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def test_scoring_separates_false_ids_duplicates_geometry_and_order() -> None:
    truth = [matrix.TagTruth(1, SQUARE), matrix.TagTruth(2, SQUARE + 20)]
    reflected = np.roll(SQUARE[::-1], 1, axis=0) + 0.3  # same square, another corner order, shifted 0.3 px
    score = cc.score_detail(truth, [(1, reflected), (1, SQUARE), (99, SQUARE)])

    assert (score["true_positive"], score["false_id"], score["duplicate"], score["false_negative"]) == (1, 1, 1, 1)
    match = score["matched"][1]
    assert match["worst_px"] == pytest.approx(0.3 * np.sqrt(2))
    assert match["alignment"] == "rs1" and match["offset"] == pytest.approx([0.3, 0.3])
    assert cc.score_detail(truth, [(2, SQUARE + 20)])["matched"][2]["alignment"] == "s0"


def test_adapter_attribute_errors_are_not_reported_as_missing_libraries(monkeypatch):
    def broken():
        raise AttributeError("adapter configuration is broken")

    monkeypatch.setattr(cb, "registry", lambda threads: {"napa": broken})
    with pytest.raises(AttributeError, match="adapter configuration"):
        cb.build_backends(["napa"], 1)


def test_opencv_without_aruco_is_an_explicit_optional_skip(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(setNumThreads=lambda threads: None))
    backends, skipped = cb.build_backends(["opencv"], 1)
    assert backends == []
    assert "cv2.aruco" in skipped["opencv"]


def test_source_archives_do_not_inherit_an_unrelated_parent_git_revision(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("revision provenance check requires git")
    repo = tmp_path / "repo"
    subprocess.run([git, "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(
        [
            git,
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            f"core.hooksPath={tmp_path / 'no-hooks'}",
            "commit",
            "--allow-empty",
            "-qm",
            "test: record fixture",
        ],
        check=True,
        capture_output=True,
    )
    assert cb._git_head(repo)
    archive = repo / "work/source-archive"
    archive.mkdir(parents=True)
    assert cb._git_head(archive) is None


def test_napa_adapters_honor_the_requested_native_thread_count(monkeypatch):
    from napa_vision import napa_detector

    constructor = napa_detector.NapaDetector
    created = []

    def detector(**options):
        instance = constructor(**options)
        created.append(instance)
        return instance

    monkeypatch.setattr(napa_detector, "NapaDetector", detector)
    try:
        backends, skipped = cb.build_backends(["napa", "napa-mtp16"], threads=4)
        assert skipped == {}
        for backend in backends:
            assert backend.config["threads"] == 4
            assert backend.factory()(np.zeros((16, 16), np.uint8)) == []
        assert len(created) == 4 and all(instance.threads == 4 for instance in created)
    finally:
        for instance in created:
            instance.close()


@pytest.mark.skipif(sys.platform != "linux", reason="loader-name regression uses Linux LD_LIBRARY_PATH")
@pytest.mark.parametrize("shadow", [False, True])
def test_napa_provenance_identifies_the_mapped_library_not_a_same_named_cwd_file(tmp_path, shadow):
    from napa_vision.napa_detector import _library

    libraries, cwd = tmp_path / "libraries", tmp_path / "cwd"
    libraries.mkdir()
    cwd.mkdir()
    loaded = libraries / "benchmark-core.so"
    shutil.copyfile(Path(_library()._name).resolve(), loaded)
    if shadow:
        (cwd / loaded.name).write_bytes(b"an unrelated file")
    env = os.environ.copy()
    env["NAPA_CORE_LIBRARY"] = loaded.name
    env["LD_LIBRARY_PATH"] = str(libraries) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(map(str, (root, root / "src")))
    script = """
import sys
from pathlib import Path
from benchmarks import cpu_backends as cb
from napa_vision import NapaDetector
detector = NapaDetector()
try:
    config = cb.napa_config(detector)
    expected = Path(sys.argv[1])
    assert config['core_library_loaded'] == [cb.public_text(str(expected))]
    assert config['core_library_sha256'] == [cb.sha256(expected)]
finally:
    detector.close()
"""
    subprocess.run([sys.executable, "-c", script, str(loaded)], cwd=cwd, env=env, check=True, capture_output=True)


@pytest.mark.parametrize(
    "payload,message",
    [(b"", "count"), (struct.pack("<i", -1), "negative"), (struct.pack("<i", 1) + bytes(40), "truncated")],
)
def test_vulkan_worker_failures_are_reported_as_protocol_errors(tmp_path, monkeypatch, payload, message):
    (tmp_path / "vkapriltag_worker").touch()
    monkeypatch.setattr(cb, "SHIM_DIR", tmp_path)

    class Worker:
        def __init__(self, *args, **kwargs):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b"ready\n" + payload)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(cb.subprocess, "Popen", Worker)
    detect = cb.vkapriltag_backend("vkapriltag-d1", 1).factory()
    with pytest.raises(RuntimeError, match=message):
        detect(np.zeros((16, 16), np.uint8))


@pytest.mark.parametrize("module", ["benchmarks.cpu_backends", "benchmarks.cpu_comparison"])
@pytest.mark.parametrize("preset", [None, "3"])
def test_openblas_is_configured_before_numpy_import(module, preset):
    script = """
import builtins
import importlib
import os
import sys
original = builtins.__import__
observed = []
def checked(name, *args, **kwargs):
    if name == "numpy" and "numpy" not in sys.modules:
        assert os.environ.get("OPENBLAS_NUM_THREADS") == sys.argv[2]
        observed.append(name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
importlib.import_module(sys.argv[1])
assert observed
"""
    env = os.environ.copy()
    env.pop("OPENBLAS_NUM_THREADS", None)
    if preset:
        env["OPENBLAS_NUM_THREADS"] = preset
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(map(str, (root, root / "src")))
    subprocess.run([sys.executable, "-c", script, module, preset or "1"], env=env, check=True, capture_output=True)


def _real_row(path):
    return {
        "frame": "real-unannotated-32x24-000",
        "scenario": cc.REAL,
        "resolution": "32x24",
        "backend": "napa",
        "tags": 0,
        "true_positive": 0,
        "tp_ids": "",
        "corner_errors_px": "",
        "alignments": "",
        "detected_ids": "1",
        "stable_across_repetitions": True,
        "median_ms": 1.0,
        "samples_ns": "1000000",
        "cpu_ns": "1000000",
        "detail_path": str(path),
        "detections_json": json.dumps([[1, SQUARE.tolist()]]),
    }


def test_real_report_shows_outputs_without_scoring_unlabelled_frames():
    report = {"backends": {"napa": {}}, **cc.aggregate([_real_row("image.png")])}
    text = cc.markdown_report(report, {}, [cc.REAL])
    assert "Outputs" in text and "1.000" in text
    assert "Recall" not in text and "Corner err" not in text


@pytest.mark.parametrize("relative", [False, True])
def test_real_agreement_resolves_recorded_paths_from_any_working_directory(tmp_path, monkeypatch, relative):
    root = tmp_path / "repo"
    image = root / "frames/image.png"
    image.parent.mkdir(parents=True)
    coords = " ".join(f"({x + 0.5} {y + 0.5})" for x, y in SQUARE)
    image.with_suffix(".txt").write_text(f"1, {coords}\n")
    monkeypatch.setattr(cc, "ROOT", root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    recorded = image.relative_to(root) if relative else image
    result = cc.real_agreement([_real_row(recorded)])["backends"]["napa"]
    assert result["matched"] == result["reference_tags"] == 1
    assert result["median_worst_corner_vs_reference_px"] == 0


def test_real_frame_can_be_selected_without_reloading_other_inputs(tmp_path):
    import cv2

    image = np.arange(17 * 23, dtype=np.uint8).reshape(17, 23)
    path = tmp_path / "image.png"
    assert cv2.imwrite(str(path), image)
    frame = cc.frame_by_name("real-unannotated-23x17-001", real_images=[tmp_path / "unused.png", path])
    assert frame.name == "real-unannotated-23x17-001"
    np.testing.assert_array_equal(frame.image, image)
    with pytest.raises(ValueError, match="real-images"):
        cc.frame_by_name("real-unannotated-23x17-001")


def test_stress_frames_are_reproducible_and_keep_their_bands() -> None:
    tiny = [cc.render_stress("stress-tiny", i, (640, 480)) for i in range(4)]
    assert all(12 * 0.95 < t.side_px < 31 * 1.02 for f in tiny for t in f.tags)
    assert np.array_equal(tiny[2].image, cc.render_stress("stress-tiny", 2, (640, 480)).image)
    assert len(cc.render_stress("stress-dense", 0, (1280, 720)).tags) >= 20
    assert cc.render_stress("stress-clutter-noisy", 0, (640, 480)).tags == []
    assert cc.render_stress("stress-heavy-noise", 0, (640, 480)).details["noise_sigma"] >= 15


def test_validation_truth_follows_a_whole_frame_rotation() -> None:
    frames = cc.validation_frames()
    turned = next(f for f in frames if f.details.get("rot90"))
    found = cb.napa_backend("napa").factory()(turned.image)
    score = cc.score_detail(turned.tags, found)
    assert score["true_positive"] == 1 and score["matched"][586]["alignment"] == "s0"
    assert score["matched"][586]["worst_px"] < 0.2


def test_frame_ratios_compare_the_same_frame() -> None:
    rows = [
        {"frame": "a", "backend": "napa", "scenario": "one-tag", "resolution": "640x480", "median_ms": 1.0},
        {"frame": "a", "backend": "other", "scenario": "one-tag", "resolution": "640x480", "median_ms": 4.0},
        {"frame": "b", "backend": "napa", "scenario": "one-tag", "resolution": "640x480", "median_ms": 2.0},
        {"frame": "b", "backend": "other", "scenario": "one-tag", "resolution": "640x480", "median_ms": 1.0},
    ]
    (entry,) = cc.frame_ratios(rows, "napa", lambda row: "g")
    assert (entry["min"], entry["max"], entry["frames_baseline_slower"]) == (0.5, 4.0, 1)


@pytest.mark.parametrize("name", list(cb.TUNING))
def test_adapter_conventions_on_known_geometry(name: str) -> None:
    """Every backend that can be built: right IDs, pixel origin within 0.15 px, one corner order, no blanks.

    Recall is the benchmark's job, not this test's: RapidTag 0.1.12 and OpenCV 4.7 miss the noise-free,
    near-axis-aligned 120 px tags here (they find them once sensor noise is added), so only most must be found.
    """

    backends, skipped = cb.build_backends([name], threads=1)
    if not backends:
        pytest.skip(skipped[name])
    entry = cc.validate(backends)[name]
    tagged = [case for case in entry["cases"] if case["expected"]]
    found = sum(case["true_positive"] for case in tagged)
    assert found > len(tagged) // 2
    assert entry["false_ids"] == entry["duplicates"] == entry["blank_detections"] == 0
    assert np.all(np.abs(entry["mean_offset_px"]) < 0.15)
    assert entry["order_consistent"]


def test_resource_summary_keeps_skipped_backends_apart_from_measured_ones() -> None:
    run = {
        "backend": "napa",
        "round": 0,
        "skipped": {},
        "config": {},
        "mode": "serial",
        "workers": 1,
        "threads_per_detector": 1,
        "cpus": [2],
        "wall_ms_per_frame": 1.0,
        "cpu_ms_per_frame": 1.0,
        "frames_per_s": 1000.0,
        "mean_busy_cpus": 1.0,
        "latency_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0},
        "setup": {"wall_ms": 1.0},
        "frame_median_ms": [1.0],
        "threads_end": [{"cpus": [2]}],
        "rss_kib": {"before_backend": 0, "after_construct": 1024, "after_warmup": 2048, "end": 2048, "peak": 2048},
    }
    summary = cc.resources_summary({"results": [run, {"backend": "locus", "round": 0, "skip_reason": "no locus"}]})
    measured, skipped = summary
    assert measured["wall_ms_per_frame"] == 1.0 and measured["rss_peak_over_constructed_mib"] == 1.0
    assert skipped == {"backend": "locus", "failures": [], "skip_reason": ["no locus"]}
    assert "skipped: no locus" in cc.resources_markdown(summary, "t")


def test_published_tables_report_recall_wrong_ids_speedups_and_range() -> None:
    def row(frame, scenario, backend, ms, found, tags=2, false_id=0, side=""):
        return {
            "frame": frame,
            "scenario": scenario,
            "resolution": "640x480",
            "backend": backend,
            "tags": tags,
            "true_positive": found,
            "false_id": false_id,
            "duplicate": 0,
            "median_ms": ms,
            "samples_ns": str(int(ms * 1e6)),
            "corner_errors_px": ";".join(["0.02"] * found),
            "detail_side_px": side,
        }

    rows = [
        row("a", "one-tag", "napa", 1.0, 2),
        row("a", "one-tag", "locus", 4.0, 1, false_id=1),
        row("b", "noise", "napa", 2.0, 2),
        row("b", "noise", "locus", 3.0, 2),
        row("r0", "range", "napa", 1.0, 0, 1, side="12.0"),
        row("r0", "range", "locus", 1.0, 1, 1, side="12.0"),
        row("r1", "range", "napa", 1.0, 1, 1, side="16.0"),
        row("r1", "range", "locus", 1.0, 1, 1, side="16.0"),
    ]
    text = cc.published_tables(rows)
    assert "| Tags found | 75.0% | 100.0% |" in text  # Locus first (label order), NAPA last
    assert "| Smallest tag found every time (range test) | 12 px | 16 px |" in text
    assert "| Wrong IDs in 2 frames | 1 | 0 |" in text
    assert "| All frames | 2.8× (1.5×) |" in text  # per-frame ratios 4.0 and 1.5: median 2.75, slowest 1.5
    assert "| one-tag | 2 | 1 (1) | 2 |" in text


@pytest.mark.parametrize("sigma", [8, 15, 25])
def test_dense_noise_floor_preserves_twelve_pixel_tags(sigma: int) -> None:
    from napa_vision import NapaDetector

    detector = NapaDetector(min_tag_px=16)
    for index in range(12):
        rng = np.random.default_rng((12, sigma, index, 7319))
        image = matrix._uniform(rng, 640, 480)
        tags = matrix.place_tags(rng, 640, 480, 1, (12, 12), max_tilt_deg=10)
        for tag in tags:
            matrix.render_tag(image, tag.tag_id, tag.corners, black=30, white=220)
        noisy = matrix._to_uint8(matrix.add_noise(rng, image, sigma))
        observations = detector.detect(noisy)
        assert [o.tag_id for o in observations] == [t.tag_id for t in tags]


def test_dense_band_keeps_small_motion_blurred_tag() -> None:
    from napa_vision import NapaDetector

    frame = matrix.render_frame("motion-blur", 11, (640, 480), 2468)
    found = NapaDetector(min_tag_px=16).detect(frame.image)
    assert {o.tag_id for o in found} == {t.tag_id for t in frame.tags}
