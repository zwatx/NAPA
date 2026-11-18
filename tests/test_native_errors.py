"""Native callers get error statuses for invalid buffers and behavior-changing allocation failures."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector, napa_detector
from tests.test_camera_gamma import _streaked
from tests.test_napa_detector import _scene
from tests.test_small_tag_regrowth import _blurred_small_tag


@pytest.mark.parametrize(
    "entry",
    ["napa_detect", "napa_detect_shadows", "napa_detect_shadows_configured", "napa_detect_soft", "napa_detect_options"],
)
@pytest.mark.parametrize("capacity", [-1, 0, 1])
@pytest.mark.parametrize("tagged", [False, True])
def test_native_null_output_is_valid_only_at_zero_capacity(entry, capacity, tagged):
    image = _scene(42, 96)[0] if tagged else np.zeros((32, 32), np.uint8)
    detector = NapaDetector()
    extra = {
        "napa_detect": (),
        "napa_detect_shadows": (),
        "napa_detect_shadows_configured": (0,),
        "napa_detect_soft": (0, 0),
        "napa_detect_options": (0, 0, 0),
    }[entry]
    try:
        status = getattr(detector._library, entry)(
            detector._handle,
            image.ctypes.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            ctypes.byref(detector._params),
            None,
            capacity,
            *extra,
        )
        assert status == (0 if capacity == 0 else -1)
    finally:
        detector.close()


@pytest.fixture(scope="module")
def fault_library(tmp_path_factory):
    compiler = shutil.which("cc")
    root = Path(__file__).resolve().parents[1]
    if compiler is None or not (root / "src/napa_vision/_native/napa_core.c").exists():
        pytest.skip("native fault injection requires a C compiler and source checkout")
    output = tmp_path_factory.mktemp("allocation-failures") / "core.so"
    subprocess.run(
        [
            compiler,
            "-std=c99",
            "-O2",
            "-pthread",
            "-fno-wrapv",
            "-shared",
            "-fPIC",
            str(root / "tests/native_allocation_failures.c"),
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.mark.parametrize("case", ["gamma-initial", "gamma-resize", "regrow-stack", "regrow-marks", "tone-table"])
@pytest.mark.parametrize(
    "entry_options",
    [{}, {"recover_shadows": True}, {"soft_decode": True}, {"align_corners": True}],
    ids=["ordinary", "shadows", "soft", "aligned"],
)
def test_behavior_allocations_fail_explicitly_and_can_retry(fault_library, monkeypatch, case, entry_options):
    monkeypatch.setenv("NAPA_CORE_LIBRARY", str(fault_library))
    napa_detector._library.cache_clear()
    try:
        library = napa_detector._library()
        library.napa_test_fail_allocation.argtypes = [ctypes.c_char_p, ctypes.c_int]
        library.napa_test_fail_allocation.restype = None
        library.napa_test_allocation_hits.restype = ctypes.c_int
        gamma = case.startswith("gamma")
        image = (
            _streaked(24 if case == "tone-table" else 80, 20, 0.9, 25, 2.2)[0]
            if gamma or case == "tone-table"
            else _blurred_small_tag(7, 20, 25, 95)[0]
        )
        options = {"min_tag_px": 16, "camera_gamma": 2.2} if gamma else {"min_tag_px": 22, "regrow_small_tags": True}
        if case == "tone-table":
            options = {"min_tag_px": 16}
        options.update(entry_options)
        if case == "gamma-resize":
            image = cv2.copyMakeBorder(image, 0, 32, 0, 64, cv2.BORDER_REPLICATE)
        library.napa_test_fail_allocation(None, 0)
        fresh = NapaDetector(**options)
        try:
            expected = fresh.detect(image)
        finally:
            fresh.close()
        assert [o.tag_id for o in expected] == [42 if gamma or case == "tone-table" else 7]
        detector = NapaDetector(**options)
        try:
            if case == "gamma-resize":
                assert detector.detect(image[:480, :640])
            stage = (
                b"refine_decoded_candidate" if gamma else b"tone_prepare" if case == "tone-table" else b"regrow_small"
            )
            library.napa_test_fail_allocation(stage, int(gamma or case == "regrow-marks"))
            status = detector._detect(
                detector._handle,
                image.ctypes.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                ctypes.byref(detector._params),
                detector._results,
                len(detector._results),
                *detector._shadow_args,
            )
            assert library.napa_test_allocation_hits() == 1
            assert status == -2
            with pytest.raises(MemoryError, match="status -2"):
                detector.detect(image)
            assert library.napa_test_allocation_hits() == 2
            library.napa_test_fail_allocation(None, 0)
            actual = detector.detect(image)
            assert [o.tag_id for o in actual] == [o.tag_id for o in expected]
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a.corners, b.corners)
        finally:
            library.napa_test_fail_allocation(None, 0)
            detector.close()
    finally:
        napa_detector._library.cache_clear()
