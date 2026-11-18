"""Exercise the Python wrapper against the original five-field native parameter layout."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from napa_vision import NapaDetector, napa_detector


@pytest.fixture
def legacy_library(tmp_path, monkeypatch):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("legacy ABI check requires a C compiler")
    source = tmp_path / "legacy.c"
    source.write_text(
        """
        typedef struct { int minimum, hamming, contrast, border, sharpening; } params;
        static int calls;
        int legacy_calls(void) { return calls; }
        int napa_detect(void *det, const unsigned char *img, int w, int h, int stride,
                        const params *p, void *out, int max_out) {
            calls++;
            return p->minimum == 22 && p->hamming == 3 && p->contrast == 20 &&
                   p->border == 22 && p->sharpening == 25 ? 0 : -1;
        }
        """
    )
    path = tmp_path / "legacy.so"
    subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(path)], check=True, capture_output=True)
    legacy = ctypes.CDLL(str(path))
    current = napa_detector._library()
    legacy.napa_detect.argtypes = current.napa_detect.argtypes
    legacy.napa_detect.restype = ctypes.c_int

    class Legacy:
        def __getattr__(self, name):
            if name == "napa_params_size":
                raise AttributeError(name)
            return legacy.napa_detect if name == "napa_detect" else getattr(current, name)

    monkeypatch.setattr(napa_detector, "_library", lambda: Legacy())
    return legacy


@pytest.mark.parametrize("split", [True, "auto"])
def test_legacy_core_rejects_requested_splitting(legacy_library, split):
    with pytest.raises(napa_detector.NativeLibraryError, match="split_touching.*rebuild"):
        NapaDetector(split_touching=split)
    assert legacy_library.legacy_calls() == 0


def test_legacy_core_still_accepts_explicitly_disabled_splitting(legacy_library):
    detector = NapaDetector(split_touching=False)
    try:
        assert detector.detect(np.zeros((32, 32), np.uint8)) == []
    finally:
        detector.close()
    assert legacy_library.legacy_calls() == 1


@pytest.mark.parametrize("split", [False, True, "auto"])
def test_parameter_layout_mismatch_is_rejected(monkeypatch, split):
    current = napa_detector._library()

    class WrongLayout:
        def napa_params_size(self):
            return ctypes.sizeof(napa_detector._Params) + ctypes.sizeof(ctypes.c_int)

        def __getattr__(self, name):
            return getattr(current, name)

    monkeypatch.setattr(napa_detector, "_library", lambda: WrongLayout())
    with pytest.raises(napa_detector.NativeLibraryError, match="layout.*rebuild"):
        NapaDetector(split_touching=split)
