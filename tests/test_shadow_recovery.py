"""Recovery supplements ordinary detection; it never changes existing observations."""

import ctypes

import numpy as np
import pytest

from napa_vision import NapaDetector
from tests.test_napa_detector import _scene


def _shadow(tag_id, side, angle):
    image, corners = _scene(tag_id, side, angle_deg=angle)
    yy, xx = np.indices(image.shape)
    image = np.where(xx + 0.4 * (yy - 240) < 320, image * 0.4, image).astype(np.uint8)
    return image, corners


@pytest.mark.parametrize("tag_id", [0, 17, 137, 586])
@pytest.mark.parametrize("angle", [0, 17, 45])
def test_shadow_recovers_exact_codes(tag_id, angle):
    image, corners = _shadow(tag_id, 96, angle)
    assert NapaDetector(min_tag_px=16, recover_shadows=False).detect(image) == []
    observations = NapaDetector(min_tag_px=16, recover_shadows=True).detect(image)
    # Some shadows still need corrected payload bits: recovery deliberately rejects those.
    expected = tag_id == 137 or (tag_id == 0 and angle != 45)
    assert all(o.tag_id == tag_id for o in observations)
    if expected:
        assert len(observations) == 1
    for observation in observations:
        assert observation.hamming == 0
        np.testing.assert_allclose(observation.corners, corners, atol=0.7)


@pytest.mark.parametrize("split", [False, True])
def test_shadow_preserves_prefix_and_capacity(split):
    left, _ = _scene(5, 96)
    right, _ = _shadow(137, 96, 17)
    image = np.hstack([left, right])
    ordinary = NapaDetector(min_tag_px=16, split_touching=split, recover_shadows=False).detect(image)
    assert [o.tag_id for o in ordinary] == [5]
    recovered = NapaDetector(min_tag_px=16, split_touching=split, recover_shadows=True).detect(image)
    assert [o.tag_id for o in recovered] == [5, 137]
    assert ordinary[0].hamming == recovered[0].hamming
    assert ordinary[0].decision_margin == recovered[0].decision_margin
    np.testing.assert_array_equal(ordinary[0].corners, recovered[0].corners)
    with pytest.warns(RuntimeWarning, match="max_detections"):
        capped = NapaDetector(min_tag_px=16, split_touching=split, recover_shadows=True, max_detections=1).detect(image)
    assert [o.tag_id for o in capped] == [5]
    np.testing.assert_array_equal(capped[0].corners, ordinary[0].corners)


def test_shadow_validation_and_empty_input():
    with pytest.raises(ValueError, match="recover_shadows"):
        NapaDetector(recover_shadows=1)
    d = NapaDetector(recover_shadows=True)
    assert d.detect(np.zeros((0, 0), np.uint8)) == []
    assert d.detect(np.zeros((16, 16), np.uint8)) == []
    image = np.zeros((32, 32), np.uint8)
    f = d._library.napa_detect_shadows
    args = (d._handle, image.ctypes.data, 32, 32, 32, ctypes.byref(d._params))
    assert f(*args, None, 1) == -1
    assert f(*args, None, 0) == 0
    assert f(*args, d._results, -1) == -1
    assert f(None, *args[1:], d._results, 64) == -1
    d.close()
    with pytest.raises(RuntimeError, match="close"):
        d.detect(image)


@pytest.mark.parametrize("side,size,contrast", [(32, (256, 128), 30), (48, (128, 128), 40)])
def test_shadow_preserves_noisy_low_contrast_detections(side, size, contrast):
    image, _ = _scene(42, side, size=size)
    rng = np.random.default_rng((side, 12, contrast, *size))
    noisy = np.clip(np.rint(100 + image.astype(float) * contrast / 255 + rng.normal(0, 12, image.shape)), 0, 255)
    image = noisy.astype(np.uint8)
    ordinary = NapaDetector(min_tag_px=16, recover_shadows=False).detect(image)
    recovered = NapaDetector(min_tag_px=16, recover_shadows=True).detect(image)
    assert [o.tag_id for o in ordinary] == [42]
    assert [o.tag_id for o in recovered] == [42]
    np.testing.assert_allclose(ordinary[0].corners, recovered[0].corners, atol=0.1)


@pytest.mark.parametrize("minimum", [None, 16, 24, 32, 30000])
@pytest.mark.parametrize("split", [False, True])
def test_shadow_sampling_preserves_ordinary(minimum, split):
    left, _ = _scene(5, 96)
    right, _ = _shadow(137, 96, 17)
    image = np.hstack([left, right])
    ordinary = NapaDetector(min_tag_px=16, split_touching=split, recover_shadows=False).detect(image)
    d = NapaDetector(min_tag_px=16, split_touching=split, recover_shadows=True, shadow_min_tag_px=minimum)
    result = d.detect(image)
    assert [o.tag_id for o in result[: len(ordinary)]] == [o.tag_id for o in ordinary]
    for a, b in zip(ordinary, result, strict=False):
        assert (a.hamming, a.decision_margin) == (b.hamming, b.decision_margin)
        np.testing.assert_array_equal(a.corners, b.corners)
    assert all(o.tag_id in (5, 137) for o in result)
    if minimum in (None, 16, 24, 32):
        assert [o.tag_id for o in result] == [5, 137]
    full = NapaDetector(min_tag_px=16, recover_shadows=True, split_touching=split).detect(image)
    if minimum in (None, 16):
        for a, b in zip(full, result, strict=True):
            assert (a.tag_id, a.hamming, a.decision_margin) == (b.tag_id, b.hamming, b.decision_margin)
            np.testing.assert_array_equal(a.corners, b.corners)
    with pytest.warns(RuntimeWarning, match="max_detections"):
        capped = NapaDetector(
            min_tag_px=16, recover_shadows=True, shadow_min_tag_px=minimum, split_touching=split, max_detections=1
        ).detect(image)
    np.testing.assert_array_equal(capped[0].corners, ordinary[0].corners)


@pytest.mark.parametrize("minimum", [True, False, 15, 0, -1, 30001, 24.0, "24"])
def test_shadow_sampling_invalid(minimum):
    with pytest.raises(ValueError, match="shadow_min_tag_px"):
        NapaDetector(min_tag_px=16, recover_shadows=True, shadow_min_tag_px=minimum)


def test_shadow_sampling_requires_recovery():
    with pytest.raises(ValueError, match="requires recover_shadows"):
        NapaDetector(shadow_min_tag_px=48)


def test_shadow_sampling_native_validation():
    d = NapaDetector(min_tag_px=16, recover_shadows=True, shadow_min_tag_px=24)
    image = np.zeros((32, 32), np.uint8)
    f = d._library.napa_detect_shadows_configured
    args = (d._handle, image.ctypes.data, 32, 32, 32, ctypes.byref(d._params))
    for minimum in (-1, 1, 15, 30001):
        assert f(*args, None, 0, minimum) == -1
    for minimum in (0, 16, 24, 30000):
        assert f(*args, None, 0, minimum) == 0
        assert f(*args, None, 1, minimum) == -1
        assert f(*args, d._results, -1, minimum) == -1
    assert f(*args[:5], None, d._results, 64, 24) == -1


def test_shadow_sampling_stale_library(monkeypatch):
    import napa_vision.napa_detector as native
    from napa_vision.napa_detector import NativeLibraryError

    library = native._library()

    class OldLibrary:
        def __getattr__(self, name):
            if name == "napa_detect_shadows_configured":
                raise AttributeError(name)
            return getattr(library, name)

    monkeypatch.setattr(native, "_library", lambda: OldLibrary())
    NapaDetector(recover_shadows=True).close()
    NapaDetector().close()
    with pytest.raises(NativeLibraryError, match="shadow_min_tag_px requires an updated"):
        NapaDetector(min_tag_px=16, recover_shadows=True, shadow_min_tag_px=24)


def test_default_runs_no_shadow_pass_and_coarse_pass_is_opt_in():
    image, _ = _shadow(137, 96, 17)
    assert NapaDetector().detect(image) == []
    assert NapaDetector(recover_shadows=False).detect(image) == []
    assert [o.tag_id for o in NapaDetector(recover_shadows=True, shadow_min_tag_px=64).detect(image)] == [137]


def test_shadow_recovery_is_off_by_default():
    detector = NapaDetector()
    assert detector.recover_shadows is False and detector.shadow_min_tag_px is None
    assert NapaDetector(min_tag_px=128).recover_shadows is False
    with pytest.raises(ValueError, match="recover_shadows must be True or False"):
        NapaDetector(recover_shadows=None)
