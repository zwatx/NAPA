"""NapaDetector(threads=...): worker threads change latency only, never results."""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from napa_vision import NapaDetector
from napa_vision.tags import tag_image

# Small crops of the busy frames can fill a default max_detections; the tests compare results, not counts.
pytestmark = pytest.mark.filterwarnings("ignore:a frame filled all:RuntimeWarning")


def _busy_frame(family: str = "tag36h11", seed: int = 7, size: tuple[int, int] = (1280, 960)) -> np.ndarray:
    """Tags of many sizes and tilts on texture and noise, large enough that every level is split into bands."""

    rng = np.random.default_rng(seed)
    width, height = size
    image = cv2.resize(rng.integers(90, 200, (height // 16, width // 16), dtype=np.uint8), (width, height))
    for _ in range(40):
        y, x = rng.integers(0, height - 30), rng.integers(0, width - 30)
        image[y : y + rng.integers(4, 30), x : x + rng.integers(4, 30)] = rng.integers(0, 80)
    for _ in range(24):
        tile = tag_image(int(rng.integers(0, 30)), family=family, cell_px=int(rng.integers(2, 16)))
        side = tile.shape[0]
        if side >= min(width, height) // 2:
            continue
        angle = float(rng.uniform(0, 360))
        matrix = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
        x, y = int(rng.integers(0, width - side)), int(rng.integers(0, height - side))
        matrix[:, 2] += (x, y)
        warped = cv2.warpAffine(tile, matrix, (width, height), flags=cv2.INTER_AREA, borderValue=0)
        mask = cv2.warpAffine(np.ones_like(tile), matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        image = np.where(mask > 0, warped, image)
    noisy = image.astype(np.int16) + rng.normal(0, 4, image.shape).astype(np.int16)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _results(detector: NapaDetector, image: np.ndarray) -> list[tuple]:
    return [(o.tag_id, o.hamming, o.decision_margin, o.corners.tobytes()) for o in detector.detect(image)]


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"min_tag_px": 16},
        {"min_tag_px": 32},
        {"min_tag_px": 16, "recover_shadows": True},
        {"min_tag_px": 16, "split_touching": True, "recover_shadows": True, "shadow_min_tag_px": 24},
        {"min_tag_px": 16, "tag_shape": "tall"},
        {"family": "tag25h9", "min_tag_px": 16},
        {"family": "tagStandard41h12", "min_tag_px": 16},
    ],
    ids=["default", "m16", "m32", "shadows", "preset", "tall", "tag25h9", "reversed"],
)
def test_threads_give_identical_results(options: dict) -> None:
    image = _busy_frame(options.get("family", "tag36h11"))
    expected = _results(NapaDetector(max_detections=256, **options), image)
    assert len(expected) >= 5
    for threads in (2, 3, 4, 8):
        detector = NapaDetector(max_detections=256, threads=threads, **options)
        try:
            for _ in range(2):  # the second call reuses the detector's band buffers
                assert _results(detector, image) == expected
            assert _results(detector, image[:300, :500]) == _results(NapaDetector(**options), image[:300, :500])
        finally:
            detector.close()  # joins its worker threads now rather than at garbage collection


def test_threads_across_detectors_running_at_once() -> None:
    frames = [_busy_frame(seed=seed) for seed in range(3)]
    expected = [_results(NapaDetector(min_tag_px=16), frame) for frame in frames]
    errors: list[str] = []

    def run(index: int) -> None:
        detector = NapaDetector(min_tag_px=16, threads=3)
        try:
            for _ in range(3):
                if _results(detector, frames[index]) != expected[index]:
                    errors.append(f"frame {index}")
        finally:
            detector.close()

    workers = [threading.Thread(target=run, args=(index,)) for index in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert errors == []


@pytest.mark.parametrize("threads", [0, -1, 17, 2.0, True, "4"])
def test_threads_must_be_a_count_from_1_to_16(threads) -> None:
    with pytest.raises(ValueError, match="threads"):
        NapaDetector(threads=threads)


def test_threads_attribute() -> None:
    assert NapaDetector().threads == 1
    assert NapaDetector(threads=4).threads == 4
