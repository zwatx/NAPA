"""Compare optimized private C kernels to exact reference algorithms."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _shape_growth_input(tmp_path: Path, small: bool = False) -> Path:
    """65 disjoint thin tags force both deferred-output buffers beyond their initial 64 entries."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if small:
        # The first stride-three level is terminal; 65 tiny outlines grow its queue during replay setup.
        image = np.full((288, 324), 255, np.uint8)
        marker = cv2.aruco.generateImageMarker(dictionary, 424, 18)
        for i in range(65):
            row, col = divmod(i, 9)
            y, x = row * 36 + 9, col * 36 + 12
            image[y : y + 18, x : x + 18] = marker
        path = tmp_path / "small-growth.raw"
        image.tofile(path)
        return path
    tile = np.full((328, 88), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(dictionary, 137, 48, borderBits=1)
    tile[20:308, 20:68] = cv2.resize(marker, (48, 288), interpolation=cv2.INTER_NEAREST)
    matrix = cv2.getRotationMatrix2D((44, 164), 45, 1)
    matrix[:, 2] += [84, -36]
    tile = cv2.warpAffine(tile, matrix, (256, 256), flags=cv2.INTER_LINEAR, borderValue=255)
    image = np.full((2048, 2304), 255, np.uint8)
    for i in range(65):
        y, x = divmod(i, 9)
        image[y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256] = tile
    path = tmp_path / "shape-growth.raw"
    image.tofile(path)
    return path


@pytest.mark.parametrize("scalar", [False, True], ids=["native", "scalar"])
@pytest.mark.parametrize(
    "source_name",
    [
        "native_speed_kernels.c",
        "native_curve_kernels.c",
        "native_contact_kernels.c",
        "native_coverage_kernels.c",
        "native_footprint_kernels.c",
        "native_shape_storage.c",
        "native_run_kernels.c",
        "native_small_outline.c",
        "native_borrowed_gray.c",
        "native_border_batch.c",
        "native_box_rake.c",
        "native_band_kernels.c",
        "native_pool.c",
        "native_extrema_cache.c",
        "native_nearest_code.c",
    ],
)
def test_native_speed_kernels(tmp_path: Path, scalar: bool, source_name: str) -> None:
    _run_native_kernels(tmp_path, scalar, source_name)


@pytest.mark.parametrize("scalar", [False, True], ids=["native", "scalar"])
def test_small_proposal_storage(tmp_path: Path, scalar: bool) -> None:
    _run_native_kernels(tmp_path, scalar, "native_shape_storage.c", small_storage=True)


def _run_native_kernels(tmp_path: Path, scalar: bool, source_name: str, small_storage: bool = False) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("standalone C kernel checks require a cc compiler")
    root = Path(__file__).resolve().parents[1]
    if not (root / "src/napa_vision/_native/napa_core.c").is_file():
        pytest.skip("private C kernel checks require a source checkout")
    executable = tmp_path / "kernels"
    command = [compiler, "-std=c99", "-O2", "-fno-wrapv", "-pthread"]
    if scalar:
        command.append("-DNAPA_NO_SIMD")
    if source_name == "native_shape_storage.c":
        command.append(f"-DNAPA_SHAPE_GROWTH_INPUT={json.dumps(str(_shape_growth_input(tmp_path, small_storage)))}")
        if small_storage:
            command.append("-DNAPA_SMALL_STORAGE")
    command.extend([str(root / "tests" / source_name), "-lm", "-o", str(executable)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    if source_name == "native_curve_kernels.c":
        assert "curved-edge numerical guard cases passed" in result.stdout
    elif source_name == "native_contact_kernels.c":
        assert "pixel connectivity oracle cases passed" in result.stdout
    elif source_name == "native_coverage_kernels.c":
        assert "coverage cache exact-oracle cases passed" in result.stdout
    elif source_name == "native_footprint_kernels.c":
        assert "30000 independent convex footprint/query oracle cases passed" in result.stdout
    elif source_name == "native_shape_storage.c":
        assert "65 distinct shapes, capped prefixes, six allocation-failure/retry cases passed" in result.stdout
    elif source_name == "native_run_kernels.c":
        assert "459292 exact pixel-scan run oracle cases passed" in result.stdout
    elif source_name == "native_border_batch.c":
        assert (
            "2560 sampler cases, 49408 scalar-oracle points, 152 family profiles, 5016 profile points passed"
            in result.stdout
        )
    elif source_name == "native_pool.c":
        assert "2400 pool jobs at 1-8 threads, 16 thread settings, 800 contended jobs passed" in result.stdout
        assert "4032 banded labelling cases match the serial pass" in result.stdout
    elif source_name == "native_band_kernels.c":
        assert "20480 exact dark-bit cases, 13608 exact tile-extrema rows passed" in result.stdout
        assert (
            "tile rows paired from exact half-tile extrema, 42 half-tile levels match the definition" in result.stdout
        )
    elif source_name == "native_box_rake.c":
        assert "6000 box_rake cases match the block-mean definition" in result.stdout
    elif source_name == "native_extrema_cache.c":
        assert "14 recovery cases at 1 and 4 threads match fresh thresholds, 20 saved levels reused" in result.stdout
    elif source_name == "native_nearest_code.c":
        assert "737973 reads match the nearest rotated code at max_bits 0-5" in result.stdout
    elif source_name == "native_borrowed_gray.c":
        receipt = json.loads(result.stdout)
        assert receipt["status"] == "PASS"
        assert receipt["cases"] == 48
        assert receipt["guarded_inputs"] in (0, 12)
    elif source_name == "native_small_outline.c":
        receipt = json.loads(result.stdout)
        assert receipt["status"] == "PASS"
        assert receipt["cases"] == 13765
        assert receipt["accepted"] == 2352
        assert receipt["rejected_exact_restore"] == 11413
        assert receipt["wrong_order_exact_restore"] == 11280
        assert receipt["canary_checks"] == 110120
    else:
        assert "786612 independent scalar samples, 90 independent projected samples passed" in result.stdout
        assert "14700 independent floor cases, 23700 regional values;" in result.stdout
        assert "600 connectivity cases" in result.stdout
        assert "10000 profile/median/paired sampling cases passed" in result.stdout
