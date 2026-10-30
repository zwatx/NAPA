"""CPU tag36h11 detector adapters for ``cpu_comparison.py``, and the isolated resource worker.

Every adapter constructs its detector once and returns ``detect(image) -> [(id, corners)]`` where ``corners``
is a (4, 2) float64 array in the benchmark's convention: OpenCV pixel centres (the centre of the top-left
pixel is (0, 0)), in the detector's own corner order. Detectors documented to put pixel centres at +0.5
(upstream AprilTag, Locus) are shifted by -0.5 here; ``validate`` in ``cpu_comparison.py`` checks every
shift and order against known geometry before any score is trusted.

This module imports neither OpenCV nor the corpus renderer, so a resource worker measures only its own
backend's libraries. Thread pools are configured by the caller's environment (``RAYON_NUM_THREADS``,
``RAPIDTAG_CORES``, ``OMP_NUM_THREADS``) before the first backend import; ``configure_threads`` does that.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
SHIM_DIR = Path(os.environ.get("NAPA_BENCH_SHIMS", WORK / "build" / "shims"))
THIRD_PARTY = Path(os.environ.get("NAPA_BENCH_THIRD_PARTY", WORK / "thirdparty"))
MAX_OUT = 1024
SKIPPED_EXIT = 3  # resource worker: the backend's library or shim isn't available

Detection = tuple[int, np.ndarray]
DetectFn = Callable[[np.ndarray], list[Detection]]


@dataclass(frozen=True, slots=True)
class Backend:
    """Same shape as ``performance_matrix.Backend``: a name, a once-per-instance factory, and its settings."""

    name: str
    factory: Callable[[], DetectFn]
    config: dict[str, object]


# ------------------------------------------------------------------------------------------ helpers


def configure_threads(threads: int, cpus: list[int]) -> dict[str, str]:
    """Pin hidden pools before any backend is imported; returns the values in effect."""

    os.environ.setdefault("RAYON_NUM_THREADS", str(threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # RapidTag pins its Rayon workers to the CPUs with the highest max frequency (CPUs 0-11 on an i7-12700H),
    # overriding the process affinity, unless this names the cores. Name exactly the permitted CPUs.
    os.environ.setdefault("RAPIDTAG_CORES", ",".join(map(str, cpus)))
    return {
        k: os.environ[k] for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAPIDTAG_CORES")
    }


def thread_report() -> list[dict[str, object]]:
    """Every OS thread of this process: name, allowed CPUs, and CPU time so far."""

    ticks = os.sysconf("SC_CLK_TCK")
    report = []
    for tid in sorted(int(t) for t in os.listdir("/proc/self/task")):
        try:
            name = Path(f"/proc/self/task/{tid}/comm").read_text().strip()
            fields = Path(f"/proc/self/task/{tid}/stat").read_text().rsplit(")", 1)[1].split()
            cpu_s = (int(fields[11]) + int(fields[12])) / ticks
            report.append({"tid": tid, "name": name, "cpus": sorted(os.sched_getaffinity(tid)), "cpu_s": cpu_s})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return report


def public_text(text: str) -> str:
    """Result files name paths relative to the repository (and the home directory as ~), not absolutely."""

    return text.replace(str(ROOT) + os.sep, "").replace(str(Path.home()), "~")


def loaded_libraries(*needles: str) -> list[str]:
    """Shared objects mapped into this process whose path contains any needle (Linux only; else empty)."""

    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    paths = set()
    for line in maps.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6 and any(n in parts[5] for n in needles):
            paths.add(parts[5])
    return sorted(paths)


def memory_status() -> dict[str, int]:
    """VmRSS and VmHWM (peak RSS since start or the last reset), in KiB."""

    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            status[key] = int(value.split()[0])
    return status


def reset_peak_rss() -> bool:
    """Reset VmHWM to the current RSS (Linux clear_refs 5), so a later peak covers only what follows."""

    try:
        Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _contiguous(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("expected a 2-D uint8 grayscale image")
    return image if image.flags["C_CONTIGUOUS"] else np.ascontiguousarray(image)


class _Outputs:
    """Preallocated ctypes output buffers for the C shims."""

    def __init__(self) -> None:
        self.ids = np.zeros(MAX_OUT, np.int32)
        self.hamming = np.zeros(MAX_OUT, np.int32)
        self.corners = np.zeros((MAX_OUT, 4, 2), np.float64)

    def rows(self, count: int, offset: float) -> list[Detection]:
        if count < 0:
            raise RuntimeError(f"detector returned status {count}")
        if count > MAX_OUT:
            raise RuntimeError(f"{count} detections exceed the {MAX_OUT}-entry buffer")
        corners = self.corners[:count] + offset
        return [(int(i), c) for i, c in zip(self.ids[:count].tolist(), corners, strict=True)]


_INT_P = ctypes.POINTER(ctypes.c_int)
_DOUBLE_P = ctypes.POINTER(ctypes.c_double)


# ------------------------------------------------------------------------------------------- Napa


def napa_backend(name: str, min_tag_px: int = 32) -> Backend:
    from napa_vision import __version__
    from napa_vision.napa_detector import NapaDetector, _library

    library = _library()  # loads now, so the config can name the file that actually loaded
    core = loaded_libraries("napa_core")

    def factory() -> DetectFn:
        detector = NapaDetector(min_tag_px=min_tag_px)
        return lambda image: [(o.tag_id, o.corners) for o in detector.detect(image)]

    probe = NapaDetector(min_tag_px=min_tag_px)
    params = probe._params
    config = {
        "library": "NAPA (napa_vision.NapaDetector, public Python API)",
        "version": __version__,
        "commit": _git_head(ROOT),
        "core_library_env": os.environ.get("NAPA_CORE_LIBRARY"),
        "core_library_loaded": core,
        "core_library_sha256": [sha256(Path(p)) for p in core],
        "core_handle": str(library),
        "family": "tag36h11",
        "min_tag_px": params.min_tag_px,
        # napa_core.c: base = min_tag_px * 3 / (4 * W), W = 8 for tag36h11; at least 1.
        "base_sampling_stride_px": max(1, params.min_tag_px * 3 // 32),
        "noisy_frame_box_averaging": "only when the base stride is 2-16 px",
        "max_hamming": params.max_hamming,
        "min_contrast": params.min_contrast,
        "min_border_cells": params.min_border_cells,
        "sharpening": params.sharpening_x100 / 100,
        "decimation": "none (sparse rake; every level fits corners at full resolution)",
        "corner_refinement": "always: full-resolution edge fit",
        "threads": 1,
        "threading": "none inside a frame; one instance per thread",
        "pixel_offset_applied": 0.0,
    }
    probe.close()
    return Backend(name, factory, config)


# ------------------------------------------------------------------------------- upstream AprilTag


def _apriltag_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(SHIM_DIR / "libapriltag_shim.so"))
    library.at_create.restype = ctypes.c_void_p
    library.at_create.argtypes = [
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
    ]
    library.at_settings.argtypes = [ctypes.c_void_p, _DOUBLE_P]
    library.at_detect.restype = ctypes.c_int
    library.at_detect.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _INT_P,
        _INT_P,
        _DOUBLE_P,
        ctypes.c_int,
    ]
    library.at_destroy.argtypes = [ctypes.c_void_p]
    return library


def apriltag_backend(name: str, decimate: float, threads: int) -> Backend:
    library = _apriltag_library()
    defaults = {"quad_sigma": 0.0, "refine_edges": 1, "decode_sharpening": 0.25, "bits_corrected": 2}

    def create() -> int:
        handle = library.at_create(
            threads,
            decimate,
            defaults["quad_sigma"],
            defaults["refine_edges"],
            defaults["decode_sharpening"],
            defaults["bits_corrected"],
        )
        if not handle:
            raise MemoryError("at_create failed")
        return handle

    def factory() -> DetectFn:
        handle, out = create(), _Outputs()

        def detect(image: np.ndarray) -> list[Detection]:
            image = _contiguous(image)
            count = library.at_detect(
                handle,
                image.ctypes.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                out.ids.ctypes.data_as(_INT_P),
                out.hamming.ctypes.data_as(_INT_P),
                out.corners.ctypes.data_as(_DOUBLE_P),
                MAX_OUT,
            )
            # AprilTag puts pixel centres at +0.5 (apriltag.c value_for_pixel, quad fit "pixel center bias").
            return out.rows(count, -0.5)

        detect.handle = handle  # keep the native detector alive with the closure
        return detect

    handle = create()
    values = (ctypes.c_double * 10)()
    library.at_settings(handle, values)
    library.at_destroy(handle)
    keys = (
        "nthreads",
        "quad_decimate",
        "quad_sigma",
        "refine_edges",
        "decode_sharpening",
        "min_cluster_pixels",
        "min_white_black_diff",
        "max_line_fit_mse",
        "critical_rad",
        "deglitch",
    )
    config = {
        "library": "AprilTag 3 (AprilRobotics/apriltag, C library via a ctypes shim)",
        "commit": _git_head(THIRD_PARTY / "apriltag"),
        "build": "CMake Release (-O3 -DNDEBUG -std=gnu99), shared",
        "family": "tag36h11",
        "bits_corrected": defaults["bits_corrected"],
        **{f"effective_{k}": v for k, v in zip(keys, values, strict=True)},
        "corner_refinement": "refine_edges (on)",
        "threads": threads,
        "pixel_offset_applied": -0.5,
    }
    return Backend(name, factory, config)


# ----------------------------------------------------------------------------------- WPILib AprilTag


def wpilib_backend(name: str, decimate: float, threads: int) -> Backend:
    """WPILib's build of AprilTag 3 (robotpy-apriltag) with WPILib's own defaults, as FRC robot code runs it."""

    from robotpy_apriltag import AprilTagDetector

    def create():
        detector = AprilTagDetector()
        if not detector.addFamily("tag36h11", 2):
            raise RuntimeError("robotpy-apriltag could not load tag36h11")
        config = detector.getConfig()
        config.numThreads, config.quadDecimate = threads, decimate
        detector.setConfig(config)
        return detector

    def factory() -> DetectFn:
        detector = create()

        def detect(image: np.ndarray) -> list[Detection]:
            # AprilTag puts pixel centres at +0.5, as upstream does.
            return [
                (int(d.getId()), np.array([[d.getCorner(i).x, d.getCorner(i).y] for i in range(4)]) - 0.5)
                for d in detector.detect(image)
            ]

        return detect

    probe = create()
    config, quad = probe.getConfig(), probe.getQuadThresholdParameters()
    settings = {
        "library": "AprilTag 3, WPILib build (robotpy-apriltag)",
        "version": _package_version("robotpy-apriltag"),
        "family": "tag36h11",
        "bits_corrected": 2,
        **{f"effective_{k}": getattr(config, k) for k in ("numThreads", "quadDecimate", "quadSigma", "refineEdges")},
        **{f"effective_{k}": getattr(config, k) for k in ("decodeSharpening",)},
        # WPILib's defaults differ from upstream's: clusters of 300 px (upstream 24) and a 45 degree critical angle
        # (upstream 10), so small and steep tags are dropped before decoding.
        **{f"effective_quad_{k}": getattr(quad, k) for k in ("minClusterPixels", "criticalAngle", "maxLineFitMSE")},
        **{f"effective_quad_{k}": getattr(quad, k) for k in ("minWhiteBlackDiff", "deglitch")},
        "corner_refinement": "refineEdges (on)",
        "threads": threads,
        "pixel_offset_applied": -0.5,
    }
    return Backend(name, factory, settings)


# --------------------------------------------------------------------------------------------- Locus


def locus_backend(name: str, threads: int, decimation: int | None = None, profile: str = "standard") -> Backend:
    import locus

    def create():
        return locus.Detector(
            profile=profile, decimation=decimation, threads=threads, families=[locus.TagFamily.AprilTag36h11]
        )

    def factory() -> DetectFn:
        detector = create()

        def detect(image: np.ndarray) -> list[Detection]:
            batch = detector.detect(image)
            # Locus puts pixel centres at +0.5 (docs/explanation/coordinates.md).
            corners = np.asarray(batch.corners, np.float64) - 0.5
            return [(int(i), c) for i, c in zip(np.asarray(batch.ids).tolist(), corners, strict=True)]

        return detect

    effective = create().config().model_dump(mode="json")
    config = {
        "library": "Locus (locus-tag, Rust via PyO3)",
        "version": _package_version("locus-tag"),
        "build": "PyPI abi3 manylinux wheel (maturin --release, no target-cpu flags)",
        "profile": profile,
        "decimation": decimation or 1,
        "threads": threads,
        "threading": "scoped Rayon pool of `threads` workers per detector",
        "family": "tag36h11",
        "max_hamming": "family default (2 for tag36h11)"
        if effective["decoder"]["max_hamming_error"] is None
        else effective["decoder"]["max_hamming_error"],
        "corner_refinement": f"{effective['decoder']['refinement_mode']} + corner_subpix="
        f"{effective['decoder']['corner_subpix']}",
        "effective_config": effective,
        "pixel_offset_applied": -0.5,
    }
    return Backend(name, factory, config)


# -------------------------------------------------------------------------------------- OpenCV ArUco


_REFINE = {"none": 0, "subpix": 1, "contour": 2, "apriltag": 3}


def opencv_backend(
    name: str,
    threads: int,
    refine: str = "none",
    aruco3: bool = False,
    aruco3_scale: float | None = None,
    canonical_side: int = 32,
) -> Backend:
    """``aruco3_scale`` detects on an image scaled by that factor (ArUco3 paper, equation 2): with a canonical
    side of c px, the smallest marker found is about c / scale px in the original image."""

    import cv2

    cv2.setNumThreads(threads)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    def parameters(max_side: int | None) -> object:
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = _REFINE[refine]
        p.useAruco3Detection = aruco3
        p.minSideLengthCanonicalImg = canonical_side
        if aruco3_scale is not None and max_side:
            # aruco_detector.cpp: fxfy = c / (c + max_side * ratio), so ratio = c * (1 / scale - 1) / max_side.
            p.minMarkerLengthRatioOriginalImg = canonical_side * (1 / aruco3_scale - 1) / max_side
        return p

    def factory() -> DetectFn:
        detectors: dict[int, object] = {}

        def detect(image: np.ndarray) -> list[Detection]:
            key = max(image.shape) if aruco3_scale is not None else 0
            detector = detectors.get(key)
            if detector is None:
                detector = detectors[key] = cv2.aruco.ArucoDetector(dictionary, parameters(key))
            corners, ids, _rejected = detector.detectMarkers(image)
            if ids is None:
                return []
            return [
                (int(i), np.asarray(c, np.float64).reshape(4, 2)) for i, c in zip(ids.reshape(-1), corners, strict=True)
            ]

        return detect

    p = parameters(None)
    fields = (
        "adaptiveThreshWinSizeMin",
        "adaptiveThreshWinSizeMax",
        "adaptiveThreshWinSizeStep",
        "adaptiveThreshConstant",
        "minMarkerPerimeterRate",
        "maxMarkerPerimeterRate",
        "minOtsuStdDev",
        "errorCorrectionRate",
        "maxErroneousBitsInBorderRate",
        "cornerRefinementMethod",
        "useAruco3Detection",
        "minSideLengthCanonicalImg",
        "minMarkerLengthRatioOriginalImg",
    )
    config = {
        "library": "OpenCV ArUco (cv2.aruco.ArucoDetector)",
        "version": cv2.__version__,
        "build": "opencv-python-headless wheel",
        "dictionary": "DICT_APRILTAG_36h11",
        "threads": cv2.getNumThreads(),
        **{f"param_{f}": getattr(p, f) for f in fields},
        # aruco_detector.cpp (5.0.0): useAruco3Detection forces CORNER_REFINE_SUBPIX, but its refinement
        # (findCornerInPyrImage) runs cornerSubPix only on pyramid levels above the one detected on: at the
        # default ratio of 0 detection is at full scale, so no refinement runs. Without ArUco3,
        # minSideLengthCanonicalImg and minMarkerLengthRatioOriginalImg are zeroed.
        "effective_corner_refinement": (
            "subpix on the upsampling pyramid levels"
            if aruco3_scale
            else "none (forced SUBPIX runs on zero pyramid levels at full-scale detection)"
            if aruco3
            else refine
        ),
        "aruco3_detection_scale": aruco3_scale,
        "effective_min_marker_side": (
            f"about {canonical_side / aruco3_scale:g} px (canonical side {canonical_side} px at scale {aruco3_scale:g};"
            " the contour gate is 4 x canonical side in the scaled image)"
            if aruco3_scale
            else f"{canonical_side} px (minSideLengthCanonicalImg)"
            if aruco3
            else "perimeter >= 0.03 * max(w, h): about 5 px at 640, 14 px at 1920"
        ),
        "max_correctable_bits": "floor(0.6 * 5) = 3 for tag36h11 (errorCorrectionRate * (hamming distance - 1) / 2)",
        "pixel_offset_applied": 0.0,
    }
    return Backend(name, factory, config)


# --------------------------------------------------------------------------------------- ArUco Nano


def _nano_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(SHIM_DIR / "libaruco_nano_shim.so"))
    library.an_create.restype = ctypes.c_void_p
    library.an_create.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int]
    library.an_settings.argtypes = [ctypes.c_void_p, _DOUBLE_P]
    library.an_set_threads.argtypes = [ctypes.c_int]
    library.an_get_threads.restype = ctypes.c_int
    library.an_opencv_version.restype = ctypes.c_char_p
    library.an_detect.restype = ctypes.c_int
    library.an_detect.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _INT_P,
        _DOUBLE_P,
        ctypes.c_int,
    ]
    library.an_destroy.argtypes = [ctypes.c_void_p]
    return library


def aruco_nano_backend(name: str, threads: int) -> Backend:
    library = _nano_library()
    library.an_set_threads(threads)

    def factory() -> DetectFn:
        handle, out = library.an_create(-1, -1.0, -1), _Outputs()

        def detect(image: np.ndarray) -> list[Detection]:
            image = _contiguous(image)
            count = library.an_detect(
                handle,
                image.ctypes.data,
                image.shape[1],
                image.shape[0],
                image.strides[0],
                out.ids.ctypes.data_as(_INT_P),
                out.corners.ctypes.data_as(_DOUBLE_P),
                MAX_OUT,
            )
            return out.rows(count, 0.0)

        detect.handle = handle
        return detect

    handle = library.an_create(-1, -1.0, -1)
    values = (ctypes.c_double * 11)()
    library.an_settings(handle, values)
    library.an_destroy(handle)
    keys = (
        "boxFilterSize",
        "thres",
        "minSize",
        "maxAttemptsPerCandidate",
        "maxTimesRevisited",
        "errorCorrectionRate",
        "maxErroneousBitsInBorderRate",
        "detectInvertedMarker",
        "dictionaries",
        "ArucoNanoVersion",
        "opencv_threads",
    )
    config = {
        "library": "ArUco Nano (rmsalinas/aruco_nano, header-only C++ via a ctypes shim)",
        "commit": _git_head(THIRD_PARTY / "aruco_nano"),
        "opencv": library.an_opencv_version().decode(),
        "build": "g++ -O3 -std=c++17 against OpenCV 4.14.0 (Release, IPP-ICV, pthreads, baseline SSE3 + dispatch)",
        "dictionary": "DICT_APRILTAG_36h11 only",
        **{f"effective_{k}": v for k, v in zip(keys, values, strict=True)},
        "corner_refinement": "always: line fit per side (refineCornersEdges)",
        "threads": library.an_get_threads(),
        "pixel_offset_applied": 0.0,
    }
    return Backend(name, factory, config)


# ------------------------------------------------------------------------------------------ RapidTag


def rapidtag_backend(name: str, refine: str = "none") -> Backend:
    import rapidtag

    def make_parameters():
        p = rapidtag.DetectorParameters()
        p.corner_refinement_method = _REFINE[refine]
        return p

    def factory() -> DetectFn:
        p = make_parameters()

        def detect(image: np.ndarray) -> list[Detection]:
            corners, ids = rapidtag.detect_markers(image, "DICT_APRILTAG_36h11", p)
            return [(int(i), np.asarray(c, np.float64).reshape(4, 2)) for i, c in zip(ids, corners, strict=True)]

        return detect

    p = make_parameters()
    fields = (
        "adaptive_thresh_win_size_min",
        "adaptive_thresh_win_size_max",
        "adaptive_thresh_win_size_step",
        "adaptive_thresh_constant",
        "error_correction_rate",
        "min_side_length_canonical_img",
        "corner_refinement_method",
        "detect_inverted_marker",
    )
    config = {
        "library": "RapidTag (rapidtag, Rust via PyO3)",
        "version": _package_version("rapidtag"),
        "build": "PyPI abi3 manylinux wheel (portable; no target-cpu=native)",
        "dictionary": "DICT_APRILTAG_36h11",
        **{f"param_{f}": getattr(p, f) for f in fields},
        # detector.rs: min perimeter is 4 * min_side_length_canonical_img whenever it is non-zero.
        "effective_min_marker_side": "32 px (4 * min_side_length_canonical_img perimeter)",
        "input_copy": "copies the frame into a Vec on every call (lib.rs extract_frame)",
        "threads": int(os.environ.get("RAYON_NUM_THREADS", "0")) or None,
        "threading": "global Rayon pool over threshold scales; RAPIDTAG_CORES=" + os.environ.get("RAPIDTAG_CORES", ""),
        "pixel_offset_applied": 0.0,
    }
    return Backend(name, factory, config)


# ------------------------------------------------------------------------------------------ registry


NAPA_SWEEP = (16, 22, 32, 43, 64)  # base strides 1, 2, 3, 4, 6


def registry(threads: int) -> dict[str, Callable[[], Backend]]:
    """Every backend configuration by name. Building one imports and probes its library."""

    entries: dict[str, Callable[[], Backend]] = {
        "napa": lambda: napa_backend("napa", 32),
        **{f"napa-mtp{m}": (lambda m=m: napa_backend(f"napa-mtp{m}", m)) for m in NAPA_SWEEP if m != 32},
        "apriltag3-d2": lambda: apriltag_backend("apriltag3-d2", 2.0, threads),
        "apriltag3-d1": lambda: apriltag_backend("apriltag3-d1", 1.0, threads),
        "wpilib-d2": lambda: wpilib_backend("wpilib-d2", 2.0, threads),
        "locus": lambda: locus_backend("locus", threads),
        "locus-d2": lambda: locus_backend("locus-d2", threads, decimation=2),
        "opencv": lambda: opencv_backend("opencv", threads),
        "opencv-subpix": lambda: opencv_backend("opencv-subpix", threads, refine="subpix"),
        "opencv-aruco3": lambda: opencv_backend("opencv-aruco3", threads, aruco3=True),
        # Historical name: half-scale detection with the default 32 px canonical side, so the effective minimum is
        # about 64 px, not 32. Kept so existing result files stay comparable.
        "opencv-aruco3-min32": lambda: opencv_backend("opencv-aruco3-min32", threads, aruco3=True, aruco3_scale=0.5),
        "opencv-aruco3-half16": lambda: opencv_backend(
            "opencv-aruco3-half16", threads, aruco3=True, aruco3_scale=0.5, canonical_side=16
        ),
        "aruco-nano": lambda: aruco_nano_backend("aruco-nano", threads),
        "rapidtag": lambda: rapidtag_backend("rapidtag"),
        "rapidtag-subpix": lambda: rapidtag_backend("rapidtag-subpix", refine="subpix"),
    }
    return entries


TUNING = {
    "napa": "default",
    **{f"napa-mtp{m}": f"tuned: min_tag_px={m}" for m in NAPA_SWEEP if m != 32},
    "apriltag3-d2": "default (quad_decimate=2)",
    "apriltag3-d1": "tuned: quad_decimate=1 (full resolution)",
    "wpilib-d2": "WPILib's defaults (quad_decimate=2, min cluster 300 px, critical angle 45 degrees)",
    "locus": "default (standard profile)",
    "locus-d2": "tuned for speed: standard + decimation=2",
    "opencv": "default (no corner refinement)",
    "opencv-subpix": "tuned: CORNER_REFINE_SUBPIX",
    "opencv-aruco3": "tuned: useAruco3Detection (defaults; no refinement runs at full scale)",
    "opencv-aruco3-min32": "tuned for speed: ArUco3 at half scale, canonical 32 px (effective min side ~64 px)",
    "opencv-aruco3-half16": "tuned for speed: ArUco3 at half scale, canonical 16 px (effective min side ~32 px)",
    "aruco-nano": "default (36h11 dictionary only)",
    "rapidtag": "default (no corner refinement)",
    "rapidtag-subpix": "tuned: CORNER_REFINE_SUBPIX",
}


def build_backends(names: list[str], threads: int) -> tuple[list[Backend], dict[str, str]]:
    entries = registry(threads)
    backends, skipped = [], {}
    for name in names:
        if name not in entries:
            raise ValueError(f"unknown backend {name!r}; choose from {sorted(entries)}")
        try:
            backend = entries[name]()
            backend.config["tuning"] = TUNING[name]
            backends.append(backend)
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
    return backends, skipped


# ----------------------------------------------------------------------------------- resource worker


def resource_main(argv: list[str] | None = None) -> int:
    """Measure one backend alone: construction, CPU time, wall time, and peak RSS over a frame set.

    ``serial`` runs every frame on one detector in this thread. ``parallel`` gives each of ``--workers``
    Python threads its own detector (one internal thread each) and lets them pull frames from a shared index.
    """

    parser = argparse.ArgumentParser(description=resource_main.__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--frames", type=Path, required=True, help=".npz written by cpu_comparison.py corpus")
    parser.add_argument("--threads", type=int, default=1, help="Threads inside each detector")
    parser.add_argument("--mode", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    # Thread pools started by NumPy's import (OpenBLAS) must already be limited by the parent's environment;
    # recorded here so a run with an unlimited BLAS pool is visible.
    threads_at_start = thread_report()
    cpus = sorted(os.sched_getaffinity(0))
    env = configure_threads(args.threads, cpus)
    with np.load(args.frames) as data:
        keys = sorted(key for key in data.files if key.startswith("image_"))
        images = [np.ascontiguousarray(data[key]) for key in keys]
        names = [str(n) for n in data["names"]] if "names" in data.files else keys
    workers = args.workers if args.mode == "parallel" else 1
    work = [(index, image) for _ in range(args.passes) for index, image in enumerate(images)]
    latencies: list[tuple[int, int]] = []  # (frame index, ns)
    lock = threading.Lock()

    # Phases, each measured at its boundary: import + probe, construction, then (after resetting the peak)
    # warm-up and the timed passes. "Peak over constructed" therefore covers warm-up and detection only.
    rss_before = memory_status()["VmRSS"]
    cpu0, wall0 = time.process_time_ns(), time.perf_counter_ns()
    backends, skipped = build_backends([args.backend], args.threads)  # imports + probes the library
    if not backends:
        print(skipped[args.backend], file=sys.stderr)
        return SKIPPED_EXIT
    (backend,) = backends
    import_ms = (time.perf_counter_ns() - wall0) / 1e6
    rss_after_import = memory_status()["VmRSS"]

    if args.mode == "serial":
        start = time.perf_counter_ns()
        detect = backend.factory()
        construct_ms = (time.perf_counter_ns() - start) / 1e6
        rss_after_construct = memory_status()["VmRSS"]
        peak_reset = reset_peak_rss()
        for image in images:  # warm-up pass: first-touch buffers, lazy pools, caches
            detect(image)
        rss_after_warmup = memory_status()["VmRSS"]
        threads_after_warmup = thread_report()
        gc.disable()
        cpu0, wall0 = time.process_time_ns(), time.perf_counter_ns()
        for index, image in work:
            start = time.perf_counter_ns()
            detect(image)
            latencies.append((index, time.perf_counter_ns() - start))
        wall_ns, cpu_ns = time.perf_counter_ns() - wall0, time.process_time_ns() - cpu0
        gc.enable()
    else:
        # Each worker builds its own detector in its own thread (some bindings, e.g. Locus, are unsendable).
        # Barriers separate construction, warm-up, and the timed region across all workers.
        cursor = iter(range(len(work)))
        built, warm, warmed, go = (threading.Barrier(workers + 1) for _ in range(4))
        errors: list[BaseException] = []

        def run(worker: int) -> None:
            detect = None
            try:
                detect = backend.factory()
            except BaseException as exc:  # reported after the barriers, never left hanging
                errors.append(exc)
            built.wait()
            warm.wait()
            try:
                if detect is not None:
                    for image in images[worker % 4 :: 4] or images:
                        detect(image)
            except BaseException as exc:
                errors.append(exc)
                detect = None
            warmed.wait()
            go.wait()
            if detect is None:
                return
            local = []
            try:
                while True:
                    with lock:
                        index = next(cursor, None)
                    if index is None:
                        break
                    frame, image = work[index]
                    start = time.perf_counter_ns()
                    detect(image)
                    local.append((frame, time.perf_counter_ns() - start))
            except BaseException as exc:
                errors.append(exc)
            with lock:
                latencies.extend(local)

        pool = [threading.Thread(target=run, args=(w,)) for w in range(workers)]
        start = time.perf_counter_ns()
        for thread in pool:
            thread.start()
        built.wait()
        construct_ms = (time.perf_counter_ns() - start) / 1e6
        rss_after_construct = memory_status()["VmRSS"]
        peak_reset = reset_peak_rss()
        warm.wait()
        warmed.wait()
        rss_after_warmup = memory_status()["VmRSS"]
        threads_after_warmup = thread_report()
        gc.disable()
        cpu0, wall0 = time.process_time_ns(), time.perf_counter_ns()
        go.wait()
        for thread in pool:
            thread.join()
        wall_ns, cpu_ns = time.perf_counter_ns() - wall0, time.process_time_ns() - cpu0
        gc.enable()
        if errors:
            raise RuntimeError(f"worker failed: {errors[0]!r}")
    if len(latencies) != len(work):
        raise RuntimeError(f"{len(latencies)} timed calls completed of {len(work)} planned")
    memory = memory_status()
    values = np.asarray([ns for _, ns in latencies], np.float64) / 1e6
    per_frame: dict[int, list[int]] = {}
    for frame, ns in latencies:
        per_frame.setdefault(frame, []).append(ns)
    result = {
        "backend": args.backend,
        "config": backend.config,
        "skipped": skipped,
        "mode": args.mode,
        "workers": workers,
        "threads_per_detector": args.threads,
        "cpus": cpus,
        "env": env,
        "frames": len(images),
        "calls": len(work),
        "setup": {"import_ms": import_ms, "construct_ms": construct_ms, "wall_ms": import_ms + construct_ms},
        "gc_disabled_while_timing": True,
        "wall_s": wall_ns / 1e9,
        "cpu_s": cpu_ns / 1e9,
        "frames_per_s": len(work) / (wall_ns / 1e9),
        "cpu_ms_per_frame": cpu_ns / 1e6 / len(work),
        "wall_ms_per_frame": wall_ns / 1e6 / len(work),
        "mean_busy_cpus": cpu_ns / wall_ns,
        "latency_ms": {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(values.mean()),
        },
        "frame_names": names,
        "frame_median_ms": [float(np.median(per_frame[i])) / 1e6 for i in range(len(images))],
        "rss_kib": {
            "before_backend": rss_before,
            "after_import": rss_after_import,
            "after_construct": rss_after_construct,
            "after_warmup": rss_after_warmup,
            "end": memory["VmRSS"],
            # VmHWM was reset at after_construct, so this peak covers warm-up and the timed passes only.
            # VmHWM can read a few pages below a sampled RSS: report the larger of the two.
            "vmhwm": memory["VmHWM"],
            "peak": max(memory["VmHWM"], rss_after_warmup, memory["VmRSS"]),
            "peak_minus_before": max(memory["VmHWM"], rss_after_warmup, memory["VmRSS"]) - rss_before,
            "peak_reset_ok": peak_reset,
        },
        "threads_at_start": threads_at_start,
        "threads_after_warmup": threads_after_warmup,
        "threads_end": thread_report(),
        "loadavg": os.getloadavg(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(public_text(json.dumps(result, indent=2, default=str)))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(resource_main())
