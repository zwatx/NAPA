"""Napa's native tag36h11 detector, loaded from the bundled C library with ctypes."""

from __future__ import annotations

import ctypes
import os
import warnings
import weakref
from functools import cache
from pathlib import Path

import numpy as np

from .types import FiducialImageObservation

_LIBRARY_ENV = "NAPA_CORE_LIBRARY"
_LIBRARY_PATTERNS = ("_napa_core*.so", "_napa_core*.pyd", "_napa_core*.dylib")
_MAX_SIDE = 30000


class _Params(ctypes.Structure):
    _fields_ = [
        ("min_tag_px", ctypes.c_int),
        ("max_hamming", ctypes.c_int),
        ("min_contrast", ctypes.c_int),
        ("min_border_cells", ctypes.c_int),
        ("sharpening_x100", ctypes.c_int),
        ("split_touching", ctypes.c_int),
    ]


class _Detection(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("hamming", ctypes.c_int),
        ("decision_margin", ctypes.c_float),
        ("corners", ctypes.c_double * 8),
    ]


class _FamilyLayout(ctypes.Structure):
    _fields_ = [
        ("ncodes", ctypes.c_int),
        ("nbits", ctypes.c_int),
        ("width", ctypes.c_int),
        ("total_width", ctypes.c_int),
        ("reversed", ctypes.c_int),
        ("min_distance", ctypes.c_int),
        ("max_hamming", ctypes.c_int),
        ("recommended_hamming", ctypes.c_int),
        ("x", ctypes.c_int8 * 64),
        ("y", ctypes.c_int8 * 64),
    ]


class NativeLibraryError(RuntimeError):
    """Raised when the compiled Napa core library is unavailable."""


@cache
def _library() -> ctypes.CDLL:
    override = os.environ.get(_LIBRARY_ENV)
    native_dir = Path(__file__).resolve().parent / "_native"
    if override:
        # An explicit override is used or fails loudly; falling back would hide a broken build.
        candidates = [Path(override)]
    else:
        candidates = [path for pattern in _LIBRARY_PATTERNS for path in sorted(native_dir.glob(pattern))]
    errors: list[str] = []
    for candidate in candidates:
        try:
            library = ctypes.CDLL(str(candidate))
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        library.napa_create.restype = ctypes.c_void_p
        library.napa_create.argtypes = []
        library.napa_destroy.restype = None
        library.napa_destroy.argtypes = [ctypes.c_void_p]
        library.napa_create_family.restype = ctypes.c_void_p
        library.napa_create_family.argtypes = [ctypes.c_char_p]
        library.napa_family_count.restype = ctypes.c_int
        library.napa_family_count.argtypes = []
        library.napa_family_name.restype = ctypes.c_char_p
        library.napa_family_name.argtypes = [ctypes.c_int]
        library.napa_family_info.restype = ctypes.c_int
        library.napa_family_info.argtypes = [ctypes.c_char_p, ctypes.POINTER(_FamilyLayout)]
        library.napa_family_code.restype = ctypes.c_int
        library.napa_family_code.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
        library.napa_detect.restype = ctypes.c_int
        library.napa_detect.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_Params),
            ctypes.POINTER(_Detection),
            ctypes.c_int,
        ]
        return library
    detail = "; ".join(errors) if errors else f"no compiled library in {native_dir}"
    raise NativeLibraryError(
        f"The Napa core library is unavailable ({detail}). Reinstall napa-vision with a C compiler "
        f"available, or set {_LIBRARY_ENV} to a built library."
    )


def families() -> tuple[str, ...]:
    """The tag families Napa detects, tag36h11 (the default) first."""

    library = _library()
    return tuple(library.napa_family_name(index).decode() for index in range(library.napa_family_count()))


def native_available() -> bool:
    """Return whether the compiled Napa core can be loaded on this platform."""

    try:
        _library()
    except NativeLibraryError:
        return False
    return True


class NapaDetector:
    """Sparse-sampling AprilTag detector implemented in portable C, for one tag family.

    ``family`` is any of ``families()``; the default, tag36h11, is the FRC family. The detector samples
    one pixel every three quarters of a cell in each direction, so every border crossing of a tag at
    least ``min_tag_px`` pixels wide contains a sample from the border's dark core, even with mild blur.
    ``min_tag_px`` is the side of the square whose corners are reported (the tag size for poses). Smaller
    tags may be missed. Corners are fitted at full resolution and returned in OpenCV's AprilTag corner
    order, in pixel-centre coordinates.

    ``split_touching=True`` also finds tags whose border touches another dark shape at a corner, as in
    AprilGrid calibration boards and tags set in a checkerboard: such a tag and its neighbours form one blob,
    which isn't tag-shaped. The extra tags come after the others, which are unchanged. Splitting roughly
    doubles detection time on natural scenes, so leave it off unless tags touch other shapes.

    One instance holds reusable buffers: serialize calls on each instance, and use one instance
    per thread or camera stream. The native call releases the GIL.
    """

    def __init__(
        self,
        *,
        family: str = "tag36h11",
        min_tag_px: int = 32,
        max_hamming: int | None = None,
        min_contrast: int = 20,
        min_border_cells: int = 22,
        sharpening: float = 0.25,
        max_detections: int = 64,
        split_touching: bool = False,
    ) -> None:
        if isinstance(min_tag_px, bool) or not isinstance(min_tag_px, int) or not 8 <= min_tag_px <= _MAX_SIDE:
            raise ValueError(f"min_tag_px must be an integer from 8 to {_MAX_SIDE}")
        if max_hamming is not None and (
            isinstance(max_hamming, bool) or not isinstance(max_hamming, int) or not 0 <= max_hamming <= 3
        ):
            raise ValueError("max_hamming must be None or an integer from 0 to 3")
        if isinstance(min_contrast, bool) or not isinstance(min_contrast, int) or not 1 <= min_contrast <= 255:
            raise ValueError("min_contrast must be an integer from 1 to 255")
        if (
            isinstance(min_border_cells, bool)
            or not isinstance(min_border_cells, int)
            or not 0 <= min_border_cells <= 28
        ):
            raise ValueError("min_border_cells must be an integer from 0 to 28")
        if not 0.0 <= float(sharpening) <= 1.0:
            raise ValueError("sharpening must be between 0 and 1")
        if isinstance(max_detections, bool) or not isinstance(max_detections, int) or max_detections < 1:
            raise ValueError("max_detections must be a positive integer")
        if not isinstance(split_touching, bool):
            raise ValueError("split_touching must be True or False")
        library = _library()
        if not isinstance(family, str) or family not in families():
            raise ValueError(f"family must be one of {', '.join(families())}")
        layout = _FamilyLayout()
        library.napa_family_info(family.encode(), ctypes.byref(layout))
        # The family's default keeps chance matches as rare as tag36h11's at 3 bits; larger values are capped.
        max_hamming = layout.recommended_hamming if max_hamming is None else min(max_hamming, layout.max_hamming)
        handle = library.napa_create_family(family.encode())
        if not handle:
            raise MemoryError("napa_create_family failed")
        self.family = family
        self._library = library
        self._handle = ctypes.c_void_p(handle)
        self._finalizer = weakref.finalize(self, library.napa_destroy, self._handle)
        self._params = _Params(
            min_tag_px, max_hamming, min_contrast, min_border_cells, round(float(sharpening) * 100), split_touching
        )
        self._results = (_Detection * max_detections)()
        self._rows = np.ctypeslib.as_array(self._results)  # a numpy view of the same buffer
        self.min_tag_px = min_tag_px
        self.max_hamming = max_hamming
        self.split_touching = split_touching

    def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
        if not self._finalizer.alive:
            raise RuntimeError("NapaDetector.detect() called after close()")
        gray = _grayscale(image)
        if gray.shape[0] > _MAX_SIDE or gray.shape[1] > _MAX_SIDE:
            raise ValueError(f"frames larger than {_MAX_SIDE} pixels per side are not supported")
        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return []
        count = self._library.napa_detect(
            self._handle,
            gray.ctypes.data,
            gray.shape[1],
            gray.shape[0],
            gray.strides[0],
            ctypes.byref(self._params),
            self._results,
            len(self._results),
        )
        if count < 0:
            raise RuntimeError(f"napa_detect failed with status {count}")
        if count == len(self._results):
            warnings.warn(
                f"a frame filled all {count} results (max_detections), so any further tags in it were dropped; "
                "pass a larger max_detections to keep them",
                RuntimeWarning,
                stacklevel=2,
            )
        if count == 0:
            return []
        rows = self._rows[:count]
        corners = rows["corners"].reshape(count, 4, 2).copy()
        return [
            FiducialImageObservation(tag_id, tag_corners, hamming=hamming, decision_margin=margin, family=self.family)
            for tag_id, tag_corners, hamming, margin in zip(
                rows["id"].tolist(), corners, rows["hamming"].tolist(), rows["decision_margin"].tolist(), strict=True
            )
        ]

    def close(self) -> None:
        """Release native buffers now instead of at garbage collection."""

        self._finalizer()


def _grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("AprilTag detection requires an 8-bit image")
    if image.ndim == 3 and image.shape[2] in {3, 4}:
        image = _bgr_to_gray(image)
    elif image.ndim != 2:
        raise ValueError("AprilTag detection requires a grayscale, BGR, or BGRA image")
    if image.strides[1] != 1 or image.strides[0] < image.shape[1]:
        image = np.ascontiguousarray(image)
    return image


def _bgr_to_gray(image: np.ndarray) -> np.ndarray:
    """BGR(A) to grayscale with BT.601 weights (within one grey level of OpenCV, which is used when installed)."""

    try:
        import cv2
    except ImportError:
        gray = np.empty(image.shape[:2], dtype=np.uint8)
        if gray.size == 0:
            return gray
        # Keep the two uint32 scratch blocks within 256 KiB for ordinary frame widths,
        # instead of allocating full-resolution channel arrays and arithmetic temporaries.
        rows = min(image.shape[0], max(1, 32768 // image.shape[1]))
        total = np.empty((rows, image.shape[1]), dtype=np.uint32)
        scratch = np.empty_like(total)
        for y in range(0, image.shape[0], rows):
            block = image[y : y + rows]
            acc, tmp = total[: len(block)], scratch[: len(block)]
            np.multiply(block[..., 0], 1868, out=acc, dtype=np.uint32)
            np.multiply(block[..., 1], 9617, out=tmp, dtype=np.uint32)
            acc += tmp
            np.multiply(block[..., 2], 4899, out=tmp, dtype=np.uint32)
            acc += tmp
            acc += 1 << 13
            acc >>= 14
            gray[y : y + rows] = acc
        return gray
    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(np.ascontiguousarray(image), conversion)
