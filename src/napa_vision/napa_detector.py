"""Napa's native tag36h11 detector, loaded from the bundled C library with ctypes."""

from __future__ import annotations

import ctypes
import os
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
    ]


class _Detection(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("hamming", ctypes.c_int),
        ("decision_margin", ctypes.c_float),
        ("corners", ctypes.c_double * 8),
    ]


class NativeLibraryError(RuntimeError):
    """Raised when the compiled Napa core library is unavailable."""


@cache
def _library() -> ctypes.CDLL:
    override = os.environ.get(_LIBRARY_ENV)
    candidates = [Path(override)] if override else []
    native_dir = Path(__file__).resolve().parent / "_native"
    for pattern in _LIBRARY_PATTERNS:
        candidates.extend(sorted(native_dir.glob(pattern)))
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


def native_available() -> bool:
    """Return whether the compiled Napa core can be loaded on this platform."""

    try:
        _library()
    except NativeLibraryError:
        return False
    return True


class NapaDetector:
    """Sparse-sampling tag36h11 detector implemented in portable C.

    The detector samples one pixel every three quarters of a cell (``3 * min_tag_px / 32``) in each
    direction, so every border crossing of a tag at least ``min_tag_px`` pixels wide contains a
    sample from the border's dark core, even with mild blur. Smaller tags may be missed. Corners
    are fitted at full resolution and returned in OpenCV's AprilTag corner order, in pixel-centre
    coordinates.

    One instance holds reusable buffers: serialize calls on each instance, and use one instance
    per thread or camera stream. The native call releases the GIL.
    """

    def __init__(
        self,
        *,
        min_tag_px: int = 32,
        max_hamming: int = 3,
        min_contrast: int = 20,
        min_border_cells: int = 22,
        sharpening: float = 0.25,
        max_detections: int = 64,
    ) -> None:
        if isinstance(min_tag_px, bool) or not isinstance(min_tag_px, int) or min_tag_px < 8:
            raise ValueError("min_tag_px must be an integer of at least 8")
        if isinstance(max_hamming, bool) or not isinstance(max_hamming, int) or not 0 <= max_hamming <= 3:
            raise ValueError("max_hamming must be an integer from 0 to 3")
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
        library = _library()
        handle = library.napa_create()
        if not handle:
            raise MemoryError("napa_create failed")
        self._library = library
        self._handle = ctypes.c_void_p(handle)
        self._finalizer = weakref.finalize(self, library.napa_destroy, self._handle)
        self._params = _Params(min_tag_px, max_hamming, min_contrast, min_border_cells, round(float(sharpening) * 100))
        self._results = (_Detection * max_detections)()
        self.min_tag_px = min_tag_px
        self.max_hamming = max_hamming

    def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
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
        observations: list[FiducialImageObservation] = []
        for result in self._results[:count]:
            corners = np.array(result.corners, dtype=np.float64).reshape(4, 2)
            observations.append(FiducialImageObservation(int(result.id), corners))
        return observations

    def close(self) -> None:
        """Release native buffers now instead of at garbage collection."""

        self._finalizer()


def _grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("AprilTag detection requires an 8-bit image")
    if image.ndim == 3 and image.shape[2] in {3, 4}:
        from . import _import_cv2

        cv2 = _import_cv2()
        conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, conversion)
    elif image.ndim != 2:
        raise ValueError("AprilTag detection requires a grayscale, BGR, or BGRA image")
    if image.strides[1] != 1 or image.strides[0] < image.shape[1]:
        image = np.ascontiguousarray(image)
    return image
