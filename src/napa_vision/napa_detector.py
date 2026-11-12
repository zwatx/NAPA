"""Napa's native tag36h11 detector, loaded from the bundled C library with ctypes."""

from __future__ import annotations

import ctypes
import os
import warnings
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from .types import FiducialImageObservation

_LIBRARY_ENV = "NAPA_CORE_LIBRARY"
_LIBRARY_PATTERNS = ("_napa_core*.so", "_napa_core*.pyd", "_napa_core*.dylib")
_MAX_SIDE = 30000
_TAG_SHAPES = {"tall": 1, "wide": 2}  # napa_core.h NAPA_TAG_SHAPE_*
_OPTION_SOFT_DECODE = 1
_OPTION_ALIGN_CORNERS = 2


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
        if hasattr(library, "napa_detect_shadows"):
            library.napa_detect_shadows.restype = ctypes.c_int
            library.napa_detect_shadows.argtypes = library.napa_detect.argtypes
        if hasattr(library, "napa_detect_shadows_configured"):
            library.napa_detect_shadows_configured.restype = ctypes.c_int
            library.napa_detect_shadows_configured.argtypes = [*library.napa_detect.argtypes, ctypes.c_int]
        if hasattr(library, "napa_detect_soft"):
            library.napa_detect_soft.restype = ctypes.c_int
            library.napa_detect_soft.argtypes = [*library.napa_detect.argtypes, ctypes.c_int, ctypes.c_int]
        if hasattr(library, "napa_set_camera_gamma"):
            library.napa_set_camera_gamma.restype = ctypes.c_int
            library.napa_set_camera_gamma.argtypes = [ctypes.c_void_p, ctypes.c_double]
        if hasattr(library, "napa_set_tag_shape"):
            library.napa_set_tag_shape.restype = ctypes.c_int
            library.napa_set_tag_shape.argtypes = [ctypes.c_void_p, ctypes.c_int]
            library.napa_border_bias_cells.restype = ctypes.c_int
            library.napa_border_bias_cells.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(_Params),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
        if hasattr(library, "napa_detect_options"):
            library.napa_detect_options.restype = ctypes.c_int
            library.napa_detect_options.argtypes = [
                *library.napa_detect.argtypes,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
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
    order, in pixel-centre coordinates. The default is 22; ``min_tag_px=32`` runs at about 0.5.1's speed and finds
    fewer small tags.

    ``split_touching=True`` also finds tags whose border touches another dark shape at a corner, as in
    AprilGrid calibration boards and tags set in a checkerboard: such a tag and its neighbours form one blob,
    which isn't tag-shaped. The extra tags come after the others, which are unchanged. It adds about 15-30% to
    frames without touching tags, and boards take about 2.5-3x as long.

    ``recover_shadows=True`` (experimental, off by default) adds a lower-threshold proposal pass for tags whose
    border merges into a dim background, as in shade or against a bright window. Ordinary results stay first,
    exactly as they are without recovery. Additions require zero corrected bits. Short-code families with high
    chance-match rates skip recovery. Expect extra full-frame work, incomplete shadow recovery, and no guarantee
    against false positives. ``shadow_min_tag_px`` optionally raises the recovery-only sampling minimum (from
    ``min_tag_px`` through 30000) to trade sensitivity for speed; ``None`` keeps the full sensitivity. A cheap coarse
    pass for larger tags is ``recover_shadows=True, shadow_min_tag_px=64``. Recovered outputs need not be a nested
    subset.

    ``soft_decode=True`` (experimental, tag36h11 only, off by default) adds tiny tags, about 8 to 19 pixels across,
    whose bits can't be read one by one, by comparing the whole tag with every code. They follow the ordinary
    results, which are unchanged, and have ``soft_decoded=True`` and ``hamming=None``. Structured non-tags can pass,
    so treat them as weaker evidence; see the detection guide. It only helps at about ``min_tag_px=16``, apart from
    tags with a flat patch blocking part of the payload, which it can also read at any size.

    ``camera_gamma`` (experimental, tag36h11 only, unset by default) is the gamma of a camera that outputs
    gamma-encoded video, such as most USB and sRGB cameras. Corners are then refined in linear light, which removes
    most of the error blur causes on those cameras. Leave it unset for a linear machine-vision camera, where a value
    makes corners worse. ``estimate_camera_gamma`` reads it from a recording; see the detection guide.

    ``tag_shape="tall"`` samples every other row while searching, for tags turned about the vertical axis, which look
    narrow and tall; ``"wide"`` samples every other column instead, for a camera mounted on its side. It needs
    ``min_tag_px`` up to 31 for tag36h11. At 16 it finds most of what ``min_tag_px=16`` adds over 22 at a fraction of
    its extra time; at 22 it takes about as long as ``min_tag_px=32`` and finds more of the narrow tags. The wrong
    direction is worse than not setting it; see the detection guide.

    ``align_corners=True`` (experimental, tag36h11 only, off by default) moves the corners of blurred tags onto
    their own picture after decoding. It mainly helps motion blur, costs time on every blurred tag, and has no check
    on a real camera; see the detection guide.

    One instance holds reusable buffers: serialize calls on each instance, and use one instance
    per thread or camera stream. The native call releases the GIL.
    """

    def __init__(
        self,
        *,
        family: str = "tag36h11",
        min_tag_px: int = 22,
        max_hamming: int | None = None,
        min_contrast: int = 20,
        min_border_cells: int = 22,
        sharpening: float = 0.25,
        max_detections: int = 64,
        split_touching: bool = False,
        recover_shadows: bool = False,
        shadow_min_tag_px: int | None = None,
        soft_decode: bool = False,
        align_corners: bool = False,
        camera_gamma: float | None = None,
        tag_shape: str | None = None,
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
        if not isinstance(recover_shadows, bool):
            raise ValueError("recover_shadows must be True or False")
        if shadow_min_tag_px is not None:
            if (
                isinstance(shadow_min_tag_px, bool)
                or not isinstance(shadow_min_tag_px, int)
                or not min_tag_px <= shadow_min_tag_px <= _MAX_SIDE
            ):
                raise ValueError(f"shadow_min_tag_px must be an integer from min_tag_px to {_MAX_SIDE}, or None")
            if not recover_shadows:
                raise ValueError("shadow_min_tag_px requires recover_shadows=True")
        if not isinstance(soft_decode, bool):
            raise ValueError("soft_decode must be True or False")
        if soft_decode and family != "tag36h11":
            raise ValueError("soft_decode is calibrated for tag36h11 only")
        if not isinstance(align_corners, bool):
            raise ValueError("align_corners must be True or False")
        if align_corners and family != "tag36h11":
            raise ValueError("align_corners is calibrated for tag36h11 only")
        if camera_gamma is not None:
            if (
                isinstance(camera_gamma, bool)
                or not isinstance(camera_gamma, (int, float))
                or not 1.0 <= camera_gamma <= 4.0
            ):
                raise ValueError("camera_gamma must be None or a number from 1 to 4")
            if family != "tag36h11":
                raise ValueError("camera_gamma is calibrated for tag36h11 only")
        if tag_shape not in (None, "tall", "wide"):
            raise ValueError('tag_shape must be None, "tall" or "wide"')
        library = _library()
        if recover_shadows and not hasattr(library, "napa_detect_shadows"):
            raise NativeLibraryError("recover_shadows requires an updated native library; rebuild napa-vision")
        self._detect = library.napa_detect_shadows if recover_shadows else library.napa_detect
        self._shadow_args = ()
        if shadow_min_tag_px is not None:
            if not hasattr(library, "napa_detect_shadows_configured"):
                raise NativeLibraryError("shadow_min_tag_px requires an updated native library; rebuild napa-vision")
            self._detect = library.napa_detect_shadows_configured
            self._shadow_args = (shadow_min_tag_px,)
        if soft_decode or align_corners:
            if not hasattr(library, "napa_detect_options"):
                raise NativeLibraryError(
                    "soft_decode and align_corners require an updated native library; rebuild napa-vision"
                )
            options = (_OPTION_SOFT_DECODE if soft_decode else 0) | (_OPTION_ALIGN_CORNERS if align_corners else 0)
            self._shadow_args = (int(recover_shadows), shadow_min_tag_px or 0, options)
            self._detect = library.napa_detect_options
        if not isinstance(family, str) or family not in families():
            raise ValueError(f"family must be one of {', '.join(families())}")
        layout = _FamilyLayout()
        library.napa_family_info(family.encode(), ctypes.byref(layout))
        if tag_shape is not None and min_tag_px * 3 // (4 * layout.width) > 2:
            # Sampling every pixel or every other one across: coarser loses more than it saves.
            limit = (12 * layout.width - 1) // 3
            raise ValueError(f"tag_shape needs fine sampling: min_tag_px from 8 to {limit} for {family}")
        # The family's default keeps chance matches as rare as tag36h11's at 3 bits; larger values are capped.
        max_hamming = layout.recommended_hamming if max_hamming is None else min(max_hamming, layout.max_hamming)
        handle = library.napa_create_family(family.encode())
        if not handle:
            raise MemoryError("napa_create_family failed")
        self.family = family
        self._library = library
        self._handle = ctypes.c_void_p(handle)
        if camera_gamma is not None:
            if not hasattr(library, "napa_set_camera_gamma"):
                library.napa_destroy(self._handle)
                raise NativeLibraryError("camera_gamma requires an updated native library; rebuild napa-vision")
            library.napa_set_camera_gamma(self._handle, float(camera_gamma))
        if tag_shape is not None:
            if not hasattr(library, "napa_set_tag_shape"):
                library.napa_destroy(self._handle)
                raise NativeLibraryError("tag_shape requires an updated native library; rebuild napa-vision")
            library.napa_set_tag_shape(self._handle, _TAG_SHAPES[tag_shape])
        self._finalizer = weakref.finalize(self, library.napa_destroy, self._handle)
        self._params = _Params(
            min_tag_px, max_hamming, min_contrast, min_border_cells, round(float(sharpening) * 100), split_touching
        )
        self._results = (_Detection * max_detections)()
        self._rows = np.ctypeslib.as_array(self._results)  # a numpy view of the same buffer
        self.min_tag_px = min_tag_px
        self.max_hamming = max_hamming
        self.split_touching = split_touching
        self.recover_shadows = recover_shadows
        self.shadow_min_tag_px = shadow_min_tag_px
        self.soft_decode = soft_decode
        self.align_corners = align_corners
        self.camera_gamma = camera_gamma
        self.tag_shape = tag_shape

    def detect(self, image: np.ndarray) -> list[FiducialImageObservation]:
        if not self._finalizer.alive:
            raise RuntimeError("NapaDetector.detect() called after close()")
        gray = _grayscale(image)
        if gray.shape[0] > _MAX_SIDE or gray.shape[1] > _MAX_SIDE:
            raise ValueError(f"frames larger than {_MAX_SIDE} pixels per side are not supported")
        if gray.shape[0] < 16 or gray.shape[1] < 16:
            return []
        count = self._detect(
            self._handle,
            gray.ctypes.data,
            gray.shape[1],
            gray.shape[0],
            gray.strides[0],
            ctypes.byref(self._params),
            self._results,
            len(self._results),
            *self._shadow_args,
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
        if self.soft_decode:
            return [
                FiducialImageObservation(
                    tag_id,
                    tag_corners,
                    hamming=None if hamming < 0 else hamming,
                    decision_margin=margin,
                    family=self.family,
                    soft_decoded=hamming < 0,
                )
                for tag_id, tag_corners, hamming, margin in zip(
                    rows["id"].tolist(),
                    corners,
                    rows["hamming"].tolist(),
                    rows["decision_margin"].tolist(),
                    strict=True,
                )
            ]
        return [
            FiducialImageObservation(tag_id, tag_corners, hamming=hamming, decision_margin=margin, family=self.family)
            for tag_id, tag_corners, hamming, margin in zip(
                rows["id"].tolist(), corners, rows["hamming"].tolist(), rows["decision_margin"].tolist(), strict=True
            )
        ]

    def close(self) -> None:
        """Release native buffers now instead of at garbage collection."""

        self._finalizer()


@dataclass(frozen=True)
class CameraGammaEstimate:
    """A camera's gamma read from its tags.

    ``gamma`` is ``None`` when fewer than the required frames showed a usable tag, 1.0 when the camera reads as
    linear (leave ``camera_gamma`` unset), and otherwise the value to pass. ``low`` and ``high`` bound the middle
    80% of estimates from frames resampled with replacement, which reflects scatter between frames only; on a real
    camera the estimate can sit about 10% below the true value. ``frames`` and ``tags`` count the frames and tag
    sightings used at gamma 1, and ``bias`` maps each assumed gamma to the mean border-width bias, in cells, with it
    undone.
    """

    gamma: float | None
    low: float | None
    high: float | None
    frames: int
    tags: int
    bias: dict[float, float]


_GAMMA_GRID = (1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1)
_LINEAR_BELOW = 1.15
_MAX_EDGE_CELLS = 1.0  # blur wider than this makes the border read narrow at any gamma


def _bias_crossing(means: dict[float, float]) -> float | None:
    seen = sorted(means)
    if means[seen[0]] <= 0:
        return 1.0
    for low, high in zip(seen, seen[1:], strict=False):
        if means[low] > 0 >= means[high]:
            crossing = low + (high - low) * means[low] / (means[low] - means[high])
            return 1.0 if crossing < _LINEAR_BELOW else float(crossing)
    return None


def estimate_camera_gamma(
    frames: Iterable[np.ndarray],
    *,
    min_frames: int = 30,
    **detector_options: Any,
) -> CameraGammaEstimate:
    """Estimate a camera's gamma from its frames, for ``NapaDetector(camera_gamma=...)``.

    A tag's black border is one cell wide. On a camera that encodes light with a curve, blur makes it read narrow, and
    undoing the right gamma removes that. Each frame is detected with gammas 1.0 to 3.1 undone (steps of 0.3), and the
    result is where the mean border bias crosses zero. Tags whose edges are blurred over more than a cell are left out,
    since their border reads narrow at any gamma. Fewer than ``min_frames`` frames with a usable tag give ``None``.

    Use frames from one camera at its usual settings, showing tag36h11 tags. Mild blur helps, since a sharp border
    looks the same at every gamma. It's an estimate from the tags, not a calibration of the camera.
    """

    library = _library()
    if not hasattr(library, "napa_border_bias_cells"):
        raise NativeLibraryError("estimate_camera_gamma requires an updated native library; rebuild napa-vision")
    if isinstance(min_frames, bool) or not isinstance(min_frames, int) or min_frames < 1:
        raise ValueError("min_frames must be a positive integer")
    detector = NapaDetector(family="tag36h11", **detector_options)
    grid = _GAMMA_GRID
    tables = [np.rint(255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8) for g in grid]
    sums, counts = [], []  # per frame with a usable tag: bias sum and tag count at each gamma
    corners = (ctypes.c_double * 8)()
    bias, edge = ctypes.c_double(), ctypes.c_double()
    try:
        for frame in frames:
            gray = _grayscale(frame)
            frame_sums, frame_counts = np.zeros(len(grid)), np.zeros(len(grid))
            for index, table in enumerate(tables):
                undone = gray if index == 0 else np.ascontiguousarray(table[gray])
                for observation in detector.detect(undone):
                    ctypes.memmove(corners, np.asarray(observation.corners, np.float64).tobytes(), 64)
                    status = library.napa_border_bias_cells(
                        detector._handle,
                        undone.ctypes.data,
                        undone.shape[1],
                        undone.shape[0],
                        undone.strides[0],
                        ctypes.byref(detector._params),
                        corners,
                        ctypes.byref(bias),
                        ctypes.byref(edge),
                    )
                    if status == 0 and edge.value <= _MAX_EDGE_CELLS:
                        frame_sums[index] += bias.value
                        frame_counts[index] += 1
            if frame_counts[0]:
                sums.append(frame_sums)
                counts.append(frame_counts)
    finally:
        detector.close()
    sums, counts = np.array(sums).reshape(-1, len(grid)), np.array(counts).reshape(-1, len(grid))

    def crossing(rows: np.ndarray) -> float | None:
        total, seen = sums[rows].sum(0), counts[rows].sum(0)
        return _bias_crossing({g: total[i] / seen[i] for i, g in enumerate(grid) if seen[i]})

    total, seen = sums.sum(0), counts.sum(0)
    means = {g: float(total[i] / seen[i]) for i, g in enumerate(grid) if seen[i]}
    used, tags = len(sums), int(seen[0]) if len(seen) else 0
    if used < min_frames:
        return CameraGammaEstimate(None, None, None, used, tags, means)
    rng = np.random.default_rng(0)
    resampled = [crossing(rng.integers(0, used, used)) for _ in range(200)]
    spread = [value for value in resampled if value is not None]
    low, high = (float(v) for v in np.percentile(spread, [10, 90])) if spread else (None, None)
    return CameraGammaEstimate(crossing(np.arange(used)), low, high, used, tags, means)


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
