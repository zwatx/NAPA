"""Draw tag36h11 tags to print: as pixel arrays, PNG files, or SVG at an exact printed size."""

from __future__ import annotations

import ctypes
import struct
import zlib
from functools import cache

import numpy as np

from .napa_detector import _library

TAG36H11_COUNT = 587


@cache
def _codes() -> tuple[int, ...]:
    function = _library().napa_tag36h11_code
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    code = ctypes.c_uint64()
    codes = []
    for tag_id in range(TAG36H11_COUNT):
        if function(tag_id, ctypes.byref(code)) != 0:
            raise RuntimeError(f"the Napa core has no code for tag {tag_id}")
        codes.append(code.value)
    return tuple(codes)


def _check_id(tag_id: int) -> None:
    if isinstance(tag_id, bool) or not isinstance(tag_id, int) or not 0 <= tag_id < TAG36H11_COUNT:
        raise ValueError(f"tag_id must be an integer from 0 to {TAG36H11_COUNT - 1}")


def _check_quiet_zone(quiet_zone_cells: int) -> None:
    if isinstance(quiet_zone_cells, bool) or not isinstance(quiet_zone_cells, int) or quiet_zone_cells < 0:
        raise ValueError("quiet_zone_cells must be a non-negative integer")


def tag_cells(tag_id: int, *, quiet_zone_cells: int = 1) -> np.ndarray:
    """The tag's cells as a boolean array, ``True`` for white, in the official artwork's orientation.

    The black square is 8 cells across (a one-cell black border around the 6x6 payload), surrounded by
    ``quiet_zone_cells`` of white on every side. Napa and other AprilTag detectors need at least one.
    """

    _check_id(tag_id)
    _check_quiet_zone(quiet_zone_cells)
    code = _codes()[tag_id]
    payload = np.array([(code >> (35 - bit)) & 1 for bit in range(36)], dtype=bool).reshape(6, 6)
    cells = np.ones((8 + 2 * quiet_zone_cells,) * 2, dtype=bool)
    cells[quiet_zone_cells : quiet_zone_cells + 8, quiet_zone_cells : quiet_zone_cells + 8] = False
    cells[quiet_zone_cells + 1 : quiet_zone_cells + 7, quiet_zone_cells + 1 : quiet_zone_cells + 7] = payload
    return cells


def tag_image(tag_id: int, *, cell_px: int = 1, quiet_zone_cells: int = 1) -> np.ndarray:
    """The tag as a ``uint8`` grayscale image (0 black, 255 white), ``cell_px`` pixels per cell.

    With the defaults it is 10x10 pixels, like the official tag36h11 images; scale it up with
    ``cell_px`` rather than with an image editor's smoothing, which blurs the cells.
    """

    if isinstance(cell_px, bool) or not isinstance(cell_px, int) or cell_px < 1:
        raise ValueError("cell_px must be a positive integer")
    cells = tag_cells(tag_id, quiet_zone_cells=quiet_zone_cells)
    image = np.where(cells, np.uint8(255), np.uint8(0))
    return np.repeat(np.repeat(image, cell_px, axis=0), cell_px, axis=1)


def tag_svg(tag_id: int, *, size_mm: float = 165.1, quiet_zone_cells: int = 1, label: bool = True) -> str:
    """The tag as an SVG document whose black square prints exactly ``size_mm`` millimetres wide.

    The default, 165.1 mm (6.5 in), is the FRC tag size. Measure this black square, not the whole page,
    for the tag size you give pose solvers. ``label`` adds the family and ID in small grey text below
    the tag, outside its quiet zone.
    """

    if not np.isfinite(size_mm) or size_mm <= 0:
        raise ValueError("size_mm must be a positive number")
    cells = tag_cells(tag_id, quiet_zone_cells=quiet_zone_cells)
    side = cells.shape[0]
    cell_mm = size_mm / 8.0
    caption = 1.0 if label else 0.0  # cells reserved below the quiet zone for the label
    rects = []
    for row in range(side):
        column = 0
        while column < side:
            if cells[row, column]:
                column += 1
                continue
            start = column
            while column < side and not cells[row, column]:
                column += 1
            rects.append(f'<rect x="{start}" y="{row}" width="{column - start}" height="1"/>')
    text = ""
    if label:
        text = (
            f'<text x="{side / 2:g}" y="{side + 0.65:g}" font-family="sans-serif" font-size="0.5" '
            f'text-anchor="middle" fill="#888">tag36h11 · ID {tag_id} · {size_mm:g} mm</text>'
        )
    width_mm, height_mm = side * cell_mm, (side + caption) * cell_mm
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:.4f}mm" height="{height_mm:.4f}mm" '
        f'viewBox="0 0 {side} {side + caption:g}" shape-rendering="crispEdges">\n'
        f"<title>tag36h11 ID {tag_id}</title>\n"
        f'<rect width="{side}" height="{side + caption:g}" fill="#fff"/>\n'
        f'<g fill="#000">{"".join(rects)}</g>\n'
        f"{text}\n"
        "</svg>\n"
    )


def png_bytes(image: np.ndarray) -> bytes:
    """Encode a ``uint8`` grayscale image as PNG, without any imaging library."""

    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8 or image.ndim != 2 or 0 in image.shape:
        raise ValueError("png_bytes expects a non-empty uint8 grayscale image")
    height, width = image.shape
    rows = np.zeros((height, width + 1), np.uint8)  # each row starts with filter type 0
    rows[:, 1:] = image

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 9))
        + chunk(b"IEND", b"")
    )
