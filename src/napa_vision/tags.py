"""Draw AprilTags to print: as pixel arrays, PNG files, or SVG at an exact printed size."""

from __future__ import annotations

import ctypes
import struct
import zlib
from dataclasses import dataclass
from functools import cache

import numpy as np

from .napa_detector import _FamilyLayout, _library, families

TAG36H11_COUNT = 587


@dataclass(frozen=True)
class FamilyLayout:
    """A tag family's geometry, from the Napa core."""

    name: str
    ncodes: int
    nbits: int
    width: int  # cells across the detected square: its side is the tag size
    total_width: int  # cells across the printed tag
    reversed: bool  # the square prints white inside a black frame
    min_distance: int
    cells: tuple[tuple[int, int], ...]  # each code bit's cell, from the square's top-left, in the core's orientation
    codes: tuple[int, ...]


@cache
def family_layout(family: str = "tag36h11") -> FamilyLayout:
    """The layout and codes of one of ``families()``."""

    if family not in families():
        raise ValueError(f"family must be one of {', '.join(families())}")
    library = _library()
    info = _FamilyLayout()
    if library.napa_family_info(family.encode(), ctypes.byref(info)) != 0:
        raise RuntimeError(f"the Napa core has no layout for {family}")
    code = ctypes.c_uint64()
    codes = []
    for tag_id in range(info.ncodes):
        if library.napa_family_code(family.encode(), tag_id, ctypes.byref(code)) != 0:
            raise RuntimeError(f"the Napa core has no code for {family} tag {tag_id}")
        codes.append(code.value)
    cells = tuple((info.x[k], info.y[k]) for k in range(info.nbits))
    return FamilyLayout(
        family,
        info.ncodes,
        info.nbits,
        info.width,
        info.total_width,
        bool(info.reversed),
        info.min_distance,
        cells,
        tuple(codes),
    )


def _check_quiet_zone(quiet_zone_cells: int) -> None:
    if isinstance(quiet_zone_cells, bool) or not isinstance(quiet_zone_cells, int) or quiet_zone_cells < 0:
        raise ValueError("quiet_zone_cells must be a non-negative integer")


def tag_cells(tag_id: int, *, family: str = "tag36h11", quiet_zone_cells: int = 1) -> np.ndarray:
    """The tag's cells as a boolean array, ``True`` for white, oriented as the official artwork.

    That is the orientation of AprilRobotics' tag images and of printed FRC tags, which field layouts and
    Napa's pose solvers assume. (OpenCV's ``DICT_APRILTAG_36h11`` draws tags turned 180 degrees from it.)
    The printed pattern is surrounded by ``quiet_zone_cells`` of white on every side. For classic families
    (tag16h5, tag25h9, tag36h10, tag36h11) the pattern is the black square, a one-cell border around the
    payload, and detectors need at least one quiet cell around it. tagStandard, tagCircle, and tagCustom
    tags carry their own quiet ring inside the pattern, with data cells outside it.
    """

    layout = family_layout(family)
    if isinstance(tag_id, bool) or not isinstance(tag_id, int) or not 0 <= tag_id < layout.ncodes:
        raise ValueError(f"tag_id must be an integer from 0 to {layout.ncodes - 1}")
    _check_quiet_zone(quiet_zone_cells)
    total, width = layout.total_width, layout.width
    outer = (total - width) // 2
    # Upstream AprilTag's drawing: black, then a one-cell white ring (around the square for classic
    # families, as the square's own border for reversed ones), then every white data cell.
    pattern = np.zeros((total, total), dtype=bool)
    ring = width + (0 if layout.reversed else 2)
    start = (total - ring) // 2
    pattern[start, start : start + ring] = pattern[start + ring - 1, start : start + ring] = True
    pattern[start : start + ring, start] = pattern[start : start + ring, start + ring - 1] = True
    code = layout.codes[tag_id]
    for k, (x, y) in enumerate(layout.cells):
        pattern[y + outer, x + outer] = bool(code >> (layout.nbits - 1 - k) & 1)
    if layout.reversed:
        # Cells that are neither data, the square's border, nor the black frame around it carry nothing:
        # the official artwork leaves them blank (white on paper), such as tagCircle tags' corners.
        data = np.zeros_like(pattern)
        for x, y in layout.cells:
            data[y + outer, x + outer] = True
        index = np.arange(total) - outer
        x, y = np.meshgrid(index, index)
        border = np.maximum(np.abs(x - (width - 1) / 2), np.abs(y - (width - 1) / 2)) == (width - 1) / 2
        frame = np.maximum(np.abs(x - (width - 1) / 2), np.abs(y - (width - 1) / 2)) == (width + 1) / 2
        pattern[~(data | border | frame)] = True
    else:
        pattern = pattern[1:-1, 1:-1]  # the white ring is the quiet zone, drawn below
    cells = np.pad(pattern, quiet_zone_cells, constant_values=True)
    # The core stores codes in OpenCV's orientation, which fixes Napa's corner order; turn them to the official one.
    return np.rot90(cells, 2).copy()


def tag_image(tag_id: int, *, family: str = "tag36h11", cell_px: int = 1, quiet_zone_cells: int = 1) -> np.ndarray:
    """The tag as a ``uint8`` grayscale image (0 black, 255 white), ``cell_px`` pixels per cell.

    With the defaults, a tag36h11 tag is 10x10 pixels, like the official images; scale it up with
    ``cell_px`` rather than with an image editor's smoothing, which blurs the cells.
    """

    if isinstance(cell_px, bool) or not isinstance(cell_px, int) or cell_px < 1:
        raise ValueError("cell_px must be a positive integer")
    cells = tag_cells(tag_id, family=family, quiet_zone_cells=quiet_zone_cells)
    image = np.where(cells, np.uint8(255), np.uint8(0))
    return np.repeat(np.repeat(image, cell_px, axis=0), cell_px, axis=1)


def tag_svg(
    tag_id: int, *, family: str = "tag36h11", size_mm: float = 165.1, quiet_zone_cells: int = 1, label: bool = True
) -> str:
    """The tag as an SVG document whose tag-size square prints exactly ``size_mm`` millimetres wide.

    The default, 165.1 mm (6.5 in), is the FRC tag size. The tag size is the side of the square whose
    corners detectors report: the black square of a classic family, the white one inside the black frame
    of tagStandard, tagCircle, and tagCustom tags. Measure it, not the whole page, for pose solvers.
    ``label`` adds the family, ID, and size in small grey text below the tag, outside its quiet zone.
    """

    if not np.isfinite(size_mm) or size_mm <= 0:
        raise ValueError("size_mm must be a positive number")
    cells = tag_cells(tag_id, family=family, quiet_zone_cells=quiet_zone_cells)
    side = cells.shape[0]
    cell_mm = size_mm / family_layout(family).width
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
            f'text-anchor="middle" fill="#888">{family} · ID {tag_id} · {size_mm:g} mm</text>'
        )
    width_mm, height_mm = side * cell_mm, (side + caption) * cell_mm
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:.4f}mm" height="{height_mm:.4f}mm" '
        f'viewBox="0 0 {side} {side + caption:g}" shape-rendering="crispEdges">\n'
        f"<title>{family} ID {tag_id}</title>\n"
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
