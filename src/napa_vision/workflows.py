from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def calibration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate a camera for Napa from checkerboard photographs")
    parser.add_argument("images", help="Directory containing checkerboard images")
    parser.add_argument("--output", required=True, help="Destination calibration JSON")
    parser.add_argument("--columns", type=_positive_int, required=True, help="Checkerboard inner-corner columns")
    parser.add_argument("--rows", type=_positive_int, required=True, help="Checkerboard inner-corner rows")
    parser.add_argument(
        "--square-size-m", type=_positive_float, required=True, help="Checkerboard square size in meters"
    )
    parser.add_argument("--minimum-views", type=_positive_int, default=8, help="Minimum complete checkerboard images")
    return parser


def calibration_main(argv: Sequence[str] | None = None) -> int:
    from .calibration import (
        CalibrationError,
        CheckerboardSpec,
        calibrate_checkerboard_images,
        discover_checkerboard_images,
        save_calibration_data,
    )

    args = calibration_parser().parse_args(argv)
    try:
        images = discover_checkerboard_images(args.images)
        calibration = calibrate_checkerboard_images(
            images,
            CheckerboardSpec(args.columns, args.rows, args.square_size_m),
            minimum_views=args.minimum_views,
        )
        save_calibration_data(args.output, calibration)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"napa-calibrate: error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "output": str(Path(args.output).expanduser().resolve()),
        "image_size": calibration.image_size,
        "views_used": calibration.views_used,
        "rejected_images": len(calibration.rejected_images),
        "rms_reprojection_error_px": calibration.rms_reprojection_error_px,
        "mean_reprojection_error_px": calibration.mean_reprojection_error_px,
        "provenance": calibration.provenance.to_dict(),
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


def _tag_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in value.split(","):
        first, dash, last = part.strip().partition("-")
        try:
            start = int(first)
            stop = int(last) if dash else start
        except ValueError:
            raise argparse.ArgumentTypeError(f"{part!r} is not an ID or a range like 0-9") from None
        if stop < start:
            raise argparse.ArgumentTypeError(f"{part!r} is an empty range")
        ids.extend(range(start, stop + 1))
    return ids


def tag_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write tag36h11 tags to print, as SVG (exact printed size) or PNG files",
        epilog="Example: napa-tag 0-7 --size-mm 165.1 --output tags/",
    )
    parser.add_argument("ids", type=_tag_ids, nargs="+", help="Tag IDs or ranges, e.g. 3 or 0-9 or 1,4,7-8")
    parser.add_argument("--output", default=".", help="Directory to write into (created if missing)")
    parser.add_argument("--family", default="tag36h11", help="Tag family (default: tag36h11, the FRC family)")
    parser.add_argument("--format", choices=("svg", "png"), default="svg", help="File format (default: svg)")
    parser.add_argument(
        "--size-mm", type=_positive_float, default=165.1, help="SVG: printed black-square side (default: 165.1, FRC)"
    )
    parser.add_argument("--cell-px", type=_positive_int, default=40, help="PNG: pixels per cell (default: 40)")
    parser.add_argument("--quiet-zone", type=int, default=1, help="White cells around the black square (default: 1)")
    parser.add_argument("--no-label", action="store_true", help="SVG: leave out the ID caption")
    return parser


def tag_main(argv: Sequence[str] | None = None) -> int:
    from .tags import family_layout, png_bytes, tag_image, tag_svg

    args = tag_parser().parse_args(argv)
    output = Path(args.output).expanduser()
    written = []
    try:
        output.mkdir(parents=True, exist_ok=True)
        digits = len(str(family_layout(args.family).ncodes - 1))
        for tag_id in dict.fromkeys(tag_id for group in args.ids for tag_id in group):
            path = output / f"{args.family}-{tag_id:0{digits}d}.{args.format}"
            if args.format == "svg":
                path.write_text(
                    tag_svg(
                        tag_id,
                        family=args.family,
                        size_mm=args.size_mm,
                        quiet_zone_cells=args.quiet_zone,
                        label=not args.no_label,
                    ),
                    encoding="utf-8",
                )
            else:
                image = tag_image(tag_id, family=args.family, cell_px=args.cell_px, quiet_zone_cells=args.quiet_zone)
                path.write_bytes(png_bytes(image))
            written.append(str(path))
    except (OSError, ValueError) as exc:
        print(f"napa-tag: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"written": written}, separators=(",", ":")))
    return 0
