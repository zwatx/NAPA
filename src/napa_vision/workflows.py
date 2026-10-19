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
