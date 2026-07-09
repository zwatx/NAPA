from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .calibration import (
    CalibrationError,
    CheckerboardSpec,
    calibrate_checkerboard_images,
    discover_checkerboard_images,
    save_calibration_data,
)
from .field_layout import FieldLayoutError, load_wpilib_apriltag_layout
from .provenance import ArtifactProvenance, ProvenanceKind


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
    parser = argparse.ArgumentParser(description="Calibrate a Northstar camera from checkerboard photographs")
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
        print(f"northstar-calibrate: error: {exc}", file=sys.stderr)
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


def layout_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a WPILib AprilTag field-layout JSON")
    parser.add_argument("layout", help="WPILib AprilTag layout JSON")
    parser.add_argument(
        "--official-source",
        help="Explicit authoritative source label; marks the validated layout as official",
    )
    parser.add_argument(
        "--require-production",
        action="store_true",
        help="Fail unless --official-source marks the layout as production-ready",
    )
    parser.add_argument("--report", help="Optional destination for the validation report JSON")
    return parser


def layout_main(argv: Sequence[str] | None = None) -> int:
    args = layout_parser().parse_args(argv)
    provenance = None
    if args.official_source:
        provenance = ArtifactProvenance(ProvenanceKind.OFFICIAL, args.official_source)
    try:
        layout = load_wpilib_apriltag_layout(args.layout, provenance=provenance)
        if args.require_production:
            layout.require_production()
    except (FieldLayoutError, OSError, ValueError) as exc:
        print(f"northstar-layout: error: {exc}", file=sys.stderr)
        return 2
    report = {
        "layout": str(Path(args.layout).expanduser().resolve()),
        "field_length_m": layout.field_length_m,
        "field_width_m": layout.field_width_m,
        "tag_count": len(layout.tag_poses),
        "tag_ids": sorted(layout.tag_poses),
        "provenance": layout.provenance.to_dict(),
        "production_ready": layout.provenance.kind is ProvenanceKind.OFFICIAL,
    }
    encoded = json.dumps(report, indent=2 if args.report else None, separators=None if args.report else (",", ":"))
    if args.report:
        destination = Path(args.report).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0
