from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from . import __version__
from .config import ProfileError, available_profiles, load_profile
from .output import make_output_publisher
from .service import NorthstarVisionService

LOGGER = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect AprilTags and publish Northstar-compatible observations")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--profile", default="laptop", help="Bundled profile name or path to a JSON profile")
    parser.add_argument("--list-profiles", action="store_true", help="List bundled profile names and exit")
    parser.add_argument(
        "--check-config", action="store_true", help="Validate the selected profile without opening a camera"
    )
    parser.add_argument("--replay", help="Read a video or image directory instead of a live camera")
    parser.add_argument("--loop-replay", action="store_true", help="Loop replay input instead of stopping at EOF")
    parser.add_argument("--show", action="store_true", help="Show an OpenCV debug overlay window")
    parser.add_argument("--output", choices=["jsonl", "nt", "none"], help="Override the profile output mode")
    parser.add_argument("--output-path", help="JSONL destination (only used for --output jsonl)")
    parser.add_argument("--robot-yaw", type=float, default=0.0, help="Robot yaw in WPILib radians for trig output")
    parser.add_argument("--max-frames", type=_positive_int, help="Stop after this many successfully processed frames")
    skip_scan = parser.add_mutually_exclusive_group()
    skip_scan.add_argument("--enable-skip-scan", action="store_true", help="Enable ROI prefiltering for this run")
    skip_scan.add_argument("--disable-skip-scan", action="store_true", help="Run detection on each full frame")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Runtime log verbosity",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")
    if args.list_profiles:
        print("\n".join(available_profiles()))
        return 0
    try:
        profile = load_profile(args.profile)
        if args.check_config:
            print(f"Profile {profile.name!r} is valid ({profile.camera.width}x{profile.camera.height})")
            return 0
        if args.enable_skip_scan:
            profile.skip_scan.enabled = True
        elif args.disable_skip_scan:
            profile.skip_scan.enabled = False
        output_mode = args.output or profile.output_mode
        output = make_output_publisher(
            output_mode,
            profile.device_id,
            profile.nt_server,
            output_path=args.output_path or profile.output_path,
        )
        try:
            service = NorthstarVisionService(
                profile,
                output,
                replay_path=args.replay,
                loop_replay=args.loop_replay,
                show=args.show,
                robot_yaw_rad=args.robot_yaw,
            )
        except BaseException:
            output.close()
            raise
        service.run(max_frames=args.max_frames)
    except ProfileError as exc:
        LOGGER.error("Invalid profile: %s", exc)
        return 2
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0
