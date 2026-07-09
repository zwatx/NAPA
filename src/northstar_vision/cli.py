from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Sequence
from dataclasses import replace

from . import __version__
from .config import ProfileError, available_profiles, load_profile
from .orientation import NetworkTablesRobotYawSource
from .output import make_output_publisher
from .service import NorthstarVisionService

LOGGER = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > 65535:
        raise argparse.ArgumentTypeError("must be between 0 and 65535")
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
    parser.add_argument("--nt-server", help="Override the profile NetworkTables server")
    parser.add_argument("--nt-server-port", type=_port, help="NetworkTables server port (0 uses default)")
    parser.add_argument(
        "--nt-close-drain",
        type=_nonnegative_float,
        help="Bounded seconds to drain the final NetworkTables update during shutdown",
    )
    parser.add_argument(
        "--nt-delivery",
        choices=["upstream", "latest"],
        default=None,
        help="Preserve every upstream-compatible update or bound delivery to the latest value",
    )
    parser.add_argument("--jsonl-queue-size", type=_positive_int, help="Bounded recorder queue depth")
    parser.add_argument(
        "--jsonl-drop-policy",
        choices=["drop_oldest", "drop_newest"],
        default=None,
        help="Recorder behavior when its queue is full",
    )
    parser.add_argument(
        "--jsonl-flush-interval",
        type=_nonnegative_float,
        default=None,
        help="Seconds between recorder flushes (0 flushes every record)",
    )
    parser.add_argument(
        "--jsonl-fsync-interval",
        type=_nonnegative_float,
        default=None,
        help="Seconds between durable recorder syncs (0 disables fsync)",
    )
    parser.add_argument(
        "--jsonl-max-size-mib",
        type=_nonnegative_float,
        default=None,
        help="Rotate the active recording at this size (0 disables size rotation)",
    )
    parser.add_argument(
        "--jsonl-rotate-seconds",
        type=_nonnegative_float,
        default=None,
        help="Rotate the active recording after this duration (0 disables time rotation)",
    )
    parser.add_argument(
        "--jsonl-retention-files",
        type=_nonnegative_int,
        default=None,
        help="Number of rotated JSONL archives retained",
    )
    yaw = parser.add_mutually_exclusive_group()
    yaw.add_argument("--robot-yaw", type=float, help="Static WPILib yaw in radians for controlled bench tests")
    yaw.add_argument("--robot-yaw-topic", help="Absolute NetworkTables double topic containing robot yaw radians")
    parser.add_argument(
        "--yaw-max-age",
        type=_positive_float,
        default=None,
        help="Maximum yaw sample age in seconds before trig localization is suppressed",
    )
    parser.add_argument(
        "--yaw-queue-size",
        type=_positive_int,
        default=None,
        help="Timestamped robot-yaw samples retained for interpolation",
    )
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
        nt_server = args.nt_server or profile.nt_server
        nt_server_port = profile.nt_server_port if args.nt_server_port is None else args.nt_server_port
        nt_send_all = profile.nt_send_all if args.nt_delivery is None else args.nt_delivery == "upstream"
        nt_close_drain_s = profile.nt_close_drain_s if args.nt_close_drain is None else args.nt_close_drain
        jsonl_config = replace(
            profile.jsonl_recorder,
            queue_size=(profile.jsonl_recorder.queue_size if args.jsonl_queue_size is None else args.jsonl_queue_size),
            drop_policy=(
                profile.jsonl_recorder.drop_policy if args.jsonl_drop_policy is None else args.jsonl_drop_policy
            ),
            flush_interval_s=(
                profile.jsonl_recorder.flush_interval_s
                if args.jsonl_flush_interval is None
                else args.jsonl_flush_interval
            ),
            fsync_interval_s=(
                profile.jsonl_recorder.fsync_interval_s
                if args.jsonl_fsync_interval is None
                else (None if args.jsonl_fsync_interval == 0 else args.jsonl_fsync_interval)
            ),
            max_file_size_bytes=(
                profile.jsonl_recorder.max_file_size_bytes
                if args.jsonl_max_size_mib is None
                else (None if args.jsonl_max_size_mib == 0 else int(args.jsonl_max_size_mib * 1024 * 1024))
            ),
            rotation_interval_s=(
                profile.jsonl_recorder.rotation_interval_s
                if args.jsonl_rotate_seconds is None
                else (None if args.jsonl_rotate_seconds == 0 else args.jsonl_rotate_seconds)
            ),
            retention_files=(
                profile.jsonl_recorder.retention_files
                if args.jsonl_retention_files is None
                else args.jsonl_retention_files
            ),
        )
        yaw_topic = None if args.robot_yaw is not None else (args.robot_yaw_topic or profile.robot_yaw_topic)
        yaw_max_age_s = profile.yaw_max_age_s if args.yaw_max_age is None else args.yaw_max_age
        yaw_queue_size = profile.yaw_queue_size if args.yaw_queue_size is None else args.yaw_queue_size
        output = None
        yaw_source = None
        try:
            output = make_output_publisher(
                output_mode,
                profile.device_id,
                nt_server,
                output_path=args.output_path or profile.output_path,
                nt_server_port=nt_server_port,
                nt_send_all=nt_send_all,
                nt_close_drain_s=nt_close_drain_s,
                jsonl_config=jsonl_config,
            )
            if yaw_topic:
                yaw_source = NetworkTablesRobotYawSource(
                    profile.device_id,
                    nt_server,
                    topic=yaw_topic,
                    max_age_s=yaw_max_age_s,
                    queue_size=yaw_queue_size,
                    server_port=nt_server_port,
                )
            service = NorthstarVisionService(
                profile,
                output,
                replay_path=args.replay,
                loop_replay=args.loop_replay,
                show=args.show,
                robot_yaw_rad=args.robot_yaw,
                yaw_source=yaw_source,
            )
        except BaseException:
            if yaw_source is not None:
                yaw_source.close()
            if output is not None:
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
