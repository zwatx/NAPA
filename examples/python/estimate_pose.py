"""Estimate camera-to-tag poses in a photo from a measured camera calibration."""

import argparse
import json
from pathlib import Path

import cv2

from napa_vision import NapaDetector, VisionProcessor, load_calibration_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--calibration", type=Path, required=True, help="camera JSON from napa-calibrate")
    parser.add_argument(
        "--tag-size-m",
        type=float,
        required=True,
        help="measured edge of the tag-size square (the black square on tag36h11), in metres",
    )
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--min-tag-px", type=int, default=32)
    args = parser.parse_args()

    calibration = load_calibration_data(args.calibration)
    calibration.require_production()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"could not read image: {args.image}")

    detector = NapaDetector(family=args.family, min_tag_px=args.min_tag_px)
    try:
        processor = VisionProcessor(detector=detector, calibration=calibration, tag_size_m=args.tag_size_m)
        # A photo has no capture time, so 0.0 stands in. On a robot, pass each frame's exposure timestamp.
        result = processor.process(image, timestamp=0.0)
    finally:
        detector.close()

    poses = []
    for single in result.single_tag_poses:
        solutions = []
        for index, error in enumerate((single.error_0, single.error_1)):
            if error is None:
                continue
            # camera_to_tag() converts from OpenCV axes to WPILib camera axes (forward, left, up).
            pose = single.camera_to_tag(index)
            solutions.append(
                {
                    "solution": index,
                    "translation_m": pose.translation.tolist(),
                    "camera_to_tag": pose.transform.tolist(),
                    "covariance": pose.covariance.tolist(),
                    "reprojection_error_px": error,
                }
            )
        poses.append({"tag_id": single.tag_id, "ambiguity": single.ambiguity, "solutions": solutions})
    solved_ids = {single.tag_id for single in result.single_tag_poses}
    print(
        json.dumps(
            {
                "axes": "WPILib camera: +X forward, +Y left, +Z up",
                "covariance_order": ["x", "y", "z", "rx", "ry", "rz"],
                "poses": poses,
                "unsolved_tag_ids": [tag.tag_id for tag in result.image_observations if tag.tag_id not in solved_ids],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
