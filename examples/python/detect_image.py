"""Detect tags in an image, print JSON, and optionally save an annotated copy."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from napa_vision import NapaDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, help="save an annotated image, e.g. detected.png")
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--min-tag-px", type=int, default=32, help="lower to 16 for smaller or distant tags")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"could not read image: {args.image}")

    detector = NapaDetector(family=args.family, min_tag_px=args.min_tag_px)
    try:
        detections = detector.detect(image)  # BGR is accepted directly; no conversion needed.
    finally:
        detector.close()

    if args.output:
        for tag in detections:
            corners = np.rint(tag.corners).astype(np.int32)
            cv2.polylines(image, [corners], True, (0, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(tag.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not cv2.imwrite(str(args.output), image):
            parser.error(f"could not write image: {args.output}")

    print(
        json.dumps(
            [
                {
                    "family": tag.family,
                    "tag_id": tag.tag_id,
                    "corners_px": tag.corners.tolist(),
                    "hamming": tag.hamming,
                    "decision_margin": tag.decision_margin,
                }
                for tag in detections
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
