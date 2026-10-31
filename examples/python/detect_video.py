"""Detect tags in webcam or video frames and print one JSON line per frame."""

import argparse
import json
from time import perf_counter

import cv2
import numpy as np

from napa_vision import NapaDetector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="camera index, usually 0")
    source.add_argument("--video", help="path to a recorded video")
    parser.add_argument(
        "--display",
        action="store_true",
        help="show detections in a window; press q or Escape to stop (needs the gui extra)",
    )
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--min-tag-px", type=int, default=32)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")

    detector = NapaDetector(family=args.family, min_tag_px=args.min_tag_px)
    capture = cv2.VideoCapture()
    frame_index = 0
    try:
        if not capture.open(args.camera if args.camera is not None else args.video):
            parser.error("could not open capture source")
        while args.max_frames is None or frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                if args.camera is not None or frame_index == 0:
                    parser.error("capture source did not return a frame")
                break  # End of a recorded video.
            start = perf_counter()
            detections = detector.detect(frame)  # One detector reuses its buffers for every frame.
            detection_ms = (perf_counter() - start) * 1000
            print(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "detection_ms": detection_ms,
                        "tags": [
                            {"tag_id": tag.tag_id, "family": tag.family, "corners_px": tag.corners.tolist()}
                            for tag in detections
                        ],
                    }
                ),
                flush=True,
            )
            frame_index += 1
            if args.display:
                for tag in detections:
                    corners = np.rint(tag.corners).astype(np.int32)
                    cv2.polylines(frame, [corners], True, (0, 200, 0), 2, cv2.LINE_AA)
                    cv2.putText(
                        frame, str(tag.tag_id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )
                cv2.imshow("NAPA — q to quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        detector.close()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        raise SystemExit(str(exc)) from exc
