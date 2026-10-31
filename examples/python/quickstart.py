"""Generate a tag and detect it with only NAPA and NumPy, no camera or OpenCV."""

import numpy as np

from napa_vision import NapaDetector
from napa_vision.tags import tag_image


def main() -> None:
    image = np.full((480, 640), 255, dtype=np.uint8)
    image[120:360, 200:440] = tag_image(7, cell_px=24)

    detector = NapaDetector()
    try:
        for tag in detector.detect(image):
            print(f"{tag.family} ID {tag.tag_id}, hamming={tag.hamming}, margin={tag.decision_margin:.1f}")
            print("Corners (x, y) in pixels:")
            print(tag.corners)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
