import ctypes

import cv2
import numpy as np
import pytest

from napa_vision import ArtifactProvenance, CameraCalibration, NapaDetector, ProvenanceKind, VisionProcessor
from napa_vision.napa_detector import _library
from tests.test_corner_bias_correction import _worst_corner_error
from tests.test_napa_detector import DICTIONARY, _scene

K = np.array([[420.0, 0.0, 319.5], [0.0, 420.0, 239.5], [0.0, 0.0, 1.0]])
BROWN_CONRADY = np.array([-0.32, 0.12, 0.0005, -0.0003, -0.02])
FISHEYE = np.array([0.02, -0.01, 0.003, -0.001])


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.deg2rad([pitch, yaw, roll]))
    return rotation


def _lens_render(distortion: np.ndarray, fisheye: bool, center: tuple[float, float], side_m: float, *, tag_id: int = 7):
    """A tag seen through a lens, rendered by inverse mapping: every one of 4x4 sub-pixels is undistorted with OpenCV,
    met with the tag's plane and read from 40-pixel-per-cell artwork, then averaged. Truth is OpenCV's projection of
    the black square's corners."""

    size, cells, scale = (640, 480), 10, 40
    marker = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, 8 * scale, borderBits=1)
    art = np.full((cells * scale, cells * scale), 255, np.uint8)
    art[scale : 9 * scale, scale : 9 * scale] = marker
    # The tag faces the camera at 0.6 m, turned a little, its middle seen at `center` (normalized coordinates).
    rotation = _rotation(18.0, -12.0, 9.0)
    direction = np.array([center[0], center[1], 1.0])
    translation = 0.6 * direction / np.linalg.norm(direction)
    half = side_m / 2
    square = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    rvec, _ = cv2.Rodrigues(rotation)
    if fisheye:
        truth, _ = cv2.fisheye.projectPoints(square.reshape(-1, 1, 3), rvec, translation, K, distortion)
    else:
        truth, _ = cv2.projectPoints(square, rvec, translation, K, distortion)
    truth = truth.reshape(4, 2)
    x0, y0 = np.floor(truth.min(0) - 0.3 * np.ptp(truth, 0)).astype(int)
    x1, y1 = np.ceil(truth.max(0) + 0.3 * np.ptp(truth, 0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size[0] - 1), min(y1, size[1] - 1)
    sub = (np.arange(4) + 0.5) / 4 - 0.5
    xs = (np.arange(x0, x1 + 1)[:, None] + sub).ravel()
    ys = (np.arange(y0, y1 + 1)[:, None] + sub).ravel()
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 1, 2)
    exact = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)
    if fisheye:
        normal = cv2.fisheye.undistortPoints(grid, K, distortion, criteria=exact).reshape(-1, 2)
    else:
        # OpenCV 4 takes stopping criteria only in undistortPointsIter; OpenCV 5 folded them into undistortPoints.
        if hasattr(cv2, "undistortPointsIter"):
            normal = cv2.undistortPointsIter(grid, K, distortion, None, None, exact)
        else:
            normal = cv2.undistortPoints(grid, K, distortion, criteria=exact)
        normal = normal.reshape(-1, 2)
    # Normalized point to the tag's plane coordinates: the inverse of [r1 r2 t].
    plane = np.linalg.inv(np.column_stack([rotation[:, 0], rotation[:, 1], translation]))
    tag = np.column_stack([normal, np.ones(len(normal))]) @ plane.T
    tag = tag[:, :2] / tag[:, 2:]
    # Plane metres to artwork pixels: the black square spans 8 of the artwork's 10 cells.
    pixels = (tag / side_m * 8 + 5) * scale - 0.5
    shape = (len(ys), len(xs))
    sampled = cv2.remap(
        art.astype(np.float32),
        pixels[:, 0].reshape(shape).astype(np.float32),
        pixels[:, 1].reshape(shape).astype(np.float32),
        cv2.INTER_LINEAR,
        borderValue=255,
    )
    image = np.full(size[::-1], 255.0)
    image[y0 : y1 + 1, x0 : x1 + 1] = cv2.resize(sampled, (x1 - x0 + 1, y1 - y0 + 1), interpolation=cv2.INTER_AREA)
    return np.clip(np.rint(30 + image * 0.78), 0, 255).astype(np.uint8), truth


def _detect(image: np.ndarray, **options):
    detector = NapaDetector(min_tag_px=16, **options)
    try:
        return detector.detect(image)
    finally:
        detector.close()


def _calibration(distortion: np.ndarray) -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=K,
        distortion_coefficients=distortion,
        image_size=(640, 480),
        provenance=ArtifactProvenance(ProvenanceKind.EXAMPLE, "synthetic lens test"),
    )


@pytest.mark.parametrize(
    "camera",
    [
        "pinhole",
        (K,),
        (K, BROWN_CONRADY, "kannala"),
        (K[:2], BROWN_CONRADY),
        (np.array([[420.0, 0, 320], [0, -420.0, 240], [0, 0, 1]]), BROWN_CONRADY),
        (K, np.zeros(6)),
        (K, BROWN_CONRADY, "fisheye"),
        (K, np.r_[BROWN_CONRADY, np.zeros(7), 0.01, 0.0]),
        (K, [np.nan, 0, 0, 0]),
    ],
)
def test_rejects_invalid_cameras(camera) -> None:
    with pytest.raises(ValueError, match="camera"):
        NapaDetector(camera=camera)


def test_native_setter_validates_and_clears() -> None:
    library = _library()
    handle = ctypes.c_void_p(library.napa_create_family(b"tag36h11"))
    try:
        intrinsics = (ctypes.c_double * 5)(420, 420, 319.5, 239.5, 0)
        coefficients = (ctypes.c_double * 5)(*BROWN_CONRADY)
        assert library.napa_set_lens(handle, 1, intrinsics, coefficients, 5) == 0
        assert library.napa_set_lens(handle, 2, intrinsics, coefficients, 5) == -1  # fisheye takes 4
        assert library.napa_set_lens(handle, 1, intrinsics, coefficients, 6) == -1
        assert library.napa_set_lens(handle, 3, intrinsics, coefficients, 4) == -1
        assert library.napa_set_lens(handle, 1, None, coefficients, 4) == -1
        bad = (ctypes.c_double * 5)(0, 420, 319.5, 239.5, 0)
        assert library.napa_set_lens(handle, 1, bad, coefficients, 5) == -1
        assert library.napa_set_lens(handle, 0, None, None, 0) == 0
    finally:
        library.napa_destroy(handle)


def test_a_lens_that_bends_nothing_leaves_results_unchanged() -> None:
    # Zero distortion keeps every side straight, so each tag is fitted in pixels exactly as without a camera.
    image, _ = _scene(42, 120, angle_deg=23.0)
    plain = _detect(image)
    for camera in ((K, np.zeros(5)), _calibration(np.zeros(5)), (K, np.zeros(4))):
        other = _detect(image, camera=camera)
        assert [o.tag_id for o in other] == [o.tag_id for o in plain] == [42]
        np.testing.assert_array_equal(other[0].corners, plain[0].corners)


@pytest.mark.parametrize(
    "distortion,fisheye,center,side",
    [
        (BROWN_CONRADY, False, (0.45, 0.3), 0.3),
        (BROWN_CONRADY, False, (-0.5, -0.25), 0.3),
        (FISHEYE, True, (0.45, 0.3), 0.3),
        (FISHEYE, True, (-0.5, -0.25), 0.22),
    ],
    ids=["brown-conrady-a", "brown-conrady-b", "fisheye-a", "fisheye-b"],
)
def test_lens_aware_fit_finds_distorted_corners(distortion, fisheye, center, side) -> None:
    # Tags 110-200 px across, off the image middle, where fitting in pixels misses a corner by 0.27-0.93 px. Through
    # the lens every corner lands within 0.01 px here; 0.05 leaves room for platform arithmetic.
    image, truth = _lens_render(distortion, fisheye, center, side)
    camera = (K, distortion, "fisheye") if fisheye else (K, distortion)
    plain, lens = _detect(image), _detect(image, camera=camera)
    assert [o.tag_id for o in plain] == [o.tag_id for o in lens] == [7]
    before, after = _worst_corner_error(plain[0].corners, truth), _worst_corner_error(lens[0].corners, truth)
    assert before > 0.2 and after < 0.05


def test_a_mildly_wrong_calibration_still_helps() -> None:
    image, truth = _lens_render(BROWN_CONRADY, False, (-0.5, -0.25), 0.3)
    plain = _worst_corner_error(_detect(image)[0].corners, truth)
    for scale in (0.8, 1.2):
        distortion = BROWN_CONRADY.copy()
        distortion[0] *= scale
        assert _worst_corner_error(_detect(image, camera=(K, distortion))[0].corners, truth) < 0.5 * plain


def test_accepts_a_camera_calibration_and_reports_it() -> None:
    calibration = _calibration(BROWN_CONRADY)
    detector = NapaDetector(camera=calibration)
    try:
        assert detector.camera is calibration
    finally:
        detector.close()


def test_vision_processor_gives_its_calibration_to_its_own_detector() -> None:
    calibration = _calibration(BROWN_CONRADY)
    processor = VisionProcessor(calibration=calibration, tag_size_m=0.16)
    assert processor.detector.camera is calibration
    supplied = NapaDetector()
    assert VisionProcessor(detector=supplied, calibration=calibration, tag_size_m=0.16).detector.camera is None


def test_a_calibration_holds_only_at_its_resolution() -> None:
    calibration = _calibration(BROWN_CONRADY)  # measured at 640x480
    detector = NapaDetector(camera=calibration)
    try:
        assert detector.detect(np.full((480, 640), 255, np.uint8)) == []
        with pytest.raises(ValueError, match="640x480, not 1280x720"):
            detector.detect(np.full((720, 1280), 255, np.uint8))
        with pytest.raises(ValueError, match="not 480x640"):
            detector.detect(np.full((640, 480, 3), 255, np.uint8))  # transposed colour frame
    finally:
        detector.close()
    # The bare (K, D) form records no resolution, so nothing is checked.
    tuple_detector = NapaDetector(camera=(K, BROWN_CONRADY))
    try:
        assert tuple_detector.detect(np.full((720, 1280), 255, np.uint8)) == []
    finally:
        tuple_detector.close()
