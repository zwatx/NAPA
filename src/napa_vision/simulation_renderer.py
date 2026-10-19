from __future__ import annotations

import math

import cv2
import numpy as np

from .config import CameraConfig
from .geometry import invert_transform, tag_field_corners, transform_points, wpilib_points_to_opencv
from .simulation import FrameEffects, RenderedFrame, SimulationFrameSpec, SimulationTruth


class SyntheticAprilTagRenderer:
    """Render pinhole/homographic AprilTag scenes for hardware-free testing."""

    def __init__(
        self,
        camera: CameraConfig,
        tag_size_m: float,
        tag_layout: dict[int, np.ndarray],
        *,
        seed: int = 2026,
        supersample: int = 2,
        background: int = 230,
        marker_pixels: int = 256,
    ) -> None:
        if supersample < 1 or marker_pixels < 32 or not 0 <= background <= 255:
            raise ValueError("invalid simulation renderer settings")
        self.camera = camera
        self.tag_size_m = float(tag_size_m)
        self.tag_layout = {
            tag_id: np.asarray(transform, dtype=np.float64).copy() for tag_id, transform in tag_layout.items()
        }
        self.seed = int(seed)
        self.supersample = int(supersample)
        self.background = int(background)
        self.marker_pixels = int(marker_pixels)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._markers: dict[int, np.ndarray] = {}
        self._field_corners = {
            tag_id: tag_field_corners(transform, self.tag_size_m) for tag_id, transform in self.tag_layout.items()
        }
        self._source_corners = np.array(
            [
                [0, 0],
                [self.marker_pixels - 1, 0],
                [self.marker_pixels - 1, self.marker_pixels - 1],
                [0, self.marker_pixels - 1],
            ],
            dtype=np.float32,
        )
        self._marker_mask = np.full((self.marker_pixels, self.marker_pixels), 255, dtype=np.uint8)
        self._scaled_calibration_matrix = self.camera.calibration_matrix.copy() * self.supersample
        self._scaled_calibration_matrix[2, 2] = 1.0

    def render(self, spec: SimulationFrameSpec) -> RenderedFrame:
        scale = self.supersample
        width, height = self.camera.width * scale, self.camera.height * scale
        image = np.full((height, width), self.background, dtype=np.uint8)
        camera_to_field = invert_transform(spec.field_to_camera)
        render_items: list[tuple[float, int, np.ndarray]] = []
        layout_corners: dict[int, np.ndarray] = {}
        image_corners: dict[int, np.ndarray] = {}

        for tag_id in spec.tag_ids:
            if tag_id not in self.tag_layout:
                raise ValueError(f"simulation tag {tag_id} is not present in the layout")
            field_corners = self._field_corners[tag_id]
            camera_corners = transform_points(camera_to_field, field_corners)
            opencv_corners = wpilib_points_to_opencv(camera_corners)
            if np.any(opencv_corners[:, 2] <= 0.02) or not self._front_facing(tag_id, spec.field_to_camera):
                continue
            projected, _ = cv2.projectPoints(
                opencv_corners,
                np.zeros(3),
                np.zeros(3),
                self._scaled_calibration_matrix,
                self.camera.distortion_coefficients,
            )
            projected = projected.reshape(4, 2)
            layout_corners[tag_id] = projected / scale
            rendered = projected.copy()
            if spec.effects.displaced_tag_id == tag_id:
                rendered += np.asarray(spec.effects.displaced_tag_offset_px) * scale
            if not self._visible_polygon(rendered, width, height, scale):
                continue
            image_corners[tag_id] = rendered / scale
            render_items.append((float(np.mean(opencv_corners[:, 2])), tag_id, rendered))

        for _depth, tag_id, corners in sorted(render_items, reverse=True):
            self._warp_marker(image, tag_id, corners)
        self._apply_occlusion(image, spec.effects, image_corners, scale)
        if scale > 1:
            image = cv2.resize(image, (self.camera.width, self.camera.height), interpolation=cv2.INTER_AREA)
        image = self._apply_effects(image, spec)
        expected = spec.expected_tag_ids
        if expected is None:
            expected = tuple(sorted(image_corners))
        truth = SimulationTruth(
            index=spec.index,
            name=spec.name,
            field_to_camera=np.asarray(spec.field_to_camera, dtype=np.float64).copy(),
            rendered_tag_ids=tuple(sorted(image_corners)),
            expected_tag_ids=tuple(sorted(expected)),
            expected_rejected_tag_ids=tuple(sorted(spec.expected_rejected_tag_ids)),
            image_corners=image_corners,
            layout_corners=layout_corners,
            effects=spec.effects,
            expect_pose=spec.expect_pose,
        )
        return RenderedFrame(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), truth)

    def _front_facing(self, tag_id: int, field_to_camera: np.ndarray) -> bool:
        tag = self.tag_layout[tag_id]
        face_normal = tag[:3, 0]
        tag_to_camera = np.asarray(field_to_camera)[:3, 3] - tag[:3, 3]
        return float(np.dot(face_normal, tag_to_camera)) > 0.0

    @staticmethod
    def _visible_polygon(corners: np.ndarray, width: int, height: int, scale: int) -> bool:
        if abs(cv2.contourArea(corners.astype(np.float32))) < (4 * scale) ** 2:
            return False
        x0, y0 = np.min(corners, axis=0)
        x1, y1 = np.max(corners, axis=0)
        return x1 >= 0 and y1 >= 0 and x0 < width and y0 < height

    def _marker(self, tag_id: int) -> np.ndarray:
        if tag_id not in self._markers:
            self._markers[tag_id] = cv2.aruco.generateImageMarker(self.dictionary, tag_id, self.marker_pixels)
        return self._markers[tag_id]

    def _warp_marker(self, image: np.ndarray, tag_id: int, corners: np.ndarray) -> None:
        height, width = image.shape
        x0 = max(0, int(math.floor(float(np.min(corners[:, 0])))) - 1)
        y0 = max(0, int(math.floor(float(np.min(corners[:, 1])))) - 1)
        x1 = min(width, int(math.ceil(float(np.max(corners[:, 0])))) + 2)
        y1 = min(height, int(math.ceil(float(np.max(corners[:, 1])))) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        destination = corners.astype(np.float32) - np.array([x0, y0], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(self._source_corners, destination)
        size = (x1 - x0, y1 - y0)
        marker = self._marker(tag_id)
        warped = cv2.warpPerspective(marker, transform, size, flags=cv2.INTER_NEAREST, borderValue=255)
        mask = cv2.warpPerspective(
            self._marker_mask,
            transform,
            size,
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        roi = image[y0:y1, x0:x1]
        roi[mask > 0] = warped[mask > 0]

    @staticmethod
    def _apply_occlusion(
        image: np.ndarray,
        effects: FrameEffects,
        corners_by_id: dict[int, np.ndarray],
        scale: int,
    ) -> None:
        corners = corners_by_id.get(effects.occluded_tag_id) if effects.occluded_tag_id is not None else None
        if corners is None or effects.occlusion_fraction <= 0.0:
            return
        scaled = corners * scale
        minimum = np.floor(np.min(scaled, axis=0)).astype(int)
        maximum = np.ceil(np.max(scaled, axis=0)).astype(int)
        start_x = int(maximum[0] - (maximum[0] - minimum[0]) * effects.occlusion_fraction)
        cv2.rectangle(image, (start_x, minimum[1]), (maximum[0], maximum[1]), 245, thickness=-1)

    def _apply_effects(self, image: np.ndarray, spec: SimulationFrameSpec) -> np.ndarray:
        effects = spec.effects
        if (
            effects.brightness == 0.0
            and effects.contrast == 1.0
            and effects.blur_sigma == 0.0
            and effects.noise_sigma == 0.0
            and effects.motion_blur_px <= 1
            and effects.jpeg_quality is None
        ):
            return image
        work = np.clip(image.astype(np.float32) * effects.contrast + effects.brightness, 0, 255)
        if effects.blur_sigma > 0.0:
            work = cv2.GaussianBlur(work, (0, 0), effects.blur_sigma)
        if effects.motion_blur_px > 1:
            size = effects.motion_blur_px
            kernel = np.zeros((size, size), dtype=np.float32)
            kernel[size // 2, :] = 1.0 / size
            work = cv2.filter2D(work, -1, kernel)
        if effects.noise_sigma > 0.0:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, spec.index]))
            work += rng.normal(0.0, effects.noise_sigma, work.shape).astype(np.float32)
        output = np.clip(work, 0, 255).astype(np.uint8)
        if effects.jpeg_quality is not None:
            ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, effects.jpeg_quality])
            if not ok:
                raise RuntimeError("OpenCV could not encode the simulated JPEG frame")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise RuntimeError("OpenCV could not decode the simulated JPEG frame")
            output = decoded
        return output
