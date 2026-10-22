"""A drop-in for the ``pupil_apriltags`` / ``dt_apriltags`` Python API, backed by Napa.

Change ``from pupil_apriltags import Detector`` to ``from napa_vision.compat import Detector``. Results use
upstream AprilTag's conventions, not the rest of Napa's: corners in upstream order (counter-clockwise in the
image, starting from the official artwork's bottom-left corner), with ``(0, 0)`` at the top-left pixel's
corner (Napa's other APIs put it at that pixel's centre).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .napa_detector import NapaDetector

# Napa's corner order is upstream's p[1], p[0], p[3], p[2]; this indexes Napa's corners in upstream order.
_UPSTREAM_ORDER = [1, 0, 3, 2]
_HALF_PIXEL = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])


class Detection:
    """One detected tag, with the attributes ``pupil_apriltags.Detection`` has.

    ``pose_R``, ``pose_t``, and ``pose_err`` are set only when ``detect`` was asked to estimate poses.
    """

    __slots__ = (
        "tag_family",
        "tag_id",
        "hamming",
        "decision_margin",
        "homography",
        "center",
        "corners",
        "pose_R",
        "pose_t",
        "pose_err",
    )

    def __init__(self) -> None:
        self.tag_family: bytes | None = None
        self.tag_id: int | None = None
        self.hamming: int | None = None
        self.decision_margin: float | None = None
        self.homography: np.ndarray | None = None
        self.center: np.ndarray | None = None
        self.corners: np.ndarray | None = None
        self.pose_R: np.ndarray | None = None
        self.pose_t: np.ndarray | None = None
        self.pose_err: float | None = None

    def __repr__(self) -> str:
        fields = "\n".join(f"{name} = {getattr(self, name)}" for name in self.__slots__)
        return f"Detection object:\n{fields}\n"


class Detector:
    """``pupil_apriltags.Detector``'s interface on Napa's tag36h11 detector.

    Only ``families="tag36h11"`` is supported. ``decode_sharpening`` is used; ``nthreads``,
    ``quad_decimate``, ``quad_sigma``, ``refine_edges``, ``debug``, and ``searchpath`` are accepted and
    ignored, since Napa has no equivalent (it never decimates, and always refines edges). Napa's own
    options, such as ``min_tag_px`` or ``max_hamming``, can be passed as extra keyword arguments.
    """

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 2.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
        debug: int = 0,
        searchpath: Any = None,
        **napa_options: Any,
    ) -> None:
        unsupported = [family for family in families.split() if family != "tag36h11"]
        if unsupported or not families.split():
            raise ValueError(f"Napa detects only tag36h11 tags; got families={families!r}")
        self.params = {
            "families": families.split(),
            "nthreads": nthreads,
            "quad_decimate": quad_decimate,
            "quad_sigma": quad_sigma,
            "refine_edges": refine_edges,
            "decode_sharpening": decode_sharpening,
            "debug": debug,
        }
        self._detector = NapaDetector(sharpening=decode_sharpening, **napa_options)

    def detect(
        self,
        img: np.ndarray,
        estimate_tag_pose: bool = False,
        camera_params: tuple[float, float, float, float] | None = None,
        tag_size: float | None = None,
    ) -> list[Detection]:
        """Detect tags in a ``uint8`` grayscale image.

        With ``estimate_tag_pose=True``, also pass ``camera_params`` as ``(fx, fy, cx, cy)`` and
        ``tag_size``, the black square's side in metres. Each pose maps tag points into the camera frame
        (+x right, +y down, +z forward), with upstream's tag frame (+x right and +y down on the official
        artwork, +z into the tag), as upstream's ``estimate_tag_pose`` does: the better of the two planar
        solutions by upstream's object-space error, ``pose_err``. Poses need OpenCV. As upstream, lens
        distortion is not modelled; undistort the image first for a wide-angle lens.
        """

        img = np.asarray(img)
        if img.ndim != 2 or img.dtype != np.uint8:
            raise ValueError("img must be a uint8 grayscale image")
        if estimate_tag_pose:
            if camera_params is None:
                raise ValueError("camera_params must be provided to detect if estimate_tag_pose is set to True")
            if tag_size is None:
                raise ValueError("tag_size must be provided to detect if estimate_tag_pose is set to True")
        detections = []
        for observation in self._detector.detect(img):
            detection = Detection()
            detection.tag_family = b"tag36h11"
            detection.tag_id = observation.tag_id
            detection.hamming = observation.hamming
            detection.decision_margin = observation.decision_margin
            homography = _HALF_PIXEL @ observation.homography
            detection.homography = homography / homography[2, 2]
            detection.center = observation.center + 0.5
            detection.corners = observation.corners[_UPSTREAM_ORDER] + 0.5
            if estimate_tag_pose:
                detection.pose_R, detection.pose_t, detection.pose_err = _estimate_pose(
                    detection.corners, camera_params, tag_size
                )
            detections.append(detection)
        return detections


def _estimate_pose(
    corners: np.ndarray, camera_params: tuple[float, float, float, float], tag_size: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """As upstream's ``estimate_tag_pose``: both planar solutions, refined by orthogonal iteration; the better kept."""

    from . import _import_cv2

    cv2 = _import_cv2()
    fx, fy, cx, cy = (float(value) for value in camera_params)
    half = 0.5 * float(tag_size)
    # Upstream's tag frame: +x right and +y down on the official artwork, +z into the tag. p[0] is the
    # artwork's bottom-left corner, and p[0..3] run counter-clockwise as seen in the image.
    points = np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]])
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        points, np.ascontiguousarray(corners, dtype=np.float64), camera_matrix, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    rays = np.column_stack([(corners[:, 0] - cx) / fx, (corners[:, 1] - cy) / fy, np.ones(4)])
    projectors = np.einsum("ni,nj->nij", rays, rays) / np.einsum("ni,ni->n", rays, rays)[:, None, None]
    best = None
    for index in range(count):
        rotation = cv2.Rodrigues(rvecs[index])[0]
        rotation, translation, error = _orthogonal_iteration(points, projectors, rotation)
        if best is None or error < best[2]:
            best = (rotation, translation.reshape(3, 1), error)
    if best is None:
        raise RuntimeError("no pose solution for the tag's corners")
    return best


def _orthogonal_iteration(
    points: np.ndarray, projectors: np.ndarray, rotation: np.ndarray, iterations: int = 50
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lu, Hager and Mjolsness's orthogonal iteration, which upstream uses: minimise the object-space error
    sum ||(I - F_i)(R p_i + t)||^2, where F_i projects onto pixel i's line of sight."""

    identity = np.eye(3)
    residual_maps = identity - projectors  # (4, 3, 3)
    solve_t = np.linalg.inv(identity - projectors.mean(axis=0))
    centred = points - points.mean(axis=0)
    error = np.inf
    translation = np.zeros(3)
    for _ in range(iterations):
        rotated = points @ rotation.T
        translation = solve_t @ np.einsum("nij,nj->i", projectors - identity, rotated) / len(points)
        new_error = float(np.sum(np.einsum("nij,nj->ni", residual_maps, rotated + translation) ** 2))
        if new_error >= error - 1e-15:
            error = min(error, new_error)
            break
        error = new_error
        projected = np.einsum("nij,nj->ni", projectors, rotated + translation)
        u, _, vt = np.linalg.svd((projected - projected.mean(axis=0)).T @ centred)
        rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    rotated = points @ rotation.T
    translation = solve_t @ np.einsum("nij,nj->i", projectors - identity, rotated) / len(points)
    error = float(np.sum(np.einsum("nij,nj->ni", residual_maps, rotated + translation) ** 2))
    return rotation, translation, error
