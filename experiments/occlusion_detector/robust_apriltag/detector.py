"""Integrated occlusion-tolerant detector for standard tag36h11 AprilTags."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .family import TAG36H11, decode_observed, make_code_from_bits


@dataclass
class Detection:
    id: int
    corners: np.ndarray
    homography: np.ndarray
    pose: dict[str, np.ndarray] | None
    hamming: int
    decision_margin: int
    occlusion_score: float
    used_anchor_count: int
    erased_bit_count: int
    score: float
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Candidate:
    corners: np.ndarray
    source: str
    area: float


class Robust36h11Detector:
    """Detect tag36h11 with one multi-anchor solve per candidate.

    The implementation deliberately does not call UMich AprilTag, OpenCV ArUco,
    or any other detector at runtime. Candidate regions are generated from dark
    component and edge evidence, then scored jointly against border, payload,
    and geometry constraints.
    """

    def __init__(
        self,
        *,
        max_hamming: int = 2,
        max_occlusion_ratio: float = 0.25,
        max_erased_bits: int = 0,
        min_anchor_count: int = 12,
        refine_pose: bool = True,
        min_tag_area: float = 400.0,
        max_candidates: int = 80,
        debug: bool = False,
    ) -> None:
        self.max_hamming = max_hamming
        self.max_occlusion_ratio = max_occlusion_ratio
        self.max_erased_bits = max_erased_bits
        self.min_anchor_count = min_anchor_count
        self.refine_pose = refine_pose
        self.min_tag_area = min_tag_area
        self.max_candidates = max_candidates
        self.debug = debug

        self._canonical_corners = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
        self._black_border_points = self._make_black_border_points()
        self._white_border_points = self._make_white_border_points()
        self._edge_points = self._make_edge_points()
        self._edge_canonical = np.asarray([point for point, _inward, _side in self._edge_points], dtype=np.float32)
        self._edge_inward = np.asarray([inward for _point, inward, _side in self._edge_points], dtype=np.float32)
        self._edge_sides = np.asarray([side for _point, _inward, side in self._edge_points], dtype=np.int8)
        self._edge_offsets = np.linspace(-0.9, 0.9, 31, dtype=np.float32)
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._black_sample_end = len(self._black_border_points)
        self._white_sample_end = self._black_sample_end + len(self._white_border_points)
        self._sample_points = np.concatenate(
            (self._black_border_points, self._white_border_points, TAG36H11.bit_centers)
        )

    def detect(
        self,
        gray_image: np.ndarray,
        camera_params: dict[str, float] | None = None,
        tag_size: float | None = None,
        candidate_corners: list[np.ndarray] | None = None,
    ) -> list[Detection]:
        start = perf_counter()
        gray = self._normalize_gray(gray_image)
        if candidate_corners is None:
            dark_mask, edges = self._preprocess(gray)
            candidates = self._generate_candidates(dark_mask, edges, gray.shape)
        else:
            candidates = [
                _Candidate(self._order_points(np.asarray(corners, dtype=np.float32)), "seeded-corners", abs(float(cv2.contourArea(np.asarray(corners, dtype=np.float32)))))
                for corners in candidate_corners
                if self._accept_quad(np.asarray(corners, dtype=np.float32), gray.shape)
            ]

        detections: list[Detection] = []
        for candidate in candidates:
            detection = self._solve_candidate(gray, candidate, camera_params, tag_size)
            if detection is not None:
                detections.append(detection)

        detections.sort(key=lambda d: d.score, reverse=True)
        detections = self._suppress_duplicates(detections)

        if self.debug:
            elapsed_ms = (perf_counter() - start) * 1000.0
            for det in detections:
                det.debug.setdefault("timing_ms", elapsed_ms)
                det.debug.setdefault("candidate_count", len(candidates))
        return detections

    def _normalize_gray(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _preprocess(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        block_size = max(15, ((min(gray.shape) // 24) | 1))
        block_size = min(block_size, 61)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            5,
        )
        dark = cv2.bitwise_not(binary)
        cv2.morphologyEx(dark, cv2.MORPH_CLOSE, self._close_kernel, dst=dark)
        edges = cv2.Canny(blurred, 40, 120)
        return dark, edges

    def _generate_candidates(
        self,
        dark_mask: np.ndarray,
        edges: np.ndarray,
        image_shape: tuple[int, int],
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        seen: list[np.ndarray] = []
        for mask, source in ((dark_mask, "dark-component"), (edges, "edge-contour")):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_tag_area:
                    continue
                hull = cv2.convexHull(contour)
                hull_area = float(cv2.contourArea(hull))
                if hull_area < self.min_tag_area:
                    continue
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect).astype(np.float32)
                if self._accept_quad(box, image_shape) and not self._near_duplicate(box, seen):
                    ordered = self._order_points(box)
                    seen.append(ordered)
                    candidates.append(_Candidate(ordered, source, hull_area))

                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.035 * peri, True)
                if len(approx) == 4:
                    quad = approx.reshape(4, 2).astype(np.float32)
                    if self._accept_quad(quad, image_shape) and not self._near_duplicate(quad, seen):
                        ordered = self._order_points(quad)
                        seen.append(ordered)
                        candidates.append(_Candidate(ordered, f"{source}-hull", hull_area))

        candidates.sort(key=lambda c: c.area, reverse=True)
        return candidates[: self.max_candidates]

    def _solve_candidate(
        self,
        gray: np.ndarray,
        candidate: _Candidate,
        camera_params: dict[str, float] | None,
        tag_size: float | None,
    ) -> Detection | None:
        h = cv2.getPerspectiveTransform(self._canonical_corners, candidate.corners.astype(np.float32))
        edge_support = 0
        if candidate.source != "seeded-corners":
            refined_h, edge_support = self._refine_homography_from_edges(gray, h)
            if refined_h is not None and edge_support >= max(8, self.min_anchor_count // 2):
                h = refined_h

        samples = self._sample_all(gray, h)
        if samples is None:
            return None
        black_vals, white_vals, bit_vals = samples
        contrast = float(np.median(white_vals) - np.median(black_vals))
        if contrast < 18:
            return None
        threshold = float((np.median(white_vals) + np.median(black_vals)) * 0.5)
        min_margin = max(6.0, contrast * 0.16)

        black_ok, black_erased = self._classify_expected(black_vals, threshold, expected_white=False, min_margin=min_margin)
        white_ok, white_erased = self._classify_expected(white_vals, threshold, expected_white=True, min_margin=min_margin)
        border_total = len(black_vals) + len(white_vals)
        border_used = black_ok + white_ok
        border_erased = black_erased + white_erased

        bit_margins = np.abs(bit_vals - threshold)
        bit_mask = bit_margins >= min_margin
        reliable_bits = int(bit_mask.sum())
        erased_bits = TAG36H11.nbits - reliable_bits
        if erased_bits > self.max_erased_bits or reliable_bits < TAG36H11.nbits - 10:
            return None
        bit_values = (bit_vals > threshold).astype(np.uint8)
        observed_code, observed_mask = make_code_from_bits(bit_values, bit_mask)
        decoded = decode_observed(
            observed_code,
            observed_mask,
            max_hamming=self.max_hamming,
            min_reliable_bits=max(24, TAG36H11.nbits - 10),
        )
        if decoded is None:
            return None

        tag_id, hamming, rotation, decision_margin = decoded
        occluded = border_erased + erased_bits + hamming
        total_anchors = border_total + TAG36H11.nbits
        occlusion_score = occluded / max(1, total_anchors)
        used_anchor_count = border_used + reliable_bits
        if used_anchor_count < self.min_anchor_count:
            return None
        if occlusion_score > self.max_occlusion_ratio:
            return None

        corners = self._project_points(self._canonical_corners, h).astype(np.float32)
        if candidate.source == "seeded-corners":
            repaired_corners, repaired_h = self._repair_square_corners(gray, candidate.corners, h)
            if repaired_corners is not None and repaired_h is not None:
                corners = repaired_corners.astype(np.float32)
                h = repaired_h
        pose = self._estimate_pose(corners, camera_params, tag_size) if self.refine_pose else None
        avg_margin = float(np.mean(bit_margins[bit_mask])) if bit_mask.any() else 0.0
        score = (
            decision_margin * 8.0
            + avg_margin * 0.18
            + edge_support * 0.4
            + border_used * 0.2
            - hamming * 12.0
            - erased_bits * 1.5
        )

        debug: dict[str, Any] = {}
        if self.debug:
            debug = {
                "source": candidate.source,
                "edge_support": edge_support,
                "contrast": contrast,
                "threshold": threshold,
                "rotation": rotation,
                "bit_mask": bit_mask.copy(),
                "candidate_corners": candidate.corners.copy(),
                "corners": corners.copy(),
            }
        return Detection(
            id=tag_id,
            corners=corners,
            homography=h,
            pose=pose,
            hamming=hamming,
            decision_margin=decision_margin,
            occlusion_score=occlusion_score,
            used_anchor_count=used_anchor_count,
            erased_bit_count=erased_bits,
            score=score,
            debug=debug,
        )

    def _repair_square_corners(
        self,
        gray: np.ndarray,
        corners: np.ndarray,
        h: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        edge_sets = self._collect_seeded_edge_points(gray, h)
        if sum(len(points) >= 5 for points in edge_sets) < 2:
            return None, None

        lines = [self._fit_line(points) if len(points) >= 5 else None for points in edge_sets]
        repaired = np.asarray(corners, dtype=np.float32).copy()
        for i in range(4):
            previous_line = lines[(i - 1) % 4]
            next_line = lines[i]
            if previous_line is not None and next_line is not None:
                point = self._intersect_lines(previous_line, next_line)
                if point is not None:
                    repaired[i] = point
                continue
            if previous_line is None and next_line is None:
                repaired[i] = repaired[(i - 1) % 4] + repaired[(i + 1) % 4] - repaired[(i + 2) % 4]
            elif previous_line is None:
                repaired[i] = repaired[(i - 1) % 4] + repaired[(i + 1) % 4] - repaired[(i + 2) % 4]
            elif next_line is None:
                repaired[i] = repaired[(i - 1) % 4] + repaired[(i + 1) % 4] - repaired[(i + 2) % 4]

        if not self._plausible_repair(corners, repaired, gray.shape):
            return None, None
        repaired_h = cv2.getPerspectiveTransform(self._canonical_corners, repaired.astype(np.float32))
        return repaired, repaired_h.astype(np.float64)

    def _collect_seeded_edge_points(self, gray: np.ndarray, h: np.ndarray) -> list[list[np.ndarray]]:
        projected_corners = self._project_points(self._canonical_corners, h)
        side_lengths = np.linalg.norm(np.roll(projected_corners, -1, axis=0) - projected_corners, axis=1)
        cell_span = max(6.0, float(np.median(side_lengths)) / 8.0)
        edge_sets: list[list[np.ndarray]] = [[], [], [], []]
        points, valid = self._search_edges(gray, h, cell_span=cell_span)
        for found, side in zip(points[valid], self._edge_sides[valid]):
            edge_sets[int(side)].append(found)
        return edge_sets

    def _fit_line(self, points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 2:
            return None
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        direction = np.array([vx, vy], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        return np.array([x0, y0], dtype=np.float32), direction / norm

    def _intersect_lines(
        self,
        line_a: tuple[np.ndarray, np.ndarray],
        line_b: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray | None:
        p, r = line_a
        q, s = line_b
        cross = float(r[0] * s[1] - r[1] * s[0])
        if abs(cross) < 1e-5:
            return None
        qmp = q - p
        t = float((qmp[0] * s[1] - qmp[1] * s[0]) / cross)
        return (p + t * r).astype(np.float32)

    def _plausible_repair(self, original: np.ndarray, repaired: np.ndarray, image_shape: tuple[int, int]) -> bool:
        if not self._accept_quad(repaired, image_shape):
            return False
        original_side = float(np.median(np.linalg.norm(np.roll(original, -1, axis=0) - original, axis=1)))
        movement = np.linalg.norm(repaired - original, axis=1)
        if float(np.max(movement)) > max(16.0, original_side * 0.35):
            return False
        original_area = abs(float(cv2.contourArea(original.astype(np.float32))))
        repaired_area = abs(float(cv2.contourArea(repaired.astype(np.float32))))
        if original_area < 1e-6:
            return False
        area_ratio = repaired_area / original_area
        return 0.65 <= area_ratio <= 1.45

    def _sample_all(self, gray: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        values = self._bilinear(gray, self._project_points(self._sample_points, h), keep_nan=True)
        black_vals = values[: self._black_sample_end]
        white_vals = values[self._black_sample_end : self._white_sample_end]
        bit_vals = values[self._white_sample_end :]
        black_vals = black_vals[np.isfinite(black_vals)]
        white_vals = white_vals[np.isfinite(white_vals)]
        if black_vals.size < 10 or white_vals.size < 10 or bit_vals.size < TAG36H11.nbits:
            return None
        return black_vals, white_vals, bit_vals

    def _refine_homography_from_edges(self, gray: np.ndarray, h: np.ndarray) -> tuple[np.ndarray | None, int]:
        canonical: list[np.ndarray] = []
        observed: list[np.ndarray] = []
        projected_corners = self._project_points(self._canonical_corners, h)
        side_lengths = np.linalg.norm(np.roll(projected_corners, -1, axis=0) - projected_corners, axis=1)
        cell_span = max(6.0, float(np.median(side_lengths)) / 8.0)
        points, valid = self._search_edges(gray, h, cell_span=cell_span)
        canonical.extend(self._edge_canonical[valid])
        observed.extend(points[valid])
        if len(observed) < 8:
            return None, len(observed)
        h_refined, status = cv2.findHomography(
            np.asarray(canonical, dtype=np.float32),
            np.asarray(observed, dtype=np.float32),
            method=0,
        )
        if h_refined is None or not np.all(np.isfinite(h_refined)):
            return None, len(observed)
        if status is not None and int(status.sum()) < 8:
            return None, int(status.sum())
        return h_refined.astype(np.float64), len(observed)

    def _search_edges(self, gray: np.ndarray, h: np.ndarray, *, cell_span: float) -> tuple[np.ndarray, np.ndarray]:
        projected = self._project_points(
            np.concatenate((self._edge_canonical, self._edge_canonical + self._edge_inward * 0.35)), h
        )
        points = projected[: len(self._edge_canonical)]
        normals = projected[len(self._edge_canonical) :] - points
        norms = np.linalg.norm(normals, axis=1)
        usable_normals = norms >= 1e-6
        normals[usable_normals] /= norms[usable_normals, None]
        search_points = points[:, None, :] + (self._edge_offsets * cell_span)[None, :, None] * normals[:, None, :]
        vals = self._bilinear(gray, search_points.reshape(-1, 2), keep_nan=True).reshape(len(points), -1)
        valid = usable_normals & (np.isnan(vals).sum(axis=1) <= vals.shape[1] // 3)
        # Moving along the inward normal, the correct outer edge should be a
        # white-to-black transition. This avoids snapping to internal payload
        # edges that happen to have a stronger opposite-polarity gradient.
        diffs = vals[:, :-1] - vals[:, 1:]
        safe_diffs = np.where(np.isnan(diffs), -np.inf, diffs)
        indices = np.argmax(safe_diffs, axis=1)
        strengths = safe_diffs[np.arange(len(points)), indices]
        valid &= strengths >= 10
        rows = np.arange(len(points))
        found = (search_points[rows, indices] + search_points[rows, indices + 1]) * 0.5
        return found.astype(np.float32), valid

    def _estimate_pose(
        self,
        corners: np.ndarray,
        camera_params: dict[str, float] | None,
        tag_size: float | None,
    ) -> dict[str, np.ndarray] | None:
        if not camera_params or tag_size is None:
            return None
        fx = float(camera_params["fx"])
        fy = float(camera_params["fy"])
        cx = float(camera_params["cx"])
        cy = float(camera_params["cy"])
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.asarray(camera_params.get("dist", np.zeros(5)), dtype=np.float64)
        half = tag_size / 2.0
        obj = np.array(
            [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]],
            dtype=np.float64,
        )
        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), camera_matrix, dist, flags=flag)
        if not ok:
            return None
        return {"rvec": rvec, "tvec": tvec}

    def _make_black_border_points(self) -> np.ndarray:
        pts: list[tuple[float, float]] = []
        for i in range(8):
            pts.append((i + 0.5, 0.5))
            pts.append((i + 0.5, 7.5))
            pts.append((0.5, i + 0.5))
            pts.append((7.5, i + 0.5))
        return np.unique(np.asarray(pts, dtype=np.float32), axis=0)

    def _make_white_border_points(self) -> np.ndarray:
        pts: list[tuple[float, float]] = []
        for i in range(8):
            pts.append((i + 0.5, -0.5))
            pts.append((i + 0.5, 8.5))
            pts.append((-0.5, i + 0.5))
            pts.append((8.5, i + 0.5))
        return np.asarray(pts, dtype=np.float32)

    def _make_edge_points(self) -> list[tuple[np.ndarray, np.ndarray, int]]:
        pts: list[tuple[np.ndarray, np.ndarray, int]] = []
        samples = np.linspace(0.5, 7.5, 20)
        for t in samples:
            pts.append((np.array([t, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32), 0))
            pts.append((np.array([8.0, t], dtype=np.float32), np.array([-1.0, 0.0], dtype=np.float32), 1))
            pts.append((np.array([t, 8.0], dtype=np.float32), np.array([0.0, -1.0], dtype=np.float32), 2))
            pts.append((np.array([0.0, t], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32), 3))
        return pts

    def _classify_expected(
        self,
        values: np.ndarray,
        threshold: float,
        *,
        expected_white: bool,
        min_margin: float,
    ) -> tuple[int, int]:
        margins = np.abs(values - threshold)
        reliable = margins >= min_margin
        observed_white = values > threshold
        ok = reliable & (observed_white == expected_white)
        erased_or_wrong = ~ok
        return int(ok.sum()), int(erased_or_wrong.sum())

    def _accept_quad(self, quad: np.ndarray, image_shape: tuple[int, int]) -> bool:
        quad = self._order_points(quad)
        area = abs(float(cv2.contourArea(quad.astype(np.float32))))
        if area < self.min_tag_area:
            return False
        h, w = image_shape
        if np.any(quad[:, 0] < -2) or np.any(quad[:, 0] > w + 2):
            return False
        if np.any(quad[:, 1] < -2) or np.any(quad[:, 1] > h + 2):
            return False
        side_lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if np.min(side_lengths) < 12:
            return False
        ratio = float(np.max(side_lengths) / max(1e-6, np.min(side_lengths)))
        return ratio < 3.2

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        ordered = pts[np.argsort(angles)]
        start = int(np.argmin(ordered.sum(axis=1)))
        ordered = np.roll(ordered, -start, axis=0)
        if cv2.contourArea(ordered) < 0:
            ordered[[1, 3]] = ordered[[3, 1]]
        return ordered.astype(np.float32)

    def _near_duplicate(self, quad: np.ndarray, seen: list[np.ndarray]) -> bool:
        ordered = self._order_points(quad)
        scale = max(20.0, np.sqrt(abs(cv2.contourArea(ordered))))
        for other in seen:
            dist = float(np.mean(np.linalg.norm(ordered - other, axis=1)))
            if dist < 0.08 * scale:
                return True
        return False

    def _suppress_duplicates(self, detections: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        kept_centers: list[np.ndarray] = []
        for det in detections:
            center = det.corners.mean(axis=0)
            side = float(np.mean(np.linalg.norm(np.roll(det.corners, -1, axis=0) - det.corners, axis=1)))
            if not any(
                float(np.linalg.norm(center - other_center)) < 0.35 * side for other_center in kept_centers
            ):
                kept.append(det)
                kept_centers.append(center)
        return kept

    def _project_points(self, pts: np.ndarray, h: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, h.astype(np.float64)).reshape(-1, 2)

    def _bilinear(self, gray: np.ndarray, pts: np.ndarray, *, keep_nan: bool = False) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        valid = (x0 >= 0) & (y0 >= 0) & (x1 < gray.shape[1]) & (y1 < gray.shape[0])
        out = np.full(len(pts), np.nan, dtype=np.float32)
        if np.any(valid):
            xv = x[valid]
            yv = y[valid]
            x0v = x0[valid]
            y0v = y0[valid]
            x1v = x1[valid]
            y1v = y1[valid]
            wx = xv - x0v
            wy = yv - y0v
            v00 = gray[y0v, x0v].astype(np.float32)
            v10 = gray[y0v, x1v].astype(np.float32)
            v01 = gray[y1v, x0v].astype(np.float32)
            v11 = gray[y1v, x1v].astype(np.float32)
            out[valid] = (
                v00 * (1 - wx) * (1 - wy)
                + v10 * wx * (1 - wy)
                + v01 * (1 - wx) * wy
                + v11 * wx * wy
            )
        return out if keep_nan else out[np.isfinite(out)]
