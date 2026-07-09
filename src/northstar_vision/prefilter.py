from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(slots=True)
class SkipScanConfig:
    enabled: bool = False
    stride: int = 4
    min_global_std: float = 3.0
    min_edge_strength: float = 18.0
    dilation_px: int = 56
    padding_px: int = 48
    min_roi_size_px: int = 48
    max_rois: int = 8
    max_roi_area_ratio: float = 0.45


class BlockSkipPrefilter:
    """Boyer-Moore-like skip gate for AprilTag detection.

    This does not decode tags. It cheaply samples the image on a stride and
    finds high-transition regions worth sending to OpenCV's detector. If the
    image is too busy, it returns None so the caller can run full-frame
    detection instead of thrashing through many tiny ROIs.
    """

    def __init__(self, config: SkipScanConfig) -> None:
        self.config = config

    def find_rois(self, image: np.ndarray) -> list[Roi] | None:
        if not self.config.enabled:
            return None
        gray = image
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        stride = max(1, int(self.config.stride))
        sampled = gray[::stride, ::stride]
        sampled_std = 0.0 if sampled.size == 0 else float(cv2.meanStdDev(sampled)[1][0, 0])
        if sampled_std < self.config.min_global_std:
            return []

        gradient = self._sampled_gradient(sampled)
        if float(np.max(gradient)) < self.config.min_edge_strength:
            return []

        threshold = max(self.config.min_edge_strength, float(np.mean(gradient) + np.std(gradient)))
        mask = (gradient >= threshold).astype(np.uint8) * 255
        kernel_size = max(3, int(round(self.config.dilation_px / stride)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        rois: list[Roi] = []
        min_sampled = max(2, int(round(self.config.min_roi_size_px / stride)))
        padding = int(self.config.padding_px)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if w < min_sampled or h < min_sampled or area < min_sampled:
                continue
            x0 = max(0, int(x * stride) - padding)
            y0 = max(0, int(y * stride) - padding)
            x1 = min(width, int((x + w) * stride) + padding)
            y1 = min(height, int((y + h) * stride) + padding)
            if x1 - x0 >= self.config.min_roi_size_px and y1 - y0 >= self.config.min_roi_size_px:
                rois.append(Roi(x0, y0, x1 - x0, y1 - y0))

        if not rois:
            return []
        rois = self._merge_rois(rois, width, height)
        total_area = sum(roi.area for roi in rois)
        if len(rois) > self.config.max_rois or total_area / float(width * height) > self.config.max_roi_area_ratio:
            return None
        return rois

    @staticmethod
    def _sampled_gradient(sampled: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(sampled, (3, 3), 0)
        grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
        return (np.abs(grad_x) + np.abs(grad_y)).astype(np.float32)

    @staticmethod
    def _merge_rois(rois: list[Roi], width: int, height: int) -> list[Roi]:
        merged: list[Roi] = []
        for roi in sorted(rois, key=lambda item: item.area, reverse=True):
            current = roi
            changed = True
            while changed:
                changed = False
                remaining: list[Roi] = []
                for other in merged:
                    if _overlaps_or_touches(current, other):
                        current = _union(current, other, width, height)
                        changed = True
                    else:
                        remaining.append(other)
                merged = remaining
            merged.append(current)
        return sorted(merged, key=lambda item: item.area, reverse=True)


def _overlaps_or_touches(a: Roi, b: Roi) -> bool:
    return not (a.x + a.width < b.x or b.x + b.width < a.x or a.y + a.height < b.y or b.y + b.height < a.y)


def _union(a: Roi, b: Roi, width: int, height: int) -> Roi:
    x0 = max(0, min(a.x, b.x))
    y0 = max(0, min(a.y, b.y))
    x1 = min(width, max(a.x + a.width, b.x + b.width))
    y1 = min(height, max(a.y + a.height, b.y + b.height))
    return Roi(x0, y0, x1 - x0, y1 - y0)
