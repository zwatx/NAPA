from __future__ import annotations

import cv2
import numpy as np

from .types import FrameResult


def draw_overlay(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    out = frame.copy()
    for obs in result.image_observations:
        corners = obs.corners.astype(np.int32)
        cv2.polylines(out, [corners], True, (0, 255, 0), 2)
        center = tuple(corners.mean(axis=0).astype(int))
        cv2.putText(out, f"id={obs.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    health = result.health
    text = (
        f"FPS {health.get('fps', 0):.1f} | latency {health.get('latency_ms', 0):.1f} ms | "
        f"tags {len(result.image_observations)}"
    )
    cv2.putText(out, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    return out
