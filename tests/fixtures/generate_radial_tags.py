"""Regenerate the owned radial fixtures, or verify them with --check."""

import argparse
from pathlib import Path

import numpy as np

from napa_vision.tags import tag_cells


def render(radial: float) -> dict[str, np.ndarray]:
    cells = tag_cells(23, quiet_zone_cells=0)
    center, scale = np.array([320.0, 320.0]), 320.0
    lo, hi = np.array([300.0, 130.0]), np.array([510.0, 360.0])
    yy, xx = np.mgrid[:640, :640]
    image = np.zeros((640, 640), dtype=float)
    # Pixel centers have integer coordinates. Invert the monotone radial map at 16 area samples.
    for oy in [-0.375, -0.125, 0.125, 0.375]:
        for ox in [-0.375, -0.125, 0.125, 0.375]:
            xd, yd = (xx + ox - center[0]) / scale, (yy + oy - center[1]) / scale
            rd = np.hypot(xd, yd)
            ru = rd.copy()
            for _ in range(10):
                ru -= (ru * (1 + radial * ru * ru) - rd) / (1 + 3 * radial * ru * ru)
            factor = np.divide(ru, rd, out=np.ones_like(rd), where=rd > 0)
            x, y = center[0] + scale * xd * factor, center[1] + scale * yd * factor
            u, v = (x - lo[0]) / (hi[0] - lo[0]), (y - lo[1]) / (hi[1] - lo[1])
            inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
            ix, iy = np.clip((u * 8).astype(int), 0, 7), np.clip((v * 8).astype(int), 0, 7)
            image += np.where(inside, np.where(cells[iy, ix], 235, 30), 235)
    quad = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
    delta = (quad - center) / scale
    corners = center + scale * delta * (1 + radial * (delta * delta).sum(axis=1))[:, None]
    return dict(
        image=np.rint(image / 16).astype(np.uint8),
        corners=corners,
        tag_id=np.array(23),
        radial_k=np.array(radial),
        center=center,
        scale=np.array(scale),
        unwarped_corners=quad,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for radial in [0.0, 0.2]:
        path = Path(__file__).parent / f"radial-tag23-{radial:.1f}.npz"
        generated = render(radial)
        if args.check:
            with np.load(path) as existing:
                assert set(existing.files) == set(generated)
                for key, value in generated.items():
                    np.testing.assert_array_equal(existing[key], value, err_msg=key)
        else:
            np.savez_compressed(path, **generated)
