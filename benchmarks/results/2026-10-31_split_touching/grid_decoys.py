"""False-ID stress for split_touching: AprilGrid-layout boards whose tags all carry random payloads.

Every detection is a false ID. Payloads within Hamming 4 of a tag36h11 code in any rotation are redrawn.
--real draws real codes instead, as a positive control. A label ending in "+s" enables split_touching.
From the repository root: python benchmarks/results/2026-10-31_split_touching/grid_decoys.py \
    base=LIB on+s=LIB --boards 1200   (LIB: a built core, e.g. the bundled _napa_core*.so)
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)


def valid_codes():
    from napa_vision.tags import tag_cells

    out = set()
    for i in range(587):
        p = tag_cells(i, quiet_zone_cells=0)[1:7, 1:7].astype(np.uint8)
        for k in range(4):
            out.add(int("".join(map(str, np.rot90(p, k).ravel())), 2))
    return np.array(sorted(out), np.uint64)


CODES = None
REAL = False
PLACED = []


def payload(rng):
    if REAL:
        from napa_vision.tags import tag_cells

        i = int(rng.integers(0, 587))
        PLACED.append(i)
        return 1 - tag_cells(i, quiet_zone_cells=0)[1:7, 1:7].astype(np.uint8)  # 1 = dark, as decoys below
    while True:
        bits = rng.integers(0, 2, (6, 6)).astype(np.uint8)
        v = np.uint64(int("".join(map(str, bits.ravel())), 2))
        x = CODES ^ v
        d = np.array([bin(int(t)).count("1") for t in x])
        if d.min() > 4:
            return bits


def board(rng, n, layout):
    """Cell pattern (True = dark). grid: AprilGrid, 2-cell gaps with corner squares. checker: tags in the white
    squares of a checkerboard whose black squares touch every tag corner."""
    if layout == "grid":
        size = n * 8 + (n + 1) * 2
        b = np.zeros((size, size), bool)
        for i in range(n + 1):
            for j in range(n + 1):
                b[i * 10 : i * 10 + 2, j * 10 : j * 10 + 2] = True
        for i in range(n):
            for j in range(n):
                y, x = 2 + i * 10, 2 + j * 10
                t = np.ones((8, 8), bool)
                t[1:7, 1:7] = payload(rng) == 1
                b[y : y + 8, x : x + 8] = t
    else:
        size = 10 * (2 * n)
        b = np.zeros((size, size), bool)
        for i in range(2 * n):
            for j in range(2 * n):
                if (i + j) % 2:
                    b[i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = True
                else:
                    t = np.ones((8, 8), bool)
                    t[1:7, 1:7] = payload(rng) == 1
                    b[i * 10 + 1 : i * 10 + 9, j * 10 + 1 : j * 10 + 9] = t
    return np.pad(b, 3)


def render(rng, layout, small):
    from benchmarks import performance_matrix as pm

    W, H = 1280, 720
    n = int(rng.integers(3, 7))
    cells = board(rng, n, layout)
    c = cells.shape[0]
    px = 8
    src = np.where(cells, 30, 235).astype(np.uint8).repeat(px, 0).repeat(px, 1)
    side = float(rng.uniform(150, 350) if small else rng.uniform(300, 900))
    side = min(side, 640)
    center = (
        float(rng.uniform(side * 0.6, W - side * 0.6)),
        float(rng.uniform(min(side * 0.55, H / 2), max(H - side * 0.55, H / 2))),
    )
    dst = pm.tag_pose(rng, side, center, max_tilt_deg=45.0) * 4
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [c * px, 0], [c * px, c * px], [0, c * px]]), np.float32(dst))
    bg = float(rng.uniform(60, 200))
    big = cv2.warpPerspective(src, M, (W * 4, H * 4), flags=cv2.INTER_LINEAR, borderValue=bg)
    img = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    s = float(rng.uniform(0, 1.2))
    if s > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), s)
    img += rng.normal(0, rng.uniform(0.5, 4), img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), n * n if layout == "grid" else 2 * n * n


_D = {}


def init(libs, real):
    global CODES, REAL
    CODES = valid_codes()
    REAL = real
    from napa_vision.napa_detector import NapaDetector, _library

    for label, lib in libs:
        for m in (32, 16):
            os.environ["NAPA_CORE_LIBRARY"] = lib
            _library.cache_clear()
            extra = {"split_touching": True} if label.endswith("+s") else {}
            _D[label, m] = NapaDetector(min_tag_px=m, max_detections=1024, **extra)


def work(task):
    i, layout = task
    rng = np.random.default_rng((4711, i, layout == "grid"))
    PLACED.clear()
    img, k = render(rng, layout, small=i % 2 == 1)
    placed = set(PLACED)
    res = {key: [(o.tag_id, o.hamming) for o in d.detect(img)] for key, d in _D.items()}
    if REAL:  # correct = placed ids; false = anything else
        return k, {key: [(t, h if t not in placed else -1) for t, h in r] for key, r in res.items()}
    return k, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("libs", nargs="+")
    ap.add_argument("--boards", type=int, default=400)
    ap.add_argument("--real", action="store_true", help="positive control: real codes")
    a = ap.parse_args()
    libs = [(s.split("=", 1)[0], str(Path(s.split("=", 1)[1]).resolve())) for s in a.libs]
    with get_context("fork").Pool(6, init, (libs, a.real)) as pool:
        out = pool.map(work, [(i, "grid") for i in range(a.boards)], chunksize=4)
    decoys = sum(k for k, _ in out)
    print(f"{a.boards} boards, {decoys} tags")
    for key in sorted(out[0][1]):
        fs = [h for _, r in out for _, h in r[key] if h >= 0]
        ok = sum(1 for _, r in out for _, h in r[key] if h < 0)
        hamming = np.bincount(fs, minlength=4).tolist()
        print(f"  {key[0]:8s} min{key[1]}: correct {ok}  false IDs {len(fs)}  by hamming {hamming}")


if __name__ == "__main__":
    main()
