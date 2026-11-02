"""Render the first frames of every case of the 2026-10-30 fixed corpus (seed 2468) to a zstd-per-frame file,
for fuzz.c and subset.py.

    dump_corpus.py OUT.zst [FRAMES_PER_CASE]

Record: u16 name_len, name, u32 w, u32 h, u32 ntags, ntags * i32 id, u32 zsize, zstd(w*h bytes).
Order matches cpu_comparison.iter_corpus: main + stress per resolution, then the 18x50 range sweep."""

import struct
import sys
from multiprocessing import Pool
from pathlib import Path

import zstandard

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from benchmarks import cpu_comparison as cc  # noqa: E402
from benchmarks import performance_matrix as pm  # noqa: E402

SEED = 2468
RES = [(640, 480), (1280, 720), (1920, 1080)]


def jobs(limit):
    for res in RES:
        for name in pm.SCENARIO_NAMES:
            for i in range(limit):
                yield ("main", name, i, res)
        for name in cc.STRESS:
            for i in range(limit):
                yield ("stress", name, i, res)
    for i in range(18 * 50):
        if i % 50 < limit:
            yield ("range", None, i, None)


def render(job):
    kind, name, i, res = job
    if kind == "main":
        f = pm.render_frame(name, i, res, SEED)
    elif kind == "stress":
        f = cc.render_stress(name, i, res, SEED)
    else:
        f = pm.render_range_frame(i, resolution=(640, 480), cases=50, seed=SEED)
    name = f.name.encode()
    h, w = f.image.shape
    px = zstandard.ZstdCompressor(level=3).compress(f.image.tobytes())
    rec = struct.pack("<H", len(name)) + name + struct.pack("<III", w, h, len(f.tags))
    rec += b"".join(struct.pack("<i", t.tag_id) for t in f.tags)
    return rec + struct.pack("<I", len(px)) + px


def main():
    out = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    n = 0
    with Pool(24) as pool, out.open("wb") as fh:
        for rec in pool.imap(render, list(jobs(limit)), chunksize=4):
            fh.write(rec)
            n += 1
    print("frames", n)


if __name__ == "__main__":
    main()
