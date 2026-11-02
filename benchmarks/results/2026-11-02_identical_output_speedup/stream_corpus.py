"""Stream the full 2026-10-30 corpus (seed 2468) to stdout, rendered in memory, nothing on disk.

Record: u16 name_len, name, u32 w, u32 h, u32 ntags, ntags * i32 id, w*h pixel bytes. Order: per resolution the
20 main scenes then 4 stress scenes (50 each), then the 18 x 50 range sweep, as cpu_comparison.iter_corpus."""

import struct
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from benchmarks import cpu_comparison as cc  # noqa: E402
from benchmarks import performance_matrix as pm  # noqa: E402

SEED, N = 2468, 50


def jobs():
    for res in [(640, 480), (1280, 720), (1920, 1080)]:
        for name in pm.SCENARIO_NAMES:
            for i in range(N):
                yield ("main", name, i, res)
        for name in cc.STRESS:
            for i in range(N):
                yield ("stress", name, i, res)
    for i in range(18 * N):
        yield ("range", None, i, None)


def render(job):
    kind, name, i, res = job
    if kind == "main":
        f = pm.render_frame(name, i, res, SEED)
    elif kind == "stress":
        f = cc.render_stress(name, i, res, SEED)
    else:
        f = pm.render_range_frame(i, resolution=(640, 480), cases=N, seed=SEED)
    label = f.name.encode()
    h, w = f.image.shape
    head = struct.pack("<H", len(label)) + label + struct.pack("<III", w, h, len(f.tags))
    return head + b"".join(struct.pack("<i", t.tag_id) for t in f.tags) + f.image.tobytes()


if __name__ == "__main__":
    out = sys.stdout.buffer
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for rec in pool.imap(render, jobs(), chunksize=2):
            out.write(rec)
    out.flush()
