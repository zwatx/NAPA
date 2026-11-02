# Identical-output speedup: reproduction

Baseline core: `de613ad`. Candidate core: the commit that adds this folder. Both are built with the release flags of the
[2026-10-30 run](../2026-10-30_cpu_full_corpus/run.sh), timed on one core of an i9-14900K (CPU 4). From the
repository root, with the benchmark environment of the [original comparison](../2026-10-28_cpu_comparison/commands.md)
and `zstandard` installed:

```bash
mkdir -p work
git show de613ad:src/napa_vision/_native/napa_core.c > work/baseline.c
build() {
  cc -fno-strict-overflow -DNDEBUG -g -O2 -Wall -Wextra -fPIC -O3 -std=c99 -fvisibility=hidden -fno-wrapv \
    -shared -Isrc/napa_vision/_native -o "$1" "$2" -lm
}
build work/baseline.so work/baseline.c
build work/candidate.so src/napa_vision/_native/napa_core.c
D=benchmarks/results/2026-11-02_identical_output_speedup
```

## Timing: `summary.json`, `latency.csv`

`run_ab.py` renders the fixed corpus with `cpu_comparison.iter_corpus` (seed 2468: 20 scenes and 4 stress cases, 50
frames each at 640×480, 1280×720 and 1920×1080, and the 18 × 50 range sweep) and times it with
`cpu_comparison.run_interleaved`: one warm-up and ten timed calls per frame, backends in shuffled order. Its four
backends are each core at `min_tag_px` 32 and 16, all through `NapaDetector`, each bound to its own library. About
15 minutes.

```bash
AB_CPU=4 OPENBLAS_NUM_THREADS=1 python $D/run_ab.py "$PWD/work/baseline.so" "$PWD/work/candidate.so" work/ab
```

## Identical output on the full corpus

`stream_corpus.py` renders the same 4,500 frames in memory and pipes them to `exact`, which runs both cores at both
settings and compares every field of every detection (ID, hamming, margin and corners, bit for bit). Nothing is
written to disk.

```bash
cc -O2 -Isrc/napa_vision/_native -o work/exact $D/exact.c -ldl
python $D/stream_corpus.py 12 | work/exact work/baseline.so work/candidate.so
```

## Fuzz

`fuzz.sh` compares both cores under AddressSanitizer and UBSan on random crops of corpus frames: odd sizes down to
16 pixels, strides padded by up to 70 bytes, three families (one reversed) and `min_tag_px` 8 to 48. The corpus
file holds the first 20 frames of every case (1,800 frames, about 900 MB).

```bash
python $D/dump_corpus.py work/corpus20.zst 20
$D/fuzz.sh work/baseline.c work/corpus20.zst 5 6
```

## aarch64 and `NAPA_NO_SIMD`

`dump.c` writes every detection of a 90-frame raw subset at both settings. Build it with each core for aarch64
(here `zig cc`, run under `qemu-aarch64-static`) and for x86 with `-DNAPA_NO_SIMD`, then compare the outputs.

```bash
python $D/subset.py work/corpus20.zst work/subset.raw
for v in baseline candidate; do
  src=$([ $v = baseline ] && echo work/baseline.c || echo src/napa_vision/_native/napa_core.c)
  python -m ziglang cc -target aarch64-linux-musl -O3 -std=c99 -fno-wrapv -static -Isrc/napa_vision/_native \
    -o work/dump_$v.arm $D/dump.c $src -lm
  cc -O3 -std=c99 -fno-wrapv -DNAPA_NO_SIMD -Isrc/napa_vision/_native -o work/dump_$v.nosimd $D/dump.c $src -lm
  qemu-aarch64-static work/dump_$v.arm work/subset.raw > work/out_$v.arm
  work/dump_$v.nosimd work/subset.raw > work/out_$v.nosimd
done
cmp work/out_baseline.arm work/out_candidate.arm && cmp work/out_baseline.nosimd work/out_candidate.nosimd
```
