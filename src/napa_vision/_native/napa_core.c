/*
 * Napa core: a sparse-sampling ("rake") AprilTag 36h11 detector in portable C99.
 *
 * Pipeline per frame, level by level (each level doubles the sampling stride):
 *   1. Rake: point-sample the image every 3/4 of the smallest cell. A tag's black border is one
 *      cell wide, so every border crossing contains a sample from its dark core. Coarser levels
 *      take every second sample of the level before.
 *   2. Threshold each sample against its 4x4-tile min/max over a 3x3 tile neighbourhood. Tiles
 *      without contrast are skipped outright: they cannot contain a tag edge. Rows of dark
 *      samples become runs (SIMD compare and bit masks), joined into blobs by union-find.
 *   3. Keep tag-sized, tag-shaped blobs; each one's extreme points give a rough quad.
 *   4. Fit each side to its full-resolution edge along the normal, decode once cheaply to reject
 *      non-tags, then refine coarse to fine and intersect the four lines for the corners.
 *   5. Sample the 8x8 cell grid through the homography and decode. Multi-index hashing narrows
 *      the 2,348 rotated codewords to the few sharing an exact block with the payload.
 *   6. Repair the cases a plain fit gets wrong: a quiet zone darker than the background, a side
 *      fitted one cell out, and edges biased by clipped (saturated) exposure.
 *
 * The library has no dependencies and exposes a plain C ABI.
 */

#ifdef NAPA_PROFILE
#define _POSIX_C_SOURCE 199309L
#include <time.h>
#endif
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* SSE2 is part of every x86-64 CPU; MSVC doesn't define __SSE2__, so check its macros too. */
#if !defined(NAPA_NO_SIMD) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define NAPA_SSE2 1
#endif

#ifdef NAPA_SSE2
#include <emmintrin.h>
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <tmmintrin.h>
#define NAPA_SSSE3_DISPATCH 1 /* SSSE3 isn't baseline x86-64: compiled per function, chosen at runtime */
#endif
#elif defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
#include <arm_neon.h>
#endif

#include "napa_core.h"
#include "tag36h11_codes.h"

#define NAPA_ROTATED (NAPA_TAG36H11_COUNT * 4)

/* The builtin is one instruction on ARM64 and x86 with POPCNT, but a library call on baseline x86-64. */
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__POPCNT__) || !(defined(__x86_64__) || defined(__i386__)))
#define napa_popcount64(x) __builtin_popcountll(x)
#else
static int napa_popcount64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
}
#endif

/* A horizontal run of dark samples in one level row, and a blob's sample count and bounding box. */
typedef struct {
    uint16_t x0, x1, y, pad;
} napa_run;

typedef struct {
    int32_t count;
    uint16_t minx, maxx, miny, maxy;
} blob_box;

typedef struct {
    int count;
    int minx, maxx, miny, maxy;
    int ext[16][2]; /* extreme points along the 16 directions of comp_dirs, in angular order */
    int extv[16];
    int row_y, row_x0, row_x1; /* the blob's current row: its leftmost and rightmost dark sample */
} component;

struct napa_detector {
    uint64_t codes[NAPA_ROTATED];
    int16_t ids[NAPA_ROTATED];
    int8_t rotations[NAPA_ROTATED];
    /* Multi-index hashing: a code within H bits matches at least one of H + 1 disjoint blocks
     * exactly, so only codes sharing a block are compared. Six 6-bit blocks cover H <= 5 and
     * four 9-bit blocks cover H <= 3. Lists are stored compressed by bucket. */
    uint16_t mi6_start[6][65];
    uint16_t mi6_list[6][NAPA_ROTATED];
    uint16_t mi9_start[4][513];
    uint16_t mi9_list[4][NAPA_ROTATED];
    uint32_t stamp[NAPA_ROTATED];
    uint32_t stamp_now;
    int ssse3; /* the CPU supports SSSE3 (x86 only; picked once at creation) */

    uint8_t *level;
    uint8_t *tile_min;
    uint8_t *tile_max;
    uint8_t *nb_lo; /* one band's tile neighbourhood min, max, and threshold (row buffers) */
    uint8_t *nb_hi;
    uint8_t *nb_t;
    uint8_t *nb_f;
    napa_run *runs;
    int32_t *parent; /* union-find over runs; a blob's root is its first run in raster order */
    blob_box *boxes;
    int32_t *slot;   /* root run -> candidate index, or -1 */
    int32_t *roots;  /* the root runs, in raster order */
    component *cands;
    size_t cand_capacity;
    uint64_t *row_bits;
    uint8_t *row_min;
    uint8_t *row_max;
    uint8_t *row_thr;
    size_t row_capacity;
    size_t run_capacity;
    size_t level_capacity;
    uint8_t *floors; /* tile contrast floor of each noise region, row-major */
    size_t floor_alloc;
    int floor_cols, floor_rows, floor_uniform; /* uniform: every region has the same floor */
    uint16_t *box_rows; /* the averaging rake's two scratch rows */
    size_t box_capacity;
    size_t tile_capacity;
};

typedef struct {
    double x, y;
} vec2;

/* ---------------------------------------------------------------- family */

static uint64_t rotate_ccw(uint64_t code) {
    /* code holds a 6x6 row-major grid, MSB = row 0 col 0. Return it rotated 90 degrees counter-clockwise. */
    uint64_t out = 0;
    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) {
            /* rot90 (CCW): out[r][c] = in[c][5 - r] */
            int src = c * 6 + (5 - r);
            uint64_t bit = (code >> (35 - src)) & 1u;
            out = (out << 1) | bit;
        }
    }
    return out;
}

static void build_index(const uint64_t *codes, int blocks, int bits, uint16_t *start, uint16_t *list) {
    int buckets = 1 << bits;
    for (int b = 0; b < blocks; b++) {
        uint16_t *st = start + b * (buckets + 1);
        uint16_t *ls = list + b * NAPA_ROTATED;
        memset(st, 0, sizeof(uint16_t) * (size_t)(buckets + 1));
        for (int i = 0; i < NAPA_ROTATED; i++) st[((codes[i] >> (b * bits)) & (uint64_t)(buckets - 1)) + 1]++;
        for (int v = 0; v < buckets; v++) st[v + 1] = (uint16_t)(st[v + 1] + st[v]);
        uint16_t fill[512];
        memcpy(fill, st, sizeof(uint16_t) * (size_t)buckets);
        for (int i = 0; i < NAPA_ROTATED; i++) {
            int v = (int)((codes[i] >> (b * bits)) & (uint64_t)(buckets - 1));
            ls[fill[v]++] = (uint16_t)i;
        }
    }
}

/* Index of the rotated code within max_bits (at most 5) of `code`, or -1. tag36h11 codes (with
 * rotations) are at least 11 bits apart, so any match is unique. */
static int nearest_code(napa_detector *det, uint64_t code, int max_bits, int *distance) {
    int blocks, bits;
    const uint16_t *start, *list;
    if (max_bits <= 3) {
        blocks = 4, bits = 9, start = &det->mi9_start[0][0], list = &det->mi9_list[0][0];
    } else {
        blocks = 6, bits = 6, start = &det->mi6_start[0][0], list = &det->mi6_list[0][0];
    }
    if (++det->stamp_now == 0) {
        memset(det->stamp, 0, sizeof det->stamp);
        det->stamp_now = 1;
    }
    int buckets = 1 << bits;
    for (int b = 0; b < blocks; b++) {
        const uint16_t *st = start + b * (buckets + 1);
        const uint16_t *ls = list + b * NAPA_ROTATED;
        int v = (int)((code >> (b * bits)) & (uint64_t)(buckets - 1));
        for (int e = st[v]; e < st[v + 1]; e++) {
            int i = ls[e];
            if (det->stamp[i] == det->stamp_now) continue;
            det->stamp[i] = det->stamp_now;
            int d = napa_popcount64(code ^ det->codes[i]);
            if (d <= max_bits) {
                *distance = d;
                return i;
            }
        }
    }
    return -1;
}

static void load_family(napa_detector *det) {
    for (int rot = 0; rot < 4; rot++) {
        for (int id = 0; id < NAPA_TAG36H11_COUNT; id++) {
            uint64_t code = napa_tag36h11_codes[id];
            for (int k = 0; k < rot; k++) {
                code = rotate_ccw(code);
            }
            int index = rot * NAPA_TAG36H11_COUNT + id;
            det->codes[index] = code;
            det->ids[index] = (int16_t)id;
            det->rotations[index] = (int8_t)rot;
        }
    }
    build_index(det->codes, 6, 6, &det->mi6_start[0][0], &det->mi6_list[0][0]);
    build_index(det->codes, 4, 9, &det->mi9_start[0][0], &det->mi9_list[0][0]);
}

/* ---------------------------------------------------------------- lifecycle */

napa_detector *napa_create(void) {
    napa_detector *det = (napa_detector *)calloc(1, sizeof(napa_detector));
    if (det != NULL) {
        load_family(det);
#ifdef NAPA_SSSE3_DISPATCH
        __builtin_cpu_init();
        det->ssse3 = __builtin_cpu_supports("ssse3") != 0;
#endif
    }
    return det;
}

void napa_destroy(napa_detector *det) {
    if (det == NULL) {
        return;
    }
    free(det->level);
    free(det->tile_min);
    free(det->tile_max);
    free(det->nb_lo);
    free(det->nb_hi);
    free(det->nb_t);
    free(det->nb_f);
    free(det->runs);
    free(det->parent);
    free(det->boxes);
    free(det->slot);
    free(det->roots);
    free(det->cands);
    free(det->row_bits);
    free(det->floors);
    free(det->box_rows);
    free(det->row_min);
    free(det->row_max);
    free(det->row_thr);
    free(det);
}

static int ensure_buffers(napa_detector *det, size_t pixels, size_t tiles, size_t row, size_t max_runs) {
    if (max_runs > det->run_capacity) {
        void *p;
        if ((p = realloc(det->runs, max_runs * sizeof(napa_run))) == NULL) return -1;
        det->runs = (napa_run *)p;
        if ((p = realloc(det->parent, max_runs * sizeof(int32_t))) == NULL) return -1;
        det->parent = (int32_t *)p;
        if ((p = realloc(det->boxes, max_runs * sizeof(blob_box))) == NULL) return -1;
        det->boxes = (blob_box *)p;
        if ((p = realloc(det->slot, max_runs * sizeof(int32_t))) == NULL) return -1;
        det->slot = (int32_t *)p;
        if ((p = realloc(det->roots, max_runs * sizeof(int32_t))) == NULL) return -1;
        det->roots = (int32_t *)p;
        det->run_capacity = max_runs;
    }
    if (row > det->row_capacity) {
        uint8_t *p;
        /* Row buffers run 128 bytes past the row: whole tiles plus whole 64-sample mask words. */
        if ((p = (uint8_t *)realloc(det->row_min, row + 128)) == NULL) return -1;
        det->row_min = p;
        if ((p = (uint8_t *)realloc(det->row_max, row + 128)) == NULL) return -1;
        det->row_max = p;
        if ((p = (uint8_t *)realloc(det->row_thr, row + 128)) == NULL) return -1;
        det->row_thr = p;
        /* Tile rows: row / 4 + 1 tiles plus one padding tile at each end. */
        if ((p = (uint8_t *)realloc(det->nb_lo, row / 4 + 3)) == NULL) return -1;
        det->nb_lo = p;
        if ((p = (uint8_t *)realloc(det->nb_hi, row / 4 + 3)) == NULL) return -1;
        det->nb_hi = p;
        if ((p = (uint8_t *)realloc(det->nb_t, row / 4 + 3)) == NULL) return -1;
        det->nb_t = p;
        if ((p = (uint8_t *)realloc(det->nb_f, row / 4 + 3)) == NULL) return -1;
        det->nb_f = p;
        uint64_t *bits = (uint64_t *)realloc(det->row_bits, (row / 64 + 2) * sizeof(uint64_t));
        if (bits == NULL) return -1;
        det->row_bits = bits;
        det->row_capacity = row;
    }
    if (pixels > det->level_capacity) {
        /* 64 spare bytes: the last row's SIMD compares read up to 63 samples past its end. */
        uint8_t *level = (uint8_t *)realloc(det->level, pixels + 64);
        if (level == NULL) return -1;
        det->level = level;
        det->level_capacity = pixels;
    }
    if (tiles > det->tile_capacity) {
        uint8_t *p;
        if ((p = (uint8_t *)realloc(det->tile_min, tiles)) == NULL) return -1;
        det->tile_min = p;
        if ((p = (uint8_t *)realloc(det->tile_max, tiles)) == NULL) return -1;
        det->tile_max = p;
        det->tile_capacity = tiles;
    }
    return 0;
}

/* ---------------------------------------------------------------- sampling */

static inline double sample_bilinear(const uint8_t *img, int w, int h, int stride, double x, double y) {
    /* Negated compares also clamp NaN (from a degenerate homography) instead of reading out of bounds. */
    if (!(x >= 0)) x = 0;
    if (!(y >= 0)) y = 0;
    if (!(x <= w - 1.001)) x = w - 1.001;
    if (!(y <= h - 1.001)) y = h - 1.001;
    int x0 = (int)x;
    int y0 = (int)y;
    double fx = x - x0;
    double fy = y - y0;
    const uint8_t *p = img + (size_t)y0 * stride + x0;
    double a = p[0] + (p[1] - p[0]) * fx;
    double b = p[stride] + (p[stride + 1] - p[stride]) * fx;
    return a + (b - a) * fy;
}

/* ---------------------------------------------------------------- threshold */

#define TILE 4
/* Tuning. A level keeps blobs whose bounding box spans up to NAPA_HI samples: a tag stays isolated
 * from its background only where its one-cell white margin spans about two samples (a side of 16+
 * samples), and its border stays solid under the tile threshold up to a side of about 64 samples,
 * which a rotated or tilted tag stretches to a box of about 96. Levels continue until the frame is
 * under NAPA_LAST samples across, so a frame-filling tag is under 64 samples at the last level. */
#define NAPA_HI 96
#define NAPA_LAST 144
/* Edges whose ramp is wider than this many cells overlap the next edge's ramp (motion or defocus
 * blur): locate them by midpoint crossing instead of the gradient's centre of mass. */
#define NAPA_CROSSING_CELLS 0.6
/* The final fit takes a profile every 1.5 px along each side, up to this many. */
#define NAPA_FINAL_POINTS 96
/* Midpoint-crossing passes: the plateau levels are re-read around each new estimate. */
#define NAPA_CROSS_ITERS 3
/* Noise: the tile contrast floor is NAPA_FLOOR_SIGMAS times the noise measured in the tile's region
 * (NAPA_REGION samples square at the base level), at most NAPA_FLOOR_CAP (a cap, because texture
 * inflates the noise estimate). Regions keep a clean part of the frame sensitive when another part is
 * noisy or textured. A frame-wide floor above NAPA_NOISY_FLOOR (noise sigma of about 7) re-rakes the
 * frame with each sample averaging its block, which divides the noise by the stride; that level's
 * floors can then be NAPA_BOX_SIGMAS, still far below the point rake's in grey levels, which keeps
 * noise blobs out. */
#define NAPA_REGION 64
#define NAPA_FLOOR_SIGMAS 6.0
#define NAPA_FLOOR_CAP 64
#define NAPA_NOISY_FLOOR 42
#define NAPA_BOX_SIGMAS 7.0

/* Debug-only candidate trace (compile with -DNAPA_TRACE): one record per non-trivial blob with the
 * stage that rejected it, for attributing missed tags to a pipeline stage. Not thread-safe. */
#ifdef NAPA_TRACE
enum { TRACE_CAP = 1 << 16 };
static int napa_trace_buf[TRACE_CAP][6];
static int napa_trace_n;
NAPA_API int napa_trace_read(int *out, int max) {
    int n = napa_trace_n < max ? napa_trace_n : max;
    memcpy(out, napa_trace_buf, sizeof(int) * 6 * (size_t)n);
    napa_trace_n = 0;
    return n;
}
#define TRACE(stage)                                                                                         \
    do {                                                                                                     \
        if (napa_trace_n < TRACE_CAP) {                                                                      \
            int *r_ = napa_trace_buf[napa_trace_n++];                                                        \
            r_[0] = s, r_[1] = t_minx * s, r_[2] = t_miny * s, r_[3] = t_maxx * s, r_[4] = t_maxy * s,      \
            r_[5] = (stage);                                                                                 \
        }                                                                                                    \
    } while (0)
#else
#define TRACE(stage) ((void)0)
#endif

/* Debug-only stage timers (compile with -DNAPA_PROFILE): cumulative nanoseconds per stage. */
#ifdef NAPA_PROFILE
enum { PROF_RAKE, PROF_THRESHOLD, PROF_BLOBS, PROF_QUAD, PROF_FIT1, PROF_DECODE1, PROF_REFINE, PROF_DECODE2, PROF_N };
static uint64_t napa_prof[PROF_N + 1];
static uint64_t prof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}
NAPA_API int napa_profile_read(uint64_t *out, int max) {
    int n = PROF_N + 1 < max ? PROF_N + 1 : max;
    memcpy(out, napa_prof, sizeof(uint64_t) * (size_t)n);
    memset(napa_prof, 0, sizeof napa_prof);
    return n;
}
#define PROF_START uint64_t prof_t_ = prof_now(), prof_c_ = 0
#define PROF_MARK(stage)                                                                                     \
    do {                                                                                                     \
        prof_c_ = prof_now();                                                                                \
        napa_prof[stage] += prof_c_ - prof_t_;                                                               \
        prof_t_ = prof_c_;                                                                                   \
    } while (0)
#else
#define PROF_START ((void)0)
#define PROF_MARK(stage) ((void)0)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define napa_ctz64(x) __builtin_ctzll(x)
#else
static int napa_ctz64(uint64_t x) {
    int n = 0;
    while (!(x & 1)) x >>= 1, n++;
    return n;
}
#endif

/* ---------------------------------------------------------------- rake */

/* dst[x] = src[2x] for x < n. Coarser levels are every second sample of the level before. */
static void gather2(uint8_t *dst, const uint8_t *src, int n) {
    int x = 0;
#ifdef NAPA_SSE2
    const __m128i even = _mm_set1_epi16(0x00ff);
    for (; x + 16 <= n; x += 16) {
        __m128i a = _mm_and_si128(_mm_loadu_si128((const __m128i *)(src + 2 * x)), even);
        __m128i b = _mm_and_si128(_mm_loadu_si128((const __m128i *)(src + 2 * x + 16)), even);
        _mm_storeu_si128((__m128i *)(dst + x), _mm_packus_epi16(a, b));
    }
#elif defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    for (; x + 16 <= n; x += 16) vst1q_u8(dst + x, vld2q_u8(src + 2 * x).val[0]);
#endif
    for (; x < n; x++) dst[x] = src[2 * x];
}

#ifdef NAPA_SSSE3_DISPATCH
__attribute__((target("ssse3"))) static int gather3_ssse3(uint8_t *dst, const uint8_t *src, int n) {
    /* Bytes 0, 3, ..., 45 of three 16-byte loads, each shuffled into its slice of the output. */
    const __m128i m0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m128i m1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    const __m128i m2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    int x = 0;
    for (; x + 16 <= n; x += 16) {
        const uint8_t *p = src + 3 * x;
        __m128i v = _mm_or_si128(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)p), m0),
                                 _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 16)), m1));
        v = _mm_or_si128(v, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(p + 32)), m2));
        _mm_storeu_si128((__m128i *)(dst + x), v);
    }
    return x;
}
#endif

/* dst[x] = src[s * x] for x < n, the rake along one image row. src must be readable to
 * src[s * (n - 1)] only, so vector paths stop a full vector before the end. */
static void gather_row(uint8_t *dst, const uint8_t *src, int n, int s, int ssse3) {
    int x = 0;
    if (s == 1) {
        memcpy(dst, src, (size_t)n);
        return;
    }
    if (s == 2) {
        int safe = n - 1; /* the SIMD loads read src[2x + 1] */
        gather2(dst, src, safe > 0 ? safe : 0);
        x = safe > 0 ? safe : 0;
    } else if (s == 3) {
#ifdef NAPA_SSSE3_DISPATCH
        if (ssse3 && n > 1) x = gather3_ssse3(dst, src, n - 1); /* the loads read up to src[3x + 47] */
#elif defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
        for (; x + 16 <= n - 1; x += 16) vst1q_u8(dst + x, vld3q_u8(src + 3 * x).val[0]);
#endif
    }
    (void)ssse3;
    for (; x < n; x++) dst[x] = src[(size_t)x * s];
}

/* The averaging rake: level[y][x] = mean of the s x s image block centred on (x * s, y * s), clipped
 * to the image. Each sample stands for its whole block, so sensor noise drops by a factor of s, and a
 * border cell's core still leaves every sample at least two thirds dark. Needs s <= 16 (16-bit sums)
 * and two scratch rows of w samples: acc (column sums) and hs (horizontal box sums). */
static void box_rake(uint8_t *level, int lw, int lh, const uint8_t *img, int w, int h, int stride, int s,
                     uint16_t *acc, uint16_t *hs) {
    int half = (s - 1) / 2;
    uint32_t m = (65536u + (uint32_t)(s * s) / 2) / (uint32_t)(s * s); /* 1 / (s * s) in 0.16 fixed point */
    for (int y = 0; y < lh; y++) {
        int r0 = y * s - half, r1 = r0 + s;
        if (r0 < 0) r0 = 0;
        if (r1 > h) r1 = h;
        const uint8_t *row = img + (size_t)r0 * stride;
        for (int x = 0; x < w; x++) acc[x] = row[x];
        for (int r = r0 + 1; r < r1; r++) {
            row = img + (size_t)r * stride;
            for (int x = 0; x < w; x++) acc[x] = (uint16_t)(acc[x] + row[x]);
        }
        /* hs[c] = acc[c] + ... + acc[c + s - 1], for every block start c that fits in the row. */
        for (int c = 0; c + s <= w; c++) hs[c] = acc[c];
        for (int k = 1; k < s; k++)
            for (int c = 0; c + s <= w; c++) hs[c] = (uint16_t)(hs[c] + acc[c + k]);
        uint8_t *dst = level + (size_t)y * lw;
        /* Whole blocks are x_lo <= x < x_hi; the rest are clipped by the image edge. */
        int x_lo = (half + s - 1) / s, x_hi = (w - s + half) / s + 1;
        if (x_hi > lw) x_hi = lw;
        if (r1 - r0 < s) x_lo = x_hi = 0; /* a clipped row band: every block is partial */
        for (int x = x_lo; x < x_hi; x++) dst[x] = (uint8_t)((hs[x * s - half] * m + 32768u) >> 16);
        for (int x = 0; x < lw; x++) {
            if (x == x_lo && x_lo < x_hi) x = x_hi;
            if (x >= lw) break;
            int c0 = x * s - half, c1 = c0 + s, sum = 0;
            if (c0 < 0) c0 = 0;
            if (c1 > w) c1 = w;
            for (int c = c0; c < c1; c++) sum += acc[c];
            int cnt = (c1 - c0) * (r1 - r0);
            dst[x] = (uint8_t)((sum + cnt / 2) / cnt);
        }
    }
}

/* Bit x of bits[x / 64] is set where row[x] < thr[x], for x < lw; thr is zero (no sample is dark)
 * from lw up to the next multiple of 64, and both rows are readable that far. One compare and
 * movemask turn 16 samples into 16 bits (SSE2, which every x86-64 CPU has; NEON on ARM). */
static void dark_bits(const uint8_t *row, const uint8_t *thr, int lw, uint64_t *bits) {
    int words = (lw + 63) / 64;
    for (int w = 0; w < words; w++) {
        const uint8_t *r = row + 64 * w, *t = thr + 64 * w;
        uint64_t word = 0;
#ifdef NAPA_SSE2
        for (int k = 0; k < 4; k++) {
            __m128i a = _mm_loadu_si128((const __m128i *)(r + 16 * k));
            __m128i b = _mm_loadu_si128((const __m128i *)(t + 16 * k));
            __m128i ge = _mm_cmpeq_epi8(_mm_max_epu8(a, b), a); /* a >= b, unsigned */
            word |= (uint64_t)(uint16_t)~_mm_movemask_epi8(ge) << (16 * k);
        }
#elif defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
        static const uint8_t weights[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
        const uint8x16_t wv = vld1q_u8(weights);
        for (int k = 0; k < 4; k++) {
            uint8x16_t lt = vandq_u8(vcltq_u8(vld1q_u8(r + 16 * k), vld1q_u8(t + 16 * k)), wv);
            uint8x8_t sum = vpadd_u8(vget_low_u8(lt), vget_high_u8(lt));
            sum = vpadd_u8(sum, sum);
            sum = vpadd_u8(sum, sum); /* lane 0: bits of samples 0-7, lane 1: samples 8-15 */
            word |= (uint64_t)(vget_lane_u8(sum, 0) | (vget_lane_u8(sum, 1) << 8)) << (16 * k);
        }
#else
        for (int k = 0; k < 64; k++) word |= (uint64_t)(r[k] < t[k]) << k;
#endif
        bits[w] = word;
    }
}


/* Append row y's runs of set bits as napa_runs. Run starts and ends are the set bits of
 * bits ^ (bits << 1), found with count-trailing-zeros, so the cost is per run, not per sample. */
static int extract_runs(const uint64_t *bits, int lw, int y, napa_run *R, int n) {
    int open = 0, x0 = 0;
    for (int w = 0; w * 64 < lw; w++) {
        uint64_t edges = bits[w] ^ ((bits[w] << 1) | (uint64_t)open);
        while (edges) {
            int b = napa_ctz64(edges);
            edges &= edges - 1;
            if (!open) {
                x0 = 64 * w + b, open = 1;
            } else {
                R[n].x0 = (uint16_t)x0, R[n].x1 = (uint16_t)(64 * w + b - 1), R[n].y = (uint16_t)y;
                n++, open = 0;
            }
        }
    }
    if (open) {
        R[n].x0 = (uint16_t)x0, R[n].x1 = (uint16_t)(lw - 1), R[n].y = (uint16_t)y;
        n++;
    }
    return n;
}

/* dst[4i..4i+3] = src[i] for i < n: each tile's threshold for its four sample columns. */
static void expand4(uint8_t *dst, const uint8_t *src, int n) {
    int i = 0;
#ifdef NAPA_SSE2
    for (; i + 16 <= n; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(src + i));
        __m128i lo = _mm_unpacklo_epi8(v, v), hi = _mm_unpackhi_epi8(v, v);
        _mm_storeu_si128((__m128i *)(dst + 4 * i), _mm_unpacklo_epi16(lo, lo));
        _mm_storeu_si128((__m128i *)(dst + 4 * i + 16), _mm_unpackhi_epi16(lo, lo));
        _mm_storeu_si128((__m128i *)(dst + 4 * i + 32), _mm_unpacklo_epi16(hi, hi));
        _mm_storeu_si128((__m128i *)(dst + 4 * i + 48), _mm_unpackhi_epi16(hi, hi));
    }
#elif defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    for (; i + 16 <= n; i += 16) {
        uint8x16_t v = vld1q_u8(src + i);
        uint8x16x2_t a = vzipq_u8(v, v);
        uint8x16x2_t b = vzipq_u8(a.val[0], a.val[0]), c = vzipq_u8(a.val[1], a.val[1]);
        vst1q_u8(dst + 4 * i, b.val[0]);
        vst1q_u8(dst + 4 * i + 16, b.val[1]);
        vst1q_u8(dst + 4 * i + 32, c.val[0]);
        vst1q_u8(dst + 4 * i + 48, c.val[1]);
    }
#endif
    for (; i < n; i++) memset(dst + 4 * i, src[i], 4);
}

static inline int32_t find_root(int32_t *P, int32_t i) {
    while (P[i] != i) {
        P[i] = P[P[i]];
        i = P[i];
    }
    return i;
}

/* Threshold the level and label its dark samples as runs joined by union-find (8-connected:
 * runs in adjacent rows touch if they overlap within one sample). Returns the run count. */
static int threshold_level(napa_detector *det, int lw, int lh, int factor) {
    const uint8_t *L = det->level;
    napa_run *R = det->runs;
    int32_t *P = det->parent;
    int n = 0, prev_y = -2, prev_start = 0, prev_end = 0;
    int tw = (lw + TILE - 1) / TILE;
    int th = (lh + TILE - 1) / TILE;
    uint8_t *vmin = det->row_min, *vmax = det->row_max, *trow = det->row_thr;

    /* Tile min/max: a vertical pass over each band of TILE rows (contiguous, so it vectorises),
     * then a short horizontal pass over the band's TILE-wide groups. */
    for (int ty = 0; ty < th; ty++) {
        int y0 = ty * TILE;
        int y1 = y0 + TILE < lh ? y0 + TILE : lh;
        memcpy(vmin, L + (size_t)y0 * lw, (size_t)lw);
        memcpy(vmax, L + (size_t)y0 * lw, (size_t)lw);
        for (int y = y0 + 1; y < y1; y++) {
            const uint8_t *row = L + (size_t)y * lw;
            for (int x = 0; x < lw; x++) {
                uint8_t v = row[x];
                vmin[x] = v < vmin[x] ? v : vmin[x];
                vmax[x] = v > vmax[x] ? v : vmax[x];
            }
        }
        uint8_t *tmin = det->tile_min + (size_t)ty * tw;
        uint8_t *tmax = det->tile_max + (size_t)ty * tw;
        /* Rows are padded past lw by repeating the last sample, so every tile reads four. */
        for (int x = lw; x < 4 * tw; x++) vmin[x] = vmin[lw - 1], vmax[x] = vmax[lw - 1];
        for (int tx = 0; tx < tw; tx++) {
            const uint8_t *a = vmin + 4 * tx, *b = vmax + 4 * tx;
            uint8_t l01 = a[0] < a[1] ? a[0] : a[1], l23 = a[2] < a[3] ? a[2] : a[3];
            uint8_t h01 = b[0] > b[1] ? b[0] : b[1], h23 = b[2] > b[3] ? b[2] : b[3];
            tmin[tx] = l01 < l23 ? l01 : l23;
            tmax[tx] = h01 > h23 ? h01 : h23;
        }
    }
    /* 3x3 tile neighbourhood, separably and without branches so the loops vectorise: vertically
     * into a row padded by one tile at each end (repeating the edge tile), then horizontally. */
    uint8_t *nlo = det->nb_lo, *nhi = det->nb_hi, *tt = det->nb_t, *fl = det->nb_f;
    /* A tile here spans 4 * factor base-level samples; look up the floor of the region it lies in. */
    int rows = det->floor_rows, cols = det->floor_cols;
    if (det->floor_uniform) memset(fl, det->floors[0], (size_t)tw);
    for (int ty = 0; ty < th; ty++) {
        int ry = ty * TILE * factor / NAPA_REGION;
        const uint8_t *frow = det->floors + (size_t)(ry < rows ? ry : rows - 1) * cols;
        for (int tx = 0; tx < tw && !det->floor_uniform; tx++) {
            int rx = tx * TILE * factor / NAPA_REGION;
            fl[tx] = frow[rx < cols ? rx : cols - 1];
        }
        const uint8_t *lo_up = det->tile_min + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *lo_mid = det->tile_min + (size_t)ty * tw;
        const uint8_t *lo_dn = det->tile_min + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        const uint8_t *hi_up = det->tile_max + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *hi_mid = det->tile_max + (size_t)ty * tw;
        const uint8_t *hi_dn = det->tile_max + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        for (int tx = 0; tx < tw; tx++) {
            uint8_t l = lo_mid[tx], h = hi_mid[tx];
            l = lo_up[tx] < l ? lo_up[tx] : l;
            l = lo_dn[tx] < l ? lo_dn[tx] : l;
            h = hi_up[tx] > h ? hi_up[tx] : h;
            h = hi_dn[tx] > h ? hi_dn[tx] : h;
            nlo[tx + 1] = l;
            nhi[tx + 1] = h;
        }
        nlo[0] = nlo[1], nhi[0] = nhi[1], nlo[tw + 1] = nlo[tw], nhi[tw + 1] = nhi[tw];
        uint8_t any = 0;
        for (int tx = 0; tx < tw; tx++) {
            uint8_t l = nlo[tx + 1], h = nhi[tx + 1];
            l = nlo[tx] < l ? nlo[tx] : l;
            l = nlo[tx + 2] < l ? nlo[tx + 2] : l;
            h = nhi[tx] > h ? nhi[tx] : h;
            h = nhi[tx + 2] > h ? nhi[tx + 2] : h;
            /* Boyer-Moore-style skip: a flat neighbourhood cannot hold a tag edge, so its
             * threshold is 0 and no sample can fall below it. */
            uint8_t t = (uint8_t)(h - l) >= fl[tx] ? (uint8_t)((l + h + 1) >> 1) : 0;
            tt[tx] = t;
            any |= t;
        }
        int y0 = ty * TILE;
        int y1 = y0 + TILE < lh ? y0 + TILE : lh;
        if (!any) continue; /* no contrast anywhere in this band: no dark samples, no runs */
        expand4(trow, tt, tw);
        memset(trow + lw, 0, (size_t)(4 * tw + 64 - lw)); /* nothing past the row is dark */
        uint64_t *bits = det->row_bits;
        for (int y = y0; y < y1; y++) {
            dark_bits(L + (size_t)y * lw, trow, lw, bits);
            int start = n;
            n = extract_runs(bits, lw, y, R, n);
            for (int i = start; i < n; i++) P[i] = i;
            if (prev_y == y - 1) {
                int a = prev_start, b = start;
                while (a < prev_end && b < n) {
                    if (R[a].x1 + 1 < R[b].x0) {
                        a++;
                    } else if (R[b].x1 + 1 < R[a].x0) {
                        b++;
                    } else {
                        /* A run of this row that touches nothing yet is a singleton (runs of one row
                         * join only through the row above): hang it under a's parent, which still
                         * precedes it. Runs whose parents match are already joined. */
                        if (P[b] == b) {
                            P[b] = P[a];
                        } else if (P[b] != P[a]) {
                            int32_t ra = find_root(P, a), rb = find_root(P, b);
                            if (ra < rb) P[rb] = ra;
                            else if (rb < ra) P[ra] = rb;
                        }
                        if (R[a].x1 < R[b].x1) a++;
                        else b++;
                    }
                }
            }
            prev_y = y, prev_start = start, prev_end = n;
        }
    }
    return n;
}

/* ---------------------------------------------------------------- geometry */

/* Fit each side to the outer border edge searched from r_out outside to r_in inside the given quad. */
/* Edge levels for the midpoint-crossing estimator; black >= white disables it. */
typedef struct {
    double black, white;
} edge_levels;

static int fit_sides(const uint8_t *img, int w, int h, int stride, const vec2 in[4], vec2 out[4], double r_out,
                     double r_in, int min_edge, int outermost, int max_points, double *edge_width,
                     const edge_levels *levels) {
    double width_sum = 0;
    int width_n = 0;
    double reach = 0.5 * (r_out + r_in);
    enum { MAXP = 96, KMAX = 15 };
    /* The first (outermost) pass only has to land within a fraction of a cell for the early
     * decode, so it uses a shorter profile. */
    const int K = outermost ? 11 : KMAX;
    vec2 cen = {0, 0};
    for (int k = 0; k < 4; k++) {
        cen.x += in[k].x * 0.25;
        cen.y += in[k].y * 0.25;
    }
    double lx[4], ly[4], dx[4], dy[4];
    for (int k = 0; k < 4; k++) {
        vec2 a = in[k], b = in[(k + 1) & 3];
        double ex = b.x - a.x, ey = b.y - a.y;
        double len = sqrt(ex * ex + ey * ey);
        if (len < 6.0) return 0;
        ex /= len;
        ey /= len;
        double nx = -ey, ny = ex;
        if ((cen.x - a.x) * nx + (cen.y - a.y) * ny < 0) {
            nx = -nx;
            ny = -ny;
        }
        /* Dense passes (max_points > 24) take a profile every 1.5 px, averaging down sensor noise. */
        int np = (int)(len / (max_points > 24 ? 1.5 : 4.0));
        if (np < 6) np = 6;
        if (np > max_points) np = max_points;
        if (np > MAXP) np = MAXP;
        double px[MAXP], py[MAXP], wt[MAXP];
        int good = 0;
        double step = (r_out + r_in) / (K - 1);
        double dt = 0.76 / np, min_g = min_edge * 0.5 * step * 16777216.0; /* best / (2 step) >= min_edge / 4 */
        /* The profile step in 16.16 fixed point, rounded to nearest. */
        double ixd = nx * step * 65536.0, iyd = ny * step * 65536.0;
        int32_t ixp = (int32_t)(ixd < 0 ? ixd - 0.5 : ixd + 0.5), iyp = (int32_t)(iyd < 0 ? iyd - 0.5 : iyd + 0.5);
        for (int i = 0; i < np && good + (np - i) >= 3; i++) { /* stop once 3 edge points are out of reach */
            double t = 0.12 + dt * (i + 0.5);
            double bx = a.x + (b.x - a.x) * t, by = a.y + (b.y - a.y) * t;
            int64_t prof[KMAX]; /* grey levels in 40.24 fixed point */
            double sx0 = bx - nx * r_out, sy0 = by - ny * r_out;
            double sx1 = bx + nx * r_in, sy1 = by + ny * r_in;
            /* Whole profile inside the image: skip per-sample clamping, and step in 16.16 fixed point
             * with 12-bit interpolation weights (1/8192 px); the profile and its gradient stay in fixed
             * point, with no conversion per sample. The fixed-point
             * walk is checked end to end, since its rounded step can drift past a profile that ends
             * right at the border (the floating-point test comes first and also rejects NaN). */
            int32_t fxp = 0, fyp = 0;
            int inside = sx0 >= 0 && sy0 >= 0 && sx1 >= 0 && sy1 >= 0 && sx0 < w - 1 && sx1 < w - 1 &&
                         sy0 < h - 1 && sy1 < h - 1;
            if (inside) {
                fxp = (int32_t)(sx0 * 65536.0), fyp = (int32_t)(sy0 * 65536.0);
                int64_t endx = fxp + (int64_t)(K - 1) * ixp, endy = fyp + (int64_t)(K - 1) * iyp;
                inside = endx >= 0 && endy >= 0 && endx < ((int64_t)(w - 1) << 16) && endy < ((int64_t)(h - 1) << 16);
            }
            if (inside) {
                for (int j = 0; j < K; j++, fxp += ixp, fyp += iyp) {
                    int x0 = fxp >> 16, y0 = fyp >> 16;
                    int32_t fx = (fxp >> 4) & 4095, fy = (fyp >> 4) & 4095;
                    const uint8_t *q = img + (size_t)y0 * stride + x0;
                    int32_t top = (q[0] << 12) + (q[1] - q[0]) * fx;
                    int32_t bot = (q[stride] << 12) + (q[stride + 1] - q[stride]) * fx;
                    prof[j] = ((int64_t)top << 12) + (int64_t)(bot - top) * fy;
                }
            } else {
                for (int j = 0; j < K; j++) {
                    double o = -r_out + j * step;
                    prof[j] = (int64_t)(sample_bilinear(img, w, h, stride, bx + nx * o, by + ny * o) * 16777216.0 + 0.5);
                }
            }
            /* outside (white) minus inside (black): the outer border edge is a positive peak. */
            int64_t best = 0;
            int bj = -1;
            int64_t g[KMAX];
            for (int j = 1; j < K - 1; j++) {
                g[j] = prof[j - 1] - prof[j + 1];
                if (g[j] > best) {
                    best = g[j];
                    bj = j;
                }
            }
            /* The outer border is the first step met going inward that lands on the darkest level:
             * payload cells further in produce equally strong steps (so prefer the outermost), and a
             * background brighter than the tag's white margin produces an outer step that lands only
             * on the margin's grey (so require a dark landing). */
            if (outermost && bj >= 0) {
                for (int j = 1; j < K - 1; j++) {
                    if (5 * g[j] >= 3 * best && (j == K - 2 || g[j] >= g[j + 1])) {
                        bj = j;
                        best = g[j];
                        break;
                    }
                }
            }
            /* Normalise by the derivative span so the strength test is independent of the step. */
            if (bj < 0 || (double)best < min_g) continue;
            /* Centre of mass of the positive gradient lobe around the peak. For an area-sampled edge
             * this is exactly the edge at any blur width; trimming or squaring the weights biases
             * sharp edges. The lobe ends where the gradient changes sign, and the final window is
             * capped below half a cell, so the next payload edge never contributes. */
            int jl = bj, jr = bj;
            while (jl > 1 && 20 * g[jl - 1] > best) jl--;
            while (jr < K - 2 && 20 * g[jr + 1] > best) jr++;
            int64_t sg = 0, sj = 0;
            for (int j = jl; j <= jr; j++) {
                sg += g[j];
                sj += g[j] * j;
            }
            double cj = sg > 0 ? (double)sj / (double)sg : bj;
            /* When the edge ramp is wide against a cell (motion or defocus blur), the next edge's ramp
             * overlaps this one and biases the centre of mass. The crossing of the midpoint between
             * the tag's black and white stays exact for any symmetric blur narrower than two cells.
             * Black is scaled by the local white, so it follows lighting that varies across the tag. */
            if (levels != NULL && levels->black < levels->white &&
                (jr - jl + 1) * step > NAPA_CROSSING_CELLS * len / 8.0) {
                /* Levels from the centres of the cells either side (white margin, black border), half
                 * a cell from the edge: outside the blurred ramp the window would sit in. */
                double half = len / 16.0;
                for (int iter = 0; iter < NAPA_CROSS_ITERS; iter++) {
                    double pos0 = -r_out + cj * step;
                    double wl = sample_bilinear(img, w, h, stride, bx + nx * (pos0 - half), by + ny * (pos0 - half));
                    double bl = sample_bilinear(img, w, h, stride, bx + nx * (pos0 + half), by + ny * (pos0 + half));
                    double mid = 0.5 * (wl + bl) * 16777216.0;
                    double best_c = -1, best_d = 1e9;
                    for (int j = jl - 1; j <= jr && j + 1 < K; j++) {
                        if (j < 0 || !((double)prof[j] >= mid && (double)prof[j + 1] < mid)) continue;
                        double c = j + ((double)prof[j] - mid) / (double)(prof[j] - prof[j + 1]);
                        if (fabs(c - cj) < best_d) best_d = fabs(c - cj), best_c = c;
                    }
                    if (best_c < 0) break;
                    int settled = fabs(best_c - cj) < 1e-3; /* the next pass would re-read the same levels */
                    cj = best_c;
                    if (settled) break;
                }
            }
            width_sum += (jr - jl + 1) * step;
            width_n++;
            double pos = -r_out + cj * step;
            px[good] = bx + nx * pos;
            py[good] = by + ny * pos;
            wt[good] = (double)best * (1.0 / (1 << 24));
            good++;
        }
        if (good < 3) return 0;
        /* Weighted total-least-squares line, then one pass dropping outliers. */
        for (int pass = 0; pass < 2; pass++) {
            double sw = 0, mx = 0, my = 0;
            for (int i = 0; i < good; i++) {
                sw += wt[i];
                mx += wt[i] * px[i];
                my += wt[i] * py[i];
            }
            if (sw <= 0) return 0;
            mx /= sw;
            my /= sw;
            double sxx = 0, syy = 0, sxy = 0;
            for (int i = 0; i < good; i++) {
                double qx = px[i] - mx, qy = py[i] - my;
                sxx += wt[i] * qx * qx;
                syy += wt[i] * qy * qy;
                sxy += wt[i] * qx * qy;
            }
            /* Principal axis (cos t, sin t) with tan 2t = 2 sxy / (sxx - syy), from the half-angle
             * identities instead of atan2, cos and sin: cos t sin t = sxy / r. */
            double spread = sxx - syy, r = sqrt(spread * spread + 4 * sxy * sxy);
            double ct = 1, st = 0;
            if (r > 0) {
                if (spread >= 0) {
                    ct = sqrt(0.5 * (1 + spread / r));
                    st = sxy / (r * ct);
                } else {
                    st = copysign(sqrt(0.5 * (1 - spread / r)), sxy);
                    ct = sxy / (r * st);
                }
            }
            lx[k] = mx;
            ly[k] = my;
            dx[k] = ct;
            dy[k] = st;
            if (pass == 0) {
                int kept = 0;
                for (int i = 0; i < good; i++) {
                    double dist = fabs((px[i] - mx) * (-dy[k]) + (py[i] - my) * dx[k]);
                    if (dist <= 1.0 + 0.1 * reach) {
                        px[kept] = px[i];
                        py[kept] = py[i];
                        wt[kept] = wt[i];
                        kept++;
                    }
                }
                /* Drop outliers only when most samples agree; otherwise keep the full fit. */
                if (kept >= 3 && 2 * kept >= good) good = kept;
            }
        }
    }
    *edge_width = width_n ? width_sum / width_n : 0;
    for (int k = 0; k < 4; k++) {
        int p = (k + 3) & 3; /* corner k joins side k-1 and side k */
        double cross = dx[p] * dy[k] - dy[p] * dx[k];
        if (fabs(cross) < 0.2) return 0;
        double ddx = lx[k] - lx[p], ddy = ly[k] - ly[p];
        double t = (ddx * dy[k] - ddy * dx[k]) / cross;
        out[k].x = lx[p] + t * dx[p];
        out[k].y = ly[p] + t * dy[p];
        double mv = fabs(out[k].x - in[k].x) + fabs(out[k].y - in[k].y);
        if (mv > 4.0 * reach + 2.0) return 0;
    }
    return 1;
}

static double signed_area(const vec2 q[4]) {
    double s = 0;
    for (int k = 0; k < 4; k++) {
        s += q[k].x * q[(k + 1) & 3].y - q[k].y * q[(k + 1) & 3].x;
    }
    return 0.5 * s;
}

/* Projective map of the unit square (0..1) onto quad q (Heckbert). */
static void square_to_quad(const vec2 q[4], double H[9]) {
    double x0 = q[0].x, y0 = q[0].y, x1 = q[1].x, y1 = q[1].y;
    double x2 = q[2].x, y2 = q[2].y, x3 = q[3].x, y3 = q[3].y;
    double sx = x0 - x1 + x2 - x3, sy = y0 - y1 + y2 - y3;
    double dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2;
    double g = 0, hh = 0;
    double den = dx1 * dy2 - dx2 * dy1;
    if (fabs(den) > 1e-12 && (fabs(sx) > 1e-12 || fabs(sy) > 1e-12)) {
        g = (sx * dy2 - dx2 * sy) / den;
        hh = (dx1 * sy - sx * dy1) / den;
    }
    H[0] = x1 - x0 + g * x1;
    H[1] = x3 - x0 + hh * x3;
    H[2] = x0;
    H[3] = y1 - y0 + g * y1;
    H[4] = y3 - y0 + hh * y3;
    H[5] = y0;
    H[6] = g;
    H[7] = hh;
    H[8] = 1;
}

static inline vec2 project(const double H[9], double u, double v) {
    double iz = 1.0 / (H[6] * u + H[7] * v + H[8]);
    vec2 p = {(H[0] * u + H[1] * v + H[2]) * iz, (H[3] * u + H[4] * v + H[5]) * iz};
    return p;
}

/* Median of v[0..n) for even n (the mean of the two middle values), by quickselect. Reorders v. */
static double median_even(double *v, int n) {
    int k = n / 2 - 1, lo = 0, hi = n - 1;
    while (lo < hi) {
        double pivot = v[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (v[i] < pivot) i++;
            while (v[j] > pivot) j--;
            if (i <= j) {
                double t = v[i];
                v[i++] = v[j];
                v[j--] = t;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    double upper = v[k + 1];
    for (int m = k + 2; m < n; m++) upper = v[m] < upper ? v[m] : upper;
    return 0.5 * (v[k] + upper);
}

/* Whether the set bits of a 36-bit payload mask (row-major 6x6, most significant bit first) form
 * one 8-connected group of cells. */
static int errors_connected(uint64_t diff) {
    int cells[36], n = 0;
    for (int i = 0; i < 36; i++)
        if (diff >> (35 - i) & 1) cells[n++] = i;
    if (n <= 1) return 1;
    int reached = 1, frontier[36], nf = 1, seen = 1; /* bitmask over cells[] indices */
    frontier[0] = 0;
    while (nf) {
        int c = cells[frontier[--nf]];
        for (int j = 0; j < n; j++) {
            if (seen >> j & 1) continue;
            int dr = cells[j] / 6 - c / 6, dc = cells[j] % 6 - c % 6;
            if (dr >= -1 && dr <= 1 && dc >= -1 && dc <= 1) {
                seen |= 1 << j;
                frontier[nf++] = j;
                reached++;
            }
        }
    }
    return reached == n;
}

/* Centre of mass of the strongest falling (white to black going along the profile) or rising
 * gradient lobe in prof[from..to], the same estimator fit_sides uses. Returns -1 if there is none. */
static double lobe_centroid(const double *prof, int from, int to, int falling) {
    double g[64], best = 0;
    int bj = -1;
    for (int j = from; j <= to; j++) {
        g[j] = falling ? prof[j - 1] - prof[j + 1] : prof[j + 1] - prof[j - 1];
        if (g[j] > best) best = g[j], bj = j;
    }
    if (bj < 0) return -1;
    int jl = bj, jr = bj;
    while (jl > from && g[jl - 1] > 0.05 * best) jl--;
    while (jr < to && g[jr + 1] > 0.05 * best) jr++;
    double sg = 0, sj = 0;
    for (int j = jl; j <= jr; j++) sg += g[j], sj += g[j] * j;
    return sj / sg;
}

/* Clipping (or any symmetric edge bias) shrinks every black region by delta: the outline by delta
 * per side and the one-cell border by 2 * delta. Measuring the border's width where a white payload
 * cell lies inside it gives delta = (cell - width) / 1.75 against the measured outline. Returns the
 * number of border measurements averaged into *delta. */
static int border_bias(const uint8_t *img, int w, int h, int stride, const vec2 q[4], double black, double white,
                       double *delta) {
    enum { N = 33 };
    double H[9], sum = 0, side_sum[4] = {0, 0, 0, 0};
    int n = 0, side_n[4] = {0, 0, 0, 0};
    square_to_quad(q, H);
    double mid = 0.5 * (black + white);
    for (int side = 0; side < 4; side++) {
        for (int c = 1; c < 7; c++) {
            /* (a, t): position along the side and depth inward, in cells; side 0 is the top (v = 0). */
            double a = c + 0.5;
#define UNIT(t, uu, vv)                                                                                      \
    do {                                                                                                     \
        double t_ = (t) / 8.0, a_ = a / 8.0;                                                                 \
        uu = side == 0 ? a_ : side == 1 ? 1 - t_ : side == 2 ? a_ : t_;                                      \
        vv = side == 0 ? t_ : side == 1 ? a_ : side == 2 ? 1 - t_ : a_;                                      \
    } while (0)
            double u, v;
            UNIT(1.5, u, v);
            vec2 pc = project(H, u, v);
            if (sample_bilinear(img, w, h, stride, pc.x, pc.y) < mid) continue; /* dark payload cell */
            double prof[N];
            vec2 pts[N];
            for (int j = 0; j < N; j++) {
                UNIT(-0.5 + 2.0 * j / (N - 1), u, v);
                pts[j] = project(H, u, v);
                prof[j] = sample_bilinear(img, w, h, stride, pts[j].x, pts[j].y);
            }
            /* Outer edge within half a cell of t = 0, inner edge within half a cell of t = 1. */
            double jo = lobe_centroid(prof, 1, N / 2 - 1, 1), ji = lobe_centroid(prof, N / 2 + 1, N - 2, 0);
            if (jo < 0 || ji < 0) continue;
            vec2 a0 = pts[(int)jo], a1 = pts[(int)jo + 1], b0 = pts[(int)ji], b1 = pts[(int)ji + 1];
            double fo = jo - (int)jo, fi = ji - (int)ji;
            double width_px = hypot(b0.x + (b1.x - b0.x) * fi - a0.x - (a1.x - a0.x) * fo,
                                    b0.y + (b1.y - b0.y) * fi - a0.y - (a1.y - a0.y) * fo);
            vec2 e0 = pts[8], e1 = pts[24]; /* t = 0 and t = 1 */
            double cell_px = hypot(e1.x - e0.x, e1.y - e0.y);
            double d = (cell_px - width_px) / 1.75;
            sum += d, side_sum[side] += d;
            n++, side_n[side]++;
#undef UNIT
        }
    }
    if (!n) return 0;
    *delta = sum / n;
    /* Clipping moves every side alike. Sides that disagree mean the border simply isn't one cell
     * wide (a rasterised tag whose cells were rounded to whole pixels): leave such a tag alone. */
    for (int side = 0; side < 4; side++) {
        if (side_n[side] && fabs(side_sum[side] / side_n[side] - *delta) > 0.05 + 0.25 * fabs(*delta)) return 0;
    }
    return n;
}

/* Move every side of a convex quad outward by d pixels. */
static void grow_quad(const vec2 q[4], double d, vec2 out[4]) {
    double cx = 0, cy = 0, px[4], py[4], dx[4], dy[4];
    for (int k = 0; k < 4; k++) cx += 0.25 * q[k].x, cy += 0.25 * q[k].y;
    for (int k = 0; k < 4; k++) {
        vec2 a = q[k], b = q[(k + 1) & 3];
        double ex = b.x - a.x, ey = b.y - a.y, len = hypot(ex, ey);
        ex /= len, ey /= len;
        double nx = ey, ny = -ex;
        if ((a.x - cx) * nx + (a.y - cy) * ny < 0) nx = -nx, ny = -ny;
        px[k] = a.x + nx * d, py[k] = a.y + ny * d, dx[k] = ex, dy[k] = ey;
    }
    for (int k = 0; k < 4; k++) {
        int p = (k + 3) & 3;
        double cross = dx[p] * dy[k] - dy[p] * dx[k];
        double t = ((px[k] - px[p]) * dy[k] - (py[k] - py[p]) * dx[k]) / cross;
        out[k].x = px[p] + t * dx[p];
        out[k].y = py[p] + t * dy[p];
    }
}

/* Decode the tag inside quad q (reordered in place to a consistent winding). `slack` loosens the
 * border and Hamming limits, for a cheap early decode on unrefined corners. Optionally reports each
 * side's border level (side_level) and the tag's black and white levels. */
static int decode_quad(napa_detector *det, const uint8_t *img, int w, int h, int stride, vec2 q[4],
                       const napa_params *params, int slack, napa_detection *out, double side_level[4],
                       edge_levels *levels) {
    if (signed_area(q) < 0) {
        vec2 t = q[1];
        q[1] = q[3];
        q[3] = t;
    }
    double H[9];
    square_to_quad(q, H);
    double grid[64];
    double border[28];
    double ring[36];
    int nb = 0, nr = 0;
    /* Border cells and the quiet-zone ring first: most candidates are rejected on their contrast
     * or border before the 36 payload cells are worth sampling. */
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c += (r == 0 || r == 7) ? 1 : 7) {
            vec2 p = project(H, (c + 0.5) / 8.0, (r + 0.5) / 8.0);
            double v = sample_bilinear(img, w, h, stride, p.x, p.y);
            grid[r * 8 + c] = v;
            border[nb++] = v;
        }
    }
    for (int i = 0; i < 10; i++) {
        double u = (i - 0.5) / 8.0;
        vec2 p;
        p = project(H, u, -0.5 / 8.0);
        ring[nr++] = sample_bilinear(img, w, h, stride, p.x, p.y);
        p = project(H, u, 8.5 / 8.0);
        ring[nr++] = sample_bilinear(img, w, h, stride, p.x, p.y);
    }
    for (int i = 1; i < 9; i++) {
        double v = (i - 0.5) / 8.0;
        vec2 p;
        p = project(H, -0.5 / 8.0, v);
        ring[nr++] = sample_bilinear(img, w, h, stride, p.x, p.y);
        p = project(H, 8.5 / 8.0, v);
        ring[nr++] = sample_bilinear(img, w, h, stride, p.x, p.y);
    }
    double bs[28], rs[36];
    memcpy(bs, border, sizeof bs);
    memcpy(rs, ring, sizeof rs);
    double black = median_even(bs, 28);
    double white = median_even(rs, 36);
    if (white - black < params->min_contrast) return 0;
    if (levels != NULL) levels->black = black, levels->white = white;
    double thr = 0.5 * (black + white);
    int dark = 0;
    for (int i = 0; i < 28; i++) dark += border[i] < thr;
    if (dark < params->min_border_cells - slack) return 0;
    for (int r = 1; r < 7; r++) {
        for (int c = 1; c < 7; c++) {
            vec2 p = project(H, (c + 0.5) / 8.0, (r + 0.5) / 8.0);
            grid[r * 8 + c] = sample_bilinear(img, w, h, stride, p.x, p.y);
        }
    }
    double raw[64];
    memcpy(raw, grid, sizeof raw);
    /* Sharpen the cell samples, as upstream AprilTag does, to undo blur between neighbouring cells. */
    if (params->sharpening_x100 > 0) {
        double k = params->sharpening_x100 / 100.0;
        double sharp[64];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                double v = grid[r * 8 + c], lap = 4 * v;
                lap -= r > 0 ? grid[(r - 1) * 8 + c] : v;
                lap -= r < 7 ? grid[(r + 1) * 8 + c] : v;
                lap -= c > 0 ? grid[r * 8 + c - 1] : v;
                lap -= c < 7 ? grid[r * 8 + c + 1] : v;
                sharp[r * 8 + c] = v + k * lap;
            }
        }
        memcpy(grid, sharp, sizeof sharp);
    }
    if (side_level != NULL) {
        /* Each side's non-corner border cells (top, right, bottom, left, unsharpened), as a fraction of
         * the way from the tag's black to its white: about 0 for a true border side. */
        for (int k = 0; k < 4; k++) side_level[k] = 0;
        for (int k = 1; k < 7; k++) {
            side_level[0] += raw[k];
            side_level[1] += raw[k * 8 + 7];
            side_level[2] += raw[56 + k];
            side_level[3] += raw[k * 8];
        }
        for (int k = 0; k < 4; k++) side_level[k] = (side_level[k] / 6.0 - black) / (white - black);
    }

    uint64_t code = 0;
    for (int r = 1; r < 7; r++) {
        for (int c = 1; c < 7; c++) {
            code = (code << 1) | (grid[r * 8 + c] > thr ? 1u : 0u);
        }
    }
    int best = 64;
    int bi = nearest_code(det, code, params->max_hamming + slack, &best);
    if (bi < 0) return 0;
    /* Three corrected bits are accepted only when the wrong cells touch (8-connected): real damage,
     * a finger, cable, or glare spot, is spatially coherent, while a random pattern that happens to
     * fall within three bits of a code has its errors scattered: only 420 of the 7,140 three-cell
     * placements are connected, so this rejects ~94% of chance three-bit matches. */
    if (slack == 0 && best == 3 && !errors_connected(code ^ det->codes[bi])) return 0;
    int rot = det->rotations[bi];
    /* Observed grid == rot90(canonical, rot); the canonical top-left sits at corner index (4 - rot) % 4. */
    for (int k = 0; k < 4; k++) {
        vec2 c = q[(k - rot + 4) & 3];
        out->corners[2 * k] = c.x;
        out->corners[2 * k + 1] = c.y;
    }
    out->id = det->ids[bi];
    out->hamming = best;
    /* As upstream AprilTag: the smaller of the white and black payload bits' mean distance from the
     * threshold, in grey levels. Low values mean bits were nearly ambiguous. */
    double white_sum = 0, black_sum = 0;
    int white_n = 0, black_n = 0;
    for (int r = 1; r < 7; r++) {
        for (int c = 1; c < 7; c++) {
            double v = grid[r * 8 + c] - thr;
            if (v > 0) white_sum += v, white_n++;
            else black_sum -= v, black_n++;
        }
    }
    double wm = white_n ? white_sum / white_n : 0, bm = black_n ? black_sum / black_n : 0;
    out->decision_margin = (float)(wm < bm ? wm : bm);
    return 1;
}

/* ---------------------------------------------------------------- components */

/* Directions (dx, dy) whose extreme points seed the rough quad, in clockwise angular order (y points
 * down), about 22.5 degrees apart: a corner lies within about 11 degrees of one of them, so its extreme
 * can't slide far along a shallow side. */
static const int comp_dirs[16][2] = {{1, 0},  {2, 1},   {1, 1},   {1, 2},  {0, 1},  {-1, 2}, {-1, 1}, {-2, 1},
                                     {-1, 0}, {-2, -1}, {-1, -1}, {-1, -2}, {0, -1}, {1, -2}, {1, -1}, {2, -1}};

/* Fold the blob's current row into its extremes. Every direction's extreme over a row lies at the
 * row's leftmost or rightmost dark sample, so this runs once per row, not once per run. The stores
 * are unconditional selects: a growing blob moves its extremes on most rows, unpredictably. */
static void comp_flush_row(component *c) {
    int x0 = c->row_x0, x1 = c->row_x1, y = c->row_y;
    for (int k = 0; k < 16; k++) {
        int x = comp_dirs[k][0] > 0 ? x1 : x0;
        int v = comp_dirs[k][0] * x + comp_dirs[k][1] * y;
        int up = v > c->extv[k];
        c->extv[k] = up ? v : c->extv[k];
        c->ext[k][0] = up ? x : c->ext[k][0];
        c->ext[k][1] = up ? y : c->ext[k][1];
    }
}

/* Runs arrive in raster order. extv starts at INT32_MIN and row_y at -1. */
static void comp_add_run(component *c, const napa_run *r) {
    if (r->y != c->row_y) {
        if (c->row_y >= 0) comp_flush_row(c);
        c->row_y = r->y, c->row_x0 = r->x0;
    }
    c->row_x1 = r->x1;
    c->count += r->x1 - r->x0 + 1;
}

static double quad_area_pts(int p[4][2]) {
    double s = 0;
    for (int k = 0; k < 4; k++) {
        s += (double)p[k][0] * p[(k + 1) & 3][1] - (double)p[k][1] * p[(k + 1) & 3][0];
    }
    return fabs(0.5 * s);
}

static int covered(const napa_detection *out, int n, double x, double y, double radius) {
    for (int i = 0; i < n; i++) {
        double ox = 0, oy = 0;
        for (int k = 0; k < 4; k++) {
            ox += out[i].corners[2 * k] * 0.25;
            oy += out[i].corners[2 * k + 1] * 0.25;
        }
        if (fabs(ox - x) + fabs(oy - y) < radius) return 1;
    }
    return 0;
}

static int already_found(const napa_detection *out, int n, const napa_detection *d) {
    double cx = 0, cy = 0;
    for (int k = 0; k < 4; k++) {
        cx += d->corners[2 * k] * 0.25;
        cy += d->corners[2 * k + 1] * 0.25;
    }
    double size = fabs(d->corners[0] - d->corners[4]) + fabs(d->corners[1] - d->corners[5]);
    for (int i = 0; i < n; i++) {
        double ox = 0, oy = 0;
        for (int k = 0; k < 4; k++) {
            ox += out[i].corners[2 * k] * 0.25;
            oy += out[i].corners[2 * k + 1] * 0.25;
        }
        if (fabs(ox - cx) + fabs(oy - cy) < 0.25 * size + 2.0) return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------- detect */

/* Tile contrast floor from a histogram of neighbouring-sample differences (clamped to 63): `sigmas`
 * times the noise. The median absolute difference of neighbouring samples is about 0.95 sigma for
 * Gaussian noise, and a flat noisy tile neighbourhood spans about 5.5 sigma, so a floor of 6 sigma
 * keeps noise from forming dark networks. Capped well below real tag contrast. */
static int floor_from(const uint32_t hist[64], uint32_t n, int min_contrast, double sigmas) {
    uint32_t seen = 0;
    int median = 0;
    while (median < 63 && (seen += hist[median]) < (n + 1) / 2) median++;
    int floor = (int)(sigmas * median / 0.954 + 0.5);
    if (floor > NAPA_FLOOR_CAP) floor = NAPA_FLOOR_CAP;
    return floor > min_contrast ? floor : min_contrast;
}

/* out[i] = min(|row[2i + 1] - row[2i]|, 63) for i < n: neighbouring-sample differences, 16 at a time. */
static void pair_diffs(const uint8_t *row, int n, uint8_t *out) {
    int i = 0;
#ifdef NAPA_SSE2
    const __m128i even = _mm_set1_epi16(0x00ff), cap = _mm_set1_epi8(63);
    for (; i + 16 <= n; i += 16) {
        __m128i a = _mm_loadu_si128((const __m128i *)(row + 2 * i));
        __m128i b = _mm_loadu_si128((const __m128i *)(row + 2 * i + 16));
        __m128i ae = _mm_and_si128(a, even), ao = _mm_srli_epi16(a, 8);
        __m128i be = _mm_and_si128(b, even), bo = _mm_srli_epi16(b, 8);
        __m128i da = _mm_sub_epi16(_mm_max_epi16(ae, ao), _mm_min_epi16(ae, ao));
        __m128i db = _mm_sub_epi16(_mm_max_epi16(be, bo), _mm_min_epi16(be, bo));
        _mm_storeu_si128((__m128i *)(out + i), _mm_min_epu8(_mm_packus_epi16(da, db), cap));
    }
#elif defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    for (; i + 16 <= n; i += 16) {
        uint8x16x2_t v = vld2q_u8(row + 2 * i);
        vst1q_u8(out + i, vminq_u8(vabdq_u8(v.val[1], v.val[0]), vdupq_n_u8(63)));
    }
#endif
    for (; i < n; i++) {
        int d = row[2 * i + 1] - row[2 * i];
        d = d < 0 ? -d : d;
        out[i] = (uint8_t)(d < 63 ? d : 63);
    }
}

/* Each region's floor into det->floors (from pairs of samples on every 8th row), at most the frame's,
 * which is returned; a region too small to measure takes the frame's. Returns -1 if out of memory. */
static int contrast_floors(napa_detector *det, const uint8_t *L, int lw, int lh, int min_contrast, double sigmas) {
    int cols = (lw + NAPA_REGION - 1) / NAPA_REGION, rows = (lh + NAPA_REGION - 1) / NAPA_REGION;
    if ((size_t)cols * rows > det->floor_alloc) {
        uint8_t *p = (uint8_t *)realloc(det->floors, (size_t)cols * rows);
        if (p == NULL) return -1;
        det->floors = p;
        det->floor_alloc = (size_t)cols * rows;
    }
    det->floor_cols = cols, det->floor_rows = rows;
    uint32_t all[64] = {0}, n_all = 0;
    for (int ry = 0; ry < rows; ry++) {
        for (int rx = 0; rx < cols; rx++) {
            /* Four interleaved histograms: a flat region repeats the same few differences, and
             * consecutive increments of one counter would wait on each other. */
            uint32_t h4[4][64], hist[64], n = 0;
            memset(h4, 0, sizeof h4);
            int x0 = rx * NAPA_REGION, x1 = x0 + NAPA_REGION < lw ? x0 + NAPA_REGION : lw;
            int y1 = (ry + 1) * NAPA_REGION < lh ? (ry + 1) * NAPA_REGION : lh;
            int pairs = (x1 - x0) / 2;
            for (int y = ry * NAPA_REGION; y < y1; y += 8) {
                uint8_t d[NAPA_REGION / 2];
                pair_diffs(L + (size_t)y * lw + x0, pairs, d);
                for (int k = 0; k < pairs; k++) h4[k & 3][d[k]]++;
            }
            for (int k = 0; k < 64; k++) {
                hist[k] = h4[0][k] + h4[1][k] + h4[2][k] + h4[3][k];
                n += hist[k], all[k] += hist[k];
            }
            n_all += n;
            det->floors[ry * cols + rx] = (uint8_t)(n >= 16 ? floor_from(hist, n, min_contrast, sigmas) : 0);
        }
    }
    int frame = n_all ? floor_from(all, n_all, min_contrast, sigmas) : min_contrast;
    det->floor_uniform = 1;
    for (int i = 0; i < cols * rows; i++) {
        /* Never above the frame's: a region only lowers its floor, where it is cleaner than the frame
         * (a region's estimate, from fewer samples, also varies more). */
        if (det->floors[i] == 0 || det->floors[i] > frame) det->floors[i] = (uint8_t)frame;
        det->floor_uniform &= det->floors[i] == det->floors[0];
    }
    return frame;
}

int napa_detect(napa_detector *det, const uint8_t *img, int width, int height, int stride, const napa_params *params,
                napa_detection *out, int max_out) {
    if (det == NULL || img == NULL || params == NULL) return -1;
    if (stride < width || max_out < 0) return -1;
    if (width > 30000 || height > 30000) return -1;
    if (params->min_tag_px < 8 || params->min_tag_px > 30000 || params->max_hamming < 0 || params->max_hamming > 3 ||
        params->min_contrast < 1 || params->min_contrast > 255 || params->min_border_cells < 0 ||
        params->min_border_cells > 28 || params->sharpening_x100 < 0 || params->sharpening_x100 > 100)
        return -1;
    if (width < 16 || height < 16) return 0;
    /* A tag cell is min_tag_px / 8 wide, but blur turns its edges grey, so only its core is
     * reliably dark. Sampling every 3/4 cell guarantees a sample inside that core. */
    int base = params->min_tag_px * 3 / 32;
    if (base < 1) base = 1;
    int found = 0;
    int contrast = params->min_contrast;
    /* Size band in level samples. Adjacent levels overlap (7-24 here is 14-48 one level finer),
     * because a blob's measured extent varies by a sample or two with phase, blur and rotation. */
    const int lo = 7, hi = NAPA_HI;

    int prev_lw = 0;
    for (int s = base;; s *= 2) {
        int lw = (width + s - 1) / s;
        int lh = (height + s - 1) / s;
        int last = (lw < NAPA_LAST) || (lh < NAPA_LAST);
        size_t pixels = (size_t)lw * lh;
        size_t tiles = (size_t)((lw + TILE - 1) / TILE) * ((lh + TILE - 1) / TILE);
        if (ensure_buffers(det, pixels, tiles, (size_t)lw, pixels / 2 + (size_t)lh + 1) != 0) return -2;

        PROF_START;
        /* The rake: one point sample per stride, row by row. A coarser level is every second
         * sample of the level before (s doubles), so it comes from the small, cached level
         * buffer, in place (each output sample sits at or before its source). */
        if (s == base) {
            for (int y = 0; y < lh; y++)
                gather_row(det->level + (size_t)y * lw, img + (size_t)y * s * stride, lw, s, det->ssse3);
        } else {
            for (int y = 0; y < lh; y++) gather2(det->level + (size_t)y * lw, det->level + (size_t)(2 * y) * prev_lw, lw);
        }
        prev_lw = lw;
        PROF_MARK(PROF_RAKE);
        if (s == base) {
            contrast = contrast_floors(det, det->level, lw, lh, params->min_contrast, NAPA_FLOOR_SIGMAS);
            /* A noisy frame: rake again, averaging each sample's block, and re-measure the noise. */
            if (contrast > NAPA_NOISY_FLOOR && s > 1 && s <= 16) {
                if ((size_t)width > det->box_capacity) {
                    uint16_t *p = (uint16_t *)realloc(det->box_rows, sizeof(uint16_t) * 2 * (size_t)width);
                    if (p == NULL) return -2;
                    det->box_rows = p;
                    det->box_capacity = (size_t)width;
                }
                box_rake(det->level, lw, lh, img, width, height, stride, s, det->box_rows, det->box_rows + width);
                contrast = contrast_floors(det, det->level, lw, lh, params->min_contrast, NAPA_BOX_SIGMAS);
            }
            if (contrast < 0) return -2;
        }
        int nruns = threshold_level(det, lw, lh, s / base);
        PROF_MARK(PROF_THRESHOLD);

        /* Blobs: each run's root, then every blob's count and bounding box. */
        napa_run *R = det->runs;
        int32_t *P = det->parent;
        blob_box *B = det->boxes;
        int32_t *roots = det->roots, nroots = 0;
        for (int i = 0; i < nruns; i++) {
            /* A parent precedes its child and was resolved first, so P[P[i]] is already the root. */
            int32_t r = P[P[i]];
            P[i] = r;
            if (r == i) {
                det->slot[i] = -1;
                roots[nroots++] = i;
                B[i].count = R[i].x1 - R[i].x0 + 1;
                B[i].minx = R[i].x0, B[i].maxx = R[i].x1, B[i].miny = B[i].maxy = R[i].y;
            } else {
                B[r].count += R[i].x1 - R[i].x0 + 1;
                B[r].minx = R[i].x0 < B[r].minx ? R[i].x0 : B[r].minx;
                B[r].maxx = R[i].x1 > B[r].maxx ? R[i].x1 : B[r].maxx;
                B[r].maxy = R[i].y; /* runs arrive row by row */
            }
        }
        /* Keep tag-sized, tag-shaped blobs as candidates, in raster order of their first sample. A tag
         * tilted by t looks about 1 / cos(t) times longer than it is wide, more with perspective: a 4:1
         * box admits tilts to about 70-75 degrees, beyond which the cells are too thin to read anyway. */
        int min_size = lo;
        int max_size = last ? 1 << 30 : hi;
        int ncand = 0;
        for (int ri = 0; ri < nroots; ri++) {
            int i = roots[ri];
            int t_minx = B[i].minx, t_maxx = B[i].maxx, t_miny = B[i].miny, t_maxy = B[i].maxy;
            int bw = t_maxx - t_minx + 1, bh = t_maxy - t_miny + 1;
            int big = bw > bh ? bw : bh, small = bw < bh ? bw : bh;
            if (small < min_size || big > max_size || big > 4 * small) {
                if (big >= 4) TRACE(small < min_size ? 1 : big > max_size ? 2 : 3);
                continue;
            }
            if ((double)B[i].count / ((double)bw * bh) < 0.2) {
                TRACE(4);
                continue;
            }
            if ((size_t)ncand >= det->cand_capacity) {
                size_t cap = det->cand_capacity ? 2 * det->cand_capacity : 256;
                component *c = (component *)realloc(det->cands, cap * sizeof(component));
                if (c == NULL) return -2;
                det->cands = c;
                det->cand_capacity = cap;
            }
            component *c = &det->cands[ncand];
            c->count = 0;
            for (int k = 0; k < 16; k++) c->extv[k] = INT32_MIN;
            c->row_y = -1;
            c->minx = t_minx, c->maxx = t_maxx, c->miny = t_miny, c->maxy = t_maxy;
            det->slot[i] = ncand++;
        }
        /* Extreme points only for the candidates: one more pass over the runs. */
        for (int i = 0; i < nruns; i++) {
            int32_t c = det->slot[P[i]];
            if (c >= 0) comp_add_run(&det->cands[c], &R[i]);
        }
        for (int ci = 0; ci < ncand; ci++) comp_flush_row(&det->cands[ci]);
        PROF_MARK(PROF_BLOBS);

        for (int ci = 0; ci < ncand; ci++) {
            component comp = det->cands[ci];
            int t_minx = comp.minx, t_maxx = comp.maxx, t_miny = comp.miny, t_maxy = comp.maxy;
            int bw = t_maxx - t_minx + 1, bh = t_maxy - t_miny + 1;
            int big = bw > bh ? bw : bh;
            /* Levels run fine to coarse; skip a blob whose centre a finer level already decoded. */
            double ccx = 0.5 * (comp.minx + comp.maxx) * s, ccy = 0.5 * (comp.miny + comp.maxy) * s;
            if (covered(out, found < max_out ? found : max_out, ccx, ccy, 0.3 * big * s + 2.0)) {
                TRACE(5);
                continue;
            }
            /* Rough quad: of the extreme points (in clockwise angular order), the four that span the
             * largest quadrilateral. Start from the best of the eight axis and diagonal extremes, then
             * let each vertex move to any of the 16 extremes between its neighbours while the area
             * grows: axis or diagonal extremes alone slide along a side near 22 and 67 degrees. */
            /* Twice the area (positive, as the extremes run clockwise with y down) is the sum of
             * cross(p[t], p[t + 1]), so moving vertex v to point m changes it by a term linear in m:
             * cross(m, next - prev). Each move is one dot product. */
            int(*E)[2] = comp.ext;
            int idx[4] = {0, 4, 8, 12};
            int64_t best2 = -1; /* twice the area: (p2 - p0) x (p3 - p1) */
            for (int i = 0; i < 10; i += 2) {
                for (int j = i + 2; j < 12; j += 2) {
                    for (int k = j + 2; k < 14; k += 2) {
                        for (int l = k + 2; l < 16; l += 2) {
                            int64_t a2 = (int64_t)(E[k][0] - E[i][0]) * (E[l][1] - E[j][1]) -
                                         (int64_t)(E[k][1] - E[i][1]) * (E[l][0] - E[j][0]);
                            if (a2 > best2) best2 = a2, idx[0] = i, idx[1] = j, idx[2] = k, idx[3] = l;
                        }
                    }
                }
            }
            for (int moved = 1, sweep = 0; moved && sweep < 4; sweep++) {
                moved = 0;
                for (int v = 0; v < 4; v++) {
                    int pv = idx[(v + 3) & 3], nv = idx[(v + 1) & 3];
                    int ex = E[nv][0] - E[pv][0], ey = E[nv][1] - E[pv][1];
                    int64_t cur = (int64_t)E[idx[v]][0] * ey - (int64_t)E[idx[v]][1] * ex;
                    for (int m = (pv + 1) & 15; m != nv; m = (m + 1) & 15) {
                        int64_t gain = (int64_t)E[m][0] * ey - (int64_t)E[m][1] * ex;
                        if (gain > cur) cur = gain, idx[v] = m, moved = 1;
                    }
                }
            }
            int pick[4][2];
            for (int t = 0; t < 4; t++) pick[t][0] = E[idx[t]][0], pick[t][1] = E[idx[t]][1];
            double best_area = quad_area_pts(pick);
            /* Every tag36h11 code has at least 10 white payload cells, so at most 54/64 of a tag's
             * square is dark (0.85 measured, against the quad grown by the half sample its corner
             * samples sit inside). Solid strokes and filled shapes run well past that. */
            double perim_q = 0;
            for (int k = 0; k < 4; k++) {
                int ex = pick[(k + 1) & 3][0] - pick[k][0], ey = pick[(k + 1) & 3][1] - pick[k][1];
                perim_q += sqrt((double)(ex * ex + ey * ey));
            }
            if (comp.count > 0.95 * (best_area + 0.5 * perim_q + 1.0)) {
                PROF_MARK(PROF_QUAD);
                TRACE(6);
                continue;
            }
            if (best_area < 0.3 * bw * bh) {
                PROF_MARK(PROF_QUAD);
                TRACE(6);
                continue;
            }
            PROF_MARK(PROF_QUAD);
            vec2 rough[4], fine[4], finer[4];
            for (int k = 0; k < 4; k++) {
                rough[k].x = pick[k][0] * (double)s;
                rough[k].y = pick[k][1] * (double)s;
            }
            /* Rough corners are dark samples, so the true border lies outward of them by at most
             * one stride: search mostly outward, which keeps payload edges out of the window. */
            double reach = 0.5 * ((1.5 * s + 1.0) + (0.5 * s + 1.0));
            double edge_w = 0;
            if (!fit_sides(img, width, height, stride, rough, fine, 1.5 * s + 1.0, 0.5 * s + 1.0, params->min_contrast, 1, 6, &edge_w, NULL)) {
                PROF_MARK(PROF_FIT1);
                TRACE(7);
                continue;
            }
            /* The first fit is already within a fraction of a cell: decode now, with two bits of
             * slack, and spend the refinement passes only on quads that are tags. A random quad
             * passes this only about 1.5% of the time. */
            PROF_MARK(PROF_FIT1);
            vec2 probe[4];
            napa_detection early;
            edge_levels lv = {255, 0};
            memcpy(probe, fine, sizeof probe);
            if (!decode_quad(det, img, width, height, stride, probe, params, 2, &early, NULL, &lv)) {
                /* The blob can include the white quiet zone when that margin is darker than the
                 * background around it (a grey card on a bright wall): then this quad is the margin's
                 * outline and the black square is its inner 8/10. Refit there and try once more. */
                int rescued = 0;
                double Hq[9], perim = 0;
                square_to_quad(fine, Hq);
                vec2 inner[4] = {project(Hq, 0.1, 0.1), project(Hq, 0.9, 0.1), project(Hq, 0.9, 0.9),
                                 project(Hq, 0.1, 0.9)};
                for (int k = 0; k < 4; k++) perim += hypot(fine[(k + 1) & 3].x - fine[k].x, fine[(k + 1) & 3].y - fine[k].y);
                double r = 0.45 * perim / 40.0; /* 0.45 of a cell, if the quad spans 10 cells */
                if (r >= 1.5 && fit_sides(img, width, height, stride, inner, finer, r, r, params->min_contrast, 0, 8, &edge_w, NULL)) {
                    memcpy(probe, finer, sizeof probe);
                    if (decode_quad(det, img, width, height, stride, probe, params, 2, &early, NULL, &lv)) {
                        memcpy(fine, finer, sizeof fine);
                        reach = r;
                        rescued = 1;
                    }
                }
                if (!rescued) {
                    PROF_MARK(PROF_DECODE1);
                    TRACE(8);
                    continue;
                }
            }
            PROF_MARK(PROF_DECODE1);
            double cell = 0;
            for (int k = 0; k < 4; k++) {
                double ex = fine[(k + 1) & 3].x - fine[k].x, ey = fine[(k + 1) & 3].y - fine[k].y;
                cell += sqrt(ex * ex + ey * ey) / 32.0;
            }
            /* Coarse-to-fine: shrink the symmetric window to the previous pass's resolution, but
             * never below the measured edge width, so the whole ramp stays inside it. */
            int ok = 1;
            for (int pass = 0; pass < 5; pass++) {
                double step = 2.0 * reach / 14.0;
                double next = 1.5 * step;
                if (next < 1.5) next = 1.5;
                if (next < 1.25 * edge_w) next = 1.25 * edge_w;
                /* ...but within half a cell, so the window never reaches the next payload edge. */
                if (next > 0.45 * cell && 0.45 * cell >= 1.0) next = 0.45 * cell;
                if (pass > 0 && (next >= 0.9 * reach || step <= 0.35)) break;
                reach = next;
                ok = fit_sides(img, width, height, stride, fine, finer, reach, reach, params->min_contrast, 0, 8, &edge_w, &lv);
                if (!ok) break;
                memcpy(fine, finer, sizeof fine);
            }
            /* One more pass at the converged window, re-centred on the last estimate, with every
             * edge sample. */
            if (ok) ok = fit_sides(img, width, height, stride, fine, finer, reach, reach, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv);
            if (!ok) {
                PROF_MARK(PROF_REFINE);
                TRACE(9);
                continue;
            }
            PROF_MARK(PROF_REFINE);
            napa_detection d;
            double side_level[4];
            edge_levels final_lv = {255, 0};
            int decoded_ = decode_quad(det, img, width, height, stride, finer, params, 0, &d, side_level, &final_lv);
            PROF_MARK(PROF_DECODE2);
            if (!decoded_) {
                TRACE(10);
                continue;
            }
            /* A side whose border cells read white was fitted one cell out, on the quiet zone's outer
             * edge (the payload still decodes through error correction). Move it in by a cell, refit,
             * and keep the result if it decodes the same tag with a darker border. */
            int bad = 0;
            double level_before = 0;
            for (int k = 0; k < 4; k++) {
                bad |= (side_level[k] > 0.3) << k;
                level_before += side_level[k];
            }
            if (bad && bad != 15) {
                double Hs[9], perim = 0;
                square_to_quad(finer, Hs);
                double u0 = (bad & 8) ? 0.125 : 0, v0 = (bad & 1) ? 0.125 : 0;
                double u1 = (bad & 2) ? 0.875 : 1, v1 = (bad & 4) ? 0.875 : 1;
                vec2 moved[4] = {project(Hs, u0, v0), project(Hs, u1, v0), project(Hs, u1, v1), project(Hs, u0, v1)};
                for (int k = 0; k < 4; k++) perim += hypot(moved[(k + 1) & 3].x - moved[k].x, moved[(k + 1) & 3].y - moved[k].y);
                double r = 0.45 * perim / 32.0;
                vec2 fixed[4];
                napa_detection d2;
                double side2[4], level_after = 0;
                /* The moved side starts tilted (it came from a skewed quad): fit, then refit once re-centred
                 * in a window just wide enough for the edge, as the main refinement converges. */
                double r2 = 1.25 * edge_w > 1.5 ? 1.25 * edge_w : 1.5;
                vec2 first[4];
                int refit = r >= 1.0 && fit_sides(img, width, height, stride, moved, first, r, r, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv);
                if (refit && r2 < r) refit = fit_sides(img, width, height, stride, first, fixed, r2, r2, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv);
                else if (refit) memcpy(fixed, first, sizeof fixed);
                if (refit && decode_quad(det, img, width, height, stride, fixed, params, 0, &d2, side2, NULL) && d2.id == d.id) {
                    for (int k = 0; k < 4; k++) level_after += side2[k];
                    if (level_after < level_before - 0.2) {
                        d = d2;
                        memcpy(finer, fixed, sizeof finer);
                    }
                }
            }
            /* A clipped white (or black) level biases every edge toward the clipped side; measure the
             * bias from the border's own width and undo it. Only when clipping is visible. */
            if (final_lv.white >= 250 || final_lv.black <= 4) {
                double delta = 0;
                if (border_bias(img, width, height, stride, finer, final_lv.black, final_lv.white, &delta) >= 4 &&
                    fabs(delta) > 0.02 && fabs(delta) < 1.5) {
                    vec2 grown[4];
                    napa_detection d3;
                    grow_quad(finer, delta, grown);
                    if (decode_quad(det, img, width, height, stride, grown, params, 0, &d3, NULL, NULL) && d3.id == d.id) d = d3;
                }
            }
            if (already_found(out, found < max_out ? found : max_out, &d)) {
                TRACE(11);
                continue;
            }
            TRACE(12);
            if (found < max_out) out[found] = d;
            found++;
        }
        PROF_MARK(PROF_BLOBS);
        if (last) break;
    }
    return found < max_out ? found : max_out;
}

/* setuptools links every extension with a module-init export on Windows. The library is loaded
 * with ctypes, never imported, so this placeholder only satisfies the linker. */
NAPA_API void *PyInit__napa_core(void) { return 0; }
