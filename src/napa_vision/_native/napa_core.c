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

#if defined(__GNUC__) || defined(__clang__)
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
    int ext[8][2]; /* extremes: +x, -x, +y, -y, +(x+y), -(x+y), +(x-y), -(x-y) */
    int extv[8];
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
    uint8_t *tile_lo;
    uint8_t *tile_hi;
    napa_run *runs;
    int32_t *parent; /* union-find over runs; a blob's root is its first run in raster order */
    blob_box *boxes;
    int32_t *slot;   /* root run -> candidate index, or -1 */
    component *cands;
    size_t cand_capacity;
    uint64_t *row_bits;
    uint8_t *row_min;
    uint8_t *row_max;
    uint8_t *row_thr;
    size_t row_capacity;
    size_t run_capacity;
    size_t level_capacity;
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
    free(det->tile_lo);
    free(det->tile_hi);
    free(det->runs);
    free(det->parent);
    free(det->boxes);
    free(det->slot);
    free(det->cands);
    free(det->row_bits);
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
        if ((p = (uint8_t *)realloc(det->tile_lo, tiles)) == NULL) return -1;
        det->tile_lo = p;
        if ((p = (uint8_t *)realloc(det->tile_hi, tiles)) == NULL) return -1;
        det->tile_hi = p;
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

static inline int32_t find_root(int32_t *P, int32_t i) {
    while (P[i] != i) {
        P[i] = P[P[i]];
        i = P[i];
    }
    return i;
}

/* Threshold the level and label its dark samples as runs joined by union-find (8-connected:
 * runs in adjacent rows touch if they overlap within one sample). Returns the run count. */
static int threshold_level(napa_detector *det, int lw, int lh, int min_contrast) {
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
    /* 3x3 tile neighbourhood, separably: horizontal then vertical. */
    for (int ty = 0; ty < th; ty++) {
        const uint8_t *a = det->tile_min + (size_t)ty * tw, *b = det->tile_max + (size_t)ty * tw;
        uint8_t *lo = det->tile_lo + (size_t)ty * tw, *hi = det->tile_hi + (size_t)ty * tw;
        for (int tx = 0; tx < tw; tx++) {
            uint8_t l = a[tx], h = b[tx];
            if (tx > 0) {
                l = a[tx - 1] < l ? a[tx - 1] : l;
                h = b[tx - 1] > h ? b[tx - 1] : h;
            }
            if (tx + 1 < tw) {
                l = a[tx + 1] < l ? a[tx + 1] : l;
                h = b[tx + 1] > h ? b[tx + 1] : h;
            }
            lo[tx] = l;
            hi[tx] = h;
        }
    }
    for (int ty = 0; ty < th; ty++) {
        const uint8_t *lo_up = det->tile_lo + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *lo_mid = det->tile_lo + (size_t)ty * tw;
        const uint8_t *lo_dn = det->tile_lo + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        const uint8_t *hi_up = det->tile_hi + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *hi_mid = det->tile_hi + (size_t)ty * tw;
        const uint8_t *hi_dn = det->tile_hi + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        int any = 0;
        for (int tx = 0; tx < tw; tx++) {
            uint8_t l = lo_mid[tx], h = hi_mid[tx];
            l = lo_up[tx] < l ? lo_up[tx] : l;
            l = lo_dn[tx] < l ? lo_dn[tx] : l;
            h = hi_up[tx] > h ? hi_up[tx] : h;
            h = hi_dn[tx] > h ? hi_dn[tx] : h;
            /* Boyer-Moore-style skip: a flat neighbourhood cannot hold a tag edge, so its
             * threshold is 0 and no sample can fall below it. */
            uint8_t t = (h - l >= min_contrast) ? (uint8_t)((l + h + 1) / 2) : 0;
            any |= t;
            memset(trow + 4 * tx, t, 4);
        }
        int y0 = ty * TILE;
        int y1 = y0 + TILE < lh ? y0 + TILE : lh;
        if (!any) continue; /* no contrast anywhere in this band: no dark samples, no runs */
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
                        int32_t ra = find_root(P, a), rb = find_root(P, b);
                        if (ra < rb) P[rb] = ra;
                        else if (rb < ra) P[ra] = rb;
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
        for (int i = 0; i < np; i++) {
            double t = 0.12 + 0.76 * (i + 0.5) / np;
            double bx = a.x + (b.x - a.x) * t, by = a.y + (b.y - a.y) * t;
            double prof[KMAX];
            double sx0 = bx - nx * r_out, sy0 = by - ny * r_out;
            double sx1 = bx + nx * r_in, sy1 = by + ny * r_in;
            if (sx0 >= 0 && sy0 >= 0 && sx1 >= 0 && sy1 >= 0 && sx0 < w - 1 && sx1 < w - 1 && sy0 < h - 1 &&
                sy1 < h - 1) {
                /* Whole profile inside the image: skip per-sample clamping. */
                double sx = sx0, sy = sy0, ix = nx * step, iy = ny * step;
                for (int j = 0; j < K; j++, sx += ix, sy += iy) {
                    int x0 = (int)sx, y0 = (int)sy;
                    double fx = sx - x0, fy = sy - y0;
                    const uint8_t *q = img + (size_t)y0 * stride + x0;
                    double top = q[0] + (q[1] - q[0]) * fx;
                    double bot = q[stride] + (q[stride + 1] - q[stride]) * fx;
                    prof[j] = top + (bot - top) * fy;
                }
            } else {
                for (int j = 0; j < K; j++) {
                    double o = -r_out + j * step;
                    prof[j] = sample_bilinear(img, w, h, stride, bx + nx * o, by + ny * o);
                }
            }
            /* outside (white) minus inside (black): the outer border edge is a positive peak. */
            double best = 0;
            int bj = -1;
            double g[KMAX];
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
                    if (g[j] >= 0.6 * best && (j == K - 2 || g[j] >= g[j + 1])) {
                        bj = j;
                        best = g[j];
                        break;
                    }
                }
            }
            /* Normalise by the derivative span so the strength test is independent of the step. */
            if (bj < 0 || best / (2.0 * step) < min_edge * 0.25) continue;
            /* Centre of mass of the positive gradient lobe around the peak. For an area-sampled edge
             * this is exactly the edge at any blur width; trimming or squaring the weights biases
             * sharp edges. The lobe ends where the gradient changes sign, and the final window is
             * capped below half a cell, so the next payload edge never contributes. */
            int jl = bj, jr = bj;
            while (jl > 1 && g[jl - 1] > 0.05 * best) jl--;
            while (jr < K - 2 && g[jr + 1] > 0.05 * best) jr++;
            double sg = 0, sj = 0;
            for (int j = jl; j <= jr; j++) {
                double wv = g[j];
                sg += wv;
                sj += wv * j;
            }
            double cj = sg > 1e-9 ? sj / sg : bj;
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
                    double mid = 0.5 * (wl + bl);
                    double best_c = -1, best_d = 1e9;
                    for (int j = jl - 1; j <= jr && j + 1 < K; j++) {
                        if (j < 0 || !(prof[j] >= mid && prof[j + 1] < mid)) continue;
                        double c = j + (prof[j] - mid) / (prof[j] - prof[j + 1]);
                        if (fabs(c - cj) < best_d) best_d = fabs(c - cj), best_c = c;
                    }
                    if (best_c < 0) break;
                    cj = best_c;
                }
            }
            width_sum += (jr - jl + 1) * step;
            width_n++;
            double pos = -r_out + cj * step;
            px[good] = bx + nx * pos;
            py[good] = by + ny * pos;
            wt[good] = best;
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
            double theta = 0.5 * atan2(2 * sxy, sxx - syy);
            lx[k] = mx;
            ly[k] = my;
            dx[k] = cos(theta);
            dy[k] = sin(theta);
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
    double z = H[6] * u + H[7] * v + H[8];
    vec2 p = {(H[0] * u + H[1] * v + H[2]) / z, (H[3] * u + H[4] * v + H[5]) / z};
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

static void comp_add_run(component *c, const napa_run *r) {
    int x0 = r->x0, x1 = r->x1, y = r->y;
    /* The extreme of each direction over a run lies at one of its ends. */
    int px[8] = {x1, x0, x0, x0, x1, x0, x1, x0};
    int vals[8] = {x1, -x0, y, -y, x1 + y, -(x0 + y), x1 - y, y - x0};
    for (int k = 0; k < 8; k++) {
        if (c->count == 0 || vals[k] > c->extv[k]) {
            c->extv[k] = vals[k];
            c->ext[k][0] = px[k];
            c->ext[k][1] = y;
        }
    }
    c->count += x1 - x0 + 1;
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

/* Tile contrast floor that rises with sensor noise. The median absolute difference of
 * neighbouring samples is about 0.95 sigma for Gaussian noise, and a flat noisy tile
 * neighbourhood spans about 5.5 sigma, so a floor of 6 sigma keeps noise from forming
 * dark networks. Capped well below real tag contrast. */
static int contrast_floor(const uint8_t *L, int lw, int lh, int min_contrast) {
    uint32_t hist[256] = {0};
    uint32_t n = 0;
    for (int y = 0; y < lh; y += 8) {
        const uint8_t *row = L + (size_t)y * lw;
        for (int x = 0; x + 1 < lw; x += 2) {
            int d = row[x + 1] - row[x];
            hist[d < 0 ? -d : d]++;
            n++;
        }
    }
    if (n == 0) return min_contrast;
    uint32_t seen = 0;
    int median = 0;
    while (median < 255 && (seen += hist[median]) < (n + 1) / 2) median++;
    int floor = (int)(6.0 * median / 0.954 + 0.5);
    if (floor > 64) floor = 64;
    return floor > min_contrast ? floor : min_contrast;
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
        if (s == base) contrast = contrast_floor(det->level, lw, lh, params->min_contrast);
        int nruns = threshold_level(det, lw, lh, contrast);
        PROF_MARK(PROF_THRESHOLD);

        /* Blobs: each run's root, then every blob's count and bounding box. */
        napa_run *R = det->runs;
        int32_t *P = det->parent;
        blob_box *B = det->boxes;
        for (int i = 0; i < nruns; i++) {
            int32_t r = find_root(P, i);
            P[i] = r;
            if (r == i) {
                B[i].count = R[i].x1 - R[i].x0 + 1;
                B[i].minx = R[i].x0, B[i].maxx = R[i].x1, B[i].miny = B[i].maxy = R[i].y;
            } else {
                B[r].count += R[i].x1 - R[i].x0 + 1;
                B[r].minx = R[i].x0 < B[r].minx ? R[i].x0 : B[r].minx;
                B[r].maxx = R[i].x1 > B[r].maxx ? R[i].x1 : B[r].maxx;
                B[r].maxy = R[i].y; /* runs arrive row by row */
            }
        }
        /* Keep tag-sized, tag-shaped blobs as candidates, in raster order of their first sample. */
        int min_size = lo;
        int max_size = last ? 1 << 30 : hi;
        int ncand = 0;
        for (int i = 0; i < nruns; i++) {
            det->slot[i] = -1;
            if (P[i] != i) continue;
            int t_minx = B[i].minx, t_maxx = B[i].maxx, t_miny = B[i].miny, t_maxy = B[i].maxy;
            int bw = t_maxx - t_minx + 1, bh = t_maxy - t_miny + 1;
            int big = bw > bh ? bw : bh, small = bw < bh ? bw : bh;
            if (small < min_size || big > max_size || big > 3 * small) {
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
            c->minx = t_minx, c->maxx = t_maxx, c->miny = t_miny, c->maxy = t_maxy;
            det->slot[i] = ncand++;
        }
        /* Extreme points only for the candidates: one more pass over the runs. */
        for (int i = 0; i < nruns; i++) {
            int32_t c = det->slot[P[i]];
            if (c >= 0) comp_add_run(&det->cands[c], &R[i]);
        }
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
            /* Rough quad: of the eight extreme points (in clockwise angular order), the four that
             * span the largest quadrilateral. Axis or diagonal extremes alone degrade near 22 and
             * 67 degrees; the best mix tracks the corners at any rotation. */
            static const int order[8] = {0, 4, 2, 7, 1, 5, 3, 6};
            int pick[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
            double best_area = -1;
            for (int i = 0; i < 5; i++) {
                for (int j = i + 1; j < 6; j++) {
                    for (int k = j + 1; k < 7; k++) {
                        for (int l = k + 1; l < 8; l++) {
                            int q[4][2] = {{comp.ext[order[i]][0], comp.ext[order[i]][1]},
                                           {comp.ext[order[j]][0], comp.ext[order[j]][1]},
                                           {comp.ext[order[k]][0], comp.ext[order[k]][1]},
                                           {comp.ext[order[l]][0], comp.ext[order[l]][1]}};
                            double area = quad_area_pts(q);
                            if (area > best_area) {
                                best_area = area;
                                memcpy(pick, q, sizeof pick);
                            }
                        }
                    }
                }
            }
            /* Every tag36h11 code has at least 10 white payload cells, so at most 54/64 of a tag's
             * square is dark (0.85 measured, against the quad grown by the half sample its corner
             * samples sit inside). Solid strokes and filled shapes run well past that. */
            double perim_q = 0;
            for (int k = 0; k < 4; k++) perim_q += hypot(pick[(k + 1) & 3][0] - pick[k][0], pick[(k + 1) & 3][1] - pick[k][1]);
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
