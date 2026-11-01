/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
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
#include "families.h"

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

/* The fitted quad, in pixels, of a candidate that failed its early decode, and its level's stride. */
typedef struct {
    float xy[8];
    int s;
} failed_quad;

/* Multi-index hashing: a code within H bits of another matches at least one of H + 1 disjoint blocks
 * of bits exactly, so only codes sharing a block are compared. Lists are stored compressed by bucket. */
typedef struct {
    int blocks;
    int probe; /* 1: also look up every one-bit change of each block (multi-probe), so fewer, larger blocks work */
    int shift[6], bits[6];
    uint32_t *start[6]; /* (1 << bits) + 1 offsets per block */
    uint32_t *list[6];  /* rotated-code indices, by bucket */
} code_index;

/* The family a detector reads, laid out for decoding. Cells are in grid coordinates: the square's
 * top-left cell is (outer, outer), and the grid spans `grid` cells, the square plus any data rings
 * outside its quiet ring. */
typedef struct {
    const napa_family *f;
    int width, outer, grid, nbits, ncodes, nrot, max_hamming;
    int recommended_hamming; /* the default correction: chance matches no likelier than tag36h11's at 3 */
    int soft_hamming; /* families whose default corrects fewer bits than 3: the most bits a soft correction may fix */
    int piece_hamming; /* split_touching: the most bits a piece may correct, or -1 if pieces aren't decoded */
    double cells;      /* width as a double */
    double fill_limit; /* largest dark fraction a blob may have of its grown quad */
    double bias_slack; /* border_bias's side-agreement tolerance, relative to tag36h11's */
    int8_t cell[64];   /* each bit's grid index, most significant first */
    int8_t is_data[256];
    uint64_t near[64]; /* bits whose cells touch each bit's cell (8-connected), for errors_connected */
    uint64_t mask;     /* the low nbits bits */
    uint64_t *codes;   /* nrot codes: every code in 4 rotations, as observed (complemented if reversed) */
    int32_t *ids;
    int8_t *rotations;
    code_index index3, index5; /* for up to 3 and up to 5 wrong bits */
    uint32_t *stamp;
    uint32_t stamp_now;
} family_layout;

struct napa_detector {
    family_layout fam;
    int ssse3; /* the CPU supports SSSE3 (x86 only; picked once at creation) */

    uint8_t *level;
    uint8_t *tile_min;
    uint8_t *tile_max;
    uint8_t *nb_lo; /* one band's tile neighbourhood min, max, and threshold (row buffers) */
    uint8_t *nb_hi;
    uint8_t *nb_t;
    uint8_t *nb_f;
    napa_run *runs;
    int32_t *parent; /* backward union links, then -dense component index - 1 after aggregation */
    blob_box *boxes; /* packed components, in first-run raster order */
    int32_t *slot;   /* dense component -> candidate index, or -1 */
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
    failed_quad *failed; /* this frame's failed candidates, for skipping the same blob at a coarser level */
    size_t failed_n, failed_capacity;
    /* split_touching: run pairs of this level that touch only at a corner, or through a one-sample neck
     * (then their pieces); each neck pair's pieces' roots once the necks are joined; and each piece
     * proposed on its own: its component, its root, its blob's candidate, and the next candidate its runs
     * belong to (the larger piece holding it, or its blob's candidate). */
    int32_t *diag, *neck, *neck_root;
    size_t diag_n, neck_n, diag_capacity, neck_capacity;
    component *pieces;
    int32_t *piece_root, *piece_cand, *piece_next;
    size_t piece_capacity;
    napa_detection *piece_out; /* decoded pieces, appended after the other detections */
    size_t piece_out_capacity;
    uint8_t *fates; /* before necks are split: each level's candidates' and pieces' fates, for redoing it */
    size_t fate_capacity;
};

typedef struct {
    double x, y;
} vec2;

/* ---------------------------------------------------------------- family */

static uint64_t rotate_bits(const uint64_t code, const int8_t *perm, int nbits) {
    uint64_t out = 0;
    for (int k = 0; k < nbits; k++) out = (out << 1) | ((code >> (nbits - 1 - perm[k])) & 1u);
    return out;
}

static void free_index(code_index *ix) {
    for (int b = 0; b < 6; b++) {
        free(ix->start[b]);
        free(ix->list[b]);
        ix->start[b] = NULL, ix->list[b] = NULL;
    }
}

/* Split nbits into `blocks` blocks of bits (low bits first; the larger blocks last) and index every
 * rotated code by each block's value. Returns -1 if out of memory. */
static int build_index(code_index *ix, const uint64_t *codes, int n, int nbits, int blocks, int probe) {
    ix->blocks = blocks;
    ix->probe = probe;
    for (int b = 0, shift = 0; b < blocks; b++) {
        ix->shift[b] = shift;
        ix->bits[b] = nbits / blocks + (b >= blocks - nbits % blocks ? 1 : 0);
        shift += ix->bits[b];
        size_t buckets = (size_t)1 << ix->bits[b];
        uint32_t *st = (uint32_t *)calloc(buckets + 1, sizeof(uint32_t));
        uint32_t *ls = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
        ix->start[b] = st, ix->list[b] = ls;
        if (st == NULL || ls == NULL) return -1;
        uint64_t m = buckets - 1;
        for (int i = 0; i < n; i++) st[((codes[i] >> ix->shift[b]) & m) + 1]++;
        for (size_t v = 0; v < buckets; v++) st[v + 1] += st[v];
        /* Fill each bucket in code order, using the next bucket's start as a cursor, then shift back. */
        for (int i = 0; i < n; i++) ls[st[(codes[i] >> ix->shift[b]) & m]++] = (uint32_t)i;
        for (size_t v = buckets; v > 0; v--) st[v] = st[v - 1];
        st[0] = 0;
    }
    return 0;
}

/* Index of the rotated code within max_bits (at most 5) of `code`, or -1. When codes are at least
 * 2 * max_bits + 1 apart (the final decode), any match is unique. */
static int nearest_code(napa_detector *det, uint64_t code, int max_bits, int *distance) {
    family_layout *L = &det->fam;
    const code_index *ix = max_bits <= 3 ? &L->index3 : &L->index5;
    if (++L->stamp_now == 0) {
        memset(L->stamp, 0, sizeof(uint32_t) * (size_t)L->nrot);
        L->stamp_now = 1;
    }
    for (int b = 0; b < ix->blocks; b++) {
        const uint32_t *st = ix->start[b];
        const uint32_t *ls = ix->list[b];
        uint64_t v0 = (code >> ix->shift[b]) & (((uint64_t)1 << ix->bits[b]) - 1);
        for (int flip = -1; flip < (ix->probe ? ix->bits[b] : 0); flip++) {
            uint64_t v = flip < 0 ? v0 : v0 ^ ((uint64_t)1 << flip);
            for (uint32_t e = st[v]; e < st[v + 1]; e++) {
                uint32_t i = ls[e];
                if (L->stamp[i] == L->stamp_now) continue;
                L->stamp[i] = L->stamp_now;
                int d = napa_popcount64(code ^ L->codes[i]);
                if (d <= max_bits) {
                    *distance = d;
                    return (int)i;
                }
            }
        }
    }
    return -1;
}

static int errors_connected(const family_layout *L, uint64_t diff);

static const napa_family *find_family(const char *name) {
    if (name == NULL) return NULL;
    for (int i = 0; i < NAPA_FAMILY_COUNT; i++)
        if (strcmp(napa_families[i].name, name) == 0) return &napa_families[i];
    return NULL;
}

static void free_family(family_layout *L) {
    free(L->codes);
    free(L->ids);
    free(L->rotations);
    free(L->stamp);
    free_index(&L->index3);
    free_index(&L->index5);
}

/* Geometry and correction limits need no code tables, including for metadata-only queries. */
static int init_layout(family_layout *L, const napa_family *f, int8_t perm[64]) {
    memset(L, 0, sizeof *L);
    L->f = f;
    L->width = f->width, L->nbits = f->nbits, L->ncodes = f->ncodes, L->nrot = 4 * f->ncodes;
    L->cells = f->width;
    /* Each side's clipping estimate averages about (width - 2) / 2 border measurements: allow its noise. */
    L->bias_slack = sqrt(6.0 / (f->width - 2));
    L->max_hamming = (f->min_distance - 1) / 2;
    L->mask = f->nbits == 64 ? ~(uint64_t)0 : (((uint64_t)1 << f->nbits) - 1);
    int lo = 0, hi = f->width - 1;
    for (int k = 0; k < f->nbits; k++) {
        lo = f->x[k] < lo ? f->x[k] : lo, lo = f->y[k] < lo ? f->y[k] : lo;
        hi = f->x[k] > hi ? f->x[k] : hi, hi = f->y[k] > hi ? f->y[k] : hi;
    }
    L->outer = -lo, L->grid = hi - lo + 1;
    if (L->outer != hi - (f->width - 1) || L->grid > 16 || f->nbits > 64) return -1; /* not symmetric */
    for (int k = 0; k < f->nbits; k++) {
        int gx = f->x[k] + L->outer, gy = f->y[k] + L->outer;
        L->cell[k] = (int8_t)(gy * L->grid + gx);
        L->is_data[gy * L->grid + gx] = 1;
    }
    for (int k = 0; k < f->nbits; k++) {
        int gx = L->cell[k] % L->grid, gy = L->cell[k] / L->grid;
        /* Turned 90 degrees counter-clockwise, as numpy's rot90: out[y][x] = in[x][grid - 1 - y]. */
        int src = gx * L->grid + (L->grid - 1 - gy);
        perm[k] = -1;
        for (int j = 0; j < f->nbits; j++)
            if (L->cell[j] == src) perm[k] = (int8_t)j;
        if (perm[k] < 0) return -1; /* a layout that doesn't rotate onto itself */
        for (int j = 0; j < f->nbits; j++) {
            int dx = L->cell[j] % L->grid - gx, dy = L->cell[j] / L->grid - gy;
            if (j != k && dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1) L->near[k] |= (uint64_t)1 << (f->nbits - 1 - j);
        }
    }
    /* How likely a random pattern is to fall within h bits of some rotated code: nrot * sum_k C(n, k) / 2^n,
     * with three-bit corrections counted only for touching cells (errors_connected). Recommend the
     * largest h whose chance is at most 1e-4, about three times tag36h11's at its default of 3. */
    double placements[4] = {1, f->nbits, 0.5 * f->nbits * (f->nbits - 1), 0};
    for (int a = 0; a < f->nbits; a++)
        for (int b = a + 1; b < f->nbits; b++)
            for (int c = b + 1; c < f->nbits; c++) {
                uint64_t bits = ((uint64_t)1 << (f->nbits - 1 - a)) | ((uint64_t)1 << (f->nbits - 1 - b)) |
                                ((uint64_t)1 << (f->nbits - 1 - c));
                placements[3] += errors_connected(L, bits);
            }
    double chance = 0;
    L->piece_hamming = -1;
    for (int h = 0; h <= 3 && h <= L->max_hamming; h++) {
        chance += L->nrot * placements[h] / ldexp(1.0, f->nbits);
        if (h == 0 || chance <= 1e-4) L->recommended_hamming = h;
        /* split_touching proposes many more blobs, mostly not tags: its pieces may correct only as many
         * bits as keep chance matches no likelier than tag36h11's at 2 (2.3e-5). Short codes (tag16h5,
         * the 21h7 families) match random pieces too often even exactly, and aren't split. */
        if (h <= 2 && chance <= 2.5e-5) L->piece_hamming = h;
        if (chance > 1e-4) break;
    }
    /* Families that can't afford three hard corrections get soft ones (decode_quad), up to 3 bits. */
    int cap = L->max_hamming < 3 ? L->max_hamming : 3;
    L->soft_hamming = L->recommended_hamming < cap ? cap : 0;
    return 0;
}

/* Allocate and populate the tables only when creating a detector. */
static int load_family(family_layout *L, const napa_family *f) {
    int8_t perm[64];
    if (init_layout(L, f, perm) != 0) return -1;
    L->codes = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)L->nrot);
    L->ids = (int32_t *)malloc(sizeof(int32_t) * (size_t)L->nrot);
    L->rotations = (int8_t *)malloc((size_t)L->nrot);
    L->stamp = (uint32_t *)calloc((size_t)L->nrot, sizeof(uint32_t));
    if (L->codes == NULL || L->ids == NULL || L->rotations == NULL || L->stamp == NULL) return -1;
    int max_dark = 0;
    for (int id = 0; id < f->ncodes; id++) {
        /* Napa reads reversed families on the inverted image, where a printed white cell is dark. */
        uint64_t code = f->reversed ? ~f->codes[id] & L->mask : f->codes[id];
        int dark = 0;
        for (int k = 0; k < f->nbits; k++) {
            int gx = L->cell[k] % L->grid - L->outer, gy = L->cell[k] / L->grid - L->outer;
            int inside = gx > 0 && gy > 0 && gx < f->width - 1 && gy < f->width - 1;
            dark += inside && !((code >> (f->nbits - 1 - k)) & 1);
        }
        max_dark = dark > max_dark ? dark : max_dark;
        for (int rot = 0; rot < 4; rot++) {
            int index = rot * f->ncodes + id;
            L->codes[index] = code;
            L->ids[index] = id;
            L->rotations[index] = (int8_t)rot;
            code = rotate_bits(code, perm, f->nbits);
        }
    }
    /* Interior cells that aren't data (tagCustom48h12's centre) are left blank on the official artwork:
     * printed white, they're dark on the inverted image. Count them as dark. */
    for (int gy = 1; gy < f->width - 1; gy++)
        for (int gx = 1; gx < f->width - 1; gx++) max_dark += !L->is_data[(gy + L->outer) * L->grid + gx + L->outer];
    /* A tag36h11 tag is at most 54/64 dark (every code has at least 10 white payload cells); blobs are
     * rejected above 0.95 of the grown quad. Other families keep the same slack over their own bound. */
    int dark_cells = 4 * (f->width - 1) + max_dark;
    L->fill_limit = 0.95 + ((double)dark_cells / (f->width * f->width) - 54.0 / 64.0);
    if (build_index(&L->index3, L->codes, L->nrot, f->nbits, 4, 0) != 0) return -1;
    /* Up to 5 wrong bits: six exact blocks, or for large families three blocks probed with one changed bit
     * each (five errors leave some block with at most one), whose buckets are far smaller. */
    int large = L->nrot > 4096;
    if (build_index(&L->index5, L->codes, L->nrot, f->nbits, large ? 3 : 6, large) != 0) return -1;
    return 0;
}

int napa_tag36h11_code(int id, uint64_t *code) {
    return napa_family_code("tag36h11", id, code);
}

int napa_family_code(const char *family, int id, uint64_t *code) {
    const napa_family *f = find_family(family);
    if (f == NULL || code == NULL || id < 0 || id >= f->ncodes) return -1;
    *code = f->codes[id];
    return 0;
}

int napa_family_count(void) { return NAPA_FAMILY_COUNT; }

const char *napa_family_name(int index) {
    return index >= 0 && index < NAPA_FAMILY_COUNT ? napa_families[index].name : NULL;
}

int napa_family_info(const char *family, napa_family_layout *info) {
    const napa_family *f = find_family(family);
    if (f == NULL || info == NULL) return -1;
    info->ncodes = f->ncodes, info->nbits = f->nbits, info->width = f->width, info->total_width = f->total_width;
    info->reversed = f->reversed, info->min_distance = f->min_distance;
    family_layout L;
    int8_t perm[64];
    if (init_layout(&L, f, perm) != 0) return -1;
    info->max_hamming = L.max_hamming < 3 ? L.max_hamming : 3;
    info->recommended_hamming = L.recommended_hamming;
    for (int k = 0; k < 64; k++) {
        info->x[k] = k < f->nbits ? f->x[k] : 0;
        info->y[k] = k < f->nbits ? f->y[k] : 0;
    }
    return 0;
}

/* ---------------------------------------------------------------- lifecycle */

napa_detector *napa_create(void) { return napa_create_family("tag36h11"); }

napa_detector *napa_create_family(const char *family) {
    const napa_family *f = find_family(family);
    if (f == NULL) return NULL;
    napa_detector *det = (napa_detector *)calloc(1, sizeof(napa_detector));
    if (det != NULL) {
        if (load_family(&det->fam, f) != 0) {
            napa_destroy(det);
            return NULL;
        }
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
    free(det->failed);
    free_family(&det->fam);
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
    free(det->cands);
    free(det->diag);
    free(det->neck);
    free(det->neck_root);
    free(det->pieces);
    free(det->piece_root);
    free(det->piece_cand);
    free(det->piece_next);
    free(det->piece_out);
    free(det->fates);
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

/* The frame as the detector reads it. Reversed families read it inverted, one sample at a time, so the
 * frame is never copied. */
typedef struct {
    const uint8_t *p;
    int w, h, stride, invert;
} image_view;

static inline double sample_bilinear(const image_view *im, double x, double y) {
    const uint8_t *img = im->p;
    const int w = im->w, h = im->h, stride = im->stride;
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
    double v = a + (b - a) * fy;
    return im->invert ? 255.0 - v : v;
}

/* ---------------------------------------------------------------- threshold */

#define TILE 4
/* Tuning. A level keeps blobs whose bounding box spans up to NAPA_HI samples: a tag stays isolated
 * from its background only where its one-cell white margin spans about two samples (a side of 16+
 * samples), and its border stays solid under the tile threshold up to a side of about 64 samples,
 * which a rotated or tilted tag stretches to a box of about 96. Keep an overlapping band up to 192
 * so large low-contrast tags can be proposed before their thin quiet zone merges into the background.
 * Levels continue until the frame is under NAPA_LAST samples across. */
#define NAPA_HI 192
#define NAPA_LAST 144
/* Edges whose ramp is wider than this many cells overlap the next edge's ramp (motion or defocus
 * blur): locate them by midpoint crossing instead of the gradient's centre of mass. */
#define NAPA_CROSSING_CELLS 0.6
/* The final fit takes a profile every 1.5 px along each side, up to this many. */
#define NAPA_FINAL_POINTS 96
/* Midpoint-crossing passes: the plateau levels are re-read around each new estimate. */
#define NAPA_CROSS_ITERS 3
/* Noise floors are measured per region on the current grid. Sparse grids cap the floor at
 * NAPA_FLOOR_CAP because texture can inflate the estimate. A noisy sparse grid re-rakes with
 * block averages, then re-measures. The dense grid keeps original pixels for tiny borders and
 * permits a higher floor; stride two supplies an averaged recovery pass for low-contrast tags. */
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
 * and two scratch rows of w samples: acc (column sums) and hs (horizontal box sums).
 * The dense detector keeps stride-one proposals unfiltered for tiny tags, then uses a 3x3 kernel
 * at stride two to recover low-contrast tags. Edge fitting and decoding always read the original. */
static void box_rake(uint8_t *level, int lw, int lh, const uint8_t *img, int w, int h, int stride, int s, int kernel,
                     uint16_t *acc, uint16_t *hs) {
    int half = (kernel - 1) / 2;
    uint32_t m = (65536u + (uint32_t)(kernel * kernel) / 2) / (uint32_t)(kernel * kernel); /* 1 / kernel^2 in 0.16 fixed point */
    for (int y = 0; y < lh; y++) {
        int r0 = y * s - half, r1 = r0 + kernel;
        if (r0 < 0) r0 = 0;
        if (r1 > h) r1 = h;
        const uint8_t *row = img + (size_t)r0 * stride;
        for (int x = 0; x < w; x++) acc[x] = row[x];
        for (int r = r0 + 1; r < r1; r++) {
            row = img + (size_t)r * stride;
            for (int x = 0; x < w; x++) acc[x] = (uint16_t)(acc[x] + row[x]);
        }
        /* hs[c] = acc[c] + ... + acc[c + kernel - 1], for every block start c that fits in the row. */
        for (int c = 0; c + kernel <= w; c++) hs[c] = acc[c];
        for (int k = 1; k < kernel; k++)
            for (int c = 0; c + kernel <= w; c++) hs[c] = (uint16_t)(hs[c] + acc[c + k]);
        uint8_t *dst = level + (size_t)y * lw;
        /* Whole blocks are x_lo <= x < x_hi; the rest are clipped by the image edge. */
        int x_lo = (half + s - 1) / s, x_hi = (w - kernel + half) / s + 1;
        if (x_hi > lw) x_hi = lw;
        if (r1 - r0 < kernel) x_lo = x_hi = 0; /* a clipped row band: every block is partial */
        for (int x = x_lo; x < x_hi; x++) dst[x] = (uint8_t)((hs[x * s - half] * m + 32768u) >> 16);
        for (int x = 0; x < lw; x++) {
            if (x == x_lo && x_lo < x_hi) x = x_hi;
            if (x >= lw) break;
            int c0 = x * s - half, c1 = c0 + kernel, sum = 0;
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

#if defined(__GNUC__) || defined(__clang__)
#define NAPA_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define NAPA_FORCE_INLINE __forceinline
#else
#define NAPA_FORCE_INLINE inline
#endif

static inline int32_t find_root(int32_t *P, int32_t i) {
    while (P[i] != i) {
        P[i] = P[P[i]];
        i = P[i];
    }
    return i;
}

/* Join the runs a .. a_end - 1 of one row and b .. b_end - 1 of the next where they touch, except, with split,
 * corner contacts (and with split 2, one-sample necks), which are recorded instead. Inlined with a constant
 * split, so ordinary labelling has no test for it. */
static NAPA_FORCE_INLINE void join_rows(napa_detector *det, const napa_run *R, int32_t *P, int a, int a_end, int b,
                                        int b_end, const int split) {
    while (a < a_end && b < b_end) {
        if (R[a].x1 + 1 < R[b].x0) {
            a++;
        } else if (R[b].x1 + 1 < R[a].x0) {
            b++;
        } else if (split == 2 ? (R[a].x1 < R[b].x1 ? R[a].x1 : R[b].x1) <= (R[a].x0 > R[b].x0 ? R[a].x0 : R[b].x0)
                              : split && (R[a].x1 + 1 == R[b].x0 || R[b].x1 + 1 == R[a].x0)) {
            /* A corner contact (no overlap) or, with split 2, a neck (one sample). */
            int corner = R[a].x1 + 1 == R[b].x0 || R[b].x1 + 1 == R[a].x0;
            int32_t *pairs = corner ? det->diag : det->neck;
            size_t *count = corner ? &det->diag_n : &det->neck_n;
            pairs[(*count)++] = a;
            pairs[(*count)++] = b;
            if (R[a].x1 < R[b].x1) a++;
            else b++;
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

/* Threshold the level and label its dark samples as runs joined by union-find (8-connected:
 * runs in adjacent rows touch if they overlap within one sample). Returns the run count, or -1 when out
 * of memory. With split, runs that touch only at a corner (split 1), or also overlap by one sample (split 2),
 * aren't joined: each such pair is recorded in det->diag or det->neck, for split_pieces. */
static int threshold_level(napa_detector *det, int lw, int lh, int factor, const int split) {
    const uint8_t *L = det->level;
    napa_run *R = det->runs;
    int32_t *P = det->parent;
    int n = 0, prev_y = -2, prev_start = 0, prev_end = 0;
    det->diag_n = det->neck_n = 0;
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
    /* A tile spans 4 * factor samples of the grid where the floors were measured. */
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
                /* Two adjacent rows touch at most once per run of either. */
                size_t most = 2 * ((size_t)(prev_end - prev_start) + (size_t)(n - start));
                if (split && det->diag_n + most > det->diag_capacity) {
                    size_t cap = det->diag_capacity ? 2 * det->diag_capacity : 4096;
                    while (cap < det->diag_n + most) cap *= 2;
                    int32_t *p = (int32_t *)realloc(det->diag, cap * sizeof(int32_t));
                    if (p == NULL) return -1;
                    det->diag = p;
                    det->diag_capacity = cap;
                }
                if (split == 2 && det->neck_n + most > det->neck_capacity) {
                    size_t cap = det->neck_capacity ? 2 * det->neck_capacity : 4096;
                    while (cap < det->neck_n + most) cap *= 2;
                    int32_t *p = (int32_t *)realloc(det->neck, cap * sizeof(int32_t));
                    if (p == NULL) return -1;
                    det->neck = p;
                    if ((p = (int32_t *)realloc(det->neck_root, cap * sizeof(int32_t))) == NULL) return -1;
                    det->neck_root = p;
                    det->neck_capacity = cap;
                }
                if (split == 2) join_rows(det, R, P, prev_start, prev_end, start, n, 2);
                else if (split) join_rows(det, R, P, prev_start, prev_end, start, n, 1);
                else join_rows(det, R, P, prev_start, prev_end, start, n, 0);
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

static int fit_sides(const image_view *im, const vec2 in[4], vec2 out[4], double r_out, double r_in, int min_edge,
                     int outermost, int max_points, double *edge_width, const edge_levels *levels, double cells) {
    const uint8_t *img = im->p;
    const int w = im->w, h = im->h, stride = im->stride;
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
                if (im->invert)
                    for (int j = 0; j < K; j++) prof[j] = ((int64_t)255 << 24) - prof[j];
            } else {
                for (int j = 0; j < K; j++) {
                    double o = -r_out + j * step;
                    prof[j] = (int64_t)(sample_bilinear(im, bx + nx * o, by + ny * o) * 16777216.0 + 0.5);
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
                (jr - jl + 1) * step > NAPA_CROSSING_CELLS * len / cells) {
                /* Levels from the centres of the cells either side (white margin, black border), half
                 * a cell from the edge: outside the blurred ramp the window would sit in. */
                double half = len / (2.0 * cells);
                for (int iter = 0; iter < NAPA_CROSS_ITERS; iter++) {
                    double pos0 = -r_out + cj * step;
                    double wl = sample_bilinear(im, bx + nx * (pos0 - half), by + ny * (pos0 - half));
                    double bl = sample_bilinear(im, bx + nx * (pos0 + half), by + ny * (pos0 + half));
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
                if (kept == good) break; /* unchanged samples would produce the same fit again */
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

/* Whether the set bits of a code difference (bits as the family's cells) form one 8-connected group. */
static int errors_connected(const family_layout *L, uint64_t diff) {
    if (napa_popcount64(diff) <= 1) return 1;
    uint64_t reached = diff & (~diff + 1), last = 0;
    while (reached != last) {
        last = reached;
        for (int k = 0; k < L->nbits; k++)
            if (reached >> (L->nbits - 1 - k) & 1) reached |= L->near[k] & diff;
    }
    return reached == diff;
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
static int border_bias(const family_layout *L, const image_view *im, const vec2 q[4], double black, double white,
                       double *delta) {
    enum { N = 33 };
    double H[9], sum = 0, side_sum[4] = {0, 0, 0, 0};
    int n = 0, side_n[4] = {0, 0, 0, 0};
    square_to_quad(q, H);
    double mid = 0.5 * (black + white);
    for (int side = 0; side < 4; side++) {
        for (int c = 1; c < L->width - 1; c++) {
            /* (a, t): position along the side and depth inward, in cells; side 0 is the top (v = 0). */
            double a = c + 0.5;
#define UNIT(t, uu, vv)                                                                                      \
    do {                                                                                                     \
        double t_ = (t) / L->cells, a_ = a / L->cells;                                                       \
        uu = side == 0 ? a_ : side == 1 ? 1 - t_ : side == 2 ? a_ : t_;                                      \
        vv = side == 0 ? t_ : side == 1 ? a_ : side == 2 ? 1 - t_ : a_;                                      \
    } while (0)
            double u, v;
            UNIT(1.5, u, v);
            vec2 pc = project(H, u, v);
            if (sample_bilinear(im, pc.x, pc.y) < mid) continue; /* dark payload cell */
            double prof[N];
            vec2 pts[N];
            for (int j = 0; j < N; j++) {
                UNIT(-0.5 + 2.0 * j / (N - 1), u, v);
                pts[j] = project(H, u, v);
                prof[j] = sample_bilinear(im, pts[j].x, pts[j].y);
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
        if (side_n[side] && fabs(side_sum[side] / side_n[side] - *delta) > (0.05 + 0.25 * fabs(*delta)) * L->bias_slack)
            return 0;
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

/* Read the square's border cells (into grid and border) and the quiet ring half a cell outside them, for
 * the square mapped by H: 28 and 36 samples for an 8-cell square. Black and white are their medians.
 * Returns how many border cells are darker than the midpoint, or -1 if white - black is under
 * min_contrast. */
static int read_border(const family_layout *L, const image_view *im, const double H[9], int min_contrast,
                       double *grid, double *border, double *black_out, double *white_out) {
    const int W = L->width, o = L->outer, G = L->grid;
    const double cells = L->cells;
    double ring[4 * 16 + 4];
    int nb = 0, nr = 0;
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c += (r == 0 || r == W - 1) ? 1 : W - 1) {
            vec2 p = project(H, (c + 0.5) / cells, (r + 0.5) / cells);
            double v = sample_bilinear(im, p.x, p.y);
            grid[(r + o) * G + c + o] = v;
            border[nb++] = v;
        }
    }
    for (int i = 0; i < W + 2; i++) {
        double u = (i - 0.5) / cells;
        vec2 p;
        p = project(H, u, -0.5 / cells);
        ring[nr++] = sample_bilinear(im, p.x, p.y);
        p = project(H, u, (W + 0.5) / cells);
        ring[nr++] = sample_bilinear(im, p.x, p.y);
    }
    for (int i = 1; i < W + 1; i++) {
        double v = (i - 0.5) / cells;
        vec2 p;
        p = project(H, -0.5 / cells, v);
        ring[nr++] = sample_bilinear(im, p.x, p.y);
        p = project(H, (W + 0.5) / cells, v);
        ring[nr++] = sample_bilinear(im, p.x, p.y);
    }
    double bs[4 * 16];
    memcpy(bs, border, sizeof(double) * (size_t)nb);
    double black = median_even(bs, nb);
    double white = median_even(ring, nr);
    *black_out = black, *white_out = white;
    if (white - black < min_contrast) return -1;
    double thr = 0.5 * (black + white);
    int dark = 0;
    for (int i = 0; i < nb; i++) dark += border[i] < thr;
    return dark;
}

/* The params' min_border_cells counts out of tag36h11's 28 border cells; scale it to the family's. */
static int min_border(const family_layout *L, int min_border_cells) {
    return (min_border_cells * 4 * (L->width - 1) + 14) / 28;
}

/* Whether quad q has a plausible tag border, before any edge fitting: its border cells read dark against
 * the ring outside, with at least min_dark of them dark. */
static int border_plausible(const family_layout *L, const image_view *im, const vec2 q[4], int min_contrast,
                            int min_dark) {
    double H[9], grid[256], border[64], black, white;
    square_to_quad(q, H);
    return read_border(L, im, H, min_contrast, grid, border, &black, &white) >= min_dark;
}

/* Decode the tag inside quad q (reordered in place to a consistent winding). `slack` loosens the
 * border and Hamming limits, for a cheap early decode on unrefined corners. Optionally reports each
 * side's border level (side_level) and the tag's black and white levels. */
static int decode_quad(napa_detector *det, const image_view *im, vec2 q[4],
                       const napa_params *params, int slack, napa_detection *out, double side_level[4],
                       edge_levels *levels) {
    const family_layout *L = &det->fam;
    const int W = L->width, o = L->outer, G = L->grid;
    if (signed_area(q) < 0) {
        vec2 t = q[1];
        q[1] = q[3];
        q[3] = t;
    }
    double H[9];
    square_to_quad(q, H);
    double grid[256];
    double border[64];
    double black, white;
    /* Border cells and the quiet-zone ring first: most candidates are rejected on their contrast
     * or border before the payload cells are worth sampling. */
    int dark = read_border(L, im, H, params->min_contrast, grid, border, &black, &white);
    if (dark < 0) return 0;
    if (levels != NULL) levels->black = black, levels->white = white;
    double thr = 0.5 * (black + white);
    if (dark < min_border(L, params->min_border_cells) - slack) return 0;
    /* Every other cell of the grid: the payload, and for families with data outside the quiet ring,
     * that ring and the data around it. */
    for (int r = 1 - o; r < W - 1 + o; r++) {
        for (int c = 1 - o; c < W - 1 + o; c++) {
            if (o > 0 && r >= 0 && r < W && c >= 0 && c < W && (r == 0 || r == W - 1 || c == 0 || c == W - 1))
                continue; /* a border cell, already read */
            vec2 p = project(H, (c + 0.5) / L->cells, (r + 0.5) / L->cells);
            grid[(r + o) * G + c + o] = sample_bilinear(im, p.x, p.y);
        }
    }
    if (o > 0) {
        /* The grid's outermost ring, outside the quiet ring. */
        for (int r = -o; r < W + o; r++) {
            for (int c = -o; c < W + o; c += (r == -o || r == W + o - 1) ? 1 : W + 2 * o - 1) {
                vec2 p = project(H, (c + 0.5) / L->cells, (r + 0.5) / L->cells);
                grid[(r + o) * G + c + o] = sample_bilinear(im, p.x, p.y);
            }
        }
    }
    if (side_level != NULL) {
        /* Each side's non-corner border cells (top, right, bottom, left, unsharpened), as a fraction of
         * the way from the tag's black to its white: about 0 for a true border side. */
        for (int k = 0; k < 4; k++) side_level[k] = 0;
        for (int k = 1; k < W - 1; k++) {
            side_level[0] += grid[o * G + k + o];
            side_level[1] += grid[(k + o) * G + W - 1 + o];
            side_level[2] += grid[(W - 1 + o) * G + k + o];
            side_level[3] += grid[(k + o) * G + o];
        }
        for (int k = 0; k < 4; k++) side_level[k] = (side_level[k] / (W - 2) - black) / (white - black);
    }
    /* Sharpen the cell samples, as upstream AprilTag does, to undo blur between neighbouring cells. */
    if (params->sharpening_x100 > 0) {
        double k = params->sharpening_x100 / 100.0;
        double sharp[256];
        for (int r = 0; r < G; r++) {
            for (int c = 0; c < G; c++) {
                double v = grid[r * G + c], lap = 4 * v;
                lap -= r > 0 ? grid[(r - 1) * G + c] : v;
                lap -= r < G - 1 ? grid[(r + 1) * G + c] : v;
                lap -= c > 0 ? grid[r * G + c - 1] : v;
                lap -= c < G - 1 ? grid[r * G + c + 1] : v;
                sharp[r * G + c] = v + k * lap;
            }
        }
        memcpy(grid, sharp, sizeof(double) * (size_t)(G * G));
    }

    uint64_t code = 0;
    for (int k = 0; k < L->nbits; k++) code = (code << 1) | (grid[L->cell[k]] > thr ? 1u : 0u);
    int max_hamming = params->max_hamming < L->max_hamming ? params->max_hamming : L->max_hamming;
    int soft = L->soft_hamming > max_hamming ? L->soft_hamming : max_hamming;
    int best = 64;
    int bi = nearest_code(det, code, (slack ? soft : max_hamming) + slack, &best);
    if (bi < 0 && slack == 0 && soft > max_hamming) {
        /* Soft decision, for families whose codes are too close for more hard corrections: accept up to
         * `soft` wrong bits only if they are exactly the least certain readings (nearest the threshold),
         * as blur and compression make them. A random pattern's wrong bits are that set by chance only
         * once in C(n, d), so each extra bit adds about the chance-match rate of exact decoding. */
        bi = nearest_code(det, code, soft, &best);
        if (bi >= 0) {
            uint64_t wrong = code ^ L->codes[bi];
            double worst_wrong = 0, best_right = 1e300;
            for (int k = 0; k < L->nbits; k++) {
                double certainty = fabs(grid[L->cell[k]] - thr);
                if (wrong >> (L->nbits - 1 - k) & 1) worst_wrong = certainty > worst_wrong ? certainty : worst_wrong;
                else best_right = certainty < best_right ? certainty : best_right;
            }
            if (worst_wrong >= best_right) bi = -1;
        }
    }
    if (bi < 0) return 0;
    /* Three corrected bits are accepted only when the wrong cells touch (8-connected): real damage,
     * a finger, cable, or glare spot, is spatially coherent, while a random pattern that happens to
     * fall within three bits of a code has its errors scattered: for tag36h11, only 420 of the 7,140
     * three-cell placements are connected, so this rejects ~94% of chance three-bit matches. */
    if (slack == 0 && best == 3 && !errors_connected(L, code ^ L->codes[bi])) return 0;
    int rot = L->rotations[bi];
    /* Observed grid == rot90(canonical, rot); the canonical top-left sits at corner index (4 - rot) % 4. */
    for (int k = 0; k < 4; k++) {
        vec2 c = q[(k - rot + 4) & 3];
        out->corners[2 * k] = c.x;
        out->corners[2 * k + 1] = c.y;
    }
    out->id = L->ids[bi];
    out->hamming = best;
    /* As upstream AprilTag: the smaller of the white and black payload bits' mean distance from the
     * threshold, in grey levels. Low values mean bits were nearly ambiguous. */
    double white_sum = 0, black_sum = 0;
    int white_n = 0, black_n = 0;
    for (int k = 0; k < L->nbits; k++) {
        double v = grid[L->cell[k]] - thr;
        if (v > 0) white_sum += v, white_n++;
        else black_sum -= v, black_n++;
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

/* The component holding piece r, once split_pieces has joined pieces: a joined piece's count holds
 * -(parent + 1), and parents are earlier pieces, so the root is the component's first. */
static int32_t joined_root(blob_box *B, int32_t r) {
    while (B[r].count < 0) {
        int32_t p = -B[r].count - 1;
        if (B[p].count < 0) B[r].count = B[p].count; /* path halving */
        r = p;
    }
    return r;
}

static int grow_pieces(napa_detector *det, size_t n) {
    if (n <= det->piece_capacity) return 0;
    size_t cap = det->piece_capacity ? 2 * det->piece_capacity : 64;
    void *p;
    if ((p = realloc(det->pieces, cap * sizeof(component))) == NULL) return -1;
    det->pieces = (component *)p;
    if ((p = realloc(det->piece_root, cap * sizeof(int32_t))) == NULL) return -1;
    det->piece_root = (int32_t *)p;
    if ((p = realloc(det->piece_cand, cap * sizeof(int32_t))) == NULL) return -1;
    det->piece_cand = (int32_t *)p;
    if ((p = realloc(det->piece_next, cap * sizeof(int32_t))) == NULL) return -1;
    det->piece_next = (int32_t *)p;
    det->piece_capacity = cap;
    return 0;
}

/* Propose component r as piece n if it passes the candidate size and shape test on its own, the first time it
 * is seen this stage (a negated count marks it seen). Returns the new piece count, or -1 when out of memory. */
static int propose_piece(napa_detector *det, int32_t r, int n, int lo, int hi, int dense) {
    blob_box *B = det->boxes;
    if (B[r].count < 0) return n;
    B[r].count = -B[r].count;
    int w = B[r].maxx - B[r].minx + 1, h = B[r].maxy - B[r].miny + 1;
    int small = w < h ? w : h, big = w > h ? w : h;
    if (small < lo || big > hi || big > 8 * small || (dense && small > 72) || (double)-B[r].count < 0.2 * (double)w * h)
        return n;
    if (grow_pieces(det, (size_t)n + 1) != 0) return -1;
    component *c = &det->pieces[n];
    c->count = 0;
    for (int k = 0; k < 16; k++) c->extv[k] = INT32_MIN;
    c->row_y = -1;
    c->minx = B[r].minx, c->maxx = B[r].maxx, c->miny = B[r].miny, c->maxy = B[r].maxy;
    det->piece_root[n] = r;
    return n + 1;
}

/* One stage of split_pieces: E holds *n_pairs run pairs. Rewrite them as pairs of the components they
 * join so far (dropping pairs already joined), propose each such component that passes the candidate
 * size and shape test on its own, then join the pairs. Returns the new piece count, or -1. */
static int join_pairs(napa_detector *det, int32_t *E, size_t *n_pairs, int n, int lo, int hi, int dense) {
    blob_box *B = det->boxes;
    const int32_t *P = det->parent;
    size_t ne = 0;
    /* Resolve every pair before marking any component: a mark would read as a join. */
    for (size_t e = 0; e < *n_pairs; e += 2) {
        int32_t ra = joined_root(B, -P[E[e]] - 1), rb = joined_root(B, -P[E[e + 1]] - 1);
        if (ra == rb) continue; /* joined another way */
        E[ne++] = ra;
        E[ne++] = rb;
    }
    for (size_t e = 0; e < ne; e++)
        if ((n = propose_piece(det, E[e], n, lo, hi, dense)) < 0) return -1;
    *n_pairs = ne;
    for (size_t e = 0; e < ne; e++)
        if (B[E[e]].count < 0) B[E[e]].count = -B[E[e]].count;
    /* The later component merges into the earlier one. */
    for (size_t e = 0; e < ne; e += 2) {
        int32_t ra = joined_root(B, E[e]), rb = joined_root(B, E[e + 1]);
        if (ra == rb) continue;
        int32_t root = ra < rb ? ra : rb, child = ra < rb ? rb : ra;
        blob_box *t = &B[root], *c = &B[child];
        t->count += c->count;
        if (c->minx < t->minx) t->minx = c->minx;
        if (c->miny < t->miny) t->miny = c->miny;
        if (c->maxx > t->maxx) t->maxx = c->maxx;
        if (c->maxy > t->maxy) t->maxy = c->maxy;
        c->count = -root - 1;
    }
    return n;
}

/* split_touching: the level was labelled with necks and corner contacts left unjoined. A tag whose border
 * touches another dark shape only at a corner (an AprilGrid's corner squares, a checkerboard) is one
 * eight-connected blob with it, which isn't tag-shaped; anti-aliasing often leaves a one-sample neck at
 * such a corner instead. Join the necks, then the corners, proposing the pieces on either side of each
 * before it is joined: the finest pieces separate tags at necks, the next keep a tag whole where its own
 * thin border has necks. Afterwards the boxes are the ordinary blobs again. Returns the number of pieces,
 * neck-stage pieces first (*fine of them), or -1 when out of memory. */
static int split_pieces(napa_detector *det, int lo, int hi, int dense, int *fine) {
    int n = join_pairs(det, det->neck, &det->neck_n, 0, lo, hi, dense);
    if (n < 0) return -1;
    *fine = n;
    /* Each neck pair's components' roots now, before corner joins (and path halving) pass them by. */
    for (size_t e = 0; e < det->neck_n; e++) det->neck_root[e] = joined_root(det->boxes, det->neck[e]);
    return join_pairs(det, det->diag, &det->diag_n, n, lo, hi, dense);
}

/* Whether a found tag touches dark shapes at its corners, as grid tags do: just outside at least two corners,
 * diagonally, the image is darker than halfway between the tag's border there and its quiet zone. Families with
 * data cells outside the border have them where these samples fall, so their boards may not be recognised. */
static int corners_touch(const image_view *view, const napa_detection *d, double cells) {
    const double *c = d->corners;
    int dark = 0;
    for (int k = 0; k < 4; k++) {
        int a = (k + 3) & 3, b = (k + 1) & 3, n = (k + 2) & 3;
        /* One cell along each side, away from the corner's two neighbours. */
        double ux = (c[2 * k] - c[2 * b]) / cells, uy = (c[2 * k + 1] - c[2 * b + 1]) / cells;
        double vx = (c[2 * k] - c[2 * a]) / cells, vy = (c[2 * k + 1] - c[2 * a + 1]) / cells;
        double outside = sample_bilinear(view, c[2 * k] + 0.6 * (ux + vx), c[2 * k + 1] + 0.6 * (uy + vy));
        double border = sample_bilinear(view, c[2 * k] - 0.5 * (ux + vx), c[2 * k + 1] - 0.5 * (uy + vy));
        /* The quiet zone beside the side from this corner to the next. */
        double qx = 0.5 * (c[2 * k] + c[2 * b]) + 0.6 * 0.5 * (vx + (c[2 * b] - c[2 * n]) / cells);
        double qy = 0.5 * (c[2 * k + 1] + c[2 * b + 1]) + 0.6 * 0.5 * (vy + (c[2 * b + 1] - c[2 * n + 1]) / cells);
        dark += outside < 0.5 * (border + sample_bilinear(view, qx, qy));
    }
    return dark >= 2;
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
 * keeps noise from forming dark networks. The caller supplies the cap for this grid. */
static int floor_from(const uint32_t hist[64], uint32_t n, int min_contrast, double sigmas, int cap) {
    uint32_t seen = 0;
    int median = 0;
    while (median < 63 && (seen += hist[median]) < (n + 1) / 2) median++;
    int floor = (int)(sigmas * median / 0.954 + 0.5);
    if (floor > cap) floor = cap;
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
static int contrast_floors(napa_detector *det, const uint8_t *L, int lw, int lh, int min_contrast, double sigmas, int cap) {
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
            det->floors[ry * cols + rx] = (uint8_t)(n >= 16 ? floor_from(hist, n, min_contrast, sigmas, cap) : 0);
        }
    }
    int frame = n_all ? floor_from(all, n_all, min_contrast, sigmas, cap) : min_contrast;
    det->floor_uniform = 1;
    for (int i = 0; i < cols * rows; i++) {
        /* Never above the frame's: a region only lowers its floor, where it is cleaner than the frame
         * (a region's estimate, from fewer samples, also varies more). */
        if (det->floors[i] == 0 || det->floors[i] > frame) det->floors[i] = (uint8_t)frame;
        det->floor_uniform &= det->floors[i] == det->floors[0];
    }
    return frame;
}

static void invert_bytes(uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(255 - p[i]);
}

/* Remember a failed candidate's fitted quad. Best effort: without memory, nothing is remembered. */
static void remember_failure(napa_detector *det, const vec2 q[4], int s) {
    if (det->failed_n == det->failed_capacity) {
        size_t cap = det->failed_capacity ? 2 * det->failed_capacity : 64;
        failed_quad *p = (failed_quad *)realloc(det->failed, cap * sizeof(failed_quad));
        if (p == NULL) return;
        det->failed = p, det->failed_capacity = cap;
    }
    failed_quad *f = &det->failed[det->failed_n++];
    for (int k = 0; k < 4; k++) f->xy[2 * k] = (float)q[k].x, f->xy[2 * k + 1] = (float)q[k].y;
    f->s = s;
}

/* Whether a finer level already rejected this quad: every corner of q within tol pixels of a failed
 * quad's corner, in any cyclic order of the same winding. */
static int failed_before(const napa_detector *det, const vec2 q[4], int s, double tol) {
    for (size_t i = 0; i < det->failed_n; i++) {
        const failed_quad *f = &det->failed[i];
        if (f->s >= s) continue;
        for (int shift = 0; shift < 4; shift++) {
            int k = 0;
            for (; k < 4; k++) {
                int j = (k + shift) & 3;
                if (fabs(q[k].x - f->xy[2 * j]) > tol || fabs(q[k].y - f->xy[2 * j + 1]) > tol) break;
            }
            if (k == 4) return 1;
        }
    }
    return 0;
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
    if (params->split_touching != 0 && params->split_touching != 1) return -1;
    if (width < 16 || height < 16) return 0;
    const int split = params->split_touching && det->fam.piece_hamming >= 0;
    const int W = det->fam.width;
    const double cells = det->fam.cells;
    /* Reversed families print the square light in a dark frame: read inverted, they look like every other
     * family, a dark square in a light quiet ring. The raked levels are inverted as they're made, and
     * full-resolution reads sample by sample (image_view). */
    const image_view view = {img, width, height, stride, det->fam.f->reversed};
    /* A tag cell is min_tag_px / W wide, but blur turns its edges grey, so only its core is
     * reliably dark. Sampling every 3/4 cell guarantees a sample inside that core. */
    int base = params->min_tag_px * 3 / (4 * W);
    if (base < 1) base = 1;
    int found = 0, piece_found = 0;
    /* split_touching splits at corners only until it finds a tag that touches dark shapes at its corners:
     * one-sample necks (anti-aliased corners) cost far more and only pay where tags do touch. Then the frame
     * starts over splitting necks too, and the levels already done (to redone) decode only their new pieces. */
    int necks = 0, redone = 0, touching = 0;
    size_t fates_kept = 0; /* fates written before the restart */
    det->failed_n = 0;
    /* Size band in level samples. Adjacent levels overlap (7-24 here is 14-48 one level finer),
     * because a blob's measured extent varies by a sample or two with phase, blur and rotation. */
    const int lo = (7 * W + 4) / 8, hi = NAPA_HI * W / 8;

start:;
    int contrast = params->min_contrast;
    size_t fate_off = 0; /* where this level's fates are kept in det->fates */
    int prev_lw = 0, floor_stride = base;
    for (int s = base;; s *= 2) {
        int lw = (width + s - 1) / s;
        int lh = (height + s - 1) / s;
        int last = (lw < NAPA_LAST * W / 8) || (lh < NAPA_LAST * W / 8);
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
            if (view.invert) invert_bytes(det->level, pixels);
        } else {
            for (int y = 0; y < lh; y++) gather2(det->level + (size_t)y * lw, det->level + (size_t)(2 * y) * prev_lw, lw);
        }
        prev_lw = lw;
        PROF_MARK(PROF_RAKE);
        if (s == base || (base == 1 && s == 2 && contrast > NAPA_NOISY_FLOOR)) {
            /* Do not blur stride-one proposals: a tiny border can be just one pixel wide. Its
             * higher floor suppresses raw noise; the next averaged level restores low contrast.
             * A terminal level has no recovery pass, so keep its original contrast floor. */
            floor_stride = s;
            double sigmas = s == 1 && !last ? 7.0 : NAPA_FLOOR_SIGMAS;
            int cap = s == 1 && !last ? 255 : NAPA_FLOOR_CAP;
            contrast = contrast_floors(det, det->level, lw, lh, params->min_contrast, sigmas, cap);
            /* A noisy frame: rake again, averaging each sample's block, and re-measure the noise. */
            if (contrast > NAPA_NOISY_FLOOR && s > 1 && s <= 16) {
                if ((size_t)width > det->box_capacity) {
                    uint16_t *p = (uint16_t *)realloc(det->box_rows, sizeof(uint16_t) * 2 * (size_t)width);
                    if (p == NULL) return -2;
                    det->box_rows = p;
                    det->box_capacity = (size_t)width;
                }
                box_rake(det->level, lw, lh, img, width, height, stride, s, base == 1 && s == 2 ? 3 : s, det->box_rows, det->box_rows + width);
                if (view.invert) invert_bytes(det->level, pixels);
                /* Stride-two 3x3 means share one column: their difference measures sqrt(2/3)
                 * times their noise sigma. Disjoint averaging blocks need no correction. */
                double sigmas = base == 1 && s == 2 ? NAPA_BOX_SIGMAS * sqrt(1.5) : NAPA_BOX_SIGMAS;
                contrast = contrast_floors(det, det->level, lw, lh, params->min_contrast, sigmas, NAPA_FLOOR_CAP);
            }
            if (contrast < 0) return -2;
        }
        const int redo = s <= redone;
        int nruns = threshold_level(det, lw, lh, s / floor_stride, split ? 1 + necks : 0);
        if (nruns < 0) return -2;
        PROF_MARK(PROF_THRESHOLD);

        /* Resolve runs into packed components, preserving root raster order. */
        napa_run *R = det->runs;
        int32_t *P = det->parent;
        blob_box *B = det->boxes;
        int32_t nroots = 0;
        for (int i = 0; i < nruns; i++) {
            int32_t r;
            if (P[i] == i) {
                r = nroots++;
                B[r].count = R[i].x1 - R[i].x0 + 1;
                B[r].minx = R[i].x0, B[r].maxx = R[i].x1;
                B[r].miny = B[r].maxy = R[i].y;
            } else {
                /* Union always points to an earlier run. That run already carries its dense
                 * component index, encoded negatively to distinguish it from a run index. */
                r = -P[P[i]] - 1;
                B[r].count += R[i].x1 - R[i].x0 + 1;
                B[r].minx = R[i].x0 < B[r].minx ? R[i].x0 : B[r].minx;
                B[r].maxx = R[i].x1 > B[r].maxx ? R[i].x1 : B[r].maxx;
                B[r].maxy = R[i].y;
            }
            P[i] = -r - 1;
        }
        int npieces = 0, fine = 0;
        if (det->diag_n || det->neck_n) {
            npieces = split_pieces(det, lo, last ? 1 << 30 : hi, !last && base == 1 && s <= 2, &fine);
            if (npieces < 0) return -2;
        }
        /* Keep tag-sized, tag-shaped blobs as candidates, in raster order of their first sample. A tag
         * tilted by t looks about 1 / cos(t) times longer than it is wide, more with perspective: an 8:1
         * box admits tilts to about 80 degrees. Clean tags stay readable past 75 degrees whenever their
         * narrow side is still min_tag_px or more, and decoding rejects thin blobs that aren't tags. */
        int min_size = lo;
        /* Dense levels propose small tags only. Larger tags have wider cells and are cheaper to
         * propose on the next level; the overlapping bands retain steeply tilted small tags. */
        int max_size = last ? 1 << 30 : hi;
        int ncand = 0;
        for (int i = 0; i < nroots; i++) {
            det->slot[i] = -1;
            if (B[i].count < 0) continue; /* a piece joined into an earlier one */
            int t_minx = B[i].minx, t_maxx = B[i].maxx, t_miny = B[i].miny, t_maxy = B[i].maxy;
            int bw = t_maxx - t_minx + 1, bh = t_maxy - t_miny + 1;
            int big = bw > bh ? bw : bh, small = bw < bh ? bw : bh;
            if (!last && base == 1 && s <= 2 && small > 72) continue;
            if (small < min_size || big > max_size || big > 8 * small) {
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
        if (det->diag_n || det->neck_n) {
            /* Runs carry their piece. A joined piece takes the slot of what joined it: its blob's
             * candidate, or the corner-stage piece holding it. A proposed piece takes ncand + its index,
             * and passes its runs on to that slot next. Blobs' candidates are read before any changes. */
            for (int k = 0; k < npieces; k++) det->piece_cand[k] = det->slot[joined_root(B, det->piece_root[k])];
            for (size_t e = 0; e < det->diag_n; e++) {
                int32_t r = det->diag[e];
                if (B[r].count < 0) det->slot[r] = det->slot[joined_root(B, r)];
            }
            /* At stride one, pieces of candidate blobs are never decoded (below): their runs skip them. */
            for (int k = fine; k < npieces; k++) {
                det->piece_next[k] = det->slot[det->piece_root[k]];
                if (s > 1 || det->piece_cand[k] < 0) det->slot[det->piece_root[k]] = ncand + k;
            }
            for (size_t e = 0; e < det->neck_n; e++)
                if (det->neck_root[e] != det->neck[e]) det->slot[det->neck[e]] = det->slot[det->neck_root[e]];
            for (int k = 0; k < fine; k++) {
                det->piece_next[k] = det->slot[det->piece_root[k]];
                if (s > 1 || det->piece_cand[k] < 0) det->slot[det->piece_root[k]] = ncand + k;
            }
        }
        if (npieces == 0) {
            for (int i = 0; i < nruns; i++) {
                int32_t c = det->slot[-P[i] - 1];
                if (c >= 0) comp_add_run(&det->cands[c], &R[i]);
            }
        } else {
            for (int i = 0; i < nruns; i++) {
                int32_t c = det->slot[-P[i] - 1];
                for (; c >= ncand; c = det->piece_next[c - ncand]) comp_add_run(&det->pieces[c - ncand], &R[i]);
                if (c >= 0) comp_add_run(&det->cands[c], &R[i]);
            }
            for (int k = 0; k < npieces; k++)
                if (det->pieces[k].row_y >= 0) comp_flush_row(&det->pieces[k]); /* not: pieces given no runs */
        }
        for (int ci = 0; ci < ncand; ci++) comp_flush_row(&det->cands[ci]);
        if (npieces) {
            /* row_x1 is spent after the flush: it counts each candidate blob's proposed pieces. */
            for (int ci = 0; ci < ncand; ci++) det->cands[ci].row_x1 = 0;
            for (int k = 0; k < npieces; k++)
                if (det->piece_cand[k] >= 0) det->cands[det->piece_cand[k]].row_x1++;
        }
        if (redo && fate_off + (size_t)ncand + (size_t)(npieces - fine) <= fates_kept) {
            /* The first pass's fates: the corner-stage pieces are the same pieces, in the same order. */
            for (int ci = 0; ci < ncand; ci++) det->cands[ci].row_y = -det->fates[fate_off + ci];
            for (int k = fine; k < npieces; k++) det->pieces[k].row_y = -det->fates[fate_off + ncand + k - fine];
        }
        PROF_MARK(PROF_BLOBS);

        for (int ci = 0; ci < ncand + npieces; ci++) {
            const int piece = ci >= ncand;
            /* Corner-stage pieces first: a tag kept whole there is found as without necks. */
            int k = ci - ncand < npieces - fine ? fine + ci - ncand : ci - ncand - (npieces - fine);
            /* Redone levels decode only the pieces not tried before (whose fate reads -1, below). */
            if (redo && (!piece || (k >= fine && det->pieces[k].row_y != -1))) continue;
            component *self = piece ? &det->pieces[k] : &det->cands[ci];
            component comp = *self;
            /* row_y is spent after the extrema pass: it records how far each proposal got (-TRACE stage). */
            self->row_y = -5;
            if (piece) {
                /* A blob or larger piece that failed its rough quad or its refinement, or decoded, never
                 * held another tag in its pieces; at stride one, nor did any candidate blob (only tiny
                 * tags in large blobs are new there). */
                int next = det->piece_next[k];
                const component *whole = next >= ncand ? &det->pieces[next - ncand]
                                         : det->piece_cand[k] >= 0 ? &det->cands[det->piece_cand[k]] : NULL;
                int fate = whole ? -whole->row_y : 0;
                if (det->piece_cand[k] >= 0 && s == 1) continue;
                /* Until tags are seen touching, the lone piece of a blob that failed its first fit waits:
                 * outside boards, whose blobs hold many pieces, it never held a tag. Untried, it is decoded
                 * if the level is redone. */
                if (!necks && det->piece_cand[k] >= 0 && fate == 7 && det->cands[det->piece_cand[k]].row_x1 == 1) {
                    self->row_y = -1;
                    continue;
                }
                if (fate == 6 || fate >= 9) {
                    self->row_y = whole->row_y; /* its own pieces share the fate */
                    continue;
                }
            }
            int t_minx = comp.minx, t_maxx = comp.maxx, t_miny = comp.miny, t_maxy = comp.maxy;
            int bw = t_maxx - t_minx + 1, bh = t_maxy - t_miny + 1;
            int big = bw > bh ? bw : bh;
            /* Levels run fine to coarse; skip a blob whose centre a finer level already decoded. */
            double ccx = 0.5 * (comp.minx + comp.maxx) * s, ccy = 0.5 * (comp.miny + comp.maxy) * s;
            if (covered(out, found < max_out ? found : max_out, ccx, ccy, 0.3 * big * s + 2.0) ||
                (piece && covered(det->piece_out, piece_found, ccx, ccy, 0.3 * big * s + 2.0))) {
                TRACE(5);
                self->row_y = -5;
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
            if (comp.count > det->fam.fill_limit * (best_area + 0.5 * perim_q + 1.0)) {
                PROF_MARK(PROF_QUAD);
                TRACE(6);
                self->row_y = -6;
                continue;
            }
            if (best_area < 0.3 * bw * bh) {
                PROF_MARK(PROF_QUAD);
                TRACE(6);
                self->row_y = -6;
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
            /* Sparse paired border samples reject noise before the expensive edge profiles.
             * Permit one weak side and leave terminal proposals unscreened: their ring can have
             * local clutter that a later level would normally resolve. Final decode is unchanged. */
            if ((piece || (s <= 2 && !last)) && params->min_border_cells >= 22) {
                vec2 q[4];
                double H[9];
                grow_quad(rough, 0.5 * s, q);
                square_to_quad(q, H);
                int good_sides = 0;
                double half_cell = 0.5 / cells;
                for (int side = 0; side < 4; side++) {
                    double contrast_sum = 0;
                    for (int j = 0; j < 2; j++) {
                        double along = 0.25 + 0.5 * j;
                        double u = side == 1 ? 1.0 : side == 3 ? 0.0 : along;
                        double v = side == 0 ? 0.0 : side == 2 ? 1.0 : along;
                        double du = side == 1 ? half_cell : side == 3 ? -half_cell : 0;
                        double dv = side == 2 ? half_cell : side == 0 ? -half_cell : 0;
                        vec2 white = project(H, u + du, v + dv), black = project(H, u - du, v - dv);
                        contrast_sum += sample_bilinear(&view, white.x, white.y) - sample_bilinear(&view, black.x, black.y);
                    }
                    good_sides += contrast_sum >= params->min_contrast * 0.25;
                }
                if (good_sides < 3) {
                    PROF_MARK(PROF_FIT1);
                    TRACE(7);
                    self->row_y = -7;
                    continue;
                }
            }
            double reach = 0.5 * ((1.5 * s + 1.0) + (0.5 * s + 1.0));
            double edge_w = 0;
            if (!fit_sides(&view, rough, fine, 1.5 * s + 1.0, 0.5 * s + 1.0, params->min_contrast, 1, 6, &edge_w, NULL, cells)) {
                PROF_MARK(PROF_FIT1);
                TRACE(7);
                self->row_y = -7;
                continue;
            }
            /* The first fit is already within a fraction of a cell: decode now, with two bits of
             * slack, and spend the refinement passes only on quads that are tags. A random quad
             * passes this only about 1.5% of the time. */
            PROF_MARK(PROF_FIT1);
            /* Levels overlap, so a blob that failed to decode at a finer level often comes back at this one.
             * The early decode depends only on the fitted quad: if it's within 0.3 px of a failed one, it
             * fails again. (The rough quads can match while the fits differ: the window grows with s.) */
            if (failed_before(det, fine, s, 0.3)) {
                PROF_MARK(PROF_DECODE1);
                TRACE(8);
                self->row_y = -8;
                continue;
            }
            vec2 probe[4];
            napa_detection early;
            edge_levels lv = {255, 0};
            memcpy(probe, fine, sizeof probe);
            if (!decode_quad(det, &view, probe, params, 2, &early, NULL, &lv)) {
                /* The blob can include the white quiet zone when that margin is darker than the
                 * background around it (a grey card on a bright wall): then this quad is the margin's
                 * outline and the black square is its inner 8/10. Refit there and try once more. */
                int rescued = 0;
                double Hq[9], perim = 0;
                square_to_quad(fine, Hq);
                double m0 = 1.0 / (W + 2), m1 = (W + 1.0) / (W + 2);
                vec2 inner[4] = {project(Hq, m0, m0), project(Hq, m1, m0), project(Hq, m1, m1), project(Hq, m0, m1)};
                for (int k = 0; k < 4; k++) perim += hypot(fine[(k + 1) & 3].x - fine[k].x, fine[(k + 1) & 3].y - fine[k].y);
                double r = 0.45 * perim / (4.0 * (W + 2)); /* 0.45 of a cell, if the quad spans W + 2 cells */
                /* Most failed decodes aren't tags at all. Refit only if the unfitted inner square already
                 * shows a dark border inside a lighter margin, with loose limits (half the contrast, six
                 * more light border cells): this skips 86% of retries and kept every rescue tested. */
                if (r >= 1.5 &&
                    border_plausible(&det->fam, &view, inner, params->min_contrast / 2,
                                     min_border(&det->fam, params->min_border_cells) - 6) &&
                    fit_sides(&view, inner, finer, r, r, params->min_contrast, 0, 8, &edge_w, NULL, cells)) {
                    memcpy(probe, finer, sizeof probe);
                    if (decode_quad(det, &view, probe, params, 2, &early, NULL, &lv)) {
                        memcpy(fine, finer, sizeof fine);
                        reach = r;
                        rescued = 1;
                    }
                }
                if (!rescued) {
                    PROF_MARK(PROF_DECODE1);
                    TRACE(8);
                    self->row_y = -8;
                    /* Pieces only read the list, so the other detections can't depend on them. */
                    if (!piece) remember_failure(det, fine, s);
                    continue;
                }
            }
            PROF_MARK(PROF_DECODE1);
            double cell = 0;
            for (int k = 0; k < 4; k++) {
                double ex = fine[(k + 1) & 3].x - fine[k].x, ey = fine[(k + 1) & 3].y - fine[k].y;
                cell += sqrt(ex * ex + ey * ey) / (4.0 * W);
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
                ok = fit_sides(&view, fine, finer, reach, reach, params->min_contrast, 0, 8, &edge_w, &lv, cells);
                if (!ok) break;
                memcpy(fine, finer, sizeof fine);
            }
            /* One more pass at the converged window, re-centred on the last estimate, with every
             * edge sample. */
            if (ok) ok = fit_sides(&view, fine, finer, reach, reach, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv, cells);
            if (!ok) {
                PROF_MARK(PROF_REFINE);
                TRACE(9);
                self->row_y = -9;
                continue;
            }
            PROF_MARK(PROF_REFINE);
            napa_detection d;
            double side_level[4];
            edge_levels final_lv = {255, 0};
            int decoded_ = decode_quad(det, &view, finer, params, 0, &d, side_level, &final_lv);
            PROF_MARK(PROF_DECODE2);
            if (!decoded_) {
                TRACE(10);
                self->row_y = -10;
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
                double one = 1.0 / W, rest = (W - 1.0) / W;
                double u0 = (bad & 8) ? one : 0, v0 = (bad & 1) ? one : 0;
                double u1 = (bad & 2) ? rest : 1, v1 = (bad & 4) ? rest : 1;
                vec2 moved[4] = {project(Hs, u0, v0), project(Hs, u1, v0), project(Hs, u1, v1), project(Hs, u0, v1)};
                for (int k = 0; k < 4; k++) perim += hypot(moved[(k + 1) & 3].x - moved[k].x, moved[(k + 1) & 3].y - moved[k].y);
                double r = 0.45 * perim / (4.0 * W);
                vec2 fixed[4];
                napa_detection d2;
                double side2[4], level_after = 0;
                /* The moved side starts tilted (it came from a skewed quad): fit, then refit once re-centred
                 * in a window just wide enough for the edge, as the main refinement converges. */
                double r2 = 1.25 * edge_w > 1.5 ? 1.25 * edge_w : 1.5;
                vec2 first[4];
                int refit = r >= 1.0 && fit_sides(&view, moved, first, r, r, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv, cells);
                if (refit && r2 < r) refit = fit_sides(&view, first, fixed, r2, r2, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv, cells);
                else if (refit) memcpy(fixed, first, sizeof fixed);
                if (refit && decode_quad(det, &view, fixed, params, 0, &d2, side2, NULL) && d2.id == d.id) {
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
                if (border_bias(&det->fam, &view, finer, final_lv.black, final_lv.white, &delta) >= 4 &&
                    fabs(delta) > 0.02 && fabs(delta) < 1.5) {
                    vec2 grown[4];
                    napa_detection d3;
                    grow_quad(finer, delta, grown);
                    if (decode_quad(det, &view, grown, params, 0, &d3, NULL, NULL) && d3.id == d.id) d = d3;
                }
            }
            if (piece) {
                /* Pieces are many and mostly not tags: correct fewer bits (in tag36h11, three admitted
                 * chance matches from random-payload grids), and keep them apart until the end. */
                self->row_y = -12;
                if (d.hamming > det->fam.piece_hamming || already_found(det->piece_out, piece_found, &d)) continue;
                if ((size_t)piece_found == det->piece_out_capacity) {
                    size_t cap = det->piece_out_capacity ? 2 * det->piece_out_capacity : 64;
                    napa_detection *p = (napa_detection *)realloc(det->piece_out, cap * sizeof(napa_detection));
                    if (p == NULL) return -2;
                    det->piece_out = p;
                    det->piece_out_capacity = cap;
                }
                if (!necks && !touching) touching = corners_touch(&view, &d, cells);
                det->piece_out[piece_found++] = d;
                continue;
            }
            if (already_found(out, found < max_out ? found : max_out, &d)) {
                TRACE(11);
                self->row_y = -11;
                continue;
            }
            TRACE(12);
            self->row_y = -12;
            if (split && !necks && !touching) touching = corners_touch(&view, &d, cells);
            if (found < max_out) out[found] = d;
            found++;
        }
        PROF_MARK(PROF_BLOBS);
        if (split && !necks) {
            /* Keep this level's fates, in case the frame is redone with necks. */
            size_t n = (size_t)ncand + (size_t)npieces;
            if (fate_off + n > det->fate_capacity) {
                size_t cap = det->fate_capacity ? 2 * det->fate_capacity : 4096;
                while (cap < fate_off + n) cap *= 2;
                uint8_t *f = (uint8_t *)realloc(det->fates, cap);
                if (f == NULL) return -2;
                det->fates = f;
                det->fate_capacity = cap;
            }
            for (int ci = 0; ci < ncand; ci++) det->fates[fate_off + ci] = (uint8_t)-det->cands[ci].row_y;
            for (int k = 0; k < npieces; k++) det->fates[fate_off + ncand + k] = (uint8_t)-det->pieces[k].row_y;
            fates_kept = fate_off + n;
        }
        fate_off += (size_t)ncand + (size_t)(npieces - fine);
        if (split && !necks && touching) {
            necks = 1;
            redone = s;
            goto start;
        }
        if (last) break;
    }
    /* Pieces come last, so the other detections are unchanged by split_touching. */
    for (int k = 0; k < piece_found && found < max_out; k++)
        if (!already_found(out, found, &det->piece_out[k])) out[found++] = det->piece_out[k];
    return found < max_out ? found : max_out;
}

/* setuptools links every extension with a module-init export on Windows. The library is loaded
 * with ctypes, never imported, so this placeholder only satisfies the linker. */
NAPA_API void *PyInit__napa_core(void) { return 0; }
