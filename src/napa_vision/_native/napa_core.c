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
    /* Spent once the extrema are flushed; napa_detect then reuses row_y for how far the blob got and
     * row_x1 for how many split pieces it has. */
} component;

typedef struct fine_whole fine_whole;

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

typedef struct {
    double x, y;
} vec2;

/* A decode attempt worth a second look by the soft decoder. `order` keeps the first of exact duplicates. */
typedef struct {
    float quad[8], black, white, grid[64];
    int order;
} soft_candidate;

typedef struct soft_tables soft_tables;

struct napa_detector {
    family_layout fam;
    int tag_shape;        /* NAPA_TAG_SHAPE_*: the ordinary pass samples twice as far apart down (tall) or across (wide) */
    uint8_t *shrunk;      /* the frame halved that way, area-averaged */
    size_t shrunk_capacity;
    int has_camera_gamma;
    uint8_t linearize[256]; /* undoes the camera's tone curve; used only while refining a tag */
    uint8_t *linear_frame;  /* frame-sized so a tag keeps its coordinates; only each tag's own rectangle is ever filled */
    size_t linear_capacity;
    soft_tables *soft;
    int align_corners; /* during napa_detect_options: pull corners onto each blurred tag's own picture */
    soft_candidate *blocked_candidates, *grid_probe; /* probe: where a decode attempt copies its samples */
    size_t blocked_count, blocked_capacity;
    uint64_t *blocked_codes; /* every rotated code as a 6x6 bitboard, row by row */
    float (*tone_blur)[30][10]; /* blur tables for measuring blur and tone curve on a decoded tag; built on first use */
    soft_candidate *soft_candidates;
    size_t soft_count, soft_capacity;
    int soft_capturing, soft_failed;
    int ssse3; /* the CPU supports SSSE3 (x86 only; picked once at creation) */

    uint8_t *level; /* owned scratch, even when the finest gray pixels are borrowed */
    const uint8_t *level_view; /* non-owning, only during threshold_level; NULL uses owned scratch */
    /* Each level's tile extrema from the ordinary pass, reused by recovery on the same samples. */
    uint8_t *saved_extrema[32];
    size_t saved_extrema_capacity[32];
    uint8_t *level_extrema_out; /* where this level's extrema go, or NULL */
    const uint8_t *level_extrema_in; /* the saved extrema to reuse, or NULL */
    int extrema_wanted, extrema_ready;
    int saved_boxed[32]; /* whether that level was replaced by box averages */
    uint8_t level_contrast_min;
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
    /* Newly admitted whole shapes wait until the existing whole detections are complete. */
    napa_detection *shape_out;
    vec2 *shape_centers;
    size_t shape_out_capacity, shape_center_capacity;
    fine_whole *fine_wholes; /* outlines and confidence alternatives, replayed after proposals */
    size_t fine_whole_capacity;
    vec2 *out_centers, *piece_centers; /* accepted detections, in output order */
    size_t out_center_capacity, piece_center_capacity;
    uint8_t *fates; /* before necks are split: each level's candidates' and pieces' fates, for redoing it */
    size_t fate_capacity;
};

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
    for (int i = 0; i < 32; i++) free(det->saved_extrema[i]);
    free(det->level);
    free(det->failed);
    free(det->tone_blur);
    free(det->blocked_candidates);
    free(det->blocked_codes);
    free(det->linear_frame);
    free(det->shrunk);
    free(det->soft);
    free(det->soft_candidates);
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
    free(det->shape_out);
    free(det->shape_centers);
    free(det->fine_wholes);
    free(det->out_centers);
    free(det->piece_centers);
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

/* Exact integer-to-double values for scalar byte samples and signed differences. */
static const double sample_integer_values[511] = {
    -255.0, -254.0, -253.0, -252.0, -251.0, -250.0, -249.0, -248.0,
    -247.0, -246.0, -245.0, -244.0, -243.0, -242.0, -241.0, -240.0,
    -239.0, -238.0, -237.0, -236.0, -235.0, -234.0, -233.0, -232.0,
    -231.0, -230.0, -229.0, -228.0, -227.0, -226.0, -225.0, -224.0,
    -223.0, -222.0, -221.0, -220.0, -219.0, -218.0, -217.0, -216.0,
    -215.0, -214.0, -213.0, -212.0, -211.0, -210.0, -209.0, -208.0,
    -207.0, -206.0, -205.0, -204.0, -203.0, -202.0, -201.0, -200.0,
    -199.0, -198.0, -197.0, -196.0, -195.0, -194.0, -193.0, -192.0,
    -191.0, -190.0, -189.0, -188.0, -187.0, -186.0, -185.0, -184.0,
    -183.0, -182.0, -181.0, -180.0, -179.0, -178.0, -177.0, -176.0,
    -175.0, -174.0, -173.0, -172.0, -171.0, -170.0, -169.0, -168.0,
    -167.0, -166.0, -165.0, -164.0, -163.0, -162.0, -161.0, -160.0,
    -159.0, -158.0, -157.0, -156.0, -155.0, -154.0, -153.0, -152.0,
    -151.0, -150.0, -149.0, -148.0, -147.0, -146.0, -145.0, -144.0,
    -143.0, -142.0, -141.0, -140.0, -139.0, -138.0, -137.0, -136.0,
    -135.0, -134.0, -133.0, -132.0, -131.0, -130.0, -129.0, -128.0,
    -127.0, -126.0, -125.0, -124.0, -123.0, -122.0, -121.0, -120.0,
    -119.0, -118.0, -117.0, -116.0, -115.0, -114.0, -113.0, -112.0,
    -111.0, -110.0, -109.0, -108.0, -107.0, -106.0, -105.0, -104.0,
    -103.0, -102.0, -101.0, -100.0, -99.0, -98.0, -97.0, -96.0,
    -95.0, -94.0, -93.0, -92.0, -91.0, -90.0, -89.0, -88.0,
    -87.0, -86.0, -85.0, -84.0, -83.0, -82.0, -81.0, -80.0,
    -79.0, -78.0, -77.0, -76.0, -75.0, -74.0, -73.0, -72.0,
    -71.0, -70.0, -69.0, -68.0, -67.0, -66.0, -65.0, -64.0,
    -63.0, -62.0, -61.0, -60.0, -59.0, -58.0, -57.0, -56.0,
    -55.0, -54.0, -53.0, -52.0, -51.0, -50.0, -49.0, -48.0,
    -47.0, -46.0, -45.0, -44.0, -43.0, -42.0, -41.0, -40.0,
    -39.0, -38.0, -37.0, -36.0, -35.0, -34.0, -33.0, -32.0,
    -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0, -24.0,
    -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0,
    -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0,
    -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0,
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
    33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
    41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
    49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0,
    57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0,
    65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0,
    73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0,
    81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0,
    89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0,
    97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
    105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0,
    113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
    121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0,
    129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0,
    137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0,
    145.0, 146.0, 147.0, 148.0, 149.0, 150.0, 151.0, 152.0,
    153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0,
    161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0,
    169.0, 170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0,
    177.0, 178.0, 179.0, 180.0, 181.0, 182.0, 183.0, 184.0,
    185.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0,
    193.0, 194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 200.0,
    201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0,
    217.0, 218.0, 219.0, 220.0, 221.0, 222.0, 223.0, 224.0,
    225.0, 226.0, 227.0, 228.0, 229.0, 230.0, 231.0, 232.0,
    233.0, 234.0, 235.0, 236.0, 237.0, 238.0, 239.0, 240.0,
    241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0,
    249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0,
};

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
    const double *values = sample_integer_values + 255;
    double a = values[p[0]] + values[p[1] - p[0]] * fx;
    double b = values[p[stride]] + values[p[stride + 1] - p[stride]] * fx;
    double v = a + (b - a) * fy;
    return im->invert ? 255.0 - v : v;
}

/* ---------------------------------------------------------------- threshold */

#define TILE 4
/* The finest level, and the next unless that one is every pixel, only search neighbourhoods at least this many
 * times the minimum contrast. A tag's border against its quiet zone clears it easily; flat, noisy texture doesn't.
 * With every pixel sampled, tiny tags need the next level's weak signal to be searched at all. */
#define NAPA_FINEST_CONTRAST_NUM 12
#define NAPA_FINEST_CONTRAST_DEN 5
/* X-corners are refined only on tags whose edges curve at least this far from straight lines. */
#define NAPA_CURVED_TAG_MIN 0.3
/* Short edge runs beside a corner: most samples in a run, the least contrast worth trusting, how far an edge must
 * sit from the straight side to justify a fit, the smallest move worth keeping, and the bias worth acting on when
 * the sides disagreed about it. Beyond NAPA_LOCAL_MAX_BOW the border-width estimate of that bias stops holding. */
#define NAPA_LOCAL_EDGE_SAMPLES 32
#define NAPA_LOCAL_EDGE_CONTRAST 20.0
#define NAPA_LOCAL_EDGE_DISAGREE 0.2
#define NAPA_LOCAL_MIN_MOVE 0.2
#define NAPA_LOCAL_BIAS_MIN 0.25
#define NAPA_LOCAL_MAX_BOW 3.5
/* A camera gamma: the tag's rectangle in linear pixels reaches this far past its corners, far enough for every refinement read,
 * and is worth trying only if the tag's contrast there is at least this many times the contrast floor. */
#define NAPA_LINEAR_MARGIN_CELLS 2.5
#define NAPA_LINEAR_MARGIN_PIXELS 10
#define NAPA_LINEAR_MIN_CONTRAST 1.0
/* Each correction's decision fades across these widths instead of switching at its limit, so a tag near a limit moves part
 * of the way and noise can't flip it between two answers: the local refit's bias, side width ratio, bow, edge offset and
 * move; the whole-tag correction's side agreement, clipped bias and bias size. */
#define NAPA_FADE_BIAS 0.1
#define NAPA_FADE_WIDTH_RATIO 0.1
#define NAPA_FADE_BOW 0.1
#define NAPA_FADE_EDGE 0.1
#define NAPA_FADE_MOVE 0.1
#define NAPA_FADE_AGREEMENT 0.3
#define NAPA_FADE_CLIPPED_BIAS 0.015
#define NAPA_FADE_BIAS_STEP 0.05
/* The alternate side line: how fast outliers fade, as a share of the usual rejection distance, and how many passes. */
#define NAPA_SOFT_LINE_SCALE 0.7
#define NAPA_SOFT_LINE_PASSES 4
/* A real bow moves at least three corners. Fewer than three moving by this share of the largest shift, when that shift is
 * over this many cells, is one side's curve fit gone wrong on a straight tag. */
#define NAPA_BOW_MIN_SHARE 0.3
#define NAPA_BOW_LOPSIDED_CELLS 0.12
/* Motion blur widens some sides' edges and not others, which also makes the sides disagree about the border's width;
 * that disagreement says nothing about the tone curve, so a straight tag's refit stays out of it. */
#define NAPA_LOCAL_MAX_WIDTH_RATIO 1.3
/* Near the frame: the most edge samples per side, then what a shared radial lens must achieve to replace the fit: this
 * much bow, this much less residual than straight sides, and this close to four independent quadratics. */
#define NAPA_SIDE_SAMPLES 96
#define NAPA_LENS_MIN_KAPPA 0.02
#define NAPA_LENS_MIN_GAIN 0.3
#define NAPA_LENS_MAX_EXCESS 1.5
/* Blur shifts a tag's apparent edges by an amount set by the tone curve. After decoding, the tag's own picture gives its
 * blur and tone, and these limits say when to trust that: smallest tag side in pixels, fit quality, how clearly curved
 * the tone must be, the least blur (in cells) on a side, and the edge width below which a tag is too sharp to bother. */
#define NAPA_TONE_MIN_SIDE 16.0
#define NAPA_TONE_MIN_FIT 0.94719
#define NAPA_TONE_MAX_CURVATURE (-0.1)
#define NAPA_TONE_SCREEN_CURVATURE 0.1
#define NAPA_TONE_MIN_BLUR 0.2
#define NAPA_TONE_MIN_EDGE_CELLS 0.7
#define NAPA_TONE_MAX_CELL 98.0
#define NAPA_TONE_SIGMAS 11
#define NAPA_TONE_START 4
/* Straight tones: the widest edge, in cells, worth a search past the screen, and the curvature range the straight model covers. */
#define NAPA_TONE_STRAIGHT_EDGE_CELLS 0.8
#define NAPA_TONE_STRAIGHT_MIN_CURVATURE 0.0
#define NAPA_TONE_STRAIGHT_MAX_CURVATURE 0.5
/* When a tag's picture is trusted enough to align to: how well its white ring fits, how well the whole picture fits, and how
 * blurred, or how unevenly blurred, it must be to need it. Then the gain in correlation that makes a move worthwhile. */
#define NAPA_ALIGN_MAX_RING_ERROR 0.35
#define NAPA_ALIGN_MIN_FIT 0.93
#define NAPA_ALIGN_MIN_BLUR 0.35
#define NAPA_ALIGN_MIN_ANISOTROPY 1.5
#define NAPA_ALIGN_MIN_GAIN 0.001
#define NAPA_ALIGN_ITERATIONS 2
/* Soft decoding, calibrated for tag36h11 only. A candidate must be this small and look this much like a tag before
 * its score is worth computing: mean side in pixels, tag white minus black, spread of its payload cells, how many of the
 * 28 border cells are dark, and how far the brightest border cell may rise past the midpoint. */
#define NAPA_SOFT_MIN_SIDE 8.172
#define NAPA_SOFT_MAX_SIDE 18.933
#define NAPA_SOFT_MIN_CONTRAST 50.0
#define NAPA_SOFT_MAX_CONTRAST 238.118
#define NAPA_SOFT_MIN_SPREAD 0.4338
#define NAPA_SOFT_MAX_SPREAD 1.2
#define NAPA_SOFT_MIN_DARK_BORDER 24
#define NAPA_SOFT_MAX_BORDER_Z 0.75
/* What the winning code must achieve: correlation with the whole tag, lead over the best different ID, separation of
 * the code's white and black payload cells, and lead over the non-tag patterns tags are most often confused with. */
#define NAPA_SOFT_MIN_NCC 0.900182
#define NAPA_SOFT_MIN_LEAD 0.626128
#define NAPA_SOFT_MIN_SEPARATION 0.606306
#define NAPA_SOFT_MIN_NONTAG_LEAD 0.058770
#define NAPA_SOFT_SAMPLES 30
#define NAPA_SOFT_AREA (NAPA_SOFT_SAMPLES * NAPA_SOFT_SAMPLES)
/* Blur bins, in steps of 0.02 cells: only those the side limits above can reach. */
#define NAPA_SOFT_FIRST_BIN 15
#define NAPA_SOFT_BINS 20
#define NAPA_SOFT_NONTAGS 8
#define NAPA_SOFT_IDS 587
#define NAPA_SOFT_CODES (4 * NAPA_SOFT_IDS)
/* A blocked payload: a code explains a failed read if every wrong cell lies in one rectangle of at most this many cells
 * whose samples are this flat, as a fraction of the tag's contrast. A random payload does that rarely, whatever the size. */
#define NAPA_BLOCKED_MAX_AREA 16
#define NAPA_BLOCKED_MAX_SPAN 0.2
#define NAPA_BLOCKED_MIN_WRONG 4
#define NAPA_BLOCKED_MAX_WRONG 16
/* Fit the payload cells freely: how much of their amplitude may contradict the winning code. */
#define NAPA_SOFT_FIT 38
#define NAPA_SOFT_MAX_DISAGREEMENT 0.0406
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
/* The final fit leaves out each end of a side: half a search window (which follows the edge's width), at least 3 px
 * and 2% of the side, at most the coarse passes' 12%. Next to a corner the neighbouring side bends the profile. */
#define NAPA_SIDE_MARGIN 0.12
#define NAPA_SIDE_MIN_MARGIN 0.02
#define NAPA_SIDE_MARGIN_WINDOWS 0.5
#define NAPA_SIDE_MARGIN_PIXELS 3.0
/* An end joins the fit only where the border runs through it unbroken: edge strengths within this band of the middle
 * of the side's mean, and edges within this share of a cell (at least 0.5 px) of the line through the middle. */
#define NAPA_SIDE_END_MIN_STRENGTH 0.6
#define NAPA_SIDE_END_MAX_STRENGTH 1.67
#define NAPA_SIDE_END_REACH_CELLS 0.1
/* A corner refitted on its own is only trusted where the edges are sharp against a cell: under heavy blur the
 * corner's neighbourhood holds no separate edges to fit. */
#define NAPA_POLISH_MAX_EDGE_CELLS 0.6
/* Border cells whose width estimate is further than this share of a cell from the middle one are left out. */
#define NAPA_BORDER_CELL_REACH 0.07
#define NAPA_BORDER_CELLS_MAX 64
/* A balanced crossing is used on cells at least this wide, and only if it stays this close, in pixels, to the plain one. */
#define NAPA_BALANCE_MIN_CELL 2.4
#define NAPA_BALANCE_MAX_MOVE 0.12
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

/* The sums of `count` adjacent blocks of `kernel` bytes (2, 4, 8 or 16), reading each byte once. */
static void block_sums(const uint8_t *row, uint16_t *sum, int count, int kernel) {
    int i = 0;
#ifdef NAPA_SSE2
    const __m128i zero = _mm_setzero_si128(), low = _mm_set1_epi16(0x00ff), one = _mm_set1_epi16(1);
    if (kernel == 2) {
        for (; i + 8 <= count; i += 8) {
            __m128i a = _mm_loadu_si128((const __m128i *)(row + 2 * i));
            _mm_storeu_si128((__m128i *)(sum + i), _mm_add_epi16(_mm_and_si128(a, low), _mm_srli_epi16(a, 8)));
        }
    } else if (kernel == 4) {
        for (; i + 8 <= count; i += 8) {
            __m128i a = _mm_loadu_si128((const __m128i *)(row + 4 * i)), b = _mm_loadu_si128((const __m128i *)(row + 4 * i + 16));
            a = _mm_madd_epi16(_mm_add_epi16(_mm_and_si128(a, low), _mm_srli_epi16(a, 8)), one);
            b = _mm_madd_epi16(_mm_add_epi16(_mm_and_si128(b, low), _mm_srli_epi16(b, 8)), one);
            _mm_storeu_si128((__m128i *)(sum + i), _mm_packs_epi32(a, b));
        }
    } else if (kernel == 8) {
        for (; i + 2 <= count; i += 2) {
            __m128i a = _mm_sad_epu8(_mm_loadu_si128((const __m128i *)(row + 8 * i)), zero);
            sum[i] = (uint16_t)_mm_cvtsi128_si32(a), sum[i + 1] = (uint16_t)_mm_extract_epi16(a, 4);
        }
    } else if (kernel == 16) {
        for (; i < count; i++) {
            __m128i a = _mm_sad_epu8(_mm_loadu_si128((const __m128i *)(row + 16 * i)), zero);
            sum[i] = (uint16_t)(_mm_cvtsi128_si32(a) + _mm_extract_epi16(a, 4));
        }
    }
#elif defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    if (kernel == 2) {
        for (; i + 8 <= count; i += 8) vst1q_u16(sum + i, vpaddlq_u8(vld1q_u8(row + 2 * i)));
    } else {
        /* Each step halves the count from two bytes per sum: 16 x 2 bytes make 8 sums of 4, then 4 of 8, then 2 of 16. */
        int per = 32 / kernel;
        for (; i + per <= count; i += per) {
            uint16x8_t a = vpaddlq_u8(vld1q_u8(row + kernel * i)), b = vpaddlq_u8(vld1q_u8(row + kernel * i + 16));
            uint16x8_t fours = vcombine_u16(vpadd_u16(vget_low_u16(a), vget_high_u16(a)), vpadd_u16(vget_low_u16(b), vget_high_u16(b)));
            if (kernel == 4) {
                vst1q_u16(sum + i, fours);
                continue;
            }
            uint16x4_t eights = vpadd_u16(vget_low_u16(fours), vget_high_u16(fours));
            if (kernel == 8) {
                vst1_u16(sum + i, eights);
            } else {
                uint16x4_t sixteens = vpadd_u16(eights, eights);
                sum[i] = vget_lane_u16(sixteens, 0), sum[i + 1] = vget_lane_u16(sixteens, 1);
            }
        }
    }
#endif
    for (; i < count; i++) {
        unsigned total = 0;
        for (int k = 0; k < kernel; k++) total += row[kernel * i + k];
        sum[i] = (uint16_t)total;
    }
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
    /* Blocks of a power-of-two width that tile the row are summed along the row first, which reads each pixel once. */
    int shift = 0;
    if (kernel == s && (s == 2 || s == 4 || s == 8 || s == 16)) shift = s == 2 ? 2 : s == 4 ? 4 : s == 8 ? 6 : 8;
    for (int y = 0; y < lh; y++) {
        int r0 = y * s - half, r1 = r0 + kernel;
        if (r0 < 0) r0 = 0;
        if (r1 > h) r1 = h;
        if (shift && r1 - r0 == kernel) {
            int x_lo = (half + s - 1) / s, x_hi = (w - kernel + half) / s + 1;
            if (x_hi > lw) x_hi = lw;
            int count = x_hi - x_lo;
            uint8_t *dst = level + (size_t)y * lw;
            if (count > 0) {
                const size_t first = (size_t)x_lo * s - half;
                block_sums(img + (size_t)r0 * stride + first, acc, count, kernel);
                for (int r = r0 + 1; r < r1; r++) {
                    block_sums(img + (size_t)r * stride + first, hs, count, kernel);
                    for (int i = 0; i < count; i++) acc[i] = (uint16_t)(acc[i] + hs[i]);
                }
                const uint16_t round = (uint16_t)(1u << (shift - 1));
                uint8_t *whole = dst + x_lo;
                for (int i = 0; i < count; i++) whole[i] = (uint8_t)((uint16_t)(acc[i] + round) >> shift);
                for (int x = 0; x < lw; x++) {
                    if (x == x_lo) x = x_hi;
                    if (x >= lw) break;
                    int c0 = x * s - half, c1 = c0 + kernel, sum = 0;
                    if (c0 < 0) c0 = 0;
                    if (c1 > w) c1 = w;
                    for (int r = r0; r < r1; r++) {
                        const uint8_t *p = img + (size_t)r * stride;
                        for (int c = c0; c < c1; c++) sum += p[c];
                    }
                    int cnt = (c1 - c0) * (r1 - r0);
                    dst[x] = (uint8_t)((sum + cnt / 2) / cnt);
                }
                continue;
            }
        }
        const uint8_t *row = img + (size_t)r0 * stride;
        for (int x = 0; x < w; x++) acc[x] = row[x];
        for (int r = r0 + 1; r < r1; r++) {
            row = img + (size_t)r * stride;
            for (int x = 0; x < w; x++) acc[x] = (uint16_t)(acc[x] + row[x]);
        }
        /* hs[c] = acc[c] + ... + acc[c + kernel - 1], for every block start c that fits in the row. */
        /* Built by doubling, so a kernel of k takes about log2(k) passes instead of k. */
        for (int c = 0; c < w; c++) hs[c] = acc[c];
        int span = 1;
        for (int bit = 3; bit >= 0; bit--) {
            if ((kernel >> (bit + 1)) == 0) continue; /* bits at or above the top one */
            int doubled = 2 * span;
            for (int c = 0; c + doubled <= w; c++) hs[c] = (uint16_t)(hs[c] + hs[c + span]);
            span = doubled;
            if ((kernel >> bit) & 1) {
                for (int c = 0; c + span < w; c++) hs[c] = (uint16_t)(hs[c] + acc[c + span]);
                span++;
            }
        }
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
#define NAPA_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define NAPA_FORCE_INLINE __forceinline
#define NAPA_NOINLINE __declspec(noinline)
#else
#define NAPA_FORCE_INLINE inline
#define NAPA_NOINLINE
#endif

static inline int32_t find_root(int32_t *P, int32_t i) {
    while (P[i] != i) {
        P[i] = P[P[i]];
        i = P[i];
    }
    return i;
}

/* Require filled support beyond both corners before separating a diagonal contact.
 * A binary 2x2 saddle alone cannot distinguish a thin stroke from touching objects. */
static NAPA_FORCE_INLINE int filled_corner_contact(const uint8_t *L, int w, int h, int y, int x, int upper_left_dark, int floor) {
    if (x < 1 || x + 2 >= w || y < 2 || y + 1 >= h) return 0;
    int a = L[(size_t)(y - 1) * w + x], b = L[(size_t)(y - 1) * w + x + 1];
    int c = L[(size_t)y * w + x], d = L[(size_t)y * w + x + 1];
    int dark1 = upper_left_dark ? a : b, dark2 = upper_left_dark ? d : c;
    int white1 = upper_left_dark ? b : a, white2 = upper_left_dark ? c : d;
    int dark = dark1 > dark2 ? dark1 : dark2;
    int white = white1 < white2 ? white1 : white2;
    if (white - dark < floor) return 0;
    int limit = dark;
    int ax = upper_left_dark ? x : x + 1;
    int bx = upper_left_dark ? x + 1 : x;
    int da = upper_left_dark ? -1 : 1, db = -da;
    return L[(size_t)(y - 1) * w + ax + da] <= limit &&
           L[(size_t)(y - 2) * w + ax] <= limit &&
           L[(size_t)(y - 2) * w + ax + da] <= limit &&
           L[(size_t)y * w + bx + db] <= limit &&
           L[(size_t)(y + 1) * w + bx] <= limit &&
           L[(size_t)(y + 1) * w + bx + db] <= limit;
}

/* Both paths must omit the same edges, including across regional noise-floor boundaries. */
static NAPA_NOINLINE int separate_corner_contact(const napa_detector *det, int lw, int lh, int factor, int a, int b) {
    const napa_run *R = det->runs;
    if (R[a].x1 <= R[a].x0 || R[b].x1 <= R[b].x0) return 0;
    int left = R[a].x1 + 1 == R[b].x0;
    int x = left ? R[b].x0 - 1 : R[b].x1;
    int y = R[b].y, floor = det->floors[0];
    if (!det->floor_uniform) {
        int cols = det->floor_cols, rows = det->floor_rows;
        int x0 = (x / TILE) * TILE * factor / NAPA_REGION;
        int x1 = ((x + 1) / TILE) * TILE * factor / NAPA_REGION;
        int y0 = ((y - 1) / TILE) * TILE * factor / NAPA_REGION;
        int y1 = (y / TILE) * TILE * factor / NAPA_REGION;
        if (x0 >= cols) x0 = cols - 1;
        if (x1 >= cols) x1 = cols - 1;
        if (y0 >= rows) y0 = rows - 1;
        if (y1 >= rows) y1 = rows - 1;
        const uint8_t *up = det->floors + (size_t)y0 * cols;
        const uint8_t *dn = det->floors + (size_t)y1 * cols;
        floor = up[x0] > up[x1] ? up[x0] : up[x1];
        if (dn[x0] > floor) floor = dn[x0];
        if (dn[x1] > floor) floor = dn[x1];
    }
    return filled_corner_contact(det->level_view ? det->level_view : det->level, lw, lh, y, x, left, floor);
}

/* Join adjacent rows for optional splitting. Supported filled-corner contacts stay separate, as in
 * ordinary indexing. Other corner contacts (and, with split 2, one-sample necks) are deferred as pairs. */
static NAPA_FORCE_INLINE void join_rows(napa_detector *det, const napa_run *R, int32_t *P, int a, int a_end, int b,
                                        int b_end, const int split, int lw, int lh, int factor, int shadow) {
    while (a < a_end && b < b_end) {
        if (R[a].x1 + 1 < R[b].x0) {
            a++;
        } else if (R[b].x1 + 1 < R[a].x0) {
            b++;
        } else if (split && !shadow && (R[a].x1 + 1 == R[b].x0 || R[b].x1 + 1 == R[a].x0) &&
                   separate_corner_contact(det, lw, lh, factor, a, b)) {
            /* These contacts also stay separate in ordinary detection: do not defer them. */
            if (R[a].x1 < R[b].x1) a++;
            else b++;
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

/* Threshold the level and label its dark samples as runs joined by union-find. Adjacent-row runs
 * connect within one sample, except supported filled-corner contacts in ordinary detection. Returns
 * the run count, or -1 when out of memory. Optional splitting defers the remaining corner contacts
 * (split 1), or also one-sample necks (split 2), in det->diag / det->neck for split_pieces. */
/* Exact 4x4 tile extrema, with short bands and the final partial tile repeated at their edges.
 * Adapted with sampling/median and edge indexing from PR23,
 * 7735f7a2e41b40584cd48a7f07c0bd947c4993c2 (same project/license).
 * Single-row runs, split-piece ordering and restart bookkeeping remain the existing representation. */
static void tile_row_minmax(const uint8_t *L, size_t lstride, int lw, int lh, int ty, int tw, uint8_t *vmin,
                            uint8_t *vmax, uint8_t *tmin, uint8_t *tmax) {
    int y0 = ty * TILE;
    int y1 = y0 + TILE < lh ? y0 + TILE : lh;
    int x = 0;
#ifdef NAPA_SSE2
    /* Whole 64-sample blocks in one pass: each sample's min and max over the band's rows (a short
     * band repeats its first row), then each tile's four within its 32-bit lane: two shift-and-min
     * steps leave the tile's value in the lane's low byte, and packing those gives 16 tiles. */
    const uint8_t *row[TILE];
    for (int k = 0; k < TILE; k++) row[k] = L + (size_t)(y0 + (y0 + k < y1 ? k : 0)) * lstride;
    const __m128i low = _mm_set1_epi32(0xff);
    for (; x + 64 <= lw; x += 64) {
        __m128i lo[4], hi[4];
        for (int c = 0; c < 4; c++) {
            const int at = x + 16 * c;
            __m128i a = _mm_loadu_si128((const __m128i *)(row[0] + at)), b = _mm_loadu_si128((const __m128i *)(row[1] + at));
            __m128i d = _mm_loadu_si128((const __m128i *)(row[2] + at)), e = _mm_loadu_si128((const __m128i *)(row[3] + at));
            __m128i mn = _mm_min_epu8(_mm_min_epu8(a, b), _mm_min_epu8(d, e));
            __m128i mx = _mm_max_epu8(_mm_max_epu8(a, b), _mm_max_epu8(d, e));
            mn = _mm_min_epu8(mn, _mm_srli_epi32(mn, 8));
            mx = _mm_max_epu8(mx, _mm_srli_epi32(mx, 8));
            lo[c] = _mm_and_si128(_mm_min_epu8(mn, _mm_srli_epi32(mn, 16)), low);
            hi[c] = _mm_and_si128(_mm_max_epu8(mx, _mm_srli_epi32(mx, 16)), low);
        }
        _mm_storeu_si128((__m128i *)(tmin + x / 4),
                         _mm_packus_epi16(_mm_packs_epi32(lo[0], lo[1]), _mm_packs_epi32(lo[2], lo[3])));
        _mm_storeu_si128((__m128i *)(tmax + x / 4),
                         _mm_packus_epi16(_mm_packs_epi32(hi[0], hi[1]), _mm_packs_epi32(hi[2], hi[3])));
    }
    if (x >= lw) return;
#endif
    /* The rest of the row: a vertical pass over the band's rows (contiguous, so it vectorises), then
     * a short horizontal pass over its TILE-wide groups. */
    const int n = lw - x, tiles = tw - x / TILE;
    memcpy(vmin, L + (size_t)y0 * lstride + x, (size_t)n);
    memcpy(vmax, L + (size_t)y0 * lstride + x, (size_t)n);
    for (int y = y0 + 1; y < y1; y++) {
        const uint8_t *src = L + (size_t)y * lstride + x;
        for (int i = 0; i < n; i++) {
            uint8_t v = src[i];
            vmin[i] = v < vmin[i] ? v : vmin[i];
            vmax[i] = v > vmax[i] ? v : vmax[i];
        }
    }
    for (int i = n; i < TILE * tiles; i++) vmin[i] = vmin[n - 1], vmax[i] = vmax[n - 1];
    for (int t = 0; t < tiles; t++) {
        const uint8_t *a = vmin + 4 * t, *b = vmax + 4 * t;
        uint8_t l01 = a[0] < a[1] ? a[0] : a[1], l23 = a[2] < a[3] ? a[2] : a[3];
        uint8_t h01 = b[0] > b[1] ? b[0] : b[1], h23 = b[2] > b[3] ? b[2] : b[3];
        tmin[x / TILE + t] = l01 < l23 ? l01 : l23;
        tmax[x / TILE + t] = h01 > h23 ? h01 : h23;
    }
}

#ifdef _MSC_VER
#define NAPA_RESTRICT __restrict
#else
#define NAPA_RESTRICT restrict
#endif
/* Independent output rows let the compiler vectorize the vertical 3x3 extrema pass. */
static void min3_max3(uint8_t *NAPA_RESTRICT lo, uint8_t *NAPA_RESTRICT hi, const uint8_t *a, const uint8_t *b,
                      const uint8_t *c, const uint8_t *d, const uint8_t *e, const uint8_t *f, int n) {
    for (int i = 0; i < n; i++) {
        uint8_t l = b[i], h = e[i];
        l = a[i] < l ? a[i] : l;
        l = c[i] < l ? c[i] : l;
        h = d[i] > h ? d[i] : h;
        h = f[i] > h ? f[i] : h;
        lo[i] = l;
        hi[i] = h;
    }
}

#ifdef NAPA_SSSE3_DISPATCH
#define INDEX_POPCOUNT(hw, x) ((hw) ? __builtin_popcountll(x) : napa_popcount64(x))
#else
#define INDEX_POPCOUNT(hw, x) napa_popcount64(x)
#endif
/* Label one complete run using the row's original ranks and union order.
 * Its complete record is already stored when contact checks inspect it.
 * Edges alternate start/end+1. Inclusive word counts minus the suffix from q give
 * counts before q, including at word and sentinel boundaries. Counts before b.x0
 * and b.x1+2 give the eight-connected range above; ordinary indexing then trims
 * supported corner contacts. Visit b and its neighbors in raster order so backward
 * roots retain the original packed order. Single-row runs need no row-pair mask. */
static NAPA_FORCE_INLINE void label_indexed_run(napa_detector *det, int lw, int lh, int factor,
                                  int b, int prev_start, const uint64_t *mask, const int32_t *prefix, int shadow, int hw) {
    napa_run *R = det->runs;
    int32_t *P = det->parent;
    (void)hw;
    int q = R[b].x0;
    int f = prev_start + ((prefix[q >> 6] - INDEX_POPCOUNT(hw, mask[q >> 6] >> (q & 63))) >> 1);
    q = R[b].x1 + 2;
    int g = prev_start + ((prefix[q >> 6] - INDEX_POPCOUNT(hw, mask[q >> 6] >> (q & 63)) + 1) >> 1);
    if (!shadow && f < g) {
        if (R[f].x1 + 1 == R[b].x0 && separate_corner_contact(det, lw, lh, factor, f, b)) f++;
        if (f < g && R[g - 1].x0 == R[b].x1 + 1 && separate_corner_contact(det, lw, lh, factor, g - 1, b)) g--;
    }
    P[b] = f < g ? P[f] : b;
    for (int a = f + 1; a < g; a++) {
        if (P[b] != P[a]) {
            int32_t ra = find_root(P, a), rb = find_root(P, b);
            if (ra < rb) P[rb] = ra;
            else if (rb < ra) P[ra] = rb;
        }
    }
}

static NAPA_FORCE_INLINE int extract_labeled_runs(napa_detector *det, const uint64_t *bits,
                                    int lw, int lh, int factor, int y, int n, int prev_start,
                                    const uint64_t *mask, const int32_t *prefix, const int shadow, int hw) {
    napa_run *R = det->runs;
    int open = 0, x0 = 0;
    for (int w = 0; w * 64 < lw; w++) {
        uint64_t edges = bits[w] ^ ((bits[w] << 1) | (uint64_t)open);
        while (edges) {
            int bit = napa_ctz64(edges);
            edges &= edges - 1;
            if (!open) {
                x0 = 64 * w + bit, open = 1;
            } else {
                R[n].x0 = (uint16_t)x0, R[n].x1 = (uint16_t)(64 * w + bit - 1), R[n].y = (uint16_t)y;
                label_indexed_run(det, lw, lh, factor, n, prev_start, mask, prefix, shadow, hw);
                n++, open = 0;
            }
        }
    }
    if (open) {
        R[n].x0 = (uint16_t)x0, R[n].x1 = (uint16_t)(lw - 1), R[n].y = (uint16_t)y;
        label_indexed_run(det, lw, lh, factor, n, prev_start, mask, prefix, shadow, hw);
        n++;
    }
    return n;
}

static NAPA_FORCE_INLINE int threshold_index_impl(napa_detector *det, int lw, int lh, int factor, const int split, int shadow, const int hw) {
    const uint8_t *L = det->level_view ? det->level_view : det->level;
    napa_run *R = det->runs;
    int32_t *P = det->parent;
    /* Image dimensions are validated at <= 30000; q may reach lw + 1. */
    enum { INDEX_WORDS = (30000 + 63) / 64 + 2 };
    uint64_t masks[2][INDEX_WORDS];
    int32_t prefixes[2][INDEX_WORDS];
    (void)hw; /* portable builds use the generic population count */
    int n = 0, prev_y = -2, prev_start = 0, prev_end = 0;
    det->diag_n = det->neck_n = 0;
    int tw = (lw + TILE - 1) / TILE;
    int th = (lh + TILE - 1) / TILE;
    uint8_t *vmin = det->row_min, *vmax = det->row_max, *trow = det->row_thr;

    /* 3x3 tile neighbourhood, separably and without branches so the loops vectorise: vertically
     * into a row padded by one tile at each end (repeating the edge tile), then horizontally. */
    uint8_t *nlo = det->nb_lo, *nhi = det->nb_hi, *tt = det->nb_t, *fl = det->nb_f;
    const uint8_t contrast_min = det->level_contrast_min;
    /* A tile spans 4 * factor samples of the grid where the floors were measured. */
    int rows = det->floor_rows, cols = det->floor_cols;
    if (det->floor_uniform) memset(fl, det->floors[0], (size_t)tw);
    const uint8_t *saved = det->level_extrema_in;
    const size_t tile_count = (size_t)tw * th;
    const uint8_t *tile_min = saved ? saved : det->tile_min;
    const uint8_t *tile_max = saved ? saved + tile_count : det->tile_max;
    if (!saved) tile_row_minmax(L, lw, lw, lh, 0, tw, vmin, vmax, det->tile_min, det->tile_max);
    for (int ty = 0; ty < th; ty++) {
        if (!saved && ty + 1 < th)
            tile_row_minmax(L, lw, lw, lh, ty + 1, tw, vmin, vmax,
                            det->tile_min + (size_t)(ty + 1) * tw, det->tile_max + (size_t)(ty + 1) * tw);
        int ry = ty * TILE * factor / NAPA_REGION;
        const uint8_t *frow = det->floors + (size_t)(ry < rows ? ry : rows - 1) * cols;
        for (int tx = 0; tx < tw && !det->floor_uniform; tx++) {
            int rx = tx * TILE * factor / NAPA_REGION;
            fl[tx] = frow[rx < cols ? rx : cols - 1];
        }
        const uint8_t *lo_up = tile_min + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *lo_mid = tile_min + (size_t)ty * tw;
        const uint8_t *lo_dn = tile_min + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        const uint8_t *hi_up = tile_max + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *hi_mid = tile_max + (size_t)ty * tw;
        const uint8_t *hi_dn = tile_max + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        min3_max3(nlo + 1, nhi + 1, lo_up, lo_mid, lo_dn, hi_up, hi_mid, hi_dn, tw);
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
            uint8_t t = (uint8_t)(h - l) >= fl[tx] && (uint8_t)(h - l) >= contrast_min ? (uint8_t)(shadow ? l + ((h - l + 2) >> 2) : (l + h + 1) >> 1) : 0;
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
            if (det->level_view && (size_t)(lh - y) * lw < ((size_t)lw + 63) / 64 * 64) {
                /* Borrowed input need not have the owned level's 64 padding bytes. For narrow
                 * images, rounded word loads can cross the array before the final row. */
                int whole = lw & ~63;
                uint8_t tail[64] = {0};
                memcpy(tail, L + (size_t)y * lw + whole, (size_t)(lw - whole));
                dark_bits(L + (size_t)y * lw, trow, whole, bits);
                dark_bits(tail, trow + whole, 64, bits + whole / 64);
            } else dark_bits(L + (size_t)y * lw, trow, lw, bits);
            if (!split) {
                uint64_t carry = 0;
                int count = 0, words = (lw + 63) / 64;
                for (int w = 0; w < words; w++) {
                    uint64_t x = bits[w], e = x ^ ((x << 1) | carry);
                    carry = x >> 63;
                    masks[y & 1][w] = e;
                    count += INDEX_POPCOUNT(hw, e);
                    prefixes[y & 1][w] = count; /* through this word */
                }
                masks[y & 1][words] = 0;
                prefixes[y & 1][words] = count;
            }
            int start = n;
            if (!split && prev_y == y - 1 && prev_start < prev_end)
                n = extract_labeled_runs(det, bits, lw, lh, factor, y, n, prev_start,
                                        masks[(y & 1) ^ 1], prefixes[(y & 1) ^ 1], shadow, hw);
            else n = extract_runs(bits, lw, y, R, n);
            /* Adjacent ordinary runs were labeled during emission; other paths keep their setup. */
            if (split || prev_y != y - 1 || prev_start == prev_end)
                for (int i = start; i < n; i++) P[i] = i;
            if (prev_y == y - 1 && split) {
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
                if (split == 2) join_rows(det, R, P, prev_start, prev_end, start, n, 2, lw, lh, factor, shadow);
                else join_rows(det, R, P, prev_start, prev_end, start, n, 1, lw, lh, factor, shadow);
            }
            prev_y = y, prev_start = start, prev_end = n;
        }
    }
    if (det->level_extrema_out) {
        memcpy(det->level_extrema_out, det->tile_min, tile_count);
        memcpy(det->level_extrema_out + tile_count, det->tile_max, tile_count);
    }
    return n;
}
static int threshold_index_generic(napa_detector *det, int lw, int lh, int factor, int split, int shadow) {
    /* Keep the common ordinary loop free of recovery-mode branches. */
    if (!split && !shadow) return threshold_index_impl(det, lw, lh, factor, 0, 0, 0);
    return threshold_index_impl(det, lw, lh, factor, split, shadow, 0);
}
#ifdef NAPA_SSSE3_DISPATCH
__attribute__((target("popcnt"))) static int threshold_index_popcnt(napa_detector *det, int lw, int lh, int factor, int split, int shadow) {
    /* Keep the common ordinary loop free of recovery-mode branches. */
    if (!split && !shadow) return threshold_index_impl(det, lw, lh, factor, 0, 0, 1);
    return threshold_index_impl(det, lw, lh, factor, split, shadow, 1);
}
#endif
static int threshold_level(napa_detector *det, int lw, int lh, int factor, int split, int shadow) {
#ifdef NAPA_SSSE3_DISPATCH
    if (__builtin_cpu_supports("popcnt")) return threshold_index_popcnt(det, lw, lh, factor, split, shadow);
#endif
    return threshold_index_generic(det, lw, lh, factor, split, shadow);
}


/* PR23 7735f7a2e41b40584cd48a7f07c0bd947c4993c2: exact four-point interpolation. */
static void profile_fixed(const uint8_t *img, int stride, int32_t fxp, int32_t fyp, int32_t ixp, int32_t iyp, int K,
                          int64_t *prof) {
#ifdef NAPA_SSE2
    /* Vectors are built in registers, not stored point by point and loaded back four at a time: a load
     * spanning several smaller pending stores can't be forwarded from them and waits for them to retire.
     * Positions are the running sums as lane offsets; four points' pixel pairs pack into one 64-bit word. */
    const __m128i zero = _mm_setzero_si128(), one = _mm_set1_epi32(4096), low15 = _mm_set1_epi32(0x7fff);
    __m128i px = _mm_add_epi32(_mm_set1_epi32(fxp), _mm_setr_epi32(0, ixp, 2 * ixp, 3 * ixp));
    __m128i py = _mm_add_epi32(_mm_set1_epi32(fyp), _mm_setr_epi32(0, iyp, 2 * iyp, 3 * iyp));
    const __m128i px4 = _mm_set1_epi32(4 * ixp), py4 = _mm_set1_epi32(4 * iyp);
    for (int g = 0; g < K; g += 4, px = _mm_add_epi32(px, px4), py = _mm_add_epi32(py, py4)) {
        uint64_t u = 0, d = 0; /* lanes past K stay zero; their results are never read */
        for (int k = 0; k < 4 && g + k < K; k++, fxp += ixp, fyp += iyp) {
            const uint8_t *q = img + (size_t)(fyp >> 16) * stride + (fxp >> 16);
            uint16_t a, b;
            memcpy(&a, q, 2), memcpy(&b, q + stride, 2); /* x86 is little-endian: low byte left */
            u |= (uint64_t)a << (16 * k), d |= (uint64_t)b << (16 * k);
        }
        /* Weights as 16-bit pairs (4096 - f, f): f is bits 4-15 of the position. */
        __m128i fx = _mm_srli_epi32(_mm_slli_epi32(px, 16), 20), fy = _mm_srli_epi32(_mm_slli_epi32(py, 16), 20);
        __m128i wx = _mm_or_si128(_mm_sub_epi32(one, fx), _mm_slli_epi32(fx, 16));
        __m128i wy = _mm_or_si128(_mm_sub_epi32(one, fy), _mm_slli_epi32(fy, 16));
        __m128i top = _mm_madd_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&u), zero), wx);
        __m128i bot = _mm_madd_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)&d), zero), wx);
        __m128i lo = _mm_or_si128(_mm_and_si128(top, low15), _mm_slli_epi32(_mm_and_si128(bot, low15), 16));
        __m128i hi = _mm_or_si128(_mm_srli_epi32(top, 15), _mm_slli_epi32(_mm_srli_epi32(bot, 15), 16));
        __m128i v = _mm_add_epi32(_mm_slli_epi32(_mm_madd_epi16(hi, wy), 15), _mm_madd_epi16(lo, wy));
        _mm_storeu_si128((__m128i *)(prof + g), _mm_unpacklo_epi32(v, zero));
        _mm_storeu_si128((__m128i *)(prof + g + 2), _mm_unpackhi_epi32(v, zero));
    }
#else
    for (int j = 0; j < K; j++, fxp += ixp, fyp += iyp) {
        int x0 = fxp >> 16, y0 = fyp >> 16;
        int32_t fx = (fxp >> 4) & 4095, fy = (fyp >> 4) & 4095;
        const uint8_t *q = img + (size_t)y0 * stride + x0;
        int32_t top = (q[0] << 12) + (q[1] - q[0]) * fx;
        int32_t bot = (q[stride] << 12) + (q[stride + 1] - q[stride]) * fx;
        prof[j] = ((int64_t)top << 12) + (int64_t)(bot - top) * fy;
    }
#endif
}

/* ---------------------------------------------------------------- geometry */

/* Fit each side to the outer border edge searched from r_out outside to r_in inside the given quad. */
/* Edge levels for the midpoint-crossing estimator; black >= white disables it. */
typedef struct {
    double black, white;
    double midpoint[3], independent_midpoint[3], cx, cy, scale;
    int has_model, model_pending;
    const family_layout *model_family;
} edge_levels;

enum {
    FINE_WHOLE_OUTLINE,
    FINE_WHOLE_CONFIDENCE_PENDING,
    FINE_WHOLE_CONFIDENCE_FINISHED,
    FINE_WHOLE_CONFIDENCE_REJECTED
};

/* Outlines and exact early alternatives share the existing deferred allocation.
 * Confidence detections stay quarantined until established wholes are complete. */
struct fine_whole {
    union {
        component shape;
        struct {
            vec2 fine[4];
            double reach, edge_width;
            edge_levels lv;
            int early_id;
        } confidence;
        napa_detection completed;
    } data;
    int stride, terminal, kind;
};

/* Keep the rare illumination fit's sample arrays off the hot refinement stack. */
static NAPA_NOINLINE void refresh_edge_light(const image_view *im, const vec2 q[4], edge_levels *levels);

/* Retain original initial-fit samples before the existing TLS compacts them. */
typedef struct {
    int count[4];
    unsigned rejection_mask;
    double x[4][6], y[4][6], weight[4][6];
} initial_capture;

static NAPA_FORCE_INLINE int fit_side_line_impl(double px[], double py[], double wt[], int good, double reach,
                     double *lx, double *ly, double *dx, double *dy, int *rejected) {
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
        (*lx) = mx;
        (*ly) = my;
        (*dx) = ct;
        (*dy) = st;
        if (pass == 0) {
            int kept = 0;
            for (int i = 0; i < good; i++) {
                double dist = fabs((px[i] - mx) * (-(*dy)) + (py[i] - my) * (*dx));
                if (dist <= 1.0 + 0.1 * reach) {
                    px[kept] = px[i];
                    py[kept] = py[i];
                    wt[kept] = wt[i];
                    kept++;
                }
            }
            if (rejected != NULL) *rejected = kept != good;
            /* Shorten the second pass only with enough inliers. Retain the established
             * partial-compaction fallback when too few samples survive. */
            if (kept == good) break; /* unchanged samples would produce the same fit again */
            if (kept >= 3 && 2 * kept >= good) good = kept;
        }
    }
    return 1;
}

static NAPA_FORCE_INLINE int fit_side_line(double px[], double py[], double wt[], int good, double reach,
                     double *lx, double *ly, double *dx, double *dy) {
    return fit_side_line_impl(px, py, wt, good, reach, lx, ly, dx, dy, NULL);
}

/* Report initial sample rejection without changing arithmetic, compaction or
 * the established partial-compaction fallback. */
static NAPA_FORCE_INLINE int fit_initial_side_observed(double px[], double py[], double wt[], int good, double reach,
                     double *lx, double *ly, double *dx, double *dy, unsigned *mask, unsigned side) {
    int rejected = 0;
    int ok = fit_side_line_impl(px, py, wt, good, reach, lx, ly, dx, dy, &rejected);
    if (rejected) *mask |= 1u << side;
    return ok;
}

/* Final edge samples can follow a lens-curved border rather than a straight line.
 * Fit normal displacement in pixels against tangent position / side length - 0.5.
 * Keep TLS unless the same raw samples strongly support a quadratic model. */
typedef struct {
    vec2 a;
    double ex, ey, nx, ny, len, c[3];
    int qualified;
} curve_side;

static void fit_curve_side(curve_side *s, vec2 a, vec2 b, const double *px, const double *py,
                           const double *wt, int n) {
    memset(s, 0, sizeof *s);
    s->a = a;
    s->len = hypot(b.x - a.x, b.y - a.y);
    if (n < 12 || !(s->len >= 6)) return;
    s->ex = (b.x - a.x) / s->len;
    s->ey = (b.y - a.y) / s->len;
    s->nx = -s->ey;
    s->ny = s->ex;
    double m[5] = {0}, v[3] = {0}, half[2] = {0};
    for (int i = 0; i < n; i++) {
        double u = ((px[i] - a.x) * s->ex + (py[i] - a.y) * s->ey) / s->len - 0.5;
        double y = (px[i] - a.x) * s->nx + (py[i] - a.y) * s->ny, p = 1;
        if (!isfinite(u) || !isfinite(y) || !(wt[i] > 0) || !isfinite(wt[i])) return;
        half[u >= 0] += wt[i];
        for (int j = 0; j < 5; j++) {
            m[j] += wt[i] * p;
            if (j < 3) v[j] += wt[i] * y * p;
            p *= u;
        }
    }
    /* Require meaningful weighted support across the side and a conditioned tangent spread. */
    if (!(m[0] > 0) || half[0] < 0.15 * m[0] || half[1] < 0.15 * m[0]) return;
    double variance = m[2] / m[0] - (m[1] / m[0]) * (m[1] / m[0]);
    if (!(variance >= 0.02)) return;
    double det = m[0] * m[2] - m[1] * m[1];
    double line[2] = {(v[0] * m[2] - v[1] * m[1]) / det,
                      (v[1] * m[0] - v[0] * m[1]) / det};
    double A[3][4];
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) A[j][k] = m[j + k] / m[0];
        A[j][3] = v[j] / m[0];
    }
    for (int j = 0; j < 3; j++) {
        int pivot = j;
        for (int k = j + 1; k < 3; k++)
            if (fabs(A[k][j]) > fabs(A[pivot][j])) pivot = k;
        if (!(fabs(A[pivot][j]) > 1e-7)) return;
        for (int k = j; k < 4; k++) {
            double tmp = A[pivot][k];
            A[pivot][k] = A[j][k];
            A[j][k] = tmp;
        }
        double p = A[j][j];
        for (int k = j; k < 4; k++) A[j][k] /= p;
        for (int k = 0; k < 3; k++) {
            if (k == j) continue;
            double f = A[k][j];
            for (int z = j; z < 4; z++) A[k][z] -= f * A[j][z];
        }
    }
    for (int j = 0; j < 3; j++) {
        s->c[j] = A[j][3];
        if (!isfinite(s->c[j])) return;
    }
    double sse1 = 0, sse2 = 0;
    for (int i = 0; i < n; i++) {
        double u = ((px[i] - a.x) * s->ex + (py[i] - a.y) * s->ey) / s->len - 0.5;
        double y = (px[i] - a.x) * s->nx + (py[i] - a.y) * s->ny;
        double r1 = y - line[0] - line[1] * u;
        double r2 = y - s->c[0] - u * (s->c[1] + u * s->c[2]);
        sse1 += wt[i] * r1 * r1;
        sse2 += wt[i] * r2 * r2;
    }
    /* Avoid fitting small sampling variations: at least 0.1 px RMS, with 75% less residual energy. */
    s->qualified = isfinite(sse1) && isfinite(sse2) && sse1 >= 0.01 * m[0] && sse2 <= 0.25 * sse1;
}

static void curve_value(const curve_side *s, double lx, double ly, double dx, double dy,
                        vec2 p, double *value, double *gx, double *gy) {
    if (!s->qualified) {
        *value = (p.x - lx) * (-dy) + (p.y - ly) * dx;
        *gx = -dy;
        *gy = dx;
        return;
    }
    double u = ((p.x - s->a.x) * s->ex + (p.y - s->a.y) * s->ey) / s->len - 0.5;
    *value = (p.x - s->a.x) * s->nx + (p.y - s->a.y) * s->ny - s->c[0] - u * (s->c[1] + u * s->c[2]);
    double slope = (s->c[1] + 2 * u * s->c[2]) / s->len;
    *gx = s->nx - slope * s->ex;
    *gy = s->ny - slope * s->ey;
}

static int intersect_curves(const curve_side sides[4], const vec2 in[4], const vec2 tls[4],
                            const double lx[4], const double ly[4], const double dx[4], const double dy[4],
                            double reach, vec2 result[4]) {
    int qualified = 0;
    for (int k = 0; k < 4; k++) qualified += sides[k].qualified;
    if (!qualified) return 0;
    for (int k = 0; k < 4; k++) {
        int p = (k + 3) & 3, settled = 0;
        vec2 x = tls[k];
        for (int iter = 0; iter < 8; iter++) {
            double f[2], gx[2], gy[2];
            int z[2] = {p, k};
            for (int j = 0; j < 2; j++)
                curve_value(&sides[z[j]], lx[z[j]], ly[z[j]], dx[z[j]], dy[z[j]], x, &f[j], &gx[j], &gy[j]);
            if (!isfinite(f[0]) || !isfinite(f[1]) || !isfinite(gx[0]) || !isfinite(gx[1]) ||
                !isfinite(gy[0]) || !isfinite(gy[1])) return 0;
            double cross = gx[0] * gy[1] - gy[0] * gx[1];
            double norms = (gx[0] * gx[0] + gy[0] * gy[0]) * (gx[1] * gx[1] + gy[1] * gy[1]);
            if (!(norms > 0) || !isfinite(norms) || cross * cross < 0.04 * norms) return 0;
            if (fabs(f[0]) <= 1e-5 && fabs(f[1]) <= 1e-5) {
                settled = 1;
                break;
            }
            x.x -= (f[0] * gy[1] - f[1] * gy[0]) / cross;
            x.y -= (gx[0] * f[1] - gx[1] * f[0]) / cross;
            if (!isfinite(x.x) || !isfinite(x.y) ||
                fabs(x.x - in[k].x) + fabs(x.y - in[k].y) > 4 * reach + 2) return 0;
        }
        if (!settled || fabs(x.x - in[k].x) + fabs(x.y - in[k].y) > 4 * reach + 2) return 0;
        result[k] = x;
    }
    double sign = 0;
    for (int k = 0; k < 4; k++) {
        vec2 a = result[k], b = result[(k + 1) & 3], c = result[(k + 2) & 3];
        double cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        double original = (in[(k + 1) & 3].x - in[k].x) * (in[(k + 2) & 3].y - in[(k + 1) & 3].y) -
                          (in[(k + 1) & 3].y - in[k].y) * (in[(k + 2) & 3].x - in[(k + 1) & 3].x);
        if (!(hypot(b.x - a.x, b.y - a.y) >= 6) || !isfinite(cross) || cross * original <= 0 ||
            (k && cross * sign <= 0)) return 0;
        sign = cross;
    }
    return 1;
}

static NAPA_NOINLINE int confidence_quad(initial_capture *capture, const vec2 rough[4],
                                         double reach, vec2 out[4]) {
    if (capture->rejection_mask == 0) return 0;
    double lx[4],ly[4],dx[4],dy[4];
    for (int k=0;k<4;k++) {
        int n=capture->count[k];
        if(n<3 || n>6)return 0;
        for(int i=0;i<n;i++) capture->weight[k][i]*=capture->weight[k][i];
        if(!fit_side_line(capture->x[k],capture->y[k],capture->weight[k],n,reach,
                         &lx[k],&ly[k],&dx[k],&dy[k]))return 0;
    }
    for(int k=0;k<4;k++) {
        int p=(k+3)&3;
        double cross=dx[p]*dy[k]-dy[p]*dx[k];
        if(!(fabs(cross)>=0.2))return 0;
        double t=((lx[k]-lx[p])*dy[k]-(ly[k]-ly[p])*dx[k])/cross;
        out[k].x=lx[p]+t*dx[p];out[k].y=ly[p]+t*dy[p];
        double move=fabs(out[k].x-rough[k].x)+fabs(out[k].y-rough[k].y);
        if(!(move<=4.0*reach+2.0))return 0;
    }
    return 1;
}
static inline int inside_frame(const image_view *view, vec2 p) {
    return p.x >= 1.0 && p.y >= 1.0 && p.x <= view->w - 2.0 && p.y <= view->h - 2.0;
}

static int compare_doubles(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Gauss-Jordan on an augmented 3x3 matrix, in place. */
static int solve_3x3(double a[3][4], double x[3]) {
    for (int p = 0; p < 3; p++) {
        int pivot = p;
        for (int r = p + 1; r < 3; r++) if (fabs(a[r][p]) > fabs(a[pivot][p])) pivot = r;
        if (fabs(a[pivot][p]) < 1e-12) return 0;
        if (pivot != p) for (int q = 0; q < 4; q++) { double w = a[p][q]; a[p][q] = a[pivot][q]; a[pivot][q] = w; }
        for (int r = 0; r < 3; r++) {
            if (r == p) continue;
            const double f = a[r][p] / a[p][p];
            for (int q = p; q < 4; q++) a[r][q] -= f * a[p][q];
        }
    }
    for (int r = 0; r < 3; r++) x[r] = a[r][3] / a[r][r];
    return 1;
}

/* Every side's final edge samples, kept for fits that need all four sides at once. */
typedef struct {
    int n[4];
    double x[4][NAPA_SIDE_SAMPLES], y[4][NAPA_SIDE_SAMPLES], w[4][NAPA_SIDE_SAMPLES];
} side_samples;

/* Weighted total-least-squares line through one side; returns the weighted squared residual. */
static double side_line(const double *x, const double *y, const double *w, int n, double kappa, double cx, double cy,
                        double scale, double *mx, double *my, double *dx, double *dy) {
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    for (int i = 0; i < n; i++) {
        const double ox = x[i] - cx, oy = y[i] - cy, grow = 1 + kappa * (ox * ox + oy * oy) / (scale * scale);
        const double px = ox * grow, py = oy * grow;
        sw += w[i], sx += w[i] * px, sy += w[i] * py;
        sxx += w[i] * px * px, sxy += w[i] * px * py, syy += w[i] * py * py;
    }
    const double ax = sx / sw, ay = sy / sw;
    const double cxx = sxx - sw * ax * ax, cxy = sxy - sw * ax * ay, cyy = syy - sw * ay * ay;
    const double spread = sqrt((cxx - cyy) * (cxx - cyy) + 4 * cxy * cxy), least = 0.5 * (cxx + cyy - spread);
    if (mx != NULL) {
        const double angle = 0.5 * atan2(2 * cxy, cxx - cyy);
        *mx = ax + cx, *my = ay + cy, *dx = cos(angle), *dy = sin(angle);
    }
    return least > 0 ? least : 0;
}

/* The same samples with four independent quadratics, the most a side can be explained without a shared lens. */
static int side_quadratic_residual(const double *x, const double *y, const double *w, int n, double *sse) {
    double mx, my, dx, dy, lo = HUGE_VAL, hi = -HUGE_VAL, a[3][4] = {{0}}, c[3];
    side_line(x, y, w, n, 0, 0, 0, 1, &mx, &my, &dx, &dy);
    for (int i = 0; i < n; i++) {
        const double u = (x[i] - mx) * dx + (y[i] - my) * dy;
        if (u < lo) lo = u;
        if (u > hi) hi = u;
    }
    const double span = hi > lo ? hi - lo : 1;
    for (int i = 0; i < n; i++) {
        const double u = ((x[i] - mx) * dx + (y[i] - my) * dy) / span, v = (y[i] - my) * dx - (x[i] - mx) * dy;
        const double b[3] = {1, u, u * u};
        for (int r = 0; r < 3; r++) {
            for (int q = 0; q < 3; q++) a[r][q] += w[i] * b[r] * b[q];
            a[r][3] += w[i] * b[r] * v;
        }
    }
    if (!solve_3x3(a, c)) return 0;
    for (int i = 0; i < n; i++) {
        const double u = ((x[i] - mx) * dx + (y[i] - my) * dy) / span, v = (y[i] - my) * dx - (x[i] - mx) * dy;
        const double e = v - c[0] - u * (c[1] + u * c[2]);
        *sse += w[i] * e * e;
    }
    return 1;
}

/* A side cut by the frame can't show its own bow near the cut, but a lens bends all four sides the same way. Find the
 * one radial term that straightens them together, about the image centre, and intersect the straightened sides. */
static int shared_lens_quad(const image_view *im, side_samples *s, const int side_clear[4], vec2 out[4]) {
    const double cx = 0.5 * im->w, cy = 0.5 * im->h, scale = 0.5 * (im->w > im->h ? im->w : im->h);
    double sse_quadratic = 0;
    for (int k = 0; k < 4; k++) {
        double mx, my, dx, dy, e[NAPA_SIDE_SAMPLES], sorted[NAPA_SIDE_SAMPLES];
        int kept = 0;
        for (int i = 0; i < s->n[k]; i++)
            if (s->w[k][i] > 0) s->x[k][kept] = s->x[k][i], s->y[k][kept] = s->y[k][i], s->w[k][kept] = s->w[k][i], kept++;
        if (kept < 6) return 0;
        side_line(s->x[k], s->y[k], s->w[k], kept, 0, 0, 0, 1, &mx, &my, &dx, &dy);
        for (int i = 0; i < kept; i++) sorted[i] = e[i] = (s->y[k][i] - my) * dx - (s->x[k][i] - mx) * dy;
        qsort(sorted, (size_t)kept, sizeof *sorted, compare_doubles);
        const double median = sorted[kept / 2];
        for (int i = 0; i < kept; i++) sorted[i] = fabs(e[i] - median);
        qsort(sorted, (size_t)kept, sizeof *sorted, compare_doubles);
        double limit = 3 * 1.4826 * sorted[kept / 2];
        if (limit < 0.35) limit = 0.35;
        int n = 0;
        for (int i = 0; i < kept; i++)
            if (fabs(e[i]) <= limit) s->x[k][n] = s->x[k][i], s->y[k][n] = s->y[k][i], s->w[k][n] = s->w[k][i], n++;
        if (n < 6 || !side_quadratic_residual(s->x[k], s->y[k], s->w[k], n, &sse_quadratic)) return 0;
        s->n[k] = n;
    }
    enum { STEPS = 41 };
    double cost[STEPS];
    int best = 0;
    for (int g = 0; g < STEPS; g++) {
        const double kappa = -0.4 + 0.02 * g;
        cost[g] = 0;
        for (int k = 0; k < 4; k++) cost[g] += side_line(s->x[k], s->y[k], s->w[k], s->n[k], kappa, cx, cy, scale, NULL, NULL, NULL, NULL);
        if (cost[g] < cost[best]) best = g;
    }
    double kappa = -0.4 + 0.02 * best;
    if (best > 0 && best < STEPS - 1) {
        const double curve = cost[best - 1] - 2 * cost[best] + cost[best + 1];
        if (curve > 0) kappa += 0.5 * (cost[best - 1] - cost[best + 1]) / curve * 0.02;
    }
    double mx[4], my[4], dx[4], dy[4], sse = 0;
    for (int k = 0; k < 4; k++) sse += side_line(s->x[k], s->y[k], s->w[k], s->n[k], kappa, cx, cy, scale, &mx[k], &my[k], &dx[k], &dy[k]);
    /* Straight sides have nothing to gain, and a lens this model can't describe leaves the shared fit well behind. The
     * sides the frame didn't cut must gain too: a cut side's own bend can't vouch for itself. */
    double clear_straight = 0, clear_bent = 0;
    for (int k = 0; k < 4; k++) {
        if (!side_clear[k]) continue;
        clear_straight += side_line(s->x[k], s->y[k], s->w[k], s->n[k], 0, cx, cy, scale, NULL, NULL, NULL, NULL);
        clear_bent += side_line(s->x[k], s->y[k], s->w[k], s->n[k], kappa, cx, cy, scale, NULL, NULL, NULL, NULL);
    }
    if (!(fabs(kappa) >= NAPA_LENS_MIN_KAPPA) || !(sse <= (1 - NAPA_LENS_MIN_GAIN) * cost[20]) || !(sse <= NAPA_LENS_MAX_EXCESS * sse_quadratic) ||
        !(clear_straight > 0) || !(clear_bent <= (1 - NAPA_LENS_MIN_GAIN) * clear_straight)) return 0;
    for (int k = 0; k < 4; k++) {
        const int p = (k + 3) & 3;
        const double cross = dx[p] * dy[k] - dy[p] * dx[k];
        if (fabs(cross) < 0.2) return 0;
        const double t = ((mx[k] - mx[p]) * dy[k] - (my[k] - my[p]) * dx[k]) / cross;
        const double qx = mx[p] + t * dx[p] - cx, qy = my[p] + t * dy[p] - cy;
        double ox = qx, oy = qy;
        for (int iter = 0; iter < 8; iter++) {
            const double grow = 1 + kappa * (ox * ox + oy * oy) / (scale * scale);
            ox = qx / grow, oy = qy / grow;
        }
        out[k].x = ox + cx, out[k].y = oy + cy;
        if (!isfinite(out[k].x) || !isfinite(out[k].y)) return 0;
    }
    return 1;
}


/* Where the profile's area either side of the half level balances over one pixel each way. A level crossing read off
 * bilinear samples moves with a sharp edge's pixel phase, which on a side along the pixel grid doesn't average out. */
static double balanced_crossing(const int64_t *prof, int K, double step, double c, double mid) {
    const double h = 1.0 / step, start = c;
    for (int it = 0; it < 3; it++) {
        const double a = c - h, b = c + h;
        if (!(a >= 0) || !(b <= K - 1)) return start;
        const int ja = (int)a, jb = (int)b;
        const double fa = a - ja, fb = b - jb;
        const double pa = (double)prof[ja] + ((double)prof[ja + (ja < K - 1)] - (double)prof[ja]) * fa;
        const double pb = (double)prof[jb] + ((double)prof[jb + (jb < K - 1)] - (double)prof[jb]) * fb;
        double area = 0, x0 = a, v0 = pa;
        for (int j = ja + 1; j <= jb; j++) {
            area += 0.5 * (v0 + (double)prof[j] - 2 * mid) * (j - x0);
            x0 = j, v0 = (double)prof[j];
        }
        area += 0.5 * (v0 + pb - 2 * mid) * (b - x0);
        if (!(pa - pb > 0)) return start;
        const double move = area / (pa - pb);
        c += move;
        if (fabs(move) < 1e-3) break;
    }
    return fabs(c - start) < NAPA_BALANCE_MAX_MOVE * h ? c : start;
}

/* The same line fit with outliers faded out instead of dropped at a limit, so a point near the limit can't flip the fit. */
static int fit_side_line_soft(const double px[], const double py[], const double wt[], int good, double reach,
                              double *lx, double *ly, double *dx, double *dy) {
    const double limit = NAPA_SOFT_LINE_SCALE * (1.0 + 0.1 * reach);
    double ct = 1, st = 0, mx = 0, my = 0;
    for (int pass = 0; pass < NAPA_SOFT_LINE_PASSES; pass++) {
        double sw = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        double w[NAPA_SIDE_SAMPLES];
        for (int i = 0; i < good; i++) {
            double fall = 1;
            if (pass > 0) {
                const double dist = ((px[i] - mx) * (-st) + (py[i] - my) * ct) / limit;
                fall = 1 / (1 + dist * dist * dist * dist);
            }
            w[i] = wt[i] * fall;
            sw += w[i], sx += w[i] * px[i], sy += w[i] * py[i];
        }
        if (sw <= 0) return 0;
        mx = sx / sw, my = sy / sw;
        for (int i = 0; i < good; i++) {
            const double qx = px[i] - mx, qy = py[i] - my;
            sxx += w[i] * qx * qx, syy += w[i] * qy * qy, sxy += w[i] * qx * qy;
        }
        const double spread = sxx - syy, r = sqrt(spread * spread + 4 * sxy * sxy);
        ct = 1, st = 0;
        if (r > 0) {
            if (spread >= 0) ct = sqrt(0.5 * (1 + spread / r)), st = sxy / (r * ct);
            else st = copysign(sqrt(0.5 * (1 - spread / r)), sxy), ct = sxy / (r * st);
        }
    }
    *lx = mx, *ly = my, *dx = ct, *dy = st;
    return 1;
}

static inline double fade(double x, double lo, double hi);
static NAPA_FORCE_INLINE int fit_sides_impl(const image_view *im, const vec2 in[4], vec2 out[4], double r_out, double r_in, int min_edge,
                     int outermost, int max_points, double *edge_width, edge_levels *levels, double cells, vec2 alternate[4], vec2 curved[4], vec2 lens[4], int *lens_ok, double *width_ratio, initial_capture *capture) {
    const uint8_t *img = im->p;
    const int w = im->w, h = im->h, stride = im->stride;
    double width_sum = 0;
    int width_n = 0;
    double reach = 0.5 * (r_out + r_in);
    enum { MAXP = NAPA_SIDE_SAMPLES, KMAX = 15 };
    /* The first (outermost) pass only has to land within a fraction of a cell for the early
     * decode, so it uses a shorter profile. */
    const int K = outermost ? 11 : KMAX;
    vec2 cen = {0, 0};
    for (int k = 0; k < 4; k++) {
        cen.x += in[k].x * 0.25;
        cen.y += in[k].y * 0.25;
    }
    double lx[4], ly[4], dx[4], dy[4];
    double alx[4], aly[4], adx[4], ady[4];
    int any_alternate = 0;
    curve_side curves[4];
    side_samples samples;
    double side_width[4] = {0, 0, 0, 0};
    int side_widths[4] = {0, 0, 0, 0};
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
        double apx[MAXP], apy[MAXP], awt[MAXP];
        int side_changed = 0;
        int good = 0;
        double step = (r_out + r_in) / (K - 1);
        double margin = NAPA_SIDE_MARGIN;
        if (max_points > 24) {
            margin = NAPA_SIDE_MARGIN_WINDOWS * (r_out + r_in) / len;
            if (margin < NAPA_SIDE_MARGIN_PIXELS / len) margin = NAPA_SIDE_MARGIN_PIXELS / len;
            if (margin < NAPA_SIDE_MIN_MARGIN) margin = NAPA_SIDE_MIN_MARGIN;
            if (margin > NAPA_SIDE_MARGIN) margin = NAPA_SIDE_MARGIN;
        }
        double dt = (1 - 2 * margin) / np, min_g = min_edge * 0.5 * step * 16777216.0; /* best / (2 step) >= min_edge / 4 */
        /* The profile step in 16.16 fixed point, rounded to nearest. */
        double ixd = nx * step * 65536.0, iyd = ny * step * 65536.0;
        int32_t ixp = (int32_t)(ixd < 0 ? ixd - 0.5 : ixd + 0.5), iyp = (int32_t)(iyd < 0 ? iyd - 0.5 : iyd + 0.5);
        unsigned char zone[MAXP]; /* 0: the middle of the side; 1, 2: the ends the coarse passes leave out */
        int tried[3] = {0, 0, 0}, got[3] = {0, 0, 0}, retaken = 0;
        const double kept_width_sum = width_sum, kept_side_width = side_width[k];
        const int kept_width_n = width_n, kept_side_widths = side_widths[k];
retake:
        for (int i = 0; i < np && good + (np - i) >= 3; i++) { /* stop once 3 edge points are out of reach */
            double t = margin + dt * (i + 0.5);
            const int z = t < NAPA_SIDE_MARGIN ? 1 : t > 1 - NAPA_SIDE_MARGIN ? 2 : 0;
            tried[z]++;
            double bx = a.x + (b.x - a.x) * t, by = a.y + (b.y - a.y) * t;
            int64_t prof[(KMAX + 3) & ~3]; /* grey levels in 40.24 fixed point */
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
                profile_fixed(img, stride, fxp, fyp, ixp, iyp, K, prof);
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
             * this is the edge for an isolated symmetric ramp; trimming or squaring the weights biases
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
            double acj = 0;
            int alternate_crossing = 0;
            /* When the edge ramp is wide against a cell (motion or defocus blur), the next edge's ramp
             * overlaps this one and biases the centre of mass. The crossing of the midpoint between
             * the tag's black and white reduces that bias when those local levels are measured accurately.
             * Black is scaled by the local white, so it follows lighting that varies across the tag. */
            if (levels != NULL && levels->black < levels->white &&
                (jr - jl + 1) * step > NAPA_CROSSING_CELLS * len / cells) {
                /* Levels from the centres of the cells either side (white margin, black border), half
                 * a cell from the edge: outside the blurred ramp the window would sit in. */
                if (alternate != NULL) {
                    if (levels->model_pending) {
                        refresh_edge_light(im, in, levels);
                        levels->model_pending = 0;
                    }
                    if (levels->has_model) {
                    double local_mid = levels->midpoint[0] +
                        levels->midpoint[1] * (bx - levels->cx) / levels->scale +
                        levels->midpoint[2] * (by - levels->cy) / levels->scale;
                    double mid = local_mid * 16777216.0;
                    double best_d = 1e9;
                    for (int j = jl - 1; j <= jr && j + 1 < K; j++) {
                        if (j < 0 || !((double)prof[j] >= mid && (double)prof[j + 1] < mid)) continue;
                        double c = j + ((double)prof[j] - mid) / (double)(prof[j] - prof[j + 1]);
                        if (fabs(c - cj) < best_d) {
                            best_d = fabs(c - cj); acj = c; alternate_crossing = 1;
                        }
                    }
                    if (alternate_crossing && len / cells >= NAPA_BALANCE_MIN_CELL) acj = balanced_crossing(prof, K, step, acj, mid);
                    }
                }
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
                    if (settled || iter == NAPA_CROSS_ITERS - 1) {
                        if (len / cells >= NAPA_BALANCE_MIN_CELL) cj = balanced_crossing(prof, K, step, cj, mid);
                        break;
                    }
                }
            }
            width_sum += (jr - jl + 1) * step;
            width_n++;
            side_width[k] += (jr - jl + 1) * step, side_widths[k]++;
            double pos = -r_out + cj * step;
            px[good] = bx + nx * pos;
            py[good] = by + ny * pos;
            wt[good] = (double)best * (1.0 / (1 << 24));
            if (alternate != NULL && (alternate_crossing || side_changed)) {
                if (!side_changed) {
                    memcpy(apx, px, (size_t)good * sizeof(double));
                    memcpy(apy, py, (size_t)good * sizeof(double));
                    memcpy(awt, wt, (size_t)good * sizeof(double));
                }
                double apos = alternate_crossing ? -r_out + acj * step : pos;
                apx[good] = bx + nx * apos; apy[good] = by + ny * apos; awt[good] = wt[good];
                side_changed |= alternate_crossing;
            }
            zone[good] = (unsigned char)z, got[z]++;
            good++;
        }
        if (good < 3) return 0;
        if (margin < NAPA_SIDE_MARGIN && !retaken && got[0] >= 4) {
            /* An end of the side joins the fit only where the border runs through it unbroken: every profile found an
             * edge, each about as strong as the middle of the side's and close to the line through the middle. Anything
             * lying across the border near a corner breaks one of the three, and the side is taken again as before. */
            double sw = 0, su = 0, sv = 0, suu = 0, suv = 0;
            for (int i = 0; i < good; i++) {
                if (zone[i]) continue;
                const double u = (px[i] - a.x) * ex + (py[i] - a.y) * ey, v = (px[i] - a.x) * nx + (py[i] - a.y) * ny;
                sw += wt[i], su += wt[i] * u, sv += wt[i] * v, suu += wt[i] * u * u, suv += wt[i] * u * v;
            }
            const double typical = sw / got[0], den = sw * suu - su * su;
            int broken = got[1] < tried[1] || got[2] < tried[2] || !(den > 0);
            if (!broken) {
                const double slope = (sw * suv - su * sv) / den, offset = (sv - slope * su) / sw;
                double reach_px = NAPA_SIDE_END_REACH_CELLS * len / cells;
                if (reach_px < 0.5) reach_px = 0.5;
                for (int i = 0; i < good && !broken; i++) {
                    if (!zone[i]) continue;
                    const double u = (px[i] - a.x) * ex + (py[i] - a.y) * ey, v = (px[i] - a.x) * nx + (py[i] - a.y) * ny;
                    broken = !(wt[i] >= NAPA_SIDE_END_MIN_STRENGTH * typical && wt[i] <= NAPA_SIDE_END_MAX_STRENGTH * typical) ||
                             fabs(v - offset - slope * u) > reach_px;
                }
            }
            if (broken) {
                retaken = 1, margin = NAPA_SIDE_MARGIN, dt = (1 - 2 * margin) / np, good = 0, side_changed = 0;
                tried[0] = tried[1] = tried[2] = got[0] = got[1] = got[2] = 0;
                width_sum = kept_width_sum, width_n = kept_width_n, side_width[k] = kept_side_width, side_widths[k] = kept_side_widths;
                goto retake;
            }
        }
        /* TLS compacts its arrays, so assess curvature from the original weighted samples first. */
        if (curved != NULL) fit_curve_side(&curves[k], a, b, px, py, wt, good);
        if (lens != NULL) {
            samples.n[k] = good;
            memcpy(samples.x[k], px, (size_t)good * sizeof(double));
            memcpy(samples.y[k], py, (size_t)good * sizeof(double));
            memcpy(samples.w[k], wt, (size_t)good * sizeof(double));
        }
        if (capture != NULL) {
            if (good > 6) return 0;
            capture->count[k]=good;
            for (int i = 0; i < good; i++) {
                capture->x[k][i] = px[i];
                capture->y[k][i] = py[i];
                capture->weight[k][i] = wt[i];
            }
        }
        if (capture != NULL) {
            if (!fit_initial_side_observed(px, py, wt, good, reach, &lx[k], &ly[k], &dx[k], &dy[k],
                                           &capture->rejection_mask, (unsigned)k)) return 0;
        } else {
            if (!fit_side_line(px, py, wt, good, reach, &lx[k], &ly[k], &dx[k], &dy[k])) return 0;
        }
        if (alternate != NULL) {
            any_alternate |= side_changed;
            if (!side_changed || !fit_side_line_soft(apx, apy, awt, good, reach, &alx[k], &aly[k], &adx[k], &ady[k])) {
                alx[k] = lx[k]; aly[k] = ly[k]; adx[k] = dx[k]; ady[k] = dy[k];
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
    if (alternate != NULL && !any_alternate) memcpy(alternate, out, 4 * sizeof(vec2));
    if (alternate != NULL && any_alternate) {
        int valid = 1;
        for (int k = 0; k < 4; k++) {
            int p = (k + 3) & 3;
            double cross = adx[p] * ady[k] - ady[p] * adx[k];
            if (fabs(cross) < 0.2) { valid = 0; break; }
            double ddx = alx[k] - alx[p], ddy = aly[k] - aly[p];
            double t = (ddx * ady[k] - ddy * adx[k]) / cross;
            alternate[k].x = alx[p] + t * adx[p];
            alternate[k].y = aly[p] + t * ady[p];
            double mv = fabs(alternate[k].x - in[k].x) + fabs(alternate[k].y - in[k].y);
            if (!(mv <= 4.0 * reach + 2.0)) { valid = 0; break; }
        }
        if (!valid) memcpy(alternate, out, 4 * sizeof(vec2));
    }
    if (curved != NULL) {
        vec2 proposal[4];
        memcpy(curved, out, 4 * sizeof(vec2));
        if (intersect_curves(curves, in, out, lx, ly, dx, dy, reach, proposal)) memcpy(curved, proposal, sizeof proposal);
    }
    if (width_ratio != NULL) {
        double narrow = HUGE_VAL, wide = 0;
        for (int k = 0; k < 4; k++) {
            const double mean = side_widths[k] ? side_width[k] / side_widths[k] : 0;
            if (mean < narrow) narrow = mean;
            if (mean > wide) wide = mean;
        }
        width_ratio[0] = narrow > 0 ? wide / narrow : 1;
        width_ratio[1] = wide;
    }
    if (lens != NULL) {
        int near_frame = 0, corner_clear[4], side_clear[4];
        for (int k = 0; k < 4; k++) {
            corner_clear[k] = inside_frame(im, out[k]) && inside_frame(im, curved[k]);
            near_frame |= !corner_clear[k];
        }
        for (int k = 0; k < 4; k++) side_clear[k] = corner_clear[k] && corner_clear[(k + 1) & 3];
        *lens_ok = near_frame && shared_lens_quad(im, &samples, side_clear, lens);
    }
    return 1;
}

static int fit_sides(const image_view *im, const vec2 in[4], vec2 out[4], double r_out, double r_in, int min_edge,
                     int outermost, int max_points, double *edge_width, edge_levels *levels, double cells) {
    return fit_sides_impl(im, in, out, r_out, r_in, min_edge, outermost, max_points, edge_width, levels, cells, NULL, NULL, NULL, NULL, NULL, NULL);
}

static int fit_initial_captured(const image_view *im, const vec2 in[4], vec2 out[4],
                                double r_out, double r_in, int min_edge, double *width,
                                double cells, initial_capture *capture) {
    if (capture != NULL) capture->rejection_mask = 0;
    /* A short later side will fail the same fit after sampling earlier sides.
     * Check those lengths before spending profiles; side zero is checked first
     * by the fitter itself. Failed initial-fit outputs are never consumed. */
    for (int k = 1; k < 4; k++) {
        double x = in[(k + 1) & 3].x - in[k].x, y = in[(k + 1) & 3].y - in[k].y;
        if (x * x + y * y < 36.0) return 0;
    }
    return fit_sides_impl(im,in,out,r_out,r_in,min_edge,1,6,width,NULL,cells,NULL,NULL,NULL,NULL,NULL,capture);
}

static int fit_sides_alternate(const image_view *im, const vec2 in[4], vec2 out[4], double r_out, double r_in, int min_edge,
                     int outermost, int max_points, double *edge_width, edge_levels *levels, double cells, vec2 alternate[4], vec2 curved[4], vec2 lens[4], int *lens_ok, double *width_ratio) {
    return fit_sides_impl(im, in, out, r_out, r_in, min_edge, outermost, max_points, edge_width, levels, cells, alternate, curved, lens, lens_ok, width_ratio, NULL);
}

/* The shared-white model denoises geometry after a tag already decodes.
 * When ordinary final decoding fails, retain recovery from independent light
 * planes. This cold refit uses the same input quad, image, samples and window;
 * it does not rerun proposals or add a frame pass. */
static NAPA_NOINLINE int fit_independent_alternate(const image_view *im, const vec2 in[4], vec2 alternate[4],
                         double reach, int min_edge, double cells, const edge_levels *levels) {
    edge_levels independent = *levels;
    memcpy(independent.midpoint, levels->independent_midpoint, sizeof independent.midpoint);
    independent.model_pending = 0;
    vec2 ordinary[4];
    double width = 0;
    return fit_sides_alternate(im, in, ordinary, reach, reach, min_edge, 0, NAPA_FINAL_POINTS,
                              &width, &independent, cells, alternate, NULL, NULL, NULL, NULL);
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
/* Project/sample two points with the scalar operation order and NaN-safe clamps.
 * All families, including inverted families, retain the same floating-point values. */
static void sample_points_impl(const image_view *im, const double H[9], double *u, double *v, int n,
                               double *out, vec2 *rawcoords) {
    int i = 0;
#ifdef NAPA_SSE2
    const uint8_t *img = im->p;
    const int stride = im->stride;
    const __m128d h0 = _mm_set1_pd(H[0]), h1 = _mm_set1_pd(H[1]), h2 = _mm_set1_pd(H[2]), h3 = _mm_set1_pd(H[3]);
    const __m128d h4 = _mm_set1_pd(H[4]), h5 = _mm_set1_pd(H[5]), h6 = _mm_set1_pd(H[6]), h7 = _mm_set1_pd(H[7]);
    const __m128d h8 = _mm_set1_pd(H[8]), one = _mm_set1_pd(1.0), zero = _mm_setzero_pd();
    const __m128d xmax = _mm_set1_pd(im->w - 1.001), ymax = _mm_set1_pd(im->h - 1.001);
    for (; i + 2 <= n; i += 2) {
        __m128d U = _mm_loadu_pd(u + i), V = _mm_loadu_pd(v + i);
        __m128d iz = _mm_div_pd(one, _mm_add_pd(_mm_add_pd(_mm_mul_pd(h6, U), _mm_mul_pd(h7, V)), h8));
        __m128d x = _mm_mul_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(h0, U), _mm_mul_pd(h1, V)), h2), iz);
        __m128d y = _mm_mul_pd(_mm_add_pd(_mm_add_pd(_mm_mul_pd(h3, U), _mm_mul_pd(h4, V)), h5), iz);
        if (rawcoords) {
            rawcoords[i].x = _mm_cvtsd_f64(x);
            rawcoords[i].y = _mm_cvtsd_f64(y);
            rawcoords[i + 1].x = _mm_cvtsd_f64(_mm_unpackhi_pd(x, x));
            rawcoords[i + 1].y = _mm_cvtsd_f64(_mm_unpackhi_pd(y, y));
        }
        x = _mm_and_pd(_mm_cmpge_pd(x, zero), x); /* !(x >= 0): 0 */
        y = _mm_and_pd(_mm_cmpge_pd(y, zero), y);
        __m128d in = _mm_cmple_pd(x, xmax); /* !(x <= w - 1.001): w - 1.001 */
        x = _mm_or_pd(_mm_and_pd(in, x), _mm_andnot_pd(in, xmax));
        in = _mm_cmple_pd(y, ymax);
        y = _mm_or_pd(_mm_and_pd(in, y), _mm_andnot_pd(in, ymax));
        __m128i xi = _mm_cvttpd_epi32(x), yi = _mm_cvttpd_epi32(y);
        const uint8_t *pa = img + (size_t)_mm_cvtsi128_si32(yi) * stride + _mm_cvtsi128_si32(xi);
        const uint8_t *pb = img + (size_t)_mm_cvtsi128_si32(_mm_srli_si128(yi, 4)) * stride +
                            _mm_cvtsi128_si32(_mm_srli_si128(xi, 4));
        __m128d p0 = _mm_cvtepi32_pd(_mm_setr_epi32(pa[0], pb[0], 0, 0));
        __m128d d01 = _mm_cvtepi32_pd(_mm_setr_epi32(pa[1] - pa[0], pb[1] - pb[0], 0, 0));
        __m128d p2 = _mm_cvtepi32_pd(_mm_setr_epi32(pa[stride], pb[stride], 0, 0));
        __m128d d23 = _mm_cvtepi32_pd(_mm_setr_epi32(pa[stride + 1] - pa[stride], pb[stride + 1] - pb[stride], 0, 0));
        __m128d fx = _mm_sub_pd(x, _mm_cvtepi32_pd(xi)), fy = _mm_sub_pd(y, _mm_cvtepi32_pd(yi));
        __m128d a = _mm_add_pd(p0, _mm_mul_pd(d01, fx)), b = _mm_add_pd(p2, _mm_mul_pd(d23, fx));
        __m128d val = _mm_add_pd(a, _mm_mul_pd(_mm_sub_pd(b, a), fy));
        if (im->invert) val = _mm_sub_pd(_mm_set1_pd(255.0), val);
        _mm_storeu_pd(out + i, val);
    }
#endif
    for (; i < n; i++) {
        vec2 q = project(H, u[i], v[i]);
        if (rawcoords) rawcoords[i] = q;
        out[i] = sample_bilinear(im, q.x, q.y);
    }
}

static void sample_points(const image_view *im, const double H[9], double *u, double *v, int n, double *out) {
    sample_points_impl(im, H, u, v, n, out, NULL);
}

/* Non-mutating median of an even number of finite samples, 2 <= n <= 68.
 * Partition into separate buffers; repeated values occupy the pivot's equal range. */
static double median_even(const double *v, int n) {
    double buf[3][68];
    const double *src = v;
    int m = n, k = n / 2 - 1, cur = -1;
    double above = HUGE_VAL; /* the least value dropped above the range so far */
    for (;;) {
        const double p = src[m / 2];
        const int lo_b = cur == 0 ? 1 : 0, hi_b = cur == 2 || cur < 0 ? 1 : 2; /* the buffers src isn't */
        double *lo = buf[lo_b], *hi = buf[hi_b];
        int nl = 0, nh = 0;
        for (int i = 0; i < m; i++) {
            const double x = src[i];
            lo[nl] = x, hi[nh] = x;
            nl += x < p, nh += x > p;
        }
        const int ne = m - nl - nh; /* at least 1: the pivot itself */
        if (k < nl) {
            above = p, src = lo, m = nl, cur = lo_b;
        } else if (k < nl + ne) {
            double next = above;
            if (k + 1 < nl + ne) next = p;
            else
                for (int i = 0; i < nh; i++) next = hi[i] < next ? hi[i] : next;
            return 0.5 * (p + next);
        } else {
            k -= nl + ne, src = hi, m = nh, cur = hi_b;
        }
    }
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
                       double *delta, double *disagreement) {
    enum { N = 33 };
    double cell_d[NAPA_BORDER_CELLS_MAX], mean_cell = 0;
    unsigned char cell_side[NAPA_BORDER_CELLS_MAX];
    double H[9], sum = 0, side_sum[4] = {0, 0, 0, 0};
    int n = 0, side_n[4] = {0, 0, 0, 0};
    square_to_quad(q, H);
    double mid = 0.5 * (black + white);
    for (int side = 0; side < 4; side++) {
        /* grow_quad moves sides along their normals; measure border widths in those same units.
         * A projected cell's direction is generally oblique to its edge under perspective. */
        double nx = q[side].y - q[(side + 1) & 3].y;
        double ny = q[(side + 1) & 3].x - q[side].x;
        double side_len = hypot(nx, ny);
        nx /= side_len; ny /= side_len;
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
            double payload = sample_bilinear(im, pc.x, pc.y);
            if (payload < mid) continue; /* dark payload cell */
            double prof[N], us[N - 1], vs[N - 1];
            vec2 pts[N];
            for (int j = 0; j < N - 1; j++) {
                UNIT(-0.5 + 2.0 * j / (N - 1), us[j], vs[j]);
            }
            sample_points_impl(im, H, us, vs, N - 1, prof, pts);
            pts[N - 1] = pc;
            prof[N - 1] = payload;
            /* Outer edge within half a cell of t = 0, inner edge within half a cell of t = 1. */
            double jo = lobe_centroid(prof, 1, N / 2 - 1, 1), ji = lobe_centroid(prof, N / 2 + 1, N - 2, 0);
            if (jo < 0 || ji < 0) continue;
            vec2 a0 = pts[(int)jo], a1 = pts[(int)jo + 1], b0 = pts[(int)ji], b1 = pts[(int)ji + 1];
            double fo = jo - (int)jo, fi = ji - (int)ji;
            double width_px = fabs((b0.x + (b1.x - b0.x) * fi - a0.x - (a1.x - a0.x) * fo) * nx +
                                   (b0.y + (b1.y - b0.y) * fi - a0.y - (a1.y - a0.y) * fo) * ny);
            vec2 e0 = pts[8], e1 = pts[24]; /* t = 0 and t = 1 */
            double cell_px = fabs((e1.x - e0.x) * nx + (e1.y - e0.y) * ny);
            double d = (cell_px - width_px) / 1.75;
            if (n < NAPA_BORDER_CELLS_MAX) cell_d[n] = d, cell_side[n] = (unsigned char)side;
            n++;
#undef UNIT
        }
    }
    if (!n) return 0;
    if (n > NAPA_BORDER_CELLS_MAX) n = NAPA_BORDER_CELLS_MAX;
    /* Something lying across the border (a cable, a strap) spoils the cells under it by a large part of a cell, while
     * clipping, gamma and lenses move every cell by nearly the same amount. Leave out cells far from the middle value. */
    double sorted[NAPA_BORDER_CELLS_MAX];
    memcpy(sorted, cell_d, (size_t)n * sizeof(double));
    for (int i = 1; i < n; i++) {
        const double v = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > v) sorted[j + 1] = sorted[j], j--;
        sorted[j + 1] = v;
    }
    const double middle = n & 1 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    for (int k = 0; k < 4; k++) mean_cell += 0.25 * hypot(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y) / L->cells;
    double reach = NAPA_BORDER_CELL_REACH * mean_cell;
    if (reach < 0.1) reach = 0.1;
    int kept = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(cell_d[i] - middle) > reach) continue;
        sum += cell_d[i], side_sum[cell_side[i]] += cell_d[i], side_n[cell_side[i]]++, kept++;
    }
    n = kept;
    if (!n) return 0;
    *delta = sum / n;
    /* Clipping moves every side alike. Sides that disagree mean the border simply isn't one cell
     * wide (a rasterised tag whose cells were rounded to whole pixels): leave such a tag alone. */
    /* disagreement: the worst side's distance from the mean, as a share of what is allowed (1 = at the limit). */
    *disagreement = 0;
    for (int side = 0; side < 4; side++) {
        if (!side_n[side]) continue;
        const double share = fabs(side_sum[side] / side_n[side] - *delta) / ((0.05 + 0.25 * fabs(*delta)) * L->bias_slack);
        if (share > *disagreement) *disagreement = share;
    }
    return n;
}

/* A tone bias (clipping, gamma under blur) moves every black-white edge of the tag alike, payload
 * edges included. A border that simply is not one cell wide moves only its own inner edge. Measure how far the
 * payload's own edges sit from the cell grid, signed so + means black regions are smaller, in pixels per edge.
 * Edges entering and leaving black are averaged separately so a small grid shift cancels. */
static int payload_bias(const family_layout *L, const image_view *im, const vec2 q[4], double black, double white, double *delta) {
    enum { PER = 3, MAXC = 16 };
    double H[9], A[2][3][3] = {{{0}}}, B[2][3] = {{0}};
    int n[2] = {0, 0};
    square_to_quad(q, H);
    const double mid = 0.5 * (black + white);
    const int cells = L->cells, steps = (cells - 3) * PER;
    for (int axis = 0; axis < 2; axis++)
        for (int line = 1; line < cells - 1; line++) {
            double prev = 0, xprev = 0, xs[MAXC], px[MAXC];
            int dark_after[MAXC], count = 0;
            const double across = (line + 0.5) / cells;
            for (int j = 0; j <= steps && count < MAXC; j++) {
                const double x = 1.5 + (double)j / PER;
                const vec2 p = axis ? project(H, across, x / cells) : project(H, x / cells, across);
                const double val = sample_bilinear(im, p.x, p.y);
                if (j > 0 && (prev < mid) != (val < mid)) {
                    double xa = xprev, xb = x, va = prev, vb = val;
                    for (int it = 0; it < 5; it++) {
                        const double xm = 0.5 * (xa + xb);
                        const vec2 pm = axis ? project(H, across, xm / cells) : project(H, xm / cells, across);
                        const double vm = sample_bilinear(im, pm.x, pm.y);
                        if ((vm < mid) == (va < mid)) xa = xm, va = vm; else xb = xm, vb = vm;
                    }
                    xs[count] = vb != va ? xa + (mid - va) / (vb - va) * (xb - xa) : 0.5 * (xa + xb);
                    const vec2 p0 = axis ? project(H, across, (xs[count] - 0.5) / cells) : project(H, (xs[count] - 0.5) / cells, across);
                    const vec2 p1 = axis ? project(H, across, (xs[count] + 0.5) / cells) : project(H, (xs[count] + 0.5) / cells, across);
                    px[count] = hypot(p1.x - p0.x, p1.y - p0.y);
                    dark_after[count++] = val < mid;
                }
                prev = val, xprev = x;
            }
            for (int k = 0; k < count; k++) {
                const double g = floor(xs[k] + 0.5), off = (xs[k] - g) * px[k];
                if (fabs(xs[k] - g) >= 0.35) continue;
                /* off = a + b (g - centre) + delta * (+1 entering black, -1 leaving it): a and b absorb a shifted or
                 * stretched grid, delta is the part that depends only on which side is black. */
                const double row[3] = {1, g - 0.5 * cells, dark_after[k] ? 1 : -1};
                for (int i = 0; i < 3; i++) {
                    for (int j2 = 0; j2 < 3; j2++) A[axis][i][j2] += row[i] * row[j2];
                    B[axis][i] += row[i] * off;
                }
                n[dark_after[k]]++;
            }
        }
    if (n[0] < 6 || n[1] < 6) return 0;
    double total = 0, weight = 0;
    for (int axis = 0; axis < 2; axis++) {
        double (*M)[3] = A[axis], *r = B[axis];
        const double det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                           M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        if (fabs(det) < 1e-6 * M[0][0] * M[1][1] * M[2][2] || M[0][0] < 6) continue;
        const double d2 = M[0][0] * (M[1][1] * r[2] - r[1] * M[2][1]) - M[0][1] * (M[1][0] * r[2] - r[1] * M[2][0]) +
                          r[0] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        total += M[0][0] * d2 / det, weight += M[0][0];
    }
    if (weight <= 0) return 0;
    *delta = total / weight;
    return n[0] + n[1];
}

/* Where a strongly curved tag touches a neighbour at a corner (an X-corner on a board seen through a wide
 * lens), lines fitted along the whole side miss the corner. Gradients around an X-corner point at it, so
 * solve for the spot every nearby gradient is orthogonal to the offset from. Returns 1 on convergence. */
/* Where the dark-to-light edge crosses half level on a short ray through (x, y), as signed distance along the ray. */
static int edge_crossing(const image_view *view, double x, double y, double nx, double ny, double *offset) {
    enum { N = 13 };
    const double start = -3.0, step = 0.5;
    double v[N], best = 0;
    int peak = -1;
    for (int i = 0; i < N; i++) v[i] = sample_bilinear(view, x + (start + step * i) * nx, y + (start + step * i) * ny);
    for (int i = 2; i < N - 2; i++)
        if (v[i + 1] - v[i - 1] > best) best = v[i + 1] - v[i - 1], peak = i;
    if (peak < 0) return 0;
    double lo = v[peak], hi = v[peak];
    for (int i = peak > 4 ? peak - 4 : 0; i <= peak; i++) if (v[i] < lo) lo = v[i];
    for (int i = peak; i < N && i <= peak + 4; i++) if (v[i] > hi) hi = v[i];
    if (hi - lo < NAPA_LOCAL_EDGE_CONTRAST) return 0;
    const double half = 0.5 * (lo + hi);
    int j = peak;
    while (j > 0 && v[j] > half) j--;
    while (j < N - 1 && v[j + 1] <= half) j++;
    if (j >= N - 1 || v[j + 1] == v[j]) return 0;
    *offset = start + step * j + step * (half - v[j]) / (v[j + 1] - v[j]);
    return 1;
}

/* Robust quadratic y = c0 + c1 u + c2 u^2 through a run of edge samples, with u the distance divided by scale. */
static int fit_edge_run(const double *t, const double *y, int n, double scale, double c[3]) {
    unsigned char keep[NAPA_LOCAL_EDGE_SAMPLES];
    memset(keep, 1, (size_t)n);
    for (int pass = 0; pass < 3; pass++) {
        double a[3][4] = {{0}};
        int used = 0;
        for (int i = 0; i < n; i++) {
            if (!keep[i]) continue;
            const double u = t[i] / scale, b[3] = {1, u, u * u};
            for (int r = 0; r < 3; r++) {
                for (int q = 0; q < 3; q++) a[r][q] += b[r] * b[q];
                a[r][3] += b[r] * y[i];
            }
            used++;
        }
        if (used < 5) return 0;
        if (!solve_3x3(a, c)) return 0;
        double residual[NAPA_LOCAL_EDGE_SAMPLES], sorted[NAPA_LOCAL_EDGE_SAMPLES];
        for (int i = 0; i < n; i++) {
            const double u = t[i] / scale;
            residual[i] = fabs(y[i] - (c[0] + u * (c[1] + u * c[2])));
            sorted[i] = residual[i];
        }
        qsort(sorted, (size_t)n, sizeof *sorted, compare_doubles);
        double cutoff = 1.5 * sorted[n / 2];
        if (cutoff < 0.08) cutoff = 0.08;
        cutoff *= 2.5;
        for (int i = 0; i < n; i++) keep[i] = residual[i] < cutoff;
    }
    return 1;
}

/* The two sides meeting at corner k: their directions, outward normals, and where along them sampling can begin,
 * far enough out that the edges don't blur into each other. */
typedef struct {
    double ux[2], uy[2], nx[2], ny[2], near_end;
} corner_sides;

static int measure_corner_sides(const vec2 q[4], int k, double cell, corner_sides *f) {
    const vec2 c = q[k], ends[2] = {q[(k + 1) & 3], q[(k + 3) & 3]};
    const double mx = 0.25 * (q[0].x + q[1].x + q[2].x + q[3].x), my = 0.25 * (q[0].y + q[1].y + q[2].y + q[3].y);
    for (int e = 0; e < 2; e++) {
        const double len = hypot(ends[e].x - c.x, ends[e].y - c.y);
        if (len < 1) return 0;
        f->ux[e] = (ends[e].x - c.x) / len, f->uy[e] = (ends[e].y - c.y) / len;
        f->nx[e] = -f->uy[e], f->ny[e] = f->ux[e];
        if (f->nx[e] * (c.x + 5 * f->ux[e] - mx) + f->ny[e] * (c.y + 5 * f->uy[e] - my) < 0)
            f->nx[e] = -f->nx[e], f->ny[e] = -f->ny[e];
    }
    double dot = f->ux[0] * f->ux[1] + f->uy[0] * f->uy[1];
    dot = dot > 1 ? 1 : dot < -1 ? -1 : dot;
    double wedge = 2 * tan(0.5 * acos(dot));
    if (wedge < 0.2) wedge = 0.2;
    f->near_end = 3.5 / wedge;
    if (f->near_end < 0.45 * cell) f->near_end = 0.45 * cell;
    return 1;
}

/* 0 below lo, 1 above hi, straight in between: a decision that fades in instead of switching. */
static inline double fade(double x, double lo, double hi) { return x <= lo ? 0 : x >= hi ? 1 : (x - lo) / (hi - lo); }

/* How far the edges beside the corner sit from where the straight sides say, in pixels (the larger of the two sides). */
static double corner_sides_offset(const image_view *view, const vec2 c, const corner_sides *f, double cell, double shift) {
    double worst = 0;
    for (int e = 0; e < 2; e++) {
        double sum = 0;
        int n = 0;
        for (int i = 0; i < 3; i++) {
            const double d = f->near_end + 1.5 * cell * i;
            double offset;
            if (edge_crossing(view, c.x + d * f->ux[e], c.y + d * f->uy[e], f->nx[e], f->ny[e], &offset))
                sum += offset + shift, n++;
        }
        if (n && fabs(sum / n) > worst) worst = fabs(sum / n);
    }
    return worst;
}

/* Re-find a corner from the two edge runs beside it. A fit along a whole side averages its bow, and a sharp corner
 * magnifies what's left; short runs follow the edges right where they meet. shift is the sides' measured bias. */
static int refit_corner(const image_view *view, const vec2 c, const corner_sides *f, double cell, double shift, vec2 *out) {
    double fits[2][3];
    const double scale = f->near_end + 3 * cell;
    double spacing = 3 * cell / NAPA_LOCAL_EDGE_SAMPLES;
    if (spacing < 1) spacing = 1;
    for (int e = 0; e < 2; e++) {
        double t[NAPA_LOCAL_EDGE_SAMPLES], y[NAPA_LOCAL_EDGE_SAMPLES];
        int n = 0;
        for (double d = f->near_end; d < scale && n < NAPA_LOCAL_EDGE_SAMPLES; d += spacing) {
            double offset;
            if (edge_crossing(view, c.x + d * f->ux[e], c.y + d * f->uy[e], f->nx[e], f->ny[e], &offset))
                t[n] = d, y[n] = offset, n++;
        }
        if (n < 8 || !fit_edge_run(t, y, n, scale, fits[e])) return 0;
        fits[e][0] += shift;
    }
    double a = 0, b = 0;
    for (int iter = 0; iter < 10; iter++) {
        const double ua = a / scale, ub = b / scale;
        const double ya = fits[0][0] + ua * (fits[0][1] + ua * fits[0][2]), yb = fits[1][0] + ub * (fits[1][1] + ub * fits[1][2]);
        const double sa = (fits[0][1] + 2 * ua * fits[0][2]) / scale, sb = (fits[1][1] + 2 * ub * fits[1][2]) / scale;
        const double rx = b * f->ux[1] + yb * f->nx[1] - a * f->ux[0] - ya * f->nx[0];
        const double ry = b * f->uy[1] + yb * f->ny[1] - a * f->uy[0] - ya * f->ny[0];
        const double dax = f->ux[0] + sa * f->nx[0], day = f->uy[0] + sa * f->ny[0];
        const double dbx = f->ux[1] + sb * f->nx[1], dby = f->uy[1] + sb * f->ny[1];
        const double det = -dax * dby + dbx * day;
        if (fabs(det) < 1e-6) return 0;
        const double da = (-rx * dby + dbx * ry) / det, db = (dax * ry - day * rx) / det;
        a += da, b += db;
        if (fabs(da) + fabs(db) < 1e-3) break;
    }
    const double ua = a / scale, ya = fits[0][0] + ua * (fits[0][1] + ua * fits[0][2]);
    const double dx = a * f->ux[0] + ya * f->nx[0], dy = a * f->uy[0] + ya * f->ny[0];
    if (!(fabs(a) < 6 && fabs(b) < 6) || !(hypot(dx, dy) < 4)) return 0;
    out->x = c.x + dx, out->y = c.y + dy;
    return 1;
}

static int refine_x_corner(const image_view *view, double *x, double *y) {
    enum { WINDOW = 3 };
    const double start_x = *x, start_y = *y;
    double cx = start_x, cy = start_y;
    for (int iter = 0; iter < 20; iter++) {
        double a = 0, b = 0, c = 0, b1 = 0, b2 = 0;
        for (int j = -WINDOW; j <= WINDOW; j++) {
            for (int i = -WINDOW; i <= WINDOW; i++) {
                double px = cx + i, py = cy + j;
                double tx = sample_bilinear(view, px + 1, py) - sample_bilinear(view, px - 1, py);
                double ty = sample_bilinear(view, px, py + 1) - sample_bilinear(view, px, py - 1);
                double m = exp(-(double)(i * i + j * j) / (WINDOW * WINDOW));
                double gxx = tx * tx * m, gxy = tx * ty * m, gyy = ty * ty * m;
                a += gxx, b += gxy, c += gyy;
                b1 += gxx * i + gxy * j;
                b2 += gxy * i + gyy * j;
            }
        }
        double det = a * c - b * b;
        if (fabs(det) < 1e-6) return 0;
        double dx = (c * b1 - b * b2) / det, dy = (a * b2 - b * b1) / det;
        cx += dx, cy += dy;
        if (fabs(cx - start_x) > WINDOW || fabs(cy - start_y) > WINDOW) return 0;
        if (dx * dx + dy * dy < 1e-4) break;
    }
    *x = cx, *y = cy;
    return 1;
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
                       double *grid, double *border, double *black_out, double *white_out, double *ring_out) {
    const int W = L->width, o = L->outer, G = L->grid;
    const double cells = L->cells;
    double ring[4 * 16 + 4], u[4 * 16 + 4], v[4 * 16 + 4];
    int nb = 0, nr = 0;
    for (int r = 0; r < W; r++) {
        for (int c = 0; c < W; c += (r == 0 || r == W - 1) ? 1 : W - 1) {
            u[nb] = (c + 0.5) / cells, v[nb] = (r + 0.5) / cells;
            nb++;
        }
    }
    sample_points(im, H, u, v, nb, border);
    for (int r = 0, k = 0; r < W; r++)
        for (int c = 0; c < W; c += (r == 0 || r == W - 1) ? 1 : W - 1) grid[(r + o) * G + c + o] = border[k++];
    for (int i = 0; i < W + 2; i++) {
        u[nr] = (i - 0.5) / cells, v[nr++] = -0.5 / cells;
        u[nr] = (i - 0.5) / cells, v[nr++] = (W + 0.5) / cells;
    }
    for (int i = 1; i < W + 1; i++) {
        u[nr] = -0.5 / cells, v[nr++] = (i - 0.5) / cells;
        u[nr] = (W + 0.5) / cells, v[nr++] = (i - 0.5) / cells;
    }
    sample_points(im, H, u, v, nr, ring);
    if (ring_out != NULL) memcpy(ring_out, ring, (size_t)nr * sizeof(double));
    double black = median_even(border, nb);
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
    return read_border(L, im, H, min_contrast, grid, border, &black, &white, NULL) >= min_dark;
}


/* Fit a local illumination plane with bounded asymmetric least-squares iterations.
 * Mixed samples are biased toward the other colour: downweight dark white-ring samples and
 * bright black-border samples. This estimates local levels; it does not assume a flat exposure. */
static int light_plane(double points[][3], int n, int white, double out[3]) {
    out[0] = out[1] = out[2] = 0;
    for (int pass = 0; pass < 4; pass++) {
        double a[3][4] = {{0}};
        for (int i = 0; i < n; i++) {
            double x[3] = {1, points[i][0], points[i][1]}, value = points[i][2];
            double residual = value - (out[0] + out[1] * x[1] + out[2] * x[2]);
            double weight = pass == 0 ? 1.0 : ((white ? residual < 0 : residual > 0) ? 0.1 : 0.9);
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) a[r][c] += weight * x[r] * x[c];
                a[r][3] += weight * x[r] * value;
            }
        }
        for (int c = 0; c < 3; c++) {
            int pivot = c;
            for (int r = c + 1; r < 3; r++) if (fabs(a[r][c]) > fabs(a[pivot][c])) pivot = r;
            if (fabs(a[pivot][c]) < 1e-9) return 0;
            for (int k = c; k < 4; k++) { double t = a[c][k]; a[c][k] = a[pivot][k]; a[pivot][k] = t; }
            double scale = a[c][c];
            for (int k = c; k < 4; k++) a[c][k] /= scale;
            for (int r = 0; r < 3; r++) if (r != c) {
                double scale = a[r][c];
                for (int k = c; k < 4; k++) a[r][k] -= scale * a[c][k];
            }
        }
        for (int k = 0; k < 3; k++) out[k] = a[k][3];
    }
    return 1;
}

static void edge_light_model(const family_layout *L, const double H[9], const vec2 q[4],
                             const double *ring, const double *border, edge_levels *levels) {
    double bp[64][3], wp[68][3], b[3], w[3];
    int nb = 0, nw = 0, W = L->width;
    levels->cx = levels->cy = levels->scale = 0;
    for (int k = 0; k < 4; k++) {
        levels->cx += q[k].x * 0.25; levels->cy += q[k].y * 0.25;
        levels->scale += hypot(q[k].x-q[(k+1)&3].x, q[k].y-q[(k+1)&3].y) * 0.25;
    }
    if (!(levels->scale > 0)) return;
    for (int r = 0; r < W; r++) for (int c = 0; c < W; c += (r == 0 || r == W - 1) ? 1 : W - 1) {
        vec2 pos = project(H, (c+0.5)/L->cells, (r+0.5)/L->cells);
        bp[nb][0] = (pos.x-levels->cx)/levels->scale; bp[nb][1] = (pos.y-levels->cy)/levels->scale;
        bp[nb][2] = border[nb]; nb++;
    }
    for (int i = 0; i < W + 2; i++) for (int side = 0; side < 2; side++) {
        vec2 pos = project(H, (i-0.5)/L->cells, (side ? W+0.5 : -0.5)/L->cells);
        wp[nw][0] = (pos.x-levels->cx)/levels->scale; wp[nw][1] = (pos.y-levels->cy)/levels->scale;
        wp[nw][2] = ring[nw]; nw++;
    }
    for (int i = 1; i < W + 1; i++) for (int side = 0; side < 2; side++) {
        vec2 pos = project(H, (side ? W+0.5 : -0.5)/L->cells, (i-0.5)/L->cells);
        wp[nw][0] = (pos.x-levels->cx)/levels->scale; wp[nw][1] = (pos.y-levels->cy)/levels->scale;
        wp[nw][2] = ring[nw]; nw++;
    }
    if (light_plane(bp, nb, 0, b) && light_plane(wp, nw, 1, w)) {
        for (int k = 0; k < 3; k++) if (!isfinite(b[k]) || !isfinite(w[k])) return;
        for (int k = 0; k < 4; k++) {
            double x = (q[k].x - levels->cx) / levels->scale, y = (q[k].y - levels->cy) / levels->scale;
            double black = b[0] + b[1] * x + b[2] * y, white = w[0] + w[1] * x + w[2] * y;
            if (!(black >= 0 && white <= 255 && white > black)) return;
        }
        for (int k = 0; k < 3; k++) levels->midpoint[k] = 0.5 * (b[k] + w[k]);
        /* Thin border samples mix with payload under blur. For normal print,
         * use the white quiet ring's spatial drift, retaining both fitted levels.
         * Image inversion does not preserve this multiplicative-light model. */
        if (!L->f->reversed) {
            memcpy(levels->independent_midpoint, levels->midpoint, sizeof levels->midpoint);
            double scale = levels->midpoint[0] / w[0]; /* validated white > black >= 0 at every corner */
            for (int k = 1; k < 3; k++) levels->midpoint[k] = scale * w[k];
        }
        levels->has_model = 1;
    }
}

static NAPA_NOINLINE void refresh_edge_light(const image_view *im, const vec2 q[4], edge_levels *levels) {
    double H[9], grid[256], border[64], ring[68], black, white;
    square_to_quad(q, H);
    if (read_border(levels->model_family, im, H, 0, grid, border, &black, &white, ring) >= 0 && white > black)
        edge_light_model(levels->model_family, H, q, ring, border, levels);
}

/* Keep a decode attempt for the soft decoder if it could be a tiny tag. The cells are the unsharpened samples. */
static void soft_capture(napa_detector *det, const vec2 q[4], double black, double white, const double *grid) {
    const double contrast = white - black;
    if (contrast < NAPA_SOFT_MIN_CONTRAST || contrast > NAPA_SOFT_MAX_CONTRAST) return;
    double side = 0;
    for (int k = 0; k < 4; k++) side += 0.25 * hypot(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
    if (!(side >= NAPA_SOFT_MIN_SIDE && side <= NAPA_SOFT_MAX_SIDE)) return;
    const double mid = 0.5 * (black + white), half = 0.5 * contrast;
    int dark = 0;
    double brightest = -HUGE_VAL, sum = 0, squares = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double z = (grid[r * 8 + c] - mid) / half;
            if (r == 0 || r == 7 || c == 0 || c == 7) {
                dark += z < 0;
                if (z > brightest) brightest = z;
            } else {
                z = z < -1.5 ? -1.5 : z > 1.5 ? 1.5 : z;
                sum += z, squares += z * z;
            }
        }
    }
    if (dark < NAPA_SOFT_MIN_DARK_BORDER || brightest > NAPA_SOFT_MAX_BORDER_Z) return;
    const double variance = squares / 36 - (sum / 36) * (sum / 36);
    if (variance < NAPA_SOFT_MIN_SPREAD * NAPA_SOFT_MIN_SPREAD || variance > NAPA_SOFT_MAX_SPREAD * NAPA_SOFT_MAX_SPREAD) return;
    if (det->soft_count == det->soft_capacity) {
        const size_t cap = det->soft_capacity ? 2 * det->soft_capacity : 256;
        soft_candidate *grown = (soft_candidate *)realloc(det->soft_candidates, cap * sizeof *grown);
        if (grown == NULL) { det->soft_failed = 1; return; }
        det->soft_candidates = grown, det->soft_capacity = cap;
    }
    soft_candidate *e = &det->soft_candidates[det->soft_count];
    for (int k = 0; k < 4; k++) e->quad[2 * k] = (float)q[k].x, e->quad[2 * k + 1] = (float)q[k].y;
    e->black = (float)black, e->white = (float)white, e->order = (int)det->soft_count;
    for (int k = 0; k < 64; k++) e->grid[k] = (float)grid[k];
    det->soft_count++;
}

static void blocked_capture(napa_detector *det, const vec2 q[4], double black, double white, const double *grid) {
    /* The soft decoder's own test for a dark tag border: without it most candidates aren't tag-like and the search is wasted. */
    if (white - black < NAPA_SOFT_MIN_CONTRAST) return;
    {
        const double mid = 0.5 * (black + white), half = 0.5 * (white - black);
        double brightest = -HUGE_VAL;
        int dark = 0;
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++) {
                if (r != 0 && r != 7 && c != 0 && c != 7) continue;
                const double z = (grid[r * 8 + c] - mid) / half;
                dark += z < 0;
                if (z > brightest) brightest = z;
            }
        if (dark < NAPA_SOFT_MIN_DARK_BORDER || brightest > NAPA_SOFT_MAX_BORDER_Z) return;
    }
    if (det->blocked_count == det->blocked_capacity) {
        const size_t cap = det->blocked_capacity ? 2 * det->blocked_capacity : 256;
        soft_candidate *grown = (soft_candidate *)realloc(det->blocked_candidates, cap * sizeof *grown);
        if (grown == NULL) { det->soft_failed = 1; return; }
        det->blocked_candidates = grown, det->blocked_capacity = cap;
    }
    soft_candidate *e = &det->blocked_candidates[det->blocked_count];
    for (int k = 0; k < 4; k++) e->quad[2 * k] = (float)q[k].x, e->quad[2 * k + 1] = (float)q[k].y;
    e->black = (float)black, e->white = (float)white, e->order = (int)det->blocked_count;
    for (int k = 0; k < 64; k++) e->grid[k] = (float)grid[k];
    det->blocked_count++;
}

/* Decode the tag inside quad q (reordered in place to a consistent winding). `slack` loosens the
 * border and Hamming limits, for a cheap early decode on unrefined corners. Optionally reports each
 * side's border level (side_level) and the tag's black and white levels. */
static int decode_quad_impl(napa_detector *det, const image_view *im, vec2 q[4],
                       const napa_params *params, int slack, napa_detection *out, double side_level[4],
                       edge_levels *levels, int exact) {
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
    int dark = read_border(L, im, H, params->min_contrast, grid, border, &black, &white, NULL);
    if (dark < 0) return 0;
    if (levels != NULL) { levels->black = black; levels->white = white; levels->has_model = 0; levels->model_pending = 0; }
    double thr = 0.5 * (black + white);
    if (dark < min_border(L, params->min_border_cells) - slack) return 0;
    /* Every other cell of the grid: the payload, and for families with data outside the quiet ring,
     * that ring and the data around it. */
    {
        double u[256], v[256], val[256];
        int at[256], n = 0;
        for (int r = 1 - o; r < W - 1 + o; r++) {
            for (int c = 1 - o; c < W - 1 + o; c++) {
                if (o > 0 && r >= 0 && r < W && c >= 0 && c < W && (r == 0 || r == W - 1 || c == 0 || c == W - 1))
                    continue; /* a border cell, already read */
                u[n] = (c + 0.5) / L->cells, v[n] = (r + 0.5) / L->cells, at[n++] = (r + o) * G + c + o;
            }
        }
        sample_points(im, H, u, v, n, val);
        for (int k = 0; k < n; k++) grid[at[k]] = val[k];
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
    if (det->soft_capturing) {
        soft_capture(det, q, black, white, grid);
        blocked_capture(det, q, black, white, grid);
    }
    if (det->grid_probe != NULL) {
        soft_candidate *e = det->grid_probe;
        for (int k = 0; k < 4; k++) e->quad[2 * k] = (float)q[k].x, e->quad[2 * k + 1] = (float)q[k].y;
        e->black = (float)black, e->white = (float)white, e->order = 1;
        for (int k = 0; k < 64; k++) e->grid[k] = (float)grid[k];
    }
    /* Sharpen the cell samples, as upstream AprilTag does, to undo blur between neighbouring cells. */
    double unsharpened[256];
    if (params->sharpening_x100 > 0) memcpy(unsharpened, grid, sizeof(double) * (size_t)(G * G));
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
    int bi = nearest_code(det, code, exact ? 0 : (slack ? soft : max_hamming) + slack, &best);
    if (bi < 0 && slack == 0 && params->sharpening_x100 > 0 && L->piece_hamming >= 0) {
        /* Sharpening can flip a cell that read correctly before it. Only an exact code counts here: a corrected one
         * would raise the chance of a false match for a very small gain. Families with short codes don't retry. */
        uint64_t unsharpened_code = 0;
        for (int k = 0; k < L->nbits; k++) unsharpened_code = (unsharpened_code << 1) | (unsharpened[L->cell[k]] > thr ? 1u : 0u);
        int unsharpened_best = 64, match = nearest_code(det, unsharpened_code, 0, &unsharpened_best);
        if (match >= 0) bi = match, best = unsharpened_best, code = unsharpened_code, memcpy(grid, unsharpened, sizeof(double) * (size_t)(G * G));
    }
    if (!exact && bi < 0 && slack == 0 && soft > max_hamming) {
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
    if (levels != NULL && slack) {
        levels->model_family = L;
        levels->model_pending = 1;
    }
    return 1;
}

static int decode_quad(napa_detector *det, const image_view *im, vec2 q[4],
                       const napa_params *params, int slack, napa_detection *out, double side_level[4],
                       edge_levels *levels) {
    return decode_quad_impl(det,im,q,params,slack,out,side_level,levels,0);
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

/* Level coordinates stay below 30000, so a box's minx has a spare top bit: it marks a component seen this stage. */
#define PIECE_SEEN 0x8000u

/* Preserve the old box-density decision for existing split-piece dependencies. */
static int box_is_sparse(int count, int width, int height) {
    return (double)count / ((double)width * height) < 0.2;
}

/* These wholes used to wait for a coarser grid. Their finer outlines may
 * merge into the background there; retain them as additional proposals. */
static int dense_whole_is_new(int base, int s, int last, int width, int height) {
    return base == 1 && s <= 2 && !last && width > 72 && height > 72;
}

static int reserve_fine_whole(napa_detector *det, size_t index) {
    if (index >= det->fine_whole_capacity) {
        size_t cap = det->fine_whole_capacity ? 2 * det->fine_whole_capacity : 64;
        if (cap < det->fine_whole_capacity || cap <= index || cap > SIZE_MAX / sizeof(fine_whole)) return 0;
        fine_whole *p = (fine_whole *)realloc(det->fine_wholes, cap * sizeof(fine_whole));
        if (p == NULL) return 0;
        det->fine_wholes = p;
        det->fine_whole_capacity = cap;
    }
    return 1;
}

static int remember_fine_whole(napa_detector *det, size_t index, const component *shape, int stride, int terminal) {
    if (!reserve_fine_whole(det, index)) return 0;
    det->fine_wholes[index].data.shape = *shape;
    det->fine_wholes[index].stride = stride;
    det->fine_wholes[index].terminal = terminal;
    det->fine_wholes[index].kind = FINE_WHOLE_OUTLINE;
    return 1;
}

static int remember_confidence(napa_detector *det, size_t index, const vec2 fine[4],
                               double reach, double edge_width, const edge_levels *lv,
                               int early_id, int stride, int terminal) {
    if (!reserve_fine_whole(det, index)) return 0;
    fine_whole *entry = &det->fine_wholes[index];
    memcpy(entry->data.confidence.fine, fine, sizeof entry->data.confidence.fine);
    entry->data.confidence.reach = reach;
    entry->data.confidence.edge_width = edge_width;
    entry->data.confidence.lv = *lv;
    entry->data.confidence.early_id = early_id;
    entry->stride = stride;
    entry->terminal = terminal;
    entry->kind = FINE_WHOLE_CONFIDENCE_PENDING;
    return 1;
}

/* Propose component r as piece n if it passes the candidate size and shape test on its own, the first time it
 * is seen this stage. Returns the new piece count, or -1 when out of memory. */
static int propose_piece(napa_detector *det, int32_t r, int n, int lo, int hi, int dense) {
    blob_box *B = det->boxes;
    if (B[r].minx & PIECE_SEEN) return n;
    int w = B[r].maxx - B[r].minx + 1, h = B[r].maxy - B[r].miny + 1;
    B[r].minx |= PIECE_SEEN;
    int small = w < h ? w : h, big = w > h ? w : h;
    if (small < lo || big > hi || big > 8 * small || (dense && small > 72) || (double)B[r].count < 0.2 * (double)w * h)
        return n;
    if (grow_pieces(det, (size_t)n + 1) != 0) return -1;
    component *c = &det->pieces[n];
    c->count = 0;
    for (int k = 0; k < 16; k++) c->extv[k] = INT32_MIN;
    c->row_y = -1;
    c->minx = B[r].minx & ~PIECE_SEEN, c->maxx = B[r].maxx, c->miny = B[r].miny, c->maxy = B[r].maxy;
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
    for (size_t e = 0; e < *n_pairs; e += 2) {
        int32_t ra = joined_root(B, -P[E[e]] - 1), rb = joined_root(B, -P[E[e + 1]] - 1);
        if (ra == rb) continue; /* joined another way */
        E[ne++] = ra;
        E[ne++] = rb;
        if ((n = propose_piece(det, ra, n, lo, hi, dense)) < 0 || (n = propose_piece(det, rb, n, lo, hi, dense)) < 0)
            return -1;
    }
    *n_pairs = ne;
    /* The later component merges into the earlier one. Every component a join touches is the root of a pair
     * already reached here, so clearing marks pair by pair leaves none in a box that is read. */
    for (size_t e = 0; e < ne; e += 2) {
        B[E[e]].minx &= ~PIECE_SEEN;
        B[E[e + 1]].minx &= ~PIECE_SEEN;
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
 * thin border has necks. Supported filled-corner edges were never recorded, so afterwards the boxes
 * are the same blobs as ordinary detection. Returns the number of pieces,
 * neck-stage pieces first (*fine of them), or -1 when out of memory. */
static int split_pieces(napa_detector *det, int lo, int hi, int dense, int *fine) {
    int n = join_pairs(det, det->neck, &det->neck_n, 0, lo, hi, dense);
    if (n < 0) return -1;
    *fine = n;
    /* Each neck pair's components' roots now, before corner joins (and path halving) pass them by. */
    for (size_t e = 0; e < det->neck_n; e++) det->neck_root[e] = joined_root(det->boxes, det->neck[e]);
    return join_pairs(det, det->diag, &det->diag_n, n, lo, hi, dense);
}

/* Whether a found tag touches dark shapes at its corners, as grid tags do: just outside at least three corners,
 * diagonally, the image is darker than halfway between the tag's border there and its quiet zone. (A tag lying on
 * grass or against an edge has dark beyond one or two corners.) Families with
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
    return dark >= 3;
}

/* Coverage is queried for every proposal. Compute each accepted centre once, using exactly
 * the original ordered sum; caches have the same logical lengths as their output buffers. */
static int save_center(vec2 **centers, size_t *capacity, int index, const napa_detection *d) {
    if ((size_t)index >= *capacity) {
        size_t cap = *capacity ? 2 * *capacity : 64;
        if (cap <= (size_t)index) cap = (size_t)index + 1;
        if (cap > SIZE_MAX / sizeof(vec2)) return 0;
        vec2 *p = (vec2 *)realloc(*centers, cap * sizeof(vec2));
        if (p == NULL) return 0;
        *centers = p;
        *capacity = cap;
    }
    vec2 c = {0, 0};
    for (int k = 0; k < 4; k++) {
        c.x += d->corners[2 * k] * 0.25;
        c.y += d->corners[2 * k + 1] * 0.25;
    }
    (*centers)[index] = c;
    return 1;
}

/* A centre-distance hit alone does not establish overlap for an oblique tag.
 * Only suppress a point in the decoded convex footprint; accept either winding. */
static NAPA_FORCE_INLINE int detection_contains(const napa_detection *d, double x, double y) {
    int positive = 0, negative = 0;
    for (int k = 0; k < 4; k++) {
        int j = (k + 1) & 3;
        double ax = d->corners[2 * k], ay = d->corners[2 * k + 1];
        double bx = d->corners[2 * j], by = d->corners[2 * j + 1];
        double cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
        if (!isfinite(cross)) return 0;
        positive |= cross > 0;
        negative |= cross < 0;
        if (positive && negative) return 0;
    }
    return positive || negative;
}

/* Return the first distance hit as index+1, keeping the common negative search
 * independent of quad loads and the cold footprint calculation. */
static NAPA_FORCE_INLINE int first_near(const vec2 *centers, int n, double x, double y, double radius) {
    for (int i = 0; i < n; i++)
        if (fabs(centers[i].x - x) + fabs(centers[i].y - y) < radius) return i + 1;
    return 0;
}

/* Start at a known distance hit, then search only its unvisited tail. Every centre
 * is tested once even when the first close tag has a disjoint oblique footprint. */
static NAPA_NOINLINE int covered_tail(const napa_detection *out, const vec2 *centers, int n, int start,
                                     double x, double y, double radius) {
    for (int i = start; i < n;) {
        if (detection_contains(&out[i], x, y)) return 1;
        int next = first_near(centers + i + 1, n - i - 1, x, y, radius);
        if (!next) return 0;
        i += next;
    }
    return 0;
}

static NAPA_FORCE_INLINE int covered_footprint(const napa_detection *out, const vec2 *centers, int n,
                                              double x, double y, double radius) {
    int first = first_near(centers, n, x, y, radius);
    return first && covered_tail(out, centers, n, first - 1, x, y, radius);
}

static int duplicate_near(const vec2 *centers, int n, const napa_detection *d, vec2 *point, double *radius) {
    double cx = 0, cy = 0;
    for (int k = 0; k < 4; k++) {
        cx += d->corners[2 * k] * 0.25;
        cy += d->corners[2 * k + 1] * 0.25;
    }
    double size = fabs(d->corners[0] - d->corners[4]) + fabs(d->corners[1] - d->corners[5]);
    point->x = cx, point->y = cy;
    *radius = 0.25 * size + 2.0;
    return first_near(centers, n, cx, cy, *radius);
}

static int already_found_footprint(const napa_detection *out, const vec2 *centers, int n, const napa_detection *d) {
    vec2 point;
    double radius;
    int first = duplicate_near(centers, n, d, &point, &radius);
    return first && covered_tail(out, centers, n, first - 1, point.x, point.y, radius);
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
                int k = 0;
                for (; k + 4 <= pairs; k += 4) {
                    h4[0][d[k]]++;
                    h4[1][d[k + 1]]++;
                    h4[2][d[k + 2]]++;
                    h4[3][d[k + 3]]++;
                }
                for (; k < pairs; k++) h4[k & 3][d[k]]++;
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

/* A sparsely sampled small outline can retain three box corners while its
 * fourth rough pick lies along an edge or inside the box. Complete that corner
 * from the same component; never read a tag ID or introduce a second fit.
 * Keep the old seed unless the completed quad has the established winding. */
static int complete_small_outline(int pick[4][2], int minx, int maxx, int miny, int maxy) {
    if (minx >= maxx || miny >= maxy) return 0;
    unsigned seen = 0;
    int missing = -1;
    for (int i = 0; i < 4; i++) {
        int x = pick[i][0], y = pick[i][1];
        if (x < minx || x > maxx || y < miny || y > maxy) return 0;
        int xb = x == minx ? 0 : x == maxx ? 1 : -1;
        int yb = y == miny ? 0 : y == maxy ? 1 : -1;
        if (xb < 0 || yb < 0) {
            if (missing >= 0) return 0;
            missing = i;
        } else {
            unsigned bit = 1u << (xb + 2 * yb);
            if (seen & bit) return 0;
            seen |= bit;
        }
    }
    if (missing < 0) return 0;
    /* Three unique corners imply exactly one absent bit. */
    unsigned absent = 15u ^ seen;
    int corner = 0;
    while (!(absent & (1u << corner))) corner++;
    int oldx = pick[missing][0], oldy = pick[missing][1];
    pick[missing][0] = corner & 1 ? maxx : minx;
    pick[missing][1] = corner & 2 ? maxy : miny;
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) & 3, k = (i + 2) & 3;
        int64_t ax = (int64_t)pick[j][0] - pick[i][0];
        int64_t ay = (int64_t)pick[j][1] - pick[i][1];
        int64_t bx = (int64_t)pick[k][0] - pick[j][0];
        int64_t by = (int64_t)pick[k][1] - pick[j][1];
        if (ax * by - ay * bx <= 0) {
            pick[missing][0] = oldx;
            pick[missing][1] = oldy;
            return 0;
        }
    }
    return 1;
}

static void soft_sample(const image_view *view, const vec2 q[4], float S[NAPA_SOFT_AREA]);
static float standardize(float *a, int n);
static int solve_8x8(double a[8][9], double x[8]);
static double convex_area(const vec2 q[4]);
static double normal_cdf(double x);
static inline double soft_position(int i);

static int tone_prepare(napa_detector *det) {
    if (det->tone_blur != NULL) return 1;
    float (*t)[30][10] = (float (*)[30][10])malloc(NAPA_TONE_SIGMAS * sizeof *t);
    if (t == NULL) return 0;
    for (int b = 0; b < NAPA_TONE_SIGMAS; b++) {
        const double sigma = 0.06 * (b + 1);
        for (int r = 0; r < 30; r++)
            for (int c = 0; c < 10; c++)
                t[b][r][c] = (float)(normal_cdf((soft_position(r) - (c - 1.0)) / sigma) - normal_cdf((soft_position(r) - c) / sigma));
    }
    det->tone_blur = t;
    return 1;
}

/* Whether one side shows a whole edge: a dark border, its white ring, and a rising step between them, all in frame.
 * Tags on boards, with no white ring, and tags at the frame's edge stop here. */
#ifndef NAPA_TONE_VISIBLE_ROWS
#define NAPA_TONE_VISIBLE_ROWS 8
#endif
static int tone_side_visible(const image_view *view, vec2 a, vec2 b, vec2 center, double *cell, double *nx, double *ny, double *tx, double *ty) {
    enum { MAX_OFFSETS = 1024 };
    const double length = hypot(b.x - a.x, b.y - a.y);
    if (length < 8) return 0;
    const double c = length / 8;
    if (c > NAPA_TONE_MAX_CELL) return 0;
    const double ex = (b.x - a.x) / length, ey = (b.y - a.y) / length;
    double px = -ey, py = ex;
    if (px * (0.5 * (a.x + b.x) - center.x) + py * (0.5 * (a.y + b.y) - center.y) < 0) px = -px, py = -py;
    /* Every pixel of a long edge says the same thing: a fixed number of samples per cell and a few rows are enough. */
    const double dp = c / 8 > 0.25 ? c / 8 : 0.25;
    const int offsets = (int)floor(2 * 1.3 * c / dp + 1e-6) + 1, full_rows = (int)floor(0.6 * length / 4.5 + 1e-6) + 1;
    const int rows = full_rows < NAPA_TONE_VISIBLE_ROWS ? full_rows : NAPA_TONE_VISIBLE_ROWS;
    const double row_gap = rows == full_rows ? 4.5 : 4.5 * (full_rows - 1) / (rows - 1);
    if (rows < 3 || offsets < 8 || offsets > MAX_OFFSETS) return 0;
    /* The whole measured band must lie inside the frame, whichever samples are read. The band is a rectangle, so its corners decide. */
    const double far_offset = -1.3 * c + 0.25 * floor(2 * 1.3 * c / 0.25 + 1e-6), far_row = 0.2 * length + 4.5 * (full_rows - 1);
    for (int corner = 0; corner < 4; corner++) {
        const double o = corner & 1 ? far_offset : -1.3 * c, d = corner & 2 ? far_row : 0.2 * length, x = a.x + ex * d + px * o, y = a.y + ey * d + py * o;
        if (x < 1 || y < 1 || x > view->w - 2 || y > view->h - 2) return 0;
    }
    double profile[MAX_OFFSETS];
    for (int j = 0; j < offsets; j++) profile[j] = 0;
    for (int i = 0; i < rows; i++) {
        const double d = 0.2 * length + row_gap * i;
        for (int j = 0; j < offsets; j++) {
            const double o = -1.3 * c + dp * j, x = a.x + ex * d + px * o, y = a.y + ey * d + py * o;
            if (x < 1 || y < 1 || x > view->w - 2 || y > view->h - 2) return 0;
            profile[j] += sample_bilinear(view, x, y);
        }
    }
    double white = 0, black = 0;
    int white_n = 0, black_n = 0, peak = -1;
    double best = 0;
    for (int j = 0; j < offsets; j++) {
        profile[j] /= rows;
        const double o = -1.3 * c + dp * j;
        if (o > 0.35 * c && o < 0.9 * c) white += profile[j], white_n++;
        if (o > -0.9 * c && o < -0.35 * c) black += profile[j], black_n++;
    }
    if (!white_n || !black_n) return 0;
    white /= white_n, black /= black_n;
    if (!(white - black > 12)) return 0;
    for (int j = 0; j + 1 < offsets; j++) {
        const double gradient = (profile[j + 1] - profile[j]) / dp, middle = -1.3 * c + dp * (j + 0.5);
        if (fabs(middle) < 0.6 * c && gradient > best) best = gradient, peak = j;
    }
    if (peak < 0) return 0;
    int lo = peak, hi = peak;
    while (lo > 0 && (profile[lo] - profile[lo - 1]) / dp > 0.05 * best) lo--;
    while (hi < offsets - 2 && (profile[hi + 2] - profile[hi + 1]) / dp > 0.05 * best) hi++;
    const double mid = 0.5 * (white + black);
    int crossing = 0;
    for (int j = lo; j <= hi && !crossing; j++) crossing = profile[j] <= mid && mid < profile[j + 1];
    if (!crossing) return 0;
    *cell = c, *nx = px, *ny = py, *tx = ex, *ty = ey;
    return 1;
}

/* How far the sides of a blurred tag have moved, as a fraction of a cell, for blur s in cells and tone curvature k. */
static double tone_side_shift(double s, double k) {
    return -0.00567 + s * (0.09974 + s * (-0.55038 + s * 0.53232)) +
           k * (-0.01674 + s * (0.19018 + s * (0.77825 + s * -0.04047))) + k * k * (-0.02427 + s * (0.44033 + s * -0.77273));
}

/* On a straight tone curve, how far blur pushes a tag's sides outward, in cells, for blur s in cells. */
static double tone_straight_shift(double s) {
    return 0.015747 + s * (-0.166991 + s * (0.386275 + s * 0.339914));
}

/* The tag's picture against its code, blurred separately along each axis and passed through a quadratic tone curve.
 * Returns the residual, and the tone terms through c. */
/* across[r][col] for the lattice columns in use: the cells of each row, blurred along it. Sums per column rather
 * than per sample so they vectorize; each still adds its terms in the same order, so the rounding doesn't change. */
static void tone_across(const float cells[10][10], const float (*blur)[10], int step, float across[10][30]) {
    float by_column[10][10];
    for (int r = 0; r < 10; r++)
        for (int k = 0; k < 10; k++) by_column[k][r] = cells[r][k];
    for (int col = 0; col < 30; col += step) {
        float value[10];
        memset(value, 0, sizeof value);
        for (int k = 0; k < 10; k++) {
            const float weight = blur[col][k];
            for (int r = 0; r < 10; r++) value[r] += by_column[k][r] * weight;
        }
        for (int r = 0; r < 10; r++) across[r][col] = value[r];
    }
}

static double tone_fit(const float y[NAPA_SOFT_AREA], const float across[10][30], const float (*down)[10], int step, double c[3]) {
    double m[5] = {0, 0, 0, 0, 0}, v[2] = {0, 0}, a[3][4];
    int count = 0;
    const int columns = (30 + step - 1) / step;
    float packed[10][30];
    for (int k = 0; k < 10; k++)
        for (int i = 0; i < columns; i++) packed[k][i] = across[k][i * step];
    for (int r = 0; r < 30; r += step) {
        float values[30];
        memset(values, 0, sizeof values);
        for (int k = 0; k < 10; k++) {
            const float weight = down[r][k];
            for (int i = 0; i < columns; i++) values[i] += weight * packed[k][i];
        }
        for (int i = 0; i < columns; i++) {
            const int col = i * step;
            const float value = values[i];
            const double t = 0.5 * value + 0.5, t2 = t * t;
            m[1] += t, m[2] += t2, m[3] += t2 * t, m[4] += t2 * t2;
            v[0] += y[r * 30 + col] * t, v[1] += y[r * 30 + col] * t2;
            count++;
        }
    }
    m[0] = count;
    for (int r = 0; r < 3; r++) for (int q = 0; q < 3; q++) a[r][q] = m[r + q];
    a[0][3] = 0, a[1][3] = v[0], a[2][3] = v[1];
    if (!solve_3x3(a, c)) return HUGE_VAL;
    return count - (c[1] * v[0] + c[2] * v[1]);
}

/* The tag's lattice at every step-th row and column, standardized over those samples. */
static void tone_sample(const image_view *view, const vec2 q[4], int step, float y[NAPA_SOFT_AREA]) {
    double H[9], sum = 0, squares = 0;
    int n = 0;
    square_to_quad(q, H);
    for (int r = 0; r < 30; r += step) {
        for (int c = 0; c < 30; c += step) {
            const vec2 p = project(H, soft_position(c) / 8, soft_position(r) / 8);
            const double v = sample_bilinear(view, p.x, p.y);
            y[r * 30 + c] = (float)v, sum += v, squares += v * v, n++;
        }
    }
    const double mean = sum / n, sigma = sqrt(squares / n - mean * mean);
    for (int r = 0; r < 30; r += step)
        for (int c = 0; c < 30; c += step) y[r * 30 + c] = sigma > 1e-6 ? (float)((y[r * 30 + c] - mean) / sigma) : 0;
}

/* Blur sweep along one axis, keeping the best fit so far. */
static void tone_sweep(const napa_detector *det, const float cells[10][10], const float y[NAPA_SOFT_AREA], int step, int vary_v, int *u, int *v,
                       double *best, double tone[3]) {
    float across[10][30];
    const int was_u = *u, was_v = *v;
    for (int b = 0; b < NAPA_TONE_SIGMAS; b++) {
        const int su = vary_v ? was_u : b, sv = vary_v ? b : was_v;
        if (!vary_v || b == 0) tone_across(cells, (const float (*)[10])det->tone_blur[su], step, across);
        double c[3];
        const double sse = tone_fit(y, across, (const float (*)[10])det->tone_blur[sv], step, c);
        if (sse < *best) *best = sse, *u = su, *v = sv, memcpy(tone, c, 3 * sizeof *tone);
    }
}

/* What a decoded tag's own picture says about its blur and the camera's tone curve. */
typedef struct {
    float cells[10][10], picture[NAPA_SOFT_AREA], seen[NAPA_SOFT_AREA]; /* picture: the code blurred to match the tag, from 0 to 1 */
    double seen_at[8];
    double tone[3], fit, curvature, ring_error, blur_u, blur_v, mean_side;
    double cell[4], nx[4], ny[4], tx[4], ty[4];
    int sides_visible, straight_only;
    vec2 center;
} tone_measure;

static void tone_picture(const napa_detector *det, const float cells[10][10], int u, int v, float picture[NAPA_SOFT_AREA]) {
    float across[10][30];
    for (int r = 0; r < 10; r++)
        for (int col = 0; col < 30; col++) {
            float value = 0;
            for (int k = 0; k < 10; k++) value += cells[r][k] * det->tone_blur[u][col][k];
            across[r][col] = value;
        }
    for (int r = 0; r < 30; r++)
        for (int col = 0; col < 30; col++) {
            float value = 0;
            for (int k = 0; k < 10; k++) value += det->tone_blur[v][r][k] * across[k][col];
            picture[r * 30 + col] = 0.5f * value + 0.5f;
        }
}

static int tone_sides_visible(const image_view *view, const vec2 q[4], tone_measure *m) {
    for (int k = 0; k < 4; k++)
        if (!tone_side_visible(view, q[k], q[(k + 1) & 3], m->center, &m->cell[k], &m->nx[k], &m->ny[k], &m->tx[k], &m->ty[k])) return 0;
    return 1;
}

/* Measure a decoded tag's blur along each axis and its tone curve, or 0 if it's too small, sharp or straight-toned to matter.
 * Size correction alone needs every side's whole edge, which is cheaper to check before the search; alignment doesn't. */
static int tone_measure_tag(napa_detector *det, const image_view *view, const napa_detection *d, double edge_width, int aligning, tone_measure *m) {
    vec2 q[4];
    m->center.x = m->center.y = m->mean_side = 0;
    for (int k = 0; k < 4; k++) q[k].x = d->corners[2 * k], q[k].y = d->corners[2 * k + 1], m->center.x += 0.25 * q[k].x, m->center.y += 0.25 * q[k].y;
    for (int k = 0; k < 4; k++) m->mean_side += 0.25 * hypot(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
    if (m->mean_side < NAPA_TONE_MIN_SIDE || edge_width < NAPA_TONE_MIN_EDGE_CELLS * m->mean_side / 8 || signed_area(q) <= 0) return 0;
    if (strcmp(det->fam.f->name, "tag36h11") != 0) return 0;
    if (!tone_prepare(det)) return 0;
    const family_layout *L = &det->fam;
    float y[NAPA_SOFT_AREA];
    for (int r = 0; r < 10; r++) for (int c = 0; c < 10; c++) m->cells[r][c] = (r == 0 || r == 9 || c == 0 || c == 9) ? 1.0f : -1.0f;
    const uint64_t code = L->f->codes[d->id];
    for (int k = 0; k < 36; k++) m->cells[L->cell[k] / L->grid + 1][L->cell[k] % L->grid + 1] = (code >> (35 - k) & 1) ? 1.0f : -1.0f;
    int u = NAPA_TONE_START, v = NAPA_TONE_START;
    double best = HUGE_VAL, tone[3] = {0, 0, 0};
    /* A few samples and blurs are enough to see that a tone curve is straight, which is most tags. */
    {
        float across[10][30];
        tone_across((const float (*)[10])m->cells, (const float (*)[10])det->tone_blur[NAPA_TONE_START], 1, across);
        tone_sample(view, q, 3, y);
        for (int b = 2; b <= 6; b += 2) {
            double c[3];
            const double sse = tone_fit(y, across, (const float (*)[10])det->tone_blur[b], 3, c);
            if (sse < best) best = sse, memcpy(tone, c, sizeof tone);
        }
    }
    /* A screen reading above the limit means a straight tone, and only wide edges then have a size bias worth measuring. */
    m->straight_only = fabs(tone[1] + tone[2]) > 1e-9 && tone[2] / (tone[1] + tone[2]) > NAPA_TONE_SCREEN_CURVATURE;
    if (m->straight_only && !(edge_width >= NAPA_TONE_STRAIGHT_EDGE_CELLS * m->mean_side / 8)) return 0;
    m->sides_visible = 1;
    if (!aligning && !tone_sides_visible(view, q, m)) return 0;
    tone_sample(view, q, 2, y);
    best = HUGE_VAL, u = NAPA_TONE_START, v = NAPA_TONE_START;
    /* If the second sweep keeps u, a third would repeat the first exactly. */
    for (int round = 0, kept_u = u; round < 3 && (round < 2 || u != kept_u); round++) {
        if (round == 1) kept_u = u;
        tone_sweep(det, (const float (*)[10])m->cells, y, 2, !(round & 1), &u, &v, &best, tone);
    }
    /* The search ran on a quarter of the lattice; judge its answer on all of it. */
    tone_sample(view, q, 1, y);
    {
        float across[10][30];
        tone_across((const float (*)[10])m->cells, (const float (*)[10])det->tone_blur[u], 1, across);
        best = tone_fit(y, across, (const float (*)[10])det->tone_blur[v], 1, tone);
    }
    memcpy(m->tone, tone, sizeof tone);
    m->fit = 1 - best / NAPA_SOFT_AREA;
    const double slope = tone[1] + tone[2];
    m->curvature = fabs(slope) > 1e-9 ? tone[2] / slope : 0;
    m->blur_u = 0.06 * (u + 1), m->blur_v = 0.06 * (v + 1);
    if (aligning) {
        m->sides_visible = tone_sides_visible(view, q, m);
        memcpy(m->seen, y, sizeof m->seen), memcpy(m->seen_at, d->corners, sizeof m->seen_at);
        tone_picture(det, (const float (*)[10])m->cells, u, v, m->picture);
        double ring = 0;
        int ring_n = 0;
        for (int r = 0; r < 30; r++) {
            for (int c = 0; c < 30; c++) {
                if (r > 2 && r < 27 && c > 2 && c < 27) continue;
                const double p = m->picture[r * 30 + c], e = y[r * 30 + c] - (tone[0] + p * (tone[1] + p * tone[2]));
                ring += e * e, ring_n++;
            }
        }
        m->ring_error = sqrt(ring / ring_n);
    }
    return 1;
}

/* Corners with the size bias of blur and tone taken out, or 0 to keep them. */
static int tone_corrected_quad(const napa_detection *d, const tone_measure *m, vec2 out[4]) {
    if (!m->sides_visible || !(m->fit >= NAPA_TONE_MIN_FIT)) return 0;
    const int straight = m->curvature > NAPA_TONE_STRAIGHT_MIN_CURVATURE;
    if (m->curvature > NAPA_TONE_MAX_CURVATURE && !straight) return 0;
    if (straight && !(m->curvature < NAPA_TONE_STRAIGHT_MAX_CURVATURE)) return 0;
    const double mean_cell = m->mean_side / 8;
    /* Uneven blur, as from motion, moves the sides less than the fitted widths predict. */
    const double most = m->blur_u > m->blur_v ? m->blur_u : m->blur_v, least = m->blur_u < m->blur_v ? m->blur_u : m->blur_v;
    const double unevenness = most / least, even = 1.3 / (unevenness * unevenness) < 1 ? 1.3 / (unevenness * unevenness) : 1;
    double lx[4], ly[4];
    int moved = 0;
    for (int k = 0; k < 4; k++) {
        const double s = ((k & 1) ? m->blur_u : m->blur_v) * mean_cell / m->cell[k];
        const double sc = s < 0.6 ? s : 0.6;
        const double shift = s < NAPA_TONE_MIN_BLUR ? 0 : straight ? even * m->cell[k] * tone_straight_shift(sc) : m->cell[k] * tone_side_shift(s, m->curvature);
        moved |= shift != 0;
        lx[k] = d->corners[2 * k] - m->nx[k] * shift, ly[k] = d->corners[2 * k + 1] - m->ny[k] * shift;
    }
    if (!moved) return 0;
    for (int k = 0; k < 4; k++) {
        const int p = (k + 3) & 3;
        const double cross = m->tx[p] * m->ty[k] - m->ty[p] * m->tx[k];
        if (fabs(cross) < 0.2) return 0;
        const double t = ((lx[k] - lx[p]) * m->ty[k] - (ly[k] - ly[p]) * m->tx[k]) / cross;
        out[k].x = lx[p] + t * m->tx[p], out[k].y = ly[p] + t * m->ty[p];
    }
    return 1;
}

/* Pull the corners onto the tag's own blurred, tone-mapped picture, which whole-side fits can't follow under uneven blur.
 * A white ring occupied by neighbours would mislead it, so the ring has to fit the picture. The Jacobian comes once from
 * the sampled picture's slopes, with no further image reads. */
static int tone_aligned_quad(const image_view *view, const napa_detection *d, const tone_measure *m, vec2 out[4]) {
    const double most = m->blur_u > m->blur_v ? m->blur_u : m->blur_v, least = m->blur_u < m->blur_v ? m->blur_u : m->blur_v;
    if (m->straight_only) return 0; /* the size shift alone serves these, and alignment would cost far more */
    if (!(m->ring_error <= NAPA_ALIGN_MAX_RING_ERROR) || !(m->fit >= NAPA_ALIGN_MIN_FIT) || !(m->curvature > -1 && m->curvature < 0.5) ||
        !(most >= NAPA_ALIGN_MIN_BLUR || most >= NAPA_ALIGN_MIN_ANISOTROPY * least)) return 0;
    float t[NAPA_SOFT_AREA], now[NAPA_SOFT_AREA], jacobian[8][NAPA_SOFT_AREA];
    double x[8], origin[8], kept[8];
    for (int i = 0; i < NAPA_SOFT_AREA; i++) {
        const double p = m->picture[i];
        t[i] = (float)(m->tone[0] + p * (m->tone[1] + p * m->tone[2]));
    }
    if (standardize(t, NAPA_SOFT_AREA) <= 1e-6f) return 0;
    for (int k = 0; k < 8; k++) x[k] = origin[k] = kept[k] = d->corners[k];
    const double limit = m->mean_side / 16 > 1 ? m->mean_side / 16 : 1;
    vec2 q[4];
    for (int k = 0; k < 4; k++) q[k].x = x[2 * k], q[k].y = x[2 * k + 1];
    if (memcmp(m->seen_at, d->corners, sizeof m->seen_at) == 0) memcpy(now, m->seen, sizeof now);
    else tone_sample(view, q, 1, now);
    double start = 0;
    for (int i = 0; i < NAPA_SOFT_AREA; i++) start += (double)now[i] * t[i];
    start /= NAPA_SOFT_AREA;
    double best = start, jtj[8][8], trace = 0;
    for (int iter = 0; iter < NAPA_ALIGN_ITERATIONS; iter++) {
        if (iter == 0) {
            /* Slopes of the picture across the lattice, then how each corner slides the lattice under it. */
            const float *g = now;
            double H[9], Hi[9], Hk[9], M[9];
            square_to_quad(q, H);
            Hi[0] = H[4] * H[8] - H[5] * H[7], Hi[1] = H[2] * H[7] - H[1] * H[8], Hi[2] = H[1] * H[5] - H[2] * H[4];
            Hi[3] = H[5] * H[6] - H[3] * H[8], Hi[4] = H[0] * H[8] - H[2] * H[6], Hi[5] = H[2] * H[3] - H[0] * H[5];
            Hi[6] = H[3] * H[7] - H[4] * H[6], Hi[7] = H[1] * H[6] - H[0] * H[7], Hi[8] = H[0] * H[4] - H[1] * H[3];
            for (int r = 0; r < 30; r++)
                for (int c = 0; c < 30; c++) {
                    const int i = r * 30 + c, cl = c > 0 ? c - 1 : c, cr = c < 29 ? c + 1 : c, ru = r > 0 ? r - 1 : r, rd = r < 29 ? r + 1 : r;
                    jacobian[0][i] = (g[r * 30 + cr] - g[r * 30 + cl]) * 24.0f / (cr - cl);
                    jacobian[1][i] = (g[rd * 30 + c] - g[ru * 30 + c]) * 24.0f / (rd - ru);
                }
            float gu[NAPA_SOFT_AREA], gv[NAPA_SOFT_AREA];
            memcpy(gu, jacobian[0], sizeof gu), memcpy(gv, jacobian[1], sizeof gv);
            for (int k = 0; k < 8; k++) {
                double *c = k & 1 ? &q[k / 2].y : &q[k / 2].x;
                const double was = *c;
                *c = was + 0.25;
                square_to_quad(q, Hk);
                *c = was;
                for (int r = 0; r < 3; r++)
                    for (int cc = 0; cc < 3; cc++) M[3 * r + cc] = Hi[3 * r] * Hk[cc] + Hi[3 * r + 1] * Hk[3 + cc] + Hi[3 * r + 2] * Hk[6 + cc];
                for (int r = 0; r < 30; r++)
                    for (int cc = 0; cc < 30; cc++) {
                        const double u = soft_position(cc) / 8, v = soft_position(r) / 8;
                        const vec2 p = project(M, u, v);
                        jacobian[k][r * 30 + cc] = (float)((gu[r * 30 + cc] * (p.x - u) + gv[r * 30 + cc] * (p.y - v)) / 0.25);
                    }
            }
            for (int r = 0; r < 8; r++) {
                for (int c = r; c < 8; c++) {
                    double v = 0;
                    for (int i = 0; i < NAPA_SOFT_AREA; i++) v += (double)jacobian[r][i] * jacobian[c][i];
                    jtj[r][c] = jtj[c][r] = v;
                }
                trace += jtj[r][r];
            }
        }
        double a[8][9], step[8];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) a[r][c] = jtj[r][c];
            double v = 0;
            for (int i = 0; i < NAPA_SOFT_AREA; i++) v += (double)jacobian[r][i] * (t[i] - now[i]);
            a[r][8] = v;
        }
        for (int r = 0; r < 8; r++) a[r][r] += 1e-2 * trace / 8;
        if (!solve_8x8(a, step)) break;
        for (int k = 0; k < 8; k++) x[k] += step[k];
        for (int k = 0; k < 4; k++) {
            const double dx = x[2 * k] - origin[2 * k], dy = x[2 * k + 1] - origin[2 * k + 1], len = hypot(dx, dy);
            if (len > limit) x[2 * k] = origin[2 * k] + dx * limit / len, x[2 * k + 1] = origin[2 * k + 1] + dy * limit / len;
        }
        for (int k = 0; k < 4; k++) q[k].x = x[2 * k], q[k].y = x[2 * k + 1];
        tone_sample(view, q, 1, now);
        double value = 0;
        for (int i = 0; i < NAPA_SOFT_AREA; i++) value += (double)now[i] * t[i];
        value /= NAPA_SOFT_AREA;
        if (value > best) best = value, memcpy(kept, x, sizeof kept);
    }
    if (!(best - start >= NAPA_ALIGN_MIN_GAIN)) return 0;
    for (int k = 0; k < 4; k++) out[k].x = kept[2 * k], out[k].y = kept[2 * k + 1];
    return signed_area(out) > 0 && convex_area(out) > 0;
}

/* Decoding in linear light needs the tag's contrast to survive the table, or the attempt only costs a second pass. */
static int linear_contrast_enough(const napa_detector *det, const edge_levels *lv, const napa_params *params) {
    const int black = lv->black < 0 ? 0 : lv->black > 255 ? 255 : (int)(lv->black + 0.5), white = lv->white < 0 ? 0 : lv->white > 255 ? 255 : (int)(lv->white + 0.5);
    return (double)det->linearize[white] - det->linearize[black] >= NAPA_LINEAR_MIN_CONTRAST * params->min_contrast;
}

/* The frame halved down (rows averaged in pairs) or across (columns in pairs) into det->shrunk; an odd last row or
 * column stands alone. Returns 0, or -2 when out of memory. */
static int halve_frame(napa_detector *det, const uint8_t *img, int width, int height, int stride, int across, int *ow,
                       int *oh) {
    const int w = across ? (width + 1) / 2 : width, h = across ? height : (height + 1) / 2;
    if ((size_t)w * h > det->shrunk_capacity) {
        uint8_t *p = (uint8_t *)realloc(det->shrunk, (size_t)w * h);
        if (p == NULL) return -2;
        det->shrunk = p, det->shrunk_capacity = (size_t)w * h;
    }
    for (int r = 0; r < h; r++) {
        uint8_t *dst = det->shrunk + (size_t)r * w;
        if (across) {
            const uint8_t *a = img + (size_t)r * stride;
            const int pairs = width / 2;
            for (int i = 0; i < pairs; i++) dst[i] = (uint8_t)((a[2 * i] + a[2 * i + 1] + 1) >> 1);
            if (width & 1) dst[pairs] = a[width - 1];
        } else {
            const uint8_t *a = img + (size_t)(2 * r) * stride, *b = 2 * r + 1 < height ? a + stride : a;
            for (int x = 0; x < width; x++) dst[x] = (uint8_t)((a[x] + b[x] + 1) >> 1);
        }
    }
    *ow = w, *oh = h;
    return 0;
}

static int detect_pass(napa_detector *det, const uint8_t *img, int width, int height, int stride, const napa_params *params,
                napa_detection *out, int max_out, int found, int shadow) {
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
    /* tag_shape: the ordinary pass samples a copy of the frame halved down (tall) or across (wide), so its levels are
     * twice as coarse that way, while fitting and decoding read the frame itself. A tag turned about the vertical axis
     * is narrow and tall: it keeps its height, so sampling it every other row loses little. Level sample (i, j) covers
     * pixels [i sx, (i + 1) sx) x [j sy, (j + 1) sy), centred (sx - 1) / 2 and (sy - 1) / 2 past the corner. */
    const int shaped = !shadow && det->tag_shape != 0;
    const double sx = shaped && det->tag_shape == NAPA_TAG_SHAPE_WIDE ? 2.0 : 1.0;
    const double sy = shaped && det->tag_shape == NAPA_TAG_SHAPE_TALL ? 2.0 : 1.0;
    const double offx = 0.5 * (sx - 1.0), offy = 0.5 * (sy - 1.0), smax = sx > sy ? sx : sy;
    const uint8_t *pimg = img;
    int pw = width, ph = height, pstride = stride;
    if (shaped) {
        if (halve_frame(det, img, width, height, stride, sx > 1.0, &pw, &ph) != 0) return -2;
        pimg = det->shrunk, pstride = pw;
    }
    int piece_found = 0, shape_found = 0;
    size_t fine_whole_count = 0;
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
    const uint8_t *gray = NULL; /* reset on split restart; never owns caller pixels */
    for (int s = base;; s *= 2) {
        int lazy_rows = 0;
        int lw = (pw + s - 1) / s;
        int lh = (ph + s - 1) / s;
        int last = (lw < NAPA_LAST * W / 8) || (lh < NAPA_LAST * W / 8);
        size_t pixels = (size_t)lw * lh;
        size_t tiles = (size_t)((lw + TILE - 1) / TILE) * ((lh + TILE - 1) / TILE);
        if (ensure_buffers(det, pixels, tiles, (size_t)lw, pixels / 2 + (size_t)lh + 1) != 0) return -2;

        PROF_START;
        /* Contiguous, non-inverted stride-one gray can be read directly. Coarser
         * samples always go into owned scratch; later levels keep the in-place rake. */
        if (s == base) {
            if (s == 1 && pstride == pw && !view.invert) gray = pimg;
            else {
                /* The noise floors read every 8th row. Sample only those first: a noisy frame re-rakes
                 * every sample with block averages, and the rest would be thrown away. */
                lazy_rows = s > 1 && s <= 16;
                for (int y = 0; y < lh; y += lazy_rows ? 8 : 1)
                    gather_row(det->level + (size_t)y * lw, pimg + (size_t)y * s * pstride, lw, s, det->ssse3);
                if (view.invert && !lazy_rows) invert_bytes(det->level, pixels);
                gray = det->level;
            }
        } else {
            for (int y = 0; y < lh; y++) {
                const uint8_t *src = gray + (size_t)(2 * y) * prev_lw;
                if (gray == det->level) gather2(det->level + (size_t)y * lw, src, lw);
                /* Unlike owned scratch, an odd borrowed final row has no spare byte. */
                else gather_row(det->level + (size_t)y * lw, src, lw, 2, det->ssse3);
            }
            gray = det->level;
        }
        prev_lw = lw;
        PROF_MARK(PROF_RAKE);
        int boxed = 0;
        if (s == base || (base == 1 && s == 2 && contrast > NAPA_NOISY_FLOOR)) {
            /* Do not blur stride-one proposals: a tiny border can be just one pixel wide. Its
             * higher floor suppresses raw noise; the next averaged level restores low contrast.
             * A terminal level has no recovery pass, so keep its original contrast floor. */
            floor_stride = s;
            double sigmas = s == 1 && !last ? 7.0 : NAPA_FLOOR_SIGMAS;
            int cap = s == 1 && !last ? 255 : NAPA_FLOOR_CAP;
            contrast = contrast_floors(det, gray, lw, lh, params->min_contrast, sigmas, cap);
            if (lazy_rows && !(contrast > NAPA_NOISY_FLOOR && s > 1 && s <= 16)) {
                for (int y = 0; y < lh; y++)
                    if (y & 7) gather_row(det->level + (size_t)y * lw, pimg + (size_t)y * s * pstride, lw, s, det->ssse3);
                if (view.invert) invert_bytes(det->level, pixels);
            }
            /* A noisy frame: rake again, averaging each sample's block, and re-measure the noise. */
            if (contrast > NAPA_NOISY_FLOOR && s > 1 && s <= 16) {
                if ((size_t)pw > det->box_capacity) {
                    uint16_t *p = (uint16_t *)realloc(det->box_rows, sizeof(uint16_t) * 2 * (size_t)pw);
                    if (p == NULL) return -2;
                    det->box_rows = p;
                    det->box_capacity = (size_t)pw;
                }
                boxed = 1;
                box_rake(det->level, lw, lh, pimg, pw, ph, pstride, s, base == 1 && s == 2 ? 3 : s, det->box_rows, det->box_rows + pw);
                if (view.invert) invert_bytes(det->level, pixels);
                /* Stride-two 3x3 means share one column: their difference measures sqrt(2/3)
                 * times their noise sigma. Disjoint averaging blocks need no correction. */
                double sigmas = base == 1 && s == 2 ? NAPA_BOX_SIGMAS * sqrt(1.5) : NAPA_BOX_SIGMAS;
                contrast = contrast_floors(det, det->level, lw, lh, params->min_contrast, sigmas, NAPA_FLOOR_CAP);
            }
            if (contrast < 0) return -2;
        }
        det->level_contrast_min = !shadow && (s == base || (base > 1 && s == 2 * base)) && !last
            ? (uint8_t)(params->min_contrast * NAPA_FINEST_CONTRAST_NUM / NAPA_FINEST_CONTRAST_DEN > 255
                            ? 255 : params->min_contrast * NAPA_FINEST_CONTRAST_NUM / NAPA_FINEST_CONTRAST_DEN)
            : 0;
        const int redo = s <= redone;
        det->level_view = gray == det->level ? NULL : gray;
        /* Keyed by sampling step: the passes can start from different steps. */
        const int level_index = s < 32 ? s : 32;
        det->level_extrema_out = NULL;
        det->level_extrema_in = ((shadow && det->extrema_ready) || (!shadow && redo && det->extrema_wanted)) && level_index < 32 && !boxed && !det->saved_boxed[level_index] ? det->saved_extrema[level_index] : NULL;
        if (det->extrema_wanted && !det->level_extrema_in && level_index < 32) {
            if (2 * tiles > det->saved_extrema_capacity[level_index]) {
                uint8_t *p = (uint8_t *)realloc(det->saved_extrema[level_index], 2 * tiles);
                if (p == NULL) return -2;
                det->saved_extrema[level_index] = p;
                det->saved_extrema_capacity[level_index] = 2 * tiles;
            }
            det->level_extrema_out = det->saved_extrema[level_index];
            det->saved_boxed[level_index] = boxed;
        }
        int nruns = threshold_level(det, lw, lh, s / floor_stride, split ? 1 + necks : 0, shadow);
        det->level_view = NULL; /* clear even on threshold allocation failure */
        det->level_extrema_out = NULL, det->level_extrema_in = NULL;
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
        /* Only the finest grid tries smaller wholes; coarser bands already overlap the
         * established finer candidates. Defer these additions and require an exact payload. */
        int min_size = base > 1 && s == base ? (5 * W + 4) / 8 : lo;
        /* Overlapping size bands retain tilted tags. Large fine-grid outlines wait until
         * established proposals finish, because they can merge into their background later. */
        int max_size = last ? 1 << 30 : hi;
        int ncand = 0;
        /* Unsplit candidate boxes are checked against this stable prefix before extrema. */
        const int covered_prefix = !split ? (found < max_out ? found : max_out) : 0;
        for (int i = 0; i < nroots; i++) {
            det->slot[i] = -1;
            if (B[i].count < 0) continue; /* a piece joined into an earlier one */
            int t_minx = B[i].minx, t_maxx = B[i].maxx, t_miny = B[i].miny, t_maxy = B[i].maxy;
            int bw = t_maxx - t_minx + 1, bh = t_maxy - t_miny + 1;
            int big = bw > bh ? bw : bh, small = bw < bh ? bw : bh;
            /* Larger fine-grid wholes are retained below; split pieces keep their size cap. */
            if (small < min_size || big > max_size || big > 8 * small) {
                if (big >= 4) TRACE(small < min_size ? 1 : big > max_size ? 2 : 3);
                continue;
            }
            /* Without split-piece dependencies, a candidate covered by an earlier level
             * will still be covered when decoded outputs grow later in this level. */
            if ((shadow || !split) && covered_footprint(out, det->out_centers, found < max_out ? found : max_out,
                                  0.5 * (t_minx + t_maxx) * s * sx + offx, 0.5 * (t_miny + t_maxy) * s * sy + offy,
                                  0.3 * s * (bw * sx > bh * sy ? bw * sx : bh * sy) + 2.0)) continue;
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
            for (int k = 0; k < npieces; k++) {
                int32_t root = joined_root(B, det->piece_root[k]);
                const blob_box *box = &B[root];
                int candidate = det->slot[root];
                /* A newly admitted sparse or fine parent was not an old candidate. Keep its extrema
                 * chain, but do not let its fate or stride-one shortcut suppress old pieces. */
                if (candidate >= 0 && (box_is_sparse(box->count, box->maxx - box->minx + 1,
                                                    box->maxy - box->miny + 1) ||
                    dense_whole_is_new(base, s, last, box->maxx - box->minx + 1,
                                      box->maxy - box->miny + 1))) candidate = -1;
                det->piece_cand[k] = candidate;
            }
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

        const int level_stride = s, terminal_level = last;
        const int ordinary_count = ncand + npieces;
        if (last && fine_whole_count > (size_t)(INT32_MAX - ordinary_count)) return -2;
        int candidate_count = ordinary_count + (last ? (int)fine_whole_count : 0);
        for (int ci = 0; ci < candidate_count; ci++) {
            const int delayed = ci >= ordinary_count;
            const size_t pending_index = delayed ? (size_t)(ci - ordinary_count) : 0;
            const int confidence = delayed && det->fine_wholes[pending_index].kind != FINE_WHOLE_OUTLINE;
            /* Process confidence entries once, even when neck splitting restarts the levels. */
            if (confidence && det->fine_wholes[pending_index].kind != FINE_WHOLE_CONFIDENCE_PENDING) continue;
            const int piece = !delayed && ci >= ncand;
            /* A saved outline retains its original coordinate grid and terminal fit policy. */
            const int s = delayed ? det->fine_wholes[pending_index].stride : level_stride;
            const int last = delayed ? det->fine_wholes[pending_index].terminal : terminal_level;
            const int coverage_start = delayed ? 0 : covered_prefix;
            /* Corner-stage pieces first: a tag kept whole there is found as without necks. */
            int k = piece ? (ci - ncand < npieces - fine ? fine + ci - ncand : ci - ncand - (npieces - fine)) : 0;
            /* Redone levels decode only the pieces not tried before (whose fate reads -1, below). */
            if (!delayed && redo && (!piece || (k >= fine && det->pieces[k].row_y != -1))) continue;
            /* These locals are shared with the predecoded confidence entry below. Every
             * value read after refine_candidate is assigned on both incoming paths. */
            component confidence_self;
            component *self;
            component comp;
            int new_small = 0, require_exact = confidence, extended_shape = 0;
            int t_minx, t_maxx, t_miny, t_maxy;
            napa_params strict_border;
            const napa_params *decode_params = params;
            vec2 rough[4], fine[4], finer[4];
            image_view lin_view = view;
            const image_view *rview = &view;
            int lin_tried = 0;
            vec2 raw_fine[4];
            double raw_reach = 0, raw_edge_w = 0;
            edge_levels raw_lv;
            double reach, edge_w, cell;
            napa_detection early;
            edge_levels lv;
            if (confidence) {
                /* Mark rejected before refinement: each failure continue keeps that state.
                 * No pointer into the deferred buffer survives an append/reallocation. */
                det->fine_wholes[pending_index].kind = FINE_WHOLE_CONFIDENCE_REJECTED;
                if (found >= max_out) continue;
                memcpy(fine, det->fine_wholes[pending_index].data.confidence.fine, sizeof fine);
                reach = det->fine_wholes[pending_index].data.confidence.reach;
                edge_w = det->fine_wholes[pending_index].data.confidence.edge_width;
                lv = det->fine_wholes[pending_index].data.confidence.lv;
                early.id = det->fine_wholes[pending_index].data.confidence.early_id;
                strict_border = *params;
                strict_border.max_hamming = 0;
                decode_params = &strict_border;
                /* The shared tail only writes a failure fate through self. Keep those
                 * writes local instead of changing an established component's fate. */
                confidence_self.row_y = -8;
                self = &confidence_self;
#ifdef NAPA_TRACE
                t_minx = t_maxx = (int)floor(fine[0].x / s);
                t_miny = t_maxy = (int)floor(fine[0].y / s);
                for (int side = 0; side < 4; side++) {
                    int x0 = (int)floor(fine[side].x / s), x1 = (int)ceil(fine[side].x / s);
                    int y0 = (int)floor(fine[side].y / s), y1 = (int)ceil(fine[side].y / s);
                    if (x0 < t_minx) t_minx = x0;
                    if (x1 > t_maxx) t_maxx = x1;
                    if (y0 < t_miny) t_miny = y0;
                    if (y1 > t_maxy) t_maxy = y1;
                }
#endif
                goto refine_candidate;
            }
            self = delayed ? &det->fine_wholes[pending_index].data.shape
                   : piece ? &det->pieces[k] : &det->cands[ci];
            comp = *self;
            new_small = !piece && base > 1 &&
                (comp.maxx - comp.minx + 1 < lo || comp.maxy - comp.miny + 1 < lo);
            require_exact = new_small || confidence;
            if (!delayed && !piece && (new_small || dense_whole_is_new(base, s, last,
                    comp.maxx - comp.minx + 1, comp.maxy - comp.miny + 1))) {
                if (!remember_fine_whole(det, fine_whole_count, &comp, s, last)) return -2;
                fine_whole_count++;
                /* Terminal entries join the replay after all normal candidates, including
                 * entries discovered in this same terminal candidate loop. */
                if (terminal_level) {
                    if (fine_whole_count > (size_t)(INT32_MAX - ordinary_count)) return -2;
                    candidate_count = ordinary_count + (int)fine_whole_count;
                }
                self->row_y = -6; /* absent legacy parents must not acquire new child dependencies */
                continue;
            }
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
                /* Until tags are seen touching, a piece that is nearly all of its blob, and its blob's only
                 * piece, waits when the blob failed its first fit: it fails the same way. (A tag touching a
                 * bar or a board is a small part of its blob.) Untried, it is decoded if the level is redone. */
                const component *blob = det->piece_cand[k] >= 0 ? &det->cands[det->piece_cand[k]] : NULL;
                if (!necks && blob && fate == 7 && blob->row_x1 == 1 &&
                    5 * (comp.maxx - comp.minx + 1) * (comp.maxy - comp.miny + 1) >=
                        4 * (blob->maxx - blob->minx + 1) * (blob->maxy - blob->miny + 1)) {
                    self->row_y = -1;
                    continue;
                }
                if (fate == 6 || fate >= 9) {
                    self->row_y = whole->row_y; /* its own pieces share the fate */
                    continue;
                }
            }
            t_minx = comp.minx, t_maxx = comp.maxx, t_miny = comp.miny, t_maxy = comp.maxy;
            int bw = t_maxx - t_minx + 1, bh = t_maxy - t_miny + 1;
            /* Levels run fine to coarse; skip a blob whose centre a finer level already decoded. */
            double ccx = 0.5 * (comp.minx + comp.maxx) * s * sx + offx, ccy = 0.5 * (comp.miny + comp.maxy) * s * sy + offy;
            const double big_px = s * (bw * sx > bh * sy ? bw * sx : bh * sy);
            /* The earlier check ruled out the initial unsplit prefix; only appended centres
             * can cover this box now. The count guard also keeps pointer arithmetic off NULL. */
            int covered_count = found < max_out ? found : max_out;
            if ((covered_count > coverage_start &&
                 covered_footprint(out + coverage_start, det->out_centers + coverage_start, covered_count - coverage_start,
                         ccx, ccy, 0.3 * big_px + 2.0)) ||
                (piece && covered_footprint(det->piece_out, det->piece_centers, piece_found, ccx, ccy, 0.3 * big_px + 2.0))) {
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
            if (new_small) complete_small_outline(pick, comp.minx, comp.maxx, comp.miny, comp.maxy);
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
            /* The axis-aligned box grows when a thin tag rotates. Keep the old path for
             * ordinary shapes; new sparse or slender proposals need quad-relative fill and
             * compactness. A rectangle of aspect eight has area/perimeter^2 = 2/81. */
            int old_bad_area = best_area < 0.3 * bw * bh;
            extended_shape = !piece && (new_small || dense_whole_is_new(base, s, last, bw, bh) ||
                                           box_is_sparse(comp.count, bw, bh) || old_bad_area);
            if ((piece && old_bad_area) || (extended_shape &&
                (81.0 * best_area < 2.0 * perim_q * perim_q || comp.count < 0.2 * best_area))) {
                PROF_MARK(PROF_QUAD);
                TRACE(6);
                self->row_y = -6;
                continue;
            }
            if (extended_shape) {
                /* These wholes had no legacy candidate, or were rough-rejected with fate 6.
                 * Their additional fitting must not change old piece eligibility. */
                self->row_y = -6;
                if (covered_footprint(det->shape_out, det->shape_centers, shape_found,
                                      ccx, ccy, 0.3 * big_px + 2.0)) continue;
            }
            /* Short-code families match clutter more often even without bit correction.
             * A new shape in those families needs a complete dark border, using the same
             * samples as decoding. Existing shape paths retain the caller's threshold. */
            if (extended_shape && ((det->fam.piece_hamming < 0 && params->min_border_cells < 28) ||
                                   (new_small && params->max_hamming != 0))) {
                strict_border = *params;
                if (det->fam.piece_hamming < 0) strict_border.min_border_cells = 28;
                if (new_small) strict_border.max_hamming = 0;
                decode_params = &strict_border;
            }
            PROF_MARK(PROF_QUAD);
            for (int k = 0; k < 4; k++) {
                rough[k].x = pick[k][0] * (double)s * sx + offx;
                rough[k].y = pick[k][1] * (double)s * sy + offy;
            }
            /* Ordinary rough corners are dark samples. A completed small corner is a
             * geometric hypothesis from the same component bounds, not a new dark sample.
             * Retain the established mostly-outward fit window and final payload validation. */
            /* Sparse paired border samples reject noise before the expensive edge profiles.
             * Permit one weak side and leave terminal proposals unscreened: their ring can have
             * local clutter that a later level would normally resolve. Final decode is unchanged. */
            if ((shadow || piece || ((s <= 2 || (s <= 6 && !new_small)) && !last)) && params->min_border_cells >= 22) {
                vec2 q[4];
                double H[9];
                grow_quad(rough, 0.5 * s * smax, q);
                square_to_quad(q, H);
                int good_sides = 0;
                double all_sides = 0;
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
                    all_sides += contrast_sum;
                }
                /* A tag's quiet zone is brighter than its border all the way round; the interior
                 * fragments and clutter that dominate proposals average near zero. */
                if (good_sides < 3 || ((shadow || s <= 2 || (s <= 6 && !new_small)) && all_sides < (shadow ? 6.0 : 4.8) * params->min_contrast)) {
                    PROF_MARK(PROF_FIT1);
                    TRACE(7);
                    self->row_y = extended_shape ? -6 : -7;
                    continue;
                }
            }
            reach = 0.5 * ((1.5 * s * smax + 1.0) + (0.5 * s * smax + 1.0));
            edge_w = 0;
            initial_capture initial_samples;
            const int confidence_eligible = s > 1 && !shadow && !piece && !delayed &&
                !extended_shape && det->fam.piece_hamming >= 0 && found < max_out;
            if (!fit_initial_captured(&view, rough, fine, 1.5 * s * smax + 1.0, 0.5 * s * smax + 1.0,
                    params->min_contrast, &edge_w, cells, confidence_eligible ? &initial_samples : NULL)) {
                PROF_MARK(PROF_FIT1);
                TRACE(7);
                self->row_y = extended_shape ? -6 : -7;
                continue;
            }
            const double initial_width = edge_w;
            /* The first fit is already within a fraction of a cell: decode now, with two bits of
             * slack, and spend the refinement passes only on quads that are tags. A random quad
             * passes this only about 1.5% of the time. */
            PROF_MARK(PROF_FIT1);
            /* Levels overlap, so a blob that failed to decode at a finer level often comes back at this one.
             * Cached early failures use the caller's thresholds: a fitted quad within 0.3 px fails
             * again, including when the new proposal requires a stricter border. (The rough quads can match while the fits differ: the window grows with s.) */
            if (failed_before(det, fine, s, 0.3)) {
                PROF_MARK(PROF_DECODE1);
                TRACE(8);
                self->row_y = extended_shape ? -6 : -8;
                continue;
            }
            vec2 probe[4];
            lv.black = 255; lv.white = 0; lv.has_model = 0; lv.model_pending = 0;
            memcpy(probe, fine, sizeof probe);
            if (!decode_quad(det, &view, probe, decode_params, 2, &early, NULL, &lv)) {
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
                    if (decode_quad(det, &view, probe, decode_params, 2, &early, NULL, &lv)) {
                        memcpy(fine, finer, sizeof fine);
                        reach = r;
                        rescued = 1;
                    }
                }
                if (!rescued) {
                    PROF_MARK(PROF_DECODE1);
                    TRACE(8);
                    self->row_y = extended_shape ? -6 : -8;
                    /* Pieces only read the list. A stricter border failure also cannot prove
                     * that an ordinary proposal at another level will fail the caller's check. */
                    if (!piece && !extended_shape && decode_params == params) remember_failure(det, fine, s);
                    if (confidence_eligible) {
                        vec2 confidence[4];
                        if (confidence_quad(&initial_samples,rough,reach,confidence)) {
                            double movement = 0;
                            for (int side = 0; side < 4; side++)
                                movement += fabs(confidence[side].x - fine[side].x) +
                                            fabs(confidence[side].y - fine[side].y);
                            if (movement <= 0.02) continue;
                            napa_detection trial;
                            edge_levels trial_lv = {0};
                            trial_lv.black = 255;
                            if (decode_quad_impl(det,&view,confidence,decode_params,2,&trial,NULL,&trial_lv,1) &&
                                trial.hamming==0) {
                                if (!remember_confidence(det, fine_whole_count, confidence, reach,
                                        initial_width, &trial_lv, trial.id, s, last)) return -2;
                                fine_whole_count++;
                                if (terminal_level) {
                                    if (fine_whole_count > (size_t)(INT32_MAX - ordinary_count)) return -2;
                                    candidate_count = ordinary_count + (int)fine_whole_count;
                                }
                            }
                        }
                    }
                    continue;
                }
            }
            PROF_MARK(PROF_DECODE1);
refine_candidate:
            cell = 0;
            for (int k = 0; k < 4; k++) {
                double ex = fine[(k + 1) & 3].x - fine[k].x, ey = fine[(k + 1) & 3].y - fine[k].y;
                cell += sqrt(ex * ex + ey * ey) / (4.0 * W);
            }
            /* With a camera gamma set, refine on a tone-linearized copy of the tag's own rectangle: blur is linear in light,
             * so edges there sit where the fits expect. A tag too dark to clear the contrast floor in linear terms stays raw. */
            if (det->has_camera_gamma && !lin_tried && linear_contrast_enough(det, &lv, params)) {
                memcpy(raw_fine, fine, sizeof raw_fine), raw_reach = reach, raw_edge_w = edge_w, raw_lv = lv;
                double x0 = fine[0].x, x1 = fine[0].x, y0 = fine[0].y, y1 = fine[0].y;
                for (int k = 1; k < 4; k++) {
                    if (fine[k].x < x0) x0 = fine[k].x;
                    if (fine[k].x > x1) x1 = fine[k].x;
                    if (fine[k].y < y0) y0 = fine[k].y;
                    if (fine[k].y > y1) y1 = fine[k].y;
                }
                const double margin = NAPA_LINEAR_MARGIN_CELLS * cell + NAPA_LINEAR_MARGIN_PIXELS;
                int ix0 = (int)(x0 - margin), ix1 = (int)(x1 + margin) + 1, iy0 = (int)(y0 - margin), iy1 = (int)(y1 + margin) + 1;
                if (ix0 < 0) ix0 = 0;
                if (iy0 < 0) iy0 = 0;
                if (ix1 > width - 1) ix1 = width - 1;
                if (iy1 > height - 1) iy1 = height - 1;
                if (det->linear_capacity < (size_t)width * height) {
                    free(det->linear_frame);
                    det->linear_frame = (uint8_t *)calloc((size_t)width * height, 1);
                    det->linear_capacity = det->linear_frame != NULL ? (size_t)width * height : 0;
                }
                if (det->linear_frame != NULL && ix1 > ix0 && iy1 > iy0) {
                    for (int y = iy0; y <= iy1; y++) {
                        const uint8_t *src = img + (size_t)y * stride;
                        uint8_t *dst = det->linear_frame + (size_t)y * width;
                        for (int x = ix0; x <= ix1; x++) dst[x] = det->linearize[src[x]];
                    }
                    lin_view.p = det->linear_frame, lin_view.stride = width;
                    rview = &lin_view;
                }
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
                ok = fit_sides(rview, fine, finer, reach, reach, params->min_contrast, 0, 8, &edge_w, &lv, cells);
                if (!ok) break;
                memcpy(fine, finer, sizeof fine);
            }
            /* One more pass at the converged window, re-centred on the last estimate, with every
             * edge sample. */
            vec2 alternative[4], curved[4], lens_quad[4];
            int lens_ok = 0;
            double side_widths_seen[2] = {1, 0}; /* the widest side's edge over the narrowest, and the widest */
            int accepted_blur = 0, accepted_clip = 0;
            if (ok) ok = fit_sides_alternate(rview, fine, finer, reach, reach, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv, cells, alternative, curved, lens_quad, &lens_ok, side_widths_seen);
            if (!ok && rview != &view) { /* the linear patch failed: go again on the frame's own pixels */
                lin_tried = 1, rview = &view;
                memcpy(fine, raw_fine, sizeof fine), reach = raw_reach, edge_w = raw_edge_w, lv = raw_lv;
                goto refine_candidate;
            }
            if (!ok) {
                PROF_MARK(PROF_REFINE);
                TRACE(9);
                self->row_y = extended_shape ? -6 : -9;
                continue;
            }
            PROF_MARK(PROF_REFINE);
            napa_detection d;
            double side_level[4];
            edge_levels final_lv; final_lv.black = 255; final_lv.white = 0; final_lv.has_model = 0; final_lv.model_pending = 0;
            int decoded_ = decode_quad_impl(det, rview, finer, decode_params, 0, &d, side_level, &final_lv, confidence);
            if (decoded_ && require_exact && d.hamming != 0) decoded_ = 0;
            /* The ordinary fit may fail decoding because its edge levels were mixed by blur.
             * Reuse the already-fitted alternate, with the same strict decode and path limits. */
            if (!decoded_ && lv.has_model) {
                /* Preserve independent-light recovery when ordinary decoding failed. */
                int alternative_ok = view.invert || fit_independent_alternate(rview, fine, alternative,
                                            reach, params->min_contrast, cells, &lv);
                if (alternative_ok) {
                    if (signed_area(alternative) < 0) {
                        vec2 t = alternative[1]; alternative[1] = alternative[3]; alternative[3] = t;
                    }
                    double move = 0;
                    for (int k = 0; k < 4; k++)
                        move += fabs(alternative[k].x - finer[k].x) + fabs(alternative[k].y - finer[k].y);
                    if (move > 0.02 && decode_quad_impl(det, rview, alternative, decode_params, 0, &d, side_level, &final_lv, confidence) &&
                        (!require_exact || d.hamming == 0) && d.id == early.id && d.hamming <= params->max_hamming && d.hamming <= det->fam.max_hamming &&
                        (!shadow || d.hamming == 0) && (!piece || d.hamming <= det->fam.piece_hamming)) {
                        accepted_blur = 1;
                        memcpy(finer, alternative, sizeof finer);
                        decoded_ = 1;
                    }
                }
            }
            PROF_MARK(PROF_DECODE2);
            if (!decoded_ && rview != &view) {
                lin_tried = 1, rview = &view;
                memcpy(fine, raw_fine, sizeof fine), reach = raw_reach, edge_w = raw_edge_w, lv = raw_lv;
                goto refine_candidate;
            }
            if (!decoded_) {
                TRACE(10);
                self->row_y = extended_shape ? -6 : -10;
                continue;
            }
            /* A side whose border cells read white was fitted one cell out, on the quiet zone's outer
             * edge (the payload still decodes through error correction). Move it in by a cell, refit,
             * and keep the result if it decodes the same tag with a darker border. */
            int bad = 0, corrected_border = 0;
            double edge_shift = 0; /* the measured bias moved every side by this, without visible clipping */
            double measured_bias = 0, whole_share = 0;
            double level_before = 0;
            for (int k = 0; k < 4; k++) {
                bad |= (side_level[k] > 0.3) << k;
                level_before += side_level[k];
            }
            double accepted_border_level = level_before;
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
                int refit = r >= 1.0 && fit_sides(rview, moved, first, r, r, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv, cells);
                if (refit && r2 < r) refit = fit_sides(rview, first, fixed, r2, r2, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv, cells);
                else if (refit) memcpy(fixed, first, sizeof fixed);
                if (refit && decode_quad_impl(det, rview, fixed, decode_params, 0, &d2, side2, NULL, confidence) &&
                    (!require_exact || d2.hamming == 0) && d2.id == d.id) {
                    for (int k = 0; k < 4; k++) level_after += side2[k];
                    if (level_after < level_before - 0.2) {
                        d = d2;
                        corrected_border = 1;
                        accepted_border_level = level_after;
                        memcpy(finer, fixed, sizeof finer);
                    }
                }
            }
            /* Clipped levels, and gamma-encoded images under blur, bias every edge the same way: the half-level
             * isn't the true edge. Measure the bias from the border's own width, which is one cell whatever the
             * camera, and undo it. Without visible clipping only a clear narrowing counts: gamma can only narrow a
             * black border, while a border that looks wider than a cell is blur, which needs no correction. */
            double bias_estimate = 0; /* set even when the sides disagree and the correction is set aside */
            {
                double disagreement = 0;
                const int samples = border_bias(&det->fam, rview, finer, final_lv.black, final_lv.white, &bias_estimate, &disagreement);
                const double delta = bias_estimate;
                const int clipped = final_lv.white >= 250 || final_lv.black <= 4;
                /* How much of the correction to apply: all of it when the sides clearly agree and the bias is clearly there. */
                double share = samples >= 4 && fabs(delta) < 1.5
                    ? (1 - fade(disagreement, 1 - NAPA_FADE_AGREEMENT, 1 + NAPA_FADE_AGREEMENT)) *
                      (clipped ? fade(fabs(delta), 0.02 - NAPA_FADE_CLIPPED_BIAS, 0.02 + NAPA_FADE_CLIPPED_BIAS) : fade(delta, 0.1 - NAPA_FADE_BIAS_STEP, 0.1 + NAPA_FADE_BIAS_STEP))
                    : 0;
                /* The border also looks narrow when the artwork's border simply is not one cell wide. A real tone bias
                 * shows up at the payload's edges by the same amount; keep the correction only as far as they agree. */
                double payload_delta = 0;
                /* A bowed tag's payload lines aren't on the straight grid this measures against, so it isn't asked. */
                vec2 bowed[4];
                double bow = 0;
                memcpy(bowed, curved, sizeof bowed);
                if (signed_area(bowed) < 0) { vec2 swap = bowed[1]; bowed[1] = bowed[3]; bowed[3] = swap; }
                for (int k = 0; k < 4; k++) {
                    const double span = hypot(bowed[k].x - finer[k].x, bowed[k].y - finer[k].y);
                    if (span > bow) bow = span;
                }
                if (share > 0 && bow < NAPA_CURVED_TAG_MIN && payload_bias(&det->fam, rview, finer, final_lv.black, final_lv.white, &payload_delta))
                    share *= fade(payload_delta / delta, 0.25, 0.75);
                if (share > 0) {
                    vec2 grown[4];
                    napa_detection d3;
                    grow_quad(finer, share * delta, grown);
                    if (decode_quad_impl(det, rview, grown, decode_params, 0, &d3, NULL, NULL, confidence) &&
                        (!require_exact || d3.hamming == 0) && d3.id == d.id) {
                        d = d3;
                        measured_bias = delta;
                        whole_share = share;
                        if (clipped) accepted_clip = 1;
                        else edge_shift = share * delta;
                    }
                }
            }
            if (signed_area(alternative) < 0) { vec2 t = alternative[1]; alternative[1] = alternative[3]; alternative[3] = t; }
            double alternate_move = 0;
            for (int k = 0; k < 4; k++) alternate_move += fabs(alternative[k].x - finer[k].x) + fabs(alternative[k].y - finer[k].y);
            if (alternate_move > 0.02 && final_lv.white < 250 && final_lv.black > 4) {
                napa_detection refined;
                double alternative_border[4];
                /* The fits share the same biased edge location, so they share its correction. */
                if (edge_shift != 0) grow_quad(alternative, edge_shift, alternative);
                if (decode_quad_impl(det, rview, alternative, decode_params, 0, &refined, alternative_border, NULL, confidence) &&
                    (!require_exact || refined.hamming == 0) && refined.id == d.id && refined.hamming <= params->max_hamming &&
                    refined.hamming <= det->fam.max_hamming && (!shadow || refined.hamming == 0) &&
                    (!piece || refined.hamming <= det->fam.piece_hamming)) {
                    double border_sum = 0;
                    for (int k = 0; k < 4; k++) border_sum += alternative_border[k];
                    if (!corrected_border || border_sum <= accepted_border_level) { d = refined; accepted_blur = 1; }
                }
            }
            double curve_span = 0;
            int corner_out[4], near_frame = 0;
            {
                vec2 ordered[4];
                double clear_span = 0;
                memcpy(ordered, curved, sizeof ordered);
                if (signed_area(ordered) < 0) { vec2 swap = ordered[1]; ordered[1] = ordered[3]; ordered[3] = swap; }
                for (int k = 0; k < 4; k++) {
                    corner_out[k] = !inside_frame(rview, finer[k]) || !inside_frame(rview, ordered[k]);
                    near_frame |= corner_out[k];
                }
                for (int k = 0; k < 4; k++) {
                    double span = hypot(ordered[k].x - finer[k].x, ordered[k].y - finer[k].y);
                    if (span > curve_span) curve_span = span;
                    if (span > clear_span && !corner_out[k] && !corner_out[(k + 1) & 3] && !corner_out[(k + 3) & 3]) clear_span = span;
                }
                /* A side cut by the frame has no samples at its true end, so its curve is extrapolated. Believe it
                 * only if a corner whose two sides both lie inside the frame shows the bow too. */
                if (clear_span >= NAPA_CURVED_TAG_MIN) near_frame = 0;
                /* A lens bows every side; one side bowing alone, by a large part of a cell, is a poor curve fit. */
                if (curve_span >= NAPA_CURVED_TAG_MIN) {
                    double span[4], top = 0;
                    int moved_corners = 0;
                    for (int k = 0; k < 4; k++) {
                        span[k] = hypot(ordered[k].x - finer[k].x, ordered[k].y - finer[k].y);
                        if (span[k] > top) top = span[k];
                    }
                    for (int k = 0; k < 4; k++) moved_corners += span[k] >= NAPA_BOW_MIN_SHARE * top;
                    if (moved_corners < 3 && top > NAPA_BOW_LOPSIDED_CELLS * cell) curve_span = 0;
                }
            }
            if (near_frame) curve_span = 0;
            if (lens_ok) {
                napa_detection proposal;
                double lens_bias = measured_bias;
                if (lens_bias == 0 && final_lv.white < 250 && bias_estimate >= NAPA_LOCAL_BIAS_MIN && bias_estimate < 1.5) lens_bias = bias_estimate;
                if (signed_area(lens_quad) < 0) { vec2 swap = lens_quad[1]; lens_quad[1] = lens_quad[3]; lens_quad[3] = swap; }
                if (lens_bias != 0) grow_quad(lens_quad, lens_bias, lens_quad);
                if (decode_quad_impl(det, rview, lens_quad, decode_params, 0, &proposal, NULL, NULL, confidence) &&
                    (!require_exact || proposal.hamming == 0) && proposal.id == d.id && proposal.hamming <= d.hamming &&
                    (!shadow || proposal.hamming == 0) && (!piece || proposal.hamming <= det->fam.piece_hamming)) {
                    d = proposal;
                    near_frame = 1, curve_span = 0; /* no later step should move these corners */
                }
            }
            /* The final profile points precede these corrections, so keep any later geometry winner. */
            if (!corrected_border && !accepted_clip && !accepted_blur) {
                if (signed_area(curved) < 0) {
                    vec2 swap = curved[1]; curved[1] = curved[3]; curved[3] = swap;
                }
                double curve_move = 0;
                for (int k = 0; k < 4; k++)
                    curve_move += fabs(curved[k].x - finer[k].x) + fabs(curved[k].y - finer[k].y);
                if (curve_move > 0.02 && !near_frame) {
                    napa_detection proposal;
                    if (edge_shift != 0) grow_quad(curved, edge_shift, curved);
                    if (decode_quad_impl(det, rview, curved, decode_params, 0, &proposal, NULL, NULL, confidence) &&
                        (!require_exact || proposal.hamming == 0) && proposal.id == d.id && proposal.hamming <= d.hamming &&
                        proposal.hamming <= params->max_hamming && proposal.hamming <= det->fam.max_hamming &&
                        (!shadow || proposal.hamming == 0) && (!piece || proposal.hamming <= det->fam.piece_hamming))
                        d = proposal;
                }
            }
            {
                vec2 local[4];
                double mx = 0, my = 0, side = 0;
                for (int k = 0; k < 4; k++) {
                    local[k].x = d.corners[2 * k], local[k].y = d.corners[2 * k + 1];
                    mx += 0.25 * local[k].x, my += 0.25 * local[k].y;
                }
                for (int k = 0; k < 4; k++)
                    side += 0.25 * hypot(local[(k + 1) & 3].x - local[k].x, local[(k + 1) & 3].y - local[k].y);
                const double one_cell = side / W;
                int moved = 0;
                vec2 start[4];
                memcpy(start, local, sizeof start);
                /* Sides that disagree about the border's width set aside the correction for the whole tag, as on a
                 * bowed one. A large narrowing still says how far the half level sits from the edge; widening is blur. */
                /* How much each reason to refit applies, from 0 to 1, so a tag near a limit moves part of the way. */
                double straight_weight = 0;
                if (whole_share < 1 && final_lv.white < 250 && bias_estimate < 1.5) {
                    straight_weight = (1 - whole_share) * fade(bias_estimate, NAPA_LOCAL_BIAS_MIN - NAPA_FADE_BIAS, NAPA_LOCAL_BIAS_MIN + NAPA_FADE_BIAS) * fade(1.5 - bias_estimate, 0, 0.2) *
                                      (1 - fade(side_widths_seen[0], NAPA_LOCAL_MAX_WIDTH_RATIO - NAPA_FADE_WIDTH_RATIO, NAPA_LOCAL_MAX_WIDTH_RATIO + NAPA_FADE_WIDTH_RATIO));
                    if (straight_weight > 0 && measured_bias == 0) measured_bias = bias_estimate;
                }
                const double bowed_weight = near_frame || curve_span > NAPA_LOCAL_MAX_BOW ? 0 : fade(curve_span, NAPA_CURVED_TAG_MIN - NAPA_FADE_BOW, NAPA_CURVED_TAG_MIN + NAPA_FADE_BOW);
                /* Straight tags whose whole-tag correction stood are already right. */
                const int refittable = !near_frame && (bowed_weight > 0 || straight_weight > 0);
                for (int k = 0; k < 4; k++) {
                    vec2 fitted;
                    corner_sides sides;
                    if (corner_out[k]) continue;
                    if (refittable && measure_corner_sides(start, k, one_cell, &sides)) {
                        double weight = bowed_weight;
                        if (weight < 1 && straight_weight > 0) {
                            const double disagree = straight_weight * fade(corner_sides_offset(rview, start[k], &sides, one_cell, measured_bias), NAPA_LOCAL_EDGE_DISAGREE - NAPA_FADE_EDGE, NAPA_LOCAL_EDGE_DISAGREE + NAPA_FADE_EDGE);
                            if (disagree > weight) weight = disagree;
                        }
                        if (weight > 0 && refit_corner(rview, start[k], &sides, one_cell, measured_bias, &fitted)) {
                            const double dx = fitted.x - start[k].x, dy = fitted.y - start[k].y;
                            weight *= fade(hypot(dx, dy), NAPA_LOCAL_MIN_MOVE - NAPA_FADE_MOVE, NAPA_LOCAL_MIN_MOVE + NAPA_FADE_MOVE);
                            if (weight > 0) local[k].x = start[k].x + weight * dx, local[k].y = start[k].y + weight * dy, moved = 1;
                            continue;
                        }
                        if (weight > 0 && bowed_weight < 1) continue;
                    }
                    if (curve_span < NAPA_CURVED_TAG_MIN) continue;
                    double px = d.corners[2 * k], py = d.corners[2 * k + 1], norm = hypot(px - mx, py - my);
                    double ux = (px - mx) / norm, uy = (py - my) / norm;
                    vec2 prev = local[(k + 3) & 3], next = local[(k + 1) & 3];
                    double ax = next.x - px, ay = next.y - py, bx = prev.x - px, by = prev.y - py;
                    double al = hypot(ax, ay), bl = hypot(bx, by);
                    ax /= al, ay /= al, bx /= bl, by /= bl;
                    double white = sample_bilinear(rview, px + 0.2 * one_cell * ux - one_cell * ax, py + 0.2 * one_cell * uy - one_cell * ay);
                    double white2 = sample_bilinear(rview, px + 0.2 * one_cell * ux - one_cell * bx, py + 0.2 * one_cell * uy - one_cell * by);
                    if (white2 > white) white = white2;
                    double inside = sample_bilinear(rview, px - one_cell * ux, py - one_cell * uy);
                    if (sample_bilinear(rview, px + one_cell * ux, py + one_cell * uy) >= 0.5 * (inside + white)) continue;
                    double nx = px, ny = py;
                    if (refine_x_corner(rview, &nx, &ny) && hypot(nx - px, ny - py) < 3.0) local[k].x = nx, local[k].y = ny, moved = 1;
                }
                napa_detection polished;
                if (moved && edge_w <= NAPA_POLISH_MAX_EDGE_CELLS * one_cell &&
                    decode_quad_impl(det, rview, local, decode_params, 0, &polished, NULL, NULL, confidence) &&
                    polished.id == d.id && polished.hamming <= d.hamming) d = polished;
            }
            {
                tone_measure tone;
                if (tone_measure_tag(det, rview, &d, det->align_corners ? side_widths_seen[1] : edge_w, det->align_corners, &tone)) {
                    vec2 sized[4], aligned[4];
                    napa_detection redone;
                    if (tone_corrected_quad(&d, &tone, sized) &&
                        decode_quad_impl(det, rview, sized, decode_params, 0, &redone, NULL, NULL, confidence) &&
                        redone.id == d.id && redone.hamming <= d.hamming) d = redone;
                    if (det->align_corners && tone_aligned_quad(rview, &d, &tone, aligned) &&
                        decode_quad_impl(det, rview, aligned, decode_params, 0, &redone, NULL, NULL, confidence) &&
                        redone.id == d.id && redone.hamming <= d.hamming) d = redone;
                }
            }
            if ((shadow || require_exact) && d.hamming != 0) continue;
            if (confidence) {
                if (d.id != early.id) continue;
                det->fine_wholes[pending_index].data.completed = d;
                det->fine_wholes[pending_index].kind = FINE_WHOLE_CONFIDENCE_FINISHED;
                continue; /* No established fates, coverage, outputs or restart changes. */
            }
            if (piece) {
                /* Pieces are many and mostly not tags: correct fewer bits (in tag36h11, three admitted
                 * chance matches from random-payload grids), and keep them apart until the end. */
                self->row_y = extended_shape ? -6 : -12;
                if (d.hamming > det->fam.piece_hamming || already_found_footprint(det->piece_out, det->piece_centers, piece_found, &d)) continue;
                if ((size_t)piece_found == det->piece_out_capacity) {
                    size_t cap = det->piece_out_capacity ? 2 * det->piece_out_capacity : 64;
                    napa_detection *p = (napa_detection *)realloc(det->piece_out, cap * sizeof(napa_detection));
                    if (p == NULL) return -2;
                    det->piece_out = p;
                    det->piece_out_capacity = cap;
                }
                if (!necks && !touching) touching = corners_touch(rview, &d, cells);
                if (!save_center(&det->piece_centers, &det->piece_center_capacity, piece_found, &d)) return -2;
                det->piece_out[piece_found++] = d;
                continue;
            }
            if (already_found_footprint(out, det->out_centers, found < max_out ? found : max_out, &d)) {
                TRACE(11);
                self->row_y = extended_shape ? -6 : -11;
                continue;
            }
            if (extended_shape) {
                if (already_found_footprint(det->shape_out, det->shape_centers, shape_found, &d)) continue;
                if ((size_t)shape_found == det->shape_out_capacity) {
                    size_t cap = det->shape_out_capacity ? 2 * det->shape_out_capacity : 64;
                    if (cap < det->shape_out_capacity || cap > SIZE_MAX / sizeof(napa_detection)) return -2;
                    napa_detection *p = (napa_detection *)realloc(det->shape_out, cap * sizeof(napa_detection));
                    if (p == NULL) return -2;
                    det->shape_out = p;
                    det->shape_out_capacity = cap;
                }
                if (!save_center(&det->shape_centers, &det->shape_center_capacity, shape_found, &d)) return -2;
                det->shape_out[shape_found++] = d;
                TRACE(12);
                continue; /* No ordinary coverage, output count or restart decision changes. */
            }
            TRACE(12);
            self->row_y = -12;
            if (split && !necks && !touching) touching = corners_touch(rview, &d, cells);
            if (found < max_out) {
                if (!save_center(&det->out_centers, &det->out_center_capacity, found, &d)) return -2;
                out[found] = d;
            }
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
    /* Retain the established whole detections and corners before adding new shapes.
     * These ordinary additions precede optional pieces to keep the unsplit prefix stable. */
    for (int k = 0; k < shape_found && found < max_out; k++) {
        if (!already_found_footprint(out, det->out_centers, found, &det->shape_out[k])) {
            if (!save_center(&det->out_centers, &det->out_center_capacity, found, &det->shape_out[k])) return -2;
            out[found++] = det->shape_out[k];
        }
    }
    /* Confidence successes cannot cover old proposals during processing. Append
     * them after old wholes/shapes and before optional pieces, preserving their prefix. */
    for (size_t k = 0; k < fine_whole_count && found < max_out; k++) {
        if (det->fine_wholes[k].kind != FINE_WHOLE_CONFIDENCE_FINISHED) continue;
        const napa_detection *d = &det->fine_wholes[k].data.completed;
        if (already_found_footprint(out, det->out_centers, found, d)) continue;
        if (!save_center(&det->out_centers, &det->out_center_capacity, found, d)) return -2;
        out[found++] = *d;
    }
    /* Pieces come last, so the other detections are unchanged by split_touching. */
    for (int k = 0; k < piece_found && found < max_out; k++) {
        if (!already_found_footprint(out, det->out_centers, found, &det->piece_out[k])) {
            if (!save_center(&det->out_centers, &det->out_center_capacity, found, &det->piece_out[k])) return -2;
            out[found++] = det->piece_out[k];
        }
    }
    return found < max_out ? found : max_out;
}

/* Keep ordinary results first and untouched; recovery shares their coverage/duplicate checks.
 * Each pass resets its failed-quad cache: a failure under one threshold is not evidence for the other. */
int napa_detect(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                const napa_params *params, napa_detection *out, int max_out) {
    return detect_pass(det, img, width, height, stride, params, out, max_out, 0, 0);
}

int napa_detect_shadows_configured(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                        const napa_params *params, napa_detection *out, int max_out, int shadow_min_tag_px) {
    if (params == NULL || (shadow_min_tag_px != 0 &&
        (shadow_min_tag_px < params->min_tag_px || shadow_min_tag_px < 8 || shadow_min_tag_px > 30000))) return -1;
    if (max_out > 0 && out == NULL) return -1;
    if (det == NULL) return -1;
    /* Recovery samples the same levels, so it reuses their extrema unless it samples more coarsely. */
    det->extrema_ready = 0;
    det->extrema_wanted = det->fam.piece_hamming >= 0;
    int n = detect_pass(det, img, width, height, stride, params, out, max_out, 0, 0);
    /* A shaped ordinary pass sampled other places than the recovery levels will: nothing of its can be reused. */
    det->extrema_ready = det->extrema_wanted && !det->tag_shape, det->extrema_wanted = 0;
    if (n < 0 || n == max_out || width < 16 || height < 16) return n;
    /* Extra proposals must decode exactly. Families whose exact-code chance-match rate exceeds
     * the split recovery budget do not get an additional search. */
    if (det->fam.piece_hamming < 0) return n;
    napa_params recovery = *params;
    if (shadow_min_tag_px) recovery.min_tag_px = shadow_min_tag_px;
    recovery.max_hamming = 0;
    recovery.split_touching = 0;
    return detect_pass(det, img, width, height, stride, &recovery, out, max_out, n, 1);
}

int napa_detect_shadows(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                        const napa_params *params, napa_detection *out, int max_out) {
    return napa_detect_shadows_configured(det, img, width, height, stride, params, out, max_out, 0);
}

/* Soft decoding: a post-pass for tiny tags whose bits can't be read one by one. Each kept decode attempt is compared,
 * as a whole picture of the tag, against every code. See docs/detection.md for what it does and doesn't promise. */
typedef struct {
    float blur[NAPA_SOFT_SAMPLES][6];  /* how much each payload row or column reaches each row or column of samples */
    float base[NAPA_SOFT_AREA];        /* the tag with its payload cells at mid grey */
    float sigma[NAPA_SOFT_IDS];        /* each ID's template spread: rotating a tag doesn't change it */
    float nontag_sigma[NAPA_SOFT_NONTAGS];
    double free_fit[NAPA_SOFT_FIT][NAPA_SOFT_FIT]; /* inverse normal matrix for [1, base, each payload cell] */
} soft_bin;

struct soft_tables {
    float payload[NAPA_SOFT_CODES][36]; /* +1 white, -1 black, for every code in every rotation as observed */
    float nontag[NAPA_SOFT_NONTAGS][36];
    soft_bin bins[NAPA_SOFT_BINS];
};

static double normal_cdf(double x) { return 0.5 * (1 + erf(x / sqrt(2.0))); }

static inline double soft_position(int i) { return (i + 0.5) / 3 - 1; } /* in cells: the tag and its white ring span -1 to 9 */

static void soft_template(const soft_bin *B, const float p[36], float out[NAPA_SOFT_AREA]) {
    float across[NAPA_SOFT_SAMPLES][6];
    for (int r = 0; r < NAPA_SOFT_SAMPLES; r++) {
        for (int c = 0; c < 6; c++) {
            float v = 0;
            for (int k = 0; k < 6; k++) v += B->blur[r][k] * p[k * 6 + c];
            across[r][c] = v;
        }
    }
    for (int r = 0; r < NAPA_SOFT_SAMPLES; r++) {
        for (int c = 0; c < NAPA_SOFT_SAMPLES; c++) {
            float v = B->base[r * NAPA_SOFT_SAMPLES + c];
            for (int k = 0; k < 6; k++) v += across[r][k] * B->blur[c][k];
            out[r * NAPA_SOFT_SAMPLES + c] = v;
        }
    }
}

static float standardize(float *a, int n) {
    float mean = 0, squares = 0;
    for (int i = 0; i < n; i++) mean += a[i];
    mean /= (float)n;
    for (int i = 0; i < n; i++) a[i] -= mean, squares += a[i] * a[i];
    const float sigma = sqrtf(squares / (float)n);
    if (sigma > 1e-6f) for (int i = 0; i < n; i++) a[i] /= sigma;
    else for (int i = 0; i < n; i++) a[i] = 0;
    return sigma;
}

static float template_spread(const soft_bin *B, const float p[36]) {
    float t[NAPA_SOFT_AREA];
    soft_template(B, p, t);
    return standardize(t, NAPA_SOFT_AREA);
}

static void nontag_payload(int kind, float p[36]) {
    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) {
            int white = 0;
            switch (kind) {
            case 1: white = c >= 3; break;
            case 2: white = c < 3; break;
            case 3: white = r < 3; break;
            case 4: white = r >= 3; break;
            case 5: white = (r < 3 && c < 3) || (r >= 3 && c >= 3); break;
            case 6: white = c >= 2 && c < 4; break;
            case 7: white = r >= 2 && r < 4; break;
            default: break; /* blank */
            }
            p[r * 6 + c] = white ? 1.0f : -1.0f;
        }
    }
}

typedef struct {
    float sum_base, projection[36];
} soft_projection;

/* Inverts a symmetric positive definite matrix in place by Gauss-Jordan elimination. */
static int invert_fit_matrix(double a[NAPA_SOFT_FIT][NAPA_SOFT_FIT]) {
    double m[NAPA_SOFT_FIT][2 * NAPA_SOFT_FIT];
    for (int r = 0; r < NAPA_SOFT_FIT; r++)
        for (int c = 0; c < NAPA_SOFT_FIT; c++) m[r][c] = a[r][c], m[r][NAPA_SOFT_FIT + c] = r == c;
    for (int p = 0; p < NAPA_SOFT_FIT; p++) {
        int pivot = p;
        for (int r = p + 1; r < NAPA_SOFT_FIT; r++) if (fabs(m[r][p]) > fabs(m[pivot][p])) pivot = r;
        if (fabs(m[pivot][p]) < 1e-12) return 0;
        if (pivot != p) for (int c = 0; c < 2 * NAPA_SOFT_FIT; c++) { double w = m[p][c]; m[p][c] = m[pivot][c]; m[pivot][c] = w; }
        const double scale = m[p][p];
        for (int c = 0; c < 2 * NAPA_SOFT_FIT; c++) m[p][c] /= scale;
        for (int r = 0; r < NAPA_SOFT_FIT; r++) {
            if (r == p) continue;
            const double f = m[r][p];
            if (f != 0) for (int c = 0; c < 2 * NAPA_SOFT_FIT; c++) m[r][c] -= f * m[p][c];
        }
    }
    for (int r = 0; r < NAPA_SOFT_FIT; r++)
        for (int c = 0; c < NAPA_SOFT_FIT; c++) a[r][c] = m[r][NAPA_SOFT_FIT + c];
    return 1;
}

static int soft_prepare_fit(soft_bin *B) {
    float *columns = (float *)malloc(NAPA_SOFT_FIT * NAPA_SOFT_AREA * sizeof *columns);
    if (columns == NULL) return 0;
    for (int i = 0; i < NAPA_SOFT_AREA; i++) columns[i] = 1, columns[NAPA_SOFT_AREA + i] = B->base[i];
    for (int k = 0; k < 36; k++)
        for (int r = 0; r < NAPA_SOFT_SAMPLES; r++)
            for (int c = 0; c < NAPA_SOFT_SAMPLES; c++)
                columns[(k + 2) * NAPA_SOFT_AREA + r * NAPA_SOFT_SAMPLES + c] = B->blur[r][k / 6] * B->blur[c][k % 6];
    for (int r = 0; r < NAPA_SOFT_FIT; r++) {
        for (int c = r; c < NAPA_SOFT_FIT; c++) {
            double v = 0;
            for (int i = 0; i < NAPA_SOFT_AREA; i++) v += (double)columns[r * NAPA_SOFT_AREA + i] * columns[c * NAPA_SOFT_AREA + i];
            B->free_fit[r][c] = B->free_fit[c][r] = v;
        }
    }
    free(columns);
    return invert_fit_matrix(B->free_fit);
}

/* The share of the freely fitted payload amplitude that contradicts the code. Blur leaves cell centres unreliable,
 * but a fit that knows the blur recovers each cell, and a real tag's cells then agree with its code almost entirely. */
static double soft_disagreement(const soft_bin *B, const soft_projection *P, const float code[36]) {
    double right[NAPA_SOFT_FIT], fit[NAPA_SOFT_FIT];
    right[0] = 0, right[1] = P->sum_base;
    for (int k = 0; k < 36; k++) right[k + 2] = P->projection[k];
    for (int r = 0; r < NAPA_SOFT_FIT; r++) {
        double v = 0;
        for (int c = 0; c < NAPA_SOFT_FIT; c++) v += B->free_fit[r][c] * right[c];
        fit[r] = v;
    }
    const double sign = fit[1] + 1e-9 >= 0 ? 1 : -1;
    double against = 0, amplitude = 0;
    for (int k = 0; k < 36; k++) {
        const double opposed = -code[k] * sign * fit[k + 2];
        if (opposed > 0) against += opposed;
        amplitude += fabs(fit[k + 2]);
    }
    return against / (amplitude > 1e-9 ? amplitude : 1e-9);
}

static int soft_prepare(napa_detector *det) {
    if (det->soft != NULL) return 1;
    soft_tables *T = (soft_tables *)malloc(sizeof *T);
    if (T == NULL) return 0;
    const family_layout *L = &det->fam;
    for (int t = 0; t < NAPA_SOFT_CODES; t++) {
        for (int k = 0; k < 36; k++) {
            const int r = L->cell[k] / L->grid - 1, c = L->cell[k] % L->grid - 1;
            T->payload[t][r * 6 + c] = (L->codes[t] >> (35 - k) & 1) ? 1.0f : -1.0f;
        }
    }
    for (int m = 0; m < NAPA_SOFT_NONTAGS; m++) nontag_payload(m, T->nontag[m]);
    for (int b = 0; b < NAPA_SOFT_BINS; b++) {
        soft_bin *B = &T->bins[b];
        const double blur = (b + NAPA_SOFT_FIRST_BIN) * 0.02;
        float full[NAPA_SOFT_SAMPLES][10], ring[10][10], row[NAPA_SOFT_SAMPLES][10];
        for (int r = 0; r < NAPA_SOFT_SAMPLES; r++) {
            for (int c = 0; c < 10; c++) {
                const double a = c - 1.0;
                full[r][c] = (float)(normal_cdf((soft_position(r) - a) / blur) - normal_cdf((soft_position(r) - a - 1) / blur));
            }
            for (int c = 0; c < 6; c++) B->blur[r][c] = full[r][c + 2];
        }
        for (int r = 0; r < 10; r++)
            for (int c = 0; c < 10; c++)
                ring[r][c] = (r == 0 || r == 9 || c == 0 || c == 9) ? 1.0f : (r >= 2 && r < 8 && c >= 2 && c < 8) ? 0.0f : -1.0f;
        for (int r = 0; r < NAPA_SOFT_SAMPLES; r++) {
            for (int c = 0; c < 10; c++) {
                float v = 0;
                for (int k = 0; k < 10; k++) v += full[r][k] * ring[k][c];
                row[r][c] = v;
            }
        }
        for (int r = 0; r < NAPA_SOFT_SAMPLES; r++) {
            for (int c = 0; c < NAPA_SOFT_SAMPLES; c++) {
                float v = 0;
                for (int k = 0; k < 10; k++) v += row[r][k] * full[c][k];
                B->base[r * NAPA_SOFT_SAMPLES + c] = v;
            }
        }
        for (int i = 0; i < NAPA_SOFT_IDS; i++) B->sigma[i] = 0;
        for (int t = 0; t < NAPA_SOFT_CODES; t++)
            if (B->sigma[L->ids[t]] == 0) B->sigma[L->ids[t]] = template_spread(B, T->payload[t]);
        for (int m = 0; m < NAPA_SOFT_NONTAGS; m++) B->nontag_sigma[m] = template_spread(B, T->nontag[m]);
        if (!soft_prepare_fit(B)) { free(T); return 0; }
    }
    det->soft = T;
    return 1;
}

static void soft_sample(const image_view *view, const vec2 q[4], float S[NAPA_SOFT_AREA]) {
    double H[9];
    square_to_quad(q, H);
    for (int r = 0; r < NAPA_SOFT_SAMPLES; r++) {
        for (int c = 0; c < NAPA_SOFT_SAMPLES; c++) {
            const vec2 p = project(H, soft_position(c) / 8, soft_position(r) / 8);
            S[r * NAPA_SOFT_SAMPLES + c] = (float)sample_bilinear(view, p.x, p.y);
        }
    }
    standardize(S, NAPA_SOFT_AREA);
}

static void soft_project(const soft_bin *B, const float S[NAPA_SOFT_AREA], soft_projection *P) {
    float by_row[NAPA_SOFT_SAMPLES][6];
    P->sum_base = 0;
    for (int i = 0; i < NAPA_SOFT_AREA; i++) P->sum_base += S[i] * B->base[i];
    for (int r = 0; r < NAPA_SOFT_SAMPLES; r++) {
        for (int j = 0; j < 6; j++) {
            float x = 0;
            for (int k = 0; k < NAPA_SOFT_SAMPLES; k++) x += S[r * NAPA_SOFT_SAMPLES + k] * B->blur[k][j];
            by_row[r][j] = x;
        }
    }
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float x = 0;
            for (int r = 0; r < NAPA_SOFT_SAMPLES; r++) x += by_row[r][j] * B->blur[r][i];
            P->projection[i * 6 + j] = x;
        }
    }
}

static inline float soft_score(const soft_projection *P, const float p[36], float sigma) {
    float d = P->sum_base;
    for (int i = 0; i < 36; i++) d += p[i] * P->projection[i];
    return d / (NAPA_SOFT_AREA * sigma);
}

/* The best code over all rotations, and the best code of any other ID. */
static void soft_rank(const family_layout *L, const soft_tables *T, const soft_bin *B, const soft_projection *P, int *best_code,
                      float *best, float *rival) {
    float score[NAPA_SOFT_CODES];
    int top = 0;
    for (int t = 0; t < NAPA_SOFT_CODES; t++) {
        score[t] = soft_score(P, T->payload[t], B->sigma[L->ids[t]]);
        if (score[t] > score[top]) top = t;
    }
    float second = -9;
    for (int t = 0; t < NAPA_SOFT_CODES; t++)
        if (L->ids[t] != L->ids[top] && score[t] > second) second = score[t];
    *best_code = top, *best = score[top], *rival = second;
}

/* Twice the signed area when the quad is convex and wound positively; otherwise 0. */
static double convex_area(const vec2 q[4]) {
    double area = 0;
    for (int k = 0; k < 4; k++) {
        const vec2 a = q[k], b = q[(k + 1) & 3], c = q[(k + 2) & 3];
        if ((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 0) return 0;
        area += a.x * b.y - b.x * a.y;
    }
    return area;
}

/* Intersection over union of two convex quads, each wound positively. */
static double quad_overlap(const vec2 a[4], const vec2 b[4]) {
    vec2 poly[12], next[12];
    int n = 4;
    memcpy(poly, a, 4 * sizeof *poly);
    for (int k = 0; k < 4 && n > 0; k++) {
        const vec2 e0 = b[k], e1 = b[(k + 1) & 3];
        int m = 0;
        for (int i = 0; i < n; i++) {
            const vec2 p = poly[i], s = poly[(i + 1) % n];
            const double dp = (e1.x - e0.x) * (p.y - e0.y) - (e1.y - e0.y) * (p.x - e0.x);
            const double ds = (e1.x - e0.x) * (s.y - e0.y) - (e1.y - e0.y) * (s.x - e0.x);
            if (dp >= 0) next[m++] = p;
            if ((dp >= 0) != (ds >= 0)) {
                const double f = dp / (dp - ds);
                next[m].x = p.x + f * (s.x - p.x), next[m].y = p.y + f * (s.y - p.y), m++;
            }
        }
        n = m;
        memcpy(poly, next, (size_t)n * sizeof *poly);
    }
    if (n < 3) return 0;
    double overlap = 0;
    for (int i = 0; i < n; i++) overlap += poly[i].x * poly[(i + 1) % n].y - poly[(i + 1) % n].x * poly[i].y;
    overlap = fabs(overlap);
    const double total = convex_area(a) + convex_area(b) - overlap;
    return total > 0 ? overlap / total : 0;
}

/* A detection's corners, wound positively, or 0 if they aren't a convex quad. */
static int detection_quad(const napa_detection *d, vec2 q[4]) {
    for (int k = 0; k < 4; k++) q[k].x = d->corners[2 * k], q[k].y = d->corners[2 * k + 1];
    if (signed_area(q) < 0) { vec2 t = q[1]; q[1] = q[3]; q[3] = t; }
    return convex_area(q) > 0;
}

static int overlaps_found(const napa_detection *out, int n, const vec2 q[4]) {
    for (int i = 0; i < n; i++) {
        vec2 other[4];
        if (detection_quad(&out[i], other) && quad_overlap(q, other) >= 0.5) return 1;
    }
    return 0;
}

static int solve_8x8(double a[8][9], double x[8]) {
    for (int p = 0; p < 8; p++) {
        int pivot = p;
        for (int r = p + 1; r < 8; r++) if (fabs(a[r][p]) > fabs(a[pivot][p])) pivot = r;
        if (fabs(a[pivot][p]) < 1e-12) return 0;
        if (pivot != p) for (int q = 0; q < 9; q++) { double w = a[p][q]; a[p][q] = a[pivot][q]; a[pivot][q] = w; }
        for (int r = 0; r < 8; r++) {
            if (r == p) continue;
            const double f = a[r][p] / a[p][p];
            for (int q = p; q < 9; q++) a[r][q] -= f * a[p][q];
        }
    }
    for (int r = 0; r < 8; r++) x[r] = a[r][8] / a[r][r];
    return 1;
}

static float soft_correlation(const float *a, const float *b) {
    float d = 0;
    for (int i = 0; i < NAPA_SOFT_AREA; i++) d += a[i] * b[i];
    return d / NAPA_SOFT_AREA;
}

/* Gauss-Newton on the eight corner coordinates, pulling the sampled tag onto the winning code's picture. */
static void soft_align(const image_view *view, const float t[NAPA_SOFT_AREA], const vec2 start[4], double limit, vec2 out[4]) {
    double x[8], origin[8], kept[8];
    float now[NAPA_SOFT_AREA], moved[NAPA_SOFT_AREA], jacobian[8][NAPA_SOFT_AREA];
    for (int k = 0; k < 4; k++) x[2 * k] = start[k].x, x[2 * k + 1] = start[k].y;
    memcpy(origin, x, sizeof origin);
    memcpy(kept, x, sizeof kept);
    memcpy(out, start, 4 * sizeof(vec2));
    soft_sample(view, out, now);
    float best = soft_correlation(now, t);
    for (int iter = 0; iter < 4; iter++) {
        vec2 q[4];
        for (int k = 0; k < 4; k++) q[k].x = x[2 * k], q[k].y = x[2 * k + 1];
        soft_sample(view, q, now);
        for (int m = 0; m < 8; m++) {
            double *coordinate = m & 1 ? &q[m / 2].y : &q[m / 2].x;
            const double was = *coordinate;
            *coordinate = was + 0.25;
            soft_sample(view, q, moved);
            *coordinate = was;
            for (int i = 0; i < NAPA_SOFT_AREA; i++) jacobian[m][i] = (moved[i] - now[i]) / 0.25f;
        }
        double a[8][9], trace = 0, step[8];
        for (int r = 0; r < 8; r++) {
            for (int c = r; c < 8; c++) {
                double v = 0;
                for (int i = 0; i < NAPA_SOFT_AREA; i++) v += (double)jacobian[r][i] * jacobian[c][i];
                a[r][c] = a[c][r] = v;
            }
            double v = 0;
            for (int i = 0; i < NAPA_SOFT_AREA; i++) v += (double)jacobian[r][i] * (t[i] - now[i]);
            a[r][8] = v;
            trace += a[r][r];
        }
        for (int r = 0; r < 8; r++) a[r][r] += 1e-2 * trace / 8;
        if (!solve_8x8(a, step)) break;
        for (int m = 0; m < 8; m++) x[m] += step[m];
        for (int k = 0; k < 4; k++) {
            const double dx = x[2 * k] - origin[2 * k], dy = x[2 * k + 1] - origin[2 * k + 1], len = hypot(dx, dy);
            if (len > limit) x[2 * k] = origin[2 * k] + dx * limit / len, x[2 * k + 1] = origin[2 * k + 1] + dy * limit / len;
        }
        for (int k = 0; k < 4; k++) q[k].x = x[2 * k], q[k].y = x[2 * k + 1];
        soft_sample(view, q, now);
        const float value = soft_correlation(now, t);
        if (value > best) best = value, memcpy(kept, x, sizeof kept);
    }
    for (int k = 0; k < 4; k++) out[k].x = kept[2 * k], out[k].y = kept[2 * k + 1];
}

typedef struct {
    vec2 q[4];
    int code, bin;
    float lead, ncc;
    double side, black, white;
} soft_accepted;

static int compare_soft_rounded(const void *a, const void *b) {
    const soft_candidate *x = (const soft_candidate *)a, *y = (const soft_candidate *)b;
    for (int k = 0; k < 8; k++) {
        const long l = lrintf(x->quad[k]), m = lrintf(y->quad[k]);
        if (l != m) return l < m ? -1 : 1;
    }
    return (x->order > y->order) - (x->order < y->order);
}

static int compare_soft_lead(const void *a, const void *b) {
    const soft_accepted *x = (const soft_accepted *)a, *y = (const soft_accepted *)b;
    return (x->lead < y->lead) - (x->lead > y->lead);
}

static int soft_pass(napa_detector *det, const image_view *view, napa_detection *out, int n, int max_out) {
    if (det->soft_count == 0 || n >= max_out) return n;
    if (!soft_prepare(det)) return -2;
    const family_layout *L = &det->fam;
    const soft_tables *T = det->soft;
    qsort(det->soft_candidates, det->soft_count, sizeof *det->soft_candidates, compare_soft_rounded);
    soft_accepted *accepted = (soft_accepted *)malloc(det->soft_count * sizeof *accepted);
    if (accepted == NULL) return -2;
    size_t count = 0;
    for (size_t index = 0; index < det->soft_count; index++) {
        const soft_candidate *e = &det->soft_candidates[index];
        if (index > 0) {
            const soft_candidate *before = &det->soft_candidates[index - 1];
            int same = 1;
            for (int k = 0; k < 8 && same; k++) same = lrintf(e->quad[k]) == lrintf(before->quad[k]);
            if (same) continue;
        }
        vec2 q[4];
        double side = 0;
        for (int k = 0; k < 4; k++) q[k].x = e->quad[2 * k], q[k].y = e->quad[2 * k + 1];
        for (int k = 0; k < 4; k++) side += 0.25 * hypot(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
        if (convex_area(q) <= 0 || overlaps_found(out, n, q)) continue;
        /* Many attempts land on one tag: one good reading is enough. */
        int known = 0;
        for (size_t j = 0; j < count && !known; j++) known = quad_overlap(q, accepted[j].q) >= 0.5;
        if (known) continue;
        int b = (int)floor(0.7 * 8 / side / 0.02 + 0.5) - NAPA_SOFT_FIRST_BIN;
        b = b < 0 ? 0 : b >= NAPA_SOFT_BINS ? NAPA_SOFT_BINS - 1 : b;
        const soft_bin *B = &T->bins[b];
        float S[NAPA_SOFT_AREA], best, rival;
        soft_projection P;
        int code;
        soft_sample(view, q, S);
        soft_project(B, S, &P);
        soft_rank(L, T, B, &P, &code, &best, &rival);
        if (!(best > NAPA_SOFT_MIN_NCC)) continue;
        const float gap = 1 - best > 1e-3f ? 1 - best : 1e-3f, lead = (best - rival) / gap;
        if (!(lead > NAPA_SOFT_MIN_LEAD)) continue;
        const float mid = 0.5f * (e->black + e->white), half = 0.5f * (e->white - e->black);
        float white_sum = 0, black_sum = 0;
        int white_cells = 0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                float z = (e->grid[(i + 1) * 8 + j + 1] - mid) / half;
                z = z < -1.5f ? -1.5f : z > 1.5f ? 1.5f : z;
                if (T->payload[code][i * 6 + j] > 0) white_sum += z, white_cells++;
                else black_sum += z;
            }
        }
        if (white_cells == 0 || white_cells == 36) continue;
        if (!(white_sum / (float)white_cells - black_sum / (float)(36 - white_cells) >= NAPA_SOFT_MIN_SEPARATION)) continue;
        float nontag = -9;
        for (int m = 0; m < NAPA_SOFT_NONTAGS; m++) {
            const float v = soft_score(&P, T->nontag[m], B->nontag_sigma[m]);
            if (v > nontag) nontag = v;
        }
        if (!(best - nontag >= NAPA_SOFT_MIN_NONTAG_LEAD)) continue;
        soft_accepted *a = &accepted[count++];
        memcpy(a->q, q, sizeof q);
        a->code = code, a->bin = b, a->lead = lead, a->ncc = best, a->side = side, a->black = e->black, a->white = e->white;
    }
    qsort(accepted, count, sizeof *accepted, compare_soft_lead);
    for (size_t index = 0; index < count && n < max_out; index++) {
        const soft_accepted *a = &accepted[index];
        const soft_bin *B = &T->bins[a->bin];
        int clash = 0;
        for (size_t j = 0; j < index && !clash; j++) clash = quad_overlap(a->q, accepted[j].q) >= 0.5;
        if (clash) continue;
        /* Line the quad up with the winning code, then score it again from scratch: it must be the same code in the same rotation. */
        float t[NAPA_SOFT_AREA], S[NAPA_SOFT_AREA], best, rival;
        soft_projection P;
        vec2 q[4];
        int code;
        soft_template(B, T->payload[a->code], t);
        standardize(t, NAPA_SOFT_AREA);
        soft_align(view, t, a->q, a->side / 8 > 2 ? a->side / 8 : 2, q);
        if (convex_area(q) <= 0 || overlaps_found(out, n, q)) continue;
        soft_sample(view, q, S);
        soft_project(B, S, &P);
        soft_rank(L, T, B, &P, &code, &best, &rival);
        if (code != a->code || !(best > NAPA_SOFT_MIN_NCC)) continue;
        if (!(soft_disagreement(B, &P, T->payload[code]) <= NAPA_SOFT_MAX_DISAGREEMENT)) continue;
        double H[9], white_sum = 0, black_sum = 0;
        int white_cells = 0;
        const double threshold = 0.5 * (a->black + a->white);
        square_to_quad(q, H);
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                const vec2 p = project(H, (j + 1.5) / 8, (i + 1.5) / 8);
                const double v = sample_bilinear(view, p.x, p.y);
                if (T->payload[code][i * 6 + j] > 0) white_sum += v - threshold, white_cells++;
                else black_sum += threshold - v;
            }
        }
        const double white_margin = white_sum / white_cells, black_margin = black_sum / (36 - white_cells);
        const int rot = L->rotations[code];
        napa_detection *d = &out[n++];
        for (int k = 0; k < 4; k++) {
            const vec2 c = q[(k - rot + 4) & 3];
            d->corners[2 * k] = c.x, d->corners[2 * k + 1] = c.y;
        }
        d->id = L->ids[code];
        d->hamming = NAPA_SOFT_DECODED;
        d->decision_margin = (float)(white_margin < black_margin ? white_margin : black_margin);
    }
    free(accepted);
    return n;
}

static int blocked_prepare(napa_detector *det) {
    if (det->blocked_codes != NULL) return 1;
    uint64_t *boards = (uint64_t *)malloc(NAPA_SOFT_CODES * sizeof *boards);
    if (boards == NULL) return 0;
    const family_layout *L = &det->fam;
    for (int t = 0; t < NAPA_SOFT_CODES; t++) {
        uint64_t board = 0;
        for (int k = 0; k < 36; k++)
            if (L->codes[t] >> (35 - k) & 1) board |= (uint64_t)1 << ((L->cell[k] / L->grid - 1) * 6 + L->cell[k] % L->grid - 1);
        boards[t] = board;
    }
    det->blocked_codes = boards;
    return 1;
}

/* The one code whose wrong cells a flat rectangle explains, or -1 if none does or more than one tag could. */
static int blocked_code(const napa_detector *det, const soft_candidate *e, int *x0_out, int *x1_out, int *y0_out, int *y1_out) {
    const family_layout *L = &det->fam;
    const float mid = 0.5f * (e->black + e->white), allowed = (float)(NAPA_BLOCKED_MAX_SPAN * (e->white - e->black > 1e-3f ? e->white - e->black : 1e-3f));
    uint64_t read = 0;
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++)
            if (e->grid[(r + 1) * 8 + c + 1] > mid) read |= (uint64_t)1 << (r * 6 + c);
    int found = -1;
    for (int t = 0; t < NAPA_SOFT_CODES; t++) {
        const uint64_t wrong = read ^ det->blocked_codes[t];
        const int count = napa_popcount64(wrong);
        if (count < NAPA_BLOCKED_MIN_WRONG || count > NAPA_BLOCKED_MAX_WRONG) continue;
        unsigned columns = 0;
        int y0 = -1, y1 = -1;
        for (int r = 0; r < 6; r++) {
            const unsigned row = (unsigned)(wrong >> (r * 6)) & 63u;
            if (row) { columns |= row; if (y0 < 0) y0 = r; y1 = r; }
        }
        const int x0 = __builtin_ctz(columns), x1 = 31 - __builtin_clz(columns);
        if ((x1 - x0 + 1) * (y1 - y0 + 1) > NAPA_BLOCKED_MAX_AREA) continue;
        float lo = HUGE_VALF, hi = -HUGE_VALF;
        for (int r = y0; r <= y1; r++)
            for (int c = x0; c <= x1; c++) {
                const float v = e->grid[(r + 1) * 8 + c + 1];
                lo = v < lo ? v : lo, hi = v > hi ? v : hi;
            }
        if (hi - lo > allowed) continue;
        if (found >= 0 && (L->ids[found] != L->ids[t] || found != t)) return -1;
        found = t, *x0_out = x0, *x1_out = x1, *y0_out = y0, *y1_out = y1;
    }
    return found;
}

/* Tags with part of the payload hidden: run after every other pass. The winner gets the ordinary side fit, then must
 * explain its samples the same way from the fitted corners. */
static int blocked_pass(napa_detector *det, const image_view *view, const napa_params *params, napa_detection *out, int n, int max_out) {
    if (det->blocked_count == 0 || n >= max_out) return n;
    if (!blocked_prepare(det)) return -2;
    const family_layout *L = &det->fam;
    qsort(det->blocked_candidates, det->blocked_count, sizeof *det->blocked_candidates, compare_soft_rounded);
    for (size_t index = 0; index < det->blocked_count && n < max_out; index++) {
        const soft_candidate *e = &det->blocked_candidates[index];
        if (index > 0) {
            const soft_candidate *before = &det->blocked_candidates[index - 1];
            int same = 1;
            for (int k = 0; k < 8 && same; k++) same = lrintf(e->quad[k]) == lrintf(before->quad[k]);
            if (same) continue;
        }
        vec2 q[4], fine[4], refined[4];
        double side = 0, edge = 0;
        for (int k = 0; k < 4; k++) q[k].x = e->quad[2 * k], q[k].y = e->quad[2 * k + 1];
        for (int k = 0; k < 4; k++) side += 0.25 * hypot(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
        if (convex_area(q) <= 0 || overlaps_found(out, n, q)) continue;
        int x0, x1, y0, y1;
        const int code = blocked_code(det, e, &x0, &x1, &y0, &y1);
        if (code < 0) continue;
        const double cell = side / 8, reach = 0.45 * cell >= 1.5 ? 0.45 * cell : 1.5;
        memcpy(fine, q, sizeof fine);
        if (!fit_sides(view, fine, refined, reach, reach, params->min_contrast, 0, 8, &edge, NULL, 8)) continue;
        memcpy(fine, refined, sizeof fine);
        if (!fit_sides(view, fine, refined, reach, reach, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge, NULL, 8)) continue;
        if (signed_area(refined) < 0) { vec2 t = refined[1]; refined[1] = refined[3]; refined[3] = t; }
        if (convex_area(refined) <= 0) continue;
        /* Read the fitted quad the way the decoder would, and ask the same question of what it sees. */
        soft_candidate probe;
        napa_detection unused;
        memset(&probe, 0, sizeof probe);
        det->grid_probe = &probe;
        decode_quad_impl(det, view, refined, params, 0, &unused, NULL, NULL, 0);
        det->grid_probe = NULL;
        if (!probe.order) continue;
        for (int k = 0; k < 4; k++) refined[k].x = probe.quad[2 * k], refined[k].y = probe.quad[2 * k + 1];
        if (blocked_code(det, &probe, &x0, &x1, &y0, &y1) != code || convex_area(refined) <= 0 || overlaps_found(out, n, refined)) continue;
        const float mid = 0.5f * (probe.black + probe.white);
        double white_sum = 0, black_sum = 0;
        int white_cells = 0, black_cells = 0;
        for (int r = 0; r < 6; r++) {
            for (int c = 0; c < 6; c++) {
                if (r >= y0 && r <= y1 && c >= x0 && c <= x1) continue;
                const float v = probe.grid[(r + 1) * 8 + c + 1];
                if (det->blocked_codes[code] >> (r * 6 + c) & 1) white_sum += v - mid, white_cells++;
                else black_sum += mid - v, black_cells++;
            }
        }
        const double white_margin = white_cells ? white_sum / white_cells : 0, black_margin = black_cells ? black_sum / black_cells : 0;
        const int rot = L->rotations[code];
        napa_detection *d = &out[n++];
        for (int k = 0; k < 4; k++) {
            const vec2 c = refined[(k - rot + 4) & 3];
            d->corners[2 * k] = c.x, d->corners[2 * k + 1] = c.y;
        }
        d->id = L->ids[code];
        d->hamming = NAPA_SOFT_DECODED;
        d->decision_margin = (float)(white_margin < black_margin ? white_margin : black_margin);
    }
    return n;
}

int napa_set_tag_shape(napa_detector *det, int shape) {
    if (det == NULL || (shape != 0 && shape != NAPA_TAG_SHAPE_TALL && shape != NAPA_TAG_SHAPE_WIDE)) return -1;
    det->tag_shape = shape;
    return 0;
}

int napa_set_camera_gamma(napa_detector *det, double gamma) {
    if (det == NULL || !isfinite(gamma) || gamma < 1 || gamma > 4 || strcmp(det->fam.f->name, "tag36h11") != 0) return -1;
    det->has_camera_gamma = gamma > 1;
    for (int v = 0; v < 256; v++) det->linearize[v] = (uint8_t)(255.0 * pow(v / 255.0, gamma) + 0.5);
    return 0;
}

int napa_border_bias_cells(napa_detector *det, const uint8_t *img, int width, int height, int stride, const napa_params *params,
                           const double corners[8], double *bias_cells, double *edge_cells) {
    if (det == NULL || img == NULL || params == NULL || corners == NULL || bias_cells == NULL) return -1;
    const image_view view = {img, width, height, stride, det->fam.f->reversed};
    vec2 q[4];
    double side = 0, delta = NAN, disagreement = 0;
    for (int k = 0; k < 4; k++) q[k].x = corners[2 * k], q[k].y = corners[2 * k + 1];
    if (signed_area(q) < 0) { vec2 t = q[1]; q[1] = q[3]; q[3] = t; }
    for (int k = 0; k < 4; k++) side += 0.25 * hypot(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
    edge_levels levels;
    napa_detection unused;
    levels.black = 255, levels.white = 0;
    decode_quad_impl(det, &view, q, params, 0, &unused, NULL, &levels, 0);
    if (!(levels.black < levels.white)) return 1;
    border_bias(&det->fam, &view, q, levels.black, levels.white, &delta, &disagreement);
    if (!isfinite(delta)) return 1;
    const double cell = side / det->fam.width;
    *bias_cells = delta / cell;
    if (edge_cells != NULL) {
        /* The blur of the tag's outer edges: profiles reach most of a cell each way, so wide blur is not cut short. */
        vec2 fitted[4];
        double width = 0, reach = 0.75 * cell + 1.0;
        *edge_cells = fit_sides(&view, q, fitted, reach, reach, params->min_contrast, 0, 8, &width, NULL, det->fam.width) && width > 0
                          ? width / cell : NAN;
    }
    return 0;
}

int napa_detect_options(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                        const napa_params *params, napa_detection *out, int max_out, int recover_shadows,
                        int shadow_min_tag_px, unsigned options) {
    if (det == NULL || (recover_shadows != 0 && recover_shadows != 1) || (options & ~(unsigned)(NAPA_OPTION_SOFT_DECODE | NAPA_OPTION_ALIGN_CORNERS)) != 0 ||
        strcmp(det->fam.f->name, "tag36h11") != 0) return -1;
    const int soft = (options & NAPA_OPTION_SOFT_DECODE) != 0;
    det->align_corners = (options & NAPA_OPTION_ALIGN_CORNERS) != 0;
    det->soft_count = 0, det->blocked_count = 0, det->soft_failed = 0, det->soft_capturing = soft;
    const int n = recover_shadows ? napa_detect_shadows_configured(det, img, width, height, stride, params, out, max_out, shadow_min_tag_px)
                                  : napa_detect(det, img, width, height, stride, params, out, max_out);
    det->soft_capturing = 0, det->align_corners = 0;
    if (n < 0) return n;
    if (det->soft_failed) return -2;
    if (!soft) return n;
    const image_view view = {img, width, height, stride, 0};
    const int with_soft = soft_pass(det, &view, out, n, max_out);
    return with_soft < 0 ? with_soft : blocked_pass(det, &view, params, out, with_soft, max_out);
}

int napa_detect_soft(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                     const napa_params *params, napa_detection *out, int max_out, int recover_shadows,
                     int shadow_min_tag_px) {
    return napa_detect_options(det, img, width, height, stride, params, out, max_out, recover_shadows, shadow_min_tag_px, NAPA_OPTION_SOFT_DECODE);
}

/* setuptools links every extension with a module-init export on Windows. The library is loaded
 * with ctypes, never imported, so this placeholder only satisfies the linker. */
NAPA_API void *PyInit__napa_core(void) { return 0; }
