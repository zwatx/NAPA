/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/*
 * Napa core: a sparse-sampling ("rake") AprilTag detector in portable C99.
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
 *   5. Sample the family's cell grid through the homography and decode. Multi-index hashing
 *      narrows the rotated codewords to those sharing a block with the payload.
 *   6. Repair the cases a plain fit gets wrong: a quiet zone darker than the background, a side
 *      fitted one cell out, and edges biased by clipped (saturated) exposure.
 *
 * The library exposes a plain C ABI and uses the C math library and POSIX or Windows threads.
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
#ifdef _MSC_VER
#include <intrin.h> /* clang-cl: __cpuid (see cpu_features) */
#endif
#endif
#elif defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
#include <arm_neon.h>
#endif

#include "napa_core.h"
#include "families.h"

/* ---------------------------------------------------------------- worker threads */

/* Optional workers for one detector (napa_set_threads). A job is n independent indices; the caller takes
 * indices too and returns once every index has run. Work is split so results never depend on which thread
 * ran an index, so outputs are identical at every thread count. Each index is told which thread runs it
 * (0 for the caller, 1.. for the workers), for jobs that keep scratch per thread. */
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef near /* legacy 16-bit pointer qualifiers that windows.h still defines; the core uses these names */
#undef far
typedef HANDLE napa_thread_t;
typedef SRWLOCK napa_mutex_t;
typedef CONDITION_VARIABLE napa_cond_t;
#define napa_mutex_init(m) InitializeSRWLock(m)
#define napa_mutex_destroy(m) ((void)0)
#define napa_mutex_lock(m) AcquireSRWLockExclusive(m)
#define napa_mutex_trylock(m) (TryAcquireSRWLockExclusive(m) != 0)
#define napa_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define napa_cond_init(c) InitializeConditionVariable(c)
#define napa_cond_destroy(c) ((void)0)
#define napa_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, 0)
#define napa_cond_broadcast(c) WakeAllConditionVariable(c)
#else
#include <pthread.h>
typedef pthread_t napa_thread_t;
typedef pthread_mutex_t napa_mutex_t;
typedef pthread_cond_t napa_cond_t;
#define napa_mutex_init(m) pthread_mutex_init(m, NULL)
#define napa_mutex_destroy(m) pthread_mutex_destroy(m)
#define napa_mutex_lock(m) pthread_mutex_lock(m)
#define napa_mutex_trylock(m) (pthread_mutex_trylock(m) == 0)
#define napa_mutex_unlock(m) pthread_mutex_unlock(m)
#define napa_cond_init(c) pthread_cond_init(c, NULL)
#define napa_cond_destroy(c) pthread_cond_destroy(c)
#define napa_cond_wait(c, m) pthread_cond_wait(c, m)
#define napa_cond_broadcast(c) pthread_cond_broadcast(c)
#endif

#define NAPA_MAX_THREADS 16
#define NAPA_MAX_SIDE 30000 /* the largest image width or height accepted */

typedef struct napa_pool {
    int workers; /* threads besides the caller */
    napa_thread_t thread[NAPA_MAX_THREADS];
    napa_mutex_t lock;
    napa_mutex_t owner; /* held by the one caller using the workers */
    napa_cond_t work, done_cond, progress; /* progress: pool_mark, for jobs waiting on lower indices (pool_await) */
    void (*fn)(void *ctx, int index, int worker);
    void *ctx;
    struct napa_pool_worker { struct napa_pool *pool; int id; } arg[NAPA_MAX_THREADS];
    int jobs, next, done, stop;
    unsigned generation;
} napa_pool;

/* Run indices of the current job until none are left. Called with the lock held; returns with it held. */
static void pool_drain(napa_pool *pool, int worker) {
    while (pool->next < pool->jobs) {
        int index = pool->next++;
        void (*fn)(void *, int, int) = pool->fn;
        void *ctx = pool->ctx;
        napa_mutex_unlock(&pool->lock);
        fn(ctx, index, worker);
        napa_mutex_lock(&pool->lock);
        if (++pool->done == pool->jobs) napa_cond_broadcast(&pool->done_cond);
    }
}

#if defined(_WIN32)
static DWORD WINAPI pool_worker(LPVOID arg) {
#else
static void *pool_worker(void *arg) {
#endif
    napa_pool *pool = ((struct napa_pool_worker *)arg)->pool;
    const int id = ((struct napa_pool_worker *)arg)->id;
    unsigned seen = 0;
    napa_mutex_lock(&pool->lock);
    for (;;) {
        while (!pool->stop && pool->generation == seen) napa_cond_wait(&pool->work, &pool->lock);
        if (pool->stop) break;
        seen = pool->generation;
        pool_drain(pool, id);
    }
    napa_mutex_unlock(&pool->lock);
    return 0;
}

static void pool_destroy(napa_pool *pool) {
    if (pool == NULL) return;
    napa_mutex_lock(&pool->lock);
    pool->stop = 1;
    napa_cond_broadcast(&pool->work);
    napa_mutex_unlock(&pool->lock);
    for (int i = 0; i < pool->workers; i++) {
#if defined(_WIN32)
        WaitForSingleObject(pool->thread[i], INFINITE);
        CloseHandle(pool->thread[i]);
#else
        pthread_join(pool->thread[i], NULL);
#endif
    }
    napa_cond_destroy(&pool->work);
    napa_cond_destroy(&pool->done_cond);
    napa_cond_destroy(&pool->progress);
    napa_mutex_destroy(&pool->lock);
    napa_mutex_destroy(&pool->owner);
    free(pool);
}

/* A pool with `workers` threads besides the caller, or NULL if threads can't be started. */
static napa_pool *pool_create(int workers) {
    napa_pool *pool = (napa_pool *)calloc(1, sizeof(napa_pool));
    if (pool == NULL) return NULL;
    napa_mutex_init(&pool->lock);
    napa_mutex_init(&pool->owner);
    napa_cond_init(&pool->work);
    napa_cond_init(&pool->done_cond);
    napa_cond_init(&pool->progress);
    for (int i = 0; i < workers; i++) {
        pool->arg[i].pool = pool, pool->arg[i].id = i + 1;
#if defined(_WIN32)
        pool->thread[i] = CreateThread(NULL, 0, pool_worker, &pool->arg[i], 0, NULL);
        int failed = pool->thread[i] == NULL;
#else
        int failed = pthread_create(&pool->thread[i], NULL, pool_worker, &pool->arg[i]) != 0;
#endif
        if (failed) {
            pool_destroy(pool); /* joins the ones already started */
            return NULL;
        }
        pool->workers = i + 1;
    }
    return pool;
}

/* fn(ctx, i, worker) for every i in [0, n), across the pool and the calling thread; serial without a pool. A detector
 * must not be used from two threads at once (its scratch is shared), but if it is, the second caller runs its
 * job alone rather than taking over the workers mid-job, which could otherwise wait forever. */
static void pool_run(napa_pool *pool, void (*fn)(void *, int, int), void *ctx, int n) {
    if (pool == NULL || n <= 1 || !napa_mutex_trylock(&pool->owner)) {
        for (int i = 0; i < n; i++) fn(ctx, i, 0);
        return;
    }
    napa_mutex_lock(&pool->lock);
    pool->fn = fn, pool->ctx = ctx;
    pool->jobs = n, pool->next = 0, pool->done = 0;
    pool->generation++;
    napa_cond_broadcast(&pool->work);
    pool_drain(pool, 0);
    while (pool->done < pool->jobs) napa_cond_wait(&pool->done_cond, &pool->lock);
    napa_mutex_unlock(&pool->lock);
    napa_mutex_unlock(&pool->owner);
}

/* Set *flag from inside an index of the pool's current job (or serially, pool NULL), for indices in pool_await. */
static void pool_mark(napa_pool *pool, int *flag) {
    if (pool == NULL) {
        *flag = 1;
        return;
    }
    napa_mutex_lock(&pool->lock);
    *flag = 1;
    napa_cond_broadcast(&pool->progress);
    napa_mutex_unlock(&pool->lock);
}

/* Wait until flags[from..to) are all set by pool_mark. Only for flags that lower indices of the same job set: those
 * were handed out first, so they are running or done (serially, done), and the wait always ends. */
static void pool_await(napa_pool *pool, const int *flags, int from, int to) {
    if (pool == NULL) return;
    napa_mutex_lock(&pool->lock);
    for (int i = from; i < to; i++)
        while (!flags[i]) napa_cond_wait(&pool->progress, &pool->lock);
    napa_mutex_unlock(&pool->lock);
}

/* From inside an index of the pool's current job (ctx): add n more indices to it. Returns 0, adding none, when the
 * job is not running on the pool (pool_run ran it serially, for a fixed count). */
static int pool_extend(napa_pool *pool, const void *ctx, int n) {
    if (pool == NULL) return 0;
    napa_mutex_lock(&pool->lock);
    const int on_pool = pool->ctx == ctx && pool->done < pool->jobs;
    if (on_pool) pool->jobs += n;
    napa_mutex_unlock(&pool->lock);
    return on_pool;
}

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

/* A horizontal run of dark samples in one level row, and a blob's sample count and bounding box.
 * Ordinary labelling (no split, no shadow) stores row-pair runs instead (see label_row_pair): samples
 * x0..x1 where row y or row y + 1 is dark, and rows says which: bit 0 row y, bit 1 row y + 1. Split and
 * shadow labelling leave rows unset. */
typedef struct {
    uint16_t x0, x1, y, rows;
} napa_run;

typedef struct {
    int32_t count;
    uint16_t minx, maxx, miny, maxy;
} blob_box;

/* Candidate stages retain the trace's numeric values; waiting for neck splitting has no trace event. */
enum {
    CANDIDATE_UNTRIED = 0,
    CANDIDATE_WAITING_FOR_NECKS = 1,
    CANDIDATE_SKIPPED = 5,
    CANDIDATE_ROUGH_REJECTED = 6,
    CANDIDATE_FIT_FAILED = 7,
    CANDIDATE_DECODE_FAILED = 8,
    CANDIDATE_REFINE_FIT_FAILED = 9,
    CANDIDATE_REFINE_DECODE_FAILED = 10,
    CANDIDATE_DUPLICATE = 11,
    CANDIDATE_DECODED = 12
};

static int suppresses_pieces(int fate) {
    return fate == CANDIDATE_ROUGH_REJECTED || fate == CANDIDATE_REFINE_FIT_FAILED ||
           fate == CANDIDATE_REFINE_DECODE_FAILED || fate == CANDIDATE_DUPLICATE || fate == CANDIDATE_DECODED;
}

typedef struct {
    int count;
    int minx, maxx, miny, maxy;
    int ext[16][2]; /* extreme points along the 16 directions of comp_dirs, in angular order */
    int extv[16];
    /* Row accumulation ends before candidate processing; the two phases share storage. */
    union {
        struct {
            int y, x0, x1;
            /* Row pairs: x0 = -1 means empty; a negative x1 encodes the union run's end
             * as -x1 - 1 until the last dark sample in that row is resolved. */
            int pair_y, top_x0, top_x1, bot_x0, bot_x1;
        } rows;
        struct {
            int fate; /* candidate stage, retained when the level is redone with neck splitting */
            int piece_count;
        } candidate;
    } work;
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

/* A vector's length. Not libm's hypot: its last bit differs between C libraries (glibc on x86-64 and on ARM disagree),
 * which made detections differ between machines. A correctly rounded square root gives the same bits everywhere, and
 * every length here is far from overflow. */
static inline double vec_length(double x, double y) { return sqrt(x * x + y * y); }

/* A calibrated camera's lens (napa_set_lens), OpenCV's conventions: pixel centres at integers, normalized coordinates
 * x = X/Z, y = Y/Z. Brown-Conrady k: k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4; fisheye (Kannala-Brandt) k: k1 k2 k3 k4. */
typedef struct lens_memo lens_memo;
typedef struct {
    int model; /* NAPA_LENS_* */
    double fx, fy, cx, cy, skew;
    double k[12];
    int rational, prism; /* Brown-Conrady: whether k4-k6 or s1-s4 are in use */
    lens_memo *memo;     /* the detector's cache of the view it last worked out for a tag, updated even through a
                          * const napa_lens */
} napa_lens;

/* One tag's view through the lens: the plane square to the ray through the tag's middle, where every straight edge of the
 * tag is a straight line, scaled to about one unit per pixel there. Plane x follows the image's x. */
typedef struct {
    const napa_lens *L;
    double e[3][3]; /* plane x, plane y, view axis */
    double scale, inv_scale;
} lens_frame;

/* A candidate's fits and bias readings each start from the corners the step before returned, so the detector keeps the
 * last tag view it worked out: the corners it was asked for (key), where they lie in the plane, and how far the lens bows
 * the sides between them. A lens fit leaves its own result as `next`, already in the same plane. */
struct lens_memo {
    int valid, chained;
    vec2 key[4], plane[4], next[4], next_plane[4];
    double bow, departure; /* departure: quad_map_init's, -1 until asked */
    lens_frame f;
};

/* What labelling a level (threshold_rows) writes besides its runs and the level's dark bits: row and tile
 * scratch. The detector has one, and one per pool thread for banded labelling (band_arena). */
typedef struct {
    uint8_t *row_min, *row_max, *row_thr; /* one level row's tile extrema and thresholds, expanded to samples */
    uint8_t *nb_lo, *nb_hi, *nb_t, *nb_f; /* one tile row's neighbourhood min, max, threshold and floor */
    uint64_t *row_bits; /* one row's dark bits, or row-pair labelling's boundary windows */
    uint16_t *row_edges; /* row-pair labelling: a pair's run edges, in order */
    int32_t *pair_rank;  /* row-pair labelling: two pairs' touchable runs in left-to-right order, then a scratch list */
    size_t row_capacity;
    uint8_t *tile_min, *tile_max; /* tile extrema: the level's, or a band's tile rows and one on each side */
    size_t tile_capacity;
    uint8_t *half; /* a level of half tiles (level_half): HALF_ROWS rows of row / 2 + 67 bytes, see threshold_index_impl */
} label_scratch;

/* Bounded access for thresholding and run labelling; no detector, pool or candidate state. */
typedef struct {
    const uint8_t *level, *floors, *level_extrema_in;
    int borrowed, floor_cols, floor_rows, floor_uniform;
    uint8_t level_contrast_min;
    uint8_t level_half; /* the detector's level_half */
    uint8_t *level_extrema_out;
    napa_run *runs;
    int32_t *parent;
    const label_scratch *scratch;
    uint64_t *level_bits;
    int popcnt; /* the CPU has POPCNT: label with threshold_index_popcnt */
} label_context;

/* Only serial split labelling can grow and append these deferred contacts. */
typedef struct {
    int32_t **diag, **neck, **neck_root;
    size_t *diag_n, *neck_n, *diag_capacity, *neck_capacity;
} label_contacts;

/* A decode attempt worth a second look by the soft decoder. `order` keeps the first of exact duplicates. */
typedef struct {
    float quad[8], black, white, grid[64];
    int order;
} soft_candidate;

typedef struct soft_tables soft_tables;

/* Banded labelling's buffers for one pool thread (worker index): the runs and parents of the bands it labels, one band
 * after another from the start of each level, and its labelling scratch. A thread's buffers stay warm across its bands
 * and frames; buffers of a band's own would come back from memory every frame. */
typedef struct {
    napa_run *runs;
    int32_t *parent;
    size_t run_capacity, used; /* used: runs of this level's bands so far */
    label_scratch scratch;
} band_arena;

/* One sampling step's tile extrema from the ordinary pass, and what its samples were made from. */
typedef struct {
    uint8_t *tiles; /* the tile minima, then the maxima */
    size_t capacity;
    int base;  /* the pass's first sampling step */
    int boxed; /* this level, or a finer one it was sampled from, was replaced by box averages */
    int valid; /* written during this frame, not merely allocated */
} saved_extrema;

struct napa_detector {
    family_layout fam;
    int threads;      /* napa_set_threads: 1 (the default) runs on the calling thread only */
    napa_pool *pool;  /* threads - 1 workers, started when threads > 1 */
    struct napa_band *bands; /* banded labelling buffers, one set per band */
    uint32_t (*floor_hist)[64]; /* contrast_floors: each region row's histogram */
    size_t floor_hist_rows;
    int band_count;
    band_arena band_arenas[NAPA_MAX_THREADS]; /* banded labelling's buffers, one per pool thread (worker index) */
    int tag_shape;        /* NAPA_TAG_SHAPE_*: the ordinary pass samples twice as far apart down (tall) or across (wide) */
    int32_t *regrow_stack; /* regrow_small's fill stack and marks, NAPA_REGROW_WINDOW squared each, made on first use */
    uint16_t *regrow_mark; /* a window pixel is marked when it holds regrow_stamp */
    uint16_t regrow_stamp;
    int regrow_left;       /* regrowths this pass may still make (NAPA_REGROW_BUDGET) */
    int regrow_small_tags; /* napa_set_regrow_small_tags: 0 (the default) never regrows */
    uint8_t *shrunk;      /* the frame halved that way, area-averaged */
    size_t shrunk_capacity;
    int has_camera_gamma;
    napa_lens lens; /* model NAPA_LENS_NONE unless napa_set_lens gave one */
    lens_memo lens_cache; /* lens.memo points here */
    uint8_t linearize[256]; /* undoes the camera's tone curve; used only while refining a tag */
    uint8_t *linear_frame;  /* frame-sized so a tag keeps its coordinates; only each tag's own rectangle is ever filled */
    size_t linear_capacity;
    soft_tables *soft;
    int align_corners; /* during napa_detect_options: pull corners onto each blurred tag's own picture */
    int touching_seen; /* split_touching 2: the last ordinary pass decoded a tag touching dark shapes at its corners */
    int split_from_necks; /* split_touching 2's second pass: tags are known to touch, so split necks from the start */
    soft_candidate *blocked_candidates, *grid_probe; /* probe: where a decode attempt copies its samples */
    size_t blocked_count, blocked_capacity;
    uint64_t *blocked_codes; /* every rotated code as a 6x6 bitboard, row by row */
    float (*tone_blur)[30][10]; /* blur tables for measuring blur and tone curve on a decoded tag; built on first use */
    soft_candidate *soft_candidates;
    size_t soft_count, soft_capacity;
    int soft_capturing, soft_failed;
    int ssse3, popcnt; /* the CPU supports SSSE3, POPCNT (x86 only; read once at creation) */

    uint8_t *level; /* owned scratch, even when the finest gray pixels are borrowed */
    const uint8_t *level_view; /* non-owning, only during threshold_level; NULL uses owned scratch */
    saved_extrema saved[32]; /* by sampling step, for recovery to reuse where its samples are the same */
    uint8_t *level_extrema_out; /* where this level's extrema go, or NULL */
    const uint8_t *level_extrema_in; /* the saved extrema to reuse, or NULL */
    int extrema_wanted, extrema_ready;
    uint8_t level_contrast_min;
    uint8_t level_half; /* this level's thresholds are also capped by half-tile neighbourhoods (threshold_index_impl) */
    label_scratch scratch;
    napa_run *runs;
    int32_t *parent; /* backward union links, then -dense component index - 1 after aggregation */
    blob_box *boxes; /* packed components, in first-run raster order */
    int32_t *slot;   /* dense component -> candidate index, or -1 */
    component *cands;
    size_t cand_capacity;
    uint64_t *level_bits; /* row-pair labelling: every labelled row's dark bits, (lw + 63) / 64 + 1 words a row */
    size_t level_bits_capacity;
    size_t run_capacity;
    size_t level_capacity;
    uint8_t *floors; /* tile contrast floor of each noise region, row-major */
    size_t floor_alloc;
    int floor_cols, floor_rows, floor_uniform; /* uniform: every region has the same floor */
    uint16_t *box_rows; /* the averaging rake's two scratch rows */
    size_t box_capacity;
    failed_quad *failed; /* this frame's failed candidates, for skipping the same blob at a coarser level */
    struct cand_memo *memo; /* threads: each candidate's rough quad, screen and first fit, worked out in parallel per level */
    int32_t *memo_order;    /* memo_capacity entries: the order the pre-pass takes candidates in when decoding ahead */
    size_t memo_capacity;
    int32_t *cand_runs, *cand_run_end; /* deferred extremes: the candidates' runs grouped by candidate, and each group's end */
    size_t cand_runs_capacity, cand_run_end_capacity;
    /* threads: a second set of level buffers, so the next level can be labelled while this one's pre-pass reads its own
     * (run_set_swap). runs_swapped: the sets are exchanged; each pass starts by swapping them back. */
    struct {
        napa_run *runs;
        int32_t *parent, *slot;
        blob_box *boxes;
        size_t run_capacity, level_bits_capacity;
        uint64_t *level_bits;
        label_scratch scratch;
        component *cands;
        size_t cand_capacity;
        struct cand_memo *memo;
        int32_t *memo_order;
        size_t memo_capacity;
        int32_t *cand_runs, *cand_run_end;
        size_t cand_runs_capacity, cand_run_end_capacity;
    } spare;
    int runs_swapped;
    /* contrast_floors_raked left the finest level's other rows to rake (rake_rows), alongside the threshold bands */
    const uint8_t *rake_img;
    int rake_stride, rake_s, rake_invert, rake_deferred;
    int *rake_done; /* per chunk, set once raked */
    int rake_next, rake_floors_left; /* the next chunk to rake; floor jobs not yet done */
    size_t rake_done_capacity;
    int cands_swapped; /* threads: the candidate sets are exchanged (cand_set_swap); each pass starts by swapping them back */
    struct delayed_memo *dmemo; /* threads, terminal level: the first dmemo_n deferred fine wholes, worked out ahead */
    size_t dmemo_capacity;
    int dmemo_n;
    int dmemo_ready; /* threads: the first deferred fine wholes, already worked out by the levels that set them aside */
    int32_t *whole_index; /* threads: each of the level's set-aside candidates' index among the deferred wholes, else -1 */
    size_t whole_index_capacity;
    /* threads: the pool threads' own detectors for decoding and refining candidates ahead of the loop. Each holds only
     * what that reads (prepare_workers) and its own code stamps; worker_count is 0 if they couldn't be made. */
    struct napa_detector *workers;
    int worker_count;
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
    /* Extended-shape detections wait until ordinary whole detections are complete. */
    napa_detection *shape_out;
    vec2 *shape_centers;
    size_t shape_out_capacity, shape_center_capacity;
    fine_whole *fine_wholes; /* deferred whole outlines, replayed after proposals */
    size_t fine_whole_capacity;
    vec2 *out_centers, *piece_centers; /* accepted detections, in output order */
    size_t out_center_capacity, piece_center_capacity;
    uint8_t *fates; /* before necks are split: each level's candidates' and pieces' fates, for redoing it */
    size_t fate_capacity;
};

/* The fewest samples a level (or frame) must have before its labelling, noise floors, row gathering or colour
 * conversion is split across threads: below it, handing work to the threads costs more than it saves. */
#define NAPA_THREAD_MIN_SAMPLES (96 * 1024)

/* The detector's thread pool for work over `samples` samples, or NULL to run it on the calling thread. */
static napa_pool *pool_for(const napa_detector *det, size_t samples) {
    return samples >= NAPA_THREAD_MIN_SAMPLES ? det->pool : NULL;
}

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

/* Index of the rotated code nearest `code` within max_bits (at most 5), the first in table order on a tie, or -1.
 * A code within the family's max_hamming is the only one that near (codes are 2 * max_hamming + 1 apart), so the
 * search stops there; a farther match must be compared with every other code within max_bits. */
static int nearest_code(napa_detector *det, uint64_t code, int max_bits, int *distance) {
    family_layout *L = &det->fam;
    const code_index *ix = max_bits <= 3 ? &L->index3 : &L->index5;
    if (++L->stamp_now == 0) {
        memset(L->stamp, 0, sizeof(uint32_t) * (size_t)L->nrot);
        L->stamp_now = 1;
    }
    int best = -1, best_d = max_bits + 1;
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
                if (d <= L->max_hamming && d <= max_bits) {
                    *distance = d;
                    return (int)i;
                }
                if (d < best_d || (d == best_d && (int)i < best)) best = (int)i, best_d = d;
            }
        }
    }
    if (best >= 0) *distance = best_d;
    return best;
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

int napa_params_size(void) { return (int)sizeof(napa_params); }

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

#ifdef NAPA_SSSE3_DISPATCH
/* The CPU features the dispatched functions need. clang-cl defines __clang__, but __builtin_cpu_supports needs clang's
 * builtins library, which clang-cl doesn't link, so under _MSC_VER CPUID is read directly. */
static void cpu_features(int *ssse3, int *popcnt) {
#ifdef _MSC_VER
    int info[4];
    __cpuid(info, 1);
    *ssse3 = (info[2] >> 9) & 1;   /* leaf 1, ECX bit 9 */
    *popcnt = (info[2] >> 23) & 1; /* leaf 1, ECX bit 23 */
#else
    __builtin_cpu_init();
    *ssse3 = __builtin_cpu_supports("ssse3") != 0;
    *popcnt = __builtin_cpu_supports("popcnt") != 0;
#endif
}
#endif

napa_detector *napa_create_family(const char *family) {
    const napa_family *f = find_family(family);
    if (f == NULL) return NULL;
    napa_detector *det = (napa_detector *)calloc(1, sizeof(napa_detector));
    if (det != NULL) {
        if (load_family(&det->fam, f) != 0) {
            napa_destroy(det);
            return NULL;
        }
        det->threads = 1;
#ifdef NAPA_SSSE3_DISPATCH
        cpu_features(&det->ssse3, &det->popcnt);
#endif
    }
    return det;
}

static void band_free(napa_detector *det);
static void scratch_free(label_scratch *s);
static void free_workers(napa_detector *det) {
    for (int w = 0; w < det->worker_count; w++) free(det->workers[w].fam.stamp);
    free(det->workers);
    det->workers = NULL, det->worker_count = 0;
}

/* n detectors for the pool's threads (see workers). Without them the threads still run, only without decoding ahead. */
static void make_workers(napa_detector *det, int n) {
    napa_detector *w = (napa_detector *)calloc((size_t)n, sizeof(napa_detector));
    if (w == NULL) return;
    det->workers = w;
    for (int i = 0; i < n; i++) {
        w[i].fam.stamp = (uint32_t *)calloc((size_t)det->fam.nrot, sizeof(uint32_t));
        if (w[i].fam.stamp == NULL) {
            free_workers(det);
            return;
        }
        det->worker_count = i + 1;
    }
}

void napa_destroy(napa_detector *det) {
    if (det == NULL) {
        return;
    }
    free_workers(det);
    free(det->memo_order), free(det->cand_runs), free(det->cand_run_end);
    free(det->spare.runs), free(det->spare.parent), free(det->spare.slot), free(det->spare.boxes), free(det->spare.level_bits);
    scratch_free(&det->spare.scratch);
    free(det->spare.cands), free(det->spare.memo), free(det->spare.memo_order), free(det->spare.cand_runs), free(det->spare.cand_run_end);
    free(det->dmemo), free(det->whole_index);
    pool_destroy(det->pool);
    free(det->rake_done);
    band_free(det);
    free(det->floor_hist);
    for (int i = 0; i < 32; i++) free(det->saved[i].tiles);
    free(det->level);
    free(det->failed);
    free(det->memo);
    free(det->regrow_stack);
    free(det->regrow_mark);
    free(det->tone_blur);
    free(det->blocked_candidates);
    free(det->blocked_codes);
    free(det->linear_frame);
    free(det->shrunk);
    free(det->soft);
    free(det->soft_candidates);
    free_family(&det->fam);
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
    free(det->level_bits);
    scratch_free(&det->scratch);
    free(det->floors);
    free(det->box_rows);
    free(det);
}

/* Half-tile scratch rows: eight of minima and eight of maxima (a ring over half rows), two of thresholds, and one each
 * for the tile thresholds, the floors and the neighbourhood minima and maxima at half-tile spacing. */
enum { HALF_ROWS = 22 };
/* Grows s for level rows of up to `row` samples and up to `tiles` tiles. */
static int scratch_reserve(label_scratch *s, size_t row, size_t tiles) {
    if (row > s->row_capacity) {
        /* Row buffers run 128 bytes past the row: whole tiles plus whole 64-sample mask words. Tile rows:
         * row / 4 + 1 tiles plus one padding tile at each end. */
        uint8_t **bytes[3] = {&s->row_min, &s->row_max, &s->row_thr}, **tile_rows[4] = {&s->nb_lo, &s->nb_hi, &s->nb_t, &s->nb_f};
        for (int k = 0; k < 3; k++) {
            uint8_t *p = (uint8_t *)realloc(*bytes[k], row + 128);
            if (p == NULL) return -1;
            *bytes[k] = p;
        }
        for (int k = 0; k < 4; k++) {
            uint8_t *p = (uint8_t *)realloc(*tile_rows[k], row / 4 + 3);
            if (p == NULL) return -1;
            *tile_rows[k] = p;
        }
        uint64_t *bits = (uint64_t *)realloc(s->row_bits, (row / 64 + 2) * sizeof(uint64_t));
        if (bits == NULL) return -1;
        s->row_bits = bits;
        /* Every edge of a row (row + 1 at most) plus decode_edges' eight spare entries. */
        uint16_t *edges = (uint16_t *)realloc(s->row_edges, (row + 16) * sizeof(uint16_t));
        if (edges == NULL) return -1;
        s->row_edges = edges;
        int32_t *rank = (int32_t *)realloc(s->pair_rank, 3 * (row / 2 + 2) * sizeof(int32_t));
        if (rank == NULL) return -1;
        s->pair_rank = rank;
        uint8_t *half = (uint8_t *)realloc(s->half, HALF_ROWS * (row / 2 + 67));
        if (half == NULL) return -1;
        s->half = half;
        s->row_capacity = row;
    }
    if (tiles > s->tile_capacity) {
        uint8_t *p;
        if ((p = (uint8_t *)realloc(s->tile_min, tiles)) == NULL) return -1;
        s->tile_min = p;
        if ((p = (uint8_t *)realloc(s->tile_max, tiles)) == NULL) return -1;
        s->tile_max = p;
        s->tile_capacity = tiles;
    }
    return 0;
}

static void scratch_free(label_scratch *s) {
    free(s->row_min), free(s->row_max), free(s->row_thr);
    free(s->nb_lo), free(s->nb_hi), free(s->nb_t), free(s->nb_f);
    free(s->row_bits), free(s->row_edges), free(s->pair_rank);
    free(s->tile_min), free(s->tile_max);
    free(s->half);
}

static int ensure_buffers(napa_detector *det, size_t pixels, size_t tiles, size_t row, size_t max_runs) {
    /* pixels / row rows of at least (row + 63) / 64 + 1 words. */
    const size_t bit_words = (row ? pixels / row : 0) * (row / 64 + 2);
    if (bit_words > det->level_bits_capacity) {
        uint64_t *p = (uint64_t *)realloc(det->level_bits, bit_words * sizeof(uint64_t));
        if (p == NULL) return -1;
        det->level_bits = p;
        det->level_bits_capacity = bit_words;
    }
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
    if (scratch_reserve(&det->scratch, row, tiles) != 0) return -1;
    if (pixels > det->level_capacity) {
        /* 64 spare bytes: the last row's SIMD compares read up to 63 samples past its end. */
        uint8_t *level = (uint8_t *)realloc(det->level, pixels + 64);
        if (level == NULL) return -1;
        det->level = level;
        det->level_capacity = pixels;
    }
    return 0;
}

/* ---------------------------------------------------------------- sampling */

/* The frame as the detector reads it. Reversed families read it inverted, one sample at a time, so the
 * frame is never copied. */
typedef struct {
    const uint8_t *p;
    int w, h, stride, invert;
    const napa_lens *lens; /* NULL: fit straight sides in pixels */
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

#ifdef NAPA_SSE2
/* sample_bilinear at two points at once: the scalar operation order and NaN-safe clamps, so the same values. */
static inline __m128d sample_bilinear_pair(const image_view *im, __m128d x, __m128d y) {
    const uint8_t *img = im->p;
    const int stride = im->stride;
    const __m128d zero = _mm_setzero_pd(), xmax = _mm_set1_pd(im->w - 1.001), ymax = _mm_set1_pd(im->h - 1.001);
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
    return im->invert ? _mm_sub_pd(_mm_set1_pd(255.0), val) : val;
}
#elif defined(__aarch64__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
/* The same pair on AArch64, with the same order and clamps. Truncating to int64 matches (int) on clamped values. */
static inline float64x2_t sample_bilinear_pair(const image_view *im, float64x2_t x, float64x2_t y) {
    const uint8_t *img = im->p;
    const int stride = im->stride;
    const float64x2_t zero = vdupq_n_f64(0), xmax = vdupq_n_f64(im->w - 1.001), ymax = vdupq_n_f64(im->h - 1.001);
    x = vbslq_f64(vcgeq_f64(x, zero), x, zero); /* !(x >= 0): 0 */
    y = vbslq_f64(vcgeq_f64(y, zero), y, zero);
    x = vbslq_f64(vcleq_f64(x, xmax), x, xmax); /* !(x <= w - 1.001): w - 1.001 */
    y = vbslq_f64(vcleq_f64(y, ymax), y, ymax);
    const int64x2_t xi = vcvtq_s64_f64(x), yi = vcvtq_s64_f64(y);
    const uint8_t *pa = img + (size_t)vgetq_lane_s64(yi, 0) * stride + vgetq_lane_s64(xi, 0);
    const uint8_t *pb = img + (size_t)vgetq_lane_s64(yi, 1) * stride + vgetq_lane_s64(xi, 1);
    const float64x2_t p0 = {pa[0], pb[0]}, d01 = {pa[1] - pa[0], pb[1] - pb[0]};
    const float64x2_t p2 = {pa[stride], pb[stride]};
    const float64x2_t d23 = {pa[stride + 1] - pa[stride], pb[stride + 1] - pb[stride]};
    const float64x2_t fx = vsubq_f64(x, vcvtq_f64_s64(xi)), fy = vsubq_f64(y, vcvtq_f64_s64(yi));
    const float64x2_t a = vaddq_f64(p0, vmulq_f64(d01, fx)), b = vaddq_f64(p2, vmulq_f64(d23, fx));
    const float64x2_t val = vaddq_f64(a, vmulq_f64(vsubq_f64(b, a), fy));
    return im->invert ? vsubq_f64(vdupq_n_f64(255.0), val) : val;
}
#endif

/* sample_bilinear at n pixel positions, two at a time where the CPU allows. */
static void sample_at(const image_view *im, const double *x, const double *y, int n, double *out) {
    int i = 0;
#ifdef NAPA_SSE2
    for (; i + 2 <= n; i += 2) _mm_storeu_pd(out + i, sample_bilinear_pair(im, _mm_loadu_pd(x + i), _mm_loadu_pd(y + i)));
#elif defined(__aarch64__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    for (; i + 2 <= n; i += 2) vst1q_f64(out + i, sample_bilinear_pair(im, vld1q_f64(x + i), vld1q_f64(y + i)));
#endif
    for (; i < n; i++) out[i] = sample_bilinear(im, x[i], y[i]);
}

/* ---------------------------------------------------------------- threshold */

#define TILE 4
/* The finest level of a search that samples every other pixel also judges each sample against the 3x3 neighbourhood
 * of 2x2 half tiles around it, a 12 px window inside the tiles' 24 px one: nothing in the brightest NAPA_HALF_CAP
 * sixteenths of that small range is dark, whatever the wide window says. A tag's grey margin, or a pale quiet zone
 * beside a white wall, is then light beside the black border instead of joining it to the background. Where the small
 * neighbourhood is flat the wide threshold stands alone, so wide borders stay solid, and a threshold is only ever
 * lowered: dark samples are a subset of what the tiles alone would give. */
#define NAPA_HALF_CAP 6
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
#define NAPA_LENS_MIN_SIDE_POINTS 6
/* Empirical noise gate for the extra radial parameter. The F-shaped score uses 48 rather than the asymptotic
 * white-noise reference of about 24 because neighbouring edge samples share blur and pixels. */
#define NAPA_LENS_MIN_F 48.0
/* The constants above are for a lens estimated from one tag's own bowed sides; those below, for the calibrated lens
 * given by napa_set_lens. With one, tags whose sides it bows less than this (pixels, at the middle) are fitted in
 * pixels. */
#define NAPA_LENS_STRAIGHT_PX 0.02
#define NAPA_LENS_STRAIGHT_COARSE_PX 0.25 /* the same for the coarse passes, which only have to converge */
#define NAPA_LENS_KNOT_PX 20.0 /* spacing of exact lens evaluations along a side */
/* The border and payload bias readings read the tag's cells through the lens only where it moves them at least this much
 * (pixels) from the plain grid. */
#define NAPA_LENS_BIAS_STRAIGHT_PX 0.05
#define NAPA_LENS_MEMO_NEAR_PX 1.0 /* corners that moved less keep a far-straighter-than-asked verdict */
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
typedef struct { uint64_t previous; } pass_timer;
#define PROF_START (timer->previous = prof_now())
#define PROF_MARK(stage)                                                                                     \
    do {                                                                                                     \
        uint64_t prof_c_ = prof_now();                                                                                \
        napa_prof[stage] += prof_c_ - timer->previous;                                                               \
        timer->previous = prof_c_;                                                                                   \
    } while (0)
#else
typedef struct { int unused; } pass_timer;
#define PROF_START ((void)timer)
#define PROF_MARK(stage) ((void)timer)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define napa_ctz64(x) __builtin_ctzll(x)
#define napa_clz64(x) __builtin_clzll(x)
#else
static int napa_ctz64(uint64_t x) {
    int n = 0;
    while (!(x & 1)) x >>= 1, n++;
    return n;
}
static int napa_clz64(uint64_t x) {
    int n = 0;
    while (!(x >> 63)) x <<= 1, n++;
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
    uint32_t count = (uint32_t)(kernel * kernel);
    /* ceil(2^32 / count): for n = sum + count/2 <= 65408, reciprocal error
     * is < n/2^32 < 1/count, so floor(n*m/2^32) equals floor(n/count). */
    uint64_t m = ((UINT64_C(1) << 32) + count - 1) / count;
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
        for (int x = x_lo; x < x_hi; x++) dst[x] = (uint8_t)(((hs[x * s - half] + count / 2) * m) >> 32);
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
#elif defined(__aarch64__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD) && \
      defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        /* Fold all four comparisons together: each final byte holds eight consecutive bits.
         * Disjoint weights keep every pairwise sum exact in an unsigned byte. The lane-to-word
         * cast is little-endian; other NEON targets use the path below. */
        static const uint8_t weights[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
        const uint8x16_t wv = vld1q_u8(weights);
        uint8x16_t a = vandq_u8(vcltq_u8(vld1q_u8(r), vld1q_u8(t)), wv);
        uint8x16_t b = vandq_u8(vcltq_u8(vld1q_u8(r + 16), vld1q_u8(t + 16)), wv);
        uint8x16_t c = vandq_u8(vcltq_u8(vld1q_u8(r + 32), vld1q_u8(t + 32)), wv);
        uint8x16_t d = vandq_u8(vcltq_u8(vld1q_u8(r + 48), vld1q_u8(t + 48)), wv);
        uint8x16_t pairs = vpaddq_u8(vpaddq_u8(a, b), vpaddq_u8(c, d));
        uint8x16_t octets = vpaddq_u8(pairs, pairs);
        word = vget_lane_u64(vreinterpret_u64_u8(vget_low_u8(octets)), 0);
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

/* dst[2i], dst[2i + 1] = src[i] for i < n: each half tile's value for its two sample columns. */
static void expand2(uint8_t *dst, const uint8_t *src, int n) {
    int i = 0;
#ifdef NAPA_SSE2
    for (; i + 16 <= n; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i *)(src + i));
        _mm_storeu_si128((__m128i *)(dst + 2 * i), _mm_unpacklo_epi8(v, v));
        _mm_storeu_si128((__m128i *)(dst + 2 * i + 16), _mm_unpackhi_epi8(v, v));
    }
#elif defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    for (; i + 16 <= n; i += 16) {
        uint8x16_t v = vld1q_u8(src + i);
        uint8x16x2_t a = vzipq_u8(v, v);
        vst1q_u8(dst + 2 * i, a.val[0]);
        vst1q_u8(dst + 2 * i + 16, a.val[1]);
    }
#endif
    for (; i < n; i++) dst[2 * i] = dst[2 * i + 1] = src[i];
}

/* lo[t] = the least of a[2t], a[2t + 1], b[2t], b[2t + 1] and hi[t] = the greatest of the same four of c and d, for
 * t < (n + 1) / 2; an odd n's last pair is its last column alone. With a = c and b = d two rows of a level this is a
 * row of 2x2 half-tile extrema; with two half rows' minima and maxima, the 4x4 tiles' (exactly tile_row_minmax's). */
static void pair_extrema(const uint8_t *a, const uint8_t *b, const uint8_t *c, const uint8_t *d, int n, uint8_t *lo,
                         uint8_t *hi) {
    int i = 0;
#ifdef NAPA_SSE2
    /* Each sample's min and max down the two rows, then across each 16-bit lane's two bytes, left in its low byte. */
    const __m128i low = _mm_set1_epi16(0x00ff);
    for (; i + 32 <= n; i += 32) {
        __m128i mn0 = _mm_min_epu8(_mm_loadu_si128((const __m128i *)(a + i)), _mm_loadu_si128((const __m128i *)(b + i)));
        __m128i mn1 = _mm_min_epu8(_mm_loadu_si128((const __m128i *)(a + i + 16)), _mm_loadu_si128((const __m128i *)(b + i + 16)));
        __m128i mx0 = _mm_max_epu8(_mm_loadu_si128((const __m128i *)(c + i)), _mm_loadu_si128((const __m128i *)(d + i)));
        __m128i mx1 = _mm_max_epu8(_mm_loadu_si128((const __m128i *)(c + i + 16)), _mm_loadu_si128((const __m128i *)(d + i + 16)));
        mn0 = _mm_and_si128(_mm_min_epu8(mn0, _mm_srli_epi16(mn0, 8)), low);
        mn1 = _mm_and_si128(_mm_min_epu8(mn1, _mm_srli_epi16(mn1, 8)), low);
        mx0 = _mm_and_si128(_mm_max_epu8(mx0, _mm_srli_epi16(mx0, 8)), low);
        mx1 = _mm_and_si128(_mm_max_epu8(mx1, _mm_srli_epi16(mx1, 8)), low);
        _mm_storeu_si128((__m128i *)(lo + i / 2), _mm_packus_epi16(mn0, mn1));
        _mm_storeu_si128((__m128i *)(hi + i / 2), _mm_packus_epi16(mx0, mx1));
    }
#elif defined(__aarch64__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    for (; i + 32 <= n; i += 32) {
        vst1q_u8(lo + i / 2, vpminq_u8(vminq_u8(vld1q_u8(a + i), vld1q_u8(b + i)), vminq_u8(vld1q_u8(a + i + 16), vld1q_u8(b + i + 16))));
        vst1q_u8(hi + i / 2, vpmaxq_u8(vmaxq_u8(vld1q_u8(c + i), vld1q_u8(d + i)), vmaxq_u8(vld1q_u8(c + i + 16), vld1q_u8(d + i + 16))));
    }
#endif
    for (int t = i / 2; 2 * t < n; t++) {
        const int k = 2 * t, k1 = k + 1 < n ? k + 1 : k;
        uint8_t l = a[k] < b[k] ? a[k] : b[k], l1 = a[k1] < b[k1] ? a[k1] : b[k1];
        uint8_t h = c[k] > d[k] ? c[k] : d[k], h1 = c[k1] > d[k1] ? c[k1] : d[k1];
        lo[t] = l < l1 ? l : l1, hi[t] = h > h1 ? h : h1;
    }
}

/* Half row r of a level: the extrema of its 2x2 half tiles, over level rows 2r and 2r + 1 (an odd level's last row alone). */
static void half_row_minmax(const uint8_t *L, int lw, int lh, int r, uint8_t *lo, uint8_t *hi) {
    const uint8_t *a = L + (size_t)(2 * r) * lw, *b = 2 * r + 1 < lh ? a + lw : a;
    pair_extrema(a, b, a, b, lw, lo, hi);
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

/* Join the components of runs a and b, linking the larger root under the smaller, so that a component's root
 * is always its earliest run, whatever order the joins come in. */
static NAPA_FORCE_INLINE void union_roots(int32_t *P, int32_t a, int32_t b) {
    const int32_t ra = find_root(P, a), rb = find_root(P, b);
    if (ra < rb) P[rb] = ra;
    else if (rb < ra) P[ra] = rb;
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

/* Labelling sees only this level's inputs and the buffers assigned to this pass. A band replaces
 * the three private buffers; its dark-bit rows remain disjoint parts of the shared level. */
static label_context label_context_for(napa_detector *det) {
    label_context ctx = {
        .level = det->level_view ? det->level_view : det->level,
        .borrowed = det->level_view != NULL,
        .floors = det->floors,
        .floor_cols = det->floor_cols, .floor_rows = det->floor_rows, .floor_uniform = det->floor_uniform,
        .level_contrast_min = det->level_contrast_min, .level_half = det->level_half,
        .level_extrema_in = det->level_extrema_in, .level_extrema_out = det->level_extrema_out,
        .runs = det->runs, .parent = det->parent, .scratch = &det->scratch, .level_bits = det->level_bits,
        .popcnt = det->popcnt
    };
    return ctx;
}

/* Point at the owning fields so successful reallocations remain published even if a later one fails. */
static label_contacts label_contacts_for(napa_detector *det) {
    label_contacts contacts = {
        &det->diag, &det->neck, &det->neck_root,
        &det->diag_n, &det->neck_n, &det->diag_capacity, &det->neck_capacity
    };
    return contacts;
}

/* The corner rule at the diagonal between columns x and x + 1 of rows y - 1 and y, upper-left and
 * lower-right dark when left is set (else upper-right and lower-left), against the larger noise floor
 * of the regions around it. Both runs must already span two samples. */
static NAPA_FORCE_INLINE int corner_contact_at(const label_context *ctx, int lw, int lh, int factor, int x, int y,
                                               int left) {
    int floor = ctx->floors[0];
    if (!ctx->floor_uniform) {
        int cols = ctx->floor_cols, rows = ctx->floor_rows;
        int x0 = (x / TILE) * TILE * factor / NAPA_REGION;
        int x1 = ((x + 1) / TILE) * TILE * factor / NAPA_REGION;
        int y0 = ((y - 1) / TILE) * TILE * factor / NAPA_REGION;
        int y1 = (y / TILE) * TILE * factor / NAPA_REGION;
        if (x0 >= cols) x0 = cols - 1;
        if (x1 >= cols) x1 = cols - 1;
        if (y0 >= rows) y0 = rows - 1;
        if (y1 >= rows) y1 = rows - 1;
        const uint8_t *up = ctx->floors + (size_t)y0 * cols;
        const uint8_t *dn = ctx->floors + (size_t)y1 * cols;
        floor = up[x0] > up[x1] ? up[x0] : up[x1];
        if (dn[x0] > floor) floor = dn[x0];
        if (dn[x1] > floor) floor = dn[x1];
    }
    return filled_corner_contact(ctx->level, lw, lh, y, x, left, floor);
}

/* Both paths must omit the same edges, including across regional noise-floor boundaries. Run a lies in
 * the row above run b, and they touch only diagonally. */
static NAPA_NOINLINE int separate_corner_contact(const label_context *ctx, int lw, int lh, int factor, int a, int b) {
    const napa_run *R = ctx->runs;
    if (R[a].x1 <= R[a].x0 || R[b].x1 <= R[b].x0) return 0;
    int left = R[a].x1 + 1 == R[b].x0;
    return corner_contact_at(ctx, lw, lh, factor, left ? R[b].x0 - 1 : R[b].x1, R[b].y, left);
}

static NAPA_FORCE_INLINE int bit_at(const uint64_t *bits, int x) { return (int)(bits[x >> 6] >> (x & 63)) & 1; }

/* separate_corner_contact for dark bit rows: upper and lower (row y) touch only diagonally between
 * columns x and x + 1, upper dark at x when left is set. The run ending at x and the run starting at
 * x + 1 must each span two samples. */
static NAPA_NOINLINE int separate_bit_contact(const label_context *ctx, int lw, int lh, int factor,
                                              const uint64_t *upper, const uint64_t *lower, int x, int y, int left) {
    const uint64_t *ends = left ? upper : lower, *starts = left ? lower : upper;
    if (x < 1 || x + 2 >= lw || !bit_at(ends, x - 1) || !bit_at(starts, x + 2)) return 0;
    return corner_contact_at(ctx, lw, lh, factor, x, y, left);
}

/* Join adjacent rows for optional splitting. Supported filled-corner contacts stay separate, as in
 * ordinary indexing. Other corner contacts (and, with split 2, one-sample necks) are deferred as pairs. */
static NAPA_FORCE_INLINE void join_rows(const label_context *ctx, const label_contacts *contacts, const napa_run *R,
                                        int32_t *P, int a, int a_end, int b, int b_end, const int split,
                                        int lw, int lh, int factor, int shadow) {
    while (a < a_end && b < b_end) {
        if (R[a].x1 + 1 < R[b].x0) {
            a++;
        } else if (R[b].x1 + 1 < R[a].x0) {
            b++;
        } else if (split && !shadow && (R[a].x1 + 1 == R[b].x0 || R[b].x1 + 1 == R[a].x0) &&
                   separate_corner_contact(ctx, lw, lh, factor, a, b)) {
            /* These contacts also stay separate in ordinary detection: do not defer them. */
            if (R[a].x1 < R[b].x1) a++;
            else b++;
        } else if (split == 2 ? (R[a].x1 < R[b].x1 ? R[a].x1 : R[b].x1) <= (R[a].x0 > R[b].x0 ? R[a].x0 : R[b].x0)
                              : split && (R[a].x1 + 1 == R[b].x0 || R[b].x1 + 1 == R[a].x0)) {
            /* A corner contact (no overlap) or, with split 2, a neck (one sample). */
            int corner = R[a].x1 + 1 == R[b].x0 || R[b].x1 + 1 == R[a].x0;
            int32_t *pairs = corner ? *contacts->diag : *contacts->neck;
            size_t *count = corner ? contacts->diag_n : contacts->neck_n;
            pairs[(*count)++] = a;
            pairs[(*count)++] = b;
            if (R[a].x1 < R[b].x1) a++;
            else b++;
        } else {
            /* A run of this row that touches nothing yet is a singleton (runs of one row
             * join only through the row above): hang it under a's parent, which still
             * precedes it. Runs whose parents match are already joined. */
            if (P[b] == b) P[b] = P[a];
            else if (P[b] != P[a]) union_roots(P, a, b);
            if (R[a].x1 < R[b].x1) a++;
            else b++;
        }
    }
}

/* Exact 4x4 tile extrema, with short bands and the final partial tile repeated at their edges. */
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
#elif defined(__aarch64__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    /* Whole 64-sample blocks: each sample's min and max over the band's rows (a short band repeats
     * its first row), then three pairwise min/max steps leave each 4-sample tile in one byte, in order. */
    const uint8_t *row[TILE];
    for (int k = 0; k < TILE; k++) row[k] = L + (size_t)(y0 + (y0 + k < y1 ? k : 0)) * lstride;
    for (; x + 64 <= lw; x += 64) {
        uint8x16_t mn[4], mx[4];
        for (int c = 0; c < 4; c++) {
            const int at = x + 16 * c;
            uint8x16_t a = vld1q_u8(row[0] + at), b = vld1q_u8(row[1] + at);
            uint8x16_t d = vld1q_u8(row[2] + at), e = vld1q_u8(row[3] + at);
            mn[c] = vminq_u8(vminq_u8(a, b), vminq_u8(d, e));
            mx[c] = vmaxq_u8(vmaxq_u8(a, b), vmaxq_u8(d, e));
        }
        vst1q_u8(tmin + x / 4, vpminq_u8(vpminq_u8(mn[0], mn[1]), vpminq_u8(mn[2], mn[3])));
        vst1q_u8(tmax + x / 4, vpmaxq_u8(vpmaxq_u8(mx[0], mx[1]), vpmaxq_u8(mx[2], mx[3])));
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

/* A population count for the labelling loops, which are compiled twice on x86: hw is a constant 1 in the copy
 * built for the POPCNT instruction (threshold_index_popcnt) and 0 elsewhere. */
#ifdef NAPA_SSSE3_DISPATCH
#define INDEX_POPCOUNT(hw, x) ((hw) ? __builtin_popcountll(x) : napa_popcount64(x))
#else
#define INDEX_POPCOUNT(hw, x) ((void)(hw), napa_popcount64(x))
#endif
/* Words of a row's edge index (index_edges): one per 64 samples, the word holding sample lw + 1, and a spare. */
enum { INDEX_WORDS = (NAPA_MAX_SIDE + 63) / 64 + 2 };

/* Edges of the runs of a | b: starts and the samples after ends, alternating. prefix counts them
 * through each word; word `words` is empty, for edges_before. */
static NAPA_FORCE_INLINE void index_edges(const uint64_t *a, const uint64_t *b, int words, uint64_t *mask,
                                          int32_t *prefix, const int hw) {
    uint64_t carry = 0;
    int count = 0;
    for (int w = 0; w < words; w++) {
        uint64_t x = a[w] | b[w], e = x ^ ((x << 1) | carry);
        carry = x >> 63;
        mask[w] = e;
        count += INDEX_POPCOUNT(hw, e);
        prefix[w] = count;
    }
    mask[words] = 0;
    prefix[words] = count;
}

/* Edges before q, for 0 <= q <= lw + 1. Halved and rounded down: runs ending before q - 1; rounded
 * up: runs starting before q. */
static NAPA_FORCE_INLINE int edges_before(const uint64_t *mask, const int32_t *prefix, int q, const int hw) {
    return prefix[q >> 6] - INDEX_POPCOUNT(hw, mask[q >> 6] >> (q & 63));
}

/* Label one complete run using the row's original ranks and union order.
 * Its complete record is already stored when contact checks inspect it.
 * Edges alternate start/end+1. Inclusive word counts minus the suffix from q give
 * counts before q, including at word and sentinel boundaries. Counts before b.x0
 * and b.x1+2 give the eight-connected range above; ordinary indexing then trims
 * supported corner contacts. Visit b and its neighbors in raster order so backward
 * roots retain the original packed order. Single-row runs need no row-pair mask. */
static NAPA_FORCE_INLINE void label_indexed_run(const label_context *ctx, int lw, int lh, int factor,
                                  int b, int prev_start, const uint64_t *mask, const int32_t *prefix, int shadow, int hw) {
    napa_run *R = ctx->runs;
    int32_t *P = ctx->parent;
    int f = prev_start + (edges_before(mask, prefix, R[b].x0, hw) >> 1);
    int g = prev_start + ((edges_before(mask, prefix, R[b].x1 + 2, hw) + 1) >> 1);
    if (!shadow && f < g) {
        if (R[f].x1 + 1 == R[b].x0 && separate_corner_contact(ctx, lw, lh, factor, f, b)) f++;
        if (f < g && R[g - 1].x0 == R[b].x1 + 1 && separate_corner_contact(ctx, lw, lh, factor, g - 1, b)) g--;
    }
    P[b] = f < g ? P[f] : b;
    for (int a = f + 1; a < g; a++)
        if (P[b] != P[a]) union_roots(P, a, b);
}

static NAPA_FORCE_INLINE int extract_labeled_runs(const label_context *ctx, const uint64_t *bits,
                                    int lw, int lh, int factor, int y, int n, int prev_start,
                                    const uint64_t *mask, const int32_t *prefix, const int shadow, int hw) {
    napa_run *R = ctx->runs;
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
                label_indexed_run(ctx, lw, lh, factor, n, prev_start, mask, prefix, shadow, hw);
                n++, open = 0;
            }
        }
    }
    if (open) {
        R[n].x0 = (uint16_t)x0, R[n].x1 = (uint16_t)(lw - 1), R[n].y = (uint16_t)y;
        label_indexed_run(ctx, lw, lh, factor, n, prev_start, mask, prefix, shadow, hw);
        n++;
    }
    return n;
}

/* Row y's dark bits. Borrowed input need not have the owned level's 64 padding bytes: for narrow
 * images, rounded word loads can cross the array before the final row. */
static NAPA_FORCE_INLINE void level_row_bits(const label_context *ctx, const uint8_t *L, int lw, int lh, int y,
                                             const uint8_t *trow, uint64_t *bits) {
    if (ctx->borrowed && (size_t)(lh - y) * lw < ((size_t)lw + 63) / 64 * 64) {
        int whole = lw & ~63;
        uint8_t tail[64] = {0};
        memcpy(tail, L + (size_t)y * lw + whole, (size_t)(lw - whole));
        dark_bits(L + (size_t)y * lw, trow, whole, bits);
        dark_bits(tail, trow + whole, 64, bits + whole / 64);
    } else dark_bits(L + (size_t)y * lw, trow, lw, bits);
}

/* ---- Row-pair labelling: ordinary detection (no split, no shadow).
 * Labelling runs of each row pair's union follows PR23's row-pair scan,
 * 7735f7a2e41b40584cd48a7f07c0bd947c4993c2 (same project/license), extended to keep corner contacts exact.
 *
 * Rows go in pairs from each tile row's first row, so a pair's top row y is even; a level's odd last row
 * is a pair alone. A run of the pair's union (samples where either row is dark) is one 8-connected
 * piece of the two rows, unless the rows meet inside it only diagonally at a corner that single-row
 * labelling keeps apart (separate_corner_contact). A pair without such a corner is labelled as its
 * union runs (napa_run.rows 1, 2 or 3), about half as many union-find nodes as single rows. A pair
 * with one (a few percent of pairs) keeps its single-row runs, top row first (rows 1, y their own
 * row), joined exactly as single-row labelling joins them. Either way the runs the next pair can
 * touch (all union runs, or the bottom row's runs) are disjoint and separated by columns where the
 * pair's bottom row is empty, which makes the window test in join_above exact.
 *
 * Runs are stored in the order single-row labelling first meets them, so union by smaller index keeps
 * each blob's root, and the blobs' numbering, as single-row labelling has them. napa_detect's resolve
 * and extremes passes (comp_add_pair_run) read the rows' extents and counts from det->level_bits. */

/* Append the runs whose edges index_edges found, as row y's runs (rows 1). */
static NAPA_FORCE_INLINE int runs_from_edges(const uint64_t *mask, int lw, int words, int y, napa_run *R, int n) {
    int open = 0, x0 = 0;
    for (int w = 0; w < words; w++) {
        for (uint64_t e = mask[w]; e; e &= e - 1) {
            int x = 64 * w + napa_ctz64(e);
            if (!open) {
                x0 = x, open = 1;
            } else {
                R[n].x0 = (uint16_t)x0, R[n].x1 = (uint16_t)(x - 1), R[n].y = (uint16_t)y, R[n].rows = 1;
                n++, open = 0;
            }
        }
    }
    if (open) {
        R[n].x0 = (uint16_t)x0, R[n].x1 = (uint16_t)(lw - 1), R[n].y = (uint16_t)y, R[n].rows = 1;
        n++;
    }
    return n;
}

/* Whether bits lo..hi (lo <= hi) hold a set bit. */
static NAPA_FORCE_INLINE int bits_any(const uint64_t *b, int lo, int hi) {
    int w = lo >> 6, last = hi >> 6;
    uint64_t v = b[w] & (~(uint64_t)0 << (lo & 63));
    while (w < last && !v) v = b[++w];
    if (w == last) v &= ~(uint64_t)0 >> (63 - (hi & 63));
    return v != 0;
}

/* Word w of the windows where rows up (y - 1) and dn (y) touch as single-row labelling joins them;
 * carry holds the rows' bits 63 of word w - 1 (bit 0 up, bit 1 dn). Window x holds samples x - 1 and
 * x of both rows, so at most one run of each row there, and the rows touch through it when both are
 * dark in it. A window dark only at opposite corners holds a diagonal contact, which the corner rule
 * may keep apart; any other holds an overlap. */
static NAPA_FORCE_INLINE uint64_t boundary_word(const label_context *ctx, int lw, int lh, int factor,
                                                const uint64_t *up, const uint64_t *dn, int w, int y, int *carry) {
    const uint64_t u = up[w], d = dn[w], ul = u << 1 | (uint64_t)(*carry & 1), dl = d << 1 | (uint64_t)(*carry >> 1);
    *carry = (int)(u >> 63) | (int)(d >> 63) << 1;
    uint64_t win = (u | ul) & (d | dl);
    /* Bit x of u2 / d2: column x - 2 of up / dn; of u1 / d1: column x + 1 (rows end in a zero padding word). A
     * diagonal at window x touches columns x - 1 and x; separate_bit_contact keeps it only when the dark sample
     * before it and the one after it are dark too, so it is called for no other. */
    const uint64_t u2 = u << 2 | (w ? up[w - 1] >> 62 : 0), d2 = d << 2 | (w ? dn[w - 1] >> 62 : 0);
    const uint64_t u1 = u >> 1 | up[w + 1] << 63, d1 = d >> 1 | dn[w + 1] << 63;
    const uint64_t kept = (ul & ~u & d & ~dl & u2 & d1) | (dl & ~d & u & ~ul & d2 & u1);
    for (uint64_t diag = win & kept; diag; diag &= diag - 1) {
        const int x = 64 * w + napa_ctz64(diag);
        if (separate_bit_contact(ctx, lw, lh, factor, up, dn, x - 1, y, (int)(ul >> (x & 63)) & 1))
            win &= ~((uint64_t)1 << (x & 63));
    }
    return win;
}

/* All the boundary windows. Window lw (in the padding word) is left empty: it holds only column
 * lw - 1, which window lw - 1 holds too. */
static NAPA_FORCE_INLINE void boundary_windows(const label_context *ctx, int lw, int lh, int factor,
                                               const uint64_t *up, const uint64_t *dn, int words, int y,
                                               uint64_t *junction) {
    int carry = 0;
    for (int w = 0; w < words; w++) junction[w] = boundary_word(ctx, lw, lh, factor, up, dn, w, y, &carry);
    junction[words] = 0;
}

/* One pass over a pair's words (has_bottom: whether row y + 1 is in it): its union's run edges, as
 * index_edges; at each run's end + 1, whether the run holds top-row (ft) or bottom-row (fb) samples (a
 * row's dark bits lie inside the union's runs, so adding them to the union carries out of exactly the
 * runs holding one); and, given the row above (up), the boundary windows, as boundary_windows. Returns
 * 0 when the pair's rows meet only diagonally somewhere single-row labelling keeps apart: the pair then
 * keeps single rows, and the windows are incomplete. */
static NAPA_FORCE_INLINE int index_pair(const label_context *ctx, int lw, int lh, int factor, const uint64_t *up,
                                        const uint64_t *top, const uint64_t *bot, int has_bottom, int words, int y,
                                        uint64_t *mask, int32_t *prefix, uint64_t *ft, uint64_t *fb,
                                        uint64_t *junction, const int hw) {
    uint64_t carry = 0, ct = 0, cb = 0;
    int count = 0, boundary = 0;
    for (int w = 0; w < words; w++) {
        const uint64_t t = top[w], b = has_bottom ? bot[w] : 0, x = t | b;
        /* Bit x: the rows meet only diagonally between columns x and x + 1 (a row's word past its end
         * is the zero padding word). */
        const uint64_t tn = t >> 1 | top[w + 1] << 63, bn = has_bottom ? b >> 1 | bot[w + 1] << 63 : 0;
        /* As in boundary_word: only diagonals whose dark sample before and after are dark too (bit x of tp / bp:
         * column x - 1; of t2 / b2: column x + 2) can be kept apart by separate_bit_contact. */
        const uint64_t tp = t << 1 | (w ? top[w - 1] >> 63 : 0), t2 = t >> 2 | top[w + 1] << 62;
        const uint64_t bp = has_bottom ? b << 1 | (w ? bot[w - 1] >> 63 : 0) : 0;
        const uint64_t b2 = has_bottom ? b >> 2 | bot[w + 1] << 62 : 0;
        for (uint64_t d = (t & ~b & bn & ~tn & tp & b2) | (b & ~t & tn & ~bn & bp & t2); d; d &= d - 1) {
            const int at = 64 * w + napa_ctz64(d);
            if (separate_bit_contact(ctx, lw, lh, factor, top, bot, at, y + 1, (int)(t >> (at & 63)) & 1)) return 0;
        }
        const uint64_t e = x ^ (x << 1 | carry);
        carry = x >> 63;
        mask[w] = e;
        count += INDEX_POPCOUNT(hw, e);
        prefix[w] = count;
        uint64_t sum = x + t, v = sum + ct;
        ct = (uint64_t)((sum < x) | (v < sum));
        ft[w] = v & ~x;
        sum = x + b, v = sum + cb;
        cb = (uint64_t)((sum < x) | (v < sum));
        fb[w] = v & ~x;
        if (up) junction[w] = boundary_word(ctx, lw, lh, factor, up, top, w, y, &boundary);
    }
    mask[words] = 0, prefix[words] = count;
    ft[words] = ct, fb[words] = cb;
    if (up) junction[words] = 0;
    return 1;
}

/* The positions of mask's set bits (index_edges' edges), in order, without a branch per edge: eight
 * at a time, bit 63 stopping count-trailing-zeros on an exhausted word (the spare entries it writes
 * past the edges are overwritten or unused). Returns how many. */
static NAPA_FORCE_INLINE int decode_edges(const uint64_t *mask, int words, uint16_t *edge, const int hw) {
    const uint64_t stop = (uint64_t)1 << 63;
    int ne = 0;
    for (int w = 0; w < words; w++) {
        uint64_t e = mask[w];
        if (!e) continue;
        uint16_t *o = edge + ne;
        ne += INDEX_POPCOUNT(hw, e);
        do {
            for (int k = 0; k < 8; k++) {
                o[k] = (uint16_t)(64 * w + napa_ctz64(e | stop));
                e &= e - 1;
            }
            o += 8;
        } while (e);
    }
    return ne;
}

/* Join run c (dark in this pair's top row) to the runs of the pair above it touches. That pair's
 * touchable runs, left to right, are prank[0...], and their edges pmask / pprefix. Windows
 * max(x0)..min(x1) + 1 of two such runs belong to those two runs, because their rows are empty beside
 * them. A run touches nothing earlier in its own pair, so the first touching run's parent, which
 * precedes it, takes it. */
static NAPA_FORCE_INLINE void join_above(const label_context *ctx, int c, const int32_t *prank, const uint64_t *pmask,
                                         const int32_t *pprefix, const uint64_t *junction, const int hw) {
    napa_run *R = ctx->runs;
    int32_t *P = ctx->parent;
    const int f = edges_before(pmask, pprefix, R[c].x0, hw) >> 1;
    const int g = (edges_before(pmask, pprefix, R[c].x1 + 2, hw) + 1) >> 1;
    for (int k = f; k < g; k++) {
        const int a = prank[k];
        int lo = R[a].x0 > R[c].x0 ? R[a].x0 : R[c].x0;
        int hi = (R[a].x1 < R[c].x1 ? R[a].x1 : R[c].x1) + 1;
        if (!bits_any(junction, lo, hi)) continue;
        if (P[c] == c) P[c] = P[a];
        else if (P[c] != P[a]) union_roots(P, a, c);
    }
}

/* Label row pair y (has_bottom: whether row y + 1 is in it), whose dark bits are in ctx->level_bits, against
 * the pair above's touchable runs (prank, edges pmask / pprefix; prank NULL when the rows are not
 * adjacent). Runs are stored in the order single-row labelling meets them first: those dark in the top
 * row, left to right, then the rest, left to right. Union by smaller index then keeps each blob's root
 * at its first sample in raster order. This pair's touchable runs go to rank, left to right, and their
 * edges to mask / prefix. The scratch rows hold the union's run flags and a single-row pair's top-row
 * edges; ctx->scratch->row_bits holds the boundary windows. */
static NAPA_FORCE_INLINE int label_row_pair(const label_context *ctx, int lw, int lh, int factor, int y, int has_bottom, int n,
                                            const int32_t *prank, const uint64_t *pmask, const int32_t *pprefix,
                                            int32_t *rank, uint64_t *mask, int32_t *prefix, uint64_t *scratch[3],
                                            int32_t *tprefix, const int hw) {
    napa_run *R = ctx->runs;
    int32_t *P = ctx->parent;
    const int words = (lw + 63) / 64, stride = words + 1, start = n;
    const uint64_t *top = ctx->level_bits + (size_t)y * stride, *bot = top + stride;
    const uint64_t *up = prank ? top - stride : NULL;
    uint64_t *junction = ctx->scratch->row_bits, *ft = scratch[0], *fb = scratch[1];
    if (index_pair(ctx, lw, lh, factor, up, top, bot, has_bottom, words, y, mask, prefix, ft, fb, junction, hw)) {
        uint16_t *edge = ctx->scratch->row_edges;
        int32_t *later = ctx->scratch->pair_rank + 2 * (size_t)(lw / 2 + 2);
        int ne = decode_edges(mask, words, edge, hw), nlater = 0;
        if (ne & 1) edge[ne++] = (uint16_t)lw; /* a run still open at the end of a whole last word */
        for (int k = 0; k < ne; k += 2) {
            const int x0 = edge[k], end = edge[k + 1];
            if (!((ft[end >> 6] >> (end & 63)) & 1)) {
                /* Dark in the bottom row only: stored after the rest. Its edges move down to entry
                 * 2 * nlater, already read. */
                edge[2 * nlater] = (uint16_t)x0, edge[2 * nlater + 1] = (uint16_t)end;
                later[nlater++] = k >> 1;
                continue;
            }
            R[n].x0 = (uint16_t)x0, R[n].x1 = (uint16_t)(end - 1), R[n].y = (uint16_t)y;
            R[n].rows = (uint16_t)(1 | ((fb[end >> 6] >> (end & 63)) & 1) << 1);
            P[n] = n;
            rank[k >> 1] = n;
            if (prank) join_above(ctx, n, prank, pmask, pprefix, junction, hw);
            n++;
        }
        for (int j = 0; j < nlater; j++, n++) {
            R[n].x0 = edge[2 * j], R[n].x1 = (uint16_t)(edge[2 * j + 1] - 1), R[n].y = (uint16_t)y, R[n].rows = 2;
            P[n] = n;
            rank[later[j]] = n;
        }
    } else {
        uint64_t *tmask = scratch[2];
        if (prank) boundary_windows(ctx, lw, lh, factor, up, top, words, y, junction);
        index_edges(top, top, words, tmask, tprefix, hw);
        n = runs_from_edges(tmask, lw, words, y, R, n);
        for (int c = start; c < n; c++) {
            P[c] = c;
            if (prank) join_above(ctx, c, prank, pmask, pprefix, junction, hw);
        }
        const int mid = n;
        index_edges(bot, bot, words, mask, prefix, hw);
        n = runs_from_edges(mask, lw, words, y + 1, R, n);
        for (int b = mid; b < n; b++) {
            label_indexed_run(ctx, lw, lh, factor, b, start, tmask, tprefix, 0, hw);
            rank[b - mid] = b;
        }
    }
    return n;
}

/* Join the first pair of a band (threshold_banded), whose runs are c..end - 1 and whose top row y is even, to
 * the pair above it (top row y - 2), the last of the band before, whose runs end at c. That pair's touchable
 * runs and their edges are rebuilt from the dark bits, and each run of the band's first pair that is dark in
 * row y joins them as label_row_pair would have: through join_above, with the same windows. */
static void join_band_pairs(const label_context *ctx, int lw, int lh, int factor, int y, int c, int end) {
    uint64_t mask[INDEX_WORDS];
    int32_t prefix[INDEX_WORDS];
    const napa_run *R = ctx->runs;
    const int words = (lw + 63) / 64, stride = words + 1;
    int above = c;
    while (above > 0 && R[above - 1].y >= y - 2) above--;
    if (above == c || c == end || R[c].y != y || !(R[c].rows & 1)) return; /* nothing labelled to join on one side */
    const uint64_t *top = ctx->level_bits + (size_t)(y - 2) * stride, *bot = top + stride;
    /* A pair kept as single rows stores its bottom row's runs last; they are its touchable runs. */
    const int single = R[c - 1].y == y - 1;
    int32_t *prank = ctx->scratch->pair_rank;
    index_edges(single ? bot : top, bot, words, mask, prefix, 0);
    for (int a = above; a < c; a++)
        if (!single || R[a].y == y - 1) prank[edges_before(mask, prefix, R[a].x0, 0) >> 1] = a;
    boundary_windows(ctx, lw, lh, factor, bot, bot + stride, words, y, ctx->scratch->row_bits);
    for (; c < end && R[c].y == y && (R[c].rows & 1); c++) join_above(ctx, c, prank, mask, prefix, ctx->scratch->row_bits, 0);
}

/* Join the first row y of a band (threshold_banded) of shadow labelling, whose runs are c..end - 1, to row
 * y - 1, the last of the band before, whose runs end at c, as label_indexed_run would have: for each run, left
 * is the first run above whose end reaches its left neighbour, right the first that starts past its right
 * one. Shadow labelling keeps corner contacts, so every run in left..right - 1 joins. */
static void join_band_rows(const label_context *ctx, int y, int c, int end) {
    const napa_run *R = ctx->runs;
    int32_t *P = ctx->parent;
    const int first = c; /* the band's first run: row y - 1's runs end here */
    int above = first;
    while (above > 0 && R[above - 1].y == y - 1) above--;
    if (above == first) return; /* nothing labelled in row y - 1 */
    int left = above, right = above;
    for (; c < end && R[c].y == y; c++) {
        while (left < first && R[left].x1 + 1 < R[c].x0) left++;
        if (right < left) right = left;
        while (right < first && R[right].x0 <= R[c].x1 + 1) right++;
        for (int a = left; a < right; a++) union_roots(P, a, c);
    }
}

/* Threshold the level and label its dark samples as runs joined by union-find. Adjacent-row runs
 * connect within one sample, except supported filled-corner contacts in ordinary detection. Returns
 * the run count, or -1 when out of memory. Optional splitting defers the remaining corner contacts
 * (split 1), or also one-sample necks (split 2), in contacts for split_pieces. */
/* Works over tile rows [ty0, ty1) of the level. Tile extrema come from level_extrema_in when saved; otherwise they are
 * computed here one tile row ahead, into ctx->scratch->tile_min and tile_max, which hold tile rows from `base`
 * on (a band keeps only its own rows and the one on each side). */
static NAPA_FORCE_INLINE int threshold_index_impl(const label_context *ctx, const label_contacts *contacts,
                                                  int lw, int lh, int factor, const int split, int shadow, const int hw,
                                                  int ty0, int ty1, int base) {
    const uint8_t *L = ctx->level;
    napa_run *R = ctx->runs;
    int32_t *P = ctx->parent;
    /* Run edges of the previous and current row (row pairs: also a fallback pair's top row). */
    uint64_t masks[3][INDEX_WORDS];
    int32_t prefixes[3][INDEX_WORDS];
    uint64_t flags[2][INDEX_WORDS]; /* row pairs: which rows each union run holds */
    uint64_t *pair_scratch[3] = {flags[0], flags[1], masks[2]};
    const int paired = !split && !shadow;
    int n = 0, prev_y = -2, prev_start = 0, prev_end = 0;
    if (contacts) *contacts->diag_n = *contacts->neck_n = 0;
    int tw = (lw + TILE - 1) / TILE;
    int th = (lh + TILE - 1) / TILE;
    uint8_t *vmin = ctx->scratch->row_min, *vmax = ctx->scratch->row_max, *trow = ctx->scratch->row_thr;

    /* 3x3 tile neighbourhood, separably and without branches so the loops vectorise: vertically
     * into a row padded by one tile at each end (repeating the edge tile), then horizontally. */
    uint8_t *nlo = ctx->scratch->nb_lo, *nhi = ctx->scratch->nb_hi, *tt = ctx->scratch->nb_t, *fl = ctx->scratch->nb_f;
    const uint8_t contrast_min = ctx->level_contrast_min;
    /* A tile spans 4 * factor samples of the grid where the floors were measured. */
    int rows = ctx->floor_rows, cols = ctx->floor_cols;
    if (ctx->floor_uniform) memset(fl, ctx->floors[0], (size_t)tw);
    int floor_row = -1;
    const uint8_t *saved = ctx->level_extrema_in;
    const size_t tile_count = (size_t)tw * th;
    if (saved) base = 0;
    const uint8_t *tile_min = saved ? saved : ctx->scratch->tile_min;
    const uint8_t *tile_max = saved ? saved + tile_count : ctx->scratch->tile_max;
    /* Half tiles (level_half): half row r's extrema live in ring rows r & 7, from the half row above the tile row to
     * the two of the tile row below, and each tile row's extrema are the pairs of its two half rows'. */
    const int half = ctx->level_half && !shadow, hw2 = (lw + 1) / 2, hh2 = (lh + 1) / 2;
    const size_t hs = (size_t)(lw / 2 + 67);
    uint8_t *const hmin = ctx->scratch->half, *const hmax = hmin + 8 * hs, *const hthr = hmax + 8 * hs;
    uint8_t *const htile = hthr + 2 * hs, *const hfloor = htile + hs, *const hlo = hfloor + hs, *const hhi = hlo + hs;
    if (half)
        for (int r = ty0 > 0 ? 2 * ty0 - 2 : 0; r < 2 * ty0 + 2 && r < hh2; r++)
            half_row_minmax(L, lw, lh, r, hmin + (r & 7) * hs, hmax + (r & 7) * hs);
    if (!saved)
        for (int r = ty0 > 0 ? ty0 - 1 : 0; r <= ty0 && r < th; r++) {
            uint8_t *tmin = ctx->scratch->tile_min + (size_t)(r - base) * tw, *tmax = ctx->scratch->tile_max + (size_t)(r - base) * tw;
            if (half) {
                const int a = (2 * r) & 7, b = 2 * r + 1 < hh2 ? (2 * r + 1) & 7 : a;
                pair_extrema(hmin + a * hs, hmin + b * hs, hmax + a * hs, hmax + b * hs, hw2, tmin, tmax);
            } else tile_row_minmax(L, lw, lw, lh, r, tw, vmin, vmax, tmin, tmax);
        }
    for (int ty = ty0; ty < ty1; ty++) {
        if (half)
            for (int r = 2 * ty + 2; r < 2 * ty + 4 && r < hh2; r++)
                half_row_minmax(L, lw, lh, r, hmin + (r & 7) * hs, hmax + (r & 7) * hs);
        if (!saved && ty + 1 < th) {
            uint8_t *tmin = ctx->scratch->tile_min + (size_t)(ty + 1 - base) * tw, *tmax = ctx->scratch->tile_max + (size_t)(ty + 1 - base) * tw;
            if (half) {
                const int a = (2 * ty + 2) & 7, b = 2 * ty + 3 < hh2 ? (2 * ty + 3) & 7 : a;
                pair_extrema(hmin + a * hs, hmin + b * hs, hmax + a * hs, hmax + b * hs, hw2, tmin, tmax);
            } else tile_row_minmax(L, lw, lw, lh, ty + 1, tw, vmin, vmax, tmin, tmax);
        }
        int ry = ty * TILE * factor / NAPA_REGION;
        ry = ry < rows ? ry : rows - 1;
        if (!ctx->floor_uniform && ry != floor_row) {
            /* Each floor region covers a run of whole tiles: fill the row's floors run by run,
             * and only when the tile row enters a new region row. */
            const uint8_t *frow = ctx->floors + (size_t)ry * cols;
            floor_row = ry;
            for (int tx = 0; tx < tw;) {
                int rx = tx * TILE * factor / NAPA_REGION;
                int end = ((rx + 1) * NAPA_REGION + TILE * factor - 1) / (TILE * factor);
                if (end > tw || rx >= cols - 1) end = tw;
                memset(fl + tx, frow[rx < cols ? rx : cols - 1], (size_t)(end - tx));
                tx = end;
            }
        }
        const uint8_t *lo_up = tile_min + (size_t)((ty > 0 ? ty - 1 : ty) - base) * tw;
        const uint8_t *lo_mid = tile_min + (size_t)(ty - base) * tw;
        const uint8_t *lo_dn = tile_min + (size_t)((ty + 1 < th ? ty + 1 : ty) - base) * tw;
        const uint8_t *hi_up = tile_max + (size_t)((ty > 0 ? ty - 1 : ty) - base) * tw;
        const uint8_t *hi_mid = tile_max + (size_t)(ty - base) * tw;
        const uint8_t *hi_dn = tile_max + (size_t)((ty + 1 < th ? ty + 1 : ty) - base) * tw;
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
        if (!any) continue; /* no contrast anywhere in this tile row: no dark samples, no runs */
        if (half) {
            /* Each of the tile row's two half rows: the tile thresholds, capped where the half tile's own 3x3
             * neighbourhood has the contrast a tile's must have. Bytes only and no branches, so the loops vectorise. */
            expand2(htile, tt, tw), expand2(hfloor, fl, tw);
            for (int k = 0; k < 2 && 2 * ty + k < hh2; k++) {
                const int r = 2 * ty + k, up = (r > 0 ? r - 1 : r) & 7, dn = (r + 1 < hh2 ? r + 1 : r) & 7;
                uint8_t *NAPA_RESTRICT out = hthr + k * hs;
                min3_max3(hlo + 1, hhi + 1, hmin + up * hs, hmin + (r & 7) * hs, hmin + dn * hs, hmax + up * hs,
                          hmax + (r & 7) * hs, hmax + dn * hs, hw2);
                hlo[0] = hlo[1], hhi[0] = hhi[1], hlo[hw2 + 1] = hlo[hw2], hhi[hw2 + 1] = hhi[hw2];
                for (int tx = 0; tx < hw2; tx++) {
                    uint8_t l = hlo[tx + 1], h = hhi[tx + 1];
                    l = hlo[tx] < l ? hlo[tx] : l;
                    l = hlo[tx + 2] < l ? hlo[tx + 2] : l;
                    h = hhi[tx] > h ? hhi[tx] : h;
                    h = hhi[tx + 2] > h ? hhi[tx + 2] : h;
                    const uint8_t c = (uint8_t)(h - l);
                    const uint8_t cap = c >= hfloor[tx] && c >= contrast_min ? (uint8_t)(h - ((c * NAPA_HALF_CAP) >> 4)) : 255;
                    out[tx] = cap < htile[tx] ? cap : htile[tx];
                }
            }
        } else {
            expand4(trow, tt, tw);
            memset(trow + lw, 0, (size_t)(4 * tw + 64 - lw)); /* nothing past the row is dark */
        }
        if (paired) {
            const int words = (lw + 63) / 64, stride = words + 1;
            for (int y = y0; y < y1; y += 2) {
                const int has_bottom = y + 1 < y1, side = (y >> 1) & 1;
                if (half) {
                    expand2(trow, hthr + ((y - y0) >> 1) * hs, hw2);
                    memset(trow + lw, 0, (size_t)(4 * tw + 64 - lw));
                }
                uint64_t *top = ctx->level_bits + (size_t)y * stride;
                level_row_bits(ctx, L, lw, lh, y, trow, top);
                top[words] = 0;
                if (has_bottom) {
                    level_row_bits(ctx, L, lw, lh, y + 1, trow, top + stride);
                    top[stride + words] = 0;
                }
                int32_t *rank = ctx->scratch->pair_rank + (size_t)side * (lw / 2 + 2);
                const int32_t *prank = prev_y == y - 1 ? ctx->scratch->pair_rank + (size_t)(side ^ 1) * (lw / 2 + 2) : NULL;
                n = label_row_pair(ctx, lw, lh, factor, y, has_bottom, n, prank, masks[side ^ 1], prefixes[side ^ 1], rank,
                                   masks[side], prefixes[side], pair_scratch, prefixes[2], hw);
                prev_y = y + has_bottom;
            }
            continue;
        }
        uint64_t *bits = ctx->scratch->row_bits;
        for (int y = y0; y < y1; y++) {
            if (half && !((y - y0) & 1)) {
                expand2(trow, hthr + ((y - y0) >> 1) * hs, hw2);
                memset(trow + lw, 0, (size_t)(4 * tw + 64 - lw));
            }
            level_row_bits(ctx, L, lw, lh, y, trow, bits);
            /* Shadow labelling labels each run as it is emitted, against the row above's edge index; split
             * labelling stores the rows' runs unlabelled and joins them below (join_rows). */
            if (!split) index_edges(bits, bits, (lw + 63) / 64, masks[y & 1], prefixes[y & 1], hw);
            int start = n;
            if (!split && prev_y == y - 1 && prev_start < prev_end)
                n = extract_labeled_runs(ctx, bits, lw, lh, factor, y, n, prev_start,
                                        masks[(y & 1) ^ 1], prefixes[(y & 1) ^ 1], shadow, hw);
            else n = extract_runs(bits, lw, y, R, n);
            if (split || prev_y != y - 1 || prev_start == prev_end)
                for (int i = start; i < n; i++) P[i] = i;
            if (prev_y == y - 1 && split) {
                /* Two adjacent rows touch at most once per run of either. */
                size_t most = 2 * ((size_t)(prev_end - prev_start) + (size_t)(n - start));
                if (*contacts->diag_n + most > *contacts->diag_capacity) {
                    size_t cap = *contacts->diag_capacity ? 2 * *contacts->diag_capacity : 4096;
                    while (cap < *contacts->diag_n + most) cap *= 2;
                    int32_t *p = (int32_t *)realloc(*contacts->diag, cap * sizeof(int32_t));
                    if (p == NULL) return -1;
                    *contacts->diag = p;
                    *contacts->diag_capacity = cap;
                }
                if (split == 2 && *contacts->neck_n + most > *contacts->neck_capacity) {
                    size_t cap = *contacts->neck_capacity ? 2 * *contacts->neck_capacity : 4096;
                    while (cap < *contacts->neck_n + most) cap *= 2;
                    int32_t *p = (int32_t *)realloc(*contacts->neck, cap * sizeof(int32_t));
                    if (p == NULL) return -1;
                    *contacts->neck = p;
                    if ((p = (int32_t *)realloc(*contacts->neck_root, cap * sizeof(int32_t))) == NULL) return -1;
                    *contacts->neck_root = p;
                    *contacts->neck_capacity = cap;
                }
                if (split == 2) join_rows(ctx, contacts, R, P, prev_start, prev_end, start, n, 2, lw, lh, factor, shadow);
                else join_rows(ctx, contacts, R, P, prev_start, prev_end, start, n, 1, lw, lh, factor, shadow);
            }
            prev_y = y, prev_start = start, prev_end = n;
        }
    }
    if (ctx->level_extrema_out) {
        memcpy(ctx->level_extrema_out, ctx->scratch->tile_min, tile_count);
        memcpy(ctx->level_extrema_out + tile_count, ctx->scratch->tile_max, tile_count);
    }
    return n;
}
static int threshold_index_generic(const label_context *ctx, const label_contacts *contacts,
                                   int lw, int lh, int factor, int split, int shadow, int ty0, int ty1, int base) {
    /* Keep the common ordinary loop free of recovery-mode branches. */
    if (!split && !shadow) return threshold_index_impl(ctx, contacts, lw, lh, factor, 0, 0, 0, ty0, ty1, base);
    return threshold_index_impl(ctx, contacts, lw, lh, factor, split, shadow, 0, ty0, ty1, base);
}
#ifdef NAPA_SSSE3_DISPATCH
__attribute__((target("popcnt"))) static int threshold_index_popcnt(const label_context *ctx, const label_contacts *contacts,
                                   int lw, int lh, int factor, int split, int shadow, int ty0, int ty1, int base) {
    /* Keep the common ordinary loop free of recovery-mode branches. */
    if (!split && !shadow) return threshold_index_impl(ctx, contacts, lw, lh, factor, 0, 0, 1, ty0, ty1, base);
    return threshold_index_impl(ctx, contacts, lw, lh, factor, split, shadow, 1, ty0, ty1, base);
}
#endif
static int threshold_context(const label_context *ctx, const label_contacts *contacts,
                                   int lw, int lh, int factor, int split, int shadow, int ty0, int ty1, int base) {
#ifdef NAPA_SSSE3_DISPATCH
    if (ctx->popcnt) return threshold_index_popcnt(ctx, contacts, lw, lh, factor, split, shadow, ty0, ty1, base);
#endif
    return threshold_index_generic(ctx, contacts, lw, lh, factor, split, shadow, ty0, ty1, base);
}

static int threshold_rows(napa_detector *det, int lw, int lh, int factor, int split, int shadow, int ty0, int ty1, int base) {
    label_context ctx = label_context_for(det);
    label_contacts contacts = label_contacts_for(det);
    return threshold_context(&ctx, &contacts, lw, lh, factor, split, shadow, ty0, ty1, base);
}

/* ---------------------------------------------------------------- banded labelling */

/* Banded labelling (napa_set_threads). The level's tile rows are cut into contiguous bands; each band is
 * thresholded and labelled on its own, into its own runs, exactly as the serial pass would, except that its
 * first row has no row above. The bands' runs are then concatenated in raster order and each band's first
 * row is joined to the row above it with the same contact rule. Every union links the larger root to the
 * smaller, so a component's root is always its first run: the components, their raster order and every
 * later stage are identical to the serial pass. Only ordinary and shadow labelling are banded; split
 * labelling defers contacts in order and stays serial. Ordinary labelling's row pairs start on each tile
 * row's first row, so a band (whole tile rows) starts on a pair, and join_band_pairs joins it to the pair
 * above; each band writes only its own rows of the shared dark bits (level_bits). */

typedef struct napa_band napa_band;
struct napa_band {
    size_t at; /* the band's runs and parents start here in band_arenas[worker] */
    int worker, ty0, ty1, n, offset; /* offset: where the band's runs go in the level's runs */
};

typedef struct {
    napa_detector *det;
    int lw, lh, factor, shadow, tw, th;
    const int *first, *need; /* with the deferred rake: band k's rows lie in chunks [first[k], need[k]) */
} band_job;

/* Room in a for at least `runs` runs; doubling, so a thread's arena settles after a few frames. */
static int arena_reserve(band_arena *a, size_t runs) {
    if (runs > a->run_capacity) {
        size_t cap = a->run_capacity ? 2 * a->run_capacity : 4096;
        while (cap < runs) cap *= 2;
        void *p;
        if ((p = realloc(a->runs, cap * sizeof(napa_run))) == NULL) return -1;
        a->runs = (napa_run *)p;
        if ((p = realloc(a->parent, cap * sizeof(int32_t))) == NULL) return -1;
        a->parent = (int32_t *)p;
        a->run_capacity = cap;
    }
    return 0;
}

static void band_free(napa_detector *det) {
    for (int w = 0; w < NAPA_MAX_THREADS; w++) {
        band_arena *a = &det->band_arenas[w];
        free(a->runs), free(a->parent);
        scratch_free(&a->scratch);
    }
    free(det->bands);
    det->bands = NULL, det->band_count = 0;
}

/* Each band receives its thread's buffers (runs and parents after those of the thread's earlier bands), shared
 * read-only inputs and its own rows of level_bits. Split contact storage is absent; saved extrema are copied by band
 * below. */
static void band_label(void *arg, int k, int worker) {
    band_job *job = (band_job *)arg;
    napa_band *b = &job->det->bands[k];
    band_arena *a = &job->det->band_arenas[worker];
    /* At most half a row's samples (plus one) start runs, in each of the band's rows. */
    if (arena_reserve(a, a->used + (size_t)(b->ty1 - b->ty0) * TILE * ((size_t)job->lw / 2 + 1) + 1) != 0) {
        b->n = -1;
        return;
    }
    label_context ctx = label_context_for(job->det);
    b->worker = worker, b->at = a->used;
    ctx.runs = a->runs + b->at, ctx.parent = a->parent + b->at;
    ctx.scratch = &a->scratch;
    ctx.level_extrema_out = NULL;
    const int base = b->ty0 > 0 ? b->ty0 - 1 : 0;
    b->n = threshold_context(&ctx, NULL, job->lw, job->lh, job->factor, 0, job->shadow, b->ty0, b->ty1, base);
    if (b->n > 0) a->used += (size_t)b->n;
    uint8_t *out = job->det->level_extrema_out;
    if (out != NULL) { /* the band's own rows, saved for a later pass over this level (never set with saved input) */
        const size_t at = (size_t)b->ty0 * job->tw, count = (size_t)(b->ty1 - b->ty0) * job->tw;
        const size_t skip = (size_t)(b->ty0 - base) * job->tw, tile_count = (size_t)job->tw * job->th;
        memcpy(out + at, ctx.scratch->tile_min + skip, count);
        memcpy(out + tile_count + at, ctx.scratch->tile_max + skip, count);
    }
}

/* Band k's runs into the level's arrays at its offset, parents shifted with them. */
static void band_place(void *arg, int k, int worker) {
    (void)worker;
    band_job *job = (band_job *)arg;
    napa_detector *det = job->det;
    const napa_band *b = &det->bands[k];
    const band_arena *a = &det->band_arenas[b->worker];
    memcpy(det->runs + b->offset, a->runs + b->at, (size_t)b->n * sizeof(napa_run));
    int32_t *P = det->parent + b->offset;
    const int32_t *bp = a->parent + b->at;
    for (int i = 0; i < b->n; i++) {
        const int32_t p = bp[i];
        P[i] = p == i ? i + b->offset : P[p]; /* P[p], p < i, is already its root */
    }
}

static void invert_bytes(uint8_t *p, size_t n);
enum { RAKE_CHUNK = 32 }; /* rows the raker marks done at a time */

/* Rows [y0, y1) of the level that contrast_floors_raked left (all but every 8th), raked and inverted as it would
 * have. */
static void rake_rows(napa_detector *det, int lw, int y0, int y1) {
    for (int y = y0; y < y1; y++) {
        if (!(y & 7)) continue;
        uint8_t *row = det->level + (size_t)y * lw;
        gather_row(row, det->rake_img + (size_t)y * det->rake_s * det->rake_stride, lw, det->rake_s, det->ssse3);
        if (det->rake_invert) invert_bytes(row, (size_t)lw);
    }
}

typedef struct {
    napa_detector *det;
    int lw, lh, from;
} rake_rest_job;

static void rake_rest(void *arg, int k, int worker) {
    (void)worker;
    const rake_rest_job *j = (const rake_rest_job *)arg;
    k += j->from;
    const int y1 = (k + 1) * RAKE_CHUNK < j->lh ? (k + 1) * RAKE_CHUNK : j->lh;
    rake_rows(j->det, j->lw, k * RAKE_CHUNK, y1);
}

/* The deferred rows, before a threshold that isn't banded. */
static void finish_rake(napa_detector *det, int lw, int lh) {
    if (!det->rake_deferred) return;
    rake_rest_job j = {det, lw, lh, det->rake_next};
    pool_run(det->pool, rake_rest, &j, (lh + RAKE_CHUNK - 1) / RAKE_CHUNK - det->rake_next);
    det->rake_deferred = 0;
}

/* Index 0 rakes the deferred rows chunk by chunk, in order, from where the floors' run left off; index k + 1 labels
 * band k once its chunks are raked. */
static void rake_band_job(void *arg, int i, int worker) {
    band_job *job = (band_job *)arg;
    napa_detector *det = job->det;
    if (i == 0) {
        for (int c = det->rake_next; c * RAKE_CHUNK < job->lh; c++) {
            const int y1 = (c + 1) * RAKE_CHUNK < job->lh ? (c + 1) * RAKE_CHUNK : job->lh;
            rake_rows(det, job->lw, c * RAKE_CHUNK, y1);
            pool_mark(det->pool, &det->rake_done[c]);
        }
        return;
    }
    pool_await(det->pool, det->rake_done, job->first[i - 1], job->need[i - 1]);
    band_label(arg, i - 1, worker);
}

static int threshold_banded(napa_detector *det, int lw, int lh, int factor, int shadow) {
    const int tw = (lw + TILE - 1) / TILE, th = (lh + TILE - 1) / TILE;
    /* A few bands per thread even out uneven rows; each keeps at least four tile rows. */
    int bands = det->threads * 3;
    if (bands > th / 4) bands = th / 4;
    if (bands < 2) {
        finish_rake(det, lw, lh);
        return threshold_rows(det, lw, lh, factor, 0, shadow, 0, th, 0);
    }
    if (bands > det->band_count) {
        napa_band *p = (napa_band *)realloc(det->bands, (size_t)bands * sizeof(napa_band));
        if (p == NULL) return -1;
        memset(p + det->band_count, 0, (size_t)(bands - det->band_count) * sizeof(napa_band));
        det->bands = p, det->band_count = bands;
    }
    /* Guided heights: band k gets (1.8 - 1.6 k / (bands - 1)) even shares, in whole tile rows, and at least four. */
    int edge[NAPA_MAX_THREADS * 3 + 1];
    edge[0] = 0, edge[bands] = th;
    for (int k = 1; k < bands; k++) {
        /* shares before band k: 1.8 k - 0.8 k (k - 1) / (bands - 1), of bands in all */
        const double before = 1.8 * k - 0.8 * k * (k - 1) / (bands - 1);
        edge[k] = (int)(th * before / bands + 0.5);
        if (edge[k] < edge[k - 1] + 4) edge[k] = edge[k - 1] + 4;
    }
    for (int k = bands - 1; k > 0; k--)
        if (edge[k] > edge[k + 1] - 4) edge[k] = edge[k + 1] - 4;
    int most = 0; /* the tallest band's tile rows */
    for (int k = 0; k < bands; k++) {
        napa_band *b = &det->bands[k];
        b->ty0 = edge[k], b->ty1 = edge[k + 1];
        if (b->ty1 - b->ty0 > most) most = b->ty1 - b->ty0;
    }
    for (int w = 0; w < det->threads; w++) { /* scratch: a band's tile rows and one on each side */
        det->band_arenas[w].used = 0;
        if (scratch_reserve(&det->band_arenas[w].scratch, (size_t)lw, (size_t)(most + 2) * tw) != 0) return -1;
    }
    band_job job = {det, lw, lh, factor, shadow, tw, th, NULL, NULL};
    if (det->rake_deferred) {
        const int chunks = (lh + RAKE_CHUNK - 1) / RAKE_CHUNK;
        int first[NAPA_MAX_THREADS * 3], need[NAPA_MAX_THREADS * 3];
        for (int k = 0; k < bands; k++) {
            /* The rows a band reads: its tile rows and one either side (tile extrema), and a tile row more. */
            const int r0 = (det->bands[k].ty0 - 1) * TILE, r1 = (det->bands[k].ty1 + 2) * TILE;
            first[k] = r0 > 0 ? r0 / RAKE_CHUNK : 0;
            need[k] = r1 < lh ? (r1 + RAKE_CHUNK - 1) / RAKE_CHUNK : chunks;
        }
        job.first = first, job.need = need;
        pool_run(det->pool, rake_band_job, &job, bands + 1);
        det->rake_deferred = 0;
    } else pool_run(det->pool, band_label, &job, bands);

    int n = 0;
    det->diag_n = det->neck_n = 0;
    for (int k = 0; k < bands; k++) {
        if (det->bands[k].n < 0) return -1;
        det->bands[k].offset = n;
        n += det->bands[k].n;
    }
    pool_run(det->pool, band_place, &job, bands);
    n = 0;
    label_context ctx = label_context_for(det);
    for (int k = 0; k < bands; k++) {
        const napa_band *b = &det->bands[k];
        const int y = b->ty0 * TILE;
        if (k > 0 && !shadow) join_band_pairs(&ctx, lw, lh, factor, y, n, n + b->n);
        else if (k > 0) join_band_rows(&ctx, y, n, n + b->n);
        n += b->n;
    }
    return n;
}

static int threshold_level(napa_detector *det, int lw, int lh, int factor, int split, int shadow) {
    if (!split && pool_for(det, (size_t)lw * lh)) return threshold_banded(det, lw, lh, factor, shadow);
    finish_rake(det, lw, lh);
    return threshold_rows(det, lw, lh, factor, split, shadow, 0, (lh + TILE - 1) / TILE, 0);
}


/* Bilinear profiles at 16.16 positions, returned at 40.24 precision. The caller checks image bounds and pads
 * prof to a multiple of four: the SIMD path writes all four lanes of the final group, even past K. */
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
#elif defined(__aarch64__) && defined(__AARCH64EL__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    /* The scalar formula four samples at a time, all in exact integer arithmetic. Byte pairs move
     * to the NEON registers as two packed words (little-endian: low byte left) instead of through
     * memory, and positions are running sums, as in the SSE2 path above. */
    static const int32_t lanes[4] = {0, 1, 2, 3};
    const int32x4_t frac = vdupq_n_s32(4095);
    int32x4_t px = vaddq_s32(vdupq_n_s32(fxp), vmulq_s32(vdupq_n_s32(ixp), vld1q_s32(lanes)));
    int32x4_t py = vaddq_s32(vdupq_n_s32(fyp), vmulq_s32(vdupq_n_s32(iyp), vld1q_s32(lanes)));
    const int32x4_t px4 = vdupq_n_s32(4 * ixp), py4 = vdupq_n_s32(4 * iyp);
    int j = 0;
    for (; j + 4 <= K; j += 4, px = vaddq_s32(px, px4), py = vaddq_s32(py, py4)) {
        uint64_t u = 0, d = 0;
        for (int k = 0; k < 4; k++, fxp += ixp, fyp += iyp) {
            const uint8_t *q = img + (size_t)(fyp >> 16) * stride + (fxp >> 16);
            uint16_t a, b;
            memcpy(&a, q, 2), memcpy(&b, q + stride, 2);
            u |= (uint64_t)a << (16 * k), d |= (uint64_t)b << (16 * k);
        }
        const uint16x8_t ub = vmovl_u8(vreinterpret_u8_u64(vcreate_u64(u)));
        const uint16x8_t db = vmovl_u8(vreinterpret_u8_u64(vcreate_u64(d)));
        /* Unzip the interleaved bytes: even lanes are each pair's left pixel, odd its right. */
        const int32x4_t a0 = vreinterpretq_s32_u32(vmovl_u16(vuzp1_u16(vget_low_u16(ub), vget_high_u16(ub))));
        const int32x4_t a1 = vreinterpretq_s32_u32(vmovl_u16(vuzp2_u16(vget_low_u16(ub), vget_high_u16(ub))));
        const int32x4_t b0 = vreinterpretq_s32_u32(vmovl_u16(vuzp1_u16(vget_low_u16(db), vget_high_u16(db))));
        const int32x4_t b1 = vreinterpretq_s32_u32(vmovl_u16(vuzp2_u16(vget_low_u16(db), vget_high_u16(db))));
        const int32x4_t fx = vandq_s32(vshrq_n_s32(px, 4), frac), fy = vandq_s32(vshrq_n_s32(py, 4), frac);
        const int32x4_t top = vaddq_s32(vshlq_n_s32(a0, 12), vmulq_s32(vsubq_s32(a1, a0), fx));
        const int32x4_t bot = vaddq_s32(vshlq_n_s32(b0, 12), vmulq_s32(vsubq_s32(b1, b0), fx));
        const int32x4_t diff = vsubq_s32(bot, top);
        vst1q_s64(prof + j, vaddq_s64(vshlq_n_s64(vmovl_s32(vget_low_s32(top)), 12),
                                      vmull_s32(vget_low_s32(diff), vget_low_s32(fy))));
        vst1q_s64(prof + j + 2, vaddq_s64(vshlq_n_s64(vmovl_s32(vget_high_s32(top)), 12),
                                          vmull_s32(vget_high_s32(diff), vget_high_s32(fy))));
    }
    for (; j < K; j++, fxp += ixp, fyp += iyp) {
        int x0 = fxp >> 16, y0 = fyp >> 16;
        int32_t fx = (fxp >> 4) & 4095, fy = (fyp >> 4) & 4095;
        const uint8_t *q = img + (size_t)y0 * stride + x0;
        int32_t top = (q[0] << 12) + (q[1] - q[0]) * fx;
        int32_t bot = (q[stride] << 12) + (q[stride + 1] - q[stride]) * fx;
        prof[j] = ((int64_t)top << 12) + (int64_t)(bot - top) * fy;
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

/* No edge along side a-b: if it lies on the frame edge, the black border runs to the edge of
 * the image (the camera cut off the white margin, or part of the border), so the dark region ends at the image
 * boundary. Take that boundary as the side for now (its line in *lx, *ly, *dx, *dy); napa_detect measures where
 * the cut-off corners really are (extend_cut_sides). A tilted side can show part of its edge to a later,
 * narrower pass, so fit_sides keeps the mark for the whole fit. Returns 0 if the side isn't on the frame edge.
 * Rare, so kept out of fit_sides' loop. */
static NAPA_NOINLINE int frame_side(const image_view *im, vec2 a, vec2 b, double r_out, double *lx, double *ly,
                                    double *dx, double *dy) {
    double tol = r_out + 1.0, right = im->w - 0.5, bottom = im->h - 0.5;
    if (fabs(b.x - a.x) < fabs(b.y - a.y)) { /* a vertical side: the left or right edge */
        if (fabs(a.x + 0.5) <= tol && fabs(b.x + 0.5) <= tol) *lx = -0.5;
        else if (fabs(a.x - right) <= tol && fabs(b.x - right) <= tol) *lx = right;
        else return 0;
        *ly = a.y, *dx = 0, *dy = 1;
    } else {
        if (fabs(a.y + 0.5) <= tol && fabs(b.y + 0.5) <= tol) *ly = -0.5;
        else if (fabs(a.y - bottom) <= tol && fabs(b.y - bottom) <= tol) *ly = bottom;
        else return 0;
        *lx = a.x, *dx = 1, *dy = 0;
    }
    return 1;
}

/* Edge levels for the midpoint-crossing estimator; black >= white disables it. */
typedef struct {
    double black, white;
    double midpoint[3], independent_midpoint[3], cx, cy, scale;
    int has_model, model_pending;
    const family_layout *model_family;
} edge_levels;

/* A whole set aside for the terminal replay, with the stride and fit policy of the level that found it. */
struct fine_whole {
    component shape;
    int stride, terminal;
};

/* Keep the rare illumination fit's sample arrays off the hot refinement stack. */
static NAPA_NOINLINE void refresh_edge_light(const image_view *im, const vec2 q[4], edge_levels *levels, int visible_only);

enum { INITIAL_SIDE_POINTS = 6 };

static NAPA_FORCE_INLINE int fit_side_line(double px[], double py[], double wt[], int good, double reach,
                     double *lx, double *ly, double *dx, double *dy) {
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
                if (dist <= 1.0 + 0.1 * reach) kept++;
            }
            /* Keep the original fit and samples unless enough inliers support a refit. */
            if (kept == good || kept < 3 || 2 * kept < good) break;
            kept = 0;
            for (int i = 0; i < good; i++) {
                double dist = fabs((px[i] - mx) * (-(*dy)) + (py[i] - my) * (*dx));
                if (dist <= 1.0 + 0.1 * reach) {
                    px[kept] = px[i];
                    py[kept] = py[i];
                    wt[kept] = wt[i];
                    kept++;
                }
            }
            good = kept;
        }
    }
    return 1;
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
    s->len = vec_length(b.x - a.x, b.y - a.y);
    if (n < 12 || !(s->len >= 6)) return;
    s->ex = (b.x - a.x) / s->len;
    s->ey = (b.y - a.y) / s->len;
    s->nx = -s->ey;
    s->ny = s->ex;
    double m[5] = {0}, v[3] = {0}, half[2] = {0};
    double us[NAPA_SIDE_SAMPLES], ys[NAPA_SIDE_SAMPLES]; /* the caller's px holds at most this many */
    const int kept = n <= NAPA_SIDE_SAMPLES;
    for (int i = 0; i < n; i++) {
        double u = ((px[i] - a.x) * s->ex + (py[i] - a.y) * s->ey) / s->len - 0.5;
        double y = (px[i] - a.x) * s->nx + (py[i] - a.y) * s->ny;
        if (!isfinite(u) || !isfinite(y) || !(wt[i] > 0) || !isfinite(wt[i])) return;
        if (kept) us[i] = u, ys[i] = y;
        half[u >= 0] += wt[i];
        const double w = wt[i], wy = w * y, u2 = u * u, u3 = u2 * u, u4 = u3 * u;
        m[0] += w, m[1] += w * u, m[2] += w * u2, m[3] += w * u3, m[4] += w * u4;
        v[0] += wy, v[1] += wy * u, v[2] += wy * u2;
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
        double u = kept ? us[i] : ((px[i] - a.x) * s->ex + (py[i] - a.y) * s->ey) / s->len - 0.5;
        double y = kept ? ys[i] : (px[i] - a.x) * s->nx + (py[i] - a.y) * s->ny;
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
        if (!(vec_length(b.x - a.x, b.y - a.y) >= 6) || !isfinite(cross) || cross * original <= 0 ||
            (k && cross * sign <= 0)) return 0;
        sign = cross;
    }
    return 1;
}

static inline int inside_frame(const image_view *view, vec2 p) {
    return p.x >= 1.0 && p.y >= 1.0 && p.x <= view->w - 2.0 && p.y <= view->h - 2.0;
}

/* Observation domain in pixel-center coordinates, independent of the sampler's address guard. */
static inline int sample_is_observed(const image_view *im, double x, double y) {
    return x >= 0 && y >= 0 && x <= im->w - 1 && y <= im->h - 1;
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

/* Weighted total-least-squares line through one side's samples moved by the radial map o (1 + kappa |o|^2 / scale^2)
 * about (cx, cy); returns the weighted squared residual in pixels. The map stretches a sample's offset from the line by
 * |J n|, its Jacobian along the line's normal, so with kappa a second pass divides each weight by that squared, using
 * the first pass's normal. Only a first-order measure: kappa must keep the map well clear of folding at the samples. */
static double side_line(const double *x, const double *y, const double *w, int n, double kappa, double cx, double cy,
                        double scale, double *mx, double *my, double *dx, double *dy) {
    double ax = 0, ay = 0, angle = 0, least = 0;
    for (int pass = 0; pass < (kappa != 0 ? 2 : 1); pass++) {
        const double nx = -sin(angle), ny = cos(angle);
        double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        for (int i = 0; i < n; i++) {
            const double ox = x[i] - cx, oy = y[i] - cy, grow = 1 + kappa * (ox * ox + oy * oy) / (scale * scale);
            const double px = ox * grow, py = oy * grow, along = 2 * kappa * (ox * nx + oy * ny) / (scale * scale);
            const double jx = grow * nx + along * ox, jy = grow * ny + along * oy, v = pass ? w[i] / (jx * jx + jy * jy) : w[i];
            sw += v, sx += v * px, sy += v * py;
            sxx += v * px * px, sxy += v * px * py, syy += v * py * py;
        }
        ax = sx / sw, ay = sy / sw;
        const double cxx = sxx - sw * ax * ax, cxy = sxy - sw * ax * ay, cyy = syy - sw * ay * ay;
        const double spread = sqrt((cxx - cyy) * (cxx - cyy) + 4 * cxy * cxy);
        least = 0.5 * (cxx + cyy - spread), angle = 0.5 * atan2(2 * cxy, cxx - cyy);
    }
    if (mx != NULL) *mx = ax + cx, *my = ay + cy, *dx = cos(angle), *dy = sin(angle);
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
static int shared_lens_quad(const image_view *im, side_samples *s, const int side_clear[4], const vec2 in[4], double reach,
                            vec2 out[4]) {
    const double cx = 0.5 * im->w, cy = 0.5 * im->h, scale = 0.5 * (im->w > im->h ? im->w : im->h);
    double sse_quadratic = 0;
    for (int k = 0; k < 4; k++) {
        double mx, my, dx, dy, e[NAPA_SIDE_SAMPLES], sorted[NAPA_SIDE_SAMPLES];
        int kept = 0;
        for (int i = 0; i < s->n[k]; i++)
            if (s->w[k][i] > 0) s->x[k][kept] = s->x[k][i], s->y[k][kept] = s->y[k][i], s->w[k][kept] = s->w[k][i], kept++;
        if (kept < NAPA_LENS_MIN_SIDE_POINTS) return 0;
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
        if (n < NAPA_LENS_MIN_SIDE_POINTS || !side_quadratic_residual(s->x[k], s->y[k], s->w[k], n, &sse_quadratic)) return 0;
        s->n[k] = n;
    }
    /* A lens maps the whole frame one to one, so 1 + 3 kappa r^2 > 0 out to its corners. */
    enum { STEPS = 41 };
    const double corner2 = (cx * cx + cy * cy) / (scale * scale);
    double cost[STEPS];
    int best = 0;
    for (int g = 0; g < STEPS; g++) {
        const double kappa = -0.4 + 0.02 * g;
        cost[g] = HUGE_VAL;
        if (!(1 + 3 * kappa * corner2 > 0)) continue;
        cost[g] = 0;
        for (int k = 0; k < 4; k++) cost[g] += side_line(s->x[k], s->y[k], s->w[k], s->n[k], kappa, cx, cy, scale, NULL, NULL, NULL, NULL);
        if (cost[g] < cost[best]) best = g;
    }
    /* A least cost at the end of the range, or at the fold, locates no lens: it may lie beyond. */
    if (best == 0 || best == STEPS - 1 || !(cost[best - 1] < HUGE_VAL)) return 0;
    const double curve = cost[best - 1] - 2 * cost[best] + cost[best + 1];
    double kappa = -0.4 + 0.02 * best;
    if (curve > 0) kappa += 0.5 * (cost[best - 1] - cost[best + 1]) / curve * 0.02;
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
    const int samples = s->n[0] + s->n[1] + s->n[2] + s->n[3];
    if (!(fabs(kappa) >= NAPA_LENS_MIN_KAPPA) || !(sse <= (1 - NAPA_LENS_MIN_GAIN) * cost[20]) || !(sse <= NAPA_LENS_MAX_EXCESS * sse_quadratic) ||
        !((cost[20] - sse) * (samples - 9) >= NAPA_LENS_MIN_F * sse) || !(clear_straight > 0) ||
        !(clear_bent <= (1 - NAPA_LENS_MIN_GAIN) * clear_straight)) return 0;
    for (int k = 0; k < 4; k++) {
        const int p = (k + 3) & 3;
        const double cross = dx[p] * dy[k] - dy[p] * dx[k];
        if (fabs(cross) < 0.2) return 0;
        const double t = ((mx[k] - mx[p]) * dy[k] - (my[k] - my[p]) * dx[k]) / cross;
        /* Back through the lens: Newton on r (1 + kappa r^2 / scale^2) = |q| from r = |q| approaches the root from one
         * side while the map stays one to one; a corner past the fold has no preimage. */
        const double qx = mx[p] + t * dx[p] - cx, qy = my[p] + t * dy[p] - cy, rq = vec_length(qx, qy);
        double r = rq, step = HUGE_VAL;
        for (int iter = 0; iter < 30 && !(fabs(step) <= 1e-9 * (1 + rq)); iter++) {
            const double slope = 1 + 3 * kappa * r * r / (scale * scale);
            if (!(slope > 0)) return 0;
            step = (r * (1 + kappa * r * r / (scale * scale)) - rq) / slope, r -= step;
        }
        if (!(fabs(step) <= 1e-9 * (1 + rq)) || !(1 + 3 * kappa * r * r / (scale * scale) > 0)) return 0;
        const double ratio = rq > 0 ? r / rq : 1;
        out[k].x = cx + qx * ratio, out[k].y = cy + qy * ratio;
        if (!(fabs(out[k].x - in[k].x) + fabs(out[k].y - in[k].y) <= 4.0 * reach + 2.0)) return 0;
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

/* ---------------------------------------------------------------- lens */

/* Brown-Conrady distortion of the normalized point (a, b): the distorted point (x, y) and, with J, its derivatives
 * (J[0]: dx/da, dx/db; J[1]: dy/da, dy/db). Returns 0 where the rational terms' denominator isn't positive.
 * The flags only skip work: napa_set_lens sets rational or prism whenever those coefficients aren't all zero,
 * so the Jacobian's prism terms, left ungated, are zero without them. */
static NAPA_FORCE_INLINE int bc_distort(const napa_lens *L, double a, double b, double *x, double *y, double J[2][2]) {
    const double *k = L->k, r2 = a * a + b * b;
    double num = 1 + r2 * (k[0] + r2 * (k[1] + r2 * k[4])), den = 1, radial = num;
    if (L->rational) den = 1 + r2 * (k[5] + r2 * (k[6] + r2 * k[7])), radial = num / den;
    *x = a * radial + 2 * k[2] * a * b + k[3] * (r2 + 2 * a * a);
    *y = b * radial + k[2] * (r2 + 2 * b * b) + 2 * k[3] * a * b;
    if (L->prism) *x += r2 * (k[8] + r2 * k[9]), *y += r2 * (k[10] + r2 * k[11]);
    if (J != NULL) {
        const double dnum = k[0] + r2 * (2 * k[1] + 3 * r2 * k[4]), dden = k[5] + r2 * (2 * k[6] + 3 * r2 * k[7]);
        const double dR = L->rational ? (dnum * den - num * dden) / (den * den) : dnum; /* d radial / d r2 */
        const double dsu = k[8] + 2 * r2 * k[9], dsv = k[10] + 2 * r2 * k[11];
        J[0][0] = radial + 2 * a * a * dR + 2 * k[2] * b + 6 * k[3] * a + 2 * a * dsu;
        J[0][1] = 2 * a * b * dR + 2 * k[2] * a + 2 * k[3] * b + 2 * b * dsu;
        J[1][0] = 2 * a * b * dR + 2 * k[2] * a + 2 * k[3] * b + 2 * a * dsv;
        J[1][1] = radial + 2 * b * b * dR + 6 * k[2] * b + 2 * k[3] * a + 2 * b * dsv;
    }
    return den > 0;
}

/* The pixel where the lens images a camera ray (any length, pointing into the scene); 0 if it can't. With D, also the
 * pixel's derivatives by the ray (d pixel x / d r, then d pixel y / d r). */
static NAPA_FORCE_INLINE int lens_project_d(const napa_lens *L, const double r[3], vec2 *p, double D[2][3]) {
    const double *k = L->k;
    double x = 0, y = 0;
    if (L->model == NAPA_LENS_FISHEYE) {
        const double rho = sqrt(r[0] * r[0] + r[1] * r[1]);
        double g, dg[3];
        if (D != NULL && r[2] > 0 && rho < 1e-4 * r[2]) {
            /* Near the axis, g = td / rho = (1 + (k1 - 1/3) rho^2 / z^2) / z to within (rho / z)^4. */
            const double iz = 1 / r[2], c1 = k[0] - 1.0 / 3, u = rho * rho * iz * iz;
            g = iz * (1 + c1 * u);
            dg[0] = 2 * c1 * r[0] * iz * iz * iz, dg[1] = 2 * c1 * r[1] * iz * iz * iz, dg[2] = -iz * iz * (1 + 3 * c1 * u);
        } else if (rho > 0) {
            const double t = atan2(rho, r[2]), t2 = t * t;
            const double td = t * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
            g = td / rho;
            if (D != NULL) {
                /* theta = atan2(rho, z): d theta = (z d rho - rho dz) / (rho^2 + z^2) */
                const double dtd = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
                const double n2 = rho * rho + r[2] * r[2], a = dtd / (rho * rho * n2), c = td / (rho * rho * rho);
                dg[0] = r[0] * (a * r[2] - c), dg[1] = r[1] * (a * r[2] - c), dg[2] = -a * rho * rho;
            }
        } else if (!(r[2] > 0) || D != NULL) {
            return 0;
        } else {
            g = 0, dg[0] = dg[1] = dg[2] = 0;
        }
        x = g * r[0], y = g * r[1];
        if (D != NULL)
            for (int j = 0; j < 3; j++) {
                const double dx = r[0] * dg[j] + (j == 0 ? g : 0), dy = r[1] * dg[j] + (j == 1 ? g : 0);
                D[0][j] = L->fx * dx + L->skew * dy, D[1][j] = L->fy * dy;
            }
    } else {
        if (!(r[2] > 0)) return 0;
        const double iz = 1 / r[2], a = r[0] * iz, b = r[1] * iz;
        if (!(a * a + b * b < 1e12)) return 0;
        double J[2][2];
        /* Projection keeps OpenCV's value even where the rational terms' denominator isn't positive; only lens_ray,
         * inverting the lens, rejects such points. */
        (void)bc_distort(L, a, b, &x, &y, D != NULL ? J : NULL);
        if (D != NULL) {
            /* The derivatives by (a, b), then (a, b) by the ray. */
            const double px_a = L->fx * J[0][0] + L->skew * J[1][0], px_b = L->fx * J[0][1] + L->skew * J[1][1];
            const double py_a = L->fy * J[1][0], py_b = L->fy * J[1][1];
            D[0][0] = px_a * iz, D[0][1] = px_b * iz, D[0][2] = -(px_a * a + px_b * b) * iz;
            D[1][0] = py_a * iz, D[1][1] = py_b * iz, D[1][2] = -(py_a * a + py_b * b) * iz;
        }
    }
    p->x = L->fx * x + L->skew * y + L->cx, p->y = L->fy * y + L->cy;
    return isfinite(p->x) && isfinite(p->y);
}

static int lens_project(const napa_lens *L, const double r[3], vec2 *p) { return lens_project_d(L, r, p, NULL); }

/* The camera ray a pixel sees, by Newton's method from the distorted point; 0 if it doesn't settle or the lens folds there. */
static int lens_ray(const napa_lens *L, vec2 p, double r[3]) {
    const double *k = L->k, yd = (p.y - L->cy) / L->fy, xd = (p.x - L->cx - L->skew * yd) / L->fx;
    if (L->model == NAPA_LENS_FISHEYE) {
        const double td = vec_length(xd, yd);
        double t = td < 3.0 ? td : 3.0, f = 0;
        if (!(td > 1e-12)) {
            r[0] = 0, r[1] = 0, r[2] = 1;
            return td >= 0;
        }
        for (int i = 0; i < 30; i++) {
            const double t2 = t * t;
            f = t * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - td;
            const double g = 1 + t2 * (3 * k[0] + t2 * (5 * k[1] + t2 * (7 * k[2] + t2 * 9 * k[3])));
            if (!(g > 0)) return 0;
            const double step = f / g;
            t -= step;
            if (!(t > 0 && t < 3.14159265358979)) return 0;
            if (fabs(step) < 1e-14) break;
        }
        if (!(fabs(f) < 1e-9)) return 0;
        const double s = sin(t) / td;
        r[0] = xd * s, r[1] = yd * s, r[2] = cos(t);
        return 1;
    }
    double x = xd, y = yd;
    for (int i = 0; i < 30; i++) {
        double u, v, J[2][2];
        if (!bc_distort(L, x, y, &u, &v, J)) return 0;
        u -= xd, v -= yd;
        const double ux = J[0][0], uy = J[0][1], vx = J[1][0], vy = J[1][1], det = ux * vy - uy * vx;
        if (!(det > 0)) return 0;
        const double sx = (u * vy - v * uy) / det, sy = (v * ux - u * vx) / det;
        x -= sx, y -= sy;
        if (!isfinite(x) || !isfinite(y)) return 0;
        if (fabs(sx) + fabs(sy) < 1e-14) break;
        if (i == 29 && !(fabs(u) + fabs(v) < 1e-9)) return 0;
    }
    r[0] = x, r[1] = y, r[2] = 1;
    return 1;
}

static int lens_to_plane(const lens_frame *f, const double r[3], vec2 *q) {
    const double z = r[0] * f->e[2][0] + r[1] * f->e[2][1] + r[2] * f->e[2][2];
    if (!(z > 0.05 * sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]))) return 0;
    q->x = f->scale * (r[0] * f->e[0][0] + r[1] * f->e[0][1] + r[2] * f->e[0][2]) / z;
    q->y = f->scale * (r[0] * f->e[1][0] + r[1] * f->e[1][1] + r[2] * f->e[1][2]) / z;
    return 1;
}

static int lens_from_plane(const lens_frame *f, double x, double y, vec2 *p) {
    double r[3];
    for (int i = 0; i < 3; i++) r[i] = f->e[2][i] + (x * f->e[0][i] + y * f->e[1][i]) * f->inv_scale;
    return lens_project(f->L, r, p);
}

/* The same with the pixel's derivatives by the plane: x by x, x by y, y by x, y by y. */
static int lens_from_plane_d(const lens_frame *f, double x, double y, vec2 *p, double J[4]) {
    double r[3], D[2][3];
    for (int i = 0; i < 3; i++) r[i] = f->e[2][i] + (x * f->e[0][i] + y * f->e[1][i]) * f->inv_scale;
    if (!lens_project_d(f->L, r, p, D)) return 0;
    for (int i = 0; i < 2; i++) {
        J[2 * i] = (D[i][0] * f->e[0][0] + D[i][1] * f->e[0][1] + D[i][2] * f->e[0][2]) * f->inv_scale;
        J[2 * i + 1] = (D[i][0] * f->e[1][0] + D[i][1] * f->e[1][1] + D[i][2] * f->e[1][2]) * f->inv_scale;
    }
    return 1;
}

static int lens_frame_build(const napa_lens *L, const vec2 in[4], lens_frame *f, vec2 plane[4]) {
    double ray[4][3], m[3] = {0, 0, 0};
    for (int k = 0; k < 4; k++) {
        if (!lens_ray(L, in[k], ray[k])) return 0;
        const double n = sqrt(ray[k][0] * ray[k][0] + ray[k][1] * ray[k][1] + ray[k][2] * ray[k][2]);
        for (int i = 0; i < 3; i++) m[i] += ray[k][i] / n;
    }
    double n = sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
    if (!(n > 0)) return 0;
    for (int i = 0; i < 3; i++) f->e[2][i] = m[i] / n;
    const double *z = f->e[2];
    double x[3] = {1 - z[0] * z[0], -z[0] * z[1], -z[0] * z[2]};
    n = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    if (!(n > 0.1)) return 0;
    for (int i = 0; i < 3; i++) f->e[0][i] = x[i] / n;
    f->e[1][0] = z[1] * f->e[0][2] - z[2] * f->e[0][1];
    f->e[1][1] = z[2] * f->e[0][0] - z[0] * f->e[0][2];
    f->e[1][2] = z[0] * f->e[0][1] - z[1] * f->e[0][0];
    f->L = L, f->scale = 1, f->inv_scale = 1;
    vec2 c;
    double J[4];
    if (!lens_from_plane_d(f, 0, 0, &c, J)) return 0;
    const double area = J[0] * J[3] - J[1] * J[2];
    if (!(area > 0)) return 0;
    f->scale = sqrt(area), f->inv_scale = 1 / f->scale;
    for (int k = 0; k < 4; k++)
        if (!lens_to_plane(f, ray[k], &plane[k])) return 0;
    return 1;
}

/* How far the lens bows the sides between the corners `in` (pixels, at the middle of the side); -1 if it can't image one. */
static double lens_bow(const lens_frame *f, const vec2 in[4], const vec2 plane[4]) {
    double bow = 0;
    for (int k = 0; k < 4; k++) {
        const vec2 a = plane[k], b = plane[(k + 1) & 3];
        vec2 m;
        if (!lens_from_plane(f, 0.5 * (a.x + b.x), 0.5 * (a.y + b.y), &m)) return -1;
        const double d = vec_length(m.x - 0.5 * (in[k].x + in[(k + 1) & 3].x), m.y - 0.5 * (in[k].y + in[(k + 1) & 3].y));
        if (d > bow) bow = d;
    }
    return bow;
}

static int same_corners(const vec2 a[4], const vec2 b[4]) { return memcmp(a, b, 4 * sizeof(vec2)) == 0; }

/* The tag view for corners `in`, with their places in its plane. Returns 0 where the lens leaves every side straight to
 * within `straight` pixels: pixels serve as well there and cost less. The detector's memo answers the questions a
 * candidate's steps ask in turn: the same corners again, the corners the last lens fit returned (already in its plane),
 * and nearly the same corners where the sides were far straighter than this asks. */
static int lens_frame_init(const napa_lens *L, const vec2 in[4], lens_frame *f, vec2 plane[4], double straight) {
    lens_memo *m = L->memo;
    if (m != NULL && m->valid) {
        if (same_corners(in, m->key)) {
            if (!(m->bow >= straight)) return 0;
            *f = m->f;
            memcpy(plane, m->plane, sizeof m->plane);
            return 1;
        }
        if (m->chained && same_corners(in, m->next)) {
            const double bow = lens_bow(&m->f, in, m->next_plane);
            if (bow < 0) {
                m->valid = 0;
                return 0;
            }
            /* The fit's result becomes the memo's key, in the same view; quad_map_init's departure is redone. */
            memcpy(m->key, in, sizeof m->key);
            memcpy(m->plane, m->next_plane, sizeof m->plane);
            m->bow = bow;
            m->chained = 0;
            m->departure = -1;
            if (!(bow >= straight)) return 0;
            *f = m->f;
            memcpy(plane, m->plane, sizeof m->plane);
            return 1;
        }
        if (m->bow < 0.5 * straight) {
            double moved = 0;
            for (int k = 0; k < 4; k++) moved = fmax(moved, fabs(in[k].x - m->key[k].x) + fabs(in[k].y - m->key[k].y));
            if (moved < NAPA_LENS_MEMO_NEAR_PX) return 0;
        }
    }
    lens_frame g;
    vec2 q[4];
    const int built = lens_frame_build(L, in, &g, q);
    const double bow = built ? lens_bow(&g, in, q) : -1;
    if (m != NULL) {
        m->valid = bow >= 0;
        m->chained = 0;
        if (m->valid) {
            memcpy(m->key, in, sizeof m->key);
            memcpy(m->plane, q, sizeof q);
            m->bow = bow;
            m->f = g;
            m->departure = -1;
        }
    }
    if (!(bow >= straight)) return 0;
    *f = g;
    memcpy(plane, q, sizeof q);
    return 1;
}

/* After a lens fit from corners `in` in view f: its result, already in f's plane, for the next step to start from. */
static void lens_memo_chain(const lens_frame *f, const vec2 in[4], const vec2 out[4], const vec2 plane_out[4]) {
    lens_memo *m = f->L->memo;
    if (m == NULL || !m->valid || !same_corners(in, m->key)) return;
    memcpy(m->next, out, sizeof m->next);
    memcpy(m->next_plane, plane_out, sizeof m->next_plane);
    m->chained = 1;
}

/* One side as the lens draws it: at knots about NAPA_LENS_KNOT_PX apart, the pixel, its rate along the side, the side's
 * pixel normal there (oriented like the caller's), and the plane offset per pixel moved along that normal. Profiles
 * between knots interpolate them (cubic Hermite for the position), which costs far less than projecting every profile and
 * is exact to well under a thousandth of a pixel. */
enum { NAPA_LENS_KNOTS = 33 };
typedef struct {
    int m;
    vec2 p[NAPA_LENS_KNOTS], dp[NAPA_LENS_KNOTS];
    double n[NAPA_LENS_KNOTS][4]; /* normal x, y; plane offset x, y per pixel along it */
    unsigned char ok[NAPA_LENS_KNOTS]; /* 0 where the lens folds */
    double sx, sy;                /* the side's unit direction in the plane */
} lens_side;

static NAPA_NOINLINE int lens_side_init(const lens_frame *f, vec2 a, vec2 b, double pixel_len, double cnx, double cny, lens_side *ls) {
    const double ex = b.x - a.x, ey = b.y - a.y, len = vec_length(ex, ey);
    int m = (int)(pixel_len / NAPA_LENS_KNOT_PX) + 2;
    if (m > NAPA_LENS_KNOTS) m = NAPA_LENS_KNOTS;
    if (!(len > 0)) return 0;
    ls->m = m, ls->sx = ex / len, ls->sy = ey / len;
    for (int i = 0; i < m; i++) {
        const double t = (double)i / (m - 1);
        double J[4];
        if (!lens_from_plane_d(f, a.x + ex * t, a.y + ey * t, &ls->p[i], J)) return 0;
        ls->dp[i].x = J[0] * ex + J[1] * ey, ls->dp[i].y = J[2] * ex + J[3] * ey;
        /* Square to the drawn side, and back to the plane through the inverse of J. */
        const double det = J[0] * J[3] - J[1] * J[2], tx = J[0] * ls->sx + J[1] * ls->sy, ty = J[2] * ls->sx + J[3] * ls->sy;
        const double tl = sqrt(tx * tx + ty * ty);
        ls->ok[i] = det > 0 && tl > 0;
        if (!ls->ok[i]) continue;
        double mx = -ty / tl, my = tx / tl;
        if (mx * cnx + my * cny < 0) mx = -mx, my = -my;
        ls->n[i][0] = mx, ls->n[i][1] = my, ls->n[i][2] = (J[3] * mx - J[1] * my) / det, ls->n[i][3] = (J[0] * my - J[2] * mx) / det;
    }
    return 1;
}

/* A profile across the side at t (0 to 1 along it): the pixel there, the side's pixel normal there, and the plane offset
 * per pixel moved along that normal. */
static inline int lens_profile(const lens_side *ls, double t, vec2 *b, double *nx, double *ny, double *back_x, double *back_y) {
    const double g = t * (ls->m - 1);
    int i = (int)g;
    if (i < 0) i = 0;
    if (i > ls->m - 2) i = ls->m - 2;
    if (!ls->ok[i] || !ls->ok[i + 1]) return 0;
    const double s = g - i, step = 1.0 / (ls->m - 1), s2 = s * s, s3 = s2 * s;
    const double h00 = 2 * s3 - 3 * s2 + 1, h10 = (s3 - 2 * s2 + s) * step, h01 = 3 * s2 - 2 * s3, h11 = (s3 - s2) * step;
    b->x = h00 * ls->p[i].x + h10 * ls->dp[i].x + h01 * ls->p[i + 1].x + h11 * ls->dp[i + 1].x;
    b->y = h00 * ls->p[i].y + h10 * ls->dp[i].y + h01 * ls->p[i + 1].y + h11 * ls->dp[i + 1].y;
    const double *u = ls->n[i], *v = ls->n[i + 1];
    *nx = u[0] + s * (v[0] - u[0]), *ny = u[1] + s * (v[1] - u[1]);
    *back_x = u[2] + s * (v[2] - u[2]), *back_y = u[3] + s * (v[3] - u[3]);
    return 1;
}

/* Every profile of one pass along a side, worked out before its samples are read: where it crosses the side in the plane
 * (q) and in pixels (b), its pixel normal, the plane offset per pixel along it, and its 16.16 fixed-point step. */
typedef struct {
    vec2 q[NAPA_SIDE_SAMPLES], b[NAPA_SIDE_SAMPLES], n[NAPA_SIDE_SAMPLES], back[NAPA_SIDE_SAMPLES];
    int32_t ix[NAPA_SIDE_SAMPLES], iy[NAPA_SIDE_SAMPLES];
    unsigned char ok[NAPA_SIDE_SAMPLES];
} lens_profile_set;

static NAPA_NOINLINE void lens_profiles(const lens_side *ls, vec2 a, vec2 b, double margin, double dt, int np, double step, lens_profile_set *set) {
    for (int i = 0; i < np; i++) {
        const double t = margin + dt * (i + 0.5);
        set->q[i].x = a.x + (b.x - a.x) * t, set->q[i].y = a.y + (b.y - a.y) * t;
        set->ok[i] = (unsigned char)lens_profile(ls, t, &set->b[i], &set->n[i].x, &set->n[i].y, &set->back[i].x, &set->back[i].y);
        if (!set->ok[i]) continue;
        const double sx = set->n[i].x * step * 65536.0, sy = set->n[i].y * step * 65536.0;
        set->ix[i] = (int32_t)(sx < 0 ? sx - 0.5 : sx + 0.5), set->iy[i] = (int32_t)(sy < 0 ? sy - 0.5 : sy + 0.5);
    }
}

/* Nearest falling midpoint crossing; equal distances retain the first crossing. */
static double nearest_midpoint_crossing(const int64_t *prof, int K, int jl, int jr, double mid, double center) {
    double best_c = -1, best_d = 1e9;
    /* prof holds fixed-point grey levels (under 2^33), which a double holds exactly, so for an
     * entry p the tests (double)p >= mid and (double)p < mid are p >= ceil(mid) and
     * p < ceil(mid): scan in integers and convert only a found crossing. */
    if (mid >= -9223372036854775808.0 && mid < 9223372036854775808.0) {
        /* ceil(mid) without libm: truncation toward zero already is it below zero. */
        int64_t level = (int64_t)mid;
        if ((double)level < mid) level++;
        for (int j = jl - 1; j <= jr && j + 1 < K; j++) {
            if (j < 0 || !(prof[j] >= level && prof[j + 1] < level)) continue;
            double c = j + ((double)prof[j] - mid) / (double)(prof[j] - prof[j + 1]);
            double d = fabs(c - center);
            if (d < best_d) best_d = d, best_c = c;
        }
        return best_c;
    }
    for (int j = jl - 1; j <= jr && j + 1 < K; j++) {
        if (j < 0 || !((double)prof[j] >= mid && (double)prof[j + 1] < mid)) continue;
        double c = j + ((double)prof[j] - mid) / (double)(prof[j] - prof[j + 1]);
        double d = fabs(c - center);
        if (d < best_d) best_d = d, best_c = c;
    }
    return best_c;
}

/* A partial profile can locate a measured intensity crossing without observing the gradient's whole
 * tail. Only use image-supported samples, including every sample the existing balance correction reads.
 * The caller supplies a midpoint measured from the tag's visible black/white cells. */
static int observed_profile_edge(const image_view *im, const int64_t *prof, int K, vec2 base, vec2 normal,
                                 double r_out, double step, double mid, double min_edge,
                                 double *position, double *weight) {
    int lo = K, hi = -1;
    for (int j = 0; j < K; j++) {
        const double o = -r_out + j * step;
        if (sample_is_observed(im, base.x + normal.x * o, base.y + normal.y * o)) {
            if (lo == K) lo = j;
            hi = j;
        }
    }
    if (lo >= hi) return 0;
    mid *= 16777216.0;
    double crossing = nearest_midpoint_crossing(prof + lo, hi - lo + 1, 1, hi - lo - 1,
                                                mid, r_out / step - lo);
    /* The balance estimator needs one observed pixel on each side; a bracket alone is insufficient. */
    if (!(crossing >= 0) || crossing < 1.0 / step || crossing + 1.0 / step > hi - lo) return 0;
    const int j = lo + (int)crossing;
    const double strength = (double)(prof[j] - prof[j + 1]) / 16777216.0;
    /* Same derivative-strength requirement as the ordinary central difference, over one step. */
    if (!(strength >= 0.25 * min_edge * step)) return 0;
    crossing = lo + balanced_crossing(prof + lo, hi - lo + 1, step, crossing, mid);
    *position = -r_out + crossing * step;
    *weight = 2 * strength;
    return 1;
}

/* Rank falling two-plateau fits of an initial profile by their reduction
 * in squared residuals: n_left*n_right/K * (white-black)^2. This ranks
 * splits; it does not certify a dark landing. A sharp sampled step's
 * central difference equals its plateau amplitude, so the existing
 * gradient-strength gate applies to the fitted amplitude too. The original
 * gradient-lobe centroid still supplies the subpixel location. */
static int initial_plateau_peak(const int64_t *prof, const int64_t *g, int K, double min_drop) {
    int64_t total = 0, left = 0;
    for (int j = 0; j < K; j++) total += prof[j];
    double best_score = 0;
    int split = -1;
    for (int j = 1; j < K; j++) {
        left += prof[j - 1];
        double product = (double)j * (K - j);
        double drop = (double)left * (K - j) - (double)(total - left) * j;
        if (drop <= 0 || drop < min_drop * product) continue;
        double score = drop * drop / product;
        if (score > best_score) best_score = score, split = j;
    }
    if (split < 0) return -1;
    int peak = split - 1;
    if (peak < 1) peak = 1;
    if (split > 1 && peak < K - 2 && g[peak + 1] > g[peak]) peak++;
    return peak;
}

/* Fit each side to the outer border edge searched from r_out outside to r_in inside the given quad. Sides taken
 * from the frame edge instead are added to *frame_sides (bit k: side k, from in[k] to in[k + 1]); with frame_sides
 * NULL, a side with no edge fails the fit. */
static NAPA_FORCE_INLINE int fit_sides_impl(const image_view *im, const vec2 in[4], vec2 out[4], double r_out, double r_in, int min_edge,
                     int outermost, int max_points, double *edge_width, edge_levels *levels, double cells,
                     vec2 alternate[4], vec2 curved[4], vec2 est_lens[4], int *est_ok, double *width_ratio,
                     int *frame_sides) {
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
    /* Through a known lens the sides are fitted where they are straight: in the plane facing the tag (lens_frame). */
    lens_frame lens_view;
    vec2 plane[4], plane_out[4];
    const int use_lens = im->lens != NULL && !outermost && lens_frame_init(im->lens, in, &lens_view, plane,
                                                        max_points > 24 ? NAPA_LENS_STRAIGHT_PX : NAPA_LENS_STRAIGHT_COARSE_PX);
    side_samples samples, crossing_samples;
    /* A normal window stays in the frame when every input corner clears its reach. */
    int partial_possible = 0, needs_partial = 0;
    const double profile_reach = fmax(r_out, r_in);
    for (int k = 0; k < 4; k++)
        partial_possible |= in[k].x < profile_reach || in[k].y < profile_reach ||
            in[k].x > w - 1 - profile_reach || in[k].y > h - 1 - profile_reach;
    edge_levels partial_levels = {.black = 255, .white = 0};
    const family_layout *partial_family = est_lens != NULL && !use_lens && levels != NULL &&
        (levels->model_pending || levels->has_model) ? levels->model_family : NULL;
    int partial_levels_ready = 0;
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
        /* The side's own frame for the samples (fit_*): the pixels, or through a lens the plane. Three normals
         * follow: nx, ny the side's in pixels; pnx, pny each profile's in pixels (through a lens, square to the
         * side where the lens draws it); fit_nx, fit_ny the side's in the fit frame. */
        vec2 fit_a = a;
        double fit_ex = ex, fit_ey = ey, fit_nx = nx, fit_ny = ny;
        lens_side lside;
        lens_profile_set lprof;
        if (use_lens) {
            if (!lens_side_init(&lens_view, plane[k], plane[(k + 1) & 3], len, nx, ny, &lside)) return 0;
            fit_a = plane[k], fit_ex = lside.sx, fit_ey = lside.sy, fit_nx = -fit_ey, fit_ny = fit_ex;
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
        double side_width_sum = width_sum;
        int side_width_n = width_n;
        double step = (r_out + r_in) / (K - 1);
        double margin = NAPA_SIDE_MARGIN;
        if (max_points > 24) {
            margin = NAPA_SIDE_MARGIN_WINDOWS * (r_out + r_in) / len;
            if (margin < NAPA_SIDE_MARGIN_PIXELS / len) margin = NAPA_SIDE_MARGIN_PIXELS / len;
            if (margin < NAPA_SIDE_MIN_MARGIN) margin = NAPA_SIDE_MIN_MARGIN;
            if (margin > NAPA_SIDE_MARGIN) margin = NAPA_SIDE_MARGIN;
        }
        double dt = (1 - 2 * margin) / np, min_g = min_edge * 0.5 * step * 16777216.0; /* best / (2 step) >= min_edge / 4 */
        const double cell_len = len / cells, cross_limit = NAPA_CROSSING_CELLS * len / cells, half_cell = len / (2.0 * cells);
        /* The profile step in 16.16 fixed point, rounded to nearest. */
        double ixd = nx * step * 65536.0, iyd = ny * step * 65536.0;
        int32_t ixp = (int32_t)(ixd < 0 ? ixd - 0.5 : ixd + 0.5), iyp = (int32_t)(iyd < 0 ? iyd - 0.5 : iyd + 0.5);
        unsigned char zone[MAXP]; /* 0: the middle of the side; 1, 2: the ends the coarse passes leave out */
        int tried[3] = {0, 0, 0}, got[3] = {0, 0, 0}, retaken = 0;
        const double kept_width_sum = width_sum, kept_side_width = side_width[k];
        const int kept_width_n = width_n, kept_side_widths = side_widths[k];
retake:
        if (use_lens) lens_profiles(&lside, fit_a, plane[(k + 1) & 3], margin, dt, np, step, &lprof);
        for (int i = 0; i < np && good + (np - i) >= 3; i++) { /* stop once 3 edge points are out of reach */
            double t = margin + dt * (i + 0.5);
            const int z = t < NAPA_SIDE_MARGIN ? 1 : t > 1 - NAPA_SIDE_MARGIN ? 2 : 0;
            tried[z]++;
            double bx, by, pnx = nx, pny = ny, back_x = 0, back_y = 0;
            int32_t pixp = ixp, piyp = iyp;
            vec2 q = {0, 0};
            if (use_lens) {
                /* The profile crosses the side where the lens draws it, square to it there. */
                if (!lprof.ok[i]) continue;
                q = lprof.q[i], bx = lprof.b[i].x, by = lprof.b[i].y, pnx = lprof.n[i].x, pny = lprof.n[i].y;
                back_x = lprof.back[i].x, back_y = lprof.back[i].y, pixp = lprof.ix[i], piyp = lprof.iy[i];
            } else {
                bx = a.x + (b.x - a.x) * t, by = a.y + (b.y - a.y) * t;
            }
            int64_t prof[(KMAX + 3) & ~3]; /* grey levels in 40.24 fixed point */
            double sx0 = bx - pnx * r_out, sy0 = by - pny * r_out;
            double sx1 = bx + pnx * r_in, sy1 = by + pny * r_in;
            /* Whole profile inside the image: skip per-sample clamping, and step in 16.16 fixed point
             * with 12-bit interpolation weights (1/8192 px); the profile and its gradient stay in fixed
             * point, with no conversion per sample. The fixed-point
             * walk is checked end to end, since its rounded step can drift past a profile that ends
             * right at the border (the floating-point test comes first and also rejects NaN). */
            int32_t fxp = 0, fyp = 0;
            const int complete = sx0 >= 0 && sy0 >= 0 && sx1 >= 0 && sy1 >= 0 && sx0 <= w - 1 && sx1 <= w - 1 &&
                                 sy0 <= h - 1 && sy1 <= h - 1;
            int inside = complete && sx0 < w - 1 && sx1 < w - 1 && sy0 < h - 1 && sy1 < h - 1;
            if (inside) {
                fxp = (int32_t)(sx0 * 65536.0), fyp = (int32_t)(sy0 * 65536.0);
                int64_t endx = fxp + (int64_t)(K - 1) * pixp, endy = fyp + (int64_t)(K - 1) * piyp;
                inside = endx >= 0 && endy >= 0 && endx < ((int64_t)(w - 1) << 16) && endy < ((int64_t)(h - 1) << 16);
            }
            if (inside) {
                profile_fixed(img, stride, fxp, fyp, pixp, piyp, K, prof);
                if (im->invert)
                    for (int j = 0; j < K; j++) prof[j] = ((int64_t)255 << 24) - prof[j];
            } else {
                for (int j = 0; j < K; j++) {
                    double o = -r_out + j * step;
                    prof[j] = (int64_t)(sample_bilinear(im, bx + pnx * o, by + pny * o) * 16777216.0 + 0.5);
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
            if (outermost) {
                bj = initial_plateau_peak(prof, g, K, min_g);
                if (bj >= 0) best = g[bj];
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
                (jr - jl + 1) * step > cross_limit) {
                /* Levels from the centres of the cells either side (white margin, black border), half
                 * a cell from the edge: outside the blurred ramp the window would sit in. */
                if (alternate != NULL) {
                    if (levels->model_pending) {
                        refresh_edge_light(im, in, levels, 0);
                        levels->model_pending = 0;
                    }
                    if (levels->has_model) {
                        double local_mid = levels->midpoint[0] +
                            levels->midpoint[1] * (bx - levels->cx) / levels->scale +
                            levels->midpoint[2] * (by - levels->cy) / levels->scale;
                        double mid = local_mid * 16777216.0;
                        double crossing = nearest_midpoint_crossing(prof, K, jl, jr, mid, cj);
                        if (crossing >= 0) acj = crossing, alternate_crossing = 1;
                        if (alternate_crossing && cell_len >= NAPA_BALANCE_MIN_CELL) acj = balanced_crossing(prof, K, step, acj, mid);
                    }
                }
                /* The midpoint between the plateau levels half a cell either side of the centre of mass.
                 * One reading: re-reading the levels around each new estimate was slower and found fewer
                 * small tags. */
                double pos0 = -r_out + cj * step, level[2]; /* white outside, black inside */
                const double level_x[2] = {bx + pnx * (pos0 - half_cell), bx + pnx * (pos0 + half_cell)};
                const double level_y[2] = {by + pny * (pos0 - half_cell), by + pny * (pos0 + half_cell)};
                sample_at(im, level_x, level_y, 2, level);
                double mid = 0.5 * (level[0] + level[1]) * 16777216.0;
                double best_c = nearest_midpoint_crossing(prof, K, jl, jr, mid, cj);
                if (best_c >= 0) {
                    cj = best_c;
                    if (cell_len >= NAPA_BALANCE_MIN_CELL) cj = balanced_crossing(prof, K, step, cj, mid);
                }
            }
            width_sum += (jr - jl + 1) * step;
            width_n++;
            side_width[k] += (jr - jl + 1) * step, side_widths[k]++;
            double pos = -r_out + cj * step;
            if (use_lens) px[good] = q.x + back_x * pos, py[good] = q.y + back_y * pos;
            else px[good] = bx + pnx * pos, py[good] = by + pny * pos;
            wt[good] = (double)best * (1.0 / (1 << 24));
            if (est_lens != NULL && !use_lens) {
                samples.x[k][good] = px[good], samples.y[k][good] = py[good];
                samples.w[k][good] = complete ? wt[good] : 0;
                crossing_samples.x[k][good] = px[good], crossing_samples.y[k][good] = py[good];
                crossing_samples.w[k][good] = 0;
                if (partial_possible && levels != NULL && levels->white > levels->black) {
                    if (!partial_levels_ready) {
                        if (partial_family != NULL) {
                            partial_levels = (edge_levels){.model_family = partial_family};
                            refresh_edge_light(im, in, &partial_levels, 1);
                        } else partial_levels.black = 255, partial_levels.white = 0;
                        partial_levels_ready = 1;
                    }
                    if (partial_levels.white - partial_levels.black >= min_edge) {
                        const double mid = partial_levels.has_model ? partial_levels.midpoint[0] +
                            partial_levels.midpoint[1] * (bx - partial_levels.cx) / partial_levels.scale +
                            partial_levels.midpoint[2] * (by - partial_levels.cy) / partial_levels.scale :
                            0.5 * (partial_levels.white + partial_levels.black);
                        double offset, weight;
                        if (observed_profile_edge(im, prof, K, (vec2){bx, by}, (vec2){pnx, pny},
                                                  r_out, step, mid, min_edge, &offset, &weight)) {
                            crossing_samples.x[k][good] = bx + pnx * offset, crossing_samples.y[k][good] = by + pny * offset;
                            crossing_samples.w[k][good] = weight;
                        }
                    }
                }
            }
            if (alternate != NULL && (alternate_crossing || side_changed)) {
                if (!side_changed) {
                    memcpy(apx, px, (size_t)good * sizeof(double));
                    memcpy(apy, py, (size_t)good * sizeof(double));
                    memcpy(awt, wt, (size_t)good * sizeof(double));
                }
                double apos = alternate_crossing ? -r_out + acj * step : pos;
                if (use_lens) apx[good] = q.x + back_x * apos, apy[good] = q.y + back_y * apos;
                else apx[good] = bx + pnx * apos, apy[good] = by + pny * apos;
                awt[good] = wt[good];
                side_changed |= alternate_crossing;
            }
            zone[good] = (unsigned char)z, got[z]++;
            good++;
        }
        if (good < 3) {
            width_sum = side_width_sum, width_n = side_width_n; /* this side's few edge points aren't its edge */
            /* Only a caller that tracks the mark (and so measures the side later) may use the frame edge. */
            /* A frame boundary is in pixels, not the lens fit's tangent plane. */
            if (use_lens || frame_sides == NULL || !frame_side(im, a, b, r_out, &lx[k], &ly[k], &dx[k], &dy[k])) return 0;
            *frame_sides |= 1 << k;
            curves[k].qualified = 0, samples.n[k] = crossing_samples.n[k] = 0;
            side_width[k] = 0, side_widths[k] = 0;
            alx[k] = lx[k], aly[k] = ly[k], adx[k] = dx[k], ady[k] = dy[k];
            continue;
        }
        if (margin < NAPA_SIDE_MARGIN && !retaken && got[0] >= 4) {
            /* An end of the side joins the fit only where the border runs through it unbroken: every profile found an
             * edge, each about as strong as the middle of the side's and close to the line through the middle. Anything
             * lying across the border near a corner breaks one of the three, and the side is taken again as before. */
            double sw = 0, su = 0, sv = 0, suu = 0, suv = 0;
            for (int i = 0; i < good; i++) {
                if (zone[i]) continue;
                const double u = (px[i] - fit_a.x) * fit_ex + (py[i] - fit_a.y) * fit_ey, v = (px[i] - fit_a.x) * fit_nx + (py[i] - fit_a.y) * fit_ny;
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
                    const double u = (px[i] - fit_a.x) * fit_ex + (py[i] - fit_a.y) * fit_ey, v = (px[i] - fit_a.x) * fit_nx + (py[i] - fit_a.y) * fit_ny;
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
        if (curved != NULL) fit_curve_side(&curves[k], fit_a, use_lens ? plane[(k + 1) & 3] : b, px, py, wt, good);
        if (est_lens != NULL && !use_lens) {
            samples.n[k] = crossing_samples.n[k] = good;
            /* Prefer complete gradients. If one side needs recovery, use crossings on all four sides:
             * mixing the two position estimators can make pixel-phase differences look like a lens. */
            int complete_points = 0;
            for (int i = 0; i < good; i++) complete_points += samples.w[k][i] > 0;
            needs_partial |= complete_points < NAPA_LENS_MIN_SIDE_POINTS;
        }
        if (!fit_side_line(px, py, wt, good, reach, &lx[k], &ly[k], &dx[k], &dy[k])) return 0;
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
        if (use_lens) {
            plane_out[k] = out[k];
            if (!lens_from_plane(&lens_view, out[k].x, out[k].y, &out[k])) return 0;
        }
        double mv = fabs(out[k].x - in[k].x) + fabs(out[k].y - in[k].y);
        if (mv > 4.0 * reach + 2.0) return 0;
    }
    if (use_lens) lens_memo_chain(&lens_view, in, out, plane_out);
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
            if (use_lens && !lens_from_plane(&lens_view, alternate[k].x, alternate[k].y, &alternate[k])) { valid = 0; break; }
            double mv = fabs(alternate[k].x - in[k].x) + fabs(alternate[k].y - in[k].y);
            if (!(mv <= 4.0 * reach + 2.0)) { valid = 0; break; }
        }
        if (!valid) memcpy(alternate, out, 4 * sizeof(vec2));
    }
    if (curved != NULL) {
        vec2 proposal[4];
        memcpy(curved, out, 4 * sizeof(vec2));
        if (!use_lens) {
            if (intersect_curves(curves, in, out, lx, ly, dx, dy, reach, proposal)) memcpy(curved, proposal, sizeof proposal);
        } else if (intersect_curves(curves, plane, plane_out, lx, ly, dx, dy, reach, proposal)) {
            /* What bow is left in the plane is the tag's own (a bent sheet), not the lens. */
            int mapped = 1;
            for (int k = 0; k < 4 && mapped; k++) mapped = lens_from_plane(&lens_view, proposal[k].x, proposal[k].y, &proposal[k]);
            if (mapped) memcpy(curved, proposal, sizeof proposal);
        }
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
    if (est_lens != NULL && use_lens) *est_ok = 0; /* the lens is already known */
    else if (est_lens != NULL) {
        int near_frame = 0, corner_clear[4], side_clear[4];
        for (int k = 0; k < 4; k++) {
            corner_clear[k] = inside_frame(im, out[k]) && inside_frame(im, curved[k]);
            near_frame |= !corner_clear[k];
        }
        for (int k = 0; k < 4; k++) side_clear[k] = corner_clear[k] && corner_clear[(k + 1) & 3];
        *est_ok = near_frame && shared_lens_quad(im, needs_partial ? &crossing_samples : &samples, side_clear, in, reach, est_lens);
    }
    return 1;
}

static int fit_sides(const image_view *im, const vec2 in[4], vec2 out[4], double r_out, double r_in, int min_edge,
                     int outermost, int max_points, double *edge_width, edge_levels *levels, double cells, int *frame_sides) {
    return fit_sides_impl(im, in, out, r_out, r_in, min_edge, outermost, max_points, edge_width, levels, cells,
                          NULL, NULL, NULL, NULL, NULL, frame_sides);
}

static int fit_initial_sides(const image_view *im, const vec2 in[4], vec2 out[4],
                             double r_out, double r_in, int min_edge, double *width,
                             double cells, int *frame_sides) {
    /* A short later side will fail the same fit after sampling earlier sides.
     * Check those lengths before spending profiles; side zero is checked first
     * by the fitter itself. Failed initial-fit outputs are never consumed. */
    for (int k = 1; k < 4; k++) {
        double x = in[(k + 1) & 3].x - in[k].x, y = in[(k + 1) & 3].y - in[k].y;
        if (x * x + y * y < 36.0) return 0;
    }
    return fit_sides_impl(im, in, out, r_out, r_in, min_edge, 1, INITIAL_SIDE_POINTS, width, NULL, cells,
                          NULL, NULL, NULL, NULL, NULL, frame_sides);
}

/* fit_sides with the alternatives the final fit weighs. est_lens / est_ok: near the frame and without a calibrated
 * lens, the corners a radial lens estimated from the tag's own bowed sides would give (shared_lens_quad), and whether
 * one was found. */
static int fit_sides_alternate(const image_view *im, const vec2 in[4], vec2 out[4], double r_out, double r_in, int min_edge,
                     int outermost, int max_points, double *edge_width, edge_levels *levels, double cells, vec2 alternate[4], vec2 curved[4], vec2 est_lens[4], int *est_ok, double *width_ratio, int *frame_sides) {
    return fit_sides_impl(im, in, out, r_out, r_in, min_edge, outermost, max_points, edge_width, levels, cells,
                          alternate, curved, est_lens, est_ok, width_ratio, frame_sides);
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
                              &width, &independent, cells, alternate, NULL, NULL, NULL, NULL, NULL);
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

/* Project/sample two points with the scalar operation order and NaN-safe clamps.
 * All families, including inverted families, retain the same floating-point values. */
static void sample_points_impl(const image_view *im, const double H[9], double *u, double *v, int n,
                               double *out, vec2 *rawcoords) {
    int i = 0;
#ifdef NAPA_SSE2
    const __m128d h0 = _mm_set1_pd(H[0]), h1 = _mm_set1_pd(H[1]), h2 = _mm_set1_pd(H[2]), h3 = _mm_set1_pd(H[3]);
    const __m128d h4 = _mm_set1_pd(H[4]), h5 = _mm_set1_pd(H[5]), h6 = _mm_set1_pd(H[6]), h7 = _mm_set1_pd(H[7]);
    const __m128d h8 = _mm_set1_pd(H[8]), one = _mm_set1_pd(1.0);
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
        _mm_storeu_pd(out + i, sample_bilinear_pair(im, x, y));
    }
#elif defined(__aarch64__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    const float64x2_t h0 = vdupq_n_f64(H[0]), h1 = vdupq_n_f64(H[1]), h2 = vdupq_n_f64(H[2]), h3 = vdupq_n_f64(H[3]);
    const float64x2_t h4 = vdupq_n_f64(H[4]), h5 = vdupq_n_f64(H[5]), h6 = vdupq_n_f64(H[6]), h7 = vdupq_n_f64(H[7]);
    const float64x2_t h8 = vdupq_n_f64(H[8]), one = vdupq_n_f64(1.0);
    for (; i + 2 <= n; i += 2) {
        const float64x2_t U = vld1q_f64(u + i), V = vld1q_f64(v + i);
        const float64x2_t iz = vdivq_f64(one, vaddq_f64(vaddq_f64(vmulq_f64(h6, U), vmulq_f64(h7, V)), h8));
        const float64x2_t x = vmulq_f64(vaddq_f64(vaddq_f64(vmulq_f64(h0, U), vmulq_f64(h1, V)), h2), iz);
        const float64x2_t y = vmulq_f64(vaddq_f64(vaddq_f64(vmulq_f64(h3, U), vmulq_f64(h4, V)), h5), iz);
        if (rawcoords) {
            rawcoords[i].x = vgetq_lane_f64(x, 0);
            rawcoords[i].y = vgetq_lane_f64(y, 0);
            rawcoords[i + 1].x = vgetq_lane_f64(x, 1);
            rawcoords[i + 1].y = vgetq_lane_f64(y, 1);
        }
        vst1q_f64(out + i, sample_bilinear_pair(im, x, y));
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

/* A tag's cell grid in the frame: the homography from the unit square, through the lens when the view has one (the
 * homography then maps to the plane facing the tag, where the grid is straight). */
typedef struct {
    double H[9];
    int lens;
    lens_frame f;
} quad_map;

static void quad_map_init(const image_view *im, const vec2 q[4], quad_map *m) {
    vec2 plane[4];
    square_to_quad(q, m->H);
    m->lens = 0;
    if (im->lens == NULL || !lens_frame_init(im->lens, q, &m->f, plane, NAPA_LENS_STRAIGHT_PX)) return;
    /* The readings shift the whole tag's edges alike, so they keep the plain grid wherever the lens moves it by too little
     * to matter. The sides' bow doesn't tell: fisheye views squeeze a tag's middle while leaving its sides straight. */
    lens_memo *memo = im->lens->memo;
    double Hp[9], departure = -1;
    square_to_quad(plane, Hp);
    if (memo != NULL && memo->valid && same_corners(q, memo->key)) departure = memo->departure;
    if (departure < 0) {
        static const double at[5][2] = {{0.5, 0}, {1, 0.5}, {0.5, 1}, {0, 0.5}, {0.5, 0.5}};
        departure = 0;
        for (int k = 0; k < 5; k++) {
            const vec2 a = project(Hp, at[k][0], at[k][1]), b = project(m->H, at[k][0], at[k][1]);
            vec2 c;
            if (!lens_from_plane(&m->f, a.x, a.y, &c)) {
                departure = HUGE_VAL;
                break;
            }
            departure = fmax(departure, vec_length(c.x - b.x, c.y - b.y));
        }
        if (memo != NULL && memo->valid && same_corners(q, memo->key)) memo->departure = departure;
    }
    if (departure < NAPA_LENS_BIAS_STRAIGHT_PX) return;
    m->lens = 1;
    memcpy(m->H, Hp, sizeof Hp);
}

static inline vec2 quad_point(const quad_map *m, double u, double v) {
    vec2 p = project(m->H, u, v);
    if (m->lens) {
        vec2 d = {-1, -1}; /* not imaged: read as the frame's corner pixel, like any point outside it */
        lens_from_plane(&m->f, p.x, p.y, &d);
        return d;
    }
    return p;
}

/* Cubic coefficients (power basis in s) through the values at s = 0, 1, 2, 3, for x and y. */
static void cubic_through(const vec2 node[4], vec2 c[4]) {
    const vec2 d1 = {node[1].x - node[0].x, node[1].y - node[0].y};
    const vec2 d2 = {node[2].x - 2 * node[1].x + node[0].x, node[2].y - 2 * node[1].y + node[0].y};
    const vec2 d3 = {node[3].x - 3 * node[2].x + 3 * node[1].x - node[0].x, node[3].y - 3 * node[2].y + 3 * node[1].y - node[0].y};
    c[0] = node[0];
    c[1].x = d1.x - d2.x / 2 + d3.x / 3, c[1].y = d1.y - d2.y / 2 + d3.y / 3;
    c[2].x = (d2.x - d3.x) / 2, c[2].y = (d2.y - d3.y) / 2;
    c[3].x = d3.x / 6, c[3].y = d3.y / 6;
}

/* Sample 4 <= n <= 64 points (u, v) of the square, evenly spaced along one line of it. Through a lens, four of them are projected
 * and the rest follow the cubic through those, which over the short lines this reads is as good as projecting each. */
static void quad_sample_line(const image_view *im, const quad_map *m, double *u, double *v, int n, double *out, vec2 *raw) {
    if (!m->lens) {
        sample_points_impl(im, m->H, u, v, n, out, raw);
        return;
    }
    vec2 node[4], c[4];
    for (int k = 0; k < 4; k++) node[k] = quad_point(m, u[0] + (u[n - 1] - u[0]) * k / 3, v[0] + (v[n - 1] - v[0]) * k / 3);
    cubic_through(node, c);
    /* The cubic in the sample index (s = 3 i / (n - 1)), stepped by forward differences, then read like a homography's. */
    const double h = 3.0 / (n - 1), a1x = c[1].x * h, a2x = c[2].x * h * h, a3x = c[3].x * h * h * h;
    const double a1y = c[1].y * h, a2y = c[2].y * h * h, a3y = c[3].y * h * h * h;
    double x = c[0].x, y = c[0].y, f1x = a1x + a2x + a3x, f1y = a1y + a2y + a3y, f2x = 2 * a2x + 6 * a3x, f2y = 2 * a2y + 6 * a3y;
    const double f3x = 6 * a3x, f3y = 6 * a3y;
    double xs[64], ys[64];
    for (int i = 0; i < n; i++) {
        xs[i] = x, ys[i] = y;
        x += f1x, y += f1y, f1x += f2x, f1y += f2y, f2x += f3x, f2y += f3y;
    }
    static const double identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    sample_points_impl(im, identity, xs, ys, n, out, raw);
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

enum { BORDER_PROFILE_SAMPLES = 33 };

/* The value at rank k (0-based) in sorted order of a[0..n), without mutating a.
 * Scratch must hold n doubles. Exact selection: the same element a sort would place there. */
static double select_kth(const double *a, int n, int k, double *scratch) {
    memcpy(scratch, a, (size_t)n * sizeof(double));
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const double p = scratch[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (scratch[i] < p) i++;
            while (scratch[j] > p) j--;
            if (i <= j) { const double t = scratch[i]; scratch[i] = scratch[j]; scratch[j] = t; i++; j--; }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    return scratch[k];
}

/* Position along a side and depth inward, in cells; side 0 is the top (v = 0). */
static vec2 border_uv(double cells, int side, int c, double depth) {
    double t = depth / cells, a = (c + 0.5) / cells;
    vec2 uv = {side == 0 ? a : side == 1 ? 1 - t : side == 2 ? a : t,
               side == 0 ? t : side == 1 ? a : side == 2 ? 1 - t : a};
    return uv;
}

/* Sample from half a cell outside the outline to the payload centre. Skip dark payloads before
 * gathering the rest of the profile: they cannot measure the border's inner edge. */
static int border_profile(const image_view *im, const quad_map *M, double cells, int side, int c, double mid,
                          double prof[BORDER_PROFILE_SAMPLES], vec2 pts[BORDER_PROFILE_SAMPLES]) {
    enum { N = BORDER_PROFILE_SAMPLES };
    vec2 uv = border_uv(cells, side, c, 1.5), pc = quad_point(M, uv.x, uv.y);
    double payload = sample_bilinear(im, pc.x, pc.y);
    if (payload < mid) return 0;
    double us[N - 1], vs[N - 1];
    for (int j = 0; j < N - 1; j++) {
        uv = border_uv(cells, side, c, -0.5 + 2.0 * j / (N - 1));
        us[j] = uv.x; vs[j] = uv.y;
    }
    quad_sample_line(im, M, us, vs, N - 1, prof, pts);
    pts[N - 1] = pc;
    prof[N - 1] = payload;
    return 1;
}

/* Clipping (or any symmetric edge bias) shrinks every black region by delta: the outline by delta
 * per side and the one-cell border by 2 * delta. Measuring the border's width where a white payload
 * cell lies inside it gives delta = (cell - width) / 1.75 against the measured outline. Returns the
 * number of border measurements averaged into *delta. */
static int border_bias(const family_layout *L, const image_view *im, const vec2 q[4], double black, double white,
                       double *delta, double *disagreement) {
    enum { N = BORDER_PROFILE_SAMPLES };
    double cell_d[NAPA_BORDER_CELLS_MAX], mean_cell = 0;
    unsigned char cell_side[NAPA_BORDER_CELLS_MAX];
    quad_map M;
    double sum = 0, side_sum[4] = {0, 0, 0, 0};
    int n = 0, side_n[4] = {0, 0, 0, 0};
    quad_map_init(im, q, &M);
    double mid = 0.5 * (black + white);
    for (int side = 0; side < 4; side++) {
        /* grow_quad moves sides along their normals; measure border widths in those same units.
         * A projected cell's direction is generally oblique to its edge under perspective. */
        double nx = q[side].y - q[(side + 1) & 3].y;
        double ny = q[(side + 1) & 3].x - q[side].x;
        double side_len = vec_length(nx, ny);
        nx /= side_len; ny /= side_len;
        for (int c = 1; c < L->width - 1; c++) {
            double prof[N];
            vec2 pts[N];
            if (!border_profile(im, &M, L->cells, side, c, mid, prof, pts)) continue;
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
        }
    }
    if (!n) return 0;
    if (n > NAPA_BORDER_CELLS_MAX) n = NAPA_BORDER_CELLS_MAX;
    /* Something lying across the border (a cable, a strap) spoils the cells under it by a large part of a cell, while
     * clipping, gamma and lenses move every cell by nearly the same amount. Leave out cells far from the middle value. */
    double sorted[NAPA_BORDER_CELLS_MAX];
    const double middle_hi = select_kth(cell_d, n, n / 2, sorted);
    const double middle = n & 1 ? middle_hi : 0.5 * (select_kth(cell_d, n, n / 2 - 1, sorted) + middle_hi);
    for (int k = 0; k < 4; k++) mean_cell += 0.25 * vec_length(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y) / L->cells;
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

/* One line of the square, across fixed and x running along it in cells: (u, v) = (x, across) / cells, or swapped. Through a
 * lens it is the cubic through four projected points, which over one tag is as good as projecting each point. */
typedef struct {
    const quad_map *m;
    int axis;
    double across, cells, x0, inv_step;
    vec2 c[4]; /* the cubic in s = (x - x0) / step */
} quad_line;

static void quad_line_init(quad_line *l, const quad_map *m, int axis, double across, double cells, double x0, double x1) {
    const double step = (x1 - x0) / 3;
    l->m = m, l->axis = axis, l->across = across, l->cells = cells, l->x0 = x0, l->inv_step = 1 / step;
    if (m->lens) {
        vec2 node[4];
        for (int k = 0; k < 4; k++) {
            const double x = (x0 + k * step) / cells;
            node[k] = axis ? quad_point(m, across, x) : quad_point(m, x, across);
        }
        cubic_through(node, l->c);
    }
}

static inline vec2 quad_line_point(const quad_line *l, double x) {
    if (!l->m->lens) return l->axis ? project(l->m->H, l->across, x / l->cells) : project(l->m->H, x / l->cells, l->across);
    const double s = (x - l->x0) * l->inv_step;
    const vec2 p = {l->c[0].x + s * (l->c[1].x + s * (l->c[2].x + s * l->c[3].x)), l->c[0].y + s * (l->c[1].y + s * (l->c[2].y + s * l->c[3].y))};
    return p;
}

/* A tone bias (clipping, gamma under blur) moves every black-white edge of the tag alike, payload
 * edges included. A border that simply is not one cell wide moves only its own inner edge. Measure how far the
 * payload's own edges sit from the cell grid, signed so + means black regions are smaller, in pixels per edge.
 * Edges entering and leaving black are averaged separately so a small grid shift cancels. */
static int payload_bias(const family_layout *L, const image_view *im, const vec2 q[4], double black, double white, double *delta) {
    enum { PER = 3, MAXC = 16 };
    quad_map M;
    double A[2][3][3] = {{{0}}}, B[2][3] = {{0}};
    int n[2] = {0, 0};
    quad_map_init(im, q, &M);
    const double mid = 0.5 * (black + white);
    const int cells = L->cells, steps = (cells - 3) * PER;
    for (int axis = 0; axis < 2; axis++)
        for (int line = 1; line < cells - 1; line++) {
            double prev = 0, xprev = 0, xs[MAXC], px[MAXC];
            int dark_after[MAXC], count = 0;
            const double across = (line + 0.5) / cells;
            quad_line ln;
            quad_line_init(&ln, &M, axis, across, cells, 1.0, cells - 1.0);
            double scan[MAXC * PER + 1], su[MAXC * PER + 1], sv[MAXC * PER + 1];
            if (!M.lens) { /* the scan's points are plain projections: read them two at a time, the same values */
                for (int j = 0; j <= steps; j++) {
                    const double x = 1.5 + (double)j / PER;
                    su[j] = axis ? across : x / cells, sv[j] = axis ? x / cells : across;
                }
                sample_points(im, M.H, su, sv, steps + 1, scan);
            }
            for (int j = 0; j <= steps && count < MAXC; j++) {
                const double x = 1.5 + (double)j / PER;
                double val;
                if (M.lens) {
                    const vec2 p = quad_line_point(&ln, x);
                    val = sample_bilinear(im, p.x, p.y);
                } else val = scan[j];
                if (j > 0 && (prev < mid) != (val < mid)) {
                    double xa = xprev, xb = x, va = prev, vb = val;
                    for (int it = 0; it < 5; it++) {
                        const double xm = 0.5 * (xa + xb);
                        const vec2 pm = quad_line_point(&ln, xm);
                        const double vm = sample_bilinear(im, pm.x, pm.y);
                        if ((vm < mid) == (va < mid)) xa = xm, va = vm; else xb = xm, vb = vm;
                    }
                    xs[count] = vb != va ? xa + (mid - va) / (vb - va) * (xb - xa) : 0.5 * (xa + xb);
                    const vec2 p0 = quad_line_point(&ln, xs[count] - 0.5);
                    const vec2 p1 = quad_line_point(&ln, xs[count] + 0.5);
                    px[count] = vec_length(p1.x - p0.x, p1.y - p0.y);
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

/* Where the dark-to-light edge crosses half level on a short ray through (x, y), as signed distance along the ray. */
static int edge_crossing(const image_view *view, double x, double y, double nx, double ny, double *offset) {
    enum { N = 13 };
    const double start = -3.0, step = 0.5;
    double v[N], xs[N], ys[N], best = 0;
    int peak = -1;
    for (int i = 0; i < N; i++) xs[i] = x + (start + step * i) * nx, ys[i] = y + (start + step * i) * ny;
    sample_at(view, xs, ys, N, v);
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
        if (pass == 2) break;
        double residual[NAPA_LOCAL_EDGE_SAMPLES], sorted[NAPA_LOCAL_EDGE_SAMPLES];
        for (int i = 0; i < n; i++) { /* only the middle residual is read: select it, don't sort */
            const double u = t[i] / scale;
            residual[i] = fabs(y[i] - (c[0] + u * (c[1] + u * c[2])));
        }
        double cutoff = 1.5 * select_kth(residual, n, n / 2, sorted);
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
        const double len = vec_length(ends[e].x - c.x, ends[e].y - c.y);
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
    if (!(fabs(a) < 6 && fabs(b) < 6) || !(vec_length(dx, dy) < 4)) return 0;
    out->x = c.x + dx, out->y = c.y + dy;
    return 1;
}

/* Where a strongly curved tag touches a neighbour at a corner (an X-corner on a board seen through a wide
 * lens), lines fitted along the whole side miss the corner. Gradients around an X-corner point at it, so
 * solve for the spot every nearby gradient is orthogonal to the offset from. Returns 1 for a nonsingular fit
 * that stays within the search window. */
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
        double ex = b.x - a.x, ey = b.y - a.y, len = vec_length(ex, ey);
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

/* A visible-only fit excludes clamped cell samples and also measures fallback levels from those left. */
static void edge_light_model(const family_layout *L, const double H[9], const vec2 q[4],
                             const double *ring, const double *border, edge_levels *levels, const image_view *visible) {
    double bp[64][3], wp[68][3], b[3], w[3];
    int nb = 0, nw = 0, bi = 0, wi = 0, W = L->width;
    levels->cx = levels->cy = levels->scale = 0;
    for (int k = 0; k < 4; k++) {
        levels->cx += q[k].x * 0.25; levels->cy += q[k].y * 0.25;
        levels->scale += vec_length(q[k].x-q[(k+1)&3].x, q[k].y-q[(k+1)&3].y) * 0.25;
    }
    if (!(levels->scale > 0)) return;
    for (int r = 0; r < W; r++) for (int c = 0; c < W; c += (r == 0 || r == W - 1) ? 1 : W - 1) {
        vec2 pos = project(H, (c+0.5)/L->cells, (r+0.5)/L->cells);
        const double value = border[bi++];
        if (visible != NULL && !sample_is_observed(visible, pos.x, pos.y)) continue;
        bp[nb][0] = (pos.x-levels->cx)/levels->scale; bp[nb][1] = (pos.y-levels->cy)/levels->scale;
        bp[nb][2] = value; nb++;
    }
    for (int i = 0; i < W + 2; i++) for (int side = 0; side < 2; side++) {
        vec2 pos = project(H, (i-0.5)/L->cells, (side ? W+0.5 : -0.5)/L->cells);
        const double value = ring[wi++];
        if (visible != NULL && !sample_is_observed(visible, pos.x, pos.y)) continue;
        wp[nw][0] = (pos.x-levels->cx)/levels->scale; wp[nw][1] = (pos.y-levels->cy)/levels->scale;
        wp[nw][2] = value; nw++;
    }
    for (int i = 1; i < W + 1; i++) for (int side = 0; side < 2; side++) {
        vec2 pos = project(H, (side ? W+0.5 : -0.5)/L->cells, (i-0.5)/L->cells);
        const double value = ring[wi++];
        if (visible != NULL && !sample_is_observed(visible, pos.x, pos.y)) continue;
        wp[nw][0] = (pos.x-levels->cx)/levels->scale; wp[nw][1] = (pos.y-levels->cy)/levels->scale;
        wp[nw][2] = value; nw++;
    }
    if (visible != NULL) {
        if (nb < 3 || nw < 3) return;
        double values[68];
        for (int i = 0; i < nb; i++) values[i] = bp[i][2];
        qsort(values, (size_t)nb, sizeof *values, compare_doubles);
        levels->black = 0.5 * (values[(nb - 1) / 2] + values[nb / 2]);
        for (int i = 0; i < nw; i++) values[i] = wp[i][2];
        qsort(values, (size_t)nw, sizeof *values, compare_doubles);
        levels->white = 0.5 * (values[(nw - 1) / 2] + values[nw / 2]);
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

static NAPA_NOINLINE void refresh_edge_light(const image_view *im, const vec2 q[4], edge_levels *levels, int visible_only) {
    double H[9], grid[256], border[64], ring[68], black, white;
    if (visible_only) {
        levels->black = 255, levels->white = 0;
        levels->has_model = levels->model_pending = 0;
    }
    square_to_quad(q, H);
    const int dark = read_border(levels->model_family, im, H, 0, grid, border, &black, &white, ring);
    if (visible_only || (dark >= 0 && white > black))
        edge_light_model(levels->model_family, H, q, ring, border, levels, visible_only ? im : NULL);
}

/* Keep a decode attempt for the soft decoder if it could be a tiny tag. The cells are the unsharpened samples. */
static void soft_capture(napa_detector *det, const vec2 q[4], double black, double white, const double *grid) {
    const double contrast = white - black;
    if (contrast < NAPA_SOFT_MIN_CONTRAST || contrast > NAPA_SOFT_MAX_CONTRAST) return;
    double side = 0;
    for (int k = 0; k < 4; k++) side += 0.25 * vec_length(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
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
 * side's border level (side_level), the tag's black and white levels, and its matched code in q's
 * observed orientation (code_out). */
static int decode_quad(napa_detector *det, const image_view *im, vec2 q[4],
                       const napa_params *params, int slack, napa_detection *out, double side_level[4],
                       edge_levels *levels, uint64_t *code_out) {
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
    if (params->sharpening_x100 > 0) {
        memcpy(unsharpened, grid, sizeof(double) * (size_t)(G * G));
        double k = params->sharpening_x100 / 100.0;
        for (int r = 0; r < G; r++) {
            for (int c = 0; c < G; c++) {
                double v = unsharpened[r * G + c], lap = 4 * v;
                lap -= r > 0 ? unsharpened[(r - 1) * G + c] : v;
                lap -= r < G - 1 ? unsharpened[(r + 1) * G + c] : v;
                lap -= c > 0 ? unsharpened[r * G + c - 1] : v;
                lap -= c < G - 1 ? unsharpened[r * G + c + 1] : v;
                grid[r * G + c] = v + k * lap;
            }
        }
    }

    uint64_t code = 0;
    for (int k = 0; k < L->nbits; k++) code = (code << 1) | (grid[L->cell[k]] > thr ? 1u : 0u);
    int max_hamming = params->max_hamming < L->max_hamming ? params->max_hamming : L->max_hamming;
    int soft = L->soft_hamming > max_hamming ? L->soft_hamming : max_hamming;
    int best = 64;
    int bi = nearest_code(det, code, (slack ? soft : max_hamming) + slack, &best);
    if (bi < 0 && slack == 0 && params->sharpening_x100 > 0 && L->piece_hamming >= 0) {
        /* Sharpening can flip a cell that read correctly before it. Only an exact code counts here: a corrected one
         * would raise the chance of a false match for a very small gain. Families with short codes don't retry. */
        uint64_t unsharpened_code = 0;
        for (int k = 0; k < L->nbits; k++) unsharpened_code = (unsharpened_code << 1) | (unsharpened[L->cell[k]] > thr ? 1u : 0u);
        int unsharpened_best = 64, match = nearest_code(det, unsharpened_code, 0, &unsharpened_best);
        if (match >= 0) bi = match, best = unsharpened_best, code = unsharpened_code, memcpy(grid, unsharpened, sizeof(double) * (size_t)(G * G));
    }
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
    if (code_out != NULL) *code_out = L->codes[bi];
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

/* Points on the inner edge of the square's border along one side (0 top, v = 0, then clockwise), for the square
 * mapped by H: where the decoded code (as observed) has a white cell just inside the border, the rising step from
 * the border's black to that white, searched t0..t1 cells in from the outer edge. Each profile must start dark
 * and end light, so a window that starts inside the cell (the frame cut off the edge there too) is skipped. Writes
 * up to max points and their depths in cells (depth, if not NULL); returns the count, and in *cells how many
 * white cells gave one. */
static int inner_edge(const family_layout *L, const image_view *im, const double H[9], int side, uint64_t code,
                      double black, double white, double t0, double t1, double *px, double *py, double *depth,
                      int max, int *cells) {
    enum { N = 25, PER_CELL = 3 };
    const int W = L->width, o = L->outer, G = L->grid;
    int n = 0;
    *cells = 0;
    for (int c = 1; c < W - 1 && n + PER_CELL <= max; c++) {
        int row = side == 0 ? 1 : side == 2 ? W - 2 : c, col = side == 3 ? 1 : side == 1 ? W - 2 : c;
        int g = (row + o) * G + col + o, white_cell = 0;
        for (int k = 0; k < L->nbits; k++)
            if (L->cell[k] == g) white_cell = (int)(code >> (L->nbits - 1 - k) & 1);
        if (!white_cell) continue;
        int got = 0;
        for (int i = 0; i < PER_CELL; i++) {
            double a = (c + (i + 1.0) / (PER_CELL + 1)) / L->cells, prof[N];
            vec2 pts[N];
            for (int j = 0; j < N; j++) {
                double t = (t0 + (t1 - t0) * j / (N - 1)) / L->cells;
                double u = side == 0 || side == 2 ? a : side == 1 ? 1 - t : t;
                double v = side == 1 || side == 3 ? a : side == 2 ? 1 - t : t;
                pts[j] = project(H, u, v);
                prof[j] = sample_bilinear(im, pts[j].x, pts[j].y);
            }
            if (prof[0] > black + 0.3 * (white - black) || prof[N - 1] < black + 0.6 * (white - black)) continue;
            double jc = lobe_centroid(prof, 1, N - 2, 0);
            if (jc < 0) continue;
            int j0 = (int)jc;
            double f = jc - j0;
            px[n] = pts[j0].x + (pts[j0 + 1].x - pts[j0].x) * f;
            py[n] = pts[j0].y + (pts[j0 + 1].y - pts[j0].y) * f;
            if (depth != NULL) depth[n] = t0 + (t1 - t0) * jc / (N - 1);
            n++, got = 1;
        }
        *cells += got;
    }
    return n;
}

/* A side taken from the frame edge leaves its two corners short of the true ones by however much of the border
 * the frame cuts off. The border's inner edge still shows against the white cells next to it, one cell in from
 * the true side: fit a line to it there, and extrapolate the square from that line and the other three sides.
 * Twice: the second pass samples at the cells' corrected positions. Then check the result against what it didn't
 * use: every other side's inner edge must lie a cell in from its outer edge, and the side opposite each cut one
 * must show it. A tilted side that also runs off the frame was fitted to the wrong edge, and a cut of more than a
 * cell, with a row of dark cells inside it that passed for the border, puts the square a cell off: both fail.
 * Returns 0 if a cut side shows its inner edge at fewer than four points or two white cells, the check fails, or a
 * corner would move by more than a cell and a quarter. */
static int extend_cut_sides(const family_layout *L, const image_view *im, const vec2 q[4], int cut, uint64_t code,
                            double black, double white, vec2 out[4]) {
    enum { MAXPTS = 48 };
    double lx[4], ly[4], dx[4], dy[4], inset[4], cell_px = 0, px[MAXPTS], py[MAXPTS], depth[MAXPTS];
    for (int k = 0; k < 4; k++) {
        vec2 a = q[k], b = q[(k + 1) & 3];
        double len = vec_length(b.x - a.x, b.y - a.y);
        lx[k] = a.x, ly[k] = a.y, dx[k] = (b.x - a.x) / len, dy[k] = (b.y - a.y) / len, inset[k] = 0;
        cell_px += len / (4.0 * L->width);
    }
    vec2 cur[4];
    memcpy(cur, q, sizeof cur);
    double H[9];
    for (int pass = 0; pass < 2; pass++) {
        square_to_quad(cur, H);
        /* Depth inward from the outer edge, in cells. The first pass's quad is squeezed toward the frame edge by
         * the cut (up to 0.9 cell): its window holds the inner edge, but not the next cell's rising edge. */
        double t0 = pass ? 0.55 : 0.1, t1 = pass ? 1.45 : 1.35;
        for (int side = 0; side < 4; side++) {
            if (!(cut >> side & 1)) continue;
            int cells, n = inner_edge(L, im, H, side, code, black, white, t0, t1, px, py, NULL, MAXPTS, &cells);
            if (cells < 2 || n < 4) return 0;
            /* Total-least-squares line, then once more without points off it (blur from a dark neighbour). The
             * corrected pass must agree: most of its points on one line. */
            const int measured = n;
            for (int fit = 0; fit < 2; fit++) {
                double mx = 0, my = 0, sxx = 0, syy = 0, sxy = 0;
                for (int i = 0; i < n; i++) mx += px[i] / n, my += py[i] / n;
                for (int i = 0; i < n; i++) {
                    double ex = px[i] - mx, ey = py[i] - my;
                    sxx += ex * ex, syy += ey * ey, sxy += ex * ey;
                }
                double th = 0.5 * atan2(2 * sxy, sxx - syy);
                lx[side] = mx, ly[side] = my, dx[side] = cos(th), dy[side] = sin(th);
                int kept = 0;
                double limit = 0.1 * cell_px > 0.5 ? 0.1 * cell_px : 0.5;
                for (int i = 0; i < n; i++)
                    if (fabs((px[i] - mx) * dy[side] - (py[i] - my) * dx[side]) <= limit) px[kept] = px[i], py[kept++] = py[i];
                if (pass && 4 * kept < 3 * measured) return 0;
                if (kept == n || kept < 4) break;
                n = kept;
            }
            inset[side] = 1.0 / L->cells;
        }
        /* The four lines bound the square inset by a cell on each cut side; map the whole square through it. */
        vec2 in[4];
        for (int k = 0; k < 4; k++) {
            int p = (k + 3) & 3;
            double cross = dx[p] * dy[k] - dy[p] * dx[k];
            if (fabs(cross) < 0.2) return 0;
            double t = ((lx[k] - lx[p]) * dy[k] - (ly[k] - ly[p]) * dx[k]) / cross;
            in[k].x = lx[p] + t * dx[p], in[k].y = ly[p] + t * dy[p];
        }
        double u0 = inset[3], u1 = 1 - inset[1], v0 = inset[0], v1 = 1 - inset[2], Hi[9];
        square_to_quad(in, Hi);
        for (int k = 0; k < 4; k++) {
            double u = (k == 1 || k == 2) ? 1 : 0, v = k >= 2 ? 1 : 0;
            cur[k] = project(Hi, (u - u0) / (u1 - u0), (v - v0) / (v1 - v0));
        }
    }
    square_to_quad(cur, H);
    for (int side = 0; side < 4; side++) {
        if (cut >> side & 1) continue;
        int cells, n = inner_edge(L, im, H, side, code, black, white, 0.55, 1.45, px, py, depth, MAXPTS, &cells);
        if (cells < 2) {
            if (cut >> ((side + 2) & 3) & 1) return 0; /* the one side that tests the extrapolation */
            /* Unchecked, and within a cell of the frame edge: it may run off the frame too. */
            for (int e = 0; e < 2; e++) {
                vec2 c = cur[(side + e) & 3];
                if (c.x < cell_px - 0.5 || c.y < cell_px - 0.5 || c.x > im->w - 0.5 - cell_px || c.y > im->h - 0.5 - cell_px)
                    return 0;
            }
            continue;
        }
        for (int i = 1; i < n; i++) /* median depth: a point blurred by a dark neighbour can't sink it */
            for (int j = i; j > 0 && depth[j - 1] > depth[j]; j--) {
                double t = depth[j];
                depth[j] = depth[j - 1], depth[j - 1] = t;
            }
        double mid = n & 1 ? depth[n / 2] : 0.5 * (depth[n / 2 - 1] + depth[n / 2]);
        if (fabs(mid - 1.0) > 0.1) return 0;
    }
    for (int k = 0; k < 4; k++)
        if (!(vec_length(cur[k].x - q[k].x, cur[k].y - q[k].y) <= 1.25 * cell_px)) return 0;
    memcpy(out, cur, sizeof cur);
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
    int x0 = c->work.rows.x0, x1 = c->work.rows.x1, y = c->work.rows.y;
    for (int k = 0; k < 16; k++) {
        int x = comp_dirs[k][0] > 0 ? x1 : x0;
        int v = comp_dirs[k][0] * x + comp_dirs[k][1] * y;
        int up = v > c->extv[k];
        c->extv[k] = up ? v : c->extv[k];
        c->ext[k][0] = up ? x : c->ext[k][0];
        c->ext[k][1] = up ? y : c->ext[k][1];
    }
}

/* Runs arrive in raster order. extv starts at INT32_MIN and the pending row at -1. */
static void comp_add_run(component *c, const napa_run *r) {
    if (r->y != c->work.rows.y) {
        if (c->work.rows.y >= 0) comp_flush_row(c);
        c->work.rows.y = r->y, c->work.rows.x0 = r->x0;
    }
    c->work.rows.x1 = r->x1;
    c->count += r->x1 - r->x0 + 1;
}

/* Row-pair runs (label_row_pair). The first dark bit at or after x, and the last at or before x, of a
 * row known to have one there. */
static int first_bit(const uint64_t *b, int x) {
    int w = x >> 6;
    uint64_t v = b[w] & (~(uint64_t)0 << (x & 63));
    while (!v) v = b[++w];
    return 64 * w + napa_ctz64(v);
}

static int last_bit(const uint64_t *b, int x) {
    int w = x >> 6;
    uint64_t v = b[w] & (~(uint64_t)0 >> (63 - (x & 63)));
    while (!v) v = b[--w];
    return 64 * w + 63 - napa_clz64(v);
}

/* Samples x0..x1 dark in both rows a and b. */
static int bits_count_both(const uint64_t *a, const uint64_t *b, int x0, int x1) {
    int w = x0 >> 6, last = x1 >> 6, count = 0;
    uint64_t v = a[w] & b[w] & (~(uint64_t)0 << (x0 & 63));
    for (; w < last; w++, v = a[w] & b[w]) count += napa_popcount64(v);
    return count + napa_popcount64(v & (~(uint64_t)0 >> (63 - (x1 & 63))));
}

/* Both rows of a pair, y (x0t..x1t) and y + 1 (x0b..x1b), in one update: comp_flush_row on the top row, then on
 * the bottom row. Per direction the bottom row's value replaces the top row's only when strictly greater, so the
 * winner of the two is the one the two strict updates in raster order would keep. */
static void comp_flush_two(component *c, int y, int x0t, int x1t, int x0b, int x1b) {
    for (int k = 0; k < 16; k++) {
        const int dx = comp_dirs[k][0], dy = comp_dirs[k][1];
        const int xt = dx > 0 ? x1t : x0t, xb = dx > 0 ? x1b : x0b;
        const int vt = dx * xt + dy * y, vb = dx * xb + dy * (y + 1);
        const int lower = vb > vt;
        const int v = lower ? vb : vt, x = lower ? xb : xt, yy = y + lower;
        const int up = v > c->extv[k];
        c->extv[k] = up ? v : c->extv[k];
        c->ext[k][0] = up ? x : c->ext[k][0];
        c->ext[k][1] = up ? yy : c->ext[k][1];
    }
}

/* Fold the pending row pair into the extremes: its top row, then its bottom row, as comp_add_run's
 * flushes would. A negative x1 is -x1 - 1 of a union run whose last dark sample is still to find. */
static void comp_flush_pair(component *c, const uint64_t *bits, int stride) {
    if (c->work.rows.pair_y < 0) return;
    const uint64_t *t = bits + (size_t)c->work.rows.pair_y * stride;
    const int y = c->work.rows.pair_y;
    const int tx0 = c->work.rows.top_x0, bx0 = c->work.rows.bot_x0;
    const int top_x1 = c->work.rows.top_x1, bot_x1 = c->work.rows.bot_x1;
    const int tx1 = tx0 < 0 ? 0 : top_x1 >= 0 ? top_x1 : last_bit(t, -top_x1 - 1);
    const int bx1 = bx0 < 0 ? 0 : bot_x1 >= 0 ? bot_x1 : last_bit(t + stride, -bot_x1 - 1);
    if (tx0 >= 0 && bx0 >= 0) {
        comp_flush_two(c, y, tx0, tx1, bx0, bx1);
        c->work.rows.y = y + 1, c->work.rows.x0 = bx0, c->work.rows.x1 = bx1;
        return;
    }
    if (tx0 >= 0) c->work.rows.y = y, c->work.rows.x0 = tx0, c->work.rows.x1 = tx1;
    else if (bx0 >= 0) c->work.rows.y = y + 1, c->work.rows.x0 = bx0, c->work.rows.x1 = bx1;
    else return;
    comp_flush_row(c);
}

/* comp_add_run for row-pair runs, which arrive pair by pair: runs dark in the top row left to right,
 * then the rest left to right (label_row_pair). A union run dark in one row only is exactly that row's
 * run. One dark in both counts its samples as its length plus the columns dark in both rows, and its
 * rows' extents come from the dark bits: the top row's last only at the flush. Runs dark in the bottom
 * row arrive in two sweeps, so its extents take the outermost. */
static NAPA_FORCE_INLINE void comp_add_pair_run(component *c, const napa_run *r, const uint64_t *bits, int stride) {
    const int pair_y = r->y & ~1;
    if (pair_y != c->work.rows.pair_y) {
        comp_flush_pair(c, bits, stride);
        c->work.rows.pair_y = pair_y, c->work.rows.top_x0 = c->work.rows.bot_x0 = -1;
    }
    if (r->rows == 3) {
        const uint64_t *t = bits + (size_t)r->y * stride, *b = t + stride;
        c->count += r->x1 - r->x0 + 1 + bits_count_both(t, b, r->x0, r->x1);
        if (c->work.rows.top_x0 < 0) c->work.rows.top_x0 = first_bit(t, r->x0);
        c->work.rows.top_x1 = -r->x1 - 1;
        /* The first such run is the leftmost dark in the bottom row so far: only top-row runs came before. */
        if (c->work.rows.bot_x0 < 0) c->work.rows.bot_x0 = first_bit(b, r->x0);
        c->work.rows.bot_x1 = -r->x1 - 1;
        return;
    }
    c->count += r->x1 - r->x0 + 1;
    if (!(r->y & 1) && r->rows == 1) {
        if (c->work.rows.top_x0 < 0) c->work.rows.top_x0 = r->x0;
        c->work.rows.top_x1 = r->x1;
    } else if (c->work.rows.bot_x0 < 0) {
        c->work.rows.bot_x0 = r->x0, c->work.rows.bot_x1 = r->x1;
    } else {
        /* Runs are disjoint: compare with the range of the run holding the current extreme. */
        if (r->x0 < c->work.rows.bot_x0) c->work.rows.bot_x0 = r->x0;
        if (r->x1 > (c->work.rows.bot_x1 >= 0 ? c->work.rows.bot_x1 : -c->work.rows.bot_x1 - 1)) c->work.rows.bot_x1 = r->x1;
    }
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

/* Level coordinates stay below NAPA_MAX_SIDE, so a box's minx has a spare top bit: it marks a component seen this stage. */
#define PIECE_SEEN 0x8000u
typedef char napa_side_fits_15_bits[NAPA_MAX_SIDE < 0x8000 ? 1 : -1]; /* coordinates leave PIECE_SEEN's bit free */

/* Split-piece eligibility uses box density, independently of the whole quad's fill. */
static int box_is_sparse(int count, int width, int height) {
    return (double)count / ((double)width * height) < 0.2;
}

/* Large fine-grid outlines can merge into the background at coarser levels.
 * Defer them until ordinary proposals finish, retaining their finer outlines. */
static int dense_whole_needs_deferral(int base, int s, int last, int width, int height) {
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
    det->fine_wholes[index].shape = *shape;
    det->fine_wholes[index].stride = stride;
    det->fine_wholes[index].terminal = terminal;
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
    c->work.rows.y = -1;
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

/* Coverage is queried for every proposal. Compute each accepted centre once, preserving
 * corner summation order; caches have the same logical lengths as their output buffers. */
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

struct gather_job;
typedef struct {
    napa_detector *det;
    const uint8_t *L;
    int lw, lh, cols, min_contrast, cap;
    double sigmas;
    uint32_t (*all)[64]; /* per region row: the sum of its regions' histograms */
    const struct gather_job *rake; /* threads: also rake the rows the floors read, and the rest (contrast_floors_raked) */
} floor_job;

/* The noise floors of region row ry, and that row's share of the frame histogram. */
static void floor_row_regions(void *arg, int ry, int worker) {
    (void)worker;
    floor_job *job = (floor_job *)arg;
    const int lw = job->lw, lh = job->lh, cols = job->cols;
    uint32_t *all = job->all[ry];
    memset(all, 0, 64 * sizeof(uint32_t));
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
            pair_diffs(job->L + (size_t)y * lw + x0, pairs, d);
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
        job->det->floors[ry * cols + rx] = (uint8_t)(n >= 16 ? floor_from(hist, n, job->min_contrast, job->sigmas, job->cap) : 0);
    }
}

static void invert_bytes(uint8_t *p, size_t n);
typedef struct gather_job {
    napa_detector *det;
    const uint8_t *img;
    int stride, lw, lh, s, invert;
} gather_job;

/* Chunk k (64 rows) of the rows the noise floors skipped (all but every 8th); inverted for reversed families. */
/* On ARM, ask for the next source row while this one is gathered: on a wide frame each row is a fresh stretch of
 * memory. A prefetch is only a hint; it reads nothing and never faults. */
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(NAPA_NO_SIMD)
#define ROW_PREFETCH(p) (__builtin_prefetch((p), 0, 0), __builtin_prefetch((p) + 64, 0, 0))
#else
#define ROW_PREFETCH(p) ((void)0)
#endif

static void gather_rest(void *arg, int k, int worker) {
    (void)worker;
    gather_job *job = (gather_job *)arg;
    const int y1 = (k + 1) * 64 < job->lh ? (k + 1) * 64 : job->lh;
    for (int y = k * 64; y < y1; y++) {
        uint8_t *row = job->det->level + (size_t)y * job->lw;
        if (y + 1 < job->lh) ROW_PREFETCH(job->img + (size_t)(y + 1) * job->s * job->stride);
        if (y & 7) gather_row(row, job->img + (size_t)y * job->s * job->stride, job->lw, job->s, job->det->ssse3);
        if (job->invert) invert_bytes(row, (size_t)job->lw);
    }
}

/* contrast_floors_raked's job ry + 1: region row ry's every-8th rows raked, then its floors, then those rows inverted
 * for reversed families. Job 0 rakes the level's other rows in chunks while floor jobs are left, and the threshold's
 * raker carries on from the next chunk (rake_band_job). Where job 0 stops depends only on timing: either way every
 * row is raked before a band reads it. */
static void floor_rake_job(void *arg, int ry, int worker) {
    floor_job *job = (floor_job *)arg;
    const gather_job *g = job->rake;
    napa_detector *det = g->det;
    if (ry-- == 0) {
        napa_pool *pool = det->pool;
        for (;;) {
            if (pool != NULL) napa_mutex_lock(&pool->lock);
            const int left = det->rake_floors_left;
            if (pool != NULL) napa_mutex_unlock(&pool->lock);
            const int c = det->rake_next;
            if (left == 0 || c * RAKE_CHUNK >= job->lh) return;
            rake_rows(det, job->lw, c * RAKE_CHUNK, (c + 1) * RAKE_CHUNK < job->lh ? (c + 1) * RAKE_CHUNK : job->lh);
            pool_mark(pool, &det->rake_done[c]);
            det->rake_next = c + 1;
        }
    }
    const int y0 = ry * NAPA_REGION, y1 = y0 + NAPA_REGION < job->lh ? y0 + NAPA_REGION : job->lh;
    for (int y = y0; y < y1; y += 8)
        gather_row(g->det->level + (size_t)y * g->lw, g->img + (size_t)y * g->s * g->stride, g->lw, g->s, g->det->ssse3);
    floor_row_regions(arg, ry, worker);
    if (g->invert)
        for (int y = y0; y < y1; y += 8) invert_bytes(g->det->level + (size_t)y * g->lw, (size_t)g->lw);
    if (det->pool != NULL) napa_mutex_lock(&det->pool->lock);
    det->rake_floors_left--;
    if (det->pool != NULL) napa_mutex_unlock(&det->pool->lock);
}

/* Each region's floor into det->floors (from pairs of samples on every 8th row), at most the frame's,
 * which is returned; a region too small to measure takes the frame's. Returns -1 if out of memory. With rake set
 * (threads), the same pool run rakes the every-8th rows and, while floor jobs are left, other rows chunk by chunk,
 * inverted for reversed families; the threshold's raker samples the rest. */
static int contrast_floors_raked(napa_detector *det, const uint8_t *L, int lw, int lh, int min_contrast, double sigmas,
                                 int cap, const gather_job *rake) {
    int cols = (lw + NAPA_REGION - 1) / NAPA_REGION, rows = (lh + NAPA_REGION - 1) / NAPA_REGION;
    if ((size_t)cols * rows > det->floor_alloc) {
        uint8_t *p = (uint8_t *)realloc(det->floors, (size_t)cols * rows);
        if (p == NULL) return -1;
        det->floors = p;
        det->floor_alloc = (size_t)cols * rows;
    }
    if ((size_t)rows > det->floor_hist_rows) {
        uint32_t (*p)[64] = (uint32_t (*)[64])realloc(det->floor_hist, (size_t)rows * sizeof *p);
        if (p == NULL) return -1;
        det->floor_hist = p;
        det->floor_hist_rows = (size_t)rows;
    }
    det->floor_cols = cols, det->floor_rows = rows;
    floor_job job = {det, L, lw, lh, cols, min_contrast, cap, sigmas, det->floor_hist, rake};
    /* Region rows are independent and the frame histogram is a sum, so threads change nothing. */
    if (rake) {
        /* The floors' rows, and as many of the rest as one job rakes meanwhile; threshold_level rakes the others
         * alongside its bands (rake_band_job). */
        const int chunks = (lh + RAKE_CHUNK - 1) / RAKE_CHUNK;
        if ((size_t)chunks > det->rake_done_capacity) {
            int *p = (int *)realloc(det->rake_done, (size_t)chunks * sizeof(int));
            if (p == NULL) return -1;
            det->rake_done = p, det->rake_done_capacity = (size_t)chunks;
        }
        memset(det->rake_done, 0, (size_t)chunks * sizeof(int));
        det->rake_img = rake->img, det->rake_stride = rake->stride, det->rake_s = rake->s, det->rake_invert = rake->invert;
        det->rake_next = 0, det->rake_floors_left = rows;
        det->rake_deferred = 1;
        pool_run(det->pool, floor_rake_job, &job, rows + 1); /* index 0 rakes ahead (floor_rake_job) */
    }
    else pool_run(pool_for(det, (size_t)lw * lh), floor_row_regions, &job, rows);
    uint32_t all[64] = {0}, n_all = 0;
    for (int ry = 0; ry < rows; ry++)
        for (int k = 0; k < 64; k++) all[k] += det->floor_hist[ry][k], n_all += det->floor_hist[ry][k];
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

static int contrast_floors(napa_detector *det, const uint8_t *L, int lw, int lh, int min_contrast, double sigmas, int cap) {
    return contrast_floors_raked(det, L, lw, lh, min_contrast, sigmas, cap, NULL);
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
    const double length = vec_length(b.x - a.x, b.y - a.y);
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
    double profile[MAX_OFFSETS], xs[MAX_OFFSETS], ys[MAX_OFFSETS], row[MAX_OFFSETS];
    for (int j = 0; j < offsets; j++) profile[j] = 0;
    for (int i = 0; i < rows; i++) {
        const double d = 0.2 * length + row_gap * i;
        for (int j = 0; j < offsets; j++) {
            const double o = -1.3 * c + dp * j, x = a.x + ex * d + px * o, y = a.y + ey * d + py * o;
            if (x < 1 || y < 1 || x > view->w - 2 || y > view->h - 2) return 0;
            xs[j] = x, ys[j] = y;
        }
        sample_at(view, xs, ys, offsets, row);
        for (int j = 0; j < offsets; j++) profile[j] += row[j];
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
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
        float32x4_t v0 = vdupq_n_f32(0), v1 = vdupq_n_f32(0);
        float32x2_t v2 = vdup_n_f32(0);
        for (int k = 0; k < 10; k++) {
            const float weight = blur[col][k];
            const float32x4_t w = vdupq_n_f32(weight);
            v0 = vaddq_f32(v0, vmulq_f32(vld1q_f32(by_column[k]), w));
            v1 = vaddq_f32(v1, vmulq_f32(vld1q_f32(by_column[k] + 4), w));
            v2 = vadd_f32(v2, vmul_f32(vld1_f32(by_column[k] + 8), vdup_n_f32(weight)));
        }
        vst1q_f32(value, v0), vst1q_f32(value + 4, v1), vst1_f32(value + 8, v2);
#else
        memset(value, 0, sizeof value);
        for (int k = 0; k < 10; k++) {
            const float weight = blur[col][k];
            for (int r = 0; r < 10; r++) value[r] += by_column[k][r] * weight;
        }
#endif
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
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
            const float32x4_t w = vdupq_n_f32(weight);
            const float *pk = packed[k];
            int i = 0;
            for (; i + 4 <= columns; i += 4)
                vst1q_f32(values + i, vaddq_f32(vld1q_f32(values + i), vmulq_f32(w, vld1q_f32(pk + i))));
            for (; i < columns; i++) values[i] += weight * packed[k][i];
#else
            for (int i = 0; i < columns; i++) values[i] += weight * packed[k][i];
#endif
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

/* The tag's lattice at every step-th row and column, as read; points of every have-th row and column are already in
 * raw (have 0: none are). */
static void tone_read(const image_view *view, const vec2 q[4], int step, int have, double raw[NAPA_SOFT_AREA]) {
    double H[9], u[30], v[30], value[30];
    int column[30];
    square_to_quad(q, H);
    for (int r = 0; r < 30; r += step) {
        int n = 0;
        for (int c = 0; c < 30; c += step) {
            if (have && r % have == 0 && c % have == 0) continue;
            u[n] = soft_position(c) / 8, v[n] = soft_position(r) / 8, column[n++] = c;
        }
        sample_points(view, H, u, v, n, value); /* the values project and sample_bilinear give, two at a time */
        for (int i = 0; i < n; i++) raw[r * 30 + column[i]] = value[i];
    }
}

/* Every step-th row and column of a read lattice, standardized over those samples. */
static void tone_standardize(const double raw[NAPA_SOFT_AREA], int step, float y[NAPA_SOFT_AREA]) {
    double sum = 0, squares = 0;
    int n = 0;
    for (int r = 0; r < 30; r += step) {
        for (int c = 0; c < 30; c += step) {
            const double v = raw[r * 30 + c];
            y[r * 30 + c] = (float)v, sum += v, squares += v * v, n++;
        }
    }
    const double mean = sum / n, sigma = sqrt(squares / n - mean * mean);
#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    if (step == 1) {
        if (sigma > 1e-6) {
            const float64x2_t mv = vdupq_n_f64(mean), sv = vdupq_n_f64(sigma);
            for (int i = 0; i < NAPA_SOFT_AREA; i += 2) {
                const float32x2_t f = vld1_f32(y + i);
                const float64x2_t d = vdivq_f64(vsubq_f64(vcvt_f64_f32(f), mv), sv);
                vst1_f32(y + i, vcvt_f32_f64(d));
            }
        } else
            for (int i = 0; i < NAPA_SOFT_AREA; i++) y[i] = 0;
        return;
    }
#endif
    for (int r = 0; r < 30; r += step)
        for (int c = 0; c < 30; c += step) y[r * 30 + c] = sigma > 1e-6 ? (float)((y[r * 30 + c] - mean) / sigma) : 0;
}

/* The tag's lattice at every step-th row and column, standardized over those samples. */
static void tone_sample(const image_view *view, const vec2 q[4], int step, float y[NAPA_SOFT_AREA]) {
    double raw[NAPA_SOFT_AREA];
    tone_read(view, q, step, 0, raw);
    tone_standardize(raw, step, y);
}

/* Blur sweep along one axis, keeping the best fit so far. */
static void tone_sweep(const napa_detector *det, const float cells[10][10], const float y[NAPA_SOFT_AREA], int step, int vary_v, int *u, int *v,
                       double *best, double tone[3]) {
    float across[10][30];
    const int was_u = *u, was_v = *v;
    for (int b = 0; b < NAPA_TONE_SIGMAS; b++) {
        const int su = vary_v ? was_u : b, sv = vary_v ? b : was_v;
        const int incumbent = *best != HUGE_VAL && su == *u && sv == *v;
        if ((!vary_v && !incumbent) || b == 0) tone_across(cells, (const float (*)[10])det->tone_blur[su], step, across);
        if (incumbent) continue; /* this pair already produced *best; refitting cannot improve it */
        double c[3];
        const double sse = tone_fit(y, (const float (*)[30])across, (const float (*)[10])det->tone_blur[sv], step, c);
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
 * Returns -2 on allocation failure.
 * Size correction alone needs every side's whole edge, which is cheaper to check before the search; alignment doesn't. */
static int tone_measure_tag(napa_detector *det, const image_view *view, const napa_detection *d, double edge_width, int aligning, tone_measure *m) {
    vec2 q[4];
    m->center.x = m->center.y = m->mean_side = 0;
    for (int k = 0; k < 4; k++) q[k].x = d->corners[2 * k], q[k].y = d->corners[2 * k + 1], m->center.x += 0.25 * q[k].x, m->center.y += 0.25 * q[k].y;
    for (int k = 0; k < 4; k++) m->mean_side += 0.25 * vec_length(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
    if (m->mean_side < NAPA_TONE_MIN_SIDE || edge_width < NAPA_TONE_MIN_EDGE_CELLS * m->mean_side / 8 || signed_area(q) <= 0) return 0;
    if (strcmp(det->fam.f->name, "tag36h11") != 0) return 0;
    if (!tone_prepare(det)) return -2;
    const family_layout *L = &det->fam;
    float y[NAPA_SOFT_AREA];
    double raw[NAPA_SOFT_AREA]; /* the lattice is read once: the screen's third of it, then the rest */
    for (int r = 0; r < 10; r++) for (int c = 0; c < 10; c++) m->cells[r][c] = (r == 0 || r == 9 || c == 0 || c == 9) ? 1.0f : -1.0f;
    const uint64_t code = L->f->codes[d->id];
    for (int k = 0; k < 36; k++) m->cells[L->cell[k] / L->grid + 1][L->cell[k] % L->grid + 1] = (code >> (35 - k) & 1) ? 1.0f : -1.0f;
    int u = NAPA_TONE_START, v = NAPA_TONE_START;
    double best = HUGE_VAL, tone[3] = {0, 0, 0};
    /* A few samples and blurs are enough to see that a tone curve is straight, which is most tags. */
    {
        float across[10][30];
        tone_across((const float (*)[10])m->cells, (const float (*)[10])det->tone_blur[NAPA_TONE_START], 3, across);
        tone_read(view, q, 3, 0, raw);
        tone_standardize(raw, 3, y);
        for (int b = 2; b <= 6; b += 2) {
            double c[3];
            const double sse = tone_fit(y, (const float (*)[30])across, (const float (*)[10])det->tone_blur[b], 3, c);
            if (sse < best) best = sse, memcpy(tone, c, sizeof tone);
        }
    }
    /* A screen reading above the limit means a straight tone, and only wide edges then have a size bias worth measuring. */
    m->straight_only = fabs(tone[1] + tone[2]) > 1e-9 && tone[2] / (tone[1] + tone[2]) > NAPA_TONE_SCREEN_CURVATURE;
    if (m->straight_only && !(edge_width >= NAPA_TONE_STRAIGHT_EDGE_CELLS * m->mean_side / 8)) return 0;
    m->sides_visible = 1;
    if (!aligning && !tone_sides_visible(view, q, m)) return 0;
    tone_read(view, q, 1, 3, raw);
    tone_standardize(raw, 2, y);
    best = HUGE_VAL, u = NAPA_TONE_START, v = NAPA_TONE_START;
    /* If the second sweep keeps u, a third would repeat the first exactly. */
    for (int round = 0, kept_u = u; round < 3 && (round < 2 || u != kept_u); round++) {
        if (round == 1) kept_u = u;
        tone_sweep(det, (const float (*)[10])m->cells, y, 2, !(round & 1), &u, &v, &best, tone);
    }
    /* The search ran on a quarter of the lattice; judge its answer on all of it. */
    tone_standardize(raw, 1, y);
    {
        float across[10][30];
        tone_across((const float (*)[10])m->cells, (const float (*)[10])det->tone_blur[u], 1, across);
        best = tone_fit(y, (const float (*)[30])across, (const float (*)[10])det->tone_blur[v], 1, tone);
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
            float gu[NAPA_SOFT_AREA], gv[NAPA_SOFT_AREA];
            for (int r = 0; r < 30; r++)
                for (int c = 0; c < 30; c++) {
                    const int i = r * 30 + c, cl = c > 0 ? c - 1 : c, cr = c < 29 ? c + 1 : c, ru = r > 0 ? r - 1 : r, rd = r < 29 ? r + 1 : r;
                    gu[i] = (g[r * 30 + cr] - g[r * 30 + cl]) * 24.0f / (cr - cl);
                    gv[i] = (g[rd * 30 + c] - g[ru * 30 + c]) * 24.0f / (rd - ru);
                }
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
            const double dx = x[2 * k] - origin[2 * k], dy = x[2 * k + 1] - origin[2 * k + 1], len = vec_length(dx, dy);
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

/* A tag only a little bigger than the finest level's sampling can come apart there: its thin black border falls
 * between samples, and the candidate is a fragment of the tag, whose quad fits nothing that reads. Sampling every
 * pixel keeps the border whole. So when a small candidate's code doesn't read, regrow it at full resolution: threshold
 * a window around it at the window's own midpoint, follow the dark pixels connected to the fragment, and take that
 * component's extremes along the 16 directions. Returns 0 when it can't: no contrast, no dark pixel at the fragment,
 * or a component reaching the window's edge (merged with its surroundings, which a tag's quiet zone prevents).
 * Returns -2 on allocation failure. */
#define NAPA_REGROW_MAX_PX 28.0 /* the largest candidate regrown, in pixels: bigger tags keep their border at the finest level */
#define NAPA_REGROW_WINDOW 128  /* the largest window side */
/* Regrowths a pass may make. A frame needs a few to a few dozen; a 4K frame of tiny unreadable shapes can raise
 * thousands, at a few microseconds each, and the budget keeps that from multiplying its time. */
#define NAPA_REGROW_BUDGET 32

static int regrow_small(napa_detector *det, const image_view *view, const vec2 rough[4], double side, int min_contrast,
                        component *c, int *ox, int *oy) {
    if (det->regrow_left <= 0) return 0;
    det->regrow_left--;
    if (det->regrow_stack == NULL) {
        det->regrow_stack = (int32_t *)malloc(sizeof(int32_t) * NAPA_REGROW_WINDOW * NAPA_REGROW_WINDOW);
        det->regrow_mark = (uint16_t *)calloc(NAPA_REGROW_WINDOW * NAPA_REGROW_WINDOW, sizeof(uint16_t));
        if (det->regrow_stack == NULL || det->regrow_mark == NULL) {
            free(det->regrow_stack), free(det->regrow_mark);
            det->regrow_stack = NULL, det->regrow_mark = NULL;
            return -2;
        }
    }
    double cx = 0, cy = 0;
    for (int k = 0; k < 4; k++) cx += 0.25 * rough[k].x, cy += 0.25 * rough[k].y;
    const int r = (int)(1.3 * side + 6.0);
    int x0 = (int)cx - r, y0 = (int)cy - r, x1 = (int)cx + r + 1, y1 = (int)cy + r + 1;
    x0 = x0 < 0 ? 0 : x0, y0 = y0 < 0 ? 0 : y0;
    x1 = x1 > view->w ? view->w : x1, y1 = y1 > view->h ? view->h : y1;
    const int w = x1 - x0, h = y1 - y0;
    if (w < 8 || h < 8 || w > NAPA_REGROW_WINDOW || h > NAPA_REGROW_WINDOW) return 0;
    const uint8_t *base = view->p + (size_t)y0 * view->stride + x0;
    const int flip = view->invert ? 255 : 0; /* reversed families are read inverted */
    int lo = 255, hi = 0;
    for (int y = 0; y < h; y += 2) { /* every other row and column: a tag's border and quiet zone are wider */
        const uint8_t *row = base + (size_t)y * view->stride;
        for (int x = 0; x < w; x += 2) {
            const int v = row[x] ^ flip;
            lo = v < lo ? v : lo, hi = v > hi ? v : hi;
        }
    }
    if (hi - lo < min_contrast) return 0;
    const int t = (lo + hi + 1) >> 1;
    uint16_t *mark = det->regrow_mark;
    int32_t *stack = det->regrow_stack;
    if (++det->regrow_stamp == 0) { /* the stamp wrapped: clear the old marks */
        memset(mark, 0, sizeof(uint16_t) * NAPA_REGROW_WINDOW * NAPA_REGROW_WINDOW);
        det->regrow_stamp = 1;
    }
    const uint16_t stamp = det->regrow_stamp;
    int seed = -1;
    for (int k = 0; k < 4 && seed < 0; k++) { /* the rough corners are dark samples of the fragment */
        const int sx = (int)floor(rough[k].x + 0.5) - x0, sy = (int)floor(rough[k].y + 0.5) - y0;
        for (int dy = -1; dy <= 1 && seed < 0; dy++)
            for (int dx = -1; dx <= 1 && seed < 0; dx++) {
                const int x = sx + dx, y = sy + dy;
                if (x >= 0 && y >= 0 && x < w && y < h && (base[(size_t)y * view->stride + x] ^ flip) < t) seed = y * w + x;
            }
    }
    if (seed < 0) return 0;
    c->count = 0;
    c->minx = c->miny = 1 << 30, c->maxx = c->maxy = -1;
    for (int k = 0; k < 16; k++) c->extv[k] = INT32_MIN;
    int top = 0;
    stack[top++] = seed, mark[seed] = stamp;
    while (top) {
        const int i = stack[--top], x = i % w, y = i / w;
        if (x == 0 || y == 0 || x == w - 1 || y == h - 1) return 0;
        c->count++;
        c->minx = x < c->minx ? x : c->minx, c->maxx = x > c->maxx ? x : c->maxx;
        c->miny = y < c->miny ? y : c->miny, c->maxy = y > c->maxy ? y : c->maxy;
        for (int k = 0; k < 16; k++) {
            const int v = comp_dirs[k][0] * x + comp_dirs[k][1] * y;
            if (v > c->extv[k]) c->extv[k] = v, c->ext[k][0] = x, c->ext[k][1] = y;
        }
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                const int j = i + dy * w + dx; /* in the window: x and y aren't on its edge */
                if (mark[j] != stamp && (base[(size_t)(y + dy) * view->stride + x + dx] ^ flip) < t) mark[j] = stamp, stack[top++] = j;
            }
    }
    *ox = x0, *oy = y0;
    return 1;
}

/* The rough quad of a candidate blob, from its extreme points; shared by the candidate loop and its threaded
 * pre-pass, so both compute it the same way. */
static NAPA_NOINLINE void rough_pick(const component *comp, int undersized_whole, int pick[4][2]) {
    /* Rough quad: of the extreme points (in clockwise angular order), the four that span the
     * largest quadrilateral. Start from the best of the eight axis and diagonal extremes, then
     * let each vertex move to any of the 16 extremes between its neighbours while the area
     * grows: axis or diagonal extremes alone slide along a side near 22 and 67 degrees. */
    /* Twice the area (positive, as the extremes run clockwise with y down) is the sum of
     * cross(p[t], p[t + 1]), so moving vertex v to point m changes it by a term linear in m:
     * cross(m, next - prev). Each move is one dot product. */
    const int(*E)[2] = comp->ext;
    int idx[4] = {0, 4, 8, 12};
    int64_t best2 = -1; /* twice the area: (p2 - p0) x (p3 - p1) */
#if defined(__aarch64__) && !defined(NAPA_NO_SIMD)
    /* The same 70 quads in the same order, written out: with constant indices the A76 keeps the extremes in registers
     * instead of reloading them for every quad. */
#define ROUGH_AREA(i, j, k, l)                                                                                         \
    do {                                                                                                               \
        const int64_t a2 = (int64_t)(E[k][0] - E[i][0]) * (E[l][1] - E[j][1]) -                                        \
                           (int64_t)(E[k][1] - E[i][1]) * (E[l][0] - E[j][0]);                                         \
        if (a2 > best2) best2 = a2, idx[0] = i, idx[1] = j, idx[2] = k, idx[3] = l;                                    \
    } while (0)
    ROUGH_AREA(0, 2, 4, 6); ROUGH_AREA(0, 2, 4, 8); ROUGH_AREA(0, 2, 4, 10); ROUGH_AREA(0, 2, 4, 12);
    ROUGH_AREA(0, 2, 4, 14); ROUGH_AREA(0, 2, 6, 8); ROUGH_AREA(0, 2, 6, 10); ROUGH_AREA(0, 2, 6, 12);
    ROUGH_AREA(0, 2, 6, 14); ROUGH_AREA(0, 2, 8, 10); ROUGH_AREA(0, 2, 8, 12); ROUGH_AREA(0, 2, 8, 14);
    ROUGH_AREA(0, 2, 10, 12); ROUGH_AREA(0, 2, 10, 14); ROUGH_AREA(0, 2, 12, 14); ROUGH_AREA(0, 4, 6, 8);
    ROUGH_AREA(0, 4, 6, 10); ROUGH_AREA(0, 4, 6, 12); ROUGH_AREA(0, 4, 6, 14); ROUGH_AREA(0, 4, 8, 10);
    ROUGH_AREA(0, 4, 8, 12); ROUGH_AREA(0, 4, 8, 14); ROUGH_AREA(0, 4, 10, 12); ROUGH_AREA(0, 4, 10, 14);
    ROUGH_AREA(0, 4, 12, 14); ROUGH_AREA(0, 6, 8, 10); ROUGH_AREA(0, 6, 8, 12); ROUGH_AREA(0, 6, 8, 14);
    ROUGH_AREA(0, 6, 10, 12); ROUGH_AREA(0, 6, 10, 14); ROUGH_AREA(0, 6, 12, 14); ROUGH_AREA(0, 8, 10, 12);
    ROUGH_AREA(0, 8, 10, 14); ROUGH_AREA(0, 8, 12, 14); ROUGH_AREA(0, 10, 12, 14); ROUGH_AREA(2, 4, 6, 8);
    ROUGH_AREA(2, 4, 6, 10); ROUGH_AREA(2, 4, 6, 12); ROUGH_AREA(2, 4, 6, 14); ROUGH_AREA(2, 4, 8, 10);
    ROUGH_AREA(2, 4, 8, 12); ROUGH_AREA(2, 4, 8, 14); ROUGH_AREA(2, 4, 10, 12); ROUGH_AREA(2, 4, 10, 14);
    ROUGH_AREA(2, 4, 12, 14); ROUGH_AREA(2, 6, 8, 10); ROUGH_AREA(2, 6, 8, 12); ROUGH_AREA(2, 6, 8, 14);
    ROUGH_AREA(2, 6, 10, 12); ROUGH_AREA(2, 6, 10, 14); ROUGH_AREA(2, 6, 12, 14); ROUGH_AREA(2, 8, 10, 12);
    ROUGH_AREA(2, 8, 10, 14); ROUGH_AREA(2, 8, 12, 14); ROUGH_AREA(2, 10, 12, 14); ROUGH_AREA(4, 6, 8, 10);
    ROUGH_AREA(4, 6, 8, 12); ROUGH_AREA(4, 6, 8, 14); ROUGH_AREA(4, 6, 10, 12); ROUGH_AREA(4, 6, 10, 14);
    ROUGH_AREA(4, 6, 12, 14); ROUGH_AREA(4, 8, 10, 12); ROUGH_AREA(4, 8, 10, 14); ROUGH_AREA(4, 8, 12, 14);
    ROUGH_AREA(4, 10, 12, 14); ROUGH_AREA(6, 8, 10, 12); ROUGH_AREA(6, 8, 10, 14); ROUGH_AREA(6, 8, 12, 14);
    ROUGH_AREA(6, 10, 12, 14); ROUGH_AREA(8, 10, 12, 14);
#undef ROUGH_AREA
#else
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
#endif
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
    for (int t = 0; t < 4; t++) pick[t][0] = E[idx[t]][0], pick[t][1] = E[idx[t]][1];
    if (undersized_whole) complete_small_outline(pick, comp->minx, comp->maxx, comp->miny, comp->maxy);
}

/* A rough quad's corners in pixels. */
static NAPA_NOINLINE void rough_corners(int pick[4][2], int s, double sx, double sy, double offx, double offy,
                                        vec2 rough[4]) {
    for (int k = 0; k < 4; k++) {
        rough[k].x = pick[k][0] * (double)s * sx + offx;
        rough[k].y = pick[k][1] * (double)s * sy + offy;
    }
}

/* A maximum-area inscribed quad can slide a corner along a long side, leaving
 * another component extremum beyond the short side's outward fit window. After
 * that fit fails, restore the nearest endpoint from the retained support point.
 * Bound the replacement by the fitter's existing movement limit and keep convex winding. */
static int restore_rough_corner(const component *comp, int pick[4][2], double step_x, double step_y,
                                double outward, double move_limit) {
    double most = outward;
    int corner = -1, extreme = -1;
    for (int k = 0; k < 4; k++) {
        int next = (k + 1) & 3;
        double ex = (pick[next][0] - pick[k][0]) * step_x;
        double ey = (pick[next][1] - pick[k][1]) * step_y;
        double len = vec_length(ex, ey);
        if (!(len > 0)) return 0;
        for (int j = 0; j < 16; j++) {
            double x = (comp->ext[j][0] - pick[k][0]) * step_x;
            double y = (comp->ext[j][1] - pick[k][1]) * step_y;
            double outside = (x * ey - y * ex) / len;
            if (outside > most) {
                most = outside;
                corner = x * x + y * y < (x - ex) * (x - ex) + (y - ey) * (y - ey) ? k : next;
                extreme = j;
            }
        }
    }
    if (corner < 0) return 0;
    int oldx = pick[corner][0], oldy = pick[corner][1];
    if (fabs((comp->ext[extreme][0] - oldx) * step_x) +
        fabs((comp->ext[extreme][1] - oldy) * step_y) > move_limit) return 0;
    pick[corner][0] = comp->ext[extreme][0], pick[corner][1] = comp->ext[extreme][1];
    for (int k = 0; k < 4; k++) {
        int j = (k + 1) & 3, n = (k + 2) & 3;
        int64_t ax = (int64_t)pick[j][0] - pick[k][0], ay = (int64_t)pick[j][1] - pick[k][1];
        int64_t bx = (int64_t)pick[n][0] - pick[j][0], by = (int64_t)pick[n][1] - pick[j][1];
        if (ax * by - ay * bx <= 0) {
            pick[corner][0] = oldx, pick[corner][1] = oldy;
            return 0;
        }
    }
    return 1;
}

/* Sparse paired border samples, before the expensive edge profiles: whether the rough quad (grown by `grow`) has a
 * quiet zone brighter than its border on three sides, and over all four on average when `strict`. */
static NAPA_NOINLINE int quiet_zone_passes(const image_view *view, const vec2 rough[4], double grow, double cells,
                                           int min_contrast, int strict, int shadow) {
    vec2 q[4];
    double H[9];
    grow_quad(rough, grow, q);
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
            contrast_sum += sample_bilinear(view, white.x, white.y) - sample_bilinear(view, black.x, black.y);
        }
        good_sides += contrast_sum >= min_contrast * 0.25;
        all_sides += contrast_sum;
    }
    return !(good_sides < 3 || (strict && all_sides < (shadow ? 6.0 : 4.8) * min_contrast));
}

/* These decisions are shared by serial candidates and the parallel pre-pass. */
static int rough_quad_too_full(const family_layout *family, const component *comp, int pick[4][2],
                               double *area, double *perimeter) {
    *area = quad_area_pts(pick);
    *perimeter = 0;
    for (int k = 0; k < 4; k++) {
        int ex = pick[(k + 1) & 3][0] - pick[k][0], ey = pick[(k + 1) & 3][1] - pick[k][1];
        *perimeter += sqrt((double)(ex * ex + ey * ey));
    }
    return comp->count > family->fill_limit * (*area + 0.5 * *perimeter + 1.0);
}

enum { QUIET_ZONE_SKIP, QUIET_ZONE_CHECK, QUIET_ZONE_STRICT };

static int rough_screen_mode(int s, int last, int shadow, int piece, int undersized_whole, int min_border_cells) {
    const int strict = shadow || s <= 2 || (s <= 6 && !undersized_whole);
    if (min_border_cells < 22 || !(shadow || piece || (strict && !last))) return QUIET_ZONE_SKIP;
    return strict ? QUIET_ZONE_STRICT : QUIET_ZONE_CHECK;
}

static int fit_initial_from_level(const image_view *view, const vec2 rough[4], vec2 fine[4], int s, double scale,
                                   int min_contrast, double *width, double cells, int *frame_sides) {
    return fit_initial_sides(view, rough, fine, 1.5 * s * scale + 1.0, 0.5 * s * scale + 1.0,
                             min_contrast, width, cells, frame_sides);
}

/* Geometry passed from the early decode into final fitting. It owns no component or queue pointers. */
typedef struct {
    vec2 fine[4];
    double reach, edge_w;
    edge_levels lv;
    int frame_sides;
    napa_detection early;
} candidate_fit;

typedef struct {
    const napa_params *params, *decode_params;
    int shadow, piece, require_exact;
} decode_policy;

/* Whether a decode corrects few enough bits under policy p: an undersized tag must read exactly (require_exact 1),
 * or within one bit when its first fit read the same code (require_exact 2, see refine_decoded_candidate). */
static int exact_enough(const decode_policy *p, int hamming) { return !p->require_exact || hamming <= p->require_exact - 1; }

typedef enum {
    REFINE_NO_MEMORY = -2,
    REFINE_ACCEPTED = 0,
    REFINE_FIT_FAILED,
    REFINE_DECODE_FAILED,
    REFINE_CUT_FAILED
} refinement_result;

/* What decoding and refining a fitted candidate came to (decode_and_refine). */
typedef struct {
    int decoded;               /* decode_initial_candidate: 1 decoded, 0 not */
    refinement_result refined; /* refine_decoded_candidate, when decoded */
    napa_detection d;
    image_view view;           /* the view d was refined in */
} candidate_outcome;

static int decode_and_refine(napa_detector *det, const image_view *view, const decode_policy *policy, candidate_fit *fit,
                             const vec2 rough[4], double regrow_extent, candidate_outcome *out, pass_timer *timer);

/* Whether a whole candidate is an extended shape (set aside as undersized or deferred, sparse, or filling under 0.3
 * of its box), which gets stricter tests in process_level_candidates. */
static int whole_is_extended(int base, int s, int last, int undersized_whole, const component *comp, int bw, int bh,
                             double best_area) {
    return undersized_whole || dense_whole_needs_deferral(base, s, last, bw, bh) || box_is_sparse(comp->count, bw, bh) ||
        best_area < 0.3 * bw * bh;
}

/* An extended shape is kept only if it is compact and filled enough to be a tag. A rectangle of aspect eight has
 * area/perimeter^2 = 2/81. */
static int extended_shape_rejected(const component *comp, double best_area, double perim_q) {
    return 81.0 * best_area < 2.0 * perim_q * perim_q || comp->count < 0.2 * best_area;
}

/* The decode thresholds for a whole candidate: *params, or *strict filled in from it. Short-code families match
 * clutter more often even without bit correction, so an extended shape in those families needs a complete dark
 * border, using the same samples as decoding; an undersized one needs an exact code. */
static const napa_params *candidate_decode_params(const napa_detector *det, const napa_params *params, int extended_shape,
                                                  int undersized_whole, napa_params *strict) {
    if (!extended_shape || !((det->fam.piece_hamming < 0 && params->min_border_cells < 28) ||
                             (undersized_whole && params->max_hamming != 0))) return params;
    *strict = *params;
    if (det->fam.piece_hamming < 0) strict->min_border_cells = 28;
    if (undersized_whole) strict->max_hamming = 0;
    return strict;
}

/* How far a small candidate's early decode may regrow its blob (0: not at all). */
static double regrow_extent_for(int regrown, int shadow, int piece, int base, int shaped, int s, double big_px) {
    return !regrown && !shadow && !piece && (base >= 2 || shaped) && s == base && big_px + s <= NAPA_REGROW_MAX_PX ?
        big_px + s : 0;
}

/* threads: a candidate's rough quad, screen and first fit, for the candidate loop. screen and fit are -1 where the
 * pre-pass didn't get that far (the loop then works them out itself, if it gets there). When decoding ahead, ahead
 * says whether outcome holds what the loop's decode_and_refine would make of that first fit. */
typedef struct cand_memo {
    int pick[4][2];
    signed char screen, fit, ahead;
    int frame_sides;
    vec2 fine[4];
    double edge_w;
    candidate_outcome outcome;
    candidate_fit fit_after; /* the fit as decode_and_refine left it */
} cand_memo;

/* threads, terminal level: a deferred fine whole's first-fit stage (stage: first_fit_stage's 1 or 0, with rough, fit and
 * regrown; -1 where the pre-pass stopped before it) and, when ahead is set, its decode and refinement. */
typedef struct delayed_memo {
    signed char stage, regrown, ahead;
    vec2 rough[4];
    candidate_fit fit;
    candidate_outcome outcome;
    candidate_fit fit_after;
} delayed_memo;

enum { MEMO_CHUNK = 4 };

typedef struct memo_job {
    napa_detector *det;
    const image_view *view;
    const napa_params *params;
    int s, base, lo, last, shadow, shaped;
    double sx, sy, offx, offy, smax, cells;
    int n;
    int ahead;            /* also decode and refine, one candidate per index, on each thread's own detector */
    const int32_t *order; /* when decoding ahead: index k takes candidate order[k] */
    const napa_run *runs; /* deferred extremes: the level's runs and dark bits, else NULL */
    const uint64_t *bits;
    int bit_stride;
    struct next_level_job *next; /* index 0 samples and labels the next level instead (sample_next_level), else NULL */
    const struct pass_config *config;
    int ordinary_jobs; /* the indices after these (and after next) each take a deferred fine whole (decode_whole_ahead) */
    int wholes_now;    /* set-aside candidates are worked out now, into dmemo[det->whole_index[ci]] */
    component *cands;  /* the level's candidate set (cand_set_swap) */
    struct cand_memo *memo;
    const int32_t *cand_runs, *cand_run_end;
    struct memo_job *coarse; /* indices from coarse_from on are the terminal level's (sample_next_level), else NULL */
    int coarse_from;
    int fit_only; /* one candidate per index, but no decoding ahead (the terminal level assembled early) */
} memo_job;

struct next_level_job;
static void sample_next_level(struct next_level_job *job, int worker);
static void decode_whole_ahead(const memo_job *j, napa_detector *wd, int index);
static void decode_shape_ahead(const memo_job *j, napa_detector *wd, const component *comp, int s, int last,
                               struct delayed_memo *dm);

/* The loop's decode_and_refine for a candidate whose first fit the pre-pass made, worked out ahead on this thread's
 * detector wd, with the same rough quad, fit, thresholds and regrowth extent the loop would use. Whether the loop
 * gets that far (coverage, failed quads, duplicates) is still decided there, in candidate order. */
static void decode_ahead(const memo_job *j, napa_detector *wd, cand_memo *m, const component *comp, int bw, int bh,
                         double best_area, double perim_q, const vec2 rough[4]) {
    const int s = j->s;
    const int extended_shape = whole_is_extended(j->base, s, j->last, 0, comp, bw, bh, best_area);
    if (extended_shape && extended_shape_rejected(comp, best_area, perim_q)) return; /* the loop stops before decoding */
    napa_params strict_border;
    const napa_params *decode_params = candidate_decode_params(wd, j->params, extended_shape, 0, &strict_border);
    candidate_fit fit;
    fit.reach = 0.5 * ((1.5 * s * j->smax + 1.0) + (0.5 * s * j->smax + 1.0));
    fit.edge_w = m->edge_w, fit.frame_sides = m->frame_sides;
    memcpy(fit.fine, m->fine, sizeof fit.fine);
    const double big_px = s * (bw * j->sx > bh * j->sy ? bw * j->sx : bh * j->sy);
    const decode_policy policy = {j->params, decode_params, j->shadow, 0, 0};
    pass_timer timing = {0};
    const double regrow_extent = regrow_extent_for(0, j->shadow, 0, j->base, j->shaped, s, big_px);
    m->ahead = decode_and_refine(wd, j->view, &policy, &fit, rough, regrow_extent, &m->outcome, &timing) == 0;
    m->fit_after = fit;
}

/* One index of the pre-pass, for the ordinary candidates (det->cands), as the candidate loop treats them: MEMO_CHUNK
 * candidates, or one when decoding ahead. */
static void memo_candidates(void *ctx, int index, int worker) {
    const memo_job *j = (const memo_job *)ctx;
    napa_detector *det = j->det;
    if (j->coarse != NULL && index >= j->coarse_from) {
        memo_candidates(j->coarse, index - j->coarse_from, worker);
        return;
    }
    if (j->next != NULL && index-- == 0) {
        sample_next_level(j->next, worker);
        return;
    }
    if (index >= j->ordinary_jobs) { /* decoding ahead implies a detector per pool thread (can_decode_ahead) */
        decode_whole_ahead(j, worker == 0 ? det : &det->workers[worker - 1], det->dmemo_ready + index - j->ordinary_jobs);
        return;
    }
    const int chunk = j->ahead || j->fit_only ? 1 : MEMO_CHUNK;
    const int end = (index + 1) * chunk < j->n ? (index + 1) * chunk : j->n;
    for (int k = index * chunk; k < end; k++) {
        const int ci = j->order ? j->order[k] : k;
        cand_memo *m = &j->memo[ci];
        if (j->runs != NULL) {
            /* The extremes assemble_level_candidates left: this candidate's runs, in the order it would have added them. */
            component *c = &j->cands[ci];
            const int32_t *next_run = j->cand_runs, tail = j->cand_run_end[ci];
            int32_t e = tail;
            do {
                e = next_run[e];
                comp_add_pair_run(c, &j->runs[e], j->bits, j->bit_stride);
            } while (e != tail);
            comp_flush_pair(c, j->bits, j->bit_stride);
            c->work.candidate.fate = CANDIDATE_UNTRIED, c->work.candidate.piece_count = 0;
        }
        const component *comp = &j->cands[ci];
        m->screen = m->fit = -1;
        m->ahead = 0;
        m->frame_sides = 0;
        const int bw = comp->maxx - comp->minx + 1, bh = comp->maxy - comp->miny + 1;
        const int undersized_whole = j->base > 1 && (bw < j->lo || bh < j->lo);
        rough_pick(comp, undersized_whole, m->pick);
        if (undersized_whole || dense_whole_needs_deferral(j->base, j->s, j->last, bw, bh)) { /* set aside by the loop */
            if (j->wholes_now)
                decode_shape_ahead(j, worker == 0 ? det : &det->workers[worker - 1], comp, j->s, j->last,
                                   &det->dmemo[det->whole_index[ci]]);
            continue;
        }
        double best_area, perim_q;
        if (rough_quad_too_full(&det->fam, comp, m->pick, &best_area, &perim_q)) continue;
        vec2 rough[4], fine[4];
        rough_corners(m->pick, j->s, j->sx, j->sy, j->offx, j->offy, rough);
        const int s = j->s;
        const int screen_mode = rough_screen_mode(s, j->last, j->shadow, 0, undersized_whole, j->params->min_border_cells);
        if (screen_mode != QUIET_ZONE_SKIP) {
            m->screen = (signed char)quiet_zone_passes(j->view, rough, 0.5 * s * j->smax, j->cells, j->params->min_contrast,
                                                       screen_mode == QUIET_ZONE_STRICT, j->shadow);
            if (!m->screen) continue;
        }
        double width = 0;
        m->fit = (signed char)fit_initial_from_level(j->view, rough, fine, s, j->smax,
                                                   j->params->min_contrast, &width, j->cells, &m->frame_sides);
        memcpy(m->fine, fine, sizeof fine);
        m->edge_w = width;
        if (j->ahead && m->fit) /* decoding ahead implies a detector per pool thread (can_decode_ahead) */
            decode_ahead(j, worker == 0 ? det : &det->workers[worker - 1], m, comp, bw, bh, best_area, perim_q, rough);
    }
}

/* Decoding ahead on the pool's threads needs every path decoding and refining can take to leave the detector alone:
 * no camera gamma (its linear frame), small-tag regrowth budget, soft capture or grid probe, and the tone tables built
 * first. Those paths, and the code stamps, are all the shared state that decoding and refining write. */
static int can_decode_ahead(napa_detector *det) {
#ifdef NAPA_PROFILE
    (void)det;
    return 0; /* the stage timers are one set for the process */
#else
    return det->worker_count == det->threads - 1 && !det->has_camera_gamma && det->regrow_left <= 0 &&
        !det->soft_capturing && det->grid_probe == NULL && tone_prepare(det);
#endif
}

/* The pool threads' detectors: the family tables (with their own code stamps) and the settings that decoding and
 * refining read. */
static void prepare_workers(napa_detector *det) {
    for (int w = 0; w < det->worker_count; w++) {
        napa_detector *wd = &det->workers[w];
        uint32_t *stamp = wd->fam.stamp;
        const uint32_t now = wd->fam.stamp_now;
        wd->fam = det->fam, wd->fam.stamp = stamp, wd->fam.stamp_now = now;
        wd->align_corners = det->align_corners, wd->tone_blur = det->tone_blur;
        memcpy(wd->linearize, det->linearize, sizeof wd->linearize);
    }
}

/* Room for the pre-pass's results for ncand candidates. */
static int memo_reserve(napa_detector *det, int ncand) {
    if ((size_t)ncand > det->memo_capacity) {
        cand_memo *m = (cand_memo *)realloc(det->memo, (size_t)ncand * sizeof(cand_memo));
        if (m != NULL) det->memo = m;
        int32_t *o = m != NULL ? (int32_t *)realloc(det->memo_order, (size_t)ncand * sizeof(int32_t)) : NULL;
        if (o != NULL) det->memo_order = o, det->memo_capacity = (size_t)ncand;
    }
    return (size_t)ncand <= det->memo_capacity;
}

/* log2 of a candidate's box area, largest first. */
static int area_rank(const component *c) {
    unsigned a = (unsigned)(c->maxx - c->minx + 1) * (unsigned)(c->maxy - c->miny + 1);
    int b = 0;
    while (a > 1 && b < 31) a >>= 1, b++;
    return 31 - b;
}

static int params_valid(const napa_params *params) {
    return params != NULL && params->min_tag_px >= 8 && params->min_tag_px <= 30000 &&
        params->max_hamming >= 0 && params->max_hamming <= 3 &&
        params->min_contrast >= 1 && params->min_contrast <= 255 &&
        params->min_border_cells >= 0 && params->min_border_cells <= 28 &&
        params->sharpening_x100 >= 0 && params->sharpening_x100 <= 100 &&
        params->split_touching >= 0 && params->split_touching <= 2;
}

/* The converged profile and its alternate geometric hypotheses. */
typedef struct {
    vec2 finer[4], alternative[4], curved[4], lens_quad[4];
    double edge_w, cell, side_widths_seen[2], side_level[4];
    edge_levels lv, final_lv;
    int lens_ok, cut, accepted_blur;
    napa_detection d;
} refined_quad;

typedef struct {
    int corrected_border, accepted_clip, near_frame, corner_out[4];
    double edge_shift, measured_bias, whole_share, bias_estimate, accepted_border_level, curve_span;
} border_correction;

/* Each side's two border cells at each end (the corner cell and its neighbour along the side) as a fraction of the way
 * from the tag's black to its white: about 0 where the side runs along the border, about 1 where it runs out on the quiet
 * zone. levels[side][end], sides top, right, bottom, left of q's square; end 0 at the side's first corner. 0 if the
 * border can't be read. */
static int side_end_levels_whiter(const family_layout *L, const image_view *im, const vec2 q[4], int min_contrast,
                                  int whiter, double levels[4][2]) {
    const int W = L->width, o = L->outer, G = L->grid;
    double H[9], grid[256], border[64], black, white;
    square_to_quad(q, H);
    if (read_border(L, im, H, min_contrast, grid, border, &black, &white, NULL) < 0 || white <= black) return 0;
    for (int k = 0; k < 4; k++)
        for (int e = 0; e < 2; e++) {
            double cell[3];
            for (int j = 0; j <= 2; j++) {
                const int along = e ? W - 1 - j : j;
                cell[j] = grid[((k == 0 ? 0 : k == 2 ? W - 1 : along) + o) * G + (k == 1 ? W - 1 : k == 3 ? 0 : along) + o];
            }
            /* whiter: the corner cell also lies on the other side's border, so when only this side pivots out it
             * stays dark; judge the end by the whiter of its two pairs. The corner repair keeps the plain pair. */
            const double pair = whiter && cell[1] + cell[2] > cell[0] + cell[1] ? cell[1] + cell[2] : cell[0] + cell[1];
            levels[k][e] = (0.5 * pair - black) / (white - black);
        }
    return 1;
}

static int side_end_levels(const family_layout *L, const image_view *im, const vec2 q[4], int min_contrast,
                           double levels[4][2]) {
    return side_end_levels_whiter(L, im, q, min_contrast, 0, levels);
}

static void correct_border_geometry(napa_detector *det, const image_view *rview, const decode_policy *policy,
                                    refined_quad *q, border_correction *c) {
    const int W = det->fam.width;
    const double cells = det->fam.cells;
    /* A side whose border cells read white was fitted one cell out, on the quiet zone's outer
     * edge (the payload still decodes through error correction). Move it in by a cell, refit,
     * and keep the result if it decodes the same tag with a darker border. */
    int bad = 0;
    c->corrected_border = 0;
    c->accepted_clip = 0;
    c->edge_shift = 0; /* the measured bias moved every side by this, without visible clipping */
    c->measured_bias = 0; c->whole_share = 0;
    double level_before = 0;
    for (int k = 0; k < 4; k++) {
        bad |= (q->side_level[k] > 0.3) << k;
        level_before += q->side_level[k];
    }
    /* A side that pivoted out reads black at the end it pivots on (that end runs along the border) and white at
     * the end that left it: moving the whole side in a cell would pull the anchored end into the tag, so leave
     * such sides to the corner repair below. A side that merely reads grey at both ends (a blurred edge the fit
     * straddles) is not anchored anywhere and still takes the whole-side move. */
    double end_levels[4][2];
    if (bad && side_end_levels_whiter(&det->fam, rview, q->finer, policy->params->min_contrast, 1, end_levels))
        for (int k = 0; k < 4; k++) {
            const double lo = end_levels[k][0] < end_levels[k][1] ? end_levels[k][0] : end_levels[k][1];
            const double hi = end_levels[k][0] < end_levels[k][1] ? end_levels[k][1] : end_levels[k][0];
            if (((bad >> k) & 1) && lo < 0.15 && hi > 0.6) bad &= ~(1 << k);
        }
    c->accepted_border_level = level_before;
    if (bad && bad != 15 && !q->cut) {
        double Hs[9], perim = 0;
        square_to_quad(q->finer, Hs);
        double one = 1.0 / W, rest = (W - 1.0) / W;
        double u0 = (bad & 8) ? one : 0, v0 = (bad & 1) ? one : 0;
        double u1 = (bad & 2) ? rest : 1, v1 = (bad & 4) ? rest : 1;
        vec2 moved[4] = {project(Hs, u0, v0), project(Hs, u1, v0), project(Hs, u1, v1), project(Hs, u0, v1)};
        for (int k = 0; k < 4; k++) perim += vec_length(moved[(k + 1) & 3].x - moved[k].x, moved[(k + 1) & 3].y - moved[k].y);
        double r = 0.45 * perim / (4.0 * W);
        vec2 fixed[4];
        napa_detection d2;
        double side2[4], level_after = 0;
        /* The moved side starts tilted (it came from a skewed quad): fit, then refit once re-centred
         * in a window just wide enough for the edge, as the main refinement converges. */
        double r2 = 1.25 * q->edge_w > 1.5 ? 1.25 * q->edge_w : 1.5;
        vec2 first[4];
        int refit = r >= 1.0 && fit_sides(rview, moved, first, r, r, policy->params->min_contrast, 0, NAPA_FINAL_POINTS, &q->edge_w, &q->lv, cells, NULL);
        if (refit && r2 < r) refit = fit_sides(rview, first, fixed, r2, r2, policy->params->min_contrast, 0, NAPA_FINAL_POINTS, &q->edge_w, &q->lv, cells, NULL);
        else if (refit) memcpy(fixed, first, sizeof fixed);
        if (refit && decode_quad(det, rview, fixed, policy->decode_params, 0, &d2, side2, NULL, NULL) &&
            exact_enough(policy, d2.hamming) && d2.id == q->d.id) {
            for (int k = 0; k < 4; k++) level_after += side2[k];
            if (level_after < level_before - 0.2) {
                q->d = d2;
                c->corrected_border = 1;
                c->accepted_border_level = level_after;
                memcpy(q->finer, fixed, sizeof q->finer);
            }
        }
    }
    /* A corner fitted out on the quiet zone (a speck of the margin joined the blob there): one side pivots out at that
     * end, or both sides meeting there do. The sides as a whole needn't read white, but their border cells next to that
     * corner do. Step the corner in a cell at a time, square to each side whose end there reads white, until those
     * cells read dark, refit twice in a narrow window, and keep the result if the same tag decodes with them dark and
     * the border as a whole no lighter. Steps may pass through a decode with more corrected bits; the repair as a
     * whole must not end with more. */
    const refined_quad before_ends = *q;
    const border_correction before_c = *c;
    /* The two side ends that meet at each corner: (side, end). Corner 0 (0,0) is side 0's start and side 3's start. */
    static const int corner_ends[4][2][2] = {{{0, 0}, {3, 0}}, {{0, 1}, {1, 0}}, {{1, 1}, {2, 1}}, {{2, 0}, {3, 1}}};
    for (int rep = 0, changed = 1; rep < 3 && changed && !q->cut && bad != 15; rep++) {
        changed = 0;
        double levels[4][2], end_white = 0.6;
        int corner = -1;
        if (side_end_levels(&det->fam, rview, q->finer, policy->params->min_contrast, levels))
            for (int k = 0; k < 4; k++)
                for (int j = 0; j < 2; j++)
                    if (levels[corner_ends[k][j][0]][corner_ends[k][j][1]] > end_white)
                        end_white = levels[corner_ends[k][j][0]][corner_ends[k][j][1]], corner = k;
        if (corner < 0) break;
        /* Step the corner in square to each side whose end there reads white: one side when only that side pivoted
         * out, both when the corner itself sits out on the margin. */
        int across[2];
        for (int j = 0; j < 2; j++) across[j] = levels[corner_ends[corner][j][0]][corner_ends[corner][j][1]] > 0.6;
        vec2 trial[4];
        memcpy(trial, q->finer, sizeof trial);
        double level = end_white;
        for (int step = 0; step < 3 && level > 0.6; step++) {
            double Hs[9], cu = (corner == 1 || corner == 2) ? 1.0 : 0.0, cv = corner >= 2 ? 1.0 : 0.0;
            for (int j = 0; j < 2; j++) {
                if (!across[j]) continue;
                const int side = corner_ends[corner][j][0];
                if (side == 0) cv += 1.0 / W; else if (side == 2) cv -= 1.0 / W; else if (side == 1) cu -= 1.0 / W; else cu += 1.0 / W;
            }
            square_to_quad(trial, Hs);
            trial[corner] = project(Hs, cu, cv);
            if (!side_end_levels(&det->fam, rview, trial, policy->params->min_contrast, levels)) break;
            level = 0;
            for (int j = 0; j < 2; j++)
                if (across[j] && levels[corner_ends[corner][j][0]][corner_ends[corner][j][1]] > level)
                    level = levels[corner_ends[corner][j][0]][corner_ends[corner][j][1]];
        }
        if (level >= end_white - 0.3) continue;
        const double reach = 1.25 * q->edge_w > 1.5 ? 1.25 * q->edge_w : 1.5;
        vec2 fixed[4];
        napa_detection d2;
        double side2[4], level_after = 0, edge_w = q->edge_w;
        edge_levels lv = q->lv;
        vec2 first[4];
        if (!fit_sides(rview, trial, first, reach, reach, policy->params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv, cells, NULL) ||
            !fit_sides(rview, first, fixed, reach, reach, policy->params->min_contrast, 0, NAPA_FINAL_POINTS, &edge_w, &lv, cells, NULL) ||
            signed_area(fixed) * signed_area(q->finer) <= 0 ||
            !decode_quad(det, rview, fixed, policy->decode_params, 0, &d2, side2, NULL, NULL) ||
            !exact_enough(policy, d2.hamming) || d2.id != q->d.id ||
            d2.hamming > policy->params->max_hamming || d2.hamming > det->fam.max_hamming ||
            (policy->shadow && d2.hamming != 0) || (policy->piece && d2.hamming > det->fam.piece_hamming))
            continue;
        for (int k = 0; k < 4; k++) level_after += side2[k];
        if (!side_end_levels(&det->fam, rview, fixed, policy->params->min_contrast, levels)) continue;
        double end_after = 0;
        for (int j = 0; j < 2; j++)
            if (across[j] && levels[corner_ends[corner][j][0]][corner_ends[corner][j][1]] > end_after)
                end_after = levels[corner_ends[corner][j][0]][corner_ends[corner][j][1]];
        if (end_after < 0.5 && end_after < end_white - 0.3 && level_after <= c->accepted_border_level) {
            q->d = d2;
            c->corrected_border = 1;
            c->accepted_border_level = level_after;
            q->edge_w = edge_w, q->lv = lv;
            memcpy(q->finer, fixed, sizeof q->finer);
            /* The fits' alternatives described the quad before the repair. */
            memcpy(q->alternative, fixed, sizeof q->alternative);
            memcpy(q->curved, fixed, sizeof q->curved);
            q->lens_ok = 0;
            changed = 1;
        }
    }
    if (q->d.hamming > before_ends.d.hamming) *q = before_ends, *c = before_c;
    /* Clipped levels, and gamma-encoded images under blur, bias every edge the same way: the half-level
     * isn't the true edge. Measure the bias from the border's own width, which is one cell whatever the
     * camera, and undo it. Without visible clipping only a clear narrowing counts: gamma can only narrow a
     * black border, while a border that looks wider than a cell is blur, which needs no correction. */
    c->bias_estimate = 0; /* set even when the sides disagree and the correction is set aside */
    if (!q->cut) {
        double disagreement = 0;
        const int samples = border_bias(&det->fam, rview, q->finer, q->final_lv.black, q->final_lv.white, &c->bias_estimate, &disagreement);
        const double delta = c->bias_estimate;
        const int clipped = q->final_lv.white >= 250 || q->final_lv.black <= 4;
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
        memcpy(bowed, q->curved, sizeof bowed);
        if (signed_area(bowed) < 0) { vec2 swap = bowed[1]; bowed[1] = bowed[3]; bowed[3] = swap; }
        for (int k = 0; k < 4; k++) {
            const double span = vec_length(bowed[k].x - q->finer[k].x, bowed[k].y - q->finer[k].y);
            if (span > bow) bow = span;
        }
        if (share > 0 && bow < NAPA_CURVED_TAG_MIN && payload_bias(&det->fam, rview, q->finer, q->final_lv.black, q->final_lv.white, &payload_delta))
            share *= fade(payload_delta / delta, 0.25, 0.75);
        if (share > 0) {
            vec2 grown[4];
            napa_detection d3;
            grow_quad(q->finer, share * delta, grown);
            if (decode_quad(det, rview, grown, policy->decode_params, 0, &d3, NULL, NULL, NULL) &&
                exact_enough(policy, d3.hamming) && d3.id == q->d.id) {
                q->d = d3;
                c->measured_bias = delta;
                c->whole_share = share;
                if (clipped) c->accepted_clip = 1;
                else c->edge_shift = share * delta;
            }
        }
    }
    if (signed_area(q->alternative) < 0) { vec2 t = q->alternative[1]; q->alternative[1] = q->alternative[3]; q->alternative[3] = t; }
    double alternate_move = 0;
    for (int k = 0; k < 4; k++) alternate_move += fabs(q->alternative[k].x - q->finer[k].x) + fabs(q->alternative[k].y - q->finer[k].y);
    if (!q->cut && alternate_move > 0.02 && q->final_lv.white < 250 && q->final_lv.black > 4) {
        napa_detection refined;
        double alternative_border[4];
        /* The fits share the same biased edge location, so they share its correction. */
        if (c->edge_shift != 0) grow_quad(q->alternative, c->edge_shift, q->alternative);
        if (decode_quad(det, rview, q->alternative, policy->decode_params, 0, &refined, alternative_border, NULL,
                        NULL) &&
            exact_enough(policy, refined.hamming) && refined.id == q->d.id && refined.hamming <= policy->params->max_hamming &&
            refined.hamming <= det->fam.max_hamming && (!policy->shadow || refined.hamming == 0) &&
            (!policy->piece || refined.hamming <= det->fam.piece_hamming)) {
            double border_sum = 0;
            for (int k = 0; k < 4; k++) border_sum += alternative_border[k];
            if (!c->corrected_border || border_sum <= c->accepted_border_level) { q->d = refined; q->accepted_blur = 1; }
        }
    }
}

static void select_curve_geometry(napa_detector *det, const image_view *rview, const decode_policy *policy,
                                    refined_quad *q, border_correction *c) {
    c->curve_span = 0;
    c->near_frame = 0;
    {
        vec2 ordered[4];
        double clear_span = 0;
        memcpy(ordered, q->curved, sizeof ordered);
        if (signed_area(ordered) < 0) { vec2 swap = ordered[1]; ordered[1] = ordered[3]; ordered[3] = swap; }
        for (int k = 0; k < 4; k++) {
            c->corner_out[k] = !inside_frame(rview, q->finer[k]) || !inside_frame(rview, ordered[k]);
            c->near_frame |= c->corner_out[k];
        }
        for (int k = 0; k < 4; k++) {
            double span = vec_length(ordered[k].x - q->finer[k].x, ordered[k].y - q->finer[k].y);
            if (span > c->curve_span) c->curve_span = span;
            if (span > clear_span && !c->corner_out[k] && !c->corner_out[(k + 1) & 3] && !c->corner_out[(k + 3) & 3]) clear_span = span;
        }
        /* A side cut by the frame has no samples at its true end, so its curve is extrapolated. Believe it
         * only if a corner whose two sides both lie inside the frame shows the bow too. */
        if (clear_span >= NAPA_CURVED_TAG_MIN) c->near_frame = 0;
        /* A lens bows every side; one side bowing alone, by a large part of a cell, is a poor curve fit. */
        if (c->curve_span >= NAPA_CURVED_TAG_MIN) {
            double span[4], top = 0;
            int moved_corners = 0;
            for (int k = 0; k < 4; k++) {
                span[k] = vec_length(ordered[k].x - q->finer[k].x, ordered[k].y - q->finer[k].y);
                if (span[k] > top) top = span[k];
            }
            for (int k = 0; k < 4; k++) moved_corners += span[k] >= NAPA_BOW_MIN_SHARE * top;
            if (moved_corners < 3 && top > NAPA_BOW_LOPSIDED_CELLS * q->cell) c->curve_span = 0;
        }
    }
    if (q->cut) c->near_frame = 1; /* cut sides were measured from their inner edges */
    if (c->near_frame) c->curve_span = 0;
    if (!q->cut && q->lens_ok) {
        napa_detection proposal;
        double lens_bias = c->measured_bias;
        if (lens_bias == 0 && q->final_lv.white < 250 && c->bias_estimate >= NAPA_LOCAL_BIAS_MIN && c->bias_estimate < 1.5) lens_bias = c->bias_estimate;
        if (signed_area(q->lens_quad) < 0) { vec2 swap = q->lens_quad[1]; q->lens_quad[1] = q->lens_quad[3]; q->lens_quad[3] = swap; }
        if (lens_bias != 0) grow_quad(q->lens_quad, lens_bias, q->lens_quad);
        if (decode_quad(det, rview, q->lens_quad, policy->decode_params, 0, &proposal, NULL, NULL, NULL) &&
            exact_enough(policy, proposal.hamming) && proposal.id == q->d.id && proposal.hamming <= q->d.hamming &&
            (!policy->shadow || proposal.hamming == 0) && (!policy->piece || proposal.hamming <= det->fam.piece_hamming)) {
            q->d = proposal;
            c->near_frame = 1, c->curve_span = 0; /* no later step should move these corners */
        }
    }
    /* The final profile points precede these corrections, so keep any later geometry winner. */
    if (!q->cut && !c->corrected_border && !c->accepted_clip && !q->accepted_blur) {
        if (signed_area(q->curved) < 0) {
            vec2 swap = q->curved[1]; q->curved[1] = q->curved[3]; q->curved[3] = swap;
        }
        double curve_move = 0;
        for (int k = 0; k < 4; k++)
            curve_move += fabs(q->curved[k].x - q->finer[k].x) + fabs(q->curved[k].y - q->finer[k].y);
        if (curve_move > 0.02 && !c->near_frame) {
            napa_detection proposal;
            if (c->edge_shift != 0) grow_quad(q->curved, c->edge_shift, q->curved);
            if (decode_quad(det, rview, q->curved, policy->decode_params, 0, &proposal, NULL, NULL, NULL) &&
                exact_enough(policy, proposal.hamming) && proposal.id == q->d.id && proposal.hamming <= q->d.hamming &&
                proposal.hamming <= policy->params->max_hamming && proposal.hamming <= det->fam.max_hamming &&
                (!policy->shadow || proposal.hamming == 0) && (!policy->piece || proposal.hamming <= det->fam.piece_hamming))
                q->d = proposal;
        }
    }
}

static int polish_decoded_corners(napa_detector *det, const image_view *rview, const decode_policy *policy,
                                    refined_quad *q, border_correction *c) {
    const int W = det->fam.width;
    if (!q->cut) {
        vec2 local[4];
        double mx = 0, my = 0, side = 0;
        for (int k = 0; k < 4; k++) {
            local[k].x = q->d.corners[2 * k], local[k].y = q->d.corners[2 * k + 1];
            mx += 0.25 * local[k].x, my += 0.25 * local[k].y;
        }
        for (int k = 0; k < 4; k++)
            side += 0.25 * vec_length(local[(k + 1) & 3].x - local[k].x, local[(k + 1) & 3].y - local[k].y);
        const double one_cell = side / W;
        int moved = 0;
        vec2 start[4];
        memcpy(start, local, sizeof start);
        /* Sides that disagree about the border's width set aside the correction for the whole tag, as on a
         * bowed one. A large narrowing still says how far the half level sits from the edge; widening is blur. */
        /* How much each reason to refit applies, from 0 to 1, so a tag near a limit moves part of the way. */
        double straight_weight = 0;
        if (c->whole_share < 1 && q->final_lv.white < 250 && c->bias_estimate < 1.5) {
            straight_weight = (1 - c->whole_share) * fade(c->bias_estimate, NAPA_LOCAL_BIAS_MIN - NAPA_FADE_BIAS, NAPA_LOCAL_BIAS_MIN + NAPA_FADE_BIAS) * fade(1.5 - c->bias_estimate, 0, 0.2) *
                              (1 - fade(q->side_widths_seen[0], NAPA_LOCAL_MAX_WIDTH_RATIO - NAPA_FADE_WIDTH_RATIO, NAPA_LOCAL_MAX_WIDTH_RATIO + NAPA_FADE_WIDTH_RATIO));
            if (straight_weight > 0 && c->measured_bias == 0) c->measured_bias = c->bias_estimate;
        }
        const double bowed_weight = c->near_frame || c->curve_span > NAPA_LOCAL_MAX_BOW ? 0 : fade(c->curve_span, NAPA_CURVED_TAG_MIN - NAPA_FADE_BOW, NAPA_CURVED_TAG_MIN + NAPA_FADE_BOW);
        /* Straight tags whose whole-tag correction stood are already right. */
        const int refittable = !c->near_frame && (bowed_weight > 0 || straight_weight > 0);
        for (int k = 0; k < 4; k++) {
            vec2 fitted;
            corner_sides sides;
            if (c->corner_out[k]) continue;
            if (refittable && measure_corner_sides(start, k, one_cell, &sides)) {
                double weight = bowed_weight;
                if (weight < 1 && straight_weight > 0) {
                    const double disagree = straight_weight * fade(corner_sides_offset(rview, start[k], &sides, one_cell, c->measured_bias), NAPA_LOCAL_EDGE_DISAGREE - NAPA_FADE_EDGE, NAPA_LOCAL_EDGE_DISAGREE + NAPA_FADE_EDGE);
                    if (disagree > weight) weight = disagree;
                }
                if (weight > 0 && refit_corner(rview, start[k], &sides, one_cell, c->measured_bias, &fitted)) {
                    const double dx = fitted.x - start[k].x, dy = fitted.y - start[k].y;
                    /* A small local move can still correct a bowed edge; fade it as bow confidence falls. */
                    const double move_weight = fade(vec_length(dx, dy), NAPA_LOCAL_MIN_MOVE - NAPA_FADE_MOVE, NAPA_LOCAL_MIN_MOVE + NAPA_FADE_MOVE);
                    weight *= bowed_weight + (1 - bowed_weight) * move_weight;
                    if (weight > 0) local[k].x = start[k].x + weight * dx, local[k].y = start[k].y + weight * dy, moved = 1;
                    continue;
                }
                if (weight > 0 && bowed_weight < 1) continue;
            }
            if (c->curve_span < NAPA_CURVED_TAG_MIN) continue;
            double px = q->d.corners[2 * k], py = q->d.corners[2 * k + 1], norm = vec_length(px - mx, py - my);
            double ux = (px - mx) / norm, uy = (py - my) / norm;
            vec2 prev = local[(k + 3) & 3], next = local[(k + 1) & 3];
            double ax = next.x - px, ay = next.y - py, bx = prev.x - px, by = prev.y - py;
            double al = vec_length(ax, ay), bl = vec_length(bx, by);
            ax /= al, ay /= al, bx /= bl, by /= bl;
            double white = sample_bilinear(rview, px + 0.2 * one_cell * ux - one_cell * ax, py + 0.2 * one_cell * uy - one_cell * ay);
            double white2 = sample_bilinear(rview, px + 0.2 * one_cell * ux - one_cell * bx, py + 0.2 * one_cell * uy - one_cell * by);
            if (white2 > white) white = white2;
            double inside = sample_bilinear(rview, px - one_cell * ux, py - one_cell * uy);
            if (sample_bilinear(rview, px + one_cell * ux, py + one_cell * uy) >= 0.5 * (inside + white)) continue;
            double nx = px, ny = py;
            if (refine_x_corner(rview, &nx, &ny) && vec_length(nx - px, ny - py) < 3.0) local[k].x = nx, local[k].y = ny, moved = 1;
        }
        napa_detection polished;
        if (moved && q->edge_w <= NAPA_POLISH_MAX_EDGE_CELLS * one_cell &&
            decode_quad(det, rview, local, policy->decode_params, 0, &polished, NULL, NULL, NULL) &&
            polished.id == q->d.id && polished.hamming <= q->d.hamming) q->d = polished;
    }
    if (!q->cut) {
        tone_measure tone;
        int measured = tone_measure_tag(det, rview, &q->d, det->align_corners ? q->side_widths_seen[1] : q->edge_w, det->align_corners, &tone);
        if (measured < 0) return -2;
        if (measured) {
            vec2 sized[4], aligned[4];
            napa_detection redone;
            if (tone_corrected_quad(&q->d, &tone, sized) &&
                decode_quad(det, rview, sized, policy->decode_params, 0, &redone, NULL, NULL, NULL) &&
                redone.id == q->d.id && redone.hamming <= q->d.hamming) q->d = redone;
            if (det->align_corners && tone_aligned_quad(rview, &q->d, &tone, aligned) &&
                decode_quad(det, rview, aligned, policy->decode_params, 0, &redone, NULL, NULL, NULL) &&
                redone.id == q->d.id && redone.hamming <= q->d.hamming) q->d = redone;
        }
    }
    return 0;
}

static refinement_result refine_decoded_candidate(napa_detector *det, const image_view *view, const decode_policy *policy,
                                    candidate_fit *fit, napa_detection *result, image_view *result_view, pass_timer *timer) {
    const int W = det->fam.width, width = view->w, height = view->h, stride = view->stride;
    const double cells = det->fam.cells;
    const uint8_t *img = view->p;
    /* On cells of a pixel or two, moving a side by a tenth of a pixel can flip one bit, so the refined corners can
     * miss a bit the first fit read. An undersized tag whose first fit read its code with at most one wrong bit may
     * then correct one bit, and only to that same ID. Families whose codes are too short for pieces to decode
     * (piece_hamming < 0) stay exact, since a chance match within one bit is not rare enough there. */
    decode_policy one_bit_policy;
    napa_params one_bit;
    if (policy->require_exact == 1 && fit->early.hamming <= 1 && det->fam.piece_hamming >= 0 &&
        policy->decode_params->max_hamming == 0 && det->fam.max_hamming >= 1) {
        one_bit = *policy->decode_params, one_bit.max_hamming = 1;
        one_bit_policy = *policy, one_bit_policy.decode_params = &one_bit, one_bit_policy.require_exact = 2;
        policy = &one_bit_policy;
    }
    refined_quad q;
    q.edge_w = fit->edge_w;
    q.lv = fit->lv;
    image_view lin_view = *view;
    const image_view *rview = view;
    int lin_tried = 0, raw_frame_sides = 0;
    vec2 raw_fine[4];
    double raw_reach = 0, raw_edge_w = 0;
    edge_levels raw_lv;
refine_candidate:
    q.cell = 0;
    for (int k = 0; k < 4; k++) {
        double ex = fit->fine[(k + 1) & 3].x - fit->fine[k].x, ey = fit->fine[(k + 1) & 3].y - fit->fine[k].y;
        q.cell += sqrt(ex * ex + ey * ey) / (4.0 * W);
    }
    /* With a camera gamma set, refine on a tone-linearized copy of the tag's own rectangle: blur is linear in light,
     * so edges there sit where the fits expect. A tag too dark to clear the contrast floor in linear terms stays raw. */
    if (det->has_camera_gamma && !lin_tried && linear_contrast_enough(det, &q.lv, policy->params)) {
        memcpy(raw_fine, fit->fine, sizeof raw_fine), raw_reach = fit->reach, raw_edge_w = q.edge_w, raw_lv = q.lv;
        raw_frame_sides = fit->frame_sides;
        double x0 = fit->fine[0].x, x1 = fit->fine[0].x, y0 = fit->fine[0].y, y1 = fit->fine[0].y;
        for (int k = 1; k < 4; k++) {
            if (fit->fine[k].x < x0) x0 = fit->fine[k].x;
            if (fit->fine[k].x > x1) x1 = fit->fine[k].x;
            if (fit->fine[k].y < y0) y0 = fit->fine[k].y;
            if (fit->fine[k].y > y1) y1 = fit->fine[k].y;
        }
        const double margin = NAPA_LINEAR_MARGIN_CELLS * q.cell + NAPA_LINEAR_MARGIN_PIXELS;
        int ix0 = (int)(x0 - margin), ix1 = (int)(x1 + margin) + 1, iy0 = (int)(y0 - margin), iy1 = (int)(y1 + margin) + 1;
        if (ix0 < 0) ix0 = 0;
        if (iy0 < 0) iy0 = 0;
        if (ix1 > width - 1) ix1 = width - 1;
        if (iy1 > height - 1) iy1 = height - 1;
        if (det->linear_capacity < (size_t)width * height) {
            free(det->linear_frame);
            det->linear_frame = (uint8_t *)calloc((size_t)width * height, 1);
            det->linear_capacity = det->linear_frame != NULL ? (size_t)width * height : 0;
            if (det->linear_frame == NULL) return REFINE_NO_MEMORY;
        }
        if (ix1 > ix0 && iy1 > iy0) {
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
        double step = 2.0 * fit->reach / 14.0;
        double next = 1.5 * step;
        if (next < 1.5) next = 1.5;
        if (next < 1.25 * q.edge_w) next = 1.25 * q.edge_w;
        /* ...but within half a cell, so the window never reaches the next payload edge. */
        if (next > 0.45 * q.cell && 0.45 * q.cell >= 1.0) next = 0.45 * q.cell;
        if (pass > 0 && (next >= 0.9 * fit->reach || step <= 0.35)) break;
        fit->reach = next;
        ok = fit_sides(rview, fit->fine, q.finer, fit->reach, fit->reach, policy->params->min_contrast, 0, 8, &q.edge_w, &q.lv, cells, &fit->frame_sides);
        if (!ok) break;
        memcpy(fit->fine, q.finer, sizeof fit->fine);
    }
    /* One more pass at the converged window, re-centred on the last estimate, with every
     * edge sample. */

    q.lens_ok = 0;
    q.side_widths_seen[0] = 1; q.side_widths_seen[1] = 0; /* the widest side's edge over the narrowest, and the widest */
    q.accepted_blur = 0;
    if (ok) ok = fit_sides_alternate(rview, fit->fine, q.finer, fit->reach, fit->reach, policy->params->min_contrast, 0, NAPA_FINAL_POINTS, &q.edge_w, &q.lv, cells, q.alternative, q.curved, q.lens_quad, &q.lens_ok, q.side_widths_seen, &fit->frame_sides);
    if (!ok && rview != view) { /* the linear patch failed: go again on the frame's own pixels */
        lin_tried = 1, rview = view;
        memcpy(fit->fine, raw_fine, sizeof fit->fine), fit->reach = raw_reach, q.edge_w = raw_edge_w, q.lv = raw_lv;
        fit->frame_sides = raw_frame_sides;
        goto refine_candidate;
    }
    if (!ok) {
        PROF_MARK(PROF_REFINE);
        return REFINE_FIT_FAILED;
    }
    PROF_MARK(PROF_REFINE);

    q.final_lv.black = 255; q.final_lv.white = 0; q.final_lv.has_model = 0; q.final_lv.model_pending = 0;
    uint64_t code = 0;
    vec2 second = q.finer[1];
    int decoded_ = decode_quad(det, rview, q.finer, policy->decode_params, 0, &q.d, q.side_level, &q.final_lv, &code);
    if (decoded_ && (!exact_enough(policy, q.d.hamming) || (policy->require_exact == 2 && q.d.id != fit->early.id))) decoded_ = 0;
    /* The ordinary fit may fail decoding because its edge levels were mixed by blur.
     * Reuse the already-fitted alternate, with the same strict decode and path limits. */
    if (!decoded_ && !fit->frame_sides && q.lv.has_model) {
        /* Preserve independent-light recovery when ordinary decoding failed. */
        int alternative_ok = view->invert || fit_independent_alternate(rview, fit->fine, q.alternative,
                                    fit->reach, policy->params->min_contrast, cells, &q.lv);
        if (alternative_ok) {
            if (signed_area(q.alternative) < 0) {
                vec2 t = q.alternative[1]; q.alternative[1] = q.alternative[3]; q.alternative[3] = t;
            }
            double move = 0;
            for (int k = 0; k < 4; k++)
                move += fabs(q.alternative[k].x - q.finer[k].x) + fabs(q.alternative[k].y - q.finer[k].y);
            if (move > 0.02 &&
                decode_quad(det, rview, q.alternative, policy->decode_params, 0, &q.d, q.side_level, &q.final_lv,
                            &code) &&
                exact_enough(policy, q.d.hamming) && q.d.id == fit->early.id && q.d.hamming <= policy->params->max_hamming && q.d.hamming <= det->fam.max_hamming &&
                (!policy->shadow || q.d.hamming == 0) && (!policy->piece || q.d.hamming <= det->fam.piece_hamming)) {
                q.accepted_blur = 1;
                memcpy(q.finer, q.alternative, sizeof q.finer);
                decoded_ = 1;
            }
        }
    }
    PROF_MARK(PROF_DECODE2);
    if (!decoded_ && rview != view) {
        lin_tried = 1, rview = view;
        memcpy(fit->fine, raw_fine, sizeof fit->fine), fit->reach = raw_reach, q.edge_w = raw_edge_w, q.lv = raw_lv;
        fit->frame_sides = raw_frame_sides;
        goto refine_candidate;
    }
    if (!decoded_) {
        return REFINE_DECODE_FAILED;
    }
    /* A side on the frame edge stands in for a cut-off one: measure where its corners really are. Decoding
     * may have reversed the winding (swapped corners 1 and 3), which reverses the side order. */
    q.cut = fit->frame_sides;
    if (q.cut && (q.finer[1].x != second.x || q.finer[1].y != second.y))
        q.cut = (q.cut & 1) << 3 | (q.cut & 2) << 1 | (q.cut & 4) >> 1 | (q.cut & 8) >> 3;
    if (q.cut) {
        vec2 extended[4];
        napa_detection d1;
        if (!extend_cut_sides(&det->fam, rview, q.finer, q.cut, code, q.final_lv.black, q.final_lv.white, extended) ||
            !decode_quad(det, rview, extended, policy->decode_params, 0, &d1, NULL, NULL, NULL) ||
            !exact_enough(policy, d1.hamming) || d1.id != q.d.id) {
            return REFINE_CUT_FAILED; /* cut extension failed without changing the component fate */
        }
        q.d = d1;
        memcpy(q.finer, extended, sizeof q.finer);
    }
    border_correction correction;
    correct_border_geometry(det, rview, policy, &q, &correction);
    select_curve_geometry(det, rview, policy, &q, &correction);
    if (polish_decoded_corners(det, rview, policy, &q, &correction) < 0) return REFINE_NO_MEMORY;
    *result_view = *rview;
    *result = q.d;
    return REFINE_ACCEPTED;
}

typedef struct pass_config {
    image_view view;
    const napa_params *params;
    const uint8_t *pimg;
    int pw, ph, pstride, base, lo, hi, split, watch, shadow, shaped;
    double sx, sy, offx, offy, smax;
} pass_config;

/* Counts and restart state persist while each level's scratch is reused. */
typedef struct {
    napa_detection *out;
    int max_out, found, piece_found, shape_found;
    size_t fine_whole_count, fate_off, fates_kept;
    int necks, redone, touching;
} pass_outputs;

typedef struct {
    const uint8_t *gray;
    int prev_lw, floor_stride, contrast;
    int boxed; /* some level so far was replaced by box averages, so every later one samples them */
    int prev_s; /* the previous level's stride */
} level_sampling;

typedef struct {
    int s, lw, lh, last, redo;
    int nruns, ncand, npieces, fine, covered_prefix;
    int extremes_deferred; /* threads: the candidates' extremes are left to the pre-pass (memo_candidates) */
} candidate_level;

static int sample_threshold_level(napa_detector *det, const pass_config *config, const pass_outputs *outputs,
                                  level_sampling *sampling, candidate_level *level, pass_timer *timer) {
    image_view view = config->view;
    const napa_params *params = config->params;
    const uint8_t *pimg = config->pimg;
    int pw = config->pw;
    int ph = config->ph;
    int pstride = config->pstride;
    int base = config->base;
    int split = config->split;
    int shadow = config->shadow;
    int lazy_rows = 0;
    det->rake_deferred = 0;

    size_t pixels = (size_t)level->lw * level->lh;
    size_t tiles = (size_t)((level->lw + TILE - 1) / TILE) * ((level->lh + TILE - 1) / TILE);
    if (ensure_buffers(det, pixels, tiles, (size_t)level->lw, pixels / 2 + (size_t)level->lh + 1) != 0) return -2;

    PROF_START;
    /* Contiguous, non-inverted stride-one gray can be read directly. Coarser
     * samples always go into owned scratch; later levels keep the in-place rake. */
    if (level->s == base) {
        if (level->s == 1 && pstride == pw && !view.invert) sampling->gray = pimg;
        else {
            /* The noise floors read every 8th row. Sample only those first: a noisy frame re-rakes
             * every sample with block averages, and the rest would be thrown away. */
            lazy_rows = level->s > 1 && level->s <= 16;
            /* With threads, the floors' pool run rakes the every-8th rows and starts on the rest, which the
             * threshold's raker finishes (contrast_floors_raked, threshold_banded). */
            if (lazy_rows && pool_for(det, pixels) != NULL) lazy_rows = 2;
            else for (int y = 0; y < level->lh; y += lazy_rows ? 8 : 1)
                gather_row(det->level + (size_t)y * level->lw, pimg + (size_t)y * level->s * pstride, level->lw, level->s, det->ssse3);
            if (view.invert && !lazy_rows) invert_bytes(det->level, pixels);
            sampling->gray = det->level;
        }
    } else {
        for (int y = 0; y < level->lh; y++) {
            const int step = level->s / sampling->prev_s; /* 2, or 4 where the ordinary pass skips a level */
            const uint8_t *src = sampling->gray + (size_t)(step * y) * sampling->prev_lw;
            if (sampling->gray == det->level && step == 2) gather2(det->level + (size_t)y * level->lw, src, level->lw);
            /* Unlike owned scratch, an odd borrowed final row has no spare byte. */
            else gather_row(det->level + (size_t)y * level->lw, src, level->lw, step, det->ssse3);
        }
        sampling->gray = det->level;
    }
    sampling->prev_lw = level->lw, sampling->prev_s = level->s;
    PROF_MARK(PROF_RAKE);
    if (level->s == base || (base == 1 && level->s == 2 && sampling->contrast > NAPA_NOISY_FLOOR)) {
        /* Do not blur stride-one proposals: a tiny border can be just one pixel wide. Its
         * higher floor suppresses raw noise; the next averaged level restores low contrast.
         * A terminal level has no recovery pass, so keep its original contrast floor. */
        sampling->floor_stride = level->s;
        double sigmas = level->s == 1 && !level->last ? 7.0 : NAPA_FLOOR_SIGMAS;
        int cap = level->s == 1 && !level->last ? 255 : NAPA_FLOOR_CAP;
        const gather_job rake = {det, pimg, pstride, level->lw, level->lh, level->s, view.invert};
        sampling->contrast = contrast_floors_raked(det, sampling->gray, level->lw, level->lh, params->min_contrast, sigmas, cap,
                                                   lazy_rows == 2 ? &rake : NULL);
        if (lazy_rows == 1 && !(sampling->contrast > NAPA_NOISY_FLOOR && level->s > 1 && level->s <= 16)) {
            gather_job job = {det, pimg, pstride, level->lw, level->lh, level->s, view.invert};
            pool_run(pool_for(det, pixels), gather_rest, &job, (level->lh + 63) / 64);
        }
        /* A noisy frame: rake again, averaging each sample's block, and re-measure the noise. */
        if (sampling->contrast > NAPA_NOISY_FLOOR && level->s > 1 && level->s <= 16) {
            if ((size_t)pw > det->box_capacity) {
                uint16_t *p = (uint16_t *)realloc(det->box_rows, sizeof(uint16_t) * 2 * (size_t)pw);
                if (p == NULL) return -2;
                det->box_rows = p;
                det->box_capacity = (size_t)pw;
            }
            det->rake_deferred = 0; /* the box rake replaces every row */
            sampling->boxed = 1;
            box_rake(det->level, level->lw, level->lh, pimg, pw, ph, pstride, level->s, base == 1 && level->s == 2 ? 3 : level->s, det->box_rows, det->box_rows + pw);
            if (view.invert) invert_bytes(det->level, pixels);
            /* Stride-two 3x3 means share one column: their difference measures sqrt(2/3)
             * times their noise sigma. Disjoint averaging blocks need no correction. */
            double sigmas = base == 1 && level->s == 2 ? NAPA_BOX_SIGMAS * sqrt(1.5) : NAPA_BOX_SIGMAS;
            sampling->contrast = contrast_floors(det, det->level, level->lw, level->lh, params->min_contrast, sigmas, NAPA_FLOOR_CAP);
        }
        if (sampling->contrast < 0) return -2;
    }
    det->level_contrast_min = !shadow && (level->s == base || (base > 1 && level->s == 2 * base)) && !level->last
        ? (uint8_t)(params->min_contrast * NAPA_FINEST_CONTRAST_NUM / NAPA_FINEST_CONTRAST_DEN > 255
                        ? 255 : params->min_contrast * NAPA_FINEST_CONTRAST_NUM / NAPA_FINEST_CONTRAST_DEN)
        : 0;
    /* The ordinary pass's stride-two finest level, and the stride-eight level that follows it (stride four is skipped,
     * see detect_pass): there a half tile's neighbourhood spans the 48 px the skipped level's tiles did. */
    det->level_half = !shadow && base == 2 && (level->s == base || (!config->split && level->s == 4 * base));

    det->level_view = sampling->gray == det->level ? NULL : sampling->gray;
    /* Keyed by sampling step: the passes can start from different steps. On one frame, with one contrast setting, a
     * step holds the same samples in two passes that started from the same step (every boxing decision repeats), or
     * in two that never box-averaged (both took every s-th pixel). Otherwise one may hold averages the other doesn't. */
    saved_extrema *saved = level->s < 32 ? &det->saved[level->s] : NULL;
    const int reuse = saved != NULL && saved->valid && (saved->base == base || (!saved->boxed && !sampling->boxed)) &&
        ((shadow && det->extrema_ready) || (!shadow && level->redo && det->extrema_wanted));
    det->level_extrema_in = reuse ? saved->tiles : NULL;
    det->level_extrema_out = NULL;
    if (det->extrema_wanted && !reuse && saved != NULL) {
        if (2 * tiles > saved->capacity) {
            uint8_t *p = (uint8_t *)realloc(saved->tiles, 2 * tiles);
            if (p == NULL) return -2;
            saved->tiles = p;
            saved->capacity = 2 * tiles;
        }
        saved->base = base, saved->boxed = sampling->boxed, saved->valid = 0;
        det->level_extrema_out = saved->tiles;
    }
    level->nruns = threshold_level(det, level->lw, level->lh, level->s / sampling->floor_stride, split ? 1 + outputs->necks : 0, shadow);
    if (level->nruns >= 0 && det->level_extrema_out) saved->valid = 1;
    det->level_view = NULL; /* clear even on threshold allocation failure */
    det->level_extrema_out = NULL, det->level_extrema_in = NULL;
    if (level->nruns < 0) return -2;
    PROF_MARK(PROF_THRESHOLD);
    return 0;
}

/* Exchange the level buffers (runs, parents, slots, boxes, dark bits, labelling scratch) with the spare set. */
static void run_set_swap(napa_detector *det) {
#define NAPA_SWAP(T, a, b) do { T t_ = a; a = b; b = t_; } while (0)
    NAPA_SWAP(napa_run *, det->runs, det->spare.runs);
    NAPA_SWAP(int32_t *, det->parent, det->spare.parent);
    NAPA_SWAP(int32_t *, det->slot, det->spare.slot);
    NAPA_SWAP(blob_box *, det->boxes, det->spare.boxes);
    NAPA_SWAP(size_t, det->run_capacity, det->spare.run_capacity);
    NAPA_SWAP(uint64_t *, det->level_bits, det->spare.level_bits);
    NAPA_SWAP(size_t, det->level_bits_capacity, det->spare.level_bits_capacity);
    NAPA_SWAP(label_scratch, det->scratch, det->spare.scratch);
#undef NAPA_SWAP
    det->runs_swapped = !det->runs_swapped;
}

/* Exchange the candidate buffers (candidates, pre-pass results and order, run lists) with the spare set. */
static void cand_set_swap(napa_detector *det) {
#define NAPA_SWAP(T, a, b) do { T t_ = a; a = b; b = t_; } while (0)
    NAPA_SWAP(component *, det->cands, det->spare.cands);
    NAPA_SWAP(size_t, det->cand_capacity, det->spare.cand_capacity);
    NAPA_SWAP(cand_memo *, det->memo, det->spare.memo);
    NAPA_SWAP(int32_t *, det->memo_order, det->spare.memo_order);
    NAPA_SWAP(size_t, det->memo_capacity, det->spare.memo_capacity);
    NAPA_SWAP(int32_t *, det->cand_runs, det->spare.cand_runs);
    NAPA_SWAP(int32_t *, det->cand_run_end, det->spare.cand_run_end);
    NAPA_SWAP(size_t, det->cand_runs_capacity, det->spare.cand_runs_capacity);
    NAPA_SWAP(size_t, det->cand_run_end_capacity, det->spare.cand_run_end_capacity);
#undef NAPA_SWAP
    det->cands_swapped = !det->cands_swapped;
}

/* threads: the next level of an ordinary pass, sampled and labelled alongside this level's pre-pass. It depends on
 * nothing this level's candidates decide: sampling reads the frame, the previous level's samples and settings fixed for
 * the pass. The pre-pass reads this level's runs and dark bits through its job, and the next level gets the spare set. */
typedef struct next_level_job {
    napa_detector *det;
    const pass_config *config;
    const pass_outputs *outputs;
    level_sampling *sampling;
    candidate_level *level;
    int status, done;
    /* early: the next level is the terminal one, and its candidates (the spare candidate set) are assembled here too,
     * before this level's detections are known; its pre-pass joins this level's pool (pool_ctx) as job coarse. */
    int early, assembled, memo_ready;
    const void *pool_ctx;
    memo_job coarse;
} next_level_job;

static int assemble_level_candidates(napa_detector *det, const pass_config *config, const pass_outputs *outputs,
                                     candidate_level *level, pass_timer *timer);

static void sample_next_level(next_level_job *job, int worker) {
    pass_timer timing = {0};
    job->status = sample_threshold_level(job->det, job->config, job->outputs, job->sampling, job->level, &timing);
    if (job->status == 0 && job->early) {
        /* The outputs are this level's starting ones until its loop: assembly drops what they cover now, and
         * keep_uncovered what this level's loop adds. */
        napa_detector *det = job->det;
        const pass_config *config = job->config;
        candidate_level *level = job->level;
        job->status = assemble_level_candidates(det, config, job->outputs, level, &timing);
        job->assembled = job->status == 0;
        const int ncand = level->ncand;
        if (job->assembled && ncand > 0 && memo_reserve(det, ncand)) {
            int at[33] = {0}; /* largest boxes first, as process_level_candidates orders them */
            for (int ci = 0; ci < ncand; ci++) at[area_rank(&det->cands[ci]) + 1]++;
            for (int b = 0; b < 32; b++) at[b + 1] += at[b];
            for (int ci = 0; ci < ncand; ci++) det->memo_order[at[area_rank(&det->cands[ci])]++] = ci;
            const memo_job c = {det, &config->view, config->params, level->s, config->base, config->lo, level->last,
                                config->shadow, config->shaped, config->sx, config->sy, config->offx, config->offy,
                                config->smax, det->fam.cells, ncand, 0, det->memo_order,
                                level->extremes_deferred ? det->runs : NULL, det->level_bits, (level->lw + 63) / 64 + 1,
                                NULL, config, ncand, 0, det->cands, det->memo, det->cand_runs, det->cand_run_end, NULL, 0, 1};
            job->coarse = c;
            job->memo_ready = 1;
            if (!pool_extend(det->pool, job->pool_ctx, ncand))
                for (int k = 0; k < ncand; k++) memo_candidates(&job->coarse, k, worker);
        }
    }
    job->done = 1;
}

/* The terminal level's kept candidates with a first fit (sample_next_level's fit_only pre-pass), decoded and refined
 * ahead on the pool, one per index, as memo_candidates would have. */
typedef struct {
    memo_job job;
    const int32_t *list;
} kept_decode_job;

static void decode_kept(void *ctx, int index, int worker) {
    const kept_decode_job *k = (const kept_decode_job *)ctx;
    const memo_job *j = &k->job;
    napa_detector *det = j->det;
    const int ci = k->list[index];
    cand_memo *m = &j->memo[ci];
    const component *comp = &j->cands[ci];
    const int bw = comp->maxx - comp->minx + 1, bh = comp->maxy - comp->miny + 1;
    double best_area, perim_q;
    rough_quad_too_full(&det->fam, comp, m->pick, &best_area, &perim_q); /* false: the pre-pass went on to fit */
    vec2 rough[4];
    rough_corners(m->pick, j->s, j->sx, j->sy, j->offx, j->offy, rough);
    decode_ahead(j, worker == 0 ? det : &det->workers[worker - 1], m, comp, bw, bh, best_area, perim_q, rough);
}

/* The terminal level assembled early (sample_next_level): drop the candidates the previous level's detections cover,
 * as assembly would have, keeping the rest and their pre-pass results in order. */
static void keep_uncovered(napa_detector *det, const pass_config *config, const pass_outputs *outputs,
                           candidate_level *level, int memo) {
    const int s = level->s, found = outputs->found < outputs->max_out ? outputs->found : outputs->max_out;
    const double sx = config->sx, sy = config->sy, offx = config->offx, offy = config->offy;
    int n = 0;
    for (int ci = 0; ci < level->ncand; ci++) {
        const component *c = &det->cands[ci];
        const int t_minx = c->minx, t_maxx = c->maxx, t_miny = c->miny, t_maxy = c->maxy;
        const int bw = t_maxx - t_minx + 1, bh = t_maxy - t_miny + 1;
        if (covered_footprint(outputs->out, det->out_centers, found,
                              0.5 * (t_minx + t_maxx) * s * sx + offx, 0.5 * (t_miny + t_maxy) * s * sy + offy,
                              0.3 * s * (bw * sx > bh * sy ? bw * sx : bh * sy) + 2.0)) continue;
        if (n != ci) {
            det->cands[n] = det->cands[ci];
            if (memo) det->memo[n] = det->memo[ci];
        }
        n++;
    }
    level->ncand = n;
    level->covered_prefix = found;
    level->extremes_deferred = 0; /* complete: the pre-pass added them */
    /* Two or more left to decode: decode them ahead on the pool, as the pre-pass would have. */
    int nk = 0;
    if (memo && can_decode_ahead(det))
        for (int ci = 0; ci < n; ci++) nk += det->memo[ci].fit > 0;
    if (nk >= 2) {
        int32_t *list = det->memo_order; /* free: the pre-pass is done */
        nk = 0;
        for (int ci = 0; ci < n; ci++) if (det->memo[ci].fit > 0) list[nk++] = ci;
        kept_decode_job kj;
        memset(&kj, 0, sizeof kj);
        kj.job.det = det, kj.job.view = &config->view, kj.job.params = config->params, kj.job.s = s;
        kj.job.base = config->base, kj.job.lo = config->lo, kj.job.last = level->last, kj.job.shadow = config->shadow;
        kj.job.shaped = config->shaped, kj.job.sx = sx, kj.job.sy = sy, kj.job.offx = offx, kj.job.offy = offy;
        kj.job.smax = config->smax, kj.job.cells = det->fam.cells, kj.job.n = n, kj.job.ahead = 1, kj.job.config = config;
        kj.job.cands = det->cands, kj.job.memo = det->memo;
        kj.list = list;
        prepare_workers(det);
        pool_run(det->pool, decode_kept, &kj, nk);
    }
}

/* Whether a level's ordinary candidates get the threaded pre-pass (memo_candidates). (A camera's lens memo isn't
 * shared safely.) */
static int level_pre_pass_possible(const napa_detector *det, const image_view *view, const candidate_level *level) {
    return det->pool != NULL && view->lens == NULL && !level->redo;
}
static int level_pre_pass(const napa_detector *det, const image_view *view, const candidate_level *level) {
    return level_pre_pass_possible(det, view, level) && level->ncand >= 2 * MEMO_CHUNK;
}

/* Room for the candidate run lists (memo_candidates): a next run for each of the level's runs, and a tail for each of
 * its components. Returns 0 when out of memory. */
static int reserve_run_lists(napa_detector *det, int nruns) {
    if ((size_t)nruns > det->cand_runs_capacity) {
        int32_t *p = (int32_t *)realloc(det->cand_runs, (size_t)nruns * sizeof(int32_t));
        if (p == NULL) return 0;
        det->cand_runs = p, det->cand_runs_capacity = (size_t)nruns;
    }
    if ((size_t)nruns > det->cand_run_end_capacity) { /* candidates are at most the components, at most the runs */
        int32_t *p = (int32_t *)realloc(det->cand_run_end, (size_t)nruns * sizeof(int32_t));
        if (p == NULL) return 0;
        det->cand_run_end = p, det->cand_run_end_capacity = (size_t)nruns;
    }
    return 1;
}

static int assemble_level_candidates(napa_detector *det, const pass_config *config, const pass_outputs *outputs,
                                     candidate_level *level, pass_timer *timer) {
    level->extremes_deferred = 0;
    int base = config->base;
    int lo = config->lo;
    int hi = config->hi;
    int split = config->split;
    int shadow = config->shadow;
    double sx = config->sx;
    double sy = config->sy;
    double offx = config->offx;
    double offy = config->offy;
    const int W = det->fam.width;
    const int s = level->s;
    /* Resolve runs into packed components, preserving root raster order. */
    napa_run *R = det->runs;
    int32_t *P = det->parent;
    blob_box *B = det->boxes;
    int32_t nroots = 0;
    const int paired = !split && !shadow; /* row-pair runs (label_row_pair) */
    /* With threads, the candidates' extremes may be left to the pre-pass: thread each component's runs into a
     * circular list in run order as they resolve (next run in cand_runs; the tail in the box's count, which only
     * splitting reads), for each candidate's job to walk (memo_candidates). */
    const int listed = paired && level_pre_pass_possible(det, &config->view, level) && reserve_run_lists(det, level->nruns);
    int32_t *next_run = det->cand_runs;
    if (listed) {
        for (int i = 0; i < level->nruns; i++) {
            const uint16_t miny = (uint16_t)(R[i].y + !(R[i].rows & 1)), maxy = (uint16_t)(R[i].y + (R[i].rows >> 1));
            int32_t r;
            if (P[i] == i) {
                r = nroots++;
                B[r].count = i, next_run[i] = i;
                B[r].minx = R[i].x0, B[r].maxx = R[i].x1;
                B[r].miny = miny, B[r].maxy = maxy;
            } else {
                r = -P[P[i]] - 1;
                const int32_t tail = B[r].count;
                next_run[i] = next_run[tail], next_run[tail] = i, B[r].count = i;
                B[r].minx = R[i].x0 < B[r].minx ? R[i].x0 : B[r].minx;
                B[r].maxx = R[i].x1 > B[r].maxx ? R[i].x1 : B[r].maxx;
                B[r].maxy = maxy > B[r].maxy ? maxy : B[r].maxy;
            }
            P[i] = -r - 1;
        }
    } else if (paired) {
        /* Row-pair runs: the same, with each run's rows giving its first and last row. A blob's
         * first run holds its top row; later runs can end above earlier ones. Splitting alone
         * reads blob counts. */
        for (int i = 0; i < level->nruns; i++) {
            const uint16_t miny = (uint16_t)(R[i].y + !(R[i].rows & 1)), maxy = (uint16_t)(R[i].y + (R[i].rows >> 1));
            int32_t r;
            if (P[i] == i) {
                r = nroots++;
                B[r].count = 0;
                B[r].minx = R[i].x0, B[r].maxx = R[i].x1;
                B[r].miny = miny, B[r].maxy = maxy;
            } else {
                r = -P[P[i]] - 1;
                B[r].minx = R[i].x0 < B[r].minx ? R[i].x0 : B[r].minx;
                B[r].maxx = R[i].x1 > B[r].maxx ? R[i].x1 : B[r].maxx;
                B[r].maxy = maxy > B[r].maxy ? maxy : B[r].maxy;
            }
            P[i] = -r - 1;
        }
    } else for (int i = 0; i < level->nruns; i++) {
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
    level->npieces = 0; level->fine = 0;
    if (det->diag_n || det->neck_n) {
        level->npieces = split_pieces(det, lo, level->last ? 1 << 30 : hi, !level->last && base == 1 && s <= 2, &level->fine);
        if (level->npieces < 0) return -2;
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
    int max_size = level->last ? 1 << 30 : hi;
    level->ncand = 0;
    /* Unsplit candidate boxes are checked against this stable prefix before extrema. */
    level->covered_prefix = !split ? (outputs->found < outputs->max_out ? outputs->found : outputs->max_out) : 0;
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
        if ((shadow || !split) && covered_footprint(outputs->out, det->out_centers, outputs->found < outputs->max_out ? outputs->found : outputs->max_out,
                              0.5 * (t_minx + t_maxx) * s * sx + offx, 0.5 * (t_miny + t_maxy) * s * sy + offy,
                              0.3 * s * (bw * sx > bh * sy ? bw * sx : bh * sy) + 2.0)) continue;
        if ((size_t)level->ncand >= det->cand_capacity) {
            size_t cap = det->cand_capacity ? 2 * det->cand_capacity : 256;
            component *c = (component *)realloc(det->cands, cap * sizeof(component));
            if (c == NULL) return -2;
            det->cands = c;
            det->cand_capacity = cap;
        }
        component *c = &det->cands[level->ncand];
        c->count = 0;
        for (int k = 0; k < 16; k++) c->extv[k] = INT32_MIN;
        c->work.rows.y = c->work.rows.pair_y = -1;
        c->minx = t_minx, c->maxx = t_maxx, c->miny = t_miny, c->maxy = t_maxy;
        if (listed) det->cand_run_end[level->ncand] = B[i].count;
        det->slot[i] = level->ncand++;
    }
    if (listed) for (int i = 0; i < nroots; i++) B[i].count = 0;
    /* Extreme points only for the candidates: one more pass over the runs. */
    if (det->diag_n || det->neck_n) {
        /* Runs carry their piece. A joined piece takes the slot of what joined it: its blob's
         * candidate, or the corner-stage piece holding it. A proposed piece takes ncand + its index,
         * and passes its runs on to that slot next. Blobs' candidates are read before any changes. */
        for (int k = 0; k < level->npieces; k++) {
            int32_t root = joined_root(B, det->piece_root[k]);
            const blob_box *box = &B[root];
            int candidate = det->slot[root];
            /* Sparse or deferred fine parents contribute extrema without controlling their
             * pieces' eligibility through a failure fate or the stride-one shortcut. */
            if (candidate >= 0 && (box_is_sparse(box->count, box->maxx - box->minx + 1,
                                                box->maxy - box->miny + 1) ||
                dense_whole_needs_deferral(base, s, level->last, box->maxx - box->minx + 1,
                                  box->maxy - box->miny + 1))) candidate = -1;
            det->piece_cand[k] = candidate;
        }
        for (size_t e = 0; e < det->diag_n; e++) {
            int32_t r = det->diag[e];
            if (B[r].count < 0) det->slot[r] = det->slot[joined_root(B, r)];
        }
        /* At stride one, pieces of candidate blobs are never decoded (below): their runs skip them. */
        for (int k = level->fine; k < level->npieces; k++) {
            det->piece_next[k] = det->slot[det->piece_root[k]];
            if (s > 1 || det->piece_cand[k] < 0) det->slot[det->piece_root[k]] = level->ncand + k;
        }
        for (size_t e = 0; e < det->neck_n; e++)
            if (det->neck_root[e] != det->neck[e]) det->slot[det->neck[e]] = det->slot[det->neck_root[e]];
        for (int k = 0; k < level->fine; k++) {
            det->piece_next[k] = det->slot[det->piece_root[k]];
            if (s > 1 || det->piece_cand[k] < 0) det->slot[det->piece_root[k]] = level->ncand + k;
        }
    }
    /* With threads, each candidate's pre-pass job adds its own runs instead (memo_candidates). */
    level->extremes_deferred = listed && level->npieces == 0 && level_pre_pass(det, &config->view, level) &&
        memo_reserve(det, level->ncand);
    if (level->extremes_deferred) {
        /* fates and piece counts too, once each candidate's extremes are complete */
    } else if (paired) {
        const uint64_t *bits = det->level_bits;
        const int stride = (level->lw + 63) / 64 + 1;
        /* The candidates' runs, gathered without a branch per run into the blob boxes' storage (no longer read
         * without splitting), then added in the same order. */
        int32_t *list = (int32_t *)(void *)B;
        int nl = 0;
        for (int i = 0; i < level->nruns; i++) {
            const int32_t c = det->slot[-P[i] - 1];
            list[2 * nl] = i, list[2 * nl + 1] = c;
            nl += c >= 0;
        }
        for (int k = 0; k < nl; k++) comp_add_pair_run(&det->cands[list[2 * k + 1]], &R[list[2 * k]], bits, stride);
        for (int ci = 0; ci < level->ncand; ci++) comp_flush_pair(&det->cands[ci], bits, stride);
    } else if (level->npieces == 0) {
        for (int i = 0; i < level->nruns; i++) {
            int32_t c = det->slot[-P[i] - 1];
            if (c >= 0) comp_add_run(&det->cands[c], &R[i]);
        }
    } else {
        for (int i = 0; i < level->nruns; i++) {
            int32_t c = det->slot[-P[i] - 1];
            for (; c >= level->ncand; c = det->piece_next[c - level->ncand]) comp_add_run(&det->pieces[c - level->ncand], &R[i]);
            if (c >= 0) comp_add_run(&det->cands[c], &R[i]);
        }
        for (int k = 0; k < level->npieces; k++)
            if (det->pieces[k].work.rows.y >= 0) comp_flush_row(&det->pieces[k]); /* not: pieces given no runs */
    }
    if (!paired) for (int ci = 0; ci < level->ncand; ci++) comp_flush_row(&det->cands[ci]);
    /* All row extrema are complete. Candidates now carry rejection stages and piece counts. */
    if (!level->extremes_deferred) for (int ci = 0; ci < level->ncand; ci++) {
        det->cands[ci].work.candidate.fate = CANDIDATE_UNTRIED;
        det->cands[ci].work.candidate.piece_count = 0;
    }
    for (int k = 0; k < level->npieces; k++) {
        det->pieces[k].work.candidate.fate = CANDIDATE_UNTRIED;
        if (det->piece_cand[k] >= 0) det->cands[det->piece_cand[k]].work.candidate.piece_count++;
    }
    if (level->redo && outputs->fate_off + (size_t)level->ncand + (size_t)(level->npieces - level->fine) <= outputs->fates_kept) {
        /* The first pass's fates: the corner-stage pieces are the same pieces, in the same order. */
        for (int ci = 0; ci < level->ncand; ci++) det->cands[ci].work.candidate.fate = det->fates[outputs->fate_off + ci];
        for (int k = level->fine; k < level->npieces; k++) det->pieces[k].work.candidate.fate = det->fates[outputs->fate_off + level->ncand + k - level->fine];
    }
    PROF_MARK(PROF_BLOBS);
    return 0;
}

/* Regrowth uses full-resolution extrema and its own fixed fit window. */
static int regrow_and_fit(napa_detector *det, const image_view *view, const napa_params *params,
                          const vec2 rough[4], double extent, vec2 fitted[4], double *edge_width) {
    component grown;
    int ox = 0, oy = 0;
    int status = regrow_small(det, view, rough, extent, params->min_contrast, &grown, &ox, &oy);
    if (status <= 0) return status;
    if (grown.maxx - grown.minx + 1 < 8 || grown.maxy - grown.miny + 1 < 8) return 0;
    int pick[4][2];
    vec2 quad[4];
    rough_pick(&grown, 0, pick);
    for (int k = 0; k < 4; k++) quad[k].x = pick[k][0] + ox, quad[k].y = pick[k][1] + oy;
    *edge_width = 0;
    return fit_initial_sides(view, quad, fitted, 2.5, 1.5, params->min_contrast, edge_width,
                             det->fam.cells, NULL);
}

static int decode_initial_candidate(napa_detector *det, const image_view *view, const napa_params *params,
                                    const napa_params *decode_params, candidate_fit *fit,
                                    const vec2 rough[4], double regrow_extent) {
    const int W = det->fam.width;
    const double cells = det->fam.cells;
    vec2 finer[4];
    vec2 probe[4];
    fit->lv.black = 255; fit->lv.white = 0; fit->lv.has_model = 0; fit->lv.model_pending = 0;
    memcpy(probe, fit->fine, sizeof probe);
    if (decode_quad(det, view, probe, decode_params, 2, &fit->early, NULL, &fit->lv, NULL)) return 1;
    /* The blob can include the white quiet zone when that margin is darker than the
     * background around it (a grey card on a bright wall): then this quad is the margin's
     * outline and the black square is its inner 8/10. Refit there and try once more. */
    int rescued = 0, rescue_sides = 0;
    /* A small candidate may be a fragment of a tag sampled every other pixel (or row): regrow it at full
     * resolution, fit that, and read again. */
    if (regrow_extent > 0) {
        double grown_width;
        int regrown = regrow_and_fit(det, view, params, rough, regrow_extent, finer, &grown_width);
        if (regrown < 0) return -2;
        if (regrown) {
            memcpy(probe, finer, sizeof probe);
            if (decode_quad(det, view, probe, decode_params, 2, &fit->early, NULL, &fit->lv, NULL)) {
                memcpy(fit->fine, finer, sizeof fit->fine);
                fit->edge_w = grown_width, fit->reach = 2.0, rescued = 1, fit->frame_sides = 0;
            }
        }
    }
    double Hq[9], perim = 0;
    square_to_quad(fit->fine, Hq);
    double m0 = 1.0 / (W + 2), m1 = (W + 1.0) / (W + 2);
    vec2 inner[4] = {project(Hq, m0, m0), project(Hq, m1, m0), project(Hq, m1, m1), project(Hq, m0, m1)};
    for (int k = 0; k < 4; k++) perim += vec_length(fit->fine[(k + 1) & 3].x - fit->fine[k].x, fit->fine[(k + 1) & 3].y - fit->fine[k].y);
    double r = 0.45 * perim / (4.0 * (W + 2)); /* 0.45 of a cell, if the quad spans W + 2 cells */
    /* Most failed decodes aren't tags at all. Refit only if the unfitted inner square already
     * shows a dark border inside a lighter margin, with loose limits (half the contrast, six
     * more light border cells): this skips 86% of retries and kept every rescue tested. */
    if (!rescued && r >= 1.5 &&
        border_plausible(&det->fam, view, inner, params->min_contrast / 2,
                         min_border(&det->fam, params->min_border_cells) - 6) &&
        fit_sides(view, inner, finer, r, r, params->min_contrast, 0, 8, &fit->edge_w, NULL, cells, &rescue_sides)) {
        memcpy(probe, finer, sizeof probe);
        if (decode_quad(det, view, probe, decode_params, 2, &fit->early, NULL, &fit->lv, NULL)) {
            memcpy(fit->fine, finer, sizeof fit->fine);
            fit->reach = r;
            rescued = 1;
            fit->frame_sides = rescue_sides;
        }
    }
    return rescued;
}

/* decode_initial_candidate, then refine_decoded_candidate if it decoded, updating *fit as they do. Returns -2 when out
 * of memory. */
static int decode_and_refine(napa_detector *det, const image_view *view, const decode_policy *policy, candidate_fit *fit,
                             const vec2 rough[4], double regrow_extent, candidate_outcome *out, pass_timer *timer) {
    out->view = *view;
    out->decoded = decode_initial_candidate(det, view, policy->params, policy->decode_params, fit, rough, regrow_extent);
    if (out->decoded < 0) return -2;
    PROF_MARK(PROF_DECODE1);
    if (!out->decoded) return 0;
    out->refined = refine_decoded_candidate(det, view, policy, fit, &out->d, &out->view, timer);
    return out->refined == REFINE_NO_MEMORY ? -2 : 0;
}

/* A whole candidate's or a piece's way from its rough corners (pick) to a first fit of its sides, as the candidate loop
 * takes it: the quiet-zone screen, the first fit, and for a whole that fails it, a restored corner or a regrown blob.
 * mm, when not NULL, holds the pre-pass's screen and first fit. Returns 1 with rough, *fit and *regrown set, 0 when the
 * candidate goes no further, -2 when out of memory. */
static int first_fit_stage(napa_detector *det, const image_view *view_in, const pass_config *config, const component *comp,
                           int pick[4][2], int s, int last, int piece, int undersized_whole, int extended_shape,
                           const cand_memo *mm, vec2 rough[4], candidate_fit *out, int *regrown_out) {
    const image_view view = *view_in;
    const napa_params *params = config->params;
    const int base = config->base, shaped = config->shaped, shadow = config->shadow;
    const double sx = config->sx, sy = config->sy, offx = config->offx, offy = config->offy, smax = config->smax;
    const double cells = det->fam.cells;
    const int bw = comp->maxx - comp->minx + 1, bh = comp->maxy - comp->miny + 1;
    const double big_px = s * (bw * sx > bh * sy ? bw * sx : bh * sy);
    candidate_fit fit = *out;
    rough_corners(pick, s, sx, sy, offx, offy, rough);
    /* Ordinary rough corners are dark samples. A completed small corner is a
     * geometric hypothesis from the same component bounds, not a new dark sample.
     * Retain the established mostly-outward fit window and final payload validation. */
    /* Sparse paired border samples reject noise before the expensive edge profiles.
     * Permit one weak side and leave terminal proposals unscreened: their ring can have
     * local clutter that a later level would normally resolve. Final decode is unchanged. */
    const int screen_mode = rough_screen_mode(s, last, shadow, piece, undersized_whole, params->min_border_cells);
    if (screen_mode != QUIET_ZONE_SKIP) {
        /* A tag's quiet zone is brighter than its border all the way round; the interior
         * fragments and clutter that dominate proposals average near zero. */
        const int pass = mm && mm->screen >= 0 ? mm->screen
            : quiet_zone_passes(&view, rough, 0.5 * s * smax, cells, params->min_contrast,
                                screen_mode == QUIET_ZONE_STRICT, shadow);
        if (!pass) return 0;
    }
    fit.reach = 0.5 * ((1.5 * s * smax + 1.0) + (0.5 * s * smax + 1.0));
    fit.edge_w = 0;
    int first_fit;
    if (mm && mm->fit >= 0) {
        first_fit = mm->fit;
        fit.frame_sides = mm->frame_sides;
        if (first_fit) {
            memcpy(fit.fine, mm->fine, sizeof fit.fine);
            fit.edge_w = mm->edge_w;
        }
    } else {
        first_fit = fit_initial_from_level(&view, rough, fit.fine, s, smax,
            params->min_contrast, &fit.edge_w, cells, &fit.frame_sides);
    }
    if (!first_fit && !piece && !shadow && !extended_shape) {
        int supported_pick[4][2];
        memcpy(supported_pick, pick, sizeof supported_pick);
        double supported_area, supported_perimeter;
        if (restore_rough_corner(comp, supported_pick, s * sx, s * sy,
                                 1.5 * s * smax + 1.0, 4.0 * fit.reach + 2.0) &&
            !rough_quad_too_full(&det->fam, comp, supported_pick, &supported_area, &supported_perimeter) &&
            supported_area >= 0.3 * bw * bh) {
            vec2 supported[4];
            rough_corners(supported_pick, s, sx, sy, offx, offy, supported);
            if (quiet_zone_passes(&view, supported, 0.5 * s * smax, cells, params->min_contrast, 1, 0)) {
                fit.edge_w = 0, fit.frame_sides = 0;
                first_fit = fit_initial_from_level(&view, supported, fit.fine, s, smax,
                    params->min_contrast, &fit.edge_w, cells, &fit.frame_sides);
                if (first_fit) memcpy(rough, supported, sizeof supported);
            }
        }
    }
    /* The same for a small candidate whose first fit fails: a fragment's quad may not fit at all. */
    int regrown = 0;
    if (!first_fit && !shadow && !piece && (base >= 2 || shaped) && s == base && big_px + s <= NAPA_REGROW_MAX_PX) {
        regrown = regrow_and_fit(det, &view, params, rough, big_px + s, fit.fine, &fit.edge_w);
        if (regrown < 0) return -2;
        if (regrown) first_fit = 1, fit.reach = 2.0, fit.frame_sides = 0;
    }
    *out = fit, *regrown_out = regrown;
    return first_fit;
}

/* The candidate loop's work on a deferred fine whole at the terminal level, up to its refinement, worked out ahead on
 * this thread's detector wd with the same steps the loop takes (see process_level_candidates). The loop's checks
 * against other candidates still decide, in order, whether it gets that far. */
static void decode_whole_ahead(const memo_job *j, napa_detector *wd, int index) {
    const fine_whole *whole = &j->det->fine_wholes[index];
    decode_shape_ahead(j, wd, &whole->shape, whole->stride, whole->terminal, &j->det->dmemo[index]);
}

static void decode_shape_ahead(const memo_job *j, napa_detector *wd, const component *comp, int s, int last,
                               delayed_memo *dm) {
    const pass_config *config = j->config;
    const int base = config->base, shadow = config->shadow;
    const int bw = comp->maxx - comp->minx + 1, bh = comp->maxy - comp->miny + 1;
    const int undersized_whole = base > 1 && (bw < config->lo || bh < config->lo);
    dm->stage = -1, dm->ahead = 0;
    int pick[4][2];
    rough_pick(comp, undersized_whole, pick);
    double best_area, perim_q;
    if (rough_quad_too_full(&wd->fam, comp, pick, &best_area, &perim_q)) return;
    const int extended_shape = whole_is_extended(base, s, last, undersized_whole, comp, bw, bh, best_area);
    if (extended_shape && extended_shape_rejected(comp, best_area, perim_q)) return;
    napa_params strict_border;
    const napa_params *decode_params = candidate_decode_params(wd, config->params, extended_shape, undersized_whole,
                                                               &strict_border);
    candidate_fit fit;
    fit.frame_sides = 0;
    int regrown = 0;
    const int stage = first_fit_stage(wd, j->view, config, comp, pick, s, last, 0, undersized_whole, extended_shape, NULL,
                                      dm->rough, &fit, &regrown);
    if (stage < 0) return;
    dm->stage = (signed char)stage, dm->regrown = (signed char)regrown, dm->fit = fit;
    if (!stage) return;
    const double big_px = s * (bw * config->sx > bh * config->sy ? bw * config->sx : bh * config->sy);
    const decode_policy policy = {config->params, decode_params, shadow, 0, undersized_whole};
    pass_timer timing = {0};
    const double regrow_extent = regrow_extent_for(regrown, shadow, 0, base, config->shaped, s, big_px);
    dm->ahead = decode_and_refine(wd, j->view, &policy, &fit, dm->rough, regrow_extent, &dm->outcome, &timing) == 0;
    dm->fit_after = fit;
}

static int process_level_candidates(napa_detector *det, const pass_config *config, pass_outputs *outputs,
                                    const candidate_level *level, pass_timer *timer, next_level_job *next,
                                    int memo_ready) {
    image_view view = config->view;
    const napa_params *params = config->params;
    int base = config->base;
    int lo = config->lo;
    int split = config->split;
    int watch = config->watch;
    int shadow = config->shadow;
    int shaped = config->shaped;
    double sx = config->sx;
    double sy = config->sy;
    double offx = config->offx;
    double offy = config->offy;
    double smax = config->smax;
    const double cells = det->fam.cells;
    const int s = level->s;
    const int last = level->last;
    const int redo = level->redo;
    const int ncand = level->ncand;
    const int npieces = level->npieces;
    const int fine = level->fine;
    const int covered_prefix = level->covered_prefix;
    const int level_stride = s, terminal_level = last;
    const int ordinary_count = ncand + npieces;
    /* With threads, work out every candidate's rough quad, screen and first fit in parallel first, and where it can,
     * its decode and refinement too; the loop below takes them in order as it would have computed them. (A camera's
     * lens memo isn't shared safely.) */
    int memo_level = 0;
    det->dmemo_n = last ? det->dmemo_ready : 0;
    /* A level with a next level to label alongside, or the terminal level with deferred wholes to work out ahead, runs
     * the pre-pass for any candidates at all. */
    const size_t pending = last ? outputs->fine_whole_count : 0;
    if (memo_ready) {
        memo_level = 1; /* the previous level's pool worked it out (sample_next_level) */
    } else if (level->extremes_deferred || level_pre_pass(det, &view, level) ||
        (det->pool != NULL && view.lens == NULL && !redo && ((next != NULL && ncand >= 1) || pending > 0))) {
        if (memo_reserve(det, ncand)) { /* always, when the extremes were deferred: assemble_level_candidates reserved it */
            const int ahead = can_decode_ahead(det);
            if (ahead && pending > 0 && pending <= (size_t)INT32_MAX / 2) {
                if (pending > det->dmemo_capacity) {
                    delayed_memo *p = (delayed_memo *)realloc(det->dmemo, pending * sizeof(delayed_memo));
                    if (p != NULL) det->dmemo = p, det->dmemo_capacity = pending;
                }
                if (pending <= det->dmemo_capacity) det->dmemo_n = (int)pending;
            }
            /* A level of an ordinary pass with a next level works out the wholes it sets aside in its own pre-pass,
             * when every earlier level's were. */
            int wholes_now = 0, wholes_all = 0; /* wholes_all: none are left for the terminal level's pre-pass */
            const int ready = det->dmemo_ready;
            if (ahead && next != NULL && outputs->fine_whole_count == (size_t)ready) {
                if ((size_t)ncand > det->whole_index_capacity) {
                    int32_t *p = (int32_t *)realloc(det->whole_index, (size_t)ncand * sizeof(int32_t));
                    if (p != NULL) det->whole_index = p, det->whole_index_capacity = (size_t)ncand;
                }
                int count = 0;
                if ((size_t)ncand <= det->whole_index_capacity) {
                    for (int ci = 0; ci < ncand; ci++) {
                        const component *c = &det->cands[ci];
                        const int bw = c->maxx - c->minx + 1, bh = c->maxy - c->miny + 1;
                        det->whole_index[ci] = (base > 1 && (bw < lo || bh < lo)) || dense_whole_needs_deferral(base, s, last, bw, bh)
                            ? ready + count++ : -1;
                    }
                    const size_t total = (size_t)ready + (size_t)count;
                    if (total > det->dmemo_capacity && total <= (size_t)INT32_MAX / 2) {
                        delayed_memo *p = (delayed_memo *)realloc(det->dmemo, total * sizeof(delayed_memo));
                        if (p != NULL) det->dmemo = p, det->dmemo_capacity = total;
                    }
                    wholes_now = count > 0 && total <= det->dmemo_capacity;
                    wholes_all = count == 0 || wholes_now;
                }
                if (wholes_now) det->dmemo_ready = ready + count;
            }
            if (ahead) {
                prepare_workers(det);
                /* Largest boxes first: the few candidates that go on to refine are among them, and a refinement
                 * started last would finish last. Which thread takes a candidate never changes its outcome. */
                int at[33] = {0};
                for (int ci = 0; ci < ncand; ci++) at[area_rank(&det->cands[ci]) + 1]++;
                for (int b = 0; b < 32; b++) at[b + 1] += at[b];
                for (int ci = 0; ci < ncand; ci++) det->memo_order[at[area_rank(&det->cands[ci])]++] = ci;
            }
            memo_job job = {det, &view, params, s, base, lo, last, shadow, shaped, sx, sy, offx, offy, smax, cells, ncand,
                            ahead, ahead ? det->memo_order : NULL, level->extremes_deferred ? det->runs : NULL,
                            det->level_bits, (level->lw + 63) / 64 + 1, next, config,
                            ahead ? ncand : (ncand + MEMO_CHUNK - 1) / MEMO_CHUNK, wholes_now,
                            det->cands, det->memo, det->cand_runs, det->cand_run_end, NULL, 0, 0};
            const int jobs = (next != NULL) + job.ordinary_jobs + (last ? det->dmemo_n - det->dmemo_ready : 0);
            /* The terminal level next, with nothing deferred to its pre-pass: assemble it and work out its pre-pass
             * in this pool too (sample_next_level), in the spare candidate set. */
            const int early = next != NULL && ahead && wholes_all && next->level->last;
            if (early) next->early = 1, next->pool_ctx = &job, job.coarse = &next->coarse, job.coarse_from = jobs;
            if (next != NULL) run_set_swap(det); /* the job holds this level's buffers; the next level takes the others */
            if (early) cand_set_swap(det);
            pool_run(det->pool, memo_candidates, &job, jobs);
            if (early) cand_set_swap(det);
            memo_level = 1;
        }
    }
    if (last && outputs->fine_whole_count > (size_t)(INT32_MAX - ordinary_count)) return -2;
    int candidate_count = ordinary_count + (last ? (int)outputs->fine_whole_count : 0);
    for (int ci = 0; ci < candidate_count; ci++) {
        const int delayed = ci >= ordinary_count;
        const size_t pending_index = delayed ? (size_t)(ci - ordinary_count) : 0;
        const int piece = !delayed && ci >= ncand;
        /* A saved outline retains its original coordinate grid and terminal fit policy. */
        const int s = delayed ? det->fine_wholes[pending_index].stride : level_stride;
        const int last = delayed ? det->fine_wholes[pending_index].terminal : terminal_level;
        const int coverage_start = delayed ? 0 : covered_prefix;
        /* Corner-stage pieces first: a tag kept whole there is found as without necks. */
        int k = piece ? (ci - ncand < npieces - fine ? fine + ci - ncand : ci - ncand - (npieces - fine)) : 0;
        /* Redone levels decode only the pieces deferred until neck splitting. */
        if (!delayed && redo && (!piece || (k >= fine && det->pieces[k].work.candidate.fate != CANDIDATE_WAITING_FOR_NECKS))) continue;
        component comp;
        int t_minx, t_maxx, t_miny, t_maxy;
        napa_params strict_border;
        const napa_params *decode_params = params;
        vec2 rough[4];
        candidate_fit fit;
        image_view lin_view = view;
        const image_view *rview = &view;
        fit.frame_sides = 0;
        component *self = delayed ? &det->fine_wholes[pending_index].shape
                          : piece ? &det->pieces[k] : &det->cands[ci];
        int *candidate_fate = &self->work.candidate.fate;
        comp = *self;
        const int undersized_whole = !piece && base > 1 &&
            (comp.maxx - comp.minx + 1 < lo || comp.maxy - comp.miny + 1 < lo);
        const int require_exact = undersized_whole;
        int extended_shape = 0;
        if (!delayed && !piece && (undersized_whole || dense_whole_needs_deferral(base, s, last,
                comp.maxx - comp.minx + 1, comp.maxy - comp.miny + 1))) {
            if (!remember_fine_whole(det, outputs->fine_whole_count, &comp, s, last)) return -2;
            outputs->fine_whole_count++;
            /* Terminal entries join the replay after all normal candidates, including
             * entries discovered in this same terminal candidate loop. */
            if (terminal_level) {
                if (outputs->fine_whole_count > (size_t)(INT32_MAX - ordinary_count)) return -2;
                candidate_count = ordinary_count + (int)outputs->fine_whole_count;
            }
            *candidate_fate = CANDIDATE_ROUGH_REJECTED; /* deferred wholes must not control split-piece eligibility */
            continue;
        }
        *candidate_fate = CANDIDATE_SKIPPED;
        if (piece) {
            /* A blob or larger piece that failed its rough quad or its refinement, or decoded, never
             * held another tag in its pieces; at stride one, nor did any candidate blob (only tiny
             * tags in large blobs are new there). */
            int next = det->piece_next[k];
            const component *whole = next >= ncand ? &det->pieces[next - ncand]
                                     : det->piece_cand[k] >= 0 ? &det->cands[det->piece_cand[k]] : NULL;
            int fate = whole ? whole->work.candidate.fate : CANDIDATE_UNTRIED;
            if (det->piece_cand[k] >= 0 && s == 1) continue;
            /* Until tags are seen touching, a piece that is nearly all of its blob, and its blob's only
             * piece, waits when the blob failed its first fit: it fails the same way. (A tag touching a
             * bar or a board is a small part of its blob.) Untried, it is decoded if the level is redone. */
            const component *blob = det->piece_cand[k] >= 0 ? &det->cands[det->piece_cand[k]] : NULL;
            if (!outputs->necks && blob && fate == CANDIDATE_FIT_FAILED && blob->work.candidate.piece_count == 1 &&
                5 * (comp.maxx - comp.minx + 1) * (comp.maxy - comp.miny + 1) >=
                    4 * (blob->maxx - blob->minx + 1) * (blob->maxy - blob->miny + 1)) {
                *candidate_fate = CANDIDATE_WAITING_FOR_NECKS;
                continue;
            }
            if (suppresses_pieces(fate)) {
                *candidate_fate = whole->work.candidate.fate; /* its own pieces share the fate */
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
        int covered_count = outputs->found < outputs->max_out ? outputs->found : outputs->max_out;
        if ((covered_count > coverage_start &&
             covered_footprint(outputs->out + coverage_start, det->out_centers + coverage_start, covered_count - coverage_start,
                     ccx, ccy, 0.3 * big_px + 2.0)) ||
            (piece && covered_footprint(det->piece_out, det->piece_centers, outputs->piece_found, ccx, ccy, 0.3 * big_px + 2.0))) {
            TRACE(5);
            *candidate_fate = CANDIDATE_SKIPPED;
            continue;
        }
        const cand_memo *mm = memo_level && !piece && !delayed ? &det->memo[ci] : NULL;
        const delayed_memo *dm = delayed && pending_index < (size_t)det->dmemo_n ? &det->dmemo[pending_index] : NULL;
        int pick[4][2];
        if (mm) memcpy(pick, mm->pick, sizeof pick);
        else rough_pick(&comp, undersized_whole, pick);
        double best_area, perim_q;
        /* Every tag36h11 code has at least 10 white payload cells, so at most 54/64 of a tag's
         * square is dark (0.85 measured, against the quad grown by the half sample its corner
         * samples sit inside). Solid strokes and filled shapes run well past that. */
        if (rough_quad_too_full(&det->fam, &comp, pick, &best_area, &perim_q)) {
            PROF_MARK(PROF_QUAD);
            TRACE(6);
            *candidate_fate = CANDIDATE_ROUGH_REJECTED;
            continue;
        }
        /* The axis-aligned box grows when a thin tag rotates. Sparse or slender whole
         * proposals use quad-relative fill and compactness; pieces use box-relative area. */
        int low_box_fill = best_area < 0.3 * bw * bh;
        extended_shape = !piece && whole_is_extended(base, s, last, undersized_whole, &comp, bw, bh, best_area);
        if ((piece && low_box_fill) || (extended_shape && extended_shape_rejected(&comp, best_area, perim_q))) {
            PROF_MARK(PROF_QUAD);
            TRACE(6);
            *candidate_fate = CANDIDATE_ROUGH_REJECTED;
            continue;
        }
        if (extended_shape) {
            /* Extended wholes retain rough rejection so their additional fitting
             * does not change split-piece eligibility. */
            *candidate_fate = CANDIDATE_ROUGH_REJECTED;
            if (covered_footprint(det->shape_out, det->shape_centers, outputs->shape_found,
                                  ccx, ccy, 0.3 * big_px + 2.0)) continue;
        }
        decode_params = candidate_decode_params(det, params, extended_shape, undersized_whole, &strict_border);
        PROF_MARK(PROF_QUAD);
        int regrown = 0, fitted;
        if (dm && dm->stage >= 0) { /* decode_whole_ahead's */
            fitted = dm->stage, regrown = dm->regrown, fit = dm->fit;
            memcpy(rough, dm->rough, sizeof rough);
        } else {
            fitted = first_fit_stage(det, &view, config, &comp, pick, s, last, piece, undersized_whole, extended_shape, mm,
                                     rough, &fit, &regrown);
            if (fitted < 0) return -2;
        }
        if (!fitted) {
            PROF_MARK(PROF_FIT1);
            TRACE(7);
            *candidate_fate = extended_shape ? CANDIDATE_ROUGH_REJECTED : CANDIDATE_FIT_FAILED;
            continue;
        }
        /* The first fit is already within a fraction of a cell: decode now, with two bits of
         * slack, and spend the refinement passes only on quads that are tags. A random quad
         * passes this only about 1.5% of the time. */
        PROF_MARK(PROF_FIT1);
        /* Levels overlap, so a blob that failed to decode at a finer level often comes back at this one.
         * Cached early failures use the caller's thresholds: a fitted quad within 0.3 px fails
         * again, including when the new proposal requires a stricter border. (The rough quads can match while the fits differ: the window grows with s.) */
        if (failed_before(det, fit.fine, s, 0.3)) {
            PROF_MARK(PROF_DECODE1);
            TRACE(8);
            *candidate_fate = extended_shape ? CANDIDATE_ROUGH_REJECTED : CANDIDATE_DECODE_FAILED;
            continue;
        }
        decode_policy policy = {params, decode_params, shadow, piece, require_exact};
        candidate_outcome computed;
        const candidate_outcome *outcome = &computed;
        if (mm && mm->ahead) {
            outcome = &mm->outcome, fit = mm->fit_after; /* decode_ahead's, from the same first fit (mm->fit == 1) */
        } else if (dm && dm->ahead) {
            outcome = &dm->outcome, fit = dm->fit_after; /* decode_whole_ahead's, from the same first fit */
        } else if (decode_and_refine(det, &view, &policy, &fit, rough,
                                     regrow_extent_for(regrown, shadow, piece, base, shaped, s, big_px), &computed,
                                     timer) < 0) {
            return -2;
        }
        if (!outcome->decoded) {
            TRACE(8);
            *candidate_fate = extended_shape ? CANDIDATE_ROUGH_REJECTED : CANDIDATE_DECODE_FAILED;
            /* Pieces only read the list. A stricter border failure also cannot prove
             * that an ordinary proposal at another level will fail the caller's check. */
            if (!piece && !extended_shape && decode_params == params) remember_failure(det, fit.fine, s);
            continue;
        }
        const refinement_result rejected = outcome->refined;
        if (rejected) {
            int fate = rejected == REFINE_FIT_FAILED ? CANDIDATE_REFINE_FIT_FAILED : CANDIDATE_REFINE_DECODE_FAILED;
            TRACE(fate);
            if (rejected != REFINE_CUT_FAILED) *candidate_fate = extended_shape ? CANDIDATE_ROUGH_REJECTED : fate;
            continue;
        }
        napa_detection d = outcome->d;
        lin_view = outcome->view;
        rview = &lin_view;
        const int one_bit = fit.early.hamming <= 1 && d.id == fit.early.id && det->fam.piece_hamming >= 0;
        if ((shadow && d.hamming != 0) || (require_exact && d.hamming > one_bit)) continue;
        if (piece) {
            /* Pieces are many and mostly not tags: correct fewer bits (in tag36h11, three admitted
             * chance matches from random-payload grids), and keep them apart until the end. */
            *candidate_fate = extended_shape ? CANDIDATE_ROUGH_REJECTED : CANDIDATE_DECODED;
            if (d.hamming > det->fam.piece_hamming || already_found_footprint(det->piece_out, det->piece_centers, outputs->piece_found, &d)) continue;
            if ((size_t)outputs->piece_found == det->piece_out_capacity) {
                size_t cap = det->piece_out_capacity ? 2 * det->piece_out_capacity : 64;
                napa_detection *p = (napa_detection *)realloc(det->piece_out, cap * sizeof(napa_detection));
                if (p == NULL) return -2;
                det->piece_out = p;
                det->piece_out_capacity = cap;
            }
            if (!outputs->necks && !outputs->touching) outputs->touching = corners_touch(rview, &d, cells);
            if (!save_center(&det->piece_centers, &det->piece_center_capacity, outputs->piece_found, &d)) return -2;
            det->piece_out[outputs->piece_found++] = d;
            continue;
        }
        if (already_found_footprint(outputs->out, det->out_centers, outputs->found < outputs->max_out ? outputs->found : outputs->max_out, &d)) {
            TRACE(11);
            *candidate_fate = extended_shape ? CANDIDATE_ROUGH_REJECTED : CANDIDATE_DUPLICATE;
            continue;
        }
        if (extended_shape) {
            if (already_found_footprint(det->shape_out, det->shape_centers, outputs->shape_found, &d)) continue;
            if ((size_t)outputs->shape_found == det->shape_out_capacity) {
                size_t cap = det->shape_out_capacity ? 2 * det->shape_out_capacity : 64;
                if (cap < det->shape_out_capacity || cap > SIZE_MAX / sizeof(napa_detection)) return -2;
                napa_detection *p = (napa_detection *)realloc(det->shape_out, cap * sizeof(napa_detection));
                if (p == NULL) return -2;
                det->shape_out = p;
                det->shape_out_capacity = cap;
            }
            if (!save_center(&det->shape_centers, &det->shape_center_capacity, outputs->shape_found, &d)) return -2;
            det->shape_out[outputs->shape_found++] = d;
            TRACE(12);
            continue; /* No ordinary coverage, output count or restart decision changes. */
        }
        TRACE(12);
        *candidate_fate = CANDIDATE_DECODED;
        if ((split || watch) && !outputs->necks && !outputs->touching) outputs->touching = corners_touch(rview, &d, cells);
        if (outputs->found < outputs->max_out) {
            if (!save_center(&det->out_centers, &det->out_center_capacity, outputs->found, &d)) return -2;
            outputs->out[outputs->found] = d;
        }
        outputs->found++;
    }
    if (last) det->dmemo_ready = 0;
    PROF_MARK(PROF_BLOBS);
    return 0;
}

static int save_level_fates(napa_detector *det, pass_outputs *outputs, const candidate_level *level) {
    /* Keep this level's fates, in case the frame is redone with necks. */
    size_t n = (size_t)level->ncand + (size_t)level->npieces;
    if (outputs->fate_off + n > det->fate_capacity) {
        size_t cap = det->fate_capacity ? 2 * det->fate_capacity : 4096;
        while (cap < outputs->fate_off + n) cap *= 2;
        uint8_t *f = (uint8_t *)realloc(det->fates, cap);
        if (f == NULL) return -2;
        det->fates = f;
        det->fate_capacity = cap;
    }
    for (int ci = 0; ci < level->ncand; ci++) det->fates[outputs->fate_off + ci] = (uint8_t)det->cands[ci].work.candidate.fate;
    for (int k = 0; k < level->npieces; k++) det->fates[outputs->fate_off + level->ncand + k] = (uint8_t)det->pieces[k].work.candidate.fate;
    outputs->fates_kept = outputs->fate_off + n;
    return 0;
}

static int append_deferred_outputs(napa_detector *det, pass_outputs *outputs) {
    /* Append extended shapes after ordinary wholes and before optional pieces,
     * keeping the unsplit detection prefix stable. */
    for (int k = 0; k < outputs->shape_found && outputs->found < outputs->max_out; k++) {
        if (!already_found_footprint(outputs->out, det->out_centers, outputs->found, &det->shape_out[k])) {
            if (!save_center(&det->out_centers, &det->out_center_capacity, outputs->found, &det->shape_out[k])) return -2;
            outputs->out[outputs->found++] = det->shape_out[k];
        }
    }
    /* Pieces come last, so the other detections are unchanged by split_touching. */
    for (int k = 0; k < outputs->piece_found && outputs->found < outputs->max_out; k++) {
        if (!already_found_footprint(outputs->out, det->out_centers, outputs->found, &det->piece_out[k])) {
            if (!save_center(&det->out_centers, &det->out_center_capacity, outputs->found, &det->piece_out[k])) return -2;
            outputs->out[outputs->found++] = det->piece_out[k];
        }
    }
    return outputs->found < outputs->max_out ? outputs->found : outputs->max_out;
}

static int detect_pass(napa_detector *det, const uint8_t *img, int width, int height, int stride, const napa_params *params,
                napa_detection *out, int max_out, int found, int shadow) {
    if (det == NULL || img == NULL || (max_out > 0 && out == NULL)) return -1;
    if (stride < width || max_out < 0) return -1;
    if (width > NAPA_MAX_SIDE || height > NAPA_MAX_SIDE) return -1;
    if (!params_valid(params)) return -1;
    det->touching_seen = 0;
    if (width < 16 || height < 16) return 0;
    const int split = params->split_touching == 1 && det->fam.piece_hamming >= 0;
    /* split_touching 2: an ordinary pass that notes whether a decoded tag touches dark shapes at its corners. */
    const int watch = params->split_touching == 2 && !shadow && det->fam.piece_hamming >= 0;
    const int W = det->fam.width;
    /* Reversed families print the square light in a dark frame: read inverted, they look like every other
     * family, a dark square in a light quiet ring. The raked levels are inverted as they're made, and
     * full-resolution reads sample by sample (image_view). */
    const image_view view = {img, width, height, stride, det->fam.f->reversed, det->lens.model ? &det->lens : NULL};
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
    pass_outputs outputs = {0};
    outputs.out = out; outputs.max_out = max_out; outputs.found = found;
    /* split_touching splits at corners only until it finds a tag that touches dark shapes at its corners:
     * one-sample necks (anti-aliased corners) cost far more and only pay where tags do touch. Then the frame
     * starts over splitting necks too, and the levels already done (to redone) decode only their new pieces. */
    outputs.necks = split && det->split_from_necks;
    outputs.touching = outputs.necks;
    det->failed_n = 0;
    det->regrow_left = det->regrow_small_tags ? NAPA_REGROW_BUDGET : 0;
    /* Size band in level samples. Adjacent levels overlap (7-24 here is 14-48 one level finer),
     * because a blob's measured extent varies by a sample or two with phase, blur and rotation. */
    const int lo = (7 * W + 4) / 8, hi = NAPA_HI * W / 8;

    const pass_config config = {view, params, pimg, pw, ph, pstride, base, lo, hi,
                                split, watch, shadow, shaped, sx, sy, offx, offy, smax};
    if (det->runs_swapped) run_set_swap(det); /* the finest level always gets the same set, so the spare stays small */
    if (det->cands_swapped) cand_set_swap(det);
    det->dmemo_ready = 0;
    /* With threads, an ordinary pass labels each next level alongside this level's pre-pass (next_level_job). */
#ifdef NAPA_PROFILE
    const int overlap = 0; /* the stage timers are one set for the process */
#else
    const int overlap = det->pool != NULL && !split && !shadow;
#endif
start:;
    outputs.fate_off = 0;
    level_sampling sampling = {NULL, 0, base, params->min_contrast, 0, 0};
    pass_timer timing, *timer = &timing;
    /* After a stride-two finest level, whose half-tile caps (level_half) separate what a wider window merges, the ordinary
     * pass skips stride four: strides two and eight (capped the same way) find nearly all of its tags, for a sixth less
     * time. Splitting passes (boards) keep it, because a board's large tags often come apart only there. Every other pass
     * doubles the stride level by level. */
    const int skip_four = !shadow && !split && base == 2;
    candidate_level ahead; /* the next level, once sampled alongside this one */
    int sampled_ahead = 0, assembled_ahead = 0, memo_ahead = 0; /* and assembled, and its pre-pass worked out */
    for (int s = base;; s *= skip_four && s == base ? 4 : 2) {
        candidate_level level;
        if (sampled_ahead) {
            level = ahead, sampled_ahead = 0;
        } else {
            level.s = s;
            level.lw = (pw + s - 1) / s;
            level.lh = (ph + s - 1) / s;
            level.last = (level.lw < NAPA_LAST * W / 8) || (level.lh < NAPA_LAST * W / 8);
            level.redo = s <= outputs.redone;
            if (sample_threshold_level(det, &config, &outputs, &sampling, &level, timer) != 0) return -2;
        }
        int memo_ready = 0;
        if (assembled_ahead) {
            cand_set_swap(det); /* the terminal level's candidates, in the spare set */
            keep_uncovered(det, &config, &outputs, &level, memo_ahead);
            memo_ready = memo_ahead, assembled_ahead = memo_ahead = 0;
        } else if (assemble_level_candidates(det, &config, &outputs, &level, timer) != 0) return -2;
        next_level_job next;
        memset(&next, 0, sizeof next);
        next.det = det, next.config = &config, next.outputs = &outputs, next.sampling = &sampling, next.level = &ahead;
        if (overlap && !level.last) {
            const int ns = s * (skip_four && s == base ? 4 : 2);
            ahead.s = ns;
            ahead.lw = (pw + ns - 1) / ns;
            ahead.lh = (ph + ns - 1) / ns;
            ahead.last = (ahead.lw < NAPA_LAST * W / 8) || (ahead.lh < NAPA_LAST * W / 8);
            ahead.redo = ns <= outputs.redone;
        }
        if (process_level_candidates(det, &config, &outputs, &level, timer,
                                     overlap && !level.last ? &next : NULL, memo_ready) != 0) return -2;
        if (next.done) {
            if (next.status != 0) return -2;
            sampled_ahead = 1, assembled_ahead = next.assembled, memo_ahead = next.memo_ready;
        }
        if (split && !outputs.necks && save_level_fates(det, &outputs, &level) != 0) return -2;
        outputs.fate_off += (size_t)level.ncand + (size_t)(level.npieces - level.fine);
        if (split && !outputs.necks && outputs.touching) {
            outputs.necks = 1;
            outputs.redone = s;
            goto start;
        }
        if (level.last) break;
    }
    det->touching_seen = watch && outputs.touching;
    return append_deferred_outputs(det, &outputs);
}

/* split_touching 2: frames where an ordinary pass finds tags touching dark shapes at their corners (AprilGrid boards,
 * tag mosaics) are detected again with split_touching, splitting necks from the start; other frames pay only the
 * corner checks on the tags they decode. */
static int detect_ordinary(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                           const napa_params *params, napa_detection *out, int max_out) {
    int n = detect_pass(det, img, width, height, stride, params, out, max_out, 0, 0);
    if (n >= 0 && params->split_touching == 2 && det->touching_seen) {
        napa_params split = *params;
        split.split_touching = 1;
        det->split_from_necks = 1;
        n = detect_pass(det, img, width, height, stride, &split, out, max_out, 0, 0);
        det->split_from_necks = 0;
    }
    return n;
}

int napa_detect(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                const napa_params *params, napa_detection *out, int max_out) {
    return detect_ordinary(det, img, width, height, stride, params, out, max_out);
}

/* Keep ordinary results first and untouched; recovery shares their coverage/duplicate checks.
 * Each pass resets its failed-quad cache: a failure under one threshold is not evidence for the other. */
int napa_detect_shadows_configured(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                        const napa_params *params, napa_detection *out, int max_out, int shadow_min_tag_px) {
    if (params == NULL || (shadow_min_tag_px != 0 &&
        (shadow_min_tag_px < params->min_tag_px || shadow_min_tag_px < 8 || shadow_min_tag_px > 30000))) return -1;
    if (max_out > 0 && out == NULL) return -1;
    if (det == NULL) return -1;
    /* Recovery reuses the ordinary pass's tile extrema at each step whose samples match (sample_threshold_level). */
    for (int i = 0; i < 32; i++) det->saved[i].valid = 0;
    det->extrema_ready = 0;
    det->extrema_wanted = det->fam.piece_hamming >= 0;
    int n = detect_ordinary(det, img, width, height, stride, params, out, max_out);
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
    soft_sample(view, start, now);
    float best = soft_correlation(now, t);
    for (int iter = 0; iter < 4; iter++) {
        vec2 q[4];
        for (int k = 0; k < 4; k++) q[k].x = x[2 * k], q[k].y = x[2 * k + 1];
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
            const double dx = x[2 * k] - origin[2 * k], dy = x[2 * k + 1] - origin[2 * k + 1], len = vec_length(dx, dy);
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
    float lead;
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
        for (int k = 0; k < 4; k++) side += 0.25 * vec_length(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
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
        a->code = code, a->bin = b, a->lead = lead, a->side = side, a->black = e->black, a->white = e->white;
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
        /* columns isn't 0: at least NAPA_BLOCKED_MIN_WRONG bits are wrong. */
        const int x0 = napa_ctz64(columns), x1 = 63 - napa_clz64(columns);
        if ((x1 - x0 + 1) * (y1 - y0 + 1) > NAPA_BLOCKED_MAX_AREA) continue;
        float lo = HUGE_VALF, hi = -HUGE_VALF;
        for (int r = y0; r <= y1; r++)
            for (int c = x0; c <= x1; c++) {
                const float v = e->grid[(r + 1) * 8 + c + 1];
                lo = v < lo ? v : lo, hi = v > hi ? v : hi;
            }
        if (hi - lo > allowed) continue;
        if (found >= 0) return -1;
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
        for (int k = 0; k < 4; k++) side += 0.25 * vec_length(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
        if (convex_area(q) <= 0 || overlaps_found(out, n, q)) continue;
        int x0, x1, y0, y1;
        const int code = blocked_code(det, e, &x0, &x1, &y0, &y1);
        if (code < 0) continue;
        const double cell = side / 8, reach = 0.45 * cell >= 1.5 ? 0.45 * cell : 1.5;
        if (!fit_sides(view, q, fine, reach, reach, params->min_contrast, 0, 8, &edge, NULL, 8, NULL)) continue;
        if (!fit_sides(view, fine, refined, reach, reach, params->min_contrast, 0, NAPA_FINAL_POINTS, &edge, NULL, 8, NULL)) continue;
        if (signed_area(refined) < 0) { vec2 t = refined[1]; refined[1] = refined[3]; refined[3] = t; }
        if (convex_area(refined) <= 0) continue;
        /* Read the fitted quad the way the decoder would, and ask the same question of what it sees. */
        soft_candidate probe;
        napa_detection unused;
        memset(&probe, 0, sizeof probe);
        det->grid_probe = &probe;
        decode_quad(det, view, refined, params, 0, &unused, NULL, NULL, NULL);
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

/* ---------------------------------------------------------------- input conversion */

/* OpenCV's BGR-to-grey weights, in 15-bit fixed point: cv2.cvtColor gives exactly these values. */
#define NAPA_GRAY_B 3735u
#define NAPA_GRAY_G 19235u
#define NAPA_GRAY_R 9798u

typedef struct {
    const uint8_t *src;
    uint8_t *dst;
    int width, height, stride, dst_stride, pixel_bytes, colour, chunks;
} gray_job;

static void gray_rows(void *arg, int k, int worker) {
    (void)worker;
    const gray_job *job = (const gray_job *)arg;
    const int y0 = (int)((long long)job->height * k / job->chunks), y1 = (int)((long long)job->height * (k + 1) / job->chunks);
    const int w = job->width, step = job->pixel_bytes;
#if defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
    /* Vector loads read every byte of their 16 pixels; a view's last pixel can end a byte or more early (its last
     * channel only), so the last pixel of each row is always left to the scalar loop. */
    const int vector_end = w - 1;
#endif
    for (int y = y0; y < y1; y++) {
        const uint8_t *s = job->src + (size_t)y * job->stride;
        uint8_t *d = job->dst + (size_t)y * job->dst_stride;
        int x = 0;
        if (!job->colour) { /* one byte of each pixel: a strided view, such as YUYV's luma */
#if defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
            if (step == 2)
                for (; x + 16 <= vector_end; x += 16) vst1q_u8(d + x, vld2q_u8(s + 2 * x).val[0]);
            else if (step == 3)
                for (; x + 16 <= vector_end; x += 16) vst1q_u8(d + x, vld3q_u8(s + 3 * x).val[0]);
            else if (step == 4)
                for (; x + 16 <= vector_end; x += 16) vst1q_u8(d + x, vld4q_u8(s + 4 * x).val[0]);
#endif
            for (; x < w; x++) d[x] = s[(size_t)x * step];
            continue;
        }
#if defined(__ARM_NEON) && !defined(NAPA_NO_SIMD)
        for (; x + 16 <= vector_end; x += 16) {
            uint8x16_t b, g, r;
            if (step == 3) {
                uint8x16x3_t p = vld3q_u8(s + 3 * x);
                b = p.val[0], g = p.val[1], r = p.val[2];
            } else {
                uint8x16x4_t p = vld4q_u8(s + 4 * x);
                b = p.val[0], g = p.val[1], r = p.val[2];
            }
            uint16x8_t parts[3][2] = {{vmovl_u8(vget_low_u8(b)), vmovl_u8(vget_high_u8(b))},
                                      {vmovl_u8(vget_low_u8(g)), vmovl_u8(vget_high_u8(g))},
                                      {vmovl_u8(vget_low_u8(r)), vmovl_u8(vget_high_u8(r))}};
            uint16x8_t half[2];
            for (int h = 0; h < 2; h++) {
                uint32x4_t lo = vmull_n_u16(vget_low_u16(parts[0][h]), NAPA_GRAY_B);
                uint32x4_t hi = vmull_n_u16(vget_high_u16(parts[0][h]), NAPA_GRAY_B);
                lo = vmlal_n_u16(lo, vget_low_u16(parts[1][h]), NAPA_GRAY_G);
                hi = vmlal_n_u16(hi, vget_high_u16(parts[1][h]), NAPA_GRAY_G);
                lo = vmlal_n_u16(lo, vget_low_u16(parts[2][h]), NAPA_GRAY_R);
                hi = vmlal_n_u16(hi, vget_high_u16(parts[2][h]), NAPA_GRAY_R);
                half[h] = vcombine_u16(vrshrn_n_u32(lo, 15), vrshrn_n_u32(hi, 15)); /* + 2^14, >> 15 */
            }
            vst1q_u8(d + x, vcombine_u8(vmovn_u16(half[0]), vmovn_u16(half[1])));
        }
#endif
        for (; x < w; x++) {
            const uint8_t *p = s + (size_t)x * step;
            d[x] = (uint8_t)((p[0] * NAPA_GRAY_B + p[1] * NAPA_GRAY_G + p[2] * NAPA_GRAY_R + 16384u) >> 15);
        }
    }
}

int napa_gray_from_pixels(napa_detector *det, const uint8_t *src, int width, int height, int stride, int pixel_bytes,
                          int colour, uint8_t *dst, int dst_stride) {
    if (src == NULL || dst == NULL || width < 1 || height < 1 || width > NAPA_MAX_SIDE || height > NAPA_MAX_SIDE) return -1;
    if (pixel_bytes < 1 || pixel_bytes > 4 || (colour && pixel_bytes < 3) || dst_stride < width) return -1;
    if (stride < (width - 1) * pixel_bytes + (colour ? 3 : 1)) return -1;
    const size_t pixels = (size_t)width * height;
    /* Copying one byte per pixel is all memory traffic: threads only help the arithmetic of colour. */
    const int chunks = colour && det != NULL && pool_for(det, pixels) ? det->threads * 2 : 1;
    gray_job job = {src, dst, width, height, stride, dst_stride, pixel_bytes, colour != 0, chunks};
    pool_run(chunks > 1 ? det->pool : NULL, gray_rows, &job, chunks);
    return 0;
}

int napa_set_threads(napa_detector *det, int threads) {
    if (det == NULL || threads < 1 || threads > NAPA_MAX_THREADS) return -1;
    if (threads == det->threads) return 0;
    napa_pool *pool = NULL;
    if (threads > 1 && (pool = pool_create(threads - 1)) == NULL) return -2;
    pool_destroy(det->pool);
    det->pool = pool, det->threads = threads;
    free_workers(det);
    if (threads > 1) make_workers(det, threads - 1);
    return 0;
}

int napa_set_tag_shape(napa_detector *det, int shape) {
    if (det == NULL || (shape != 0 && shape != NAPA_TAG_SHAPE_TALL && shape != NAPA_TAG_SHAPE_WIDE)) return -1;
    det->tag_shape = shape;
    return 0;
}

int napa_set_regrow_small_tags(napa_detector *det, int on) {
    if (det == NULL || (on != 0 && on != 1)) return -1;
    det->regrow_small_tags = on;
    return 0;
}

int napa_set_camera_gamma(napa_detector *det, double gamma) {
    if (det == NULL || !isfinite(gamma) || gamma < 1 || gamma > 4 || strcmp(det->fam.f->name, "tag36h11") != 0) return -1;
    det->has_camera_gamma = gamma > 1;
    for (int v = 0; v < 256; v++) det->linearize[v] = (uint8_t)(255.0 * pow(v / 255.0, gamma) + 0.5);
    return 0;
}

int napa_set_lens(napa_detector *det, int model, const double *intrinsics, const double *coefficients, int count) {
    if (det == NULL) return -1;
    if (model == NAPA_LENS_NONE) {
        memset(&det->lens, 0, sizeof det->lens);
        return 0;
    }
    if ((model != NAPA_LENS_BROWN_CONRADY && model != NAPA_LENS_FISHEYE) || intrinsics == NULL || coefficients == NULL) return -1;
    if (model == NAPA_LENS_FISHEYE ? count != 4 : count != 4 && count != 5 && count != 8 && count != 12 && count != 14) return -1;
    for (int i = 0; i < 5; i++)
        if (!isfinite(intrinsics[i])) return -1;
    for (int i = 0; i < count; i++)
        /* No real lens comes near 1e6; the limit keeps the distortion polynomials far from overflow. */
        if (!isfinite(coefficients[i]) || fabs(coefficients[i]) > 1e6) return -1;
    if (!(intrinsics[0] > 0) || !(intrinsics[1] > 0) || (count == 14 && (coefficients[12] != 0 || coefficients[13] != 0))) return -1;
    napa_lens L;
    memset(&L, 0, sizeof L);
    L.model = model, L.fx = intrinsics[0], L.fy = intrinsics[1], L.cx = intrinsics[2], L.cy = intrinsics[3];
    L.skew = model == NAPA_LENS_FISHEYE ? intrinsics[4] : 0; /* OpenCV's standard model ignores it */
    for (int i = 0; i < count && i < 12; i++) L.k[i] = coefficients[i];
    L.rational = L.k[5] != 0 || L.k[6] != 0 || L.k[7] != 0;
    L.prism = L.k[8] != 0 || L.k[9] != 0 || L.k[10] != 0 || L.k[11] != 0;
    memset(&det->lens_cache, 0, sizeof det->lens_cache);
    L.memo = &det->lens_cache;
    det->lens = L;
    return 0;
}

int napa_border_bias_cells(napa_detector *det, const uint8_t *img, int width, int height, int stride, const napa_params *params,
                           const double corners[8], double *bias_cells, double *edge_cells) {
    if (det == NULL || img == NULL || corners == NULL || bias_cells == NULL || !params_valid(params) ||
        width < 2 || height < 2 || width > NAPA_MAX_SIDE || height > NAPA_MAX_SIDE || stride < width) return -1;
    for (int k = 0; k < 8; k++)
        if (!isfinite(corners[k])) return -1;
    const image_view view = {img, width, height, stride, det->fam.f->reversed, NULL}; /* measured in pixels, as given */
    vec2 q[4];
    double side = 0, delta = NAN, disagreement = 0;
    for (int k = 0; k < 4; k++) q[k].x = corners[2 * k], q[k].y = corners[2 * k + 1];
    if (signed_area(q) < 0) { vec2 t = q[1]; q[1] = q[3]; q[3] = t; }
    for (int k = 0; k < 4; k++) side += 0.25 * vec_length(q[(k + 1) & 3].x - q[k].x, q[(k + 1) & 3].y - q[k].y);
    edge_levels levels;
    napa_detection unused;
    levels.black = 255, levels.white = 0;
    decode_quad(det, &view, q, params, 0, &unused, NULL, &levels, NULL);
    if (!(levels.black < levels.white)) return 1;
    border_bias(&det->fam, &view, q, levels.black, levels.white, &delta, &disagreement);
    if (!isfinite(delta)) return 1;
    const double cell = side / det->fam.width;
    *bias_cells = delta / cell;
    if (edge_cells != NULL) {
        /* The blur of the tag's outer edges: profiles reach most of a cell each way, so wide blur is not cut short. */
        vec2 fitted[4];
        double width = 0, reach = 0.75 * cell + 1.0;
        *edge_cells = fit_sides(&view, q, fitted, reach, reach, params->min_contrast, 0, 8, &width, NULL, det->fam.width, NULL) && width > 0
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
    const image_view view = {img, width, height, stride, 0, det->lens.model ? &det->lens : NULL};
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
