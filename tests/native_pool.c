/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* The worker pool behind napa_set_threads: every index runs exactly once per job, across many back-to-back
 * jobs, thread counts, resizes and teardown. Banded labelling gives the serial pass's runs and components at
 * every thread count, row pairs included. */
#ifdef NDEBUG
#error "The pool checks require assertions"
#endif
#include "../src/napa_vision/_native/napa_core.c"
#include <assert.h>
#include <stdio.h>

typedef struct {
    int hits[512];
    long long sum;
    napa_mutex_t lock;
} tally;

static void count(void *ctx, int index) {
    tally *t = (tally *)ctx;
    t->hits[index]++; /* each index is owned by exactly one runner: no race if the pool is right */
    volatile long long spin = 0;
    for (int k = 0; k < (index * 7919) % 2000; k++) spin += k; /* uneven work */
    napa_mutex_lock(&t->lock);
    t->sum += index;
    napa_mutex_unlock(&t->lock);
}

static uint32_t seed = 1291;
static uint32_t rnd(void) { return seed = seed * 1664525u + 1013904223u; }

/* Fill the level: random bytes, sparse or dense dark specks (many diagonal contacts, so many row pairs kept
 * as single rows), or blocks of specks on flat ground. */
static void fill_level(uint8_t *level, int w, int h, int mode) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t v;
            if (mode == 0) v = (uint8_t)(rnd() >> 24);
            else if (mode == 1) v = rnd() % 5 == 0 ? 0 : 255;
            else if (mode == 2) v = rnd() % 2 ? 20 : 230;
            else v = ((x / 23 + y / 17) & 1) ? (rnd() % 3 ? 200 : 10) : 128;
            level[(size_t)y * w + x] = v;
        }
}

/* Banded threshold_level at 2-8 threads against the serial pass (threshold_rows): ordinary, shadow, and ordinary with
 * half tiles (level_half). */
static void check_banded_labels(void) {
    static const int sizes[][2] = {{320, 320}, {385, 257}, {640, 160}, {129, 777}, {1000, 101}, {64, 1601}};
    long cases = 0, boundaries = 0, single_above = 0, single_below = 0, joined = 0;
    for (int s = 0; s < (int)(sizeof sizes / sizeof sizes[0]); s++) {
        const int w = sizes[s][0], h = sizes[s][1], tw = (w + 3) / 4, th = (h + 3) / 4;
        const size_t pixels = (size_t)w * h;
        assert(pixels >= NAPA_THREAD_MIN_SAMPLES && th / 4 >= 2);
        for (int mode = 0; mode < 4; mode++)
            for (int regional = 0; regional < 2; regional++)
                for (int kind = 0; kind < 3; kind++)
                    for (int factor = 1; factor <= 4; factor++) {
                        const int shadow = kind == 1;
                        napa_detector *det = napa_create();
                        assert(det != NULL && ensure_buffers(det, pixels, (size_t)tw * th, (size_t)w, pixels / 2 + (size_t)h + 1) == 0);
                        fill_level(det->level, w, h, mode);
                        const int cols = regional ? (w * factor + 63) / 64 : 1;
                        const int rows = regional ? (h * factor + 63) / 64 : 1;
                        det->floors = (uint8_t *)malloc((size_t)cols * rows);
                        assert(det->floors != NULL);
                        for (int i = 0; i < cols * rows; i++) det->floors[i] = (uint8_t)(regional ? rnd() % 60 : 8);
                        det->floor_cols = cols, det->floor_rows = rows, det->floor_uniform = !regional;
                        det->level_half = kind == 2;
                        const int n = threshold_rows(det, w, h, factor, 0, shadow, 0, th, 0);
                        assert(n > 0);
                        napa_run *runs = (napa_run *)malloc((size_t)n * sizeof(napa_run));
                        int32_t *roots = (int32_t *)malloc((size_t)n * sizeof(int32_t));
                        assert(runs != NULL && roots != NULL);
                        memcpy(runs, det->runs, (size_t)n * sizeof(napa_run));
                        for (int i = 0; i < n; i++) roots[i] = find_root(det->parent, i);
                        for (int threads = 2; threads <= 8; threads++) {
                            assert(napa_set_threads(det, threads) == 0);
                            assert(threshold_level(det, w, h, factor, 0, shadow) == n);
                            for (int i = 0; i < n; i++) {
                                const napa_run *a = &det->runs[i], *b = &runs[i];
                                if (a->x0 != b->x0 || a->x1 != b->x1 || a->y != b->y || (!shadow && a->rows != b->rows)) {
                                    fprintf(stderr, "%dx%d mode %d regional %d kind %d threads %d: run %d differs\n", w, h,
                                            mode, regional, kind, threads, i);
                                    exit(1);
                                }
                            }
                            for (int i = 0; i < n; i++) assert(find_root(det->parent, i) == roots[i]);
                            int bands = threads * 3 < th / 4 ? threads * 3 : th / 4;
                            for (int k = 1; k < bands && !shadow; k++) {
                                const int y = det->bands[k].ty0 * TILE;
                                int above = 0, below = 0, across = 0;
                                for (int i = 0; i < n; i++) {
                                    above |= runs[i].y == y - 1;
                                    below |= runs[i].y == y + 1;
                                    across |= runs[i].y == y && roots[i] < i && runs[roots[i]].y < y;
                                }
                                boundaries++, single_above += above, single_below += below, joined += across;
                            }
                            cases++;
                        }
                        free(runs), free(roots);
                        napa_destroy(det);
                }
    }
    assert(single_above && single_below && joined);
    printf("%ld banded labelling cases match the serial pass (%ld row-pair band boundaries: %ld joined, %ld kept "
           "single rows above, %ld below)\n", cases, boundaries, joined, single_above, single_below);
}

/* Two threads driving one pool at once (a detector misused from two threads): neither may hang, and each
 * job still runs every one of its indices exactly once (the second caller runs alone). */
typedef struct {
    napa_pool *pool;
    tally t;
    int failures;
} contender;

#if defined(_WIN32)
static DWORD WINAPI contend(LPVOID arg) {
#else
static void *contend(void *arg) {
#endif
    contender *c = (contender *)arg;
    for (int round = 0; round < 400; round++) {
        int n = 2 + round % 97;
        memset(c->t.hits, 0, sizeof c->t.hits);
        c->t.sum = 0;
        pool_run(c->pool, count, &c->t, n);
        for (int i = 0; i < n; i++) c->failures += c->t.hits[i] != 1;
        c->failures += c->t.sum != (long long)n * (n - 1) / 2;
    }
    return 0;
}

static int contended_runs(void) {
    napa_pool *pool = pool_create(3);
    assert(pool != NULL);
    static contender c[2];
    for (int k = 0; k < 2; k++) {
        c[k].pool = pool, c[k].failures = 0;
        napa_mutex_init(&c[k].t.lock);
    }
#if defined(_WIN32)
    HANDLE other = CreateThread(NULL, 0, contend, &c[1], 0, NULL);
    contend(&c[0]);
    WaitForSingleObject(other, INFINITE);
    CloseHandle(other);
#else
    pthread_t other;
    assert(pthread_create(&other, NULL, contend, &c[1]) == 0);
    contend(&c[0]);
    pthread_join(other, NULL);
#endif
    pool_destroy(pool);
    for (int k = 0; k < 2; k++) napa_mutex_destroy(&c[k].t.lock);
    return c[0].failures + c[1].failures;
}

int main(void) {
    static tally t;
    napa_mutex_init(&t.lock);
    long jobs = 0;
    for (int workers = 0; workers <= 7; workers++) {
        napa_pool *pool = workers ? pool_create(workers) : NULL;
        assert(workers == 0 || (pool != NULL && pool->workers == workers));
        for (int round = 0; round < 300; round++) {
            int n = (round * 37) % 300;
            memset(t.hits, 0, sizeof t.hits);
            t.sum = 0;
            pool_run(pool, count, &t, n);
            for (int i = 0; i < 512; i++) assert(t.hits[i] == (i < n));
            assert(t.sum == (long long)n * (n - 1) / 2);
            jobs++;
        }
        pool_destroy(pool);
    }
    napa_detector *det = napa_create();
    assert(det != NULL && det->threads == 1 && det->pool == NULL);
    assert(napa_set_threads(det, 0) == -1 && napa_set_threads(det, NAPA_MAX_THREADS + 1) == -1);
    assert(napa_set_threads(NULL, 2) == -1);
    for (int k = 1; k <= NAPA_MAX_THREADS; k++) {
        assert(napa_set_threads(det, k) == 0);
        assert(det->threads == k && (k == 1 ? det->pool == NULL : det->pool->workers == k - 1));
        memset(t.hits, 0, sizeof t.hits);
        pool_run(det->pool, count, &t, 64);
        for (int i = 0; i < 64; i++) assert(t.hits[i] == 1);
    }
    assert(napa_set_threads(det, 3) == 0 && napa_set_threads(det, 3) == 0 && det->pool->workers == 2);
    napa_destroy(det); /* joins the workers */
    napa_mutex_destroy(&t.lock);
    assert(contended_runs() == 0);
    printf("%ld pool jobs at 1-8 threads, %d thread settings, 800 contended jobs passed\n", jobs, NAPA_MAX_THREADS);
    check_banded_labels();
    return 0;
}
