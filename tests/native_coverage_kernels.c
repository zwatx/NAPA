/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Independent numeric centre-cache oracles; footprint policy is tested separately. */
#include <stdlib.h>
static int fail_realloc;
static size_t failed_size;
static void *coverage_realloc(void *p, size_t n) {
    if (fail_realloc) {
        failed_size = n;
        return NULL;
    }
    return realloc(p, n);
}
#define realloc coverage_realloc
#include "../src/napa_vision/_native/napa_core.c"
#undef realloc
#include <float.h>
#include <stdio.h>

/* Preserve the prefilters' exact arithmetic independently of the geometric decision. */
static int near_cached(const vec2 *centres, int n, double x, double y, double radius) {
    return first_near(centres, n, x, y, radius) != 0;
}
static int duplicate_near_cached(const vec2 *centres, int n, const napa_detection *d) {
    vec2 point;
    double radius;
    return duplicate_near(centres, n, d, &point, &radius) != 0;
}

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "coverage oracle failed at line %d: %s\n", __LINE__, #condition); \
    return 1; } } while (0)

static int covered_legacy(const napa_detection *out, int n, double x, double y, double radius) {
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

static int already_found_legacy(const napa_detection *out, int n, const napa_detection *d) {
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

static uint32_t seed = 97;
static uint32_t rnd(void) { return seed = seed * 1664525u + 1013904223u; }
static napa_detection point(double x, double y) {
    napa_detection d = {0};
    for (int k = 0; k < 4; k++) d.corners[2 * k] = x, d.corners[2 * k + 1] = y;
    return d;
}
static int check_center(vec2 c, const napa_detection *d) {
    double x = 0, y = 0;
    for (int k = 0; k < 4; k++) {
        x += d->corners[2 * k] * 0.25;
        y += d->corners[2 * k + 1] * 0.25;
    }
    return memcmp(&c.x, &x, sizeof(x)) == 0 && memcmp(&c.y, &y, sizeof(y)) == 0;
}

static int check_public_allocation_failure(napa_detector *det) {
    enum { W = 192, H = 192, CELL = 12, ORIGIN = 48 };
    uint8_t image[W * H];
    const napa_family *family = &napa_families[0];
    memset(image, 255, sizeof(image));
    for (int y = 0; y < family->width * CELL; y++)
        memset(image + (ORIGIN + y) * W + ORIGIN, 0, (size_t)family->width * CELL);
    for (int bit = 0; bit < family->nbits; bit++) {
        if (!(family->codes[137] >> (family->nbits - bit - 1) & 1)) continue;
        for (int y = 0; y < CELL; y++)
            memset(image + (ORIGIN + family->y[bit] * CELL + y) * W + ORIGIN + family->x[bit] * CELL,
                   255, CELL);
    }
    napa_params params = {16, 3, 20, 22, 25, 0};
    napa_detection expected[4], actual[4];
    CHECK(napa_detect(det, image, W, H, W, &params, expected, 4) == 1);
    CHECK(expected[0].id == 137);
    /* The first call primes every image buffer. Fail the new centre allocation, then retry on
     * the same detector: no partial cache may become visible to the next detection call. */
    free(det->out_centers);
    det->out_centers = NULL;
    det->out_center_capacity = 0;
    fail_realloc = 1;
    int status = napa_detect(det, image, W, H, W, &params, actual, 4);
    fail_realloc = 0;
    CHECK(status == -2 && failed_size == 64 * sizeof(vec2));
    CHECK(det->out_centers == NULL && det->out_center_capacity == 0);
    CHECK(napa_detect(det, image, W, H, W, &params, actual, 4) == 1);
    CHECK(actual[0].id == expected[0].id && actual[0].hamming == expected[0].hamming);
    CHECK(actual[0].decision_margin == expected[0].decision_margin);
    CHECK(memcmp(actual[0].corners, expected[0].corners, sizeof(actual[0].corners)) == 0);
    return 0;
}

int main(void) {
    napa_detector *det = napa_create();
    napa_detection ordinary[129], pieces[129];
    int lengths[] = {0, 1, 64, 65, 129};
    CHECK(det != NULL);
    CHECK(!near_cached(NULL, 0, 0, 0, 2));
    napa_detection query = point(0, 0);
    CHECK(!duplicate_near_cached(NULL, 0, &query));

    /* Empty and full growth failures must preserve ownership, capacity and live entries. */
    fail_realloc = 1;
    CHECK(!save_center(&det->out_centers, &det->out_center_capacity, 0, &query));
    fail_realloc = 0;
    CHECK(det->out_centers == NULL && det->out_center_capacity == 0);
    for (int i = 0; i < 129; i++) {
        ordinary[i] = point(1000.0 * i, 0);
        CHECK(save_center(&det->out_centers, &det->out_center_capacity, i, &ordinary[i]));
        CHECK(det->out_center_capacity == (i < 64 ? 64u : i < 128 ? 128u : 256u));
        if (i == 63) {
            vec2 *old = det->out_centers, saved[64];
            size_t cap = det->out_center_capacity;
            memcpy(saved, old, sizeof(saved));
            fail_realloc = 1;
            int ok = save_center(&det->out_centers, &det->out_center_capacity, 64, &query);
            fail_realloc = 0;
            CHECK(!ok && det->out_centers == old && det->out_center_capacity == cap);
            CHECK(memcmp(saved, old, sizeof(saved)) == 0);
        }
    }
    /* Prefixes exclude stale tail entries, including the first entry after each growth edge. */
    for (int k = 0; k < 5; k++) {
        int n = lengths[k];
        query = ordinary[n ? n - 1 : 0];
        CHECK(near_cached(det->out_centers, n, query.corners[0], 0, 1) == (n > 0));
        CHECK(duplicate_near_cached(det->out_centers, n, &query) == (n > 0));
        if (n < 129) {
            CHECK(!near_cached(det->out_centers, n, ordinary[n].corners[0], 0, 1));
            CHECK(!duplicate_near_cached(det->out_centers, n, &ordinary[n]));
        }
    }

    for (int frame = 0; frame < 80; frame++) {
        int n = lengths[frame % 5];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 8; k++) {
                ordinary[i].corners[k] = ldexp((int)(rnd() % 20001) - 10000, (int)(rnd() % 81) - 40);
                pieces[i].corners[k] = (int)(rnd() % 20001) - 30000;
            }
            CHECK(save_center(&det->out_centers, &det->out_center_capacity, i, &ordinary[i]));
            CHECK(save_center(&det->piece_centers, &det->piece_center_capacity, i, &pieces[i]));
            CHECK(check_center(det->out_centers[i], &ordinary[i]));
            CHECK(check_center(det->piece_centers[i], &pieces[i]));
        }
        for (int q = 0; q < 100; q++) {
            query = point((int)(rnd() % 40001) - 20000, (int)(rnd() % 40001) - 20000);
            if (n && q % 3 == 0) query = ordinary[rnd() % n];
            double x = query.corners[0], y = query.corners[1], r = rnd() % 20001;
            CHECK(near_cached(det->out_centers, n, x, y, r) == covered_legacy(ordinary, n, x, y, r));
            CHECK(duplicate_near_cached(det->out_centers, n, &query) == already_found_legacy(ordinary, n, &query));
            CHECK(near_cached(det->piece_centers, n, x, y, r) == covered_legacy(pieces, n, x, y, r));
            CHECK(duplicate_near_cached(det->piece_centers, n, &query) == already_found_legacy(pieces, n, &query));
        }
    }

    /* Ordered accumulation, scaling before addition, cancellation, subnormals and signed zero. */
    const double special[][8] = {
        {1e16, -1e16, 1, -1, -1e16, 1e16, 1, -1},
        {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX},
        {-0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0},
        {DBL_MIN, -DBL_MIN, DBL_MIN, -DBL_MIN, DBL_MIN, -DBL_MIN, DBL_MIN, -DBL_MIN},
        {nextafter(0.0, 1.0), 0, nextafter(0.0, 1.0), 0,
         nextafter(0.0, 1.0), 0, nextafter(0.0, 1.0), 0}
    };
    for (size_t i = 0; i < sizeof(special) / sizeof(special[0]); i++) {
        memcpy(query.corners, special[i], sizeof(query.corners));
        CHECK(save_center(&det->out_centers, &det->out_center_capacity, 0, &query));
        CHECK(check_center(det->out_centers[0], &query));
        CHECK(near_cached(det->out_centers, 1, 0, 0, 0.25) == covered_legacy(&query, 1, 0, 0, 0.25));
        CHECK(duplicate_near_cached(det->out_centers, 1, &query) == already_found_legacy(&query, 1, &query));
    }

    query = point(0, 0);
    CHECK(save_center(&det->out_centers, &det->out_center_capacity, 0, &query));
    double boundary[] = {nextafter(2.0, 0.0), 2.0, nextafter(2.0, INFINITY)};
    for (int i = 0; i < 3; i++) {
        napa_detection q = point(boundary[i], -0.0);
        CHECK(near_cached(det->out_centers, 1, boundary[i], 0, 2) == (i == 0));
        CHECK(duplicate_near_cached(det->out_centers, 1, &q) == (i == 0));
    }
    CHECK(!near_cached(det->out_centers, 1, 0, -0.0, 0));
    CHECK(det->out_centers != det->piece_centers);
    CHECK(check_public_allocation_failure(det) == 0);
    napa_destroy(det);
    puts("coverage cache exact-oracle cases passed");
    return 0;
}
