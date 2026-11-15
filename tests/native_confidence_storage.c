/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Exercise actual deferred confidence readings, growth and allocation failure. */
#ifndef NAPA_CONFIDENCE_GROWTH_INPUT
#error "Build with the owned 65-tag input path; pytest supplies it."
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail_enabled, fail_hits;
static void *fail_pointer;
static size_t fail_bytes;
static void *storage_realloc(void *p, size_t bytes) {
    if (fail_enabled && p == fail_pointer && bytes == fail_bytes) {
        fail_hits++;
        return NULL;
    }
    return realloc(p, bytes);
}
#define realloc storage_realloc
#include "../src/napa_vision/_native/napa_core.c"
#undef realloc

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "confidence storage line %d: %s\n", __LINE__, #c); return 1; } } while (0)

enum { PITCH_X = 384, PITCH_Y = 288, WIDTH = 9 * PITCH_X, HEIGHT = 8 * PITCH_Y, COUNT = 65 };
static uint8_t image[WIDTH * HEIGHT];
static napa_detection expected[128], actual[128];

static int same(const napa_detection *a, const napa_detection *b, int n) {
    for (int i = 0; i < n; i++)
        if (a[i].id != b[i].id || a[i].hamming != b[i].hamming ||
            a[i].decision_margin != b[i].decision_margin ||
            memcmp(a[i].corners, b[i].corners, sizeof a[i].corners)) return 0;
    return 1;
}

int main(void) {
    FILE *stream = fopen(NAPA_CONFIDENCE_GROWTH_INPUT, "rb");
    CHECK(stream && fread(image, 1, sizeof image, stream) == sizeof image);
    fclose(stream);
    napa_detector *det = napa_create();
    CHECK(det);
    napa_params p = {32, 3, 20, 22, 25, 0};
    CHECK(napa_detect(det, image, WIDTH, HEIGHT, WIDTH, &p, NULL, 0) == 0);
    int n = napa_detect(det, image, WIDTH, HEIGHT, WIDTH, &p, expected, 128);
    CHECK(n == COUNT);
    CHECK(det->fine_whole_capacity >= 128);
    unsigned char seen[72] = {0};
    for (int i = 0; i < COUNT; i++) {
        CHECK(expected[i].id == 2 && expected[i].hamming == 0);
        double x = 0, y = 0;
        for (int k = 0; k < 4; k++) x += expected[i].corners[2*k] / 4, y += expected[i].corners[2*k+1] / 4;
        int cell = (int)(y / PITCH_Y) * 9 + (int)(x / PITCH_X);
        CHECK(cell >= 0 && cell < COUNT && !seen[cell]);
        seen[cell] = 1;
    }
    for (int cap = 1; cap <= 64; cap *= 8)
        CHECK(napa_detect(det, image, WIDTH, HEIGHT, WIDTH, &p, actual, cap) == cap && same(expected, actual, cap));
    /* Fail the first deferred allocation, its 64->128 growth, and both output
     * centre allocations. Retry with the same detector after each failure. */
    for (int centers = 0; centers < 2; centers++) {
        for (int reserve = 0; reserve < 2; reserve++) {
            if (centers) {
                free(det->out_centers);
                det->out_centers = reserve ? malloc(64 * sizeof(vec2)) : NULL;
                det->out_center_capacity = reserve ? 64 : 0;
                CHECK(!reserve || det->out_centers);
                fail_pointer = det->out_centers;
            } else {
                free(det->fine_wholes);
                det->fine_wholes = reserve ? malloc(64 * sizeof(fine_whole)) : NULL;
                det->fine_whole_capacity = reserve ? 64 : 0;
                CHECK(!reserve || det->fine_wholes);
                fail_pointer = det->fine_wholes;
            }
            fail_bytes = (size_t)(reserve ? 128 : 64) * (centers ? sizeof(vec2) : sizeof(fine_whole));
            fail_enabled = 1;
            fail_hits = 0;
            n = napa_detect(det, image, WIDTH, HEIGHT, WIDTH, &p, actual, 128);
            fail_enabled = 0;
            CHECK(n == -2 && fail_hits == 1);
            if (centers) {
                CHECK(det->out_centers == fail_pointer && det->out_center_capacity == (size_t)(reserve ? 64 : 0));
            } else {
                CHECK(det->fine_wholes == fail_pointer && det->fine_whole_capacity == (size_t)(reserve ? 64 : 0));
            }
            CHECK(napa_detect(det, image, WIDTH, HEIGHT, WIDTH, &p, actual, 128) == COUNT && same(expected, actual, COUNT));
        }
    }
    napa_destroy(det);
    puts("65 actual confidence readings, growth, capped prefixes, four allocation failures and retries passed");
    return 0;
}
