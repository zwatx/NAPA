/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Independent pixel scan, including append offsets and untouched buffer suffixes. */
#include "../src/napa_vision/_native/napa_core.c"
#include <stdio.h>

/* An empty upper-row rank map isolates fused emission and self-parent writes.
 * The pixel connectivity oracle separately covers actual nonempty-row unions. */
static int isolated_labeled_runs(const uint64_t *bits, int width, int y, napa_run *runs,
                                 int prefix, int capacity) {
    napa_detector det = {0};
    det.runs = runs;
    det.parent = malloc((size_t)capacity * sizeof(int32_t));
    int words = (width + 63) / 64 + 2;
    uint64_t *mask = calloc((size_t)words, sizeof(uint64_t));
    int32_t *rank = calloc((size_t)words, sizeof(int32_t));
    if (!det.parent || !mask || !rank) exit(2);
    for (int i = 0; i < capacity; i++) det.parent[i] = -17;
    int n = extract_labeled_runs(&det, bits, width, 30000, 1, y, prefix, prefix, mask, rank, 0, 0);
    for (int i = 0; i < capacity; i++) {
        int expected = i >= prefix && i < n ? i : -17;
        if (det.parent[i] != expected) {
            fprintf(stderr,"labeled parent/sentinel width=%d slot=%d\n",width,i); exit(5);
        }
    }
    free(det.parent); free(mask); free(rank);
    return n;
}

static unsigned long cases;
static void check(const uint64_t *bits, int width) {
    int capacity = (width + 1) / 2 + 8, prefix = 3, y = 29999;
    napa_run *actual = malloc((size_t)capacity * sizeof(napa_run));
    napa_run *expected = malloc((size_t)capacity * sizeof(napa_run));
    if (!actual || !expected) exit(2);
    memset(actual, 0xa5, (size_t)capacity * sizeof(napa_run));
    for (int i = 0; i < prefix; i++) {
        actual[i].x0 = actual[i].x1 = (uint16_t)(i % width);
        actual[i].y = (uint16_t)i;
    }
    memcpy(expected, actual, (size_t)capacity * sizeof(napa_run));
    int n = prefix, x = 0;
    while (x < width) {
        while (x < width && !(bits[x / 64] >> (x % 64) & 1)) x++;
        if (x == width) break;
        int start = x;
        while (x < width && (bits[x / 64] >> (x % 64) & 1)) x++;
        expected[n].x0 = (uint16_t)start; expected[n].x1 = (uint16_t)(x - 1); expected[n].y = (uint16_t)y;
        n++;
    }
    for (int labeled = 0; labeled < 2; labeled++) {
        memset(actual, 0xa5, (size_t)capacity * sizeof(napa_run));
        memcpy(actual, expected, (size_t)prefix * sizeof(napa_run));
        int found = labeled ? isolated_labeled_runs(bits, width, y, actual, prefix, capacity)
                            : extract_runs(bits, width, y, actual, prefix);
        if (found != n) { fprintf(stderr, "run count width=%d case=%lu: %d != %d\n", width, cases, found, n); exit(3); }
        for (int i = 0; i < capacity; i++) if (actual[i].x0 != expected[i].x0 ||
                actual[i].x1 != expected[i].x1 || actual[i].y != expected[i].y) {
            fprintf(stderr, "run fields width=%d case=%lu slot=%d\n", width, cases, i); exit(4);
        }
    }
    free(actual); free(expected); cases++;
}

static uint32_t state = 83;
static uint32_t random_word(void) { return state = state * 1664525u + 1013904223u; }

int main(void) {
    const int offsets[] = {0, 48, 55, 60, 63, 64, 112};
    for (int k = 0; k < 7; k++) {
        int offset = offsets[k], width = offset + 16;
        for (unsigned mask = 0; mask < 65536; mask++) {
            uint64_t bits[3] = {0};
            for (int x = 0; x < 16; x++) if (mask & (1u << x))
                bits[(offset + x) / 64] |= (uint64_t)1 << ((offset + x) % 64);
            check(bits, width);
        }
    }
    const int widths[] = {1,2,3,4,15,16,17,63,64,65,127,128,129,29999,30000};
    for (int k = 0; k < 15; k++) {
        int width = widths[k], words = (width + 63) / 64;
        uint64_t *bits = calloc((size_t)words, sizeof(uint64_t));
        if (!bits) return 2;
        for (int pattern = 0; pattern < 36; pattern++) {
            for (int w = 0; w < words; w++) {
                uint64_t hi = random_word();
                uint64_t r = (hi << 32) | random_word();
                bits[w] = pattern == 0 ? 0 : pattern == 1 ? UINT64_MAX :
                          pattern == 2 ? UINT64_C(0xaaaaaaaaaaaaaaaa) :
                          pattern == 3 ? UINT64_C(0x5555555555555555) : r;
            }
            if (width % 64) bits[words - 1] &= ((uint64_t)1 << (width % 64)) - 1;
            check(bits, width);
        }
        free(bits);
    }
    printf("%lu exact pixel-scan run oracle cases passed (plain and labeled)\n", cases);
    return 0;
}
