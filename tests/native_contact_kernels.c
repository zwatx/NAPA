/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Independent pixel flood-fill oracle for the threshold/run connectivity kernels. */
#include "../src/napa_vision/_native/napa_core.c"
#include <stdio.h>

static int floor_at(const napa_detector *d, int x, int y, int factor) {
    int rx = (x / 4) * 4 * factor / 64, ry = (y / 4) * 4 * factor / 64;
    if (d->floor_uniform) return d->floors[0];
    if (rx >= d->floor_cols) rx = d->floor_cols - 1;
    if (ry >= d->floor_rows) ry = d->floor_rows - 1;
    return d->floors[ry * d->floor_cols + rx];
}

/* Directly scan each tile's 3x3 neighbourhood; no production threshold helpers. */
static void dark_mask(const napa_detector *d, int w, int h, int factor, int shadow, uint8_t *mask) {
    for (int ty = 0; ty < h; ty += 4) for (int tx = 0; tx < w; tx += 4) {
        int lo = 255, hi = 0;
        for (int y = ty - 4; y < ty + 8 && y < h; y++) {
            if (y < 0) continue;
            for (int x = tx - 4; x < tx + 8 && x < w; x++) {
                if (x < 0) continue;
                int v = d->level[y * w + x];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
        }
        int t = hi - lo < floor_at(d, tx, ty, factor) ? 0 :
                shadow ? lo + (hi - lo + 2) / 4 : (lo + hi + 1) / 2;
        for (int y = ty; y < ty + 4 && y < h; y++)
            for (int x = tx; x < tx + 4 && x < w; x++) mask[y * w + x] = d->level[y * w + x] < t;
    }
}

/* A diagonal pixel edge can be removed only between two supported solid blocks.
 * Scan rectangles and horizontal pixel spans instead of production run endpoints. */
static int omit_edge(const napa_detector *d, const uint8_t *mask, int w, int h,
                     int factor, int x, int y, int dx) {
    int bx = x + dx, left = x < bx ? x : bx;
    if (mask[y * w + bx] || mask[(y + 1) * w + x]) return 0;
    int len_a = 1, len_b = 1;
    for (int xx = x - dx; xx >= 0 && xx < w && mask[y * w + xx]; xx -= dx) len_a++;
    for (int xx = bx + dx; xx >= 0 && xx < w && mask[(y + 1) * w + xx]; xx += dx) len_b++;
    if (len_a < 2 || len_b < 2 || left < 1 || left + 2 >= w || y < 1 || y + 2 >= h) return 0;
    int limit = d->level[y * w + x];
    if (d->level[(y + 1) * w + bx] > limit) limit = d->level[(y + 1) * w + bx];
    int white = d->level[y * w + bx];
    if (d->level[(y + 1) * w + x] < white) white = d->level[(y + 1) * w + x];
    int floor = 0;
    for (int yy = y; yy <= y + 1; yy++) for (int xx = left; xx <= left + 1; xx++) {
        int f = floor_at(d, xx, yy, factor);
        if (f > floor) floor = f;
    }
    if (white - limit < floor) return 0;
    for (int block = 0; block < 2; block++) {
        int x0 = block ? bx : x - dx, y0 = block ? y + 1 : y - 1;
        if (dx < 0) x0--;
        for (int yy = y0; yy < y0 + 2; yy++) for (int xx = x0; xx < x0 + 2; xx++)
            if (d->level[yy * w + xx] > limit) return 0;
    }
    return 1;
}

static int cases = 0;
static void check_impl(napa_detector *d, int w, int h, int factor, const char *name, int expected, int all_modes) {
    int pixels = w * h;
    uint8_t *mask = malloc((size_t)pixels);
    int *ids = malloc((size_t)pixels * sizeof(int)), *labels = malloc((size_t)pixels * sizeof(int));
    int *queue = malloc((size_t)pixels * sizeof(int)), *roots = malloc((size_t)pixels * sizeof(int));
    napa_run *runs = malloc((size_t)pixels * sizeof(napa_run));
    if (!mask || !ids || !labels || !queue || !roots || !runs) exit(2);
    for (int shadow = 0; shadow <= (all_modes ? 1 : 0); shadow++) {
        dark_mask(d, w, h, factor, shadow, mask);
        int n = 0, components = 0;
        for (int p = 0; p < pixels; p++) ids[p] = labels[p] = -1;
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) if (mask[y * w + x]) {
            runs[n].x0 = x; runs[n].y = y;
            do { ids[y * w + x] = n; x++; } while (x < w && mask[y * w + x]);
            runs[n++].x1 = x - 1;
        }
        for (int start = 0; start < pixels; start++) if (mask[start] && labels[start] < 0) {
            int head = 0, tail = 1, root = ids[start];
            queue[0] = start; labels[start] = components;
            while (head < tail) {
                int p = queue[head++], x = p % w, y = p / w;
                if (ids[p] < root) root = ids[p];
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    int xx = x + dx, yy = y + dy;
                    if ((!dx && !dy) || xx < 0 || xx >= w || yy < 0 || yy >= h) continue;
                    int q = yy * w + xx;
                    if (!mask[q] || labels[q] >= 0) continue;
                    if (!shadow && dx && dy && omit_edge(d, mask, w, h, factor,
                            dy > 0 ? x : xx, dy > 0 ? y : yy, dy > 0 ? dx : -dx)) continue;
                    labels[q] = components; queue[tail++] = q;
                }
            }
            for (int k = 0; k < tail; k++) roots[ids[queue[k]]] = root;
            components++;
        }
        if (!shadow && expected >= 0 && components != expected) {
            fprintf(stderr, "%s: oracle fixture has %d components, expected %d\n", name, components, expected); exit(3);
        }
        for (int split = 0; split <= (all_modes ? 2 : 0); split++) for (int hw = 0; hw < 2; hw++) {
            int actual;
#ifdef NAPA_SSSE3_DISPATCH
            if (hw && __builtin_cpu_supports("popcnt")) actual = threshold_index_popcnt(d, w, h, factor, split, shadow);
            else
#endif
            actual = threshold_index_generic(d, w, h, factor, split, shadow);
            if (actual != n) { fprintf(stderr, "%s: run count %d != %d\n", name, actual, n); exit(4); }
            /* Reconstruct all deferred split edges before comparing the graph. */
            for (int kind = 0; kind < 2; kind++) {
                int32_t *pairs = kind ? d->neck : d->diag;
                size_t count = kind ? d->neck_n : d->diag_n;
                for (size_t k = 0; k < count; k += 2) {
                    int a = find_root(d->parent, pairs[k]), b = find_root(d->parent, pairs[k + 1]);
                    d->parent[a > b ? a : b] = a < b ? a : b;
                }
            }
            for (int k = 0; k < n; k++) if (d->runs[k].x0 != runs[k].x0 || d->runs[k].x1 != runs[k].x1 ||
                    d->runs[k].y != runs[k].y || find_root(d->parent, k) != roots[k]) {
                fprintf(stderr, "%s %dx%d factor=%d shadow=%d split=%d hw=%d run=%d root=%d expected=%d\n",
                        name, w, h, factor, shadow, split, hw, k, find_root(d->parent, k), roots[k]); exit(5);
            }
        }
    }
    free(mask); free(ids); free(labels); free(queue); free(roots); free(runs); cases++;
}

static void check(napa_detector *d, int w, int h, int factor, const char *name, int expected) {
    check_impl(d, w, h, factor, name, expected, 1);
}

static napa_detector *image(int w, int h, int factor) {
    napa_detector *d = napa_create();
    if (!d || ensure_buffers(d, (size_t)w * h, (size_t)((w + 3) / 4) * ((h + 3) / 4),
                             (size_t)w, (size_t)w * h) != 0) exit(2);
    memset(d->level, 240, (size_t)w * h + 64);
    d->floor_cols = (w * factor + 63) / 64; d->floor_rows = (h * factor + 63) / 64;
    d->floors = malloc((size_t)d->floor_cols * d->floor_rows);
    if (!d->floors) exit(2);
    memset(d->floors, 24, (size_t)d->floor_cols * d->floor_rows); d->floor_uniform = 1;
    return d;
}

static void block(napa_detector *d, int w, int x, int y, int width, int height, int gray) {
    for (int yy = y; yy < y + height; yy++) for (int xx = x; xx < x + width; xx++)
        d->level[yy * w + xx] = (uint8_t)gray;
}
static void reflect(napa_detector *d, int w, int h) {
    for (int y = 0; y < h; y++) for (int x = 0; x < w / 2; x++) {
        uint8_t t = d->level[y * w + x]; d->level[y * w + x] = d->level[y * w + w - 1 - x];
        d->level[y * w + w - 1 - x] = t;
    }
    for (int y = 0; y < d->floor_rows; y++) for (int x = 0; x < d->floor_cols / 2; x++) {
        int a = y * d->floor_cols + x, b = y * d->floor_cols + d->floor_cols - 1 - x;
        uint8_t t = d->floors[a]; d->floors[a] = d->floors[b]; d->floors[b] = t;
    }
}
static void reflect_vertical(napa_detector *d, int w, int h) {
    for (int y = 0; y < h / 2; y++) for (int x = 0; x < w; x++) {
        int a = y * w + x, b = (h - 1 - y) * w + x;
        uint8_t t = d->level[a]; d->level[a] = d->level[b]; d->level[b] = t;
    }
    for (int y = 0; y < d->floor_rows / 2; y++) for (int x = 0; x < d->floor_cols; x++) {
        int a = y * d->floor_cols + x, b = (d->floor_rows - 1 - y) * d->floor_cols + x;
        uint8_t t = d->floors[a]; d->floors[a] = d->floors[b]; d->floors[b] = t;
    }
}

int main(void) {
    const int rank_widths[] = {1, 2, 63, 64, 65, 127, 128, 129, 29952, 29999, 30000};
    for (int wi = 0; wi < 11; wi++) for (int pattern = 0; pattern < 6; pattern++) {
        int w = rank_widths[wi], h = 6;
        napa_detector *d = image(w, h, 1);
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            int dark = pattern == 0 ? !(x & 1) : pattern == 1 ? !((x + y) & 1) :
                       pattern == 2 ? (x + y) % 4 < 2 : pattern == 3 ? (y & 1 ? !(x & 1) : 1) :
                       pattern == 4 ? (y != 2 && !(x & 1)) : (y == 1 || y == 4 ? 0 : (x + y) % 3 < 2);
            d->level[y * w + x] = dark ? 20 : 240;
        }
        check(d, w, h, 1, "wide transition ranks", -1);
        napa_destroy(d);
    }

    const int widths[] = {1, 2, 3, 4, 5, 15, 16, 17, 63, 64, 65, 127, 128, 129};
    uint32_t state = 83;
    for (int c = 0; c < 196; c++) {
        int w = widths[c % 14], h = 1 + c % 19, factor = 1 + c % 4;
        napa_detector *d = image(w, h, factor);
        d->floor_uniform = c % 2;
        for (int k = 0; k < d->floor_cols * d->floor_rows; k++) d->floors[k] = (uint8_t)(8 + (c + k * 29) % 90);
        for (int p = 0; p < w * h; p++) {
            state = state * 1664525u + 1013904223u;
            d->level[p] = c % 3 ? (uint8_t)((state >> 24) & 255) : (uint8_t)((state >> 31) * 240);
        }
        check(d, w, h, factor, "generated", -1); napa_destroy(d);
    }
    /* Both ends of the indexed [f,g) range, including a surviving middle run. */
    for (int middle = 0; middle < 2; middle++) {
        napa_detector *d = image(20, 8, 1);
        block(d, 20, 2, 1, 2, 2, 20); block(d, 20, 12, 1, 2, 2, 20);
        block(d, 20, 4, 3, 8, 2, 20);
        if (middle) block(d, 20, 7, 1, 2, 2, 20);
        check(d, 20, 8, 1, "two trimmed contacts", 3); napa_destroy(d);
    }
    for (int variant = 0; variant < 9; variant++) for (int mirrored = 0; mirrored < 4; mirrored++) {
        int w = 128, h = 128, x = variant == 1 ? 0 : 62, y = variant == 2 ? 0 : 62;
        napa_detector *d = image(w, h, 1);
        if (variant == 4) memset(d->level, 70, (size_t)w * h);
        if (variant >= 5) memset(d->level, 160, (size_t)w * h);
        block(d, w, x, y, 2, 2, 40); block(d, w, x + 2, y + 2, 2, 2, 40);
        int expected = 2;
        if (variant == 3) { d->level[y * w + x] = 45; expected = 1; } /* support above endpoint */
        if (variant == 4) { d->level[(y + 1) * w + x + 2] = 55; expected = 1; } /* weak opposite white */
        if (variant >= 5) {
            d->floor_uniform = 0;
            d->floors[(variant == 5 || variant == 8 ? 0 : 1) * d->floor_cols + (variant >= 7 ? 0 : 1)] = 100;
            d->level[(y + 1) * w + x + 2] = 120;
            expected = 1;
        }
        if (mirrored & 1) reflect(d, w, h);
        if (mirrored & 2) reflect_vertical(d, w, h);
        char name[80];
        snprintf(name, sizeof(name), "filled/gray/boundary variant=%d reflection=%d", variant, mirrored);
        check(d, w, h, 1, name, expected); napa_destroy(d);
    }
    for (int bottom = 0; bottom < 2; bottom++) {
        napa_detector *d = image(8, 3, 1);
        block(d, 8, 1, 0, 2, 1, 40); block(d, 8, 3, 1, 2, 2, 40);
        if (bottom) reflect_vertical(d, 8, 3);
        check(d, 8, 3, 1, "truncated support", 1); napa_destroy(d);
    }
    for (int thick = 1; thick <= 2; thick++) for (int mirrored = 0; mirrored < 2; mirrored++) {
        napa_detector *d = image(70, 12, 1);
        for (int y = 0; y < 12; y++) block(d, 70, 42 + thick * y, y, thick, 1, 20);
        if (mirrored) reflect(d, 70, 12);
        check(d, 70, 12, 1, "thin staircase", 1); napa_destroy(d);
    }
    /* Every pair of byte patterns, repeated vertically to make the support blocks solid.
     * Shifts straddle 64-bit transition words; pixel BFS independently checks every contact. */
    const int shifts[] = {1, 60, 124};
    for (int at = 0; at < 3; at++) {
        int offset = shifts[at], w = offset + 9;
        napa_detector *d = image(w, 4, 1);
        for (int upper = 0; upper < 256; upper++) for (int lower = 0; lower < 256; lower++) {
            memset(d->level, 240, (size_t)w * 4);
            for (int y = 0; y < 4; y++) for (int x = 0; x < 8; x++) {
                int pattern = y < 2 ? upper : lower;
                if (pattern & (1 << x)) d->level[y * w + offset + x] = 0;
            }
            check_impl(d, w, 4, 1, "exhaustive row-pair contacts", -1, 0);
        }
        napa_destroy(d);
    }
    printf("%d pixel connectivity oracle cases passed\n", cases);
    return 0;
}
