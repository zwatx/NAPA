/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Exact independent oracles for the threshold kernels: dark-bit packing and 4x4 tile extrema,
 * in whichever SIMD path (SSE2, NEON or scalar) this build selects. */
#ifdef NDEBUG
#error "The band kernel oracle requires assertions"
#endif
#include "../src/napa_vision/_native/napa_core.c"
#include <assert.h>
#include <stdio.h>

static uint32_t state = 49281;
static uint8_t next_byte(void) {
    state = state * 1664525u + 1013904223u;
    return (uint8_t)(state >> 24);
}

static unsigned long check_dark_bits(void) {
    uint8_t row[256] = {0}, thr[256] = {0}, saved_row[256], saved_thr[256];
    unsigned long cases = 0;
    for (int width = 1; width <= 256; width++) {
        int rounded = (width + 63) / 64 * 64;
        for (int mode = 0; mode < 80; mode++) {
            for (int x = 0; x < rounded; x++) {
                row[x] = next_byte();
                thr[x] = x >= width ? 0 : mode == 0 ? 0 : mode == 1 ? 255 :
                    mode == 2 ? row[x] : mode < 67 ? (x % 64 == mode - 3 ? 255 : 0) : next_byte();
            }
            memcpy(saved_row, row, sizeof row);
            memcpy(saved_thr, thr, sizeof thr);
            uint64_t words[6];
            for (int j = 0; j < 6; j++) words[j] = UINT64_C(0xa57ca57ca57ca57c);
            dark_bits(row, thr, width, words + 1);
            assert(words[0] == UINT64_C(0xa57ca57ca57ca57c));
            assert(words[rounded / 64 + 1] == UINT64_C(0xa57ca57ca57ca57c));
            for (int x = 0; x < rounded; x++)
                assert(((words[1 + x / 64] >> (x % 64)) & 1) == (unsigned)(row[x] < thr[x]));
            assert(memcmp(row, saved_row, sizeof row) == 0);
            assert(memcmp(thr, saved_thr, sizeof thr) == 0);
            cases++;
        }
    }
    return cases;
}

/* Tile extrema over rows [4ty, min(4ty+4, lh)) and columns [4tx, min(4tx+4, lw)). */
static unsigned long check_tile_extrema(void) {
    enum { MAXW = 300, MAXH = 11, PAD = 64 };
    static uint8_t image[MAXH * (MAXW + PAD)];
    static uint8_t vmin[MAXW + PAD], vmax[MAXW + PAD];
    unsigned long cases = 0;
    for (int lw = 1; lw <= MAXW; lw += (lw < 140 ? 1 : 7)) {
        for (int lh = 1; lh <= MAXH; lh++) {
            for (int mode = 0; mode < 4; mode++) {
                const size_t stride = (size_t)lw + (mode & 1) * 13;
                for (size_t i = 0; i < sizeof image; i++)
                    image[i] = mode == 2 ? (uint8_t)(i % 3 ? 255 : 0) : mode == 3 ? (uint8_t)(128 + (next_byte() & 1)) : next_byte();
                const int tw = (lw + TILE - 1) / TILE, th = (lh + TILE - 1) / TILE;
                for (int ty = 0; ty < th; ty++) {
                    uint8_t tmin[MAXW / TILE + 20], tmax[MAXW / TILE + 20];
                    memset(tmin, 0xa5, sizeof tmin);
                    memset(tmax, 0xa5, sizeof tmax);
                    tile_row_minmax(image, stride, lw, lh, ty, tw, vmin, vmax, tmin, tmax);
                    for (int tx = 0; tx < tw; tx++) {
                        uint8_t lo = 255, hi = 0;
                        for (int y = ty * TILE; y < ty * TILE + TILE && y < lh; y++)
                            for (int x = tx * TILE; x < tx * TILE + TILE && x < lw; x++) {
                                uint8_t v = image[(size_t)y * stride + (size_t)x];
                                lo = v < lo ? v : lo;
                                hi = v > hi ? v : hi;
                            }
                        assert(tmin[tx] == lo && tmax[tx] == hi);
                    }
                    for (int tx = tw; tx < (int)sizeof tmin; tx++) assert(tmin[tx] == 0xa5 && tmax[tx] == 0xa5);
                    cases++;
                }
            }
        }
    }
    return cases;
}

/* Half-tile extrema (pair_extrema over two level rows) against the samples, and the tile rows paired from them
 * against tile_row_minmax. */
static unsigned long check_half_extrema(void) {
    enum { MAXW = 200, MAXH = 9, PAD = 64 };
    static uint8_t image[MAXH * MAXW];
    static uint8_t hmin[(MAXH + 1) / 2][MAXW / 2 + PAD], hmax[(MAXH + 1) / 2][MAXW / 2 + PAD];
    static uint8_t vmin[MAXW + PAD], vmax[MAXW + PAD];
    unsigned long cases = 0;
    for (int lw = 1; lw <= MAXW; lw += (lw < 100 ? 1 : 5)) {
        for (int lh = 1; lh <= MAXH; lh++) {
            for (int mode = 0; mode < 3; mode++) {
                for (size_t i = 0; i < sizeof image; i++)
                    image[i] = mode == 1 ? (uint8_t)(i % 3 ? 255 : 0) : mode == 2 ? (uint8_t)(128 + (next_byte() & 1)) : next_byte();
                const int hw = (lw + 1) / 2, hh = (lh + 1) / 2, tw = (lw + TILE - 1) / TILE, th = (lh + TILE - 1) / TILE;
                memset(hmin, 0xa5, sizeof hmin);
                memset(hmax, 0xa5, sizeof hmax);
                for (int r = 0; r < hh; r++) {
                    half_row_minmax(image, lw, lh, r, hmin[r], hmax[r]);
                    for (int tx = 0; tx < hw; tx++) {
                        uint8_t lo = 255, hi = 0;
                        for (int y = 2 * r; y < 2 * r + 2 && y < lh; y++)
                            for (int x = 2 * tx; x < 2 * tx + 2 && x < lw; x++) {
                                uint8_t v = image[(size_t)y * lw + x];
                                lo = v < lo ? v : lo;
                                hi = v > hi ? v : hi;
                            }
                        assert(hmin[r][tx] == lo && hmax[r][tx] == hi);
                    }
                    for (int tx = hw; tx < MAXW / 2 + PAD; tx++) assert(hmin[r][tx] == 0xa5 && hmax[r][tx] == 0xa5);
                }
                for (int ty = 0; ty < th; ty++) {
                    uint8_t tmin[MAXW / TILE + 20], tmax[MAXW / TILE + 20], pmin[MAXW / TILE + 20], pmax[MAXW / TILE + 20];
                    memset(pmin, 0xa5, sizeof pmin);
                    memset(pmax, 0xa5, sizeof pmax);
                    const int a = 2 * ty, b = 2 * ty + 1 < hh ? 2 * ty + 1 : a;
                    tile_row_minmax(image, (size_t)lw, lw, lh, ty, tw, vmin, vmax, tmin, tmax);
                    pair_extrema(hmin[a], hmin[b], hmax[a], hmax[b], hw, pmin, pmax);
                    assert(memcmp(tmin, pmin, (size_t)tw) == 0 && memcmp(tmax, pmax, (size_t)tw) == 0);
                    for (int tx = tw; tx < (int)sizeof pmin; tx++) assert(pmin[tx] == 0xa5 && pmax[tx] == 0xa5);
                    cases++;
                }
            }
        }
    }
    return cases;
}

/* A level of half tiles (level_half) against the definition, sample by sample: dark where the sample is under its
 * tile's threshold and, when its half tile's 3x3 neighbourhood has that contrast too, under that neighbourhood's cap. */
static unsigned long check_half_level(void) {
    static const int sizes[][2] = {{16, 16}, {33, 18}, {70, 21}, {131, 40}, {256, 19}, {67, 130}, {300, 77}};
    unsigned long cases = 0, capped = 0;
    for (int s = 0; s < (int)(sizeof sizes / sizeof sizes[0]); s++) {
        const int w = sizes[s][0], h = sizes[s][1], tw = (w + 3) / 4, th = (h + 3) / 4, hw = (w + 1) / 2, hh = (h + 1) / 2;
        const size_t pixels = (size_t)w * h;
        for (int mode = 0; mode < 6; mode++) {
            napa_detector *det = napa_create();
            assert(det != NULL && ensure_buffers(det, pixels, (size_t)tw * th, (size_t)w, pixels / 2 + (size_t)h + 1) == 0);
            uint8_t *L = det->level;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int v = next_byte();
                    if (mode == 1) v = ((x / 9 + y / 7) & 1) ? 30 + (v & 15) : 130 + (v & 63);
                    else if (mode == 2) v = (x / 6 + y / 6) % 3 == 0 ? 20 : (x / 6 + y / 6) % 3 == 1 ? 110 + (v & 7) : 235;
                    else if (mode == 3) v = 128 + (v & 3);
                    else if (mode == 4) v = x < w / 2 ? (v & 1 ? 0 : 255) : 90;
                    L[(size_t)y * w + x] = (uint8_t)v;
                }
            const int regional = mode & 1, cols = regional ? (w + 63) / 64 : 1, rows = regional ? (h + 63) / 64 : 1;
            det->floors = (uint8_t *)malloc((size_t)cols * rows);
            assert(det->floors != NULL);
            for (int i = 0; i < cols * rows; i++) det->floors[i] = (uint8_t)(regional ? 4 + next_byte() % 40 : 8);
            det->floor_cols = cols, det->floor_rows = rows, det->floor_uniform = !regional;
            det->level_contrast_min = (uint8_t)(mode == 5 ? 60 : 0);
            det->level_half = 1;
            const int n = threshold_rows(det, w, h, 1, 0, 0, 0, th, 0);
            assert(n >= 0);
            uint8_t *got = (uint8_t *)calloc(pixels, 1);
            assert(got != NULL);
            const int stride = (w + 63) / 64 + 1;
            for (int i = 0; i < n; i++)
                for (int row = 0; row < 2; row++) {
                    const napa_run *r = &det->runs[i];
                    if (!((r->rows >> row) & 1)) continue;
                    const uint64_t *b = det->level_bits + (size_t)(r->y + row) * stride;
                    for (int x = r->x0; x <= r->x1; x++) got[(size_t)(r->y + row) * w + x] = (uint8_t)((b[x >> 6] >> (x & 63)) & 1);
                }
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int lo = 255, hi = 0, hlo = 255, hhi = 0;
                    for (int ty = y / 4 - 1; ty <= y / 4 + 1; ty++)
                        for (int tx = x / 4 - 1; tx <= x / 4 + 1; tx++) {
                            const int cy = ty < 0 ? 0 : ty >= th ? th - 1 : ty, cx = tx < 0 ? 0 : tx >= tw ? tw - 1 : tx;
                            for (int v = 4 * cy; v < 4 * cy + 4 && v < h; v++)
                                for (int u = 4 * cx; u < 4 * cx + 4 && u < w; u++) {
                                    const int p = L[(size_t)v * w + u];
                                    lo = p < lo ? p : lo, hi = p > hi ? p : hi;
                                }
                        }
                    for (int ty = y / 2 - 1; ty <= y / 2 + 1; ty++)
                        for (int tx = x / 2 - 1; tx <= x / 2 + 1; tx++) {
                            const int cy = ty < 0 ? 0 : ty >= hh ? hh - 1 : ty, cx = tx < 0 ? 0 : tx >= hw ? hw - 1 : tx;
                            for (int v = 2 * cy; v < 2 * cy + 2 && v < h; v++)
                                for (int u = 2 * cx; u < 2 * cx + 2 && u < w; u++) {
                                    const int p = L[(size_t)v * w + u];
                                    hlo = p < hlo ? p : hlo, hhi = p > hhi ? p : hhi;
                                }
                        }
                    int ry = (y / 4) * 4 / NAPA_REGION, rx = (x / 4) * 4 / NAPA_REGION;
                    ry = ry < rows ? ry : rows - 1, rx = rx < cols ? rx : cols - 1;
                    const int floor = det->floors[(size_t)ry * cols + rx], least = det->level_contrast_min;
                    int thr = hi - lo >= floor && hi - lo >= least ? (lo + hi + 1) >> 1 : 0;
                    if (hhi - hlo >= floor && hhi - hlo >= least) {
                        const int cap = hhi - (((hhi - hlo) * NAPA_HALF_CAP) >> 4);
                        if (cap < thr) thr = cap, capped++;
                    }
                    if (got[(size_t)y * w + x] != (L[(size_t)y * w + x] < thr)) {
                        fprintf(stderr, "%dx%d mode %d: sample (%d, %d) differs\n", w, h, mode, x, y);
                        exit(1);
                    }
                }
            free(got);
            napa_destroy(det);
            cases++;
        }
    }
    assert(capped > 1000);
    return cases;
}

int main(void) {
    unsigned long bits = check_dark_bits(), tiles = check_tile_extrema();
    printf("%lu exact dark-bit cases, %lu exact tile-extrema rows passed\n", bits, tiles);
    unsigned long halves = check_half_extrema(), levels = check_half_level();
    printf("%lu tile rows paired from exact half-tile extrema, %lu half-tile levels match the definition\n", halves, levels);
    return 0;
}
