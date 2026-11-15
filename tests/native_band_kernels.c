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

int main(void) {
    unsigned long bits = check_dark_bits(), tiles = check_tile_extrema();
    printf("%lu exact dark-bit cases, %lu exact tile-extrema rows passed\n", bits, tiles);
    return 0;
}
