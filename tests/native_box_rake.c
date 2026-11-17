/* box_rake against its definition: the rounded mean of each block, clipped to the image. */
#include "../src/napa_vision/_native/napa_core.c"

#include <stdio.h>

static uint32_t seed = 2026;
static uint32_t rnd(void) { return seed = seed * 1664525u + 1013904223u, seed >> 8; }

static int expected(const uint8_t *img, int w, int h, int stride, int s, int kernel, int x, int y) {
    int half = (kernel - 1) / 2, r0 = y * s - half, r1 = r0 + kernel, c0 = x * s - half, c1 = c0 + kernel, sum = 0;
    if (r0 < 0) r0 = 0;
    if (r1 > h) r1 = h;
    if (c0 < 0) c0 = 0;
    if (c1 > w) c1 = w;
    for (int r = r0; r < r1; r++)
        for (int c = c0; c < c1; c++) sum += img[(size_t)r * stride + c];
    int count = (c1 - c0) * (r1 - r0);
    return (sum + count / 2) / count;
}

int main(void) {
    /* Real blocks for every representable sum, including ties; no reciprocal in the oracle. */
    for (int k = 2; k <= 16; k++) {
        for (int clipped = 0; clipped <= 1; clipped++) {
            uint8_t img[32 * 32] = {0}, level[4];
            uint16_t rows[64];
            int half = (k - 1) / 2, start = k - half, cols = k - clipped;
            int w = clipped ? start + cols : 2 * k, h = 2 * k, count = cols * k;
            for (int sum = 0; sum <= 255 * count; sum++) {
                if (sum) {
                    int cell = (sum - 1) / 255;
                    img[(start + cell / cols) * w + start + cell % cols]++;
                }
                box_rake(level, 2, 2, img, w, h, w, k, k, rows, rows + w);
                if (level[3] != (sum + count / 2) / count) return 1;
                for (int y = 0; y < 2; y++)
                    for (int x = 0; x < 2; x++)
                        if (level[y * 2 + x] != expected(img, w, h, w, k, k, x, y)) return 1;
            }
        }
    }
    for (int trial = 0; trial < 6000; trial++) {
        int w = 16 + (int)(rnd() % 200), h = 16 + (int)(rnd() % 80), stride = w + (rnd() % 3 ? 0 : (int)(rnd() % 9));
        int s = 2 + (int)(rnd() % 15), kernel = s == 2 && rnd() % 2 ? 3 : s, mode = (int)(rnd() % 3);
        int lw = (w + s - 1) / s, lh = (h + s - 1) / s;
        /* The last row ends at the allocation, so a sanitizer or a crash shows any read past the image. */
        size_t bytes = (size_t)stride * (h - 1) + w;
        uint8_t *img = malloc(bytes), *level = malloc((size_t)lw * lh);
        uint16_t *rows = malloc(2 * sizeof(uint16_t) * w);
        for (size_t i = 0; i < bytes; i++) img[i] = mode == 0 ? (uint8_t)rnd() : mode == 1 ? 255 : (rnd() & 1 ? 255 : 0);
        box_rake(level, lw, lh, img, w, h, stride, s, kernel, rows, rows + w);
        for (int y = 0; y < lh; y++)
            for (int x = 0; x < lw; x++)
                if (level[(size_t)y * lw + x] != expected(img, w, h, stride, s, kernel, x, y)) {
                    printf("box_rake differs: w %d h %d stride %d s %d kernel %d at %d,%d\n", w, h, stride, s, kernel, x, y);
                    return 1;
                }
        free(img), free(level), free(rows);
    }
    printf("6000 box_rake cases match the block-mean definition\n");
    return 0;
}
