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
    if (count == kernel * kernel) {
        uint32_t m = (65536u + (uint32_t)(kernel * kernel) / 2) / (uint32_t)(kernel * kernel);
        return (int)(((uint32_t)sum * m + 32768u) >> 16);
    }
    return (sum + count / 2) / count;
}

int main(void) {
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
