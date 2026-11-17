/* Recovery must agree with a fresh detector after frame dimensions or sampling steps change. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/napa_vision/_native/napa_core.h"

static void compare(napa_detector *reused, uint8_t *image, int side, int minimum, int shadow) {
    napa_detector *fresh = napa_create();
    napa_params params = {minimum, 3, 25, 20, 25, 0};
    napa_detection a[64], b[64];
    assert(fresh);
    int na = napa_detect_shadows_configured(reused, image, side, side, side, &params, a, 64, shadow);
    int nb = napa_detect_shadows_configured(fresh, image, side, side, side, &params, b, 64, shadow);
    assert(na >= 0 && na == nb);
    for (int i = 0; i < na; i++) {
        assert(a[i].id == b[i].id && a[i].hamming == b[i].hamming);
        assert(a[i].decision_margin == b[i].decision_margin);
        assert(memcmp(a[i].corners, b[i].corners, sizeof a[i].corners) == 0);
    }
    napa_destroy(fresh);
}

/* Fixed-width arithmetic keeps these two textured frames identical on every platform. */
static uint32_t next_value(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state >> 1;
}

static void shadow_frame(uint8_t *image, uint32_t seed) {
    const int side = 320;
    napa_family_layout layout;
    assert(napa_family_info("tag36h11", &layout) == 0);
    int gradient = next_value(&seed) % 3;
    for (int y = 0; y < side; y++) for (int x = 0; x < side; x++) {
        int shade = gradient == 1 ? x * 150 / side : gradient == 2 ? y * 150 / side : 0;
        image[y * side + x] = (uint8_t)(200 - shade + (int)(next_value(&seed) % 9) - 4);
    }
    for (int tag = 0; tag < 3; tag++) {
        int cell = 3 + next_value(&seed) % 6, size = 8 * cell;
        int x0 = 4 + next_value(&seed) % (side - size - 8);
        int y0 = 4 + next_value(&seed) % (side - size - 8);
        int light = 120 + next_value(&seed) % 120;
        int dark = next_value(&seed) % (light - 20);
        uint64_t code;
        assert(napa_family_code("tag36h11", next_value(&seed) % 500, &code) == 0);
        for (int y = 0; y < size; y++) for (int x = 0; x < size; x++)
            image[(y0 + y) * side + x0 + x] = (uint8_t)dark;
        for (int bit = 0; bit < layout.nbits; bit++) if ((code >> (layout.nbits - 1 - bit)) & 1) {
            for (int y = 0; y < cell; y++) for (int x = 0; x < cell; x++)
                image[(y0 + layout.y[bit] * cell + y) * side + x0 + layout.x[bit] * cell + x] = (uint8_t)light;
        }
    }
    int start = next_value(&seed) % side;
    for (int y = 0; y < side; y++) for (int x = start; x < side && x < start + side / 3; x++)
        image[y * side + x] = (uint8_t)(image[y * side + x] * (30 + next_value(&seed) % 5) / 100);
}

int main(void) {
    napa_detector *reused = napa_create();
    uint8_t *image = calloc(320 * 320, 1);
    assert(reused && image);
    /* Step three was cached in a smaller allocation, but the next ordinary pass skips it. */
    compare(reused, image, 64, 32, 0);
    compare(reused, image, 256, 22, 32);
    /* A shaded frame with different pixels must not reuse the previous frame's extrema. */
    shadow_frame(image, 1);
    compare(reused, image, 320, 32, 0);
    shadow_frame(image, 2);
    compare(reused, image, 320, 22, 32);
    uint64_t code;
    napa_family_layout layout;
    assert(napa_family_info("tag36h11", &layout) == 0);
    assert(napa_family_code("tag36h11", 7, &code) == 0);
    memset(image, 255, 256 * 256);
    for (int y = 80; y < 176; y++) for (int x = 80; x < 176; x++) image[y * 256 + x] = 0;
    for (int bit = 0; bit < layout.nbits; bit++) if ((code >> (layout.nbits - 1 - bit)) & 1) {
        for (int y = 0; y < 12; y++) for (int x = 0; x < 12; x++)
            image[(80 + layout.y[bit] * 12 + y) * 256 + 80 + layout.x[bit] * 12 + x] = 255;
    }
    napa_params params = {16, 3, 25, 20, 25, 0};
    napa_detection out[64];
    assert(napa_detect(reused, image, 256, 256, 256, &params, out, 64) > 0);
    for (int round = 0; round < 3; round++) {
        compare(reused, image, 256, 32, 0);
        compare(reused, image, 256, 22, 32);
        compare(reused, image, 256, 16, 64);
        compare(reused, image, 64, 32, 0);
    }
    napa_destroy(reused);
    free(image);
    puts("shadow cache frame transitions passed");
}
