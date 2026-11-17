/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Fuzz target for the C core: every public detection entry point on frames built from the input.
 *
 * Built with clang and libFuzzer (-DNAPA_LIBFUZZER), or as a self-contained random driver with any C compiler:
 * `./napa_fuzz 2000` runs 2000 random inputs, `./napa_fuzz FILE...` replays inputs such as libFuzzer crash
 * reproducers. CI's native-sanitizers job (.github/workflows/ci.yml) has the exact build flags.
 *
 * The first bytes choose the frame size, stride, family, parameters, entry point and how many real tag
 * codes to draw; the rest is texture. Frames are allocated at exactly stride * height bytes, so a sanitizer
 * reports any read past the caller's buffer. Every result is checked against the API contract. */
#include "../../src/napa_vision/_native/napa_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_OUT 80

static long detections_seen; /* reported by the random driver: the drawn tags must reach decoding */
static long threaded_comparisons, nonempty_comparisons;

static uint8_t byte_at(const uint8_t *data, size_t size, size_t i) { return size ? data[i % size] : (uint8_t)(i * 37); }

/* Draw a family code as a tag: border, payload and quiet ring, `cell` pixels per cell, with a per-row shear. */
static void draw_tag(uint8_t *img, int w, int h, int stride, const napa_family_layout *L, uint64_t code, int x0, int y0,
                     int cell, int shear, uint8_t dark, uint8_t light) {
    const int total = L->total_width, pad = (total - L->width) / 2;
    for (int cy = 0; cy < total; cy++)
        for (int cx = 0; cx < total; cx++) {
            int sx = cx - pad, sy = cy - pad; /* cells from the detected square's top-left */
            int white = sx < 0 || sy < 0 || sx >= L->width || sy >= L->width; /* quiet ring outside the square */
            if (!white) {
                int border = sx == 0 || sy == 0 || sx == L->width - 1 || sy == L->width - 1;
                white = 0;
                if (!border)
                    for (int k = 0; k < L->nbits; k++)
                        if (L->x[k] == sx && L->y[k] == sy) white = (int)((code >> (L->nbits - 1 - k)) & 1);
            }
            if (L->reversed) white = !white;
            for (int py = 0; py < cell; py++)
                for (int px = 0; px < cell; px++) {
                    int y = y0 + cy * cell + py, x = x0 + cx * cell + px + (shear * (cy * cell + py)) / 16;
                    if (x >= 0 && y >= 0 && x < w && y < h) img[(size_t)y * stride + x] = white ? light : dark;
                }
        }
}

static void check(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "napa_fuzz: contract violated: %s\n", what);
        abort();
    }
}

static void configure(napa_detector *det, const char *family, const uint8_t *data, size_t size,
                      int w, int h, int shape, int threads) {
    check(napa_set_tag_shape(det, shape) == 0, "tag shape");
    if (data[17] & 1) {
        int status = napa_set_camera_gamma(det, 1.0 + (data[18] % 31) / 10.0);
        check(status == (strcmp(family, "tag36h11") == 0 ? 0 : -1), "gamma family contract");
    }
    check(napa_set_regrow_small_tags(det, (data[20] >> 3) & 1) == 0, "regrow small tags");
    check(napa_set_threads(det, threads) == 0, "threads");
    if (data[23] % 3) { /* a lens model, mild to extreme, centred anywhere */
        const double f = 0.4 * w + 4.0 * (byte_at(data, size, 24) % 200);
        const double intrinsics[5] = {f, f * (0.9 + (byte_at(data, size, 25) % 21) / 100.0),
                                      (byte_at(data, size, 26) / 255.0) * w, (byte_at(data, size, 27) / 255.0) * h, 0};
        double k[8];
        for (int i = 0; i < 8; i++) k[i] = ((int)byte_at(data, size, 28 + i) - 128) / (i < 2 ? 160.0 : 900.0);
        const int fisheye = data[23] % 3 == 2;
        int status = napa_set_lens(det, fisheye ? NAPA_LENS_FISHEYE : NAPA_LENS_BROWN_CONRADY,
                                   intrinsics, k, fisheye ? 4 : 8);
        check(status == 0 || status == -1, "lens status");
    }
}

static int detect(napa_detector *det, const uint8_t *img, int w, int h, int stride, const napa_params *p,
                  napa_detection *out, int max_out, int mode, const uint8_t *data, size_t size) {
    switch (mode) {
    case 0: return napa_detect(det, img, w, h, stride, p, out, max_out);
    case 1: return napa_detect_shadows(det, img, w, h, stride, p, out, max_out);
    case 2: return napa_detect_shadows_configured(det, img, w, h, stride, p, out, max_out,
                                                data[19] % 2 ? 0 : p->min_tag_px + data[19] % 64);
    case 3: return napa_detect_soft(det, img, w, h, stride, p, out, max_out, data[19] & 1, 0);
    case 4: return napa_detect_options(det, img, w, h, stride, p, out, max_out, data[19] & 1, 0, data[20] % 4);
    default: {
        double corners[8], bias = 0, edge = 0;
        for (int k = 0; k < 8; k++) corners[k] = (double)byte_at(data, size, 21 + k) * (k & 1 ? h : w) / 255.0;
        int status = napa_border_bias_cells(det, img, w, h, stride, p, corners, &bias, &edge);
        check(status >= -1 && status <= 1, "border bias status");
        return 0;
    }
    }
}

static void validate(int n, const napa_detection *out, int max_out, int ncodes, int mode) {
    check(n >= -2 && n <= max_out, "detection count within -2..max_out");
    if (n == -1 && mode != 3 && mode != 4) check(0, "valid parameters rejected");
    for (int i = 0; i < n; i++) {
        check(out[i].id >= 0 && out[i].id < ncodes, "id in family");
        check(out[i].hamming == NAPA_SOFT_DECODED || (out[i].hamming >= 0 && out[i].hamming <= 3), "hamming range");
        check(isfinite(out[i].decision_margin), "finite margin");
        for (int k = 0; k < 8; k++) check(isfinite(out[i].corners[k]), "finite corners");
    }
}

static void compare(int n, const napa_detection *out, int nr, const napa_detection *reference, int threads) {
    /* Independent allocations may fail under different memory pressure. Successful calls must agree. */
    if (n == -2 || nr == -2) return;
    check(n == nr, "threaded count matches single-thread count");
    for (int i = 0; i < n; i++) {
        check(out[i].id == reference[i].id, "threaded ID order");
        check(out[i].hamming == reference[i].hamming, "threaded hamming");
        check(memcmp(&out[i].decision_margin, &reference[i].decision_margin, sizeof out[i].decision_margin) == 0,
              "threaded margin");
        for (int k = 0; k < 8; k++)
            check(memcmp(&out[i].corners[k], &reference[i].corners[k], sizeof out[i].corners[k]) == 0, "threaded corner");
    }
    if (threads > 1 && n >= 0) {
        threaded_comparisons++;
        if (n > 0) nonempty_comparisons++;
    }
}

static int fuzz_one(const uint8_t *data, size_t size) {
    if (size < 24) return 0;
    const int w = 16 + (data[0] | (data[1] & 3) << 8) % 640, h = 16 + (data[2] | (data[3] & 1) << 8) % 400;
    const int stride = w + data[4] % 9;
    const int family = data[5] % napa_family_count();
    napa_params p;
    p.min_tag_px = 8 + data[6] % 72;
    p.max_hamming = data[7] % 4;
    p.min_contrast = 1 + data[8] % 120;
    p.min_border_cells = data[9] % 29;
    p.sharpening_x100 = data[10] % 101;
    p.split_touching = data[11] & 1;
    const int mode = data[12] % 6, ntags = data[13] % 7, max_out = data[14] % MAX_OUT;
    const int shape = data[15] % 3, texture = data[16] % 4;

    uint8_t *img = (uint8_t *)malloc((size_t)stride * h); /* exact size: no padding to hide overreads */
    if (img == NULL) return 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < stride; x++) {
            size_t i = 24 + (size_t)y * stride + x;
            uint8_t v = byte_at(data, size, i);
            img[(size_t)y * stride + x] = texture == 0 ? v : texture == 1 ? (uint8_t)(200 + (v & 31)) :
                                          texture == 2 ? (uint8_t)(((x / 7 + y / 5) & 1) ? 230 : 25) : (uint8_t)(v & 0xf0);
        }
    const char *name = napa_family_name(family);
    napa_family_layout L;
    check(napa_family_info(name, &L) == 0, "family info");
    for (int t = 0; t < ntags; t++) {
        uint64_t code = 0;
        int id = (byte_at(data, size, 17 + t) | byte_at(data, size, 18 + t) << 8) % L.ncodes;
        check(napa_family_code(name, id, &code) == 0, "family code");
        int cell = 1 + byte_at(data, size, 19 + t) % 9;
        int x0 = byte_at(data, size, 20 + 3 * t) % w - cell, y0 = byte_at(data, size, 21 + 3 * t) % h - cell;
        int shear = (int)(byte_at(data, size, 22 + t) % 17) - 8;
        uint8_t dark = byte_at(data, size, 23 + t) & 63, light = (uint8_t)(255 - (byte_at(data, size, 24 + t) & 63));
        draw_tag(img, w, h, stride, &L, code, x0, y0, cell, shear, dark, light);
    }

    napa_detector *det = napa_create_family(name), *reference = napa_create_family(name);
    if (det == NULL || reference == NULL) {
        napa_destroy(det);
        napa_destroy(reference);
        free(img);
        return 0;
    }
    const int threads = data[21] & 1 ? 1 + data[22] % 6 : 1;
    configure(det, name, data, size, w, h, shape, threads);
    configure(reference, name, data, size, w, h, shape, 1);
    napa_detection out[MAX_OUT], expected[MAX_OUT];
    for (int round = 0; round < 2; round++) { /* a second call reuses the detector's grown buffers */
        int n = detect(det, img, w, h, stride, &p, out, max_out, mode, data, size);
        int nr = detect(reference, img, w, h, stride, &p, expected, max_out, mode, data, size);
        validate(n, out, max_out, L.ncodes, mode);
        validate(nr, expected, max_out, L.ncodes, mode);
        compare(n, out, nr, expected, threads);
        if (n > 0) detections_seen += n;
        if (round == 0 && w > 32 && h > 32) { /* then a smaller frame: buffers must not assume the larger size */
            napa_detection small[4], expected_small[4];
            int m = napa_detect(det, img, w / 2, h / 2, stride, &p, small, 4);
            int mr = napa_detect(reference, img, w / 2, h / 2, stride, &p, expected_small, 4);
            validate(m, small, 4, L.ncodes, 0);
            validate(mr, expected_small, 4, L.ncodes, 0);
            compare(m, small, mr, expected_small, threads);
        }
    }
    if (data[20] & 2) { /* the colour conversion: a BGR copy of the frame must convert back to OpenCV's grey */
        const int pixel = 3 + (data[20] & 4 ? 1 : 0), cstride = w * pixel + data[4] % 5;
        uint8_t *colour = (uint8_t *)malloc((size_t)cstride * h), *grey = (uint8_t *)malloc((size_t)w * h);
        if (colour != NULL && grey != NULL) {
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int c = 0; c < pixel; c++) colour[(size_t)y * cstride + x * pixel + c] = byte_at(data, size, 24 + (size_t)(y * w + x) * 3 + c);
            check(napa_gray_from_pixels(det, colour, w, h, cstride, pixel, 1, grey, w) == 0, "colour conversion");
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    const uint8_t *q = colour + (size_t)y * cstride + x * pixel;
                    check(grey[(size_t)y * w + x] == (uint8_t)((q[0] * 3735u + q[1] * 19235u + q[2] * 9798u + 16384u) >> 15), "grey value");
                }
            check(napa_gray_from_pixels(det, colour, w, h, cstride, pixel, 0, grey, w) == 0, "plane copy");
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) check(grey[(size_t)y * w + x] == colour[(size_t)y * cstride + x * pixel], "plane value");
        }
        free(colour);
        free(grey);
    }
    napa_destroy(det);
    napa_destroy(reference);
    free(img);
    return 0;
}

#ifdef NAPA_LIBFUZZER
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) { return fuzz_one(data, size); }
#else
static uint32_t rng = 20261112u;
static uint8_t next_byte(void) {
    rng = rng * 1664525u + 1013904223u;
    return (uint8_t)(rng >> 24);
}

int main(int argc, char **argv) {
    if (argc > 1 && argv[1][0] >= '0' && argv[1][0] <= '9') {
        long iterations = strtol(argv[1], NULL, 10);
        for (long it = 0; it < iterations; it++) {
            /* Every fourth input is a short one (24-39 bytes). Each is allocated at its exact size, so AddressSanitizer
             * catches a read past the input as libFuzzer's exactly sized inputs would. */
            size_t size = it % 4 == 0 ? 24 + (size_t)(it / 4 % 16) : 24 + (size_t)next_byte() * 16;
            uint8_t *buf = (uint8_t *)malloc(size);
            if (buf == NULL) return 2;
            for (size_t i = 0; i < size; i++) buf[i] = next_byte();
            fuzz_one(buf, size);
            free(buf);
        }
        printf("%ld random fuzz inputs passed, %ld detections, %ld threaded comparisons (%ld non-empty)\n",
               iterations, detections_seen, threaded_comparisons, nonempty_comparisons);
        return 0;
    }
    for (int a = 1; a < argc; a++) { /* replay files, e.g. libFuzzer crash reproducers */
        FILE *f = fopen(argv[a], "rb");
        if (f == NULL) return 2;
        static uint8_t buf[1 << 20];
        size_t size = fread(buf, 1, sizeof buf, f);
        fclose(f);
        fuzz_one(buf, size);
    }
    printf("%d fuzz input files passed\n", argc - 1);
    return 0;
}
#endif
