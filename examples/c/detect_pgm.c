/* Napa: Not Another Pixel Algorithm. Copyright (c) 2026 Zachary Wilson.
 * Licensed under the PolyForm Noncommercial License 1.0.0: see LICENSE. Commercial use needs a separate
 * license. Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Detect tag36h11 tags in a binary PGM (P5) image and print one line per tag.
 *
 *   napa_detect_pgm frame.pgm
 *   id 7 hamming 0 margin 201.3 corners 120.52,80.31 220.47,80.12 220.66,180.40 120.71,180.58
 *
 * Corners are in OpenCV's ArUco order (top-left, top-right, bottom-right, bottom-left of the tag's
 * artwork), with (0, 0) at the centre of the top-left pixel. */
#include <stdio.h>
#include <stdlib.h>

#include "napa_core.h"

static unsigned char *read_pgm(const char *path, int *width, int *height) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    int maxval = 0;
    unsigned char *pixels = NULL;
    if (fscanf(f, "P5 %d %d %d", width, height, &maxval) == 3 && maxval == 255 && *width > 0 && *height > 0 &&
        fgetc(f) != EOF) {
        size_t size = (size_t)*width * (size_t)*height;
        pixels = (unsigned char *)malloc(size);
        if (pixels != NULL && fread(pixels, 1, size, f) != size) {
            free(pixels);
            pixels = NULL;
        }
    }
    fclose(f);
    return pixels;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s image.pgm\n", argv[0]);
        return 2;
    }
    int width = 0, height = 0;
    unsigned char *image = read_pgm(argv[1], &width, &height);
    if (image == NULL) {
        fprintf(stderr, "%s: not an 8-bit binary PGM\n", argv[1]);
        return 2;
    }
    /* The defaults NapaDetector uses in Python. */
    napa_params params = {.min_tag_px = 32, .max_hamming = 3, .min_contrast = 20, .min_border_cells = 22,
                          .sharpening_x100 = 25};
    napa_detection found[64];
    napa_detector *detector = napa_create();
    int count = detector ? napa_detect(detector, image, width, height, width, &params, found, 64) : -2;
    for (int i = 0; i < count; i++) {
        const double *c = found[i].corners;
        printf("id %d hamming %d margin %.1f corners %.2f,%.2f %.2f,%.2f %.2f,%.2f %.2f,%.2f\n", found[i].id,
               found[i].hamming, found[i].decision_margin, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
    }
    napa_destroy(detector);
    free(image);
    if (count < 0) {
        fprintf(stderr, "napa_detect failed with status %d\n", count);
        return 1;
    }
    return 0;
}
