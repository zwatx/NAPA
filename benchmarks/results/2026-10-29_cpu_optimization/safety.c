#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "napa_core.h"

int main(void) {
    const char *families[] = {"tag36h11", "tagStandard41h12", "tag16h5"};
    const int shapes[][2] = {{16,16}, {17,31}, {65,129}, {143,127}, {144,145},
                            {641,481}, {1281,721}, {1921,1081}};
    const int minimums[] = {8,22,32,171};
    uint32_t rng = 193;
    int calls = 0;
    for (int f = 0; f < 3; f++) {
        napa_detector *det = napa_create_family(families[f]);
        if (!det) return 1;
        for (int shape = 0; shape < 8; shape++) {
            int w = shapes[shape][0], h = shapes[shape][1], stride = w + 17;
            size_t bytes = (size_t)stride * h;
            uint8_t *image = malloc(bytes), *before = malloc(bytes);
            if (!image || !before) return 2;
            for (int pattern = 0; pattern < 4; pattern++) {
                memset(image, 93, bytes);
                for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
                    rng = rng * 1664525u + 1013904223u;
                    image[(size_t)y * stride + x] = pattern == 0 ? 127 : pattern == 1 ?
                        (uint8_t)(((x / 5 + y / 7) & 1) * 255) : pattern == 2 ?
                        (uint8_t)(rng >> 24) : (uint8_t)(100 + (rng >> 27));
                }
                memcpy(before, image, bytes);
                for (int m = 0; m < 4; m++) {
                    napa_params p = {minimums[m], 3, 20, 22, 25};
                    napa_detection out[64];
                    int n = napa_detect(det, image, w, h, stride, &p, out, 64);
                    if (n < 0 || n > 64 || memcmp(before, image, bytes)) return 3;
                    calls++;
                }
            }
            free(before);
            free(image);
        }
        napa_destroy(det);
    }
    printf("%d native calls passed across odd/padded shapes, sample strides and three families\n", calls);
    return 0;
}
