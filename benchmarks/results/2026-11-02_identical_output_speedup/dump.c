/* Static driver for one napa_core build (ARM under qemu, or any no-SIMD build): run every frame of a raw
 * subset at min_tag_px 32 and 16 and write each call's count and detections to stdout, for cmp. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "napa_core.h"
int main(int argc, char **argv) {
    FILE *fh = fopen(argv[1], "rb");
    if (!fh) return 1;
    static napa_detection out[1024];
    napa_detector *det = napa_create();
    int frames = 0;
    uint16_t nl;
    while (fread(&nl, 2, 1, fh) == 1) {
        char name[256];
        uint32_t wh[2];
        if (fread(name, 1, nl, fh) != nl || fread(wh, 4, 2, fh) != 2) return 2;
        uint8_t *px = malloc((size_t)wh[0] * wh[1]);
        if (fread(px, 1, (size_t)wh[0] * wh[1], fh) != (size_t)wh[0] * wh[1]) return 3;
        for (int m = 0; m < 2; m++) {
            napa_params p = {m ? 16 : 32, 3, 20, 22, 25};
            int n = napa_detect(det, px, (int)wh[0], (int)wh[1], (int)wh[0], &p, out, 1024);
            fwrite(&n, 4, 1, stdout);
            if (n > 0) fwrite(out, sizeof(napa_detection), (size_t)n, stdout);
        }
        free(px);
        frames++;
    }
    fprintf(stderr, "frames %d\n", frames);
    napa_destroy(det);
    return 0;
}
