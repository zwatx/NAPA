/* Exactness and memory-safety fuzz: baseline and candidate cores linked side by side (symbols renamed
 * base_*), run on random crops of corpus frames with odd sizes and padded strides. Each test frame is
 * malloc'd to exactly (h - 1) * stride + w bytes, so AddressSanitizer catches any read past its end. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "napa_core.h"

/* Every field of two detection lists equal, bit for bit (struct padding is never compared). */
static int same_detections(const napa_detection *a, const napa_detection *b, int n) {
    for (int i = 0; i < n; i++)
        if (a[i].id != b[i].id || a[i].hamming != b[i].hamming ||
            memcmp(&a[i].decision_margin, &b[i].decision_margin, sizeof(float)) ||
            memcmp(a[i].corners, b[i].corners, sizeof a[i].corners))
            return 0;
    return 1;
}

size_t ZSTD_decompress(void *dst, size_t dstCapacity, const void *src, size_t compressedSize);
unsigned ZSTD_isError(size_t code);
napa_detector *base_napa_create_family(const char *family);
void base_napa_destroy(napa_detector *det);
int base_napa_detect(napa_detector *det, const uint8_t *img, int width, int height, int stride, const napa_params *params,
                     napa_detection *out, int max_out);

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(uint32_t n) {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return (uint32_t)(rng % n);
}

int main(int argc, char **argv) {
    int every = argc > 2 ? atoi(argv[2]) : 20, variants = argc > 3 ? atoi(argv[3]) : 4;
    FILE *fh = fopen(argv[1], "rb");
    const char *fams[] = {"tag36h11", "tagStandard41h12", "tag16h5"};
    const int mtps[] = {8, 12, 16, 20, 24, 32, 48};
    napa_detector *A[3], *B[3];
    for (int f = 0; f < 3; f++) A[f] = base_napa_create_family(fams[f]), B[f] = napa_create_family(fams[f]);
    static napa_detection oa[512], ob[512];
    long calls = 0, mismatches = 0, found = 0;
    uint8_t *z = NULL;
    size_t zcap = 0;
    for (int index = 0;; index++) {
        uint16_t nl;
        if (fread(&nl, 2, 1, fh) != 1) break;
        char name[256];
        uint32_t hdr[3], zs;
        if (fread(name, 1, nl, fh) != nl || fread(hdr, 4, 3, fh) != 3) return 2;
        name[nl] = 0;
        fseek(fh, 4L * hdr[2], SEEK_CUR);
        if (fread(&zs, 4, 1, fh) != 1) return 2;
        if (zs > zcap) z = realloc(z, zcap = zs);
        if (fread(z, 1, zs, fh) != zs) return 2;
        if (index % every) continue;
        int W = (int)hdr[0], H = (int)hdr[1];
        uint8_t *full = malloc((size_t)W * H);
        if (ZSTD_isError(ZSTD_decompress(full, (size_t)W * H, z, zs))) return 3;
        for (int v = 0; v < variants; v++) {
            /* Odd crops of every size class, down to the 16-pixel minimum, and strides padded up to 70. */
            int w = v == 0 ? W : 16 + (int)rnd((uint32_t)W - 15), h = v == 0 ? H : 16 + (int)rnd((uint32_t)H - 15);
            if (v == 1) w = 16 + (int)rnd(80), h = 16 + (int)rnd(80);
            int x0 = (int)rnd((uint32_t)(W - w + 1)), y0 = (int)rnd((uint32_t)(H - h + 1));
            int stride = w + (v & 1 ? (int)rnd(71) : 0);
            size_t bytes = (size_t)(h - 1) * stride + w;
            uint8_t *img = malloc(bytes);
            for (int y = 0; y < h; y++) {
                memcpy(img + (size_t)y * stride, full + (size_t)(y0 + y) * W + x0, (size_t)w);
                if (y < h - 1) memset(img + (size_t)y * stride + w, (int)rnd(256), (size_t)(stride - w));
            }
            for (int f = 0; f < 3; f++) {
                for (int m = 0; m < 7; m++) {
                    napa_params p = {mtps[m], f == 2 ? 1 : 3, 20, 22, 25};
                    int na = base_napa_detect(A[f], img, w, h, stride, &p, oa, 512);
                    int nb = napa_detect(B[f], img, w, h, stride, &p, ob, 512);
                    calls++;
                    found += nb > 0 ? nb : 0;
                    if (na != nb || (na > 0 && !same_detections(oa, ob, na))) {
                        if (mismatches < 10)
                            fprintf(stderr, "MISMATCH %s crop %dx%d+%d+%d stride %d %s mtp %d: %d vs %d\n", name, w, h, x0,
                                    y0, stride, fams[f], mtps[m], na, nb);
                        mismatches++;
                    }
                }
            }
            free(img);
        }
        free(full);
    }
    for (int f = 0; f < 3; f++) base_napa_destroy(A[f]), napa_destroy(B[f]);
    free(z);
    fclose(fh);
    printf("calls %ld, detections %ld, mismatches %ld\n", calls, found, mismatches);
    return mismatches != 0;
}
