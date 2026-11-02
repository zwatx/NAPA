/* Exactness over a streamed corpus (stream_corpus.py on stdin): baseline and candidate cores (dlopen) on
 * every frame at min_tag_px 32 and 16; every detection field compared bit for bit; IDs scored against the truth by
 * group (main, stress, range): truth IDs found and detections with an ID not in the frame. */
#include <dlfcn.h>
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
typedef napa_detector *(*create_fn)(void);
typedef int (*detect_fn)(napa_detector *, const uint8_t *, int, int, int, const napa_params *, napa_detection *, int);
int main(int argc, char **argv) {
    void *h[2];
    create_fn cr[2];
    detect_fn dt[2];
    for (int i = 0; i < 2; i++) {
        h[i] = dlopen(argv[1 + i], RTLD_NOW | RTLD_LOCAL);
        if (!h[i]) return fprintf(stderr, "%s\n", dlerror()), 1;
        cr[i] = (create_fn)dlsym(h[i], "napa_create"), dt[i] = (detect_fn)dlsym(h[i], "napa_detect");
    }
    napa_detector *d[2][2];
    for (int i = 0; i < 2; i++)
        for (int m = 0; m < 2; m++) d[i][m] = cr[i]();
    static napa_detection out[2][1024];
    long frames = 0, mism[2] = {0, 0}, tags[3] = {0}, found[2][2][3] = {{{0}}}, wrong[2][2][3] = {{{0}}};
    uint8_t *px = NULL;
    size_t cap = 0;
    uint16_t nl;
    while (fread(&nl, 2, 1, stdin) == 1) {
        char name[256];
        uint32_t hd[3];
        if (fread(name, 1, nl, stdin) != nl || fread(hd, 4, 3, stdin) != 3) return 2;
        name[nl] = 0;
        int32_t ids[256];
        if (fread(ids, 4, hd[2], stdin) != hd[2]) return 2;
        size_t n = (size_t)hd[0] * hd[1];
        if (n > cap) px = realloc(px, cap = n);
        if (fread(px, 1, n, stdin) != n) return 2;
        int g = !strncmp(name, "range", 5) ? 2 : !strncmp(name, "stress", 6) ? 1 : 0;
        tags[g] += hd[2];
        for (int m = 0; m < 2; m++) {
            napa_params p = {m ? 16 : 32, 3, 20, 22, 25};
            int c[2];
            for (int i = 0; i < 2; i++) {
                c[i] = dt[i](d[i][m], px, (int)hd[0], (int)hd[1], (int)hd[0], &p, out[i], 1024);
                int used[256] = {0};
                for (int k = 0; k < c[i]; k++) {
                    int hit = 0;
                    for (uint32_t t = 0; t < hd[2] && !hit; t++)
                        if (!used[t] && ids[t] == out[i][k].id) used[t] = hit = 1;
                    found[i][m][g] += hit, wrong[i][m][g] += !hit;
                }
            }
            if (c[0] != c[1] || (c[0] > 0 && !same_detections(out[0], out[1], c[0]))) {
                if (mism[m] < 10) fprintf(stderr, "MISMATCH mtp%d %s: %d vs %d\n", m ? 16 : 32, name, c[0], c[1]);
                mism[m]++;
            }
        }
        if (++frames % 500 == 0) fprintf(stderr, "%ld frames\n", frames);
    }
    const char *gn[3] = {"main", "stress", "range"};
    printf("frames %ld\n", frames);
    for (int m = 0; m < 2; m++) {
        printf("min_tag_px=%d: mismatches %ld\n", m ? 16 : 32, mism[m]);
        for (int g = 0; g < 3; g++)
            printf("  %-6s tags %5ld | baseline found %5ld wrong %ld | candidate found %5ld wrong %ld\n", gn[g], tags[g],
                   found[0][m][g], wrong[0][m][g], found[1][m][g], wrong[1][m][g]);
    }
    return mism[0] || mism[1];
}
