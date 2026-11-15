/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Check ordinary labelling's row-pair runs against an independent dark mask and expand them to
 * single-row runs in raster order, each with the single-row index of its blob's root. Include after
 * napa_core.c. */
#include <stdio.h>

/* mask: w x h, 1 where dark. Returns the single-row run count, or -1 (with a message) when the
 * row-pair runs do not cover the dark samples exactly, say the wrong rows, or put a blob's root
 * anywhere but the run holding its first dark sample in raster order. rows and roots hold at least
 * one entry per dark sample. */
static int expand_row_pairs(napa_detector *d, const uint8_t *mask, int w, int h, int n, napa_run *rows,
                            int32_t *roots, const char *name) {
    int pixels = w * h, count = 0, last_root = -1;
    int *owner = (int *)malloc((size_t)pixels * sizeof(int)), *first = (int *)malloc((size_t)(n + 1) * sizeof(int));
    uint8_t *single = (uint8_t *)calloc((size_t)h + 1, 1); /* pairs kept as single rows, by top row */
    if (!owner || !first || !single) exit(2);
    for (int p = 0; p < pixels; p++) owner[p] = -1;
    for (int i = 0; i < n; i++) if (d->runs[i].y & 1) single[d->runs[i].y - 1] = 1;
    for (int i = 0; i < n; i++) {
        const napa_run *r = &d->runs[i];
        int prev_y = i ? d->runs[i - 1].y & ~1 : 0;
        if (r->x0 > r->x1 || r->x1 >= w || r->rows < 1 || r->rows > 3 || (r->rows & 2 && (r->y & 1)) ||
            (r->y & ~1) < prev_y || (r->y & 1 ? r->rows != 1 : r->y + (r->rows >> 1) >= h)) {
            fprintf(stderr, "%s: malformed row-pair run %d\n", name, i);
            return -1;
        }
        for (int k = 0; k < 2; k++) {
            int y = r->y + k, any = 0;
            if (!(r->rows >> k & 1)) {
                /* A union run holds every dark sample of both rows in its columns. */
                if ((r->y & 1) || single[r->y]) continue;
                for (int x = r->x0; y < h && x <= r->x1; x++) any |= mask[y * w + x];
                if (any) { fprintf(stderr, "%s: run %d misses dark row %d\n", name, i, y); return -1; }
                continue;
            }
            for (int x = r->x0; x <= r->x1; x++) {
                if (!mask[y * w + x]) continue;
                if (owner[y * w + x] >= 0) { fprintf(stderr, "%s: sample %d,%d in two runs\n", name, x, y); return -1; }
                owner[y * w + x] = i, any = 1;
            }
            if (!any) { fprintf(stderr, "%s: run %d has no dark sample in row %d\n", name, i, y); return -1; }
        }
        /* Its rows are dark in every column of the run and in neither column beside it. */
        for (int x = r->x0 - 1; x <= r->x1 + 1; x++) {
            int dark = 0;
            for (int k = 0; k < 2; k++)
                if (r->rows >> k & 1 && x >= 0 && x < w) dark |= mask[(r->y + k) * w + x];
            if (dark != (x >= r->x0 && x <= r->x1)) { fprintf(stderr, "%s: run %d is not one piece\n", name, i); return -1; }
        }
    }
    for (int i = 0; i < n; i++) first[i] = -1;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        if (!mask[y * w + x]) continue;
        int item = owner[y * w + x];
        if (item < 0) { fprintf(stderr, "%s: dark sample %d,%d in no run\n", name, x, y); return -1; }
        rows[count].x0 = (uint16_t)x, rows[count].y = (uint16_t)y;
        do {
            if (owner[y * w + x] != item) { fprintf(stderr, "%s: single run split across runs\n", name); return -1; }
            x++;
        } while (x < w && mask[y * w + x]);
        rows[count].x1 = (uint16_t)(x - 1);
        int root = find_root(d->parent, item);
        if (root > item) { fprintf(stderr, "%s: root after its run\n", name); return -1; }
        if (first[root] < 0) {
            /* The blob's first dark sample in raster order: its root must hold it, and roots follow
             * that order, as blob numbers do. */
            if (root != item || root < last_root) {
                fprintf(stderr, "%s: root %d is not the first run of its blob in order\n", name, root);
                return -1;
            }
            first[root] = count, last_root = root;
        }
        roots[count++] = first[root];
    }
    free(owner);
    free(first);
    free(single);
    return count;
}
