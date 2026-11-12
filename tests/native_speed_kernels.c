/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Standalone exact-oracle tests for private speed kernels. The legacy threshold/median
 * below deliberately retain the pre-optimization algorithms as independent oracles.
 * Run through test_native_speed_kernels.py; no packaged library is modified. */
#include "../src/napa_vision/_native/napa_core.c"
#include <stdio.h>
#include <assert.h>
static int threshold_legacy(napa_detector *det, int lw, int lh, int factor, const int split, int shadow) {
    const uint8_t *L = det->level;
    napa_run *R = det->runs;
    int32_t *P = det->parent;
    int n = 0, prev_y = -2, prev_start = 0, prev_end = 0;
    det->diag_n = det->neck_n = 0;
    int tw = (lw + TILE - 1) / TILE;
    int th = (lh + TILE - 1) / TILE;
    uint8_t *vmin = det->row_min, *vmax = det->row_max, *trow = det->row_thr;

    /* Tile min/max: a vertical pass over each band of TILE rows (contiguous, so it vectorises),
     * then a short horizontal pass over the band's TILE-wide groups. */
    for (int ty = 0; ty < th; ty++) {
        int y0 = ty * TILE;
        int y1 = y0 + TILE < lh ? y0 + TILE : lh;
        memcpy(vmin, L + (size_t)y0 * lw, (size_t)lw);
        memcpy(vmax, L + (size_t)y0 * lw, (size_t)lw);
        for (int y = y0 + 1; y < y1; y++) {
            const uint8_t *row = L + (size_t)y * lw;
            for (int x = 0; x < lw; x++) {
                uint8_t v = row[x];
                vmin[x] = v < vmin[x] ? v : vmin[x];
                vmax[x] = v > vmax[x] ? v : vmax[x];
            }
        }
        uint8_t *tmin = det->tile_min + (size_t)ty * tw;
        uint8_t *tmax = det->tile_max + (size_t)ty * tw;
        /* Rows are padded past lw by repeating the last sample, so every tile reads four. */
        for (int x = lw; x < 4 * tw; x++) vmin[x] = vmin[lw - 1], vmax[x] = vmax[lw - 1];
        for (int tx = 0; tx < tw; tx++) {
            const uint8_t *a = vmin + 4 * tx, *b = vmax + 4 * tx;
            uint8_t l01 = a[0] < a[1] ? a[0] : a[1], l23 = a[2] < a[3] ? a[2] : a[3];
            uint8_t h01 = b[0] > b[1] ? b[0] : b[1], h23 = b[2] > b[3] ? b[2] : b[3];
            tmin[tx] = l01 < l23 ? l01 : l23;
            tmax[tx] = h01 > h23 ? h01 : h23;
        }
    }
    /* 3x3 tile neighbourhood, separably and without branches so the loops vectorise: vertically
     * into a row padded by one tile at each end (repeating the edge tile), then horizontally. */
    uint8_t *nlo = det->nb_lo, *nhi = det->nb_hi, *tt = det->nb_t, *fl = det->nb_f;
    /* A tile spans 4 * factor samples of the grid where the floors were measured. */
    int rows = det->floor_rows, cols = det->floor_cols;
    if (det->floor_uniform) memset(fl, det->floors[0], (size_t)tw);
    for (int ty = 0; ty < th; ty++) {
        int ry = ty * TILE * factor / NAPA_REGION;
        const uint8_t *frow = det->floors + (size_t)(ry < rows ? ry : rows - 1) * cols;
        for (int tx = 0; tx < tw && !det->floor_uniform; tx++) {
            int rx = tx * TILE * factor / NAPA_REGION;
            fl[tx] = frow[rx < cols ? rx : cols - 1];
        }
        const uint8_t *lo_up = det->tile_min + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *lo_mid = det->tile_min + (size_t)ty * tw;
        const uint8_t *lo_dn = det->tile_min + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        const uint8_t *hi_up = det->tile_max + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *hi_mid = det->tile_max + (size_t)ty * tw;
        const uint8_t *hi_dn = det->tile_max + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        for (int tx = 0; tx < tw; tx++) {
            uint8_t l = lo_mid[tx], h = hi_mid[tx];
            l = lo_up[tx] < l ? lo_up[tx] : l;
            l = lo_dn[tx] < l ? lo_dn[tx] : l;
            h = hi_up[tx] > h ? hi_up[tx] : h;
            h = hi_dn[tx] > h ? hi_dn[tx] : h;
            nlo[tx + 1] = l;
            nhi[tx + 1] = h;
        }
        nlo[0] = nlo[1], nhi[0] = nhi[1], nlo[tw + 1] = nlo[tw], nhi[tw + 1] = nhi[tw];
        uint8_t any = 0;
        for (int tx = 0; tx < tw; tx++) {
            uint8_t l = nlo[tx + 1], h = nhi[tx + 1];
            l = nlo[tx] < l ? nlo[tx] : l;
            l = nlo[tx + 2] < l ? nlo[tx + 2] : l;
            h = nhi[tx] > h ? nhi[tx] : h;
            h = nhi[tx + 2] > h ? nhi[tx + 2] : h;
            /* Boyer-Moore-style skip: a flat neighbourhood cannot hold a tag edge, so its
             * threshold is 0 and no sample can fall below it. */
            uint8_t t = (uint8_t)(h - l) >= fl[tx] ? (uint8_t)(shadow ? l + ((h - l + 2) >> 2) : (l + h + 1) >> 1) : 0;
            tt[tx] = t;
            any |= t;
        }
        int y0 = ty * TILE;
        int y1 = y0 + TILE < lh ? y0 + TILE : lh;
        if (!any) continue; /* no contrast anywhere in this band: no dark samples, no runs */
        expand4(trow, tt, tw);
        memset(trow + lw, 0, (size_t)(4 * tw + 64 - lw)); /* nothing past the row is dark */
        uint64_t *bits = det->row_bits;
        for (int y = y0; y < y1; y++) {
            dark_bits(L + (size_t)y * lw, trow, lw, bits);
            int start = n;
            n = extract_runs(bits, lw, y, R, n);
            for (int i = start; i < n; i++) P[i] = i;
            if (prev_y == y - 1) {
                /* Two adjacent rows touch at most once per run of either. */
                size_t most = 2 * ((size_t)(prev_end - prev_start) + (size_t)(n - start));
                if (split && det->diag_n + most > det->diag_capacity) {
                    size_t cap = det->diag_capacity ? 2 * det->diag_capacity : 4096;
                    while (cap < det->diag_n + most) cap *= 2;
                    int32_t *p = (int32_t *)realloc(det->diag, cap * sizeof(int32_t));
                    if (p == NULL) return -1;
                    det->diag = p;
                    det->diag_capacity = cap;
                }
                if (split == 2 && det->neck_n + most > det->neck_capacity) {
                    size_t cap = det->neck_capacity ? 2 * det->neck_capacity : 4096;
                    while (cap < det->neck_n + most) cap *= 2;
                    int32_t *p = (int32_t *)realloc(det->neck, cap * sizeof(int32_t));
                    if (p == NULL) return -1;
                    det->neck = p;
                    if ((p = (int32_t *)realloc(det->neck_root, cap * sizeof(int32_t))) == NULL) return -1;
                    det->neck_root = p;
                    det->neck_capacity = cap;
                }
                if (split == 2) join_rows(det, R, P, prev_start, prev_end, start, n, 2, lw, lh, factor, shadow);
                else if (split) join_rows(det, R, P, prev_start, prev_end, start, n, 1, lw, lh, factor, shadow);
                else join_rows(det, R, P, prev_start, prev_end, start, n, 0, lw, lh, factor, shadow);
            }
            prev_y = y, prev_start = start, prev_end = n;
        }
    }
    return n;
}
static double median_legacy(double *v, int n) {
    int k = n / 2 - 1, lo = 0, hi = n - 1;
    while (lo < hi) {
        double pivot = v[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (v[i] < pivot) i++;
            while (v[j] > pivot) j--;
            if (i <= j) {
                double t = v[i];
                v[i++] = v[j];
                v[j--] = t;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    double upper = v[k + 1];
    for (int m = k + 2; m < n; m++) upper = v[m] < upper ? v[m] : upper;
    return 0.5 * (v[k] + upper);
}

/* These oracles retain literal byte arithmetic and sorted samples, independent of
 * production interpolation tables, pair_diffs and histogram accumulation. */
static double sample_literal(const image_view *im, double x, double y) {
    if (!(x >= 0)) x = 0;
    if (!(y >= 0)) y = 0;
    if (!(x <= im->w - 1.001)) x = im->w - 1.001;
    if (!(y <= im->h - 1.001)) y = im->h - 1.001;
    int ix = (int)x, iy = (int)y;
    double fx = x - ix, fy = y - iy;
    const uint8_t *p = im->p + (size_t)iy * im->stride + ix;
    double a = p[0] + (p[1] - p[0]) * fx;
    double b = p[im->stride] + (p[im->stride + 1] - p[im->stride]) * fx;
    double value = a + (b - a) * fy;
    return im->invert ? 255.0 - value : value;
}
static void sample_equal(double actual, double expected) {
    assert(memcmp(&actual, &expected, sizeof actual) == 0);
}
static void independent_samples(void) {
    uint8_t bytes[10], before[10];
    memset(bytes, 0xa7, sizeof bytes);
    image_view im = {bytes + 1, 2, 2, 3, 0};
    const double phases[] = {-0.0, 0.0, .125, .5, .875, .999};
    unsigned count = 0, points = 0;
    for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) {
        bytes[1] = (uint8_t)a; bytes[2] = (uint8_t)b;
        bytes[4] = (uint8_t)(255 - a); bytes[5] = (uint8_t)(255 - b);
        memcpy(before, bytes, sizeof bytes);
        for (im.invert = 0; im.invert < 2; im.invert++) for (int k = 0; k < 6; k++) {
            double x = phases[k], y = phases[(k + 2) % 6];
            sample_equal(sample_bilinear(&im, x, y), sample_literal(&im, x, y)); count++;
        }
        assert(memcmp(bytes, before, sizeof bytes) == 0);
    }
    bytes[1] = 0; bytes[2] = 255; bytes[4] = 255; bytes[5] = 0;
    memcpy(before, bytes, sizeof bytes);
    double u[] = {-0.0, 0, -1, .999, 1, 2, NAN, INFINITY, -INFINITY};
    double v[] = {0, -0.0, 2, 1, .999, -1, INFINITY, -INFINITY, NAN};
    double H[9] = {1.1, .3, -.1, .7, 1.2, .6, .001, .002, 1}, out[9];
    for (im.invert = 0; im.invert < 2; im.invert++) for (int n = 0; n <= 9; n++) {
        sample_points(&im, H, u, v, n, out);
        for (int i = 0; i < n; i++) {
            vec2 p = project(H, u[i], v[i]);
            sample_equal(out[i], sample_literal(&im, p.x, p.y)); points++;
        }
        for (int i = 0; i < 9; i++) {
            sample_equal(sample_bilinear(&im, u[i], v[i]), sample_literal(&im, u[i], v[i])); count++;
        }
    }
    assert(memcmp(bytes, before, sizeof bytes) == 0);
    printf("%u independent scalar samples, %u independent projected samples passed\n", count, points);
}
static int compare_byte(const void *a, const void *b) {
    return (int)*(const uint8_t *)a - (int)*(const uint8_t *)b;
}
static int sorted_floor(uint8_t *values, size_t n, int minimum, double sigmas, int cap) {
    assert(n);
    qsort(values, n, 1, compare_byte);
    int floor = (int)(sigmas * values[(n - 1) / 2] / .954 + .5);
    if (floor > cap) floor = cap;
    return floor > minimum ? floor : minimum;
}
static void independent_floors(void) {
    const int heights[] = {1, 9, 57, 63, 64, 65, 129};
    const int extra_widths[] = {127, 128, 129, 193};
    const struct { int minimum; double sigmas; int cap; } settings[] = {
        {0, 6, 64}, {1, 7, 255}, {16, 0, 64}, {100, 6, 64}, {1, 1, 4}, {255, 7, 255}
    };
    unsigned calls = 0, regions = 0, unmeasured = 0, clipped = 0, nonuniform = 0, empty = 0;
    for (int wi = 1; wi <= 70; wi++) for (int hi = 0; hi < 7; hi++) for (int pattern = 0; pattern < 5; pattern++) {
        int w = wi <= 66 ? wi : extra_widths[wi - 67], h = heights[hi];
        size_t pixels = (size_t)w * h;
        uint8_t *owner = malloc(pixels + 38), *copy = malloc(pixels + 38);
        assert(owner && copy);
        uint8_t *image = owner + 19;
        memset(owner, 0xa7, pixels + 38);
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            unsigned value = pattern == 0 ? 0 : pattern == 1 ? ((x & 1) ? 255 : 0) :
                pattern == 2 ? (unsigned)(x * 13 + y * 31) : pattern == 3 ?
                (unsigned)((x & 1) ? ((x / 64 + y / 64) % 3 ? 40 : 1) : 0) :
                (unsigned)((x & 1) ? (x / 2 * 17 + y * 7) % 96 : 0);
            image[(size_t)y * w + x] = (uint8_t)value;
        }
        memcpy(copy, owner, pixels + 38);
        int cols = (w + 63) / 64, rows = (h + 63) / 64, total = cols * rows;
        uint8_t guarded[50], want[12], values[256], all[2048];
        assert((size_t)total <= sizeof want && (size_t)total + 38 <= sizeof guarded);
        napa_detector det = {0}; det.floors = guarded + 19; det.floor_alloc = (size_t)total;
        for (int s = 0; s < 6; s++) {
            memset(guarded, 0xa7, sizeof guarded);
            size_t all_n = 0;
            for (int ry = 0; ry < rows; ry++) for (int rx = 0; rx < cols; rx++) {
                size_t n = 0;
                for (int y = ry * 64; y < h && y < (ry + 1) * 64; y += 8)
                    for (int x = rx * 64; x + 1 < w && x + 1 < (rx + 1) * 64; x += 2) {
                        int d = abs((int)image[(size_t)y * w + x + 1] - image[(size_t)y * w + x]);
                        values[n++] = (uint8_t)(d > 63 ? 63 : d);
                    }
                assert(all_n + n <= sizeof all); memcpy(all + all_n, values, n); all_n += n;
                want[ry * cols + rx] = (uint8_t)(n < 16 ? 0 :
                    sorted_floor(values, n, settings[s].minimum, settings[s].sigmas, settings[s].cap));
                unmeasured += n < 16;
            }
            int frame = all_n ? sorted_floor(all, all_n, settings[s].minimum, settings[s].sigmas, settings[s].cap) : settings[s].minimum;
            empty += all_n == 0;
            int uniform = 1;
            for (int i = 0; i < total; i++) {
                clipped += want[i] > frame;
                if (!want[i] || want[i] > frame) want[i] = (uint8_t)frame;
                uniform &= want[i] == want[0];
            }
            assert(contrast_floors(&det, image, w, h, settings[s].minimum, settings[s].sigmas, settings[s].cap) == frame);
            assert(det.floor_cols == cols && det.floor_rows == rows && det.floor_uniform == uniform);
            assert(memcmp(det.floors, want, (size_t)total) == 0);
            for (int i = 0; i < (int)sizeof guarded; i++) if (i < 19 || i >= 19 + total) assert(guarded[i] == 0xa7);
            assert(memcmp(owner, copy, pixels + 38) == 0);
            calls++; regions += (unsigned)total; nonuniform += !uniform;
        }
        free(copy); free(owner);
    }
    assert(unmeasured && clipped && nonuniform && empty);
    printf("%u independent floor cases, %u regional values; unmeasurable=%u clipped=%u nonuniform=%u empty=%u passed\n",
           calls, regions, unmeasured, clipped, nonuniform, empty);
}

static uint32_t seed = 83;
static uint32_t rnd(void) { return seed = seed * 1664525u + 1013904223u; }
int main(void) {
    int widths[] = {1,2,3,4,15,16,17,63,64,65,127,128,129,29999,30000};
    for (int caseid=0; caseid<600; caseid++) {
        int w=widths[caseid%15], h=1+(int)(rnd()%17), nmax=((w+1)/2)*h+1;
        napa_detector *d=napa_create();
        d->level=calloc((size_t)w*h+64,1);
        d->runs=calloc(nmax,sizeof(napa_run)); d->parent=calloc(nmax,sizeof(int32_t));
        int tw=(w+3)/4, th=(h+3)/4;
        d->tile_min=malloc(tw*th); d->tile_max=malloc(tw*th);
        d->row_min=malloc(w+4); d->row_max=malloc(w+4); d->row_thr=malloc(w+68);
        d->row_bits=malloc(((w+63)/64+1)*sizeof(uint64_t));
        d->nb_lo=malloc(tw+2); d->nb_hi=malloc(tw+2); d->nb_t=malloc(tw+2); d->nb_f=malloc(tw+2);
        d->floors=malloc(1); d->floors[0]=caseid%32; d->floor_cols=d->floor_rows=d->floor_uniform=1;
        for (int i=0;i<w*h;i++) d->level[i]=(caseid%3==0)?(rnd()>>24):((rnd()>>(caseid%17))&1)*255;
        int n=threshold_legacy(d,w,h,1,0,caseid&1);
        napa_run *runs=malloc((n+1)*sizeof(napa_run)); int32_t *roots=malloc((n+1)*sizeof(int32_t));
        memcpy(runs,d->runs,n*sizeof(napa_run));
        for (int i=0;i<n;i++) roots[i]=find_root(d->parent,i);
        for (int path=0;path<2;path++) {
            int nn;
#ifdef NAPA_SSSE3_DISPATCH
            if(path && __builtin_cpu_supports("popcnt")) nn=threshold_index_popcnt(d,w,h,1,0,caseid&1);
            else
#endif
            nn=threshold_index_generic(d,w,h,1,0,caseid&1);
            if(nn!=n) return 10;
            for (int i=0;i<n;i++) {
                int root = find_root(d->parent,i);
                /* Thresholds and raster runs stay exact. Ordinary supported-corner cuts may
                 * partition an old component; shadow retains the original eight-connected graph.
                 * native_contact_kernels.c independently checks the new graph with pixel BFS. */
                if(runs[i].x0!=d->runs[i].x0 || runs[i].x1!=d->runs[i].x1 || runs[i].y!=d->runs[i].y ||
                   root < 0 || root > i || roots[root] != roots[i] || ((caseid&1) && roots[i]!=root)) {
                    printf("label failure %d %d\n",caseid,i);return 11;
                }
            }
        }
        free(runs);free(roots);napa_destroy(d);
    }
    uint8_t im[71*64]; for(int i=0;i<71*64;i++) im[i]=rnd()>>24;
    for(int t=0;t<10000;t++) {
        int k=t&1?11:15;
        int32_t x=(20<<16)+(rnd()&65535), y=(20<<16)+(rnd()&65535);
        int32_t dx=(int32_t)(rnd()%131071)-65535, dy=(int32_t)(rnd()%131071)-65535;
        int64_t p[16]; profile_fixed(im,71,x,y,dx,dy,k,p);
        for(int j=0;j<k;j++,x+=dx,y+=dy) {
            const uint8_t *q=im+(y>>16)*71+(x>>16);
            int fx=(x>>4)&4095,fy=(y>>4)&4095;
            int32_t a=(q[0]<<12)+(q[1]-q[0])*fx,b=(q[71]<<12)+(q[72]-q[71])*fx;
            if(p[j]!=((int64_t)a<<12)+(int64_t)(b-a)*fy) return 12;
        }
        int n=2+2*(rnd()%34); double a[68],b[68];
        for(int i=0;i<n;i++) a[i]=b[i]=(rnd()%1024)*.25;
        if(median_even(a,n)!=median_legacy(b,n)) return 13;
        image_view view={im,64,64,71,t&1};
        double H[9]={1.1,.3,-10.0,.7,1.2,6.0,.001,.002,1},u[68],v[68],out[68];
        for(int i=0;i<n;i++) u[i]=(int)(rnd()%200)-100,v[i]=(int)(rnd()%200)-100;
        if (t % 5 == 0) u[0] = NAN;
        if (t % 7 == 0) v[1] = INFINITY;
        sample_points(&view,H,u,v,n,out);
        for(int i=0;i<n;i++) {vec2 p=project(H,u[i],v[i]);if(out[i]!=sample_bilinear(&view,p.x,p.y)) return 14;}
    }
    independent_samples();
    independent_floors();
    puts("600 connectivity cases, generic/hardware paths; 10000 profile/median/paired sampling cases passed");
    return 0;
}
