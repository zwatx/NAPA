/*
 * Napa core: a sparse-sampling ("rake") AprilTag 36h11 detector in portable C99.
 *
 * Pipeline per frame:
 *   1. Rake: point-sample the image every 3/4 of the smallest cell. A tag's black border is
 *      one cell wide, so every border crossing contains a sample from its dark core.
 *   2. Threshold the sample from 4x4-tile min/max over a 3x3 tile neighbourhood. Tiles
 *      without contrast are skipped outright: they cannot contain a tag edge.
 *   3. Flood-fill dark components and keep tag-sized ones. Each component's extreme
 *      points give a rough quad.
 *   4. Fit each side to the strongest full-resolution edge along its normal (sub-pixel),
 *      and intersect the four lines for the corners.
 *   5. Sample the 8x8 cell grid through the homography and decode against all 2,348
 *      rotated codewords with a hardware popcount.
 *
 * The library has no dependencies and exposes a plain C ABI.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "napa_core.h"
#include "tag36h11_codes.h"

#define NAPA_ROTATED (NAPA_TAG36H11_COUNT * 4)

#if defined(__GNUC__) || defined(__clang__)
#define napa_popcount64(x) __builtin_popcountll(x)
#else
static int napa_popcount64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
}
#endif

struct napa_detector {
    uint64_t codes[NAPA_ROTATED];
    int16_t ids[NAPA_ROTATED];
    int8_t rotations[NAPA_ROTATED];
    /* Multi-index hashing: a code within H bits matches at least one of H + 1 disjoint blocks
     * exactly, so only codes sharing a block are compared. Six 6-bit blocks cover H <= 5 and
     * four 9-bit blocks cover H <= 3. Lists are stored compressed by bucket. */
    uint16_t mi6_start[6][65];
    uint16_t mi6_list[6][NAPA_ROTATED];
    uint16_t mi9_start[4][513];
    uint16_t mi9_list[4][NAPA_ROTATED];
    uint32_t stamp[NAPA_ROTATED];
    uint32_t stamp_now;

    uint8_t *level;
    uint8_t *mask;
    uint8_t *tile_min;
    uint8_t *tile_max;
    uint8_t *tile_lo;
    uint8_t *tile_hi;
    int32_t *stack;
    uint8_t *row_min;
    uint8_t *row_max;
    uint8_t *row_thr;
    size_t row_capacity;
    size_t mask_capacity;
    size_t level_capacity;
    size_t tile_capacity;
};

typedef struct {
    double x, y;
} vec2;

/* ---------------------------------------------------------------- family */

static uint64_t rotate_ccw(uint64_t code) {
    /* code holds a 6x6 row-major grid, MSB = row 0 col 0. Return it rotated 90 degrees counter-clockwise. */
    uint64_t out = 0;
    for (int r = 0; r < 6; r++) {
        for (int c = 0; c < 6; c++) {
            /* rot90 (CCW): out[r][c] = in[c][5 - r] */
            int src = c * 6 + (5 - r);
            uint64_t bit = (code >> (35 - src)) & 1u;
            out = (out << 1) | bit;
        }
    }
    return out;
}

static void build_index(const uint64_t *codes, int blocks, int bits, uint16_t *start, uint16_t *list) {
    int buckets = 1 << bits;
    for (int b = 0; b < blocks; b++) {
        uint16_t *st = start + b * (buckets + 1);
        uint16_t *ls = list + b * NAPA_ROTATED;
        memset(st, 0, sizeof(uint16_t) * (size_t)(buckets + 1));
        for (int i = 0; i < NAPA_ROTATED; i++) st[((codes[i] >> (b * bits)) & (uint64_t)(buckets - 1)) + 1]++;
        for (int v = 0; v < buckets; v++) st[v + 1] = (uint16_t)(st[v + 1] + st[v]);
        uint16_t fill[512];
        memcpy(fill, st, sizeof(uint16_t) * (size_t)buckets);
        for (int i = 0; i < NAPA_ROTATED; i++) {
            int v = (int)((codes[i] >> (b * bits)) & (uint64_t)(buckets - 1));
            ls[fill[v]++] = (uint16_t)i;
        }
    }
}

/* Index of the rotated code within max_bits of `code`, or -1. tag36h11 codes (with rotations)
 * are at least 11 bits apart, so for max_bits <= 5 any match is unique. */
static int nearest_code(napa_detector *det, uint64_t code, int max_bits, int *distance) {
    int blocks, bits;
    const uint16_t *start, *list;
    if (max_bits <= 3) {
        blocks = 4, bits = 9, start = &det->mi9_start[0][0], list = &det->mi9_list[0][0];
    } else if (max_bits <= 5) {
        blocks = 6, bits = 6, start = &det->mi6_start[0][0], list = &det->mi6_list[0][0];
    } else {
        int best = 64, bi = -1;
        for (int i = 0; i < NAPA_ROTATED; i++) {
            int d = napa_popcount64(code ^ det->codes[i]);
            if (d < best) best = d, bi = i;
        }
        *distance = best;
        return best <= max_bits ? bi : -1;
    }
    if (++det->stamp_now == 0) {
        memset(det->stamp, 0, sizeof det->stamp);
        det->stamp_now = 1;
    }
    int buckets = 1 << bits;
    for (int b = 0; b < blocks; b++) {
        const uint16_t *st = start + b * (buckets + 1);
        const uint16_t *ls = list + b * NAPA_ROTATED;
        int v = (int)((code >> (b * bits)) & (uint64_t)(buckets - 1));
        for (int e = st[v]; e < st[v + 1]; e++) {
            int i = ls[e];
            if (det->stamp[i] == det->stamp_now) continue;
            det->stamp[i] = det->stamp_now;
            int d = napa_popcount64(code ^ det->codes[i]);
            if (d <= max_bits) {
                *distance = d;
                return i;
            }
        }
    }
    return -1;
}

static void load_family(napa_detector *det) {
    for (int rot = 0; rot < 4; rot++) {
        for (int id = 0; id < NAPA_TAG36H11_COUNT; id++) {
            uint64_t code = napa_tag36h11_codes[id];
            for (int k = 0; k < rot; k++) {
                code = rotate_ccw(code);
            }
            int index = rot * NAPA_TAG36H11_COUNT + id;
            det->codes[index] = code;
            det->ids[index] = (int16_t)id;
            det->rotations[index] = (int8_t)rot;
        }
    }
    build_index(det->codes, 6, 6, &det->mi6_start[0][0], &det->mi6_list[0][0]);
    build_index(det->codes, 4, 9, &det->mi9_start[0][0], &det->mi9_list[0][0]);
}

/* ---------------------------------------------------------------- lifecycle */

napa_detector *napa_create(void) {
    napa_detector *det = (napa_detector *)calloc(1, sizeof(napa_detector));
    if (det != NULL) load_family(det);
    return det;
}

void napa_destroy(napa_detector *det) {
    if (det == NULL) {
        return;
    }
    free(det->level);
    free(det->mask);
    free(det->tile_min);
    free(det->tile_max);
    free(det->tile_lo);
    free(det->tile_hi);
    free(det->stack);
    free(det->row_min);
    free(det->row_max);
    free(det->row_thr);
    free(det);
}

static int ensure_buffers(napa_detector *det, size_t pixels, size_t tiles, size_t row, size_t mask_pixels) {
    if (mask_pixels > det->mask_capacity) {
        uint8_t *mask = (uint8_t *)realloc(det->mask, mask_pixels);
        if (mask == NULL) return -1;
        det->mask = mask;
        det->mask_capacity = mask_pixels;
    }
    if (row > det->row_capacity) {
        uint8_t *p;
        if ((p = (uint8_t *)realloc(det->row_min, row)) == NULL) return -1;
        det->row_min = p;
        if ((p = (uint8_t *)realloc(det->row_max, row)) == NULL) return -1;
        det->row_max = p;
        if ((p = (uint8_t *)realloc(det->row_thr, row)) == NULL) return -1;
        det->row_thr = p;
        det->row_capacity = row;
    }
    if (pixels > det->level_capacity) {
        uint8_t *level = (uint8_t *)realloc(det->level, pixels);
        if (level == NULL) return -1;
        det->level = level;
        int32_t *stack = (int32_t *)realloc(det->stack, (pixels + 1) * sizeof(int32_t));
        if (stack == NULL) return -1;
        det->stack = stack;
        det->level_capacity = pixels;
    }
    if (tiles > det->tile_capacity) {
        uint8_t *p;
        if ((p = (uint8_t *)realloc(det->tile_min, tiles)) == NULL) return -1;
        det->tile_min = p;
        if ((p = (uint8_t *)realloc(det->tile_max, tiles)) == NULL) return -1;
        det->tile_max = p;
        if ((p = (uint8_t *)realloc(det->tile_lo, tiles)) == NULL) return -1;
        det->tile_lo = p;
        if ((p = (uint8_t *)realloc(det->tile_hi, tiles)) == NULL) return -1;
        det->tile_hi = p;
        det->tile_capacity = tiles;
    }
    return 0;
}

/* ---------------------------------------------------------------- sampling */

static inline double sample_bilinear(const uint8_t *img, int w, int h, int stride, double x, double y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > w - 1.001) x = w - 1.001;
    if (y > h - 1.001) y = h - 1.001;
    int x0 = (int)x;
    int y0 = (int)y;
    double fx = x - x0;
    double fy = y - y0;
    const uint8_t *p = img + (size_t)y0 * stride + x0;
    double a = p[0] + (p[1] - p[0]) * fx;
    double b = p[stride] + (p[stride + 1] - p[stride]) * fx;
    return a + (b - a) * fy;
}

/* ---------------------------------------------------------------- threshold */

#define TILE 4

static void threshold_level(napa_detector *det, int lw, int lh, int min_contrast) {
    const uint8_t *L = det->level;
    uint8_t *M = det->mask;
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
        for (int tx = 0; tx < tw; tx++) {
            int x0 = tx * TILE;
            int x1 = x0 + TILE < lw ? x0 + TILE : lw;
            uint8_t lo = vmin[x0], hi = vmax[x0];
            for (int x = x0 + 1; x < x1; x++) {
                lo = vmin[x] < lo ? vmin[x] : lo;
                hi = vmax[x] > hi ? vmax[x] : hi;
            }
            tmin[tx] = lo;
            tmax[tx] = hi;
        }
    }
    /* 3x3 tile neighbourhood, separably: horizontal then vertical. */
    for (int ty = 0; ty < th; ty++) {
        const uint8_t *a = det->tile_min + (size_t)ty * tw, *b = det->tile_max + (size_t)ty * tw;
        uint8_t *lo = det->tile_lo + (size_t)ty * tw, *hi = det->tile_hi + (size_t)ty * tw;
        for (int tx = 0; tx < tw; tx++) {
            uint8_t l = a[tx], h = b[tx];
            if (tx > 0) {
                l = a[tx - 1] < l ? a[tx - 1] : l;
                h = b[tx - 1] > h ? b[tx - 1] : h;
            }
            if (tx + 1 < tw) {
                l = a[tx + 1] < l ? a[tx + 1] : l;
                h = b[tx + 1] > h ? b[tx + 1] : h;
            }
            lo[tx] = l;
            hi[tx] = h;
        }
    }
    for (int ty = 0; ty < th; ty++) {
        const uint8_t *lo_up = det->tile_lo + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *lo_mid = det->tile_lo + (size_t)ty * tw;
        const uint8_t *lo_dn = det->tile_lo + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        const uint8_t *hi_up = det->tile_hi + (size_t)(ty > 0 ? ty - 1 : ty) * tw;
        const uint8_t *hi_mid = det->tile_hi + (size_t)ty * tw;
        const uint8_t *hi_dn = det->tile_hi + (size_t)(ty + 1 < th ? ty + 1 : ty) * tw;
        int any = 0;
        for (int tx = 0; tx < tw; tx++) {
            uint8_t l = lo_mid[tx], h = hi_mid[tx];
            l = lo_up[tx] < l ? lo_up[tx] : l;
            l = lo_dn[tx] < l ? lo_dn[tx] : l;
            h = hi_up[tx] > h ? hi_up[tx] : h;
            h = hi_dn[tx] > h ? hi_dn[tx] : h;
            /* Boyer-Moore-style skip: a flat neighbourhood cannot hold a tag edge, so its
             * threshold is 0 and no sample can fall below it. */
            uint8_t t = (h - l >= min_contrast) ? (uint8_t)((l + h + 1) / 2) : 0;
            any |= t;
            int x0 = tx * TILE;
            int x1 = x0 + TILE < lw ? x0 + TILE : lw;
            for (int x = x0; x < x1; x++) trow[x] = t;
        }
        int y0 = ty * TILE;
        int y1 = y0 + TILE < lh ? y0 + TILE : lh;
        if (!any) continue; /* the padded mask was cleared for this level */
        for (int y = y0; y < y1; y++) {
            const uint8_t *row = L + (size_t)y * lw;
            uint8_t *out = M + (size_t)(y + 1) * (lw + 2) + 1;
            for (int x = 0; x < lw; x++) out[x] = row[x] < trow[x];
        }
    }
}

/* ---------------------------------------------------------------- geometry */

/* Fit each side to the outer border edge searched from r_out outside to r_in inside the given quad. */
static int fit_sides(const uint8_t *img, int w, int h, int stride, const vec2 in[4], vec2 out[4], double r_out,
                     double r_in, int min_edge, int outermost, int max_points, double *edge_width) {
    double width_sum = 0;
    int width_n = 0;
    double reach = 0.5 * (r_out + r_in);
    enum { MAXP = 24, KMAX = 15 };
    /* The first (outermost) pass only has to land within a fraction of a cell for the early
     * decode, so it uses a shorter profile. */
    const int K = outermost ? 11 : KMAX;
    vec2 cen = {0, 0};
    for (int k = 0; k < 4; k++) {
        cen.x += in[k].x * 0.25;
        cen.y += in[k].y * 0.25;
    }
    double lx[4], ly[4], dx[4], dy[4];
    for (int k = 0; k < 4; k++) {
        vec2 a = in[k], b = in[(k + 1) & 3];
        double ex = b.x - a.x, ey = b.y - a.y;
        double len = sqrt(ex * ex + ey * ey);
        if (len < 6.0) return 0;
        ex /= len;
        ey /= len;
        double nx = -ey, ny = ex;
        if ((cen.x - a.x) * nx + (cen.y - a.y) * ny < 0) {
            nx = -nx;
            ny = -ny;
        }
        int np = (int)(len / 4.0);
        if (np < 6) np = 6;
        if (np > max_points) np = max_points;
        if (np > MAXP) np = MAXP;
        double px[MAXP], py[MAXP], wt[MAXP];
        int good = 0;
        double step = (r_out + r_in) / (K - 1);
        for (int i = 0; i < np; i++) {
            double t = 0.12 + 0.76 * (i + 0.5) / np;
            double bx = a.x + (b.x - a.x) * t, by = a.y + (b.y - a.y) * t;
            double prof[KMAX];
            double sx0 = bx - nx * r_out, sy0 = by - ny * r_out;
            double sx1 = bx + nx * r_in, sy1 = by + ny * r_in;
            if (sx0 >= 0 && sy0 >= 0 && sx1 >= 0 && sy1 >= 0 && sx0 < w - 1 && sx1 < w - 1 && sy0 < h - 1 &&
                sy1 < h - 1) {
                /* Whole profile inside the image: skip per-sample clamping. */
                double sx = sx0, sy = sy0, ix = nx * step, iy = ny * step;
                for (int j = 0; j < K; j++, sx += ix, sy += iy) {
                    int x0 = (int)sx, y0 = (int)sy;
                    double fx = sx - x0, fy = sy - y0;
                    const uint8_t *q = img + (size_t)y0 * stride + x0;
                    double top = q[0] + (q[1] - q[0]) * fx;
                    double bot = q[stride] + (q[stride + 1] - q[stride]) * fx;
                    prof[j] = top + (bot - top) * fy;
                }
            } else {
                for (int j = 0; j < K; j++) {
                    double o = -r_out + j * step;
                    prof[j] = sample_bilinear(img, w, h, stride, bx + nx * o, by + ny * o);
                }
            }
            /* outside (white) minus inside (black): the outer border edge is a positive peak. */
            double best = 0;
            int bj = -1;
            double g[KMAX];
            for (int j = 1; j < K - 1; j++) {
                g[j] = prof[j - 1] - prof[j + 1];
                if (g[j] > best) {
                    best = g[j];
                    bj = j;
                }
            }
            /* The outer border is the first white-to-black step met going inward; payload cells
             * further in produce equally strong steps, so take the outermost local peak near the max. */
            for (int j = 1; outermost && j < K - 1; j++) {
                if (g[j] >= 0.6 * best && (j == K - 2 || g[j] >= g[j + 1])) {
                    bj = j;
                    best = g[j];
                    break;
                }
            }
            /* Normalise by the derivative span so the strength test is independent of the step. */
            if (bj < 0 || best / (2.0 * step) < min_edge * 0.25) continue;
            /* Centre of mass of the positive gradient lobe around the peak. For an area-sampled edge
             * this is exactly the edge at any blur width; trimming or squaring the weights biases
             * sharp edges. The lobe ends where the gradient changes sign, and the final window is
             * capped below half a cell, so the next payload edge never contributes. */
            int jl = bj, jr = bj;
            while (jl > 1 && g[jl - 1] > 0.05 * best) jl--;
            while (jr < K - 2 && g[jr + 1] > 0.05 * best) jr++;
            double sg = 0, sj = 0;
            for (int j = jl; j <= jr; j++) {
                double wv = g[j];
                sg += wv;
                sj += wv * j;
            }
            double cj = sg > 1e-9 ? sj / sg : bj;
            width_sum += (jr - jl + 1) * step;
            width_n++;
            double pos = -r_out + cj * step;
            px[good] = bx + nx * pos;
            py[good] = by + ny * pos;
            wt[good] = best;
            good++;
        }
        if (good < 3) return 0;
        /* Weighted total-least-squares line, then one pass dropping outliers. */
        for (int pass = 0; pass < 2; pass++) {
            double sw = 0, mx = 0, my = 0;
            for (int i = 0; i < good; i++) {
                sw += wt[i];
                mx += wt[i] * px[i];
                my += wt[i] * py[i];
            }
            if (sw <= 0) return 0;
            mx /= sw;
            my /= sw;
            double sxx = 0, syy = 0, sxy = 0;
            for (int i = 0; i < good; i++) {
                double qx = px[i] - mx, qy = py[i] - my;
                sxx += wt[i] * qx * qx;
                syy += wt[i] * qy * qy;
                sxy += wt[i] * qx * qy;
            }
            double theta = 0.5 * atan2(2 * sxy, sxx - syy);
            lx[k] = mx;
            ly[k] = my;
            dx[k] = cos(theta);
            dy[k] = sin(theta);
            if (pass == 0) {
                int kept = 0;
                for (int i = 0; i < good; i++) {
                    double dist = fabs((px[i] - mx) * (-dy[k]) + (py[i] - my) * dx[k]);
                    if (dist <= 1.0 + 0.1 * reach) {
                        px[kept] = px[i];
                        py[kept] = py[i];
                        wt[kept] = wt[i];
                        kept++;
                    }
                }
                /* Drop outliers only when most samples agree; otherwise keep the full fit. */
                if (kept >= 3 && 2 * kept >= good) good = kept;
            }
        }
    }
    *edge_width = width_n ? width_sum / width_n : 0;
    for (int k = 0; k < 4; k++) {
        int p = (k + 3) & 3; /* corner k joins side k-1 and side k */
        double cross = dx[p] * dy[k] - dy[p] * dx[k];
        if (fabs(cross) < 0.2) return 0;
        double ddx = lx[k] - lx[p], ddy = ly[k] - ly[p];
        double t = (ddx * dy[k] - ddy * dx[k]) / cross;
        out[k].x = lx[p] + t * dx[p];
        out[k].y = ly[p] + t * dy[p];
        double mv = fabs(out[k].x - in[k].x) + fabs(out[k].y - in[k].y);
        if (mv > 4.0 * reach + 2.0) return 0;
    }
    return 1;
}

static double signed_area(const vec2 q[4]) {
    double s = 0;
    for (int k = 0; k < 4; k++) {
        s += q[k].x * q[(k + 1) & 3].y - q[k].y * q[(k + 1) & 3].x;
    }
    return 0.5 * s;
}

/* Projective map of the unit square (0..1) onto quad q (Heckbert). */
static void square_to_quad(const vec2 q[4], double H[9]) {
    double x0 = q[0].x, y0 = q[0].y, x1 = q[1].x, y1 = q[1].y;
    double x2 = q[2].x, y2 = q[2].y, x3 = q[3].x, y3 = q[3].y;
    double sx = x0 - x1 + x2 - x3, sy = y0 - y1 + y2 - y3;
    double dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2;
    double g = 0, hh = 0;
    double den = dx1 * dy2 - dx2 * dy1;
    if (fabs(den) > 1e-12 && (fabs(sx) > 1e-12 || fabs(sy) > 1e-12)) {
        g = (sx * dy2 - dx2 * sy) / den;
        hh = (dx1 * sy - sx * dy1) / den;
    }
    H[0] = x1 - x0 + g * x1;
    H[1] = x3 - x0 + hh * x3;
    H[2] = x0;
    H[3] = y1 - y0 + g * y1;
    H[4] = y3 - y0 + hh * y3;
    H[5] = y0;
    H[6] = g;
    H[7] = hh;
    H[8] = 1;
}

static inline vec2 project(const double H[9], double u, double v) {
    double z = H[6] * u + H[7] * v + H[8];
    vec2 p = {(H[0] * u + H[1] * v + H[2]) / z, (H[3] * u + H[4] * v + H[5]) / z};
    return p;
}

static void sort_small(double *v, int n) {
    for (int i = 1; i < n; i++) {
        double x = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > x) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = x;
    }
}

/* `slack` loosens the border and Hamming limits, for a cheap early decode on unrefined corners. */
static int decode_quad(napa_detector *det, const uint8_t *img, int w, int h, int stride, vec2 q[4],
                       const napa_params *params, int slack, napa_detection *out) {
    if (signed_area(q) < 0) {
        vec2 t = q[1];
        q[1] = q[3];
        q[3] = t;
    }
    double H[9];
    square_to_quad(q, H);
    double grid[64];
    double border[28];
    double ring[36];
    int nb = 0, nr = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            vec2 p = project(H, (c + 0.5) / 8.0, (r + 0.5) / 8.0);
            double v = sample_bilinear(img, w, h, stride, p.x, p.y);
            grid[r * 8 + c] = v;
            if (r == 0 || r == 7 || c == 0 || c == 7) border[nb++] = v;
        }
    }
    for (int i = 0; i < 10; i++) {
        double u = (i - 0.5) / 8.0;
        vec2 p;
        p = project(H, u, -0.5 / 8.0);
        ring[nr++] = sample_bilinear(img, w, h, stride, p.x, p.y);
        p = project(H, u, 8.5 / 8.0);
        ring[nr++] = sample_bilinear(img, w, h, stride, p.x, p.y);
    }
    for (int i = 1; i < 9; i++) {
        double v = (i - 0.5) / 8.0;
        vec2 p;
        p = project(H, -0.5 / 8.0, v);
        ring[nr++] = sample_bilinear(img, w, h, stride, p.x, p.y);
        p = project(H, 8.5 / 8.0, v);
        ring[nr++] = sample_bilinear(img, w, h, stride, p.x, p.y);
    }
    /* Sharpen the cell samples, as upstream AprilTag does, to undo blur between neighbouring cells. */
    if (params->sharpening_x100 > 0) {
        double k = params->sharpening_x100 / 100.0;
        double sharp[64];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                double v = grid[r * 8 + c], lap = 4 * v;
                lap -= r > 0 ? grid[(r - 1) * 8 + c] : v;
                lap -= r < 7 ? grid[(r + 1) * 8 + c] : v;
                lap -= c > 0 ? grid[r * 8 + c - 1] : v;
                lap -= c < 7 ? grid[r * 8 + c + 1] : v;
                sharp[r * 8 + c] = v + k * lap;
            }
        }
        memcpy(grid, sharp, sizeof sharp);
    }
    double bs[28], rs[36];
    memcpy(bs, border, sizeof bs);
    memcpy(rs, ring, sizeof rs);
    sort_small(bs, 28);
    sort_small(rs, 36);
    double black = 0.5 * (bs[13] + bs[14]);
    double white = 0.5 * (rs[17] + rs[18]);
    if (white - black < params->min_contrast) return 0;
    double thr = 0.5 * (black + white);
    int dark = 0;
    for (int i = 0; i < 28; i++) dark += border[i] < thr;
    if (dark < params->min_border_cells - slack) return 0;

    uint64_t code = 0;
    for (int r = 1; r < 7; r++) {
        for (int c = 1; c < 7; c++) {
            code = (code << 1) | (grid[r * 8 + c] > thr ? 1u : 0u);
        }
    }
    int best = 64;
    int bi = nearest_code(det, code, params->max_hamming + slack, &best);
    if (bi < 0) return 0;
    int rot = det->rotations[bi];
    /* Observed grid == rot90(canonical, rot); the canonical top-left sits at corner index (4 - rot) % 4. */
    for (int k = 0; k < 4; k++) {
        vec2 c = q[(k - rot + 4) & 3];
        out->corners[2 * k] = c.x;
        out->corners[2 * k + 1] = c.y;
    }
    out->id = det->ids[bi];
    out->hamming = best;
    out->decision_margin = (float)(white - black);
    return 1;
}

/* ---------------------------------------------------------------- components */

typedef struct {
    int count;
    int minx, maxx, miny, maxy;
    int ext[8][2]; /* extremes: +x, -x, +y, -y, +(x+y), -(x+y), +(x-y), -(x-y) */
    int extv[8];
} component;

static void comp_add(component *c, int x, int y) {
    int vals[8] = {x, -x, y, -y, x + y, -(x + y), x - y, -(x - y)};
    if (c->count == 0) {
        c->minx = c->maxx = x;
        c->miny = c->maxy = y;
        for (int k = 0; k < 8; k++) {
            c->extv[k] = vals[k];
            c->ext[k][0] = x;
            c->ext[k][1] = y;
        }
    } else {
        if (x < c->minx) c->minx = x;
        if (x > c->maxx) c->maxx = x;
        if (y < c->miny) c->miny = y;
        if (y > c->maxy) c->maxy = y;
        for (int k = 0; k < 8; k++) {
            if (vals[k] > c->extv[k]) {
                c->extv[k] = vals[k];
                c->ext[k][0] = x;
                c->ext[k][1] = y;
            }
        }
    }
    c->count++;
}

static double quad_area_pts(int p[4][2]) {
    double s = 0;
    for (int k = 0; k < 4; k++) {
        s += (double)p[k][0] * p[(k + 1) & 3][1] - (double)p[k][1] * p[(k + 1) & 3][0];
    }
    return fabs(0.5 * s);
}

static int covered(const napa_detection *out, int n, double x, double y, double radius) {
    for (int i = 0; i < n; i++) {
        double ox = 0, oy = 0;
        for (int k = 0; k < 4; k++) {
            ox += out[i].corners[2 * k] * 0.25;
            oy += out[i].corners[2 * k + 1] * 0.25;
        }
        if (fabs(ox - x) + fabs(oy - y) < radius) return 1;
    }
    return 0;
}

static int already_found(const napa_detection *out, int n, const napa_detection *d) {
    double cx = 0, cy = 0;
    for (int k = 0; k < 4; k++) {
        cx += d->corners[2 * k] * 0.25;
        cy += d->corners[2 * k + 1] * 0.25;
    }
    double size = fabs(d->corners[0] - d->corners[4]) + fabs(d->corners[1] - d->corners[5]);
    for (int i = 0; i < n; i++) {
        double ox = 0, oy = 0;
        for (int k = 0; k < 4; k++) {
            ox += out[i].corners[2 * k] * 0.25;
            oy += out[i].corners[2 * k + 1] * 0.25;
        }
        if (fabs(ox - cx) + fabs(oy - cy) < 0.25 * size + 2.0) return 1;
    }
    return 0;
}

/* ---------------------------------------------------------------- detect */

/* Tile contrast floor that rises with sensor noise. The median absolute difference of
 * neighbouring samples is about 0.95 sigma for Gaussian noise, and a flat noisy tile
 * neighbourhood spans about 5.5 sigma, so a floor of 6 sigma keeps noise from forming
 * dark networks. Capped well below real tag contrast. */
static int contrast_floor(const uint8_t *L, int lw, int lh, int min_contrast) {
    uint32_t hist[256] = {0};
    uint32_t n = 0;
    for (int y = 0; y < lh; y += 8) {
        const uint8_t *row = L + (size_t)y * lw;
        for (int x = 0; x + 1 < lw; x += 2) {
            int d = row[x + 1] - row[x];
            hist[d < 0 ? -d : d]++;
            n++;
        }
    }
    if (n == 0) return min_contrast;
    uint32_t seen = 0;
    int median = 0;
    while (median < 255 && (seen += hist[median]) < (n + 1) / 2) median++;
    int floor = (int)(6.0 * median / 0.954 + 0.5);
    if (floor > 64) floor = 64;
    return floor > min_contrast ? floor : min_contrast;
}

int napa_detect(napa_detector *det, const uint8_t *img, int width, int height, int stride, const napa_params *params,
                napa_detection *out, int max_out) {
    if (det == NULL || img == NULL || params == NULL) return -1;
    if (stride < width) return -1;
    if (width > 30000 || height > 30000) return -1;
    if (width < 16 || height < 16) return 0;
    /* A tag cell is min_tag_px / 8 wide, but blur turns its edges grey, so only its core is
     * reliably dark. Sampling every 3/4 cell guarantees a sample inside that core. */
    int base = params->min_tag_px * 3 / 32;
    if (base < 1) base = 1;
    int found = 0;
    int contrast = params->min_contrast;
    /* Size band in level samples. Adjacent levels overlap (7-24 here is 14-48 one level finer),
     * because a blob's measured extent varies by a sample or two with phase, blur and rotation. */
    const int lo = 7, hi = 24;

    for (int s = base;; s *= 2) {
        int lw = (width + s - 1) / s;
        int lh = (height + s - 1) / s;
        int last = (lw < 2 * hi * 2) || (lh < 2 * hi * 2);
        size_t pixels = (size_t)lw * lh;
        size_t tiles = (size_t)((lw + TILE - 1) / TILE) * ((lh + TILE - 1) / TILE);
        if (ensure_buffers(det, pixels, tiles, (size_t)lw, (size_t)(lw + 2) * (lh + 2)) != 0) return -2;

        /* The rake: one point sample per stride, row by row. */
        for (int y = 0; y < lh; y++) {
            const uint8_t *src = img + (size_t)y * s * stride;
            uint8_t *dst = det->level + (size_t)y * lw;
            if (s == 1) {
                memcpy(dst, src, (size_t)lw);
            } else {
                for (int x = 0; x < lw; x++) dst[x] = src[(size_t)x * s];
            }
        }
        /* The mask carries a one-sample zero border so the flood fill needs no bounds checks. */
        const int mw = lw + 2;
        const size_t mpixels = (size_t)mw * (lh + 2);
        memset(det->mask, 0, mpixels);
        if (s == base) contrast = contrast_floor(det->level, lw, lh, params->min_contrast);
        threshold_level(det, lw, lh, contrast);

        uint8_t *M = det->mask;
        int32_t *stack = det->stack;
        int min_size = lo;
        int max_size = last ? 1 << 30 : hi;
        const int offsets[8] = {-mw - 1, -mw, -mw + 1, -1, 1, mw - 1, mw, mw + 1};
        /* List entries pack padded mask coordinates (y + 1) << 16 | (x + 1), which are never
         * negative, so a neighbour is entry + step with no borrow between the halves. */
        const int32_t steps[8] = {-65537, -65536, -65535, -1, 1, 65535, 65536, 65537};
        for (const uint8_t *hit = memchr(M, 1, mpixels); hit != NULL;
             hit = memchr(hit + 1, 1, mpixels - (size_t)(hit + 1 - M))) {
            int start = (int)(hit - M);
            /* Breadth-first fill into a list, tracking only the bounding box and count; the list
             * keeps the blob's samples so extremes are computed only for tag-sized blobs. */
            int head = 0, tail = 0;
            int minx = start % mw, maxx = minx, miny = start / mw, maxy = miny;
            stack[tail++] = (miny << 16) | minx;
            M[start] = 2;
            while (head < tail) {
                int32_t packed = stack[head++];
                int x = packed & 0xffff, y = packed >> 16;
                minx = x < minx ? x : minx;
                maxx = x > maxx ? x : maxx;
                miny = y < miny ? y : miny;
                maxy = y > maxy ? y : maxy;
                int idx = y * mw + x;
                /* Branchless: on noisy frames the neighbour test is unpredictable, and mispredicted
                 * branches would dominate the fill. The list has one spare slot for the last write. */
                for (int k = 0; k < 8; k++) {
                    int j = idx + offsets[k];
                    int grow = M[j] == 1;
                    M[j] = (uint8_t)(M[j] + grow);
                    stack[tail] = packed + steps[k];
                    tail += grow;
                }
            }
            component comp;
            comp.count = tail;
            comp.minx = minx - 1;
            comp.maxx = maxx - 1;
            comp.miny = miny - 1;
            comp.maxy = maxy - 1;
            int bw = comp.maxx - comp.minx + 1, bh = comp.maxy - comp.miny + 1;
            int big = bw > bh ? bw : bh, small = bw < bh ? bw : bh;
            if (small < min_size || big > max_size || big > 3 * small) continue;
            double fill = (double)comp.count / ((double)bw * bh);
            if (fill < 0.2) continue;
            /* Levels run fine to coarse; skip a blob whose centre a finer level already decoded. */
            double ccx = 0.5 * (comp.minx + comp.maxx) * s, ccy = 0.5 * (comp.miny + comp.maxy) * s;
            if (covered(out, found < max_out ? found : max_out, ccx, ccy, 0.3 * big * s + 2.0)) continue;
            comp.count = 0;
            for (int i = 0; i < tail; i++) comp_add(&comp, (stack[i] & 0xffff) - 1, (stack[i] >> 16) - 1);
            /* Rough quad: of the eight extreme points (in clockwise angular order), the four that
             * span the largest quadrilateral. Axis or diagonal extremes alone degrade near 22 and
             * 67 degrees; the best mix tracks the corners at any rotation. */
            static const int order[8] = {0, 4, 2, 7, 1, 5, 3, 6};
            int pick[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
            double best_area = -1;
            for (int i = 0; i < 5; i++) {
                for (int j = i + 1; j < 6; j++) {
                    for (int k = j + 1; k < 7; k++) {
                        for (int l = k + 1; l < 8; l++) {
                            int q[4][2] = {{comp.ext[order[i]][0], comp.ext[order[i]][1]},
                                           {comp.ext[order[j]][0], comp.ext[order[j]][1]},
                                           {comp.ext[order[k]][0], comp.ext[order[k]][1]},
                                           {comp.ext[order[l]][0], comp.ext[order[l]][1]}};
                            double area = quad_area_pts(q);
                            if (area > best_area) {
                                best_area = area;
                                memcpy(pick, q, sizeof pick);
                            }
                        }
                    }
                }
            }
            if (best_area < 0.3 * bw * bh) continue;
            vec2 rough[4], fine[4], finer[4];
            for (int k = 0; k < 4; k++) {
                rough[k].x = pick[k][0] * (double)s;
                rough[k].y = pick[k][1] * (double)s;
            }
            /* Rough corners are dark samples, so the true border lies outward of them by at most
             * one stride: search mostly outward, which keeps payload edges out of the window. */
            double reach = 0.5 * ((1.5 * s + 1.0) + (0.5 * s + 1.0));
            double edge_w = 0;
            if (!fit_sides(img, width, height, stride, rough, fine, 1.5 * s + 1.0, 0.5 * s + 1.0, params->min_contrast, 1, 6, &edge_w))
                continue;
            /* The first fit is already within a fraction of a cell: decode now, with two bits of
             * slack, and spend the refinement passes only on quads that are tags. A random quad
             * passes this only about 1.5% of the time. */
            vec2 probe[4];
            napa_detection early;
            memcpy(probe, fine, sizeof probe);
            if (!decode_quad(det, img, width, height, stride, probe, params, 2, &early)) continue;
            double cell = 0;
            for (int k = 0; k < 4; k++) {
                double ex = fine[(k + 1) & 3].x - fine[k].x, ey = fine[(k + 1) & 3].y - fine[k].y;
                cell += sqrt(ex * ex + ey * ey) / 32.0;
            }
            /* Coarse-to-fine: shrink the symmetric window to the previous pass's resolution, but
             * never below the measured edge width, so the whole ramp stays inside it. */
            int ok = 1;
            for (int pass = 0; pass < 5; pass++) {
                double step = 2.0 * reach / 14.0;
                double next = 1.5 * step;
                if (next < 1.5) next = 1.5;
                if (next < 1.25 * edge_w) next = 1.25 * edge_w;
                /* ...but within half a cell, so the window never reaches the next payload edge. */
                if (next > 0.45 * cell && 0.45 * cell >= 1.0) next = 0.45 * cell;
                if (pass > 0 && (next >= 0.9 * reach || step <= 0.35)) break;
                reach = next;
                ok = fit_sides(img, width, height, stride, fine, finer, reach, reach, params->min_contrast, 0, 8, &edge_w);
                if (!ok) break;
                memcpy(fine, finer, sizeof fine);
            }
            /* One more pass at the converged window, re-centred on the last estimate, with every
             * edge sample. */
            if (ok) ok = fit_sides(img, width, height, stride, fine, finer, reach, reach, params->min_contrast, 0, 12, &edge_w);
            if (!ok) continue;
            napa_detection d;
            if (!decode_quad(det, img, width, height, stride, finer, params, 0, &d)) continue;
            if (already_found(out, found < max_out ? found : max_out, &d)) continue;
            if (found < max_out) out[found] = d;
            found++;
        }
        if (last) break;
    }
    return found < max_out ? found : max_out;
}

/* setuptools links every extension with a module-init export on Windows. The library is loaded
 * with ctypes, never imported, so this placeholder only satisfies the linker. */
NAPA_API void *PyInit__napa_core(void) { return 0; }
