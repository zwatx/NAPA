/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Independent scalar oracle for paired samples and their unclamped coordinates. */
#ifndef NAPA_BORDER_BATCH_SOURCE
#define NAPA_BORDER_BATCH_SOURCE "../src/napa_vision/_native/napa_core.c"
#endif
#include NAPA_BORDER_BATCH_SOURCE
#include <assert.h>
#include <float.h>
#include <stdio.h>

static uint32_t seed = 83;
static uint32_t rnd(void) { return seed = seed * 1664525u + 1013904223u; }
static uint64_t bits(double x) { uint64_t b; memcpy(&b, &x, sizeof b); return b; }
static void same(double a, double b) {
    if (isnan(a)) assert(isnan(b));
    else if (isinf(a)) assert(isinf(b) && !!signbit(a) == !!signbit(b));
    else assert(bits(a) == bits(b)); /* includes the sign of zero */
}
/* Keep this scalar operation order independent of the implementation under test. */
static vec2 project_reference(const double H[9], double u, double v) {
    double iz = 1.0 / (H[6] * u + H[7] * v + H[8]);
    vec2 p = {(H[0] * u + H[1] * v + H[2]) * iz, (H[3] * u + H[4] * v + H[5]) * iz};
    return p;
}
static double sample_reference(const image_view *im, double x, double y) {
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
static void canaries(const void *base, size_t item, int n) {
    const uint8_t *p = base;
    for (size_t i = 0; i < item; i++) assert(p[i] == 0xa5);
    for (size_t i = (size_t)(n + 1) * item; i < 70 * item; i++) assert(p[i] == 0xa5);
}
static void unit(double cells, int side, int c, double t, double *u, double *v) {
    double a = c + 0.5, t_ = t / cells, a_ = a / cells;
    *u = side == 0 ? a_ : side == 1 ? 1 - t_ : side == 2 ? a_ : t_;
    *v = side == 0 ? t_ : side == 1 ? a_ : side == 2 ? 1 - t_ : a_;
}
int main(void) {
    const int lengths[] = {0,1,2,3,7,16,31,32,33,68};
    const double special[] = {0.0,-0.0,-1000,1000,DBL_MIN,-DBL_MIN,DBL_MAX,-DBL_MAX,
                              INFINITY,-INFINITY,NAN,0x1p-1074,15.999,11.999};
    int cases = 0, points = 0, endpoints = 0;
    for (int padded = 0; padded < 2; padded++) for (int invert = 0; invert < 2; invert++) {
        uint8_t pixels[24 * 13], saved_pixels[24 * 13];
        for (size_t i = 0; i < sizeof pixels; i++) pixels[i] = (uint8_t)(rnd() >> 24);
        memcpy(saved_pixels,pixels,sizeof pixels);
        image_view im = {pixels,17,13,padded ? 24 : 17,invert};
        for (int t = 0; t < 64; t++) for (unsigned k = 0; k < sizeof lengths / sizeof lengths[0]; k++) {
            int n = lengths[k];
            double H[9] = {1,0,0,0,1,0,0,0,1}, saved_H[9];
            if (t == 1) for (int j = 0; j < 6; j++) H[j] = -0.0;
            if (t == 2) H[8] = 0;
            if (t == 3) H[0] = DBL_MAX, H[4] = -DBL_MAX, H[8] = DBL_MIN;
            if (t == 4) H[2] = NAN, H[5] = INFINITY;
            if (t == 5) H[6] = INFINITY, H[7] = -INFINITY;
            if (t >= 6) for (int j = 0; j < 9; j++) H[j] = ((int32_t)rnd()) / 1048576.0;
            memcpy(saved_H,H,sizeof H);
            double ub[70], vb[70], out[70], wrapped[70], saved_u[70], saved_v[70];
            vec2 raw[70];
            memset(ub,0xa5,sizeof ub); memset(vb,0xa5,sizeof vb);
            memset(out,0xa5,sizeof out); memset(wrapped,0xa5,sizeof wrapped); memset(raw,0xa5,sizeof raw);
            for (int j = 0; j < n; j++) {
                ub[j+1] = j < 14 ? special[j] : ((int32_t)rnd()) / 1048576.0;
                vb[j+1] = j < 14 ? special[(j+7)%14] : ((int32_t)rnd()) / 1048576.0;
            }
            /* Both coordinates +0 ensure the all-negative-zero homography returns raw -0. */
            if (n && t == 1) ub[1] = vb[1] = 0.0;
            memcpy(saved_u,ub,sizeof ub); memcpy(saved_v,vb,sizeof vb);
            sample_points_impl(&im,H,ub+1,vb+1,n,out+1,raw+1);
            sample_points(&im,H,ub+1,vb+1,n,wrapped+1);
            for (int j = 1; j <= n; j++) {
                vec2 p = project_reference(H,ub[j],vb[j]);
                double value = sample_reference(&im,p.x,p.y);
                same(raw[j].x,p.x); same(raw[j].y,p.y);
                same(out[j],value); same(wrapped[j],value); points++;
            }
            assert(!memcmp(ub,saved_u,sizeof ub) && !memcmp(vb,saved_v,sizeof vb));
            assert(!memcmp(H,saved_H,sizeof H) && !memcmp(pixels,saved_pixels,sizeof pixels));
            canaries(out,sizeof *out,n); canaries(wrapped,sizeof *wrapped,n); canaries(raw,sizeof *raw,n);
            cases++;
        }
    }
    for (int f = 0; f < NAPA_FAMILY_COUNT; f++) for (int side = 0; side < 4; side++)
        for (int c = 1; c < napa_families[f].width - 1; c++) {
            double u0,v0,u1,v1, cells = napa_families[f].width;
            unit(cells,side,c,1.5,&u0,&v0);
            unit(cells,side,c,-0.5+2.0*32/32,&u1,&v1);
            same(u0,u1); same(v0,v1); endpoints++;
        }
    printf("%d sampler cases, %d scalar-oracle points, %d family endpoint checks passed\n",cases,points,endpoints);
    return 0;
}
