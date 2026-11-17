/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* shared_lens_quad on known geometry: four straight sides drawn through the radial map F(x) = c + (x - c)(1 + k |x - c|^2
 * / s^2), pixel noise, one corner past the frame. Straight sides must never yield a lens; real ones must give the corners. */
#include "../src/napa_vision/_native/napa_core.c"
#include <assert.h>
#include <stdio.h>

enum { W = 640, H = 480 };
static const double CX = 0.5 * W, CY = 0.5 * H, S = 0.5 * W;
static uint64_t rng;
static double uni(void) {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return (rng >> 11) * (1.0 / 9007199254740992.0);
}
static double gauss(void) {
    const double u = uni(), v = uni();
    return sqrt(-2 * log(u > 1e-300 ? u : 1e-300)) * cos(6.283185307179586 * v);
}
/* The pixel F maps to p, by bisection where r (1 + k r^2 / s^2) rises. */
static vec2 unbend(vec2 p, double k) {
    const double qx = p.x - CX, qy = p.y - CY, rq = hypot(qx, qy);
    double lo = 0, hi = k < 0 ? S / sqrt(-3 * k) : rq;
    for (int i = 0; i < 100; i++) {
        const double mid = 0.5 * (lo + hi);
        if (mid * (1 + k * mid * mid / (S * S)) < rq) lo = mid; else hi = mid;
    }
    const double f = rq > 0 ? 0.5 * (lo + hi) / rq : 1;
    return (vec2){CX + qx * f, CY + qy * f};
}
static int inside(vec2 p) { return p.x >= 0 && p.y >= 0 && p.x <= W - 1 && p.y <= H - 1; }

/* Returns how many of `trials` tags gave a lens; *worst is the largest corner error among them. */
static int run(uint64_t seed, int trials, double k, double smin, double smax, double sigma, double *worst) {
    const image_view im = {.w = W, .h = H, .stride = W};
    int accepted = 0;
    rng = 0x9E3779B97F4A7C15ull ^ seed * 0xD1B54A32D192ED03ull;
    *worst = 0;
    for (int t = 0; t < trials; t++) {
        const double side = smin + (smax - smin) * uni(), turn = 0.12 + 0.43 * uni(), c = cos(turn), s = sin(turn);
        const double over = 0.3 + 1.7 * uni(), along = 0.35 + 0.3 * uni();
        const int edge = (int)(uni() * 4);
        vec2 flat[4] = {{0, 0}, {side * c, side * s}, {side * (c - s), side * (s + c)}, {-side * s, side * c}}, truth[4], in[4];
        /* Shift the straightened square until its extreme pixel corner overhangs the chosen frame edge by `over`. */
        double sx = along * W - 0.5 * side * (c - s), sy = along * H - 0.5 * side * (s + c);
        for (int it = 0; it < 30; it++) {
            double lo = HUGE_VAL, hi = -HUGE_VAL;
            for (int q = 0; q < 4; q++) {
                const vec2 p = unbend((vec2){flat[q].x + sx, flat[q].y + sy}, k);
                lo = fmin(lo, edge < 2 ? p.x : p.y), hi = fmax(hi, edge < 2 ? p.x : p.y);
            }
            if (edge == 0) sx -= over + lo;
            else if (edge == 1) sx += W - 1 + over - hi;
            else if (edge == 2) sy -= over + lo;
            else sy += H - 1 + over - hi;
        }
        int side_clear[4];
        for (int q = 0; q < 4; q++) flat[q].x += sx, flat[q].y += sy, truth[q] = unbend(flat[q], k);
        for (int q = 0; q < 4; q++) {
            side_clear[q] = inside(truth[q]) && inside(truth[(q + 1) & 3]);
            in[q] = (vec2){truth[q].x + 0.3 * gauss(), truth[q].y + 0.3 * gauss()};
        }
        side_samples samples;
        for (int q = 0; q < 4; q++) {
            const vec2 a = flat[q], b = flat[(q + 1) & 3];
            const int n = (int)fmin(0.65 * side, NAPA_SIDE_SAMPLES);
            samples.n[q] = 0;
            for (int i = 0; i < n; i++) {
                const double u = 0.05 + 0.9 * (i + 0.5) / n;
                vec2 p = unbend((vec2){a.x + u * (b.x - a.x), a.y + u * (b.y - a.y)}, k);
                p.x += sigma * gauss(), p.y += sigma * gauss();
                if (!inside(p)) continue;
                const int j = samples.n[q]++;
                samples.x[q][j] = p.x, samples.y[q][j] = p.y, samples.w[q][j] = 1;
            }
        }
        vec2 out[4];
        if (!shared_lens_quad(&im, &samples, side_clear, in, 2.0, out)) continue;
        accepted++;
        for (int q = 0; q < 4; q++) *worst = fmax(*worst, hypot(out[q].x - truth[q].x, out[q].y - truth[q].y));
    }
    return accepted;
}

int main(void) {
    double worst;
    /* Straight sides: noise on small and mid-size tags once passed for a lens and moved corners by up to 70 px. */
    const double sigmas[3] = {0.05, 0.1, 0.2};
    for (int i = 0; i < 3; i++) {
        assert(run(21 + i, 400, 0, 20, 40, sigmas[i], &worst) == 0);
        assert(run(31 + i, 400, 0, 40, 80, sigmas[i], &worst) == 0);
    }
    /* Barrel and pincushion bows on large tags: most give a lens, and every one puts the corners where the lens does. */
    const double bows[3] = {0.2, 0.1, -0.1};
    for (int i = 0; i < 3; i++) {
        const int accepted = run(41 + i, 200, bows[i], 120, 136, 0.08, &worst);
        printf("bow %+.1f: %d/200 within %.3f px\n", bows[i], accepted, worst);
        assert(accepted >= 160 && worst < 0.2);
    }
    printf("shared lens estimate known-geometry cases passed\n");
    return 0;
}
