/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
#include "../src/napa_vision/_native/napa_core.c"
#include <assert.h>
#include <stdio.h>

/* Fit both means and sum squared residuals independently for every split. */
static int profile_residual_oracle(const int64_t p[11], double min_drop, int *chosen_split) {
    const double unit = 16777216.0;
    double least = INFINITY;
    *chosen_split = -1;
    for (int split = 1; split < 11; split++) {
        double white = 0, black = 0;
        for (int i = 0; i < split; i++) white += p[i] / unit;
        for (int i = split; i < 11; i++) black += p[i] / unit;
        white /= split;
        black /= 11 - split;
        if (white <= black || white - black < min_drop / unit) continue;
        double residual = 0;
        for (int i = 0; i < 11; i++) {
            double error = p[i] / unit - (i < split ? white : black);
            residual += error * error;
        }
        if (residual < least) least = residual, *chosen_split = split;
    }
    if (*chosen_split < 0) return -1;
    int peak = -1;
    int64_t largest = INT64_MIN;
    for (int j = 1; j < 10; j++) {
        if (!(j == *chosen_split || j + 1 == *chosen_split)) continue;
        int64_t derivative = p[j - 1] - p[j + 1];
        if (derivative > largest) largest = derivative, peak = j;
    }
    return peak;
}

static int check_profile_against_residuals(const int values[11], double min_g, int *split) {
    int64_t p[11], g[11] = {0};
    for (int i = 0; i < 11; i++) p[i] = (int64_t)values[i] * INT64_C(16777216);
    for (int j = 1; j < 10; j++) g[j] = p[j - 1] - p[j + 1];
    int expected = profile_residual_oracle(p, min_g, split);
    int actual = initial_plateau_peak(p, g, 11, min_g);
    assert(actual == expected);
    return actual;
}

static void check_initial_plateau_peak(void) {
    const double unit = 16777216.0;
    int p[11], split;
    /* Both end splits have only one adjacent central derivative. */
    for (int i = 0; i < 11; i++) p[i] = i == 0 ? 140 : 100;
    assert(check_profile_against_residuals(p, 10 * unit, &split) == 1 && split == 1);
    for (int i = 0; i < 11; i++) p[i] = i == 10 ? 100 : 140;
    assert(check_profile_against_residuals(p, 10 * unit, &split) == 9 && split == 10);

    /* Equality passes; one fixed-point unit above the measured step does not. */
    for (int i = 0; i < 11; i++) p[i] = i < 6 ? 140 : 100;
    assert(check_profile_against_residuals(p, 40 * unit, &split) == 5 && split == 6);
    assert(check_profile_against_residuals(p, 40 * unit + 1, &split) == -1);

    /* A strong plateau fit does not bypass the caller's derivative-strength gate.
     * In particular, split1 must not borrow the stronger nonadjacent derivative2. */
    const int boundary[11] = {197, 74, 195, 19, 185, 125, 127, 125, 149, 92, 97};
    int peak = check_profile_against_residuals(boundary, 10 * unit, &split);
    assert(split == 1 && peak == 1);
    assert((boundary[peak - 1] - boundary[peak + 1]) * unit < 10 * unit);
    for (int i = 0; i < 11; i++) p[i] = 140 - 4 * i;
    peak = check_profile_against_residuals(p, 20 * unit, &split);
    assert(peak >= 1 && (p[peak - 1] - p[peak + 1]) * unit < 20 * unit);

    for (int i = 0; i < 11; i++) p[i] = 100;
    assert(check_profile_against_residuals(p, 10 * unit, &split) == -1);
    for (int i = 0; i < 11; i++) p[i] = i < 6 ? 100 : 140;
    assert(check_profile_against_residuals(p, 10 * unit, &split) == -1);

    /* General profiles exercise the objective, not a universal notch-rejection rule. */
    uint32_t state = UINT32_C(0x12345678);
    for (int trial = 0; trial < 512; trial++) {
        for (int i = 0; i < 11; i++) {
            state = UINT32_C(1664525) * state + UINT32_C(1013904223);
            p[i] = (int)(state >> 24);
        }
        (void)check_profile_against_residuals(p, (trial % 4 + 1) * 5 * unit, &split);
    }
}

static void check_x_corners(void) {
    uint8_t pixels[64 * 64];
    image_view view = {pixels, 64, 64, 64, 0, NULL};
    for (int turn = 0; turn < 4; turn++) for (int phase = 0; phase < 4; phase++) {
        const double angle = 0.3 * turn, tx = 31 + 0.25 * phase, ty = 32.75 - 0.25 * phase;
        /* Gaussian-blurred crossing of two orthogonal edges, integrated over each pixel.
         * Its intersection is (tx, ty), independently of the gradient fit under test. */
        for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) {
            double sum = 0;
            for (int sy = 0; sy < 4; sy++) for (int sx = 0; sx < 4; sx++) {
                double ox = x + (sx + 0.5) / 4 - 0.5 - tx, oy = y + (sy + 0.5) / 4 - 0.5 - ty;
                double u = cos(angle) * ox + sin(angle) * oy, v = -sin(angle) * ox + cos(angle) * oy;
                sum += 127.5 + 110 * erf(u / (sqrt(2.0) * 0.7)) * erf(v / (sqrt(2.0) * 0.7));
            }
            pixels[y * 64 + x] = (uint8_t)floor(sum / 16 + 0.5);
        }
        for (int invert = 0; invert < 2; invert++) for (int start = 0; start < 4; start++) {
            double x = tx + (start & 1 ? 0.8 : -0.8), y = ty + (start & 2 ? 0.8 : -0.8);
            view.invert = invert;
            assert(refine_x_corner(&view, &x, &y));
            assert(hypot(x - tx, y - ty) < 0.1);
        }
    }
    /* Flat patches and a lone edge do not determine a corner, and must leave the input unchanged. */
    view.invert = 0;
    for (int edge = 0; edge < 2; edge++) {
        for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++)
            pixels[y * 64 + x] = (uint8_t)(edge && x < 32 ? 0 : 255);
        double x = 31.5, y = 32.5;
        assert(!refine_x_corner(&view, &x, &y));
        assert(x == 31.5 && y == 32.5);
    }
}

int main(void) {
    check_initial_plateau_peak();
    check_x_corners();
    double px[48], py[48], wt[48];
    curve_side side;
    vec2 a = {0, 0}, b = {200, 0};
    for (int i = 0; i < 48; i++) {
        double u = -0.38 + 0.76 * (i + 0.5) / 48;
        px[i] = 200 * (u + 0.5);
        py[i] = 20 * u * u;
        wt[i] = 1;
    }
    fit_curve_side(&side, a, b, px, py, wt, 48);
    assert(side.qualified && fabs(side.c[2] - 20) < 1e-8);

    for (int i = 0; i < 48; i++) py[i] = 0.02 * ((i * 17) % 11 - 5);
    fit_curve_side(&side, a, b, px, py, wt, 48);
    assert(!side.qualified);
    for (int i = 0; i < 48; i++) py[i] = 3;
    fit_curve_side(&side, a, b, px, py, wt, 48);
    assert(!side.qualified);

    for (int i = 0; i < 48; i++) {
        double u = px[i] / 200 - 0.5;
        py[i] = 20 * u * u;
        wt[i] = i < 24 ? 1e-12 : 1;
    }
    fit_curve_side(&side, a, b, px, py, wt, 48);
    assert(!side.qualified); /* a nearly weightless half cannot support an extrapolated endpoint */
    for (int i = 0; i < 48; i++) wt[i] = 1;
    fit_curve_side(&side, a, b, px, py, wt, 11);
    assert(!side.qualified);

    curve_side sides[4];
    memset(sides, 0, sizeof sides);
    fit_curve_side(&sides[0], a, b, px, py, wt, 48);
    vec2 input[4] = {{0, 0}, {200, 0}, {200, 200}, {0, 200}}, result[4];
    double lx[4] = {100, 200, 100, 0}, ly[4] = {0, 100, 200, 100};
    double dx[4] = {1, 0, -1, 0}, dy[4] = {0, 1, 0, -1};
    assert(intersect_curves(sides, input, input, lx, ly, dx, dy, 1, result));
    /* The quadratic y=20*(x/200-0.5)^2 meets the unchanged vertical sides at y=5. */
    assert(fabs(result[0].y - 5) < 1e-6 && fabs(result[1].y - 5) < 1e-6 && result[2].y == 200);
    assert(!intersect_curves(sides, input, input, lx, ly, dx, dy, 0.1, result));
    dx[3] = 1; dy[3] = 0;
    assert(!intersect_curves(sides, input, input, lx, ly, dx, dy, 1, result));
    dx[3] = 0; dy[3] = -1;
    sides[0].c[2] = NAN;
    assert(!intersect_curves(sides, input, input, lx, ly, dx, dy, 1, result));
    puts("curved-edge numerical guard cases passed");
    return 0;
}
