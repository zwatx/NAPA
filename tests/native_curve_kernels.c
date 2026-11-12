/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
#include "../src/napa_vision/_native/napa_core.c"
#include <assert.h>
#include <stdio.h>

int main(void) {
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
