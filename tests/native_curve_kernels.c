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

/* Analytic blurred edges: the unseen profile entries are deliberately unrelated to the image.
 * They cannot affect a supported crossing, its balance correction, or its derivative weight. */
static void check_observed_profiles(void) {
    const image_view im = {.w = 64, .h = 64};
    const vec2 bases[4] = {{2.2, 32}, {60.8, 32}, {32, 2.2}, {32, 60.8}};
    const vec2 normals[4] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    const double step = 10.0 / 14;
    for (int side = 0; side < 4; side++) for (int phase = -2; phase <= 2; phase++) {
        const double truth = phase * 0.2;
        double first_position = 0, first_weight = 0;
        for (int poison = 0; poison < 2; poison++) {
            int64_t prof[15];
            for (int j = 0; j < 15; j++) {
                const double o = -5 + j * step;
                const double x = bases[side].x + normals[side].x * o, y = bases[side].y + normals[side].y * o;
                double value = 20 + 100 * erfc((o - truth) / (sqrt(2.0) * 0.8));
                if (x < 0 || y < 0 || x > 63 || y > 63) value = ((j + poison) & 1) ? 255 : 0;
                prof[j] = (int64_t)(value * 16777216.0 + 0.5);
            }
            double position, weight;
            assert(observed_profile_edge(&im, prof, 15, bases[side], normals[side], 5, step, 120, 20,
                                         &position, &weight));
            assert(fabs(position - truth) < 0.04 && weight > 0);
            if (!poison) first_position = position, first_weight = weight;
            else assert(position == first_position && weight == first_weight);
        }
    }
    int64_t prof[11];
    for (int j = 0; j < 11; j++) prof[j] = (int64_t)((20 + 100 * erfc((j - 5) / sqrt(2.0))) * 16777216.0 + 0.5);
    double position, weight;
    /* A boundary pixel is observed, but a bracket without the symmetric balance window is insufficient. */
    assert(sample_is_observed(&im, 63, 32));
    assert(!observed_profile_edge(&im, prof, 11, (vec2){63, 32}, (vec2){-1, 0}, 5, 1, 120, 20, &position, &weight));
    assert(observed_profile_edge(&im, prof, 11, (vec2){62, 32}, (vec2){-1, 0}, 5, 1, 120, 20, &position, &weight));
    assert(position == 0);
    assert(!observed_profile_edge(&im, prof, 11, (vec2){64.5, 32}, (vec2){-1, 0}, 5, 1, 120, 20, &position, &weight));
    for (int j = 0; j < 11; j++) prof[j] = (int64_t)((100 + erfc((j - 5) / sqrt(2.0))) * 16777216.0 + 0.5);
    assert(!observed_profile_edge(&im, prof, 11, (vec2){60.8, 32}, (vec2){-1, 0}, 5, 1, 101, 20, &position, &weight));
}

/* Independent order-statistic definition, including ties, for odd and even observed counts. */
static double visible_median_oracle(const double *values, int n) {
    double ranks[2] = {0, 0};
    for (int i = 0; i < n; i++) {
        int below = 0, equal = 0;
        for (int j = 0; j < n; j++) below += values[j] < values[i], equal += values[j] == values[i];
        const int wanted[2] = {(n - 1) / 2, n / 2};
        for (int k = 0; k < 2; k++) if (below <= wanted[k] && wanted[k] < below + equal) ranks[k] = values[i];
    }
    return 0.5 * (ranks[0] + ranks[1]);
}

static void check_visible_light(void) {
    napa_detector *det = napa_create();
    assert(det != NULL);
    const family_layout *L = &det->fam;
    uint8_t pixels[64 * 64];
    memset(pixels, 90, sizeof pixels);
    const image_view im = {pixels, 64, 64, 64, 0, NULL};
    const vec2 q[4] = {{-4, 13}, {36, 17}, {39, 57}, {-1, 54}};
    double H[9], border[64], ring[68], observed_black[64], observed_white[68];
    square_to_quad(q, H);
    edge_levels reference = {0};
    int odd_seen = 0;
    for (int poison = 0; poison < 2; poison++) {
        int bi = 0, wi = 0, nb = 0, nw = 0;
        const int W = L->width;
        for (int r = 0; r < W; r++) for (int c = 0; c < W; c += (r == 0 || r == W - 1) ? 1 : W - 1) {
            const vec2 pos = project(H, (c + 0.5) / L->cells, (r + 0.5) / L->cells);
            const double value = 20 * (1 + 0.001 * pos.x + 0.002 * pos.y);
            if (pos.x >= 0 && pos.y >= 0 && pos.x <= 63 && pos.y <= 63)
                observed_black[nb++] = border[bi++] = value;
            else border[bi++] = poison ? 255 : 0;
        }
        for (int group = 0; group < 2; group++) {
            const int begin = group ? 1 : 0, end = group ? W + 1 : W + 2;
            for (int i = begin; i < end; i++) for (int side = 0; side < 2; side++) {
                const double a = (i - 0.5) / L->cells, b = (side ? W + 0.5 : -0.5) / L->cells;
                const vec2 pos = project(H, group ? b : a, group ? a : b);
                const double value = 200 * (1 + 0.001 * pos.x + 0.002 * pos.y);
                if (pos.x >= 0 && pos.y >= 0 && pos.x <= 63 && pos.y <= 63)
                    observed_white[nw++] = ring[wi++] = value;
                else ring[wi++] = poison ? 255 : 0;
            }
        }
        odd_seen |= (nb & 1) || (nw & 1);
        edge_levels levels = {0};
        edge_light_model(L, H, q, ring, border, &levels, &im);
        assert(levels.has_model);
        assert(levels.black == visible_median_oracle(observed_black, nb));
        assert(levels.white == visible_median_oracle(observed_white, nw));
        for (int i = 0; i < 4; i++) {
            const double mid = levels.midpoint[0] + levels.midpoint[1] * (q[i].x - levels.cx) / levels.scale +
                               levels.midpoint[2] * (q[i].y - levels.cy) / levels.scale;
            assert(fabs(mid - 110 * (1 + 0.001 * q[i].x + 0.002 * q[i].y)) < 1e-8);
        }
        if (!poison) reference = levels;
        else {
            assert(levels.black == reference.black && levels.white == reference.white);
            for (int i = 0; i < 3; i++) assert(levels.midpoint[i] == reference.midpoint[i]);
        }
    }
    assert(odd_seen);
    /* An uninformative visible image must not retain an earlier clamped illumination model. */
    reference.model_family = L;
    reference.model_pending = reference.has_model = 1;
    refresh_edge_light(&im, q, &reference, 1);
    assert(!reference.has_model && !reference.model_pending && reference.black >= reference.white);
    /* Most virtual cells clamp onto the dark frame boundary, hiding the contrast of the three black
     * and five white cell centers actually visible in this corner. The visible fit uses those cells. */
    for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++)
        pixels[y * 64 + x] = (uint8_t)((x >= 62 || y >= 62) ? 0 : x >= 50 && y >= 50 ? 20 : 200);
    const vec2 mostly_outside[4] = {{50, 50}, {106, 50}, {106, 106}, {50, 106}};
    double aggregate_grid[256], aggregate_border[64], aggregate_black, aggregate_white;
    square_to_quad(mostly_outside, H);
    (void)read_border(L, &im, H, 0, aggregate_grid, aggregate_border, &aggregate_black, &aggregate_white, NULL);
    assert(aggregate_white <= aggregate_black);
    refresh_edge_light(&im, mostly_outside, &reference, 1);
    assert(reference.has_model && !reference.model_pending);
    assert(reference.black == 20 && reference.white == 200);
    napa_destroy(det);
}

int main(void) {
    check_observed_profiles();
    check_visible_light();
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
