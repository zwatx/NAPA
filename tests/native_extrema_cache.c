/* Recovery reuses the ordinary pass's tile extrema only where both passes sampled the same values. On a noisy frame
 * the passes box-average different levels, so each recovery level must threshold exactly as it does without the
 * saved extrema, while steps both passes sampled the same way must still be reused. Includes the core for its
 * private sampling loop. Run through test_native_speed_kernels.py. */
#include "../src/napa_vision/_native/napa_core.c"
#include <assert.h>
#include <stdio.h>

enum { W = 1280, H = 960, MAX_OUT = 256, MAX_LEVELS = 16 };

static uint32_t next_value(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state >> 8;
}

/* 48 low-contrast tag36h11 tags of 16-80 pixels on a dim gradient. The noise is the sum of four uniform values,
 * with a standard deviation of about `noise` grey levels, in fixed-width integers so every platform draws it alike. */
static void noisy_frame(uint8_t *image, int noise, uint32_t seed) {
    napa_family_layout layout;
    assert(napa_family_info("tag36h11", &layout) == 0);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) image[y * W + x] = (uint8_t)(90 + 40 * x / W);
    for (int id = 0; id < 48; id++) {
        int cell = 2 + id % 9, black = 20 + 8 * (id % 7), white = black + 25 + 10 * (id % 5);
        int ox = 20 + id % 8 * 155, oy = 20 + id / 8 * 155;
        uint64_t code;
        assert(napa_family_code("tag36h11", id, &code) == 0);
        for (int cy = -1; cy < 9; cy++) for (int cx = -1; cx < 9; cx++) {
            int v = cx >= 0 && cy >= 0 && cx < 8 && cy < 8 ? black : white;
            for (int k = 0; k < layout.nbits; k++)
                if (layout.x[k] == cx && layout.y[k] == cy) v = (code >> (layout.nbits - 1 - k)) & 1 ? white : black;
            for (int y = 0; y < cell; y++) for (int x = 0; x < cell; x++) {
                int px = ox + (cx + 1) * cell + x, py = oy + (cy + 1) * cell + y;
                if (px < W && py < H) image[py * W + px] = (uint8_t)v;
            }
        }
    }
    const int a = (7 * noise + 4) / 8; /* four uniform values in [-a, a] have a standard deviation of about 1.15 a */
    for (int i = 0; i < W * H; i++) {
        int v = image[i];
        for (int k = 0; k < 4; k++) v += (int)(next_value(&seed) % (uint32_t)(2 * a + 1)) - a;
        image[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}

static napa_params recovery_params(const napa_params *params, int shadow_px) {
    napa_params recovery = *params;
    if (shadow_px) recovery.min_tag_px = shadow_px;
    recovery.max_hamming = 0, recovery.split_touching = 0;
    return recovery;
}

/* napa_detect_shadows_configured with nothing saved for recovery to reuse. */
static int detect_uncached(napa_detector *det, const uint8_t *image, const napa_params *params, int shadow_px,
                           napa_detection *out) {
    int n = napa_detect(det, image, W, H, W, params, out, MAX_OUT);
    assert(n >= 0 && n < MAX_OUT);
    napa_params recovery = recovery_params(params, shadow_px);
    det->extrema_ready = 0;
    return detect_pass(det, image, W, H, W, &recovery, out, MAX_OUT, n, 1);
}

typedef struct {
    int s, nruns, boxed;
    uint64_t hash;
} level_digest;

/* Sample and threshold every recovery level as detect_pass does, recording each level's dark runs. */
static int recovery_levels(napa_detector *det, const uint8_t *image, const napa_params *recovery, level_digest *digest) {
    const int cells = det->fam.width;
    int base = recovery->min_tag_px * 3 / (4 * cells);
    if (base < 1) base = 1;
    const pass_config config = {.view = {image, W, H, W, 0, NULL}, .params = recovery, .pimg = image,
                                .pw = W, .ph = H, .pstride = W, .base = base, .shadow = 1};
    pass_outputs outputs = {0};
    level_sampling sampling = {NULL, 0, base, recovery->min_contrast, 0};
    pass_timer timer;
    int n = 0;
    for (int s = base;; s *= 2) {
        candidate_level level = {0};
        level.s = s, level.lw = (W + s - 1) / s, level.lh = (H + s - 1) / s;
        level.last = level.lw < NAPA_LAST * cells / 8 || level.lh < NAPA_LAST * cells / 8;
        assert(n < MAX_LEVELS && sample_threshold_level(det, &config, &outputs, &sampling, &level, &timer) == 0);
        uint64_t hash = 1469598103934665603u;
        for (int i = 0; i < level.nruns; i++) {
            const napa_run *r = &det->runs[i];
            hash = (hash ^ ((uint64_t)r->x0 | (uint64_t)r->x1 << 16 | (uint64_t)r->y << 32)) * 1099511628211u;
        }
        digest[n++] = (level_digest){s, level.nruns, sampling.boxed, hash};
        if (level.last) return n;
    }
}

static int same_level(level_digest a, level_digest b) {
    return a.s == b.s && a.nruns == b.nruns && a.hash == b.hash;
}

static void same_detections(const napa_detection *a, int na, const napa_detection *b, int nb) {
    assert(na >= 0 && na == nb);
    for (int i = 0; i < na; i++) {
        assert(a[i].id == b[i].id && a[i].hamming == b[i].hamming);
        assert(a[i].decision_margin == b[i].decision_margin);
        assert(memcmp(a[i].corners, b[i].corners, sizeof a[i].corners) == 0);
    }
}

typedef enum { REUSE_ALL, REUSE_NONE } reuse_expected;

/* Returns the number of recovery levels that read saved extrema. */
static int check_case(napa_detector *det, const uint8_t *image, int min_px, int shadow_px, reuse_expected expected) {
    static napa_detection cached[MAX_OUT], fresh[MAX_OUT];
    const napa_params params = {min_px, 2, 20, 22, 25, 0};
    int nc = napa_detect_shadows_configured(det, image, W, H, W, &params, cached, MAX_OUT, shadow_px);
    int nf = detect_uncached(det, image, &params, shadow_px, fresh);
    same_detections(cached, nc, fresh, nf);

    /* Saved again by an ordinary pass, then every recovery level with, without, and with poisoned saved extrema. */
    assert(napa_detect_shadows_configured(det, image, W, H, W, &params, cached, MAX_OUT, shadow_px) == nc);
    const napa_params recovery = recovery_params(&params, shadow_px);
    level_digest with[MAX_LEVELS], without[MAX_LEVELS], poisoned[MAX_LEVELS];
    int levels = recovery_levels(det, image, &recovery, with);
    det->extrema_ready = 0;
    assert(recovery_levels(det, image, &recovery, without) == levels);
    det->extrema_ready = 1;
    /* Where reuse would be wrong, recovery averages from its first step on: its samples differ at every step. */
    assert(expected == REUSE_ALL || with[0].boxed);
    int reused = 0;
    for (int k = 0; k < levels; k++) {
        assert(same_level(with[k], without[k]));
        saved_extrema *saved = &det->saved[with[k].s];
        if (!saved->valid) {
            /* An ordinary pass from step two skips step four (detect_pass), so recovery thresholds that step itself. */
            assert(with[k].s == 4 && min_px * 3 / (4 * 8) == 2);
            continue;
        }
        /* Full-range extrema threshold every tile at mid-grey: a level that reads them labels other runs. */
        const int lw = (W + with[k].s - 1) / with[k].s, lh = (H + with[k].s - 1) / with[k].s;
        const size_t tiles = (size_t)((lw + TILE - 1) / TILE) * ((lh + TILE - 1) / TILE);
        uint8_t *copy = (uint8_t *)malloc(2 * tiles);
        assert(copy != NULL);
        memcpy(copy, saved->tiles, 2 * tiles);
        memset(saved->tiles, 0, tiles), memset(saved->tiles + tiles, 255, tiles);
        assert(recovery_levels(det, image, &recovery, poisoned) == levels);
        memcpy(saved->tiles, copy, 2 * tiles);
        free(copy);
        int read = !same_level(poisoned[k], with[k]);
        for (int j = 0; j < levels; j++) if (j != k) assert(same_level(poisoned[j], with[j]));
        assert(read == (expected == REUSE_ALL));
        reused += read;
    }
    return reused;
}

int main(void) {
    uint8_t *noisy = (uint8_t *)malloc(W * H), *clean = (uint8_t *)malloc(W * H);
    assert(noisy != NULL && clean != NULL);
    noisy_frame(noisy, 12, 1);
    noisy_frame(clean, 2, 2);
    int cases = 0, reused = 0;
    for (int threads = 1; threads <= 4; threads += 3) {
        napa_detector *det = napa_create();
        assert(det != NULL && napa_set_threads(det, threads) == 0);
        /* The same first step repeats every boxing decision, including the box-averaged step two itself. */
        reused += check_case(det, noisy, 16, 0, REUSE_ALL);
        assert(det->saved[2].boxed && !det->saved[1].boxed && det->saved[4].boxed);
        reused += check_case(det, noisy, 22, 0, REUSE_ALL);
        assert(det->saved[2].boxed);
        /* Steps one and two against two: the ordinary pass averages 3x3 blocks at step two, recovery 2x2 blocks. */
        reused += check_case(det, noisy, 16, 22, REUSE_NONE);
        reused += check_case(det, noisy, 8, 22, REUSE_NONE);
        /* Steps two and four, each averaging its own blocks. */
        reused += check_case(det, noisy, 22, 43, REUSE_NONE);
        /* Nothing averaged: both passes took every s-th pixel, so the shared steps are reused. */
        reused += check_case(det, clean, 16, 22, REUSE_ALL);
        assert(!det->saved[1].boxed && !det->saved[2].boxed);
        reused += check_case(det, clean, 22, 43, REUSE_ALL);
        cases += 7;
        napa_destroy(det);
    }
    free(noisy), free(clean);
    printf("%d recovery cases at 1 and 4 threads match fresh thresholds, %d saved levels reused\n", cases, reused);
    return 0;
}
