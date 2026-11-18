/* nearest_code against a brute-force search of every rotated code: the nearest code within max_bits, the first in
 * table order on a tie, at every max_bits decode_quad asks for (0-5). Short families are read with every pattern of
 * up to three wrong bits, the rest with up to one or two and some random reads of three to five. nearest_code stops
 * at a code within max_hamming, which is right only if codes are min_distance apart: that is checked where quick.
 * Run through test_native_speed_kernels.py. */
#include "../src/napa_vision/_native/napa_core.c"
#include <assert.h>
#include <stdio.h>

typedef struct {
    napa_detector *det;
    long reads, beyond, ties;
} checker;

static void check_read(checker *c, uint64_t code) {
    const family_layout *L = &c->det->fam;
    int want = -1, want_d = 65, at_best = 0;
    for (int i = 0; i < L->nrot; i++) {
        int d = napa_popcount64(code ^ L->codes[i]);
        if (d < want_d) want = i, want_d = d, at_best = 1;
        else if (d == want_d) at_best++;
    }
    for (int max_bits = 0; max_bits <= 5; max_bits++) {
        int d = -1, got = nearest_code(c->det, code, max_bits, &d);
        if (want_d > max_bits) assert(got == -1);
        else assert(got == want && d == want_d);
    }
    c->reads++;
    c->beyond += want_d > L->max_hamming && want_d <= 5;
    c->ties += at_best > 1 && want_d <= 5;
}

/* Every read of `code` with up to `left` more wrong bits, each set of bits once. */
static void check_flips(checker *c, uint64_t code, int from, int left) {
    check_read(c, code);
    if (left > 0)
        for (int k = from; k < c->det->fam.nbits; k++) check_flips(c, code ^ (uint64_t)1 << k, k + 1, left - 1);
}

static int min_distance(const family_layout *L) {
    int least = 64;
    for (int i = 0; i < L->nrot; i++)
        for (int j = i + 1; j < L->nrot; j++) {
            int d = napa_popcount64(L->codes[i] ^ L->codes[j]);
            least = d < least ? d : least;
        }
    return least;
}

int main(void) {
    long reads = 0, beyond = 0, ties = 0;
    uint32_t seed = 12345;
    for (int f = 0; f < napa_family_count(); f++) {
        napa_detector *det = napa_create_family(napa_family_name(f));
        assert(det != NULL);
        const family_layout *L = &det->fam;
        checker c = {det, 0, 0, 0};
        if (L->nrot <= 10000) assert(min_distance(L) >= L->f->min_distance);
        if (L->nbits <= 25) {
            for (int i = 0; i < L->nrot; i++) check_flips(&c, L->codes[i], 0, 3);
        } else {
            const int sample = L->nrot <= 10000 ? 16 : 2, randoms = L->nrot <= 10000 ? 4000 : 100;
            for (int i = 0; i < L->nrot; i += L->nrot / sample) check_flips(&c, L->codes[i], 0, L->nrot <= 10000 ? 2 : 1);
            for (int r = 0; r < randoms; r++) {
                seed = seed * 1664525u + 1013904223u;
                const uint64_t code = L->codes[(seed >> 8) % (uint32_t)L->nrot];
                uint64_t wrong = 0;
                for (int left = 3 + r % 3; left > 0;) {
                    seed = seed * 1664525u + 1013904223u;
                    uint64_t bit = (uint64_t)1 << ((seed >> 8) % (uint32_t)L->nbits);
                    if (!(wrong & bit)) wrong |= bit, left--;
                }
                check_read(&c, code ^ wrong);
            }
        }
        reads += c.reads, beyond += c.beyond, ties += c.ties;
        napa_destroy(det);
    }
    /* Reads nearer no code than max_hamming, and ties, are the cases the first match used to get wrong. */
    assert(beyond > 0 && ties > 0);
    printf("%ld reads match the nearest rotated code at max_bits 0-5 (%ld past max_hamming, %ld ties)\n", reads, beyond, ties);
    return 0;
}
