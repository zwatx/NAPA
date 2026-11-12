/* Public detection on exact-size read-only inputs must match padded owned input.
 * Vertical contrast forces the final threshold bands through their word loads. */
#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#define HAVE_GUARD_PAGES 1
#endif
#ifndef NAPA_TEST_SOURCE
#define NAPA_TEST_SOURCE "../src/napa_vision/_native/napa_core.c"
#endif
#include NAPA_TEST_SOURCE

int main(void) {
    const int widths[] = {16, 17, 21, 31, 33, 63, 64, 65, 191, 192, 193, 30000};
    napa_detector *detector = napa_create_family("tag36h11");
    assert(detector);
    int cases = 0, guards = 0;
    for (size_t i = 0; i < sizeof widths / sizeof widths[0]; i++) {
        int w = widths[i], h = 19, stride = w + 19;
        size_t bytes = (size_t)w * h;
        uint8_t *input, *allocation;
#ifdef HAVE_GUARD_PAGES
        long page = sysconf(_SC_PAGESIZE);
        assert(page > 0);
        size_t span = (bytes + (size_t)page - 1) / (size_t)page * (size_t)page;
        allocation = mmap(NULL, span + (size_t)page, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(allocation != MAP_FAILED);
        input = allocation + span - bytes;
#else
        allocation = malloc(bytes);
        assert(allocation);
        input = allocation;
#endif
        uint8_t *padded = malloc((size_t)stride * h), *saved = malloc(bytes);
        assert(padded && saved);
        memset(padded, 177, (size_t)stride * h);
        for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
            uint8_t value = x < w / 2 ? 20 : 230;
            input[(size_t)y * w + x] = value;
            padded[(size_t)y * stride + x] = value;
        }
        memcpy(saved, input, bytes);
#ifdef HAVE_GUARD_PAGES
        assert(mprotect(allocation, span, PROT_READ) == 0);
        assert(mprotect(allocation + span, (size_t)page, PROT_NONE) == 0);
        guards++;
#endif
        for (int minimum = 16; minimum <= 32; minimum *= 2) for (int split = 0; split <= 1; split++) {
            napa_params params = {minimum, 0, 10, 20, 25, split};
            napa_detection output[16];
            assert(napa_detect(detector, input, w, h, w, &params, output, 16) == 0);
            assert(napa_detect(detector, padded, w, h, stride, &params, output, 16) == 0);
            assert(memcmp(input, saved, bytes) == 0);
            cases++;
        }
        free(saved);
        free(padded);
#ifdef HAVE_GUARD_PAGES
        assert(munmap(allocation, span + (size_t)page) == 0);
#else
        free(allocation);
#endif
    }
    /* Context destruction follows release of every caller buffer. */
    napa_destroy(detector);
    printf("{\"status\":\"PASS\",\"cases\":%d,\"guarded_inputs\":%d}\n", cases, guards);
    return 0;
}
