/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Fail one allocation in a named processing stage without changing its normal implementation. */
#include <stdlib.h>
#include <string.h>

static const char *fail_stage;
static int fail_calloc, fail_hits;

void napa_test_fail_allocation(const char *stage, int zeroed) {
    fail_stage = stage;
    fail_calloc = zeroed;
    fail_hits = 0;
}
int napa_test_allocation_hits(void) { return fail_hits; }

static int fail_here(const char *stage, int zeroed) {
    if (fail_stage && zeroed == fail_calloc && strcmp(stage, fail_stage) == 0) {
        fail_hits++;
        return 1;
    }
    return 0;
}
static void *stage_malloc(size_t bytes, const char *stage) {
    return fail_here(stage, 0) ? NULL : malloc(bytes);
}
static void *stage_calloc(size_t count, size_t bytes, const char *stage) {
    return fail_here(stage, 1) ? NULL : calloc(count, bytes);
}
#define malloc(bytes) stage_malloc(bytes, __func__)
#define calloc(count, bytes) stage_calloc(count, bytes, __func__)
#include "../src/napa_vision/_native/napa_core.c"
#undef malloc
#undef calloc
