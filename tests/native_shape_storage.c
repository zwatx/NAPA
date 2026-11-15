/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Exercise new storage through real detections, including partial-allocation retry. */
#include <stdlib.h>
#include <stdio.h>
#include "../src/napa_vision/_native/napa_core.h"
static int fail_enabled, fail_hits;
static void *fail_pointer;
static size_t fail_bytes;
static void *storage_realloc(void *p, size_t n) {
    if (fail_enabled && p == fail_pointer && n == fail_bytes) {
        fail_hits++;
        return NULL;
    }
    return realloc(p,n);
}
#define realloc storage_realloc
#include "../src/napa_vision/_native/napa_core.c"
#undef realloc

#define CHECK(c) do { if (!(c)) { fprintf(stderr,"shape storage line%d: %s\n",__LINE__,#c); return 1; } } while (0)
#ifndef NAPA_SHAPE_GROWTH_INPUT
#error "Build with a generated 65-tag input path; pytest supplies it."
#endif
#ifdef NAPA_SMALL_STORAGE
enum { W=324, H=288, MINIMUM=32, IDENT=424, PITCH=36 };
#else
enum { W=2304, H=2048, MINIMUM=16, IDENT=137, PITCH=256 };
#endif
static uint8_t image[W*H];
static napa_detection expected[128], actual[128];

static int same(const napa_detection *a, const napa_detection *b, int n) {
    for (int i=0;i<n;i++)
        if (a[i].id!=b[i].id || a[i].hamming!=b[i].hamming ||
            a[i].decision_margin!=b[i].decision_margin ||
            memcmp(a[i].corners,b[i].corners,sizeof(a[i].corners))) return 0;
    return 1;
}

static int reset_shapes(napa_detector *det, int reserve) {
    free(det->shape_out); free(det->shape_centers);
    det->shape_out=NULL; det->shape_centers=NULL;
    det->shape_out_capacity=det->shape_center_capacity=0;
    if (reserve) {
        det->shape_out=(napa_detection *)malloc(64*sizeof(napa_detection));
        det->shape_centers=(vec2 *)malloc(64*sizeof(vec2));
        if (!det->shape_out || !det->shape_centers) return 0;
        det->shape_out_capacity=det->shape_center_capacity=64;
    }
    return 1;
}

int main(void) {
    FILE *f=fopen(NAPA_SHAPE_GROWTH_INPUT,"rb");
    CHECK(f && fread(image,1,sizeof(image),f)==sizeof(image)); fclose(f);
    napa_detector *det=napa_create(); CHECK(det);
    napa_params p={MINIMUM,3,20,22,25,0};
    CHECK(napa_detect(det,image,W,H,W,&p,expected,128)==65);
    CHECK(det->shape_out_capacity>=128 && det->shape_center_capacity>=128);
    CHECK(det->fine_whole_capacity>=128);
    unsigned char seen[72]={0};
    for (int i=0;i<65;i++) {
        CHECK(expected[i].id==IDENT && expected[i].hamming==0);
        double x=0,y=0;
        for (int k=0;k<4;k++) x+=expected[i].corners[2*k]/4, y+=expected[i].corners[2*k+1]/4;
        int cell=(int)(y/PITCH)*9+(int)(x/PITCH);
        CHECK(cell>=0 && cell<65 && !seen[cell]); seen[cell]=1;
    }
    CHECK(napa_detect(det,image,W,H,W,&p,actual,1)==1 && same(expected,actual,1));
    CHECK(napa_detect(det,image,W,H,W,&p,actual,64)==64 && same(expected,actual,64));
    for (int mode=0;mode<4;mode++) {
        CHECK(reset_shapes(det,mode>=2));
        int centers=mode&1;
        void *old_shape=det->shape_out, *old_centers=det->shape_centers;
        fail_pointer=centers?(void *)det->shape_centers:(void *)det->shape_out;
        fail_bytes=(size_t)(mode>=2?128:64)*(centers?sizeof(vec2):sizeof(napa_detection));
        fail_hits=0; fail_enabled=1;
        int n=napa_detect(det,image,W,H,W,&p,actual,128);
        fail_enabled=0;
        CHECK(n==-2 && fail_hits==1);
        if (!centers) {
            CHECK(det->shape_out==old_shape && det->shape_centers==old_centers);
            CHECK(det->shape_out_capacity==(size_t)(mode>=2?64:0));
        } else {
            CHECK(det->shape_out && det->shape_out_capacity==(size_t)(mode>=2?128:64));
            CHECK(det->shape_centers==old_centers);
        }
        CHECK(det->shape_center_capacity==(size_t)(mode>=2?64:0));
        CHECK(napa_detect(det,image,W,H,W,&p,actual,128)==65 && same(expected,actual,65));
    }
    for (int reserve=0;reserve<2;reserve++) {
        free(det->fine_wholes);
        det->fine_wholes=reserve?(fine_whole *)malloc(64*sizeof(fine_whole)):NULL;
        det->fine_whole_capacity=reserve?64:0;
        CHECK(!reserve || det->fine_wholes);
        void *old_pending=det->fine_wholes;
        fail_pointer=old_pending;
        fail_bytes=(size_t)(reserve?128:64)*sizeof(fine_whole);
        fail_hits=0; fail_enabled=1;
        int n=napa_detect(det,image,W,H,W,&p,actual,128);
        fail_enabled=0;
        CHECK(n==-2 && fail_hits==1);
        CHECK(det->fine_wholes==old_pending && det->fine_whole_capacity==(size_t)(reserve?64:0));
        CHECK(napa_detect(det,image,W,H,W,&p,actual,128)==65 && same(expected,actual,65));
    }
    napa_destroy(det);
    puts("65 distinct shapes, capped prefixes, six allocation-failure/retry cases passed");
    return 0;
}
