/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Integer triangle areas independently classify convex footprints and their queries. */
#include "../src/napa_vision/_native/napa_core.c"
#include <stdio.h>

static uint32_t state = 219;
static uint32_t random_word(void) { return state = state * 1664525u + 1013904223u; }
static int64_t absolute(int64_t x) { return x < 0 ? -x : x; }

/* Vertices have half-pixel coordinates; centroids can have eighth-pixel coordinates.
 * A point belongs to a convex polygon iff its unsigned edge-triangle areas sum
 * to the polygon area. This does not reuse the production sign test. */
static int reference_contains(const napa_detection *d, double x, double y) {
    int64_t px = (int64_t)(8*x), py = (int64_t)(8*y), area = 0, triangles = 0;
    for (int k = 0; k < 4; k++) {
        int j = (k+1)&3;
        int64_t ax = (int64_t)(8*d->corners[2*k]), ay = (int64_t)(8*d->corners[2*k+1]);
        int64_t bx = (int64_t)(8*d->corners[2*j]), by = (int64_t)(8*d->corners[2*j+1]);
        area += ax*by-ay*bx;
        triangles += absolute((ax-px)*(by-py)-(ay-py)*(bx-px));
    }
    return absolute(area) > 0 && absolute(area) == triangles;
}

static int reference_covered(const napa_detection *out, const vec2 *centres, int n,
                             double x, double y, double radius) {
    for (int i = 0; i < n; i++)
        if (fabs(centres[i].x-x)+fabs(centres[i].y-y) < radius && reference_contains(&out[i],x,y)) return 1;
    return 0;
}

static napa_detection quad(void) {
    napa_detection d = {0};
    const int signs[4][2] = {{-1,-1},{1,-1},{1,1},{-1,1}};
    for (;;) {
        int cx = (int)(random_word()%601)-300, cy = (int)(random_word()%601)-300;
        int ux = (int)(random_word()%101)-50, uy = (int)(random_word()%101)-50;
        int vx = (int)(random_word()%101)-50, vy = (int)(random_word()%101)-50;
        int xy[4][2], positive = 0, negative = 0;
        for (int k = 0; k < 4; k++) {
            xy[k][0] = cx+signs[k][0]*ux+signs[k][1]*vx+(int)(random_word()%9)-4;
            xy[k][1] = cy+signs[k][0]*uy+signs[k][1]*vy+(int)(random_word()%9)-4;
        }
        for (int k = 0; k < 4; k++) {
            int j = (k+1)&3, l = (k+2)&3;
            int cross = (xy[j][0]-xy[k][0])*(xy[l][1]-xy[j][1])-
                        (xy[j][1]-xy[k][1])*(xy[l][0]-xy[j][0]);
            positive += cross > 0;
            negative += cross < 0;
        }
        if (positive != 4 && negative != 4) continue;
        for (int k = 0; k < 4; k++) {
            d.corners[2*k] = 0.5*xy[k][0];
            d.corners[2*k+1] = 0.5*xy[k][1];
        }
        return d;
    }
}

static vec2 center(const napa_detection *d) {
    vec2 p = {0,0};
    for (int k = 0; k < 4; k++) {
        p.x += 0.25*d->corners[2*k];
        p.y += 0.25*d->corners[2*k+1];
    }
    return p;
}

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr,"footprint failure line%d case%d\n",__LINE__,t); return 1; } } while (0)

int main(void) {
    int t = 0;
    int winding[2] = {0,0}, general = 0;
    napa_detection query = quad();
    CHECK(!covered_footprint(NULL,NULL,0,0,0,1));
    CHECK(!already_found_footprint(NULL,NULL,0,&query));
    napa_detection pair[2] = {{0},{0}};
    const double corners[8] = {-1,-1,1,-1,1,1,-1,1};
    memcpy(pair[0].corners,corners,sizeof(corners));
    memcpy(pair[1].corners,corners,sizeof(corners));
    for (int k = 0; k < 4; k++) pair[1].corners[2*k] += 2;
    vec2 near[2] = {{0,0},{2,0}};
    CHECK(!covered_footprint(pair,near,1,2,0,10));
    CHECK(covered_footprint(pair,near,2,2,0,10));
    CHECK(covered_footprint(pair+1,near+1,1,2,0,10));
    CHECK(covered_footprint(pair,near,1,1,0,2)); /* Footprint edge is included. */
    CHECK(!covered_footprint(pair,near,1,1,0,1)); /* Distance threshold is strict. */
    CHECK(!already_found_footprint(pair,near,1,&pair[1]));
    CHECK(already_found_footprint(pair,near,2,&pair[1]));
    for (t = 0; t < 30000; t++) {
        napa_detection out[5]; vec2 centres[5];
        for (int i = 0; i < 5; i++) {
            out[i] = quad(); centres[i] = center(&out[i]);
            const double *q = out[i].corners;
            double cross = (q[2]-q[0])*(q[5]-q[3])-(q[3]-q[1])*(q[4]-q[2]);
            winding[cross > 0]++;
            general += q[0]+q[4] != q[2]+q[6] || q[1]+q[5] != q[3]+q[7];
        }
        int offset = t%3, n = (t/3)%(6-offset);
        double x = 0.5*((int)(random_word()%1001)-500), y = 0.5*((int)(random_word()%1001)-500);
        double radius = 0.5*(random_word()%601);
        int i = (t/5)%5, k = (t/25)%4, j = (k+1)&3;
        if (t%4 == 0) { x = out[i].corners[2*k]; y = out[i].corners[2*k+1]; }
        if (t%4 == 1) {
            x = 0.5*(out[i].corners[2*k]+out[i].corners[2*j]);
            y = 0.5*(out[i].corners[2*k+1]+out[i].corners[2*j+1]);
        }
        if (t%4 == 2) { x = centres[i].x; y = centres[i].y; }
        for (i = 0; i < 5; i++) CHECK(detection_contains(&out[i],x,y) == reference_contains(&out[i],x,y));
        CHECK(covered_footprint(out+offset,centres+offset,n,x,y,radius) ==
              reference_covered(out+offset,centres+offset,n,x,y,radius));
        query = quad(); vec2 c = center(&query);
        double r = 0.25*(fabs(query.corners[0]-query.corners[4])+fabs(query.corners[1]-query.corners[5]))+2;
        CHECK(already_found_footprint(out+offset,centres+offset,n,&query) ==
              reference_covered(out+offset,centres+offset,n,c.x,c.y,r));
    }
    query = quad(); query.corners[0] = NAN;
    CHECK(!detection_contains(&query,0,0));
    query = quad(); query.corners[3] = INFINITY;
    CHECK(!detection_contains(&query,0,0));
    query = quad(); CHECK(!detection_contains(&query,INFINITY,0));
    memset(&query,0,sizeof(query)); CHECK(!detection_contains(&query,0,0));
    CHECK(winding[0] > 10000 && winding[1] > 10000 && general > 10000);
    puts("30000 independent convex footprint/query oracle cases passed");
    return 0;
}
