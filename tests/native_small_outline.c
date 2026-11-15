/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
/* Independent geometry, rejection-restore and bounds checks for small outline seeds. */
#include "../src/napa_vision/_native/napa_core.c"
#include <stdio.h>

static unsigned long total, accepted, rejected, edge_cases, interior_cases, order_rejects;
static unsigned long malformed_cases, trace_cases, invalid_bounds, canary_checks;
static const uint64_t before_guard = UINT64_C(0x7a16f2045de398bc);
static const uint64_t after_guard = UINT64_C(0xd0496b31e82c57af);

static void demand(int condition, const char *what, const char *group) {
    if (!condition) {
        fprintf(stderr, "FAIL %s [%s] case %lu\n", what, group, total);
        exit(1);
    }
}

/* Positive oracle: a completed rectangle has unique bbox vertices, rectilinear
 * consecutive sides and the independently known rectangle shoelace area. */
static void rectangle_invariants(int q[4][2], int x0, int x1, int y0, int y1, const char *group) {
    int64_t area2 = 0;
    for (int i = 0; i < 4; i++) {
        int j = (i + 1) % 4;
        demand((q[i][0] == x0 || q[i][0] == x1) && (q[i][1] == y0 || q[i][1] == y1), "non-bbox output", group);
        demand((q[i][0] == q[j][0]) != (q[i][1] == q[j][1]), "non-rectangle edge", group);
        for (int k = 0; k < i; k++)
            demand(q[i][0] != q[k][0] || q[i][1] != q[k][1], "duplicate output corner", group);
        area2 += (int64_t)q[i][0] * q[j][1] - (int64_t)q[i][1] * q[j][0];
    }
    demand(area2 == 2 * (int64_t)(x1 - x0) * (y1 - y0), "rectangle area/winding", group);
}

static void check(const char *group, int input[4][2], int x0, int x1, int y0, int y1,
                  int expected, int changed_index, int corner_x, int corner_y) {
    struct guarded {
        uint64_t before[8];
        int q[4][2];
        uint64_t after[8];
    } storage;
    for (int k = 0; k < 8; k++) storage.before[k] = before_guard, storage.after[k] = after_guard;
    memcpy(storage.q, input, sizeof storage.q);
    total++;
    int result = complete_small_outline(storage.q, x0, x1, y0, y1);
    demand(result == expected, "return value", group);
    for (int k = 0; k < 8; k++) {
        demand(storage.before[k] == before_guard && storage.after[k] == after_guard, "output canary", group);
        canary_checks++;
    }
    if (!expected) {
        rejected++;
        demand(memcmp(storage.q, input, sizeof storage.q) == 0, "rejection did not restore entire array", group);
        return;
    }
    accepted++;
    int changes = 0;
    for (int i = 0; i < 4; i++) {
        int different = memcmp(storage.q[i], input[i], sizeof storage.q[i]) != 0;
        changes += different;
        demand(different == (i == changed_index), "changed a fixed vertex", group);
    }
    demand(changes == 1, "not exactly one changed vertex", group);
    demand(storage.q[changed_index][0] == corner_x && storage.q[changed_index][1] == corner_y, "wrong completed corner", group);
    rectangle_invariants(storage.q, x0, x1, y0, y1, group);
}

/* Expected winding comes from known vertex labels TL,TR,BR,BL in y-down
 * coordinates, independent of the helper's cross-product implementation. */
static int clockwise_labels(const int order[4]) {
    for (int k = 1; k < 4; k++) if (order[k] != (order[0] + k) % 4) return 0;
    return 1;
}

static void permutation_cases(int x0, int x1, int y0, int y1, int px, int py) {
    const int corners[4][2] = {{x0,y0},{x1,y0},{x1,y1},{x0,y1}};
    for (int missing = 0; missing < 4; missing++) {
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++)
        for (int c = 0; c < 4; c++) for (int d = 0; d < 4; d++) {
            int order[4] = {a,b,c,d};
            if (a == b || a == c || a == d || b == c || b == d || c == d) continue;
            int q[4][2], index = -1;
            for (int k = 0; k < 4; k++) {
                q[k][0] = corners[order[k]][0]; q[k][1] = corners[order[k]][1];
                if (order[k] == missing) q[k][0] = px, q[k][1] = py, index = k;
            }
            int expected = clockwise_labels(order);
            if (expected) {
                if (px == x0 || px == x1 || py == y0 || py == y1) edge_cases++;
                else interior_cases++;
            } else order_rejects++;
            check("permuted missing corner", q, x0,x1,y0,y1, expected,index,corners[missing][0],corners[missing][1]);
        }
    }
}

static void grids(void) {
    const int boxes[][4] = {
        {0,4,0,4}, {71,76,54,59}, {72,77,54,59},
        {0,30000,0,30000}, {29990,30000,29980,30000},
        {0,1,0,30000}, {0,30000,0,1}, {300,301,200,202}, {0,1,0,1}
    };
    for (size_t b = 0; b < sizeof boxes / sizeof boxes[0]; b++) {
        int x0=boxes[b][0],x1=boxes[b][1],y0=boxes[b][2],y1=boxes[b][3];
        if (x1-x0 <= 5 && y1-y0 <= 5) {
            for (int x=x0;x<=x1;x++) for(int y=y0;y<=y1;y++) {
                if ((x==x0 || x==x1) && (y==y0 || y==y1)) continue;
                permutation_cases(x0,x1,y0,y1,x,y);
            }
        } else {
            int xs[5]={x0,x0+1,x0+(x1-x0)/2,x1-1,x1};
            int ys[5]={y0,y0+1,y0+(y1-y0)/2,y1-1,y1};
            for(int i=0;i<5;i++)for(int j=0;j<5;j++) {
                int seen=0;
                for(int k=0;k<i;k++) if(xs[k]==xs[i])seen=1;
                for(int k=0;k<j;k++) if(ys[k]==ys[j])seen=1;
                if(seen || ((xs[i]==x0 || xs[i]==x1) && (ys[j]==y0 || ys[j]==y1)))continue;
                permutation_cases(x0,x1,y0,y1,xs[i],ys[j]);
            }
        }
        int full[4][2]={{x0,y0},{x1,y0},{x1,y1},{x0,y1}};
        check("already complete rectangle unchanged",full,x0,x1,y0,y1,0,-1,0,0);malformed_cases++;
    }
}

static void malformed(void) {
    int base[4][2]={{4,0},{4,2},{0,4},{0,0}};
    for(int i=0;i<4;i++) for(int side=0;side<4;side++) {
        int q[4][2];memcpy(q,base,sizeof q);
        if(side==0)q[i][0]=-1;
        if(side==1)q[i][0]=5;
        if(side==2)q[i][1]=-1;
        if(side==3)q[i][1]=5;
        check("out of bbox",q,0,4,0,4,0,-1,0,0);malformed_cases++;
    }
    const int bad[][4][2]={
        {{4,0},{2,2},{0,4},{4,0}}, /* duplicate corner */
        {{4,0},{2,2},{0,4},{2,2}}, /* duplicate noncorner */
        {{4,0},{4,2},{0,2},{0,0}}, /* two distinct noncorners */
        {{2,2},{3,3},{1,1},{1,2}}, /* all noncorners */
        {{4,0},{4,0},{4,0},{4,0}}, /* all same */
        {{0,0},{4,0},{0,0},{2,4}}  /* duplicate with edge point */
    };
    for(size_t i=0;i<sizeof bad/sizeof bad[0];i++) {
        int q[4][2];memcpy(q,bad[i],sizeof q);
        check("malformed vertices",q,0,4,0,4,0,-1,0,0);malformed_cases++;
    }
    const int bounds[][4]={{0,0,0,4},{0,4,0,0},{4,0,0,4},{0,4,4,0},{0,0,0,0},{30000,30000,0,30000}};
    for(size_t i=0;i<sizeof bounds/sizeof bounds[0];i++) {
        check("zero or inverted bounds",base,bounds[i][0],bounds[i][1],bounds[i][2],bounds[i][3],0,-1,0,0);
        invalid_bounds++;
    }
}

static void real_traces(void) {
    const int traces[6][4][2]={
        {{76,54},{76,57},{71,59},{71,54}}, /* 18px42 point */
        {{76,54},{76,59},{71,59},{71,55}}, /* 18px42 mean */
        {{76,54},{76,57},{71,59},{71,54}}, /* 18px137 point */
        {{76,54},{76,59},{71,59},{71,55}}, /* 18px137 mean */
        {{77,54},{76,58},{72,59},{72,54}}, /* 20px424 point */
        {{77,54},{77,56},{72,59},{72,54}}  /* 20px424 mean */
    };
    for(int t=0;t<6;t++) for(int rotation=0;rotation<4;rotation++) {
        int source[4][2],x0=t<4?71:72,x1=t<4?76:77,y0=54,y1=59;
        memcpy(source,traces[t],sizeof source);
        for(int r=0;r<rotation;r++) {
            int oldx0=x0,oldy0=y0,oldx1=x1,oldy1=y1;
            for(int i=0;i<4;i++) {
                int x=source[i][0],y=source[i][1];
                source[i][0]=oldy1-y;source[i][1]=x-oldx0;
            }
            x0=0;y0=0;x1=oldy1-oldy0;y1=oldx1-oldx0;
        }
        int corners[4][2]={{x0,y0},{x1,y0},{x1,y1},{x0,y1}},missing=-1,index=-1;
        for(int c=0;c<4;c++) {
            int seen=0;for(int i=0;i<4;i++)if(source[i][0]==corners[c][0]&&source[i][1]==corners[c][1])seen=1;
            if(!seen)missing=c;
        }
        for(int i=0;i<4;i++) {
            int corner=0;for(int c=0;c<4;c++)if(source[i][0]==corners[c][0]&&source[i][1]==corners[c][1])corner=1;
            if(!corner)index=i;
        }
        demand(missing>=0 && index>=0,"invalid real fixture","trace");
        for(int shift=0;shift<4;shift++)for(int reverse=0;reverse<2;reverse++) {
            int q[4][2],new_index=-1;
            for(int k=0;k<4;k++) {
                int from=(shift+(reverse?4-k:k))%4;
                memcpy(q[k],source[from],sizeof q[k]);if(from==index)new_index=k;
            }
            check("real trace rotated/cyclic/reversed",q,x0,x1,y0,y1,!reverse,new_index,corners[missing][0],corners[missing][1]);
            trace_cases++;
        }
    }
}

int main(void) {
    grids();malformed();real_traces();
    printf("{\"status\":\"PASS\",\"cases\":%lu,\"accepted\":%lu,\"rejected_exact_restore\":%lu,\"positive_edge\":%lu,\"positive_interior\":%lu,\"wrong_order_exact_restore\":%lu,\"malformed\":%lu,\"invalid_bounds\":%lu,\"trace_rotations_cyclic_reverse\":%lu,\"canary_checks\":%lu}\n",total,accepted,rejected,edge_cases,interior_cases,order_rejects,malformed_cases,invalid_bounds,trace_cases,canary_checks);
    return 0;
}
