/* Independent atan2 TLS oracle; rejected fits must not duplicate or lose samples. */
#include "../src/napa_vision/_native/napa_core.c"
#include <assert.h>
#include <stdio.h>

static void check(const int *order, int n, const double *x, const double *y, const double *w) {
    double px[12], py[12], wt[12], sw = 0, mx = 0, my = 0;
    for (int i = 0; i < n; i++) {
        px[i] = x[order[i]]; py[i] = y[order[i]]; wt[i] = w[order[i]];
        sw += wt[i]; mx += wt[i]*px[i]; my += wt[i]*py[i];
    }
    mx /= sw; my /= sw;
    double xx = 0, yy = 0, xy = 0;
    for (int i = 0; i < n; i++) {
        xx += wt[i]*(px[i]-mx)*(px[i]-mx);
        yy += wt[i]*(py[i]-my)*(py[i]-my);
        xy += wt[i]*(px[i]-mx)*(py[i]-my);
    }
    double angle = .5 * atan2(2*xy, xx-yy), dx = cos(angle), dy = sin(angle);
    int kept = 0;
    for (int i = 0; i < n; i++) kept += fabs(-(px[i]-mx)*dy + (py[i]-my)*dx) <= 1;
    assert(kept == (n == 6 ? 2 : 3));
    double lx, ly, ux, uy; int rejected = 0;
    assert(fit_side_line_impl(px, py, wt, n, 0, &lx, &ly, &ux, &uy, &rejected));
    assert(rejected && fabs(lx-mx) < 1e-12 && fabs(ly-my) < 1e-12);
    assert(fabs(ux*dy-uy*dx) < 1e-12);
    for (int i = 0; i < n; i++) {
        assert(px[i] == x[order[i]] && py[i] == y[order[i]] && wt[i] == w[order[i]]);
    }
}
static void permute(int *p, int at) {
    const double x[] = {-30,-20,-10,10,20,30}, y[] = {3,-3,0,0,-3,3}, w[] = {1,1,1,1,1,1};
    if (at == 6) { check(p, 6, x, y, w); return; }
    for (int i = at; i < 6; i++) {
        int t=p[at]; p[at]=p[i]; p[i]=t;
        permute(p, at+1);
        t=p[at]; p[at]=p[i]; p[i]=t;
    }
}
int main(void) {
    int p[] = {0,1,2,3,4,5,6,7,8};
    permute(p, 0);
    /* Rigid transforms preserve the fallback, including a nearly vertical principal axis. */
    for (int turn = 0; turn < 4; turn++) {
        const double sx[] = {-30,-20,-10,10,20,30}, sy[] = {3,-3,0,0,-3,3};
        const double weights[] = {1,1,1,1,1,1};
        double tx[6], ty[6], angle = 0.51 * turn;
        for (int i = 0; i < 6; i++) {
            tx[i] = 100 + cos(angle) * sx[i] - sin(angle) * sy[i];
            ty[i] = -40 + sin(angle) * sx[i] + cos(angle) * sy[i];
        }
        check(p, 6, tx, ty, weights);
    }
    const double x[] = {-40,-30,-20,-10,0,10,20,30,40};
    const double y[] = {3,-3,0,3,0,3,0,-3,3};
    const double w[] = {1,1,2,1,2,1,2,1,1};
    check(p, 9, x, y, w);
    /* A supported inlier set still refits to its own exact horizontal line. */
    double xx[] = {-30,-20,-10,10,20,30}, yy[] = {0,0,0,0,0,5}, ww[] = {1,1,1,1,1,1};
    double lx,ly,dx,dy; int rejected;
    assert(fit_side_line_impl(xx,yy,ww,6,0,&lx,&ly,&dx,&dy,&rejected));
    assert(rejected && fabs(ly)<1e-12 && fabs(dy)<1e-12);
    puts("TLS independent fallback, permutation, weighted and refit cases passed");
    return 0;
}
