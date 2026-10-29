/* Thin C ABI over upstream AprilTag 3 (AprilRobotics/apriltag) for benchmarks/cpu_comparison.py.
 *
 * The detector and family are created once; each call wraps the caller's buffer in an image_u8_t without
 * copying, runs apriltag_detector_detect, and copies IDs, Hamming distances, and corners out. Corners are
 * returned exactly as AprilTag reports them (pixel centres at +0.5); the Python adapter converts them. */
#include <stdint.h>
#include <stdlib.h>

#include "apriltag.h"
#include "tag36h11.h"

typedef struct {
    apriltag_detector_t *td;
    apriltag_family_t *tf;
} at_handle;

void *at_create(int nthreads, float quad_decimate, float quad_sigma, int refine_edges, double decode_sharpening,
                int bits_corrected) {
    at_handle *h = (at_handle *)calloc(1, sizeof(at_handle));
    if (h == NULL) return NULL;
    h->tf = tag36h11_create();
    h->td = apriltag_detector_create();
    apriltag_detector_add_family_bits(h->td, h->tf, bits_corrected);
    h->td->nthreads = nthreads;
    h->td->quad_decimate = quad_decimate;
    h->td->quad_sigma = quad_sigma;
    h->td->refine_edges = refine_edges != 0;
    h->td->decode_sharpening = decode_sharpening;
    return h;
}

/* Effective settings, read back from the detector: nthreads, quad_decimate, quad_sigma, refine_edges,
 * decode_sharpening, min_cluster_pixels, min_white_black_diff, max_line_fit_mse, critical_rad, deglitch. */
void at_settings(void *handle, double *out) {
    at_handle *h = (at_handle *)handle;
    out[0] = h->td->nthreads;
    out[1] = h->td->quad_decimate;
    out[2] = h->td->quad_sigma;
    out[3] = h->td->refine_edges;
    out[4] = h->td->decode_sharpening;
    out[5] = h->td->qtp.min_cluster_pixels;
    out[6] = h->td->qtp.min_white_black_diff;
    out[7] = h->td->qtp.max_line_fit_mse;
    out[8] = h->td->qtp.critical_rad;
    out[9] = h->td->qtp.deglitch;
}

int at_detect(void *handle, uint8_t *buf, int width, int height, int stride, int *ids, int *hamming,
              double *corners, int max_out) {
    at_handle *h = (at_handle *)handle;
    image_u8_t im = {.width = width, .height = height, .stride = stride, .buf = buf};
    zarray_t *detections = apriltag_detector_detect(h->td, &im);
    if (detections == NULL) return -1;
    int n = zarray_size(detections);
    int count = n < max_out ? n : max_out;
    for (int i = 0; i < count; i++) {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        ids[i] = det->id;
        hamming[i] = det->hamming;
        for (int c = 0; c < 4; c++) {
            corners[i * 8 + c * 2] = det->p[c][0];
            corners[i * 8 + c * 2 + 1] = det->p[c][1];
        }
    }
    apriltag_detections_destroy(detections);
    return n;
}

void at_destroy(void *handle) {
    at_handle *h = (at_handle *)handle;
    if (h == NULL) return;
    apriltag_detector_destroy(h->td);
    tag36h11_destroy(h->tf);
    free(h);
}
