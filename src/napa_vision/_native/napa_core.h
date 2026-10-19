/* Napa core: plain C ABI for the sparse-sampling AprilTag 36h11 detector. */
#ifndef NAPA_CORE_H
#define NAPA_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define NAPA_API __declspec(dllexport)
#else
#define NAPA_API __attribute__((visibility("default")))
#endif

typedef struct napa_detector napa_detector;

typedef struct {
    int min_tag_px;       /* smallest black-square side to guarantee, in pixels (8-30000) */
    int max_hamming;      /* corrected payload bits (0-3) */
    int min_contrast;     /* grey levels between tag black and the white quiet zone (1-255) */
    int min_border_cells; /* dark border cells required out of 28 (0-28) */
    int sharpening_x100;  /* cell sharpening strength x100 (0-100); upstream AprilTag uses 25 */
} napa_params;

typedef struct {
    int id;              /* tag36h11 ID, 0-586 */
    int hamming;         /* payload bits corrected */
    float decision_margin; /* min of white and black payload bits' mean distance from the threshold */
    double corners[8]; /* x0, y0, ... in OpenCV corner order, pixel-centre coordinates */
} napa_detection;

NAPA_API napa_detector *napa_create(void);
NAPA_API void napa_destroy(napa_detector *det);
/* Detect tag36h11 tags in an 8-bit grayscale image. Returns the number of detections written
 * (at most max_out), -1 on invalid input or parameters (see napa_params for ranges), or -2 if
 * allocation fails. A detector handle is not thread-safe; use one per thread or camera stream. */
NAPA_API int napa_detect(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                         const napa_params *params, napa_detection *out, int max_out);

#ifdef __cplusplus
}
#endif

#endif
