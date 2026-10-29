/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * Licensed under the PolyForm Noncommercial License 1.0.0: see LICENSE. Commercial use needs a separate
 * license. Copies and works based on this file must keep this notice and credit Zachary Wilson. */
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
    int max_hamming;      /* corrected payload bits (0-3), capped by the family (napa_family_layout) */
    int min_contrast;     /* grey levels between tag black and the white quiet zone (1-255) */
    int min_border_cells; /* dark border cells required out of 28 (0-28) */
    int sharpening_x100;  /* cell sharpening strength x100 (0-100); upstream AprilTag uses 25 */
} napa_params;

typedef struct {
    int id;              /* the tag's ID in the detector's family */
    int hamming;         /* payload bits corrected */
    float decision_margin; /* min of white and black payload bits' mean distance from the threshold */
    double corners[8]; /* x0, y0, ... in OpenCV corner order, pixel-centre coordinates */
} napa_detection;

/* A tag family's layout, for drawing its tags (napa_family_info). Cell (x[k], y[k]) holds bit k of a
 * code, most significant first, in cells from the detected square's top-left cell (negative, or at least
 * `width`, outside it). As napa_tag36h11_code, the layout is the official artwork turned 180 degrees. */
typedef struct {
    int ncodes, nbits;
    int width;        /* cells across the detected square, whose side is the tag size */
    int total_width;  /* cells across the printed tag */
    int reversed;     /* the square prints white inside a black frame (tagStandard, tagCircle, tagCustom) */
    int min_distance; /* minimum Hamming distance between codes */
    int max_hamming;  /* the most payload bits napa_params.max_hamming may correct: larger values are capped */
    int recommended_hamming; /* a default for max_hamming: chance matches no likelier than tag36h11's at 3 */
    int8_t x[64], y[64];
} napa_family_layout;

/* A detector for tag36h11, the default family. */
NAPA_API napa_detector *napa_create(void);
/* A detector for one family by name ("tag36h11", "tag16h5", "tag25h9", "tag36h10", "tagStandard41h12",
 * "tagStandard52h13", "tagCircle21h7", "tagCircle49h12", "tagCustom48h12"). NULL if the name is unknown
 * or memory runs out. */
NAPA_API napa_detector *napa_create_family(const char *family);
NAPA_API int napa_family_count(void);
NAPA_API const char *napa_family_name(int index); /* NULL outside 0 .. napa_family_count() - 1 */
NAPA_API int napa_family_info(const char *family, napa_family_layout *info); /* 0, or -1 if unknown */
/* Code `id` of a family, bits as napa_family_layout describes, a set bit for a white cell. 0 or -1. */
NAPA_API int napa_family_code(const char *family, int id, uint64_t *code);
NAPA_API void napa_destroy(napa_detector *det);
/* Detect the detector's family of tags in an 8-bit grayscale image. Returns the number of detections written
 * (at most max_out), -1 on invalid input or parameters (see napa_params for ranges), or -2 if
 * allocation fails. A detector handle is not thread-safe; use one per thread or camera stream. */
NAPA_API int napa_detect(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                         const napa_params *params, napa_detection *out, int max_out);
/* The payload of tag36h11 tag `id` (0-586): the 6x6 cells inside the black border, row-major, most
 * significant of the 36 bits first, a set bit for a white cell. The grid is in OpenCV's orientation, which
 * is the official artwork turned 180 degrees: its first bit is the official tag's bottom-right payload
 * cell. Returns 0, or -1 for an invalid id or a NULL code. */
NAPA_API int napa_tag36h11_code(int id, uint64_t *code);

#ifdef __cplusplus
}
#endif

#endif
