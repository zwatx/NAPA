/* NAPA. Copyright (c) 2026 Zachary Wilson.
 * SPDX-License-Identifier: GPL-3.0-only. Also available under a commercial license: see NOTICE.
 * Copies and works based on this file must keep this notice and credit Zachary Wilson. */
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
    int split_touching;   /* 1: also decode dark shapes that touch others only at a corner, as tags do in
                           * AprilGrid boards and checkerboards; slower. 0 (the default): off */
} napa_params;

/* napa_detection.hamming of a tag read as a whole picture (napa_detect_soft): no bits were counted. */
#define NAPA_SOFT_DECODED (-1)

typedef struct {
    int id;              /* the tag's ID in the detector's family */
    int hamming;         /* payload bits corrected, or NAPA_SOFT_DECODED from napa_detect_soft */
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
/* Experimental shadow recovery: ordinary napa_detect results first, then exact-code detections
 * from a lower proposal threshold, up to max_out. Original corners are unchanged. Same errors as
 * napa_detect; on -2 the output is incomplete and must not be used. No extra search for families
 * whose exact-code chance-match rate exceeds the split recovery budget. Shadow recovery adds no
 * further napa_params changes; callers built before split_touching was added must rebuild against
 * this header. This new symbol requires an updated library. */
NAPA_API int napa_detect_shadows(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                                 const napa_params *params, napa_detection *out, int max_out);
/* Recovery sampling minimum: 0 preserves full sensitivity; otherwise params->min_tag_px through
 * 30000 inclusive. Larger values trade sensitivity for speed. Ordinary outputs are unchanged.
 * Invalid settings return -1, including at zero capacity. */
NAPA_API int napa_detect_shadows_configured(napa_detector *det, const uint8_t *img, int width, int height,
                                           int stride, const napa_params *params, napa_detection *out,
                                           int max_out, int shadow_min_tag_px);
/* Options for napa_detect_options. */
#define NAPA_OPTION_SOFT_DECODE 1u
#define NAPA_OPTION_ALIGN_CORNERS 2u

/* Experimental, tag36h11 only (-1 otherwise). The ordinary results of napa_detect, or of
 * napa_detect_shadows_configured when recover_shadows is 1, then tiny tags (sides of about 8-19 pixels) recognised
 * by comparing the whole tag with every code, and tags with a flat patch over part of the payload. Those carry hamming
 * NAPA_SOFT_DECODED; they are not exact matches and structured non-tags can pass. The first call on a detector builds about 700 KB of tables. */
NAPA_API int napa_detect_soft(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                              const napa_params *params, napa_detection *out, int max_out, int recover_shadows,
                              int shadow_min_tag_px);
/* As napa_detect_soft with any of the NAPA_OPTION flags. NAPA_OPTION_ALIGN_CORNERS re-aligns blurred tags' corners after
 * decoding, at some cost per tag. Experimental, tag36h11 only (-1 otherwise); -1 for unknown flags. */
NAPA_API int napa_detect_options(napa_detector *det, const uint8_t *img, int width, int height, int stride,
                                 const napa_params *params, napa_detection *out, int max_out, int recover_shadows,
                                 int shadow_min_tag_px, unsigned options);
/* The camera's tone curve, as a gamma from 1 (linear, the default: no change) to 4. Experimental, tag36h11 only (-1 otherwise).
 * Corners are then refined on a linearized copy of each tag. Applies to every later call on this detector. A wrong value is
 * worse than none on a linear camera. */
NAPA_API int napa_set_camera_gamma(napa_detector *det, double gamma);
/* How tags look in the frame, for the ordinary pass's sampling: 0 (the default) samples alike both ways. TALL samples
 * half as many rows, for tags turned about the vertical axis, which look narrow and tall; WIDE half as many columns,
 * for a camera on its side. The sampling across the narrow side is unchanged; fitting and decoding read the frame at
 * full resolution. Applies to every later call on this detector. 0, or -1 for an unknown shape. */
#define NAPA_TAG_SHAPE_TALL 1
#define NAPA_TAG_SHAPE_WIDE 2
NAPA_API int napa_set_tag_shape(napa_detector *det, int shape);
/* A tag's mean border-width bias, in cells, at the given corners: positive when the black border reads narrow. If edge_cells is
 * not NULL it receives the width of the blurred outer edges in cells, or NaN if they could not be fitted. 0 on success, 1 if
 * nothing could be measured, -1 on invalid input. */
NAPA_API int napa_border_bias_cells(napa_detector *det, const uint8_t *img, int width, int height, int stride, const napa_params *params,
                                    const double corners[8], double *bias_cells, double *edge_cells);
/* The payload of tag36h11 tag `id` (0-586): the 6x6 cells inside the black border, row-major, most
 * significant of the 36 bits first, a set bit for a white cell. The grid is in OpenCV's orientation, which
 * is the official artwork turned 180 degrees: its first bit is the official tag's bottom-right payload
 * cell. Returns 0, or -1 for an invalid id or a NULL code. */
NAPA_API int napa_tag36h11_code(int id, uint64_t *code);

#ifdef __cplusplus
}
#endif

#endif
