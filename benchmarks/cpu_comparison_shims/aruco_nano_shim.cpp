// Thin C ABI over ArUco Nano (rmsalinas/aruco_nano, header-only) for benchmarks/cpu_comparison.py.
//
// Parameters are built once with only the AprilTag 36h11 dictionary (ArUco Nano's default dictionary list is
// ARUCO_MIP_36h12, which would otherwise be searched too). Each call wraps the caller's buffer in a cv::Mat
// header without copying. Corners are returned as ArUco Nano reports them (OpenCV pixel centres).
#include <cstdint>
#include <opencv2/core.hpp>

#include "aruco_nano.h"

namespace {
struct Handle {
    aruco_nano::DetectorParameters params;
};
}  // namespace

extern "C" {

void *an_create(int min_size, double error_correction_rate, int max_attempts) {
    auto *h = new Handle();
    h->params.dicts = {cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11)};
    if (min_size >= 0) h->params.minSize = min_size;
    if (error_correction_rate >= 0) h->params.errorCorrectionRate = error_correction_rate;
    if (max_attempts >= 0) h->params.maxAttemptsPerCandidate = max_attempts;
    return h;
}

// Effective settings: boxFilterSize, thres, minSize, maxAttemptsPerCandidate, maxTimesRevisited,
// errorCorrectionRate, maxErroneousBitsInBorderRate, detectInvertedMarker, number of dictionaries,
// ArucoNanoVersion, OpenCV threads.
void an_settings(void *handle, double *out) {
    const auto &p = static_cast<Handle *>(handle)->params;
    out[0] = p.boxFilterSize;
    out[1] = p.thres;
    out[2] = p.minSize;
    out[3] = p.maxAttemptsPerCandidate;
    out[4] = p.maxTimesRevisited;
    out[5] = p.errorCorrectionRate;
    out[6] = p.maxErroneousBitsInBorderRate;
    out[7] = p.detectInvertedMarker;
    out[8] = double(p.dicts.size());
    out[9] = ArucoNanoVersion;
    out[10] = cv::getNumThreads();
}

void an_set_threads(int n) { cv::setNumThreads(n); }

int an_get_threads() { return cv::getNumThreads(); }

const char *an_opencv_version() { return CV_VERSION; }

int an_detect(void *handle, uint8_t *buf, int width, int height, int stride, int *ids, double *corners,
              int max_out) {
    const auto &p = static_cast<Handle *>(handle)->params;
    cv::Mat image(height, width, CV_8UC1, buf, size_t(stride));
    std::vector<aruco_nano::Marker> markers = aruco_nano::MarkerDetector::detect(image, p);
    int n = int(markers.size());
    int count = n < max_out ? n : max_out;
    for (int i = 0; i < count; i++) {
        ids[i] = markers[i].id;
        for (int c = 0; c < 4; c++) {
            corners[i * 8 + c * 2] = markers[i][c].x;
            corners[i * 8 + c * 2 + 1] = markers[i][c].y;
        }
    }
    return n;
}

void an_destroy(void *handle) { delete static_cast<Handle *>(handle); }
}
