// vkapriltag (github.com/yojobama/vkapriltag) as a worker process for benchmarks/cpu_backends.py: its released static
// library isn't position-independent, so it can't be loaded as a shared shim. Configured as that project's own validation
// tool: refine_edges on, DetectorConfig defaults.
//
// usage: vkapriltag_worker FAMILY DECIMATION
// Prints "ready\n", then for each frame reads int32 width, int32 height and width * height bytes from stdin and writes
// int32 count, then count records of int32 id, int32 hamming, double corners[8]. Exits at end of input.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

extern "C" {
#include "apriltag.h"
#include "tag16h5.h"
#include "tag36h11.h"
}
#include "vkapriltag/TagDecoder.h"
#include "vkapriltag/gpu/GpuDetector.h"
#include "vkapriltag/gpu/QuadDecode.h"
#include "vkapriltag/vk/Context.h"

namespace {

struct Pipeline {
  apriltag_vulkan::DetectorConfig config;
  std::unique_ptr<apriltag_vulkan::GpuDetector> detector;
  std::unique_ptr<apriltag_vulkan::QuadDecode> quad_decode;
};

bool read_all(void *data, size_t bytes) { return std::fread(data, 1, bytes, stdin) == bytes; }

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  apriltag_family_t *family = std::strcmp(argv[1], "tag16h5") == 0 ? tag16h5_create() : tag36h11_create();
  const auto decimation = static_cast<uint32_t>(std::atoi(argv[2]));
  apriltag_detector_t *tags = apriltag_detector_create();
  apriltag_detector_add_family(tags, family);
  tags->refine_edges = true;
  tags->nthreads = 1;
  try {
    apriltag_vulkan::vk::Context context;
    apriltag_vulkan::TagDecoder decoder(tags, decimation);
    std::map<std::pair<uint32_t, uint32_t>, Pipeline> pipelines;
    std::vector<uint8_t> gray;
    std::fputs("ready\n", stdout);
    std::fflush(stdout);
    int32_t size[2];
    while (read_all(size, sizeof size)) {
      const auto width = static_cast<uint32_t>(size[0]), height = static_cast<uint32_t>(size[1]);
      gray.resize(static_cast<size_t>(width) * height);
      if (!read_all(gray.data(), gray.size())) return 1;
      Pipeline &pipeline = pipelines[{width, height}];
      if (!pipeline.detector) {
        pipeline.config.width = width;
        pipeline.config.height = height;
        pipeline.config.decimation = decimation;
        pipeline.config.tag_width = static_cast<uint32_t>(family->width_at_border);
        pipeline.config.reversed_border = family->reversed_border;
        pipeline.config.normal_border = !family->reversed_border;
        pipeline.detector = std::make_unique<apriltag_vulkan::GpuDetector>(context, pipeline.config);
        pipeline.quad_decode = std::make_unique<apriltag_vulkan::QuadDecode>(pipeline.config);
      }
      pipeline.detector->Detect(gray.data());
      auto quads = pipeline.quad_decode->Decode(pipeline.detector->last_line_fit_points);
      zarray_t *found = decoder.Decode(quads, gray.data(), width, height, pipeline.config.reversed_border);  // owned by the decoder
      const int32_t count = zarray_size(found);
      std::fwrite(&count, sizeof count, 1, stdout);
      for (int i = 0; i < count; ++i) {
        apriltag_detection_t *detection;
        zarray_get(found, i, &detection);
        const int32_t head[2] = {detection->id, detection->hamming};
        std::fwrite(head, sizeof head, 1, stdout);
        std::fwrite(detection->p, sizeof(double), 8, stdout);
      }
      std::fflush(stdout);
    }
  } catch (const std::exception &error) {
    std::cerr << "vkapriltag: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
