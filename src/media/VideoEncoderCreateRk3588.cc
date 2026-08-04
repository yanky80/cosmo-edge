#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "media/VideoEncoder.h"
#include "media/VideoEncoderCpu.h"

namespace cosmo::media {

std::shared_ptr<VideoEncoder> VideoEncoder::Create(void* mediaHandle) {
    static_cast<void>(mediaHandle);
    // Preview/OSD/record encode with the generic host H.264 encoder: decoded
    // frames are converted to host memory on demand. MPP hardware encode is
    // deferred until profiling shows host encode is a bottleneck.
    return std::make_shared<VideoEncoderCpu>();
}

}  // namespace cosmo::media
