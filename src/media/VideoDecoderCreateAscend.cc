// VideoDecoderCreateAscend.cc — Ascend backend factory for VideoDecoder.
// Compiled only when COSMO_MEDIA_USE_ASCEND_BACKEND is ON (CMake file-level switching).

#include "media/VideoDecoder.h"
#include "media/VideoDecoderAscend.h"
#include "util/Log.h"

namespace cosmo {
namespace media {

    std::unique_ptr<VideoDecoder> VideoDecoder::Create(size_t name, void* mediaHandle) {
        static_cast<void>(mediaHandle);  // Ascend decoder pins the Gate 0 device id at open.
        if (name != 0) {
            LOG_WARN("Ascend backend only supports the Gate 0 device (requested device id {})", name);
            return nullptr;
        }
        return std::make_unique<VideoDecoderAscend>(name);
    }

}  // namespace media
}  // namespace cosmo
