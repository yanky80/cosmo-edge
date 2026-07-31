#include "media/VideoDecoder.h"
#include "media/VideoDecoderRk3588.h"

namespace cosmo {
namespace media {

    std::unique_ptr<VideoDecoder> VideoDecoder::Create(size_t name, void* mediaHandle) {
        static_cast<void>(mediaHandle);
        return std::make_unique<VideoDecoderRk3588>(name);
    }

}  // namespace media
}  // namespace cosmo
