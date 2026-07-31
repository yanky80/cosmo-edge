// RK3588 profile placeholder: runtime backend lands in follow-up issues.

#include "media/VideoDecoder.h"

namespace cosmo {
namespace media {

    std::unique_ptr<VideoDecoder> VideoDecoder::Create(size_t name, void* mediaHandle) {
        static_cast<void>(name);
        static_cast<void>(mediaHandle);
        return nullptr;
    }

}  // namespace media
}  // namespace cosmo
