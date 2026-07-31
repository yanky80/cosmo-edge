// RK3588 profile placeholder: runtime backend lands in follow-up issues.

#include "media/VideoEncoder.h"

namespace cosmo {
namespace media {

    std::shared_ptr<VideoEncoder> VideoEncoder::Create(void* mediaHandle) {
        static_cast<void>(mediaHandle);
        return nullptr;
    }

}  // namespace media
}  // namespace cosmo
