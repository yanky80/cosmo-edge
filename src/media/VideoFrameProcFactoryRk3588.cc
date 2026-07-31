// RK3588 profile placeholder: runtime backend lands in follow-up issues.

#include "media/IOsdTextRenderer.h"
#include "media/VideoFrameProcFactory.h"
#include "mem/IDeviceContext.h"

namespace cosmo {
namespace media {

    std::unique_ptr<IVideoFrameProc> CreateVideoFrameProc(mem::IDeviceContext& ctx, IOsdTextRenderer& osd) {
        static_cast<void>(ctx);
        static_cast<void>(osd);
        return nullptr;
    }

}  // namespace media
}  // namespace cosmo
