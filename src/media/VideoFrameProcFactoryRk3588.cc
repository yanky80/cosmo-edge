#include "media/IOsdTextRenderer.h"
#include "media/VideoFrameProcFactory.h"
#include "media/VideoFrameProcRk3588.h"
#include "mem/IDeviceContext.h"

namespace cosmo::media {

std::unique_ptr<IVideoFrameProc> CreateVideoFrameProc(mem::IDeviceContext& ctx, IOsdTextRenderer& osd) {
    static_cast<void>(ctx);  // Host conversion does not need an NPU device handle.
    return std::make_unique<VideoFrameProcRk3588>(osd);
}

}  // namespace cosmo::media
