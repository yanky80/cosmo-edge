// VideoDecoderAscendHostCopy.cc — on-demand D2H download of an Ascend device
// (AV_PIX_FMT_ASCEND) FrameSurface for preview/snapshot/OSD consumers.
//
// Compiled only when the Ascend media backend is built (CANN headers/libs
// available), so the decoder's hermetic contract check (system FFmpeg only)
// stays buildable. The inference path (decoder -> DVPP image_to_tensor) never
// calls this function: it exists so host consumers can request a copy without
// changing the inference path.

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "media/FrameSurface.h"
#include "media/VideoDecoderAscend.h"

namespace {

// Gate 0 device is pinned at open time by the decoder factory.
constexpr int kAscendDeviceId = 0;

struct HostCopyContext {
    aclrtContext context = nullptr;
    bool ready           = false;
};

HostCopyContext& HostCopyContextInstance() {
    static HostCopyContext instance;
    return instance;
}

std::mutex& HostCopyMutex() {
    static std::mutex mutex;
    return mutex;
}

bool EnsureHostCopyContext(std::string& error) {
    std::lock_guard<std::mutex> guard(HostCopyMutex());
    HostCopyContext& state = HostCopyContextInstance();
    if (state.ready) {
        return true;
    }
    // aclInit once per process; a repeat init from the FFmpeg decoder or the
    // NN backend is not an error (ACL_ERROR_REPEAT_INITIALIZE = 100002).
    const aclError init_ret = aclInit(nullptr);
    if (init_ret != ACL_SUCCESS && init_ret != 100002) {
        error = "aclInit failed ret=" + std::to_string(init_ret);
        return false;
    }
    const aclError dev_ret = aclrtSetDevice(kAscendDeviceId);
    if (dev_ret != ACL_SUCCESS) {
        error = "aclrtSetDevice failed ret=" + std::to_string(dev_ret);
        return false;
    }
    // One process-lifetime context for host copies; the decoder's own
    // FFmpeg context stays private to the fork and is not reused here.
    if (aclrtCreateContext(&state.context, kAscendDeviceId) != ACL_SUCCESS ||
        aclrtSetCurrentContext(state.context) != ACL_SUCCESS) {
        error         = "host-copy ACL context creation failed";
        state.context = nullptr;
        return false;
    }
    state.ready = true;
    return true;
}

}  // namespace

namespace cosmo::media {

bool DownloadAscendDeviceSurfaceToHost(const FrameSurface& surface, int frame_width, int frame_height,
                                       std::vector<uint8_t>& host_nv12, std::string& error) {
    if (surface.memory_type != FrameSurfaceMemoryType::Device) {
        error = "host copy requires a device (AV_PIX_FMT_ASCEND) surface";
        return false;
    }
    if (!surface.IsValid() || surface.planes.size() != 2) {
        error = "host copy requires a valid two-plane device surface";
        return false;
    }
    if (frame_width <= 0 || frame_height <= 0 || (frame_width % 2) != 0 || (frame_height % 2) != 0) {
        error = "host copy requires positive even frame dimensions";
        return false;
    }
    if (surface.lifetime == nullptr) {
        error = "host copy requires a surface with a bound lifetime (expired device frame)";
        return false;
    }

    const auto& luma   = surface.planes[0];
    const auto& chroma = surface.planes[1];
    if (luma.pitch < static_cast<size_t>(frame_width) || chroma.pitch != luma.pitch ||
        luma.vertical_stride < static_cast<size_t>(frame_height) ||
        chroma.vertical_stride < static_cast<size_t>(frame_height) / 2) {
        error = "host copy device surface geometry does not cover the frame";
        return false;
    }

    if (!EnsureHostCopyContext(error)) {
        return false;
    }

    const size_t luma_bytes   = luma.pitch * static_cast<size_t>(frame_height);
    const size_t chroma_bytes = chroma.pitch * static_cast<size_t>(frame_height) / 2;
    try {
        host_nv12.resize(luma_bytes + chroma_bytes);
    } catch (const std::bad_alloc&) {
        error = "host copy buffer allocation failed";
        return false;
    }

    const uint8_t* y_addr  = surface.GetPlaneData(0);
    const uint8_t* uv_addr = surface.GetPlaneData(1);
    if (y_addr == nullptr || uv_addr == nullptr) {
        error = "host copy device plane addresses missing";
        return false;
    }

    const aclError y_ret = aclrtMemcpy(host_nv12.data(), luma_bytes, const_cast<uint8_t*>(y_addr), luma_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST);
    if (y_ret != ACL_SUCCESS) {
        error = "host copy Y plane failed ret=" + std::to_string(y_ret);
        return false;
    }
    const aclError uv_ret =
        aclrtMemcpy(host_nv12.data() + luma_bytes, chroma_bytes, const_cast<uint8_t*>(uv_addr), chroma_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST);
    if (uv_ret != ACL_SUCCESS) {
        error = "host copy UV plane failed ret=" + std::to_string(uv_ret);
        return false;
    }
    return true;
}

}  // namespace cosmo::media
