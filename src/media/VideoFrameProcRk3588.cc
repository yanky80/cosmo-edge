// VideoFrameProcRk3588 — on-demand host conversion for the RK3588 media
// backend. Decoded frames are NV12 DRM PRIME DMA-BUF surfaces; this wrapper
// materializes host I420 copies only when a preview/capture/OSD/record
// consumer needs them and delegates the actual work to the generic CPU
// implementation. The zero-copy inference path never touches this class.

#include "media/VideoFrameProcRk3588.h"

#include <sys/mman.h>

#include <cstdlib>
#include <cstring>

#include "media/FrameSurface.h"
#include "media/PixelFormat.h"
#include "util/Log.h"

namespace cosmo::media {
namespace {

    // Converts a validated NV12 DMA-BUF surface into packed I420 host layout
    // (Y plane, then U plane, then V plane), matching what the generic CPU
    // implementation and upload/OSD consumers expect.
    bool Nv12DmaToHostI420(const VideoFrame& frame, uint8_t* dst, size_t dst_size) {
        const auto* surface = frame.GetSurface().get();
        if (!surface || surface->memory_type != FrameSurfaceMemoryType::DmaBuf ||
            surface->planes.size() != 2) {
            return false;
        }
        const size_t width  = frame.GetWidth();
        const size_t height = frame.GetHeight();
        if (width == 0 || height == 0 || (width % 2) != 0 || (height % 2) != 0) {
            LOG_WARN("RK3588 host conversion requires even dimensions, got {}x{}", width, height);
            return false;
        }

        const auto& y_plane  = surface->planes[0];
        const auto& uv_plane = surface->planes[1];
        if (y_plane.fd < 0 || uv_plane.fd < 0 || y_plane.pitch < width || uv_plane.pitch < width ||
            y_plane.offset >= y_plane.size || uv_plane.offset >= uv_plane.size) {
            LOG_WARN("{}", "RK3588 host conversion rejected invalid DMA surface planes");
            return false;
        }

        const size_t expected = width * height * 3 / 2;
        if (dst == nullptr || dst_size < expected) {
            LOG_WARN("RK3588 host conversion target buffer too small ({} < {})", dst_size, expected);
            return false;
        }

        const size_t map_length = y_plane.size;
        void* mapping           = mmap(nullptr, map_length, PROT_READ, MAP_SHARED, y_plane.fd, 0);
        if (mapping == MAP_FAILED) {
            LOG_WARN("RK3588 host conversion mmap failed for fd {} size {}", y_plane.fd, map_length);
            return false;
        }
        struct MappingGuard {
            void* value;
            size_t length;
            ~MappingGuard() {
                munmap(value, length);
            }
        } guard{mapping, map_length};

        const auto* y_src  = static_cast<const uint8_t*>(mapping) + y_plane.offset;
        const auto* uv_src = static_cast<const uint8_t*>(mapping) + uv_plane.offset;
        for (size_t row = 0; row < height; ++row) {
            std::memcpy(dst + row * width, y_src + row * y_plane.pitch, width);
        }

        uint8_t* u_dst = dst + width * height;
        uint8_t* v_dst = u_dst + (width / 2) * (height / 2);
        for (size_t row = 0; row < height / 2; ++row) {
            const auto* uv_row = uv_src + row * uv_plane.pitch;
            for (size_t col = 0; col < width / 2; ++col) {
                u_dst[row * (width / 2) + col] = uv_row[col * 2];
                v_dst[row * (width / 2) + col] = uv_row[col * 2 + 1];
            }
        }
        return true;
    }

}  // namespace

VideoFrameProcRk3588::VideoFrameProcRk3588(IOsdTextRenderer& osd_service) : cpu_(osd_service) {}

VideoFrameProcRk3588::~VideoFrameProcRk3588() = default;

VideoFramePtr VideoFrameProcRk3588::MaterializeHost(VideoFramePtr frame) const {
    if (!frame || !frame->Active()) {
        return nullptr;
    }
    if (frame->GetData() != nullptr) {
        return frame;  // Already host-backed.
    }
    const auto* surface = frame->GetSurface().get();
    if (!surface || surface->memory_type != FrameSurfaceMemoryType::DmaBuf || surface->planes.size() != 2) {
        LOG_WARN("{}", "RK3588 host conversion requires a NV12 DMA-BUF surface");
        return nullptr;
    }

    auto host = std::make_shared<VideoFrame>(static_cast<int>(frame->GetWidth()),
                                             static_cast<int>(frame->GetHeight()), PixelFormat::PIXEL_I420,
                                             frame->GetFrameIndex(), frame->GetTimestamp());
    if (!VideoFrameValid(host)) {
        LOG_WARN("{}", "RK3588 host conversion could not allocate an I420 frame");
        return nullptr;
    }
    if (!Nv12DmaToHostI420(*frame, host->GetData(), host->GetSize())) {
        return nullptr;
    }
    host->SetStreamIndex(frame->GetStreamIndex());
    return host;
}

bool VideoFrameProcRk3588::EnsureHostData(VideoFramePtr frame) {
    if (!frame || !frame->Active()) {
        return false;
    }
    if (frame->GetHostData() != nullptr) {
        return true;
    }
    const auto* surface = frame->GetSurface().get();
    if (surface && surface->memory_type == FrameSurfaceMemoryType::DmaBuf && surface->planes.size() == 2) {
        const size_t size = frame->GetWidth() * frame->GetHeight() * 3 / 2;
        auto* host        = static_cast<uint8_t*>(std::malloc(size));
        if (host == nullptr) {
            LOG_WARN("{}", "RK3588 EnsureHostData malloc failed");
            return false;
        }
        if (!Nv12DmaToHostI420(*frame, host, size)) {
            std::free(host);
            return false;
        }
        frame->SetHostData(host);  // VideoFrame owns and frees this buffer.
        return true;
    }
    return cpu_.EnsureHostData(frame);
}

VideoFramePtr VideoFrameProcRk3588::CopyFrame(VideoFramePtr srcImage) {
    // The materialized host frame is already a private copy of the DMA-BUF
    // surface, so return it directly instead of copying again.
    return MaterializeHost(std::move(srcImage));
}

VideoFramePtr VideoFrameProcRk3588::BGR2I420(VideoFramePtr srcImage) {
    auto host = MaterializeHost(std::move(srcImage));
    return host ? cpu_.BGR2I420(host) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::RGB2I420(VideoFramePtr srcImage) {
    auto host = MaterializeHost(std::move(srcImage));
    return host ? cpu_.RGB2I420(host) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::I4202BGR(VideoFramePtr srcImage) {
    auto host = MaterializeHost(std::move(srcImage));
    return host ? cpu_.I4202BGR(host) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::I4202RGB(VideoFramePtr srcImage) {
    auto host = MaterializeHost(std::move(srcImage));
    return host ? cpu_.I4202RGB(host) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::NV12ToI420(VideoFramePtr srcImage) {
    return MaterializeHost(std::move(srcImage));
}

VideoFramePtr VideoFrameProcRk3588::Crop(const VideoFramePtr srcPicture, const util::Box roi) {
    auto host = MaterializeHost(srcPicture);
    return host ? cpu_.Crop(host, roi) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::Resize(VideoFramePtr src, int dst_height, int dst_width) {
    auto host = MaterializeHost(std::move(src));
    return host ? cpu_.Resize(host, dst_height, dst_width) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::Padding(VideoFramePtr src, size_t top, size_t bottom, size_t left,
                                            size_t right, Color color) {
    auto host = MaterializeHost(std::move(src));
    return host ? cpu_.Padding(host, top, bottom, left, right, color) : nullptr;
}

std::vector<u_char> VideoFrameProcRk3588::EncodeJpeg(const VideoFramePtr srcPicture) {
    auto host = MaterializeHost(srcPicture);
    return host ? cpu_.EncodeJpeg(host) : std::vector<u_char>{};
}

VideoFramePtr VideoFrameProcRk3588::DecodeJpeg(const std::vector<u_int8_t>& data) {
    return cpu_.DecodeJpeg(data);
}

VideoFramePtr VideoFrameProcRk3588::DrawBox(VideoFramePtr srcImage, const util::Box imageRect,
                                            const Color& color, int lineWidth) {
    auto host = MaterializeHost(std::move(srcImage));
    return host ? cpu_.DrawBox(host, imageRect, color, lineWidth) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::DrawPoint(VideoFramePtr srcImage, util::Point point, const Color& color,
                                              int lineWidth) {
    auto host = MaterializeHost(std::move(srcImage));
    return host ? cpu_.DrawPoint(host, point, color, lineWidth) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::DrawLines(VideoFramePtr srcImage,
                                              std::vector<std::pair<util::Point, util::Point>> lines,
                                              const Color& color, int lineWidth) {
    auto host = MaterializeHost(std::move(srcImage));
    return host ? cpu_.DrawLines(host, std::move(lines), color, lineWidth) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::DrawRects(VideoFramePtr srcImage, const std::vector<util::Box>& rects,
                                              const Color& color, int lineWidth) {
    auto host = MaterializeHost(std::move(srcImage));
    return host ? cpu_.DrawRects(host, rects, color, lineWidth) : nullptr;
}

VideoFramePtr VideoFrameProcRk3588::DrawText(VideoFramePtr srcImage, int x, int y, const std::string& text,
                                             const Color& color, int fontSize) {
    auto host = MaterializeHost(std::move(srcImage));
    return host ? cpu_.DrawText(host, x, y, text, color, fontSize) : nullptr;
}

bool VideoFrameProcRk3588::BeginOSD(VideoFramePtr frame) {
    auto host = MaterializeHost(std::move(frame));
    return host && cpu_.BeginOSD(host);
}

void VideoFrameProcRk3588::OSDDrawLines(std::vector<std::pair<util::Point, util::Point>> lines,
                                        const Color& color, int lineWidth) {
    cpu_.OSDDrawLines(std::move(lines), color, lineWidth);
}

void VideoFrameProcRk3588::OSDDrawText(int x, int y, const std::string& text, const Color& color,
                                       int fontSize) {
    cpu_.OSDDrawText(x, y, text, color, fontSize);
}

void VideoFrameProcRk3588::OSDDrawTextEx(int x, int y, const std::string& text, const Color& color,
                                         int fontSize, const Color& bgColor, uint8_t bgAlpha, bool outline,
                                         int bgPadding) {
    cpu_.OSDDrawTextEx(x, y, text, color, fontSize, bgColor, bgAlpha, outline, bgPadding);
}

void VideoFrameProcRk3588::EndOSD() {
    cpu_.EndOSD();
}

}  // namespace cosmo::media
