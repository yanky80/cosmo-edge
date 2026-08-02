#include "catch_amalgamated.hpp"

#ifdef COSMO_MEDIA_USE_RK3588_BACKEND

#include <cstring>
#include <memory>

extern "C" {
#include "libavutil/buffer.h"
#include "libavutil/frame.h"
#include "libavutil/hwcontext_drm.h"
}

#include <drm/drm_fourcc.h>

#include "media/FrameSurface.h"
#include "media/VideoCodecType.h"
#include "media/VideoDecoderRk3588.h"

namespace cosmo::media {
namespace {

    std::shared_ptr<AVBufferRef> MakeDescriptorBuffer(const AVDRMFrameDescriptor& descriptor) {
        auto* raw = static_cast<uint8_t*>(av_mallocz(sizeof(AVDRMFrameDescriptor)));
        REQUIRE(raw != nullptr);
        std::memcpy(raw, &descriptor, sizeof(descriptor));
        AVBufferRef* buffer = av_buffer_create(
            raw, sizeof(AVDRMFrameDescriptor), [](void*, uint8_t* data) { av_free(data); }, nullptr, 0);
        REQUIRE(buffer != nullptr);
        return std::shared_ptr<AVBufferRef>(buffer, [](AVBufferRef* value) {
            if (value != nullptr) {
                av_buffer_unref(&value);
            }
        });
    }

    AVFrame MakeDrmPrimeFrame(const AVDRMFrameDescriptor& descriptor, int width, int height) {
        AVFrame frame{};
        frame.format      = AV_PIX_FMT_DRM_PRIME;
        frame.width       = width;
        frame.height      = height;
        auto buffer       = MakeDescriptorBuffer(descriptor);
        frame.buf[0]      = av_buffer_ref(buffer.get());
        frame.data[0]     = frame.buf[0]->data;
        frame.linesize[0] = static_cast<int>(descriptor.layers[0].planes[0].pitch);
        return frame;
    }

}  // namespace

TEST_CASE("RK3588 decoder picks the expected rkmpp codec names", "[rk3588][decoder]") {
    REQUIRE(std::string(Rk3588DecoderNameForCodec(VideoCodecType::kH264)) == "h264_rkmpp");
    REQUIRE(std::string(Rk3588DecoderNameForCodec(VideoCodecType::kH265)) == "hevc_rkmpp");
    REQUIRE(Rk3588DecoderNameForCodec(VideoCodecType::kMjpeg) == nullptr);
}

TEST_CASE("RK3588 decoder negotiation refuses non-DRM output", "[rk3588][decoder]") {
    const AVPixelFormat drm_formats[] = {AV_PIX_FMT_NV12, AV_PIX_FMT_DRM_PRIME, AV_PIX_FMT_NONE};
    REQUIRE(SelectRkDrmPrimePixelFormat(drm_formats) == AV_PIX_FMT_DRM_PRIME);

    const AVPixelFormat host_formats[] = {AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE};
    REQUIRE(SelectRkDrmPrimePixelFormat(host_formats) == AV_PIX_FMT_NONE);
}

TEST_CASE("RK3588 DRM PRIME surface extraction keeps NV12 plane metadata", "[rk3588][decoder]") {
    AVDRMFrameDescriptor descriptor{};
    descriptor.nb_objects                       = 1;
    descriptor.objects[0].fd                    = 17;
    descriptor.objects[0].size                  = 6144;
    descriptor.nb_layers                        = 1;
    descriptor.layers[0].format                 = DRM_FORMAT_NV12;
    descriptor.layers[0].nb_planes              = 2;
    descriptor.layers[0].planes[0].object_index = 0;
    descriptor.layers[0].planes[0].offset       = 0;
    descriptor.layers[0].planes[0].pitch        = 128;
    descriptor.layers[0].planes[1].object_index = 0;
    descriptor.layers[0].planes[1].offset       = 4096;
    descriptor.layers[0].planes[1].pitch        = 128;

    AVFrame frame = MakeDrmPrimeFrame(descriptor, 64, 32);
    FrameSurface surface;
    std::string error;
    REQUIRE(BuildRkDrmPrimeSurface(frame, surface, error));
    REQUIRE(surface.memory_type == FrameSurfaceMemoryType::DmaBuf);
    REQUIRE(surface.planes.size() == 2);
    CHECK(surface.planes[0].fd == 17);
    CHECK(surface.planes[0].offset == 0);
    CHECK(surface.planes[0].pitch == 128);
    CHECK(surface.planes[0].vertical_stride == 32);
    CHECK(surface.planes[0].size == 6144);
    CHECK(surface.planes[1].offset == 4096);
    CHECK(surface.planes[1].pitch == 128);
    CHECK(surface.planes[1].vertical_stride == 16);
    CHECK(surface.planes[1].size == 6144);
    av_frame_unref(&frame);
}

TEST_CASE("RK3588 DRM PRIME surface extraction matches real MPP NV12 layout", "[rk3588][decoder]") {
    // Descriptor shape observed from h264_rkmpp/hevc_rkmpp on the RK3588
    // board for 1920x1080: one DMA-BUF of 4,177,920 bytes, luma rows span
    // [0, 2,088,960) and chroma starts at 2,088,960 (pitch 1920, 1088 luma
    // rows). The chroma plane's vertical stride is half the luma stride.
    AVDRMFrameDescriptor descriptor{};
    descriptor.nb_objects                       = 1;
    descriptor.objects[0].fd                    = 21;
    descriptor.objects[0].size                  = 4177920;
    descriptor.nb_layers                        = 1;
    descriptor.layers[0].format                 = DRM_FORMAT_NV12;
    descriptor.layers[0].nb_planes              = 2;
    descriptor.layers[0].planes[0].object_index = 0;
    descriptor.layers[0].planes[0].offset       = 0;
    descriptor.layers[0].planes[0].pitch        = 1920;
    descriptor.layers[0].planes[1].object_index = 0;
    descriptor.layers[0].planes[1].offset       = 2088960;
    descriptor.layers[0].planes[1].pitch        = 1920;

    AVFrame frame = MakeDrmPrimeFrame(descriptor, 1920, 1080);
    FrameSurface surface;
    std::string error;
    REQUIRE(BuildRkDrmPrimeSurface(frame, surface, error));
    REQUIRE(surface.memory_type == FrameSurfaceMemoryType::DmaBuf);
    REQUIRE(surface.planes.size() == 2);
    CHECK(surface.planes[0].fd == 21);
    CHECK(surface.planes[0].offset == 0);
    CHECK(surface.planes[0].pitch == 1920);
    CHECK(surface.planes[0].vertical_stride == 1088);
    CHECK(surface.planes[0].size == 4177920);
    CHECK(surface.planes[1].offset == 2088960);
    CHECK(surface.planes[1].pitch == 1920);
    CHECK(surface.planes[1].vertical_stride == 544);
    CHECK(surface.planes[1].size == 4177920);
    av_frame_unref(&frame);
}

TEST_CASE("RK3588 DRM PRIME surface extraction rejects invalid decoder output", "[rk3588][decoder]") {
    SECTION("non DRM frame") {
        AVFrame frame{};
        frame.format = AV_PIX_FMT_NV12;
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildRkDrmPrimeSurface(frame, surface, error));
        CHECK(error == "decoded frame is not DRM PRIME");
    }

    SECTION("non NV12 layer") {
        AVDRMFrameDescriptor descriptor{};
        descriptor.nb_objects                       = 1;
        descriptor.objects[0].fd                    = 3;
        descriptor.objects[0].size                  = 4096;
        descriptor.nb_layers                        = 1;
        descriptor.layers[0].format                 = DRM_FORMAT_XRGB8888;
        descriptor.layers[0].nb_planes              = 1;
        descriptor.layers[0].planes[0].object_index = 0;
        descriptor.layers[0].planes[0].offset       = 0;
        descriptor.layers[0].planes[0].pitch        = 256;

        AVFrame frame = MakeDrmPrimeFrame(descriptor, 64, 32);
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildRkDrmPrimeSurface(frame, surface, error));
        CHECK(error == "DRM PRIME layer is not NV12");
        av_frame_unref(&frame);
    }

    SECTION("invalid object reference") {
        AVDRMFrameDescriptor descriptor{};
        descriptor.nb_objects                       = 1;
        descriptor.objects[0].fd                    = 4;
        descriptor.objects[0].size                  = 6144;
        descriptor.nb_layers                        = 1;
        descriptor.layers[0].format                 = DRM_FORMAT_NV12;
        descriptor.layers[0].nb_planes              = 2;
        descriptor.layers[0].planes[0].object_index = 1;
        descriptor.layers[0].planes[0].pitch        = 128;
        descriptor.layers[0].planes[1].object_index = 0;
        descriptor.layers[0].planes[1].offset       = 4096;
        descriptor.layers[0].planes[1].pitch        = 128;

        AVFrame frame = MakeDrmPrimeFrame(descriptor, 64, 32);
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildRkDrmPrimeSurface(frame, surface, error));
        CHECK(error == "DRM PRIME plane references an invalid object");
        av_frame_unref(&frame);
    }

    SECTION("luma span is not whole rows") {
        AVDRMFrameDescriptor descriptor{};
        descriptor.nb_objects                       = 1;
        descriptor.objects[0].fd                    = 5;
        descriptor.objects[0].size                  = 6145;
        descriptor.nb_layers                        = 1;
        descriptor.layers[0].format                 = DRM_FORMAT_NV12;
        descriptor.layers[0].nb_planes              = 2;
        descriptor.layers[0].planes[0].object_index = 0;
        descriptor.layers[0].planes[0].pitch        = 128;
        descriptor.layers[0].planes[1].object_index = 0;
        descriptor.layers[0].planes[1].offset       = 4097;
        descriptor.layers[0].planes[1].pitch        = 128;

        AVFrame frame = MakeDrmPrimeFrame(descriptor, 64, 32);
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildRkDrmPrimeSurface(frame, surface, error));
        CHECK(error == "NV12 luma plane size must be a whole number of rows");
        av_frame_unref(&frame);
    }

    SECTION("chroma plane exceeds the DMA-BUF") {
        AVDRMFrameDescriptor descriptor{};
        descriptor.nb_objects                       = 1;
        descriptor.objects[0].fd                    = 6;
        descriptor.objects[0].size                  = 6000;
        descriptor.nb_layers                        = 1;
        descriptor.layers[0].format                 = DRM_FORMAT_NV12;
        descriptor.layers[0].nb_planes              = 2;
        descriptor.layers[0].planes[0].object_index = 0;
        descriptor.layers[0].planes[0].pitch        = 128;
        descriptor.layers[0].planes[1].object_index = 0;
        descriptor.layers[0].planes[1].offset       = 4096;
        descriptor.layers[0].planes[1].pitch        = 128;

        AVFrame frame = MakeDrmPrimeFrame(descriptor, 64, 32);
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildRkDrmPrimeSurface(frame, surface, error));
        CHECK(error == "NV12 chroma plane exceeds the DMA-BUF allocation");
        av_frame_unref(&frame);
    }
}

}  // namespace cosmo::media

#endif
