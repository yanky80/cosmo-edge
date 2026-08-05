#include "catch_amalgamated.hpp"

#ifdef COSMO_MEDIA_USE_ASCEND_BACKEND

#include <cstring>
#include <memory>

extern "C" {
#include "libavutil/buffer.h"
#include "libavutil/frame.h"
}

#include "media/FrameSurface.h"
#include "media/VideoCodecType.h"
#include "media/VideoDecoderAscend.h"

namespace cosmo::media {
namespace {

    AVFrame MakeHostNv12Frame(int width, int height, int luma_pitch = 0, int chroma_pitch = 0) {
        AVFrame frame{};
        frame.format      = AV_PIX_FMT_NV12;
        frame.width       = width;
        frame.height      = height;
        frame.linesize[0] = luma_pitch == 0 ? width : luma_pitch;
        frame.linesize[1] = chroma_pitch == 0 ? width : chroma_pitch;

        const size_t luma_size   = static_cast<size_t>(frame.linesize[0]) * static_cast<size_t>(height);
        const size_t chroma_size = static_cast<size_t>(frame.linesize[1]) * static_cast<size_t>(height / 2);
        auto* luma               = static_cast<uint8_t*>(av_mallocz(luma_size));
        auto* chroma             = static_cast<uint8_t*>(av_mallocz(chroma_size));
        REQUIRE(luma != nullptr);
        REQUIRE(chroma != nullptr);
        frame.buf[0] =
            av_buffer_create(luma, luma_size, [](void*, uint8_t* data) { av_free(data); }, nullptr, 0);
        frame.buf[1] =
            av_buffer_create(chroma, chroma_size, [](void*, uint8_t* data) { av_free(data); }, nullptr, 0);
        REQUIRE(frame.buf[0] != nullptr);
        REQUIRE(frame.buf[1] != nullptr);
        frame.data[0] = frame.buf[0]->data;
        frame.data[1] = frame.buf[1]->data;
        return frame;
    }

}  // namespace

TEST_CASE("Ascend decoder picks the Gate 0 hardware codec names", "[ascend][decoder]") {
    REQUIRE(std::string(AscendDecoderNameForCodec(VideoCodecType::kH264)) == "h264_ascend");
    REQUIRE(std::string(AscendDecoderNameForCodec(VideoCodecType::kH265)) == "h265_ascend");
    REQUIRE(AscendDecoderNameForCodec(VideoCodecType::kMjpeg) == nullptr);
}

TEST_CASE("Ascend decoder negotiation refuses non-NV12 output", "[ascend][decoder]") {
    const AVPixelFormat nv12_formats[] = {AV_PIX_FMT_NV12, AV_PIX_FMT_NONE};
    REQUIRE(SelectAscendNv12PixelFormat(nv12_formats) == AV_PIX_FMT_NV12);

    const AVPixelFormat mixed_formats[] = {AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12, AV_PIX_FMT_NONE};
    REQUIRE(SelectAscendNv12PixelFormat(mixed_formats) == AV_PIX_FMT_NV12);

    const AVPixelFormat host_formats[] = {AV_PIX_FMT_NV12, AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE};
    REQUIRE(SelectAscendNv12PixelFormat(host_formats) == AV_PIX_FMT_NV12);

    const AVPixelFormat no_nv12[] = {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_NONE};
    REQUIRE(SelectAscendNv12PixelFormat(no_nv12) == AV_PIX_FMT_NONE);
    REQUIRE(SelectAscendNv12PixelFormat(nullptr) == AV_PIX_FMT_NONE);
}

TEST_CASE("Ascend NV12 host surface extraction keeps plane metadata", "[ascend][decoder]") {
    AVFrame frame = MakeHostNv12Frame(64, 32);
    FrameSurface surface;
    std::string error;
    REQUIRE(BuildAscendNv12Surface(frame, surface, error));
    REQUIRE(surface.memory_type == FrameSurfaceMemoryType::Host);
    REQUIRE(surface.planes.size() == 2);
    CHECK(surface.planes[0].offset == 0);
    CHECK(surface.planes[0].pitch == 64);
    CHECK(surface.planes[0].vertical_stride == 32);
    CHECK(surface.planes[0].size == 64 * 32);
    CHECK(surface.planes[0].virt_addr == frame.data[0]);
    CHECK(surface.planes[1].offset == 0);
    CHECK(surface.planes[1].pitch == 64);
    CHECK(surface.planes[1].vertical_stride == 16);
    CHECK(surface.planes[1].size == 64 * 16);
    CHECK(surface.planes[1].virt_addr == frame.data[1]);
    av_frame_unref(&frame);
}

TEST_CASE("Ascend NV12 host surface extraction keeps padded plane metadata", "[ascend][decoder]") {
    // Ascend decoders align rows (e.g. 1920x1080 frames carry pitch 1920 but
    // vertical strides of 1088/544, mirroring the MPP layout contract).
    AVFrame frame = MakeHostNv12Frame(1920, 1080, 1920, 1920);
    av_buffer_unref(&frame.buf[0]);
    av_buffer_unref(&frame.buf[1]);
    auto* luma   = static_cast<uint8_t*>(av_mallocz(static_cast<size_t>(1920) * 1088));
    auto* chroma = static_cast<uint8_t*>(av_mallocz(static_cast<size_t>(1920) * 544));
    REQUIRE(luma != nullptr);
    REQUIRE(chroma != nullptr);
    frame.buf[0] = av_buffer_create(
        luma, static_cast<size_t>(1920) * 1088, [](void*, uint8_t* data) { av_free(data); }, nullptr, 0);
    frame.buf[1] = av_buffer_create(
        chroma, static_cast<size_t>(1920) * 544, [](void*, uint8_t* data) { av_free(data); }, nullptr, 0);
    REQUIRE(frame.buf[0] != nullptr);
    REQUIRE(frame.buf[1] != nullptr);
    frame.data[0] = frame.buf[0]->data;
    frame.data[1] = frame.buf[1]->data;

    FrameSurface surface;
    std::string error;
    REQUIRE(BuildAscendNv12Surface(frame, surface, error));
    CHECK(surface.planes[0].vertical_stride == 1080);
    CHECK(surface.planes[0].size == 1920 * 1088);
    CHECK(surface.planes[1].vertical_stride == 540);
    CHECK(surface.planes[1].size == 1920 * 544);
    av_frame_unref(&frame);
}

TEST_CASE("Ascend NV12 host surface extraction rejects invalid decoder output", "[ascend][decoder]") {
    SECTION("non NV12 frame") {
        AVFrame frame{};
        frame.format = AV_PIX_FMT_YUV420P;
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildAscendNv12Surface(frame, surface, error));
        CHECK(error == "decoded frame format is not NV12");
    }

    SECTION("device-backed frame context") {
        AVFrame frame = MakeHostNv12Frame(64, 32);
        // av_frame_unref releases hw_frames_ctx alongside buf[], so no
        // separate unref is needed for the context.
        frame.hw_frames_ctx = av_buffer_allocz(1);
        REQUIRE(frame.hw_frames_ctx != nullptr);
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildAscendNv12Surface(frame, surface, error));
        CHECK(error == "device-backed frame context is not supported by the host NV12 surface path");
        av_frame_unref(&frame);
    }

    SECTION("odd dimensions") {
        AVFrame frame = MakeHostNv12Frame(63, 32);
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildAscendNv12Surface(frame, surface, error));
        CHECK(error == "decoded frame has invalid dimensions");
        av_frame_unref(&frame);
    }

    SECTION("missing plane pointer") {
        AVFrame frame = MakeHostNv12Frame(64, 32);
        frame.data[1] = nullptr;
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildAscendNv12Surface(frame, surface, error));
        CHECK(error == "NV12 frame plane pointers missing");
        av_frame_unref(&frame);
    }

    SECTION("missing AVBufferRef backing") {
        AVFrame frame = MakeHostNv12Frame(64, 32);
        av_buffer_unref(&frame.buf[1]);
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildAscendNv12Surface(frame, surface, error));
        CHECK(error == "NV12 frame planes must have AVBufferRef backing");
        av_frame_unref(&frame);
    }

    SECTION("pitch shorter than width") {
        AVFrame frame     = MakeHostNv12Frame(64, 32);
        frame.linesize[1] = 32;
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildAscendNv12Surface(frame, surface, error));
        CHECK(error == "NV12 frame plane pitch is shorter than the decoded width");
        av_frame_unref(&frame);
    }

    SECTION("plane pointer outside backing buffer") {
        AVFrame frame = MakeHostNv12Frame(64, 32);
        frame.data[0] = frame.buf[0]->data + frame.buf[0]->size;
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildAscendNv12Surface(frame, surface, error));
        CHECK(error == "NV12 plane pointer is outside its backing buffer");
        av_frame_unref(&frame);
    }

    SECTION("plane span exceeds backing buffer") {
        AVFrame frame     = MakeHostNv12Frame(64, 32);
        frame.linesize[1] = 128;
        FrameSurface surface;
        std::string error;
        REQUIRE_FALSE(BuildAscendNv12Surface(frame, surface, error));
        CHECK(error == "NV12 plane span exceeds its backing buffer");
        av_frame_unref(&frame);
    }
}

}  // namespace cosmo::media

#endif
