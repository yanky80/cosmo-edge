#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "catch_amalgamated.hpp"
#include "media/EncodedImageInfo.h"
#include "media/FrameSurface.h"
#include "media/PixelFormatUtils.h"
#include "media/VideoDecoder.h"
#include "media/VideoFrame.h"
#include "util/CipherUtil.h"
#include "util/VideoInfo.h"

#ifdef COSMO_NN_USE_SOPHON_BACKEND
#include <algorithm>
#include <chrono>
#include <vector>

#include "bmlib_runtime.h"
#include "media/IOsdTextRenderer.h"
#include "media/VideoEncoder.h"
#include "media/VideoFrameProcSophon.h"
#include "mem/DeviceContext.h"
#include "mem/IDeviceContext.h"
#include "service/detail/ServiceRegistry.h"
#include "service/infra/impl/MemoryPoolServiceImpl.h"
#endif

namespace cosmo::media {

namespace {

    class RejectingDecoder final : public VideoDecoder {
    public:
        RejectingDecoder() : VideoDecoder(0) {}

        bool Open() override {
            return true;
        }

        bool Close() override {
            return true;
        }

        bool IsOpened() override {
            return true;
        }

        bool SendPacket(const uint8_t*, size_t, int64_t) override {
            return false;
        }

        VideoFramePtr GetFrame() override {
            return nullptr;
        }
    };

}  // namespace

TEST_CASE("Decoder failure logging handles empty and short packets", "[video-frame-safety]") {
    RejectingDecoder decoder;
    bool result = true;
    REQUIRE(decoder.Decode(nullptr, 0, 1, result) == nullptr);
    REQUIRE_FALSE(result);

    const uint8_t byte = 0x7f;
    result             = true;
    REQUIRE(decoder.Decode(&byte, 1, 2, result) == nullptr);
    REQUIRE_FALSE(result);
}

TEST_CASE("Frame size calculation rejects unsafe dimensions", "[video-frame-safety]") {
    REQUIRE_FALSE(PixelFormatUtils::CalculateFrameSize(-1, 1080, PixelFormat::PIXEL_I420));
    REQUIRE_FALSE(PixelFormatUtils::CalculateFrameSize(1920, 0, PixelFormat::PIXEL_I420));
    REQUIRE_FALSE(PixelFormatUtils::CalculateFrameSize(1920, 1080, PixelFormat::PIXEL_UNKNOWN));
    REQUIRE_FALSE(PixelFormatUtils::CalculateFrameSize(1919, 1080, PixelFormat::PIXEL_I420));
    REQUIRE_FALSE(PixelFormatUtils::CalculateFrameSize(1920, 1079, PixelFormat::PIXEL_NV12));
    REQUIRE_FALSE(PixelFormatUtils::CalculateFrameSize(1919, 1080, PixelFormat::PIXEL_YUYV));
    REQUIRE_FALSE(PixelFormatUtils::CalculateFrameSize(
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), PixelFormat::PIXEL_RGB32F));

    const auto max_rgb_size =
        PixelFormatUtils::CalculateFrameSize(kVideoMaxWidth, kVideoMaxHeight, PixelFormat::PIXEL_RGB8);
    REQUIRE(max_rgb_size);
    REQUIRE(*max_rgb_size == static_cast<size_t>(kVideoFrameMaxSize));
    REQUIRE_FALSE(
        PixelFormatUtils::CalculateFrameSize(kVideoMaxWidth, kVideoMaxHeight, PixelFormat::PIXEL_RGBA8));

    const auto i420_size = PixelFormatUtils::CalculateFrameSize(1920, 1080, PixelFormat::PIXEL_I420);
    REQUIRE(i420_size);
    REQUIRE(*i420_size == 3110400);
}

TEST_CASE("Encoded image headers are checked before decoded-frame allocation",
          "[video-frame-safety][image]") {
    const auto one_pixel_png = util::DecBase64Vec(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAusB9Y9Zl1sAAAAASUVORK5CYII=");
    EncodedImageInfo info;
    REQUIRE(InspectEncodedImage(one_pixel_png, info));
    CHECK(info.width == 1);
    CHECK(info.height == 1);
    CHECK(info.pixel_count == 1);
    CHECK(IsEncodedImageWithinFrameCapability(info));

    std::vector<std::uint8_t> oversized_bmp(54, 0);
    oversized_bmp[0]  = 'B';
    oversized_bmp[1]  = 'M';
    oversized_bmp[2]  = 54;
    oversized_bmp[10] = 54;
    oversized_bmp[14] = 40;
    // BITMAPINFOHEADER width=8000, height=8000, planes=1, bpp=24.
    oversized_bmp[18] = 0x40;
    oversized_bmp[19] = 0x1f;
    oversized_bmp[22] = 0x40;
    oversized_bmp[23] = 0x1f;
    oversized_bmp[26] = 1;
    oversized_bmp[28] = 24;

    REQUIRE(InspectEncodedImage(oversized_bmp, info));
    CHECK(info.pixel_count == 64'000'000);
    CHECK_FALSE(IsEncodedImageWithinFrameCapability(info));
    CHECK(info.pixel_count > MaxDecodedImagePixels());

    CHECK_FALSE(InspectEncodedImage({0, 1, 2, 3}, info));
}

TEST_CASE("VideoFrame rejects unsafe dimensions before allocation", "[video-frame-safety]") {
    VideoFrame negative_width(-1, 1080, PixelFormat::PIXEL_I420);
    REQUIRE_FALSE(negative_width.Active());
    REQUIRE(negative_width.GetWidth() == 0);
    REQUIRE(negative_width.GetHeight() == 0);
    REQUIRE(negative_width.GetSize() == 0);

    VideoFrame oversized(kVideoMaxWidth, kVideoMaxHeight, PixelFormat::PIXEL_RGBA8);
    REQUIRE_FALSE(oversized.Active());
    REQUIRE(oversized.GetWidth() == 0);
    REQUIRE(oversized.GetHeight() == 0);
    REQUIRE(oversized.GetSize() == 0);
}

TEST_CASE("FrameSurface rejects invalid metadata before VideoFrame wraps it",
          "[video-frame-safety][frame-surface]") {
    FrameSurface host_surface;
    host_surface.memory_type = FrameSurfaceMemoryType::Host;
    FramePlane host_plane;
    host_plane.size  = 16;
    host_plane.pitch = 4;
    host_surface.planes.push_back(host_plane);
    REQUIRE_FALSE(host_surface.IsValid());

    FrameSurface dma_surface;
    dma_surface.memory_type = FrameSurfaceMemoryType::DmaBuf;
    FramePlane dma_plane;
    dma_plane.virt_addr = reinterpret_cast<uint8_t*>(0x1);
    dma_plane.size      = 16;
    dma_plane.pitch     = 4;
    dma_surface.planes.push_back(dma_plane);
    REQUIRE_FALSE(dma_surface.IsValid());

    FrameSurface bad_offsets;
    bad_offsets.memory_type = FrameSurfaceMemoryType::Host;
    FramePlane bad_offset_plane;
    bad_offset_plane.virt_addr = reinterpret_cast<uint8_t*>(0x1);
    bad_offset_plane.offset    = 8;
    bad_offset_plane.pitch     = 9;
    bad_offset_plane.size      = 16;
    bad_offsets.planes.push_back(bad_offset_plane);
    REQUIRE_FALSE(bad_offsets.IsValid());
}

TEST_CASE("VideoFrame wraps an external host surface without breaking host access",
          "[video-frame-safety][frame-surface]") {
    std::vector<uint8_t> storage(24, 0x5a);
    auto surface              = std::make_shared<FrameSurface>();
    surface->memory_type      = FrameSurfaceMemoryType::Host;
    FramePlane plane;
    plane.virt_addr = storage.data();
    plane.pitch     = 4;
    plane.size      = storage.size();
    surface->planes.push_back(plane);

    VideoFrame frame(4, 4, PixelFormat::PIXEL_I420, surface, 7, 99);
    REQUIRE(frame.Active());
    REQUIRE(frame.GetSurface() == surface);
    REQUIRE(frame.GetSize() == storage.size());
    REQUIRE(frame.GetData() == storage.data());
    REQUIRE(frame.GetHostData() == storage.data());
    REQUIRE(frame.GetFrameIndex() == 7);
    REQUIRE(frame.GetTimestamp() == 99);
}

TEST_CASE("VideoFrame rejects undersized external surfaces", "[video-frame-safety][frame-surface]") {
    std::vector<uint8_t> storage(8, 0x5a);
    auto surface         = std::make_shared<FrameSurface>();
    surface->memory_type = FrameSurfaceMemoryType::Host;
    FramePlane plane;
    plane.virt_addr = storage.data();
    plane.pitch     = 4;
    plane.size      = storage.size();
    surface->planes.push_back(plane);

    VideoFrame frame(4, 4, PixelFormat::PIXEL_I420, surface);
    REQUIRE_FALSE(frame.Active());
    REQUIRE(frame.GetData() == nullptr);
}

TEST_CASE("VideoFrame keeps external surface lifetime shared across consumers",
          "[video-frame-safety][frame-surface]") {
    auto storage         = std::make_shared<std::vector<uint8_t>>(24, 0x33);
    std::weak_ptr<void> weak_storage = storage;
    auto surface         = std::make_shared<FrameSurface>();
    surface->memory_type = FrameSurfaceMemoryType::Host;
    surface->lifetime    = storage;
    FramePlane plane;
    plane.virt_addr = storage->data();
    plane.pitch     = 4;
    plane.size      = storage->size();
    surface->planes.push_back(plane);

    auto frame    = std::make_shared<VideoFrame>(4, 4, PixelFormat::PIXEL_I420, surface, 1, 2);
    auto consumer = frame->GetSurface();
    storage.reset();
    surface.reset();

    REQUIRE_FALSE(weak_storage.expired());
    frame.reset();
    REQUIRE_FALSE(weak_storage.expired());
    consumer.reset();
    REQUIRE(weak_storage.expired());
}

TEST_CASE("VideoFrame move assignment transfers wrapped surface ownership once",
          "[video-frame-safety][frame-surface]") {
    auto storage    = std::make_shared<std::vector<uint8_t>>(24, 0x7c);
    auto surface    = std::make_shared<FrameSurface>();
    surface->memory_type = FrameSurfaceMemoryType::Host;
    surface->lifetime    = storage;
    FramePlane plane;
    plane.virt_addr = storage->data();
    plane.pitch     = 4;
    plane.size      = storage->size();
    surface->planes.push_back(plane);

    VideoFrame src(4, 4, PixelFormat::PIXEL_I420, surface, 8, 11);
    VideoFrame dst(-1, 2, PixelFormat::PIXEL_I420);
    dst = std::move(src);

    REQUIRE(dst.Active());
    REQUIRE(dst.GetSurface() == surface);
    REQUIRE(dst.GetData() == storage->data());
    REQUIRE_FALSE(src.Active());
    REQUIRE(src.GetSurface() == nullptr);
}

TEST_CASE("Crop ROI normalization safely clips and aligns recoverable input", "[video-frame-safety]") {
    constexpr int kMinDimension = 16;
    constexpr int kMaxDimension = 8192;
    const auto normalize        = [=](const util::Box& roi) {
        return PixelFormatUtils::NormalizeCropRoi(64, 64, PixelFormat::PIXEL_I420, roi, kMinDimension,
                                                         kMaxDimension);
    };

    REQUIRE(normalize({-1, 0, 16, 16}) == util::Box(0, 0, 16, 16));
    REQUIRE(normalize({0, -1, 16, 16}) == util::Box(0, 0, 16, 16));
    REQUIRE(normalize({0, 0, 6, 16}) == util::Box(0, 0, 16, 16));
    REQUIRE(normalize({0, 0, 15, 15}) == util::Box(0, 0, 16, 16));
    REQUIRE(normalize({30, 30, 1, 1}) == util::Box(23, 23, 16, 16));
    REQUIRE(normalize({60, 0, 8, 8}) == util::Box(48, 0, 16, 16));
    REQUIRE(normalize({0, 60, 8, 8}) == util::Box(0, 48, 16, 16));
    REQUIRE_FALSE(normalize({0, 0, 0, 16}));
    REQUIRE_FALSE(normalize({64, 0, 16, 16}));
    REQUIRE_FALSE(normalize({std::numeric_limits<int>::max(), 0, 16, 16}));
    REQUIRE_FALSE(normalize({1, 1, std::numeric_limits<int>::max(), 16}));
}

#ifdef COSMO_NN_USE_SOPHON_BACKEND
namespace {

    class StubOsdTextRenderer final : public IOsdTextRenderer {
    public:
        bool Init(const std::string&) override {
            return true;
        }

        bool IsReady() const override {
            return true;
        }

        TextBitmap RenderString(const std::string&, float) const override {
            return {};
        }

        OutlinedTextBitmap RenderStringWithOutline(const std::string&, float) const override {
            return {};
        }
    };

    class SophonMediaFixture {
    public:
        SophonMediaFixture() {
            device_context_ = std::make_unique<mem::DeviceContext>();
            service::ServiceRegistry::Instance().Set<mem::IDeviceContext>(device_context_.get());
            memory_pool_ = std::make_unique<service::MemoryPoolServiceImpl>();
            frame_proc_  = std::make_unique<VideoFrameProcSophon>(device_context_->GetMediaHandle(), osd_);
        }

        ~SophonMediaFixture() {
            frame_proc_.reset();
            memory_pool_.reset();
            service::ServiceRegistry::Instance().Set<mem::IDeviceContext>(nullptr);
            device_context_.reset();
        }

        SophonMediaFixture(const SophonMediaFixture&)            = delete;
        SophonMediaFixture& operator=(const SophonMediaFixture&) = delete;

        VideoFrameProcSophon& FrameProc() {
            return *frame_proc_;
        }

        void* MediaHandle() {
            return device_context_->GetMediaHandle();
        }

    private:
        StubOsdTextRenderer osd_;
        std::unique_ptr<mem::DeviceContext> device_context_;
        std::unique_ptr<service::MemoryPoolServiceImpl> memory_pool_;
        std::unique_ptr<VideoFrameProcSophon> frame_proc_;
    };

}  // namespace

TEST_CASE("Sophon crop rejects unsafe ROIs before VPP", "[sophon-crop][.device]") {
    SophonMediaFixture fixture;
    auto source = std::make_shared<VideoFrame>(64, 64, PixelFormat::PIXEL_I420, 42, 123456);
    REQUIRE(VideoFrameValid(source, true));

    const util::Box invalid_rois[] = {
        {0, 0, 0, 16},
        {0, 0, 16, -2},
        {64, 0, 16, 16},
        {0, 64, 16, 16},
        {std::numeric_limits<int>::max(), 0, 16, 16},
        {0, std::numeric_limits<int>::max(), 16, 16},
        {1, 1, std::numeric_limits<int>::max(), 16},
        {1, 1, 16, std::numeric_limits<int>::max()},
    };

    for (const auto& roi : invalid_rois) {
        INFO("ROI x=" << roi.x << " y=" << roi.y << " width=" << roi.width << " height=" << roi.height);
        REQUIRE(fixture.FrameProc().Crop(source, roi) == nullptr);
    }
}

TEST_CASE("Sophon crop normalizes recoverable ROIs", "[sophon-crop][.device]") {
    SophonMediaFixture fixture;
    auto source = std::make_shared<VideoFrame>(64, 64, PixelFormat::PIXEL_I420, 42, 123456);
    REQUIRE(VideoFrameValid(source, true));

    struct CropCase {
        util::Box roi;
        size_t expected_width;
        size_t expected_height;
    };
    const CropCase crop_cases[] = {
        {{1, 3, 15, 15}, 16, 16},
        {{60, 60, 8, 8}, 16, 16},
        {{-1, 0, 16, 16}, 16, 16},
        {{48, 48, 16, 16}, 16, 16},
    };

    for (const auto& crop_case : crop_cases) {
        INFO("ROI x=" << crop_case.roi.x << " y=" << crop_case.roi.y << " width=" << crop_case.roi.width
                      << " height=" << crop_case.roi.height);
        auto result = fixture.FrameProc().Crop(source, crop_case.roi);
        REQUIRE(VideoFrameValid(result, true));
        REQUIRE(result->GetWidth() == crop_case.expected_width);
        REQUIRE(result->GetHeight() == crop_case.expected_height);
        REQUIRE(result->GetFrameIndex() == source->GetFrameIndex());
        REQUIRE(result->GetTimestamp() == source->GetTimestamp());
    }
}

TEST_CASE("Sophon encoder survives startup without immediate output", "[sophon-encoder][.device]") {
    constexpr int kWidth  = 1280;
    constexpr int kHeight = 720;

    SophonMediaFixture fixture;
    auto encoder = VideoEncoder::Create(fixture.MediaHandle());
    REQUIRE(encoder);
    encoder->Set(VideoCodecType::kH264, kWidth, kHeight);
    REQUIRE(encoder->Open());

    auto frame = std::make_shared<VideoFrame>(kWidth, kHeight, PixelFormat::PIXEL_I420);
    REQUIRE(VideoFrameValid(frame, true));
    auto* device_memory = reinterpret_cast<bm_device_mem_t*>(frame->GetData());
    REQUIRE(device_memory != nullptr);

    std::vector<uint8_t> gray_frame(frame->GetSize(), 128);
    std::fill_n(gray_frame.begin(), static_cast<size_t>(kWidth) * kHeight, 16);
    REQUIRE(bm_memcpy_s2d_partial(reinterpret_cast<bm_handle_t>(fixture.MediaHandle()), *device_memory,
                                  gray_frame.data(),
                                  static_cast<unsigned int>(gray_frame.size())) == BM_SUCCESS);

    VideoPacketPtr last_packet;
    size_t packet_count                   = 0;
    size_t last_output_frame_index        = 0;
    size_t submitted_frame_count          = 0;
    constexpr size_t kExpectedPacketCount = 3;
    constexpr size_t kFrameCount          = 64;
    const auto deadline                   = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (size_t frame_index = 0; frame_index < kFrameCount; ++frame_index) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        frame->SetFrameIndex(frame_index);
        frame->SetTimestamp(static_cast<int64_t>(frame_index) * 40);
        auto packet = encoder->Encode(frame);
        ++submitted_frame_count;
        if (packet) {
            REQUIRE_FALSE(packet->data.empty());
            last_packet             = std::move(packet);
            last_output_frame_index = frame_index;
            ++packet_count;
        }
    }

    REQUIRE(submitted_frame_count == kFrameCount);
    REQUIRE(packet_count >= kExpectedPacketCount);
    REQUIRE(last_output_frame_index >= (kFrameCount * 3) / 4);
    REQUIRE(last_packet);
    REQUIRE(last_packet->width == kWidth);
    REQUIRE(last_packet->height == kHeight);
    REQUIRE(last_packet->codec_type == VideoCodecType::kH264);
}
#endif

}  // namespace cosmo::media
