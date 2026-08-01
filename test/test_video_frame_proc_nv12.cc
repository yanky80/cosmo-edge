#include "catch_amalgamated.hpp"

#if defined(COSMO_MEDIA_USE_CPU_BACKEND) || defined(COSMO_MEDIA_USE_RK3588_BACKEND)

#include <cstdint>
#include <memory>

#include "media/IOsdTextRenderer.h"
#include "media/PixelFormat.h"
#include "media/VideoFrame.h"
#include "media/VideoFrameProcCpu.h"
#include "mem/AllocatorCpu.h"
#include "mem/MemoryPoolMng.h"

namespace {

class StubOsdTextRenderer final : public cosmo::media::IOsdTextRenderer {
public:
    bool Init(const std::string&) override {
        return true;
    }
    bool IsReady() const override {
        return true;
    }
    cosmo::media::IOsdTextRenderer::TextBitmap RenderString(const std::string&, float) const override {
        return {};
    }
    cosmo::media::IOsdTextRenderer::OutlinedTextBitmap RenderStringWithOutline(const std::string&,
                                                                               float) const override {
        return {};
    }
};

class ScopedMemoryPool {
public:
    ScopedMemoryPool() {
        pool_ = std::make_unique<cosmo::mem::MemoryPoolMng>(std::make_unique<cosmo::mem::AllocatorCpu>(),
                                                            std::vector<int>{16, 64, 256, 1024, 4096});
        cosmo::mem::SetMemoryPoolContext(pool_.get());
    }

    ~ScopedMemoryPool() {
        cosmo::mem::SetMemoryPoolContext(nullptr);
    }

private:
    std::unique_ptr<cosmo::mem::MemoryPoolMng> pool_;
};

}  // namespace

TEST_CASE("CPU NV12ToI420 converts Y and interleaved UV planes", "[media][nv12]") {
    ScopedMemoryPool pool;
    StubOsdTextRenderer osd;
    cosmo::media::VideoFrameProcCpu proc(osd);

    constexpr int width  = 4;
    constexpr int height = 4;
    auto nv12 =
        std::make_shared<cosmo::media::VideoFrame>(width, height, cosmo::media::PixelFormat::PIXEL_NV12);
    REQUIRE(nv12 != nullptr);
    REQUIRE(nv12->Active());

    auto* data = nv12->GetData();
    REQUIRE(data != nullptr);
    // Y plane: byte value = row * width + col.
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            data[row * width + col] = static_cast<uint8_t>(row * width + col);
        }
    }
    // Interleaved UV plane: even bytes U, odd bytes V.
    for (int row = 0; row < height / 2; ++row) {
        for (int col = 0; col < width; ++col) {
            data[width * height + row * width + col] = static_cast<uint8_t>(200 + row * 2 * width + col);
        }
    }

    auto i420 = proc.NV12ToI420(nv12);
    REQUIRE(i420 != nullptr);
    REQUIRE(i420->Active());
    CHECK(i420->GetPixelFormat() == cosmo::media::PixelFormat::PIXEL_I420);
    CHECK(i420->GetWidth() == static_cast<size_t>(width));
    CHECK(i420->GetHeight() == static_cast<size_t>(height));

    auto* out = i420->GetData();
    REQUIRE(out != nullptr);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            CHECK(out[row * width + col] == static_cast<uint8_t>(row * width + col));
        }
    }
    const auto* u = out + width * height;
    const auto* v = u + (width / 2) * (height / 2);
    for (int row = 0; row < height / 2; ++row) {
        for (int col = 0; col < width / 2; ++col) {
            CHECK(u[row * (width / 2) + col] == static_cast<uint8_t>(200 + row * 2 * width + col * 2));
            CHECK(v[row * (width / 2) + col] == static_cast<uint8_t>(200 + row * 2 * width + col * 2 + 1));
        }
    }
}

TEST_CASE("CPU NV12ToI420 rejects non-NV12 input", "[media][nv12]") {
    ScopedMemoryPool pool;
    StubOsdTextRenderer osd;
    cosmo::media::VideoFrameProcCpu proc(osd);

    auto i420 = std::make_shared<cosmo::media::VideoFrame>(4, 4, cosmo::media::PixelFormat::PIXEL_I420);
    REQUIRE(i420->Active());
    auto result = proc.NV12ToI420(i420);
    CHECK(result == nullptr);
}

#endif  // CPU or RK3588 media backend
