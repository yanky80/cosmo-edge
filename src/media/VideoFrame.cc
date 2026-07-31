// VideoFrame — Legacy global alias — kept for backward compatibility.

#include "media/VideoFrame.h"

#include <thread>

#include "media/PixelFormatUtils.h"
#include "mem/MemoryPoolMng.h"
#include "util/Log.h"
#include "util/TimingConstants.h"
#include "util/UuidUtil.h"

namespace cosmo {
namespace media {

    VideoFrame::VideoFrame(int w, int h, PixelFormat fmt, uint64_t frameIndex, int64_t ts) {
        pixel_fmt_            = fmt;
        const auto frame_size = PixelFormatUtils::CalculateFrameSize(w, h, fmt);
        if (!frame_size) {
            LOG_WARN("Invalid frame dimensions or format: {}x{}, format {}", w, h, static_cast<int>(fmt));
            return;
        }

        width_   = static_cast<size_t>(w);
        height_  = static_cast<size_t>(h);
        channel_ = PixelFormatUtils::PixelFormatChannels(fmt);
        size_    = *frame_size;

        int index = 0;
        while (index++ <= 10) {
            auto block__ = mem::GetMemoryPool().Acquire(size_);
            if (block__) {
                block_ = block__;
                break;
            }

            std::this_thread::sleep_for(timing::kFastPollInterval);
        }

        if (!block_) {
            LOG_WARN("Malloc {} Failed", size_);
            return;
        }

        frame_index_ = frameIndex;
        timestamp_   = ts;
        uuid_        = util::GenerateUUID();
        active_      = true;
    }

    VideoFrame::VideoFrame(int w, int h, PixelFormat fmt, FrameSurfacePtr surface, uint64_t frameIndex,
                           int64_t ts) {
        pixel_fmt_ = fmt;
        if (!surface || !surface->IsValid()) {
            LOG_WARN("Invalid frame surface for {}x{}, format {}", w, h, static_cast<int>(fmt));
            return;
        }

        const auto frame_size = PixelFormatUtils::CalculateFrameSize(w, h, fmt);
        if (!frame_size) {
            LOG_WARN("Invalid frame dimensions or format: {}x{}, format {}", w, h, static_cast<int>(fmt));
            return;
        }

        width_   = static_cast<size_t>(w);
        height_  = static_cast<size_t>(h);
        channel_ = PixelFormatUtils::PixelFormatChannels(fmt);
        size_    = surface->TotalSize();
        if (size_ < *frame_size) {
            LOG_WARN("Surface size {} is too small for {}x{}, expected at least {}", size_, w, h,
                     *frame_size);
            return;
        }

        surface_     = std::move(surface);
        frame_index_ = frameIndex;
        timestamp_   = ts;
        uuid_        = util::GenerateUUID();
        active_      = true;
    }

    VideoFrame::VideoFrame(VideoFrame&& data) noexcept {
        *this = std::move(data);
    }

    VideoFrame& VideoFrame::operator=(VideoFrame&& data) noexcept {
        if (&data != this) {
            ReleaseOwnedResources();
            size_        = data.GetSize();
            width_       = data.GetWidth();
            height_      = data.GetHeight();
            channel_     = data.GetChannel();
            pixel_fmt_   = data.GetPixelFormat();
            frame_index_ = data.GetFrameIndex();
            timestamp_   = data.GetTimestamp();
            stream_index_ = data.GetStreamIndex();
            active_      = data.Active();
            uuid_        = util::GenerateUUID();

            block_           = data.block_;
            host_frame_data_ = data.host_frame_data_;
            surface_         = std::move(data.surface_);

            data.block_           = nullptr;
            data.host_frame_data_ = nullptr;

            data.Clear();
        }
        return *this;
    }

    void VideoFrame::Clear() {
        active_      = false;
        size_        = 0;
        width_       = 0;
        height_      = 0;
        channel_     = 3;
        pixel_fmt_   = PixelFormat::PIXEL_I420;
        frame_index_ = 0;
        timestamp_   = 0;
        stream_index_ = -1;
        surface_.reset();
    }

    void VideoFrame::ReleaseOwnedResources() {
        if (block_) {
            mem::GetMemoryPool().Recycle(block_);
            block_ = nullptr;
        }
        if (host_frame_data_) {
            free(host_frame_data_);
            host_frame_data_ = nullptr;
        }
    }

    VideoFrame::~VideoFrame() {
        ReleaseOwnedResources();
        Clear();
    }

    size_t VideoFrame::GetSize() const {
        return size_;
    }

    size_t VideoFrame::GetWidth() const {
        return width_;
    }

    void VideoFrame::SetWidth(size_t w) {
        width_ = w;
    }

    size_t VideoFrame::GetHeight() const {
        return height_;
    }

    void VideoFrame::SetHeight(size_t h) {
        height_ = h;
    }

    PixelFormat VideoFrame::GetPixelFormat() const {
        return pixel_fmt_;
    }

    void VideoFrame::SetFramePixelFormat(PixelFormat fmt) {
        pixel_fmt_ = fmt;
    }

    size_t VideoFrame::GetChannel() const {
        return channel_;
    }

    uint64_t VideoFrame::GetFrameIndex() const {
        return frame_index_;
    }

    void VideoFrame::SetFrameIndex(uint64_t sequence) {
        frame_index_ = sequence;
    }

    int64_t VideoFrame::GetTimestamp() const {
        return timestamp_;
    }

    void VideoFrame::SetTimestamp(int64_t t) {
        timestamp_ = t;
    }

    int64_t VideoFrame::GetStreamIndex() const {
        return stream_index_;
    }

    void VideoFrame::SetStreamIndex(int64_t streamIndex) {
        stream_index_ = streamIndex;
    }

    bool VideoFrame::Active() const {
        return active_;
    }

    std::string VideoFrame::GetName() const {
        return uuid_;
    }

    uint8_t* VideoFrame::GetHostData() {
        if (host_frame_data_) {
            return host_frame_data_;
        }
        if (surface_ && surface_->memory_type == FrameSurfaceMemoryType::Host) {
            return surface_->GetContiguousData();
        }
        return host_frame_data_;
    }

    void VideoFrame::SetHostData(uint8_t* data) {
        host_frame_data_ = data;
    }

    uint8_t* VideoFrame::GetData() {
        if (block_) {
            return block_->data;
        }
        if (surface_ && surface_->memory_type == FrameSurfaceMemoryType::Host) {
            return surface_->GetContiguousData();
        }
        return nullptr;
    }

    FrameSurfacePtr VideoFrame::GetSurface() const {
        return surface_;
    }

}  // namespace media
}  // namespace cosmo

bool VideoFrameValid(VideoFramePtr frame, bool blog) {
    if (!frame) {
        if (blog)
            LOG_WARN("{}", "Frame Invalid");
        return false;
    }

    if (frame->Active()) {
        return true;
    }

    if (blog)
        LOG_WARN("Frame Data Invalid {}x{}", frame->GetWidth(), frame->GetHeight());
    return false;
}
