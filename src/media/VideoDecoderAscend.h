#pragma once

#include <cstdint>
#include <memory>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavutil/frame.h"
#ifdef __cplusplus
}
#endif

#include "media/FrameSurface.h"
#include "media/VideoDecoder.h"

namespace cosmo::media {

/// Gate 0 codec name for the test host's custom Ascend FFmpeg
/// (h264_ascend/h265_ascend), or nullptr for unsupported codecs.
const char* AscendDecoderNameForCodec(VideoCodecType codec_type);

/// Picks AV_PIX_FMT_NV12 from the decoder's offered formats, or
/// AV_PIX_FMT_NONE when the hardware decoder cannot deliver host NV12.
/// Returning NONE makes avcodec_open2 fail instead of falling back.
AVPixelFormat SelectAscendNv12PixelFormat(const AVPixelFormat* formats);

/// Validates a decoded host NV12 AVFrame and fills plane metadata
/// (pointers, pitch, rows, backing size) into a Host FrameSurface.
bool BuildAscendNv12Surface(const AVFrame& frame, FrameSurface& surface, std::string& error);

class VideoDecoderAscend final : public VideoDecoder {
public:
    explicit VideoDecoderAscend(size_t name);

    ~VideoDecoderAscend() override;

    bool Open() override;
    bool Close() override;
    bool IsOpened() override;
    bool Flush() override;
    bool ReuseAcrossStreamChange() const override {
        return true;
    }
    bool SendPacket(const uint8_t* pkt, size_t len, int64_t frame_idx) override;
    VideoFramePtr GetFrame() override;

private:
    static AVPixelFormat GetFormat(AVCodecContext* ctx, const AVPixelFormat* formats);

    AVCodecContext* codec_ctx_{nullptr};
    AVFrame* av_frame_{nullptr};
    AVPacket* av_packet_{nullptr};
    bool opened_{false};
};

}  // namespace cosmo::media
