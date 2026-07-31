#pragma once

#include <cstdint>
#include <memory>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavutil/frame.h"
#include "libavutil/hwcontext_drm.h"
#ifdef __cplusplus
}
#endif

#include "media/FrameSurface.h"
#include "media/VideoDecoder.h"

namespace cosmo::media {

const char* Rk3588DecoderNameForCodec(VideoCodecType codec_type);

AVPixelFormat SelectRkDrmPrimePixelFormat(const AVPixelFormat* formats);

bool BuildRkDrmPrimeSurface(const AVFrame& frame, FrameSurface& surface, std::string& error);

class VideoDecoderRk3588 final : public VideoDecoder {
public:
    explicit VideoDecoderRk3588(size_t name);

    ~VideoDecoderRk3588() override;

    bool Open() override;
    bool Close() override;
    bool IsOpened() override;
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
