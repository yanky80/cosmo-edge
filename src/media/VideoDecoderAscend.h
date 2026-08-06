#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

#ifndef AV_PIX_FMT_ASCEND
// The custom Ascend FFmpeg 4.4.1 fork (libavutil 56.70.100, installed as
// /opt/ffmpeg-4.4.1/ascend on the 310P3 host) inserts AV_PIX_FMT_ASCEND
// between D3D11VA_VLD and CUDA and documents frame->data[i] as device
// addresses "exactly as for system memory frames" (CUDA-style). System FFmpeg
// headers lack the enumerator, so define the fork's value (CUDA - 1) to keep
// the decoder source and its hermetic contract checks buildable off-host.
#define AV_PIX_FMT_ASCEND ((AVPixelFormat)(AV_PIX_FMT_CUDA - 1))
#endif

namespace cosmo::media {

/// Gate 0 codec name for the test host's custom Ascend FFmpeg
/// (h264_ascend/h265_ascend), or nullptr for unsupported codecs.
const char* AscendDecoderNameForCodec(VideoCodecType codec_type);

/// Picks the Gate 0 pixel format from the decoder's offered formats:
/// AV_PIX_FMT_ASCEND (device frames, direct DVPP input) preferred, host
/// AV_PIX_FMT_NV12 (stage-one transfer path) as fallback for sources that
/// cannot export a device surface. Returns AV_PIX_FMT_NONE when neither is
/// offered so avcodec_open2 fails instead of falling back to software.
AVPixelFormat SelectAscendPixelFormat(const AVPixelFormat* formats);

/// Validates a decoded host NV12 AVFrame and fills plane metadata
/// (pointers, pitch, rows, backing size) into a Host FrameSurface.
/// Stage-one path: the decoded frame lives in host memory.
bool BuildAscendNv12Surface(const AVFrame& frame, FrameSurface& surface, std::string& error);

/// Validates a decoded AV_PIX_FMT_ASCEND device AVFrame and fills plane
/// metadata into a Device FrameSurface. Per the recorded 310P3 test-host ABI
/// (issue #32, hwcontext_ascend.c ascend_get_buffer):
///   - one AVBufferRef (buf[0]) holding the whole NV12 frame, allocated with
///     hi_mpi_dvpp_malloc; buf[0]->data is a DVPP device address.
///   - data[0] = Y base (device address), data[1] = UV base at
///     data[0] + linesize[0] * vertical_stride (alignment-1 layout).
///   - hw_frames_ctx is set; the AVFrame owns the pool buffer ref.
/// The surface's lifetime must be bound to the AVFrame so the device memory
/// stays alive until DVPP consumes it.
bool BuildAscendDeviceSurface(const AVFrame& frame, FrameSurface& surface, std::string& error);

/// Downloads a Device (AV_PIX_FMT_ASCEND) FrameSurface to a contiguous host
/// NV12 buffer (pitch rows per plane, display height/2 chroma rows) for
/// preview/snapshot/OSD consumers. The inference path never touches this
/// function; a device surface can be copied on demand without changing it.
bool DownloadAscendDeviceSurfaceToHost(const FrameSurface& surface, int frame_width, int frame_height,
                                       std::vector<uint8_t>& host_nv12, std::string& error);

class VideoDecoderAscend final : public VideoDecoder {
public:
    explicit VideoDecoderAscend(size_t name);

    ~VideoDecoderAscend() override;

    bool Open() override;
    bool Close() override;
    bool IsOpened() override;
    bool Flush() override;
    bool ShouldReuseAcrossStreamChange() const override {
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
