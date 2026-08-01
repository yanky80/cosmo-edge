#pragma once

#include "media/IVideoFrameProc.h"
#include "media/VideoFrameProcCpu.h"

namespace cosmo::media {

/// RK3588 media backend frame processing.
///
/// Decoded frames arrive as NV12 DRM PRIME DMA-BUF surfaces. The zero-copy
/// inference path consumes the surface directly and never goes through this
/// class. Preview, capture, OSD and recording consumers materialize a host
/// I420 copy on demand and reuse the generic CPU implementation.
class VideoFrameProcRk3588 final : public IVideoFrameProc {
public:
    explicit VideoFrameProcRk3588(IOsdTextRenderer& osd_service);
    ~VideoFrameProcRk3588() override;

    VideoFramePtr CopyFrame(VideoFramePtr srcImage) override;
    bool EnsureHostData(VideoFramePtr frame) override;

    // Color conversion
    VideoFramePtr BGR2I420(VideoFramePtr srcImage) override;
    VideoFramePtr RGB2I420(VideoFramePtr srcImage) override;
    VideoFramePtr I4202BGR(VideoFramePtr srcImage) override;
    VideoFramePtr I4202RGB(VideoFramePtr srcImage) override;
    VideoFramePtr NV12ToI420(VideoFramePtr srcImage) override;

    // Image processing
    VideoFramePtr Crop(const VideoFramePtr srcPicture, const util::Box roi) override;
    VideoFramePtr Resize(VideoFramePtr src, int dst_height, int dst_width) override;
    VideoFramePtr Padding(VideoFramePtr src, size_t top, size_t bottom, size_t left, size_t right,
                          Color color) override;

    // JPEG codec
    std::vector<u_char> EncodeJpeg(const VideoFramePtr srcPicture) override;
    VideoFramePtr DecodeJpeg(const std::vector<u_int8_t>& data) override;

    // Drawing
    VideoFramePtr DrawBox(VideoFramePtr srcImage, const util::Box imageRect, const Color& color,
                          int lineWidth = 2) override;
    VideoFramePtr DrawPoint(VideoFramePtr srcImage, util::Point point, const Color& color,
                            int lineWidth = 2) override;
    VideoFramePtr DrawLines(VideoFramePtr srcImage, std::vector<std::pair<util::Point, util::Point>> lines,
                            const Color& color, int lineWidth) override;
    VideoFramePtr DrawRects(VideoFramePtr srcImage, const std::vector<util::Box>& rects, const Color& color,
                            int lineWidth = 2) override;
    VideoFramePtr DrawText(VideoFramePtr srcImage, int x, int y, const std::string& text, const Color& color,
                           int fontSize = 50) override;

    // Session-based OSD
    bool BeginOSD(VideoFramePtr frame) override;
    void OSDDrawLines(std::vector<std::pair<util::Point, util::Point>> lines, const Color& color,
                      int lineWidth) override;
    void OSDDrawText(int x, int y, const std::string& text, const Color& color, int fontSize = 50) override;
    void OSDDrawTextEx(int x, int y, const std::string& text, const Color& color, int fontSize,
                       const Color& bgColor, uint8_t bgAlpha, bool outline = true,
                       int bgPadding = 4) override;
    void EndOSD() override;

private:
    // Host I420 copy of a NV12 DMA-BUF surface frame; returns the input
    // unchanged when it already carries host memory.
    VideoFramePtr MaterializeHost(VideoFramePtr frame) const;

    VideoFrameProcCpu cpu_;
};

}  // namespace cosmo::media
