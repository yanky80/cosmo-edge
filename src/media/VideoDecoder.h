#pragma once

#include <memory>
#include <string>

#include "media/VideoCodecType.h"
#include "media/VideoFrame.h"

namespace cosmo {
namespace media {

    class VideoDecoder {
    public:
        VideoDecoder(size_t name);

        virtual ~VideoDecoder();

        void SetCodecType(VideoCodecType type, int valWidth, int valHeight);

        virtual bool Open()  = 0;
        virtual bool Close() = 0;

        virtual bool IsOpened() = 0;

        /// Drains any buffered input through the decoder (EOS). Synchronous
        /// backends deliver every frame during the stream, so the default is a
        /// no-op; asynchronous hardware backends must override so in-flight
        /// frames are delivered, not dropped, before teardown or stream switch.
        virtual bool Flush();

        VideoFramePtr Decode(const uint8_t* pkt, size_t len, int64_t frame_idx, bool& result);

        virtual bool SendPacket(const uint8_t* pkt, size_t len, int64_t frame_idx) = 0;

        virtual VideoFramePtr GetFrame() = 0;

        size_t GetWidth() const;

        size_t GetHeight() const;

        /// Factory — creates the correct backend decoder (Sophon, CPU,
        /// RK3588, or Ascend).
        /// @param name       Decoder index / channel ID
        /// @param mediaHandle  Device handle (used by Sophon, ignored by
        ///                     CPU/Ascend; Ascend pins the Gate 0 device id)
        static std::unique_ptr<VideoDecoder> Create(size_t name, void* mediaHandle);

    protected:
        std::string idx_name_;
        std::string url_;

        VideoCodecType codec_type_{VideoCodecType::kInvalid};

        size_t width_  = 1920;
        size_t height_ = 1080;

        size_t send_pkt_cnt_   = 0;
        size_t recv_frame_cnt_ = 0;
    };

}  // namespace media
}  // namespace cosmo
