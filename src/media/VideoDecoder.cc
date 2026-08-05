// VideoDecoder — Video Decoder implementation.

#include "media/VideoDecoder.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "util/Log.h"

static constexpr const char* kTag = "[DECODER] ";

namespace cosmo {
namespace media {

    VideoDecoder::VideoDecoder(size_t name) : idx_name_("atomicDecoder_" + std::to_string(name)) {
        LOG_INFO("{} Construction", idx_name_);
    }

    VideoDecoder::~VideoDecoder() {}

    bool VideoDecoder::Flush() {
        // Synchronous backends (CPU/Sophon/RK3588) deliver every frame during
        // the stream, so nothing is buffered to drain.
        return true;
    }

    void VideoDecoder::SetCodecType(VideoCodecType type, int valWidth, int valHeight) {
        codec_type_ = type;
        width_      = static_cast<size_t>(valWidth);
        height_     = static_cast<size_t>(valHeight);
    }

    size_t VideoDecoder::GetWidth() const {
        return width_;
    }

    size_t VideoDecoder::GetHeight() const {
        return height_;
    }

    VideoFramePtr VideoDecoder::Decode(const uint8_t* pkt, size_t len, int64_t frame_idx, bool& result) {
        auto send_result = SendPacket(pkt, len, frame_idx);
        send_pkt_cnt_ += 1;
        if (frame_idx % (25 * 60) == 0) {
            LOG_INFO("{} Decode {} Frames. Total Get Output {} Frames.", idx_name_, send_pkt_cnt_,
                     recv_frame_cnt_);
        }

        if (!send_result) {
            result = false;
            std::ostringstream prefix;
            const size_t prefix_size = pkt == nullptr ? 0 : std::min<size_t>(len, 6);
            for (size_t i = 0; i < prefix_size; ++i) {
                if (i != 0) {
                    prefix << ' ';
                }
                prefix << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(pkt[i]);
            }
            LOG_WARN("{}{} Frame Size:{} Decode Failed. frameIndex:{} data:{}", kTag, idx_name_, len,
                     frame_idx, prefix.str());
            return nullptr;
        }

        result = true;

        auto frame = GetFrame();
        if (frame) {
            recv_frame_cnt_ += 1;
        }

        return frame;
    }

}  // namespace media
}  // namespace cosmo
