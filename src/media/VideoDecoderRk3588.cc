#include "media/VideoDecoderRk3588.h"

#include <cerrno>
#include <cstring>
#include <memory>
#include <string>

#include <drm/drm_fourcc.h>

#include "util/Log.h"

namespace {

constexpr int kNv12PlaneCount = 2;

}  // namespace

namespace cosmo::media {

const char* Rk3588DecoderNameForCodec(VideoCodecType codec_type) {
    switch (codec_type) {
        case VideoCodecType::kH264:
            return "h264_rkmpp";
        case VideoCodecType::kH265:
            return "hevc_rkmpp";
        default:
            return nullptr;
    }
}

AVPixelFormat SelectRkDrmPrimePixelFormat(const AVPixelFormat* formats) {
    if (formats == nullptr) {
        return AV_PIX_FMT_NONE;
    }
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == AV_PIX_FMT_DRM_PRIME) {
            return AV_PIX_FMT_DRM_PRIME;
        }
    }
    return AV_PIX_FMT_NONE;
}

bool BuildRkDrmPrimeSurface(const AVFrame& frame, FrameSurface& surface, std::string& error) {
    if (frame.format != AV_PIX_FMT_DRM_PRIME) {
        error = "decoded frame is not DRM PRIME";
        return false;
    }
    if (frame.data[0] == nullptr) {
        error = "DRM PRIME descriptor missing";
        return false;
    }

    const auto& descriptor = *reinterpret_cast<const AVDRMFrameDescriptor*>(frame.data[0]);
    if (descriptor.nb_layers != 1) {
        error = "DRM PRIME frame must contain exactly one layer";
        return false;
    }
    if (descriptor.nb_objects <= 0) {
        error = "DRM PRIME frame has no objects";
        return false;
    }

    const auto& layer = descriptor.layers[0];
    if (layer.format != DRM_FORMAT_NV12) {
        error = "DRM PRIME layer is not NV12";
        return false;
    }
    if (layer.nb_planes != kNv12PlaneCount) {
        error = "NV12 DRM PRIME frame must expose two planes";
        return false;
    }

    const auto& luma   = layer.planes[0];
    const auto& chroma = layer.planes[1];
    if (luma.object_index < 0 || luma.object_index >= descriptor.nb_objects ||
        chroma.object_index != luma.object_index) {
        error = "NV12 DRM PRIME planes must share one DMA-BUF object";
        surface.planes.clear();
        return false;
    }

    const auto& object = descriptor.objects[luma.object_index];
    if (object.fd < 0 || object.size == 0) {
        error = "DRM PRIME object is missing a valid DMA-BUF fd";
        surface.planes.clear();
        return false;
    }
    if (luma.offset < 0 || luma.pitch <= 0 || chroma.offset < 0 || chroma.pitch <= 0 ||
        static_cast<size_t>(luma.offset) >= static_cast<size_t>(object.size) ||
        static_cast<size_t>(chroma.offset) >= static_cast<size_t>(object.size)) {
        error = "DRM PRIME plane has invalid offset or pitch";
        surface.planes.clear();
        return false;
    }

    const size_t luma_pitch = static_cast<size_t>(luma.pitch);
    if (luma_pitch != static_cast<size_t>(chroma.pitch)) {
        error = "NV12 DRM PRIME plane pitches must match";
        surface.planes.clear();
        return false;
    }
    const size_t luma_offset   = static_cast<size_t>(luma.offset);
    const size_t chroma_offset = static_cast<size_t>(chroma.offset);
    if (chroma_offset <= luma_offset) {
        error = "NV12 chroma must start after the luma plane";
        surface.planes.clear();
        return false;
    }
    // MPP returns one DMA-BUF holding both NV12 planes: the luma vertical
    // stride spans [luma.offset, chroma.offset) and the chroma plane occupies
    // half of the luma rows. `size` describes the shared backing allocation.
    const size_t luma_stride = (chroma_offset - luma_offset) / luma_pitch;
    if (luma_stride < static_cast<size_t>(frame.height) || (luma_stride % 2) != 0) {
        error = "DRM PRIME luma plane stride is shorter than the decoded height";
        surface.planes.clear();
        return false;
    }

    surface.memory_type = FrameSurfaceMemoryType::DmaBuf;
    surface.planes      = {
        {object.fd, nullptr, luma_offset, luma_pitch, luma_stride, static_cast<size_t>(object.size)},
        {object.fd, nullptr, chroma_offset, luma_pitch, luma_stride / 2, static_cast<size_t>(object.size)},
    };
    if (!surface.IsValid()) {
        error = "DRM PRIME plane size metadata is invalid";
        surface.planes.clear();
        return false;
    }
    return true;
}

VideoDecoderRk3588::VideoDecoderRk3588(size_t name) : VideoDecoder(name) {}

VideoDecoderRk3588::~VideoDecoderRk3588() {
    Close();
}

AVPixelFormat VideoDecoderRk3588::GetFormat(AVCodecContext* ctx, const AVPixelFormat* formats) {
    const AVPixelFormat selected = SelectRkDrmPrimePixelFormat(formats);
    if (selected == AV_PIX_FMT_NONE) {
        LOG_WARN("{} RK3588 decoder refused non-DRM pixel format negotiation", ctx ? ctx->codec_type : -1);
    }
    return selected;
}

bool VideoDecoderRk3588::Open() {
    const char* decoder_name = Rk3588DecoderNameForCodec(codec_type_);
    if (decoder_name == nullptr) {
        LOG_WARN("{} RK3588 decoder unsupported codec type ({})", idx_name_, static_cast<int>(codec_type_));
        return false;
    }

    const AVCodec* codec = avcodec_find_decoder_by_name(decoder_name);
    if (codec == nullptr) {
        LOG_WARN("{} RK3588 decoder {} is unavailable", idx_name_, decoder_name);
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (codec_ctx_ == nullptr) {
        LOG_WARN("{} RK3588 avcodec_alloc_context3 failed", idx_name_);
        return false;
    }

    codec_ctx_->thread_count = 1;
    codec_ctx_->opaque       = this;
    codec_ctx_->get_format   = &VideoDecoderRk3588::GetFormat;

    // Disable AFBC so MPP returns linear NV12 DMA-BUF surfaces. The validated
    // reference pipeline opens rkmpp the same way; AFBC buffers would change
    // the RGA input layout and shift detections.
    AVDictionary* decoder_options = nullptr;
    av_dict_set(&decoder_options, "afbc", "0", 0);
    const int open_ret = avcodec_open2(codec_ctx_, codec, &decoder_options);
    av_dict_free(&decoder_options);
    if (open_ret < 0) {
        LOG_WARN("{} RK3588 avcodec_open2 failed for {}", idx_name_, decoder_name);
        avcodec_free_context(&codec_ctx_);
        return false;
    }

    av_frame_  = av_frame_alloc();
    av_packet_ = av_packet_alloc();
    if (av_frame_ == nullptr || av_packet_ == nullptr) {
        LOG_WARN("{} RK3588 av_frame_alloc/av_packet_alloc failed", idx_name_);
        Close();
        return false;
    }

    opened_ = true;
    LOG_INFO("{} RK3588 decoder opened with {}", idx_name_, decoder_name);
    return true;
}

bool VideoDecoderRk3588::Close() {
    opened_ = false;
    if (av_frame_ != nullptr) {
        av_frame_free(&av_frame_);
    }
    if (av_packet_ != nullptr) {
        av_packet_free(&av_packet_);
    }
    if (codec_ctx_ != nullptr) {
        avcodec_free_context(&codec_ctx_);
    }
    return true;
}

bool VideoDecoderRk3588::IsOpened() {
    return opened_;
}

bool VideoDecoderRk3588::SendPacket(const uint8_t* pkt, size_t len, int64_t frame_idx) {
    if (!opened_ || codec_ctx_ == nullptr || av_packet_ == nullptr) {
        LOG_WARN("{} RK3588 decoder send before open", idx_name_);
        return false;
    }

    av_packet_unref(av_packet_);
    av_packet_->data = const_cast<uint8_t*>(pkt);
    av_packet_->size = static_cast<int>(len);
    av_packet_->dts  = frame_idx;
    av_packet_->pts  = frame_idx;

    const int ret = avcodec_send_packet(codec_ctx_, av_packet_);
    if (ret < 0) {
        LOG_WARN("{} RK3588 avcodec_send_packet failed: {}", idx_name_, ret);
        return false;
    }
    return true;
}

VideoFramePtr VideoDecoderRk3588::GetFrame() {
    if (!opened_ || codec_ctx_ == nullptr || av_frame_ == nullptr) {
        return nullptr;
    }

    const int ret = avcodec_receive_frame(codec_ctx_, av_frame_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return nullptr;
    }
    if (ret < 0) {
        LOG_WARN("{} RK3588 avcodec_receive_frame failed: {}", idx_name_, ret);
        return nullptr;
    }

    auto owned_frame = std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame* frame) {
        if (frame != nullptr) {
            av_frame_free(&frame);
        }
    });
    if (!owned_frame) {
        LOG_WARN("{} RK3588 failed to allocate frame lifetime holder", idx_name_);
        av_frame_unref(av_frame_);
        return nullptr;
    }
    av_frame_move_ref(owned_frame.get(), av_frame_);

    FrameSurface surface;
    std::string error;
    if (!BuildRkDrmPrimeSurface(*owned_frame, surface, error)) {
        LOG_WARN("{} RK3588 DRM PRIME validation failed: {}", idx_name_, error);
        return nullptr;
    }
    surface.lifetime = owned_frame;

    width_  = static_cast<size_t>(owned_frame->width);
    height_ = static_cast<size_t>(owned_frame->height);

    auto frame_surface = std::make_shared<FrameSurface>(std::move(surface));
    auto frame         = std::make_shared<VideoFrame>(static_cast<int>(width_), static_cast<int>(height_),
                                                      PixelFormat::PIXEL_NV12, frame_surface);
    if (!frame || !frame->Active()) {
        LOG_WARN("{} RK3588 VideoFrame allocation failed", idx_name_);
        return nullptr;
    }

    frame->SetFrameIndex(static_cast<uint64_t>(
        owned_frame->best_effort_timestamp >= 0 ? owned_frame->best_effort_timestamp : owned_frame->pkt_dts));
    return frame;
}

}  // namespace cosmo::media
