#include "media/VideoDecoderRk3588.h"

#include <cerrno>
#include <cstring>
#include <memory>
#include <string>

#include <drm/drm_fourcc.h>

#include "util/Log.h"

namespace {

constexpr int kNv12PlaneCount = 2;

size_t PlaneEndOffset(const AVDRMLayerDescriptor& layer, int plane_index, int object_index, size_t object_size) {
    size_t end_offset = object_size;
    for (int candidate_index = 0; candidate_index < layer.nb_planes; ++candidate_index) {
        if (candidate_index == plane_index) {
            continue;
        }
        const auto& candidate = layer.planes[candidate_index];
        if (candidate.object_index != object_index || candidate.offset < 0) {
            continue;
        }
        const size_t candidate_offset = static_cast<size_t>(candidate.offset);
        if (candidate_offset > static_cast<size_t>(layer.planes[plane_index].offset) && candidate_offset < end_offset) {
            end_offset = candidate_offset;
        }
    }
    return end_offset;
}

size_t PlaneSizeFromDescriptor(const AVDRMObjectDescriptor& object, const AVDRMLayerDescriptor& layer,
                               int plane_index) {
    const auto& plane = layer.planes[plane_index];
    if (plane.pitch <= 0 || plane.offset < 0 || static_cast<size_t>(plane.offset) >= object.size) {
        return 0;
    }
    const size_t end_offset = PlaneEndOffset(layer, plane_index, plane.object_index, object.size);
    if (end_offset <= static_cast<size_t>(plane.offset)) {
        return 0;
    }
    const size_t remaining = end_offset - static_cast<size_t>(plane.offset);
    return remaining - (remaining % static_cast<size_t>(plane.pitch));
}

size_t PlaneVerticalStrideFromDescriptor(const AVDRMObjectDescriptor& object, const AVDRMLayerDescriptor& layer,
                                         int plane_index) {
    const auto& plane       = layer.planes[plane_index];
    const size_t plane_size = PlaneSizeFromDescriptor(object, layer, plane_index);
    if (plane_size == 0 || plane.pitch <= 0) {
        return 0;
    }
    return plane_size / static_cast<size_t>(plane.pitch);
}

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

    surface.memory_type = FrameSurfaceMemoryType::DmaBuf;
    surface.planes.clear();
    surface.planes.reserve(static_cast<size_t>(layer.nb_planes));

    for (int plane_index = 0; plane_index < layer.nb_planes; ++plane_index) {
        const auto& plane = layer.planes[plane_index];
        if (plane.object_index < 0 || plane.object_index >= descriptor.nb_objects) {
            error = "DRM PRIME plane references an invalid object";
            surface.planes.clear();
            return false;
        }

        const auto& object = descriptor.objects[plane.object_index];
        if (object.fd < 0 || object.size == 0) {
            error = "DRM PRIME object is missing a valid DMA-BUF fd";
            surface.planes.clear();
            return false;
        }
        if (plane.offset < 0 || plane.pitch <= 0) {
            error = "DRM PRIME plane has invalid offset or pitch";
            surface.planes.clear();
            return false;
        }

        FramePlane frame_plane;
        frame_plane.fd              = object.fd;
        frame_plane.offset          = static_cast<size_t>(plane.offset);
        frame_plane.pitch           = static_cast<size_t>(plane.pitch);
        frame_plane.vertical_stride = PlaneVerticalStrideFromDescriptor(object, layer, plane_index);
        frame_plane.size            = PlaneSizeFromDescriptor(object, layer, plane_index);
        if (frame_plane.vertical_stride == 0 || frame_plane.size == 0 || !frame_plane.IsValid(surface.memory_type)) {
            error = "DRM PRIME plane size metadata is invalid";
            surface.planes.clear();
            return false;
        }

        surface.planes.push_back(frame_plane);
    }

    if (surface.planes[0].vertical_stride < static_cast<size_t>(frame.height)) {
        error = "DRM PRIME luma plane stride is shorter than decoded height";
        surface.planes.clear();
        return false;
    }
    if (surface.planes[1].vertical_stride * 2 != surface.planes[0].vertical_stride) {
        error = "DRM PRIME chroma plane stride does not match NV12 layout";
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

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
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
    auto frame = std::make_shared<VideoFrame>(static_cast<int>(width_), static_cast<int>(height_),
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
