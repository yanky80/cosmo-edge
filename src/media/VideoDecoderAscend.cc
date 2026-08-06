#include "media/VideoDecoderAscend.h"

#include <libavutil/error.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "util/Log.h"

namespace {

// Gate 0 locked baseline: the 310P3 card is device 0 (docs/development/
// ascend310p3-test-host-baseline.md). device_id is pinned at open time.
constexpr int kAscendDeviceId = 0;

std::string AvErrorText(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return text.data();
}

}  // namespace

namespace cosmo::media {

const char* AscendDecoderNameForCodec(VideoCodecType codec_type) {
    switch (codec_type) {
        case VideoCodecType::kH264:
            return "h264_ascend";
        case VideoCodecType::kH265:
            return "h265_ascend";
        default:
            return nullptr;
    }
}

AVPixelFormat SelectAscendPixelFormat(const AVPixelFormat* formats) {
    if (formats == nullptr) {
        return AV_PIX_FMT_NONE;
    }
    AVPixelFormat nv12 = AV_PIX_FMT_NONE;
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == AV_PIX_FMT_ASCEND) {
            // Device frames: DVPP consumes the AVFrame buffer directly, no
            // intermediate host image and no per-frame H2D upload.
            return AV_PIX_FMT_ASCEND;
        }
        if (nv12 == AV_PIX_FMT_NONE && *format == AV_PIX_FMT_NV12) {
            nv12 = AV_PIX_FMT_NV12;
        }
    }
    // Stage-one fallback: the source cannot export a device surface, so the
    // decoder hands out host NV12 and image_to_tensor uploads it (H2D).
    return nv12;
}

bool BuildAscendNv12Surface(const AVFrame& frame, FrameSurface& surface, std::string& error) {
    if (frame.format != AV_PIX_FMT_NV12) {
        error = "decoded frame format is not NV12";
        return false;
    }
    if (frame.hw_frames_ctx != nullptr) {
        error = "device-backed frame context is not supported by the host NV12 surface path";
        return false;
    }
    if (frame.width <= 0 || frame.height <= 0 || (frame.width % 2) != 0 || (frame.height % 2) != 0) {
        error = "decoded frame has invalid dimensions";
        return false;
    }
    if (frame.data[0] == nullptr || frame.data[1] == nullptr) {
        error = "NV12 frame plane pointers missing";
        return false;
    }
    if (frame.buf[0] == nullptr || frame.buf[1] == nullptr) {
        error = "NV12 frame planes must have AVBufferRef backing";
        return false;
    }
    if (frame.linesize[0] < frame.width || frame.linesize[1] < frame.width) {
        error = "NV12 frame plane pitch is shorter than the decoded width";
        return false;
    }

    const size_t width       = static_cast<size_t>(frame.width);
    const size_t height      = static_cast<size_t>(frame.height);
    const size_t luma_rows   = height;
    const size_t chroma_rows = height / 2;

    surface.planes.clear();
    for (int plane = 0; plane < 2; ++plane) {
        const uintptr_t plane_addr = reinterpret_cast<uintptr_t>(frame.data[plane]);
        const uintptr_t buf_addr   = reinterpret_cast<uintptr_t>(frame.buf[plane]->data);
        const size_t buf_size      = frame.buf[plane]->size;
        if (plane_addr < buf_addr || plane_addr - buf_addr >= buf_size) {
            error = "NV12 plane pointer is outside its backing buffer";
            return false;
        }
        const size_t offset = static_cast<size_t>(plane_addr - buf_addr);
        const size_t pitch  = static_cast<size_t>(frame.linesize[plane]);
        const size_t rows   = plane == 0 ? luma_rows : chroma_rows;
        if (pitch > buf_size - offset || pitch > (buf_size - offset) / rows) {
            error = "NV12 plane span exceeds its backing buffer";
            return false;
        }
        surface.planes.push_back(FramePlane{-1, frame.data[plane], offset, pitch, rows, buf_size});
    }

    surface.memory_type = FrameSurfaceMemoryType::Host;
    if (!surface.IsValid()) {
        error = "NV12 plane size metadata is invalid";
        surface.planes.clear();
        return false;
    }
    return true;
}

bool BuildAscendDeviceSurface(const AVFrame& frame, FrameSurface& surface, std::string& error) {
    if (frame.format != AV_PIX_FMT_ASCEND) {
        error = "decoded frame format is not the Ascend device format";
        return false;
    }
    if (frame.hw_frames_ctx == nullptr) {
        error = "Ascend device frame is missing its hw_frames_ctx";
        return false;
    }
    if (frame.width <= 0 || frame.height <= 0 || (frame.width % 2) != 0 || (frame.height % 2) != 0) {
        error = "decoded frame has invalid dimensions";
        return false;
    }
    if (frame.data[0] == nullptr || frame.data[1] == nullptr) {
        error = "device frame plane pointers missing";
        return false;
    }
    if (frame.buf[0] == nullptr || frame.buf[0]->data == nullptr || frame.buf[0]->size == 0) {
        error = "device frame must carry one AVBufferRef backing both planes";
        return false;
    }
    if (frame.linesize[1] != frame.linesize[0]) {
        error = "device frame planes must share the same pitch";
        return false;
    }
    const size_t pitch = static_cast<size_t>(frame.linesize[0]);
    if (pitch < static_cast<size_t>(frame.width) || (pitch % 2) != 0) {
        error = "device frame plane pitch is invalid";
        return false;
    }

    const uintptr_t buf_addr = reinterpret_cast<uintptr_t>(frame.buf[0]->data);
    const size_t buf_size    = frame.buf[0]->size;
    const uintptr_t y_addr   = reinterpret_cast<uintptr_t>(frame.data[0]);
    const uintptr_t uv_addr  = reinterpret_cast<uintptr_t>(frame.data[1]);
    const size_t y_offset    = y_addr - buf_addr;
    const size_t uv_offset   = uv_addr - buf_addr;
    if (uv_addr < y_addr || y_addr < buf_addr || uv_addr - buf_addr >= buf_size) {
        error = "device plane pointers are outside the backing buffer";
        return false;
    }
    if (uv_offset <= y_offset || uv_offset % pitch != 0) {
        error = "device frame UV plane offset is not pitch-aligned";
        return false;
    }
    // The fork's alignment-1 pool layout puts UV right after the Y rows, so
    // the true DVPP picture_height_stride is derived from the pointer
    // distance, not from frame->height (which carries the VDEC height stride
    // and can exceed the display height on padded frames).
    const size_t vertical_stride = (uv_offset - y_offset) / pitch;
    if (vertical_stride < static_cast<size_t>(frame.height) / 2 || (vertical_stride % 2) != 0 ||
        vertical_stride * pitch > buf_size - y_offset) {
        error = "device frame vertical stride or Y span exceeds its backing buffer";
        return false;
    }
    const size_t chroma_stride = vertical_stride / 2;
    if (uv_offset + chroma_stride * pitch > buf_size) {
        error = "device frame UV span exceeds its backing buffer";
        return false;
    }

    // FramePlane::GetPlaneData() is virt_addr + offset, so virt_addr is the
    // backing buffer base (buf[0]->data) and each plane carries its offset
    // within it, matching the RK3588 DMA-Buf and host NV12 plane contract.
    surface.planes.clear();
    surface.planes.push_back(FramePlane{-1, frame.buf[0]->data, y_offset, pitch, vertical_stride, buf_size});
    surface.planes.push_back(FramePlane{-1, frame.buf[0]->data, uv_offset, pitch, chroma_stride, buf_size});
    surface.memory_type = FrameSurfaceMemoryType::Device;
    if (!surface.IsValid()) {
        error = "device plane size metadata is invalid";
        surface.planes.clear();
        return false;
    }
    return true;
}

VideoDecoderAscend::VideoDecoderAscend(size_t name) : VideoDecoder(name) {}

VideoDecoderAscend::~VideoDecoderAscend() {
    Close();
}

AVPixelFormat VideoDecoderAscend::GetFormat(AVCodecContext* ctx, const AVPixelFormat* formats) {
    const AVPixelFormat selected = SelectAscendPixelFormat(formats);
    if (selected == AV_PIX_FMT_NONE) {
        LOG_WARN("{} Ascend decoder refused non-device/non-NV12 pixel format negotiation",
                 ctx == nullptr ? -1 : ctx->codec_type);
    }
    return selected;
}

bool VideoDecoderAscend::Open() {
    const char* decoder_name = AscendDecoderNameForCodec(codec_type_);
    if (decoder_name == nullptr) {
        LOG_WARN("{} Ascend decoder unsupported codec type ({})", idx_name_, static_cast<int>(codec_type_));
        return false;
    }

    // Explicitly select the Gate 0 hardware decoder by name; a missing
    // h264_ascend/h265_ascend fails the open instead of falling back to the
    // FFmpeg software codec.
    const AVCodec* codec = avcodec_find_decoder_by_name(decoder_name);
    if (codec == nullptr) {
        LOG_WARN("{} Ascend decoder {} is unavailable", idx_name_, decoder_name);
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (codec_ctx_ == nullptr) {
        LOG_WARN("{} Ascend avcodec_alloc_context3 failed", idx_name_);
        return false;
    }

    codec_ctx_->thread_count = 1;
    codec_ctx_->get_format   = &VideoDecoderAscend::GetFormat;

    // The Ascend decoder validates its resolution at open time (128x128 ~
    // 4096x4096) and rejects 0x0, so the demux-provided size is pinned here.
    codec_ctx_->width  = static_cast<int>(width_);
    codec_ctx_->height = static_cast<int>(height_);

    AVDictionary* decoder_options = nullptr;
    av_dict_set_int(&decoder_options, "device_id", kAscendDeviceId, 0);
    const int open_ret = avcodec_open2(codec_ctx_, codec, &decoder_options);
    av_dict_free(&decoder_options);
    if (open_ret < 0) {
        LOG_WARN("{} Ascend avcodec_open2 failed for {}: {}", idx_name_, decoder_name, AvErrorText(open_ret));
        avcodec_free_context(&codec_ctx_);
        return false;
    }

    av_frame_  = av_frame_alloc();
    av_packet_ = av_packet_alloc();
    if (av_frame_ == nullptr || av_packet_ == nullptr) {
        LOG_WARN("{} Ascend av_frame_alloc/av_packet_alloc failed", idx_name_);
        Close();
        return false;
    }

    opened_ = true;
    LOG_INFO("{} Ascend decoder opened with {}", idx_name_, decoder_name);
    return true;
}

bool VideoDecoderAscend::Close() {
    if (opened_ && codec_ctx_ != nullptr) {
        // Flush the hardware decoder before teardown so the HiMpi channel is
        // released cleanly; closing mid-stream without flushing leaves the
        // NPU channel wedged for the next decoder on the same device.
        Flush();
        if (av_frame_ != nullptr) {
            while (avcodec_receive_frame(codec_ctx_, av_frame_) >= 0) {
                av_frame_unref(av_frame_);
            }
        }
    }
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

bool VideoDecoderAscend::IsOpened() {
    return opened_;
}

bool VideoDecoderAscend::Flush() {
    if (!opened_ || codec_ctx_ == nullptr) {
        return false;
    }
    // End-of-stream through the BSF chain: the Ascend decoder only drains
    // frames still queued in the HiMpi channel once the stream end flag
    // reaches it, so a plain receive loop would return EAGAIN forever.
    const int ret = avcodec_send_packet(codec_ctx_, nullptr);
    if (ret < 0 && ret != AVERROR_EOF) {
        LOG_WARN("{} Ascend decoder flush failed: {}", idx_name_, AvErrorText(ret));
        return false;
    }
    return true;
}

bool VideoDecoderAscend::SendPacket(const uint8_t* pkt, size_t len, int64_t frame_idx) {
    if (!opened_ || codec_ctx_ == nullptr || av_packet_ == nullptr) {
        LOG_WARN("{} Ascend decoder send before open", idx_name_);
        return false;
    }

    av_packet_unref(av_packet_);
    av_packet_->data = const_cast<uint8_t*>(pkt);
    av_packet_->size = static_cast<int>(len);
    av_packet_->dts  = frame_idx;
    av_packet_->pts  = frame_idx;

    // avcodec_send_packet refs the payload into its own buffer synchronously,
    // so the borrowed pointer stays valid only for this call.
    const int ret = avcodec_send_packet(codec_ctx_, av_packet_);
    av_packet_unref(av_packet_);
    if (ret < 0) {
        LOG_WARN("{} Ascend avcodec_send_packet failed: {}", idx_name_, AvErrorText(ret));
        return false;
    }
    return true;
}

VideoFramePtr VideoDecoderAscend::GetFrame() {
    if (!opened_ || codec_ctx_ == nullptr || av_frame_ == nullptr) {
        return nullptr;
    }

    const int ret = avcodec_receive_frame(codec_ctx_, av_frame_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return nullptr;
    }
    if (ret < 0) {
        LOG_WARN("{} Ascend avcodec_receive_frame failed: {}", idx_name_, AvErrorText(ret));
        return nullptr;
    }

    auto owned_frame = std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame* frame) {
        if (frame != nullptr) {
            av_frame_free(&frame);
        }
    });
    if (!owned_frame) {
        LOG_WARN("{} Ascend failed to allocate frame lifetime holder", idx_name_);
        av_frame_unref(av_frame_);
        return nullptr;
    }
    av_frame_move_ref(owned_frame.get(), av_frame_);

    FrameSurface surface;
    std::string error;
    if (owned_frame->format == AV_PIX_FMT_ASCEND) {
        if (!BuildAscendDeviceSurface(*owned_frame, surface, error)) {
            LOG_WARN("{} Ascend device surface validation failed: {}", idx_name_, error);
            return nullptr;
        }
    } else if (!BuildAscendNv12Surface(*owned_frame, surface, error)) {
        LOG_WARN("{} Ascend NV12 surface validation failed: {}", idx_name_, error);
        return nullptr;
    }
    // The FrameSurface outlives this call through the AVFrame it references.
    surface.lifetime = owned_frame;

    width_  = static_cast<size_t>(owned_frame->width);
    height_ = static_cast<size_t>(owned_frame->height);

    auto frame_surface = std::make_shared<FrameSurface>(std::move(surface));
    auto frame         = std::make_shared<VideoFrame>(static_cast<int>(width_), static_cast<int>(height_),
                                                      PixelFormat::PIXEL_NV12, frame_surface);
    if (!frame || !frame->Active()) {
        LOG_WARN("{} Ascend VideoFrame allocation failed", idx_name_);
        return nullptr;
    }

    // The decoder's frame index and timestamp are both the demux packet index
    // the engine passed to SendPacket (pts/dts are set from frame_idx), which
    // is the key the engine's FrameInfoSave/FrameInfoGet uses to recover the
    // demuxed stream identity downstream.
    const int64_t timestamp =
        owned_frame->best_effort_timestamp >= 0 ? owned_frame->best_effort_timestamp : owned_frame->pkt_dts;
    frame->SetFrameIndex(static_cast<uint64_t>(timestamp));
    frame->SetTimestamp(timestamp);
    return frame;
}

}  // namespace cosmo::media
