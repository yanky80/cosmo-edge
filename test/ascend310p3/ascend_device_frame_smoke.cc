// Host-level runnable check for issue #32: decode with the custom Ascend
// FFmpeg into device (AV_PIX_FMT_ASCEND) FrameSurfaces and feed them directly
// into the real AscendImageToTensorNode DVPP pipeline, with no intermediate
// host image and no H2D upload of the decoded frame.
//
// The smoke drives the repository pieces where the full engine build is not
// available on the 310P3 host yet:
//   - media::VideoDecoderAscend (h264_ascend/h265_ascend, device NV12
//     surfaces; the recorded test-host ABI: one hi_mpi_dvpp_malloc'd
//     AVBufferRef, data[0]=Y base, data[1]=data[0]+pitch*vertical_stride,
//     hw_frames_ctx set, AVFrame lifetime bound to the surface)
//   - nn::AscendImageToTensorNode (device input: DVPP consumes the decoder
//     buffer directly; host NV12 stage-one upload path kept for sources
//     without device export)
//   - media::DownloadAscendDeviceSurfaceToHost (on-demand D2H copy for
//     preview/snapshot/OSD consumers, off the inference path)
//
// Built standalone on the 310P3 host against the CANN toolkit and the custom
// Ascend FFmpeg:
//   source /usr/local/Ascend/ascend-toolkit/set_env.sh
//   g++ -std=c++17 -O2 -DCOSMO_NN_USE_ASCEND_BACKEND \
//       -I src -I 3rd/fmt-7.1.2/include \
//       -I"${ASCEND_TOOLKIT_HOME}/include" \
//       test/ascend310p3/ascend_device_frame_smoke.cc \
//       src/nn/device/ascend/ascend_image_to_tensor_node.cc \
//       src/nn/core/status.cc src/nn/core/blob.cc src/nn/core/blob_impl.cc \
//       src/nn/core/abstract_device.cc src/nn/core/abstract_context.cc \
//       src/nn/core/shared_resource.cc src/nn/core/blob_store.cc \
//       src/nn/node/node.cc src/nn/utils/op.cc src/nn/utils/string_format.cc \
//       src/nn/utils/timer.cc src/nn/utils/dims_vector_utils.cc \
//       src/nn/utils/blob_memory_size_info.cc src/nn/utils/blob_memory_size_utils.cc \
//       src/nn/utils/data_type_utils.cc src/nn/device/naive/naive_device.cc \
//       src/nn/device/naive/naive_context.cc \
//       src/media/VideoDecoder.cc src/media/VideoDecoderCreateAscend.cc \
//       src/media/VideoDecoderAscend.cc src/media/VideoDecoderAscendHostCopy.cc \
//       src/media/VideoFrame.cc src/media/PixelFormatUtils.cc \
//       src/mem/MemoryPoolMng.cc src/mem/FixedBlockPool.cc \
//       src/mem/AllocatorCpu.cc src/mem/BlockFreqCalc.cc \
//       src/util/Thread.cc src/util/ThreadRegistry.cc src/util/TimeUtil.cc \
//       3rd/fmt-7.1.2/src/format.cc \
//       -I/opt/ffmpeg-4.4.1/ascend/include \
//       -L/opt/ffmpeg-4.4.1/ascend/lib -lavformat -lavcodec -lavutil \
//       -Wl,-rpath,/opt/ffmpeg-4.4.1/ascend/lib \
//       -L"${ASCEND_TOOLKIT_HOME}/lib64" -lascendcl -lacl_dvpp -lacl_dvpp_mpi \
//       -Wl,-rpath,"${ASCEND_TOOLKIT_HOME}/lib64" -lpthread -ldl -lm \
//       -o ascend_device_frame_smoke
//
// Usage: ascend_device_frame_smoke --video <h264|h265 file>
//                                  [--frames N] [--target WxH] [--padding R,G,B]
//                                  [--host-baseline N]
//
// Prints per-frame image_to_tensor timings for the device (direct DVPP) and
// host (stage-one upload) paths, then a summary; returns nonzero on any
// contract violation.

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
}

#include "media/FrameSurface.h"
#include "media/PixelFormat.h"
#include "media/VideoDecoder.h"
#include "media/VideoDecoderAscend.h"
#include "media/VideoFrame.h"
#include "mem/AllocatorCpu.h"
#include "mem/MemoryPoolMng.h"
#include "nn/core/blob.h"
#include "nn/device/ascend/ascend_image_to_tensor_node.h"
#include "nn/utils/data_type_utils.h"
#include "nn/utils/op.h"

// Standalone smoke: provide the logging entry point that util/Log.h declares
// (the full engine links glog; this check only needs the fmt-format callback).
namespace cosmo::log {
size_t LogFormatArg(const char* file, const char* function, int line, int module, int severity,
                    fmt::string_view format, fmt::format_args args) {
    static_cast<void>(module);
    static_cast<void>(severity);
    const std::string text = fmt::vformat(format, args);
    std::fprintf(stderr, "%s:%d %s: %s\n", file, line, function, text.c_str());
    return text.size();
}
}  // namespace cosmo::log

// The full engine links libuuid; the standalone check does not, so provide
// the small entry point used by util/Thread.
namespace cosmo::util {
std::string GenerateUUID() {
    static std::atomic<uint64_t> counter{0};
    return "ascend-device-frame-smoke-" + std::to_string(counter.fetch_add(1));
}
}  // namespace cosmo::util

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void RequireStatus(cosmo::nn::Status status, const std::string& message) {
    if (!bool(status)) {
        throw std::runtime_error(message + ": " + status.description());
    }
}

std::string AvError(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return text.data();
}

// ── Demux, mirroring the engine demux lifecycle ──────────────────────────
class Demux {
public:
    ~Demux() {
        Close();
    }

    bool Open(const std::string& url) {
        Close();
        AVDictionary* options = nullptr;
        const int ret         = avformat_open_input(&format_, url.c_str(), nullptr, &options);
        av_dict_free(&options);
        if (ret < 0 || format_ == nullptr) {
            return false;
        }
        if (avformat_find_stream_info(format_, nullptr) < 0) {
            Close();
            return false;
        }
        stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index_ < 0) {
            Close();
            return false;
        }
        const AVCodecParameters* params = format_->streams[stream_index_]->codecpar;
        const char* bsf_name            = nullptr;
        if (params->codec_id == AV_CODEC_ID_H264) {
            bsf_name = "h264_mp4toannexb";
        } else if (params->codec_id == AV_CODEC_ID_HEVC) {
            bsf_name = "hevc_mp4toannexb";
        }
        if (bsf_name != nullptr) {
            const AVBitStreamFilter* filter = av_bsf_get_by_name(bsf_name);
            if (filter == nullptr || av_bsf_alloc(filter, &bsf_ctx_) < 0 ||
                avcodec_parameters_copy(bsf_ctx_->par_in, params) < 0 || av_bsf_init(bsf_ctx_) < 0) {
                Close();
                return false;
            }
        }
        return true;
    }

    bool Next(AVPacket* packet) {
        if (format_ == nullptr) {
            return false;
        }
        while (true) {
            const int ret = av_read_frame(format_, packet);
            if (ret < 0) {
                return false;
            }
            if (packet->stream_index != stream_index_) {
                av_packet_unref(packet);
                continue;
            }
            if (bsf_ctx_ != nullptr) {
                const int send_ret = av_bsf_send_packet(bsf_ctx_, packet);
                av_packet_unref(packet);
                if (send_ret < 0) {
                    return false;
                }
                while (true) {
                    const int recv_ret = av_bsf_receive_packet(bsf_ctx_, packet);
                    if (recv_ret == AVERROR(EAGAIN) || recv_ret == AVERROR_EOF) {
                        break;
                    }
                    if (recv_ret < 0) {
                        return false;
                    }
                    return true;
                }
                continue;
            }
            return true;
        }
    }

    AVCodecID CodecId() const {
        return format_->streams[stream_index_]->codecpar->codec_id;
    }

    int Width() const {
        return format_->streams[stream_index_]->codecpar->width;
    }

    int Height() const {
        return format_->streams[stream_index_]->codecpar->height;
    }

private:
    void Close() {
        if (bsf_ctx_ != nullptr) {
            av_bsf_free(&bsf_ctx_);
            bsf_ctx_ = nullptr;
        }
        if (format_ != nullptr) {
            avformat_close_input(&format_);
            format_ = nullptr;
        }
        stream_index_ = -1;
    }

    AVFormatContext* format_ = nullptr;
    AVBSFContext* bsf_ctx_   = nullptr;
    int stream_index_        = -1;
};

// ── Issue #32 device-surface contract, per the recorded test-host ABI ────
void CheckDeviceSurface(const cosmo::media::VideoFramePtr& frame, int width, int height,
                        uint64_t packets_sent) {
    Require(frame != nullptr && frame->Active(), "decoder returned an inactive frame");
    Require(frame->GetPixelFormat() == cosmo::media::PixelFormat::PIXEL_NV12,
            "decoded frame format is not NV12");
    Require(frame->GetFrameIndex() < packets_sent, "decoded frame index does not map to a sent packet");

    const auto surface = frame->GetSurface();
    Require(surface != nullptr && surface->IsValid(), "decoded frame surface is invalid");
    Require(surface->memory_type == cosmo::media::FrameSurfaceMemoryType::Device,
            "decoded surface is not Ascend device memory");
    Require(surface->lifetime != nullptr, "decoded device surface has no AVFrame lifetime holder");
    Require(surface->planes.size() == 2, "device surface must expose two planes");
    const auto& luma   = surface->planes[0];
    const auto& chroma = surface->planes[1];
    Require(luma.virt_addr != nullptr && chroma.virt_addr != nullptr,
            "device planes must carry device addresses");
    Require(luma.pitch >= static_cast<size_t>(width), "device Y pitch is shorter than the width");
    Require(luma.pitch % 16 == 0, "device Y pitch is not DVPP 16-byte aligned");
    Require(chroma.pitch == luma.pitch, "device planes must share one pitch");
    Require(luma.vertical_stride >= static_cast<size_t>(height),
            "device Y vertical stride is shorter than the height");
    Require(chroma.vertical_stride >= static_cast<size_t>(height) / 2,
            "device UV vertical stride is shorter than the height");
    // The single AVBufferRef holds both planes; UV starts pitch-aligned after
    // the Y rows (alignment-1 pool layout of the custom FFmpeg). GetPlaneData
    // resolves virt_addr + offset, so it must land on the real device plane
    // bases (this also guards against double-counting the plane offset).
    const auto* y_ptr  = surface->GetPlaneData(0);
    const auto* uv_ptr = surface->GetPlaneData(1);
    Require(y_ptr != nullptr && uv_ptr != nullptr && uv_ptr > y_ptr,
            "device plane bases must resolve through GetPlaneData");
    const size_t uv_offset = static_cast<size_t>(uv_ptr - y_ptr);
    Require(uv_offset == luma.pitch * luma.vertical_stride,
            "device UV offset does not match pitch*vertical_stride");
    Require(luma.size >= luma.pitch * (luma.vertical_stride + chroma.vertical_stride),
            "device backing buffer is smaller than the Y+UV span");
    // No intermediate host image is constructed by the inference path.
    Require(frame->GetHostData() == nullptr, "inference frame must not carry host data");
}

std::shared_ptr<cosmo::nn::Blob> MakeSurfaceBlob(const cosmo::media::FrameSurface& surface, int width,
                                                 int height) {
    cosmo::nn::BlobDesc desc;
    desc.device_type = cosmo::nn::DEVICE_NAIVE;
    desc.data_type   = cosmo::nn::DATA_TYPE_FLOAT;
    desc.data_format = cosmo::nn::DATA_FORMAT_NHWC;
    desc.dims        = {1, height, width, 3};
    cosmo::nn::BlobHandle handle;
    handle.base = const_cast<cosmo::media::FrameSurface*>(&surface);
    return std::make_shared<cosmo::nn::Blob>(desc, handle);
}

// Wraps a host NV12 buffer (from DownloadAscendDeviceSurfaceToHost) in a Host
// FrameSurface, mirroring BuildAscendNv12Surface's plane layout.
cosmo::media::FrameSurface MakeHostSurface(const std::vector<uint8_t>& host_nv12, size_t pitch, int width,
                                           int height, std::shared_ptr<void> lifetime) {
    cosmo::media::FrameSurface surface;
    surface.memory_type     = cosmo::media::FrameSurfaceMemoryType::Host;
    surface.lifetime        = std::move(lifetime);
    const size_t luma_bytes = pitch * static_cast<size_t>(height);
    surface.planes.push_back(cosmo::media::FramePlane{-1, const_cast<uint8_t*>(host_nv12.data()), 0, pitch,
                                                      static_cast<size_t>(height), host_nv12.size()});
    surface.planes.push_back(cosmo::media::FramePlane{-1, const_cast<uint8_t*>(host_nv12.data() + luma_bytes),
                                                      luma_bytes, pitch, static_cast<size_t>(height) / 2,
                                                      host_nv12.size()});
    return surface;
}

// Validates the FP16 NCHW letterbox tensor: corners carry the padding color
// and the image area is not all padding.
void CheckTensor(const std::shared_ptr<cosmo::nn::Blob>& top, int frame_width, int frame_height,
                 int target_width, int target_height, int padding) {
    const auto desc = top->GetBlobDesc();
    Require(desc.data_type == cosmo::nn::DATA_TYPE_HALF, "image_to_tensor output is not FP16");
    Require(desc.dims.size() == 4 && desc.dims[1] == 3 &&
                desc.dims[2] == static_cast<size_t>(target_height) &&
                desc.dims[3] == static_cast<size_t>(target_width),
            "image_to_tensor output shape is not NCHW 1x3xHxW");
    const auto* data = static_cast<const uint16_t*>(top->GetHandle().base);
    Require(data != nullptr, "image_to_tensor output blob is not bound");

    int fit_width  = 0;
    int fit_height = 0;
    int pad_x      = 0;
    int pad_y      = 0;
    Require(cosmo::nn::AscendImageToTensorNode::ComputeLetterbox(
                frame_width, frame_height, target_width, target_height, fit_width, fit_height, pad_x, pad_y),
            "letterbox geometry failed");
    const uint16_t pad_fp16 = cosmo::nn::FloatToFp16(static_cast<float>(padding) / 255.0F);
    // All four corners are padding (letterbox is centered).
    for (int corner = 0; corner < 4; ++corner) {
        const int x = corner & 1 ? target_width - 1 : 0;
        const int y = corner & 2 ? target_height - 1 : 0;
        for (int c = 0; c < 3; ++c) {
            const uint16_t value = data[(static_cast<size_t>(c) * target_height + y) * target_width + x];
            Require(value == pad_fp16, "letterbox corner is not the padding color");
        }
    }
    // The resized image area must carry real content, not padding gray: count
    // pixels that differ from the padding color across the three channels.
    uint64_t content_pixels   = 0;
    const uint64_t image_area = static_cast<uint64_t>(fit_width) * fit_height;
    for (int y = pad_y; y < pad_y + fit_height; ++y) {
        for (int x = pad_x; x < pad_x + fit_width; ++x) {
            bool all_padding = true;
            for (int c = 0; c < 3; ++c) {
                if (data[(static_cast<size_t>(c) * target_height + y) * target_width + x] != pad_fp16) {
                    all_padding = false;
                    break;
                }
            }
            if (!all_padding) {
                ++content_pixels;
            }
        }
    }
    Require(content_pixels > image_area / 2, "resized image area is mostly padding gray (empty scene?)");
}

}  // namespace

struct Options {
    std::string video_path;
    int frames        = 30;
    int target_width  = 960;
    int target_height = 960;
    int padding       = 114;
    int host_baseline = 5;
};

Options ParseOptions(int argc, char** argv) {
    Options options;
    auto value = [&](int index) {
        Require(index + 1 < argc, std::string("missing value for ") + argv[index]);
        return std::string(argv[index + 1]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--video")
            options.video_path = value(i++);
        else if (arg == "--frames")
            options.frames = std::stoi(value(i++));
        else if (arg == "--target") {
            const std::string target = value(i++);
            const size_t x           = target.find('x');
            Require(x != std::string::npos, "--target must be WxH");
            options.target_width  = std::stoi(target.substr(0, x));
            options.target_height = std::stoi(target.substr(x + 1));
        } else if (arg == "--padding")
            options.padding = std::stoi(value(i++));
        else if (arg == "--host-baseline")
            options.host_baseline = std::stoi(value(i++));
        else
            throw std::runtime_error("unknown option: " + arg);
    }
    Require(!options.video_path.empty(), "--video is required");
    Require(options.frames > 0, "--frames must be positive");
    return options;
}

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);

        if (std::getenv("SMOKE_SKIP_POOL") == nullptr) {
            cosmo::mem::MemoryPoolMng memory_pool(
                std::make_unique<cosmo::mem::AllocatorCpu>(),
                {640 * 640 * 3 / 2, 1280 * 720 * 3 / 2, 1920 * 1088 * 3 / 2, 3840 * 2160 * 3 / 2});
            cosmo::mem::SetMemoryPoolContext(&memory_pool);
        }
        Require(avcodec_find_decoder_by_name("h264_ascend") != nullptr,
                "h264_ascend decoder is unavailable on this FFmpeg");

        Demux demux;
        Require(demux.Open(options.video_path), "demux open failed: " + options.video_path);
        const cosmo::media::VideoCodecType codec_type = demux.CodecId() == AV_CODEC_ID_HEVC
                                                            ? cosmo::media::VideoCodecType::kH265
                                                            : cosmo::media::VideoCodecType::kH264;
        auto decoder                                  = cosmo::media::VideoDecoder::Create(0, nullptr);
        decoder->SetCodecType(codec_type, demux.Width(), demux.Height());
        Require(decoder->Open(), "Ascend decoder open failed (no software fallback): " + options.video_path);
        std::printf("video=%s %dx%d codec=%s\n", options.video_path.c_str(), demux.Width(), demux.Height(),
                    codec_type == cosmo::media::VideoCodecType::kH265 ? "h265_ascend" : "h264_ascend");

        cosmo::nn::AscendImageToTensorNode node;
        cosmo::nn::ImageToTensor op("image_to_tensor");
        op.input_width   = options.target_width;
        op.input_height  = options.target_height;
        op.padding_color = {options.padding, options.padding, options.padding};
        node.LoadParam(&op);
        RequireStatus(node.InferTopShapes(), "InferTopShapes failed");
        Require(node.GetTopBlobDeviceType() == cosmo::nn::DEVICE_NAIVE,
                "image_to_tensor output must be host memory");

        AVPacket* packet = av_packet_alloc();
        Require(packet != nullptr, "av_packet_alloc failed");

        using clock                 = std::chrono::steady_clock;
        int64_t packets_sent        = 0;
        int device_frames           = 0;
        int host_frames             = 0;
        double device_total_us      = 0.0;
        double host_total_us        = 0.0;
        bool lifetime_guard_checked = false;

        auto run_device = [&](const cosmo::media::VideoFramePtr& frame) {
            const auto surface = frame->GetSurface();
            const auto t0      = clock::now();
            auto bottom        = MakeSurfaceBlob(*surface, demux.Width(), demux.Height());
            cosmo::nn::BlobDesc top_desc;
            top_desc.device_type = cosmo::nn::DEVICE_NAIVE;
            top_desc.data_type   = cosmo::nn::DATA_TYPE_HALF;
            top_desc.dims        = {1, 3, options.target_height, options.target_width};
            auto top             = std::make_shared<cosmo::nn::Blob>(top_desc, true);
            std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
            std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{top};
            RequireStatus(node.Forward(bottoms, tops), "device-path image_to_tensor Forward failed");
            const auto t1 = clock::now();
            CheckTensor(top, demux.Width(), demux.Height(), options.target_width, options.target_height,
                        options.padding);
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            device_total_us += us;
            ++device_frames;
            std::printf("frame=%llu device direct dvpp total=%.1fus\n",
                        static_cast<unsigned long long>(frame->GetFrameIndex()), us);

            // Invalid/expired device surfaces must fail clearly: a device
            // surface without a lifetime holder is rejected before DVPP.
            if (!lifetime_guard_checked) {
                lifetime_guard_checked                 = true;
                cosmo::media::FrameSurface no_lifetime = *surface;
                no_lifetime.lifetime                   = nullptr;
                auto bad_bottom = MakeSurfaceBlob(no_lifetime, demux.Width(), demux.Height());
                std::vector<std::shared_ptr<cosmo::nn::Blob>> bad_bottoms{bad_bottom};
                std::vector<std::shared_ptr<cosmo::nn::Blob>> bad_tops{top};
                cosmo::nn::Status bad_status = node.Forward(bad_bottoms, bad_tops);
                Require(!bool(bad_status), "device surface without lifetime must be rejected");
                std::printf("invalid surface rejected: %s\n", bad_status.description().c_str());

                cosmo::media::FrameSurface bad_geometry = *surface;
                bad_geometry.planes[0].pitch            = 100;  // not 16-byte aligned
                auto geometry_bottom = MakeSurfaceBlob(bad_geometry, demux.Width(), demux.Height());
                std::vector<std::shared_ptr<cosmo::nn::Blob>> geometry_bottoms{geometry_bottom};
                cosmo::nn::Status geometry_status = node.Forward(geometry_bottoms, bad_tops);
                Require(!bool(geometry_status), "invalid device surface geometry must be rejected");
                std::printf("invalid geometry rejected: %s\n", geometry_status.description().c_str());
            }
        };

        auto run_host_baseline = [&](const cosmo::media::VideoFramePtr& frame) {
            // Stage-one fallback: download for a host consumer, wrap it in a
            // Host surface, and prove image_to_tensor still uploads it.
            const auto surface = frame->GetSurface();
            std::vector<uint8_t> host_nv12;
            std::string copy_error;
            Require(cosmo::media::DownloadAscendDeviceSurfaceToHost(*surface, demux.Width(), demux.Height(),
                                                                    host_nv12, copy_error),
                    "device->host copy failed: " + copy_error);
            auto host_surface = MakeHostSurface(host_nv12, surface->planes[0].pitch, demux.Width(),
                                                demux.Height(), surface->lifetime);
            Require(host_surface.IsValid(), "host baseline surface is invalid");
            const auto t0 = clock::now();
            auto bottom   = MakeSurfaceBlob(host_surface, demux.Width(), demux.Height());
            cosmo::nn::BlobDesc top_desc;
            top_desc.device_type = cosmo::nn::DEVICE_NAIVE;
            top_desc.data_type   = cosmo::nn::DATA_TYPE_HALF;
            top_desc.dims        = {1, 3, options.target_height, options.target_width};
            auto top             = std::make_shared<cosmo::nn::Blob>(top_desc, true);
            std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
            std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{top};
            RequireStatus(node.Forward(bottoms, tops), "host stage-one image_to_tensor Forward failed");
            const auto t1 = clock::now();
            CheckTensor(top, demux.Width(), demux.Height(), options.target_width, options.target_height,
                        options.padding);
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            host_total_us += us;
            ++host_frames;
            std::printf("frame=%llu host stage-one total=%.1fus\n",
                        static_cast<unsigned long long>(frame->GetFrameIndex()), us);
        };

        std::vector<cosmo::media::VideoFramePtr> frames;
        const auto run_start = clock::now();
        while (frames.size() < static_cast<size_t>(options.frames) && demux.Next(packet)) {
            bool ok = false;
            const auto decoded =
                decoder->Decode(packet->data, static_cast<size_t>(packet->size), packets_sent, ok);
            ++packets_sent;
            if (decoded) {
                CheckDeviceSurface(decoded, demux.Width(), demux.Height(),
                                   static_cast<uint64_t>(packets_sent));
                frames.push_back(decoded);
            }
            av_packet_unref(packet);

            while (frames.size() < static_cast<size_t>(options.frames)) {
                auto extra = decoder->GetFrame();
                if (!extra) {
                    break;
                }
                CheckDeviceSurface(extra, demux.Width(), demux.Height(), static_cast<uint64_t>(packets_sent));
                frames.push_back(extra);
            }
        }
        Require(decoder->Flush(), "Ascend decoder flush failed");
        const auto flush_deadline = clock::now() + std::chrono::seconds(10);
        int empty_rounds          = 0;
        while (frames.size() < static_cast<size_t>(options.frames) && clock::now() < flush_deadline &&
               empty_rounds < 200) {
            auto extra = decoder->GetFrame();
            if (!extra) {
                ++empty_rounds;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            empty_rounds = 0;
            CheckDeviceSurface(extra, demux.Width(), demux.Height(), static_cast<uint64_t>(packets_sent));
            frames.push_back(extra);
        }
        av_packet_free(&packet);
        Require(frames.size() == static_cast<size_t>(options.frames),
                "source ended before " + std::to_string(options.frames) + " frames (received " +
                    std::to_string(frames.size()) + ")");
        const auto decode_done = clock::now();

        for (size_t i = 0; i < frames.size(); ++i) {
            run_device(frames[i]);
            if (options.host_baseline > 0 && static_cast<int>(i) < options.host_baseline) {
                run_host_baseline(frames[i]);
            }
        }

        const auto run_done    = clock::now();
        const double decode_ms = std::chrono::duration<double, std::milli>(decode_done - run_start).count();
        const double e2e_ms    = std::chrono::duration<double, std::milli>(run_done - run_start).count();
        std::printf(
            "summary: frames=%d device_direct_avg=%.1fus host_stage_one_avg=%.1fus "
            "decode=%d frames in %.0fms e2e=%.0fms (%.1f fps incl. decode)\n",
            device_frames, device_frames ? device_total_us / device_frames : 0.0,
            host_frames ? host_total_us / host_frames : 0.0, options.frames, decode_ms, e2e_ms,
            static_cast<double>(options.frames) / (e2e_ms / 1000.0));
        std::printf("smoke OK\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ascend device frame smoke failed: %s\n", error.what());
        return 1;
    }
}
