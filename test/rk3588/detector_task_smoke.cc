// Board-level runnable check for issue #11: deliver the first complete RK3588
// detector task from H.264/H.265 file or RTSP packets through hardware decode
// and zero-copy inference to detection consumers.
//
// The smoke drives the real engine classes where the full engine build is not
// available on the board (the board lacks the Rust toolchain required by the
// pre-existing tokenizers-cpp third-party build):
//   - media::VideoDecoderRk3588 (MPP hardware decode to DRM PRIME)
//   - RknnNetNode / RknnImageToTensorNode / Yolo26RawDecodeNode (zero-copy)
//   - InstancePool (RK DetectorPool policy: 1 instance/task, max 3)
//   - VideoFrameProcRk3588 (on-demand DMA -> host I420 conversion)
//
// Built standalone on the RK3588 board:
//   g++ -std=c++17 -O2 -I src -I 3rd/fmt-7.1.2/include -I 3rd/include \
//     test/rk3588/detector_task_smoke.cc \
//     3rd/fmt-7.1.2/src/format.cc \
//     src/media/VideoDecoder.cc \
//     src/media/VideoDecoderCreateRk3588.cc \
//     src/media/VideoDecoderRk3588.cc src/media/VideoFrame.cc \
//     src/media/VideoFrameProcCpu.cc src/media/VideoFrameProcRk3588.cc \
//     src/media/PixelFormatUtils.cc src/media/EncodedImageInfo.cc \
//     src/mem/MemoryPoolMng.cc src/mem/FixedBlockPool.cc \
//     src/mem/AllocatorCpu.cc src/mem/BlockFreqCalc.cc \
//     src/util/Thread.cc src/util/ThreadRegistry.cc \
//     src/util/TimeUtil.cc \
//     src/nn/device/rknn/rknn_net_node.cc \
//     src/nn/device/rknn/rknn_image_to_tensor_node.cc \
//     src/nn/device/rknn/rknn_node_creator.cc \
//     src/nn/core/status.cc src/nn/core/blob.cc src/nn/core/blob_impl.cc \
//     src/nn/core/abstract_device.cc src/nn/core/abstract_context.cc \
//     src/nn/core/shared_resource.cc \
//     src/nn/utils/op.cc src/nn/utils/string_format.cc src/nn/utils/timer.cc \
//     src/nn/utils/dims_vector_utils.cc src/nn/utils/blob_memory_size_info.cc \
//     src/nn/utils/blob_memory_size_utils.cc src/nn/utils/data_type_utils.cc \
//     src/nn/node/node.cc src/nn/node/net_node.cc src/nn/node/node_type_utils.cc \
//     src/nn/node/node_creator.cc src/nn/node/yolo26_raw_decode_node.cc \
//     src/nn/device/naive/naive_device.cc src/nn/device/naive/naive_context.cc \
//     -I <rknn-sdk>/include -I <rknn-sdk>/include/rga -I /usr/include/libdrm \
//     -lrknnrt -lrga -ldrm $(pkg-config --cflags --libs libavcodec libavformat libavutil libswscale libdrm) \
//     -lpthread -o detector_task_smoke

#include <signal.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext_drm.h>
}

#include <drm_fourcc.h>
#include <rknn_api.h>

#include "media/FrameSurface.h"
#include "media/IOsdTextRenderer.h"
#include "media/PixelFormat.h"
#include "media/VideoDecoderRk3588.h"
#include "media/VideoFrame.h"
#include "media/VideoFrameProcRk3588.h"
#include "mem/AllocatorCpu.h"
#include "mem/MemoryPoolMng.h"
#include "nn/core/blob.h"
#include "nn/device/rknn/rknn_image_to_tensor_node.h"
#include "nn/device/rknn/rknn_net_node.h"
#include "nn/node/yolo26_raw_decode_node.h"
#include "nn/utils/op.h"
#include "util/InstancePool.h"

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
    return "detector-smoke-" + std::to_string(counter.fetch_add(1));
}
}  // namespace cosmo::util

namespace {

using AVFramePtr = std::shared_ptr<AVFrame>;

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

std::string ReadFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    Require(file.good(), "could not open file: " + path);
    const std::streamsize size = file.tellg();
    Require(size > 0, "file is empty: " + path);
    file.seekg(0);
    std::string data(static_cast<size_t>(size), '\0');
    Require(static_cast<bool>(file.read(data.data(), size)), "could not read file: " + path);
    return data;
}

bool IsRtspUrl(const std::string& url) {
    return url.rfind("rtsp://", 0) == 0;
}

// ── Demux: file or RTSP, with re-open support for stream recovery ────────
class Demux {
public:
    ~Demux() {
        Close();
    }

    bool Open(const std::string& url) {
        Close();
        AVDictionary* options = nullptr;
        if (IsRtspUrl(url)) {
            av_dict_set(&options, "rtsp_transport", "tcp", 0);
            av_dict_set(&options, "stimeout", "5000000", 0);
            av_dict_set(&options, "max_delay", "200000", 0);
            av_dict_set(&options, "fflags", "nobuffer", 0);
        }
        const int ret = avformat_open_input(&format_, url.c_str(), nullptr, &options);
        av_dict_free(&options);
        if (ret < 0 || format_ == nullptr) {
            if (format_ != nullptr) {
                avformat_close_input(&format_);
                format_ = nullptr;
            }
            std::cerr << "demux open failed: " << AvError(ret) << "\n";
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
        // Mirror the engine demux lifecycle: MP4 containers store H.264/H.265
        // in AVCC format, so the demuxer applies the mp4toannexb bitstream
        // filter before packets reach the MPP hardware decoder. RTSP already
        // delivers Annex-B and needs no filter.
        if (!IsRtspUrl(url)) {
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
        }
        return true;
    }

    // Returns true with a packet for the video stream; false on EOF/error
    // (the caller decides whether to retry with a re-open).
    bool Next(AVPacket* packet) {
        if (format_ == nullptr) {
            return false;
        }
        while (true) {
            const int ret = av_read_frame(format_, packet);
            if (ret < 0) {
                std::cerr << "demux read failed: " << AvError(ret) << "\n";
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
                    std::cerr << "demux bsf send failed: " << AvError(send_ret) << "\n";
                    return false;
                }
                while (true) {
                    const int recv_ret = av_bsf_receive_packet(bsf_ctx_, packet);
                    if (recv_ret == AVERROR(EAGAIN) || recv_ret == AVERROR_EOF) {
                        break;
                    }
                    if (recv_ret < 0) {
                        std::cerr << "demux bsf receive failed: " << AvError(recv_ret) << "\n";
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

// ── Decode + zero-copy inference pipeline (engine classes) ───────────────
class DetectorTask {
public:
    DetectorTask(const std::string& model_path, int image_size, float confidence, float iou) {
        const std::string model_data = ReadFile(model_path);
        net_.SetModelPath(model_path);
        RequireStatus(net_.LoadWeight(model_data.data(), model_data.size()),
                      "RknnNetNode::LoadWeight failed");
        RequireStatus(net_.InferTopShapes(), "RknnNetNode::InferTopShapes failed");

        auto op           = std::make_unique<cosmo::nn::ImageToTensor>("image_to_tensor");
        op->input_width   = image_size;
        op->input_height  = image_size;
        op->padding_color = {114, 114, 114};
        pre_.LoadParam(op.get());
        RequireStatus(pre_.InferTopShapes(), "RknnImageToTensorNode::InferTopShapes failed");

        cosmo::nn::BlobDesc shared_desc;
        shared_desc.device_type = cosmo::nn::DEVICE_RKNN;
        shared_desc.data_format = cosmo::nn::DATA_FORMAT_NHWC;
        shared_desc.data_type   = cosmo::nn::DATA_TYPE_UINT8;
        shared_desc.dims        = {1, image_size, image_size, 3};
        shared_blob_            = std::make_shared<cosmo::nn::Blob>(shared_desc);
        std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{shared_blob_};
        RequireStatus(net_.BindInputBlobs(bottoms), "RknnNetNode::BindInputBlobs failed");

        const auto& shapes = net_.GetTopBlobShapes();
        Require(shapes.size() == 6, "RKNN model must expose six outputs");
        for (size_t i = 0; i < shapes.size(); ++i) {
            cosmo::nn::BlobDesc desc;
            desc.device_type = cosmo::nn::DEVICE_NAIVE;
            desc.data_type   = cosmo::nn::DATA_TYPE_INT8;
            desc.data_format = cosmo::nn::DATA_FORMAT_NCHW;
            desc.dims        = shapes[i];
            net_.UpdateTopBlobDesc(i, desc);
            top_blobs_.push_back(std::make_shared<cosmo::nn::Blob>(desc, true));
        }

        cosmo::nn::Yolo26RawPost post_op("yolo26_raw_postprocess");
        post_op.input_width        = image_size;
        post_op.input_height       = image_size;
        post_op.nms_detection_conf = confidence;
        post_op.nms_threshold      = iou;
        post_op.top_k              = 300;
        post_op.reg_max            = 1;
        for (size_t i = 0; i < top_blobs_.size(); ++i) {
            const auto& desc = top_blobs_[i]->GetBlobDesc();
            Require(desc.is_affine_quantized && desc.affine_scale > 0.0F,
                    "RKNN output quant metadata was not propagated");
            post_op.output_scales.push_back(desc.affine_scale);
            post_op.output_zero_points.push_back(desc.affine_zero_point);
        }
        decode_.LoadParam(&post_op);
        RequireStatus(decode_.InferTopShapes(), "Yolo26RawDecodeNode::InferTopShapes failed");

        cosmo::nn::BlobDesc det_desc;
        det_desc.device_type = cosmo::nn::DEVICE_NAIVE;
        det_desc.data_type   = cosmo::nn::DATA_TYPE_FLOAT;
        det_desc.dims        = {1, 300, 6};
        det_blob_            = std::make_shared<cosmo::nn::Blob>(det_desc, true);
    }

    struct Detection {
        float x1, y1, x2, y2, score;
        int class_id;
    };

    std::vector<Detection> Run(const cosmo::media::FrameSurface& surface, int width, int height) {
        cosmo::nn::BlobDesc input_desc;
        input_desc.device_type = cosmo::nn::DEVICE_RKNN;
        input_desc.data_format = cosmo::nn::DATA_FORMAT_NHWC;
        input_desc.data_type   = cosmo::nn::DATA_TYPE_UINT8;
        input_desc.dims        = {1, height, width, 3};
        cosmo::nn::BlobHandle input_handle;
        input_handle.base      = const_cast<cosmo::media::FrameSurface*>(&surface);
        input_handle.ownership = cosmo::nn::BLOB_HANDLE_EXTERNAL_OWNED;
        auto input_blob        = std::make_shared<cosmo::nn::Blob>(input_desc, input_handle);

        std::vector<std::shared_ptr<cosmo::nn::Blob>> input_blobs{input_blob};
        std::vector<std::shared_ptr<cosmo::nn::Blob>> pre_tops{shared_blob_};
        RequireStatus(pre_.Forward(input_blobs, pre_tops), "RknnImageToTensorNode::Forward failed");
        const auto& letterbox = pre_.GetLastLetterbox();

        std::vector<std::shared_ptr<cosmo::nn::Blob>> net_bottoms{shared_blob_};
        RequireStatus(net_.Forward(net_bottoms, top_blobs_), "RknnNetNode::Forward failed");

        std::vector<std::shared_ptr<cosmo::nn::Blob>> det_tops{det_blob_};
        RequireStatus(decode_.Forward(top_blobs_, det_tops), "Yolo26RawDecodeNode::Forward failed");

        std::vector<Detection> detections;
        const auto* data = static_cast<const float*>(det_blob_->GetHandle().base);
        const int rows   = det_blob_->GetBlobDesc().dims[1];
        for (int row = 0; row < rows; ++row) {
            const float score = data[row * 6 + 4];
            if (score == 0.0F) {
                break;  // unused rows are reset to zeros by the decode node
            }
            const float cx = data[row * 6 + 0];
            const float cy = data[row * 6 + 1];
            const float w  = data[row * 6 + 2];
            const float h  = data[row * 6 + 3];
            const float x1 =
                std::clamp((cx - w * 0.5F - static_cast<float>(letterbox.pad_x)) / letterbox.scale, 0.0F,
                           static_cast<float>(width - 1));
            const float y1 =
                std::clamp((cy - h * 0.5F - static_cast<float>(letterbox.pad_y)) / letterbox.scale, 0.0F,
                           static_cast<float>(height - 1));
            const float x2 =
                std::clamp((cx + w * 0.5F - static_cast<float>(letterbox.pad_x)) / letterbox.scale, 0.0F,
                           static_cast<float>(width - 1));
            const float y2 =
                std::clamp((cy + h * 0.5F - static_cast<float>(letterbox.pad_y)) / letterbox.scale, 0.0F,
                           static_cast<float>(height - 1));
            detections.push_back({x1, y1, x2, y2, score, static_cast<int>(data[row * 6 + 5])});
        }
        return detections;
    }

    const cosmo::nn::RknnImageToTensorNode::RgaStats& RgaStats() const {
        return pre_.GetRgaStats();
    }

    int TensorFd() const {
        auto* binding = static_cast<cosmo::nn::RknnTensorBinding*>(shared_blob_->GetHandle().base);
        Require(binding != nullptr && binding->memory != nullptr, "RKNN input tensor is not bound");
        return binding->memory->fd;
    }

    cosmo::nn::RknnImageToTensorNode& Preprocess() {
        return pre_;
    }

    std::shared_ptr<cosmo::nn::Blob> SharedInputBlob() {
        return shared_blob_;
    }

private:
    cosmo::nn::RknnNetNode net_;
    cosmo::nn::RknnImageToTensorNode pre_;
    cosmo::nn::Yolo26RawDecodeNode decode_;
    std::shared_ptr<cosmo::nn::Blob> shared_blob_;
    std::vector<std::shared_ptr<cosmo::nn::Blob>> top_blobs_;
    std::shared_ptr<cosmo::nn::Blob> det_blob_;
};

struct RunResult {
    std::vector<std::vector<DetectorTask::Detection>> frames;
    int last_source_fd = -1;
};

// Decodes `target_frames` frames from `url` (file or RTSP) through the engine
// hardware decoder and zero-copy inference. Returns false when the stream
// ended/errored before enough frames were produced (callers use this for the
// RTSP interruption check).
bool RunSource(DetectorTask& task, const std::string& url, int target_frames,
               std::vector<std::vector<DetectorTask::Detection>>& frames, int& last_source_fd) {
    Demux demux;
    Require(demux.Open(url), "cannot open source: " + url);
    std::cout << "demux opened: " << url << " codec=" << demux.CodecId() << " " << demux.Width() << "x"
              << demux.Height() << "\n";

    cosmo::media::VideoCodecType codec_type = cosmo::media::VideoCodecType::kMjpeg;
    if (demux.CodecId() == AV_CODEC_ID_H264) {
        codec_type = cosmo::media::VideoCodecType::kH264;
    } else if (demux.CodecId() == AV_CODEC_ID_HEVC) {
        codec_type = cosmo::media::VideoCodecType::kH265;
    } else {
        throw std::runtime_error("unsupported codec; use H.264 or H.265");
    }
    auto decoder = cosmo::media::VideoDecoder::Create(0, nullptr);
    Require(decoder != nullptr, "engine decoder factory returned null");
    decoder->SetCodecType(codec_type, demux.Width(), demux.Height());
    Require(decoder->Open(), "engine hardware decoder failed to open (no silent fallback)");

    AVPacket* packet = av_packet_alloc();
    Require(packet != nullptr, "av_packet_alloc failed");
    struct PacketGuard {
        AVPacket* value;
        ~PacketGuard() {
            av_packet_free(&value);
        }
    } packet_guard{packet};

    auto drain = [&]() {
        while (static_cast<int>(frames.size()) < target_frames) {
            auto video_frame = decoder->GetFrame();
            if (!video_frame) {
                break;  // VPU buffering; feed more packets.
            }
            const auto* surface = video_frame->GetSurface().get();
            Require(
                surface != nullptr && surface->memory_type == cosmo::media::FrameSurfaceMemoryType::DmaBuf,
                "engine decoder did not produce a DMA-BUF surface");
            last_source_fd        = surface->planes[0].fd;
            const auto detections = task.Run(*surface, static_cast<int>(video_frame->GetWidth()),
                                             static_cast<int>(video_frame->GetHeight()));
            frames.push_back(detections);
            std::cout << "decoded frame=" << frames.size() << " w=" << video_frame->GetWidth()
                      << " h=" << video_frame->GetHeight() << " detections=" << detections.size() << "\n";
        }
    };

    int64_t packet_index = 0;
    while (static_cast<int>(frames.size()) < target_frames) {
        if (!demux.Next(packet)) {
            av_packet_unref(packet);
            // Flush the hardware decoder so trailing buffered frames are
            // emitted (the engine rebuilds the decoder on stream end; the
            // smoke needs the flush for a deterministic frame count).
            decoder->SendPacket(nullptr, 0, packet_index);
            drain();
            if (static_cast<int>(frames.size()) >= target_frames) {
                return true;
            }
            return false;  // EOF or stream error: caller may reconnect.
        }
        bool sent = false;
        for (int attempt = 0; attempt < 2 && !sent; ++attempt) {
            sent = decoder->SendPacket(packet->data, static_cast<size_t>(packet->size), packet_index);
            if (!sent) {
                // MPP buffering: drain decoded frames before retrying the
                // same packet (mirrors the engine's decoder-reset recovery).
                drain();
            }
        }
        av_packet_unref(packet);
        ++packet_index;
        if (!sent) {
            std::cerr << "engine decoder SendPacket failed at packet " << packet_index << "\n";
            return false;
        }
        drain();
    }
    return true;
}

// ── Reference comparison (issue #11 acceptance: class match, |score|<=1e-3,
//    matched-box IoU >= 0.99) ────────────────────────────────────────────
float IoU(const DetectorTask::Detection& lhs, const DetectorTask::Detection& rhs) {
    const float left   = std::max(lhs.x1, rhs.x1);
    const float top    = std::max(lhs.y1, rhs.y1);
    const float right  = std::min(lhs.x2, rhs.x2);
    const float bottom = std::min(lhs.y2, rhs.y2);
    const float inter  = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
    const float area_a = std::max(0.0F, lhs.x2 - lhs.x1) * std::max(0.0F, lhs.y2 - lhs.y1);
    const float area_b = std::max(0.0F, rhs.x2 - rhs.x1) * std::max(0.0F, rhs.y2 - rhs.y1);
    return inter / (area_a + area_b - inter + 1e-7F);
}

void CompareWithReference(const std::string& path,
                          const std::vector<std::vector<DetectorTask::Detection>>& frames,
                          float iou_threshold) {
    std::ifstream file(path);
    Require(file.good(), "could not open reference JSON: " + path);
    std::vector<std::vector<DetectorTask::Detection>> reference;
    for (std::string line; std::getline(file, line);) {
        if (line.empty()) {
            continue;
        }
        std::vector<DetectorTask::Detection> detections;
        size_t pos = 0;
        while ((pos = line.find("{\"class_id\":", pos)) != std::string::npos) {
            const size_t class_start = pos + std::string("{\"class_id\":").size();
            const size_t class_end   = line.find(',', class_start);
            const int class_id       = std::stoi(line.substr(class_start, class_end - class_start));
            const size_t score_pos   = line.find("\"score\":", class_end);
            const size_t score_start = score_pos + 8;
            const size_t score_end   = line.find(',', score_start);
            const float score        = std::stof(line.substr(score_start, score_end - score_start));
            const size_t xyxy_start  = line.find("[", score_end);
            const size_t xyxy_end    = line.find("]", xyxy_start);
            std::istringstream box(line.substr(xyxy_start + 1, xyxy_end - xyxy_start - 1));
            float x1, y1, x2, y2;
            char comma;
            box >> x1 >> comma >> y1 >> comma >> x2 >> comma >> y2;
            detections.push_back({x1, y1, x2, y2, score, class_id});
            pos = xyxy_end;
        }
        reference.push_back(std::move(detections));
    }
    Require(reference.size() >= frames.size(), "reference JSON has fewer frames than the smoke run");
    for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
        const auto& mine = frames[frame_index];
        const auto& ref  = reference[frame_index];
        float min_iou    = 1.0F;
        for (const auto& detection : ref) {
            bool matched   = false;
            float best_iou = 0.0F;
            for (const auto& candidate : mine) {
                if (candidate.class_id != detection.class_id) {
                    continue;
                }
                if (std::fabs(candidate.score - detection.score) > 1e-3F) {
                    continue;
                }
                best_iou = std::max(best_iou, IoU(candidate, detection));
                if (IoU(candidate, detection) >= iou_threshold) {
                    matched = true;
                    break;
                }
            }
            min_iou = std::min(min_iou, best_iou);
            if (!matched) {
                std::cerr << "unmatched reference detection; my candidates for frame " << frame_index
                          << ":\n";
                for (const auto& candidate : mine) {
                    std::cerr << "  class=" << candidate.class_id << " score=" << candidate.score << " box=["
                              << candidate.x1 << "," << candidate.y1 << "," << candidate.x2 << ","
                              << candidate.y2 << "] iou=" << IoU(candidate, detection) << "\n";
                }
            }
            Require(matched, "reference detection is missing at frame " + std::to_string(frame_index) +
                                 " class=" + std::to_string(detection.class_id) +
                                 " score=" + std::to_string(detection.score) + " ref_box=[" +
                                 std::to_string(detection.x1) + "," + std::to_string(detection.y1) + "," +
                                 std::to_string(detection.x2) + "," + std::to_string(detection.y2) +
                                 "] mine=" + std::to_string(mine.size()));
        }
        std::cout << "frame=" << frame_index << " matched ref=" << ref.size() << " mine=" << mine.size()
                  << " (extras=" << (mine.size() - ref.size()) << ") min_iou=" << min_iou << "\n";
    }
    std::cout << "reference comparison passed for " << frames.size() << " frames (iou>=" << iou_threshold
              << ")\n";
}

void DumpDetections(const std::string& path,
                    const std::vector<std::vector<DetectorTask::Detection>>& frames) {
    std::ofstream file(path, std::ios::trunc);
    Require(file.good(), "could not open dump path: " + path);
    for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
        file << "{\"frame\":" << frame_index << ",\"detections\":[";
        for (size_t i = 0; i < frames[frame_index].size(); ++i) {
            const auto& d = frames[frame_index][i];
            if (i) {
                file << ',';
            }
            file << "{\"class_id\":" << d.class_id << ",\"score\":" << std::fixed << std::setprecision(6)
                 << d.score << ",\"xyxy\":[" << d.x1 << ',' << d.y1 << ',' << d.x2 << ',' << d.y2 << "]}";
        }
        file << "]}\n";
    }
}

// ── RK DetectorPool policy check ─────────────────────────────────────────
class CountingDetector {
public:
    static int live;

    CountingDetector(const std::string&, const std::string&, const std::string&) {}
    ~CountingDetector() {
        --live;
    }

    CountingDetector(const CountingDetector&)            = delete;
    CountingDetector& operator=(const CountingDetector&) = delete;

    cosmo::util::ErrorEnum Init() {
        ++live;
        return cosmo::util::ErrorEnum::Success;
    }
};

int CountingDetector::live = 0;

void CheckDetectorPoolPolicy() {
    CountingDetector::live = 0;
    cosmo::InstancePool<CountingDetector, std::shared_ptr<CountingDetector>> pool("rk_detector", 1, 3);
    for (int task = 0; task < 4; ++task) {
        pool.CreateTask("alg", "cfg", "model");
    }
    Require(CountingDetector::live == 3, "RK DetectorPool must cap instances at three");

    auto first  = pool.GetInst("alg", "cfg", "model", 100);
    auto second = pool.GetInst("alg", "cfg", "model", 100);
    auto third  = pool.GetInst("alg", "cfg", "model", 100);
    auto fourth = pool.GetInst("alg", "cfg", "model", 100);
    Require(first && second && third, "three tasks must each hold one instance");
    Require(fourth == nullptr, "a fourth task must not create a fourth instance");
    pool.ReturnInst(first);
    pool.ReturnInst(second);
    pool.ReturnInst(third);
    std::cout << "detector pool policy passed: one instance per task, max three\n";
}

// ── On-demand host conversion check (preview/capture/OSD/record) ─────────
class StubOsdTextRenderer final : public cosmo::media::IOsdTextRenderer {
public:
    bool Init(const std::string&) override {
        return true;
    }
    bool IsReady() const override {
        return true;
    }
    cosmo::media::IOsdTextRenderer::TextBitmap RenderString(const std::string&, float) const override {
        return {};
    }
    cosmo::media::IOsdTextRenderer::OutlinedTextBitmap RenderStringWithOutline(const std::string&,
                                                                               float) const override {
        return {};
    }
};

void CheckHostConversion(const cosmo::media::FrameSurface& surface, int width, int height) {
    auto frame =
        std::make_shared<cosmo::media::VideoFrame>(width, height, cosmo::media::PixelFormat::PIXEL_NV12,
                                                   std::make_shared<cosmo::media::FrameSurface>(surface));
    Require(frame != nullptr && frame->Active(), "host check could not wrap the DMA surface");

    StubOsdTextRenderer osd;
    cosmo::media::VideoFrameProcRk3588 proc(osd);

    auto i420 = proc.NV12ToI420(frame);
    Require(i420 != nullptr && i420->Active(), "NV12ToI420 failed on a DMA surface");
    Require(i420->GetPixelFormat() == cosmo::media::PixelFormat::PIXEL_I420,
            "host conversion did not produce I420");
    Require(i420->GetData() != nullptr, "host conversion produced no host data");
    Require(
        i420->GetWidth() == static_cast<size_t>(width) && i420->GetHeight() == static_cast<size_t>(height),
        "host conversion changed the frame dimensions");

    auto jpeg = proc.EncodeJpeg(i420);
    Require(!jpeg.empty(), "capture JPEG encode failed after host conversion");
    std::cout << "capture JPEG bytes=" << jpeg.size() << "\n";

    auto copied = proc.CopyFrame(frame);
    Require(copied != nullptr && copied->Active() && copied->GetData() != nullptr,
            "CopyFrame did not materialize host memory");

    Require(proc.EnsureHostData(frame), "EnsureHostData failed on a DMA surface");
    Require(frame->GetHostData() != nullptr, "EnsureHostData left no host buffer");
    std::cout << "host conversion passed: NV12 DMA -> I420 host, JPEG, host buffer\n";
}

// ── Error-path checks (no silent fallback) ───────────────────────────────
void CheckErrorPaths() {
    // A decoder that cannot provide an RK codec must refuse to open instead of
    // falling back to software decoding.
    auto decoder = cosmo::media::VideoDecoder::Create(0, nullptr);
    decoder->SetCodecType(cosmo::media::VideoCodecType::kMjpeg, 1920, 1080);
    Require(!decoder->Open(), "RK decoder must refuse unsupported codecs (no software fallback)");

    std::cout << "error paths passed: unsupported codec rejected with no fallback\n";
}

// image_to_tensor must reject a non-DMA-BUF surface with a staged diagnostic
// error instead of silently falling back to a host conversion.
void CheckSurfaceRejection(DetectorTask& task) {
    cosmo::nn::BlobDesc desc;
    desc.device_type = cosmo::nn::DEVICE_RKNN;
    desc.data_format = cosmo::nn::DATA_FORMAT_NHWC;
    desc.data_type   = cosmo::nn::DATA_TYPE_UINT8;
    desc.dims        = {1, 640, 640, 3};

    std::array<uint8_t, 16> storage{};
    auto surface         = std::make_shared<cosmo::media::FrameSurface>();
    surface->memory_type = cosmo::media::FrameSurfaceMemoryType::Host;
    surface->planes.push_back({-1, storage.data(), 0, 4, 4, storage.size()});

    cosmo::nn::BlobHandle handle;
    handle.base      = surface.get();
    handle.ownership = cosmo::nn::BLOB_HANDLE_EXTERNAL_OWNED;
    auto blob        = std::make_shared<cosmo::nn::Blob>(desc, handle);

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{blob};
    // The top is the RKNN-bound input tensor; the surface check must fail
    // before any RGA work happens.
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{task.SharedInputBlob()};
    auto status = task.Preprocess().Forward(bottoms, tops);
    if (bool(status)) {
        throw std::runtime_error("image_to_tensor must reject a non-DMA-BUF surface");
    }
    std::cout << "error paths passed: RGA input surface rejected with staged error: " << status.description()
              << "\n";
}

// ── RTSP server management ───────────────────────────────────────────────
// MediaMTX serves RTSP while ffmpeg publishes the looped file. Stopping the
// publisher simulates an RTSP source outage; the stream recovers when the
// publisher restarts (the demuxer re-opens instead of falling back to
// software inference).
class RtspServer {
public:
    RtspServer(const std::string& media_file, int port) : media_file_(media_file), port_(port) {}

    ~RtspServer() {
        Stop();
        if (server_pid_ > 0) {
            kill(server_pid_, SIGKILL);
            int status = 0;
            waitpid(server_pid_, &status, 0);
            server_pid_ = -1;
        }
    }

    void Start() {
        Stop();
        std::string url = "rtsp://127.0.0.1:" + std::to_string(port_) + "/stream";

        const std::string config_path = "/tmp/mediamtx-issue11-" + std::to_string(port_) + ".yml";
        {
            std::ofstream config(config_path, std::ios::trunc);
            Require(config.good(), "could not write mediamtx config: " + config_path);
            config << "paths:\n  all_others:\n";
        }

        server_pid_ = fork();
        Require(server_pid_ >= 0, "fork failed");
        if (server_pid_ == 0) {
            setenv("MEDIAMTX_CONFIG_PATH", config_path.c_str(), 1);
            // The installed MediaMTX v1.19.3 takes the config as a positional
            // argument; MEDIAMTX_CONFIG_PATH is ignored by that build, so pass
            // the path explicitly to keep the smoke independent of CWD.
            execl("/home/YTHC/bin/mediamtx", "mediamtx", config_path.c_str(), nullptr);
            _exit(127);
        }
        push_pid_ = fork();
        Require(push_pid_ >= 0, "fork failed");
        if (push_pid_ == 0) {
            execlp("ffmpeg", "ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1",
                   "-i", media_file_.c_str(), "-c", "copy", "-rtsp_transport", "tcp", "-f", "rtsp",
                   url.c_str(), nullptr);
            _exit(127);
        }
        std::cout << "rtsp server started mediamtx_pid=" << server_pid_ << " push_pid=" << push_pid_
                  << " url=" << url << "\n";
    }

    void Stop() {
        // Publisher outage: subscribers observe the RTSP session ending.
        if (push_pid_ > 0) {
            kill(push_pid_, SIGKILL);
            int status = 0;
            waitpid(push_pid_, &status, 0);
            push_pid_ = -1;
        }
    }

    std::string Url() const {
        return "rtsp://127.0.0.1:" + std::to_string(port_) + "/stream";
    }

private:
    std::string media_file_;
    int port_;
    pid_t server_pid_ = -1;
    pid_t push_pid_   = -1;
};

bool WaitForReconnect(DetectorTask& task, const std::string& url, int frames_needed,
                      std::vector<std::vector<DetectorTask::Detection>>& frames, int& last_source_fd,
                      int timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        try {
            return RunSource(task, url, static_cast<int>(frames.size()) + frames_needed, frames,
                             last_source_fd);
        } catch (const std::exception& e) {
            std::cerr << "reconnect attempt failed: " << e.what() << "\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return false;
}

}  // namespace

struct Options {
    std::string model;
    std::string h264_file;
    std::string h265_file;
    std::string rtsp_file;
    std::string compare_ref;
    std::string dump_detections;
    int rtsp_port        = 8554;
    int frames           = 3;
    int sustained_secs   = 30;
    int image_size       = 640;
    float confidence     = 0.25F;
    float iou            = 0.45F;
    float iou_threshold  = 0.99F;
    bool interrupt_rtsp  = false;
    bool run_pool_check  = true;
    bool run_host_check  = true;
    bool run_error_check = true;
};

Options ParseOptions(int argc, char** argv) {
    Options options;
    auto value = [&](int& index) -> std::string {
        Require(index + 1 < argc, std::string("missing value for ") + argv[index]);
        return argv[++index];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--model")
            options.model = value(i);
        else if (argument == "--h264")
            options.h264_file = value(i);
        else if (argument == "--h265")
            options.h265_file = value(i);
        else if (argument == "--rtsp")
            options.rtsp_file = value(i);
        else if (argument == "--rtsp-port")
            options.rtsp_port = std::stoi(value(i));
        else if (argument == "--frames")
            options.frames = std::stoi(value(i));
        else if (argument == "--sustained-secs")
            options.sustained_secs = std::stoi(value(i));
        else if (argument == "--compare-ref")
            options.compare_ref = value(i);
        else if (argument == "--dump-detections")
            options.dump_detections = value(i);
        else if (argument == "--imgsz")
            options.image_size = std::stoi(value(i));
        else if (argument == "--conf")
            options.confidence = std::stof(value(i));
        else if (argument == "--iou")
            options.iou = std::stof(value(i));
        else if (argument == "--iou-threshold")
            options.iou_threshold = std::stof(value(i));
        else if (argument == "--interrupt-rtsp")
            options.interrupt_rtsp = true;
        else if (argument == "--skip-pool-check")
            options.run_pool_check = false;
        else if (argument == "--skip-host-check")
            options.run_host_check = false;
        else if (argument == "--skip-error-check")
            options.run_error_check = false;
        else
            throw std::runtime_error("unknown option: " + argument);
    }
    Require(!options.model.empty(), "--model is required");
    Require(!options.h264_file.empty() || !options.h265_file.empty() || !options.rtsp_file.empty(),
            "one of --h264/--h265/--rtsp is required");
    Require(options.image_size > 0 && options.image_size % 32 == 0, "--imgsz must be a multiple of 32");
    return options;
}

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        std::cout << "detector-task smoke model=" << options.model << "\n";

        avformat_network_init();

        // Host frames used by the on-demand conversion check allocate from the
        // engine memory pool; cover I420/NV12 for common RK resolutions.
        cosmo::mem::MemoryPoolMng memory_pool(
            std::make_unique<cosmo::mem::AllocatorCpu>(),
            {640 * 640 * 3 / 2, 1280 * 720 * 3 / 2, 1920 * 1088 * 3 / 2, 3840 * 2160 * 3 / 2});
        cosmo::mem::SetMemoryPoolContext(&memory_pool);

        DetectorTask task(options.model, options.image_size, options.confidence, options.iou);

        if (options.run_error_check) {
            CheckErrorPaths();
            CheckSurfaceRejection(task);
        }
        if (options.run_pool_check) {
            CheckDetectorPoolPolicy();
        }

        // Local H.264 and H.265 files.
        for (const std::string& source : {options.h264_file, options.h265_file}) {
            if (source.empty()) {
                continue;
            }
            std::vector<std::vector<DetectorTask::Detection>> frames;
            int last_source_fd = -1;
            Require(RunSource(task, source, options.frames, frames, last_source_fd),
                    "file source ended before " + std::to_string(options.frames) + " frames: " + source);
            Require(frames.size() == static_cast<size_t>(options.frames),
                    "expected " + std::to_string(options.frames) + " frames");
            Require(task.RgaStats().last_dst_fd == task.TensorFd(),
                    "RGA destination fd is not the RKNN tensor fd");
            Require(task.RgaStats().last_src_fd == last_source_fd,
                    "RGA source fd is not the decoder surface fd");
            Require(task.RgaStats().last_src_fd != task.RgaStats().last_dst_fd,
                    "RGA source and RKNN destination unexpectedly share an fd");
            int total = 0;
            for (const auto& detections : frames) {
                total += static_cast<int>(detections.size());
            }
            Require(total > 0, "no detections produced from " + source);
            std::cout << "file source passed: " << source << " frames=" << frames.size()
                      << " detections=" << total << " fd_provenance=" << task.RgaStats().last_src_fd << "->"
                      << task.RgaStats().last_dst_fd << "\n";

            if (options.run_host_check && !frames.empty()) {
                // Re-decode the first frame to get the DMA surface and validate
                // the on-demand host conversion path on real hardware.
                Demux demux;
                Require(demux.Open(source), "re-open failed for host check");
                auto decoder                            = cosmo::media::VideoDecoder::Create(0, nullptr);
                cosmo::media::VideoCodecType codec_type = demux.CodecId() == AV_CODEC_ID_HEVC
                                                              ? cosmo::media::VideoCodecType::kH265
                                                              : cosmo::media::VideoCodecType::kH264;
                decoder->SetCodecType(codec_type, demux.Width(), demux.Height());
                Require(decoder->Open(), "decoder re-open failed for host check");
                AVPacket* packet = av_packet_alloc();
                Require(packet != nullptr, "av_packet_alloc failed");
                bool got_frame = false;
                int64_t index  = 0;
                while (!got_frame && demux.Next(packet)) {
                    decoder->SendPacket(packet->data, static_cast<size_t>(packet->size), index++);
                    av_packet_unref(packet);
                    auto frame = decoder->GetFrame();
                    if (frame) {
                        const auto* surface = frame->GetSurface().get();
                        CheckHostConversion(*surface, static_cast<int>(frame->GetWidth()),
                                            static_cast<int>(frame->GetHeight()));
                        got_frame = true;
                    }
                }
                av_packet_free(&packet);
                Require(got_frame, "host check could not decode a first frame from " + source);
            }
        }

        // Accuracy gate: first N valid frames of the fixed H.264 video must
        // match the reference program.
        if (!options.compare_ref.empty()) {
            Require(!options.h264_file.empty(), "--compare-ref requires --h264");
            std::vector<std::vector<DetectorTask::Detection>> frames;
            int last_source_fd = -1;
            Require(RunSource(task, options.h264_file, options.frames, frames, last_source_fd),
                    "accuracy run ended early");
            if (!options.dump_detections.empty()) {
                DumpDetections(options.dump_detections, frames);
            }
            CompareWithReference(options.compare_ref, frames, options.iou_threshold);
        }

        // RTSP: sustained run plus optional interruption/reconnect.
        if (!options.rtsp_file.empty()) {
            RtspServer server(options.rtsp_file, options.rtsp_port);
            server.Start();

            std::vector<std::vector<DetectorTask::Detection>> frames;
            int last_source_fd    = -1;
            const std::string url = server.Url();
            std::cout << "rtsp phase 1: connect and decode\n";
            bool connected = false;
            for (int attempt = 0; attempt < 15 && !connected; ++attempt) {
                try {
                    connected = RunSource(task, url, options.frames, frames, last_source_fd);
                } catch (const std::exception& e) {
                    std::cerr << "rtsp connect attempt " << attempt << " failed: " << e.what() << "\n";
                }
                if (!connected) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
            Require(connected, "rtsp initial run failed");
            std::cout << "rtsp phase 1 passed: frames=" << frames.size() << "\n";

            if (options.interrupt_rtsp) {
                std::cout << "rtsp phase 2: interrupting stream\n";
                server.Stop();
                // Reading must fail while the server is down; the engine demux
                // lifecycle re-opens on failure instead of falling back to
                // software inference.
                {
                    Demux demux;
                    const bool open_failed = !demux.Open(url);
                    if (!open_failed) {
                        // The server may still be tearing down; a read must
                        // fail within the RTSP stimeout.
                        AVPacket* packet       = av_packet_alloc();
                        const bool read_failed = !demux.Next(packet);
                        av_packet_free(&packet);
                        Require(read_failed, "stream must fail while the server is stopped");
                    }
                }
                server.Start();
                std::cout << "rtsp phase 2: server restarted, reconnecting\n";
                const size_t before_reconnect = frames.size();
                Require(WaitForReconnect(task, url, options.frames, frames, last_source_fd, 30),
                        "rtsp did not recover after reconnect");
                Require(frames.size() > before_reconnect, "rtsp reconnect produced no new frames");
                std::cout << "rtsp phase 2 passed: recovered, frames=" << frames.size() << "\n";
            }

            if (options.sustained_secs > 0) {
                std::cout << "rtsp phase 3: sustained run for " << options.sustained_secs << "s\n";
                const auto deadline =
                    std::chrono::steady_clock::now() + std::chrono::seconds(options.sustained_secs);
                const size_t before_sustained = frames.size();
                Demux demux;
                Require(demux.Open(url), "rtsp sustained run could not reopen the stream");
                auto decoder = cosmo::media::VideoDecoder::Create(0, nullptr);
                decoder->SetCodecType(demux.CodecId() == AV_CODEC_ID_HEVC
                                          ? cosmo::media::VideoCodecType::kH265
                                          : cosmo::media::VideoCodecType::kH264,
                                      demux.Width(), demux.Height());
                Require(decoder->Open(), "rtsp sustained decoder open failed");
                AVPacket* packet        = av_packet_alloc();
                int64_t index           = 0;
                size_t sustained_frames = 0;
                while (std::chrono::steady_clock::now() < deadline) {
                    if (!demux.Next(packet)) {
                        std::cerr << "rtsp sustained demux error; re-opening\n";
                        av_packet_unref(packet);
                        Require(demux.Open(url), "rtsp sustained reconnect failed");
                        continue;
                    }
                    decoder->SendPacket(packet->data, static_cast<size_t>(packet->size), index++);
                    av_packet_unref(packet);
                    while (true) {
                        auto video_frame = decoder->GetFrame();
                        if (!video_frame) {
                            break;
                        }
                        const auto* surface   = video_frame->GetSurface().get();
                        const auto detections = task.Run(*surface, static_cast<int>(video_frame->GetWidth()),
                                                         static_cast<int>(video_frame->GetHeight()));
                        ++sustained_frames;
                        if (sustained_frames % 50 == 0) {
                            std::cout << "rtsp sustained decoded=" << sustained_frames
                                      << " detections=" << detections.size() << "\n";
                        }
                    }
                }
                av_packet_free(&packet);
                Require(sustained_frames > 0, "rtsp sustained run produced no frames");
                std::cout << "rtsp phase 3 passed: sustained_frames=" << sustained_frames
                          << " (total incl. earlier phases=" << frames.size() << ")\n";
            }
        }

        std::cout << "detector-task smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
