// Host-level runnable check for issue #31: decode a local H.264/H.265 file
// with the Ascend hardware decoder, then run each decoded NV12 frame through
// the real engine graph (AscendImageToTensorNode DVPP preprocessing ->
// AscendNetNode AscendCL OM -> yolo_e2e host postprocess) on a real 310P3.
//
// The smoke drives the repository pieces where the full engine build is not
// available on the 310P3 host yet:
//   - media::VideoDecoderAscend (h264_ascend/h265_ascend, device
//     AV_PIX_FMT_ASCEND surfaces; issue #32 direct-DVPP path)
//   - nn::AscendImageToTensorNode (DVPP direct device input or host upload,
//     letterbox, NV12->RGB888, host /255 + FP16 NCHW normalization; logs
//     input mode and upload/dvpp/download stages)
//   - nn::AscendNetNode (aclInit once, per-graph context/stream/model/
//     datasets/buffers, H2D -> aclmdlExecuteAsync -> sync -> D2H + FP16->FP32)
//   - nn::YoloE2EDecodeNode (host postprocess, letterbox coordinate recovery)
// No ONNX Runtime or CPU resize exists in this path; an unsupported codec or
// a DVPP/ACL failure fails the task.
//
// Built standalone on the 310P3 host against the CANN toolkit and the custom
// Ascend FFmpeg:
//   export PATH=/root/.cargo/bin:$PATH
//   source /usr/local/Ascend/ascend-toolkit/set_env.sh
//   ln -sfn /root/cosmo-edge-issue31/fmt-7.1.2 /root/cosmo-edge-issue31/3rd/fmt-7.1.2
//   g++ -std=c++17 -O2 -DCOSMO_NN_USE_ASCEND_BACKEND \
//       -I src -I 3rd/fmt-7.1.2/include \
//       -I"${ASCEND_TOOLKIT_HOME}/include" \
//       test/ascend310p3/ascend_video_detect_smoke.cc \
//       src/nn/device/ascend/ascend_net_node.cc \
//       src/nn/device/ascend/ascend_node_creator.cc \
//       src/nn/device/ascend/ascend_image_to_tensor_node.cc \
//       src/nn/core/status.cc src/nn/core/blob.cc src/nn/core/blob_impl.cc \
//       src/nn/core/abstract_device.cc src/nn/core/abstract_context.cc \
//       src/nn/core/shared_resource.cc src/nn/core/blob_store.cc src/nn/core/graph.cc \
//       src/nn/node/node.cc src/nn/node/net_node.cc src/nn/node/node_type_utils.cc \
//       src/nn/node/node_creator.cc \
//       src/nn/node/input_node.cc src/nn/node/identity_node.cc \
//       src/nn/node/yolo_e2e_decode_node.cc \
//       src/nn/utils/op.cc src/nn/utils/string_format.cc src/nn/utils/timer.cc \
//       src/nn/utils/dims_vector_utils.cc src/nn/utils/blob_memory_size_info.cc \
//       src/nn/utils/blob_memory_size_utils.cc src/nn/utils/data_type_utils.cc \
//       src/nn/device/naive/naive_device.cc src/nn/device/naive/naive_context.cc \
//       src/media/VideoDecoder.cc src/media/VideoDecoderCreateAscend.cc \
//       src/media/VideoDecoderAscend.cc src/media/VideoFrame.cc \
//       src/media/PixelFormatUtils.cc \
//       src/mem/MemoryPoolMng.cc src/mem/FixedBlockPool.cc \
//       src/mem/AllocatorCpu.cc src/mem/BlockFreqCalc.cc \
//       src/util/Thread.cc src/util/ThreadRegistry.cc src/util/TimeUtil.cc \
//       3rd/fmt-7.1.2/src/format.cc \
//       -I/opt/ffmpeg-4.4.1/ascend/include \
//       -L/opt/ffmpeg-4.4.1/ascend/lib -lavformat -lavcodec -lavutil \
//       -Wl,-rpath,/opt/ffmpeg-4.4.1/ascend/lib \
//       -L"${ASCEND_TOOLKIT_HOME}/lib64" -lascendcl -lacl_dvpp -lacl_dvpp_mpi \
//       -Wl,-rpath,"${ASCEND_TOOLKIT_HOME}/lib64" -lpthread -ldl -lm \
//       -o ascend_video_detect_smoke
//
// Usage:
//   ascend_video_detect_smoke --om <model.om> --video <h264|h265 file>
//                              [--frames N] [--conf 0.25] [--topk 300]
//                              [--expect-detections]
//                              [--dump-frame N --dump-file path]
//
// The smoke prints per-frame stage timings (decode, image_to_tensor, acl,
// postprocess, end-to-end) and the parsed YOLO26 detections. --dump-frame N
// additionally writes the image_to_tensor FP16 NCHW output for that frame to
// --dump-file so the DVPP letterbox/color contract can be diffed against a
// reference tensor (e.g. /tmp/input_sa164_0to1.f16 from issue #29).

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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

#include "acl/acl.h"
#include "fmt/format.h"
#include "media/FrameSurface.h"
#include "media/PixelFormat.h"
#include "media/VideoDecoder.h"
#include "media/VideoFrame.h"
#include "mem/AllocatorCpu.h"
#include "mem/MemoryPoolMng.h"
#include "nn/core/blob.h"
#include "nn/core/graph.h"
#include "nn/core/status.h"
#include "nn/device/ascend/ascend_acl.h"
#include "nn/device/ascend/ascend_image_to_tensor_node.h"
#include "nn/device/ascend/ascend_net_node.h"
#include "nn/node/identity_node.h"
#include "nn/node/input_node.h"
#include "nn/node/node_creator.h"
#include "nn/node/yolo_e2e_decode_node.h"
#include "nn/utils/dims_vector_utils.h"
#include "nn/utils/net_utils.h"
#include "nn/utils/op.h"

namespace {

bool require_detections_ = false;

void Require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void RequireStatus(cosmo::nn::Status status, const std::string& message) {
    if (!bool(status))
        throw std::runtime_error(message + ": " + status.description());
}

std::string AvError(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return text.data();
}

// ── Demux (file or RTSP), mirroring the engine demux lifecycle ──────────
class Demux {
public:
    ~Demux() {
        Close();
    }

    bool Open(const std::string& url) {
        Close();
        AVDictionary* options = nullptr;
        if (url.rfind("rtsp://", 0) == 0) {
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
        // MP4 containers store H.264/H.265 in AVCC format, so the demuxer
        // applies the mp4toannexb bitstream filter before packets reach the
        // Ascend hardware decoder. RTSP already delivers Annex-B.
        if (url.rfind("rtsp://", 0) != 0) {
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

// ── OM contract probe (shared with the issue #29 smoke) ────────────────
struct OmContract {
    std::vector<int> input_dims;
    size_t input_bytes = 0;
};

OmContract ProbeOm(const std::string& om_path) {
    constexpr int kDeviceId = 0;
    Require(cosmo::nn::ascend::EnsureAclInitialized() == ACL_SUCCESS, "aclInit failed");
    Require(aclrtSetDevice(kDeviceId) == ACL_SUCCESS, "aclrtSetDevice failed");
    struct DeviceGuard {
        ~DeviceGuard() {
            (void)aclrtResetDevice(kDeviceId);
        }
    } device_guard;

    uint32_t model_id = 0;
    Require(aclmdlLoadFromFile(om_path.c_str(), &model_id) == ACL_SUCCESS,
            "aclmdlLoadFromFile failed: " + om_path);
    struct ModelGuard {
        uint32_t model_id;
        ~ModelGuard() {
            (void)aclmdlUnload(model_id);
        }
    } model_guard{model_id};

    aclmdlDesc* desc = aclmdlCreateDesc();
    Require(desc != nullptr, "aclmdlCreateDesc failed");
    struct DescGuard {
        aclmdlDesc* desc;
        ~DescGuard() {
            (void)aclmdlDestroyDesc(desc);
        }
    } desc_guard{desc};
    Require(aclmdlGetDesc(desc, model_id) == ACL_SUCCESS, "aclmdlGetDesc failed");

    Require(aclmdlGetNumInputs(desc) == 1, "probe expects exactly one input");
    aclmdlIODims io_dims{};
    Require(aclmdlGetInputDims(desc, 0, &io_dims) == ACL_SUCCESS, "aclmdlGetInputDims failed");
    OmContract contract;
    contract.input_bytes = aclmdlGetInputSizeByIndex(desc, 0);
    Require(contract.input_bytes > 0, "model input size is zero");
    for (size_t i = 0; i < io_dims.dimCount; ++i)
        contract.input_dims.push_back(static_cast<int>(io_dims.dims[i]));
    return contract;
}

// ── Per-frame node timings via the graph profiler ──────────────────────
struct StageTimes {
    double image_to_tensor_ms = 0.0;
    double acl_ms             = 0.0;
    double postprocess_ms     = 0.0;
};

class SmokeProfiler : public cosmo::nn::IProfiler {
public:
    void ReportGraphInfo(const char* msg) override {
        graph_info = msg == nullptr ? std::string() : std::string(msg);
    }

    void ReportNodeTime(const char* name, double time) override {
        const std::string node(name);
        if (node.find("image_to_tensor") != std::string::npos)
            times.image_to_tensor_ms = time;
        else if (node == "net_0")
            times.acl_ms = time;
        else if (node.find("yolo_e2e") != std::string::npos)
            times.postprocess_ms = time;
    }

    StageTimes times;
    std::string graph_info;
};

std::shared_ptr<cosmo::nn::Blob> MakeSurfaceBlob(const cosmo::media::VideoFramePtr& frame) {
    const auto surface = frame->GetSurface();
    Require(surface != nullptr && surface->IsValid(), "decoded frame surface is invalid");
    // Issue #32: the Ascend decoder hands out device (AV_PIX_FMT_ASCEND)
    // surfaces by default; the host NV12 stage-one surface is still accepted
    // for sources without device export.
    Require(surface->memory_type == cosmo::media::FrameSurfaceMemoryType::Device ||
                surface->memory_type == cosmo::media::FrameSurfaceMemoryType::Host,
            "decoded surface is neither device nor host memory");
    Require(surface->planes.size() == 2, "NV12 surface must expose two planes");

    cosmo::nn::BlobDesc desc;
    desc.data_format = cosmo::nn::DATA_FORMAT_NHWC;
    desc.data_type   = cosmo::nn::DATA_TYPE_UINT8;
    desc.dims        = {1, static_cast<int>(frame->GetHeight()), static_cast<int>(frame->GetWidth()), 3};
    desc.device_type = cosmo::nn::DEVICE_ASCEND;
    cosmo::nn::BlobHandle handle;
    handle.base      = surface.get();
    handle.ownership = cosmo::nn::BLOB_HANDLE_EXTERNAL_OWNED;
    return std::make_shared<cosmo::nn::Blob>(desc, handle);
}

void PrintDetections(const std::shared_ptr<cosmo::nn::Blob>& blob, float conf_threshold) {
    const auto desc = blob->GetBlobDesc();
    const auto dims = desc.dims;
    Require(dims.size() == 3 && dims[2] == 6, "unexpected output shape");
    const float* data = static_cast<const float*>(blob->GetHandle().base);
    int valid         = 0;
    for (int i = 0; i < dims[0] * dims[1]; ++i) {
        const float* row = data + i * 6;
        if (!(row[4] >= conf_threshold))
            continue;
        // YoloE2EDecodeNode emits (cx, cy, w, h, score, class).
        std::printf("  det[%d] cx=%.1f cy=%.1f w=%.1f h=%.1f score=%.4f class=%d\n", valid, row[0], row[1],
                    row[2], row[3], row[4], static_cast<int>(row[5]));
        ++valid;
    }
    std::printf("  parsed_detections=%d (conf>=%.2f)\n", valid, conf_threshold);
}

}  // namespace

// Standalone harness: net_utils.cc (Eigen/tokenizers-dependent) is not
// linked, so provide the one Graph-referenced entry point. The smoke graph
// declares the image_to_tensor preprocess op, which the stub keeps intact.
namespace cosmo::nn {
Status NetUtils::OptimizePreOps(std::vector<std::unique_ptr<Op>>& /*ops*/, bool /*use_skip*/) {
    return COSMO_NN_OK;
}

// Minimal host node creator for the standalone harness: the smoke graph only
// needs the pass-through input, identity, and yolo_e2e decode fallback nodes
// (the engine HostNodeCreator would drag in every host decode node). The
// Ascend node creator is the real one (linked from ascend_node_creator.cc).
class SmokeHostNodeCreator : public NodeCreator {
public:
    explicit SmokeHostNodeCreator(DeviceType device_type) : NodeCreator(device_type) {}
    ~SmokeHostNodeCreator() override = default;

    std::unique_ptr<Node> CreateNode(NodeType type) override {
        switch (type) {
            case NODE_INPUT:
                return std::make_unique<InputNode>();
            case NODE_IDENTITY:
                return std::make_unique<IdentityNode>();
            case NODE_YOLO_E2E_DECODE:
                return std::make_unique<YoloE2EDecodeNode>();
            default:
                return nullptr;
        }
    }
};

NodeCreatorRegister<SmokeHostNodeCreator> g_smoke_host_node_creator_register(DEVICE_NAIVE);
}  // namespace cosmo::nn

// Standalone smoke: provide the log entry point util/Log.h declares (the full
// engine links glog; this check only needs the fmt-format callback). The
// AscendImageToTensorNode stage timings surface through this callback.
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
    return "ascend-video-detect-smoke-" + std::to_string(counter.fetch_add(1));
}
}  // namespace cosmo::util

int main(int argc, char** argv) {
    std::string om_path;
    std::string video_path;
    float conf_threshold = 0.25F;
    int top_k            = 300;
    int frames_needed    = 30;
    int dump_frame       = -1;
    std::string dump_file;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&](const char* flag) {
            Require(i + 1 < argc, std::string("missing value for ") + flag);
            return std::string(argv[++i]);
        };
        if (arg == "--om")
            om_path = value("--om");
        else if (arg == "--video")
            video_path = value("--video");
        else if (arg == "--conf")
            conf_threshold = std::stof(value("--conf"));
        else if (arg == "--topk")
            top_k = std::stoi(value("--topk"));
        else if (arg == "--frames")
            frames_needed = std::stoi(value("--frames"));
        else if (arg == "--dump-frame")
            dump_frame = std::stoi(value("--dump-frame"));
        else if (arg == "--dump-file")
            dump_file = value("--dump-file");
        else if (arg == "--expect-detections")
            require_detections_ = true;
        else
            throw std::runtime_error("unknown argument: " + arg);
    }
    Require(!om_path.empty(), "--om <model.om> is required");
    Require(!video_path.empty(), "--video <h264|h265 file> is required");
    Require(frames_needed >= 1, "--frames must be >= 1");
    Require(top_k >= 1, "--topk must be >= 1");
    Require((dump_frame < 0) == dump_file.empty(), "--dump-frame and --dump-file must be used together");

    if (std::getenv("SMOKE_SKIP_POOL") == nullptr) {
        cosmo::mem::MemoryPoolMng memory_pool(
            std::make_unique<cosmo::mem::AllocatorCpu>(),
            {640 * 640 * 3 / 2, 1280 * 720 * 3 / 2, 1920 * 1088 * 3 / 2, 3840 * 2160 * 3 / 2});
        cosmo::mem::SetMemoryPoolContext(&memory_pool);
    }
    Require(avcodec_find_decoder_by_name("h264_ascend") != nullptr,
            "h264_ascend decoder is unavailable on this FFmpeg");
    Require(avcodec_find_decoder_by_name("h265_ascend") != nullptr,
            "h265_ascend decoder is unavailable on this FFmpeg");
    std::cout << "gate0 decoders present: h264_ascend h265_ascend\n";

    using namespace cosmo::nn;
    using clock = std::chrono::steady_clock;

    const OmContract contract = ProbeOm(om_path);
    Require(contract.input_dims.size() == 4, "model input must be 4D NCHW");
    const int input_width  = contract.input_dims[3];
    const int input_height = contract.input_dims[2];
    std::printf("om=%s input_dims=[%d,%d,%d,%d] input_bytes=%zu\n", om_path.c_str(), contract.input_dims[0],
                contract.input_dims[1], contract.input_dims[2], contract.input_dims[3], contract.input_bytes);

    std::unique_ptr<AscendImageToTensorNode> dump_node;
    if (dump_frame >= 0) {
        // Standalone DVPP preprocessing probe: produces the same FP16 NCHW
        // tensor the graph feeds to AscendNetNode, without the OM.
        dump_node = std::make_unique<AscendImageToTensorNode>();
        ImageToTensor dump_op("image_to_tensor");
        dump_op.input_width   = input_width;
        dump_op.input_height  = input_height;
        dump_op.padding_color = {114, 114, 114};
        dump_node->LoadParam(&dump_op);
        RequireStatus(dump_node->InferTopShapes(), "dump node InferTopShapes failed");
    }

    CombinedModelInfo info;
    info.algorithmcode = "yolo26_det_smoke";
    info.type          = "yolo26_det";
    ModelInfo model;
    model.name      = "yolo26_det";
    model.filename  = "model.om";
    model.max_batch = 1;
    InputNodeInfo input;
    input.name                = "images";
    input.shape               = contract.input_dims;
    input.data_type           = 1;  // fp16 (InputNodeInfo encoding, see DataTypeFromInputInfo)
    auto preprocess           = std::make_unique<ImageToTensor>("image_to_tensor");
    preprocess->input_width   = input_width;
    preprocess->input_height  = input_height;
    preprocess->padding_color = {114, 114, 114};
    input.ops.push_back(std::move(preprocess));
    model.input_node_infos.push_back(std::move(input));
    OutputNodeInfo output;
    output.name              = "output0";
    output.shape             = {1, top_k, 6};
    output.data_type         = 1;
    auto post                = std::make_unique<YoloPost>("yolo_e2e_postprocess");
    post->nms_detection_conf = conf_threshold;
    post->top_k              = top_k;
    post->input_width        = input_width;
    post->input_height       = input_height;
    output.op                = std::move(post);
    model.output_node_infos.push_back(std::move(output));
    info.models.push_back(std::move(model));

    std::printf("graph_init device=DEVICE_ASCEND...\n");
    Graph graph;
    SmokeProfiler profiler;
    graph.SetProfiler(&profiler);
    RequireStatus(graph.Init(info, om_path, DEVICE_ASCEND, "", /*device_id=*/0), "Graph::Init failed");

    // Prove the hardware stages are wired in: DVPP image_to_tensor and
    // AscendCL net nodes, with no CPU preprocess node in between.
    const std::string& graph_dump = profiler.graph_info;
    Require(graph_dump.find("image_to_tensor_0") != std::string::npos,
            "graph is missing the Ascend DVPP image_to_tensor node");
    Require(graph_dump.find("net_0") != std::string::npos, "graph is missing the AscendCL net node");
    Require(graph_dump.find("yolo_e2e_decode_0") != std::string::npos,
            "graph is missing the yolo_e2e host postprocess node");
    std::printf(
        "graph nodes: image_to_tensor_0(ascend-dvpp) -> net_0(ascendcl) -> yolo_e2e_decode_0(host)\n");

    Demux demux;
    Require(demux.Open(video_path), "demux open failed: " + video_path);
    const cosmo::media::VideoCodecType codec_type = demux.CodecId() == AV_CODEC_ID_HEVC
                                                        ? cosmo::media::VideoCodecType::kH265
                                                        : cosmo::media::VideoCodecType::kH264;
    auto decoder                                  = cosmo::media::VideoDecoder::Create(0, nullptr);
    decoder->SetCodecType(codec_type, demux.Width(), demux.Height());
    Require(decoder->Open(), "Ascend decoder open failed (no software fallback): " + video_path);
    std::printf("video=%s %dx%d codec=%s\n", video_path.c_str(), demux.Width(), demux.Height(),
                codec_type == cosmo::media::VideoCodecType::kH265 ? "h265_ascend" : "h264_ascend");

    AVPacket* packet = av_packet_alloc();
    Require(packet != nullptr, "av_packet_alloc failed");
    int64_t packets_sent = 0;
    int detected_frames  = 0;
    size_t decoded_count = 0;
    const auto run_start = clock::now();

    auto run_detection = [&](const cosmo::media::VideoFramePtr& frame) {
        const int frame_index       = static_cast<int>(frame->GetFrameIndex());
        static auto prev_frame_time = clock::now();
        const auto now              = clock::now();
        std::printf("frame=%d wall_delta=%.2fms\n", frame_index,
                    std::chrono::duration<double, std::milli>(now - prev_frame_time).count());
        prev_frame_time = now;
        if (dump_node && frame_index == dump_frame) {
            BlobDesc dump_desc;
            dump_desc.device_type = DEVICE_NAIVE;
            dump_desc.data_type   = DATA_TYPE_HALF;
            dump_desc.dims        = {1, 3, input_height, input_width};
            auto dump_top         = std::make_shared<Blob>(dump_desc, true);
            std::vector<std::shared_ptr<Blob>> dump_bottoms{MakeSurfaceBlob(frame)};
            std::vector<std::shared_ptr<Blob>> dump_tops{dump_top};
            RequireStatus(dump_node->Forward(dump_bottoms, dump_tops), "dump node Forward failed");
            std::ofstream out(dump_file, std::ios::binary);
            Require(bool(out), "cannot open dump file " + dump_file);
            out.write(static_cast<const char*>(dump_top->GetHandle().base),
                      static_cast<std::streamsize>(static_cast<size_t>(input_width) * input_height * 3 *
                                                   sizeof(uint16_t)));
            std::printf("dumped frame %d image_to_tensor tensor to %s\n", frame_index, dump_file.c_str());
        }
        std::vector<std::vector<std::shared_ptr<Blob>>> params;
        params.push_back({MakeSurfaceBlob(frame)});
        const auto graph_start = clock::now();
        RequireStatus(graph.Forward(params), "Graph::Forward failed");
        const auto graph_end = clock::now();

        auto outputs = graph.Output();
        Require(outputs.size() == 1, "expected one graph output blob");
        std::printf(
            "frame=%d e2e_graph=%.2fms image_to_tensor=%.2fms acl=%.2fms "
            "postprocess=%.2fms\n",
            frame_index, std::chrono::duration<double, std::milli>(graph_end - graph_start).count(),
            profiler.times.image_to_tensor_ms, profiler.times.acl_ms, profiler.times.postprocess_ms);
        PrintDetections(outputs[0], conf_threshold);
        if (outputs[0]->GetBlobDesc().dims[1] > 0) {
            const float* data = static_cast<const float*>(outputs[0]->GetHandle().base);
            int valid         = 0;
            for (int i = 0; i < outputs[0]->GetBlobDesc().dims[1]; ++i) {
                if (data[i * 6 + 4] >= conf_threshold)
                    ++valid;
            }
            if (valid > 0)
                ++detected_frames;
        }
        std::fflush(stdout);  // keep frame lines in the log when the run is killed
    };

    // Consume the async Ascend decoder exactly like the engine: send a
    // packet, run detection on whatever frame is ready, then drain. The
    // decoder's output surfaces are a fixed pool, so holding every frame
    // until the stream ends (as a decode-all-then-run loop would) exhausts
    // the pool and the decoder blocks waiting for a freed surface.
    auto run_decoded = [&](const cosmo::media::VideoFramePtr& frame) {
        if (!frame || decoded_count >= static_cast<size_t>(frames_needed))
            return;
        ++decoded_count;
        run_detection(frame);
    };
    while (decoded_count < static_cast<size_t>(frames_needed) && demux.Next(packet)) {
        bool ok = false;
        const auto decoded =
            decoder->Decode(packet->data, static_cast<size_t>(packet->size), packets_sent, ok);
        ++packets_sent;
        run_decoded(decoded);
        av_packet_unref(packet);
        while (decoded_count < static_cast<size_t>(frames_needed)) {
            auto extra = decoder->GetFrame();
            if (!extra)
                break;
            run_decoded(extra);
        }
    }
    av_packet_free(&packet);

    // End-of-stream: flush the Ascend HiMpi pipeline so frames still queued
    // in the hardware decoder are delivered, then drain with a short poll.
    if (decoded_count < static_cast<size_t>(frames_needed)) {
        Require(decoder->Flush(), "Ascend decoder flush failed: " + video_path);
        const auto flush_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        int empty_rounds          = 0;
        while (decoded_count < static_cast<size_t>(frames_needed) &&
               std::chrono::steady_clock::now() < flush_deadline && empty_rounds < 200) {
            auto extra = decoder->GetFrame();
            if (!extra) {
                ++empty_rounds;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            empty_rounds = 0;
            run_decoded(extra);
        }
    }
    Require(decoded_count > 0, "no frames decoded from " + video_path);

    const auto run_end = clock::now();
    std::printf(
        "e2e=%zu frames in %.2fs (%.1f fps incl. decode)\n", decoded_count,
        std::chrono::duration<double>(run_end - run_start).count(),
        static_cast<double>(decoded_count) / std::chrono::duration<double>(run_end - run_start).count());
    std::printf("detected_frames=%d/%zu (conf>=%.2f)\n", detected_frames, decoded_count, conf_threshold);
    if (require_detections_ && detected_frames == 0)
        throw std::runtime_error("expected at least one detection, got zero");
    std::printf("smoke OK\n");
    return 0;
}
