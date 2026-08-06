// Host-level runnable check for issue #33: deliver local-file and RTSP
// YOLO26 detection tasks on the Ascend 310P3 through the product task
// lifecycle pieces:
//   - media::VideoDecoder::Create (Ascend factory, h264_ascend/h265_ascend)
//   - nn::AscendImageToTensorNode (DVPP) -> nn::AscendNetNode (AscendCL) ->
//     YoloE2EDecodeNode (host postprocess), one graph per detector instance
//   - InstancePool DetectorPool policy (one instance per task, max three)
//   - media::VideoFrameProcCpu host-copy path (preview/capture/OSD consumers)
//   - RTSP interruption + reconnect through demux re-open (no software
//     inference fallback) and repeated task start/stop rounds
//
// The decoder follows the engine lifecycle: one Ascend DVPP VDEC channel per
// task, reused across demux restarts (task start/stop, RTSP reconnect). The
// 310P3 DVPP driver wedges a re-opened VDEC channel once a VPC channel has
// been created (see docs/development/ascend310p3-adaptation-plan.md), so the
// smoke never closes/reopens the decoder between rounds and feeds packets at
// a paced rate (the async VDEC delivers frames in near-real-time; an
// end-of-stream flush would permanently disable the channel for later
// streams).
//
// Built standalone on the 310P3 host against the CANN toolkit and the custom
// Ascend FFmpeg (same sources as test/ascend310p3/ascend_video_detect_smoke.cc):
//   export PATH=/root/.cargo/bin:$PATH
//   source /usr/local/Ascend/ascend-toolkit/set_env.sh
//   ln -sfn /root/cosmo-edge-issue33/fmt-7.1.2 /root/cosmo-edge-issue33/3rd/fmt-7.1.2
//   g++ -std=c++17 -O2 -DCOSMO_NN_USE_ASCEND_BACKEND \
//       -I src -I 3rd/fmt-7.1.2/include -I 3rd/include \
//       -I"${ASCEND_TOOLKIT_HOME}/include" \
//       test/ascend310p3/ascend_task_smoke.cc \
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
//       src/media/PixelFormatUtils.cc src/media/VideoFrameProcCpu.cc \
//       src/mem/MemoryPoolMng.cc src/mem/FixedBlockPool.cc \
//       src/mem/AllocatorCpu.cc src/mem/BlockFreqCalc.cc \
//       src/util/Thread.cc src/util/ThreadRegistry.cc src/util/TimeUtil.cc \
//       3rd/fmt-7.1.2/src/format.cc \
//       -I/opt/ffmpeg-4.4.1/ascend/include \
//       -L/opt/ffmpeg-4.4.1/ascend/lib -lavformat -lavcodec -lavutil -lswscale \
//       -Wl,-rpath,/opt/ffmpeg-4.4.1/ascend/lib \
//       -L"${ASCEND_TOOLKIT_HOME}/lib64" -lascendcl -lacl_dvpp -lacl_dvpp_mpi \
//       -Wl,-rpath,"${ASCEND_TOOLKIT_HOME}/lib64" -lpthread -ldl -lm \
//       -o ascend_task_smoke
//
// Usage:
//   ascend_task_smoke --om <model.om> --video <h264|h265 file|rtsp://...>
//                     [--frames N] [--conf 0.25] [--topk 300]
//                     [--instances 1|3] [--rounds R] [--expect-detections]
//                     [--preview-check]
//                     [--rtsp-reconnect --rtsp-source <annexb file>
//                      --rtsp-python <rtsp_test_source.py> --rtsp-port <port>]
//
// Always runs (cheap, hardware-free): the DetectorPool (1,3) policy check and
// the error paths (unsupported codec rejected, bad OM rejected). The video
// task itself requires the 310P3 host.

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
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
#include "media/IOsdTextRenderer.h"
#include "media/PixelFormat.h"
#include "media/VideoDecoder.h"
#include "media/VideoFrame.h"
#include "media/VideoFrameProcCpu.h"
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
#include "nn/utils/net_utils.h"
#include "nn/utils/op.h"
#include "util/InstancePool.h"

namespace {

bool require_detections_ = false;

constexpr int kDefaultRtspPort = 8554;

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

// ── Demux (file or RTSP), mirroring the engine demux lifecycle ──────────
class Demux {
public:
    Demux() = default;

    ~Demux() {
        Close();
    }

    Demux(const Demux&)            = delete;
    Demux& operator=(const Demux&) = delete;

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

// ── OM contract probe (shared with the issue #29/#31 smokes) ───────────
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
    Require(surface->memory_type == cosmo::media::FrameSurfaceMemoryType::Host,
            "decoded surface is not host memory");
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

// Builds one independent detector instance graph (own ACL context, stream,
// model and buffers) for the fixed contract probed from the OM.
std::unique_ptr<cosmo::nn::Graph> BuildGraph(const OmContract& contract, const std::string& om_path,
                                             int top_k, float conf_threshold, SmokeProfiler& profiler) {
    Require(contract.input_dims.size() == 4, "model input must be 4D NCHW");
    const int input_width  = contract.input_dims[3];
    const int input_height = contract.input_dims[2];

    cosmo::nn::CombinedModelInfo info;
    info.algorithmcode = "yolo26_det_task";
    info.type          = "yolo26_det";
    cosmo::nn::ModelInfo model;
    model.name      = "yolo26_det";
    model.filename  = "model.om";
    model.max_batch = 1;
    cosmo::nn::InputNodeInfo input;
    input.name                = "images";
    input.shape               = contract.input_dims;
    input.data_type           = 1;  // fp16 (InputNodeInfo encoding, see DataTypeFromInputInfo)
    auto preprocess           = std::make_unique<cosmo::nn::ImageToTensor>("image_to_tensor");
    preprocess->input_width   = input_width;
    preprocess->input_height  = input_height;
    preprocess->padding_color = {114, 114, 114};
    input.ops.push_back(std::move(preprocess));
    model.input_node_infos.push_back(std::move(input));
    cosmo::nn::OutputNodeInfo output;
    output.name              = "output0";
    output.shape             = {1, top_k, 6};
    output.data_type         = 1;
    auto post                = std::make_unique<cosmo::nn::YoloPost>("yolo_e2e_postprocess");
    post->nms_detection_conf = conf_threshold;
    post->top_k              = top_k;
    post->input_width        = input_width;
    post->input_height       = input_height;
    output.op                = std::move(post);
    model.output_node_infos.push_back(std::move(output));
    info.models.push_back(std::move(model));

    auto graph = std::make_unique<cosmo::nn::Graph>();
    graph->SetProfiler(&profiler);
    RequireStatus(graph->Init(info, om_path, cosmo::nn::DEVICE_ASCEND, "", /*device_id=*/0),
                  "Graph::Init failed");

    // Prove the hardware stages are wired in: DVPP image_to_tensor and
    // AscendCL net nodes, with no CPU preprocess node in between.
    const std::string& graph_dump = profiler.graph_info;
    Require(graph_dump.find("image_to_tensor_0") != std::string::npos,
            "graph is missing the Ascend DVPP image_to_tensor node");
    Require(graph_dump.find("net_0") != std::string::npos, "graph is missing the AscendCL net node");
    Require(graph_dump.find("yolo_e2e_decode_0") != std::string::npos,
            "graph is missing the yolo_e2e host postprocess node");
    return graph;
}

int CountDetections(const std::shared_ptr<cosmo::nn::Blob>& blob, float conf_threshold) {
    const auto desc = blob->GetBlobDesc();
    const auto dims = desc.dims;
    Require(dims.size() == 3 && dims[2] == 6, "unexpected output shape");
    const float* data = static_cast<const float*>(blob->GetHandle().base);
    int valid         = 0;
    for (int i = 0; i < dims[0] * dims[1]; ++i) {
        if (data[i * 6 + 4] >= conf_threshold)
            ++valid;
    }
    return valid;
}

// ── DetectorPool (1,3) policy check ─────────────────────────────────────
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
    cosmo::InstancePool<CountingDetector, std::shared_ptr<CountingDetector>> pool("ascend_detector", 1, 3);
    for (int task = 0; task < 4; ++task) {
        pool.CreateTask("alg", "cfg", "model");
    }
    Require(CountingDetector::live == 3, "Ascend DetectorPool must cap instances at three");

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

// ── On-demand host-copy check (preview/capture/OSD/record consumers) ────
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

// sws_scale-free NV12 -> I420 plane conversion (the Ascend libswscale 5 on
// the 310P3 host SIGSEGVs on NV12->I420, see issue #27), used to verify the
// host-copy preview path without the broken vendor swscale.
cosmo::media::VideoFramePtr ManualNv12ToI420(const cosmo::media::VideoFramePtr& nv12) {
    const int width  = static_cast<int>(nv12->GetWidth());
    const int height = static_cast<int>(nv12->GetHeight());
    auto dst =
        std::make_shared<cosmo::media::VideoFrame>(width, height, cosmo::media::PixelFormat::PIXEL_I420);
    if (!dst || !dst->Active()) {
        return nullptr;
    }
    // Ascend VDEC surfaces carry NV12 in two separate host planes (Y, UV);
    // fall back to plane pointers + line pitch when there is no contiguous base.
    const auto surface    = nv12->GetSurface();
    const uint8_t* y_src  = static_cast<const uint8_t*>(nv12->GetData());
    const uint8_t* uv_src = nullptr;
    size_t y_pitch        = static_cast<size_t>(width);
    size_t uv_pitch       = static_cast<size_t>(width);
    if (y_src) {
        uv_src = y_src + static_cast<size_t>(width) * height;
    } else if (surface && surface->planes.size() >= 2) {
        y_src    = surface->planes[0].virt_addr + surface->planes[0].offset;
        uv_src   = surface->planes[1].virt_addr + surface->planes[1].offset;
        y_pitch  = surface->planes[0].pitch;
        uv_pitch = surface->planes[1].pitch;
    }
    uint8_t* y = static_cast<uint8_t*>(dst->GetData());
    uint8_t* u = y + static_cast<size_t>(width) * height;
    uint8_t* v = u + static_cast<size_t>(width) * height / 4;
    for (int row = 0; row < height; ++row) {
        std::memcpy(y + static_cast<size_t>(row) * width, y_src + static_cast<size_t>(row) * y_pitch,
                    static_cast<size_t>(width));
    }
    for (int row = 0; row < height / 2; ++row) {
        const uint8_t* uv = uv_src + static_cast<size_t>(row) * uv_pitch;
        for (int col = 0; col < width / 2; ++col) {
            u[row * width / 2 + col] = uv[col * 2];
            v[row * width / 2 + col] = uv[col * 2 + 1];
        }
    }
    return dst;
}

void CheckHostConversion(const cosmo::media::VideoFramePtr& frame) {
    Require(frame != nullptr && frame->Active(), "host check needs a decoded frame");
    const auto surface = frame->GetSurface();
    Require(surface != nullptr && surface->memory_type == cosmo::media::FrameSurfaceMemoryType::Host,
            "host check needs a host NV12 surface");

    StubOsdTextRenderer osd;
    cosmo::media::VideoFrameProcCpu proc(osd);

    // Task-side host-copy consumers must see host data without a device download.
    auto copy = proc.CopyFrame(frame);
    Require(copy != nullptr && copy->Active() && copy->GetData() != nullptr, "CopyFrame failed");
    Require(proc.EnsureHostData(frame), "EnsureHostData failed on a host NV12 surface");

    // The regular consumer path is sws_scale NV12->I420, which SIGSEGVs in
    // the Ascend libswscale 5 on this host (known pre-existing env issue,
    // issue #27). Run it in a child so the crash is reported instead of
    // killing the smoke, then verify the same host-copy path without sws.
    const pid_t pid = fork();
    Require(pid >= 0, "fork failed");
    if (pid == 0) {
        auto sws_i420 = proc.NV12ToI420(frame);
        _exit(sws_i420 != nullptr && sws_i420->Active() ? 0 : 2);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        std::cout << "preview sws NV12ToI420: ok\n";
    } else {
        std::cout << "preview sws NV12ToI420: crashed/failed on this host "
                     "(known Ascend libswscale issue, verified via manual planes)\n";
    }

    auto i420 = ManualNv12ToI420(frame);
    Require(i420 != nullptr && i420->Active() && i420->GetData() != nullptr,
            "manual NV12->I420 failed on a host NV12 surface");
    Require(i420->GetWidth() == frame->GetWidth() && i420->GetHeight() == frame->GetHeight(),
            "host conversion changed the frame dimensions");

    auto jpeg = proc.EncodeJpeg(i420);
    Require(!jpeg.empty(), "capture JPEG encode failed after host conversion");
    std::cout << "preview host-copy passed: NV12 -> I420 host, capture JPEG bytes=" << jpeg.size() << "\n";
}

// ── Error paths (no silent software fallback) ───────────────────────────
void CheckErrorPaths(const std::string& om_path) {
    // Unsupported codecs must refuse to open instead of falling back to the
    // FFmpeg software decoder.
    auto decoder = cosmo::media::VideoDecoder::Create(0, nullptr);
    decoder->SetCodecType(cosmo::media::VideoCodecType::kMjpeg, 1920, 1080);
    Require(!decoder->Open(), "Ascend decoder must refuse unsupported codecs (no software fallback)");

    // A corrupt OM must fail model load and graph init with an error status.
    const std::string bad_om = "/tmp/ascend_task_smoke_bad.om";
    {
        std::ofstream out(bad_om, std::ios::binary | std::ios::trunc);
        Require(bool(out), "cannot create bad OM file");
        out << "not an om artifact";
    }
    OmContract dummy;
    constexpr size_t kOmInputBytes = 960 * 960 * 3 * 2;
    dummy.input_dims               = {1, 3, 960, 960};
    dummy.input_bytes              = kOmInputBytes;
    SmokeProfiler profiler;
    try {
        BuildGraph(dummy, bad_om, 300, 0.25F, profiler);
        throw std::runtime_error("bad OM must fail Graph::Init");
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("bad OM must fail") != std::string::npos)
            throw;
        std::cout << "error paths passed: bad OM rejected: " << e.what() << "\n";
    }

    std::cout << "error paths passed: unsupported codec rejected with no fallback\n";
}

// ── RTSP publisher management (test/ascend310p3/rtsp_test_source.py) ────
class RtspPublisher {
public:
    RtspPublisher(const std::string& python, const std::string& source, int port)
        : python_(python), source_(source), port_(port) {}
    ~RtspPublisher() {
        Stop();
    }

    RtspPublisher(const RtspPublisher&)            = delete;
    RtspPublisher& operator=(const RtspPublisher&) = delete;

    void Start() {
        if (pid_ > 0)
            return;
        const std::string port = std::to_string(port_);
        pid_                   = fork();
        Require(pid_ >= 0, "fork failed");
        if (pid_ == 0) {
            execlp("python3", "python3", python_.c_str(), "--file", source_.c_str(), "--codec",
                   source_.find(".265") != std::string::npos || source_.find(".h265") != std::string::npos
                       ? "h265"
                       : "h264",
                   "--port", port.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    void Stop() {
        if (pid_ <= 0)
            return;
        kill(pid_, SIGKILL);
        int status = 0;
        waitpid(pid_, &status, 0);
        pid_ = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

private:
    std::string python_;
    std::string source_;
    int port_  = kDefaultRtspPort;
    pid_t pid_ = -1;
};

}  // namespace

// ── Standalone harness stubs (same as ascend_video_detect_smoke.cc) ─────
namespace cosmo::nn {
Status NetUtils::OptimizePreOps(std::vector<std::unique_ptr<Op>>& /*ops*/, bool /*use_skip*/) {
    return COSMO_NN_OK;
}

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

namespace cosmo::util {
std::string GenerateUUID() {
    static std::atomic<uint64_t> counter{0};
    return "ascend-task-smoke-" + std::to_string(counter.fetch_add(1));
}
}  // namespace cosmo::util

namespace {

struct RunStats {
    size_t decoded_frames = 0;
    int detected_frames   = 0;
    double graph_ms       = 0.0;
};

// One demux session against the persistent task decoder. The engine keeps one
// Ascend DVPP VDEC channel for the whole task (AlgChannelDecode reuses it
// across stream restarts; reopening the channel while the DVPP VPC channel is
// alive wedges the next decode session on the 310P3 — see the adaptation
// plan). An RTSP outage or file EOF returns early so the caller reconnects
// through a fresh demux. `first_frame` receives the first decoded frame for
// the preview check; `mid_run_interrupt` fires once after `interrupt_after`
// decoded frames to simulate a live-source disconnect mid-task.
size_t RunSourceOnce(const std::vector<std::unique_ptr<cosmo::nn::Graph>>& graphs,
                     const std::vector<std::unique_ptr<SmokeProfiler>>& profilers,
                     cosmo::media::VideoDecoder& decoder, const std::string& video_path, int frames_needed,
                     float conf_threshold, cosmo::media::VideoFramePtr& first_frame, RunStats& stats,
                     const std::function<void()>& mid_run_interrupt, size_t interrupt_after,
                     bool* interrupt_fired) {
    Demux demux;
    if (!demux.Open(video_path)) {
        std::cerr << "demux open failed, task will retry: " << video_path << "\n";
        return 0;
    }
    AVPacket* packet = av_packet_alloc();
    Require(packet != nullptr, "av_packet_alloc failed");
    int64_t packets_sent = 0;
    size_t decoded_count = 0;
    const auto run_start = std::chrono::steady_clock::now();

    auto run_decoded = [&](const cosmo::media::VideoFramePtr& frame) {
        if (!frame || decoded_count >= static_cast<size_t>(frames_needed))
            return;
        ++decoded_count;
        if (!first_frame)
            first_frame = frame;

        std::vector<std::vector<std::shared_ptr<cosmo::nn::Blob>>> params;
        params.push_back({MakeSurfaceBlob(frame)});
        const auto graph_start = std::chrono::steady_clock::now();
        bool any_detection     = false;
        for (size_t i = 0; i < graphs.size(); ++i) {
            RequireStatus(graphs[i]->Forward(params),
                          "Graph::Forward failed (instance " + std::to_string(i) + ")");
            auto outputs = graphs[i]->Output();
            Require(outputs.size() == 1, "expected one graph output blob");
            const int detections = CountDetections(outputs[0], conf_threshold);
            if (detections > 0)
                any_detection = true;
            if (i == 0 && decoded_count % 25 == 1) {
                std::printf(
                    "frame=%zu e2e_graph=%.2fms image_to_tensor=%.2fms acl=%.2fms "
                    "postprocess=%.2fms dets=%d\n",
                    decoded_count,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - graph_start)
                        .count(),
                    profilers[i]->times.image_to_tensor_ms, profilers[i]->times.acl_ms,
                    profilers[i]->times.postprocess_ms, detections);
            }
        }
        if (any_detection)
            ++stats.detected_frames;
        stats.graph_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - graph_start).count();
        std::fflush(stdout);
    };

    auto drain = [&]() {
        while (decoded_count < static_cast<size_t>(frames_needed)) {
            auto extra = decoder.GetFrame();
            if (!extra)
                break;
            run_decoded(extra);
        }
    };

    while (decoded_count < static_cast<size_t>(frames_needed) && demux.Next(packet)) {
        bool sent = false;
        for (int attempt = 0; attempt < 2 && !sent; ++attempt) {
            sent = decoder.SendPacket(packet->data, static_cast<size_t>(packet->size), packets_sent);
            if (!sent)
                drain();  // decoder backpressure (EAGAIN): free an output surface, retry the packet
        }
        ++packets_sent;
        av_packet_unref(packet);
        if (!sent) {
            if (decoded_count < static_cast<size_t>(frames_needed)) {
                std::cerr << "Ascend decoder SendPacket failed at packet " << packets_sent
                          << ", task will retry: " << video_path << "\n";
            }
            break;
        }
        drain();
        if (mid_run_interrupt && !*interrupt_fired &&
            stats.decoded_frames + decoded_count >= interrupt_after) {
            *interrupt_fired = true;
            mid_run_interrupt();
        }
        // Pace the feed: the async Ascend VDEC delivers frames in
        // near-real-time as packets arrive; bulk-feeding leaves them queued
        // in the channel until an end-of-stream flush (which would disable
        // the channel for the next stream).
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    av_packet_free(&packet);

    // Short drain poll so frames the channel has already decoded are
    // collected before the session returns. No end-of-stream flush: the
    // Ascend FFmpeg wrapper treats a flush as final and drops all later
    // packets, so the persistent channel must never be flushed mid-task.
    if (decoded_count < static_cast<size_t>(frames_needed)) {
        const auto drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        int empty_rounds          = 0;
        while (decoded_count < static_cast<size_t>(frames_needed) &&
               std::chrono::steady_clock::now() < drain_deadline && empty_rounds < 100) {
            auto extra = decoder.GetFrame();
            if (!extra) {
                ++empty_rounds;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            empty_rounds = 0;
            run_decoded(extra);
        }
    }

    if (decoded_count == 0) {
        // Stream ended before any frame (e.g. RTSP outage at start): keep the
        // task channel open and let the caller reconnect through a fresh demux.
        std::cerr << "source session produced no frames, task will retry: " << video_path << "\n";
        return 0;
    }

    const auto run_end = std::chrono::steady_clock::now();
    stats.decoded_frames += decoded_count;
    std::printf(
        "source session done: decoded=%zu in %.2fs (%.1f fps incl. decode)\n", decoded_count,
        std::chrono::duration<double>(run_end - run_start).count(),
        static_cast<double>(decoded_count) / std::chrono::duration<double>(run_end - run_start).count());
    return decoded_count;
}

// One full task round: retries demux sessions (RTSP reconnect, file restart)
// on the persistent task decoder until `frames_needed` frames are decoded
// (the existing demux/task lifecycle; no software inference fallback).
cosmo::media::VideoFramePtr RunTaskRound(const std::vector<std::unique_ptr<cosmo::nn::Graph>>& graphs,
                                         const std::vector<std::unique_ptr<SmokeProfiler>>& profilers,
                                         cosmo::media::VideoDecoder& decoder, const std::string& video_path,
                                         int frames_needed, float conf_threshold, RunStats& stats,
                                         const std::function<void()>& mid_run_interrupt) {
    cosmo::media::VideoFramePtr first_frame;
    const auto deadline          = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    size_t round_base            = stats.decoded_frames;
    bool interrupt_fired         = false;
    const size_t interrupt_after = static_cast<size_t>(frames_needed) / 2;
    while (stats.decoded_frames - round_base < static_cast<size_t>(frames_needed)) {
        RunSourceOnce(graphs, profilers, decoder, video_path, frames_needed, conf_threshold, first_frame,
                      stats, mid_run_interrupt, interrupt_after, &interrupt_fired);
        if (stats.decoded_frames - round_base >= static_cast<size_t>(frames_needed))
            break;
        Require(std::chrono::steady_clock::now() < deadline,
                "task round timed out reconnecting to: " + video_path);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return first_frame;
}

}  // namespace

int main(int argc, char** argv) {
    // Pool-based VideoFrame allocation (preview/capture host-copy consumers)
    // routes through the global memory pool, same as the product app_init.
    cosmo::mem::MemoryPoolMng memory_pool(std::make_unique<cosmo::mem::AllocatorCpu>());
    cosmo::mem::SetMemoryPoolContext(&memory_pool);

    std::string om_path;
    std::string video_path;
    float conf_threshold = 0.25F;
    int top_k            = 300;
    int frames_needed    = 30;
    int instances        = 1;
    int rounds           = 1;
    bool preview_check   = false;
    bool rtsp_reconnect  = false;
    std::string rtsp_source;
    std::string rtsp_python;
    int rtsp_port = kDefaultRtspPort;

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
        else if (arg == "--instances")
            instances = std::stoi(value("--instances"));
        else if (arg == "--rounds")
            rounds = std::stoi(value("--rounds"));
        else if (arg == "--preview-check")
            preview_check = true;
        else if (arg == "--rtsp-reconnect")
            rtsp_reconnect = true;
        else if (arg == "--rtsp-source")
            rtsp_source = value("--rtsp-source");
        else if (arg == "--rtsp-python")
            rtsp_python = value("--rtsp-python");
        else if (arg == "--rtsp-port")
            rtsp_port = std::stoi(value("--rtsp-port"));
        else if (arg == "--expect-detections")
            require_detections_ = true;
        else
            throw std::runtime_error("unknown argument: " + arg);
    }
    Require(!om_path.empty(), "--om <model.om> is required");
    Require(!video_path.empty(), "--video <h264|h265 file|rtsp://...> is required");
    Require(frames_needed >= 1, "--frames must be >= 1");
    Require(top_k >= 1, "--topk must be >= 1");
    Require(instances == 1 || instances == 3, "--instances must be 1 or 3");
    Require(rounds >= 1, "--rounds must be >= 1");
    Require(!rtsp_reconnect || video_path.rfind("rtsp://", 0) == 0,
            "--rtsp-reconnect requires an rtsp:// --video");
    Require(!rtsp_reconnect || !rtsp_source.empty(), "--rtsp-reconnect requires --rtsp-source <annexb file>");
    Require(!rtsp_reconnect || !rtsp_python.empty(),
            "--rtsp-reconnect requires --rtsp-python <rtsp_test_source.py>");

    // ── Hardware-free gates first ───────────────────────────────────────
    CheckDetectorPoolPolicy();
    CheckErrorPaths(om_path);

    // ── 310P3 task run ──────────────────────────────────────────────────
    Require(avcodec_find_decoder_by_name("h264_ascend") != nullptr,
            "h264_ascend decoder is unavailable on this FFmpeg");
    Require(avcodec_find_decoder_by_name("h265_ascend") != nullptr,
            "h265_ascend decoder is unavailable on this FFmpeg");
    std::cout << "gate0 decoders present: h264_ascend h265_ascend\n";

    const OmContract contract = ProbeOm(om_path);
    std::printf("om=%s input_dims=[%d,%d,%d,%d] input_bytes=%zu\n", om_path.c_str(), contract.input_dims[0],
                contract.input_dims[1], contract.input_dims[2], contract.input_dims[3], contract.input_bytes);

    std::vector<std::unique_ptr<cosmo::nn::Graph>> graphs;
    std::vector<std::unique_ptr<SmokeProfiler>> profilers;
    for (int i = 0; i < instances; ++i) {
        profilers.push_back(std::make_unique<SmokeProfiler>());
        graphs.push_back(BuildGraph(contract, om_path, top_k, conf_threshold, *profilers.back()));
        std::printf("instance %d/%d graph ready (DVPP -> AscendCL -> yolo_e2e)\n", i + 1, instances);
    }

    RtspPublisher publisher(rtsp_python, rtsp_source, rtsp_port);
    if (rtsp_reconnect)
        publisher.Start();

    // One task decoder for the whole run: the engine keeps a single Ascend
    // DVPP VDEC channel per task and AlgChannelDecode reuses it across stream
    // restarts (reopening the channel while the DVPP VPC channel is alive
    // wedges the next decode session on the 310P3 — see the adaptation plan).
    int video_width                         = 0;
    int video_height                        = 0;
    cosmo::media::VideoCodecType codec_type = cosmo::media::VideoCodecType::kH264;
    {
        Demux probe;
        Require(probe.Open(video_path), "cannot probe video: " + video_path);
        video_width  = probe.Width();
        video_height = probe.Height();
        codec_type   = probe.CodecId() == AV_CODEC_ID_HEVC ? cosmo::media::VideoCodecType::kH265
                                                           : cosmo::media::VideoCodecType::kH264;
    }
    auto decoder = cosmo::media::VideoDecoder::Create(0, nullptr);
    decoder->SetCodecType(codec_type, video_width, video_height);
    Require(decoder->Open(), "Ascend decoder open failed (no software fallback): " + video_path);
    std::printf("task decoder opened: %s %dx%d (persistent across rounds)\n",
                codec_type == cosmo::media::VideoCodecType::kH265 ? "h265_ascend" : "h264_ascend",
                video_width, video_height);

    RunStats stats;
    cosmo::media::VideoFramePtr preview_frame;
    for (int round = 0; round < rounds; ++round) {
        const std::function<void()> mid_run_interrupt =
            rtsp_reconnect && round == 0 ? std::function<void()>([&]() {
                std::cout << "rtsp interrupt: killing publisher mid-stream...\n";
                publisher.Stop();
                std::this_thread::sleep_for(std::chrono::seconds(2));
                std::cout << "rtsp reconnect: restarting publisher...\n";
                publisher.Start();
            })
                                         : std::function<void()>();
        std::printf("task round %d/%d start (start/stop lifecycle)\n", round + 1, rounds);
        const size_t round_base = stats.decoded_frames;
        const auto frame        = RunTaskRound(graphs, profilers, *decoder, video_path, frames_needed,
                                               conf_threshold, stats, mid_run_interrupt);
        if (!preview_frame)
            preview_frame = frame;
        const size_t round_frames = stats.decoded_frames - round_base;
        Require(round_frames > 0, "task round produced no frames: " + video_path);
        std::printf("task round %d/%d stop (decoded %zu frames)\n", round + 1, rounds, round_frames);
    }

    if (preview_check)
        CheckHostConversion(preview_frame);

    // Release the task channel. The DVPP VDEC channel is torn down while the
    // graph VPC channels are still alive (known 310P3 driver constraint: a
    // subsequent VDEC reopen in the same process would wedge; the smoke exits
    // right after, see the adaptation plan).
    Require(decoder->Close(), "Ascend decoder close failed");

    const size_t total_frames = stats.decoded_frames;
    std::printf("e2e=%zu frames across %d instances x %d rounds (%.1f ms/frame across instances)\n",
                total_frames, instances, rounds, total_frames > 0 ? stats.graph_ms / total_frames : 0.0);
    std::printf("detected_frames=%d/%zu (conf>=%.2f)\n", stats.detected_frames, total_frames, conf_threshold);
    if (require_detections_ && stats.detected_frames == 0)
        throw std::runtime_error("expected at least one detection, got zero");
    std::printf("task smoke OK\n");
    return 0;
}
