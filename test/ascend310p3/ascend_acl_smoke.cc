// Host-level runnable check for issue #29: run one prepared FP16 NCHW input
// through the repository Graph (AscendNetNode + yolo_e2e decode) on a real
// Ascend 310P3 and print parsed YOLO26 detections.
//
// The smoke drives the real engine pieces: Graph::Init/Forward/Output, the
// AscendNetNode (aclInit once, per-graph context/stream/model/datasets/buffers,
// H2D copy -> aclmdlExecuteAsync -> stream sync -> D2H copy with FP16->FP32),
// and the yolo_e2e postprocess node. No ONNX Runtime or CPU fallback exists.
//
// Input contract (verified on the 310P3 host, 2026-08-05): the OM input is
// RGB NCHW FP16 normalized to 0..1, i.e. exactly the Ultralytics preprocessed
// tensor (/255, BGR->RGB). Feeding 0..255 raw pixels yields zero detections;
// feeding 0..1 reproduces the Ultralytics reference boxes/scores (see
// docs/development/ascend310p3-adaptation-plan.md, Issue #29 record).
//
// Built standalone on the 310P3 host against the CANN toolkit:
//   export PATH=/root/.cargo/bin:$PATH
//   source /usr/local/Ascend/ascend-toolkit/set_env.sh
//   ln -sfn /root/cosmo-edge-issue29/fmt-7.1.2 /root/cosmo-edge-issue29/3rd/fmt-7.1.2
//   g++ -std=c++17 -O2 -DCOSMO_NN_USE_ASCEND_BACKEND \
//       -I src -I 3rd/fmt-7.1.2/include \
//       -I"${ASCEND_TOOLKIT_HOME}/include" \
//       test/ascend310p3/ascend_acl_smoke.cc \
//       src/nn/device/ascend/ascend_net_node.cc \
//       src/nn/device/ascend/ascend_node_creator.cc \
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
//       3rd/fmt-7.1.2/src/format.cc \
//       -L"${ASCEND_TOOLKIT_HOME}/lib64" -lascendcl \
//       -Wl,-rpath,"${ASCEND_TOOLKIT_HOME}/lib64" -lpthread -o ascend_acl_smoke
//
// Usage:
//   ascend_acl_smoke --om <model.om> [--input <fp16_nchw.raw>] [--fill 0.5]
//                    [--iters 2] [--conf 0.25] [--topk 300] [--expect-detections]
//
// --input reads a prepared FP16 NCHW tensor in the 0..1 RGB contract above
// (bytes must match the OM input); without it a constant FP16 tensor (--fill,
// default 0.5) is generated.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "fmt/format.h"
#include "nn/core/blob.h"
#include "nn/core/graph.h"
#include "nn/core/status.h"
#include "nn/device/ascend/ascend_acl.h"
#include "nn/node/identity_node.h"
#include "nn/node/input_node.h"
#include "nn/node/node_creator.h"
#include "nn/node/yolo_e2e_decode_node.h"
#include "nn/utils/dims_vector_utils.h"
#include "nn/utils/net_utils.h"
#include "nn/utils/op.h"

namespace {

bool require_detections_ = false;

// InputNodeInfo raw data_type encoding, see DataTypeFromInputInfo
// (nn/utils/data_type_utils.h): 0=fp32, 1=fp16, 2=bfp16, 3=int8.
constexpr int kInputInfoFp16 = 1;

void Require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void RequireStatus(cosmo::nn::Status status, const std::string& message) {
    if (!bool(status))
        throw std::runtime_error(message + ": " + status.description());
}

std::string ReadFile(const std::string& path, size_t expected_size) {
    std::ifstream stream(path, std::ios::in | std::ios::binary);
    Require(bool(stream), "cannot open " + path);
    stream.seekg(0, std::ios::end);
    const std::streamoff length = stream.tellg();
    stream.seekg(0, std::ios::beg);
    Require(length > 0, path + " is empty");
    if (expected_size != 0)
        Require(static_cast<size_t>(length) == expected_size,
                path + " size " + std::to_string(length) + " != expected " + std::to_string(expected_size));
    std::string data(static_cast<size_t>(length), '\0');
    stream.read(&data[0], length);
    Require(bool(stream), "failed to read " + path);
    return data;
}

uint16_t FloatToFp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign     = (bits >> 16U) & 0x8000U;
    const uint32_t exponent = (bits >> 23U) & 0xffU;
    const uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU)
        return static_cast<uint16_t>(sign | 0x7c00U | (mantissa ? 0x0200U : 0U));
    const int32_t biased = static_cast<int32_t>(exponent) - 127 + 15;
    if (biased >= 0x1f)
        return static_cast<uint16_t>(sign | 0x7c00U);
    if (biased <= 0) {
        if (biased < -10)
            return static_cast<uint16_t>(sign);
        return static_cast<uint16_t>(sign | ((mantissa | 0x800000U) >> (14 - biased)));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(biased) << 10U) | (mantissa >> 13U));
}

// Probe the OM input contract with AscendCL before building the graph.
struct OmContract {
    std::vector<int> input_dims;
    size_t input_bytes = 0;
};

OmContract ProbeOm(const std::string& om_path) {
    constexpr int kDeviceId = 0;
    // aclInit runs once per process (shared with AscendNetNode via
    // ascend::EnsureAclInitialized); the smoke must not finalize.
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

std::shared_ptr<cosmo::nn::Blob> MakeInputBlob(const OmContract& contract, const std::string& input_path,
                                               float fill_value) {
    using namespace cosmo::nn;
    Require(contract.input_dims.size() == 4, "probe input must be 4D");
    const size_t elements =
        static_cast<size_t>(contract.input_dims[0]) * static_cast<size_t>(contract.input_dims[1]) *
        static_cast<size_t>(contract.input_dims[2]) * static_cast<size_t>(contract.input_dims[3]);
    Require(elements * sizeof(uint16_t) == contract.input_bytes,
            "probe input bytes do not match FP16 element count");

    BlobDesc desc;
    desc.device_type = DEVICE_NAIVE;
    desc.data_type   = DATA_TYPE_HALF;
    desc.dims        = contract.input_dims;
    desc.name        = "images";
    auto blob        = std::make_shared<Blob>(desc, true);
    uint16_t* data   = static_cast<uint16_t*>(blob->GetHandle().base);
    Require(data != nullptr, "input blob allocation failed");

    if (!input_path.empty()) {
        const std::string raw = ReadFile(input_path, contract.input_bytes);
        std::memcpy(data, raw.data(), raw.size());
    } else {
        const uint16_t value = FloatToFp16(fill_value);
        for (size_t i = 0; i < elements; ++i)
            data[i] = value;
    }
    return blob;
}

void PrintDetections(const std::shared_ptr<cosmo::nn::Blob>& blob, float conf_threshold) {
    using namespace cosmo::nn;
    const auto desc = blob->GetBlobDesc();
    const auto dims = desc.dims;
    Require(dims.size() == 3 && dims[2] == 6, "unexpected output shape");
    const float* data = static_cast<const float*>(blob->GetHandle().base);
    int valid         = 0;
    for (int i = 0; i < dims[0] * dims[1]; ++i) {
        const float* row = data + i * 6;
        if (!(row[4] >= conf_threshold))
            continue;
        std::printf("det[%d] cx=%.1f cy=%.1f w=%.1f h=%.1f score=%.4f class=%d\n", valid, row[0], row[1],
                    row[2], row[3], row[4], static_cast<int>(row[5]));
        ++valid;
    }
    std::printf("parsed_detections=%d (conf>=%.2f)\n", valid, conf_threshold);
    if (require_detections_ && valid == 0)
        throw std::runtime_error("expected at least one detection, got zero");
}

}  // namespace

// Standalone harness: net_utils.cc (Eigen/tokenizers-dependent) is not linked,
// so provide the one Graph-referenced entry point. The smoke graph declares no
// preprocess ops; fail loudly if that ever changes.
namespace cosmo::nn {
Status NetUtils::OptimizePreOps(std::vector<std::unique_ptr<Op>>& ops, bool /*use_skip*/) {
    if (!ops.empty())
        return Status(COSMO_NN_ERR_NOT_IMPLEMENTED,
                      "NetUtils::OptimizePreOps stub: smoke config declares preprocess ops");
    return COSMO_NN_OK;
}

// Minimal host node creator for the standalone harness: the smoke graph only
// needs the pass-through input, identity, and yolo_e2e decode fallback nodes
// (the engine HostNodeCreator would drag in every host decode node).
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
// engine links glog; this check only needs the fmt-format callback).
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

int main(int argc, char** argv) {
    std::string om_path;
    std::string input_path;
    float fill_value     = 0.5f;
    float conf_threshold = 0.25f;
    int top_k            = 300;
    int iters            = 2;
    require_detections_  = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&](const char* flag) {
            Require(i + 1 < argc, std::string("missing value for ") + flag);
            return std::string(argv[++i]);
        };
        if (arg == "--om")
            om_path = value("--om");
        else if (arg == "--input")
            input_path = value("--input");
        else if (arg == "--fill")
            fill_value = std::stof(value("--fill"));
        else if (arg == "--conf")
            conf_threshold = std::stof(value("--conf"));
        else if (arg == "--topk")
            top_k = std::stoi(value("--topk"));
        else if (arg == "--iters")
            iters = std::stoi(value("--iters"));
        else if (arg == "--expect-detections")
            require_detections_ = true;
        else
            throw std::runtime_error("unknown argument: " + arg);
    }
    Require(!om_path.empty(), "--om <model.om> is required");
    Require(iters >= 1, "--iters must be >= 1");
    Require(top_k >= 1, "--topk must be >= 1");

    using namespace cosmo::nn;

    const OmContract contract = ProbeOm(om_path);
    std::printf("om=%s input_dims=[", om_path.c_str());
    for (size_t i = 0; i < contract.input_dims.size(); ++i)
        std::printf("%s%d", i ? "," : "", contract.input_dims[i]);
    std::printf("] input_bytes=%zu\n", contract.input_bytes);
    const int height = contract.input_dims[2];
    const int width  = contract.input_dims[3];

    CombinedModelInfo info;
    info.algorithmcode = "yolo26_det_smoke";
    info.type          = "yolo26_det";
    ModelInfo model;
    model.name      = "yolo26_det";
    model.filename  = "model.om";
    model.max_batch = 1;
    InputNodeInfo input;
    input.name      = "images";
    input.shape     = contract.input_dims;
    input.data_type = kInputInfoFp16;
    model.input_node_infos.push_back(std::move(input));
    OutputNodeInfo output;
    output.name              = "output0";
    output.shape             = {1, top_k, 6};
    output.data_type         = kInputInfoFp16;
    auto post                = std::make_unique<YoloPost>("yolo_e2e_postprocess");
    post->nms_detection_conf = conf_threshold;
    post->top_k              = top_k;
    post->input_width        = width;
    post->input_height       = height;
    output.op                = std::move(post);
    model.output_node_infos.push_back(std::move(output));
    info.models.push_back(std::move(model));

    std::printf("graph_init device=DEVICE_ASCEND...\n");
    Graph graph;
    RequireStatus(graph.Init(info, om_path, DEVICE_ASCEND, "", /*device_id=*/0), "Graph::Init failed");

    auto input_blob = MakeInputBlob(contract, input_path, fill_value);

    std::vector<float> reference;
    for (int it = 0; it < iters; ++it) {
        std::vector<std::vector<std::shared_ptr<Blob>>> params;
        params.push_back({input_blob});
        RequireStatus(graph.Forward(params), "Graph::Forward failed");

        auto outputs = graph.Output();
        Require(outputs.size() == 1, "expected one graph output blob");
        auto& top           = outputs[0];
        const auto desc     = top->GetBlobDesc();
        const auto dims     = desc.dims;
        const size_t floats = DimsVectorUtils::Count(dims);
        const float* data   = static_cast<const float*>(top->GetHandle().base);

        if (it == 0) {
            reference.assign(data, data + floats);
            std::printf("iteration=%d output_dims=[%d,%d,%d] dtype=FP32\n", it, dims[0], dims[1], dims[2]);
            PrintDetections(top, conf_threshold);
        } else {
            Require(std::memcmp(reference.data(), data, floats * sizeof(float)) == 0,
                    "iteration " + std::to_string(it) + " output differs from iteration 0");
            std::printf("iteration=%d output identical to iteration 0 (buffers reused)\n", it);
        }
    }

    std::printf("smoke OK: %d iteration(s), deterministic parsed detections on device 0\n", iters);
    return 0;
}
