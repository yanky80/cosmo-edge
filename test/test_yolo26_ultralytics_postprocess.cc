#include "catch_amalgamated.hpp"

#ifdef COSMO_NN_USE_CPU_BACKEND

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "nn/core/blob.h"
#include "nn/node/node_type_utils.h"
#include "nn/pipeline/pipeline_utils.h"
#include "nn/utils/dims_vector_utils.h"
#include "nn/utils/net_utils.h"
#include "nn/utils/op.h"

namespace {

using Catch::Approx;

std::shared_ptr<cosmo::nn::Blob> MakeBlob(cosmo::nn::DataType type, const cosmo::nn::DimsVector& dims) {
    cosmo::nn::BlobDesc desc;
    desc.data_type   = type;
    desc.device_type = cosmo::nn::DEVICE_NAIVE;
    desc.data_format = cosmo::nn::DATA_FORMAT_NCHW;
    desc.dims        = dims;
    return std::make_shared<cosmo::nn::Blob>(desc, true);
}

std::uint16_t FloatToHalf(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign     = (bits >> 16) & 0x8000u;
    std::int32_t exponent        = static_cast<std::int32_t>((bits >> 23) & 0xff) - 127 + 15;
    const std::uint32_t mantissa = (bits >> 13) & 0x03ffu;
    if (exponent >= 0x1f)
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10)
            return static_cast<std::uint16_t>(sign);
        const std::uint32_t subnormal = (mantissa | 0x0400u) >> static_cast<std::uint32_t>(1 - exponent);
        return static_cast<std::uint16_t>(sign | subnormal);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | mantissa);
}

template <typename T>
void SetCell(std::shared_ptr<cosmo::nn::Blob> blob, int cell, float cx, float cy, float w, float h,
             std::initializer_list<float> scores) {
    const auto dims = blob->GetBlobDesc().dims;
    const int cells = dims.at(2);
    auto* data      = static_cast<T*>(blob->GetHandle().base);
    const auto conv = [](float v) -> T {
        if constexpr (std::is_same_v<T, std::uint16_t>)
            return FloatToHalf(v);
        else
            return v;
    };
    data[cell]             = conv(cx);
    data[cells + cell]     = conv(cy);
    data[2 * cells + cell] = conv(w);
    data[3 * cells + cell] = conv(h);
    int class_id           = 0;
    for (float score : scores)
        data[(4 + class_id++) * cells + cell] = conv(score);
}

std::unique_ptr<cosmo::nn::YoloPost> MakeOp(float conf, float nms, int top_k) {
    return cosmo::nn::pipeline_utils::MakeYolo26UltralyticsPostOp(nms, conf, top_k, 640, 640);
}

}  // namespace

TEST_CASE("YOLO26 ultralytics output_format selects single-output host postprocess", "[nn][yolo26]") {
    auto op = MakeOp(0.25f, 0.45f, 300);
    REQUIRE(op != nullptr);
    CHECK(op->name == "yolo26_ultralytics_postprocess");
    CHECK(op->nms_detection_conf == 0.25f);
    CHECK(op->nms_threshold == 0.45f);
    CHECK(op->top_k == 300);
    CHECK(op->input_width == 640);
    CHECK(op->input_height == 640);
}

TEST_CASE("YOLO26 ultralytics golden: threshold, class-aware NMS, top-k", "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_ultralytics_postprocess"), 0, 1,
        cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);
    auto op = MakeOp(0.25f, 0.45f, 300);
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    // [1, 6, 100]: rows 0-3 cx/cy/w/h in net-input pixels, rows 4-5 class scores.
    auto bottom = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 6, 100});
    std::fill(static_cast<float*>(bottom->GetHandle().base),
              static_cast<float*>(bottom->GetHandle().base) + 600, 0.0f);

    // Cell 0: class 0, score 0.9, box (cx=88, cy=88, w=40, h=24).
    SetCell<float>(bottom, 0, 88.0f, 88.0f, 40.0f, 24.0f, {0.9f, 0.1f});
    // Cell 1: same class, IoU 2/3 with cell 0 -> must be NMS-suppressed.
    SetCell<float>(bottom, 1, 96.0f, 88.0f, 40.0f, 24.0f, {0.8f, 0.0f});
    // Cell 2: class 1, score 0.95, box (cx=88, cy=96, w=48, h=64).
    SetCell<float>(bottom, 2, 88.0f, 96.0f, 48.0f, 64.0f, {0.1f, 0.95f});
    // Cell 3: below the 0.25 confidence threshold.
    SetCell<float>(bottom, 3, 100.0f, 100.0f, 10.0f, 10.0f, {0.2f, 0.1f});

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};

    auto status = node->Forward(bottoms, tops);
    REQUIRE(status == cosmo::nn::COSMO_NN_OK);

    const auto* decoded = static_cast<const float*>(tops[0]->GetHandle().base);
    // Sorted by descending score: class 1 (0.95) then class 0 (0.9).
    CHECK(decoded[0 * 6 + 4] == Approx(0.95f).epsilon(1e-6f));
    CHECK(decoded[0 * 6 + 5] == 1.0f);
    CHECK(decoded[0 * 6 + 0] == 88.0f);
    CHECK(decoded[0 * 6 + 1] == 96.0f);
    CHECK(decoded[0 * 6 + 2] == 48.0f);
    CHECK(decoded[0 * 6 + 3] == 64.0f);

    CHECK(decoded[1 * 6 + 4] == Approx(0.9f).epsilon(1e-6f));
    CHECK(decoded[1 * 6 + 5] == 0.0f);
    CHECK(decoded[1 * 6 + 0] == 88.0f);
    CHECK(decoded[1 * 6 + 1] == 88.0f);
    CHECK(decoded[1 * 6 + 2] == 40.0f);
    CHECK(decoded[1 * 6 + 3] == 24.0f);

    // Exactly two survivors; every remaining row is zero-filled.
    CHECK(decoded[2 * 6 + 4] == 0.0f);
    CHECK(decoded[299 * 6 + 4] == 0.0f);

    // top-k truncates the sorted survivors.
    auto op_top1 = MakeOp(0.25f, 0.45f, 1);
    node->LoadParam(op_top1.get());
    status = node->Forward(bottoms, tops);
    REQUIRE(status == cosmo::nn::COSMO_NN_OK);
    CHECK(decoded[0 * 6 + 4] == Approx(0.95f).epsilon(1e-6f));
    CHECK(decoded[1 * 6 + 4] == 0.0f);
}

TEST_CASE("YOLO26 ultralytics letterbox coordinate recovery", "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_ultralytics_postprocess"), 0, 1,
        cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);
    auto op = MakeOp(0.25f, 0.45f, 300);
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    auto bottom = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 6, 100});
    std::fill(static_cast<float*>(bottom->GetHandle().base),
              static_cast<float*>(bottom->GetHandle().base) + 600, 0.0f);
    SetCell<float>(bottom, 0, 88.0f, 96.0f, 48.0f, 64.0f, {0.9f, 0.1f});

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};
    auto status = node->Forward(bottoms, tops);
    REQUIRE(status == cosmo::nn::COSMO_NN_OK);

    // 640x640 net input back to a 1280x720 source via the pipeline's
    // gravity-2 AdjustSize (scale by 2, subtract the 280px vertical padding).
    std::vector<std::vector<cosmo::nn::ObjectInfoV1>> recovered;
    auto pick_status = cosmo::nn::NetUtils::PickDetectionObjects(
        tops[0], {cosmo::nn::Size(1280, 720)}, cosmo::nn::Size(640, 640), {}, {}, {}, recovered);
    REQUIRE(pick_status == cosmo::nn::COSMO_NN_OK);
    REQUIRE(recovered.size() == 1);
    REQUIRE(recovered[0].size() == 1);
    CHECK(recovered[0][0].x1 == 128.0f);
    CHECK(recovered[0][0].y1 == -152.0f);
    CHECK(recovered[0][0].x2 == 224.0f);
    CHECK(recovered[0][0].y2 == -24.0f);
    CHECK(recovered[0][0].infos[0].confidence == Approx(0.9f).epsilon(1e-6f));
    CHECK(recovered[0][0].infos[0].class_id == 0);
}

TEST_CASE("YOLO26 ultralytics FP16 output decodes like FP32", "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_ultralytics_postprocess"), 0, 1,
        cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);
    auto op = MakeOp(0.25f, 0.45f, 300);
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    auto bottom = MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 6, 100});
    std::fill(static_cast<std::uint16_t*>(bottom->GetHandle().base),
              static_cast<std::uint16_t*>(bottom->GetHandle().base) + 600, FloatToHalf(0.0f));
    SetCell<std::uint16_t>(bottom, 0, 88.0f, 96.0f, 48.0f, 64.0f, {0.9f, 0.1f});

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};
    auto status = node->Forward(bottoms, tops);
    REQUIRE(status == cosmo::nn::COSMO_NN_OK);

    const auto* decoded = static_cast<const float*>(tops[0]->GetHandle().base);
    CHECK(decoded[0 * 6 + 4] == Approx(0.9f).epsilon(1e-3f));
    CHECK(decoded[0 * 6 + 5] == 0.0f);
    CHECK(decoded[0 * 6 + 0] == 88.0f);
    CHECK(decoded[0 * 6 + 3] == 64.0f);
}

TEST_CASE("YOLO26 ultralytics rejects malformed contracts", "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_ultralytics_postprocess"), 0, 1,
        cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);
    auto op = MakeOp(0.25f, 0.45f, 300);
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    // Rank-2 output is not [1, 4+nc, N].
    {
        auto bottom = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 600});
        std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
        std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};
        auto status = node->Forward(bottoms, tops);
        CHECK(status != cosmo::nn::COSMO_NN_OK);
    }
    // Fewer than five channels (4 box + 1 class).
    {
        auto bottom = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 4, 100});
        std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
        std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};
        auto status = node->Forward(bottoms, tops);
        CHECK(status != cosmo::nn::COSMO_NN_OK);
    }
    // INT8 output is not an ultralytics FP32/FP16 export.
    {
        auto bottom = MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 6, 100});
        std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
        std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};
        auto status = node->Forward(bottoms, tops);
        CHECK(status != cosmo::nn::COSMO_NN_OK);
    }
    // Multiple outputs are not the single-tensor ultralytics contract.
    {
        auto bottom = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 6, 100});
        std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom, bottom};
        std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};
        auto status = node->Forward(bottoms, tops);
        CHECK(status != cosmo::nn::COSMO_NN_OK);
    }
    // Class scores outside [0, 1] are not sigmoid probabilities: the buffer
    // is not channel-major [1, 4+nc, N] (e.g. a cell-major tensor misread).
    {
        auto bottom = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 6, 100});
        std::fill(static_cast<float*>(bottom->GetHandle().base),
                  static_cast<float*>(bottom->GetHandle().base) + 600, 0.0f);
        SetCell<float>(bottom, 0, 88.0f, 96.0f, 48.0f, 64.0f, {0.9f, 1.5f});
        std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
        std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};
        auto status = node->Forward(bottoms, tops);
        CHECK(status != cosmo::nn::COSMO_NN_OK);
    }
    {
        auto bottom = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 6, 100});
        std::fill(static_cast<float*>(bottom->GetHandle().base),
                  static_cast<float*>(bottom->GetHandle().base) + 600, 0.0f);
        SetCell<float>(bottom, 0, 88.0f, 96.0f, 48.0f, 64.0f, {0.9f, -0.5f});
        std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{bottom};
        std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};
        auto status = node->Forward(bottoms, tops);
        CHECK(status != cosmo::nn::COSMO_NN_OK);
    }
}

#endif  // COSMO_NN_USE_CPU_BACKEND
