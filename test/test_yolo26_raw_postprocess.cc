#include "catch_amalgamated.hpp"

#ifdef COSMO_NN_USE_CPU_BACKEND

#include <array>
#include <cstdint>
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

}  // namespace

TEST_CASE("YOLO26 keeps E2E postprocess by default", "[nn][yolo26]") {
    auto op = cosmo::nn::pipeline_utils::MakeYoloE2EPostOp(0.25f, 300, 640, 640);
    REQUIRE(op != nullptr);
    CHECK(op->name == "yolo_e2e_postprocess");
}

TEST_CASE("YOLO26 raw output_format selects host raw-head postprocess", "[nn][yolo26]") {
    auto op = cosmo::nn::pipeline_utils::MakeYolo26RawPostOp(0.45f, 0.25f, 300, 1, 640, 640,
                                                             {0.5f, 0.1f, 0.5f, 0.1f, 0.5f, 0.1f},
                                                             {0, 0, 0, 0, 0, 0});
    REQUIRE(op != nullptr);
    CHECK(op->name == "yolo26_raw_postprocess");
    CHECK(op->reg_max == 1);
    CHECK(op->output_scales.size() == 6);
    CHECK(op->output_zero_points.size() == 6);
}

TEST_CASE("YOLO26 raw decode validates six INT8 affine heads and decodes surviving boxes",
          "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_raw_postprocess"), 0, 1, cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);

    auto op                = std::make_unique<cosmo::nn::Yolo26RawPost>("yolo26_raw_postprocess");
    op->input_width        = 640;
    op->input_height       = 640;
    op->nms_detection_conf = 0.25f;
    op->nms_threshold      = 0.45f;
    op->top_k              = 300;
    op->reg_max            = 1;
    op->output_scales      = {0.5f, 0.1f, 0.5f, 0.1f, 0.5f, 0.1f};
    op->output_zero_points = {0, 0, 0, 0, 0, 0};
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    auto reg0 = MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 4, 80, 80});
    auto cls0 = MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 1, 80, 80});
    auto reg1 = MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 4, 40, 40});
    auto cls1 = MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 1, 40, 40});
    auto reg2 = MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 4, 20, 20});
    auto cls2 = MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 1, 20, 20});

    auto* reg0_data = static_cast<std::int8_t*>(reg0->GetHandle().base);
    auto* cls0_data = static_cast<std::int8_t*>(cls0->GetHandle().base);
    std::fill(reg0_data, reg0_data + 4 * 80 * 80, 0);
    std::fill(cls0_data, cls0_data + 80 * 80, static_cast<std::int8_t>(-128));
    std::fill(static_cast<std::int8_t*>(reg1->GetHandle().base),
              static_cast<std::int8_t*>(reg1->GetHandle().base) + 4 * 40 * 40, 0);
    std::fill(static_cast<std::int8_t*>(cls1->GetHandle().base),
              static_cast<std::int8_t*>(cls1->GetHandle().base) + 40 * 40, static_cast<std::int8_t>(-128));
    std::fill(static_cast<std::int8_t*>(reg2->GetHandle().base),
              static_cast<std::int8_t*>(reg2->GetHandle().base) + 4 * 20 * 20, 0);
    std::fill(static_cast<std::int8_t*>(cls2->GetHandle().base),
              static_cast<std::int8_t*>(cls2->GetHandle().base) + 20 * 20, static_cast<std::int8_t>(-128));
    reg0_data[0] = 10;
    reg0_data[1] = 10;
    reg0_data[2] = 10;
    reg0_data[3] = 10;
    cls0_data[0] = 127;

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{reg0, cls0, reg1, cls1, reg2, cls2};
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};

    auto status = node->Forward(bottoms, tops);
    REQUIRE(status == cosmo::nn::COSMO_NN_OK);

    const auto* decoded = static_cast<const float*>(tops[0]->GetHandle().base);
    CHECK(decoded[4] > 0.25f);
}

TEST_CASE("YOLO26 raw decode rejects malformed contracts", "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_raw_postprocess"), 0, 1, cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);

    auto op                = std::make_unique<cosmo::nn::Yolo26RawPost>("yolo26_raw_postprocess");
    op->input_width        = 640;
    op->input_height       = 640;
    op->reg_max            = 1;
    op->output_scales      = {0.5f, 0.1f, 0.5f, 0.1f, 0.5f, 0.1f};
    op->output_zero_points = {0, 0, 0, 0, 0, 0};
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bad_bottoms{
        MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 4, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 1, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 4, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 1, 40, 39}),
    };
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};

    auto status = node->Forward(bad_bottoms, tops);
    CHECK(status != cosmo::nn::COSMO_NN_OK);
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

std::shared_ptr<cosmo::nn::Blob> MakeHalfBlob(const cosmo::nn::DimsVector& dims) {
    cosmo::nn::BlobDesc desc;
    desc.data_type   = cosmo::nn::DATA_TYPE_HALF;
    desc.device_type = cosmo::nn::DEVICE_NAIVE;
    desc.data_format = cosmo::nn::DATA_FORMAT_NCHW;
    desc.dims        = dims;
    return std::make_shared<cosmo::nn::Blob>(desc, true);
}

void FillHalf(std::shared_ptr<cosmo::nn::Blob> blob, float value) {
    auto* data = static_cast<std::uint16_t*>(blob->GetHandle().base);
    std::fill(data, data + cosmo::nn::DimsVectorUtils::Count(blob->GetBlobDesc().dims), FloatToHalf(value));
}

void SetHalf(std::shared_ptr<cosmo::nn::Blob> blob, int channel, int y, int x, float value) {
    const auto dims                                     = blob->GetBlobDesc().dims;
    auto* data                                          = static_cast<std::uint16_t*>(blob->GetHandle().base);
    data[channel * dims[2] * dims[3] + y * dims[3] + x] = FloatToHalf(value);
}

TEST_CASE("YOLO26 FP16 golden: thresholds, three scales, NMS, top-k, letterbox recovery",
          "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_raw_postprocess"), 0, 1, cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);

    auto op                = std::make_unique<cosmo::nn::Yolo26RawPost>("yolo26_raw_postprocess");
    op->input_width        = 640;
    op->input_height       = 640;
    op->nms_detection_conf = 0.25f;
    op->nms_threshold      = 0.45f;
    op->top_k              = 300;
    op->reg_max            = 1;
    // FP16 heads carry raw logits/distances; no scale or zero-point metadata.
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    const int kClasses = 2;
    auto reg0          = MakeHalfBlob({1, 4, 80, 80});
    auto cls0          = MakeHalfBlob({1, kClasses, 80, 80});
    auto reg1          = MakeHalfBlob({1, 4, 40, 40});
    auto cls1          = MakeHalfBlob({1, kClasses, 40, 40});
    auto reg2          = MakeHalfBlob({1, 4, 20, 20});
    auto cls2          = MakeHalfBlob({1, kClasses, 20, 20});

    // Everything else stays below the confidence threshold (logit -2.0) with
    // zero distances, so only the crafted cells produce candidates.
    FillHalf(reg0, 0.0f);
    FillHalf(cls0, -2.0f);
    FillHalf(reg1, 0.0f);
    FillHalf(cls1, -2.0f);
    FillHalf(reg2, 0.0f);
    FillHalf(cls2, -2.0f);

    // Scale 0 (stride 8): cell (10,10) -> class 0, logit 0.0, score 0.5,
    // box (cx=88, cy=88, w=40, h=24). Cell (11,10) -> class 0, logit -0.5,
    // score 0.378, box (cx=96, cy=88, w=40, h=24); it overlaps cell (10,10)
    // with IoU 2/3 and must be suppressed by class-aware NMS.
    const float reg_a[4] = {2, 1, 3, 2};
    for (int channel = 0; channel < 4; ++channel)
        SetHalf(reg0, channel, 10, 10, reg_a[channel]);
    SetHalf(cls0, 0, 10, 10, 0.0f);
    for (int channel = 0; channel < 4; ++channel)
        SetHalf(reg0, channel, 11, 10, reg_a[channel]);
    SetHalf(cls0, 0, 11, 10, -0.5f);

    // Scale 1 (stride 16): cell (5,5) -> class 0, logit 1.0, score 0.731,
    // box (cx=84, cy=84, w=40, h=72).
    const float reg_b[4] = {1.5f, 2.5f, 1, 2};
    for (int channel = 0; channel < 4; ++channel)
        SetHalf(reg1, channel, 5, 5, reg_b[channel]);
    SetHalf(cls1, 0, 5, 5, 1.0f);

    // Scale 2 (stride 32): cell (2,2) -> class 1, logit 2.0, score 0.881,
    // box (cx=88, cy=96, w=48, h=64).
    const float reg_c[4] = {0.5f, 0.5f, 1, 1.5f};
    for (int channel = 0; channel < 4; ++channel)
        SetHalf(reg2, channel, 2, 2, reg_c[channel]);
    SetHalf(cls2, 1, 2, 2, 2.0f);

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{reg0, cls0, reg1, cls1, reg2, cls2};
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};

    auto status = node->Forward(bottoms, tops);
    REQUIRE(status == cosmo::nn::COSMO_NN_OK);

    const auto* decoded = static_cast<const float*>(tops[0]->GetHandle().base);

    // Candidates sort by descending score: class 1 (0.881), class 0 (0.731),
    // class 0 (0.5). The overlapping 0.378 candidate is NMS-suppressed, and
    // the -2.0 logit cell is below the FP16 threshold.
    CHECK(decoded[0 * 6 + 4] == Approx(0.880797f).epsilon(1e-5f));
    CHECK(decoded[0 * 6 + 5] == 1.0f);
    CHECK(decoded[0 * 6 + 0] == 88.0f);
    CHECK(decoded[0 * 6 + 1] == 96.0f);
    CHECK(decoded[0 * 6 + 2] == 48.0f);
    CHECK(decoded[0 * 6 + 3] == 64.0f);

    CHECK(decoded[1 * 6 + 4] == Approx(0.731059f).epsilon(1e-5f));
    CHECK(decoded[1 * 6 + 5] == 0.0f);
    CHECK(decoded[1 * 6 + 0] == 84.0f);
    CHECK(decoded[1 * 6 + 1] == 84.0f);
    CHECK(decoded[1 * 6 + 2] == 40.0f);
    CHECK(decoded[1 * 6 + 3] == 72.0f);

    CHECK(decoded[2 * 6 + 4] == 0.5f);
    CHECK(decoded[2 * 6 + 5] == 0.0f);
    CHECK(decoded[2 * 6 + 0] == 88.0f);
    CHECK(decoded[2 * 6 + 1] == 88.0f);
    CHECK(decoded[2 * 6 + 2] == 40.0f);
    CHECK(decoded[2 * 6 + 3] == 24.0f);

    // Exactly three survivors: every remaining row is zero-filled.
    CHECK(decoded[3 * 6 + 4] == 0.0f);
    CHECK(decoded[299 * 6 + 4] == 0.0f);

    // top-k truncates the sorted survivors.
    op->top_k = 2;
    node->LoadParam(op.get());
    status = node->Forward(bottoms, tops);
    REQUIRE(status == cosmo::nn::COSMO_NN_OK);
    CHECK(decoded[0 * 6 + 4] == Approx(0.880797f).epsilon(1e-5f));
    CHECK(decoded[1 * 6 + 4] == Approx(0.731059f).epsilon(1e-5f));
    CHECK(decoded[2 * 6 + 4] == 0.0f);

    // Letterbox coordinate recovery: 640x640 net input back to a 1280x720
    // source via the pipeline's gravity-2 AdjustSize (scale by 2, subtract
    // the 280px vertical padding).
    std::vector<std::vector<cosmo::nn::ObjectInfoV1>> recovered;
    op->top_k = 300;
    node->LoadParam(op.get());
    auto forward_status = node->Forward(bottoms, tops);
    REQUIRE(forward_status == cosmo::nn::COSMO_NN_OK);
    auto pick_status = cosmo::nn::NetUtils::PickDetectionObjects(
        tops[0], {cosmo::nn::Size(1280, 720)}, cosmo::nn::Size(640, 640), {}, {}, {}, recovered);
    REQUIRE(pick_status == cosmo::nn::COSMO_NN_OK);
    REQUIRE(recovered.size() == 1);
    REQUIRE(recovered[0].size() == 3);

    CHECK(recovered[0][0].x1 == 128.0f);
    CHECK(recovered[0][0].y1 == -152.0f);
    CHECK(recovered[0][0].x2 == 224.0f);
    CHECK(recovered[0][0].y2 == -24.0f);
    CHECK(recovered[0][0].infos[0].confidence == Approx(0.880797f).epsilon(1e-5f));
    CHECK(recovered[0][0].infos[0].class_id == 1);

    CHECK(recovered[0][1].x1 == 128.0f);
    CHECK(recovered[0][1].y1 == -184.0f);
    CHECK(recovered[0][1].x2 == 208.0f);
    CHECK(recovered[0][1].y2 == -40.0f);

    CHECK(recovered[0][2].x1 == 136.0f);
    CHECK(recovered[0][2].y1 == -128.0f);
    CHECK(recovered[0][2].x2 == 216.0f);
    CHECK(recovered[0][2].y2 == -80.0f);
}

TEST_CASE("YOLO26 FP32 raw heads decode without quant metadata", "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_raw_postprocess"), 0, 1, cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);

    auto op                = std::make_unique<cosmo::nn::Yolo26RawPost>("yolo26_raw_postprocess");
    op->input_width        = 640;
    op->input_height       = 640;
    op->nms_detection_conf = 0.25f;
    op->nms_threshold      = 0.45f;
    op->top_k              = 300;
    op->reg_max            = 1;
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    auto reg0 = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 4, 80, 80});
    auto cls0 = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 1, 80, 80});
    auto reg1 = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 4, 40, 40});
    auto cls1 = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 1, 40, 40});
    auto reg2 = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 4, 20, 20});
    auto cls2 = MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 1, 20, 20});

    auto* reg0_data = static_cast<float*>(reg0->GetHandle().base);
    auto* cls0_data = static_cast<float*>(cls0->GetHandle().base);
    std::fill(reg0_data, reg0_data + 4 * 80 * 80, 0.0f);
    std::fill(cls0_data, cls0_data + 80 * 80, -2.0f);
    const float reg_a[4] = {2, 1, 3, 2};
    for (int channel = 0; channel < 4; ++channel)
        reg0_data[channel * 80 * 80 + 10 * 80 + 10] = reg_a[channel];
    cls0_data[10 * 80 + 10] = 0.0f;
    std::fill(static_cast<float*>(reg1->GetHandle().base),
              static_cast<float*>(reg1->GetHandle().base) + 4 * 40 * 40, 0.0f);
    std::fill(static_cast<float*>(cls1->GetHandle().base),
              static_cast<float*>(cls1->GetHandle().base) + 40 * 40, -2.0f);
    std::fill(static_cast<float*>(reg2->GetHandle().base),
              static_cast<float*>(reg2->GetHandle().base) + 4 * 20 * 20, 0.0f);
    std::fill(static_cast<float*>(cls2->GetHandle().base),
              static_cast<float*>(cls2->GetHandle().base) + 20 * 20, -2.0f);

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{reg0, cls0, reg1, cls1, reg2, cls2};
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};

    auto status = node->Forward(bottoms, tops);
    REQUIRE(status == cosmo::nn::COSMO_NN_OK);

    const auto* decoded = static_cast<const float*>(tops[0]->GetHandle().base);
    CHECK(decoded[4] == 0.5f);
    CHECK(decoded[0] == 88.0f);
    CHECK(decoded[1] == 88.0f);
    CHECK(decoded[2] == 40.0f);
    CHECK(decoded[3] == 24.0f);
    CHECK(decoded[5] == 0.0f);
    CHECK(decoded[6 + 4] == 0.0f);
}

TEST_CASE("YOLO26 raw decode rejects mixed, unsupported, and inconsistent heads",
          "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_raw_postprocess"), 0, 1, cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);

    auto op                = std::make_unique<cosmo::nn::Yolo26RawPost>("yolo26_raw_postprocess");
    op->input_width        = 640;
    op->input_height       = 640;
    op->reg_max            = 1;
    op->output_scales      = {0.5f, 0.1f, 0.5f, 0.1f, 0.5f, 0.1f};
    op->output_zero_points = {0, 0, 0, 0, 0, 0};
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};

    // Mixed dtypes across the six heads.
    std::vector<std::shared_ptr<cosmo::nn::Blob>> mixed{
        MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 4, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_INT8, {1, 1, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 20, 20}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 20, 20}),
    };
    auto mixed_status = node->Forward(mixed, tops);
    REQUIRE(mixed_status != cosmo::nn::COSMO_NN_OK);
    CHECK(mixed_status.description().find("dtype") != std::string::npos);

    // Unsupported head dtype (BFP16 is not an accepted raw-head format).
    std::vector<std::shared_ptr<cosmo::nn::Blob>> unsupported{
        MakeBlob(cosmo::nn::DATA_TYPE_BFP16, {1, 4, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_BFP16, {1, 1, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_BFP16, {1, 4, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_BFP16, {1, 1, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_BFP16, {1, 4, 20, 20}),
        MakeBlob(cosmo::nn::DATA_TYPE_BFP16, {1, 1, 20, 20}),
    };
    auto unsupported_status = node->Forward(unsupported, tops);
    REQUIRE(unsupported_status != cosmo::nn::COSMO_NN_OK);

    // Inconsistent class channels across scales.
    std::vector<std::shared_ptr<cosmo::nn::Blob>> class_mismatch{
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 2, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 20, 20}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 2, 20, 20}),
    };
    auto class_status = node->Forward(class_mismatch, tops);
    REQUIRE(class_status != cosmo::nn::COSMO_NN_OK);
    CHECK(class_status.description().find("class channel") != std::string::npos);

    // Incompatible shapes: reg/cls feature maps must match.
    std::vector<std::shared_ptr<cosmo::nn::Blob>> shape_mismatch{
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 80, 79}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 20, 20}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 20, 20}),
    };
    auto shape_status = node->Forward(shape_mismatch, tops);
    REQUIRE(shape_status != cosmo::nn::COSMO_NN_OK);

    // Wrong head count.
    std::vector<std::shared_ptr<cosmo::nn::Blob>> wrong_count{
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 80, 80}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 40, 40}),
    };
    auto count_status = node->Forward(wrong_count, tops);
    REQUIRE(count_status != cosmo::nn::COSMO_NN_OK);
}

TEST_CASE("YOLO26 raw decode rejects non-NCHW heads", "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_raw_postprocess"), 0, 1, cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);

    auto op                = std::make_unique<cosmo::nn::Yolo26RawPost>("yolo26_raw_postprocess");
    op->input_width        = 640;
    op->input_height       = 640;
    op->nms_detection_conf = 0.25f;
    op->nms_threshold      = 0.45f;
    op->top_k              = 300;
    op->reg_max            = 1;
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    auto cls0 = MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 80, 80});
    cls0->GetBlobDesc().data_format = cosmo::nn::DATA_FORMAT_NHWC;

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 80, 80}),
        cls0,
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 40, 40}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 4, 20, 20}),
        MakeBlob(cosmo::nn::DATA_TYPE_HALF, {1, 1, 20, 20}),
    };
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};

    auto status = node->Forward(bottoms, tops);
    REQUIRE(status != cosmo::nn::COSMO_NN_OK);
}

TEST_CASE("YOLO26 raw decode skips non-finite FP16 logits and distances", "[nn][yolo26][postprocess]") {
    auto node = cosmo::nn::NodeTypeUtils::CreateNode(
        cosmo::nn::NodeTypeUtils::NodeTypeFromStr("yolo26_raw_postprocess"), 0, 1, cosmo::nn::DEVICE_CPU);
    REQUIRE(node != nullptr);

    auto op                = std::make_unique<cosmo::nn::Yolo26RawPost>("yolo26_raw_postprocess");
    op->input_width        = 640;
    op->input_height       = 640;
    op->nms_detection_conf = 0.25f;
    op->nms_threshold      = 0.45f;
    op->top_k              = 300;
    op->reg_max            = 1;
    node->LoadParam(op.get());
    auto infer_status = node->InferTopShapes();
    REQUIRE(infer_status == cosmo::nn::COSMO_NN_OK);

    auto reg0 = MakeHalfBlob({1, 4, 80, 80});
    auto cls0 = MakeHalfBlob({1, 1, 80, 80});
    auto reg1 = MakeHalfBlob({1, 4, 40, 40});
    auto cls1 = MakeHalfBlob({1, 1, 40, 40});
    auto reg2 = MakeHalfBlob({1, 4, 20, 20});
    auto cls2 = MakeHalfBlob({1, 1, 20, 20});
    FillHalf(reg0, 0.0f);
    FillHalf(cls0, -2.0f);
    FillHalf(reg1, 0.0f);
    FillHalf(cls1, -2.0f);
    FillHalf(reg2, 0.0f);
    FillHalf(cls2, -2.0f);

    // Cell (0,0): FP16 quiet-NaN class logit (0x7e00) must be skipped.
    static_cast<std::uint16_t*>(cls0->GetHandle().base)[0] = 0x7e00u;
    // Cell (0,1): +Inf left distance (0x7c00) would produce a NaN box; skipped.
    static_cast<std::uint16_t*>(reg0->GetHandle().base)[1] = 0x7c00u;
    SetHalf(cls0, 0, 0, 1, 2.0f);
    // Cell (1,1): valid candidate survives: cx=12, cy=12, w=16, h=16.
    const float reg_a[4] = {1, 1, 1, 1};
    for (int channel = 0; channel < 4; ++channel)
        SetHalf(reg0, channel, 1, 1, reg_a[channel]);
    SetHalf(cls0, 0, 1, 1, 2.0f);

    std::vector<std::shared_ptr<cosmo::nn::Blob>> bottoms{
        reg0,
        cls0,
        reg1,
        cls1,
        reg2,
        cls2,
    };
    std::vector<std::shared_ptr<cosmo::nn::Blob>> tops{MakeBlob(cosmo::nn::DATA_TYPE_FLOAT, {1, 300, 6})};

    auto status = node->Forward(bottoms, tops);
    REQUIRE(status == cosmo::nn::COSMO_NN_OK);

    const auto* decoded = static_cast<const float*>(tops[0]->GetHandle().base);
    CHECK(decoded[0 * 6 + 4] == Approx(0.880797f).epsilon(1e-5f));
    CHECK(decoded[0 * 6 + 5] == 0.0f);
    CHECK(decoded[0 * 6 + 0] == 12.0f);
    CHECK(decoded[0 * 6 + 1] == 12.0f);
    CHECK(decoded[0 * 6 + 2] == 16.0f);
    CHECK(decoded[0 * 6 + 3] == 16.0f);
    // Only one candidate survives; the remaining top-k rows stay zeroed.
    for (int index = 1; index < 6; ++index) {
        CHECK(decoded[index * 6 + 0] == 0.0f);
        CHECK(decoded[index * 6 + 1] == 0.0f);
        CHECK(decoded[index * 6 + 2] == 0.0f);
        CHECK(decoded[index * 6 + 3] == 0.0f);
        CHECK(decoded[index * 6 + 4] == 0.0f);
        CHECK(decoded[index * 6 + 5] == 0.0f);
    }
}

#endif  // COSMO_NN_USE_CPU_BACKEND
