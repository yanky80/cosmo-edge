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
#include "nn/utils/op.h"

namespace {

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

#endif  // COSMO_NN_USE_CPU_BACKEND
