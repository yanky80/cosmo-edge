#include "catch_amalgamated.hpp"

#ifdef COSMO_NN_USE_ASCEND_BACKEND

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "nn/core/common.h"
#include "nn/device/ascend/ascend_image_to_tensor_node.h"
#include "nn/node/node_type_utils.h"
#include "nn/pipeline/pipeline_utils.h"
#include "nn/utils/data_type_utils.h"
#include "nn/utils/op.h"

using namespace cosmo::nn;

TEST_CASE("ascend image_to_tensor maps to the Ascend node type", "[nn][ascend][image_to_tensor]") {
    auto node = NodeTypeUtils::CreateByType(NODE_IMAGE_TO_TENSOR, DEVICE_ASCEND);
    REQUIRE(node != nullptr);
    CHECK(node->GetNodeType() == NODE_IMAGE_TO_TENSOR);
    // Host-memory output so the host-memory AscendNetNode needs no copy node.
    CHECK(node->GetTopBlobDeviceType() == DEVICE_NAIVE);
}

TEST_CASE("ascend image_to_tensor op carries input size and padding color", "[nn][ascend][image_to_tensor]") {
    auto op = pipeline_utils::MakeImageToTensorOp(960, 960, {114, 114, 114});
    REQUIRE(op != nullptr);
    CHECK(op->input_width == 960);
    CHECK(op->input_height == 960);
    CHECK(op->padding_color.size() == 3);

    AscendImageToTensorNode node;
    node.LoadParam(op.get());
    REQUIRE(bool(node.InferTopShapes()));
    CHECK(node.GetTopBlobShapes().size() == 1);
    CHECK(node.GetTopBlobShapes()[0] == DimsVector({1, 3, 960, 960}));
    CHECK(node.GetTopBlobDataTypes().size() == 1);
    CHECK(node.GetTopBlobDataTypes()[0] == DATA_TYPE_HALF);
    CHECK(node.GetBottomCount() == 1);
    CHECK(node.GetTopCount() == 1);
}

TEST_CASE("ascend image_to_tensor letterbox geometry is centered and even", "[nn][ascend][image_to_tensor]") {
    auto check = [](int fw, int fh, int tw, int th, int exp_w, int exp_h, int exp_px, int exp_py) {
        int fit_w = 0;
        int fit_h = 0;
        int pad_x = 0;
        int pad_y = 0;
        REQUIRE(AscendImageToTensorNode::ComputeLetterbox(fw, fh, tw, th, fit_w, fit_h, pad_x, pad_y));
        CHECK(fit_w == exp_w);
        CHECK(fit_h == exp_h);
        CHECK(pad_x == exp_px);
        CHECK(pad_y == exp_py);
        CHECK(fit_w % 2 == 0);
        CHECK(fit_h % 2 == 0);
        CHECK(pad_x >= 0);
        CHECK(pad_y >= 0);
        CHECK(pad_x + fit_w <= tw);
        CHECK(pad_y + fit_h <= th);
    };
    // Landscape 16:9 -> square 960x960: fit 960x540, pad top/bottom.
    check(1280, 720, 960, 960, 960, 540, 0, 210);
    // Portrait -> square: fit 540x960, pad left/right.
    check(720, 1280, 960, 960, 540, 960, 210, 0);
    // Already matching: no pad, no resize.
    check(960, 960, 960, 960, 960, 960, 0, 0);
    // 4:3 source into square.
    check(640, 480, 960, 960, 960, 720, 0, 120);
    // Smaller than the target: passthrough size, centered pad.
    check(320, 240, 960, 960, 320, 240, 320, 360);
    // Invalid geometry is rejected.
    int fit_w = 0;
    int fit_h = 0;
    int pad_x = 0;
    int pad_y = 0;
    CHECK_FALSE(AscendImageToTensorNode::ComputeLetterbox(0, 240, 960, 960, fit_w, fit_h, pad_x, pad_y));
    CHECK_FALSE(AscendImageToTensorNode::ComputeLetterbox(320, 240, 0, 960, fit_w, fit_h, pad_x, pad_y));
}

TEST_CASE("FloatToFp16 matches IEEE half on edge cases", "[nn][ascend][image_to_tensor]") {
    auto roundtrip = [](float value) { return Fp16ToFloat(FloatToFp16(value)); };
    CHECK(FloatToFp16(0.0F) == 0x0000);
    CHECK(FloatToFp16(-0.0F) == 0x8000);
    CHECK(FloatToFp16(1.0F) == 0x3C00);
    CHECK(FloatToFp16(-1.0F) == 0xBC00);
    CHECK(FloatToFp16(0.5F) == 0x3800);
    CHECK(FloatToFp16(65504.0F) == 0x7BFF);  // largest finite half
    CHECK(FloatToFp16(65520.0F) == 0x7C00);  // overflows to +inf
    CHECK(FloatToFp16(1e-7F) == 0x0002);     // subnormal, round to nearest even
    CHECK(roundtrip(0.1F) > 0.099F);
    CHECK(roundtrip(0.1F) < 0.101F);
}

TEST_CASE("ascend image_to_tensor RGB normalize produces FP16 NCHW 0..1", "[nn][ascend][image_to_tensor]") {
    // 2x2 packed RGB888, stride == width * 3.
    const uint8_t rgb[12] = {
        255, 0,   0,    // red
        0,   255, 0,    // green
        0,   0,   255,  // blue
        114, 114, 114,  // gray padding color
    };
    std::vector<uint16_t> dst(3 * 2 * 2, 0);
    AscendImageToTensorNode::NormalizeRgbToFp16Nchw(rgb, 6, 2, 2, dst.data());

    auto at   = [&](int c, int y, int x) { return dst[static_cast<size_t>((c * 2 + y) * 2 + x)]; };
    auto near = [](uint16_t half, float expected) {
        const float value = Fp16ToFloat(half);
        return std::abs(value - expected) < 1e-3F;
    };
    CHECK(near(at(0, 0, 0), 1.0F));             // R of red pixel
    CHECK(near(at(1, 0, 0), 0.0F));             // G of red pixel
    CHECK(near(at(2, 0, 0), 0.0F));             // B of red pixel
    CHECK(near(at(0, 0, 1), 0.0F));             // R of green pixel
    CHECK(near(at(1, 0, 1), 1.0F));             // G of green pixel
    CHECK(near(at(0, 1, 0), 0.0F));             // R of blue pixel
    CHECK(near(at(2, 1, 0), 1.0F));             // B of blue pixel
    CHECK(near(at(0, 1, 1), 114.0F / 255.0F));  // gray R
    CHECK(near(at(1, 1, 1), 114.0F / 255.0F));  // gray G
    CHECK(near(at(2, 1, 1), 114.0F / 255.0F));  // gray B
}

#endif  // COSMO_NN_USE_ASCEND_BACKEND
