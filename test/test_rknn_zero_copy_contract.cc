#include "catch_amalgamated.hpp"

#include <memory>
#include <string>

#include "nn/core/common.h"
#include "nn/node/node_type_utils.h"
#include "nn/pipeline/pipeline_utils.h"
#include "nn/utils/op.h"

TEST_CASE("image_to_tensor maps to the RKNN node type", "[nn][rknn][image_to_tensor]") {
    REQUIRE(cosmo::nn::NodeTypeUtils::NodeTypeFromStr("image_to_tensor") ==
            cosmo::nn::NODE_IMAGE_TO_TENSOR);
    REQUIRE(cosmo::nn::NodeTypeUtils::NodeTypeToStr(cosmo::nn::NODE_IMAGE_TO_TENSOR) ==
            "image_to_tensor");
}

TEST_CASE("image_to_tensor op carries input size and padding color", "[nn][rknn][image_to_tensor]") {
    auto op = cosmo::nn::pipeline_utils::MakeImageToTensorOp(640, 640, {114, 114, 114});
    REQUIRE(op != nullptr);
    CHECK(op->name == "image_to_tensor");
    CHECK(op->input_width == 640);
    CHECK(op->input_height == 640);
    CHECK(op->padding_color.size() == 3);
    CHECK(op->padding_color[0] == 114);
    REQUIRE_FALSE(op->Description().empty());
}

TEST_CASE("RKNN device type is a non-host backend", "[nn][rknn][device]") {
    CHECK_FALSE(cosmo::nn::UsesHostMemory(cosmo::nn::DEVICE_RKNN));
    // RknnImageToTensorNode -> RknnNetNode: same device, no copy node.
    CHECK_FALSE(cosmo::nn::NeedsCopyNode(cosmo::nn::DEVICE_RKNN, cosmo::nn::DEVICE_RKNN));
    // Host producer -> RKNN consumer still needs a copy node.
    CHECK(cosmo::nn::NeedsCopyNode(cosmo::nn::DEVICE_NAIVE, cosmo::nn::DEVICE_RKNN));
}
