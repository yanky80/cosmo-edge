#include "nn/device/rknn/rknn_node_creator.h"

#include <memory>

#include "nn/device/rknn/rknn_image_to_tensor_node.h"
#include "nn/device/rknn/rknn_net_node.h"

namespace cosmo::nn {

RknnNodeCreator::RknnNodeCreator(DeviceType device_type) : NodeCreator(device_type) {}

RknnNodeCreator::~RknnNodeCreator() {}

std::unique_ptr<Node> RknnNodeCreator::CreateNode(NodeType type) {
    switch (type) {
        case NODE_IMAGE_TO_TENSOR:
            return std::make_unique<RknnImageToTensorNode>();
        case NODE_NET:
            return std::make_unique<RknnNetNode>();
        default:
            return nullptr;
    }
}

NodeCreatorRegister<RknnNodeCreator> g_rknn_node_creator_register(DEVICE_RKNN);

}  // namespace cosmo::nn
