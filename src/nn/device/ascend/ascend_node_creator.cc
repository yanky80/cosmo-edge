#include "nn/device/ascend/ascend_node_creator.h"

#include <memory>

#include "nn/device/ascend/ascend_net_node.h"

namespace cosmo::nn {

AscendNodeCreator::AscendNodeCreator(DeviceType device_type) : NodeCreator(device_type) {}

AscendNodeCreator::~AscendNodeCreator() {}

std::unique_ptr<Node> AscendNodeCreator::CreateNode(NodeType type) {
    switch (type) {
        case NODE_NET:
            return std::make_unique<AscendNetNode>();
        default:
            return nullptr;
    }
}

NodeCreatorRegister<AscendNodeCreator> g_ascend_node_creator_register(DEVICE_ASCEND);

}  // namespace cosmo::nn
