#pragma once

#include "nn/node/node_creator.h"

namespace cosmo::nn {

class RknnNodeCreator : public NodeCreator {
public:
    explicit RknnNodeCreator(DeviceType device_type);
    ~RknnNodeCreator() override;

    std::unique_ptr<Node> CreateNode(NodeType type) override;
};

}  // namespace cosmo::nn
