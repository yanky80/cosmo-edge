#pragma once

#include "nn/node/node_creator.h"

namespace cosmo::nn {

class AscendNodeCreator : public NodeCreator {
public:
    explicit AscendNodeCreator(DeviceType device_type);
    ~AscendNodeCreator() override;

    std::unique_ptr<Node> CreateNode(NodeType type) override;
};

}  // namespace cosmo::nn
