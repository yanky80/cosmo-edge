#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "nn/device/rknn/rknn_tensor_binding.h"
#include "nn/node/net_node.h"
#include "rknn_api.h"

namespace cosmo::nn {

// RKNN net node. One instance owns exactly one rknn_context and one input
// tensor memory; inference uses rknn_run + rknn_outputs_get and never calls
// rknn_inputs_set. The six INT8 affine outputs are copied into host blobs
// that carry scale/zero-point metadata for the yolo26_raw postprocess node.
class RknnNetNode : public NetNode {
public:
    RknnNetNode();
    ~RknnNetNode() override;

    DeviceType GetInputBlobDeviceType() override;
    DeviceType GetTopBlobDeviceType() override;

    Status InferTopShapes() override;
    Status LoadWeight(const char* data, size_t size) override;
    Status BindInputBlobs(std::vector<std::shared_ptr<Blob>>& bottom_blobs) override;
    void UpdateTopBlobDesc(size_t index, BlobDesc& desc) const override;

    size_t GetBottomCount() override;
    size_t GetTopCount() override;

    Status Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                   std::vector<std::shared_ptr<Blob>>& top_blobs) override;

private:
    Status QueryModel();
    Status CreateInputTensor(std::vector<std::shared_ptr<Blob>>& bottom_blobs);
    void Destroy();

    rknn_context context_ = 0;
    rknn_tensor_attr input_attr_{};
    std::vector<rknn_tensor_attr> output_attrs_;
    rknn_tensor_mem* input_mem_ = nullptr;
    std::unique_ptr<RknnTensorBinding> binding_;
    int width_stride_ = 0;
    int height_stride_ = 0;
};

}  // namespace cosmo::nn
