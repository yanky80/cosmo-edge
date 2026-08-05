#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "nn/node/net_node.h"

namespace cosmo::nn {

// AscendCL net node. One instance owns exactly one ACL context, stream, model
// (OM), input/output datasets and their device buffers; inference reuses the
// preallocated buffers. Host blobs are consumed/produced (the node copies
// H2D, executes on its stream, synchronizes, then copies D2H), so the graph
// treats Ascend as a host-memory backend and inserts no copy nodes. aclInit
// runs once per process; aclFinalize runs at process exit.
class AscendNetNode : public NetNode {
public:
    AscendNetNode();
    ~AscendNetNode() override;

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
    Status AllocateIo();
    void Destroy();

    // Errors carry stage, device id, model path, tensor detail and acl error.
    Status MakeAclStatus(int code, const std::string& stage, const std::string& detail,
                         aclError acl_ret) const;
    std::string TensorDescription(const std::string& kind, size_t index, const aclmdlIODims& dims,
                                  aclDataType dtype, aclFormat format, size_t bytes) const;

    int device_id_                 = 0;
    bool device_set_               = false;
    aclrtContext context_          = nullptr;
    aclrtStream stream_            = nullptr;
    uint32_t model_id_             = 0;
    aclmdlDesc* desc_              = nullptr;
    aclmdlDataset* input_dataset_  = nullptr;
    aclmdlDataset* output_dataset_ = nullptr;

    std::vector<aclDataBuffer*> input_buffers_;
    std::vector<aclDataBuffer*> output_buffers_;
    std::vector<void*> input_devices_;
    std::vector<void*> output_devices_;
    std::vector<size_t> input_sizes_;
    std::vector<size_t> output_sizes_;
    std::vector<aclDataType> input_types_;
    std::vector<aclDataType> output_types_;
    std::vector<DimsVector> output_shapes_;
    std::vector<uint8_t> output_scratch_;
};

}  // namespace cosmo::nn
