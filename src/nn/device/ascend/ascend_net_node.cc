#include "nn/device/ascend/ascend_net_node.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "nn/device/ascend/ascend_acl.h"
#include "nn/node/node_type_utils.h"
#include "nn/utils/data_type_utils.h"
#include "nn/utils/dims_vector_utils.h"
#include "nn/utils/string_format.h"

namespace cosmo::nn {

namespace {

    std::string AclDataTypeName(aclDataType type) {
        switch (type) {
            case ACL_FLOAT:
                return "FP32";
            case ACL_FLOAT16:
                return "FP16";
            case ACL_INT8:
                return "INT8";
            case ACL_INT32:
                return "INT32";
            case ACL_UINT8:
                return "UINT8";
            case ACL_BF16:
                return "BF16";
            default:
                return "UNKNOWN";
        }
    }

    std::string AclFormatName(aclFormat format) {
        switch (format) {
            case ACL_FORMAT_NCHW:
                return "NCHW";
            case ACL_FORMAT_NHWC:
                return "NHWC";
            case ACL_FORMAT_ND:
                return "ND";
            default:
                return "UNDEFINED";
        }
    }

}  // namespace

AscendNetNode::AscendNetNode() : NetNode() {
    node_type = NodeType::NODE_NET;
    name      = NodeTypeUtils::NodeTypeToStr(NodeType::NODE_NET).append("_0");
}

AscendNetNode::~AscendNetNode() {
    Destroy();
}

DeviceType AscendNetNode::GetInputBlobDeviceType() {
    // Host-memory input: the node copies the bottom host blob H2D itself.
    return DeviceType::DEVICE_NAIVE;
}

DeviceType AscendNetNode::GetTopBlobDeviceType() {
    // Host-memory output: the node copies D2H into graph-owned host blobs.
    return DeviceType::DEVICE_NAIVE;
}

size_t AscendNetNode::GetBottomCount() {
    return 1;
}

size_t AscendNetNode::GetTopCount() {
    return output_shapes_.empty() ? 1 : output_shapes_.size();
}

Status AscendNetNode::InferTopShapes() {
    top_blob_shapes.clear();
    top_blob_data_types.clear();
    for (const auto& dims : output_shapes_) {
        // FP16 device output is converted to FP32 host blobs for the
        // host-side decode nodes (yolo_e2e_postprocess reads FP32).
        top_blob_shapes.push_back(dims);
        top_blob_data_types.push_back(DATA_TYPE_FLOAT);
    }
    return COSMO_NN_OK;
}

Status AscendNetNode::LoadWeight(const char* data, size_t size) {
    if (model_id_ != 0 || context_ != nullptr)
        return Status(COSMO_NN_ERR_PARAM, "AscendNetNode already loaded");
    if (data == nullptr || size == 0)
        return Status(COSMO_NN_ERR_LOAD_MODEL, "Ascend OM data is empty");

    const aclError init_ret = ascend::EnsureAclInitialized();
    if (init_ret != ACL_SUCCESS)
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_INIT, "aclInit", "model_bytes=" + std::to_string(size),
                             init_ret);

    const aclError dev_ret = ascend::AcquireDevice(device_id_);
    if (dev_ret != ACL_SUCCESS)
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_DEVICE, "aclrtSetDevice", "", dev_ret);
    device_set_ = true;

    aclError ret = aclrtCreateContext(&context_, device_id_);
    if (ret != ACL_SUCCESS) {
        Destroy();
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_CONTEXT, "aclrtCreateContext", "", ret);
    }
    ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_SUCCESS) {
        Destroy();
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_CONTEXT, "aclrtSetCurrentContext", "", ret);
    }
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        Destroy();
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_STREAM, "aclrtCreateStream", "", ret);
    }

    ret = aclmdlLoadFromMem(data, size, &model_id_);
    if (ret != ACL_SUCCESS) {
        Destroy();
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_LOAD, "aclmdlLoadFromMem",
                             "model_bytes=" + std::to_string(size), ret);
    }

    desc_ = aclmdlCreateDesc();
    if (desc_ == nullptr) {
        Destroy();
        return Status(COSMO_NN_ERR_ASCEND_QUERY, "stage=ascend-query device=" + std::to_string(device_id_) +
                                                     " model=" + GetModelPath() + " aclmdlCreateDesc failed");
    }
    ret = aclmdlGetDesc(desc_, model_id_);
    if (ret != ACL_SUCCESS) {
        Destroy();
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_QUERY, "aclmdlGetDesc", "", ret);
    }

    Status status = QueryModel();
    if (!bool(status)) {
        Destroy();
        return status;
    }
    status = AllocateIo();
    if (!bool(status)) {
        Destroy();
        return status;
    }
    return COSMO_NN_OK;
}

Status AscendNetNode::QueryModel() {
    const size_t num_inputs  = aclmdlGetNumInputs(desc_);
    const size_t num_outputs = aclmdlGetNumOutputs(desc_);
    if (num_inputs != 1) {
        return Status(COSMO_NN_ERR_ASCEND_QUERY, "Ascend model contract mismatch model=" + GetModelPath() +
                                                     " inputs=" + std::to_string(num_inputs) +
                                                     " expected 1 input");
    }
    if (num_outputs == 0) {
        return Status(COSMO_NN_ERR_ASCEND_QUERY, "Ascend model has no outputs model=" + GetModelPath());
    }
    if (num_outputs != 1) {
        return Status(COSMO_NN_ERR_ASCEND_QUERY, "Ascend model contract mismatch model=" + GetModelPath() +
                                                     " outputs=" + std::to_string(num_outputs) +
                                                     " expected 1 output");
    }

    input_sizes_.clear();
    input_types_.clear();
    for (size_t i = 0; i < num_inputs; ++i) {
        aclmdlIODims io_dims{};
        const aclError ret = aclmdlGetInputDims(desc_, i, &io_dims);
        if (ret != ACL_SUCCESS)
            return MakeAclStatus(COSMO_NN_ERR_ASCEND_QUERY, "aclmdlGetInputDims",
                                 "tensor=input[" + std::to_string(i) + "]", ret);
        const size_t bytes = aclmdlGetInputSizeByIndex(desc_, i);
        if (io_dims.dimCount != 4 || io_dims.dims[0] != 1 || io_dims.dims[1] <= 0 || io_dims.dims[2] <= 0 ||
            io_dims.dims[3] <= 0 || bytes == 0) {
            return Status(COSMO_NN_ERR_ASCEND_QUERY,
                          "Ascend input must be a fixed-batch 4D tensor model=" + GetModelPath() + " " +
                              TensorDescription("input", i, io_dims, aclmdlGetInputDataType(desc_, i),
                                                aclmdlGetInputFormat(desc_, i), bytes));
        }
        if (aclmdlGetInputFormat(desc_, i) != ACL_FORMAT_NCHW) {
            return Status(COSMO_NN_ERR_ASCEND_QUERY,
                          "Ascend input must be NCHW model=" + GetModelPath() + " " +
                              TensorDescription("input", i, io_dims, aclmdlGetInputDataType(desc_, i),
                                                aclmdlGetInputFormat(desc_, i), bytes));
        }
        if (aclmdlGetInputDataType(desc_, i) != ACL_FLOAT16) {
            return Status(COSMO_NN_ERR_ASCEND_QUERY,
                          "Ascend input must be FP16 model=" + GetModelPath() + " " +
                              TensorDescription("input", i, io_dims, aclmdlGetInputDataType(desc_, i),
                                                aclmdlGetInputFormat(desc_, i), bytes));
        }
        input_sizes_.push_back(bytes);
        input_types_.push_back(aclmdlGetInputDataType(desc_, i));
    }

    output_shapes_.clear();
    output_sizes_.clear();
    output_types_.clear();
    for (size_t i = 0; i < num_outputs; ++i) {
        aclmdlIODims io_dims{};
        const aclError ret = aclmdlGetOutputDims(desc_, i, &io_dims);
        if (ret != ACL_SUCCESS)
            return MakeAclStatus(COSMO_NN_ERR_ASCEND_QUERY, "aclmdlGetOutputDims",
                                 "tensor=output[" + std::to_string(i) + "]", ret);
        const size_t bytes = aclmdlGetOutputSizeByIndex(desc_, i);
        // Phase-1 contract: yolo26 end2end single output [1, max_det, 6]
        // (each row is x1,y1,x2,y2,score,class_id).
        if (io_dims.dimCount != 3 || io_dims.dims[0] != 1 || io_dims.dims[2] != 6 || bytes == 0) {
            return Status(COSMO_NN_ERR_ASCEND_QUERY,
                          "Ascend output must be [1,N,6] end2end model=" + GetModelPath() + " " +
                              TensorDescription("output", i, io_dims, aclmdlGetOutputDataType(desc_, i),
                                                aclmdlGetOutputFormat(desc_, i), bytes));
        }
        if (aclmdlGetOutputDataType(desc_, i) != ACL_FLOAT16) {
            return Status(COSMO_NN_ERR_ASCEND_QUERY,
                          "Ascend output must be FP16 model=" + GetModelPath() + " " +
                              TensorDescription("output", i, io_dims, aclmdlGetOutputDataType(desc_, i),
                                                aclmdlGetOutputFormat(desc_, i), bytes));
        }
        DimsVector dims;
        for (size_t d = 0; d < io_dims.dimCount; ++d)
            dims.push_back(static_cast<int>(io_dims.dims[d]));
        output_shapes_.push_back(std::move(dims));
        output_sizes_.push_back(bytes);
        output_types_.push_back(aclmdlGetOutputDataType(desc_, i));
    }
    return COSMO_NN_OK;
}

Status AscendNetNode::AllocateIo() {
    input_dataset_  = aclmdlCreateDataset();
    output_dataset_ = aclmdlCreateDataset();
    if (input_dataset_ == nullptr || output_dataset_ == nullptr)
        return Status(COSMO_NN_ERR_ASCEND_MEM, "stage=ascend-alloc device=" + std::to_string(device_id_) +
                                                   " model=" + GetModelPath() +
                                                   " aclmdlCreateDataset failed");

    for (size_t i = 0; i < input_sizes_.size(); ++i) {
        void* dev          = nullptr;
        const aclError ret = aclrtMalloc(&dev, input_sizes_[i], ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
            return MakeAclStatus(
                COSMO_NN_ERR_ASCEND_MEM, "aclrtMalloc",
                "tensor=input[" + std::to_string(i) + "] size=" + std::to_string(input_sizes_[i]), ret);
        input_devices_.push_back(dev);
        aclDataBuffer* buf = aclCreateDataBuffer(dev, input_sizes_[i]);
        if (buf == nullptr)
            return Status(COSMO_NN_ERR_ASCEND_MEM, "stage=ascend-alloc device=" + std::to_string(device_id_) +
                                                       " model=" + GetModelPath() +
                                                       " aclCreateDataBuffer input[" + std::to_string(i) +
                                                       "] failed");
        input_buffers_.push_back(buf);
        const aclError add_ret = aclmdlAddDatasetBuffer(input_dataset_, buf);
        if (add_ret != ACL_SUCCESS)
            return MakeAclStatus(COSMO_NN_ERR_ASCEND_MEM, "aclmdlAddDatasetBuffer",
                                 "tensor=input[" + std::to_string(i) + "]", add_ret);
    }

    size_t max_output_bytes = 0;
    for (size_t i = 0; i < output_sizes_.size(); ++i) {
        void* dev          = nullptr;
        const aclError ret = aclrtMalloc(&dev, output_sizes_[i], ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
            return MakeAclStatus(
                COSMO_NN_ERR_ASCEND_MEM, "aclrtMalloc",
                "tensor=output[" + std::to_string(i) + "] size=" + std::to_string(output_sizes_[i]), ret);
        output_devices_.push_back(dev);
        aclDataBuffer* buf = aclCreateDataBuffer(dev, output_sizes_[i]);
        if (buf == nullptr)
            return Status(COSMO_NN_ERR_ASCEND_MEM, "stage=ascend-alloc device=" + std::to_string(device_id_) +
                                                       " model=" + GetModelPath() +
                                                       " aclCreateDataBuffer output[" + std::to_string(i) +
                                                       "] failed");
        output_buffers_.push_back(buf);
        const aclError add_ret = aclmdlAddDatasetBuffer(output_dataset_, buf);
        if (add_ret != ACL_SUCCESS)
            return MakeAclStatus(COSMO_NN_ERR_ASCEND_MEM, "aclmdlAddDatasetBuffer",
                                 "tensor=output[" + std::to_string(i) + "]", add_ret);
        max_output_bytes = std::max(max_output_bytes, output_sizes_[i]);
    }

    try {
        output_scratch_.resize(max_output_bytes);
    } catch (const std::bad_alloc&) {
        return Status(COSMO_NN_ERR_OUT_OF_MEMORY, "stage=ascend-alloc device=" + std::to_string(device_id_) +
                                                      " model=" + GetModelPath() +
                                                      " output scratch allocation failed");
    }
    return COSMO_NN_OK;
}

Status AscendNetNode::BindInputBlobs(std::vector<std::shared_ptr<Blob>>& bottom_blobs) {
    if (model_id_ == 0)
        return Status(COSMO_NN_ERR_ASCEND_BIND, "Ascend model is not loaded");
    if (bottom_blobs.size() != 1 || !bottom_blobs[0])
        return Status(COSMO_NN_ERR_ASCEND_BIND, "Ascend input blob is missing model=" + GetModelPath());

    const auto& desc   = bottom_blobs[0]->GetBlobDesc();
    const size_t bytes = DimsVectorUtils::Count(desc.dims) * DataTypeUtils::GetBytesSize(desc.data_type);
    if (bytes != input_sizes_[0]) {
        return Status(COSMO_NN_ERR_ASCEND_BIND, "Ascend input blob size mismatch model=" + GetModelPath() +
                                                    " expected=" + std::to_string(input_sizes_[0]) +
                                                    " got=" + std::to_string(bytes));
    }
    return COSMO_NN_OK;
}

void AscendNetNode::UpdateTopBlobDesc(size_t index, BlobDesc& desc) const {
    if (index >= output_shapes_.size())
        return;
    desc.dims        = output_shapes_[index];
    desc.data_type   = DATA_TYPE_FLOAT;
    desc.data_format = DATA_FORMAT_NCHW;
}

Status AscendNetNode::Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                              std::vector<std::shared_ptr<Blob>>& top_blobs) {
    timer.Start();

    if (model_id_ == 0 || input_dataset_ == nullptr || output_dataset_ == nullptr)
        return Status(COSMO_NN_ERR_ASCEND_RUN, "Ascend model is not loaded model=" + GetModelPath());
    if (bottom_blobs.size() != 1 || !bottom_blobs[0] || !bottom_blobs[0]->GetHandle().base)
        return Status(COSMO_NN_ERR_ASCEND_BIND, "Ascend input blob is missing model=" + GetModelPath());
    if (top_blobs.size() != output_sizes_.size())
        return Status(COSMO_NN_ERR_ASCEND_RUN, "Ascend output blob count mismatch model=" + GetModelPath() +
                                                   " expected=" + std::to_string(output_sizes_.size()) +
                                                   " got=" + std::to_string(top_blobs.size()));

    const size_t input_bytes = input_sizes_[0];
    const auto& input_desc   = bottom_blobs[0]->GetBlobDesc();
    const size_t host_input_bytes =
        DimsVectorUtils::Count(input_desc.dims) * DataTypeUtils::GetBytesSize(input_desc.data_type);
    if (host_input_bytes < input_bytes)
        return Status(COSMO_NN_ERR_ASCEND_BIND, "Ascend input blob is too small model=" + GetModelPath() +
                                                    " expected=" + std::to_string(input_bytes) +
                                                    " got=" + std::to_string(host_input_bytes));

    // ACL context is thread-local; Forward may run on a different thread than
    // LoadWeight, so re-bind this graph's context before using its stream.
    const aclError ctx_ret = aclrtSetCurrentContext(context_);
    if (ctx_ret != ACL_SUCCESS)
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_CONTEXT, "aclrtSetCurrentContext", "", ctx_ret);

    aclError ret = aclrtMemcpyAsync(input_devices_[0], input_bytes, bottom_blobs[0]->GetHandle().base,
                                    input_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream_);
    if (ret != ACL_SUCCESS)
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_MEMCPY, "input-h2d",
                             "tensor=input[0] size=" + std::to_string(input_bytes), ret);

    ret = aclmdlExecuteAsync(model_id_, input_dataset_, output_dataset_, stream_);
    if (ret != ACL_SUCCESS)
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_EXECUTE, "aclmdlExecuteAsync", "", ret);

    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS)
        return MakeAclStatus(COSMO_NN_ERR_ASCEND_SYNC, "aclrtSynchronizeStream", "", ret);

    for (size_t i = 0; i < output_sizes_.size(); ++i) {
        auto& top_blob = top_blobs[i];
        if (!top_blob || !top_blob->GetHandle().base)
            return Status(COSMO_NN_ERR_ASCEND_RUN, "Ascend output blob is unallocated model=" +
                                                       GetModelPath() + " index=" + std::to_string(i));

        const size_t out_bytes = output_sizes_[i];
        ret                    = aclrtMemcpy(output_scratch_.data(), out_bytes, output_devices_[i], out_bytes,
                                             ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            return MakeAclStatus(COSMO_NN_ERR_ASCEND_MEMCPY, "output-d2h",
                                 "tensor=output[" + std::to_string(i) + "] size=" + std::to_string(out_bytes),
                                 ret);

        if (output_types_[i] == ACL_FLOAT16) {
            const size_t count     = out_bytes / sizeof(uint16_t);
            const size_t dst_bytes = DimsVectorUtils::Count(top_blob->GetBlobDesc().dims) *
                                     DataTypeUtils::GetBytesSize(top_blob->GetBlobDesc().data_type);
            if (dst_bytes < count * sizeof(float))
                return Status(COSMO_NN_ERR_ASCEND_RUN, "Ascend output blob is too small model=" +
                                                           GetModelPath() + " index=" + std::to_string(i));
            const uint16_t* src = reinterpret_cast<const uint16_t*>(output_scratch_.data());
            float* dst          = static_cast<float*>(top_blob->GetHandle().base);
            for (size_t k = 0; k < count; ++k)
                dst[k] = Fp16ToFloat(src[k]);
        } else {
            std::memcpy(top_blob->GetHandle().base, output_scratch_.data(), out_bytes);
        }
    }

    timer.Stop();
    return COSMO_NN_OK;
}

Status AscendNetNode::MakeAclStatus(int code, const std::string& stage, const std::string& detail,
                                    aclError acl_ret) const {
    std::string message = "stage=ascend-" + stage + " device=" + std::to_string(device_id_);
    if (!GetModelPath().empty())
        message += " model=" + GetModelPath();
    if (!detail.empty())
        message += " " + detail;
    message += " acl_ret=" + std::to_string(static_cast<int>(acl_ret));
    return Status(code, message);
}

std::string AscendNetNode::TensorDescription(const std::string& kind, size_t index, const aclmdlIODims& dims,
                                             aclDataType dtype, aclFormat format, size_t bytes) const {
    std::string text = "tensor=" + kind + "[" + std::to_string(index) + "]";
    if (dims.name != nullptr && dims.name[0] != '\0')
        text += " name=" + std::string(dims.name);
    text += " dims=[";
    for (size_t i = 0; i < dims.dimCount; ++i) {
        if (i)
            text += "x";
        text += std::to_string(dims.dims[i]);
    }
    text += "] dtype=" + AclDataTypeName(dtype) + " format=" + AclFormatName(format) +
            " size=" + std::to_string(bytes);
    return text;
}

void AscendNetNode::Destroy() {
    // Release in the reverse order of creation; every step is safe to run on
    // a partially initialized node.
    for (auto* buf : output_buffers_) {
        if (buf)
            (void)aclDestroyDataBuffer(buf);
    }
    output_buffers_.clear();
    for (auto* buf : input_buffers_) {
        if (buf)
            (void)aclDestroyDataBuffer(buf);
    }
    input_buffers_.clear();

    if (output_dataset_ != nullptr) {
        (void)aclmdlDestroyDataset(output_dataset_);
        output_dataset_ = nullptr;
    }
    if (input_dataset_ != nullptr) {
        (void)aclmdlDestroyDataset(input_dataset_);
        input_dataset_ = nullptr;
    }

    for (auto* dev : output_devices_) {
        if (dev)
            (void)aclrtFree(dev);
    }
    output_devices_.clear();
    for (auto* dev : input_devices_) {
        if (dev)
            (void)aclrtFree(dev);
    }
    input_devices_.clear();
    output_scratch_.clear();

    if (desc_ != nullptr) {
        (void)aclmdlDestroyDesc(desc_);
        desc_ = nullptr;
    }
    if (model_id_ != 0) {
        (void)aclmdlUnload(model_id_);
        model_id_ = 0;
    }
    if (stream_ != nullptr) {
        (void)aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (context_ != nullptr) {
        (void)aclrtDestroyContext(context_);
        context_ = nullptr;
    }
    if (device_set_) {
        (void)ascend::ReleaseDevice(device_id_);
        device_set_ = false;
    }
    input_sizes_.clear();
    output_shapes_.clear();
    output_sizes_.clear();
    input_types_.clear();
    output_types_.clear();
}

}  // namespace cosmo::nn
