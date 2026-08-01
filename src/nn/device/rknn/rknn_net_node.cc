#include "nn/device/rknn/rknn_net_node.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "nn/node/node_type_utils.h"
#include "nn/utils/string_format.h"

namespace cosmo::nn {

namespace {

constexpr int kRknnOutputCount = 6;

Status MakeRknnStatus(int code, const std::string& stage, const std::string& model_path,
                      const std::string& detail, int rknn_ret) {
    std::string message = stage;
    if (!model_path.empty())
        message += " model=" + model_path;
    if (!detail.empty())
        message += " " + detail;
    message += " rknn_ret=" + std::to_string(rknn_ret);
    return Status(code, message);
}

std::string AttrToString(const rknn_tensor_attr& attr) {
    std::string dims;
    for (uint32_t i = 0; i < attr.n_dims && i < 4; ++i) {
        if (i)
            dims += "x";
        dims += std::to_string(attr.dims[i]);
    }
    return "index=" + std::to_string(attr.index) + " fmt=" + std::to_string(attr.fmt) +
           " type=" + std::to_string(attr.type) + " dims=[" + dims + "]" +
           " size_with_stride=" + std::to_string(attr.size_with_stride) +
           " w_stride=" + std::to_string(attr.w_stride) + " h_stride=" + std::to_string(attr.h_stride);
}

}  // namespace

RknnNetNode::RknnNetNode() : NetNode() {
    node_type = NodeType::NODE_NET;
    name      = NodeTypeUtils::NodeTypeToStr(NodeType::NODE_NET).append("_0");
}

RknnNetNode::~RknnNetNode() {
    Destroy();
}

DeviceType RknnNetNode::GetInputBlobDeviceType() {
    return DeviceType::DEVICE_RKNN;
}

DeviceType RknnNetNode::GetTopBlobDeviceType() {
    // rknn_outputs_get returns host buffers, so the graph allocates plain
    // host blobs and no device->host copy node is inserted.
    return DeviceType::DEVICE_NAIVE;
}

size_t RknnNetNode::GetBottomCount() {
    return 1;
}

size_t RknnNetNode::GetTopCount() {
    return output_attrs_.empty() ? kRknnOutputCount : output_attrs_.size();
}

Status RknnNetNode::InferTopShapes() {
    top_blob_shapes.clear();
    top_blob_data_types.clear();
    for (const auto& attr : output_attrs_) {
        top_blob_shapes.push_back(
            {static_cast<int>(attr.dims[0]), static_cast<int>(attr.dims[1]),
             static_cast<int>(attr.dims[2]), static_cast<int>(attr.dims[3])});
        top_blob_data_types.push_back(DATA_TYPE_INT8);
    }
    return COSMO_NN_OK;
}

Status RknnNetNode::LoadWeight(const char* data, size_t size) {
    if (context_ != 0)
        return Status(COSMO_NN_ERR_PARAM, "RknnNetNode already loaded");
    if (data == nullptr || size == 0)
        return Status(COSMO_NN_ERR_LOAD_MODEL, "RKNN model data is empty");

    const int init_ret =
        rknn_init(&context_, const_cast<char*>(data), static_cast<uint32_t>(size), 0, nullptr);
    if (init_ret != RKNN_SUCC) {
        context_ = 0;
        return MakeRknnStatus(COSMO_NN_ERR_RKNN_INIT, "rknn_init failed", GetModelPath(),
                              "model_bytes=" + std::to_string(size), init_ret);
    }

    Status status = QueryModel();
    if (!bool(status)) {
        Destroy();
        return status;
    }

    const int core_ret = rknn_set_core_mask(context_, RKNN_NPU_CORE_AUTO);
    if (core_ret != RKNN_SUCC) {
        Destroy();
        return MakeRknnStatus(COSMO_NN_ERR_RKNN_QUERY, "rknn_set_core_mask failed", GetModelPath(),
                              "requested RKNN_NPU_CORE_AUTO", core_ret);
    }
    return COSMO_NN_OK;
}

Status RknnNetNode::QueryModel() {
    rknn_input_output_num count{};
    int ret = rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count));
    if (ret != RKNN_SUCC)
        return MakeRknnStatus(COSMO_NN_ERR_RKNN_QUERY, "rknn_query IN_OUT_NUM failed", GetModelPath(), "", ret);
    if (count.n_input != 1 || count.n_output != kRknnOutputCount) {
        return Status(COSMO_NN_ERR_RKNN_QUERY,
                      "RKNN model contract mismatch model=" + GetModelPath() + " inputs=" +
                          std::to_string(count.n_input) + " outputs=" + std::to_string(count.n_output) +
                          " expected 1 input and 6 outputs");
    }

    input_attr_        = rknn_tensor_attr{};
    input_attr_.index  = 0;
    ret                = rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
    if (ret != RKNN_SUCC)
        return MakeRknnStatus(COSMO_NN_ERR_RKNN_QUERY, "rknn_query INPUT_ATTR failed", GetModelPath(), "", ret);
    if (input_attr_.n_dims != 4 || input_attr_.fmt != RKNN_TENSOR_NHWC || input_attr_.dims[3] != 3) {
        return Status(COSMO_NN_ERR_RKNN_QUERY, "RKNN input is not NHWC RGB model=" + GetModelPath() + " " +
                                                   AttrToString(input_attr_));
    }
    const int height = static_cast<int>(input_attr_.dims[1]);
    const int width  = static_cast<int>(input_attr_.dims[2]);
    if (height <= 0 || width <= 0 || height != width) {
        return Status(COSMO_NN_ERR_RKNN_QUERY,
                      "RKNN zero-copy input must be square model=" + GetModelPath() + " " +
                          AttrToString(input_attr_));
    }
    width_stride_  = input_attr_.w_stride ? static_cast<int>(input_attr_.w_stride) : width;
    height_stride_ = input_attr_.h_stride ? static_cast<int>(input_attr_.h_stride) : height;
    if (width_stride_ < width || height_stride_ < height ||
        input_attr_.size_with_stride <
            static_cast<uint32_t>(width_stride_) * static_cast<uint32_t>(height_stride_) * 3U) {
        return Status(COSMO_NN_ERR_RKNN_QUERY, "RKNN input stride/size is invalid model=" + GetModelPath() +
                                                   " " + AttrToString(input_attr_));
    }

    output_attrs_.resize(count.n_output);
    for (uint32_t i = 0; i < count.n_output; ++i) {
        rknn_tensor_attr& attr = output_attrs_[i];
        attr.index             = i;
        ret                    = rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC)
            return MakeRknnStatus(COSMO_NN_ERR_RKNN_QUERY, "rknn_query OUTPUT_ATTR failed", GetModelPath(),
                                  "output_index=" + std::to_string(i), ret);
        if (attr.n_dims != 4 || attr.dims[0] != 1 || attr.fmt != RKNN_TENSOR_NCHW) {
            return Status(COSMO_NN_ERR_RKNN_QUERY, "RKNN output is not 4D NCHW model=" + GetModelPath() + " " +
                                                       AttrToString(attr));
        }
        if (attr.type != RKNN_TENSOR_INT8 || attr.qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
            return Status(COSMO_NN_ERR_RKNN_QUERY,
                          "RKNN output is not affine-quantized INT8 model=" + GetModelPath() + " " +
                              AttrToString(attr));
        }
        if (i % 2 == 0 && attr.dims[1] != 4) {
            return Status(COSMO_NN_ERR_RKNN_QUERY,
                          "RKNN reg output requires reg_max=1 model=" + GetModelPath() + " " +
                              AttrToString(attr));
        }
    }
    return COSMO_NN_OK;
}

Status RknnNetNode::CreateInputTensor(std::vector<std::shared_ptr<Blob>>& bottom_blobs) {
    if (bottom_blobs.size() != 1 || !bottom_blobs[0])
        return Status(COSMO_NN_ERR_RKNN_BIND, "RKNN net input blob is missing");

    input_mem_ = rknn_create_mem(context_, input_attr_.size_with_stride);
    if (input_mem_ == nullptr)
        return MakeRknnStatus(COSMO_NN_ERR_RKNN_MEM, "rknn_create_mem failed", GetModelPath(),
                              "size_with_stride=" + std::to_string(input_attr_.size_with_stride), -1);

    // Bind the tensor memory as the network input. pass_through=0 + UINT8 lets
    // the runtime quantize the RGB888 bytes written by RGA (same contract as
    // the reference rknn_mpp sample).
    input_attr_.type         = RKNN_TENSOR_UINT8;
    input_attr_.fmt          = RKNN_TENSOR_NHWC;
    input_attr_.pass_through = 0;
    input_attr_.h_stride     = static_cast<uint32_t>(height_stride_);
    const int bind_ret       = rknn_set_io_mem(context_, input_mem_, &input_attr_);
    if (bind_ret != RKNN_SUCC) {
        rknn_destroy_mem(context_, input_mem_);
        input_mem_ = nullptr;
        return MakeRknnStatus(COSMO_NN_ERR_RKNN_BIND, "rknn_set_io_mem failed", GetModelPath(),
                              AttrToString(input_attr_), bind_ret);
    }

    binding_               = std::make_unique<RknnTensorBinding>();
    binding_->context      = context_;
    binding_->memory       = input_mem_;
    binding_->width_stride = width_stride_;
    binding_->height_stride = height_stride_;
    binding_->model_width  = static_cast<int>(input_attr_.dims[2]);
    binding_->model_height = static_cast<int>(input_attr_.dims[1]);
    binding_->model_path   = GetModelPath();

    auto& desc       = bottom_blobs[0]->GetBlobDesc();
    desc.device_type = DeviceType::DEVICE_RKNN;
    desc.data_format = DATA_FORMAT_NHWC;
    desc.data_type   = DATA_TYPE_UINT8;
    desc.dims        = {1, binding_->model_height, binding_->model_width, 3};

    BlobHandle handle;
    handle.base      = binding_.get();
    handle.ownership = BLOB_HANDLE_EXTERNAL_OWNED;
    bottom_blobs[0]->SetHandle(handle);
    return COSMO_NN_OK;
}

Status RknnNetNode::BindInputBlobs(std::vector<std::shared_ptr<Blob>>& bottom_blobs) {
    if (context_ == 0)
        return Status(COSMO_NN_ERR_RKNN_BIND, "RKNN model is not loaded");
    return CreateInputTensor(bottom_blobs);
}

void RknnNetNode::UpdateTopBlobDesc(size_t index, BlobDesc& desc) const {
    if (index >= output_attrs_.size())
        return;
    const rknn_tensor_attr& attr = output_attrs_[index];
    desc.dims                    = {static_cast<int>(attr.dims[0]), static_cast<int>(attr.dims[1]),
                       static_cast<int>(attr.dims[2]), static_cast<int>(attr.dims[3])};
    desc.data_type               = DATA_TYPE_INT8;
    desc.data_format             = DATA_FORMAT_NCHW;
    desc.is_affine_quantized     = true;
    desc.affine_scale            = attr.scale;
    desc.affine_zero_point       = static_cast<int>(attr.zp);
}

Status RknnNetNode::Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                            std::vector<std::shared_ptr<Blob>>& top_blobs) {
    timer.Start();

    if (!binding_ || binding_->memory == nullptr)
        return Status(COSMO_NN_ERR_RKNN_RUN, "RKNN input tensor is not bound");
    if (top_blobs.size() != output_attrs_.size())
        return Status(COSMO_NN_ERR_RKNN_RUN, "RKNN output blob count mismatch model=" + GetModelPath() +
                                                 " expected=" + std::to_string(output_attrs_.size()) +
                                                 " got=" + std::to_string(top_blobs.size()));

    const int run_ret = rknn_run(context_, nullptr);
    if (run_ret != RKNN_SUCC)
        return MakeRknnStatus(COSMO_NN_ERR_RKNN_RUN, "rknn_run failed", GetModelPath(), "", run_ret);

    std::vector<rknn_output> outputs(output_attrs_.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i].index      = static_cast<uint32_t>(i);
        outputs[i].want_float = 0;
    }
    const int get_ret =
        rknn_outputs_get(context_, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr);
    if (get_ret != RKNN_SUCC)
        return MakeRknnStatus(COSMO_NN_ERR_RKNN_OUTPUT, "rknn_outputs_get failed", GetModelPath(), "", get_ret);

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto& top_blob = top_blobs[i];
        if (!top_blob || !top_blob->GetHandle().base) {
            (void)rknn_outputs_release(context_, static_cast<uint32_t>(outputs.size()), outputs.data());
            return Status(COSMO_NN_ERR_RKNN_OUTPUT, "RKNN output blob is unallocated model=" + GetModelPath() +
                                                        " index=" + std::to_string(i));
        }
        const size_t tensor_bytes = static_cast<size_t>(output_attrs_[i].size);
        if (outputs[i].buf == nullptr || tensor_bytes == 0) {
            (void)rknn_outputs_release(context_, static_cast<uint32_t>(outputs.size()), outputs.data());
            return Status(COSMO_NN_ERR_RKNN_OUTPUT, "RKNN output buffer is empty model=" + GetModelPath() +
                                                        " index=" + std::to_string(i));
        }
        std::memcpy(top_blob->GetHandle().base, outputs[i].buf, tensor_bytes);
    }

    const int release_ret =
        rknn_outputs_release(context_, static_cast<uint32_t>(outputs.size()), outputs.data());
    if (release_ret != RKNN_SUCC)
        return MakeRknnStatus(COSMO_NN_ERR_RKNN_RELEASE, "rknn_outputs_release failed", GetModelPath(), "",
                              release_ret);

    timer.Stop();
    return COSMO_NN_OK;
}

void RknnNetNode::Destroy() {
    if (context_ != 0 && input_mem_ != nullptr) {
        rknn_destroy_mem(context_, input_mem_);
        input_mem_ = nullptr;
    }
    if (context_ != 0) {
        rknn_destroy(context_);
        context_ = 0;
    }
    binding_.reset();
    output_attrs_.clear();
}

}  // namespace cosmo::nn
