#include "nn/node/yolo26_raw_decode_node.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "nn/node/node_type_utils.h"
#include "nn/utils/dims_vector_utils.h"
#include "nn/utils/op.h"

namespace cosmo::nn {

Yolo26RawDecodeNode::Yolo26RawDecodeNode() : Node() {
    node_type     = NODE_YOLO26_RAW_DECODE;
    name          = NodeTypeUtils::NodeTypeToStr(NODE_YOLO26_RAW_DECODE).append("_0");
    one_blob_only = false;
}

Yolo26RawDecodeNode::~Yolo26RawDecodeNode() {}

void Yolo26RawDecodeNode::LoadParam(Op* op) {
    auto* raw = dynamic_cast<Yolo26RawPost*>(op);
    if (!raw)
        return;

    nms_threshold_      = raw->nms_threshold;
    base_conf_          = raw->nms_detection_conf;
    top_k_              = raw->top_k;
    reg_max_            = raw->reg_max;
    input_width_        = raw->input_width;
    input_height_       = raw->input_height;
    output_scales_      = raw->output_scales;
    output_zero_points_ = raw->output_zero_points;
}

Status Yolo26RawDecodeNode::InferTopShapes() {
    top_blob_shapes     = {{max_batch, top_k_, top_col_}};
    top_blob_data_types = {DATA_TYPE_FLOAT};
    return COSMO_NN_OK;
}

size_t Yolo26RawDecodeNode::GetBottomCount() {
    return 6;
}

size_t Yolo26RawDecodeNode::GetTopCount() {
    return 1;
}

Status Yolo26RawDecodeNode::Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                                    std::vector<std::shared_ptr<Blob>>& top_blobs) {
    timer.Start();

    if (top_blobs.size() != 1)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw expects one top blob");

    auto top_blob = top_blobs.at(0);
    ResetTopBlob(top_blob);

    std::vector<int> strides;
    int class_count = 0;
    RETURN_ON_FAIL(ValidateBottoms(bottom_blobs, strides, class_count));

    const int batch = bottom_blobs.at(0)->GetBlobDesc().dims.at(0);
    if (batch > max_batch)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "batch size too large");
    SetCurrentBatch(top_blob, batch);

    auto* top_data = static_cast<float*>(top_blob->GetHandle().base);
    const auto top_rows = top_blob->GetBlobDesc().dims.at(1);

    std::vector<int> cls_thresholds;
    cls_thresholds.reserve(3);
    for (int scale_index = 0; scale_index < 3; ++scale_index)
        cls_thresholds.push_back(
            QuantizeThreshold(base_conf_, output_scales_.at(scale_index * 2 + 1),
                              output_zero_points_.at(scale_index * 2 + 1)));

    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        std::vector<Detection> detections;
        for (int scale_index = 0; scale_index < 3; ++scale_index) {
            const auto& reg_blob = bottom_blobs.at(scale_index * 2);
            const auto& cls_blob = bottom_blobs.at(scale_index * 2 + 1);
            const auto reg_dims  = reg_blob->GetBlobDesc().dims;
            const int grid_h     = reg_dims.at(2);
            const int grid_w     = reg_dims.at(3);
            const int stride     = strides.at(scale_index);
            const int reg_hw     = grid_h * grid_w;
            const int cls_hw     = class_count * reg_hw;

            auto* reg_data = static_cast<std::int8_t*>(reg_blob->GetHandle().base) +
                             batch_index * 4 * reg_hw;
            auto* cls_data = static_cast<std::int8_t*>(cls_blob->GetHandle().base) +
                             batch_index * cls_hw;

            const float reg_scale = output_scales_.at(scale_index * 2);
            const int reg_zp      = output_zero_points_.at(scale_index * 2);
            const float cls_scale = output_scales_.at(scale_index * 2 + 1);
            const int cls_zp      = output_zero_points_.at(scale_index * 2 + 1);
            const int cls_thresh  = cls_thresholds.at(scale_index);

            for (int y = 0; y < grid_h; ++y) {
                for (int x = 0; x < grid_w; ++x) {
                    const int cell_index = y * grid_w + x;

                    float left = std::max(0.0f, Dequantize(reg_data[cell_index], reg_scale, reg_zp));
                    float top = std::max(0.0f, Dequantize(reg_data[reg_hw + cell_index], reg_scale, reg_zp));
                    float right =
                        std::max(0.0f, Dequantize(reg_data[2 * reg_hw + cell_index], reg_scale, reg_zp));
                    float bottom =
                        std::max(0.0f, Dequantize(reg_data[3 * reg_hw + cell_index], reg_scale, reg_zp));

                    const float anchor_x = static_cast<float>(x) + 0.5f;
                    const float anchor_y = static_cast<float>(y) + 0.5f;
                    const float x1       = (anchor_x - left) * stride;
                    const float y1       = (anchor_y - top) * stride;
                    const float x2       = (anchor_x + right) * stride;
                    const float y2       = (anchor_y + bottom) * stride;

                    Detection candidate;
                    candidate.cx = (x1 + x2) * 0.5f;
                    candidate.cy = (y1 + y2) * 0.5f;
                    candidate.w  = std::max(0.0f, x2 - x1);
                    candidate.h  = std::max(0.0f, y2 - y1);

                    for (int class_id = 0; class_id < class_count; ++class_id) {
                        const int cls_index = class_id * reg_hw + cell_index;
                        if (cls_data[cls_index] < cls_thresh)
                            continue;

                        candidate.score = Sigmoid(Dequantize(cls_data[cls_index], cls_scale, cls_zp));
                        if (candidate.score < base_conf_)
                            continue;

                        candidate.class_id = class_id;
                        detections.push_back(candidate);
                    }
                }
            }
        }

        std::sort(detections.begin(), detections.end(),
                  [](const Detection& lhs, const Detection& rhs) { return lhs.score > rhs.score; });

        std::vector<Detection> kept;
        kept.reserve(std::min(top_k_, static_cast<int>(detections.size())));
        for (const auto& detection : detections) {
            bool suppressed = false;
            for (const auto& accepted : kept) {
                if (accepted.class_id != detection.class_id)
                    continue;
                if (IoU(accepted, detection) >= nms_threshold_) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept.push_back(detection);
                if (static_cast<int>(kept.size()) >= top_k_)
                    break;
            }
        }

        float* batch_top = top_data + batch_index * top_rows * top_col_;
        for (std::size_t index = 0; index < kept.size(); ++index) {
            batch_top[index * top_col_ + 0] = kept[index].cx;
            batch_top[index * top_col_ + 1] = kept[index].cy;
            batch_top[index * top_col_ + 2] = kept[index].w;
            batch_top[index * top_col_ + 3] = kept[index].h;
            batch_top[index * top_col_ + 4] = kept[index].score;
            batch_top[index * top_col_ + 5] = static_cast<float>(kept[index].class_id);
        }
    }

    timer.Stop();
    return COSMO_NN_OK;
}

void Yolo26RawDecodeNode::ResetTopBlob(std::shared_ptr<Blob> top_blob) {
    auto top_desc = top_blob->GetBlobDesc();
    auto top_dim  = top_desc.dims;
    top_dim.at(0) = max_batch;
    top_desc.dims = top_dim;
    top_blob->SetBlobDesc(top_desc);

    const int count = DimsVectorUtils::Count(top_dim);
    auto* data      = static_cast<float*>(top_blob->GetHandle().base);
    std::fill(data, data + count, 0.0f);
}

Status Yolo26RawDecodeNode::ValidateBottoms(const std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                                            std::vector<int>& strides, int& class_count) {
    if (reg_max_ != 1)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw requires reg_max=1");
    if (bottom_blobs.size() != 6)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw expects six output heads");
    if (output_scales_.size() != 6 || output_zero_points_.size() != 6)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw quant metadata must describe six outputs");
    if (input_width_ <= 0 || input_height_ <= 0)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw input_size must be positive");

    const int batch = bottom_blobs.at(0)->GetBlobDesc().dims.at(0);
    strides.clear();
    class_count = 0;

    for (int scale_index = 0; scale_index < 3; ++scale_index) {
        const auto& reg_blob = bottom_blobs.at(scale_index * 2);
        const auto& cls_blob = bottom_blobs.at(scale_index * 2 + 1);
        const auto reg_desc  = reg_blob->GetBlobDesc();
        const auto cls_desc  = cls_blob->GetBlobDesc();

        if (reg_desc.data_type != DATA_TYPE_INT8 || cls_desc.data_type != DATA_TYPE_INT8)
            return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw outputs must be INT8");
        if (reg_desc.dims.size() != 4 || cls_desc.dims.size() != 4)
            return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw outputs must be NCHW tensors");
        if (reg_desc.dims.at(0) != batch || cls_desc.dims.at(0) != batch)
            return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw batch mismatch across outputs");
        if (reg_desc.dims.at(1) != 4)
            return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw reg heads must expose four channels");
        if (cls_desc.dims.at(1) <= 0)
            return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw cls heads must expose class channels");
        if (reg_desc.dims.at(2) != cls_desc.dims.at(2) || reg_desc.dims.at(3) != cls_desc.dims.at(3))
            return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw reg/cls feature maps must match");

        const int grid_h = reg_desc.dims.at(2);
        const int grid_w = reg_desc.dims.at(3);
        if (grid_h <= 0 || grid_w <= 0 || input_width_ % grid_w != 0 || input_height_ % grid_h != 0)
            return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw feature map shape does not match input");

        const int stride_w = input_width_ / grid_w;
        const int stride_h = input_height_ / grid_h;
        if (stride_w != stride_h)
            return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw requires square strides");
        strides.push_back(stride_w);

        if (class_count == 0)
            class_count = cls_desc.dims.at(1);
        else if (class_count != cls_desc.dims.at(1))
            return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw class channel count mismatch");
    }

    return COSMO_NN_OK;
}

float Yolo26RawDecodeNode::IoU(const Detection& lhs, const Detection& rhs) const {
    const float lhs_left   = lhs.cx - lhs.w * 0.5f;
    const float lhs_right  = lhs.cx + lhs.w * 0.5f;
    const float lhs_top    = lhs.cy - lhs.h * 0.5f;
    const float lhs_bottom = lhs.cy + lhs.h * 0.5f;
    const float rhs_left   = rhs.cx - rhs.w * 0.5f;
    const float rhs_right  = rhs.cx + rhs.w * 0.5f;
    const float rhs_top    = rhs.cy - rhs.h * 0.5f;
    const float rhs_bottom = rhs.cy + rhs.h * 0.5f;

    const float inter_left   = std::max(lhs_left, rhs_left);
    const float inter_right  = std::min(lhs_right, rhs_right);
    const float inter_top    = std::max(lhs_top, rhs_top);
    const float inter_bottom = std::min(lhs_bottom, rhs_bottom);
    if (inter_left >= inter_right || inter_top >= inter_bottom)
        return 0.0f;

    const float intersection = (inter_right - inter_left) * (inter_bottom - inter_top);
    const float union_area   = lhs.w * lhs.h + rhs.w * rhs.h - intersection;
    if (union_area <= 0.0f)
        return 0.0f;
    return intersection / union_area;
}

float Yolo26RawDecodeNode::Sigmoid(float value) const {
    return 1.0f / (1.0f + std::exp(-value));
}

float Yolo26RawDecodeNode::Dequantize(std::int8_t value, float scale, int zero_point) const {
    return (static_cast<int>(value) - zero_point) * scale;
}

int Yolo26RawDecodeNode::QuantizeThreshold(float threshold, float scale, int zero_point) const {
    if (threshold <= 0.0f)
        return -128;
    if (threshold >= 1.0f)
        return 127;

    const float logit = std::log(threshold / (1.0f - threshold));
    const float quantized = logit / scale + zero_point;
    return static_cast<int>(std::clamp(std::lround(quantized), -128l, 127l));
}

}  // namespace cosmo::nn
