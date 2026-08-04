#include "nn/node/yolo26_raw_decode_node.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "nn/node/node_type_utils.h"
#include "nn/utils/data_type_utils.h"
#include "nn/utils/dims_vector_utils.h"
#include "nn/utils/op.h"

namespace cosmo::nn {

namespace {

float HalfToFloat(std::uint16_t value) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000)) << 16;
    std::uint32_t exponent   = (value >> 10) & 0x1f;
    std::uint32_t mantissa   = value & 0x03ff;
    std::uint32_t bits       = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03ff;
            bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000 | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// Reads one head element as float: INT8 heads are affine-dequantized with the
// per-head scale/zero-point, FP16 heads are widened from half precision, and
// FP32 heads are already float.
template <typename T>
float ReadValue(T value, float scale, int zero_point) {
    if constexpr (std::is_same_v<T, std::int8_t>)
        return (static_cast<int>(value) - zero_point) * scale;
    else if constexpr (std::is_same_v<T, std::uint16_t>)
        return HalfToFloat(value);
    else
        return static_cast<float>(value);
}

}  // namespace

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

template <typename T>
void Yolo26RawDecodeNode::DecodeScale(const T* reg_data, const T* cls_data, int grid_h, int grid_w,
                                      int stride, int class_count, float reg_scale, int reg_zp,
                                      float cls_scale, int cls_zp, float cls_threshold,
                                      std::vector<Detection>& detections) {
    const int reg_hw = grid_h * grid_w;

    for (int y = 0; y < grid_h; ++y) {
        for (int x = 0; x < grid_w; ++x) {
            const int cell_index = y * grid_w + x;

            // reg_max=1 distances are signed: an object edge may lie on
            // either side of its anchor cell. Clamping would shift the
            // box, so keep the raw dequantized values like the
            // reference YOLO26 decode does.
            const float left   = ReadValue(reg_data[cell_index], reg_scale, reg_zp);
            const float top    = ReadValue(reg_data[reg_hw + cell_index], reg_scale, reg_zp);
            const float right  = ReadValue(reg_data[2 * reg_hw + cell_index], reg_scale, reg_zp);
            const float bottom = ReadValue(reg_data[3 * reg_hw + cell_index], reg_scale, reg_zp);

            const float anchor_x = static_cast<float>(x) + 0.5f;
            const float anchor_y = static_cast<float>(y) + 0.5f;
            const float x1       = (anchor_x - left) * stride;
            const float y1       = (anchor_y - top) * stride;
            const float x2       = (anchor_x + right) * stride;
            const float y2       = (anchor_y + bottom) * stride;

            // One candidate per cell: the highest-scoring class,
            // matching the reference YOLO26 decode. Emitting every
            // class would change candidate ordering and therefore NMS
            // tie-breaking for identically-scored cells.
            int best_class   = 0;
            float best_value = ReadValue(cls_data[cell_index], cls_scale, cls_zp);
            for (int class_id = 1; class_id < class_count; ++class_id) {
                const float value = ReadValue(cls_data[class_id * reg_hw + cell_index], cls_scale, cls_zp);
                if (value > best_value) {
                    best_value = value;
                    best_class = class_id;
                }
            }
            if (best_value <= cls_threshold) {
                continue;  // Strictly greater, like the reference.
            }

            Detection candidate;
            candidate.cx       = (x1 + x2) * 0.5f;
            candidate.cy       = (y1 + y2) * 0.5f;
            candidate.w        = std::max(0.0f, x2 - x1);
            candidate.h        = std::max(0.0f, y2 - y1);
            candidate.score    = 1.0f / (1.0f + std::exp(-best_value));
            candidate.class_id = best_class;
            detections.push_back(candidate);
        }
    }
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
    DataType head_dtype = DATA_TYPE_FLOAT;
    RETURN_ON_FAIL(ValidateBottoms(bottom_blobs, strides, class_count, head_dtype));

    const int batch = bottom_blobs.at(0)->GetBlobDesc().dims.at(0);
    if (batch > max_batch)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "batch size too large");
    SetCurrentBatch(top_blob, batch);

    auto* top_data      = static_cast<float*>(top_blob->GetHandle().base);
    const auto top_rows = top_blob->GetBlobDesc().dims.at(1);

    const bool is_int8 = head_dtype == DATA_TYPE_INT8;
    // INT8 keeps the affine quantized threshold; FP16/FP32 compare the raw
    // class logit against logit(confidence_threshold), which is exactly the
    // dequantized form of the same cutoff.
    float logit_threshold = -std::numeric_limits<float>::infinity();
    if (base_conf_ >= 1.0f)
        logit_threshold = std::numeric_limits<float>::infinity();
    else if (base_conf_ > 0.0f)
        logit_threshold = std::log(base_conf_ / (1.0f - base_conf_));

    std::vector<float> cls_thresholds;
    cls_thresholds.reserve(3);
    for (int scale_index = 0; scale_index < 3; ++scale_index) {
        const float cls_scale = is_int8 ? output_scales_.at(scale_index * 2 + 1) : 0.0f;
        const int cls_zp      = is_int8 ? output_zero_points_.at(scale_index * 2 + 1) : 0;
        const int cls_thresh  = is_int8 ? QuantizeThreshold(base_conf_, cls_scale, cls_zp) : 0;
        // (quantized_threshold - zero_point) * scale is the dequantized logit
        // cutoff; comparing dequantized values against it is exactly
        // equivalent to comparing the raw INT8 values against the quantized
        // threshold (scale > 0 makes the affine map order-preserving).
        cls_thresholds.push_back(is_int8 ? (static_cast<float>(cls_thresh - cls_zp)) * cls_scale
                                         : logit_threshold);
    }

    const int element_bytes = DataTypeUtils::GetBytesSize(head_dtype);

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

            const float reg_scale = is_int8 ? output_scales_.at(scale_index * 2) : 0.0f;
            const int reg_zp      = is_int8 ? output_zero_points_.at(scale_index * 2) : 0;
            const float cls_scale = is_int8 ? output_scales_.at(scale_index * 2 + 1) : 0.0f;
            const int cls_zp      = is_int8 ? output_zero_points_.at(scale_index * 2 + 1) : 0;

            auto* reg_base =
                static_cast<std::uint8_t*>(reg_blob->GetHandle().base) + batch_index * 4 * reg_hw * element_bytes;
            auto* cls_base =
                static_cast<std::uint8_t*>(cls_blob->GetHandle().base) + batch_index * cls_hw * element_bytes;

            switch (head_dtype) {
                case DATA_TYPE_INT8:
                    DecodeScale(reinterpret_cast<const std::int8_t*>(reg_base),
                                reinterpret_cast<const std::int8_t*>(cls_base), grid_h, grid_w, stride,
                                class_count, reg_scale, reg_zp, cls_scale, cls_zp,
                                cls_thresholds.at(scale_index), detections);
                    break;
                case DATA_TYPE_HALF:
                    DecodeScale(reinterpret_cast<const std::uint16_t*>(reg_base),
                                reinterpret_cast<const std::uint16_t*>(cls_base), grid_h, grid_w, stride,
                                class_count, reg_scale, reg_zp, cls_scale, cls_zp,
                                cls_thresholds.at(scale_index), detections);
                    break;
                default:
                    DecodeScale(reinterpret_cast<const float*>(reg_base), reinterpret_cast<const float*>(cls_base),
                                grid_h, grid_w, stride, class_count, reg_scale, reg_zp, cls_scale, cls_zp,
                                cls_thresholds.at(scale_index), detections);
                    break;
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
                if (IoU(accepted, detection) > nms_threshold_) {
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
                                            std::vector<int>& strides, int& class_count,
                                            DataType& head_dtype) {
    if (reg_max_ != 1)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw requires reg_max=1");
    if (bottom_blobs.size() != 6)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw expects six output heads");
    if (input_width_ <= 0 || input_height_ <= 0)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_raw input_size must be positive");

    head_dtype = bottom_blobs.at(0)->GetBlobDesc().data_type;
    if (head_dtype != DATA_TYPE_INT8 && head_dtype != DATA_TYPE_HALF && head_dtype != DATA_TYPE_FLOAT)
        return Status(COSMO_NN_ERR_INVALID_INPUT,
                      "yolo26_raw head 0 dtype must be INT8, FP16 (HALF), or FP32 (FLOAT); got " +
                          DataTypeUtils::GetDataTypeString(head_dtype));
    if (head_dtype == DATA_TYPE_INT8 && (output_scales_.size() != 6 || output_zero_points_.size() != 6))
        return Status(COSMO_NN_ERR_INVALID_INPUT,
                      "yolo26_raw INT8 heads require scale and zero-point metadata for all six outputs");

    const int batch = bottom_blobs.at(0)->GetBlobDesc().dims.at(0);
    strides.clear();
    class_count = 0;

    for (int scale_index = 0; scale_index < 3; ++scale_index) {
        const auto& reg_blob = bottom_blobs.at(scale_index * 2);
        const auto& cls_blob = bottom_blobs.at(scale_index * 2 + 1);
        const auto reg_desc  = reg_blob->GetBlobDesc();
        const auto cls_desc  = cls_blob->GetBlobDesc();

        if (reg_desc.data_type != head_dtype || cls_desc.data_type != head_dtype)
            return Status(COSMO_NN_ERR_INVALID_INPUT,
                          "yolo26_raw head " + std::to_string(scale_index * 2) +
                              " dtype differs from head 0; all six heads must share one dtype");
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

int Yolo26RawDecodeNode::QuantizeThreshold(float threshold, float scale, int zero_point) const {
    if (threshold <= 0.0f)
        return -128;
    if (threshold >= 1.0f)
        return 127;

    const float logit     = std::log(threshold / (1.0f - threshold));
    const float quantized = logit / scale + zero_point;
    return static_cast<int>(std::clamp(std::lround(quantized), -128l, 127l));
}

}  // namespace cosmo::nn
