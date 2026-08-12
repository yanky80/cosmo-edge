#include "nn/node/yolo26_ultralytics_decode_node.h"

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

    template <typename T>
    float ReadValue(T value) {
        if constexpr (std::is_same_v<T, std::uint16_t>)
            return HalfToFloat(value);
        else
            return static_cast<float>(value);
    }

}  // namespace

Yolo26UltralyticsDecodeNode::Yolo26UltralyticsDecodeNode() : Node() {
    node_type     = NODE_YOLO26_ULTRALYTICS_DECODE;
    name          = NodeTypeUtils::NodeTypeToStr(NODE_YOLO26_ULTRALYTICS_DECODE).append("_0");
    one_blob_only = true;
}

Yolo26UltralyticsDecodeNode::~Yolo26UltralyticsDecodeNode() {}

void Yolo26UltralyticsDecodeNode::LoadParam(Op* op) {
    auto* post = dynamic_cast<YoloPost*>(op);
    if (!post)
        return;

    nms_threshold_ = post->nms_threshold;
    base_conf_     = post->nms_detection_conf;
    top_k_         = post->top_k;
}

Status Yolo26UltralyticsDecodeNode::InferTopShapes() {
    top_blob_shapes     = {{max_batch, top_k_, top_col_}};
    top_blob_data_types = {DATA_TYPE_FLOAT};
    return COSMO_NN_OK;
}

size_t Yolo26UltralyticsDecodeNode::GetBottomCount() {
    return 1;
}

size_t Yolo26UltralyticsDecodeNode::GetTopCount() {
    return 1;
}

template <typename T>
Status Yolo26UltralyticsDecodeNode::CollectCandidates(const T* data, int cells, int channels,
                                                      float conf_threshold,
                                                      std::vector<Detection>& detections) const {
    const int class_count = channels - 4;
    for (int cell = 0; cell < cells; ++cell) {
        const float cx = ReadValue(data[cell]);
        const float cy = ReadValue(data[cells + cell]);
        const float w  = ReadValue(data[2 * cells + cell]);
        const float h  = ReadValue(data[3 * cells + cell]);

        int best_class   = 0;
        float best_value = -1.0f;
        for (int class_id = 0; class_id < class_count; ++class_id) {
            const float value = ReadValue(data[(4 + class_id) * cells + cell]);
            // Class rows carry sigmoid probabilities in (0, 1). Any value
            // outside [0, 1] means the buffer is not the channel-major
            // ultralytics layout (e.g. a cell-major [1, N, 4+nc] tensor
            // misread here, whose "class rows" would hold box coordinates),
            // so fail with an actionable error instead of emitting garbage
            // detections.
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
                return Status(COSMO_NN_ERR_INVALID_INPUT,
                              "yolo26_ultralytics class score out of [0,1]; expected channel-major "
                              "[1, 4+num_classes, N] ultralytics output");
            if (value > best_value) {
                best_value = value;
                best_class = class_id;
            }
        }
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h))
            continue;
        if (best_value <= conf_threshold)
            continue;  // Strictly greater, like the reference decode.

        Detection candidate;
        candidate.cx       = cx;
        candidate.cy       = cy;
        candidate.w        = std::max(0.0f, w);
        candidate.h        = std::max(0.0f, h);
        candidate.score    = best_value;
        candidate.class_id = best_class;
        detections.push_back(candidate);
    }
    return COSMO_NN_OK;
}

Status Yolo26UltralyticsDecodeNode::Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                                            std::vector<std::shared_ptr<Blob>>& top_blobs) {
    timer.Start();

    if (top_blobs.size() != 1)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_ultralytics expects one top blob");

    auto top_blob = top_blobs.at(0);
    ResetTopBlob(top_blob);

    if (bottom_blobs.size() != 1)
        return Status(COSMO_NN_ERR_INVALID_INPUT,
                      "yolo26_ultralytics expects exactly one model output tensor");

    auto bottom_blob = bottom_blobs.at(0);
    auto bottom_desc = bottom_blob->GetBlobDesc();
    auto bottom_dim  = bottom_desc.dims;

    if (bottom_dim.size() != 3)
        return Status(COSMO_NN_ERR_INVALID_INPUT,
                      "yolo26_ultralytics output must be [1, 4+num_classes, N]; got rank " +
                          std::to_string(bottom_dim.size()));
    const int batch    = bottom_dim.at(0);
    const int channels = bottom_dim.at(1);
    const int cells    = bottom_dim.at(2);
    if (batch > static_cast<int>(max_batch))
        return Status(COSMO_NN_ERR_INVALID_INPUT, "batch size too large");
    if (channels < 5)
        return Status(COSMO_NN_ERR_INVALID_INPUT,
                      "yolo26_ultralytics output must expose at least 5 channels (4 box + 1 class); got " +
                          std::to_string(channels));
    if (cells <= 0)
        return Status(COSMO_NN_ERR_INVALID_INPUT, "yolo26_ultralytics output has no candidates");
    if (bottom_desc.data_type != DATA_TYPE_FLOAT && bottom_desc.data_type != DATA_TYPE_HALF)
        return Status(COSMO_NN_ERR_INVALID_INPUT,
                      "yolo26_ultralytics output dtype must be FP32 or FP16; got " +
                          DataTypeUtils::GetDataTypeString(bottom_desc.data_type));

    SetCurrentBatch(top_blob, batch);

    auto top_dim       = top_blob->GetBlobDesc().dims;
    auto top_handle    = top_blob->GetHandle();
    const int top_rows = top_dim.at(1);
    auto* top_data     = static_cast<float*>(top_handle.base);

    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        std::vector<Detection> detections;
        auto* base = static_cast<std::uint8_t*>(bottom_blob->GetHandle().base) +
                     batch_index * channels * cells * DataTypeUtils::GetBytesSize(bottom_desc.data_type);

        switch (bottom_desc.data_type) {
            case DATA_TYPE_HALF:
                RETURN_ON_FAIL(CollectCandidates(reinterpret_cast<const std::uint16_t*>(base), cells,
                                                 channels, base_conf_, detections));
                break;
            default:
                RETURN_ON_FAIL(CollectCandidates(reinterpret_cast<const float*>(base), cells, channels,
                                                 base_conf_, detections));
                break;
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

void Yolo26UltralyticsDecodeNode::ResetTopBlob(std::shared_ptr<Blob> top_blob) {
    auto top_desc = top_blob->GetBlobDesc();
    auto top_dim  = top_desc.dims;
    top_dim.at(0) = max_batch;
    top_desc.dims = top_dim;
    top_blob->SetBlobDesc(top_desc);

    const int count = DimsVectorUtils::Count(top_dim);
    auto* data      = static_cast<float*>(top_blob->GetHandle().base);
    std::fill(data, data + count, 0.0f);
}

float Yolo26UltralyticsDecodeNode::IoU(const Detection& lhs, const Detection& rhs) const {
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

}  // namespace cosmo::nn
