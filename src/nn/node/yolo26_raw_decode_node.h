#pragma once

#include <cstdint>
#include <vector>

#include "nn/core/common.h"
#include "nn/node/node.h"

namespace cosmo::nn {

class Yolo26RawDecodeNode : public Node {
public:
    Yolo26RawDecodeNode();
    virtual ~Yolo26RawDecodeNode();

    virtual void LoadParam(Op* op) override;
    virtual Status InferTopShapes() override;
    virtual Status Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                           std::vector<std::shared_ptr<Blob>>& top_blobs) override;
    virtual size_t GetBottomCount() override;
    virtual size_t GetTopCount() override;

private:
    struct Detection {
        float cx = 0.0f;
        float cy = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        float score = 0.0f;
        int class_id = 0;
    };

    void ResetTopBlob(std::shared_ptr<Blob> top_blob);
    Status ValidateBottoms(const std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                           std::vector<int>& strides, int& class_count, DataType& head_dtype);
    template <typename T>
    static void DecodeScale(const T* reg_data, const T* cls_data, int grid_h, int grid_w, int stride,
                            int class_count, float reg_scale, int reg_zp, float cls_scale, int cls_zp,
                            float cls_threshold, std::vector<Detection>& detections);
    float IoU(const Detection& lhs, const Detection& rhs) const;
    int QuantizeThreshold(float threshold, float scale, int zero_point) const;

    float nms_threshold_ = 0.45f;
    float base_conf_ = 0.25f;
    int top_k_ = 300;
    int reg_max_ = 1;
    int input_width_ = 0;
    int input_height_ = 0;
    int top_col_ = 6;
    std::vector<float> output_scales_{};
    std::vector<int> output_zero_points_{};
};

}  // namespace cosmo::nn
