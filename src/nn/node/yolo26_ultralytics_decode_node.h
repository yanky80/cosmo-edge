#pragma once

#include <cstdint>
#include <vector>

#include "nn/core/common.h"
#include "nn/node/node.h"

namespace cosmo::nn {

/**
 * @brief Host decode node for a single Ultralytics-exported detection output.
 *
 * Consumes the model's one post-processed output tensor [1, 4+nc, N]
 * (FP32/FP16, channel-major): rows 0-3 are cx, cy, w, h in net-input pixels,
 * rows 4+ are per-class sigmoid probabilities. dist2bbox and the class
 * sigmoid are already baked into the exported graph, so the host only runs
 * class-aware NMS + top-k and emits [1, top_k, 6]
 * (cx, cy, w, h, score, class_id) for the framework's PickDetectionObjects /
 * AdjustSize letterbox coordinate recovery.
 */
class Yolo26UltralyticsDecodeNode : public Node {
public:
    Yolo26UltralyticsDecodeNode();
    virtual ~Yolo26UltralyticsDecodeNode();

    virtual void LoadParam(Op* op) override;
    virtual Status InferTopShapes() override;
    virtual Status Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                           std::vector<std::shared_ptr<Blob>>& top_blobs) override;
    virtual size_t GetBottomCount() override;
    virtual size_t GetTopCount() override;

private:
    struct Detection {
        float cx     = 0.0f;
        float cy     = 0.0f;
        float w      = 0.0f;
        float h      = 0.0f;
        float score  = 0.0f;
        int class_id = 0;
    };

    template <typename T>
    Status CollectCandidates(const T* data, int cells, int channels, float conf_threshold,
                             std::vector<Detection>& detections) const;
    void ResetTopBlob(std::shared_ptr<Blob> top_blob);
    float IoU(const Detection& lhs, const Detection& rhs) const;

    float nms_threshold_ = 0.45f;
    float base_conf_     = 0.25f;
    int top_k_           = 300;
    int top_col_         = 6;
};

}  // namespace cosmo::nn
